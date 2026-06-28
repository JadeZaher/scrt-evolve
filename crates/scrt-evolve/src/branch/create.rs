//! `branch::create` — the branch factory orchestrator (track 29, Phase 2).
//!
//! **Composition-first**: this assembles the SHIPPED stages (discover → teacher-QA
//! generate → train → eval gate → GGUF export) scoped to a per-branch config, with
//! the **weight-touching span (train → eval) running inside the track-15
//! transaction** (checkpoint → eval → keep | rollback; catastrophe → quarantine by
//! `gen=branch:<name>` + halt). It adds **no new ML** — the heavy stages are
//! injected as closures (mirroring [`crate::rounds::RoundHooks`]) so the SDK stays
//! ML-free + deterministically testable; the CLI wires the real subprocess stages.
//!
//! A branch is **registered only if it passes its eval gate**. A regress rolls the
//! weights back and leaves the registry untouched; a catastrophe additionally
//! quarantines the branch's provenance and signals halt.

use std::path::{Path, PathBuf};

use crate::config::EvolveConfig;
use crate::dataset::{Dataset, GenExample};
use crate::discover::DiscoveredContext;
use crate::eval::{ProbeSet, ScoreReport};
use crate::regulate::{Regulator, StepAction};
use crate::workdir::WorkDir;

use super::manifest::{sha256_file, BranchManifest, BranchRegistry, Lineage, MANIFEST_VERSION};
use super::router::{admit, corpus_signature, AdmitOutcome};

/// Similarity at/above which a new branch is a near-duplicate of an existing one
/// and merges instead of spawning a twin (styleguide §2.5; reuses track-14 shape).
const MERGE_THRESHOLD: f32 = 0.85;

/// The injected heavy stages a create needs. Kept as closures so the SDK driver is
/// ML-free + testable (deterministic mocks) and the CLI plugs in the real
/// subprocess stages (`discover::run`, `generate::run`, the transformers trainer,
/// `eval::run_eval`, the GGUF export). Mirrors [`crate::rounds::RoundHooks`] plus an
/// `export` stage (the branch artifact).
pub struct BranchHooks<'a> {
    /// Discover the branch's domain context. Production: `discover::run`.
    pub discover: &'a dyn Fn(&EvolveConfig) -> anyhow::Result<DiscoveredContext>,
    /// Teacher-QA generate over the discovered context. Production: `generate::run`.
    pub generate: &'a dyn Fn(&EvolveConfig, &DiscoveredContext) -> anyhow::Result<Dataset>,
    /// Train the branch on the (probe-carved) training set — the weight-mutating
    /// step. Returns the `gen` provenance of its rows (the quarantine key).
    pub train: &'a dyn Fn(&EvolveConfig, &Dataset) -> anyhow::Result<Vec<String>>,
    /// Score the trained branch against its held-out probe. Production: `eval::run_eval`.
    pub score: &'a dyn Fn(&EvolveConfig) -> anyhow::Result<ScoreReport>,
    /// Export the committed branch to a GGUF at `path`; returns the path written.
    /// Production: merge adapter+base → f16 → quantize (track 27).
    pub export: &'a dyn Fn(&EvolveConfig, &Path) -> anyhow::Result<PathBuf>,
}

/// The result of a `branch create`.
#[derive(Debug, Clone)]
pub struct CreateReport {
    /// The branch name.
    pub name: String,
    /// The transaction action (`None` if it bailed before the transaction).
    pub action: Option<StepAction>,
    /// Whether the branch was admitted to the registry (Accept + roster-admitted).
    pub registered: bool,
    /// The manifest, if the branch was created (Accept).
    pub manifest: Option<BranchManifest>,
    /// Whether the schedule must halt (catastrophe).
    pub halt: bool,
    /// Passages discovered.
    pub passages: usize,
    /// Training rows (after quarantine filtering + probe carve).
    pub rows: usize,
    /// The eval metrics, if it reached eval.
    pub metrics: Option<ScoreReport>,
    /// A human-facing status note.
    pub note: String,
}

/// Create one branch. `base`/`corpus`/`domain` override the per-branch config;
/// `baseline` is the score the trained branch is gated against (typically the base
/// model's score on the branch probe); `created` is the ISO-8601 timestamp stamped
/// into the manifest (passed in for determinism — no wall-clock in the driver).
#[allow(clippy::too_many_arguments)]
pub fn create(
    cfg: &EvolveConfig,
    name: &str,
    base: Option<&str>,
    corpus: Option<&Path>,
    domain: Option<&str>,
    baseline: &ScoreReport,
    created: &str,
    hooks: &BranchHooks,
) -> anyhow::Result<CreateReport> {
    let scoped = scope_config(cfg, name, base, corpus);
    let reg = Regulator::new(&scoped)?;
    let branch_wd = WorkDir::from_config(&scoped);
    branch_wd.ensure()?;

    let bail = |note: String, passages: usize| CreateReport {
        name: name.to_string(),
        action: None,
        registered: false,
        manifest: None,
        halt: false,
        passages,
        rows: 0,
        metrics: None,
        note,
    };

    // (1) Discover the branch's domain context.
    let ctx = match (hooks.discover)(&scoped) {
        Ok(c) => c,
        Err(e) => return Ok(bail(format!("discover failed: {e}"), 0)),
    };
    if ctx.passages.is_empty() {
        return Ok(bail("no passages discovered for branch corpus".into(), 0));
    }
    let passages = ctx.passages.len();

    // (2) Teacher-QA generate, stamp provenance `gen=branch:<name>`, drop quarantined.
    let stamp = format!("branch:{name}");
    let mut dataset = match (hooks.generate)(&scoped, &ctx) {
        Ok(d) => d,
        Err(e) => return Ok(bail(format!("generate failed: {e}"), passages)),
    };
    stamp_gen(&mut dataset, &stamp);
    let quarantine = reg.quarantine()?;
    let (dataset, dropped) = quarantine.filter(&dataset);
    if dropped > 0 {
        eprintln!("branch[{name}]: dropped {dropped} quarantined row(s) before training");
    }
    if dataset.is_empty() {
        return Ok(bail(
            "dataset empty after quarantine filter".into(),
            passages,
        ));
    }

    // The branch's routing descriptor — derived from its corpus passages.
    let router_kind = cfg
        .branch
        .as_ref()
        .and_then(|b| b.router.as_ref())
        .map(|r| r.kind.as_str())
        .unwrap_or("simhash");
    let passage_texts: Vec<String> = ctx.passages.iter().map(|p| p.text.clone()).collect();
    let router_signature = corpus_signature(router_kind, &passage_texts);

    // Carve a held-out probe + training remainder (track 10). With
    // `[eval].stable_probe`, REUSE an existing probe across rounds so the
    // candidate and the stored baseline are scored on the SAME exam (a real
    // cross-round keep|rollback gate); carve a fresh one only on the first round
    // (none exists yet). The default re-carves each round (one-shot `create`).
    let ecfg = scoped.eval.clone().unwrap_or_default();
    let ppath = crate::eval::probe_path(&scoped);
    // Only `train_set` is consumed here; the probe is reloaded from disk by the
    // `score` hook (`eval::run_eval`), so the in-memory handle isn't kept.
    let (_probe, train_set) = if ecfg.stable_probe && ppath.exists() {
        let probe = ProbeSet::load(&ppath)?;
        let train_set = probe.exclude_overlap(&dataset);
        probe.assert_no_overlap(&train_set)?;
        (probe, train_set)
    } else {
        let (probe, train_set) = ProbeSet::carve(&dataset, ecfg.probe_holdout_frac)?;
        let _ = probe.write(&ppath);
        (probe, train_set)
    };
    let _ = train_set.write_jsonl(branch_wd.root().join("dataset.train.jsonl"));
    let rows = train_set.len();

    // (3) The transaction: train (mutates adapter) → eval → keep | rollback.
    let id = format!("branch-{name}");
    let outcome = reg.run_step(
        &id,
        "branch:create",
        0,
        baseline,
        || (hooks.train)(&scoped, &train_set),
        || (hooks.score)(&scoped),
    )?;

    // (4) Only a committed (gate-passed) branch is exported + registered.
    match outcome.action {
        StepAction::Commit => {
            let gguf_target = branch_wd.root().join(format!("{name}.gguf"));
            let gguf_path = (hooks.export)(&scoped, &gguf_target)?;
            let gguf_sha = sha256_file(&gguf_path).unwrap_or_default();

            let base_model = base
                .map(str::to_string)
                .or_else(|| scoped.evolve.model_path.as_deref().map(path_to_string))
                .unwrap_or_else(|| "unknown".to_string());
            let corpus_descriptor = format!(
                "{passages} passages from {}",
                corpus
                    .map(path_to_string)
                    .or_else(|| scoped.evolve.corpus_dir.as_deref().map(path_to_string))
                    .unwrap_or_else(|| "<palace>".to_string())
            );

            let manifest = BranchManifest {
                name: name.to_string(),
                base_model,
                domain: domain.unwrap_or("").to_string(),
                corpus_descriptor,
                router_signature,
                eval_report: score_to_map(outcome.metrics.as_ref()),
                lineage: Lineage::default(),
                version: MANIFEST_VERSION.to_string(),
                gguf_sha,
                created: created.to_string(),
            };
            manifest.write(branch_wd.root().join("manifest.json"))?;

            // Admit into the SHARED fleet registry (bounded; near-dup merges).
            let reg_path = registry_path(cfg);
            let mut registry = BranchRegistry::load(&reg_path)?;
            let max_branches = cfg.branch.as_ref().map(|b| b.max_branches).unwrap_or(16);
            let admitted = admit(
                &mut registry,
                manifest.clone(),
                max_branches,
                MERGE_THRESHOLD,
            );
            let (registered, note) = match &admitted {
                AdmitOutcome::Added => {
                    registry.write(&reg_path)?;
                    (true, "created + registered (eval passed)".to_string())
                }
                AdmitOutcome::Merged { into } => (
                    false,
                    format!("created but merged into near-duplicate branch '{into}' (not a twin)"),
                ),
                AdmitOutcome::Rejected { reason } => {
                    (false, format!("created but NOT registered: {reason}"))
                }
            };

            Ok(CreateReport {
                name: name.to_string(),
                action: Some(StepAction::Commit),
                registered,
                manifest: Some(manifest),
                halt: false,
                passages,
                rows,
                metrics: outcome.metrics,
                note,
            })
        }
        StepAction::Rollback => Ok(CreateReport {
            name: name.to_string(),
            action: Some(StepAction::Rollback),
            registered: false,
            manifest: None,
            halt: false,
            passages,
            rows,
            metrics: outcome.metrics,
            note: "eval gate failed — rolled back, branch NOT registered".to_string(),
        }),
        StepAction::Quarantine => Ok(CreateReport {
            name: name.to_string(),
            action: Some(StepAction::Quarantine),
            registered: false,
            manifest: None,
            halt: true,
            passages,
            rows,
            metrics: outcome.metrics,
            note: format!("CATASTROPHE — rolled back + quarantined ({stamp}) + halt"),
        }),
    }
}

/// The shared fleet registry path: `<top work_dir>/branches/registry.json` (NOT the
/// per-branch scoped work-dir — the registry is the durable fleet record).
pub fn registry_path(cfg: &EvolveConfig) -> PathBuf {
    cfg.work_dir().join("branches").join("registry.json")
}

/// The GGUF artifact path for a branch: `<top work_dir>/branches/<name>/<name>.gguf`.
/// Deterministic so `branch serve` finds what `branch create` wrote.
pub fn gguf_path(cfg: &EvolveConfig, name: &str) -> PathBuf {
    cfg.work_dir()
        .join("branches")
        .join(name)
        .join(format!("{name}.gguf"))
}

/// Build a per-branch config: override `model_path` (the small `base`), `corpus_dir`
/// (the branch's domain slice), and scope `work_dir` to `branches/<name>/` so the
/// branch's checkpoints/adapter/probe isolate from the main run + other branches.
/// Pure (no I/O) — safe to call in a loop.
fn scope_config(
    cfg: &EvolveConfig,
    name: &str,
    base: Option<&str>,
    corpus: Option<&Path>,
) -> EvolveConfig {
    let mut scoped = cfg.clone();
    if let Some(b) = base {
        scoped.evolve.model_path = Some(PathBuf::from(b));
    }
    if let Some(c) = corpus {
        scoped.evolve.corpus_dir = Some(c.to_path_buf());
    }
    let branch_root = cfg.work_dir().join("branches").join(name);
    scoped.evolve.work_dir = Some(branch_root);
    scoped
}

/// Stamp `gen = Some(stamp)` on every dataset row that carries provenance (so
/// track-15 quarantine can isolate this branch's data). Rows with no `gen` field
/// (`Completion`/`Contrastive`) are left untouched.
fn stamp_gen(dataset: &mut Dataset, stamp: &str) {
    for row in &mut dataset.rows {
        match row {
            GenExample::Qa { gen, .. }
            | GenExample::Instruction { gen, .. }
            | GenExample::ToolCall { gen, .. }
            | GenExample::Cli { gen, .. }
            | GenExample::Skill { gen, .. }
            | GenExample::ReasoningEdit { gen, .. } => *gen = Some(stamp.to_string()),
            GenExample::Completion { .. } | GenExample::Contrastive { .. } => {}
        }
    }
}

/// Flatten a `ScoreReport` into the manifest's `eval_report` map (the gates that
/// admitted the branch). Stable key order via the map.
fn score_to_map(report: Option<&ScoreReport>) -> std::collections::BTreeMap<String, f64> {
    let mut m = std::collections::BTreeMap::new();
    if let Some(r) = report {
        m.insert("correctness".to_string(), r.correctness);
        if let Some(v) = r.constitution_adherence {
            m.insert("constitution_adherence".to_string(), v);
        }
        if let Some(v) = r.perplexity {
            m.insert("perplexity".to_string(), v);
        }
        if let Some(v) = r.mean_exit_depth {
            m.insert("mean_exit_depth".to_string(), v);
        }
    }
    m
}

fn path_to_string(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}
