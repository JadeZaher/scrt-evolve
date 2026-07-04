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
use crate::judge::{dataset_signal_stats, dataset_tier};
use crate::regulate::{Regulator, StepAction};
use crate::workdir::WorkDir;

use super::manifest::{sha256_file, BranchManifest, BranchRegistry, Lineage, MANIFEST_VERSION};
use super::router::{admit, corpus_signature, AdmitOutcome};

// ── Servability preflight (PC-4) ─────────────────────────────────────────────
//
// Source of truth: `model.rs::train_impl::arch_supported` — kept in sync
// manually; if that list grows, update this mirror too.
// NOTE: `arch_supported` is `fn`-private inside `#[cfg(feature="train")]` so
// it is NOT reachable from this crate-level module.  When model.rs makes
// `arch_supported` pub (or exposes a pub `arch_is_servable`), replace this
// mirror with a direct call.
//
// Supported (llama-family + fixture, Phase A):
//   LlamaForCausalLM  — all llama/mistral/vicuna/phi checkpoints
//   LlamaModel        — HF variant without the CausalLM head wrapper
//   ScrtEvolveTinyCausalLM — random-fixture model (CI/tests)
//
// Refused (Phase B / never):
//   GraniteForCausalLM / GraniteModel — IBM Granite-4-h; not yet wired
//     (Phase B target: add Granite seam to arch.rs).
//   MambaForCausalLM / MixedMambaForCausalLM — pure-Mamba / MoE-Mamba;
//     state-space kernels incompatible with the current attention-only seam;
//     refused indefinitely until a Mamba ArchAdapter is added.
//   MixtralForCausalLM / PhiMoEForCausalLM — pure-MoE routing; no MoE seam.
//   (anything else) — unknown; refused until an explicit seam exists.

fn arch_is_servable(arch: &str) -> bool {
    matches!(
        arch,
        "LlamaForCausalLM" | "LlamaModel" | "ScrtEvolveTinyCausalLM"
    )
}

/// Why a given `architectures` string is refused — surfaced verbatim in the
/// doctor-style error message.
fn arch_refuse_reason(arch: &str) -> &'static str {
    match arch {
        "GraniteForCausalLM" | "GraniteModel" => {
            "IBM Granite is not yet wired to the native inference seam \
(Phase B target: add Granite ArchAdapter to arch.rs)"
        }
        "MambaForCausalLM" | "MixedMambaForCausalLM" => {
            "pure-Mamba / Mamba-MoE architectures use state-space kernels that \
are incompatible with the current attention-only ArchAdapter; \
refused until a Mamba seam is added"
        }
        "MixtralForCausalLM" | "PhiMoEForCausalLM" => {
            "sparse MoE routing is not supported by the current linear-layer ArchAdapter; \
refused until a MoE seam is added"
        }
        _ => "architecture has no registered native-inference seam; \
add an ArchAdapter in arch.rs before creating branches on this model",
    }
}

/// Read `<model_dir>/config.json` and confirm every listed architecture is
/// servable by native candle inference.  Returns `Ok(())` on success or an
/// `Err` with a doctor-style message that names the arch and why it is refused.
/// If `model_path` is `None` (no base model configured) the check is skipped
/// — the hook pipeline will fail later with a more specific message.
fn preflight_arch(model_path: Option<&std::path::Path>) -> anyhow::Result<()> {
    let mpath = match model_path {
        Some(p) => p,
        None => return Ok(()),
    };
    let config_json = mpath.join("config.json");
    if !config_json.exists() {
        // No config.json on disk (e.g. fixture path) — silently skip; the
        // loader will surface the missing-file error at a later stage.
        return Ok(());
    }
    let raw = std::fs::read_to_string(&config_json)
        .map_err(|e| anyhow::anyhow!("preflight: could not read {}: {e}", config_json.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("preflight: could not parse {}: {e}", config_json.display()))?;

    if let Some(archs) = v.get("architectures").and_then(|a| a.as_array()) {
        for arch_val in archs {
            if let Some(arch) = arch_val.as_str() {
                if !arch_is_servable(arch) {
                    let reason = arch_refuse_reason(arch);
                    return Err(anyhow::anyhow!(
                        "[preflight] branch creation refused — \
architecture `{arch}` is not servable by native inference.\n\
  reason : {reason}\n\
  model  : {}\n\
  fix    : choose a llama-family model (LlamaForCausalLM / LlamaModel) \
or wait for the Phase B seam for this architecture.",
                        mpath.display()
                    ));
                }
            }
        }
    }
    Ok(())
}

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

    // PC-4: trainable ⇒ servable invariant — refuse early if the base model's
    // architecture cannot be served by native candle inference.
    preflight_arch(scoped.evolve.model_path.as_deref())?;

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

            let mut eval_report = score_to_map(outcome.metrics.as_ref());
            eval_report.extend(dataset_signal_stats(&dataset.rows)); // track 37: signal stats
            let corpus_tier = dataset_tier(&dataset.rows); // track 37: data-sovereignty tier
            let manifest = BranchManifest {
                name: name.to_string(),
                base_model,
                domain: domain.unwrap_or("").to_string(),
                corpus_descriptor,
                router_signature,
                eval_report,
                lineage: Lineage::default(),
                version: MANIFEST_VERSION.to_string(),
                gguf_sha,
                created: created.to_string(),
                tier: corpus_tier,
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

#[cfg(test)]
mod preflight_tests {
    use super::{arch_is_servable, arch_refuse_reason, preflight_arch};
    use std::io::Write;

    fn tmp_model_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("scrt_evolve_pc4_tests")
            .join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &std::path::Path, archs: &[&str]) {
        let arch_strs: Vec<String> = archs.iter().map(|a| format!("\"{a}\"")).collect();
        let json = format!("{{\"architectures\": [{}]}}", arch_strs.join(", "));
        let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();
    }

    // ── arch_is_servable unit tests ────────────────────────────────────

    #[test]
    fn llama_family_is_servable() {
        assert!(arch_is_servable("LlamaForCausalLM"));
        assert!(arch_is_servable("LlamaModel"));
        assert!(arch_is_servable("ScrtEvolveTinyCausalLM"));
    }

    #[test]
    fn unsupported_arches_refused() {
        for arch in &[
            "GraniteForCausalLM",
            "GraniteModel",
            "MambaForCausalLM",
            "MixedMambaForCausalLM",
            "MixtralForCausalLM",
            "PhiMoEForCausalLM",
            "FalconForCausalLM",
            "SomeNewModel",
        ] {
            assert!(!arch_is_servable(arch), "{arch} should be refused");
        }
    }

    #[test]
    fn refuse_reason_granite_mentions_phase_b() {
        let reason = arch_refuse_reason("GraniteForCausalLM");
        assert!(
            reason.contains("Phase B"),
            "Granite reason should mention Phase B: {reason}"
        );
    }

    #[test]
    fn refuse_reason_mamba_mentions_state_space() {
        let reason = arch_refuse_reason("MambaForCausalLM");
        assert!(
            reason.contains("state-space"),
            "Mamba reason should mention state-space: {reason}"
        );
    }

    // ── preflight_arch integration tests (real config.json on tmp dir) ────────

    #[test]
    fn preflight_passes_for_llama() {
        let dir = tmp_model_dir("llama");
        write_config(&dir, &["LlamaForCausalLM"]);
        preflight_arch(Some(&dir)).expect("LlamaForCausalLM should pass preflight");
    }

    #[test]
    fn preflight_refused_for_granite() {
        let dir = tmp_model_dir("granite");
        write_config(&dir, &["GraniteForCausalLM"]);
        let err = preflight_arch(Some(&dir))
            .expect_err("GraniteForCausalLM should be refused");
        let msg = err.to_string();
        assert!(msg.contains("GraniteForCausalLM"), "error should name the arch: {msg}");
        assert!(msg.contains("Phase B"), "error should mention Phase B: {msg}");
        assert!(msg.contains("[preflight]"), "error should be tagged preflight: {msg}");
    }

    #[test]
    fn preflight_refused_for_mamba() {
        let dir = tmp_model_dir("mamba");
        write_config(&dir, &["MambaForCausalLM"]);
        let err = preflight_arch(Some(&dir))
            .expect_err("MambaForCausalLM should be refused");
        let msg = err.to_string();
        assert!(msg.contains("MambaForCausalLM"), "error should name the arch: {msg}");
        assert!(msg.contains("state-space"), "error body should explain why: {msg}");
    }

    #[test]
    fn preflight_refused_for_moe() {
        let dir = tmp_model_dir("moe");
        write_config(&dir, &["MixtralForCausalLM"]);
        let err = preflight_arch(Some(&dir))
            .expect_err("MixtralForCausalLM should be refused");
        let msg = err.to_string();
        assert!(msg.contains("MixtralForCausalLM"), "{msg}");
        assert!(msg.contains("MoE") || msg.contains("routing"), "{msg}");
    }

    #[test]
    fn preflight_skips_missing_config_json() {
        // A directory with no config.json (e.g. fixture path) — silently skip.
        let dir = tmp_model_dir("no_config");
        // Remove config.json if a previous run wrote one.
        let _ = std::fs::remove_file(dir.join("config.json"));
        preflight_arch(Some(&dir)).expect("missing config.json should be skipped");
    }

    #[test]
    fn preflight_skips_none_model_path() {
        // No model configured at all — skip preflight.
        preflight_arch(None).expect("None model_path should be skipped");
    }
}
