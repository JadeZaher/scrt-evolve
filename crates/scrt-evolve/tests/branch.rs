//! Branch factory (track 29) — Phase 0 (config + manifest/registry) and Phase 1
//! (router + signature + bounded-fleet) tests. All ML-free.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use scrt_evolve::branch::router::{admit, corpus_signature, AdmitOutcome};
use scrt_evolve::branch::{
    self, BranchHooks, BranchManifest, BranchRegistry, Lineage, RegistryError,
};
use scrt_evolve::config::BranchRouterConfig;
use scrt_evolve::dataset::{Dataset, GenExample};
use scrt_evolve::discover::{DiscoveredContext, Passage};
use scrt_evolve::eval::ScoreReport;
use scrt_evolve::regulate::StepAction;
use scrt_evolve::{BranchRouter, EvolveConfig, LocalBranchRouter, Quarantine};

fn temp_dir(suffix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "scrt-evolve-branch-{}-{suffix}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn manifest(name: &str, domain_texts: &[&str]) -> BranchManifest {
    let texts: Vec<String> = domain_texts.iter().map(|s| s.to_string()).collect();
    let mut eval = BTreeMap::new();
    eval.insert("correctness".to_string(), 0.9);
    BranchManifest {
        name: name.to_string(),
        base_model: "granite-eval-0.5b".to_string(),
        domain: format!("{name}/domain"),
        corpus_descriptor: format!("{} passages", domain_texts.len()),
        router_signature: corpus_signature("simhash", &texts),
        eval_report: eval,
        lineage: Lineage::default(),
        version: scrt_evolve::branch::MANIFEST_VERSION.to_string(),
        gguf_sha: scrt_evolve::branch::sha256_hex(name.as_bytes()),
        created: "2026-06-26T00:00:00Z".to_string(),
    }
}

// ─────────────────────────── Phase 0: config ───────────────────────────

#[test]
fn branch_config_round_trips_with_defaults() {
    let toml = r#"
[evolve]
model_path = "/models/base"

[branch]
base = "granite-eval-0.5b"
domain = "legal/tool-calling"
  [branch.router]
  kind = "simhash"
  [branch.ensemble]
  mode = "single_best"
"#;
    let cfg: EvolveConfig = toml::from_str(toml).unwrap();
    let b = cfg.branch.clone().expect("[branch] should parse");
    assert!(b.enabled, "enabled defaults true");
    assert_eq!(b.objective, "end_task", "objective default");
    assert_eq!(b.max_branches, 16, "max_branches default");
    let r = b.router.expect("[branch.router]");
    assert_eq!(r.kind, "simhash");
    assert_eq!(r.top_k, 1, "router.top_k default");
    assert!((r.confidence_floor - 0.5).abs() < f32::EPSILON);
    assert_eq!(b.ensemble.unwrap().mode, "single_best");

    // Serialize → reparse is stable.
    let out = toml::to_string(&cfg).unwrap();
    let reparsed: EvolveConfig = toml::from_str(&out).unwrap();
    assert_eq!(reparsed.branch.unwrap().objective, "end_task");
}

#[test]
fn distill_config_and_branch_mode_round_trip() {
    // `[train.distill]` + `[branch].mode = "distill"` parse with the right
    // defaults and round-trip; absent ⇒ today's behavior (covered elsewhere).
    let toml = r#"
[evolve]
model_path = "/models/tinyllama"

[train.distill]
teacher_model = "/models/Mistral-7B"

[train.fractional]
block_size = 2

[branch]
base = "tinyllama-1.1b"
mode = "distill"
"#;
    let cfg: EvolveConfig = toml::from_str(toml).unwrap();
    let d = cfg
        .train
        .as_ref()
        .and_then(|t| t.distill.clone())
        .expect("[train.distill] should parse");
    assert!(d.enabled, "distill.enabled defaults true");
    assert_eq!(d.teacher_model.as_deref(), Some("/models/Mistral-7B"));
    assert_eq!(d.layer_map, "stride", "layer_map default");
    assert_eq!(d.loss, "cosine_mse", "loss default");
    assert_eq!(d.projection, "auto", "projection default");
    assert!(d.teacher_cache.is_none());

    let b = cfg.branch.clone().expect("[branch]");
    assert_eq!(b.mode, "distill", "branch mode parsed");

    // Serialize → reparse is stable.
    let out = toml::to_string(&cfg).unwrap();
    let reparsed: EvolveConfig = toml::from_str(&out).unwrap();
    assert_eq!(
        reparsed
            .train
            .unwrap()
            .distill
            .unwrap()
            .teacher_model
            .as_deref(),
        Some("/models/Mistral-7B")
    );
    assert_eq!(reparsed.branch.unwrap().mode, "distill");
}

#[test]
fn branch_mode_defaults_to_standard() {
    let toml = r#"
[evolve]
model_path = "/m"
[branch]
base = "b"
"#;
    let cfg: EvolveConfig = toml::from_str(toml).unwrap();
    assert_eq!(cfg.branch.unwrap().mode, "standard", "mode default");
    // No [train.distill] ⇒ None (back-compat).
    assert!(cfg.train.is_none() || cfg.train.unwrap().distill.is_none());
}

#[test]
fn evolve_config_without_branch_unchanged() {
    // A config with no [branch] parses and `branch` is None — today's behavior.
    let toml = r#"
[evolve]
model_path = "/models/base"
corpus_dir = "./src"
"#;
    let cfg: EvolveConfig = toml::from_str(toml).unwrap();
    assert!(cfg.branch.is_none(), "absent [branch] ⇒ None (back-compat)");
    // Round-trips without injecting a [branch] section.
    let out = toml::to_string(&cfg).unwrap();
    assert!(!out.contains("[branch]"), "absent branch not serialized");
}

// ───────────────────── Phase 0: manifest + registry ─────────────────────

#[test]
fn manifest_round_trips() {
    let m = manifest("legal-tools", &["contract law clause", "tort liability"]);
    let json = m.to_json().unwrap();
    let back = BranchManifest::from_json(&json).unwrap();
    assert_eq!(m, back);
    // Contract field names are present (the hivemind schema, §3a).
    for field in [
        "base_model",
        "router_signature",
        "eval_report",
        "lineage",
        "gguf_sha",
        "created",
    ] {
        assert!(
            json.contains(field),
            "manifest missing contract field {field}"
        );
    }
}

#[test]
fn registry_round_trips_and_writes_atomically() {
    let dir = temp_dir("registry");
    let path = dir.join("branches").join("registry.json");

    let mut reg = BranchRegistry::empty();
    reg.upsert(manifest("a", &["alpha beta gamma"]));
    reg.upsert(manifest("b", &["delta epsilon zeta"]));
    reg.write(&path).unwrap();

    let loaded = BranchRegistry::load(&path).unwrap();
    assert_eq!(loaded, reg);
    assert_eq!(loaded.branches.len(), 2);
    assert!(loaded.get("a").is_some());

    // Atomic write leaves no `.tmp` sibling behind (§2.3).
    let leftover = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| e.file_name().to_string_lossy().contains(".tmp"));
    assert!(!leftover, "no temp file should remain after atomic write");
}

#[test]
fn registry_upsert_is_idempotent_by_name() {
    let mut reg = BranchRegistry::empty();
    assert!(
        !reg.upsert(manifest("a", &["x y z"])),
        "first insert appends"
    );
    assert!(reg.upsert(manifest("a", &["p q r"])), "same name replaces");
    assert_eq!(reg.branches.len(), 1, "no duplicate name");
}

#[test]
fn missing_registry_is_empty_not_error() {
    let dir = temp_dir("missing");
    let reg = BranchRegistry::load(dir.join("nope.json")).unwrap();
    assert!(reg.branches.is_empty());
}

#[test]
fn schema_version_mismatch_refused() {
    let dir = temp_dir("schema");
    let path = dir.join("registry.json");
    std::fs::write(&path, r#"{"schema_version": 999, "branches": []}"#).unwrap();
    let err = BranchRegistry::load(&path).unwrap_err();
    assert!(
        matches!(err, RegistryError::SchemaMismatch { found: 999, .. }),
        "expected schema mismatch, got {err:?}"
    );
}

// ─────────────────────────── Phase 1: router ───────────────────────────

#[test]
fn signature_is_stable_and_discriminates_domains() {
    let legal: Vec<String> = ["contract law clause arbitration", "tort liability damages"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cooking: Vec<String> = ["saute the onions", "preheat the oven to bake"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let s1 = corpus_signature("simhash", &legal);
    let s2 = corpus_signature("simhash", &legal);
    assert_eq!(s1, s2, "signature is deterministic");
    assert_eq!(s1.vector.len(), 64, "simhash ⇒ 64-dim vector");

    let s3 = corpus_signature("simhash", &cooking);
    assert_ne!(
        s1.vector, s3.vector,
        "distinct domains ⇒ distinct signatures"
    );
}

#[test]
fn router_resolves_matching_query_and_floors_low_confidence() {
    let mut reg = BranchRegistry::empty();
    reg.upsert(manifest(
        "legal",
        &["contract law clause arbitration tort liability damages statute"],
    ));
    reg.upsert(manifest(
        "cooking",
        &["saute onions preheat oven bake simmer roast garlic"],
    ));

    let cfg = BranchRouterConfig {
        kind: "simhash".to_string(),
        confidence_floor: 0.6,
        top_k: 1,
    };
    let router = LocalBranchRouter::new(&reg, &cfg);

    // A query in the legal domain resolves to the legal branch.
    let hits = router.resolve("what statute governs this contract arbitration clause");
    assert_eq!(hits.len(), 1, "top_k=1");
    assert_eq!(hits[0].0.name, "legal", "matched the right branch");

    // A gibberish query below the floor resolves to base-only (empty).
    let hi = BranchRouterConfig {
        kind: "simhash".to_string(),
        confidence_floor: 0.99,
        top_k: 1,
    };
    let strict = LocalBranchRouter::new(&reg, &hi);
    assert!(
        strict.resolve("zzzzz qqqqq").is_empty(),
        "low confidence ⇒ base-only"
    );
}

#[test]
fn router_off_and_empty_registry_are_base_only() {
    let empty = BranchRegistry::empty();
    let cfg = BranchRouterConfig::default();
    let router = LocalBranchRouter::new(&empty, &cfg);
    assert!(
        router.resolve("anything").is_empty(),
        "empty registry ⇒ base-only"
    );

    let off = LocalBranchRouter::off();
    assert!(off.resolve("anything").is_empty(), "router=off ⇒ base-only");
}

// ───────────────── Phase 1: bounded fleet (merge + cap) ─────────────────

#[test]
fn near_duplicate_branches_merge_not_twin() {
    let mut reg = BranchRegistry::empty();
    let a = manifest(
        "legal-a",
        &["contract law clause arbitration tort liability"],
    );
    assert_eq!(admit(&mut reg, a, 16, 0.85), AdmitOutcome::Added);

    // A near-identical domain under a different name must MERGE, not spawn a twin.
    let twin = manifest(
        "legal-b",
        &["contract law clause arbitration tort liability"],
    );
    let outcome = admit(&mut reg, twin, 16, 0.85);
    assert!(
        matches!(outcome, AdmitOutcome::Merged { ref into } if into == "legal-a"),
        "near-dup should merge into legal-a, got {outcome:?}"
    );
    assert_eq!(reg.branches.len(), 1, "two near-dup domains ⇒ one branch");
}

#[test]
fn max_branches_cap_rejects_novel_overflow() {
    let mut reg = BranchRegistry::empty();
    assert_eq!(
        admit(
            &mut reg,
            manifest("a", &["alpha beta gamma delta"]),
            1,
            0.85
        ),
        AdmitOutcome::Added
    );
    // A novel domain past the cap is rejected.
    let outcome = admit(
        &mut reg,
        manifest("b", &["xi omicron pi rho sigma"]),
        1,
        0.85,
    );
    assert!(
        matches!(outcome, AdmitOutcome::Rejected { .. }),
        "novel branch past cap should be rejected, got {outcome:?}"
    );
    assert_eq!(reg.branches.len(), 1, "cap holds");
}

#[test]
fn same_name_readmit_replaces_in_place() {
    let mut reg = BranchRegistry::empty();
    assert_eq!(
        admit(&mut reg, manifest("a", &["alpha beta"]), 1, 0.85),
        AdmitOutcome::Added
    );
    // Re-admitting the SAME name is an update even at the cap (not a twin/overflow).
    assert_eq!(
        admit(&mut reg, manifest("a", &["alpha beta gamma"]), 1, 0.85),
        AdmitOutcome::Added
    );
    assert_eq!(reg.branches.len(), 1);
}

// ─────────── Phase 4: cross-repo schema contract (SCRT-EVOLVE-INTEGRATION.md) ───────────

#[test]
fn manifest_matches_hivemind_contract_schema() {
    // The exact field set hivemind reads (SCRT-EVOLVE-INTEGRATION.md §3a). A drift
    // here is a coordinated cross-repo change — this test is the guardrail.
    let m = manifest("legal-tools", &["contract law clause"]);
    let v: serde_json::Value = serde_json::from_str(&m.to_json().unwrap()).unwrap();
    let obj = v.as_object().unwrap();

    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "base_model",
            "corpus_descriptor",
            "created",
            "domain",
            "eval_report",
            "gguf_sha",
            "lineage",
            "name",
            "router_signature",
            "version",
        ],
        "manifest top-level keys must match the hivemind contract exactly"
    );
    // router_signature = { kind, vector } (the routing descriptor hivemind matches).
    let sig = obj["router_signature"].as_object().unwrap();
    assert!(sig.contains_key("kind") && sig.contains_key("vector"));
    assert!(obj["router_signature"]["vector"].is_array());
    // lineage = { parent? }.
    assert!(obj["lineage"].is_object());
}

#[test]
fn registry_matches_contract_schema() {
    // §3b: `{ schema_version, branches: [...] }`.
    let mut reg = BranchRegistry::empty();
    reg.upsert(manifest("a", &["alpha beta"]));
    let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&reg).unwrap()).unwrap();
    let obj = v.as_object().unwrap();
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(keys, vec!["branches", "schema_version"]);
    assert_eq!(obj["schema_version"].as_u64(), Some(1));
    assert!(obj["branches"].is_array());
}

// ──────────────── Phase 2: create pipeline (composed + txn) ────────────────

fn temp_branch_cfg(tag: &str) -> (PathBuf, EvolveConfig) {
    let mut dir = std::env::temp_dir();
    dir.push(format!("scrt-evolve-create-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let toml = format!(
        r#"
[evolve]
model_path = "/m"
work_dir = {dir:?}

[regulate]
enabled = true
accept_tolerance = 0.02
catastrophe_floor = 0.10

[branch]
max_branches = 16
  [branch.router]
  kind = "simhash"
"#
    );
    let cfg = EvolveConfig::from_toml_str(&toml).unwrap();
    (dir, cfg)
}

fn report(correctness: f64) -> ScoreReport {
    ScoreReport {
        correctness,
        constitution_adherence: None,
        mean_exit_depth: None,
        perplexity: None,
        n: 10,
        probe_version: "v1".to_string(),
        backend: "test".to_string(),
    }
}

fn ctx(texts: &[&str]) -> DiscoveredContext {
    DiscoveredContext {
        passages: texts
            .iter()
            .enumerate()
            .map(|(i, t)| Passage {
                text: t.to_string(),
                source: format!("file{i}.rs"),
                score: 1.0,
                seed: "corpus:domain".to_string(),
            })
            .collect(),
        anchors: Vec::new(),
    }
}

fn qa(n: usize) -> Dataset {
    Dataset::new(
        (0..n)
            .map(|i| GenExample::Qa {
                prompt: format!("question {i} about contract law clause"),
                completion: format!("answer {i}"),
                source: None,
                gen: None,
            })
            .collect(),
    )
}

/// Hooks for a successful create: discover→generate→train→score(passing)→export.
fn good_export(_: &EvolveConfig, path: &Path) -> anyhow::Result<PathBuf> {
    std::fs::write(path, b"GGUF\x00fake-quantized-weights")?;
    Ok(path.to_path_buf())
}

#[test]
fn create_yields_gguf_manifest_registry_and_stamps_provenance() {
    let (_dir, cfg) = temp_branch_cfg("ok");

    let discover = |_: &EvolveConfig| Ok(ctx(&["contract law clause", "tort liability damages"]));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(qa(8));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["branch:legal".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.90)); // improves on baseline 0.80
    let hooks = BranchHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
        export: &good_export,
    };

    let out = branch::create(
        &cfg,
        "legal",
        Some("granite-eval-0.5b"),
        None,
        Some("legal/tool-calling"),
        &report(0.80),
        "2026-06-26T00:00:00Z",
        &hooks,
    )
    .unwrap();

    assert_eq!(out.action, Some(StepAction::Commit));
    assert!(out.registered, "passing branch is registered");
    let m = out.manifest.expect("manifest on accept");
    assert_eq!(m.name, "legal");
    assert_eq!(m.domain, "legal/tool-calling");
    assert_eq!(m.base_model, "granite-eval-0.5b");
    assert!(!m.gguf_sha.is_empty(), "gguf content-addressed");
    assert!(m.eval_report.contains_key("correctness"));

    // GGUF artifact written under the per-branch dir.
    let gguf = cfg
        .work_dir()
        .join("branches")
        .join("legal")
        .join("legal.gguf");
    assert!(gguf.exists(), "gguf artifact written");

    // Registered in the shared fleet registry.
    let reg = BranchRegistry::load(cfg.work_dir().join("branches").join("registry.json")).unwrap();
    assert!(reg.get("legal").is_some(), "branch in registry");

    // Provenance: the training rows carry gen=branch:legal (quarantine key).
    let train_jsonl = cfg
        .work_dir()
        .join("branches")
        .join("legal")
        .join("dataset.train.jsonl");
    let ds = Dataset::read_jsonl(&train_jsonl).unwrap();
    assert!(!ds.rows.is_empty());
    assert!(
        ds.rows
            .iter()
            .all(|r| matches!(r, GenExample::Qa { gen: Some(g), .. } if g == "branch:legal")),
        "all rows stamped gen=branch:legal"
    );
}

#[test]
fn eval_fail_rolls_back_and_registry_unchanged() {
    let (_dir, cfg) = temp_branch_cfg("regress");

    let discover = |_: &EvolveConfig| Ok(ctx(&["contract law clause", "tort liability"]));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(qa(8));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["branch:legal".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.50)); // big drop, above catastrophe floor ⇒ regress
    let export_should_not_run = |_: &EvolveConfig, _: &Path| -> anyhow::Result<PathBuf> {
        panic!("export must not run on regress")
    };
    let hooks = BranchHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
        export: &export_should_not_run,
    };

    let out = branch::create(
        &cfg,
        "legal",
        Some("granite-eval-0.5b"),
        None,
        Some("legal"),
        &report(0.80),
        "2026-06-26T00:00:00Z",
        &hooks,
    )
    .unwrap();

    assert_eq!(out.action, Some(StepAction::Rollback));
    assert!(!out.registered, "regressed branch is NOT registered");
    assert!(out.manifest.is_none());
    assert!(!out.halt, "soft regress does not halt");

    // The shared registry has no entry (unchanged).
    let reg = BranchRegistry::load(cfg.work_dir().join("branches").join("registry.json")).unwrap();
    assert!(reg.get("legal").is_none(), "registry unchanged on regress");
    // No GGUF produced.
    let gguf = cfg
        .work_dir()
        .join("branches")
        .join("legal")
        .join("legal.gguf");
    assert!(!gguf.exists(), "no artifact on regress");
}

#[test]
fn catastrophe_quarantines_branch_provenance_and_halts() {
    let (_dir, cfg) = temp_branch_cfg("catastrophe");

    let discover = |_: &EvolveConfig| Ok(ctx(&["contract law clause", "tort liability"]));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(qa(8));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["branch:dangerous".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.02)); // below catastrophe floor (0.10)
    let export_unused = |_: &EvolveConfig, _: &Path| -> anyhow::Result<PathBuf> {
        panic!("no export on catastrophe")
    };
    let hooks = BranchHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
        export: &export_unused,
    };

    let out = branch::create(
        &cfg,
        "dangerous",
        Some("granite-eval-0.5b"),
        None,
        Some("danger"),
        &report(0.80),
        "2026-06-26T00:00:00Z",
        &hooks,
    )
    .unwrap();

    assert_eq!(out.action, Some(StepAction::Quarantine));
    assert!(out.halt, "catastrophe halts");
    assert!(!out.registered);

    // The branch's provenance is quarantined (so a later run skips its data).
    let q = Quarantine::load(
        cfg.work_dir()
            .join("branches")
            .join("dangerous")
            .join("quarantine.json"),
    )
    .unwrap();
    assert!(
        q.contains("branch:dangerous"),
        "catastrophic branch provenance quarantined"
    );
}
