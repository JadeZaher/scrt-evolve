//! Eval harness tests (track 10). ML-free: gate, probe carve, api scorer with a
//! mock transport, StepVerdict, config round-trip.

use std::collections::BTreeSet;

use scrt_evolve::dataset::GenExample;
use scrt_evolve::eval::gate::{ExecutableGate, GateVerdict};
use scrt_evolve::eval::score::{ApiScorer, ScoreReport, Scorer};
use scrt_evolve::eval::verdict::{classify, StepVerdict, VerdictError, VerdictTolerances};
use scrt_evolve::eval::ProbeSet;
use scrt_evolve::generate::api::{ChatMessage, ChatTransport};
use scrt_evolve::{Dataset, EvolveConfig};

// --- A gate built from explicit fixtures (no scrt-core dependency in unit tests) ---

fn fixture_gate() -> ExecutableGate {
    use scrt_evolve::toolspec::ToolSchema;
    let stash = ToolSchema {
        name: "scrt_stash".to_string(),
        description: "stash".to_string(),
        parameters: serde_json::json!({}),
        required: vec!["name".to_string(), "note".to_string()],
        properties: vec![
            "name".to_string(),
            "note".to_string(),
            "tags".to_string(),
            "ttl".to_string(),
        ],
    };
    let flags: BTreeSet<String> = ["--mp-stash", "--mp-ttl", "--in", "--effort"]
        .into_iter()
        .map(String::from)
        .collect();
    ExecutableGate::from_parts(vec![stash], flags)
}

#[test]
fn gate_accepts_valid_tool_call_and_cli() {
    let g = fixture_gate();
    assert!(g
        .check_tool_call(
            "scrt_stash",
            &serde_json::json!({"name": "auth", "note": "x"})
        )
        .is_pass());
    assert!(g
        .check_cli("scrt \"auth\" --mp-stash auth --mp-ttl 4h")
        .is_pass());
    // Bare command with no flags passes.
    assert!(g.check_cli("scrt \"pattern\"").is_pass());
}

#[test]
fn gate_rejects_malformed() {
    let g = fixture_gate();
    // Unknown tool.
    assert!(!g
        .check_tool_call("scrt_nope", &serde_json::json!({"name": "x"}))
        .is_pass());
    // Missing required `note`.
    assert!(matches!(
        g.check_tool_call("scrt_stash", &serde_json::json!({"name": "x"})),
        GateVerdict::Fail(_)
    ));
    // Unknown param.
    assert!(!g
        .check_tool_call(
            "scrt_stash",
            &serde_json::json!({"name": "x", "note": "y", "bogus": 1})
        )
        .is_pass());
    // Not a scrt command.
    assert!(!g.check_cli("ls -la").is_pass());
    // Unknown flag.
    assert!(!g.check_cli("scrt \"x\" --not-a-real-flag").is_pass());
}

#[test]
fn eval_config_round_trips_and_absent_is_none() {
    let toml = r#"
[evolve]
model_path = "/m"

[eval]
probe_path = "probe.jsonl"
probe_holdout_frac = 0.2
scorer_backend = "transformers"
metrics = ["correctness", "perplexity"]
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let e = cfg.eval.as_ref().expect("[evolve.eval]");
    assert_eq!(e.probe_holdout_frac, 0.2);
    assert_eq!(e.scorer_backend, "transformers");
    assert_eq!(
        e.metrics,
        vec!["correctness".to_string(), "perplexity".to_string()]
    );

    // Round-trips.
    let ser = toml::to_string(&cfg).unwrap();
    let back = EvolveConfig::from_toml_str(&ser).unwrap();
    assert_eq!(back.eval.unwrap().scorer_backend, "transformers");

    // Absent ⇒ None (no eval, non-breaking).
    let none = EvolveConfig::from_toml_str("[evolve]\nmodel_path=\"/m\"").unwrap();
    assert!(none.eval.is_none());
}

fn sample_dataset() -> Dataset {
    let rows = (0..20)
        .map(|i| GenExample::Cli {
            prompt: format!("stash item {i}"),
            command: format!("scrt \"item{i}\" --mp-stash item{i}"),
            source: Some("fixture".to_string()),
            gen: Some("test".to_string()),
        })
        .collect();
    Dataset::new(rows)
}

#[test]
fn probe_carve_holds_out_with_zero_overlap_and_is_deterministic() {
    let ds = sample_dataset();
    let (probe, train) = ProbeSet::carve(&ds, 0.25).unwrap();

    assert!(!probe.is_empty(), "probe should get some rows at 25%");
    assert_eq!(
        probe.len() + train.len(),
        ds.len(),
        "carve partitions the rows"
    );
    // Zero overlap is asserted inside carve; re-assert here explicitly.
    probe.assert_no_overlap(&train).unwrap();

    // Deterministic: same input ⇒ identical carve + version.
    let (probe2, _train2) = ProbeSet::carve(&ds, 0.25).unwrap();
    assert_eq!(probe.version, probe2.version);
    assert_eq!(probe.items, probe2.items);
}

#[test]
fn stable_probe_excludes_overlap_to_form_the_train_remainder() {
    // The `[eval].stable_probe` path: carve a fixed probe once, then on a later
    // round filter a fresh dataset against it. The probe rows are removed (never
    // trained on) while the rest survive — the cross-round-stable analogue of carve.
    let ds = sample_dataset();
    let (probe, _train) = ProbeSet::carve(&ds, 0.25).unwrap();
    assert!(!probe.is_empty());

    // Filtering the ORIGINAL dataset against the fixed probe drops exactly the
    // probe rows, and the result has zero overlap with the probe.
    let remainder = probe.exclude_overlap(&ds);
    assert_eq!(remainder.len(), ds.len() - probe.len());
    probe.assert_no_overlap(&remainder).unwrap();

    // A disjoint fresh dataset (next round's generation) passes through whole.
    let fresh = Dataset::new(
        (100..105)
            .map(|i| GenExample::Cli {
                prompt: format!("fresh {i}"),
                command: format!("scrt \"f{i}\" --mp-stash f{i}"),
                source: Some("fixture".to_string()),
                gen: Some("test".to_string()),
            })
            .collect(),
    );
    assert_eq!(probe.exclude_overlap(&fresh).len(), fresh.len());
}

#[test]
fn probe_overlap_is_detected() {
    let ds = sample_dataset();
    let (probe, _train) = ProbeSet::carve(&ds, 0.25).unwrap();
    // Asserting overlap against the FULL dataset (which contains the probe rows)
    // must fail.
    let err = probe.assert_no_overlap(&ds).unwrap_err();
    assert!(err.to_string().contains("overlap"));
}

#[test]
fn probe_round_trips_jsonl() {
    let ds = sample_dataset();
    let (probe, _) = ProbeSet::carve(&ds, 0.3).unwrap();
    let mut p = std::env::temp_dir();
    p.push(format!("scrt-evolve-probe-{}.jsonl", std::process::id()));
    probe.write(&p).unwrap();
    let back = ProbeSet::load(&p).unwrap();
    assert_eq!(
        back.version, probe.version,
        "loaded probe recomputes same version"
    );
    assert_eq!(back.items, probe.items);
    let _ = std::fs::remove_file(&p);
}

// --- A mock transport that always emits a valid scrt CLI command ---

struct AlwaysValidCli;
impl ChatTransport for AlwaysValidCli {
    fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("scrt \"x\" --mp-stash x --mp-ttl 4h".to_string())
    }
}

struct AlwaysBogus;
impl ChatTransport for AlwaysBogus {
    fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("rm -rf / --definitely-not-a-flag".to_string())
    }
}

#[test]
fn api_scorer_scores_correctness_with_mock_model() {
    let ds = sample_dataset();
    let (probe, _) = ProbeSet::carve(&ds, 0.5).unwrap();
    assert!(!probe.is_empty());

    // A model that always emits a valid command scores ~100% on cli probes.
    let good = ApiScorer::new(AlwaysValidCli, fixture_gate());
    let report = good.score(&probe).unwrap();
    assert_eq!(report.backend, "api");
    assert_eq!(report.probe_version, probe.version);
    assert_eq!(report.n, probe.len());
    assert!(
        (report.correctness - 1.0).abs() < 1e-9,
        "all-valid model should score 1.0, got {}",
        report.correctness
    );
    // api backend computes no forward-pass metrics.
    assert!(report.perplexity.is_none());
    assert!(report.mean_exit_depth.is_none());

    // A model that always emits garbage scores 0.
    let bad = ApiScorer::new(AlwaysBogus, fixture_gate());
    assert_eq!(bad.score(&probe).unwrap().correctness, 0.0);
}

#[test]
fn api_scorer_empty_probe_is_uncovered() {
    let probe = ProbeSet::from_items(vec![]);
    let scorer = ApiScorer::new(AlwaysValidCli, fixture_gate());
    let report = scorer.score(&probe).unwrap();
    assert_eq!(report.n, 0);
    assert_eq!(report.correctness, 0.0);
}

// --- StepVerdict ---

fn report(correctness: f64, version: &str) -> ScoreReport {
    ScoreReport {
        correctness,
        constitution_adherence: None,
        mean_exit_depth: None,
        perplexity: None,
        n: 10,
        probe_version: version.to_string(),
        backend: "api".to_string(),
    }
}

#[test]
fn verdict_classifies_three_outcomes() {
    let tol = VerdictTolerances::default(); // tolerance 0.02, floor 0.10
    let base = report(0.80, "v1");

    // Improvement ⇒ accept.
    assert_eq!(
        classify(&base, &report(0.85, "v1"), &tol).unwrap(),
        StepVerdict::Accept
    );
    // Within tolerance drop ⇒ accept.
    assert_eq!(
        classify(&base, &report(0.79, "v1"), &tol).unwrap(),
        StepVerdict::Accept
    );
    // Beyond tolerance ⇒ regress.
    assert_eq!(
        classify(&base, &report(0.70, "v1"), &tol).unwrap(),
        StepVerdict::Regress
    );
    // Below catastrophe floor ⇒ catastrophic.
    assert_eq!(
        classify(&base, &report(0.05, "v1"), &tol).unwrap(),
        StepVerdict::Catastrophic
    );
    // NaN ⇒ catastrophic.
    assert_eq!(
        classify(&base, &report(f64::NAN, "v1"), &tol).unwrap(),
        StepVerdict::Catastrophic
    );
}

#[test]
fn verdict_refuses_probe_version_mismatch() {
    let tol = VerdictTolerances::default();
    let err = classify(&report(0.8, "v1"), &report(0.9, "v2"), &tol).unwrap_err();
    assert!(matches!(err, VerdictError::ProbeVersionMismatch { .. }));
}
