//! Self-regulation tests (track 15). ML-free: the transaction machinery
//! (snapshot/eval/commit/rollback/quarantine/log/halt) is proven with injected
//! step + score closures — no model, no Python.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use scrt_evolve::dataset::{GenExample, Outcome, Tier, Verdict};
use scrt_evolve::eval::ScoreReport;
use scrt_evolve::regulate::{Regulator, StepAction};
use scrt_evolve::{Dataset, EvolveConfig, Quarantine};

/// A temp work-dir + a config rooted at it, with regulate enabled.
fn temp_cfg(tag: &str) -> (PathBuf, EvolveConfig) {
    let mut dir = std::env::temp_dir();
    dir.push(format!("scrt-evolve-reg-{tag}-{}", std::process::id()));
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
"#,
    );
    let cfg = EvolveConfig::from_toml_str(&toml).unwrap();
    (dir, cfg)
}

/// Write a fake adapter dir with some content (stands in for adapter.safetensors).
fn write_adapter(work: &Path, marker: &str) {
    let adapter = work.join("adapter");
    std::fs::create_dir_all(&adapter).unwrap();
    std::fs::write(adapter.join("adapter.safetensors"), marker).unwrap();
    std::fs::write(
        adapter.join("adapter_config.json"),
        r#"{"rank":16,"alpha":32,"target_modules":["q_proj"]}"#,
    )
    .unwrap();
}

fn adapter_marker(work: &Path) -> String {
    std::fs::read_to_string(work.join("adapter").join("adapter.safetensors")).unwrap()
}

fn report(correctness: f64, version: &str) -> ScoreReport {
    ScoreReport {
        correctness,
        constitution_adherence: None,
        mean_exit_depth: None,
        perplexity: None,
        n: 10,
        probe_version: version.to_string(),
        backend: "test".to_string(),
    }
}

#[test]
fn accept_commits_and_advances_last_good() {
    let (work, cfg) = temp_cfg("accept");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();

    let baseline = report(0.80, "v1");
    let outcome = reg
        .run_step(
            "step-1",
            "train",
            1,
            &baseline,
            || {
                // The step "trains" — improves the adapter.
                write_adapter(&work, "TRAINED");
                Ok(vec!["trace:goalA".to_string()])
            },
            || Ok(report(0.90, "v1")), // candidate improved
        )
        .unwrap();

    assert_eq!(outcome.action, StepAction::Commit);
    assert!(!outcome.halt);
    assert_eq!(reg.store().last_good().as_deref(), Some("step-1"));
    // The trained adapter is kept.
    assert_eq!(adapter_marker(&work), "TRAINED");
}

#[test]
fn regress_rolls_back_to_byte_equal_state() {
    let (work, cfg) = temp_cfg("regress");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();

    let baseline = report(0.80, "v1");
    let outcome = reg
        .run_step(
            "step-1",
            "train",
            1,
            &baseline,
            || {
                write_adapter(&work, "WORSE");
                Ok(vec!["trace:goalA".to_string()])
            },
            || Ok(report(0.50, "v1")), // big drop ⇒ regress
        )
        .unwrap();

    assert_eq!(outcome.action, StepAction::Rollback);
    assert!(!outcome.halt, "soft regress must NOT halt");
    // State restored byte-equal to the pre-step adapter.
    assert_eq!(
        adapter_marker(&work),
        "BASE",
        "regress restores the pre-step adapter"
    );
    // No last_good advanced (there was none, and regress doesn't set one).
    assert_eq!(reg.store().last_good(), None);
}

#[test]
fn catastrophe_rolls_back_quarantines_and_halts() {
    let (work, cfg) = temp_cfg("catastrophe");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();

    let baseline = report(0.80, "v1");
    let outcome = reg
        .run_step(
            "step-1",
            "round:goalA",
            1,
            &baseline,
            || {
                write_adapter(&work, "COLLAPSED");
                Ok(vec!["trace:goalA".to_string()])
            },
            || Ok(report(0.02, "v1")), // below catastrophe floor (0.10)
        )
        .unwrap();

    assert_eq!(outcome.action, StepAction::Quarantine);
    assert!(outcome.halt, "catastrophe MUST signal halt");
    // Rolled back.
    assert_eq!(adapter_marker(&work), "BASE");
    // The cause is quarantined.
    let q = reg.quarantine().unwrap();
    assert!(
        q.contains("trace:goalA"),
        "the catastrophic cause is quarantined"
    );

    // A subsequent round consulting the quarantine SKIPS the cause.
    let ds = Dataset::new(vec![
        GenExample::Qa {
            prompt: "good".to_string(),
            completion: "x".to_string(),
            source: None,
            gen: Some("trace:goalB".to_string()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Qa {
            prompt: "bad".to_string(),
            completion: "y".to_string(),
            source: None,
            gen: Some("trace:goalA".to_string()), // quarantined
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
    ]);
    let (kept, dropped) = q.filter(&ds);
    assert_eq!(dropped, 1, "the quarantined-provenance row is dropped");
    assert_eq!(kept.len(), 1);
}

#[test]
fn quarantine_clear_rearms() {
    let (work, _cfg) = temp_cfg("rearm");
    let path = work.join("quarantine.json");
    let mut q = Quarantine::default();
    q.add(["trace:goalA".to_string()]);
    q.write(&path).unwrap();
    assert!(Quarantine::load(&path).unwrap().contains("trace:goalA"));

    // Clear = re-arm.
    Quarantine::default().write(&path).unwrap();
    assert!(Quarantine::load(&path).unwrap().is_empty());
}

#[test]
fn checkpoints_list_and_manifest_round_trip() {
    let (work, cfg) = temp_cfg("ckpt");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();

    reg.run_step(
        "step-1",
        "train",
        1,
        &report(0.80, "v1"),
        || {
            write_adapter(&work, "T1");
            Ok(vec!["trace:goalA".to_string()])
        },
        || Ok(report(0.85, "v1")),
    )
    .unwrap();

    let all = reg.store().list().unwrap();
    // We expect at least the pre-snapshot + the committed step.
    assert!(all.iter().any(|m| m.id == "step-1"));
    let m = reg.store().load_manifest("step-1").unwrap();
    assert_eq!(m.gen_provenance, vec!["trace:goalA".to_string()]);
    assert!(m.metrics.is_some());
}

#[test]
fn step_error_rolls_back_without_verdict() {
    let (work, cfg) = temp_cfg("steperr");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();

    let outcome = reg
        .run_step(
            "step-1",
            "train",
            1,
            &report(0.80, "v1"),
            || {
                write_adapter(&work, "PARTIAL");
                anyhow::bail!("simulated trainer crash")
            },
            || Ok(report(0.9, "v1")),
        )
        .unwrap();

    assert_eq!(outcome.action, StepAction::Rollback);
    assert!(
        outcome.verdict.is_none(),
        "a step that errors has no verdict"
    );
    assert_eq!(
        adapter_marker(&work),
        "BASE",
        "crash rolls back to pre-step state"
    );
}

#[test]
fn strict_step_error_propagates_but_still_rolls_back() {
    // run_step_strict (track 31 Q2): a train crash is restored to the pre-step
    // state (the txn guarantee is unchanged) AND returned as Err so the daemon's
    // retry/supervisor path can handle it — unlike the lenient run_step above.
    let (work, cfg) = temp_cfg("steperr-strict");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();

    let result = reg.run_step_strict(
        "step-1",
        "train",
        1,
        &report(0.80, "v1"),
        || {
            write_adapter(&work, "PARTIAL");
            anyhow::bail!("simulated trainer crash")
        },
        || Ok(report(0.9, "v1")),
    );

    assert!(result.is_err(), "strict mode propagates the step error");
    assert_eq!(
        adapter_marker(&work),
        "BASE",
        "the adapter is still rolled back to the pre-step state"
    );
}

#[test]
fn evolution_log_records_actions() {
    let (work, cfg) = temp_cfg("log");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();
    reg.run_step(
        "step-1",
        "train",
        1,
        &report(0.80, "v1"),
        || Ok(vec!["g".to_string()]),
        || Ok(report(0.9, "v1")),
    )
    .unwrap();

    let log_path = work.join("evolution-log.jsonl");
    let entries = scrt_evolve::regulate::log::read_all(&log_path).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].action, StepAction::Commit);
    assert_eq!(entries[0].checkpoint_id, "step-1");
}

// Use the Cell import (silences unused if a future edit drops it) — and prove
// a scorer closure can carry mutable round state for multi-step tests.
#[test]
fn scorer_closure_can_track_calls() {
    let calls = Cell::new(0);
    let (work, cfg) = temp_cfg("calls");
    write_adapter(&work, "BASE");
    let reg = Regulator::new(&cfg).unwrap();
    reg.run_step(
        "s1",
        "train",
        1,
        &report(0.8, "v1"),
        || Ok(vec![]),
        || {
            calls.set(calls.get() + 1);
            Ok(report(0.85, "v1"))
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 1);
}
