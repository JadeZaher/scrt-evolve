//! Eval-gated round driver + scheduler tests (track 20 slices 6–9). ML-free:
//! discover/generate/train/score are injected deterministic closures, so the
//! full gate (train → eval → keep|rollback), catastrophe→halt, quarantine-skip,
//! and the bounded weighted/round-robin scheduler are all proven without a model.

use std::cell::RefCell;
use std::path::PathBuf;

use scrt_evolve::dataset::{Dataset, GenExample};
use scrt_evolve::discover::{DiscoveredContext, Passage};
use scrt_evolve::eval::ScoreReport;
use scrt_evolve::regulate::StepAction;
use scrt_evolve::rounds::{run_round, run_schedule, RoundHooks, SchedulePolicy};
use scrt_evolve::{EvolveConfig, GoalConfig};

fn temp_cfg(tag: &str, goals_toml: &str) -> (PathBuf, EvolveConfig) {
    let mut dir = std::env::temp_dir();
    dir.push(format!("scrt-evolve-rounds-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let toml = format!(
        r#"
[evolve]
model_path = "/m"
corpus_dir = {dir:?}
palace_path = {dir:?}
work_dir = {dir:?}

[eval]
probe_holdout_frac = 0.3

[regulate]
enabled = true
accept_tolerance = 0.02
catastrophe_floor = 0.10

{goals_toml}
"#,
    );
    (dir.clone(), EvolveConfig::from_toml_str(&toml).unwrap())
}

fn ctx_with(n: usize) -> DiscoveredContext {
    let passages = (0..n)
        .map(|i| Passage {
            text: format!("passage {i} about authenticate"),
            source: format!("file{i}.md"),
            score: 1.0,
            seed: "s".to_string(),
        })
        .collect();
    DiscoveredContext {
        passages,
        anchors: vec![],
    }
}

fn dataset_with(n: usize, gen_stamp: &str) -> Dataset {
    let rows = (0..n)
        .map(|i| GenExample::Cli {
            prompt: format!("do thing {i}"),
            command: format!("scrt \"q{i}\" --mp-stash s{i}"),
            source: Some("g".to_string()),
            gen: Some(gen_stamp.to_string()),
        })
        .collect();
    Dataset::new(rows)
}

fn report(c: f64) -> ScoreReport {
    ScoreReport {
        correctness: c,
        constitution_adherence: None,
        mean_exit_depth: None,
        perplexity: None,
        n: 10,
        // version is overwritten by the probe carve in the round; for the
        // baseline we must match what the candidate reports. The round's score
        // hook controls the candidate version; we keep them equal here.
        probe_version: "fixed".to_string(),
        backend: "test".to_string(),
    }
}

fn write_adapter(work: &std::path::Path) {
    let a = work.join("adapter");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::write(a.join("adapter.safetensors"), "W").unwrap();
}

#[test]
fn round_commits_when_eval_passes() {
    let goal = GoalConfig {
        name: "g1".into(),
        topic: "authenticate".into(),
        tag: "sec".into(),
        ..Default::default()
    };
    let (work, cfg) = temp_cfg("commit", "");
    write_adapter(&work);

    let discover = |_: &EvolveConfig| Ok(ctx_with(10));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(10, "trace:g1"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["trace:g1".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.9));
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };

    let r = run_round(&cfg, &goal, 1, &report(0.8), &hooks).unwrap();
    assert_eq!(r.action, Some(StepAction::Commit));
    assert!(!r.halt);
    assert!(r.rows >= 1);
}

#[test]
fn round_rolls_back_on_regress() {
    let goal = GoalConfig {
        name: "g1".into(),
        topic: "authenticate".into(),
        tag: "sec".into(),
        ..Default::default()
    };
    let (work, cfg) = temp_cfg("regress", "");
    write_adapter(&work);

    let discover = |_: &EvolveConfig| Ok(ctx_with(10));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(10, "trace:g1"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["trace:g1".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.5)); // big drop vs 0.8 baseline
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };

    let r = run_round(&cfg, &goal, 1, &report(0.8), &hooks).unwrap();
    assert_eq!(r.action, Some(StepAction::Rollback));
    assert!(!r.halt, "soft regress does not halt");
}

#[test]
fn round_catastrophe_halts_and_quarantines() {
    let goal = GoalConfig {
        name: "g1".into(),
        topic: "authenticate".into(),
        tag: "sec".into(),
        ..Default::default()
    };
    let (work, cfg) = temp_cfg("cata", "");
    write_adapter(&work);

    let discover = |_: &EvolveConfig| Ok(ctx_with(10));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(10, "trace:g1"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["trace:g1".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.02)); // below floor
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };

    let r = run_round(&cfg, &goal, 1, &report(0.8), &hooks).unwrap();
    assert_eq!(r.action, Some(StepAction::Quarantine));
    assert!(r.halt, "catastrophe halts the schedule");

    // The provenance is quarantined → a re-run drops those rows pre-train.
    let reg = scrt_evolve::Regulator::new(&cfg).unwrap();
    assert!(reg.quarantine().unwrap().contains("trace:g1"));
}

#[test]
fn schedule_is_bounded_and_round_robins_two_goals() {
    let goals = r#"
[[goals]]
name = "g1"
topic = "authenticate"
tag = "sec"

[[goals]]
name = "g2"
topic = "cache"
tag = "perf"
"#;
    let (work, cfg) = temp_cfg("sched", goals);
    write_adapter(&work);

    let discover = |_: &EvolveConfig| Ok(ctx_with(8));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(8, "trace:x"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["trace:x".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.9));
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };
    let baseline = |_: &GoalConfig| report(0.8);

    // Budget of 4 rounds across 2 goals, round-robin ⇒ g1,g2,g1,g2.
    let rep = run_schedule(&cfg, SchedulePolicy::RoundRobin, 4, 1, &hooks, &baseline).unwrap();
    assert_eq!(rep.rounds.len(), 4, "schedule is bounded by max_rounds");
    assert!(!rep.halted);
    let names: Vec<&str> = rep.rounds.iter().map(|r| r.goal.as_str()).collect();
    assert_eq!(names, vec!["g1", "g2", "g1", "g2"]);
    assert_eq!(rep.committed(), 4);
}

#[test]
fn schedule_halts_midway_on_catastrophe() {
    let goals = r#"
[[goals]]
name = "g1"
topic = "authenticate"
tag = "sec"

[[goals]]
name = "g2"
topic = "cache"
tag = "perf"
"#;
    let (work, cfg) = temp_cfg("schedhalt", goals);
    write_adapter(&work);

    // Score collapses on EVERY round ⇒ the first round catastrophes + halts.
    let discover = |_: &EvolveConfig| Ok(ctx_with(8));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(8, "trace:x"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["trace:x".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.01));
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };
    let baseline = |_: &GoalConfig| report(0.8);

    let rep = run_schedule(&cfg, SchedulePolicy::RoundRobin, 4, 1, &hooks, &baseline).unwrap();
    assert!(rep.halted, "schedule halts on catastrophe");
    assert_eq!(
        rep.rounds.len(),
        1,
        "halts after the first (catastrophic) round"
    );
}

#[test]
fn weighted_schedule_favors_heavier_goal() {
    let goals = r#"
[[goals]]
name = "heavy"
topic = "authenticate"
tag = "sec"
weight = 3.0

[[goals]]
name = "light"
topic = "cache"
tag = "perf"
weight = 1.0
"#;
    let (work, cfg) = temp_cfg("weighted", goals);
    write_adapter(&work);

    let discover = |_: &EvolveConfig| Ok(ctx_with(8));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(8, "trace:x"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["trace:x".to_string()]);
    let score = |_: &EvolveConfig| Ok(report(0.9));
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };
    let baseline = |_: &GoalConfig| report(0.8);

    let rep = run_schedule(&cfg, SchedulePolicy::Weighted, 4, 1, &hooks, &baseline).unwrap();
    let heavy = rep.rounds.iter().filter(|r| r.goal == "heavy").count();
    let light = rep.rounds.iter().filter(|r| r.goal == "light").count();
    assert!(
        heavy > light,
        "weight=3 goal gets more rounds than weight=1 ({heavy} vs {light})"
    );
}

#[test]
fn round_bails_cleanly_when_no_passages() {
    let goal = GoalConfig {
        name: "g1".into(),
        topic: "authenticate".into(),
        tag: "sec".into(),
        ..Default::default()
    };
    let (_work, cfg) = temp_cfg("nopass", "");

    let discover = |_: &EvolveConfig| Ok(ctx_with(0));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(0, "x"));
    let train = |_: &EvolveConfig, _: &Dataset| Ok(vec![]);
    let score = |_: &EvolveConfig| Ok(report(0.9));
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };

    let r = run_round(&cfg, &goal, 1, &report(0.8), &hooks).unwrap();
    assert_eq!(r.action, None, "no passages ⇒ no transaction");
    assert!(!r.halt);
}

#[test]
fn train_is_only_called_inside_transaction() {
    // Proves the weight-mutating step runs through the regulator: train must be
    // invoked exactly once per committed round (not before discover/generate).
    let goal = GoalConfig {
        name: "g1".into(),
        topic: "authenticate".into(),
        tag: "sec".into(),
        ..Default::default()
    };
    let (work, cfg) = temp_cfg("trainonce", "");
    write_adapter(&work);
    let calls = RefCell::new(0);

    let discover = |_: &EvolveConfig| Ok(ctx_with(10));
    let generate = |_: &EvolveConfig, _: &DiscoveredContext| Ok(dataset_with(10, "trace:g1"));
    let train = |_: &EvolveConfig, _: &Dataset| {
        *calls.borrow_mut() += 1;
        Ok(vec!["trace:g1".to_string()])
    };
    let score = |_: &EvolveConfig| Ok(report(0.9));
    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };

    run_round(&cfg, &goal, 1, &report(0.8), &hooks).unwrap();
    assert_eq!(
        *calls.borrow(),
        1,
        "train runs exactly once, inside the txn"
    );
}
