//! LocalCandle backend tests (offline, `--features train` only).
//!
//! Covers the end-to-end fixture pipeline + provenance stamp, the degenerate
//! dedup/quality filter, cross-backend schema interchangeability, and sampling
//! determinism (styleguide §2.2).
#![cfg(feature = "train")]

use scrt_evolve::dataset::{Dataset, GenExample};
use scrt_evolve::discover::{DiscoveredContext, Passage};
use scrt_evolve::generate::local::{filter_degenerate, LocalCandle};
use scrt_evolve::generate::run_with_backend;
use scrt_evolve::model::{LoadedModel, ModelConfig};

fn fixture_ctx() -> DiscoveredContext {
    DiscoveredContext {
        passages: vec![Passage {
            text: "scrt --mp-stash NAME stores the current search as a named stash."
                .to_string(),
            source: "README.md".to_string(),
            score: 10.0,
            seed: "corpus:stash".to_string(),
        }],
        anchors: vec![],
    }
}

/// End-to-end: a random tiny model + LocalCandle runs offline without panic and
/// stamps `gen="local"` on whatever rows survive. A random model rarely emits
/// parseable JSON, so 0 rows is acceptable — we assert success + provenance.
#[test]
fn local_backend_produces_valid_rows_on_fixture() {
    let model = LoadedModel::random_fixture(ModelConfig::tiny(), 42).expect("fixture");
    let backend = LocalCandle::from_model(model, 32, 0.7, 7);

    let kinds = vec!["qa".to_string(), "instruction".to_string()];
    let dataset = run_with_backend(&backend, &fixture_ctx(), &kinds, 3)
        .expect("local generation must succeed offline");

    for row in &dataset.rows {
        match row {
            GenExample::Qa { gen, .. }
            | GenExample::Instruction { gen, .. }
            | GenExample::ToolCall { gen, .. }
            | GenExample::Cli { gen, .. } => {
                assert_eq!(gen.as_deref(), Some("local"), "rows must be stamped gen=local");
            }
            other => panic!("unexpected variant from prose generation: {other:?}"),
        }
    }
}

/// The dedup + quality filter drops empty, too-short, repeated-char, and
/// duplicate generations, keeping only the one good unique row.
#[test]
fn degenerate_output_is_filtered() {
    let good = GenExample::Qa {
        prompt: "How do I stash a search?".into(),
        completion: "Use scrt --mp-stash NAME to store it.".into(),
        source: Some("README.md".into()),
        gen: Some("local".into()),
    };
    let rows = vec![
        good.clone(),
        // exact duplicate of the good row
        good.clone(),
        // empty completion
        GenExample::Qa {
            prompt: "q".into(),
            completion: "".into(),
            source: None,
            gen: Some("local".into()),
        },
        // repeated single char
        GenExample::Qa {
            prompt: "q".into(),
            completion: "aaaaaaaa".into(),
            source: None,
            gen: Some("local".into()),
        },
        // answer echoes the prompt
        GenExample::Instruction {
            instruction: "Explain --mp-stash".into(),
            input: "".into(),
            output: "Explain --mp-stash".into(),
            source: None,
            gen: Some("local".into()),
        },
    ];

    let kept = filter_degenerate(rows);
    assert_eq!(kept.len(), 1, "only the good unique row survives");
    assert_eq!(kept[0], good);
}

/// A local row and an api row of the same variant round-trip through JSONL
/// identically — the dataset is backend-agnostic.
#[test]
fn local_and_api_rows_are_schema_interchangeable() {
    let local_row = GenExample::Qa {
        prompt: "How do I stash a search?".into(),
        completion: "Use scrt --mp-stash NAME.".into(),
        source: Some("README.md".into()),
        gen: Some("local".into()),
    };
    let api_row = GenExample::Qa {
        prompt: "How do I stash a search?".into(),
        completion: "Use scrt --mp-stash NAME.".into(),
        source: Some("README.md".into()),
        gen: Some("api".into()),
    };

    let ds = Dataset::new(vec![local_row.clone(), api_row.clone()]);
    let jsonl = ds.to_jsonl().expect("serialize");
    let back = Dataset::from_jsonl(&jsonl).expect("parse");

    assert_eq!(back, ds, "both backends' rows round-trip identically");
    assert_eq!(back.rows[0], local_row);
    assert_eq!(back.rows[1], api_row);
}

/// Same fixture seed + same sampling seed + same prompt → identical generated
/// dataset. Proves deterministic sampling (styleguide §2.2).
#[test]
fn same_seed_same_generation() {
    let a = LocalCandle::from_model(
        LoadedModel::random_fixture(ModelConfig::tiny(), 99).expect("fixture a"),
        24,
        0.8,
        13,
    );
    let b = LocalCandle::from_model(
        LoadedModel::random_fixture(ModelConfig::tiny(), 99).expect("fixture b"),
        24,
        0.8,
        13,
    );

    let kinds = vec!["qa".to_string()];
    let ctx = fixture_ctx();
    let ds_a = run_with_backend(&a, &ctx, &kinds, 2).expect("gen a");
    let ds_b = run_with_backend(&b, &ctx, &kinds, 2).expect("gen b");

    assert_eq!(ds_a, ds_b, "same seed must yield identical generation");
}
