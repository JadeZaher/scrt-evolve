//! Transcript harvester tests (track 20 slice 4). No ML, no network.
//!
//! A fixture transcript → captured raw file + filtered, `trace:<goal>`-stamped
//! rows that round-trip through the dataset contract. Off-goal turns are dropped
//! (capture-then-filter); duplicates collapse; provenance is stamped.

use scrt_evolve::dataset::GenExample;
use scrt_evolve::harvest::{self, TranscriptEntry};
use scrt_evolve::{Dataset, GoalConfig};

/// A goal whose topic is "scrt mind palace stash commands".
fn goal() -> GoalConfig {
    GoalConfig {
        name: "scrt-cli-fluency".to_string(),
        topic: "stash palace commands".to_string(),
        tag: "scrt-cli".to_string(),
        ..Default::default()
    }
}

/// A fixture transcript: two on-goal exchanges (one CLI, one prose), one
/// off-goal exchange (must be dropped), and a duplicate of the first (must
/// collapse).
const FIXTURE: &str = r#"
{"role":"system","text":"you are a helpful agent"}
{"role":"user","text":"How do I save a stash to the palace?"}
{"role":"assistant","text":"Run the stash command.","command":"scrt \"auth\" --mp-stash auth --mp-ttl 4h"}
{"role":"user","text":"What does mp-compose do to two stashes in the palace?"}
{"role":"assistant","text":"mp-compose unions two stashes into one result set."}
{"role":"user","text":"What's the weather in Paris today?"}
{"role":"assistant","text":"It is sunny in Paris."}
{"role":"user","text":"How do I save a stash to the palace?"}
{"role":"assistant","text":"Run the stash command.","command":"scrt \"auth\" --mp-stash auth --mp-ttl 4h"}
"#;

#[test]
fn capture_writes_raw_file_and_distills_stamped_rows() {
    let mut dir = std::env::temp_dir();
    dir.push(format!("scrt-evolve-harvest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let traces_dir = dir.join("traces").join("scrt-cli-fluency");

    let g = goal();
    let res = harvest::capture_and_harvest(&g, &traces_dir, "session-abc", "2026-06-20", FIXTURE)
        .expect("harvest runs");

    // --- Capture: the raw transcript was written verbatim to <slug>-<date>.jsonl
    assert_eq!(
        res.captured_path,
        traces_dir.join("session-abc-2026-06-20.jsonl")
    );
    assert!(res.captured_path.exists(), "captured file must exist");
    let captured = std::fs::read_to_string(&res.captured_path).unwrap();
    assert_eq!(captured, FIXTURE, "capture is byte-for-byte the raw input");

    // --- Filter: off-goal Paris turn dropped; duplicate collapsed ⇒ 2 rows.
    assert_eq!(
        res.dataset.len(),
        2,
        "only the two distinct on-goal exchanges"
    );

    // --- Distill + provenance: every row stamped gen = "trace:<goal.name>".
    for row in &res.dataset.rows {
        let gen = match row {
            GenExample::Cli { gen, .. } => gen.clone(),
            GenExample::Qa { gen, .. } => gen.clone(),
            other => panic!("unexpected row variant: {other:?}"),
        };
        assert_eq!(
            gen.as_deref(),
            Some("trace:scrt-cli-fluency"),
            "trace rows must be provenance-stamped"
        );
    }

    // The CLI exchange distilled to a Cli row with the recorded command.
    let cli = res
        .dataset
        .rows
        .iter()
        .find(|r| matches!(r, GenExample::Cli { .. }))
        .expect("a Cli row");
    if let GenExample::Cli { command, .. } = cli {
        assert!(command.contains("--mp-stash"));
    }

    // The prose exchange distilled to a Qa row.
    assert!(
        res.dataset
            .rows
            .iter()
            .any(|r| matches!(r, GenExample::Qa { .. })),
        "the prose exchange distills to a Qa row"
    );

    // No off-goal content leaked in.
    assert!(
        !res.dataset
            .to_jsonl()
            .unwrap()
            .to_lowercase()
            .contains("paris"),
        "off-goal turns must not enter the dataset"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_rows_round_trip_through_dataset_contract() {
    let g = goal();
    let entries = TranscriptEntry::parse_jsonl(FIXTURE).unwrap();
    let harvested = harvest::harvest_entries(&g, &entries);

    // Serialize to JSONL and parse back — the contract holds for trace rows.
    let jsonl = harvested.dataset.to_jsonl().unwrap();
    let back = Dataset::from_jsonl(&jsonl).unwrap();
    assert_eq!(
        harvested.dataset.rows, back.rows,
        "trace rows round-trip through the dataset.jsonl contract"
    );
}

#[test]
fn harvest_is_deterministic() {
    let g = goal();
    let entries = TranscriptEntry::parse_jsonl(FIXTURE).unwrap();
    let a = harvest::harvest_entries(&g, &entries);
    let b = harvest::harvest_entries(&g, &entries);
    assert_eq!(a.dataset.rows, b.dataset.rows, "harvest is deterministic");
}

#[test]
fn off_goal_transcript_yields_no_rows() {
    let g = goal();
    let off_goal = r#"
{"role":"user","text":"What's the capital of France?"}
{"role":"assistant","text":"Paris."}
{"role":"user","text":"And the weather?"}
{"role":"assistant","text":"Sunny."}
"#;
    let entries = TranscriptEntry::parse_jsonl(off_goal).unwrap();
    let harvested = harvest::harvest_entries(&g, &entries);
    assert!(
        harvested.dataset.is_empty(),
        "a transcript with no goal-relevant turns yields no trace rows"
    );
}
