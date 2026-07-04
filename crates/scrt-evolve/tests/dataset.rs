//! Dataset format contract tests — the generate↔train JSONL boundary.
//!
//! The JSONL schema is the cross-language contract (Rust writer ↔ Python
//! reader). These tests pin the `kind` tags, nested-value survival, the
//! 1-based line-number error reporting that `from_jsonl` promises, and the
//! contract-v1.1 additive metadata (track 37): a v1.0 line stays byte-identical
//! on re-serialize, and a v1.1 line round-trips all five metadata fields.

use scrt_evolve::dataset::{Dataset, GenExample, Outcome, Tier, Verdict};

#[test]
fn tool_call_and_cli_round_trip() {
    let ds = Dataset::new(vec![
        GenExample::ToolCall {
            prompt: "stash the auth findings".to_string(),
            tool: "scrt_stash".to_string(),
            arguments: serde_json::json!({
                "name": "auth-findings",
                "ttl": "4h",
                "tags": ["auth", "finding"],
            }),
            source: None,
            gen: Some("trace:scrt-cli-fluency".to_string()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Cli {
            prompt: "search the tree for auth and stash it".to_string(),
            command: "scrt \"auth\" --mp-stash auth --mp-ttl 4h".to_string(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
    ]);

    let jsonl = ds.to_jsonl().expect("serialize to jsonl");

    // The `kind` tags are the load-bearing contract for the Python reader.
    assert!(jsonl.contains("\"kind\":\"tool_call\""), "jsonl: {jsonl}");
    assert!(jsonl.contains("\"kind\":\"cli\""), "jsonl: {jsonl}");
    // The nested `arguments` object must survive serialization intact.
    assert!(
        jsonl.contains("\"name\":\"auth-findings\""),
        "jsonl: {jsonl}"
    );
    assert!(
        jsonl.contains("\"tags\":[\"auth\",\"finding\"]"),
        "jsonl: {jsonl}"
    );

    // Full round-trip: parse back and compare structurally.
    let parsed = Dataset::from_jsonl(&jsonl).expect("parse jsonl back");
    assert_eq!(parsed, ds);

    // And the nested JSON value specifically survives the round-trip.
    if let GenExample::ToolCall { arguments, .. } = &parsed.rows[0] {
        assert_eq!(arguments["name"], serde_json::json!("auth-findings"));
        assert_eq!(arguments["tags"], serde_json::json!(["auth", "finding"]));
    } else {
        panic!("first row should be a ToolCall, got {:?}", parsed.rows[0]);
    }
}

#[test]
fn malformed_line_errors_with_line_number() {
    // Line 1: a valid qa row. Line 2: blank (must be skipped). Line 3: garbage.
    let good = serde_json::to_string(&GenExample::Qa {
        prompt: "what is scrt?".to_string(),
        completion: "a token-budgeted context tool".to_string(),
        source: None,
        gen: None,
        outcome: Outcome::Unknown,
        judge_score: None,
        judge_verdict: Verdict::Unjudged,
        tier: Tier::Private,
        chosen_over: None,
    })
    .expect("serialize good row");
    let text = format!("{good}\n\n{{garbage}}\n");

    let err = Dataset::from_jsonl(&text).expect_err("garbage line must error");
    let msg = err.to_string();

    // The blank line 2 is skipped, so the error points at the 1-based line 3.
    assert!(
        msg.contains("line 3"),
        "error should name the 1-based garbage line; got: {msg}"
    );
}

/// Contract v1.1: a legacy (v1.0) line carrying NO metadata must parse
/// unchanged AND re-serialize byte-identically — additive optionals are
/// invisible when defaulted (all `skip_serializing_if`).
#[test]
fn v1_0_line_is_byte_identical_on_reserialize() {
    // A hand-written v1.0 line — exactly what an old writer emitted.
    let v1_0 = r#"{"kind":"cli","prompt":"list stashes","command":"scrt --mp-list","gen":"trace"}"#;
    let ds = Dataset::from_jsonl(v1_0).expect("parse v1.0 line");
    let out = ds.to_jsonl().expect("reserialize");
    assert_eq!(
        out.trim_end(),
        v1_0,
        "v1.0 line must round-trip byte-identically (no metadata leaked in)"
    );
    // The defaults are the safe ones.
    assert_eq!(ds.rows[0].outcome(), Outcome::Unknown);
    assert_eq!(ds.rows[0].judge_verdict(), Verdict::Unjudged);
    assert_eq!(ds.rows[0].tier(), Tier::Private);
    assert_eq!(ds.rows[0].judge_score(), None);
    assert_eq!(ds.rows[0].chosen_over(), None);
}

/// Contract v1.1: a fully-populated line round-trips all five metadata fields.
#[test]
fn v1_1_metadata_round_trips() {
    let mut row = GenExample::Cli {
        prompt: "run the tests".to_string(),
        command: "cargo test".to_string(),
        source: Some("transcript".to_string()),
        gen: Some("ingest:transcript".to_string()),
        outcome: Outcome::Unknown,
        judge_score: None,
        judge_verdict: Verdict::Unjudged,
        tier: Tier::Private,
        chosen_over: None,
    };
    row.set_outcome(Outcome::Success);
    row.set_judge(0.875, Verdict::Keep);
    row.set_tier(Tier::Shared);
    row.set_chosen_over("cli\u{1}run the tests\u{1}cargo tset".to_string());

    let ds = Dataset::new(vec![row]);
    let jsonl = ds.to_jsonl().expect("serialize");
    assert!(jsonl.contains("\"outcome\":\"success\""), "jsonl: {jsonl}");
    assert!(jsonl.contains("\"judge_verdict\":\"keep\""), "jsonl: {jsonl}");
    assert!(jsonl.contains("\"tier\":\"shared\""), "jsonl: {jsonl}");
    assert!(jsonl.contains("\"judge_score\":0.875"), "jsonl: {jsonl}");
    assert!(jsonl.contains("\"chosen_over\":"), "jsonl: {jsonl}");

    let parsed = Dataset::from_jsonl(&jsonl).expect("round-trip");
    assert_eq!(parsed, ds);
    assert_eq!(parsed.rows[0].outcome(), Outcome::Success);
    assert_eq!(parsed.rows[0].judge_score(), Some(0.875));
    assert_eq!(parsed.rows[0].judge_verdict(), Verdict::Keep);
    assert_eq!(parsed.rows[0].tier(), Tier::Shared);
    assert!(parsed.rows[0].chosen_over().is_some());
}

/// `Tier::most_restrictive` is the manifest rollup fold (Private wins).
#[test]
fn tier_most_restrictive_wins() {
    assert_eq!(Tier::Shared.most_restrictive(Tier::Shared), Tier::Shared);
    assert_eq!(Tier::Shared.most_restrictive(Tier::Private), Tier::Private);
    assert_eq!(Tier::Private.most_restrictive(Tier::Shared), Tier::Private);
    assert_eq!(Tier::Private.most_restrictive(Tier::Private), Tier::Private);
}
