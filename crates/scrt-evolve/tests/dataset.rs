//! Dataset format contract tests — the generate↔train JSONL boundary.
//!
//! The JSONL schema is the cross-language contract (Rust writer ↔ Python
//! reader). These tests pin the `kind` tags, nested-value survival, and the
//! 1-based line-number error reporting that `from_jsonl` promises.

use scrt_evolve::dataset::{Dataset, GenExample};

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
        },
        GenExample::Cli {
            prompt: "search the tree for auth and stash it".to_string(),
            command: "scrt \"auth\" --mp-stash auth --mp-ttl 4h".to_string(),
            source: None,
            gen: None,
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
