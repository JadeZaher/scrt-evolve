//! Generate stage tests: mocked ApiEndpoint, dataset round-trip, multi-turn
//! refine, and the missing-`api_key_env` error.

use std::cell::RefCell;

use scrt_evolve::config::{GenerateApiConfig, GenerateConfig};
use scrt_evolve::dataset::{Dataset, GenExample};
use scrt_evolve::discover::{DiscoveredContext, Passage};
use scrt_evolve::generate::api::{ApiEndpoint, ChatMessage, ChatTransport};
use scrt_evolve::generate::run_with_backend;

/// A transport that records how many times it was called and replays canned
/// responses in order (the last is repeated if calls exceed the script).
struct MockTransport {
    responses: Vec<String>,
    calls: RefCell<usize>,
    last_messages: RefCell<Vec<ChatMessage>>,
}

impl MockTransport {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(String::from).collect(),
            calls: RefCell::new(0),
            last_messages: RefCell::new(Vec::new()),
        }
    }
}

impl ChatTransport for MockTransport {
    fn complete(&self, messages: &[ChatMessage]) -> anyhow::Result<String> {
        *self.last_messages.borrow_mut() = messages.to_vec();
        let n = *self.calls.borrow();
        *self.calls.borrow_mut() = n + 1;
        let idx = n.min(self.responses.len() - 1);
        Ok(self.responses[idx].clone())
    }
}

fn fixture_ctx() -> DiscoveredContext {
    DiscoveredContext {
        passages: vec![Passage {
            text: "scrt --mp-stash NAME stores the current search as a named stash.".to_string(),
            source: "README.md".to_string(),
            score: 10.0,
            seed: "corpus:stash".to_string(),
        }],
        anchors: vec![],
    }
}

#[test]
fn mocked_backend_produces_qa_and_instruction_rows() {
    let response = r#"[
      {"kind":"qa","prompt":"How do I stash a search?","completion":"Use scrt --mp-stash NAME."},
      {"kind":"instruction","instruction":"Explain --mp-stash","input":"","output":"It stores the current search as a named stash."}
    ]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);

    let kinds = vec!["qa".to_string(), "instruction".to_string()];
    let ctx = fixture_ctx();
    let dataset = run_with_backend(&backend, &ctx, &kinds, 3).unwrap();

    assert_eq!(dataset.len(), 2);
    // Provenance is injected by the parser from the passage.
    match &dataset.rows[0] {
        GenExample::Qa {
            prompt,
            source,
            gen,
            ..
        } => {
            assert!(prompt.contains("stash"));
            assert_eq!(source.as_deref(), Some("README.md"));
            assert_eq!(gen.as_deref(), Some("api"));
        }
        other => panic!("expected qa, got {other:?}"),
    }
    match &dataset.rows[1] {
        GenExample::Instruction { source, gen, .. } => {
            assert_eq!(source.as_deref(), Some("README.md"));
            assert_eq!(gen.as_deref(), Some("api"));
        }
        other => panic!("expected instruction, got {other:?}"),
    }
}

#[test]
fn parser_tolerates_markdown_fenced_array() {
    let response =
        "Here you go:\n```json\n[{\"kind\":\"qa\",\"prompt\":\"q\",\"completion\":\"a\"}]\n```";
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let dataset = run_with_backend(&backend, &fixture_ctx(), &["qa".into()], 1).unwrap();
    assert_eq!(dataset.len(), 1);
}

#[test]
fn malformed_rows_are_skipped_not_fatal() {
    // One good row, one missing required field — the good one survives.
    let response = r#"[
      {"kind":"qa","prompt":"q","completion":"a"},
      {"kind":"qa","prompt":"only prompt, no completion"}
    ]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let dataset = run_with_backend(&backend, &fixture_ctx(), &["qa".into()], 2).unwrap();
    assert_eq!(dataset.len(), 1);
}

#[test]
fn turns_greater_than_one_issues_refine_turns() {
    let r1 = r#"[{"kind":"qa","prompt":"q1","completion":"a1"}]"#;
    let r2 = r#"[{"kind":"qa","prompt":"q1","completion":"refined a1"}]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![r1, r2]), 2);

    // Inspect the mock by reaching through: easiest is to run and check the
    // refined completion came through (proves a 2nd turn happened).
    let dataset = run_with_backend(&backend, &fixture_ctx(), &["qa".into()], 1).unwrap();
    match &dataset.rows[0] {
        GenExample::Qa { completion, .. } => assert_eq!(completion, "refined a1"),
        other => panic!("expected qa, got {other:?}"),
    }
}

#[test]
fn dataset_round_trips_through_jsonl() {
    let ds = Dataset::new(vec![
        GenExample::Qa {
            prompt: "q".into(),
            completion: "a".into(),
            source: Some("s.rs".into()),
            gen: Some("api".into()),
        },
        GenExample::Instruction {
            instruction: "do x".into(),
            input: "ctx".into(),
            output: "done".into(),
            source: None,
            gen: None,
        },
        GenExample::Completion {
            text: "raw".into(),
            source: None,
        },
        GenExample::Contrastive {
            query: "auth".into(),
            positive: "login.rs".into(),
            negatives: vec!["db.rs".into()],
            stash: Some("auth".into()),
        },
    ]);

    let jsonl = ds.to_jsonl().unwrap();
    // One object per line.
    assert_eq!(jsonl.lines().count(), 4);
    let back = Dataset::from_jsonl(&jsonl).unwrap();
    assert_eq!(ds, back);
}

#[test]
fn missing_api_key_env_is_a_clear_error() {
    let var = "SCRT_EVOLVE_DEFINITELY_UNSET_TEST_VAR";
    std::env::remove_var(var);
    let gcfg = GenerateConfig {
        backend: "api".into(),
        api: Some(GenerateApiConfig {
            base_url: Some("http://localhost:1234/v1".into()),
            model: Some("m".into()),
            api_key_env: Some(var.to_string()),
            turns: 1,
        }),
        ..Default::default()
    };
    let result = ApiEndpoint::from_config(&gcfg);
    let msg = result.err().expect("missing key must error").to_string();
    assert!(
        msg.contains(var),
        "error should name the missing var: {msg}"
    );
}

#[test]
fn tool_call_rows_validate_against_real_schemas() {
    // One valid scrt_stash call, one hallucinated tool, one missing-required.
    // Only the valid one survives — grounded in scrt-core's real schemas.
    let response = r#"[
      {"prompt":"Stash the search as auth","tool":"scrt_stash","arguments":{"name":"auth","note":"Auth findings"}},
      {"prompt":"teleport","tool":"scrt_teleport","arguments":{"x":1}},
      {"prompt":"stash with only a name","tool":"scrt_stash","arguments":{"name":"only-name-no-note"}}
    ]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let ctx = fixture_ctx();
    let dataset = run_with_backend(&backend, &ctx, &["tool_call".to_string()], 3).unwrap();

    assert_eq!(dataset.len(), 1, "only the schema-valid tool call survives");
    match &dataset.rows[0] {
        GenExample::ToolCall {
            tool,
            arguments,
            source,
            gen,
            ..
        } => {
            assert_eq!(tool, "scrt_stash");
            assert_eq!(arguments["name"], "auth");
            assert_eq!(source.as_deref(), Some("README.md"));
            assert_eq!(gen.as_deref(), Some("api"));
        }
        other => panic!("expected tool_call, got {other:?}"),
    }
}

#[test]
fn cli_rows_require_scrt_command() {
    let response = r#"[
      {"prompt":"stash the auth search for 4h","command":"scrt \"auth\" --mp-stash auth --mp-ttl 4h"},
      {"prompt":"delete everything","command":"rm -rf /"}
    ]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let dataset = run_with_backend(&backend, &fixture_ctx(), &["cli".to_string()], 2).unwrap();
    assert_eq!(dataset.len(), 1, "non-scrt commands are dropped");
    match &dataset.rows[0] {
        GenExample::Cli { command, .. } => assert!(command.starts_with("scrt ")),
        other => panic!("expected cli, got {other:?}"),
    }
}

#[test]
fn no_api_key_env_means_unauthenticated_local_endpoint_ok() {
    // LM Studio / vLLM ignore auth — omitting api_key_env must NOT error.
    let gcfg = GenerateConfig {
        backend: "api".into(),
        api: Some(GenerateApiConfig {
            base_url: Some("http://localhost:1234/v1".into()),
            model: Some("openai/gpt-oss-20b".into()),
            api_key_env: None,
            turns: 1,
        }),
        ..Default::default()
    };
    assert!(ApiEndpoint::from_config(&gcfg).is_ok());
}
