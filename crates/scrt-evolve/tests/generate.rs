//! Generate stage tests: mocked ApiEndpoint, dataset round-trip, multi-turn
//! refine, and the missing-`api_key_env` error.

use std::cell::RefCell;

use scrt_evolve::config::{GenerateApiConfig, GenerateConfig};
use scrt_evolve::dataset::{Dataset, GenExample, Outcome, Tier, Verdict};
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
fn parser_salvages_truncated_array() {
    // A small teacher emitted a valid-prefix array that got truncated (missing
    // the closing `]`) plus a trailing incomplete object — the real shape seen
    // during track-24 bring-up. The two complete objects must be salvaged.
    let response = concat!(
        "[{\"kind\":\"qa\",\"prompt\":\"q1\",\"completion\":\"a1\"}, ",
        "{\"kind\":\"qa\",\"prompt\":\"q2\",\"completion\":\"a2\"}, ",
        "{\"kind\":\"qa\",\"prompt\":\"q3 truncated"
    );
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let dataset = run_with_backend(&backend, &fixture_ctx(), &["qa".into()], 1).unwrap();
    assert_eq!(dataset.len(), 2, "the two complete objects are salvaged");
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
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Instruction {
            instruction: "do x".into(),
            input: "ctx".into(),
            output: "done".into(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Completion {
            text: "raw".into(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Contrastive {
            query: "auth".into(),
            positive: "login.rs".into(),
            negatives: vec!["db.rs".into()],
            stash: Some("auth".into()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
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

// ---------------------------------------------------------------------------
// Signal-chain integration: constitution + taste -> generate system prompt.
// Proves the end-to-end seam — config values/taste actually reach generation
// (and therefore shape the dataset and downstream training).
// ---------------------------------------------------------------------------

#[test]
fn constitution_and_taste_steer_the_generate_prompt() {
    use scrt_evolve::config::{EvolveConfig, GoalConfig};

    // A config with a GLOBAL constitution + taste, and a goal that ADDS its own.
    let mut cfg = EvolveConfig::default();
    cfg.evolve.constitution = Some("Always cite the exact scrt flag.".to_string());
    cfg.evolve.taste = Some("Answer in one terse sentence.".to_string());
    let goal = GoalConfig {
        name: "scrt-cli".into(),
        topic: "mp-stash".into(),
        tag: "scrt-cli".into(),
        project: None,
        probe_set: None,
        weight: None,
        cadence: None,
        constitution: Some("Prefer the canonical spelling --mp-stash.".into()),
        taste: Some("No marketing language.".into()),
    };

    // for_goal must layer goal values on top of global; compose_steering renders.
    let goal_cfg = cfg.for_goal(&goal);
    let steering = goal_cfg.compose_steering().expect("steering present");
    assert!(steering.contains("Always cite the exact scrt flag.")); // global constitution
    assert!(steering.contains("Prefer the canonical spelling --mp-stash.")); // goal constitution
    assert!(steering.contains("Answer in one terse sentence.")); // global taste
    assert!(steering.contains("No marketing language.")); // goal taste

    // And the steering actually reaches the backend's SYSTEM message.
    let response = r#"[{"kind":"qa","prompt":"q","completion":"a"}]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let _ = scrt_evolve::generate::run_with_backend_steered(
        &backend,
        &fixture_ctx(),
        &["qa".to_string()],
        1,
        Some(steering.as_str()),
    )
    .unwrap();

    let msgs = backend_last_system_message(&backend);
    assert!(
        msgs.contains("Prefer the canonical spelling --mp-stash.")
            && msgs.contains("No marketing language."),
        "constitution + taste must appear in the generate system prompt; got:\n{msgs}"
    );

    // No steering ⇒ None (preserves built-in-template behavior).
    assert!(EvolveConfig::default().compose_steering().is_none());
}

/// Reach into the ApiEndpoint's mock transport for the last system message.
fn backend_last_system_message(backend: &ApiEndpoint<MockTransport>) -> String {
    backend
        .transport()
        .last_messages
        .borrow()
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

// --- track 37 Phase C: rejection sampling wired into generation ---

/// A judge that scores rows by a fixed script (one score per row, in order).
struct ScriptedJudge {
    scores: Vec<f32>,
}
impl scrt_evolve::judge::PairJudge for ScriptedJudge {
    fn score(&self, rows: &[GenExample], _steering: Option<&str>) -> anyhow::Result<Vec<f32>> {
        // Repeat the script to cover however many rows the pool holds.
        Ok((0..rows.len())
            .map(|i| self.scores[i % self.scores.len()])
            .collect())
    }
}

#[test]
fn rejection_sampling_generates_n_and_keeps_top_k() {
    // Each backend call yields ONE candidate qa row; candidates_per_seed=4 ⇒ the
    // pool is 4 rows for the single passage/mode; keep_k (per_passage) = 2.
    let response = r#"[{"kind":"qa","prompt":"q","completion":"a"}]"#;
    let backend = ApiEndpoint::with_transport(
        MockTransport::new(vec![response, response, response, response]),
        1,
    );
    // Scores 0.2, 0.9, 0.4, 0.8 → top-2 above 0.5 = 0.9, 0.8.
    let judge = ScriptedJudge {
        scores: vec![0.2, 0.9, 0.4, 0.8],
    };
    let rs = scrt_evolve::generate::RejectionSampling {
        candidates_per_seed: 4,
        min_score: 0.5,
        judge: Some(&judge),
        steering: None,
    };
    let ds = scrt_evolve::generate::run_with_backend_sampled(
        &backend,
        &fixture_ctx(),
        &["qa".to_string()],
        2, // keep_k
        &[],
        &rs,
    )
    .unwrap();
    assert_eq!(ds.len(), 2, "kept top-2 of 4 candidates");
    // The backend was called 4 times (N candidates), proving fan-out.
    assert_eq!(*backend.transport().calls.borrow(), 4);
    // Kept rows carry judge scores + the rsample stamp.
    for row in &ds.rows {
        assert!(row.judge_score().is_some(), "kept row persists judge_score");
        match row {
            GenExample::Qa { gen, .. } => {
                assert_eq!(gen.as_deref(), Some("rsample:4"))
            }
            other => panic!("expected Qa, got {other:?}"),
        }
    }
    // Ranked: highest score first.
    assert_eq!(ds.rows[0].judge_score(), Some(0.9));
    assert_eq!(ds.rows[1].judge_score(), Some(0.8));
}

#[test]
fn candidates_per_seed_one_is_single_pass() {
    // n=1 (or no judge) ⇒ single backend call, no rsample stamp (byte-identical).
    let response = r#"[{"kind":"qa","prompt":"q","completion":"a"}]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let rs = scrt_evolve::generate::RejectionSampling::default();
    let ds = scrt_evolve::generate::run_with_backend_sampled(
        &backend,
        &fixture_ctx(),
        &["qa".to_string()],
        3,
        &[],
        &rs,
    )
    .unwrap();
    assert_eq!(ds.len(), 1);
    assert_eq!(*backend.transport().calls.borrow(), 1, "single pass");
    assert_eq!(ds.rows[0].judge_score(), None, "unjudged in single-pass");
}
