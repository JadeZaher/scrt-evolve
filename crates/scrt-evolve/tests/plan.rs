//! Plan-stage tests: plan types round-trip, signal extraction, planner parsing,
//! plan-driven generation with self-written prompts, and the gap critic.

use std::cell::RefCell;

use scrt_evolve::dataset::{Dataset, GenExample};
use scrt_evolve::discover::{DiscoveredContext, Passage};
use scrt_evolve::generate::api::{ApiEndpoint, ChatMessage, ChatTransport};
use scrt_evolve::generate::run_plan_with_backend;
use scrt_evolve::plan::critic::{measure, critique_with_transport};
use scrt_evolve::plan::planner::{parse_plan, plan_with_transport};
use scrt_evolve::plan::signals;
use scrt_evolve::plan::{GenPlan, GenSpec};

struct MockTransport {
    responses: Vec<String>,
    calls: RefCell<usize>,
}
impl MockTransport {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(String::from).collect(),
            calls: RefCell::new(0),
        }
    }
}
impl ChatTransport for MockTransport {
    fn complete(&self, _m: &[ChatMessage]) -> anyhow::Result<String> {
        let n = *self.calls.borrow();
        *self.calls.borrow_mut() = n + 1;
        Ok(self.responses[n.min(self.responses.len() - 1)].clone())
    }
}

fn fixture_ctx() -> DiscoveredContext {
    DiscoveredContext {
        passages: vec![
            Passage {
                text: "scrt --mp-stash NAME stores a stash. scrt_stash(name, note) is the tool."
                    .into(),
                source: "README.md".into(),
                score: 9.0,
                seed: "corpus:mp-stash".into(),
            },
            Passage {
                text: "pub fn rank_similar(...) ranks stashes by simhash.".into(),
                source: "simhash.rs".into(),
                score: 7.0,
                seed: "corpus:similar".into(),
            },
        ],
        anchors: vec![],
    }
}

#[test]
fn plan_round_trips_json() {
    let plan = GenPlan {
        round: 0,
        strategy: "test".into(),
        specs: vec![GenSpec {
            modality: "tool_call".into(),
            prompt: "Generate scrt_stash calls.".into(),
            count: 10,
            target_tools: vec!["scrt_stash".into()],
            rationale: "dense stash usage".into(),
            passage_shape: "cli_ref".into(),
        }],
    };
    let json = plan.to_json().unwrap();
    let back = GenPlan::from_json(&json).unwrap();
    assert_eq!(plan, back);
}

#[test]
fn signals_extract_tool_and_shape() {
    // Minimal config (no palace); signal extraction must still work from ctx.
    let cfg = scrt_evolve::EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"").unwrap();
    let ctx = fixture_ctx();
    let sig = signals::extract(&cfg, &ctx);

    // Tool frequency picks up scrt_stash from the first passage.
    assert!(sig.cooccurrence.tool_frequency.get("scrt_stash").copied().unwrap_or(0) >= 1);
    // Flag frequency picks up --mp-stash.
    assert!(sig.cooccurrence.flag_frequency.contains_key("--mp-stash"));
    // Two passages classified into shapes.
    assert_eq!(sig.corpus_shape.per_passage.len(), 2);
    // Summary renders without panicking and mentions the tool.
    let summary = signals::summary(&sig);
    assert!(summary.contains("scrt_stash"));
}

#[test]
fn classify_shape_routes_content_types() {
    assert_eq!(signals::classify_shape("pub fn x() {}", "a.rs"), "code");
    assert_eq!(
        signals::classify_shape("scrt --mp-stash x --mp-ttl 4h", "README.md"),
        "cli_ref"
    );
    assert_eq!(signals::classify_shape("key = 1", "evolve.toml"), "config");
    assert_eq!(
        signals::classify_shape("This explains the concept of stashes.", "doc.md"),
        "conceptual"
    );
}

#[test]
fn planner_parses_self_written_plan() {
    let response = r#"{
      "strategy": "weight tool_call toward stash workflows",
      "specs": [
        {"modality":"tool_call","prompt":"Emit scrt_stash and scrt_get_stash calls grounded in the passage.","count":20,"target_tools":["scrt_stash","scrt_get_stash"],"passage_shape":"cli_ref","rationale":"high stash co-occurrence"},
        {"modality":"qa","prompt":"Write QA about scrt concepts.","count":10,"target_tools":[],"passage_shape":"conceptual","rationale":"conceptual mass"}
      ]
    }"#;
    let plan = parse_plan(response).unwrap();
    assert_eq!(plan.specs.len(), 2);
    assert_eq!(plan.specs[0].modality, "tool_call");
    assert_eq!(plan.specs[0].count, 20);
    assert_eq!(plan.specs[0].target_tools, vec!["scrt_stash", "scrt_get_stash"]);
    assert_eq!(plan.strategy, "weight tool_call toward stash workflows");
}

#[test]
fn planner_drops_invalid_modality_specs() {
    let response = r#"{"strategy":"x","specs":[
      {"modality":"telepathy","prompt":"p","count":5},
      {"modality":"cli","prompt":"emit scrt commands","count":5}
    ]}"#;
    let plan = parse_plan(response).unwrap();
    assert_eq!(plan.specs.len(), 1);
    assert_eq!(plan.specs[0].modality, "cli");
}

#[test]
fn plan_driven_generation_uses_spec_prompt_and_routes_modality() {
    // The planner's spec says tool_call; the (mocked) generator returns a valid
    // scrt_stash call. run_plan_with_backend should produce exactly that.
    let plan = GenPlan {
        round: 0,
        strategy: "s".into(),
        specs: vec![GenSpec {
            modality: "tool_call".into(),
            prompt: "PLANNER-WRITTEN: emit scrt_stash calls.".into(),
            count: 1,
            target_tools: vec!["scrt_stash".into()],
            rationale: "r".into(),
            passage_shape: "any".into(),
        }],
    };
    let response =
        r#"[{"prompt":"stash auth","tool":"scrt_stash","arguments":{"name":"auth","note":"n"}}]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let ctx = fixture_ctx();
    let shapes = vec!["cli_ref".to_string(), "code".to_string()];

    let dataset = run_plan_with_backend(&backend, &ctx, &plan, 3, &shapes).unwrap();
    assert_eq!(dataset.len(), 1);
    match &dataset.rows[0] {
        GenExample::ToolCall { tool, .. } => assert_eq!(tool, "scrt_stash"),
        other => panic!("expected tool_call, got {other:?}"),
    }
}

#[test]
fn coverage_measure_counts_modalities_and_tools() {
    let ds = Dataset::new(vec![
        GenExample::Qa { prompt: "q".into(), completion: "a".into(), source: None, gen: None },
        GenExample::ToolCall {
            prompt: "p".into(),
            tool: "scrt_stash".into(),
            arguments: serde_json::json!({"name":"x","note":"y"}),
            source: None,
            gen: None,
        },
    ]);
    let cov = measure(&ds);
    assert_eq!(cov.total, 2);
    assert_eq!(cov.by_modality.get("qa").copied(), Some(1));
    assert_eq!(cov.by_modality.get("tool_call").copied(), Some(1));
    assert_eq!(cov.by_tool.get("scrt_stash").copied(), Some(1));
}

#[test]
fn gap_critic_plans_followup_then_empty_when_balanced() {
    let cfg = scrt_evolve::EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"").unwrap();
    let ctx = fixture_ctx();
    let sig = signals::extract(&cfg, &ctx);
    let tools = scrt_evolve::toolspec::scrt_tools().unwrap();
    let cov = measure(&Dataset::new(vec![]));

    // Round 1: critic finds a gap and plans more tool_call examples.
    let gap_resp = r#"{"strategy":"fill tool_call gap","specs":[
      {"modality":"tool_call","prompt":"emit scrt_search calls","count":10,"target_tools":["scrt_search"],"passage_shape":"any","rationale":"zero search calls produced"}
    ]}"#;
    let t1 = MockTransport::new(vec![gap_resp]);
    let follow = critique_with_transport(&t1, &sig, &cov, &tools, 1).unwrap();
    assert_eq!(follow.round, 1);
    assert_eq!(follow.specs.len(), 1);

    // Round 2: critic returns empty specs → no gap, loop should stop.
    let none_resp = r#"{"strategy":"balanced","specs":[]}"#;
    let t2 = MockTransport::new(vec![none_resp]);
    let done = critique_with_transport(&t2, &sig, &cov, &tools, 2).unwrap();
    assert!(done.specs.is_empty(), "balanced coverage yields no follow-up specs");
}

#[test]
fn planner_runs_end_to_end_with_mock() {
    let cfg = scrt_evolve::EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"").unwrap();
    let ctx = fixture_ctx();
    let sig = signals::extract(&cfg, &ctx);
    let tools = scrt_evolve::toolspec::scrt_tools().unwrap();
    let resp = r#"{"strategy":"s","specs":[{"modality":"cli","prompt":"emit scrt commands","count":3,"target_tools":[],"passage_shape":"any","rationale":"r"}]}"#;
    let t = MockTransport::new(vec![resp]);
    let directive = scrt_evolve::TrainingDirective::default();
    let plan = plan_with_transport(&t, &sig, &ctx, &tools, &directive).unwrap();
    assert_eq!(plan.specs.len(), 1);
    assert_eq!(plan.specs[0].modality, "cli");
}

#[test]
fn directive_assembles_from_answers_and_renders() {
    use scrt_evolve::interview::{assemble_directive, core_questions, Question};
    let qs = core_questions();
    let find = |id: &str| qs.iter().find(|q| q.id == id).unwrap().clone();
    let answers: Vec<(Question, String)> = vec![
        (find("goal"), "tool-calling fluency for memory traversal".into()),
        (find("priorities"), "tool_call, cli".into()),
        (find("must_cover"), "scrt_stash, scrt_get_stash".into()),
        (find("exclusions"), "destructive commands, prose trivia".into()),
        (find("max_rows"), "40".into()),
    ];
    let d = assemble_directive(&answers);
    assert_eq!(d.goal, "tool-calling fluency for memory traversal");
    assert_eq!(d.priorities, vec!["tool_call", "cli"]);
    assert_eq!(d.must_cover, vec!["scrt_stash", "scrt_get_stash"]);
    assert_eq!(d.exclusions, vec!["destructive commands", "prose trivia"]);
    assert_eq!(d.max_rows, 40);

    let block = d.prompt_block();
    assert!(block.contains("GOAL:"));
    assert!(block.contains("MUST COVER"));
    assert!(block.contains("MAX TOTAL ROWS: 40"));
}

#[test]
fn directive_exclusion_matches_text() {
    let d = scrt_evolve::TrainingDirective {
        exclusions: vec!["rm -rf".into(), "destructive".into()],
        ..Default::default()
    };
    assert!(d.excluded("scrt then rm -rf /tmp"));
    assert!(d.excluded("This is a DESTRUCTIVE example"));
    assert!(!d.excluded("scrt --mp-stash auth"));
}

#[test]
fn interview_followups_parse_and_force_notes_field() {
    use scrt_evolve::interview::parse_followups;
    let raw = r#"[
      {"id":"followup_1","text":"Should the empty palace be seeded first?","field":"goal","options":[],"multi":false},
      {"text":"Which tag clusters matter?","field":"priorities","options":[],"multi":false}
    ]"#;
    let qs = parse_followups(raw);
    assert_eq!(qs.len(), 2);
    // Follow-up answers always map to notes, regardless of what the LLM said.
    assert!(qs.iter().all(|q| q.field == "notes"));
    // Missing id is backfilled.
    assert!(!qs[1].id.is_empty());
}
