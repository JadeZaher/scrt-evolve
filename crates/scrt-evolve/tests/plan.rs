//! Plan-stage tests: plan types round-trip, signal extraction, planner parsing,
//! plan-driven generation with self-written prompts, and the gap critic.

use std::cell::RefCell;

use scrt_evolve::dataset::{Dataset, GenExample, Outcome, Tier, Verdict};
use scrt_evolve::discover::{DiscoveredContext, Passage};
use scrt_evolve::generate::api::{ApiEndpoint, ChatMessage, ChatTransport};
use scrt_evolve::generate::run_plan_with_backend;
use scrt_evolve::plan::critic::{critique_with_transport, measure};
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
    assert!(
        sig.cooccurrence
            .tool_frequency
            .get("scrt_stash")
            .copied()
            .unwrap_or(0)
            >= 1
    );
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
    assert_eq!(
        plan.specs[0].target_tools,
        vec!["scrt_stash", "scrt_get_stash"]
    );
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
        GenExample::Qa {
            prompt: "q".into(),
            completion: "a".into(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::ToolCall {
            prompt: "p".into(),
            tool: "scrt_stash".into(),
            arguments: serde_json::json!({"name":"x","note":"y"}),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
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
    assert!(
        done.specs.is_empty(),
        "balanced coverage yields no follow-up specs"
    );
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
    let domain = scrt_evolve::config::DomainConfig::default();
    let plan = plan_with_transport(&t, &sig, &ctx, &tools, &directive, &domain).unwrap();
    assert_eq!(plan.specs.len(), 1);
    assert_eq!(plan.specs[0].modality, "cli");
}

#[test]
fn directive_assembles_from_answers_and_renders() {
    use scrt_evolve::interview::{assemble_directive, core_questions, Question};
    let qs = core_questions();
    let find = |id: &str| qs.iter().find(|q| q.id == id).unwrap().clone();
    let answers: Vec<(Question, String)> = vec![
        (
            find("goal"),
            "tool-calling fluency for memory traversal".into(),
        ),
        (find("priorities"), "tool_call, cli".into()),
        (find("must_cover"), "scrt_stash, scrt_get_stash".into()),
        (
            find("exclusions"),
            "destructive commands, prose trivia".into(),
        ),
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

/// A transport that records the system message it was handed, so tests can
/// snapshot the planner's system prompt without a live endpoint.
struct CapturingTransport {
    last_system: RefCell<String>,
    response: String,
}
impl CapturingTransport {
    fn new(response: &str) -> Self {
        Self {
            last_system: RefCell::new(String::new()),
            response: response.to_string(),
        }
    }
}
impl ChatTransport for CapturingTransport {
    fn complete(&self, m: &[ChatMessage]) -> anyhow::Result<String> {
        if let Some(sys) = m.iter().find(|c| c.role == "system") {
            *self.last_system.borrow_mut() = sys.content.clone();
        }
        Ok(self.response.clone())
    }
}

// Track 37 Phase C — TASK 1: planner can EMIT the track-09 modalities.
#[test]
fn planner_accepts_skill_and_reasoning_edit_modalities() {
    let response = r#"{"strategy":"cover skills + reasoning","specs":[
      {"modality":"skill","prompt":"teach scrt skills","count":3,"target_tools":[],"passage_shape":"any","rationale":"skill signal"},
      {"modality":"reasoning_edit","prompt":"correct chains","count":2,"target_tools":[],"passage_shape":"any","rationale":"reasoning signal"}
    ]}"#;
    let plan = parse_plan(response).unwrap();
    assert_eq!(plan.specs.len(), 2, "skill + reasoning_edit both accepted");
    assert_eq!(plan.specs[0].modality, "skill");
    assert_eq!(plan.specs[1].modality, "reasoning_edit");
}

// Track 37 Phase C — TASK 1b: `completion` is no longer a plannable modality
// (doc-ingestion emits completion rows directly; planning one used to silently
// degrade to Prose).
#[test]
fn planner_rejects_completion_modality() {
    let response = r#"{"strategy":"x","specs":[
      {"modality":"completion","prompt":"p","count":5,"target_tools":[],"passage_shape":"any","rationale":"r"},
      {"modality":"qa","prompt":"q","count":5,"target_tools":[],"passage_shape":"any","rationale":"r"}
    ]}"#;
    let plan = parse_plan(response).unwrap();
    assert_eq!(plan.specs.len(), 1, "completion dropped, qa kept");
    assert_eq!(plan.specs[0].modality, "qa");
}

// Track 37 Phase C — TASK 1: a planned skill/reasoning_edit spec routes to the
// right GenMode and produces a valid row through run_plan_with_backend.
#[test]
fn plan_driven_skill_routes_to_skill_mode() {
    let plan = GenPlan {
        round: 0,
        strategy: "s".into(),
        specs: vec![GenSpec {
            modality: "skill".into(),
            prompt: "emit a skill row".into(),
            count: 1,
            target_tools: vec![],
            rationale: "r".into(),
            passage_shape: "any".into(),
        }],
    };
    // A valid skill row: named skill + non-empty invocation.
    let response = r#"[{"kind":"skill","prompt":"how do I search?","skill_name":"scrt-context","invocation":"scrt \"auth\" --in .","expected_outcome":"token-budgeted search"}]"#;
    let backend = ApiEndpoint::with_transport(MockTransport::new(vec![response]), 1);
    let ctx = fixture_ctx();
    let shapes = vec!["cli_ref".to_string(), "code".to_string()];
    let dataset = run_plan_with_backend(&backend, &ctx, &plan, 3, &shapes).unwrap();
    assert_eq!(dataset.len(), 1, "skill row routed + validated");
    match &dataset.rows[0] {
        GenExample::Skill { skill_name, .. } => assert_eq!(skill_name, "scrt-context"),
        other => panic!("expected skill row, got {other:?}"),
    }
}

// Track 37 Phase C — TASK 3a: with the DEFAULT domain the planner system prompt
// is stable AND now advertises the skill/reasoning_edit modalities. The job
// line uses the default (`scrt`) description verbatim, so existing configs get
// the same wording they always had.
#[test]
fn default_domain_planner_prompt_is_stable() {
    let cfg = scrt_evolve::EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"").unwrap();
    let ctx = fixture_ctx();
    let sig = signals::extract(&cfg, &ctx);
    let tools = scrt_evolve::toolspec::scrt_tools().unwrap();
    let directive = scrt_evolve::TrainingDirective::default();
    let domain = scrt_evolve::config::DomainConfig::default();

    let resp = r#"{"strategy":"s","specs":[{"modality":"qa","prompt":"p","count":1,"target_tools":[],"passage_shape":"any","rationale":"r"}]}"#;
    let t = CapturingTransport::new(resp);
    let _ = plan_with_transport(&t, &sig, &ctx, &tools, &directive, &domain).unwrap();
    let sys = t.last_system.borrow().clone();

    // Job line uses the default scrt description (byte-identical wording).
    assert!(
        sys.contains(
            "better at USING the `scrt` tool — both as structured tool calls and \
as a CLI — plus understanding its concepts."
        ),
        "default-domain job line must reproduce the historical scrt wording"
    );
    // The new modalities are advertised (TASK 1c).
    assert!(sys.contains("\"skill\":"), "skill modality advertised");
    assert!(
        sys.contains("\"reasoning_edit\":"),
        "reasoning_edit modality advertised"
    );
    assert!(
        sys.contains("tool_call|cli|qa|instruction|skill|reasoning_edit"),
        "output-shape enum lists the new modalities"
    );
}

// Track 37 Phase C — TASK 3a: a CUSTOM domain description replaces the job line
// while leaving the rest of the prompt scaffolding intact.
#[test]
fn custom_domain_description_drives_planner_job_line() {
    let cfg = scrt_evolve::EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"").unwrap();
    let ctx = fixture_ctx();
    let sig = signals::extract(&cfg, &ctx);
    let tools = scrt_evolve::toolspec::scrt_tools().unwrap();
    let directive = scrt_evolve::TrainingDirective::default();
    let domain = scrt_evolve::config::DomainConfig {
        name: "kubectl".into(),
        description: "the `kubectl` CLI and Kubernetes concepts".into(),
        ..Default::default()
    };

    let resp = r#"{"strategy":"s","specs":[{"modality":"qa","prompt":"p","count":1,"target_tools":[],"passage_shape":"any","rationale":"r"}]}"#;
    let t = CapturingTransport::new(resp);
    let _ = plan_with_transport(&t, &sig, &ctx, &tools, &directive, &domain).unwrap();
    let sys = t.last_system.borrow().clone();
    assert!(
        sys.contains("better at USING the `kubectl` CLI and Kubernetes concepts."),
        "custom domain description appears in the job line"
    );
    assert!(
        !sys.contains("the `scrt` tool — both as structured"),
        "the hardcoded scrt wording is gone under a custom domain"
    );
}

// Track 37 Phase C — TASK 3b: the DEFAULT domain yields identical signal counts
// to the historical hardcoded scrt tool set + `--mp-` flag recognizer.
#[test]
fn default_domain_signal_counts_match_baseline() {
    let cfg = scrt_evolve::EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"").unwrap();
    let ctx = fixture_ctx();
    let sig = signals::extract(&cfg, &ctx);
    // scrt_stash counted from the first passage; --mp-stash flag counted.
    assert_eq!(
        sig.cooccurrence.tool_frequency.get("scrt_stash").copied(),
        Some(1)
    );
    assert!(sig.cooccurrence.flag_frequency.contains_key("--mp-stash"));
    // A non-domain tool name is NOT counted.
    assert!(!sig.cooccurrence.tool_frequency.contains_key("kubectl"));
}

// Track 37 Phase C — TASK 3b: a custom domain counts its OWN tool names + flag
// prefixes in the co-occurrence signal.
#[test]
fn custom_domain_counts_its_own_tools_and_flags() {
    let toml = "[evolve]\ncorpus_dir = \".\"\n\
                [domain]\nname=\"kubectl\"\ntools=[\"kubectl_apply\"]\nflag_patterns=[\"-\"]\n";
    let cfg = scrt_evolve::EvolveConfig::from_toml_str(toml).unwrap();
    let ctx = DiscoveredContext {
        passages: vec![Passage {
            text: "run kubectl_apply -f pod.yaml then kubectl_apply -n ns".into(),
            source: "notes.md".into(),
            score: 5.0,
            seed: "corpus:k8s".into(),
        }],
        anchors: vec![],
    };
    let sig = signals::extract(&cfg, &ctx);
    assert_eq!(
        sig.cooccurrence.tool_frequency.get("kubectl_apply").copied(),
        Some(2),
        "custom tool name counted twice"
    );
    // The `-` prefix picks up `-f` / `-n` (generic `--` rule wouldn't).
    assert!(
        sig.cooccurrence.flag_frequency.contains_key("-f")
            || sig.cooccurrence.flag_frequency.contains_key("-n"),
        "custom flag prefix recognizes single-dash flags"
    );
    // The scrt default tool set is NOT counted under a custom domain.
    assert!(!sig.cooccurrence.tool_frequency.contains_key("scrt_stash"));
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
