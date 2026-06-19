//! The planner LLM: signals + tool schemas + sample passages → [`GenPlan`].
//!
//! The planner decides *what modalities to generate, in what proportion, with
//! what prompts* — it writes its own per-spec generation prompts. The output is
//! `plan.json`, inspectable and editable before the generator runs it.

use crate::config::EvolveConfig;
use crate::directive::TrainingDirective;
use crate::discover::DiscoveredContext;
use crate::generate::api::{ChatMessage, ChatTransport, HttpTransport};
use crate::generate::ApiEndpoint;
use crate::plan::signals::{self, Signals};
use crate::plan::{GenPlan, GenSpec};
use crate::toolspec;

/// Build the planner system prompt: explain the job, the available modalities,
/// the tool schemas, the human's training directive, and the required output
/// shape. The directive is the human's stated INTENT and OVERRIDES pure signal
/// inference where they conflict.
fn system_prompt(tools_block: &str, directive_block: &str) -> String {
    format!(
        "You are a TRAINING-DATA PLANNER. Your job is to decide what supervised \
fine-tuning data to generate so a model becomes better at USING the `scrt` \
tool — both as structured tool calls and as a CLI — plus understanding its \
concepts.\n\n\
You are given the HUMAN'S TRAINING DIRECTIVE, usage signals (palace structure, \
tool/flag co-occurrence, corpus shape), and the real tool schemas. The DIRECTIVE \
is the human's intent and TAKES PRECEDENCE over signal inference when they \
conflict: honor the modality priority, budget specs for every MUST-COVER item \
even if the corpus rarely mentions it, match the audience/tone, respect the \
MAX ROWS cap, and NEVER plan anything the exclusions forbid.\n\n\
TRAINING DIRECTIVE:\n{directive_block}\n\
From these, you DECIDE the mix of training data and WRITE the generation prompt \
for each part.\n\n\
Available modalities:\n\
- \"tool_call\": structured function calls (the model emits a tool name + JSON \
args). Best for high-co-occurrence tool workflows.\n\
- \"cli\": runnable `scrt …` command lines. Best for CLI-reference content.\n\
- \"qa\": question/answer prose. Best for conceptual content.\n\
- \"instruction\": instruction/output prose.\n\n\
Real tools (ground truth — never invent tools/params):\n{tools_block}\n\
Output ONLY a JSON object, no prose/markdown, of this exact shape:\n\
{{\n  \"strategy\": \"<one sentence on your overall plan>\",\n  \"specs\": [\n\
    {{\n      \"modality\": \"tool_call|cli|qa|instruction\",\n\
      \"prompt\": \"<CONTENT GUIDANCE for this batch: WHAT topics/tools/scenarios \
to cover and how to ground them in the passage. Do NOT specify output format — \
the generator already enforces a strict JSON-array schema. Write guidance, not \
format rules.>\",\n\
      \"count\": <int, how many examples>,\n\
      \"target_tools\": [\"scrt_stash\", …],   // only for tool_call, else []\n\
      \"passage_shape\": \"code|cli_ref|conceptual|config|any\",\n\
      \"rationale\": \"<why — cite the signal that motivated it>\"\n    }}\n  ]\n}}\n\n\
Rules:\n\
- Let the SIGNALS drive the mix: weight tool_call toward high co-occurrence \
workflows; weight cli toward cli_ref-heavy corpora; use qa for conceptual mass.\n\
- Write prompts that are concrete and self-contained (the generator will use \
your prompt verbatim as the teacher's system instruction).\n\
- For tool_call specs, name the target_tools from the schemas above.\n\
- Keep total count reasonable (a few specs, tens to low-hundreds of examples)."
    )
}

/// Build the planner user message from the signal summary + a few short sample
/// passages so it can ground its prompt-writing in real content. Kept compact
/// to stay within modest context windows (the planner reasons over the SIGNAL
/// SUMMARY, not the full corpus — samples are just flavor).
fn user_prompt(signals: &Signals, ctx: &DiscoveredContext) -> String {
    let mut samples = String::new();
    for (i, p) in ctx.passages.iter().take(3).enumerate() {
        samples.push_str(&format!(
            "[sample {i} | shape={} | {}]\n{}\n\n",
            signals.corpus_shape.per_passage.get(i).cloned().unwrap_or_default(),
            p.source,
            p.text.chars().take(180).collect::<String>()
        ));
    }
    format!(
        "USAGE SIGNALS:\n{}\n\nSAMPLE PASSAGES (of {} discovered):\n{}\n\
Produce the generation plan JSON now.",
        signals::summary(signals),
        ctx.passages.len(),
        samples
    )
}

/// Parse the planner's JSON object into a [`GenPlan`]. Tolerant of a markdown
/// fence wrapper; validates modalities and drops malformed specs.
pub fn parse_plan(raw: &str) -> anyhow::Result<GenPlan> {
    let json = extract_json_object(raw);
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("planner: response was not a JSON object: {e}\nraw: {raw}"))?;

    let strategy = value
        .get("strategy")
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .to_string();

    let mut specs = Vec::new();
    if let Some(arr) = value.get("specs").and_then(|s| s.as_array()) {
        for v in arr {
            match serde_json::from_value::<GenSpec>(v.clone()) {
                Ok(spec) if is_valid_modality(&spec.modality) && spec.count > 0 => {
                    specs.push(spec)
                }
                _ => continue,
            }
        }
    }
    if specs.is_empty() {
        anyhow::bail!("planner: produced no valid specs\nraw: {raw}");
    }
    Ok(GenPlan { round: 0, specs, strategy })
}

fn is_valid_modality(m: &str) -> bool {
    matches!(m, "qa" | "instruction" | "tool_call" | "cli" | "completion")
}

fn extract_json_object(raw: &str) -> &str {
    let t = raw.trim();
    if let (Some(start), Some(end)) = (t.find('{'), t.rfind('}')) {
        if start <= end {
            return &t[start..=end];
        }
    }
    t
}

/// Run the planner against a chat transport (mockable). Public for testing.
/// `directive` is the human's intent; pass an empty directive for pure
/// signal-driven planning.
pub fn plan_with_transport<T: ChatTransport>(
    transport: &T,
    signals: &Signals,
    ctx: &DiscoveredContext,
    tools: &[toolspec::ToolSchema],
    directive: &TrainingDirective,
) -> anyhow::Result<GenPlan> {
    let messages = vec![
        ChatMessage::system(system_prompt(
            &toolspec::tools_compact_block(tools),
            &directive.prompt_block(),
        )),
        ChatMessage::user(user_prompt(signals, ctx)),
    ];
    let raw = transport.complete(&messages)?;
    parse_plan(&raw)
}

/// Run the planner using the configured API backend, honoring `directive`.
pub fn run(
    cfg: &EvolveConfig,
    ctx: &DiscoveredContext,
    directive: &TrainingDirective,
) -> anyhow::Result<GenPlan> {
    let signals = signals::extract(cfg, ctx);
    let tools = toolspec::scrt_tools()?;
    let gcfg = cfg.generate.clone().unwrap_or_default();
    let transport = HttpTransport::from_api_config(&gcfg)?;
    plan_with_transport(&transport, &signals, ctx, &tools, directive)
}

// Re-export a small constructor so planner/critic can build an HttpTransport
// from config without going through ApiEndpoint.
impl HttpTransport {
    /// Build directly from `[generate]` config (same resolution as
    /// [`ApiEndpoint::from_config`]).
    pub fn from_api_config(gcfg: &crate::config::GenerateConfig) -> anyhow::Result<Self> {
        // Reuse ApiEndpoint's resolution, then extract the transport.
        let endpoint = ApiEndpoint::from_config(gcfg)?;
        Ok(endpoint.into_transport())
    }
}
