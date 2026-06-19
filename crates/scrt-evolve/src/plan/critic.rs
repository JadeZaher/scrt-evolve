//! Gap-analysis critic: inspect produced coverage → a follow-up [`GenPlan`].
//!
//! After a generation round, the critic measures what was actually produced
//! (modality mix, tool coverage) against the signals (which tools/workflows
//! matter) and asks the LLM to plan a follow-up round filling the thin spots.

use std::collections::BTreeMap;

use crate::config::EvolveConfig;
use crate::dataset::{Dataset, GenExample};
use crate::discover::DiscoveredContext;
use crate::generate::api::{ChatMessage, ChatTransport, HttpTransport};
use crate::plan::planner::parse_plan;
use crate::plan::signals::{self, Signals};
use crate::plan::GenPlan;
use crate::toolspec;

/// A measurement of what a dataset actually covers.
#[derive(Debug, Clone, Default)]
pub struct Coverage {
    pub by_modality: BTreeMap<String, usize>,
    pub by_tool: BTreeMap<String, usize>,
    pub total: usize,
}

/// Measure coverage of a produced dataset.
pub fn measure(dataset: &Dataset) -> Coverage {
    let mut by_modality: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_tool: BTreeMap<String, usize> = BTreeMap::new();
    for row in &dataset.rows {
        let modality = match row {
            GenExample::Qa { .. } => "qa",
            GenExample::Instruction { .. } => "instruction",
            GenExample::Completion { .. } => "completion",
            GenExample::Contrastive { .. } => "contrastive",
            GenExample::ToolCall { tool, .. } => {
                *by_tool.entry(tool.clone()).or_default() += 1;
                "tool_call"
            }
            GenExample::Cli { .. } => "cli",
        };
        *by_modality.entry(modality.to_string()).or_default() += 1;
    }
    Coverage {
        total: dataset.rows.len(),
        by_modality,
        by_tool,
    }
}

/// Render coverage as a compact summary for the critic prompt.
fn coverage_summary(cov: &Coverage) -> String {
    let modal: Vec<String> = cov.by_modality.iter().map(|(k, v)| format!("{k}={v}")).collect();
    let tools: Vec<String> = cov.by_tool.iter().map(|(k, v)| format!("{k}={v}")).collect();
    format!(
        "total examples: {}\nby modality: {}\ntool_call coverage by tool: {}",
        cov.total,
        modal.join(", "),
        if tools.is_empty() { "(none)".into() } else { tools.join(", ") }
    )
}

fn system_prompt(tools_block: &str) -> String {
    format!(
        "You are a COVERAGE CRITIC for training-data generation. Given the usage \
signals, the tool schemas, and a report of what was ALREADY produced, identify \
GAPS — modalities, tools, or workflows that are under-covered relative to how \
important the signals say they are — and plan a FOLLOW-UP round to fill them.\n\n\
Real tools (ground truth):\n{tools_block}\n\
Output ONLY a JSON object of the SAME shape as a generation plan:\n\
{{ \"strategy\": \"...\", \"specs\": [ {{ \"modality\":..., \"prompt\":..., \
\"count\":..., \"target_tools\":[...], \"passage_shape\":..., \"rationale\":... }} ] }}\n\n\
Rules:\n\
- ONLY plan specs that address a real gap (e.g. a high-co-occurrence tool with \
zero produced examples, or a modality the signals favor but that is thin).\n\
- If coverage is already balanced against the signals, return an empty specs \
array.\n\
- Write concrete, self-contained generation prompts, same as a planner."
    )
}

fn user_prompt(signals: &Signals, cov: &Coverage) -> String {
    format!(
        "USAGE SIGNALS:\n{}\n\nALREADY PRODUCED:\n{}\n\nPlan the follow-up round \
(JSON only). Empty specs if no meaningful gap remains.",
        signals::summary(signals),
        coverage_summary(cov)
    )
}

/// Run the critic against a transport (mockable). Returns a follow-up plan
/// (possibly with empty specs, meaning "no gap").
pub fn critique_with_transport<T: ChatTransport>(
    transport: &T,
    signals: &Signals,
    cov: &Coverage,
    tools: &[toolspec::ToolSchema],
    round: usize,
) -> anyhow::Result<GenPlan> {
    let messages = vec![
        ChatMessage::system(system_prompt(&toolspec::tools_compact_block(tools))),
        ChatMessage::user(user_prompt(signals, cov)),
    ];
    let raw = transport.complete(&messages)?;
    // The critic may legitimately return an empty plan; tolerate that.
    match parse_plan(&raw) {
        Ok(mut plan) => {
            plan.round = round;
            Ok(plan)
        }
        Err(_) => Ok(GenPlan {
            round,
            specs: vec![],
            strategy: "no gap identified".into(),
        }),
    }
}

/// Run the gap critic using the configured API backend.
pub fn run(
    cfg: &EvolveConfig,
    ctx: &DiscoveredContext,
    dataset: &Dataset,
    round: usize,
) -> anyhow::Result<GenPlan> {
    let signals = signals::extract(cfg, ctx);
    let cov = measure(dataset);
    let tools = toolspec::scrt_tools()?;
    let gcfg = cfg.generate.clone().unwrap_or_default();
    let transport = HttpTransport::from_api_config(&gcfg)?;
    critique_with_transport(&transport, &signals, &cov, &tools, round)
}
