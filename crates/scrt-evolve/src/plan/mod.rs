//! The `plan` stage — self-routing generation.
//!
//! Instead of a human picking `kinds` and the framework hardcoding a prompt per
//! modality, a **planner LLM** analyzes the discovered context + the palace +
//! usage signals and emits a [`GenPlan`]: a set of [`GenSpec`]s, each carrying
//! its *own* generation prompt, target modality, count, and a rationale. The
//! plan is a durable, inspectable artifact (`work_dir/plan.json`) — auto-routed
//! but fully steerable (read/edit it before generation runs it).
//!
//! Pipeline: `discover → plan → generate(plan) → export`, with an optional
//! gap-analysis critic round that inspects coverage and plans a follow-up.
//!
//! ```text
//!   signals (deterministic)        planner LLM            generator
//!   ┌───────────────────────┐    ┌────────────┐    ┌──────────────────┐
//!   │ palace structure      │ -> │ plan.json  │ -> │ run each GenSpec  │
//!   │ tool/flag co-occurrence│   │ {specs:[…]}│    │ with its own promt│
//!   │ corpus shape          │    └────────────┘    └──────────────────┘
//!   └───────────────────────┘          ↑                    │
//!                                 gap critic  <──────────────┘ (round 2+)
//! ```

pub mod critic;
pub mod planner;
pub mod signals;

use serde::{Deserialize, Serialize};

/// One generation directive: produce `count` examples of `modality`, using the
/// planner's own `prompt`, optionally targeting specific tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenSpec {
    /// Output modality: `qa | instruction | tool_call | cli | completion`.
    pub modality: String,
    /// The generation prompt the planner wrote for this spec. This is the
    /// system-prompt body the generator hands the teacher — the model authored
    /// it, not a hardcoded template.
    pub prompt: String,
    /// How many examples to synthesize for this spec (a budget, best-effort).
    pub count: usize,
    /// For `tool_call` specs: which tools this spec should exercise (names from
    /// scrt's tool schema). Empty = any.
    #[serde(default)]
    pub target_tools: Vec<String>,
    /// Why the planner chose this spec — kept for auditability.
    #[serde(default)]
    pub rationale: String,
    /// Optional: restrict this spec to passages matching this content shape
    /// (`code | cli_ref | conceptual | config | any`). Empty/"any" = all.
    #[serde(default)]
    pub passage_shape: String,
}

/// A generation plan: the full set of specs the planner produced, plus the
/// signal summary it reasoned over (kept for provenance).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenPlan {
    /// Which round produced this plan (0 = initial, 1+ = gap-critic rounds).
    #[serde(default)]
    pub round: usize,
    pub specs: Vec<GenSpec>,
    /// Human/agent-readable note on the overall strategy.
    #[serde(default)]
    pub strategy: String,
}

impl GenPlan {
    pub fn new(specs: Vec<GenSpec>) -> Self {
        Self {
            round: 0,
            specs,
            strategy: String::new(),
        }
    }

    /// Total planned example budget across all specs.
    pub fn total_count(&self) -> usize {
        self.specs.iter().map(|s| s.count).sum()
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn write(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn read(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Merge a follow-up plan (e.g. from the gap critic) into this one,
    /// appending its specs and bumping the round.
    pub fn merge(&mut self, mut other: GenPlan) {
        self.round = self.round.max(other.round);
        self.specs.append(&mut other.specs);
    }
}

use crate::config::EvolveConfig;
use crate::dataset::{Dataset, GenExample};
use crate::directive::TrainingDirective;
use crate::discover::DiscoveredContext;
use crate::generate::ApiEndpoint;

/// Result of a self-routed generation run: the (merged) plan that was executed
/// and the dataset it produced.
pub struct SelfRoutedResult {
    pub plan: GenPlan,
    pub dataset: Dataset,
}

/// The full self-routing pipeline:
/// 1. extract signals (deterministic),
/// 2. planner LLM → initial plan,
/// 3. generate the plan,
/// 4. for up to `gap_rounds`: gap-critic → follow-up plan → generate → merge,
///    stopping early when the critic finds no gap.
///
/// `per_passage` bounds examples per teacher call. Returns the merged plan +
/// the accumulated dataset.
pub fn generate_self_routed(
    cfg: &EvolveConfig,
    ctx: &DiscoveredContext,
    directive: &TrainingDirective,
    gap_rounds: usize,
) -> anyhow::Result<SelfRoutedResult> {
    let gcfg = cfg.generate.clone().unwrap_or_default();
    let per_passage = gcfg.per_passage;

    // Shapes are reused across rounds for the passage_shape filter.
    let sig = signals::extract(cfg, ctx);
    let shapes = sig.corpus_shape.per_passage.clone();

    // Phase 1+2: plan (honoring the human's directive).
    let mut plan = planner::run(cfg, ctx, directive)?;
    plan.round = 0;

    // The generator backend (one configured ApiEndpoint reused per round).
    let backend = ApiEndpoint::from_config(&gcfg)?;

    // Phase 3: generate the initial plan.
    let mut dataset =
        crate::generate::run_plan_with_backend(&backend, ctx, &plan, per_passage, &shapes)?;
    enforce_directive(&mut dataset, directive);

    // Phase 4: gap-critic rounds.
    for round in 1..=gap_rounds {
        // Stop early if the directive's row cap is already met.
        if directive.max_rows > 0 && dataset.rows.len() >= directive.max_rows {
            break;
        }
        let follow = critic::run(cfg, ctx, &dataset, round)?;
        if follow.specs.is_empty() {
            break; // critic found no meaningful gap
        }
        let more =
            crate::generate::run_plan_with_backend(&backend, ctx, &follow, per_passage, &shapes)?;
        dataset.rows.extend(more.rows);
        enforce_directive(&mut dataset, directive);
        plan.merge(follow);
    }

    Ok(SelfRoutedResult { plan, dataset })
}

/// Apply directive guardrails to a produced dataset: drop rows whose text
/// matches a hard exclusion, then truncate to `max_rows` if set. These are the
/// human's hard constraints — enforced regardless of what the planner chose.
fn enforce_directive(dataset: &mut Dataset, directive: &TrainingDirective) {
    if !directive.exclusions.is_empty() {
        dataset
            .rows
            .retain(|row| !directive.excluded(&row_text(row)));
    }
    if directive.max_rows > 0 && dataset.rows.len() > directive.max_rows {
        dataset.rows.truncate(directive.max_rows);
    }
}

/// Flatten a row's human-facing text for exclusion matching.
fn row_text(row: &GenExample) -> String {
    match row {
        GenExample::Qa {
            prompt, completion, ..
        } => format!("{prompt} {completion}"),
        GenExample::Instruction {
            instruction,
            input,
            output,
            ..
        } => {
            format!("{instruction} {input} {output}")
        }
        GenExample::Completion { text, .. } => text.clone(),
        GenExample::Contrastive {
            query, positive, ..
        } => format!("{query} {positive}"),
        GenExample::ToolCall {
            prompt,
            tool,
            arguments,
            ..
        } => {
            format!("{prompt} {tool} {arguments}")
        }
        GenExample::Cli {
            prompt, command, ..
        } => format!("{prompt} {command}"),
    }
}
