//! Generate stage — `GenBackend` trait + the `generate()` driver.
//!
//! The `ApiEndpoint` backend hits any OpenAI-compatible chat-completions
//! endpoint (OpenAI / Anthropic-compatible / LM Studio / vLLM …) to synthesize
//! supervised examples from discovered passages. The local candle backend
//! (track 03) plugs into the same trait. Both produce the same [`GenExample`]
//! rows, so the dataset is backend-agnostic.
//!
//! Generation is **mode-driven** by `[generate].kinds`:
//! - `qa` / `instruction` → prose supervised pairs (one teacher call covers both).
//! - `tool_call` → structured function-call rows grounded in scrt's real tool
//!   schemas ([`crate::toolspec`]).
//! - `cli` → runnable `scrt …` command lines.

pub mod api;
#[cfg(feature = "train")]
pub mod local;
pub mod prompts;

pub use api::{ApiEndpoint, ChatTransport};

use crate::config::EvolveConfig;
use crate::dataset::{Dataset, GenExample};
use crate::discover::{DiscoveredContext, Passage};
use crate::toolspec::ToolSchema;

/// Which synthesis strategy a set of kinds maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenMode {
    /// `qa` and/or `instruction` prose pairs.
    Prose,
    /// `tool_call` structured function calls.
    ToolCall,
    /// `cli` runnable command lines.
    Cli,
}

/// Context handed to a backend to synthesize examples from one passage in one
/// mode.
pub struct GenContext<'a> {
    pub passage: &'a Passage,
    pub mode: GenMode,
    /// The prose kinds requested (only meaningful for [`GenMode::Prose`]).
    pub kinds: &'a [String],
    pub per_passage: usize,
    /// scrt tool schemas, present for [`GenMode::ToolCall`].
    pub tools: &'a [ToolSchema],
    /// A planner-authored system prompt to use verbatim instead of a built-in
    /// template. Present for plan-driven generation; the `mode` still selects
    /// how the response is parsed/validated.
    pub custom_prompt: Option<&'a str>,
}

/// Where synthetic training data comes from.
pub trait GenBackend {
    /// Turn one context passage into N supervised examples for one mode.
    fn generate(&self, ctx: &GenContext) -> anyhow::Result<Vec<GenExample>>;
}

/// Plan the modes to run from the configured `kinds`. `qa`/`instruction`
/// collapse into a single Prose pass that emits both; `tool_call` and `cli` are
/// their own passes.
pub fn plan_modes(kinds: &[String]) -> Vec<GenMode> {
    let mut modes = Vec::new();
    let prose: Vec<String> = kinds
        .iter()
        .filter(|k| matches!(k.as_str(), "qa" | "instruction"))
        .cloned()
        .collect();
    if !prose.is_empty() {
        modes.push(GenMode::Prose);
    }
    if kinds.iter().any(|k| k == "tool_call") {
        modes.push(GenMode::ToolCall);
    }
    if kinds.iter().any(|k| k == "cli") {
        modes.push(GenMode::Cli);
    }
    modes
}

/// Run generation over a discovered context, producing a dataset.
pub fn run(cfg: &EvolveConfig, ctx: &DiscoveredContext) -> anyhow::Result<Dataset> {
    let gcfg = cfg.generate.clone().unwrap_or_default();
    match gcfg.backend.as_str() {
        "api" => {
            let backend = ApiEndpoint::from_config(&gcfg)?;
            run_with_backend(&backend, ctx, &gcfg.kinds, gcfg.per_passage)
        }
        "local" => {
            #[cfg(feature = "train")]
            {
                let model_path = cfg.require_model_path("generate")?;
                let backend = local::LocalCandle::from_config(&gcfg, model_path)?;
                run_with_backend(&backend, ctx, &gcfg.kinds, gcfg.per_passage)
            }
            #[cfg(not(feature = "train"))]
            {
                anyhow::bail!(
                    "generate: backend=\"local\" requires the `train` feature (track 03)"
                )
            }
        }
        other => anyhow::bail!("generate: unknown backend \"{other}\" (expected api | local)"),
    }
}

/// Drive any backend over a discovered context. Public so callers (and tests)
/// can supply a custom/mocked backend. Loads scrt's tool schemas once if any
/// requested kind needs them.
pub fn run_with_backend<B: GenBackend>(
    backend: &B,
    ctx: &DiscoveredContext,
    kinds: &[String],
    per_passage: usize,
) -> anyhow::Result<Dataset> {
    let modes = plan_modes(kinds);
    let prose_kinds: Vec<String> = kinds
        .iter()
        .filter(|k| matches!(k.as_str(), "qa" | "instruction"))
        .cloned()
        .collect();

    // Tool schemas are needed only if tool_call is requested.
    let tools = if modes.contains(&GenMode::ToolCall) {
        crate::toolspec::scrt_tools()?
    } else {
        Vec::new()
    };

    let mut rows = Vec::new();
    for passage in &ctx.passages {
        for &mode in &modes {
            let gctx = GenContext {
                passage,
                mode,
                kinds: &prose_kinds,
                per_passage,
                tools: &tools,
                custom_prompt: None,
            };
            match backend.generate(&gctx) {
                Ok(mut examples) => rows.append(&mut examples),
                Err(e) => eprintln!(
                    "generate: skipping {mode:?} for passage from {} — {e}",
                    passage.source
                ),
            }
        }
    }
    Ok(Dataset::new(rows))
}

/// Map a plan modality string to a [`GenMode`] for response parsing/validation.
fn mode_for_modality(modality: &str) -> GenMode {
    match modality {
        "tool_call" => GenMode::ToolCall,
        "cli" => GenMode::Cli,
        _ => GenMode::Prose,
    }
}

/// Execute a planner-produced [`GenPlan`](crate::plan::GenPlan): for each spec,
/// run its self-written prompt over passages (optionally filtered by the spec's
/// `passage_shape`), until the spec's `count` budget is met. The spec prompt is
/// used verbatim as the teacher's system instruction; `mode` still governs how
/// the response is parsed and validated.
pub fn run_plan_with_backend<B: GenBackend>(
    backend: &B,
    ctx: &DiscoveredContext,
    plan: &crate::plan::GenPlan,
    per_passage: usize,
    shape_of_passage: &[String],
) -> anyhow::Result<Dataset> {
    let tools = if plan.specs.iter().any(|s| s.modality == "tool_call") {
        crate::toolspec::scrt_tools()?
    } else {
        Vec::new()
    };
    let prose_kinds = ["qa".to_string(), "instruction".to_string()];

    let mut rows = Vec::new();
    for spec in &plan.specs {
        let mode = mode_for_modality(&spec.modality);
        let want = spec.count;
        let mut produced = 0usize;

        for (i, passage) in ctx.passages.iter().enumerate() {
            if produced >= want {
                break;
            }
            // Respect the spec's passage_shape filter when set.
            if !spec.passage_shape.is_empty()
                && spec.passage_shape != "any"
                && shape_of_passage.get(i).map(|s| s != &spec.passage_shape).unwrap_or(false)
            {
                continue;
            }

            let gctx = GenContext {
                passage,
                mode,
                kinds: &prose_kinds,
                per_passage,
                tools: &tools,
                custom_prompt: Some(&spec.prompt),
            };
            match backend.generate(&gctx) {
                Ok(examples) => {
                    produced += examples.len();
                    rows.extend(examples);
                }
                Err(e) => eprintln!(
                    "generate(plan): skipping spec '{}' for passage from {} — {e}",
                    spec.modality, passage.source
                ),
            }
        }
    }
    Ok(Dataset::new(rows))
}
