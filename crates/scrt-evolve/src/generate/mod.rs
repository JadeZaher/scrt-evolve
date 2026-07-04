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
    /// `skill` skill-ingestion rows (track 09, opt-in).
    Skill,
    /// `reasoning_edit` reasoning-trace evolution rows (track 09, opt-in).
    ReasoningEdit,
}

/// Context handed to a backend to synthesize examples from one passage in one
/// mode.
pub struct GenContext<'a> {
    /// The source passage to synthesize examples from.
    pub passage: &'a Passage,
    /// Which synthesis strategy this generation call uses.
    pub mode: GenMode,
    /// The prose kinds requested (only meaningful for [`GenMode::Prose`]).
    pub kinds: &'a [String],
    /// How many examples to request per passage per mode.
    pub per_passage: usize,
    /// scrt tool schemas, present for [`GenMode::ToolCall`].
    pub tools: &'a [ToolSchema],
    /// A planner-authored system prompt to use verbatim instead of a built-in
    /// template. Present for plan-driven generation; the `mode` still selects
    /// how the response is parsed/validated.
    pub custom_prompt: Option<&'a str>,
    /// Domain command prefixes a `cli` row must start with (track 37 Phase C).
    /// EMPTY ⇒ the built-in `["scrt"]` default (behavior-identical). See
    /// `src/generate/AGENTS.md` §domain and `default_cli_prefixes`.
    pub command_prefixes: &'a [String],
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
    // Opt-in modalities (track 09): absent from `kinds` ⇒ never planned, so the
    // pipeline is byte-identical to today unless the user asks for them.
    if kinds.iter().any(|k| k == "skill") {
        modes.push(GenMode::Skill);
    }
    if kinds.iter().any(|k| k == "reasoning_edit") {
        modes.push(GenMode::ReasoningEdit);
    }
    modes
}

/// Run generation over a discovered context, producing a dataset.
pub fn run(cfg: &EvolveConfig, ctx: &DiscoveredContext) -> anyhow::Result<Dataset> {
    let gcfg = cfg.generate.clone().unwrap_or_default();
    // Compose the active constitution + taste into a steering prompt that is
    // injected into every generated batch. None ⇒ today's built-in templates.
    // This is what makes [evolve]/[[goals]] constitution+taste actually shape
    // the dataset (and downstream training).
    let steering = cfg.compose_steering();
    let steer = steering.as_deref();
    // Track 37 Phase C: the domain's cli command prefixes (default `["scrt"]`)
    // and rejection-sampling fan-out flow from config into generation.
    let domain = cfg.domain.clone().unwrap_or_default();
    let prefixes = domain.command_prefixes.clone();
    match gcfg.backend.as_str() {
        "api" => {
            let backend = ApiEndpoint::from_config(&gcfg)?;
            // Rejection sampling (Phase C): only when candidates_per_seed>1 AND a
            // judge is configured. The judge reuses the same [generate.api]
            // endpoint. Absent either ⇒ single pass (byte-identical to today).
            let n = gcfg.candidates_per_seed.max(1);
            if n > 1 && cfg.judge.is_some() {
                let jc = cfg.judge.clone().unwrap_or_default();
                let judge = crate::judge::LlmPairJudge::new(
                    ApiEndpoint::from_config(&gcfg)?.into_transport(),
                    jc.batch,
                    crate::judge::OnError::from_config(Some(&jc.on_error)),
                );
                run_with_backend_sampled(
                    &backend,
                    ctx,
                    &gcfg.kinds,
                    gcfg.per_passage,
                    &prefixes,
                    &RejectionSampling {
                        candidates_per_seed: n,
                        min_score: jc.min_score,
                        judge: Some(&judge),
                        steering: steer,
                    },
                )
            } else {
                run_with_backend_domained(
                    &backend,
                    ctx,
                    &gcfg.kinds,
                    gcfg.per_passage,
                    steer,
                    &prefixes,
                )
            }
        }
        "local" => {
            #[cfg(feature = "train")]
            {
                let model_path = cfg.require_model_path("generate")?;
                let backend = local::LocalCandle::from_config(&gcfg, model_path)?;
                run_with_backend_domained(
                    &backend,
                    ctx,
                    &gcfg.kinds,
                    gcfg.per_passage,
                    steer,
                    &prefixes,
                )
            }
            #[cfg(not(feature = "train"))]
            {
                let _ = (steer, &prefixes);
                anyhow::bail!("generate: backend=\"local\" requires the `train` feature (track 03)")
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
    run_with_backend_steered(backend, ctx, kinds, per_passage, None)
}

/// Like [`run_with_backend`] but injects an optional `steering` system prompt
/// (composed constitution + taste) into every batch via `custom_prompt`. The
/// `steering` is layered as additional guidance on top of each mode's built-in
/// template (see the backend's `custom_prompt` handling), so values + taste
/// shape the generated rows without replacing the modality scaffolding.
pub fn run_with_backend_steered<B: GenBackend>(
    backend: &B,
    ctx: &DiscoveredContext,
    kinds: &[String],
    per_passage: usize,
    steering: Option<&str>,
) -> anyhow::Result<Dataset> {
    // Empty prefixes ⇒ built-in `["scrt"]` cli validation (back-compat).
    run_with_backend_domained(backend, ctx, kinds, per_passage, steering, &[])
}

/// Rejection-sampling (best-of-N) knobs for generation (track 37 Phase C). When
/// `candidates_per_seed > 1` AND `judge` is `Some`, each (passage, mode) is
/// generated N times and the pooled candidates are judge-ranked, keeping the
/// top-`per_passage` above `min_score`. `candidates_per_seed <= 1` or `judge ==
/// None` ⇒ a single pass (today's behavior, byte-identical).
pub struct RejectionSampling<'a> {
    /// How many candidate batches to generate per (passage, mode) before ranking.
    pub candidates_per_seed: usize,
    /// Minimum judge score a candidate must meet to be kept.
    pub min_score: f32,
    /// The judge used to rank candidates; `None` ⇒ single-pass (no sampling).
    pub judge: Option<&'a dyn crate::judge::PairJudge>,
    /// Optional steering prompt injected as `custom_prompt` into each candidate call.
    pub steering: Option<&'a str>,
}

impl Default for RejectionSampling<'_> {
    fn default() -> Self {
        Self {
            candidates_per_seed: 1,
            min_score: 0.5,
            judge: None,
            steering: None,
        }
    }
}

/// Like [`run_with_backend_steered`] but with the domain's cli `command_prefixes`
/// (track 37 Phase C). Empty ⇒ the built-in `["scrt"]` default.
pub fn run_with_backend_domained<B: GenBackend>(
    backend: &B,
    ctx: &DiscoveredContext,
    kinds: &[String],
    per_passage: usize,
    steering: Option<&str>,
    command_prefixes: &[String],
) -> anyhow::Result<Dataset> {
    run_with_backend_sampled(
        backend,
        ctx,
        kinds,
        per_passage,
        command_prefixes,
        &RejectionSampling {
            steering,
            ..Default::default()
        },
    )
}

/// Like [`run_with_backend_domained`] but with rejection sampling (track 37
/// Phase C). `rs.candidates_per_seed <= 1` or `rs.judge == None` ⇒ single pass.
pub fn run_with_backend_sampled<B: GenBackend>(
    backend: &B,
    ctx: &DiscoveredContext,
    kinds: &[String],
    per_passage: usize,
    command_prefixes: &[String],
    rs: &RejectionSampling,
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

    let n = rs.candidates_per_seed.max(1);
    let do_sample = n > 1 && rs.judge.is_some();

    let mut rows = Vec::new();
    for passage in &ctx.passages {
        for &mode in &modes {
            let gctx = GenContext {
                passage,
                mode,
                kinds: &prose_kinds,
                per_passage,
                tools: &tools,
                custom_prompt: rs.steering,
                command_prefixes,
            };
            if !do_sample {
                match backend.generate(&gctx) {
                    Ok(mut examples) => rows.append(&mut examples),
                    Err(e) => eprintln!(
                        "generate: skipping {mode:?} for passage from {} — {e}",
                        passage.source
                    ),
                }
                continue;
            }
            // Rejection sampling: pool N candidate batches, judge-rank, keep
            // top-`per_passage` above min_score (stamped `gen=rsample:<n>`).
            let mut pool: Vec<GenExample> = Vec::new();
            for _ in 0..n {
                match backend.generate(&gctx) {
                    Ok(mut ex) => pool.append(&mut ex),
                    Err(e) => eprintln!(
                        "generate: rsample skip {mode:?} for {} — {e}",
                        passage.source
                    ),
                }
            }
            if pool.is_empty() {
                continue;
            }
            let group = pool.len();
            let judge = match rs.judge {
                Some(j) => j,
                None => continue, // do_sample guard already checked is_some; never reached
            };
            match crate::judge::rejection_sample(
                judge,
                pool,
                group,
                per_passage,
                rs.min_score,
                rs.steering,
            ) {
                Ok(mut kept) => rows.append(&mut kept),
                Err(e) => eprintln!("generate: rsample judge failed for {} — {e}", passage.source),
            }
        }
    }
    Ok(Dataset::new(rows))
}

/// Map a plan modality string to a [`GenMode`] for response parsing/validation.
/// Valid modalities are gated upstream by `planner::is_valid_modality`, so the
/// arms here are exhaustive over what the planner can emit. `qa`/`instruction`
/// both route to Prose. The final arm is a defensive fallback (Prose) rather
/// than a silent degrade — it debug-asserts to catch any modality that slipped
/// past the planner gate (e.g. a hand-edited plan.json). See `src/plan/AGENTS.md`.
fn mode_for_modality(modality: &str) -> GenMode {
    match modality {
        "qa" | "instruction" => GenMode::Prose,
        "tool_call" => GenMode::ToolCall,
        "cli" => GenMode::Cli,
        "skill" => GenMode::Skill,
        "reasoning_edit" => GenMode::ReasoningEdit,
        other => {
            debug_assert!(
                false,
                "mode_for_modality: unexpected modality {other:?} (should be gated \
                 by planner::is_valid_modality); defaulting to Prose"
            );
            GenMode::Prose
        }
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
    // Empty ⇒ built-in `["scrt"]` cli validation. TODO(track37): caller threads
    // `cfg.domain.command_prefixes` here (signature frozen for back-compat).
    let command_prefixes: &[String] = &[];

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
                && shape_of_passage
                    .get(i)
                    .map(|s| s != &spec.passage_shape)
                    .unwrap_or(false)
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
                command_prefixes,
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
