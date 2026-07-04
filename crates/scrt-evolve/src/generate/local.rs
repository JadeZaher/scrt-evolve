//! `LocalCandle` — an offline candle text-generation [`GenBackend`].
//!
//! Loads the user's model through the shared [`crate::model`] seam and runs
//! autoregressive sampling over the SAME [`crate::generate::prompts`] templates
//! the [`crate::generate::api::ApiEndpoint`] backend uses, so local and API rows
//! are byte-for-byte schema-identical: both shape their output through
//! [`crate::generate::api::parse_examples`]. The only difference is provenance —
//! local rows are re-stamped `gen="local"` so the dataset stays honest about its
//! lower-trust origin (styleguide §2.4).
//!
//! ## Determinism (styleguide §2.2)
//!
//! candle's CPU RNG is not seedable, so temperature sampling draws from an
//! inlined SplitMix64 keyed by an explicit `seed`. Same model weights + same
//! `seed` + same prompt → identical token stream. No wall-clock entropy.
//!
//! This whole module is gated behind `--features train`; the default build
//! never compiles it (the `local` dispatch arm in the parent module bails).

use std::path::Path;

use candle_core::{IndexOp, Tensor};

use crate::config::GenerateConfig;
use crate::dataset::GenExample;
use crate::generate::api::parse_examples;
use crate::generate::prompts;
use crate::generate::{GenBackend, GenContext, GenMode};
use crate::model::LoadedModel;
use crate::toolspec;

/// Minimum non-whitespace length a generated completion/output must reach to be
/// kept. Shorter rows are treated as degenerate.
const MIN_TEXT_LEN: usize = 3;

/// The local candle generation backend.
///
/// Holds a [`LoadedModel`] plus sampling knobs. Built either from config
/// ([`Self::from_config`], loads weights off disk) or directly from a model
/// ([`Self::from_model`], used by tests with [`LoadedModel::random_fixture`]).
pub struct LocalCandle {
    model: LoadedModel,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
}

impl LocalCandle {
    /// Default sampling seed when config carries none — determinism over
    /// surprise (styleguide §2.2). `[generate.local]` has no seed field; rather
    /// than reach into config we own, we fix it here.
    const DEFAULT_SEED: u64 = 0;

    /// Load the model at `model_path` and read `[generate.local]` knobs
    /// (`max_new_tokens`, `temperature`) with sensible defaults. Loading an
    /// unsupported architecture surfaces the seam's clear error (no panic).
    pub fn from_config(gcfg: &GenerateConfig, model_path: &Path) -> anyhow::Result<Self> {
        let model = LoadedModel::load(model_path)?;
        let local = gcfg.local.clone().unwrap_or_default();
        Ok(Self {
            model,
            max_new_tokens: local.max_new_tokens,
            temperature: local.temperature,
            seed: Self::DEFAULT_SEED,
        })
    }

    /// Construct directly from an already-built model (tests pass a
    /// [`LoadedModel::random_fixture`]).
    pub fn from_model(
        model: LoadedModel,
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Self {
        Self {
            model,
            max_new_tokens,
            temperature,
            seed,
        }
    }

    /// Render the system+user prompt pair for `ctx` (identical to the API
    /// backend's match block) into a single text prompt — the local model has no
    /// chat API, so the two messages are concatenated.
    fn render_prompt(&self, ctx: &GenContext) -> String {
        let (base_system, user) = match ctx.mode {
            GenMode::Prose => (prompts::system_prompt(ctx.kinds), prompts::user_prompt(ctx)),
            GenMode::ToolCall => (
                prompts::tool_call_system_prompt(&toolspec::tools_prompt_block(ctx.tools)),
                prompts::tool_call_user_prompt(ctx),
            ),
            GenMode::Cli => (prompts::cli_system_prompt(), prompts::cli_user_prompt(ctx)),
            GenMode::Skill => (
                prompts::skill_system_prompt(),
                prompts::skill_user_prompt(ctx),
            ),
            GenMode::ReasoningEdit => (
                prompts::reasoning_edit_system_prompt(),
                prompts::reasoning_edit_user_prompt(ctx),
            ),
        };
        let system = match ctx.custom_prompt {
            Some(guidance) => format!(
                "{base_system}\n\n## Additional guidance for this batch (steers \
content, NOT format — the JSON-array schema above is mandatory):\n{guidance}"
            ),
            None => base_system,
        };
        format!("{system}\n\n{user}\n\n")
    }

    /// Autoregressively sample up to `max_new_tokens` from `prompt`, returning
    /// the detokenized GENERATED suffix only (the prompt is not echoed back).
    fn generate_text(&self, prompt: &str) -> anyhow::Result<String> {
        let max_seq = self.model.config.max_seq_len;
        let prompt_ids = self.model.tokenize(prompt)?;
        // Keep room for at least one generated step within the position table.
        let mut ids: Vec<u32> = if prompt_ids.len() >= max_seq {
            prompt_ids[prompt_ids.len() - (max_seq - 1)..].to_vec()
        } else {
            prompt_ids
        };
        let generated_start = ids.len();

        let mut rng = SplitMix64::new(self.seed);
        for _ in 0..self.max_new_tokens {
            if ids.len() >= max_seq {
                break;
            }
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.model.device)?;
            let logits = self.model.forward(&input)?;
            // Last-position logits: [1, seq, vocab] -> [vocab].
            let (_b, seq, _v) = logits.dims3()?;
            let last = logits.i((0, seq - 1))?;
            let row: Vec<f32> = last.to_vec1::<f32>()?;
            let next = sample_token(&row, self.temperature, &mut rng);
            ids.push(next);
        }

        let suffix = &ids[generated_start..];
        Ok(self.model.detokenize(suffix)?)
    }
}

impl GenBackend for LocalCandle {
    fn generate(&self, ctx: &GenContext) -> anyhow::Result<Vec<GenExample>> {
        let prompt = self.render_prompt(ctx);
        let raw = self.generate_text(&prompt)?;
        // Reuse the API parser so rows are schema-identical, then re-stamp
        // provenance and drop degenerate/duplicate output (lower-trust local
        // gen, spec §"dedup + quality filter").
        let rows = parse_examples(&raw, ctx)?;
        let rows = restamp_local(rows);
        Ok(filter_degenerate(rows))
    }
}

/// Replace each row's `gen` provenance with `"local"` (api.rs's `parse_examples`
/// stamps `"api"`). Only the variants that carry a `gen` field are touched.
fn restamp_local(rows: Vec<GenExample>) -> Vec<GenExample> {
    rows.into_iter()
        .map(|row| {
            let local = || Some("local".to_string());
            match row {
                GenExample::Qa {
                    prompt,
                    completion,
                    source,
                    gen: _,
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                } => GenExample::Qa {
                    prompt,
                    completion,
                    source,
                    gen: local(),
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                },
                GenExample::Instruction {
                    instruction,
                    input,
                    output,
                    source,
                    gen: _,
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                } => GenExample::Instruction {
                    instruction,
                    input,
                    output,
                    source,
                    gen: local(),
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                },
                GenExample::ToolCall {
                    prompt,
                    tool,
                    arguments,
                    source,
                    gen: _,
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                } => GenExample::ToolCall {
                    prompt,
                    tool,
                    arguments,
                    source,
                    gen: local(),
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                },
                GenExample::Cli {
                    prompt,
                    command,
                    source,
                    gen: _,
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                } => GenExample::Cli {
                    prompt,
                    command,
                    source,
                    gen: local(),
                    outcome,
                    judge_score,
                    judge_verdict,
                    tier,
                    chosen_over,
                },
                // Completion/Contrastive have no `gen` field — pass through.
                other => other,
            }
        })
        .collect()
}

/// Dedup + basic quality filter for lower-trust local generations.
///
/// Drops rows whose answer text is empty, too short, or degenerate (a single
/// character/token repeated, or the answer equals its own prompt), then removes
/// exact-duplicate rows. Public to the crate so it is unit-testable without
/// having to coax a model into emitting garbage. Public (under `train`) so the
/// integration test exercises it directly.
pub fn filter_degenerate(rows: Vec<GenExample>) -> Vec<GenExample> {
    let mut seen: Vec<GenExample> = Vec::new();
    for row in rows {
        if is_degenerate(&row) {
            continue;
        }
        if seen.contains(&row) {
            continue; // exact duplicate
        }
        seen.push(row);
    }
    seen
}

/// The "answer" side and (where present) the "prompt" side a row carries, used
/// by the quality heuristics. Variants without a free-text answer are never
/// considered degenerate here.
fn answer_and_prompt(row: &GenExample) -> Option<(&str, Option<&str>)> {
    match row {
        GenExample::Qa {
            completion, prompt, ..
        } => Some((completion, Some(prompt))),
        GenExample::Instruction {
            output,
            instruction,
            ..
        } => Some((output, Some(instruction))),
        GenExample::Cli {
            command, prompt, ..
        } => Some((command, Some(prompt))),
        GenExample::Completion { text, .. } => Some((text, None)),
        // ToolCall's answer is structured args (validated by parse_examples);
        // Contrastive is not model prose. Neither is text-degenerate here.
        _ => None,
    }
}

/// True if the row's answer is empty, too short, a repeated single char, or an
/// echo of its own prompt.
fn is_degenerate(row: &GenExample) -> bool {
    let (answer, prompt) = match answer_and_prompt(row) {
        Some(pair) => pair,
        None => return false,
    };
    let trimmed = answer.trim();
    if trimmed.chars().count() < MIN_TEXT_LEN {
        return true;
    }
    // Single distinct character repeated (e.g. "aaaaaa", "      ").
    let mut chars = trimmed.chars();
    if let Some(first) = chars.next() {
        if chars.all(|c| c == first) {
            return true;
        }
    }
    // Answer is just an echo of the prompt.
    if let Some(p) = prompt {
        if !p.trim().is_empty() && trimmed == p.trim() {
            return true;
        }
    }
    false
}

/// Greedy (temperature == 0) or seeded temperature sampling of the next token
/// id from a vocab-length logits row.
fn sample_token(logits: &[f32], temperature: f32, rng: &mut SplitMix64) -> u32 {
    if temperature <= 0.0 || logits.is_empty() {
        return argmax(logits);
    }
    // Softmax over scaled logits, numerically stabilized by the max.
    let inv_t = 1.0 / temperature;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - max) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if !(sum.is_finite()) || sum <= 0.0 {
        return argmax(logits);
    }
    for p in &mut probs {
        *p /= sum;
    }
    // Inverse-CDF draw from the seeded uniform stream.
    let target = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if target < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Index of the maximum logit (ties go to the lowest index).
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

/// A self-contained SplitMix64 PRNG — inlined so sampling determinism does not
/// depend on any external RNG crate's value stability (mirrors model.rs).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1) from the top 24 bits.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
}
