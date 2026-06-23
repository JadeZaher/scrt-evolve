//! `EvolveConfig` — the `evolve.toml` schema.
//!
//! One top `[evolve]` section + per-stage (`[discover]`, `[generate]`,
//! `[train]`) + per-preset sub-blocks. Every stage reads only what it needs,
//! so **partial configs work** (generate-only, train-only): each stage block
//! is `Option`, and the per-stage `model_path` requirement is enforced only
//! when that stage actually runs (see [`EvolveConfig::require_model_path`]).
//!
//! The schema mirrors DESIGN.md §Config schema field-for-field.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Errors surfaced while loading/validating an `evolve.toml`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error(
        "`{field}` looks like an inline secret. It must be the NAME of an \
         environment variable (e.g. \"SCRT_EVOLVE_API_KEY\"), never the key \
         itself — the framework reads the named env var at runtime."
    )]
    InlineSecret { field: &'static str },
    #[error("`model_path` is required for the `{stage}` stage but was not set in [evolve]")]
    MissingModelPath { stage: &'static str },
}

/// Top-level config: `[evolve]` + the three optional stage blocks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolveConfig {
    #[serde(default)]
    pub evolve: EvolveSection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discover: Option<DiscoverConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<GenerateConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train: Option<TrainConfig>,
    /// `[eval]` — the shared evaluation harness (track 10). Top-level, like the
    /// other stage blocks (`[discover]`/`[generate]`/`[train]`). Absent ⇒ the
    /// lane runs **unguarded** (a logged warning); present ⇒ rounds are scored
    /// against a held-out probe set and gated by [`crate::eval::StepVerdict`].
    /// Additive + non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<EvalConfig>,
    /// `[regulate]` — the self-regulation / transactional homeostasis layer
    /// (track 15). Makes every weight-mutating step `checkpoint → apply → eval →
    /// keep|rollback`, with catastrophe → rollback+quarantine+halt. Absent ⇒ no
    /// transaction wrapper (steps run unguarded — a logged warning). Additive +
    /// non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regulate: Option<RegulateConfig>,
    /// `[hardware]` — the compute environment for the heavy ML subprocesses.
    /// Generic + architecture-level: declares the target device, available VRAM,
    /// and which acceleration kernels are present, so the pipeline can route /
    /// warn appropriately (e.g. a hybrid-SSM model needs CUDA + mamba kernels to
    /// train; CPU forward-only is fine for eval/teacher). Absent ⇒ auto/CPU
    /// defaults. Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware: Option<HardwareConfig>,
    /// `[export]` — the config-driven model-export pipeline (merge sharded
    /// adapter → convert → quantize → place). Absent ⇒ `export-gguf` uses its
    /// CLI-flag defaults. Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportConfig>,
    /// `[runtime]` — the inference runtime (load + run a model for generation,
    /// backend-generic: GGUF via llama.cpp, or HF via transformers). Absent ⇒
    /// `infer` uses the transformers fallback. Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeConfig>,
    /// Learning-by-doing **goals** (track 20). Each `[[goals]]` table declares
    /// something a local model should evolve toward and how its traces are
    /// captured (topic ⇄ palace search, tag ⇄ palace tag). Additive + non-
    /// breaking: an absent/empty `goals` reproduces today's single-run
    /// behavior (styleguide §1). Goals drive the per-goal discover→generate
    /// pipeline; eval-gated rounds + the scheduler are lane-gated (tracks
    /// 10/15) and not yet wired.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub goals: Vec<GoalConfig>,
}

/// One learning-by-doing goal (`[[goals]]` in `evolve.toml`).
///
/// A goal declares *what to evolve toward* and *how its traces are captured*.
/// The contract is **one goal ⇄ one tag**: the paired `scrt-evolve` skill
/// stamps goal-relevant palace stashes with `tag`, and discovery pulls exactly
/// those (`palace_tags = [tag]`, `palace_search = topic`). All fields beyond
/// the three identifiers are optional scheduler/eval hints, consumed by the
/// lane-gated round driver (tracks 10/15) when it lands.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalConfig {
    /// Stable id, e.g. `scrt-cli-fluency`. Used to namespace per-goal
    /// artifacts (`work_dir/traces/<name>/`, the `gen=trace:<name>` stamp).
    pub name: String,
    /// The subject to evolve toward — feeds `discover.palace_search` and scopes
    /// the corpus sweep.
    pub topic: String,
    /// The palace tag the skill stamps on goal-relevant stashes — feeds
    /// `discover.palace_tags`. One goal ⇄ one tag.
    pub tag: String,
    /// Optional project directory scoping the corpus/transcripts to one project.
    /// When set, discover runs against this project's corpus instead of the
    /// top-level `[evolve].corpus_dir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<PathBuf>,
    /// Optional path to this goal's held-out eval probes (consumed by track
    /// 10's `Scorer` to gate the goal's rounds). Lane-gated; unused until the
    /// eval harness lands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_set: Option<PathBuf>,
    /// Optional scheduler weight (priority hint for round-robin vs weighted
    /// scheduling). `None` ⇒ equal weight. Lane-gated (the scheduler is
    /// slice 9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    /// Optional scheduler cadence hint (e.g. `"1h"`, `"daily"`). `None` ⇒
    /// on-demand. Lane-gated (the scheduler is slice 9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cadence: Option<String>,
    /// Per-goal constitution override/addition — values specific to this goal,
    /// layered on top of the global `[evolve].constitution`. Composed into the
    /// goal's generate system prompt (the steering seam).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constitution: Option<String>,
    /// Per-goal taste override/addition — representational form specific to this
    /// goal, layered on the global `[evolve].taste`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taste: Option<String>,
}

/// `[evolve]` — the top section. `model_path` is the one thing most stages
/// need, but it is kept optional at parse time so a stage that doesn't need a
/// model (e.g. a future inspect-only command) can load a config without it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolveSection {
    /// Local model directory (safetensors weights + tokenizer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// The corpus to adapt to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_dir: Option<PathBuf>,
    /// The scrt mind-palace providing the retrieval signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palace_path: Option<PathBuf>,
    /// Where datasets, adapters, and checkpoints land. Defaults to
    /// `.scrt-evolve` (see [`EvolveSection::work_dir_or_default`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_dir: Option<PathBuf>,
    /// GLOBAL constitution — values that drive HOW the model should process /
    /// answer (applied to every goal's generation). Composed into the generate
    /// system prompt (the `custom_prompt` steering seam). Minimal slice of the
    /// taste/meta-object substrate (tracks 21/22); a plain string for now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constitution: Option<String>,
    /// GLOBAL taste — the representational FORM ideas should take (style,
    /// structure, conventions). Composed into the generate system prompt
    /// alongside `constitution`. Minimal slice; a plain string for now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taste: Option<String>,
}

impl EvolveSection {
    /// The default work-dir name when `work_dir` is unset.
    pub const DEFAULT_WORK_DIR: &'static str = ".scrt-evolve";

    /// Resolve the work-dir, falling back to the default.
    pub fn work_dir_or_default(&self) -> PathBuf {
        self.work_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(Self::DEFAULT_WORK_DIR))
    }
}

/// `[eval]` — the shared evaluation harness config (track 10).
///
/// The harness scores the current model against a held-out probe set and gates
/// evolution rounds. Every field is defaulted so an empty `[eval]` block is
/// valid; an absent block means **no eval** (the lane runs unguarded with a
/// logged warning — graceful degradation, spec §Constraints).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Path to the held-out probe set (`probe.jsonl`). Defaults to
    /// `work_dir/probe.jsonl`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_path: Option<PathBuf>,
    /// Fraction of a dataset to carve into the probe when building one
    /// (`probe build --holdout`). 0.0..=1.0.
    #[serde(default = "default_probe_holdout_frac")]
    pub probe_holdout_frac: f32,
    /// Which scorer backend to use: `api` (no ML deps — correctness +
    /// constitution only) | `transformers` (Python subprocess, real forward
    /// pass for perplexity/exit-depth) | `candle` (optional, `--features train`).
    #[serde(default = "default_scorer_backend")]
    pub scorer_backend: String,
    /// Optional judge endpoint for constitution-adherence scoring (an
    /// OpenAI-compatible chat endpoint). Reuses the `[generate.api]` shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<GenerateApiConfig>,
    /// Which metrics to compute. Unknown/unsupported metrics for the active
    /// backend are skipped with a log (graceful degrade). Defaults to the
    /// always-available `correctness`.
    #[serde(default = "default_eval_metrics")]
    pub metrics: Vec<String>,
}

fn default_probe_holdout_frac() -> f32 {
    0.1
}
fn default_scorer_backend() -> String {
    "api".to_string()
}
fn default_eval_metrics() -> Vec<String> {
    vec!["correctness".to_string()]
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            probe_path: None,
            probe_holdout_frac: default_probe_holdout_frac(),
            scorer_backend: default_scorer_backend(),
            judge: None,
            metrics: default_eval_metrics(),
        }
    }
}

/// `[regulate]` — the self-regulation / transactional homeostasis config
/// (track 15).
///
/// Defaults make an empty `[regulate]` block a safe, working transaction
/// wrapper: enabled, keep a few checkpoints, rollback+quarantine+halt on
/// catastrophe. Pruning (experts/base) is a documented seam (tracks 11–14) —
/// `prune` is reserved here and unused until those land.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulateConfig {
    /// Master switch. `false` ⇒ steps run unguarded (no checkpoint/eval/rollback).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// How much correctness may drop and still `accept` (absolute). Mirrors
    /// [`crate::eval::VerdictTolerances::correctness_tolerance`].
    #[serde(default = "default_accept_tolerance")]
    pub accept_tolerance: f64,
    /// Absolute correctness floor: below ⇒ `catastrophic`.
    #[serde(default = "default_catastrophe_floor")]
    pub catastrophe_floor: f64,
    /// How many checkpoints to retain (older good ones beyond this are pruned).
    #[serde(default = "default_keep_checkpoints")]
    pub keep_checkpoints: usize,
    /// Catastrophe policy. Only `rollback_quarantine_halt` is implemented; other
    /// values are accepted but treated as the default with a log.
    #[serde(default = "default_on_catastrophe")]
    pub on_catastrophe: String,
}

fn default_true() -> bool {
    true
}
fn default_accept_tolerance() -> f64 {
    0.02
}
fn default_catastrophe_floor() -> f64 {
    0.10
}
fn default_keep_checkpoints() -> usize {
    5
}
fn default_on_catastrophe() -> String {
    "rollback_quarantine_halt".to_string()
}

impl Default for RegulateConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            accept_tolerance: default_accept_tolerance(),
            catastrophe_floor: default_catastrophe_floor(),
            keep_checkpoints: default_keep_checkpoints(),
            on_catastrophe: default_on_catastrophe(),
        }
    }
}

impl RegulateConfig {
    /// The verdict tolerances implied by this config.
    pub fn tolerances(&self) -> crate::eval::VerdictTolerances {
        crate::eval::VerdictTolerances {
            correctness_tolerance: self.accept_tolerance,
            catastrophe_floor: self.catastrophe_floor,
        }
    }
}

/// `[hardware]` — the compute environment for the heavy ML subprocesses
/// (track 24). Generic + architecture-level: nothing model-specific. Lets the
/// pipeline reason about whether a given model can TRAIN here (e.g. a hybrid-SSM
/// model's backward needs CUDA + the mamba kernels) vs only run forward
/// (eval/teacher), and record the machine a run happened on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Target device for training: `auto` | `cpu` | `cuda` | `mps`. `auto` lets
    /// the Python side pick (cuda if available, else cpu).
    #[serde(default = "default_device")]
    pub device: String,
    /// Approximate usable VRAM in GB (0 ⇒ unknown / CPU). Used to warn before
    /// loading a model that won't fit.
    #[serde(default)]
    pub vram_gb: f32,
    /// System RAM in GB (0 ⇒ unknown). For CPU/offload sizing.
    #[serde(default)]
    pub ram_gb: f32,
    /// Acceleration kernels available in the environment, e.g.
    /// `["mamba-ssm", "causal-conv1d", "flash-attn"]`. A hybrid-SSM model needs
    /// `mamba-ssm` + `causal-conv1d` to TRAIN (their absence ⇒ the naive CPU path
    /// that segfaults on backward). Empty ⇒ none / naive fallbacks.
    #[serde(default)]
    pub kernels: Vec<String>,
    /// Free-form description of the machine (CPU/GPU/OS) for provenance — what
    /// hardware a benchmark run was actually executed on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
}

fn default_device() -> String {
    "auto".to_string()
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            vram_gb: 0.0,
            ram_gb: 0.0,
            kernels: Vec::new(),
            machine: None,
        }
    }
}

impl HardwareConfig {
    /// Does this environment have a given acceleration kernel?
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernels.iter().any(|k| k.eq_ignore_ascii_case(name))
    }

    /// Whether a hybrid state-space (Mamba) model can TRAIN here: needs a non-CPU
    /// device AND the mamba/conv kernels (else the naive backward segfaults).
    /// Returns `Ok(())` if trainable, else `Err(reason)` for a clear pre-flight
    /// warning. Generic — keyed on kernels, not on any model name.
    pub fn can_train_state_space(&self) -> Result<(), String> {
        let on_gpu = matches!(self.device.as_str(), "cuda" | "mps")
            || (self.device == "auto" && self.has_kernel("mamba-ssm"));
        if !on_gpu {
            return Err(
                "state-space (Mamba) training needs a CUDA/MPS device; device is \
                 cpu/auto with no GPU kernels (the naive CPU SSM backward segfaults)"
                    .to_string(),
            );
        }
        if !(self.has_kernel("mamba-ssm") && self.has_kernel("causal-conv1d")) {
            return Err(
                "state-space (Mamba) training needs the `mamba-ssm` + `causal-conv1d` \
                 kernels installed (declare them in [hardware].kernels once installed)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// `[discover]` — corpus + palace discovery strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoverConfig {
    /// `palace | corpus | both` — where discovery starts.
    #[serde(default = "default_seed")]
    pub seed: String,
    #[serde(default = "default_max_passages")]
    pub max_passages: usize,
    /// e.g. `simhash` — use scrt's similarity to drop near-dup context.
    #[serde(default = "default_dedup")]
    pub dedup: String,
    /// Spread generation across distinct topics.
    #[serde(default = "default_cluster")]
    pub cluster: bool,
    /// Patterns to sweep the corpus with when `seed` includes `corpus`. Each
    /// becomes a scrt-search query; defaults cover common doc/comment markers
    /// so a corpus sweep finds something without configuration.
    #[serde(default = "default_corpus_patterns")]
    pub corpus_patterns: Vec<String>,
    /// When `seed` includes `palace`, restrict seeding to stashes whose name,
    /// note, search pattern, or any tag contains this case-insensitive
    /// substring (scrt's `--mp-list-search`). `None` ⇒ all stashes seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palace_search: Option<String>,
    /// When `seed` includes `palace`, restrict seeding to stashes carrying ALL
    /// of these tags. Composes with `palace_search`. Empty ⇒ no tag filter.
    #[serde(default)]
    pub palace_tags: Vec<String>,
}

fn default_seed() -> String {
    "palace".to_string()
}
fn default_max_passages() -> usize {
    500
}
fn default_dedup() -> String {
    "simhash".to_string()
}
fn default_cluster() -> bool {
    true
}
fn default_corpus_patterns() -> Vec<String> {
    // Generic markers: doc comments, headings, public items. Override in
    // `[discover].corpus_patterns` to target a specific topic.
    vec![
        r"///".to_string(),
        r"^#+\s".to_string(),
        r"pub fn".to_string(),
    ]
}

impl Default for DiscoverConfig {
    fn default() -> Self {
        Self {
            seed: default_seed(),
            max_passages: default_max_passages(),
            dedup: default_dedup(),
            cluster: default_cluster(),
            corpus_patterns: default_corpus_patterns(),
            palace_search: None,
            palace_tags: Vec::new(),
        }
    }
}

/// `[generate]` — synthetic-data generation, with per-backend sub-blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConfig {
    /// `local | api`.
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_kinds")]
    pub kinds: Vec<String>,
    /// Examples synthesized per passage.
    #[serde(default = "default_per_passage")]
    pub per_passage: usize,
    /// How `tool_call` rows are rendered at export time: `gemma` (native
    /// tool_code block) is implemented; `openai` / `anthropic` are stubbed.
    #[serde(default = "default_tool_format")]
    pub tool_format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<GenerateLocalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<GenerateApiConfig>,
}

fn default_tool_format() -> String {
    "gemma".to_string()
}

fn default_backend() -> String {
    "api".to_string()
}
fn default_kinds() -> Vec<String> {
    vec!["qa".to_string(), "instruction".to_string()]
}
fn default_per_passage() -> usize {
    3
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            kinds: default_kinds(),
            per_passage: default_per_passage(),
            tool_format: default_tool_format(),
            local: None,
            api: None,
        }
    }
}

/// `[generate.local]` — the local candle backend knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateLocalConfig {
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_new_tokens() -> usize {
    512
}
fn default_temperature() -> f32 {
    0.7
}

impl Default for GenerateLocalConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            temperature: default_temperature(),
        }
    }
}

/// `[generate.api]` — the API backend knobs. `api_key_env` is a var NAME.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateApiConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The NAME of the env var holding the key — never the key itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Multi-turn refine if > 1.
    #[serde(default = "default_turns")]
    pub turns: usize,
}

fn default_turns() -> usize {
    1
}

/// `[train]` — preset selection + per-preset sub-blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// `lora | full | pretrain | contrastive | shard`.
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full: Option<FullConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pretrain: Option<PretrainConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrastive: Option<ContrastiveConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<ShardConfig>,
    /// `[train.qat]` — quantization-aware training (track 23). Absent ⇒ plain
    /// LoRA. Additive + non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qat: Option<QatConfig>,
    /// `[train.fractional]` — single-node FRACTIONAL training: split the model
    /// into contiguous layer-block shards and train one block at a time via
    /// block-local distillation, bounding peak VRAM to a single block so a large
    /// model trains on a small GPU. Distinct from `[train.shard]` (which is
    /// multi-node distributed). Absent ⇒ dense training. Additive (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fractional: Option<FractionalConfig>,
}

/// `[train.fractional]` — fractional / sharded layer-block training.
///
/// Generic and model-agnostic: the Python side discovers the decoder-layer
/// stack, splits it into contiguous blocks, and trains each block's LoRA
/// adapters by distilling the frozen full-precision block (teacher) into the
/// adapted block (student). Only one block is ever resident on the accelerator,
/// so peak VRAM is bounded regardless of model depth. Pairs with `[train.qat]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalConfig {
    /// Master switch. `false` ⇒ behave as dense training even if this table is
    /// present (lets you keep the config but toggle the mode off).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Layers per block — the hard VRAM knob (smaller ⇒ less peak VRAM, more
    /// streaming). Takes precedence over `shards` when both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// Alternatively, split the model into this many equal blocks. Ignored if
    /// `block_size` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shards: Option<usize>,
    /// Token batches each block is distilled over (boundary activations are
    /// captured from these). More ⇒ a stronger local signal.
    #[serde(default = "default_calib_batches")]
    pub calib_batches: usize,
    /// Training granularity: `block` (default — train a whole layer-block's LoRA
    /// together) or `module` (PER-MODULE sub-layer floor — train one submodule
    /// group, e.g. attention / MoE / MLP, at a time within each layer, against
    /// the layer's frozen-output teacher). `module` is the lowest-VRAM, most-
    /// passes setting (trade time for memory); pair with `block_size = 1`.
    #[serde(default = "default_granularity")]
    pub granularity: String,
    /// Learning objective: `distill` (default — block-local MSE vs the frozen
    /// block's own output; a representation/regularization signal that does NOT
    /// impart new knowledge) or `end_task` (the FINAL shard learns real
    /// cross-entropy against the completion tokens via the LM head — the actual
    /// KNOWLEDGE signal; use this to teach the model new content). Non-final
    /// shards still distill under `end_task`.
    #[serde(default = "default_objective")]
    pub objective: String,
}

fn default_calib_batches() -> usize {
    8
}
fn default_granularity() -> String {
    "block".to_string()
}
fn default_objective() -> String {
    "distill".to_string()
}

impl Default for FractionalConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            block_size: None,
            shards: None,
            calib_batches: default_calib_batches(),
            granularity: default_granularity(),
            objective: default_objective(),
        }
    }
}

/// `[export]` — config-driven model-export pipeline: merge (sharded) adapter →
/// convert to GGUF → quantize → place. Every knob the manual pipeline needed —
/// sharding-merge rules, the merge-load dtype/device, the format conversion
/// target, source (llama.cpp) + scratch + target weight paths — lives here so
/// `scrt-evolve export-gguf` runs the whole chain from config. Absent ⇒ the CLI
/// falls back to its flag defaults (non-breaking). Generic + architecture-level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Target quantization / output format: `Q4_K_M` | `Q5_K_M` | `Q6_K` |
    /// `Q8_0` | `f16` | `none` | … (any llama.cpp quant; `f16`/`none` skip the
    /// quantize step). The "format conversion" target.
    #[serde(default = "default_export_quant")]
    pub quant: String,
    /// dtype to load the base model in during the MERGE stage. `bfloat16`
    /// (default) avoids the float32 OOM on large/hybrid models; `float32` for
    /// max fidelity on small models.
    #[serde(default = "default_export_dtype")]
    pub dtype: String,
    /// Path to a llama.cpp checkout providing `convert_hf_to_gguf.py` +
    /// `llama-quantize` (the conversion SOURCE tooling). Auto-detected if unset
    /// (`$LLAMA_CPP`, `~/llama.cpp`, `~/.unsloth/llama.cpp`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_path: Option<String>,
    /// Scratch directory for intermediates (merged HF dir + f16 GGUF). Point
    /// this at a FAST native filesystem — on WSL, a `~/…` path, NOT a `/mnt/c`
    /// 9p mount (large writes there OOM / I/O-error). Default: alongside `out`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_path: Option<String>,
    /// Final GGUF output path (the TARGET weight file). Default:
    /// `work_dir/<model>-<quant>.gguf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_path: Option<String>,
    /// Optional directory to PLACE (copy) the finished GGUF into — e.g. an LM
    /// Studio models dir. Absent ⇒ leave it at `out_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_dir: Option<String>,
    /// `save_pretrained` shard size for the merged HF dir (caps the per-file
    /// write so a big model doesn't spike RAM). Default `3GB`.
    #[serde(default = "default_max_shard_size")]
    pub max_shard_size: String,
    /// Keep the intermediate merged-HF dir + f16 GGUF (default false ⇒ cleaned).
    #[serde(default)]
    pub keep_intermediates: bool,
    /// `[export.merge_shards]` — how to combine the per-shard adapter files that
    /// fractional training emits into the single `adapter.safetensors` the merge
    /// stage consumes. Absent ⇒ assume a single-file adapter already.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_shards: Option<MergeShardsConfig>,
}

/// `[export.merge_shards]` — sharding-merge rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeShardsConfig {
    /// Master switch. `false` ⇒ skip the merge (adapter is already single-file).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Glob (relative to the adapter dir) matching the per-shard weight files.
    /// Their keys are global-layer-indexed, so the union is order-independent.
    #[serde(default = "default_shard_glob")]
    pub pattern: String,
}

fn default_export_quant() -> String {
    "Q4_K_M".to_string()
}
fn default_export_dtype() -> String {
    "bfloat16".to_string()
}
fn default_max_shard_size() -> String {
    "3GB".to_string()
}
fn default_shard_glob() -> String {
    "adapter-shard-*.safetensors".to_string()
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            quant: default_export_quant(),
            dtype: default_export_dtype(),
            llama_cpp_path: None,
            work_path: None,
            out_path: None,
            place_dir: None,
            max_shard_size: default_max_shard_size(),
            keep_intermediates: false,
            merge_shards: None,
        }
    }
}

impl Default for MergeShardsConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            pattern: default_shard_glob(),
        }
    }
}

/// `[runtime]` — the inference runtime: how to LOAD + RUN a model efficiently for
/// generation, config-driven and backend-generic. `scrt-evolve infer/run-model`
/// use this to serve the evolved model (or any model). Absent ⇒ infer falls back
/// to the transformers HF path against `[evolve].model_path`. Additive.
///
/// `backend` selects the engine by an internal registry (no brand logic):
///   - `llamacpp`  → a GGUF served via the llama.cpp `llama-cli` runner
///     (efficient quantized inference; the right path for hybrid-SSM models whose
///     naive transformers forward OOMs — llama.cpp handles SSM state properly).
///   - `transformers` → a HuggingFace model via the Python `scrt_evolve_infer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Inference engine: `llamacpp` (GGUF) | `transformers` (HF dir).
    #[serde(default = "default_runtime_backend")]
    pub backend: String,
    /// Weights to serve. For `llamacpp` a `.gguf` file; for `transformers` an HF
    /// model dir. Absent ⇒ fall back to `[export].out_path` (llamacpp) or
    /// `[evolve].model_path` (transformers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// Path to the llama.cpp checkout/build providing the `llama-cli` runner
    /// (llamacpp backend). Auto-detected if unset (shared with `[export]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_path: Option<String>,
    /// Context window (tokens). Transcript-derived prompts are long — keep ≥ 8192.
    #[serde(default = "default_n_ctx")]
    pub n_ctx: usize,
    /// Layers to offload to the GPU (llamacpp `-ngl`). 0 ⇒ pure CPU; a high
    /// value (e.g. 99) ⇒ offload all that fit. Generic VRAM/speed knob.
    #[serde(default)]
    pub n_gpu_layers: usize,
    /// CPU threads for generation. 0 ⇒ let the engine choose.
    #[serde(default)]
    pub n_threads: usize,
    /// Sampling controls for generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingConfig>,
}

/// `[runtime.sampling]` — decoding controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// 0.0 ⇒ greedy (deterministic); >0 ⇒ sampled.
    #[serde(default)]
    pub temperature: f32,
    /// Nucleus sampling cutoff (1.0 ⇒ off).
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Max new tokens to generate per prompt.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_runtime_backend() -> String {
    "llamacpp".to_string()
}
fn default_n_ctx() -> usize {
    8192
}
fn default_top_p() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    256
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: default_runtime_backend(),
            model_path: None,
            llama_cpp_path: None,
            n_ctx: default_n_ctx(),
            n_gpu_layers: 0,
            n_threads: 0,
            sampling: None,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
        }
    }
}

/// `[train.qat]` — quantization-aware training settings (track 23).
///
/// Simulates the deployment quant during the LoRA forward so the adapter adapts
/// to it. Generic: `quant` is any GGUF quant name (the Python side maps it to a
/// bit width); nothing here is model-specific.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatConfig {
    /// Master switch.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Target GGUF quant to simulate (e.g. `Q4_K_M`).
    #[serde(default = "default_qat_quant")]
    pub quant: String,
    /// Per-group affine group size for the fake-quant.
    #[serde(default = "default_qat_group_size")]
    pub group_size: usize,
    /// Calibration batches (0 ⇒ dynamic per-step absmax, no calibration pass).
    #[serde(default)]
    pub calibrate_batches: usize,
}

fn default_qat_quant() -> String {
    "Q4_K_M".to_string()
}
fn default_qat_group_size() -> usize {
    32
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            quant: default_qat_quant(),
            group_size: default_qat_group_size(),
            calibrate_batches: 0,
        }
    }
}

fn default_preset() -> String {
    "lora".to_string()
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            preset: default_preset(),
            lora: None,
            full: None,
            pretrain: None,
            contrastive: None,
            shard: None,
            qat: None,
            fractional: None,
        }
    }
}

/// `[train.lora]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    #[serde(default = "default_lora_rank")]
    pub rank: usize,
    #[serde(default = "default_lora_alpha")]
    pub alpha: usize,
    #[serde(default = "default_lora_targets")]
    pub target_modules: Vec<String>,
    #[serde(default = "default_lora_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
}

fn default_lora_rank() -> usize {
    16
}
fn default_lora_alpha() -> usize {
    32
}
fn default_lora_targets() -> Vec<String> {
    vec!["q_proj".to_string(), "v_proj".to_string()]
}
fn default_lora_lr() -> f64 {
    2e-4
}
fn default_epochs() -> usize {
    1
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: default_lora_rank(),
            alpha: default_lora_alpha(),
            target_modules: default_lora_targets(),
            lr: default_lora_lr(),
            epochs: default_epochs(),
        }
    }
}

/// `[train.full]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConfig {
    #[serde(default = "default_full_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_grad_accum")]
    pub grad_accum: usize,
}

fn default_full_lr() -> f64 {
    1e-5
}
fn default_grad_accum() -> usize {
    8
}

impl Default for FullConfig {
    fn default() -> Self {
        Self {
            lr: default_full_lr(),
            epochs: default_epochs(),
            grad_accum: default_grad_accum(),
        }
    }
}

/// `[train.pretrain]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainConfig {
    #[serde(default = "default_full_lr")]
    pub lr: f64,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
}

fn default_block_size() -> usize {
    1024
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self {
            lr: default_full_lr(),
            block_size: default_block_size(),
        }
    }
}

/// `[train.contrastive]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    #[serde(default = "default_negatives_per_row")]
    pub negatives_per_row: usize,
    #[serde(default = "default_contrastive_temperature")]
    pub temperature: f32,
}

fn default_negatives_per_row() -> usize {
    4
}
fn default_contrastive_temperature() -> f32 {
    0.05
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            negatives_per_row: default_negatives_per_row(),
            temperature: default_contrastive_temperature(),
        }
    }
}

/// `[train.shard]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// `coordinator | worker`.
    #[serde(default = "default_shard_role")]
    pub role: String,
    #[serde(default)]
    pub peers: Vec<String>,
    /// `data | layer`.
    #[serde(default = "default_shard_strategy")]
    pub shard_strategy: String,
    /// What each shard runs locally.
    #[serde(default = "default_base_preset")]
    pub base_preset: String,
}

fn default_shard_role() -> String {
    "coordinator".to_string()
}
fn default_shard_strategy() -> String {
    "data".to_string()
}
fn default_base_preset() -> String {
    "lora".to_string()
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            role: default_shard_role(),
            peers: Vec::new(),
            shard_strategy: default_shard_strategy(),
            base_preset: default_base_preset(),
        }
    }
}

impl EvolveConfig {
    /// Load + validate an `evolve.toml` from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&text)
    }

    /// Parse + validate from an in-memory TOML string (the testable core).
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let cfg: EvolveConfig = toml::from_str(text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Stage-independent validation. Stage-specific `model_path` requirements
    /// are enforced lazily by [`Self::require_model_path`] when a stage runs,
    /// so partial configs (generate-only / train-only) load fine here.
    fn validate(&self) -> Result<(), ConfigError> {
        if let Some(generate) = &self.generate {
            if let Some(api) = &generate.api {
                if let Some(key_env) = &api.api_key_env {
                    if looks_like_inline_secret(key_env) {
                        return Err(ConfigError::InlineSecret {
                            field: "generate.api.api_key_env",
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Assert `model_path` is present, for a stage that needs it. Callers in
    /// the stage drivers (discover/generate-local/train) invoke this so the
    /// requirement is enforced only when the stage actually runs.
    pub fn require_model_path(&self, stage: &'static str) -> Result<&Path, ConfigError> {
        self.evolve
            .model_path
            .as_deref()
            .ok_or(ConfigError::MissingModelPath { stage })
    }

    /// The resolved work-dir for this config.
    pub fn work_dir(&self) -> PathBuf {
        self.evolve.work_dir_or_default()
    }

    /// Derive a per-goal [`EvolveConfig`] for the buildable discover→generate
    /// pipeline (track 20 slice 3). The returned config:
    /// - sets `discover.palace_search = goal.topic` and
    ///   `discover.palace_tags = [goal.tag]` so only the goal's tagged stashes
    ///   seed (the one-goal ⇄ one-tag contract),
    /// - forces `discover.seed = "palace"` (the curated, goal-tagged traces are
    ///   the high-signal source),
    /// - scopes `corpus_dir` to `goal.project` when set (else keeps the
    ///   top-level corpus),
    ///
    /// All other settings (`[generate]`, `[train]`, `palace_path`, `work_dir`)
    /// are inherited unchanged. This is pure (no I/O, no mutation of `self`) so
    /// it is safe to call inside a bounded scheduler loop.
    pub fn for_goal(&self, goal: &GoalConfig) -> EvolveConfig {
        let mut cfg = self.clone();

        if let Some(project) = &goal.project {
            cfg.evolve.corpus_dir = Some(project.clone());
        }

        let mut dcfg = cfg.discover.unwrap_or_default();
        // Always apply the goal's palace narrowing (topic→search, tag→tags) so a
        // populated palace seeds ONLY this goal's curated stashes. Preserve the
        // corpus dimension: if the base config already sweeps the corpus
        // (seed = corpus|both), keep it as "both" so a transcript/code corpus
        // still contributes when the palace is empty (the bench case). Only when
        // the base was palace-only do we stay palace-only.
        dcfg.seed = match dcfg.seed.as_str() {
            "corpus" | "both" => "both".to_string(),
            _ => "palace".to_string(),
        };
        dcfg.palace_search = Some(goal.topic.clone());
        dcfg.palace_tags = vec![goal.tag.clone()];
        cfg.discover = Some(dcfg);

        // Layer the goal's constitution/taste ON TOP of the global ones so the
        // composed config carries the goal-specific steering (global base +
        // goal addition). The composer (`compose_steering`) renders both.
        if let Some(gc) = &goal.constitution {
            cfg.evolve.constitution = Some(match &cfg.evolve.constitution {
                Some(base) => format!("{base}\n{gc}"),
                None => gc.clone(),
            });
        }
        if let Some(gt) = &goal.taste {
            cfg.evolve.taste = Some(match &cfg.evolve.taste {
                Some(base) => format!("{base}\n{gt}"),
                None => gt.clone(),
            });
        }

        cfg
    }

    /// Compose the active constitution + taste into a single steering block to be
    /// injected as the generate `custom_prompt` (the seam that lets values +
    /// representational form shape the dataset, and thus training). Returns
    /// `None` when neither is set (preserves today's built-in-template behavior).
    pub fn compose_steering(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(c) = self.evolve.constitution.as_ref().map(|s| s.trim()) {
            if !c.is_empty() {
                parts.push(format!(
                    "## Constitution (values that drive how you answer)\n{c}"
                ));
            }
        }
        if let Some(t) = self.evolve.taste.as_ref().map(|s| s.trim()) {
            if !t.is_empty() {
                parts.push(format!("## Taste (the form ideas should take)\n{t}"));
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }
}

/// Heuristic: does this `api_key_env` value look like a literal secret rather
/// than an env-var NAME? Env var names are conventionally UPPER_SNAKE and
/// short; real API keys are long, mixed-case, and often carry provider
/// prefixes (`sk-`, `sk-ant-`) or non-identifier characters.
fn looks_like_inline_secret(value: &str) -> bool {
    let v = value.trim();
    // Known provider key prefixes are an immediate tell.
    if v.starts_with("sk-") || v.starts_with("sk_") {
        return true;
    }
    // Env var names are valid identifiers: [A-Za-z_][A-Za-z0-9_]*.
    let is_identifier = !v.is_empty()
        && v.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !is_identifier {
        // Contains spaces, dashes, slashes, etc. — not an env var name.
        return true;
    }
    // A very long all-caps-or-mixed token is suspicious; real env var names
    // are short. 40+ chars in a single identifier reads as a key, not a name.
    v.len() >= 40
}
