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
        && v.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !is_identifier {
        // Contains spaces, dashes, slashes, etc. — not an env var name.
        return true;
    }
    // A very long all-caps-or-mixed token is suspicious; real env var names
    // are short. 40+ chars in a single identifier reads as a key, not a name.
    v.len() >= 40
}
