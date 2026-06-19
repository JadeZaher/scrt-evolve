//! `evolve init` scaffold — write a commented `evolve.toml` template.
//!
//! Lives in the SDK so a library consumer can scaffold too; the CLI's `init`
//! subcommand is a thin wrapper. Mirrors DESIGN.md §Config schema field-for-
//! field so a freshly-scaffolded file round-trips through [`crate::EvolveConfig`].

use std::path::Path;

/// The commented `evolve.toml` template written by `init`.
pub const TEMPLATE: &str = r#"# scrt-evolve configuration.
# One top [evolve] section + per-stage blocks. Every stage reads only what it
# needs, so partial configs work (generate-only, train-only).

[evolve]
model_path  = "/models/my-model"      # the ONE required thing (weights + tokenizer)
corpus_dir  = "./src"                  # what to adapt to
palace_path = ".mpg/mind-palace.json"  # the retrieval signal
work_dir    = ".scrt-evolve"           # datasets, adapters, checkpoints land here

[discover]
seed = "palace"            # palace | corpus | both — where discovery starts
max_passages = 500
dedup = "simhash"          # use scrt's similarity to drop near-dup context
cluster = true             # spread generation across distinct topics

[generate]
backend = "api"            # local | api
kinds = ["qa", "instruction"]
per_passage = 3            # examples synthesized per passage
  [generate.local]
  max_new_tokens = 512
  temperature = 0.7
  [generate.api]
  base_url = "https://api.example.com/v1"
  model = "model-name"
  api_key_env = "SCRT_EVOLVE_API_KEY"   # NAME of an env var — never inline a key
  turns = 1                              # multi-turn refine if > 1

[train]
preset = "lora"            # lora | full | pretrain | contrastive | shard
  [train.lora]
  rank = 16
  alpha = 32
  target_modules = ["q_proj", "v_proj"]
  lr = 2e-4
  epochs = 1
  [train.full]
  lr = 1e-5
  epochs = 1
  grad_accum = 8
  [train.pretrain]
  lr = 1e-5
  block_size = 1024
  [train.contrastive]
  negatives_per_row = 4
  temperature = 0.05
  [train.shard]
  role = "coordinator"     # coordinator | worker
  peers = ["host:port"]
  shard_strategy = "data"  # data | layer
  base_preset = "lora"     # what each shard runs locally
"#;

/// Outcome of scaffolding, so callers can surface warnings.
#[derive(Debug)]
pub struct ScaffoldReport {
    /// True when the scaffolded `model_path` does not exist on disk yet.
    /// This is a WARNING, not an error — `init` still writes the file.
    pub model_path_missing: bool,
}

/// Write the commented scaffold to `path`. Errors if the file already exists
/// (callers should check / prompt before overwriting). The returned report
/// flags a missing `model_path` so the CLI can warn (not error).
pub fn init(path: impl AsRef<Path>) -> anyhow::Result<ScaffoldReport> {
    let path = path.as_ref();
    if path.exists() {
        anyhow::bail!(
            "{} already exists — refusing to overwrite",
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, TEMPLATE)?;

    // Parse the freshly-written scaffold to read back the model_path and warn
    // if it doesn't exist — keeps the warning logic anchored to the real value.
    let cfg = crate::EvolveConfig::from_toml_str(TEMPLATE)?;
    let model_path_missing = cfg
        .evolve
        .model_path
        .as_deref()
        .map(|p| !p.exists())
        .unwrap_or(true);

    Ok(ScaffoldReport {
        model_path_missing,
    })
}
