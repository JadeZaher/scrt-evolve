//! Train stage — `TrainingPreset` trait + the `train()` driver.
//!
//! Five presets, each with its own config block. The driver routes the right
//! dataset shape to each preset. The training loops live behind the `train`
//! feature (candle); the seams here compile ML-free.

pub mod contrastive;
pub mod full;
pub mod lora;
pub mod pretrain;
pub mod shard;

use crate::config::EvolveConfig;
use crate::dataset::Dataset;
use crate::model::LoadedModel;

/// Summary of a training run.
#[derive(Debug, Clone, Default)]
pub struct TrainReport {
    pub preset: String,
    pub steps: usize,
    pub final_loss: Option<f32>,
    /// Path to the produced artifact (adapter / weights), once written.
    pub artifact: Option<std::path::PathBuf>,
}

/// How the model is updated. Each preset carries its own config type.
pub trait TrainingPreset {
    type Config;

    fn train(
        &self,
        model: &LoadedModel,
        data: &Dataset,
        cfg: &Self::Config,
    ) -> anyhow::Result<TrainReport>;
}

/// Run training per the configured preset.
///
/// Under `--features train` this loads the base model and dispatches to the
/// configured preset (track 04 implements `lora`; other presets bail with a
/// clear later-track message). Without the feature it bails so both builds
/// compile — the CLI calls this in both.
#[cfg(feature = "train")]
pub fn run(cfg: &EvolveConfig, data: &Dataset) -> anyhow::Result<TrainReport> {
    use crate::workdir::WorkDir;

    let train_cfg = cfg.train.clone().unwrap_or_default();
    let preset = train_cfg.preset.as_str();

    match preset {
        "lora" => {
            let model_path = cfg.require_model_path("train")?;
            let model = LoadedModel::load(model_path)
                .map_err(|e| anyhow::anyhow!("train: loading model from {}: {e}", model_path.display()))?;

            let lora_cfg = train_cfg.lora.clone().unwrap_or_default();
            // Deterministic seed for adapter init + batch order (styleguide §2.2).
            // Derived from the LoRA hyperparameters so a config change is a new,
            // still-reproducible run.
            let seed = lora_seed(&lora_cfg);
            let preset = lora::LoraPreset::new(seed);

            let wd = WorkDir::from_config(cfg);
            wd.ensure()?;
            let artifact = wd.adapter_safetensors();

            preset.train_to(&model, data, &lora_cfg, Some(&artifact))
        }
        other => anyhow::bail!(
            "train: preset '{other}' not implemented yet (later track) — \
             only 'lora' is available in this build"
        ),
    }
}

/// Without the `train` feature there is no candle; bail with a clear message
/// so a default build still compiles the CLI's `train::run` call.
#[cfg(not(feature = "train"))]
pub fn run(_cfg: &EvolveConfig, _data: &Dataset) -> anyhow::Result<TrainReport> {
    anyhow::bail!(
        "train: requires the `train` feature (candle) — build with --features train"
    )
}

/// Derive a stable seed from the LoRA config so runs are reproducible and a
/// config change produces a new (still-deterministic) trajectory.
#[cfg(feature = "train")]
fn lora_seed(cfg: &crate::config::LoraConfig) -> u64 {
    let mut s: u64 = 0x5EED_5EED_5EED_5EED;
    s ^= (cfg.rank as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s = s.rotate_left(7) ^ (cfg.alpha as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s = s.rotate_left(13) ^ (cfg.epochs as u64);
    s
}
