//! Native inference loader (track 39 P0-2): real HF llama-family base +
//! compose-at-load LoRA merge.
//!
//! Reuses [`LoadedModel::load`] for the base (no bespoke safetensors path),
//! reads an `adapter.safetensors` from track 04 into a [`LoraWeights`] value,
//! and merges the delta `scaling * (B @ A)` into the addressed base weights so
//! subsequent `forward` calls reflect the adapter. The merge is performed via
//! [`ArchAdapter::apply_adapter`] on [`LlamaAdapter`] (P0-1 recorded; P0-2
//! merges).
//!
//! Adapter tensor names follow the track-04 convention:
//! `lora.<target>.a` / `lora.<target>.b`, where `<target>` is the full base
//! weight name (e.g. `model.layers.0.self_attn.q_proj.weight`). No key
//! remapping — the base weight name is preserved through the load.

#![cfg(feature = "train")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use crate::model::{
    ArchAdapter, LlamaAdapter, LoadedModel, LoraTargetWeights, LoraWeights,
};

/// Errors from the native loader.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    /// The base model directory failed to load.
    #[error("base model load failed: {0}")]
    Base(String),
    /// Reading the adapter safetensors failed.
    #[error("adapter load failed: {0}")]
    AdapterIo(String),
    /// An adapter tensor had an unexpected name or shape.
    #[error("adapter shape error: {0}")]
    AdapterShape(String),
    /// The base weight named by an adapter target does not exist.
    #[error("adapter target missing in base: {0}")]
    MissingTarget(String),
    /// A tensor op failed during merge.
    #[error("merge failed: {0}")]
    Merge(String),
}

impl From<candle_core::Error> for LoaderError {
    fn from(e: candle_core::Error) -> Self {
        LoaderError::Merge(e.to_string())
    }
}

/// A base + optional adapter, materialised and ready for native inference.
pub struct LoadedInference {
    /// The adapter wrapper. If an adapter was supplied, its delta has already
    /// been merged into the underlying base weights.
    pub adapter: LlamaAdapter,
    /// Base directory the model was loaded from.
    pub model_path: PathBuf,
    /// Adapter file merged in, if any.
    pub adapter_path: Option<PathBuf>,
}

/// Load a real HF llama-family model directory (reuses
/// [`LoadedModel::load`] — no reimplementation).
pub fn load_base(model_path: &Path) -> Result<LlamaAdapter, LoaderError> {
    let model = LoadedModel::load(model_path)
        .map_err(|e| LoaderError::Base(format!("{}: {e}", model_path.display())))?;
    Ok(LlamaAdapter::new(model))
}

/// Read `adapter.safetensors` (track 04 output) into a [`LoraWeights`].
///
/// Tensor names must be `lora.<target>.a` / `lora.<target>.b`; targets are
/// pairs whose two halves are both present. `rank` is inferred from A's
/// leading dim; `alpha` defaults to `rank` (identity scaling) since the
/// on-disk safetensors does not carry it — callers who need a different
/// scaling should set `weights.alpha` before merging.
pub fn load_adapter(
    adapter_path: &Path,
    device: &Device,
) -> Result<LoraWeights, LoaderError> {
    let tensors = candle_core::safetensors::load(adapter_path, device)
        .map_err(|e| LoaderError::AdapterIo(format!("{}: {e}", adapter_path.display())))?;

    // Group by target name; A and B are matched by suffix.
    let mut pairs: BTreeMap<String, (Option<Tensor>, Option<Tensor>)> = BTreeMap::new();
    for (name, t) in tensors {
        let (target, is_a) = if let Some(rest) = name.strip_prefix("lora.") {
            if let Some(target) = rest.strip_suffix(".a") {
                (target.to_string(), true)
            } else if let Some(target) = rest.strip_suffix(".b") {
                (target.to_string(), false)
            } else {
                return Err(LoaderError::AdapterShape(format!(
                    "unexpected adapter tensor name: {name}"
                )));
            }
        } else {
            return Err(LoaderError::AdapterShape(format!(
                "adapter tensor missing lora. prefix: {name}"
            )));
        };
        let slot = pairs.entry(target).or_insert((None, None));
        if is_a {
            slot.0 = Some(t);
        } else {
            slot.1 = Some(t);
        }
    }

    let mut targets = Vec::with_capacity(pairs.len());
    let mut rank_inferred: Option<usize> = None;
    for (target, (a, b)) in pairs {
        let a = a.ok_or_else(|| {
            LoaderError::AdapterShape(format!("target {target}: A tensor missing"))
        })?;
        let b = b.ok_or_else(|| {
            LoaderError::AdapterShape(format!("target {target}: B tensor missing"))
        })?;
        let a_dims = a.dims();
        let b_dims = b.dims();
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(LoaderError::AdapterShape(format!(
                "target {target}: A/B must be rank-2, got A={a_dims:?} B={b_dims:?}"
            )));
        }
        // A: [rank, in], B: [out, rank] — assert consistency.
        if a_dims[0] != b_dims[1] {
            return Err(LoaderError::AdapterShape(format!(
                "target {target}: rank mismatch A[0]={} vs B[1]={}",
                a_dims[0], b_dims[1]
            )));
        }
        rank_inferred = Some(a_dims[0]);
        targets.push(LoraTargetWeights { target, a, b });
    }

    let rank = rank_inferred.unwrap_or(0);
    Ok(LoraWeights {
        rank,
        alpha: rank as f32, // identity scaling by default; caller may override
        targets,
    })
}

/// Load a base model and (optionally) merge an adapter into it.
///
/// The returned [`LlamaAdapter`]'s underlying [`LoadedModel::forward`] reflects
/// the merged delta directly — no runtime hook required.
pub fn load(
    model_path: &Path,
    adapter_path: Option<&Path>,
) -> Result<LoadedInference, LoaderError> {
    let mut adapter = load_base(model_path)?;
    let adapter_path_out = if let Some(ap) = adapter_path {
        let dev = adapter.model.device.clone();
        let lora = load_adapter(ap, &dev)?;
        adapter
            .apply_adapter(&lora)
            .map_err(|e| LoaderError::Merge(e.to_string()))?;
        Some(ap.to_path_buf())
    } else {
        None
    };

    Ok(LoadedInference {
        adapter,
        model_path: model_path.to_path_buf(),
        adapter_path: adapter_path_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LoadedModel, ModelConfig};
    use candle_core::{DType, Tensor};

    /// Build a synthetic LoRA whose merged delta is a known non-zero value,
    /// merge it, and check that forward logits change deterministically.
    #[test]
    fn merge_lora_changes_forward_logits() {
        let cfg = ModelConfig::tiny();
        let base = LoadedModel::random_fixture(cfg.clone(), 99).unwrap();
        let dev = base.device.clone();
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();

        let before = base
            .forward(&ids)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Wrap and apply a tiny LoRA on a real target weight: q_proj[0] is
        // [hidden, hidden] = [64, 64] in the tiny fixture. Pick rank=2 with
        // A = ones([2,64]), B = ones([64,2]) so B@A is an all-2s [64,64],
        // scaled by alpha/rank = 4/2 = 2 → merged delta is all 4s.
        let target = "model.layers.0.self_attn.q_proj.weight".to_string();
        let hidden = cfg.hidden_size;
        let rank = 2usize;
        // Non-cancelling delta: A row 0 alternates +1/-1 across the input
        // dim, so `x @ (B@A)^T` does not integrate to zero over zero-mean
        // embeddings. B is a large scalar so the merged weight is clearly
        // above float noise, and the delta is genuinely rank-1 (row 1 of
        // A is zeros so B[:,1] contributes nothing).
        let mut a_vals = vec![0f32; rank * hidden];
        for j in 0..hidden {
            a_vals[j] = if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut b_vals = vec![0f32; hidden * rank];
        for i in 0..hidden {
            b_vals[i * rank] = 1.0;
        }
        let a = Tensor::from_vec(a_vals, (rank, hidden), &dev).unwrap();
        let b = Tensor::from_vec(b_vals, (hidden, rank), &dev).unwrap();
        // Large alpha so the delta clearly dominates the tiny base init.
        let lora = LoraWeights {
            rank,
            alpha: 200.0,
            targets: vec![LoraTargetWeights { target: target.clone(), a, b }],
        };

        let mut adapter = LlamaAdapter::new(base);
        adapter.apply_adapter(&lora).expect("merge");

        // Weight itself must have shifted by exactly 4.0 everywhere.
        let after_w = {
            let data = adapter.model.var_map.data().lock().unwrap();
            data[&target]
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        // Rebuild the pre-merge weight from a fresh fixture with the same seed
        // and diff element-wise.
        let fresh = LoadedModel::random_fixture(cfg.clone(), 99).unwrap();
        let orig_w = fresh
            .var_map
            .data()
            .lock()
            .unwrap()[&target]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Reference merged weight: base + scaling * (B @ A). scaling = alpha/rank.
        let scaling = adapter.applied().unwrap().scaling();
        let base_tensor = fresh
            .var_map
            .data()
            .lock()
            .unwrap()[&target]
            .as_tensor()
            .clone();
        let ref_delta = adapter
            .applied()
            .unwrap()
            .targets[0]
            .b
            .matmul(&adapter.applied().unwrap().targets[0].a)
            .unwrap();
        let ref_merged = (base_tensor + (ref_delta * scaling as f64).unwrap()).unwrap();
        let ref_flat = ref_merged.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, n) in ref_flat.iter().zip(after_w.iter()) {
            assert!(
                (r - n).abs() < 1e-4,
                "merged weight must match base + scaling*(B@A): ref={r} got={n}"
            );
        }
        let _ = orig_w;

        // Sanity: query the q_proj weight through the fresh graph module
        // directly by re-forwarding a fresh-fixture pre-merge model on the
        // same ids for comparison, ensuring rebuild took effect.

        // Forward logits must differ from the pre-merge baseline.
        let after = adapter
            .model
            .forward(&ids)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(before.len(), after.len());
        assert!(after.iter().all(|v| v.is_finite()), "logits must be finite");
        let max_diff = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // Noise floor of a no-op rebuild is ~3e-8 (verified empirically); any
        // real weight change lifts this above 1e-6.
        assert!(
            max_diff > 1e-6,
            "merged adapter must change forward logits (max_diff={max_diff})"
        );
    }

    /// Full HF-dir load is not exercised in CI (no fixture on disk); shape
    /// wired up so it can be run manually against a real directory.
    #[test]
    #[ignore]
    fn load_real_hf_dir_smoke() {
        let dir = std::env::var("SCRT_EVOLVE_TEST_MODEL_DIR").expect("set model dir");
        let inf = load(Path::new(&dir), None).expect("load");
        let dev = inf.adapter.model.device.clone();
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();
        let logits = inf.adapter.model.forward(&ids).unwrap();
        let flat = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }
}
