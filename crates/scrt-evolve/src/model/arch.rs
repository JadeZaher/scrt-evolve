//! Per-architecture adapter seam (P0-1 for track 39 native candle inference).
//!
//! Fronts the concrete `TinyLlama` graph in [`crate::model`] behind a stable
//! [`ArchAdapter`] trait so downstream native-inference / seam-compression code
//! (track 39 P0-2+, cross-arch distillation) can address layers by index and
//! attach adapter weights without importing the private graph types.
//!
//! Only the `train` feature carries the real ML types (candle tensors); the
//! whole module is gated behind `#[cfg(feature = "train")]` at the caller.

#![cfg(feature = "train")]

use candle_core::{Device, Tensor};

use super::{LoadedModel, ModelError};

/// The kind of computation a [`LayerDesc`] represents.
///
/// Llama-family decoders emit `Attn` (a fused attention+MLP block in our
/// impl); future backends will emit `Ssm` (Mamba), `Mlp` (MLP-only stacks),
/// and `MoE` (mixture-of-experts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Self-attention (or a block dominated by it).
    Attn,
    /// State-space model layer (e.g. Mamba).
    Ssm,
    /// Feed-forward / MLP layer.
    Mlp,
    /// Mixture-of-experts routed layer.
    MoE,
}

/// A single addressable layer inside an [`ArchAdapter`].
#[derive(Debug, Clone)]
pub struct LayerDesc {
    /// Zero-based layer index (input to [`ArchAdapter::forward_layer`]).
    pub idx: usize,
    /// What kind of computation the layer performs.
    pub kind: LayerKind,
}

/// A named boundary between layers where a seam (probe, adapter, split) can
/// be attached. `after_layer == n` means "the point immediately after layer
/// `n` has produced its output".
#[derive(Debug, Clone)]
pub struct SeamPoint {
    /// Human-readable name (e.g. `"after_layer_3"`).
    pub name: String,
    /// Zero-based index of the layer whose output this seam sits on.
    pub after_layer: usize,
}

/// One LoRA target's frozen A/B matrices, mirroring
/// [`crate::train::lora::LoraAdapter`] but as inert tensors (no `Var`s).
///
/// `a: [rank, in]`, `b: [out, rank]`. Effective delta = `(alpha/rank) * B @ A`.
#[derive(Debug, Clone)]
pub struct LoraTargetWeights {
    /// The base weight name this adapter wraps (e.g.
    /// `model.layers.0.self_attn.q_proj.weight`).
    pub target: String,
    /// `A: [rank, in]`.
    pub a: Tensor,
    /// `B: [out, rank]`.
    pub b: Tensor,
}

/// A full LoRA adapter as materialised weights (post-training, pre-merge).
///
/// Shape contract mirrors [`crate::train::lora::LoraAdapters`]: one entry per
/// target module, with `rank`/`alpha` giving the standard `scaling = alpha /
/// rank`.
#[derive(Debug, Clone)]
pub struct LoraWeights {
    /// LoRA rank (inner dimension of each A/B pair).
    pub rank: usize,
    /// LoRA alpha (scaling numerator; effective scale = `alpha / rank`).
    pub alpha: f32,
    /// One target weight per LoRA-adapted module, in deterministic order.
    pub targets: Vec<LoraTargetWeights>,
}

impl LoraWeights {
    /// Effective scaling factor `alpha / rank` (or 0 for degenerate rank).
    pub fn scaling(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }
}

/// The per-architecture seam used by native inference + seam-compression.
///
/// Implementations expose:
/// - [`ArchAdapter::layers`]: layer inventory for iteration/introspection;
/// - [`ArchAdapter::seam_points`]: named depth boundaries (probe / split targets);
/// - [`ArchAdapter::apply_adapter`]: attach a materialised [`LoraWeights`];
/// - [`ArchAdapter::forward_layer`]: run exactly one layer by index, so a
///   caller can walk the stack, probe intermediates, or splice architectures.
pub trait ArchAdapter {
    /// Ordered layer inventory (`idx` fields are monotonically increasing).
    fn layers(&self) -> Vec<LayerDesc>;

    /// Named seam points (`after_layer` boundaries) available on this arch.
    fn seam_points(&self) -> Vec<SeamPoint>;

    /// Attach a materialised LoRA adapter to the underlying model.
    ///
    /// P0-1 records the adapter but does not yet re-plumb the base graph;
    /// P0-2+ will merge `B @ A` into the addressed base weights so
    /// [`Self::forward_layer`] reflects the delta.
    fn apply_adapter(&mut self, lora: &LoraWeights) -> anyhow::Result<()>;

    /// Run exactly decoder layer `idx` on `x: [batch, seq, hidden]`.
    ///
    /// Byte-identical to the corresponding slice of the base model's fused
    /// forward. `dev` is the target device (the impl may verify `x.device()`
    /// matches).
    fn forward_layer(
        &self,
        idx: usize,
        x: &Tensor,
        dev: &Device,
    ) -> anyhow::Result<Tensor>;
}

/// [`ArchAdapter`] implementation over the TinyLlama graph in
/// [`crate::model`]. Addresses layers by index.
pub struct LlamaAdapter {
    /// The wrapped model (owns the [`candle_nn::VarMap`]).
    pub model: LoadedModel,
    /// Most recently applied adapter, if any (P0-1 stores; P0-2 will merge).
    applied: Option<LoraWeights>,
}

impl LlamaAdapter {
    /// Wrap a loaded TinyLlama model.
    pub fn new(model: LoadedModel) -> Self {
        Self {
            model,
            applied: None,
        }
    }

    /// The currently applied adapter, if any.
    pub fn applied(&self) -> Option<&LoraWeights> {
        self.applied.as_ref()
    }
}

impl ArchAdapter for LlamaAdapter {
    fn layers(&self) -> Vec<LayerDesc> {
        (0..self.model.num_layers())
            .map(|idx| LayerDesc {
                idx,
                kind: LayerKind::Attn,
            })
            .collect()
    }

    fn seam_points(&self) -> Vec<SeamPoint> {
        (0..self.model.num_layers())
            .map(|i| SeamPoint {
                name: format!("after_layer_{i}"),
                after_layer: i,
            })
            .collect()
    }

    fn apply_adapter(&mut self, lora: &LoraWeights) -> anyhow::Result<()> {
        // P0-2: compose-at-load merge. For each target, add
        //   scaling * (B @ A)   [out, rank] @ [rank, in] -> [out, in]
        // into the base weight in-place via VarMap::set_one. The base
        // `forward` then reflects the delta directly — no runtime hook.
        let scaling = lora.scaling() as f64;
        // Snapshot each target's merged tensor first (release the lock before
        // set_one, which re-acquires it).
        let mut merged: Vec<(String, candle_core::Tensor)> = Vec::new();
        {
            let data = self
                .model
                .var_map
                .data()
                .lock()
                .map_err(|e| anyhow::anyhow!("base var_map lock poisoned: {e}"))?;
            for t in &lora.targets {
                let base_var = data.get(&t.target).ok_or_else(|| {
                    anyhow::anyhow!("adapter target missing in base: {}", t.target)
                })?;
                let base_w = base_var.as_detached_tensor();
                let delta = t.b.matmul(&t.a)?;
                let delta = if scaling == 1.0 { delta } else { (delta * scaling)? };
                if base_w.dims() != delta.dims() {
                    return Err(anyhow::anyhow!(
                        "adapter shape mismatch for {}: base={:?} delta={:?}",
                        t.target,
                        base_w.dims(),
                        delta.dims()
                    ));
                }
                merged.push((t.target.clone(), (base_w + delta)?));
            }
        }
        self.model
            .overwrite_weights(&merged)
            .map_err(|e| anyhow::anyhow!("merge weights: {e}"))?;
        // Also record the applied adapter for introspection.
        self.applied = Some(LoraWeights {
            rank: lora.rank,
            alpha: lora.alpha,
            targets: lora
                .targets
                .iter()
                .map(|t| LoraTargetWeights {
                    target: t.target.clone(),
                    a: t.a.clone(),
                    b: t.b.clone(),
                })
                .collect(),
        });
        Ok(())
    }

    fn forward_layer(
        &self,
        idx: usize,
        x: &Tensor,
        _dev: &Device,
    ) -> anyhow::Result<Tensor> {
        if idx >= self.model.num_layers() {
            return Err(anyhow::anyhow!(
                "forward_layer: idx {idx} out of range (num_layers={})",
                self.model.num_layers()
            ));
        }
        let out = self
            .model
            .apply_layer(idx, x)
            .map_err(|e: ModelError| anyhow::anyhow!(e))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LoadedModel, ModelConfig};
    use candle_core::Tensor;

    #[test]
    fn forward_layer_matches_full_forward_byte_identical() {
        let cfg = ModelConfig::tiny();
        let model = LoadedModel::random_fixture(cfg.clone(), 2026).expect("fixture");
        // Reference: full forward logits from the frozen base graph.
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (1, 5), &model.device).unwrap();
        let expected = model.forward(&ids).expect("full forward");
        let expected_vec = expected
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Rebuild the same forward through the ArchAdapter seam: embed →
        // iterate every layer via forward_layer → norm + lm_head.
        let mut x = model.embed(&ids).expect("embed");
        let adapter = LlamaAdapter::new(model);
        let dev = adapter.model.device.clone();
        assert_eq!(adapter.layers().len(), cfg.num_layers);
        assert_eq!(adapter.seam_points().len(), cfg.num_layers);
        for i in 0..cfg.num_layers {
            x = adapter.forward_layer(i, &x, &dev).expect("forward_layer");
        }
        let got = adapter.model.head(&x).expect("head");
        let got_vec = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(
            expected_vec, got_vec,
            "forward_layer walk must be byte-identical to TinyLlama::forward"
        );
    }

    #[test]
    fn apply_adapter_records_weights() {
        let cfg = ModelConfig::tiny();
        let model = LoadedModel::random_fixture(cfg, 1).unwrap();
        let dev = model.device.clone();
        let mut adapter = LlamaAdapter::new(model);
        let w = LoraWeights {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTargetWeights {
                target: "model.layers.0.self_attn.q_proj.weight".to_string(),
                a: Tensor::zeros((4, 64), candle_core::DType::F32, &dev).unwrap(),
                b: Tensor::zeros((64, 4), candle_core::DType::F32, &dev).unwrap(),
            }],
        };
        adapter.apply_adapter(&w).expect("apply");
        let applied = adapter.applied().expect("recorded");
        assert_eq!(applied.rank, 4);
        assert_eq!(applied.targets.len(), 1);
        assert!((applied.scaling() - 2.0).abs() < 1e-6);
    }
}
