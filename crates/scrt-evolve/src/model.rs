//! Model loading (safetensors + tokenizer) — the per-architecture seam.
//!
//! candle has no published per-arch model loader we depend on, so this module
//! hand-builds a tiny Llama-style decoder-only causal LM in `candle-nn`. It is
//! the shared foundation two downstream tracks build on:
//!
//! - track 03 (LocalCandle generation) calls [`LoadedModel::forward`] +
//!   [`LoadedModel::tokenize`] / [`LoadedModel::detokenize`] to sample text;
//! - track 04 (LoRA training) needs the per-module `q_proj`/`v_proj` weights to
//!   be individually addressable, and needs `forward` to be differentiable.
//!
//! ## Why two construction paths
//!
//! [`LoadedModel::load`] reads real weights (`config.json` +`tokenizer.json` +
//! `model.safetensors`) from a directory. [`LoadedModel::random_fixture`]
//! deterministically seeds every weight from a `u64` and builds an in-memory
//! byte-level tokenizer — no files required — so CI and the track-04 overfit
//! smoke test run fully offline.
//!
//! ## Determinism (styleguide §2.2)
//!
//! candle's CPU RNG is **not** seedable (`Device::set_seed` bails on CPU), so we
//! do not rely on it. Every fixture weight is filled from a self-contained
//! SplitMix64 PRNG keyed by the caller's `seed`, in a fixed traversal order.
//! Same `seed` + same [`ModelConfig`] → byte-identical weights.
//!
//! ## VarMap weight-naming scheme (track 04 targets these verbatim)
//!
//! Every trainable tensor lives in the [`candle_nn::VarMap`] under a stable name.
//! LoRA target-module discovery matches on the leaf `q_proj` / `v_proj`:
//!
//! ```text
//! model.embed_tokens.weight
//! model.layers.{i}.input_layernorm.weight
//! model.layers.{i}.self_attn.q_proj.weight     <- LoRA target
//! model.layers.{i}.self_attn.k_proj.weight
//! model.layers.{i}.self_attn.v_proj.weight     <- LoRA target
//! model.layers.{i}.self_attn.o_proj.weight
//! model.layers.{i}.post_attention_layernorm.weight
//! model.layers.{i}.mlp.gate_proj.weight
//! model.layers.{i}.mlp.up_proj.weight
//! model.layers.{i}.mlp.down_proj.weight
//! model.norm.weight
//! lm_head.weight
//! ```
//!
//! `{i}` is the zero-based layer index. The names mirror the HuggingFace Llama
//! convention so real `model.safetensors` checkpoints load without remapping.

#[cfg(not(feature = "train"))]
mod placeholder {
    /// A loaded model + tokenizer, ready for inference or training.
    ///
    /// The rich fields (config, tokenizer, weights) only exist under the `train`
    /// feature; in a default build this is an inert placeholder so the stage
    /// signatures compile ML-free.
    #[derive(Debug, Default)]
    pub struct LoadedModel {
        /// The model directory the weights/tokenizer were loaded from.
        pub model_path: std::path::PathBuf,
    }
}

#[cfg(not(feature = "train"))]
pub use placeholder::LoadedModel;

#[cfg(feature = "train")]
pub use train_impl::{LoadedModel, ModelConfig, ModelError};

#[cfg(feature = "train")]
pub mod arch;
#[cfg(feature = "train")]
pub use arch::{ArchAdapter, LayerDesc, LayerKind, LlamaAdapter, LoraTargetWeights, LoraWeights, SeamPoint};

#[cfg(feature = "train")]
mod train_impl {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, VarMap};
    use serde::{Deserialize, Serialize};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::Tokenizer;

    /// RMSNorm epsilon — fixed; the architecture is a fixture, not tuned.
    const RMS_EPS: f64 = 1e-5;
    /// The leaf module names LoRA may target (track 04 reads these).
    const LORA_DEFAULT_TARGETS: [&str; 2] = ["q_proj", "v_proj"];

    /// Architecture + size hyperparameters for the tiny decoder-only LM.
    ///
    /// Maps onto the subset of a HuggingFace `config.json` we support. Loading a
    /// config whose `architectures` is not a supported causal-LM yields
    /// [`ModelError::UnsupportedArch`] rather than a panic.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct ModelConfig {
        /// Vocabulary size (embedding rows and lm_head columns).
        pub vocab_size: usize,
        /// Residual-stream width.
        pub hidden_size: usize,
        /// MLP inner width (gate/up project to this, down projects back).
        pub intermediate_size: usize,
        /// Number of decoder layers.
        pub num_layers: usize,
        /// Number of attention heads (`hidden_size` must divide evenly).
        pub num_heads: usize,
        /// Maximum sequence length the learned positional table covers.
        pub max_seq_len: usize,
    }

    impl ModelConfig {
        /// A deliberately tiny config for offline fixtures and tests.
        ///
        /// 256-token vocab, 64-wide, 128 MLP, 2 layers, 2 heads, 128 positions —
        /// small enough to init + forward in milliseconds on CPU.
        pub fn tiny() -> Self {
            Self {
                vocab_size: 256,
                hidden_size: 64,
                intermediate_size: 128,
                num_layers: 2,
                num_heads: 2,
                max_seq_len: 128,
            }
        }

        /// Per-head width. Caller must ensure `hidden_size % num_heads == 0`.
        fn head_dim(&self) -> usize {
            self.hidden_size / self.num_heads
        }
    }

    /// The HuggingFace `config.json` subset we read off disk.
    #[derive(Deserialize)]
    struct HfConfig {
        #[serde(default)]
        architectures: Vec<String>,
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        #[serde(alias = "num_hidden_layers")]
        num_layers: usize,
        #[serde(alias = "num_attention_heads")]
        num_heads: usize,
        #[serde(default = "default_max_seq_len", alias = "max_position_embeddings")]
        max_seq_len: usize,
    }

    fn default_max_seq_len() -> usize {
        2048
    }

    /// Architectures this seam can materialize. Anything else is a clear error.
    fn arch_supported(arch: &str) -> bool {
        matches!(
            arch,
            "LlamaForCausalLM" | "LlamaModel" | "ScrtEvolveTinyCausalLM"
        )
    }

    /// Errors from the model-loader seam.
    ///
    /// Library-level error *type* (styleguide §1): callers at the CLI edge may
    /// wrap this in `anyhow`, but no loader path panics on SDK-reachable input.
    #[derive(Debug, thiserror::Error)]
    pub enum ModelError {
        /// The `config.json` named an architecture this seam cannot build.
        #[error("unsupported architecture: {0}")]
        UnsupportedArch(String),
        /// A required file was missing, unreadable, or malformed.
        #[error("model load failed: {0}")]
        Load(String),
        /// Tokenizer construction, encode, or decode failed.
        #[error("tokenizer error: {0}")]
        Tokenizer(String),
        /// A tensor had an unexpected shape or a config invariant was violated.
        #[error("shape error: {0}")]
        Shape(String),
    }

    impl From<candle_core::Error> for ModelError {
        fn from(e: candle_core::Error) -> Self {
            ModelError::Load(e.to_string())
        }
    }

    /// One decoder block: pre-attn norm, self-attention, pre-MLP norm, SwiGLU MLP.
    struct DecoderLayer {
        input_layernorm: RmsNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        post_attention_layernorm: RmsNorm,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
        num_heads: usize,
        head_dim: usize,
    }

    impl DecoderLayer {
        fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self, ModelError> {
            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let attn = vb.pp("self_attn");
            let mlp = vb.pp("mlp");
            Ok(Self {
                input_layernorm: candle_nn::rms_norm(h, RMS_EPS, vb.pp("input_layernorm"))?,
                q_proj: candle_nn::linear_no_bias(h, h, attn.pp("q_proj"))?,
                k_proj: candle_nn::linear_no_bias(h, h, attn.pp("k_proj"))?,
                v_proj: candle_nn::linear_no_bias(h, h, attn.pp("v_proj"))?,
                o_proj: candle_nn::linear_no_bias(h, h, attn.pp("o_proj"))?,
                post_attention_layernorm: candle_nn::rms_norm(
                    h,
                    RMS_EPS,
                    vb.pp("post_attention_layernorm"),
                )?,
                gate_proj: candle_nn::linear_no_bias(h, inter, mlp.pp("gate_proj"))?,
                up_proj: candle_nn::linear_no_bias(h, inter, mlp.pp("up_proj"))?,
                down_proj: candle_nn::linear_no_bias(inter, h, mlp.pp("down_proj"))?,
                num_heads: cfg.num_heads,
                head_dim: cfg.head_dim(),
            })
        }

        /// `x`: `[batch, seq, hidden]`. `mask`: additive causal mask
        /// `[seq, seq]`. Returns `[batch, seq, hidden]`.
        fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
            let (b, seq, _h) = x.dims3()?;
            let residual = x;
            let normed = self.input_layernorm.forward(x)?;

            let q = self.q_proj.forward(&normed)?;
            let k = self.k_proj.forward(&normed)?;
            let v = self.v_proj.forward(&normed)?;

            // [b, seq, heads, head_dim] -> [b, heads, seq, head_dim]
            let shape = (b, seq, self.num_heads, self.head_dim);
            let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
            let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
            let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

            let scale = 1f64 / (self.head_dim as f64).sqrt();
            // [b, heads, seq, seq]
            let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let scores = scores.broadcast_add(mask)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            // [b, heads, seq, head_dim]
            let ctx = probs.matmul(&v)?;
            // back to [b, seq, hidden]
            let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((
                b,
                seq,
                self.num_heads * self.head_dim,
            ))?;
            let attn_out = self.o_proj.forward(&ctx)?;
            let x = (residual + attn_out)?;

            let residual = &x;
            let normed = self.post_attention_layernorm.forward(&x)?;
            let gate = self.gate_proj.forward(&normed)?.silu()?;
            let up = self.up_proj.forward(&normed)?;
            let mlp_out = self.down_proj.forward(&(gate * up)?)?;
            Ok((residual + mlp_out)?)
        }
    }

    /// The tiny causal-LM graph. All weights are registered in the owning
    /// [`LoadedModel`]'s [`VarMap`] under the documented names.
    struct TinyLlama {
        embed_tokens: Embedding,
        pos_embed: Embedding,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        lm_head: Linear,
        device: Device,
        max_seq_len: usize,
    }

    impl TinyLlama {
        fn new(cfg: &ModelConfig, vb: VarBuilder, device: Device) -> Result<Self, ModelError> {
            if cfg.num_heads == 0 || cfg.hidden_size % cfg.num_heads != 0 {
                return Err(ModelError::Shape(format!(
                    "hidden_size {} not divisible by num_heads {}",
                    cfg.hidden_size, cfg.num_heads
                )));
            }
            let model = vb.pp("model");
            let embed_tokens =
                candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, model.pp("embed_tokens"))?;
            let pos_embed = candle_nn::embedding(
                cfg.max_seq_len,
                cfg.hidden_size,
                model.pp("embed_positions"),
            )?;
            let mut layers = Vec::with_capacity(cfg.num_layers);
            let layers_vb = model.pp("layers");
            for i in 0..cfg.num_layers {
                layers.push(DecoderLayer::new(cfg, layers_vb.pp(i))?);
            }
            let norm = candle_nn::rms_norm(cfg.hidden_size, RMS_EPS, model.pp("norm"))?;
            let lm_head =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
            Ok(Self {
                embed_tokens,
                pos_embed,
                layers,
                norm,
                lm_head,
                device,
                max_seq_len: cfg.max_seq_len,
            })
        }

        /// `input_ids`: `[batch, seq]` of u32 token ids. Returns logits
        /// `[batch, seq, vocab]`.
        fn forward(&self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
            let (_b, seq) = input_ids.dims2()?;
            if seq > self.max_seq_len {
                return Err(ModelError::Shape(format!(
                    "sequence length {seq} exceeds max_seq_len {}",
                    self.max_seq_len
                )));
            }
            let mut x = self.embed_tokens.forward(input_ids)?;
            // Learned absolute positions 0..seq, broadcast over the batch.
            let positions = Tensor::arange(0u32, seq as u32, &self.device)?;
            let pos = self.pos_embed.forward(&positions)?.unsqueeze(0)?;
            x = x.broadcast_add(&pos)?;

            let mask = causal_mask(seq, &self.device)?;
            for layer in &self.layers {
                x = layer.forward(&x, &mask)?;
            }
            let x = self.norm.forward(&x)?;
            Ok(self.lm_head.forward(&x)?)
        }
    }

    /// Additive causal mask `[seq, seq]`: 0 on/below the diagonal, -inf above.
    fn causal_mask(seq: usize, device: &Device) -> Result<Tensor, ModelError> {
        let mut data = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        Ok(Tensor::from_vec(data, (seq, seq), device)?)
    }

    /// A loaded model + tokenizer, ready for inference or training.
    ///
    /// The [`VarMap`] owns every trainable weight (track 04 attaches LoRA
    /// adapters and gradients to it). See the module header for the weight
    /// naming scheme.
    pub struct LoadedModel {
        /// The directory the weights/tokenizer were loaded from (empty for a
        /// random fixture).
        pub model_path: PathBuf,
        /// Architecture + size hyperparameters.
        pub config: ModelConfig,
        /// Text <-> token-id codec.
        pub tokenizer: Tokenizer,
        /// All trainable weights, keyed by the documented names.
        pub var_map: VarMap,
        /// The device the weights live on (always CPU for this seam).
        pub device: Device,
        /// LoRA-targetable leaf module names (`["q_proj", "v_proj"]` by default).
        pub target_modules: Vec<String>,
        /// When true (random fixtures), tokenize/detokenize use a lossless
        /// raw-byte codec instead of the held `tokenizer`. Real loaded models
        /// drive the file-backed tokenizer instead.
        byte_level: bool,
        model: TinyLlama,
    }

    impl std::fmt::Debug for LoadedModel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("LoadedModel")
                .field("model_path", &self.model_path)
                .field("config", &self.config)
                .field("target_modules", &self.target_modules)
                .field(
                    "num_vars",
                    &self
                        .var_map
                        .data()
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len(),
                )
                .finish_non_exhaustive()
        }
    }

    impl LoadedModel {
        /// Load real weights from a model directory.
        ///
        /// Reads `config.json` (mapped to [`ModelConfig`]; an unsupported
        /// `architectures` entry yields [`ModelError::UnsupportedArch`]),
        /// `tokenizer.json`, and `model.safetensors` into a fresh [`VarMap`].
        /// Missing or malformed files yield [`ModelError::Load`] — never a panic.
        pub fn load(model_path: &Path) -> Result<LoadedModel, ModelError> {
            let cfg_path = model_path.join("config.json");
            let cfg_bytes = std::fs::read(&cfg_path)
                .map_err(|e| ModelError::Load(format!("{}: {e}", cfg_path.display())))?;
            let hf: HfConfig = serde_json::from_slice(&cfg_bytes)
                .map_err(|e| ModelError::Load(format!("{}: {e}", cfg_path.display())))?;
            if let Some(arch) = hf.architectures.iter().find(|a| !arch_supported(a)) {
                return Err(ModelError::UnsupportedArch(arch.clone()));
            }
            let config = ModelConfig {
                vocab_size: hf.vocab_size,
                hidden_size: hf.hidden_size,
                intermediate_size: hf.intermediate_size,
                num_layers: hf.num_layers,
                num_heads: hf.num_heads,
                max_seq_len: hf.max_seq_len,
            };

            let tok_path = model_path.join("tokenizer.json");
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| ModelError::Tokenizer(format!("{}: {e}", tok_path.display())))?;

            let device = Device::Cpu;
            let mut var_map = VarMap::new();
            let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
            let model = TinyLlama::new(&config, vb, device.clone())?;

            let weights_path = model_path.join("model.safetensors");
            var_map
                .load(&weights_path)
                .map_err(|e| ModelError::Load(format!("{}: {e}", weights_path.display())))?;

            Ok(LoadedModel {
                model_path: model_path.to_path_buf(),
                config,
                tokenizer,
                var_map,
                device,
                target_modules: LORA_DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect(),
                byte_level: false,
                model,
            })
        }

        /// Build a fully in-memory fixture with weights seeded from `seed`.
        ///
        /// Every weight is filled deterministically (see the module header on
        /// determinism), and a byte-level tokenizer over `0..vocab_size` is built
        /// in memory, so no files are touched. Same `seed` + `config` →
        /// byte-identical weights. Both track 03 and track 04 tests call this.
        pub fn random_fixture(config: ModelConfig, seed: u64) -> Result<LoadedModel, ModelError> {
            let device = Device::Cpu;
            let mut var_map = VarMap::new();
            let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
            // Registers every var with its name/shape (init values are replaced).
            let model = TinyLlama::new(&config, vb, device.clone())?;
            seed_varmap(&mut var_map, seed, &device)?;

            let tokenizer = byte_level_tokenizer(config.vocab_size)?;

            Ok(LoadedModel {
                model_path: PathBuf::new(),
                config,
                tokenizer,
                var_map,
                device,
                target_modules: LORA_DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect(),
                byte_level: true,
                model,
            })
        }

        /// Run a forward pass. `input_ids`: `[batch, seq]` u32 tensor. Returns
        /// logits `[batch, seq, vocab_size]`. Differentiable (track 04 loss).
        pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
            self.model.forward(input_ids)
        }

        /// Number of decoder layers (arch introspection for `arch::ArchAdapter`).
        pub(crate) fn num_layers(&self) -> usize {
            self.config.num_layers
        }

        /// Token + learned-positional embedding for `input_ids: [batch, seq]`.
        /// Returns `[batch, seq, hidden]`. Used by `arch::LlamaAdapter`.
        pub(crate) fn embed(&self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
            let (_b, seq) = input_ids.dims2()?;
            let mut x = self.model.embed_tokens.forward(input_ids)?;
            let positions = Tensor::arange(0u32, seq as u32, &self.device)?;
            let pos = self.model.pos_embed.forward(&positions)?.unsqueeze(0)?;
            x = x.broadcast_add(&pos)?;
            Ok(x)
        }

        /// Run exactly decoder layer `idx` on `x: [batch, seq, hidden]` with a
        /// causal mask matching `seq`. Byte-identical to the slice inside
        /// `TinyLlama::forward`.
        pub(crate) fn apply_layer(&self, idx: usize, x: &Tensor) -> Result<Tensor, ModelError> {
            let (_b, seq, _h) = x.dims3()?;
            let mask = causal_mask(seq, &self.device)?;
            self.model.layers[idx].forward(x, &mask)
        }

        /// Final norm + lm_head, matching the tail of `TinyLlama::forward`.
        pub(crate) fn head(&self, x: &Tensor) -> Result<Tensor, ModelError> {
            let x = self.model.norm.forward(x)?;
            Ok(self.model.lm_head.forward(&x)?)
        }

        /// Overwrite a set of named weights and re-materialise the internal
        /// graph so `forward`/`apply_layer` reflect the update.
        ///
        /// candle's built `Linear`/`Embedding` modules capture cloned tensor
        /// snapshots at construction — subsequent `Var::set` or
        /// `VarMap::set_one` calls do NOT propagate into them. The proven
        /// happy path (mirroring `LoadedModel::load`) is: save updated
        /// weights to a safetensors, then rebuild the `VarMap` + graph from
        /// disk. That's what this helper does, atomically, for the LoRA
        /// compose-at-load merge.
        pub(crate) fn overwrite_weights(
            &mut self,
            updates: &[(String, Tensor)],
        ) -> Result<(), ModelError> {
            // Snapshot the current full weight set, apply the updates, and
            // rebuild via a temp-safetensors + VarMap::load round-trip.
            let mut all: HashMap<String, Tensor> = {
                let data = self.var_map.data().lock().unwrap_or_else(|e| e.into_inner());
                data.iter()
                    .map(|(k, v)| (k.clone(), v.as_detached_tensor()))
                    .collect()
            };
            for (name, tensor) in updates {
                if !all.contains_key(name) {
                    return Err(ModelError::Shape(format!("unknown weight: {name}")));
                }
                all.insert(name.clone(), tensor.clone());
            }
            let tmp_dir = std::env::temp_dir().join(format!(
                "scrt_evolve_merge_{}_{:p}",
                std::process::id(),
                self
            ));
            std::fs::create_dir_all(&tmp_dir)
                .map_err(|e| ModelError::Load(format!("{}: {e}", tmp_dir.display())))?;
            let path = tmp_dir.join("merged.safetensors");
            candle_core::safetensors::save(&all, &path)
                .map_err(|e| ModelError::Load(format!("{}: {e}", path.display())))?;

            // Build the graph from a tensor-map VarBuilder so the created
            // Linear/Embedding modules capture the merged tensors directly at
            // construction time (VarMap-backed builders re-issue Vars that
            // don't propagate post-hoc mutations into already-built modules).
            let vb = VarBuilder::from_tensors(all.clone(), DType::F32, &self.device);
            let model = TinyLlama::new(&self.config, vb, self.device.clone())?;
            // Also rebuild the VarMap so external readers (LoRA training,
            // save_safetensors) see the merged weights.
            let mut new_map = VarMap::new();
            let vb2 = VarBuilder::from_varmap(&new_map, DType::F32, &self.device);
            let _ = TinyLlama::new(&self.config, vb2, self.device.clone())?;
            new_map
                .load(&path)
                .map_err(|e| ModelError::Load(format!("{}: {e}", path.display())))?;
            self.var_map = new_map;
            self.model = model;
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Ok(())
        }

        /// Encode `text` to token ids (no special tokens added).
        ///
        /// Random fixtures use a lossless raw-byte codec (`id = byte`, mod
        /// `vocab_size`); loaded models drive the file-backed tokenizer.
        pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, ModelError> {
            if self.byte_level {
                let modulo = self.config.vocab_size.max(1) as u32;
                return Ok(text.bytes().map(|b| (b as u32) % modulo).collect());
            }
            let enc = self
                .tokenizer
                .encode(text, false)
                .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
            Ok(enc.get_ids().to_vec())
        }

        /// Decode token ids back to text.
        ///
        /// The inverse of [`Self::tokenize`]; for fixtures this reconstructs the
        /// original UTF-8 bytes (valid for `vocab_size >= 256`).
        pub fn detokenize(&self, ids: &[u32]) -> Result<String, ModelError> {
            if self.byte_level {
                let bytes: Vec<u8> = ids.iter().map(|&id| id as u8).collect();
                return String::from_utf8(bytes)
                    .map_err(|e| ModelError::Tokenizer(format!("invalid utf-8: {e}")));
            }
            self.tokenizer
                .decode(ids, false)
                .map_err(|e| ModelError::Tokenizer(e.to_string()))
        }

        /// Full VarMap weight names whose leaf module matches one of `targets`
        /// (e.g. `["q_proj", "v_proj"]`). Track 04 uses this to enumerate the
        /// modules a LoRA adapter wraps. Output is sorted for determinism.
        pub fn target_module_names(&self, targets: &[String]) -> Vec<String> {
            let data = self
                .var_map
                .data()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut out: Vec<String> = data
                .keys()
                .filter(|name| {
                    targets.iter().any(|t| {
                        name.ends_with(&format!("{t}.weight")) || name.ends_with(&format!(".{t}"))
                    })
                })
                .cloned()
                .collect();
            out.sort();
            out
        }

        /// Write all weights to `path` in safetensors format (atomic via a
        /// sibling temp file + rename). Track 04 reuses this pattern for adapter
        /// save; having it here proves the VarMap round-trips.
        pub fn save_safetensors(&self, path: &Path) -> Result<(), ModelError> {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let tmp = parent.join(format!(
                ".{}.tmp",
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("weights.safetensors")
            ));
            self.var_map
                .save(&tmp)
                .map_err(|e| ModelError::Load(format!("{}: {e}", tmp.display())))?;
            std::fs::rename(&tmp, path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                ModelError::Load(format!("rename to {}: {e}", path.display()))
            })?;
            Ok(())
        }
    }

    /// Deterministically overwrite every var in `var_map` from `seed`.
    ///
    /// Names are sorted first so the fill order — and thus each weight's PRNG
    /// stream — is independent of HashMap iteration order. Each weight gets its
    /// own sub-stream keyed by `seed` mixed with the weight's ordinal, so adding
    /// a weight does not shift the others' values for a given seed+config.
    fn seed_varmap(var_map: &mut VarMap, seed: u64, device: &Device) -> Result<(), ModelError> {
        let names: Vec<String> = {
            let data = var_map.data().lock().unwrap_or_else(|e| e.into_inner());
            let mut n: Vec<String> = data.keys().cloned().collect();
            n.sort();
            n
        };
        for (ord, name) in names.iter().enumerate() {
            let (elem_count, shape) = {
                let data = var_map.data().lock().unwrap_or_else(|e| e.into_inner());
                let var = data
                    .get(name)
                    .ok_or_else(|| ModelError::Shape(format!("var vanished: {name}")))?;
                (var.elem_count(), var.dims().to_vec())
            };
            let mut rng = SplitMix64::new(seed ^ (ord as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut buf = Vec::with_capacity(elem_count);
            for _ in 0..elem_count {
                buf.push(rng.next_normal() * 0.02);
            }
            let tensor = Tensor::from_vec(buf, shape, device)?;
            var_map
                .set_one(name, &tensor)
                .map_err(|e| ModelError::Load(format!("set {name}: {e}")))?;
        }
        Ok(())
    }

    /// A self-contained SplitMix64 PRNG with Box-Muller normal sampling.
    ///
    /// Inlined so fixture determinism does not depend on any external RNG
    /// crate's value stability across versions.
    struct SplitMix64 {
        state: u64,
        spare: Option<f32>,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare: None,
            }
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

        /// Standard-normal f32 via Box-Muller (caches the spare draw).
        fn next_normal(&mut self) -> f32 {
            if let Some(s) = self.spare.take() {
                return s;
            }
            // u1 in (0, 1] to keep ln finite.
            let u1 = 1.0 - self.next_f32();
            let u2 = self.next_f32();
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (std::f32::consts::TAU * u2).cos();
            let z1 = mag * (std::f32::consts::TAU * u2).sin();
            self.spare = Some(z1);
            z0
        }
    }

    /// Build an in-memory byte-level tokenizer over `vocab_size` tokens.
    ///
    /// Tokens are the byte values `0..min(vocab_size, 256)` rendered as the
    /// ByteLevel unicode aliases, giving a lossless byte <-> id round-trip for
    /// the fixture without needing a `tokenizer.json` on disk.
    fn byte_level_tokenizer(vocab_size: usize) -> Result<Tokenizer, ModelError> {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;

        let n = vocab_size.min(256);
        let alias = ByteLevelPre::alphabet();
        // Stable byte -> alias-char ordering so ids are reproducible.
        let mut chars: Vec<char> = alias.into_iter().collect();
        chars.sort_unstable();
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(n);
        for (i, c) in chars.into_iter().take(n).enumerate() {
            vocab.insert(c.to_string(), i as u32);
        }
        // Guarantee an unk token exists within range for unmapped pieces.
        let unk = "<unk>".to_string();
        if !vocab.contains_key(&unk) && n > 0 {
            vocab.insert(unk.clone(), 0);
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token(unk)
            .build()
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevelPre::new(false, false, false)));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));
        Ok(tokenizer)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fixture_forward_shape_and_finite() {
            let cfg = ModelConfig::tiny();
            let model = LoadedModel::random_fixture(cfg.clone(), 42).expect("fixture");
            let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (1, 5), &model.device).unwrap();
            let logits = model.forward(&ids).expect("forward");
            assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
            let flat = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(flat.iter().all(|v| v.is_finite()), "logits contain NaN/inf");
        }

        #[test]
        fn same_seed_same_q_proj() {
            let cfg = ModelConfig::tiny();
            let a = LoadedModel::random_fixture(cfg.clone(), 7).unwrap();
            let b = LoadedModel::random_fixture(cfg, 7).unwrap();
            let name = "model.layers.0.self_attn.q_proj.weight";
            let wa = a.var_map.data().lock().unwrap()[name]
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let wb = b.var_map.data().lock().unwrap()[name]
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(wa, wb, "same seed must yield identical q_proj weights");
        }

        #[test]
        fn save_then_reload_round_trips() {
            let cfg = ModelConfig::tiny();
            let model = LoadedModel::random_fixture(cfg, 13).unwrap();
            let dir =
                std::env::temp_dir().join(format!("scrt_evolve_model_rt_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("model.safetensors");
            model.save_safetensors(&path).expect("save");

            let loaded = candle_core::safetensors::load(&path, &model.device).expect("reload");
            let name = "model.layers.0.self_attn.v_proj.weight";
            let original = model.var_map.data().lock().unwrap()[name]
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reloaded = loaded
                .get(name)
                .expect("weight present after reload")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(original.len(), reloaded.len());
            assert_eq!(original, reloaded, "weight values must round-trip");
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn unsupported_arch_errors_not_panics() {
            let dir =
                std::env::temp_dir().join(format!("scrt_evolve_model_arch_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let cfg = r#"{
                "architectures": ["MambaForCausalLM"],
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_layers": 2,
                "num_heads": 2
            }"#;
            std::fs::write(dir.join("config.json"), cfg).unwrap();
            let err = LoadedModel::load(&dir).expect_err("must reject Mamba");
            assert!(
                matches!(err, ModelError::UnsupportedArch(ref a) if a == "MambaForCausalLM"),
                "expected UnsupportedArch, got {err:?}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn target_module_names_finds_q_and_v() {
            let cfg = ModelConfig::tiny();
            let model = LoadedModel::random_fixture(cfg.clone(), 1).unwrap();
            let targets = vec!["q_proj".to_string(), "v_proj".to_string()];
            let names = model.target_module_names(&targets);
            // 2 layers * (q_proj + v_proj) = 4 names.
            assert_eq!(names.len(), cfg.num_layers * 2);
            assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
            assert!(names.contains(&"model.layers.1.self_attn.v_proj.weight".to_string()));
        }

        #[test]
        fn tokenize_detokenize_round_trips() {
            let model = LoadedModel::random_fixture(ModelConfig::tiny(), 3).unwrap();
            let text = "hello world";
            let ids = model.tokenize(text).expect("tokenize");
            assert!(!ids.is_empty());
            let back = model.detokenize(&ids).expect("detokenize");
            assert_eq!(back, text);
        }
    }
}
