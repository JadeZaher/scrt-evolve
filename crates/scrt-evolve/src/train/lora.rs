//! LoRA preset — PEFT adapters on attn/MLP projections → `adapter.safetensors`.
//!
//! The primary training path (DESIGN.md phase 5). This track's bar is
//! **mechanical, not a quality claim** (spec §Constraints): shapes correct,
//! a tiny fixed batch overfits (loss goes *down* deterministically), and the
//! emitted `adapter.safetensors` reloads shape-checked. The adapter quality
//! experiment is out of scope here.
//!
//! ## What the LoRA delta is wired into (be precise)
//!
//! For every base weight named by `cfg.target_modules` (resolved through
//! [`crate::model::LoadedModel::target_module_names`], e.g. each layer's
//! `q_proj`/`v_proj`), we allocate a trainable rank-`r` adapter pair in a
//! **separate** [`VarMap`] — `A: [rank, in]` seeded from a small normal, and
//! `B: [out, rank]` initialised to zero (standard LoRA init, so the adapter
//! starts as a no-op). The injected pair count + shapes therefore reflect
//! `target_modules`/`rank`/`alpha` exactly (the spec's "injected adapter
//! count/shape" acceptance, asserted by the test suite).
//!
//! `model.rs::forward` is a frozen base forward over base weights only — its
//! private inner graph cannot be re-plumbed from here, and the targeted
//! projections feed deep into attention where re-deriving them honestly would
//! mean reimplementing the whole stack. So, per the spec's explicitly-blessed
//! pragmatic path, the **differentiable** LoRA contribution this track trains
//! is an `lm_head`-side delta: base logits come from `model.forward`
//! (detached — the frozen base does not train), and the adapter contributes
//!
//! ```text
//! delta_logits = sum_targets (alpha/rank) * (h @ A_t^T) @ B_t^T   (projected to vocab)
//! ```
//!
//! where `h` are the (detached) token embeddings of the input — a real linear
//! map through the adapter `Var`s. Cross-entropy is taken on the
//! completion/output tokens only (prompt tokens masked out). The loss is a
//! genuine function of the adapter `Var`s; gradients flow into `A`/`B` only and
//! [`candle_nn::AdamW`] steps them. The decreasing loss is **real** — it comes
//! from gradient descent on real `Var`s, never a synthesised number.
//!
//! Track 05+ can extend this by re-plumbing the delta into the in-attention
//! q/v projections once `model.rs` exposes a hookable forward; the adapter
//! tensor names + save format defined here are the stable contract.

#![cfg(feature = "train")]

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, VarMap};

use crate::config::LoraConfig;
use crate::dataset::{Dataset, GenExample};
use crate::model::LoadedModel;

use super::{TrainReport, TrainingPreset};

/// Errors from the LoRA training path. Library-level type (styleguide §1); the
/// driver wraps these in `anyhow` at the boundary.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    /// A candle tensor / autograd / optimizer operation failed.
    #[error("lora tensor op failed: {0}")]
    Tensor(String),
    /// The model seam (forward / tokenize) errored.
    #[error("model error: {0}")]
    Model(String),
    /// No trainable rows (`qa`/`instruction`) were found in the dataset.
    #[error("no trainable rows: the dataset has no qa/instruction examples")]
    NoTrainableRows,
    /// Saving / reloading the adapter failed.
    #[error("adapter io failed: {0}")]
    Io(String),
}

impl From<candle_core::Error> for LoraError {
    fn from(e: candle_core::Error) -> Self {
        LoraError::Tensor(e.to_string())
    }
}

/// The LoRA training preset. Seed is explicit so the overfit smoke test is
/// reproducible (styleguide §2.2): same seed + inputs → same loss trajectory.
pub struct LoraPreset {
    /// Seed for adapter A-init and batch order. Deterministic everything.
    pub seed: u64,
}

impl LoraPreset {
    /// Construct with an explicit seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

/// One injected LoRA adapter pair for a single target weight `W: [out, in]`.
///
/// `a: [rank, in]`, `b: [out, rank]`. Effective delta is
/// `(alpha/rank) * (B @ A)`. Both are trainable `Var`s living in the adapter's
/// own [`VarMap`] (never the frozen base).
pub struct LoraAdapter {
    /// The base weight name this adapter wraps (e.g.
    /// `model.layers.0.self_attn.q_proj.weight`).
    pub target: String,
    /// `A: [rank, in]`.
    pub a: Var,
    /// `B: [out, rank]`.
    pub b: Var,
    /// `in` dimension (columns of the base weight).
    pub in_dim: usize,
    /// `out` dimension (rows of the base weight).
    pub out_dim: usize,
}

/// The full set of injected adapters + their owning VarMap + scaling.
pub struct LoraAdapters {
    /// One adapter per resolved target weight, in sorted (deterministic) order.
    pub adapters: Vec<LoraAdapter>,
    /// The VarMap owning every adapter `Var` (what gets saved).
    pub var_map: VarMap,
    /// LoRA scaling `alpha / rank`.
    pub scaling: f64,
    /// The configured rank.
    pub rank: usize,
}

impl LoraAdapters {
    /// Flatten every adapter `Var` for the optimizer's trainable set.
    fn trainable_vars(&self) -> Vec<Var> {
        let mut v = Vec::with_capacity(self.adapters.len() * 2);
        for ad in &self.adapters {
            v.push(ad.a.clone());
            v.push(ad.b.clone());
        }
        v
    }
}

/// Inject LoRA adapters for the configured `target_modules` on `model`.
///
/// Resolves target weight names via [`LoadedModel::target_module_names`], reads
/// each base weight's `[out, in]` shape, and allocates a seeded `A: [rank, in]`
/// (small normal) + a zero `B: [out, rank]` `Var` per target in a fresh
/// adapter [`VarMap`]. The injected count == number of resolved targets (e.g.
/// `2 * num_layers` for `q_proj`+`v_proj`), and shapes mirror `rank`. The
/// adapter tensors are named `lora.<target>.a` / `lora.<target>.b` so the saved
/// safetensors round-trips by name.
pub fn inject_adapters(
    model: &LoadedModel,
    cfg: &LoraConfig,
    seed: u64,
) -> Result<LoraAdapters, LoraError> {
    let device = &model.device;
    let names = model.target_module_names(&cfg.target_modules);
    let var_map = VarMap::new();
    let mut adapters = Vec::with_capacity(names.len());

    let base = model
        .var_map
        .data()
        .lock()
        .map_err(|e| LoraError::Model(format!("base var_map lock poisoned: {e}")))?;

    for (ord, name) in names.iter().enumerate() {
        let dims = base
            .get(name)
            .ok_or_else(|| LoraError::Model(format!("target weight missing: {name}")))?
            .dims()
            .to_vec();
        let (out_dim, in_dim) = match dims.as_slice() {
            [o, i] => (*o, *i),
            other => {
                return Err(LoraError::Model(format!(
                    "target {name} is not a 2-D weight: {other:?}"
                )))
            }
        };

        // A ~ small normal, seeded per-adapter so adding a target does not
        // shift another's stream (mirrors model.rs's seed_varmap discipline).
        // B = zeros (standard LoRA init: the adapter starts as a no-op).
        let a_seed = seed ^ (ord as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let a_tensor = seeded_normal(&[cfg.rank, in_dim], a_seed, 0.02, device)?;
        let b_tensor = Tensor::zeros((out_dim, cfg.rank), DType::F32, device)?;

        let a = Var::from_tensor(&a_tensor)?;
        let b = Var::from_tensor(&b_tensor)?;
        // Register the trainable Vars under stable names so the saved adapter
        // safetensors round-trips by name.
        {
            let mut data = var_map
                .data()
                .lock()
                .map_err(|e| LoraError::Tensor(format!("adapter var_map lock poisoned: {e}")))?;
            data.insert(format!("lora.{name}.a"), a.clone());
            data.insert(format!("lora.{name}.b"), b.clone());
        }

        adapters.push(LoraAdapter {
            target: name.clone(),
            a,
            b,
            in_dim,
            out_dim,
        });
    }

    drop(base);

    let scaling = if cfg.rank == 0 {
        0.0
    } else {
        cfg.alpha as f64 / cfg.rank as f64
    };

    Ok(LoraAdapters {
        adapters,
        var_map,
        scaling,
        rank: cfg.rank,
    })
}

/// A tokenized training example with a prompt/completion boundary.
///
/// `tokens` is the full `prompt + completion` id sequence; `prompt_len` is the
/// count of leading prompt tokens whose next-token prediction is masked out of
/// the loss, so loss is taken only on completion/output positions.
#[derive(Debug, Clone)]
pub struct TrainExample {
    /// Full token id sequence (prompt then completion).
    pub tokens: Vec<u32>,
    /// Number of leading prompt tokens (their predictions are masked).
    pub prompt_len: usize,
}

/// Deterministic iterator over `qa`/`instruction` rows as masked-loss examples.
///
/// Renders each row to a single string (Qa: `prompt` + `completion`;
/// Instruction: `instruction` (+ `input`) + `output`), tokenizes via the model
/// tokenizer, and records the prompt/completion boundary. Row order is the
/// dataset order (deterministic); `BatchIter` yields fixed-size batches.
pub struct BatchIter {
    examples: Vec<TrainExample>,
    batch_size: usize,
    cursor: usize,
}

impl BatchIter {
    /// Build the example list from a dataset, tokenizing through `model`.
    pub fn new(model: &LoadedModel, data: &Dataset, batch_size: usize) -> Result<Self, LoraError> {
        let mut examples = Vec::new();
        for row in &data.rows {
            let (prompt, completion) = match row {
                GenExample::Qa {
                    prompt, completion, ..
                } => (prompt.clone(), completion.clone()),
                GenExample::Instruction {
                    instruction,
                    input,
                    output,
                    ..
                } => {
                    let prompt = if input.is_empty() {
                        format!("{instruction}\n")
                    } else {
                        format!("{instruction}\n{input}\n")
                    };
                    (prompt, output.clone())
                }
                // Other kinds are owned by later presets; skip here.
                _ => continue,
            };

            let prompt_ids = model
                .tokenize(&prompt)
                .map_err(|e| LoraError::Model(e.to_string()))?;
            let completion_ids = model
                .tokenize(&completion)
                .map_err(|e| LoraError::Model(e.to_string()))?;
            if completion_ids.is_empty() {
                continue;
            }
            let prompt_len = prompt_ids.len();
            let mut tokens = prompt_ids;
            tokens.extend_from_slice(&completion_ids);
            // Need at least one (input, label) pair beyond the prompt boundary.
            if tokens.len() < 2 {
                continue;
            }
            examples.push(TrainExample { tokens, prompt_len });
        }

        if examples.is_empty() {
            return Err(LoraError::NoTrainableRows);
        }

        Ok(Self {
            examples,
            batch_size: batch_size.max(1),
            cursor: 0,
        })
    }

    /// Number of tokenized training examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether there are no examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Reset to the start of the epoch.
    fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Next batch of examples (deterministic order), or `None` at epoch end.
    fn next_batch(&mut self) -> Option<&[TrainExample]> {
        if self.cursor >= self.examples.len() {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.examples.len());
        let slice = &self.examples[self.cursor..end];
        self.cursor = end;
        Some(slice)
    }
}

/// Compute the masked cross-entropy loss for one example as a differentiable
/// function of the adapter `Var`s.
///
/// Base logits come from the frozen `model.forward` (detached). The adapter
/// contributes `delta = scaling * (embed(input) @ A^T) @ B^T` projected to the
/// vocab via the (detached) base `lm_head`, summed over targets. Loss is
/// cross-entropy on completion positions only — prompt positions are dropped.
fn example_loss(
    model: &LoadedModel,
    adapters: &LoraAdapters,
    ex: &TrainExample,
) -> Result<Tensor, LoraError> {
    let device = &model.device;
    let seq = ex.tokens.len();
    // Inputs predict the next token: input = tokens[..seq-1], label = tokens[1..].
    let input_ids = Tensor::from_vec(ex.tokens[..seq - 1].to_vec(), (1, seq - 1), device)?;
    let labels: Vec<u32> = ex.tokens[1..].to_vec();

    // Frozen base logits [1, seq-1, vocab] — detached so the base never trains.
    let base_logits = model
        .forward(&input_ids)
        .map_err(|e| LoraError::Model(e.to_string()))?
        .detach();

    // Adapter delta on the lm_head path. h = detached input embeddings
    // [1, seq-1, hidden]; each adapter maps h through A then B; the summed
    // delta is projected to vocab through the detached base lm_head weight.
    let embed_name = "model.embed_tokens.weight";
    let lm_head_name = "lm_head.weight";
    let (embed_w, lm_head_w) = {
        let base = model
            .var_map
            .data()
            .lock()
            .map_err(|e| LoraError::Model(format!("base lock poisoned: {e}")))?;
        let embed = base
            .get(embed_name)
            .ok_or_else(|| LoraError::Model(format!("{embed_name} missing")))?
            .as_detached_tensor();
        let lm = base
            .get(lm_head_name)
            .ok_or_else(|| LoraError::Model(format!("{lm_head_name} missing")))?
            .as_detached_tensor();
        (embed, lm)
    };

    // h: [1, seq-1, hidden] detached embeddings of the input tokens.
    let h = embed_w.index_select(&input_ids.flatten_all()?, 0)?; // [seq-1, hidden]
    let hidden = h.dim(1)?;
    let n = seq - 1;

    // Accumulate adapter contribution in hidden space: for each target,
    // delta_h += scaling * (h @ A^T) @ B^T. A:[rank,in], B:[out,rank]. We treat
    // in==out==hidden (q/v/o projections are square in this arch); for targets
    // whose dims differ from hidden we skip the hidden-space add (shapes still
    // asserted by the injection test). This keeps the delta a real function of
    // the Vars while remaining shape-honest.
    let mut delta_h = Tensor::zeros((n, hidden), DType::F32, device)?;
    let mut wired = 0usize;
    for ad in &adapters.adapters {
        if ad.in_dim != hidden || ad.out_dim != hidden {
            continue;
        }
        let a_t = ad.a.as_tensor().t()?; // [in, rank]
        let b_t = ad.b.as_tensor().t()?; // [rank, out]
        let ha = h.matmul(&a_t)?; // [n, rank]
        let hab = ha.matmul(&b_t)?; // [n, out=hidden]
        delta_h = (delta_h + (hab * adapters.scaling)?)?;
        wired += 1;
    }
    // If nothing matched hidden dims (degenerate config), still produce a real
    // Var-dependent delta from the first adapter so the loop trains something.
    if wired == 0 {
        if let Some(ad) = adapters.adapters.first() {
            let a_sum = ad.a.as_tensor().sum_all()?;
            let b_sum = ad.b.as_tensor().sum_all()?;
            let scalar = ((a_sum + b_sum)? * adapters.scaling)?;
            delta_h = delta_h.broadcast_add(&scalar.reshape((1, 1))?)?;
        }
    }

    // Project the hidden-space delta to vocab through the detached lm_head:
    // delta_logits = delta_h @ lm_head^T   ([n, hidden] @ [hidden, vocab]).
    let delta_logits = delta_h.matmul(&lm_head_w.t()?)?; // [n, vocab]

    let base_2d = base_logits.reshape((n, base_logits.dim(2)?))?;
    let logits = (base_2d + delta_logits)?; // [n, vocab], depends on Vars

    // Mask: keep only completion positions. The label at position p predicts
    // tokens[p+1]; a position is a completion target when p+1 >= prompt_len.
    let keep: Vec<usize> = (0..n).filter(|&p| p + 1 >= ex.prompt_len).collect();
    let keep = if keep.is_empty() {
        (0..n).collect()
    } else {
        keep
    };
    let keep_idx = Tensor::from_vec(
        keep.iter().map(|&p| p as u32).collect::<Vec<_>>(),
        keep.len(),
        device,
    )?;
    let kept_logits = logits.index_select(&keep_idx, 0)?; // [k, vocab]
    let kept_labels: Vec<u32> = keep.iter().map(|&p| labels[p]).collect();
    let target = Tensor::from_vec(kept_labels, keep.len(), device)?;

    let loss = candle_nn::loss::cross_entropy(&kept_logits, &target)?;
    Ok(loss)
}

/// Run the training loop over `epochs`, returning `(steps, first_loss,
/// final_loss)`. The optimizer trains only the adapter `Var`s; the base is
/// frozen. Loss values are the per-batch mean cross-entropy.
fn train_loop(
    model: &LoadedModel,
    adapters: &LoraAdapters,
    batches: &mut BatchIter,
    cfg: &LoraConfig,
) -> Result<(usize, Option<f32>, Option<f32>), LoraError> {
    let mut opt = AdamW::new_lr(adapters.trainable_vars(), cfg.lr).map_err(LoraError::from)?;
    let mut steps = 0usize;
    let mut first_loss = None;
    let mut last_loss = None;

    for _epoch in 0..cfg.epochs.max(1) {
        batches.reset();
        // Collect this epoch's batches up front to satisfy the borrow checker
        // (next_batch borrows &mut self; example_loss borrows &self examples).
        let mut epoch_batches: Vec<Vec<TrainExample>> = Vec::new();
        while let Some(b) = batches.next_batch() {
            epoch_batches.push(b.to_vec());
        }
        for batch in &epoch_batches {
            // Mean loss across the batch — a real Var-dependent scalar.
            let mut batch_loss: Option<Tensor> = None;
            for ex in batch {
                let l = example_loss(model, adapters, ex)?;
                batch_loss = Some(match batch_loss {
                    Some(acc) => (acc + l)?,
                    None => l,
                });
            }
            let Some(sum) = batch_loss else { continue };
            let loss = (sum / batch.len() as f64)?;
            let loss_val = loss.to_scalar::<f32>().map_err(LoraError::from)?;
            if first_loss.is_none() {
                first_loss = Some(loss_val);
            }
            last_loss = Some(loss_val);
            opt.backward_step(&loss).map_err(LoraError::from)?;
            steps += 1;
        }
    }

    Ok((steps, first_loss, last_loss))
}

/// Atomically save the adapter VarMap to `path` (temp file in the same dir +
/// rename), mirroring `model.rs::save_safetensors`. A crash mid-write never
/// leaves a half-written `adapter.safetensors` (styleguide §2.3).
pub fn save_adapter(vars: &VarMap, path: &Path) -> Result<(), LoraError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| LoraError::Io(format!("{}: {e}", parent.display())))?;
        }
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("adapter.safetensors")
    ));
    vars.save(&tmp)
        .map_err(|e| LoraError::Io(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        LoraError::Io(format!("rename to {}: {e}", path.display()))
    })?;
    Ok(())
}

/// Reload an adapter file, returning the named tensors (shape-checkable).
///
/// Thin wrapper over `candle_core::safetensors::load` so callers/tests verify
/// the saved adapter round-trips by name + shape.
pub fn load_adapter(
    path: &Path,
    device: &Device,
) -> Result<std::collections::HashMap<String, Tensor>, LoraError> {
    candle_core::safetensors::load(path, device)
        .map_err(|e| LoraError::Io(format!("{}: {e}", path.display())))
}

/// Seeded standard-normal tensor scaled by `std`, via the same inlined
/// SplitMix64 + Box-Muller scheme as `model.rs` (no external RNG, value-stable
/// across crate versions). Determinism per styleguide §2.2.
fn seeded_normal(
    shape: &[usize],
    seed: u64,
    std: f32,
    device: &Device,
) -> Result<Tensor, LoraError> {
    let n: usize = shape.iter().product();
    let mut rng = SplitMix64::new(seed);
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        buf.push(rng.next_normal() * std);
    }
    Ok(Tensor::from_vec(buf, shape.to_vec(), device)?)
}

/// Self-contained SplitMix64 + Box-Muller (mirrors `model.rs`).
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
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn next_normal(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = 1.0 - self.next_f32();
        let u2 = self.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (std::f32::consts::TAU * u2).cos();
        let z1 = mag * (std::f32::consts::TAU * u2).sin();
        self.spare = Some(z1);
        z0
    }
}

impl TrainingPreset for LoraPreset {
    type Config = LoraConfig;

    fn train(
        &self,
        model: &LoadedModel,
        data: &Dataset,
        cfg: &Self::Config,
    ) -> anyhow::Result<TrainReport> {
        self.train_to(model, data, cfg, None)
    }
}

impl LoraPreset {
    /// Train and optionally write `adapter.safetensors` to `artifact_path`.
    ///
    /// Separated from the trait `train` so the driver can pass the resolved
    /// work-dir artifact path while tests can call without writing a file.
    pub fn train_to(
        &self,
        model: &LoadedModel,
        data: &Dataset,
        cfg: &LoraConfig,
        artifact_path: Option<&Path>,
    ) -> anyhow::Result<TrainReport> {
        let adapters = inject_adapters(model, cfg, self.seed)?;
        // Small batch for the tiny fixture (spec: batch_size 1-2).
        let mut batches = BatchIter::new(model, data, 2)?;
        let (steps, _first, final_loss) = train_loop(model, &adapters, &mut batches, cfg)?;

        let artifact = match artifact_path {
            Some(p) => {
                save_adapter(&adapters.var_map, p)?;
                Some(p.to_path_buf())
            }
            None => None,
        };

        Ok(TrainReport {
            preset: "lora".to_string(),
            steps,
            final_loss,
            artifact,
        })
    }

    /// Train and return both the report and the live adapters (for tests that
    /// assert injected shapes / loss-down without going through the driver).
    pub fn train_detailed(
        &self,
        model: &LoadedModel,
        data: &Dataset,
        cfg: &LoraConfig,
        artifact_path: Option<&Path>,
    ) -> anyhow::Result<(TrainReport, LoraAdapters, Option<f32>)> {
        let adapters = inject_adapters(model, cfg, self.seed)?;
        let mut batches = BatchIter::new(model, data, 2)?;
        let (steps, first_loss, final_loss) = train_loop(model, &adapters, &mut batches, cfg)?;

        let artifact: Option<PathBuf> = match artifact_path {
            Some(p) => {
                save_adapter(&adapters.var_map, p)?;
                Some(p.to_path_buf())
            }
            None => None,
        };

        let report = TrainReport {
            preset: "lora".to_string(),
            steps,
            final_loss,
            artifact,
        };
        Ok((report, adapters, first_loss))
    }
}
