# Train / Full + Pretrain — Specification

## Goal
DESIGN.md phase 7. Add the two heavier generative presets: `full` (update all
weights) and `pretrain` (continued causal-LM pretraining on the **raw** corpus,
not QA pairs — domain adaptation). Both extend the `TrainingPreset` trait from
track 04.

## Scope
- `full.rs`: full-weight finetune (`[train.full]`: `lr`, `epochs`,
  `grad_accum`) → full weights artifact. Consumes `qa`/`instruction` rows like
  LoRA but updates the whole model; memory-heavy by nature.
- `pretrain.rs`: continued pretraining (`[train.pretrain]`: `lr`,
  `block_size`) over `completion`-kind rows (raw corpus passages, NOT generated
  QA). The `discover`/`generate` path can emit `completion` rows from raw
  passages, or `pretrain` reads the corpus directly.
- Both reuse the `model.rs` loader, the batch/optimizer machinery, and the
  `grad_accum` path. `full` shares most of the LoRA loop minus the adapter
  injection; factor shared training plumbing so all three presets reuse it.
- The `train()` driver routes `completion` rows to `pretrain`,
  `qa`/`instruction` to `full`/`lora`.

## Constraints
- Behind `--features train`.
- `full` is **memory-heavy** (DESIGN.md v1 bar: "works, memory-heavy") —
  support `grad_accum` and document the memory expectation; don't pretend it's
  cheap.
- `pretrain` consumes a **different dataset shape** (`completion`/raw passages)
  than the instruction presets — keep the routing explicit.
- Mechanical validation only (shapes, overfit-tiny-batch, artifact reloads);
  no quality target.

## Acceptance
- `full` preset on a tiny fixture produces a full-weights artifact that
  reloads; overfit smoke drives loss down; `grad_accum > 1` accumulates across
  micro-batches (assert step count vs optimizer-step count).
- `pretrain` preset over `completion`/raw passages runs a causal-LM loop with
  `block_size` chunking; overfit smoke drives loss down; artifact reloads.
- `train()` driver routes `completion` → `pretrain` and `qa`/`instruction` →
  `full` correctly (a mixed dataset is partitioned as expected).
- `scrt-evolve train --preset full` and `--preset pretrain` run standalone from
  on-disk data.

## Dependencies
Track 04 (`TrainingPreset` trait, model loader, shared training plumbing).
Track 02/03 for datasets (and raw-corpus `completion` rows).
