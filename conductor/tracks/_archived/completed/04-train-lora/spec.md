---
type: Track Spec
title: Train / LoRA
description: The TrainingPreset trait and the LoRA preset — the primary training path (DESIGN.md phase 5).
tags: [track-04, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Train / LoRA — Specification

## Goal
DESIGN.md phase 5 — **the primary training path**. Define the `TrainingPreset`
trait and the `train()` driver, implement the model loader + LoRA injection +
training loop, and emit `adapter.safetensors`. Deepen the **PyO3 bridge** with
a training-step seam so conventional Python tooling (`peft`/`trl`/`torch`) can
drive the preset against scrt-evolve datasets.

## Scope
- `TrainingPreset` trait + `train::run(&cfg, &dataset) -> TrainReport`
  (DESIGN.md §The three core traits #3). The driver routes dataset `kind` rows
  to the right preset.
- `lora.rs`: PEFT LoRA on attn/MLP projections (`[train.lora]`: `rank`,
  `alpha`, `target_modules`, `lr`, `epochs`). candle injection + optimizer +
  training loop + gradient handling. Output `adapter.safetensors`.
- Consumes `qa`/`instruction` rows from `dataset.jsonl`. Reuses the `model.rs`
  loader from track 03 (ONE arch first).
- **PyO3 training-step seam:** under `--features pyo3`, expose the dataset +
  enough of the loop (batch iterator, a step/forward hook, adapter save) that a
  Python `peft`/`trl` script can train using scrt-evolve as the data+config
  layer. This is the explicit merge point with the existing Python training
  stack.
- `evolve train fit --preset lora [--data dataset.jsonl]` (CLI), runnable
  standalone from an on-disk dataset.

## Constraints
- Behind `--features train` (candle) and, for the bridge, `--features pyo3`.
- candle's PEFT ecosystem is thin (DESIGN.md §Honest risks) — LoRA injection,
  optimizer, grad checkpointing are hand-built. Keep `model.rs` the clean seam;
  do NOT try to support every arch in this track.
- Validation is **mechanical, not quality-chasing**: shapes correct, overfit a
  tiny batch (loss goes down), adapter loads back. The quality experiment is
  out of scope.
- The Rust loop and the Python-driven loop must consume the **same dataset
  rows** and produce a **loadable adapter** in the same format.

## Acceptance
- `train::run` on a tiny fixture dataset + tiny model produces an
  `adapter.safetensors` that re-loads (shape-checked).
- Overfit smoke: training on a handful of examples drives loss down across a
  few steps (deterministic-seeded).
- `target_modules`/`rank`/`alpha` from config are reflected in the injected
  adapter (assert injected layer count/shape).
- Under `--features pyo3`, a Python `peft`/`trl`-style script trains one step
  on the same dataset via the bridge and saves a compatible adapter.
- `evolve train fit --preset lora --data dataset.jsonl` runs standalone.

## Dependencies
Track 02 (`Dataset`), track 03 (`model.rs` loader), track 00 (`train`/`pyo3`
features). Establishes the `TrainingPreset` trait that tracks 05/06/07 extend.
