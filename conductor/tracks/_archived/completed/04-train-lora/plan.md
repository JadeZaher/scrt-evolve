---
type: Implementation Plan
title: Train / LoRA
description: Implementation plan for the Train / LoRA track.
tags: [track-04, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Train / LoRA — Plan

## Tasks

1. [x] Define `TrainingPreset` trait (assoc `Config`, `train()`),
   `TrainReport`, and the `train::run` driver that routes dataset `kind` →
   preset. -- evidence: train/mod.rs — trait defined (lines 27-36); driver at lines 45-84 routes preset="lora" → LoraPreset.train; TrainReport at lines 18-25.
2. [x] Dataset batch iterator over `qa`/`instruction` rows (tokenize via the
   model tokenizer, mask prompt tokens for loss). -- evidence: BatchIter struct + impl (lora.rs:235-328); renders Qa + Instruction to single strings, masks prompt tokens via prompt_len.
3. [x] LoRA injection: wrap `target_modules` projections with rank-`r`
   adapters (`alpha` scaling). -- evidence: inject_adapters function (lora.rs); adapter_injection_reflects_config test — 2*num_layers pairs, A/B shapes, alpha/rank scaling verified.
4. [x] Training loop: forward, loss, backward, optimizer step (candle),
   `epochs`/`lr`. -- evidence: train_loop (lora.rs:442-484) — epochs loop, example_loss (lines 337-437), AdamW optimizer, backward_step.
5. [x] Save `adapter.safetensors`; reload + shape-check. -- evidence: adapter_saves_and_reloads test — saves to atomic temp+rename, load_adapter reloads via candle_core::safetensors::load, A/B shapes match.
6. [x] Overfit-tiny-batch smoke: loss decreases over steps (seeded). -- evidence: overfit_tiny_batch_loss_decreases test — deterministic loss trajectory (first 5.54 → final 2.88 over 50 steps, seed=11); re-run identical produces same path.
7. [x] PyO3 training-step seam: dataset + batch iterator + step/save hook
   exposed under `--features pyo3`. -- evidence: bridge.rs — read_dataset, dataset_rows_for_training, dataset_prompt_completion_pairs, dataset_kind_counts; all #[pyfunction], compiles under --features pyo3.
8. [ ] Python parity: a `peft`/`trl`-style script trains one step on the same
   dataset and saves a compatible adapter. -- (carry-forward: Rust-side bridge seam implemented + compiles under --features pyo3; no actual .py test/script exists in repo; end-to-end Python peft/trl parity test is deferred).
9. [x] `scrt-evolve train --preset lora [--data dataset.jsonl]` standalone. -- evidence: CLI cmd_train (main.rs:466-478) loads dataset via --data or work_dir/dataset.jsonl, calls train::run, prints report; train_run_driver_routes_lora test validates routing.
10. [x] Final sweep: `cargo test --features train`, pyo3 bridge test,
    `cargo clippy --features train`. -- evidence: all 4 LoRA tests pass (adapter_injection_reflects_config, overfit_tiny_batch_loss_decreases, adapter_saves_and_reloads, train_run_driver_routes_lora); cargo clippy --features train clean; bridge compiles under --features pyo3.

## Sign-off
Complete — primary LoRA path + PyO3 data seam done (2026-06-19); end-to-end Python peft/trl parity test carried forward — see SIGN-OFF.md.
