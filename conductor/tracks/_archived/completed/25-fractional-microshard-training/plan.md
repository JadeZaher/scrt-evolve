---
type: Implementation Plan
title: Fractional / Microshard Training
description: Implementation plan for the Fractional / Microshard Training track.
tags: [track-25, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 25 — Fractional / Microshard Training — Plan

## Tasks
1. [x] WSL2 GPU env that trains Granite: torch 2.5.1+cu121 + `causal-conv1d`
   1.6.2 + `mamba-ssm` 2.3.2 built from source (--no-build-isolation --no-deps;
   patched torch CUDA minor-check; made mamba-ssm's eager `Mamba3` import
   optional). Verified `loss.backward()` runs on the RTX 4060 (no segfault).
2. [x] `shard.py` block-local distillation (`granularity="block"`):
   `find_decoder_layers` (generic longest-ModuleList), `plan_shards`
   (block_size/shards → contiguous ranges), `capture_boundaries` (stream frozen
   prefix one layer at a time, LoRA disabled, to capture the block input),
   teacher/student MSE, per-shard adapter save keyed by global layer index.
   Verified: all 5 Granite shards, 96 adapters, peak VRAM ~3.1–3.4GB FLAT.
3. [x] LoRALinear robustness (trainer.py): `lora_disabled` (teacher path),
   dtype-matched lora params (bf16-safe matmul), router/gate exclusion in
   `auto_detect_targets` + `attach_lora` (fixes the fp32-router dtype clash).
4. [x] Per-module sub-layer floor (`granularity="module"`): `discover_groups`
   (layer children with ≥1 Linear), `_set_group_student` (enable one group's
   LoRA, freeze rest), `_train_block_by_module` (per-group optimizer, distill
   against the layer's frozen output, advance per-layer input frozen).
   Verified on Granite attention layer 5: trained shared_mlp + self_attn groups,
   peak VRAM **0.919 GB** (<1GB for a 6.9B model).
5. [x] CLI args: `--shard-mode/--shards/--block-size/--shard-index/
   --calib-batches/--granularity/--device/--dtype`; route to `train_sharded`.
6. [x] Rust config `[train.fractional]` (+ `granularity`) + `[hardware].device`,
   plumbed through `cmd_train_transformers`. Exported `FractionalConfig`.
7. [x] Bench `evolve.toml` + RUNBOOK updated (GPU env + fractional/granularity
   knobs + measured VRAM numbers).
8. [x] Tests + full sweep GREEN:
   - Rust: `fractional_config_round_trips_and_absent_is_none` (block + module).
   - Python `test_shard.py`: plan_shards, find_decoder_layers, router exclusion
     (auto + attach), LoRA dtype, lora_disabled teacher path, discover_groups,
     group isolation.
   - cargo test (all suites) + clippy -D warnings + fmt; Python track23 + shard.

## Status
COMPLETE + verified on real Granite, GPU. The bounded-VRAM training primitive
(block + sub-layer-module granularity) is built, config-driven, and tested.
Sign-off in this dir. Track 26 (ambient daemon + living queue) builds on it.
