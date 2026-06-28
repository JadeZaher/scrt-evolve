---
type: Track Spec
title: Fractional / Microshard Training
description: Make a large model LoRA-trainable on a small GPU by bounding peak VRAM.
tags: [track-25, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 25 — Fractional / Microshard Training — Spec

## Goal
Make a large model **LoRA-trainable on a small GPU** by bounding peak VRAM to a
single shard, with a configurable granularity floor that trades wall-clock for
memory. This is what makes the bench portable to "everyone's" hardware (and is
the substrate the ambient daemon in track 26 runs on). Built + verified on real
IBM Granite-4.0-h-tiny (hybrid Mamba-2 + MoE) on an 8GB RTX 4060.

## Background (why this track exists)
Granite-4.0-h-tiny is hybrid Mamba-2. A dense bf16 fine-tune needs ~22GB and
OOMs on 8GB. Earlier the CPU path also segfaulted in the naive Mamba backward —
RESOLVED by a CUDA torch + `mamba-ssm`/`causal-conv1d` kernels (WSL2; see
track-24 SIGN-OFF + RUNBOOK "GPU setup (WSL2)"). With backward working, the
remaining blocker was memory — which fractional training removes.

## Approach — block-local distillation, two granularities
The frozen full-precision model is the teacher; LoRA is the student. We never
hold the whole graph for a global backward; instead we distill locally so only
one unit is ever resident.

- **`granularity = "block"`** — split the decoder's layers into contiguous
  **blocks**; for block `[a,b)`, distill `block_lora(in)` → `block(in)` (MSE),
  where `in` is the activation entering layer `a` (captured by streaming the
  frozen prefix one layer at a time). Peak VRAM ≈ one block. Measured ~3.1–3.4
  GB at `block_size=8` (40 Granite layers → 5 shards), FLAT across shards.
- **`granularity = "module"`** — the SUB-LAYER floor: within each layer, train
  ONE submodule **group** (attention / MoE / shared-MLP) at a time, the rest of
  the layer frozen, distilling against that LAYER's frozen output. Peak VRAM ≈
  one layer + one group's optimizer state. Measured ~0.9 GB at `block_size=1`
  on the RTX 4060 — a 6.9B model training in **<1GB**.

Both are **generic / model-agnostic**: the decoder-layer stack is discovered as
the longest ModuleList of structured children; submodule groups are direct layer
children that contain ≥1 `nn.Linear`; LoRA targets auto-detect (routers + SSM
excluded). Shards are independent — `--shard-index N` trains one per process, so
shards can run on separate machines (decentralized).

## Scope
- `python/scrt_evolve_train/shard.py` — `find_decoder_layers`, `plan_shards`,
  `capture_boundaries` (LoRA-disabled prefix), block-level distillation,
  `discover_groups` + `_set_group_student` + `_train_block_by_module` (per-module
  floor), per-shard adapter save keyed by GLOBAL layer index (independent shards
  merge cleanly).
- `LoRALinear` additions (trainer.py): `lora_disabled` flag (teacher path),
  dtype-matched LoRA params (bf16-safe), router exclusion in `attach_lora` +
  `auto_detect_targets`.
- CLI (`__main__.py`): `--shard-mode`, `--shards`, `--block-size`,
  `--shard-index`, `--calib-batches`, `--granularity {block,module}`,
  `--device`, `--dtype`.
- Rust config: `[train.fractional]` (`enabled`/`block_size`/`shards`/
  `calib_batches`/`granularity`) + `[hardware].device`, plumbed through the
  transformers-train subprocess. Additive; absent ⇒ dense; default build ML-free.
- Bench `evolve.toml` + RUNBOOK document the GPU env + fractional knobs.

## Out of scope (→ track 26)
The always-on daemon, the living dataset queue, activity tailing, and
constitution/taste-driven generation. Track 25 is the bounded training PRIMITIVE
those build on.

## Acceptance
- Real Granite, GPU: a layer-block run trains all 5 shards (96 adapters), and a
  per-module run trains an attention layer's groups — both with backward on every
  layer incl. SSM, no segfault, peak VRAM reported (~3.3GB block / ~0.9GB module).
- Config-driven: setting `[train.fractional]` switches the pipeline with no code
  change; absent leaves dense training intact.
- Tests: Rust `fractional_config_round_trips...` (block + module granularity);
  Python `test_shard.py` (plan, generic layer discovery, router exclusion, LoRA
  dtype, teacher path, group discovery, group isolation). Full sweep green
  (Rust all suites + clippy -D warnings + fmt; Python track23 + shard).
