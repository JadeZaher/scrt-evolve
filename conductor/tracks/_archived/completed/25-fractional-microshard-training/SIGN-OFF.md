# Track 25 — Fractional / Microshard Training — SIGN-OFF

Date: 2026-06-21

Bounded-VRAM training is built and **verified on real IBM Granite-4.0-h-tiny
(6.9B, hybrid Mamba-2 + MoE) on an 8GB RTX 4060** (WSL2). A large model now
LoRA-trains on a small GPU, config-driven, at two granularities.

## What this proves
- **Granite trains on GPU — no segfault.** The CUDA torch + mamba kernels (WSL2)
  resolved the Mamba-2 backward crash. Forward + backward run on every layer.
- **Layer-block granularity:** all 5 shards of the 40-layer model trained, **96
  LoRA adapters**, **peak VRAM 3.1–3.4 GB — FLAT across shards** (the evict/
  reload bound holds regardless of which block).
- **Per-module sub-layer floor:** within a layer, one submodule group trained at
  a time against the layer's frozen-output teacher — **peak VRAM 0.919 GB** for a
  6.9B model. The "more time, less memory" knob taken to <1GB.
- **Generic + decentralized:** layer stack + groups + LoRA targets all
  discovered (no model-specific names; routers + SSM excluded). Shards are
  independent (`--shard-index N` per machine); adapters saved by global layer
  index so they merge.
- **Config-driven (user mandate):** `[train.fractional]` (enabled/block_size/
  shards/calib_batches/granularity) + `[hardware].device` switch the mode with no
  code change. Additive — absent ⇒ dense; default build stays ML-free.

## Honest limits
- The smoke datasets are tiny (6 hand QA pairs, few steps), so losses sit near
  the distillation-init floor (LoRA B=0 ⇒ student≈teacher). This validates the
  MECHANISM (group isolation, layer-boundary teacher, per-group optimizer,
  bounded VRAM, backward), NOT a measurably-improved model — that's a real
  curriculum run (the bench schedule / track 26 daemon).
- The model is still loaded whole into CPU RAM first (WSL cap raised to 26GB).
  Per-shard DISK streaming (never holding the full model in RAM) is a further
  optimization, not yet done — VRAM is bounded; host RAM is not yet.

## Verification
- Rust: all suites green; `fractional_config_round_trips_and_absent_is_none`
  covers block + module granularity. clippy -D warnings clean; fmt clean.
- Python: `test_shard.py` 8/8 (plan_shards, find_decoder_layers, router
  exclusion ×2, LoRA dtype, lora_disabled, discover_groups, group isolation);
  `test_track23.py` 6/6.
- Real GPU runs (WSL2): block all-5-shards + per-module layer-5 — outputs in the
  session transcript (peak VRAM 3.3GB / 0.919GB respectively).
