---
type: Track Spec
title: Quantized-Base Training (GGUF dequant + QAT)
description: Train a model that lives on disk only as a quantized GGUF (dequant + QAT).
tags: [track-23, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 23 — Quantized-Base Training (GGUF→HF dequant + QAT) — Spec

## Goal
Let scrt-evolve **train a model that exists on disk only as a quantized GGUF**
(e.g. IBM `granite-4.0-h-tiny-Q4_K_M.gguf`, 4.2 GB). The transformers LoRA
trainer (track 19) needs HuggingFace safetensors + a tokenizer; a GGUF has
neither in trainable form. This track bridges that gap and adds
**quantization-aware training (QAT)** so the LoRA adapter is learned against the
quantization the model will actually be deployed under — minimizing the
quality drop when the trained model is re-exported to GGUF.

This unblocks the benchmark (track 24): the user's Granite GGUF becomes a
trainable student.

## Scope

### 1. `python/scrt_evolve_dequant/` — GGUF → HF safetensors converter
- Read a GGUF with the vendored `gguf` reader (llama.cpp's `gguf-py`, already on
  PYTHONPATH via the export path).
- **Dequantize** each quantized tensor (Q4_K_M etc.) back to f16/f32 using the
  `gguf` dequantize routines. (Lossy by nature — the original f16 precision is
  already gone; we recover the *quantized* weights upcast. Documented honestly.)
- **Map GGUF tensor names → HF names** for the model architecture (granite/llama
  family: `blk.N.attn_q.weight → model.layers.N.self_attn.q_proj.weight`, etc.),
  driven by a name-map table keyed on the GGUF `general.architecture` metadata.
- Reconstruct an HF `config.json` from GGUF metadata (hidden size, n_layers,
  n_heads, n_kv_heads, vocab, rope/norm eps, etc.).
- Extract the **tokenizer** from GGUF metadata (vocab + merges/scores) into a
  `tokenizer.json`/`tokenizer.model` HF can load. (Fallback: accept a
  `--tokenizer <hf-id-or-dir>` when GGUF tokenizer extraction is incomplete.)
- Write an HF model dir (`model.safetensors` + `config.json` + tokenizer) that
  `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` loads.
- CLI: `python -m scrt_evolve_dequant --gguf <path> --out <dir> [--dtype f16]
  [--tokenizer <fallback>]`. Final stdout line = JSON {out, arch, n_tensors,
  dtype}.

### 2. Rust CLI: `evolve train dequant`
- `evolve train dequant --gguf <path> --out <dir>` shells out to the Python
  converter (track-19 subprocess pattern). Auto-detects the llama.cpp `gguf-py`
  on PYTHONPATH like `export-gguf` does.
- A convenience: if `[evolve].model_path` points at a `.gguf`, the train/eval
  drivers can auto-dequant to `work_dir/base-hf/` once and reuse it (cached;
  re-dequant only if the source GGUF is newer). Behind an explicit
  `--auto-dequant` flag to keep it opt-in.

### 3. QAT — quantization-aware LoRA training (`python/scrt_evolve_train`)
- **Fake-quant (straight-through estimator)**: during the LoRA forward, the
  effective weight `W + (alpha/r)·BA` is passed through a `quant→dequant`
  simulation of the target GGUF quant (group-wise affine for Q4_K-style), with a
  straight-through gradient (identity on the backward) so the adapter learns to
  compensate for quant error. Toggled by `--qat <quant>` (e.g. `--qat Q4_K_M`);
  absent ⇒ today's plain LoRA.
- **Calibration**: an optional pass over a sample of the dataset to pick
  per-tensor (or per-group) quant scales/zero-points the fake-quant uses, instead
  of static min/max. `--qat-calibrate <n_batches>`; absent ⇒ static per-group
  absmax scales.
- QAT is CPU-safe (float32 simulation), bounded (calibration batch count is
  explicit), and deterministic given a seed (styleguide §2.2).

### 4. `[train].qat` config
- Additive `Option<QatConfig>` on `TrainConfig`: `enabled`/`quant`
  (target quant type), `calibrate_batches`, `group_size`. Absent ⇒ no QAT
  (non-breaking, styleguide §1). The Rust `train --backend transformers` passes
  these through to the Python trainer.

## Constraints
- **Default build stays ML-free + Python-free.** All heavy work is Python
  subprocess; the Rust side only adds config + a thin `dequant` shim + flag
  pass-through. No candle, no pyo3 added.
- **Honest about lossiness.** Dequant from a Q4 GGUF cannot recover the original
  f16 weights; it recovers the quantized weights upcast. The converter logs this
  and stamps the output config so downstream knows the base is dequantized.
- **Bounded + deterministic.** Calibration batch count is explicit; QAT sim is
  seeded; no unbounded loops (styleguide §2.5).
- **Reuses:** track 19 trainer (`LoRALinear`, dataset contract), the vendored
  `gguf-py` (already used by `export.py`), the export GGUF path (round-trip:
  GGUF → HF → LoRA+QAT → merge → GGUF).

## Acceptance
- `evolve train dequant --gguf <granite Q4_K_M> --out base-hf/` produces an HF dir
  that `transformers` loads and `evolve model infer` runs (smoke).
- A LoRA train run on the dequantized base produces an `adapter.safetensors`
  loadable by infer (the existing track-19 path, base swapped).
- `--qat Q4_K_M` training runs end-to-end and the fake-quant is exercised
  (a unit test on the quant→dequant STE op: forward quantizes, backward is
  identity; a tiny tensor round-trips within quant tolerance).
- `--qat-calibrate N` runs N bounded calibration batches and the chosen scales
  are used (asserted on a fixture).
- `[train].qat` round-trips in `evolve.toml`; absent ⇒ plain LoRA (non-breaking).
- Default `cargo test` / `cargo clippy` / `cargo fmt` green (Rust side is config
  + shim only); Python converter + QAT have their own pytest/smoke.

## Future / out of scope (recorded 2026-06-20)
- **Streaming dequant** (tensor-by-tensor, bounded peak RAM) IS in scope and the
  default — folds in the user's "dequantize in parts" idea.
- **Decentralized / boundary-aware sharded training** is NOT this track. It
  belongs to the existing **track 07 (train-shard)** plus a future
  "boundary-aware sharding" track that would use **gradient/activation
  attribution** (track 13's grad path) to find which layers a goal touches.
- **LARQL is explicitly NOT used.** The repo research note
  (`.omc/research/larql-regen-swap-2026-06-17.md`) dropped LARQL
  reverse-inference / WALK as a speed/location engine (~46× slower than decode,
  undemonstrated). Any future boundary detection uses plain
  gradient/activation attribution, not LARQL.

## Dependencies
Track 19 (Python trainer/infer + dataset contract), the export GGUF path
(`scrt_evolve_gguf`, vendored `gguf-py`). Consumed by track 24 (the benchmark,
which trains Granite). Independent of the eval lane (10/15) — orthogonal capability.

## Honest risks
- **GGUF tokenizer extraction is fiddly.** Some GGUFs don't carry a clean
  HF-loadable tokenizer; the `--tokenizer` fallback (point at the HF tokenizer
  for the same base) is the escape hatch and the documented default for Granite
  if extraction is partial.
- **Name-map coverage.** The GGUF→HF tensor name map is architecture-specific;
  granite/llama is covered, exotic arches error clearly listing unmapped tensors
  rather than producing a silently-broken model.
- **QAT on CPU is slow.** The fake-quant adds per-step overhead; the bench
  budgets steps accordingly. QAT quality is an experiment (out of scope); the
  *machinery* (STE op correct, calibration bounded, config wired) is what this
  track proves.
