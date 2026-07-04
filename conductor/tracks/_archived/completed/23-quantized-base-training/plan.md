---
type: Implementation Plan
title: Quantized-Base Training (GGUF dequant + QAT)
description: Implementation plan for the Quantized-Base Training (GGUF dequant + QAT) track.
tags: [track-23, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 23 — Quantized-Base Training (GGUF→HF dequant + QAT) — Plan

## Tasks

1. [ ] `python/scrt_evolve_dequant/` converter: read GGUF (vendored `gguf-py`),
   dequantize tensors → f16/f32, map GGUF→HF tensor names (granite/llama),
   reconstruct `config.json`, write `model.safetensors` + tokenizer (or
   `--tokenizer` fallback). CLI emits JSON summary. -- evidence: dequant a real
   GGUF → HF dir loads in transformers (smoke).
2. [ ] `scrt-evolve dequant` Rust shim (subprocess, gguf-py on PYTHONPATH) +
   optional `--auto-dequant` so a `.gguf` model_path is dequantized to
   `work_dir/base-hf/` once and cached. -- evidence: CLI runs; cache reused.
3. [ ] QAT fake-quant STE op in `scrt_evolve_train`: quant→dequant simulation of
   the target quant on the effective LoRA weight, identity backward. `--qat <quant>`
   toggles. -- evidence: STE unit test (forward quantizes within tol, backward
   gradient is identity).
4. [ ] QAT calibration: bounded pass over N dataset batches → per-group scales
   used by the fake-quant. `--qat-calibrate N`. -- evidence: calibration picks +
   uses scales (fixture); bounded by N.
5. [ ] `[train].qat` config (`Option<QatConfig>`: enabled/quant/calibrate_batches/
   group_size), passed through by `train --backend transformers`. -- evidence:
   config round-trip; absent ⇒ plain LoRA.
6. [ ] Round-trip smoke: GGUF → dequant HF → LoRA(+QAT) → merge → GGUF (reuse
   `scrt_evolve_gguf`). -- evidence: documented runbook + a tiny-model smoke if
   feasible in CI budget.
7. [ ] Final sweep: `cargo test`/`clippy`/`fmt` (Rust = config + shim); Python
   dequant + QAT-STE tests. -- evidence: green.

## Build order note
Tasks 1–2 (converter + shim) unblock training Granite at all. Tasks 3–5 (QAT)
improve the exported-GGUF quality. Task 6 ties to the export path. Independent of
the eval lane.

## Status (2026-06-20)
1. [x] `scrt_evolve_dequant/` converter — **generic, registry-driven** (`archspec.py`
   `ArchSpec` registry keyed on GGUF arch; rule-based name/config maps; llama/
   mistral/qwen2 registered). `dequant.py` STREAMS tensors one at a time (bounded
   memory). Unknown arch → clear error listing supported + "register an ArchSpec".
   Honest lossiness stamp `_dequantized_from_gguf`.
2. [x] `scrt-evolve dequant` Rust shim (subprocess, auto-detects vendored gguf-py
   on PYTHONPATH). (`--auto-dequant` cache convenience: deferred — not needed
   since the bench uses cached f16 HF Granite.)
3. [x] QAT fake-quant STE (`qat.py` `fake_quantize` + `_FakeQuantSTE`): forward
   group-wise affine quant→dequant, backward identity. `--qat <quant>` toggles;
   wired into `LoRALinear.forward`. Python test: forward quantizes, backward is ones.
4. [x] QAT calibration (`Calibrator`): bounded per-group absmax over N batches,
   then frozen. `--qat-calibrate N`. Python test: bounded + scale frozen.
5. [x] `[train.qat]` config (`QatConfig`: enabled/quant/group_size/calibrate_batches)
   passed through by `train --backend transformers`. Rust test:
   `qat_config_round_trips_and_absent_is_none`.
   PLUS (generic-arch mandate): `--target-modules auto` + `auto_detect_targets`
   enumerate nn.Linear leaves (no hardcoded q_proj/v_proj) so hybrid/MoE arches
   (granitemoehybrid) train. Python test: `auto_detect_targets`.
6. [~] Round-trip smoke (GGUF→HF→LoRA+QAT→merge→GGUF): the pieces exist + unit-
   tested; a full real-model round-trip runs in track 24 (the bench), not CI (no
   small vanilla LLM GGUF on disk; mmproj/granite are unsuitable for a quick smoke).
7. [x] Final sweep: `cargo test` (19 suites), `clippy -D warnings`, `fmt --check`,
   `--features train`, `--features pyo3` — GREEN. Python: 5/5 track-23 unit tests.

## Major scope correction (2026-06-20)
The bench target **Granite-4.0-h-tiny is `granitemoehybrid`** (Mamba-2 SSM +
64-expert MoE) and its **full-precision HF safetensors are ALREADY cached**
(`~/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-tiny/`). So the
BENCH trains the cached f16 HF directly (no lossy dequant) — track 24 points
model_path there. The dequant converter is kept as a GENERAL capability for
HF-less models; the hybrid-MoE/SSM GGUF→HF name-map is a documented seam (only
llama/mistral/qwen2 specs shipped; granitehybrid spec not needed for the bench).

## Sign-off
Signed off 2026-06-20 — see SIGN-OFF.md. Generic registry-driven dequant +
QAT (STE + calibration) + auto-detect LoRA targets, all generic at the
architecture level (no model/brand-specific logic), per the user mandate.
