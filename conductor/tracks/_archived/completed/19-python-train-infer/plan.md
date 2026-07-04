---
type: Implementation Plan
title: Python train/infer (transformers LoRA + inference)
description: Implementation plan for the Python train/infer (transformers LoRA + inference) track.
tags: [track-19, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 19 — Python train/infer — Plan

## Overview

Standalone Python trainer (transformers + hand-rolled LoRA) + inference module, subprocess-driven from Rust CLI. The PRIMARY real-model training/inference path (candle = fixture-only). This plan captures the **end-to-end validation completed 2026-06-20**, pending the listed future work.

## Tasks

### Validation & Evidence (completed 2026-06-20)

- [x] **End-to-end trainer run on real TinyLlama-1.1B**
  - Evidence: TinyLlama-1.1B loaded successfully (24 layers, handles RoPE/GQA/BF16 weights)
  - 44 LoRA adapters created (q_proj + v_proj per layer)
  - Adapter tensor shapes correct: A=[rank, in], B=[out, rank]; v_proj.lora_B=[256, 16] (correct GQA)
  - Training ran 40–100 steps, loss decreased (3.55 → 2.84), real gradient descent via AdamW
  - adapter.safetensors saved and reloaded successfully

- [x] **Inference module loads + runs base + adapter, A/B compare**
  - Evidence: base model inference works; adapter LoRA injection into forward pass works
  - A/B mode outputs differ as expected (base vs base+adapter)
  - Adapter config reloaded from adapter_config.json correctly

- [x] **Rust CLI arms wired and build green**
  - `evolve train fit --backend transformers` dispatches to Python trainer via subprocess
  - `evolve model infer` dispatches to Python inference module
  - Rust build compiles; no new dependencies in default build

- [x] **Larger dataset generated** (scrt-evolve corpus discovery + generation stages)
  - dataset.jsonl generated via tracks 01/02/03 (discover → API generate)
  - Trainer successfully reads mixed qa/instruction/completion rows
  - Prompt-masked cross-entropy computed correctly on completion tokens

### Testing & quality (pending; deferred to test harness or later pass)

- [ ] **Automated trainer unit tests**
  - Mock dataset (small QA pairs, known loss trajectory)
  - Verify loss decreases over a fixed seed + 10–20 steps
  - Verify adapter format round-trips (save → load → tensor shape match)
  - Verify prompt masking (loss only on completion positions)

- [ ] **Automated inference unit tests**
  - Base model inference on a fixed prompt, fixed seed → deterministic output
  - Adapter injection consistency (same input + adapter → same output across runs)
  - A/B format validation (two output blocks, clearly labeled)

- [ ] **Python environment + requirements.txt**
  - Pinned torch >= 2.0, transformers >= 4.35, safetensors >= 0.4
  - Document recommended setup: `python -m venv .venv && .venv/bin/pip install -r python/requirements.txt`
  - Test that `pip install -r python/requirements.txt` + `python -m scrt_evolve_train --help` works in a clean venv

- [ ] **Held-out evaluation harness**
  - A committed scrt-cli probe/eval artifact (held-out test set, ~10% corpus)
  - Measure base vs base+adapter on held-out set (e.g., perplexity, token-accuracy on known-good QA pairs)
  - Report in `evaluation-log.jsonl` or similar

### Integration & CI (pending)

- [ ] **CI coverage for Python path**
  - GitHub Actions workflow: `python -m pytest python/scrt_evolve_train/ python/scrt_evolve_infer/` (or unittest equivalent)
  - Run trainer on a tiny fixture dataset (~5 examples), verify loss decreases
  - Run inference on a fixture adapter, verify output format

- [ ] **Documentation updates**
  - Add to `README.md` (top-level): "Real-model training via track 19 (Python backend). See `python/scrt_evolve_train/README.md` and `python/scrt_evolve_infer/README.md` for details."
  - Clarify that default `cargo build` is ML-free; Python path is invoked as subprocess
  - Link to DESIGN.md Amendment 2026-06-20 for architecture rationale

### Optional / future (out of scope v1)

- [ ] **GGUF export & LM Studio integration** — save trained adapter as GGUF for local LM Studio use
  - Requires GGML/llama.cpp bindings or external tool integration
  - Deferred; not blocking core acceptance

- [ ] **Multi-adapter composition** — load + merge multiple adapters in inference
  - Deferred; single adapter MVP sufficient

- [ ] **Checkpointing & resume** — save optimizer state, resume from checkpoint
  - Deferred; single-run MVP sufficient

- [ ] **Distributed training** — data-parallel or model-parallel across nodes
  - Deferred; single-node MVP sufficient; shard training is track 07

## Sign-off

**Status:** Partially validated ✅ / Pending formal test harness + CI.

**Validation evidence:** 2026-06-20 end-to-end run on TinyLlama-1.1B. Real LoRA training (loss decreased), adapter saved (correct shapes), inference works (A/B). Rust CLI dispatch functional. dataset.jsonl contract validated.

**Acceptance bar:** All end-to-end criterion items in spec.md marked ✅. Outstanding work (tests, env pinning, CI, eval harness) is documented and deferred; those are quality/rigor improvements, not core functionality blockers.

**Sign-off readiness:** Ready for integration into core workflows (self-evolve lane 10–15 and any immediate use). Test harness + CI can proceed in parallel without blocking the training path adoption.

### Next steps

1. Copy/reference this track into the main track dependency graph (already in tracks.md table).
2. Begin test harness implementation (trainer tests + inference tests + CI).
3. Document Python environment setup in top-level README.
4. Integrate with self-evolve lane (track 11+ can now target real-model training via track 19).
