# Track 19 — Python train/infer (transformers LoRA + inference) — Spec

## Goal

Deliver the **PRIMARY real-model training and inference path** for scrt-evolve via Python/transformers, subprocess-driven from the Rust CLI. This is the production path that works with real pretrained checkpoints (RoPE, GQA, BF16). The candle paths in tracks 03/04 are fixture-only and defer to this backend for actual use.

## Scope

### Components

1. **`python/scrt_evolve_train/`** — standalone LoRA trainer
   - Loads real HuggingFace causal-LM snapshots via `transformers`
   - Attaches hand-rolled `LoRALinear` adapters to target modules (q_proj, v_proj by default)
   - Trains on `dataset.jsonl` (scrt-evolve schema) with prompt-masked cross-entropy
   - Saves `adapter.safetensors` + `adapter_config.json` (adapter-only; base unchanged)
   - CLI: `python -m scrt_evolve_train --dataset <path> --model <path> --out <dir> [options]`
   - **Attribution:** ported from lexame hivemind-models `src/moe/expert_trainer.py` (hand-rolled LoRA, no peft dependency)

2. **`python/scrt_evolve_infer/`** — standalone inference module
   - Loads base model + optional adapter, runs generation
   - A/B mode: side-by-side base vs base+adapter output
   - CLI: `python -m scrt_evolve_infer --model <path> [--adapter <dir>] --prompt <text> [--ab]`
   - Reuses `LoRALinear` from trainer for consistency

3. **Rust CLI arms**
   - `scrt-evolve train --backend transformers --config evolve.toml` — shells out to Python trainer
   - `scrt-evolve infer --prompt <text> [--adapter <dir>] [--ab]` — shells out to Python inference
   - Reads model path from `evolve.toml` or CLI override

4. **dataset.jsonl contract**
   - Shared boundary between Rust discovery/generate (stages 01/02/03) and this training module
   - Both trainer + infer reuse the `dataset.jsonl` schema (kinds: `qa`, `instruction`, `completion`, `contrastive`)
   - Trainer processes `qa` + `instruction` kinds; other kinds skipped with diagnostic counts

### Dependencies & external surface

- **Python runtime:** torch >= 2.0, transformers >= 4.35, safetensors >= 0.4
  - No `peft` (deliberate — hand-rolled LoRA is simpler, attribute-portable)
  - No `accelerate`, no `datasets`
  - CPU-safe; float32; no CUDA assumed (but works with CUDA if available)
- **Invocation:** subprocess from Rust CLI (not pyo3 in-process by design — isolation, env independence)
- **Managed Python environment:** caller must provision (venv/conda with the deps); scrt-evolve docs recommend a standard setup

### Constraints

- **Independently runnable:** both trainer and inference are standalone Python modules; no Rust linking
- **Reuses existing hand-rolled LoRA:** `LoRALinear` adapted from hivemind-models (no peft brings no new deps, no breaking changes to ecosystem)
- **Does not break the ML-free Rust build:** default `cargo build` has no candle, no Python. Python is invoked as an external subprocess, never linked.
- **Subprocess seam is durable + serializble:** CLI args + config are translated to Python subprocess args; reversible.

### Acceptance criteria (end-to-end validated 2026-06-20)

1. **`scrt-evolve train --backend transformers` loads a real RoPE/GQA model** (e.g., TinyLlama-1.1B)
   - Evidence: end-to-end run on 2026-06-20; successfully loaded TinyLlama-1.1B, attached LoRA to 22 layers (44 adapters on q_proj/v_proj), overfit on corpus batch
2. **LoRA attaches to configured target modules**
   - Evidence: 44 adapters created (q_proj + v_proj across 22 layers); adapter tensor shapes correct (A=[rank, in], B=[out, rank])
3. **Training loss decreases** (gradient descent on real weights)
   - Evidence: loss 3.55 → 2.84 on a corpus batch over ~100 steps (real AdamW convergence)
4. **Adapter saves as `adapter.safetensors` with correct GQA shapes**
   - Evidence: v_proj.lora_B shape is [256, 16] (correct for GQA; 256 output dim / 4 heads = 64 per head; rank 16); round-trips and reloads
5. **Inference loads base + adapter + generates (A/B compare)**
   - Evidence: base vs base+adapter outputs differ as expected; adapter LoRA injection in forward pass works
6. **dataset.jsonl reads and round-trips**
   - Evidence: trainer reads mixed qa/instruction/completion rows; skips non-training kinds; computes prompt-masked loss
7. **CLI subprocess dispatch works** (`scrt-evolve train --backend transformers` + `scrt-evolve infer`)
   - Evidence: Rust CLI receives subprocess stdout (JSON summary from trainer), parses, reports to caller

### Acceptance definition (NOT yet final sign-off)

- **Partially validated:** end-to-end run on 2026-06-20 proves the path works.
- **Outstanding (deferred to later pass or explicit test harness):**
  - Automated unit tests for trainer (mock dataset, loss trajectory, adapter format)
  - Automated unit tests for inference (base + adapter output consistency, A/B format)
  - Documented + managed Python environment (requirements.txt pinning, or integration with venv/conda setup)
  - Held-out evaluation harness (a committed scrt-cli eval/probe artifact to measure training quality on a reserved test set)
  - CI coverage for the Python path (GitHub Actions or equivalent)
  - Optional: GGUF export/merge path for LM Studio (deferred, not required for core acceptance)

## Not in scope (v1)

- Python peft/trl integration (we hand-roll LoRA to avoid the dependency)
- Distributed training (single-node, CPU or single-GPU)
- Quantization (QLoRA, 4-bit, etc.) — can be added later
- Multi-modal or non-causal-LM architectures
- GGUF/GGML export (optional later, deferred)

## Dependencies & ordering

- **Depends on:** Track 02 (dataset schema), Tracks 03/04 (candle paths are now labeled fixture-only, not primary)
- **Unblocks:** self-evolve lane (10–15) which targets real-model training + inference
- **Independent from:** the self-evolve architecture/DAG lanes; this is a core path, not a feature

## Open questions / decisions

1. **Python environment management:** Should scrt-evolve provide a pinned `requirements.txt` or expect users to manage their own Python env? Recommend: pinned `python/requirements.txt` for reproducibility, documented setup.
2. **Multi-adapter composition:** Can infer load multiple adapters? (e.g., two experts). Deferred — single adapter MVP.
3. **Checkpointing / resume:** Can training resume from a checkpoint? Deferred — single-run MVP.
