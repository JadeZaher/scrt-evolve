---
type: spec
track: 33
title: Concurrent inference during shard training (serve-while-you-train)
status: planned
created: 2026-06-30
depends_on: [26, 25, 19, 27]
---

# Track 33 — Concurrent inference during shard training — Spec

## Goal
Let a user **run inference on their evolving model while the ambient daemon is
still training it**, interrupting the served model only at the moment a freshly
trained shard is committed (a hot adapter-swap), never during the rest of the
loop. The served model always answers from a **complete, self-consistent set of
full weights** — the last committed adapter merged over the frozen base — so the
user never sees a half-trained Frankenstein.

The pitch: "your model is always live. Training happens in the background on one
block at a time; the served copy blips only when a new block lands, then keeps
serving the improved weights."

## Why this is feasible (the load-bearing fact)
The fractional trainer (track 25, `python/scrt_evolve_train/shard.py`) already has
exactly the residency profile this needs:

- The **full-precision base model lives on CPU/disk**; only **ONE block is ever
  resident on the GPU** at a time (`shard.py` header: "only one block is ever
  resident on the accelerator, and the rest of the model stays on CPU/disk").
- Measured **peak training VRAM ≈ 3.1–3.4 GB** for `block_size=8` on the RTX 4060
  (8 GB) — leaving **~4.5 GB free during a training step**.
- Inference for the deployment path runs on the **Q4_K_M GGUF via llama.cpp**
  (`run-model`, `[runtime]`), which is far more VRAM-frugal than the bf16
  transformers forward and handles the hybrid Mamba SSM state properly.

So "serve while training" is a **VRAM-arbitration + adapter-swap** problem, not a
fundamental conflict. Two models are never both doing heavy GPU work at the same
instant if we sequence them; and the served GGUF's footprint + one training block
can plausibly co-reside in 8 GB (to be measured — see Risks).

## The three interruption models (pick one; recommend B)

**A. Never co-resident (safest, simplest).** Training and serving strictly
alternate on the GPU. `pause_on_gpu_process` already yields the GPU when another
process computes; extend it so the *served* inference process is a first-class
"foreground" that training yields to. Inference is instant when idle-training;
during an active block step, inference queues (or falls to a CPU llama.cpp path)
until the ~seconds-long block step + cooldown passes. Zero VRAM risk. Latency
cost: a request that lands mid-block waits up to one block-step.

**B. Co-resident with a VRAM budget (recommended).** Serve the Q4_K_M GGUF
pinned in VRAM (`n_gpu_layers` tuned so it fits in, say, 3.5–4 GB) AND train one
block (~3.3 GB) at the same time, arbitrated by a shared VRAM ceiling. The daemon
already self-throttles on `max_vram_gb`; add a "reserved-for-serving" carve-out so
the trainer only starts a block when `free − serve_reservation ≥ block_need`. The
served model is only *swapped* (adapter reload) at commit — inference itself never
stops. This is the "interrupt only when the shard is trained" behavior the user
asked for.

**C. Two-tier (CPU-serve + GPU-train).** Always serve from a CPU llama.cpp
instance (slow but never contends), and only hot-swap its weights at commit. GPU
is 100% training's. Simplest VRAM story, worst inference latency.

Recommendation: **B**, with **A** as the fallback the code degrades to when the
measured co-resident footprint doesn't fit (doctor-gated).

## The hot adapter-swap (the "interrupt only at shard-done" mechanism)
The commit boundary already exists: each daemon step ends with the track-15
transaction keeping or rolling back the adapter. On **keep**, the merged flat
adapter (`adapter.safetensors`, produced by the track-33-prereq merge step now in
`shard.py`) is the new truth. The served model must reload THAT.

For the GGUF serve path, a full re-merge+re-quant per step is too slow. Options,
in order of preference:
1. **llama.cpp LoRA hot-apply** — serve `base.gguf` + apply the LoRA adapter as a
   GGUF LoRA (`--lora`) that can be swapped without re-quantizing the base. Needs
   a `safetensors LoRA → GGUF LoRA` converter step at commit (seconds, small).
2. **Debounced re-export** — only re-merge→re-quant the served GGUF every N
   commits or every M minutes, not every step. The served model lags the trained
   one by a bounded amount; acceptable for "run inference while it evolves."
3. **Serve the transformers path with `apply_adapter`** (already exists) for the
   non-GGUF case — cheapest to swap (just reload the flat adapter), most VRAM.

## Deliverables
1. A `serve --live` (or `daemon serve` / `run-model --follow`) mode that:
   - starts a long-lived inference server on the current committed adapter,
   - subscribes to the evolution log / a commit signal,
   - hot-swaps the adapter at each **keep** commit, logging `served vN → vN+1`.
2. VRAM arbitration between the trainer and the live server (model B): a
   `serve_reservation_gb` carve-out honored by the daemon's `max_vram_gb`
   self-throttle, and a doctor check that measures the co-resident footprint and
   auto-selects model B or degrades to A.
3. The `safetensors LoRA → GGUF LoRA` converter (if pursuing llama.cpp hot-apply).
4. Status surface: `watch status`/`serve status` shows the **served version** vs
   the **latest committed version** (lag), plus whether the server is GPU- or
   CPU-resident right now.

## Non-goals
- Multi-user / networked serving (single local user, one server).
- Serving *mid-block* partial weights (we serve only committed, complete adapters).
- Changing the training algorithm — this is purely serving + arbitration around
  the existing fractional loop.

## Risks / open questions (resolve in plan phase)
- **Co-resident 8 GB fit (model B).** Must MEASURE: Q4_K_M GGUF at reduced
  `n_gpu_layers` + one 3.3 GB training block + CUDA context overhead. If it
  doesn't fit, model B degrades to A. This is the single biggest unknown.
- **llama.cpp GGUF-LoRA hot-apply** — verify the installed llama.cpp build
  supports `--lora` hot-apply for the granitemoehybrid arch, and that a
  `safetensors→GGUF-LoRA` path exists (may need a converter). If not → option 2/3.
- **Swap atomicity** — a swap must be all-or-nothing so a request never sees a
  torn adapter. Serve vN until vN+1 is fully loaded, then flip a pointer.
- **cpu_fallback interaction** — the Granite config sets `cpu_fallback=false`
  (Mamba CPU backward segfaults). Serving on CPU is fine (forward-only), but the
  arbitration logic must not confuse "serve on CPU" with "train on CPU".

## Acceptance
- User runs `scrt-evolve serve --live --config bench/ambient-granite.toml` and
  gets answers throughout a daemon run; the served version increments at each keep
  commit; inference is never interrupted except a bounded blip at swap.
- On this 8 GB box, either co-resident (B) works within budget, or the tool
  cleanly reports it degraded to alternating (A) with the reason.
