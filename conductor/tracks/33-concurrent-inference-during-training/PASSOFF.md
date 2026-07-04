---
type: note
track: 33
title: Inline passoff — start here next session
status: planned
created: 2026-06-30
---

# Track 33 — Inline passoff (start next session here)

## One-line
Build "serve inference on the model **while** the ambient daemon trains it,"
hot-swapping the served adapter only at each **keep-commit** (never mid-block).

## Why it was deferred (not implemented this session)
It's a real architectural feature (a live inference server + VRAM arbitration +
adapter hot-swap around the existing fractional loop), not a config tweak. Per the
user's instruction: complex → make a track, hold implementation, pass off. Spec is
in `spec.md`; this file is the "cold-start in 5 minutes" for the next session.

## The single fact that makes it feasible (don't re-derive it)
`python/scrt_evolve_train/shard.py` keeps the **full model on CPU** and moves
**ONE block to the GPU at a time** (peak ~3.1–3.4 GB for `block_size=8` on the
RTX 4060, leaving ~4.5 GB free during a step). So "serve while train" is a VRAM
arbitration + adapter-swap problem, NOT a fundamental conflict. This is the whole
premise — verify it still holds (grep the shard.py header) before designing.

## Recommended approach (from spec — model B)
Serve the Q4_K_M GGUF pinned in a VRAM carve-out AND train one block co-resident,
gated by a shared VRAM ceiling; hot-swap the served adapter at each keep-commit so
inference itself never stops. Degrade to model A (strict alternate on the GPU)
when the measured co-resident footprint doesn't fit 8 GB. See spec §"three
interruption models".

## First concrete steps (in order)
1. **MEASURE the co-resident footprint** (the biggest unknown). Load the Q4_K_M
   GGUF via `run-model`/llama.cpp at a reduced `n_gpu_layers`, note VRAM; separately
   run one daemon block step, note its ~3.3 GB peak; check whether
   `serve_footprint + block_peak + CUDA_ctx ≤ 8 GB`. If yes → model B is viable; if
   no → build model A. This measurement decides the whole design — do it FIRST.
2. **Prereq already landed:** the shards→flat-adapter merge is DONE (this session,
   `shard.py` `train_sharded` + `train_distill` now call `merge_shard_adapters`),
   so `adapter.safetensors` always exists at commit for the server to load. Don't
   rebuild that.
3. **Adapter-swap mechanism:** decide GGUF-LoRA hot-apply (`llama.cpp --lora`, needs
   a `safetensors→GGUF-LoRA` converter — check the installed llama.cpp supports it
   for granitemoehybrid) vs. debounced re-export (re-quant every N commits) vs. the
   transformers `apply_adapter` path (cheapest swap, most VRAM). Spec §"hot
   adapter-swap" ranks them.
4. **`serve --live` mode:** long-lived server that subscribes to the evolution log
   (or a commit signal file) and hot-swaps at each `action:"keep"` row. Reuse the
   existing `RunModel`/`branch serve` infra (crates/scrt-evolve-cli/src/main.rs
   ~L248 `RunModel`, ~L622 `Serve`).
5. **VRAM arbitration:** add `serve_reservation_gb` to `[daemon]`; the trainer's
   `max_vram_gb` self-throttle must subtract the reservation before starting a
   block. `doctor` measures + picks B or degrades to A.
6. **Status surface:** `serve status` / extend `watch status` to show served
   version vs latest committed version (the lag) + GPU-vs-CPU residency now.

## Landmarks (files you'll touch)
- `python/scrt_evolve_train/shard.py` — residency model (read-only for this track).
- `python/scrt_evolve_infer/infer.py` — `apply_adapter` (transformers swap path).
- `crates/scrt-evolve-cli/src/main.rs` — `RunModel` (~L248), `Serve` (~L622),
  daemon subcommands (~L644+); add `serve --live` here.
- `crates/scrt-evolve/src/daemon.rs` — `max_vram_gb` throttle + the keep/rollback
  commit boundary (the swap trigger); `daemon.run` stop-file pattern to mirror.
- `bench/ambient-granite.toml` — the live config; `[runtime]`/`[export]` in
  `bench/evolve.toml` show the GGUF serve + merge-shards wiring to build on.

## Watch-outs (from spec Risks — don't relearn the hard way)
- Granite config sets `cpu_fallback=false` (Mamba CPU **backward** segfaults).
  Serving on CPU is forward-only and FINE — don't let arbitration conflate "serve
  on CPU" with "train on CPU".
- Swap must be atomic (serve vN until vN+1 fully loaded, then flip a pointer) — a
  request must never see a torn adapter.
- The 13 GB f16 weights are Windows-side only (`/mnt/c` cache); the WSL `~/.cache`
  is weight-stripped. GGUF serving uses the exported GGUF, not the f16 — but if any
  step needs the f16, point at `/mnt/c` (see memory `granite-ambient-daemon-wsl`).
