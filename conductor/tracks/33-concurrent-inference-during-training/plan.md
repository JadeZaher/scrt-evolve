---
type: Track Plan
title: Concurrent inference during shard training — Plan
description: Phased plan for serve-while-you-train — a live inference server that hot-swaps the adapter only at each keep-commit, VRAM-arbitrated against the fractional trainer on 8 GB.
tags: [track-33, in-progress]
timestamp: 2026-07-03T00:00:00Z
resource: ./metadata.json
depends_on: [26, 25, 19, 27]
---

# Track 33 — Concurrent inference during shard training — Plan

## Design decision that shapes this plan

The spec offers three interruption models (A strict-alternate, B co-resident,
C CPU-serve) and recommends **B with A as the degrade path**. The single unknown
that picks B vs A is a **live VRAM co-residence measurement** on the 8 GB box,
which no code-authoring agent can run — it needs real hardware + Granite weights.

**Resolution: do not fork the code on the unmeasured number.** Build the runtime
to support **both A and B behind one selection point**, and make the measurement a
first-class `doctor` deliverable that *selects* the mode at runtime (B if it fits,
else A) and *persists the reason*. This means:

- Every phase below is authorable and unit-testable **without a GPU** (the heavy
  train/serve/free-vram effects stay injected closures, exactly as `daemon.rs`
  already does — see daemon.rs:22-27, `free_vram_gb` at daemon.rs:57).
- The **live acceptance run** (spec §Acceptance) is a **human-in-the-loop step**
  the user performs on the box; it is called out explicitly in Phase 6, not
  something ultrapilot claims as "done".

Per the test-once-at-end policy: **all code lands first, then ONE full sweep**
(`cargo test --workspace`, `cargo clippy --all-targets`, Python unit tests) in
Phase 6. Do not run tests after each phase.

## Phase → file-ownership map (for parallel execution)

Phase 0 is a **barrier**: it defines the shared config fields and the commit-swap
signal contract every other phase consumes. After Phase 0, Phases 1–4 own disjoint
files and parallelize; Phase 5 depends on 2+3; Phase 6 is the single final sweep.

| Phase | Owns (primary files) | Depends on |
|:--|:--|:--|
| 0 shared contract | `crates/scrt-evolve/src/config.rs` (or the config module), `bench/ambient-granite.toml`, `bench/evolve.toml` | — |
| 1 doctor measurement + mode select | `crates/scrt-evolve-cli/src/main.rs` (doctor cmd), new `arbitration.rs` selection helper | 0 |
| 2 daemon arbitration + swap-signal emit | `crates/scrt-evolve/src/daemon.rs` | 0 |
| 3 `serve --live` server | `crates/scrt-evolve-cli/src/main.rs` (Serve/RunModel path) | 0, (2's signal schema) |
| 4 adapter-swap mechanism | `python/scrt_evolve_infer/infer.py`, new converter script | 0 |
| 5 status surface | `main.rs` watch/serve status | 2, 3 |
| 6 sweep + docs + live acceptance | tests, `conductor/tracks/33-*/AGENTS.md`, README | all |

---

## Phase 0 — Shared config surface + commit-swap signal contract (BARRIER)

**Goal:** land the shared vocabulary every other phase compiles against, so the
parallel phases never race on the same struct.

### Tasks
- [ ] **Config:** add `serve_reservation_gb: Option<f64>` to the `[daemon]` config
  (the carve-out the trainer subtracts before starting a block). Default `None`
  (= behave exactly as today; no reservation → model A/degraded).
- [ ] **Config:** add a `[serve]` section (or extend `[runtime]`): `live: bool`,
  `swap_debounce_commits: u32` (0 = swap every keep), `mode: "auto" | "b" | "a"`
  (default `auto` → doctor decides).
- [ ] **Signal contract:** define the **commit-swap signal** the daemon emits and
  the server subscribes to. Reuse the existing evolution-log row where possible;
  the contract is: on a `keep` commit, a `served-ready` record carries
  `{ version: u64, adapter_path, base_path, timestamp }`. Document the schema in
  the track `AGENTS.md`. Prefer a small append-only signal file
  (`<state>/served-ready.jsonl`) so the server tails it without parsing the whole log.
- [ ] **Config:** wire the new fields into `bench/ambient-granite.toml` (commented,
  disabled by default) and `bench/evolve.toml` so the live config documents them.
- [ ] Serde defaults + validation + a unit test that round-trips the new fields.

---

## Phase 1 — Doctor co-residence measurement + mode selection

**Goal:** the harness that *measures* the co-resident footprint and *selects* B or
degrades to A, persisting the reason. The arithmetic/selection is fully unit-tested
GPU-free; the live numbers come from the user's hardware run.

### Tasks
- [ ] New `arbitration.rs` (in `scrt-evolve`): `fn select_mode(serve_footprint_gb,
  block_peak_gb, cuda_ctx_gb, ceiling_gb, reservation_gb) -> Mode` returning
  `Mode::Coresident { .. }` when `serve + block + ctx ≤ ceiling`, else
  `Mode::Alternate { reason }`. Pure function, exhaustively unit-tested.
- [ ] Extend `evolve doctor`: measure (a) served Q4_K_M GGUF footprint at the
  configured/reduced `n_gpu_layers`, (b) one training block peak (reuse the
  daemon's `free_vram_gb` probe around a dry block), (c) CUDA context overhead;
  feed them to `select_mode`; print the decision + reason; persist to
  `<state>/arbitration.json`. Behind the injected-probe pattern so tests supply
  synthetic VRAM numbers.
- [ ] Doctor honors `[serve].mode`: `auto` runs the measurement; `a`/`b` force and
  only warn if the forced mode won't fit.
- [ ] Verify (in doctor output, not code) the llama.cpp build supports `--lora`
  hot-apply for `granitemoehybrid`; if the probe can't confirm, recommend Phase-4
  option 2 (debounced re-export). Record the finding in the track `AGENTS.md`.

---

## Phase 2 — Daemon-side VRAM arbitration + swap-signal emission

**Goal:** the trainer respects the serve reservation, and every keep-commit emits
the swap signal the live server consumes. All in `daemon.rs`, all injected-closure
friendly (no real GPU in tests).

### Tasks
- [ ] In the gating logic (daemon.rs ~L221-240 `vram_ok`/`gpu_ok`), subtract
  `serve_reservation_gb` from the available headroom before a block starts:
  the trainer only proceeds when `free − reservation ≥ block_need`. Reservation
  `None` ⇒ current behavior unchanged.
- [ ] Model-A degrade path: when `mode == Alternate`, treat the served inference
  process as a first-class foreground in the `pause_on_gpu_process` check
  (daemon.rs:128, L228/L240) so training yields the GPU to a live request.
- [ ] At the **keep** branch of the transaction (daemon.rs ~L587+), after the
  merged flat `adapter.safetensors` exists, append a `served-ready` record
  (Phase-0 schema) with the incremented version. Rollback/catastrophe emit nothing.
- [ ] Add a `DaemonHooks` seam for the signal emit (injected closure) so tests
  assert "a keep emits exactly one served-ready record; a rollback emits none"
  without touching the filesystem.
- [ ] Unit tests: reservation gating (fits / doesn't fit / None), one-signal-per-keep,
  zero-signal-per-rollback, version monotonicity.

---

## Phase 3 — `serve --live` server (subscribe + atomic hot-swap)

**Goal:** a long-lived inference server that serves the current committed adapter
and hot-swaps atomically at each signal — inference never sees a torn adapter.

### Tasks
- [ ] Add `--live` (and `--config`) to the serve entrypoint in `main.rs` (reuse the
  existing `Serve`/`RunModel`/`branch serve` infra, main.rs:824/1280/3273). Non-live
  path unchanged.
- [ ] Server loop: load current committed adapter → serve → tail the Phase-0
  `served-ready.jsonl` → on a new record, load vN+1 **fully** into a staging slot,
  then **flip an atomic pointer** (swap atomicity, spec §Risks). A request in
  flight finishes on vN; the next request sees vN+1. Never serve a partially
  loaded adapter.
- [ ] Debounce: honor `swap_debounce_commits` (only swap every N records / M minutes)
  so the GGUF path isn't re-quantized every step.
- [ ] `cpu_fallback` guard: serving on CPU is forward-only and fine; the arbitration
  must not conflate "serve on CPU" with "train on CPU" (spec §Risks). Assert this
  in a test with a CPU-served config.
- [ ] Unit tests (effects injected): swap is all-or-nothing under a simulated
  mid-load failure (old version stays served); debounce coalesces N signals into
  one swap; served version advances monotonically.

---

## Phase 4 — Adapter-swap mechanism (converter + transformers path)

**Goal:** provide the actual swap payload for both serve backends.

### Tasks
- [ ] `safetensors LoRA → GGUF LoRA` converter (small script + a thin CLI hook) for
  the llama.cpp `--lora` hot-apply path (spec §hot-swap option 1). If the installed
  llama.cpp can't hot-apply for `granitemoehybrid` (Phase-1 finding), the converter
  is still authored but the server defaults to option 2 (debounced re-export).
- [ ] Wire `infer.py::apply_adapter` (infer.py:65) as the transformers swap path for
  the non-GGUF case — reload the flat adapter into the resident model in place
  (cheapest swap). Add a `reload_adapter(model, adapter_dir)` that re-applies over
  an already-loaded model without a full reload.
- [ ] Unit tests: converter produces a GGUF-LoRA whose tensor names round-trip the
  `save_adapter()` contract (infer.py:5-7); `reload_adapter` swaps weights without
  re-instantiating the base.

---

## Phase 5 — Status surface (lag + residency)

**Goal:** the user can see how far the served model lags the trained one, and where
it's resident right now.

### Tasks
- [ ] Extend `watch status` / add `serve status`: show **served version** vs
  **latest committed version** (the lag), read from `served-ready.jsonl` +
  arbitration state.
- [ ] Show current residency: GPU (co-resident, mode B) vs CPU (alternate/degraded,
  mode A), sourced from `<state>/arbitration.json`.
- [ ] Unit test the lag/residency rendering against synthetic state files.

---

## Phase 6 — Single sweep + docs + live acceptance (human-in-the-loop)

### Tasks
- [ ] **ONE** full sweep: `cargo test --workspace`, `cargo clippy --all-targets`,
  Python unit tests. Fix to green.
- [ ] Track `AGENTS.md` (new, in this track dir or the touched source dirs): document
  the signal contract, the A/B selection rule, and the `cpu_fallback` caveat, per
  the directory-level-docs convention (one-line pointers in code).
- [ ] README / RUNBOOK: a `evolve serve --live --config bench/ambient-granite.toml`
  quickstart line.
- [ ] **LIVE ACCEPTANCE (user runs on the 8 GB box, not an agent):** run the daemon +
  `serve --live` on Granite/WSL; confirm answers throughout, served version
  increments at each keep, inference blips only at swap, and doctor cleanly reports
  B-fits or A-degraded-with-reason (spec §Acceptance). This is the only step that
  proves the track; mark the track `completed` only after it passes.

---

## Risks carried from spec (resolved in-plan)
- **8 GB co-resident fit** → not a code fork; `select_mode` + doctor decide at
  runtime (Phase 1). Live number is Phase-6 acceptance.
- **llama.cpp GGUF-LoRA hot-apply support** → probed in Phase 1; converter authored
  regardless (Phase 4); debounced re-export is the always-available fallback.
- **Swap atomicity** → Phase 3 staged-load-then-flip, tested against mid-load failure.
- **cpu_fallback conflation** → Phase 3 explicit guard + test.
