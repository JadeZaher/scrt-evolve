---
type: plan
track: 39
title: Native inference engine (candle) — implementation plan
status: planned
created: 2026-07-04
updated: 2026-07-04
depends_on: [19, 27, 29]
---

# Track 39 — Native inference engine (candle) — Plan

## Overview

Evolve owns the forward pass in-process (candle, Rust). This plan builds the
runtime in four phases that are **largely sequential**: Phase 0 (parity +
coverage gate) decides the retirement schedule; Phase A ships `evolve infer` /
`evolve serve` native and executes the llama.cpp retirement checklist for
cleared families; Phase B is the load-bearing Granite Mamba2/SSD kernel work;
Phase C exercises interleaved/prefix placement across ≥2 arch families.

The organizing seam is the **`ArchAdapter` trait** — the serving mirror of
track 04's per-arch loader in `crates/scrt-evolve/src/model.rs`. All new
serving code lands under a **new `serve` module** (`crates/scrt-evolve/src/serve/`)
so file ownership is clean vs the existing `model.rs` (train-side loader),
`generate/local.rs` (existing candle sampler / parity reference), and
`config.rs` (config surface). All ML stays behind `--features train`.

Grounding (existing files this plan touches or mirrors):
- `crates/scrt-evolve/src/model.rs` — track 04 loader; `LoadedModel`,
  `DecoderLayer`, `TinyLlama`, `arch_supported()`, `ModelError`. **The
  `ArchAdapter` trait is introduced here (or a sibling `model/arch.rs`).**
- `crates/scrt-evolve/src/generate/local.rs` — `LocalCandle` autoregressive
  sampler, `sample_token`/`argmax`, `SplitMix64`. Rust-side greedy-parity ref.
- `python/scrt_evolve_infer/{__main__.py,__init__.py,infer.py}` — the parity
  target: `--ab`, greedy (temp 0), `load_base_model`/`apply_adapter`/`generate`.
- `crates/scrt-evolve/src/config.rs` — `RuntimeConfig` (`[runtime]`, llama.cpp
  keys `backend`/`n_gpu_layers`/`llama_cpp_path`), `ServeConfig` (`[serve]`,
  track 33). New `[serve.placement]` added here.
- `crates/scrt-evolve-cli/src/main.rs` — `ModelCommand::Infer` (shells to
  `python -m scrt_evolve_infer`) and `ModelCommand::Run` (`[runtime]` →
  llama.cpp). Rewired to the native engine.
- `crates/scrt-evolve/src/branch/create.rs` — `create()` + `BranchHooks`;
  where the trainable⇒servable preflight is enforced.
- `crates/scrt-evolve/src/export.rs` — GGUF export (stays an export/interop
  path; candle's quantized loader also consumes GGUF for Phase 0's Q4_K_M leg).
- `crates/scrt-evolve/src/arbitration.rs` — track 33 co-residence arbitration;
  placement `auto` mode is its VRAM-arbitration policy.

Task IDs: `P0-*` (Phase 0), `PA-*` (Phase A), `PB-*` (Phase B), `PC-*` (Phase C).

---

## Phase 0 — parity + coverage go/no-go gate

Goal: prove candle can load and greedily match the Python forward on a
llama-family branch (safetensors base+LoRA AND exported Q4_K_M), and write the
registry-wide coverage matrix. This gate decides the **retirement schedule**,
not whether the track proceeds. **This whole phase is single-lane / sequential**
— it is the spine everything else hangs off.

- [ ] **P0-1 — `ArchAdapter` trait + `LlamaAdapter` impl.**
  Goal: define `trait ArchAdapter { layers/seam_points/apply_adapter/forward_layer }`
  and a llama-family impl wrapping the existing `TinyLlama`/`DecoderLayer`
  forward, addressing layers by index.
  Files: `crates/scrt-evolve/src/model.rs` (or new `crates/scrt-evolve/src/model/arch.rs`
  + re-export from `model.rs`).
  Accept: `cargo test --features train` — `LlamaAdapter::layers()` returns
  `num_layers` typed `LayerDesc`s; `forward_layer(i, x)` on a `random_fixture`
  matches the corresponding slice of `TinyLlama::forward` (byte-identical).

- [ ] **P0-2 — native safetensors base + LoRA compose-at-load.**
  Goal: load a real HF llama-family dir (`config.json`/`tokenizer.json`/
  `model.safetensors`) + `adapter.safetensors` and compose the LoRA into the
  forward via `ArchAdapter::apply_adapter`.
  Files: `crates/scrt-evolve/src/serve/loader.rs` (new); reuses `model.rs`
  `LoadedModel::load`; mirrors LoRA parameterization from
  `crates/scrt-evolve/src/train/lora.rs`.
  Accept: loads TinyLlama base + a track-04 adapter without remapping; forward
  runs finite logits; base-vs-adapter logits differ where the adapter is active.
  Blocked by P0-1.

- [ ] **P0-3 — Q4_K_M quantized-loader leg.**
  Goal: load the track-27 exported Q4_K_M GGUF via candle's quantized loader as
  the second parity leg.
  Files: `crates/scrt-evolve/src/serve/loader.rs`; consumes `export.rs` output.
  Accept: the exported GGUF loads and produces finite logits on the probe set;
  loader errors (not panics) on a malformed GGUF. Blocked by P0-1.

- [ ] **P0-4 — greedy token-level parity harness.**
  Goal: fixed-seed, greedy (temp 0) generation over the probe set through the
  native path, compared token-for-token against `scrt_evolve_infer` output.
  Files: `crates/scrt-evolve/tests/serve_parity.rs` (new); reuses
  `generate/local.rs` `argmax`/greedy path as the Rust reference; captures the
  Python `--ab` base output as fixtures under `crates/scrt-evolve/tests/fixtures/`.
  Accept: native greedy token stream == Python greedy stream on the probe set
  for the TinyLlama branch (both base and adapter legs). A committed
  **parity report** records any divergence + cause. Blocked by P0-2, P0-3.

- [ ] **P0-5 — coverage matrix (every registry family).**
  Goal: enumerate, per arch family in registry use, what candle covers today vs
  the op gap — explicitly Granite-4-h's Mamba2/SSD scan + hybrid block wiring.
  Files: `conductor/tracks/39-native-inference/coverage-matrix.md` (new
  committed artifact); reads the branch registry via `branch/manifest.rs`.
  Accept: a written matrix listing each family as `native | needs-op-work |
  refused`, with the Granite gap named concretely (Mamba2 scan, SSD state).
  **Independent of P0-1..4** (documentation/enumeration) — see partition map.

- [ ] **P0-6 — go/no-go gate decision + retirement schedule.**
  Goal: from P0-4 (parity) + P0-5 (coverage), record which families are cleared
  for llama.cpp retirement in Phase A and which stay preflight-refused.
  Files: `conductor/tracks/39-native-inference/parity-report.md` (new).
  Accept: committed decision doc naming the per-family retirement order.
  `[checkpoint marker]` — reviewer gate; blocks Phase A. Blocked by P0-4, P0-5.

- [ ] **P0-V — Verification:** ML-free `cargo build` + `cargo clippy` green;
  `cargo test --features train` (parity + adapter tests) green; parity report +
  coverage matrix committed. `[checkpoint marker]`

---

## Phase A — `evolve infer` / `evolve serve` native + retirement checklist

Goal: chat/complete against base+adapter natively with `--ab`, in-process
adapter hot-reload at keep-commit, and execute the llama.cpp retirement
checklist for the families Phase 0 cleared. Blocked by the P0-6 gate.

- [ ] **PA-1 — native generation engine (session + sampling).**
  Goal: a `serve::engine` that owns an `ArchAdapter`, a KV/position loop, and
  greedy + temperature sampling (lifted from `generate/local.rs::sample_token`).
  Files: `crates/scrt-evolve/src/serve/engine.rs` (new); reuses
  `generate/local.rs` sampler + `SplitMix64`.
  Accept: `engine.generate(prompt, max_new_tokens, temp, seed)` reproduces the
  P0-4 greedy stream; temperature>0 is seed-deterministic. Blocked by P0-6.

- [ ] **PA-2 — `evolve model infer` native (`--ab`, mirrors Python contract).**
  Goal: rewire `ModelCommand::Infer` to the native engine — same flags
  (`--adapter`/`--prompt`/`--ab`/`--max-new-tokens`/`--temperature`/`--chat`),
  base-vs-adapter side-by-side, NO python subprocess.
  Files: `crates/scrt-evolve-cli/src/main.rs` (`ModelCommand::Infer` handler);
  `crates/scrt-evolve/src/serve/mod.rs` (public `infer` entry).
  Accept: `evolve model infer --adapter … --prompt … --ab` prints base+adapter
  blocks matching the Python `--ab` layout on a cleared family; `--help` stable.
  Blocked by PA-1.

- [ ] **PA-3 — `evolve model run` / `evolve serve` native runtime.**
  Goal: rewire the config-driven serving lane (`ModelCommand::Run`) onto the
  native engine; where track 33 `[serve]` is live, expose one-shot + the
  hot-reload path.
  Files: `crates/scrt-evolve-cli/src/main.rs` (`ModelCommand::Run` handler);
  `crates/scrt-evolve/src/serve/mod.rs`.
  Accept: `evolve model run --prompt …` serves via candle (no llama.cpp
  subprocess) for a cleared family; output finite + deterministic at temp 0.
  Blocked by PA-1.

- [ ] **PA-4 — in-process adapter hot-reload at keep-commit.**
  Goal: track 33's process-bounce swap becomes an in-process
  `ArchAdapter::apply_adapter` reload at each keep-commit.
  Files: `crates/scrt-evolve/src/serve/engine.rs`; integration seam in
  `crates/scrt-evolve/src/arbitration.rs` (track 33 co-residence).
  Accept: a swap test reloads a second adapter into a live engine and the next
  generation reflects the new weights, no reload of base/tokenizer. Blocked by
  PA-1. **Serially after PA-2/PA-3 if it edits the same CLI handlers; otherwise
  parallel (engine + arbitration only).**

- [ ] **PA-5 — `[serve.placement]` config + `[runtime]` deprecation.**
  Goal: add `[serve.placement] { mode = "auto"|"manual", gpu_shards = [...] }`;
  deprecate the `[runtime]` llama.cpp keys with a load-time warning pointing at
  `[serve.placement]`.
  Files: `crates/scrt-evolve/src/config.rs` (new `PlacementConfig` under
  `ServeConfig`; deprecation note on `RuntimeConfig`);
  `crates/scrt-evolve-cli/src/config_reference.rs` (reference doc).
  Accept: `[serve.placement]` round-trips via `EvolveConfig::from_toml_str`;
  a config with `[runtime]` keys parses AND emits the deprecation warning.
  **Independent of PA-1..4** — config-only lane.

- [ ] **PA-6 — llama.cpp retirement checklist (cleared families).**
  Goal: for Phase-0-cleared families — `run-model`/`serve` rewired to native
  (done in PA-2/3), `[runtime]` deprecated (PA-5), docs updated.
  Files: `PORTABILITY.md`, `README.md`, `DESIGN.md`, `AGENTS.md` (doc updates);
  `conductor/tracks/39-native-inference/retirement-checklist.md` (new).
  Accept: committed checklist with each item evidenced; docs no longer present
  llama.cpp as a serving dependency for cleared families (GGUF still documented
  as an export format). Blocked by PA-2, PA-3, PA-5.

- [ ] **PA-V — Verification:** native `infer`/`run` on the cleared family match
  the Python `--ab` greedy output; `cargo build` (ML-free) + `cargo test` +
  `cargo test --features train` + `cargo clippy` green; retirement checklist
  committed. `[checkpoint marker]`

---

## Phase B — Granite Mamba2/SSD arch coverage (the load-bearing work)

Goal: implement the missing Granite hybrid ops in candle so a Granite branch
serves natively with forward-pass parity vs the Python transformers forward.
**This is deep, correctness-critical kernel work and MUST stay single-lane /
sequential** (CPU reference correctness first; CUDA speed second). Until B
lands, Granite serving is explicitly preflight-refused — never a hidden
fallback. Runs after Phase A but its early tasks are independent of Phase A's
CLI wiring (see partition map).

- [ ] **PB-1 — Mamba2/SSD selective-scan CPU reference op.**
  Goal: a correct, non-optimized CPU implementation of the Mamba2/SSD selective
  scan (the load-bearing kernel), tested against a known-good reference.
  Files: `crates/scrt-evolve/src/serve/ops/ssd.rs` (new).
  Accept: scan output matches a NumPy/torch reference on fixed random inputs to
  fp tolerance; errors (not panics) on shape mismatch. Single-lane.

- [ ] **PB-2 — Granite hybrid block wiring (`GraniteAdapter`).**
  Goal: an `ArchAdapter` impl wiring Mamba layers (using PB-1) + attention
  layers + the Mamba→attention `seam_points()` boundary.
  Files: `crates/scrt-evolve/src/serve/arch/granite.rs` (new); implements
  `ArchAdapter` from P0-1; extends `model.rs::arch_supported()` for Granite.
  Accept: a Granite config loads; `layers()` returns the correct typed
  (`Ssm | Attn | Mlp`) sequence; `seam_points()` names the Mamba→attn boundary
  (the `bench/seam_distill` seam). Blocked by PB-1.

- [ ] **PB-3 — Granite forward-parity vs Python transformers.**
  Goal: greedy forward-pass parity for a Granite branch: native vs the Python
  transformers forward on the probe set.
  Files: `crates/scrt-evolve/tests/serve_granite_parity.rs` (new); fixtures
  captured from the Python forward.
  Accept: native Granite greedy logits/tokens match the Python forward on
  probes to tolerance; recorded in an updated parity report. Blocked by PB-2.

- [ ] **PB-4 — CUDA fast path for the scan (speed, correctness-preserving).**
  Goal: a CUDA kernel for PB-1's scan behind the same op interface; upstream to
  candle if accepted. CPU path remains the correctness oracle.
  Files: `crates/scrt-evolve/src/serve/ops/ssd_cuda.rs` (new, cfg-gated).
  Accept: CUDA scan matches the CPU reference (PB-1) to tolerance; falls back to
  CPU when no device. Blocked by PB-1. **Independent of PB-2/PB-3** once the op
  interface is frozen — can run parallel to Granite wiring.

- [ ] **PB-5 — Granite retirement/preflight flip.**
  Goal: flip Granite from preflight-refused to native-servable in the coverage
  matrix + preflight once PB-3 passes.
  Files: `conductor/tracks/39-native-inference/coverage-matrix.md`;
  `crates/scrt-evolve/src/branch/create.rs` (preflight table, see PC-4).
  Accept: Granite `branch create` no longer refuses; coverage matrix updated.
  Blocked by PB-3.

- [ ] **PB-V — Verification:** Granite forward parity green (`cargo test
  --features train`); CPU reference is the oracle; ML-free `cargo build` +
  clippy green; parity report updated. `[checkpoint marker]`

---

## Phase C — placement (interleaved + prefix, ≥2 families)

Goal: `auto` + `manual` interleaved placement maps honored and verified against
measured VRAM, exercised on ≥2 arch families via `ArchAdapter` (prefix AND
interleaved). Blocked by Phase A (engine) for both families; the Granite family
leg additionally needs Phase B.

- [ ] **PC-1 — placement resolver (`auto` + `manual`).**
  Goal: resolve `[serve.placement]` into a per-layer-index device map; `auto`
  probes free VRAM (same probe as `doctor`/`max_vram_gb`) and fills; `manual`
  honors `gpu_shards` exactly and refuses-fast on impossible maps.
  Files: `crates/scrt-evolve/src/serve/placement.rs` (new); VRAM probe reused
  from the `doctor`/`daemon` path; interacts with `arbitration.rs`.
  Accept: given free-VRAM `V` and layer costs, `auto` yields a valid map;
  `manual` with an over-budget `gpu_shards` returns a clear error (no panic);
  interleaved indices `[0,1,2,8,9,16]` resolve to exactly those on GPU.
  Blocked by PA-5 (config) + PA-1 (engine layer addressing).

- [ ] **PC-2 — engine honors the device map (interleaved forward).**
  Goal: `serve::engine` forwards each layer on its assigned device per the
  resolved map (arbitrary interleaving, not just a contiguous prefix).
  Files: `crates/scrt-evolve/src/serve/engine.rs`;
  `crates/scrt-evolve/src/serve/placement.rs`.
  Accept: a forward with an interleaved map produces logits identical (to fp
  tolerance) to the all-CPU forward; a prefix map likewise. Blocked by PC-1.

- [ ] **PC-3 — placement verified on ≥2 families vs measured VRAM.**
  Goal: exercise prefix AND interleaved maps on a llama-family branch and a
  second family (Granite once Phase B lands, else a second llama-family arch).
  Files: `crates/scrt-evolve/tests/serve_placement.rs` (new).
  Accept: both map shapes honored on both families; resident-layer VRAM matches
  the resolver's prediction within a documented margin. Blocked by PC-2 (+PB-3
  for the Granite leg).

- [ ] **PC-4 — `branch create` trainable⇒servable preflight.**
  Goal: enforce the invariant — a base the branch factory can train must load in
  native serve OR be refused at `branch create` preflight with the
  unsupported-arch reason (doctor-style).
  Files: `crates/scrt-evolve/src/branch/create.rs` (`create()` preflight);
  reads the coverage matrix / `arch_supported()`.
  Accept: creating a branch on an unsupported arch is refused up front with the
  reason; a supported arch proceeds. Blocked by P0-1 (adapter registry); coverage
  updated by PB-5. **Largely independent** file-wise (branch module) — see map.

- [ ] **PC-V — Verification:** interleaved + prefix maps honored on ≥2 families,
  verified vs measured VRAM; preflight invariant enforced; full sweep
  (`cargo build` ML-free + `cargo test` + `cargo test --features train` +
  `cargo clippy`) green; acceptance criteria from spec all evidenced.
  `[checkpoint marker]`

---

## DEPENDENCY GRAPH

### Sequential spine (each gates the next; DO NOT parallelize across the arrow)

```
P0-1 (ArchAdapter trait)
  └─> P0-2 (safetensors+LoRA) ─┐
  └─> P0-3 (Q4_K_M leg) ───────┴─> P0-4 (greedy parity harness)
                                        └─> P0-6 (go/no-go gate) [checkpoint]
                                              └─> PA-1 (native engine)
                                                    └─> PA-2 / PA-3 (infer/run CLI)
                                                          └─> PA-6 (retirement checklist)
Phase A gate ─> PB-1 (SSD CPU ref) ─> PB-2 (Granite wiring) ─> PB-3 (Granite parity) ─> PB-5 (flip)
PA-1 (engine) + PA-5 (placement cfg) ─> PC-1 (resolver) ─> PC-2 (interleaved forward) ─> PC-3 (verify ≥2 fam)
```

### Blockers per task

| Task | Blocked by |
| :--- | :--- |
| P0-1 | — (spine root) |
| P0-2 | P0-1 |
| P0-3 | P0-1 |
| P0-4 | P0-2, P0-3 |
| P0-5 | — (independent; enumeration/docs) |
| P0-6 | P0-4, P0-5 |
| PA-1 | P0-6 |
| PA-2 | PA-1 |
| PA-3 | PA-1 |
| PA-4 | PA-1 (serial after PA-2/3 only if it edits the same CLI handler) |
| PA-5 | — (config-only; can start any time, needed by PA-6 + PC-1) |
| PA-6 | PA-2, PA-3, PA-5 |
| PB-1 | Phase A gate (schedule) — code-independent of PA CLI |
| PB-2 | PB-1 |
| PB-3 | PB-2 |
| PB-4 | PB-1 (independent of PB-2/PB-3 once op iface frozen) |
| PB-5 | PB-3 |
| PC-1 | PA-1, PA-5 |
| PC-2 | PC-1 |
| PC-3 | PC-2 (+ PB-3 for the Granite leg) |
| PC-4 | P0-1 (coverage flipped by PB-5) |

### Genuinely independent (safe to run in parallel — disjoint files)

- **P0-5** (coverage matrix, docs only) parallel to **P0-1/2/3** (Rust code).
- **PA-5** (config.rs + config_reference.rs) parallel to **PA-1/2/3** (serve/ +
  CLI handlers) — different files.
- **PB-4** (CUDA scan) parallel to **PB-2/PB-3** (Granite wiring/parity) once the
  op interface from PB-1 is frozen.
- **PC-4** (branch/create.rs preflight) parallel to **PC-1/2/3** (serve/placement
  + tests) — different modules; only the coverage-matrix artifact is shared
  (append-only, low-conflict).

---

## PARALLELIZATION / PARTITION MAP

Five worker lanes with **disjoint file ownership**. A lane may only edit files
it owns. Cross-lane needs go through the sequential gates, not shared edits.

### Lane 1 — CORE ENGINE (spine; SINGLE-LANE, cannot be split)
Owns: `crates/scrt-evolve/src/model.rs` (+ `model/arch.rs`),
`crates/scrt-evolve/src/serve/mod.rs`, `serve/loader.rs`, `serve/engine.rs`,
`crates/scrt-evolve/tests/serve_parity.rs`.
Tasks: **P0-1 → P0-2 → P0-3 → P0-4 → PA-1 → PA-4**. This is the critical path.
Everything else depends on P0-1 (trait) then P0-6 (gate) landing from this lane.

### Lane 2 — CLI + SERVE WIRING
Owns: `crates/scrt-evolve-cli/src/main.rs` (`ModelCommand::Infer`/`Run`
handlers only), the public `infer`/`run` entry surface in `serve/mod.rs`
(coordinated with Lane 1 — Lane 1 owns the module body, Lane 2 owns the CLI
handler wiring).
Tasks: **PA-2, PA-3**. Starts once PA-1 exists. NOTE: `serve/mod.rs` is the one
coordination point with Lane 1 — keep Lane 2's edits to the CLI handlers and a
thin public entry to avoid overlap.

### Lane 3 — CONFIG + DOCS + RETIREMENT (mostly parallel from day 1)
Owns: `crates/scrt-evolve/src/config.rs`,
`crates/scrt-evolve-cli/src/config_reference.rs`, `PORTABILITY.md`,
`README.md`, `DESIGN.md`, `AGENTS.md`, and the track doc artifacts
`coverage-matrix.md`, `parity-report.md`, `retirement-checklist.md`.
Tasks: **P0-5, P0-6 (doc side), PA-5, PA-6**. P0-5 and PA-5 can start
immediately (no code dep). PA-6 waits on Lane 2 (PA-2/3) + PA-5.

### Lane 4 — GRANITE KERNEL (SINGLE-LANE for the load-bearing op)
Owns: `crates/scrt-evolve/src/serve/ops/` (`ssd.rs`, `ssd_cuda.rs`),
`crates/scrt-evolve/src/serve/arch/granite.rs`,
`crates/scrt-evolve/tests/serve_granite_parity.rs`, plus the Granite line in
`model.rs::arch_supported()` (coordinate with Lane 1 on that one function).
Tasks: **PB-1 → PB-2 → PB-3 → PB-5**, with **PB-4** as an optional sub-lane
(Lane 4b) that may run in parallel *after* PB-1 freezes the op interface.
**PB-1 correctness is the whole track's load-bearing risk — do NOT split the
CPU-reference scan across workers.**

### Lane 5 — PLACEMENT + PREFLIGHT
Owns: `crates/scrt-evolve/src/serve/placement.rs`,
`crates/scrt-evolve/tests/serve_placement.rs`,
`crates/scrt-evolve/src/branch/create.rs` (preflight only),
`crates/scrt-evolve/src/arbitration.rs` (placement/co-residence seam).
Tasks: **PC-1 → PC-2 → PC-3**, and **PC-4** (preflight) which is file-independent
and can run as soon as P0-1's trait exists. Starts once PA-1 (engine) + PA-5
(config) land.

### What MUST stay sequential / single-lane (do NOT parallelize)

1. **The Phase 0 parity gate (Lane 1, P0-1→P0-4→P0-6).** It is the spine that
   decides the retirement schedule; every other lane is downstream of P0-1 (the
   trait) and blocked from *retiring* anything until P0-6 signs off. No lane may
   touch `serve/engine.rs`/`loader.rs` before P0-1 exists.
2. **The Granite Mamba2/SSD kernel (Lane 4, PB-1→PB-2→PB-3).** Correctness-
   critical custom-op work; CPU reference is the oracle. PB-1's scan is the
   single load-bearing risk and must not be sharded across workers. Granite stays
   preflight-refused until PB-3 passes — never a hidden fallback.
3. **`serve/engine.rs` forward loop.** Owned solely by Lane 1 (and Lane 5 only
   for the placement-map integration in PC-2, strictly after Lane 1's PA-1). Two
   lanes must never edit the forward loop concurrently.
4. **The go/no-go checkpoints** (`P0-6`, `PA-V`, `PB-V`, `PC-V`) are reviewer
   gates — the next phase does not start until the checkpoint's reviewer pass
   is green.

### Safe-to-parallelize summary

| Can run concurrently | Because |
| :--- | :--- |
| Lane 3 (config/docs) with Lane 1 (engine) from day 1 | disjoint files; P0-5/PA-5 have no code dep |
| Lane 4b (PB-4 CUDA) with Lane 4 (PB-2/3) | disjoint files once PB-1 op iface frozen |
| Lane 5 PC-4 (preflight) with Lane 5 PC-1/2/3 | branch module vs serve module — different files |
| Lane 2 (CLI) with Lane 3 (config) | different files; both wait only on their own deps |
