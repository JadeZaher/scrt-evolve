---
type: spec
track: 39
title: Native inference engine (candle) — the only serving path, model-generic
status: planned
created: 2026-07-03
updated: 2026-07-03
depends_on: [19, 27, 29]
---

# Track 39 — Native inference engine (candle) — Spec

## Goal

Evolve owns its inference runtime outright: load an evolved model (base +
`adapter.safetensors` directly, or an exported GGUF via the quantized loader)
and run the forward pass **in-process in Rust via candle**. This is the ONLY
serving path — the llama.cpp sidecar is **retired**, not wrapped (decision
2026-07-03). Owning the forward pass enables what stock runtimes cannot:

1. **Arbitrary per-shard placement** — `[serve.placement].gpu_shards =
   [0, 1, 2, 8, 9, 16]`: explicit layer indices resident on GPU, rest on
   CPU/RAM, **interleaving allowed** (llama.cpp could only express a
   contiguous `n_gpu_layers` prefix).
2. **Hidden-state access at named seams** — per-arch `seam_points()` expose
   the model's structural joints (e.g. Granite's Mamba→attention boundary,
   the same seam the `bench/seam_distill` lane trains against) for in-process
   instrumentation and capture; no C++ fork, no Python round-trip.
3. **No export round-trip to prompt** — base + adapter served directly;
   in-process adapter hot-reload at each keep-commit.
4. **Arch-generic serving** — one `ArchAdapter` seam so ANY base the branch
   factory can train is also servable (the branch-from-any-model invariant).

## Decision log

- **2026-07-03 — native-only.** No dual-engine maintenance. The llama.cpp
  sidecar (`[runtime]`, `run-model`, `serve`) is retired once the Phase 0
  parity + coverage gate passes for every model family in registry use.
  **GGUF remains an EXPORT format** (track 27 — interop, lexame sharing,
  other people's runtimes; candle's quantized loader also consumes it), it
  just stops being a *serving dependency*. Consequence accepted: a model
  family candle can't run is not "fall back to llama.cpp" — it is **implement
  the missing ops in the runtime we own** (Phase B) or refuse the arch at
  `branch create` preflight. No silent half-support.

## Model-generic design: the `ArchAdapter` seam

Serving must not be hardcoded to one architecture. A per-arch adapter trait
(the serving mirror of track 04's `model.rs` per-arch loader seam):

```rust
trait ArchAdapter {
    fn layers(&self) -> Vec<LayerDesc>;       // ordered; typed: Attn | Ssm | Mlp | MoE
    fn seam_points(&self) -> Vec<SeamPoint>;  // arch-specific structural boundaries (instrumentation/capture)
    fn apply_adapter(&mut self, lora: &LoraWeights) -> Result<()>;  // compose at load
    fn forward_layer(&self, idx: usize, x: Tensor, dev: &Device) -> Result<Tensor>;
}
```

- **Placement and hooks address layers by index**, arch-agnostic — a
  `gpu_shards` array or a hook registration means the same thing on a pure
  transformer, a Mamba hybrid, or (later) an MoE.
- **Seam selection is per-arch**: pure transformers → depth boundaries;
  Granite-4-h → the Mamba→attention boundary; MoE → the router boundary
  (enumerated, not promised).
- **Family matrix (initial):** llama-family (TinyLlama, Llama, Mistral, Qwen,
  Gemma — candle-transformers has loaders) = day 1. Granite-4-h hybrid =
  Phase B op work (Mamba2/SSD scan). MoE families = out of v1, listed for the
  trait's sake.

**The branch-from-any-model invariant:** any base the branch factory
(Python/transformers, track 19/29) can train must either (a) load in native
serve, or (b) be **refused at `branch create` preflight** (doctor-style, with
the unsupported-arch reason). No branch may exist that cannot be served —
trainability and servability are checked together, up front.

## Config sketch

```toml
[serve.placement]
mode = "auto"                # auto | manual
gpu_shards = [0, 1, 2, 8, 9, 16]   # manual: interleaved layer indices on GPU; rest CPU/RAM
```

`auto` measures free VRAM at load (same probe as `doctor`/`max_vram_gb`) and
fills; `manual` honors the array exactly and refuses fast on impossible maps.
The `[runtime]` llama.cpp keys are deprecated with a migration note (config
load warns, points at `[serve.placement]`); removed at retirement.

## Phases

- **Phase 0 — parity + coverage go/no-go.** candle loads the TinyLlama branch
  (safetensors base + LoRA at load, AND the exported Q4_K_M via the quantized
  loader). **Token-level greedy parity** vs `scrt_evolve_infer` on the probe
  set (fixed seed). Enumerate Granite-4-h's op gap (Mamba2 scan, hybrid block
  wiring) → a written **coverage matrix** for every family in the registry.
  This gate decides the retirement schedule, not whether the track proceeds.
- **Phase A — `evolve infer` / `evolve serve` native.** Chat/complete against
  base+adapter, `--ab` on/off (mirrors the Python infer contract),
  **in-process adapter hot-reload at keep-commit** (track 33's swap becomes a
  reload, not a process bounce). llama.cpp **retirement checklist** executed
  for the families Phase 0 cleared: `run-model`/`serve` rewired, `[runtime]`
  deprecated, docs/PORTABILITY updated.
- **Phase B — arch coverage (the load-bearing work).** Implement the missing
  Granite Mamba2/SSD ops in candle (custom op; CPU-reference correctness
  first, CUDA speed second; upstream to candle if accepted). Acceptance: a
  Granite branch serves natively with forward-pass parity vs the Python
  transformers forward on probes. Until B lands, Granite serving is
  explicitly "not yet" (preflight says so) — never a hidden fallback.
- **Phase C — placement.** `auto` + `manual` interleaved maps, honored and
  verified against measured VRAM in tests; exercised on ≥2 arch families via
  `ArchAdapter` (prefix AND interleaved cases).
*(A fourth phase — adaptive-compute exit policies on the hook seam — was
measured 2026-07-03 and removed the same day: no exploitable depth redundancy
in Granite-4-h-tiny. That research continues outside this repo; `seam_points()`
remains as the structural/instrumentation fundamental only.)*

## Relationship to other tracks

- **Track 33 (serve-while-train):** its serving half becomes this engine —
  co-residence arbitration + in-process adapter swap at keep-commit;
  placement `auto` mode is the VRAM-arbitration policy.
- **Track 27 (export):** unchanged as an export/interop path; no longer the
  only way to prompt.
- **Track 40 (delegation contract):** independent of serving specifics — the
  capability card derives from judged probe runs; native serve is simply the
  runtime those models execute on.

## Risks

- **Mamba2 op implementation is the load-bearing risk** — real kernel work,
  accepted knowingly (native-only means no fallback). Mitigations: CPU
  reference first (correctness is the bar; the 4060 forward can be modest),
  candle upstream may land SSD ops independently, and Granite keeps its
  honest "not yet" preflight until B lands.
- **Perf gap vs llama.cpp is now user-facing** (no sidecar to hide behind).
  Document the measured gap; optimize the hot path after correctness; the
  placement + hook control is what's being bought.
- **Parity bugs** (tokenizer, RoPE variants, sampling) — the parity gate IS
  the acceptance; no parity, no retirement.
- **Retirement sequencing** — llama.cpp is removed per-family only after that
  family's parity gate, registry-wide before the code is deleted.

## Acceptance

- Phase 0 coverage matrix + parity report committed.
- Native infer of a llama-family branch matches Python `--ab` (greedy, fixed
  seed) on the probe set; llama.cpp retired for cleared families with config
  migration warnings in place.
- Granite branch serves natively with forward parity (Phase B) — or remains
  explicitly preflight-refused, never silently degraded.
- Interleaved + prefix placement maps honored, verified against measured
  VRAM, on ≥2 arch families.
- `branch create` preflight enforces the trainable⇒servable invariant.
- ML-free `cargo build` + clippy green.
