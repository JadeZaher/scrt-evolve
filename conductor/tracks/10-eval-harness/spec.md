# Evaluation Harness — Probe Set + Scorer (shared) — Specification

## Goal
The **shared evaluation substrate** the self-evolve lane depends on. Tracks 11
(regen gate), 12 (constitution judge), and 15 (regression/catastrophe detection)
all assume "score the model on a held-out probe and tell me if it got worse" —
this track builds that once, with one owner, instead of three half-built
evaluators. A fixed, seeded **probe set** + a **scorer** producing comparable
metrics across evolution steps.

This is the integration gap surfaced by the e2e cogency audit: 11/12/15 each
referenced an evaluator nobody built. It is sequenced FIRST in the lane so its
consumers can depend on it.

## Scope
- **`evolve/probe.rs` — `ProbeSet`**: a fixed, versioned, held-out set of
  evaluation items, never trained on. Item kinds mirror the dataset:
  `tool_call`/`cli` (executable-checkable), `qa`/`instruction` (reference or
  judge-checkable). Loaded from `work_dir/probe.jsonl`; deterministic ordering.
  A builder that *carves* a probe set out of a dataset (hold-out split) + asserts
  no overlap with training rows.
- **`evolve/score.rs` — `Scorer`** producing a `ScoreReport`
  {correctness, constitution_adherence, mean_exit_depth, perplexity?, n,
  probe_version}. Three signal sources, each independently optional:
  - **executable correctness** — reuse the track-11 executable gate over
    `tool_call`/`cli` probe items (does the emitted command parse/resolve).
    Works WITHOUT a live model forward pass for already-generated outputs; for
    fresh generation it needs the model (see backends).
  - **constitution adherence** — sample probe completions, judge against the
    track-12 constitution (a `judge` backend scores principle violations).
  - **depth / perplexity** — only with a real forward pass (model backend).
- **Scoring backends (the PyO3→transformers seam):** scoring that needs a real
  model forward pass (generate completions, perplexity, exit-depth, attribution)
  routes through a backend, NOT a hand-built candle loop:
  - `api` — generate probe completions via the existing `ApiEndpoint` (works
    today, no ML deps; correctness + constitution scoring only).
  - `pyo3` — **drive HuggingFace `transformers` via the PyO3 bridge** to load
    the in-training model and run forward passes (perplexity, exit-depth probes,
    logprobs). This is the directive: heavy ML scoring depends on
    transformers/torch through `bridge.rs`, not on candle being mature.
  - `candle` — optional native path, gated `--features train`, lower priority.
- **`StepVerdict` helper**: given a baseline `ScoreReport` and a candidate one +
  per-metric tolerances/floors (from the consumer's config), classify
  `accept | regress | catastrophic`. Pure function over two reports — the shared
  decision logic 11/15 both call so verdicts are consistent.
- **`[evolve.eval]` config block** — added as a field on `EvolveConfig`
  (new `Option<EvalConfig>`; `EvolveConfig` uses `#[serde(default)]` so adding a
  field is non-breaking): `probe_path`, `probe_holdout_frac`, `scorer_backend`
  (api|pyo3|candle), `judge` (api endpoint for constitution scoring),
  `metrics` (which to compute). Absent block → no eval (lane runs unguarded with
  a logged warning).
- CLI: `scrt-evolve eval --probe probe.jsonl` (score current model → report),
  `scrt-evolve probe build --from dataset.jsonl --holdout 0.1` (carve a probe set).

## Constraints
- **One harness, three consumers.** 11/12/15 MUST consume this module; they do
  NOT define their own probe/scorer. Their specs reference `ScoreReport` /
  `StepVerdict` by name.
- **PyO3→transformers for real forward passes** (the directive): perplexity /
  exit-depth / logprob scoring is implemented over `transformers` through the
  PyO3 bridge, not hand-built in candle. The `api` backend covers
  correctness+constitution with zero ML deps so the harness is useful before any
  Python/candle is wired. `candle` backend is an optional later path.
- **Probe is held-out, fixed, seeded.** No probe item may appear in any training
  dataset (overlap asserted). Probe version is stamped in every report so a
  report is only compared against same-version baselines.
- **Graceful degradation.** `api` backend → correctness + constitution only
  (no depth/perplexity), logged. No probe → harness returns "uncovered" not an
  error.
- ML-free build green: `ProbeSet`, `ScoreReport`, `StepVerdict`, config, the
  `api` scorer backend, and CLI compile with no candle and no Python. `pyo3`
  backend behind `--features pyo3`; `candle` backend behind `--features train`.
- Reuses: track 11 executable gate (correctness), track 12 constitution (judge),
  track 02 `ApiEndpoint` (api backend), `bridge.rs` (pyo3 backend), `dataset.rs`
  (`gen` provenance carried into probe items for traceability).

## Acceptance
- `probe build` carves a held-out probe set from a dataset with asserted
  zero-overlap; `ProbeSet` loads deterministically.
- `Scorer` (api backend) produces a `ScoreReport` with correctness +
  constitution_adherence over a fixture probe, no ML deps.
- `StepVerdict` classifies accept/regress/catastrophic from two reports +
  tolerances (pure-function test across the three outcomes).
- `pyo3` backend (behind `--features pyo3`) computes perplexity/exit-depth over a
  tiny `transformers` model through the bridge (or is asserted to compile +
  surface the seam if a model isn't present in CI).
- `[evolve.eval]` round-trips as an `EvolveConfig` field; absent block → no-eval
  warning, no crash.
- Probe-version mismatch between baseline and candidate reports is refused
  (asserted).
- ML-free + `--features pyo3` + `--features train` build green.
- **Styleguide gates** (code-styleguides.md): probe items are deterministic +
  held-out (§2.2 determinism, §2.1 no-ambient-state); `ScoreReport` is reproducible
  under a fixed probe version (§2.1 idempotency); the executable `gate.rs` is pure
  (no effects in arg-gen, §2.1). Built per §4 (scaffold → minimal checks → one end
  sweep → review+fix).

## Dependencies
Track 02 (`ApiEndpoint`), `bridge.rs` / track 00 (`pyo3` feature), track 11
(executable gate — note: 11 and 10 co-depend; build the gate primitive in 11 and
the harness wraps it, or lift the gate into a shared util — resolve at impl as a
small `evolve/gate.rs` both share). Foundation of the self-evolve lane;
consumed by 11, 12, 15.
