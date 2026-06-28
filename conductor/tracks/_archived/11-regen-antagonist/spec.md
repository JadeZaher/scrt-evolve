---
type: Track Spec
title: "Regen-swap Antagonist & Topology Shift"
description: A self-distillation loop driven by the model's own training-refreshed generations.
tags: [track-11, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Regen-swap Antagonist & Topology Shift — Specification

## Goal
Add a **self-distillation loop** in which the model's own training-refreshed
checkpoint is hot-swapped back into generation as a **divergent "antagonist"**
generator, and a **depth-first cheapness objective** trains the forward path to
reach the same answer at an earlier layer (the **topology shift** = lower
inference depth/resources, learned at training time — NOT a reverse-inference
trick). CLI tool-calling is the primary target; reasoning "intuition" is the
secondary kind. Provides empirical measurement of whether the cheaper path
forms, instead of asserting it.

> Framing correction (research due-diligence, see
> `.omc/research/larql-regen-swap-2026-06-17.md`): the LARQL "reverse the
> forward pass / KNN-walk FFN for speed" premise does NOT hold — the WALK path
> is ~46x *slower* than GPU decode (517ms@1.9tok/s vs 11.4ms@88tok/s) and the
> "all paths reachable, pick the best statically" search is the Logit Lens
> problem (static FFN edges ignore the live, attention-shaped residual stream).
> Path enumeration does not scale and cannot see context. The salvaged,
> buildable idea: **do not search for the cheap path — define a loss that
> prefers cheapness and let gradient descent (which respects context) find it.**
> Recursion is temporal (regen swap each round), not spatial (graph walk).

> **Cogency-audit notes (apply throughout this spec).**
> - **Config host (A1).** `[generate.regen]` is a new field on the EXISTING
>   `GenerateConfig` struct in `config.rs` (`pub regen: Option<RegenConfig>`),
>   NOT a new top-level section. `GenerateConfig` already uses `#[serde(default)]`
>   + per-field default fns, so the addition is non-breaking.
> - **Mutable-weight prerequisite (A2).** `Arc<RwLock<LoadedModel>>` + `refresh()`
>   require a MUTABLE weight container that `model.rs`/`LoadedModel` does not yet
>   provide (it is an inert placeholder). The depth-cheapen training that
>   produces the updated weights is implemented via the **PyO3→`transformers`
>   bridge** (see directive note below), so `refresh()` swaps the handle to the
>   model the Python side just updated — candle is not required for this track.
> - **Executable gate source.** The executable gate is OWNED by track 10
>   (`evolve/gate.rs`); this track CONSUMES it, not redefines it.
> - **Provenance (B5).** Every antagonist row stamps the existing
>   `GenExample.gen` field (e.g. `regen:swap<N>`) so track 15 quarantine can
>   trace a bad row to the swap that produced it.
> - **Directive — PyO3→transformers.** The depth-first early-exit training
>   (early-exit head + `CE + λ·exit_depth`) is a heavy, candle-thin workflow;
>   per the user directive it is implemented by driving HF `transformers`/`peft`
>   through `bridge.rs`, with the candle path as an optional later alternative.
>   Eval/measurement uses track 10's `Scorer` (not a bespoke evaluator).

## Scope
- **`generate/regen.rs` — `RegenAntagonist` GenBackend** (DESIGN.md §trait #2,
  third impl beside `ApiEndpoint`/`LocalCandle`). Holds the in-training model
  behind a shared handle (`Arc<RwLock<LoadedModel>>`); a `refresh()` re-points
  generation at the latest weights with no disk reload. Divergence knobs:
  high `temperature` / wide `top_p`, optional inference-time activation dropout,
  and **prompt perturbation** (pair two unrelated palace stashes → ask for a CLI
  workflow bridging them) — the source of novel `scrt mp-*` command chains.
- **Executable acceptance gate** (the convergence force on the recursion). Every
  antagonist sample is validated before entering the dataset:
  - `schema` gate: validate against real scrt tool schemas (`toolspec.rs`).
  - `execute` gate: parse / dry-run the emitted `scrt …` command; reject if it
    does not parse or args do not resolve. A novel-but-invalid command is
    DISCARDED; novel-and-valid becomes high-value signal the teacher wouldn't
    produce.
- **`[generate.regen]` config block**: `enabled`, `swap_every_steps`,
  `temperature`, `top_p`, `antagonist_ratio`, `ratio_decay` (explore→exploit
  anneal), `gate = schema|execute`, `dropout`. Mixed with the grounded teacher
  per `antagonist_ratio`.
- **Self-distilled targets + grounding nodes** (the GraphRAG seam). For each
  accepted example, record: the **full-depth output** (soft label = the "best
  path" target), the **FFN features/layers it used** (grounding nodes), and an
  **early-exit attempt**. Activated features + the seeding palace stashes become
  adjacency nodes; the antagonist generates examples bridging neighbors. Grounding
  = neighbors must be the nodes the accepted path actually used.
- **Depth-first cheapness training** (extends the `TrainingPreset` trait from
  track 04). New `regen` preset with a two-term loss:
  `(a)` CE against the full-depth self-distilled target (correctness anchor) +
  `(b)` λ·exit_depth via a self-distilled **early-exit head** (the topology-shift
  term — same answer, fewer layers). Sparsity (λ·‖active_FFN‖) is a **stubbed
  second term, OFF by default**, wired but not implemented in this track
  (depth-first, per decision).
- **Measurement**: per-swap metrics — correctness, mean exit depth, and (when
  `--features larql`) FFN-vs-attention attribution on the tool-call token via
  LARQL `TRACE … FOR <tool>`. Rising FFN share + stable correctness at lower
  exit depth = the topology shift is real, with evidence. Emitted to
  `work_dir/regen-metrics.jsonl`.
- **LARQL sidecar (OPTIONAL, `--features larql`, never on the hot path):**
  `ROUTE VERIFY` as a second-stage hallucination guard on tool *arguments*;
  `INSERT INTO EDGES` to seed the vindex with the CLI command graph so novel
  chains stay near real command adjacencies; `TRACE … FOR` for the measurement
  above. Interpretability-as-signal, not a runtime.
- CLI: `scrt-evolve run --regen` (drive the full self-distill loop) and
  `scrt-evolve generate --backend regen` (override). Standalone-runnable.

## Constraints
- Behind `--features train` (candle); the LARQL sidecar behind a further,
  independent `--features larql`. ML-free `cargo build` stays green: the seam,
  config, trait wiring, and gate plumbing compile without candle; candle bodies
  are gated.
- **The cheapness objective is a LOSS, not a search.** No per-token path
  enumeration / BFS over FFN edges anywhere — that is explicitly out of scope
  and rejected (does not scale, ignores context).
- **Anti-collapse is mandatory, not optional** (DESIGN.md echo-chamber risk).
  The loop MUST NOT train on antagonist output that fails the gate. The teacher
  (API or frozen-original) remains the base-case anchor; `antagonist_ratio`
  decays. Self-training on un-gated self-output is a defect, not a feature.
- Reuses track 03 `model.rs` loader and track 04 `TrainingPreset` trait +
  training loop; does NOT add a new arch. ONE arch first.
- Validation is **mechanical, not quality-chasing** (matches track 04): shapes
  correct, gate rejects invalid commands, early-exit head trains (loss down),
  exit-depth metric is produced and monotone on a seeded overfit fixture. The
  "does it make a usefully smarter/faster model" quality experiment is OUT of
  scope (carried as the project-wide unproven-premise risk).
- LARQL is a hard optional: all `larql`-gated code must be removable without
  affecting the core loop; absence of a vindex degrades to schema/execute gating
  + depth-only metrics, never an error.

## Acceptance
- `RegenAntagonist` implements `GenBackend`; `refresh()` re-points generation at
  updated weights without disk reload (asserted: post-refresh generation reflects
  a mutated weight). -- ML seam behind `--features train`.
- Executable gate rejects a malformed `scrt …` command and accepts a valid one
  (fixture-driven, both `schema` and `execute` modes).
- `[generate.regen]` config round-trips (load + validate + defaults); ML-free
  build with the block present stays green.
- Depth-first `regen` preset: on a tiny seeded fixture + tiny model, the
  early-exit head trains (loss down) and **mean exit depth decreases across
  swaps while correctness on a held-out tiny set does not regress** (the core
  topology-shift smoke). Sparsity term present but inert (asserted no-op).
- `regen-metrics.jsonl` is emitted with per-swap {correctness, mean_exit_depth}
  rows; under `--features larql`, rows additionally carry FFN/attn attribution.
- Anti-collapse guard: a forced gate-failing antagonist sample never appears in
  the produced dataset (asserted).
- `scrt-evolve generate --backend regen` and `scrt-evolve run --regen` run
  standalone on a fixture.
- `--features larql` compiles and is fully removable: `cargo build` and
  `cargo build --features train` (no `larql`) both green; the loop runs with
  schema/execute gating and depth-only metrics when no vindex is present.
- **Styleguide gates** (code-styleguides.md): antagonist rows stamp
  `gen=regen:swap<N>` (§2.4 provenance); the swap loop is budget-bounded (§2.5);
  generation is seeded/deterministic for the smoke test (§2.2); gate-failing
  output never trains (§2.4 quarantine-traceable, §2.1 effects gated). Built per §4.

## Dependencies
Track 03 (`model.rs` loader, `LocalCandle` patterns, `GenBackend` reuse),
track 04 (`TrainingPreset` trait + the model update path `refresh()` swaps from),
track 10 (executable `gate.rs` + `Scorer`/`ScoreReport` for measurement — this
track does NOT define its own evaluator). The depth-cheapen training depends on
the **PyO3→`transformers` bridge** (`bridge.rs`, `--features pyo3`) per the user
directive; candle is an optional later path. Introduces the optional, isolated
`larql` feature. Sequenced after 10 (eval) and 04; parallel-independent of
05/06/07.
