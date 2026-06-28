---
type: Track Spec
title: Self-Refinement (Constitutional, Dialectic)
description: A generation stage producing Constitutional-AI-grade dialectic training data.
tags: [track-12, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Self-Refinement (Constitutional, Dialectic) — Specification

## Goal
A new **generation stage** that produces Constitutional-AI-grade training data
with **no human labeling**, by having the model critique and revise its own
drafts through a **sequential dialectic** — thesis → metacognitive check →
Jungian shadow antithesis → synthesis — judged against an explicit
**constitution**. This is the stronger-than-parse outcome signal the system
evolves on: the constitution encodes the user's goals/objectives, and the model
self-refines toward them. The synthesis is the SFT target (depth-cheapened by
track 11); thesis-vs-synthesis becomes a DPO preference pair (follow-on preset).

Purpose context: this is the data engine for the product goal — **locally tune
ONE model that evolves with a user's goals across ALL their projects** (merged,
on-demand). See the DESIGN.md scope expansion this track is paired with.

> **Cogency-audit notes (apply throughout).**
> - **Config host (A1).** `[generate.refine]` is a new `Option<RefineConfig>`
>   field on the EXISTING `GenerateConfig` struct (serde-default → non-breaking),
>   NOT a new top-level section.
> - **Merged corpus (B4).** "Across all projects, merged" requires discovery to
>   union multiple project dirs. That needs `corpus_dirs: Vec<PathBuf>` (or
>   `projects`) on `EvolveSection` (today: single `corpus_dir`) — added in the
>   track 01 revision referenced here; this track CONSUMES the unioned
>   `DiscoveredContext`, it does not re-implement the union.
> - **max_revisions cap (C8).** `max_revisions` DEFAULT = 1; >1 is EXPERIMENTAL,
>   off by default (verbose multi-round critique fights the depth-cheapen goal).
> - **Provenance (B5).** `refined`/`preference` rows stamp `GenExample.gen`
>   (`refine:synthesis` / `refine:pref`) for quarantine traceability (track 15).
> - **Directive — PyO3→transformers.** The DPO/preference TrainingPreset is
>   implemented via HF `trl`/`transformers` through `bridge.rs` (`--features
>   pyo3`), candle optional later. Constitution adherence scoring uses track 10's
>   `Scorer`, not a bespoke evaluator. References to "track 09" for depth-cheapen
>   throughout this spec mean **track 11** (regen) post-renumber.

## Scope
- **Constitution artifact** (`constitution.toml`, versioned, the durable thing
  the model evolves AGAINST). Two cleanly-separated tiers:
  - **base** — authored, NON-NEGOTIABLE principles (safety: confirm destructive
    ops; correctness: tool calls must parse & resolve; humility: flag
    uncertainty, don't fabricate). The induction in the next tier CANNOT
    weaken or override base principles.
  - **overlay** — user preferences, either user-authored or **mined** from the
    user's corpus/palace/history (idioms, minimalism, project conventions).
    Mined principles are tagged with provenance + confidence and are subordinate
    to base.
  - Loader validates the tiering invariant (no overlay principle may contradict
    a base principle; conflicts resolve to base, logged).
- **`generate/refine.rs` — the dialectic pipeline** (a generation strategy, not
  a new GenBackend; it composes existing backends). Per input draft:
  1. **THESIS** — produce a draft answer / tool-call (reuses the antagonist
     from track 09 or the teacher).
  2. **METACOGNITION** — the model critiques its OWN reasoning: is the chain
     sound, is confidence calibrated, which principles apply.
  3. **ANTITHESIS (Jungian shadow)** — surface what the draft AVOIDS or
     rationalizes: the blind spot, the over-confident leap, the rejected
     alternative (the "shadow"); name the projection.
  4. **SYNTHESIS** — a revised answer that resolves thesis↔antithesis and cites
     the principles it now satisfies.
  Each stage is a recorded turn with the principle(s) cited.
- **Two new dataset row kinds** (additive to the `#[serde(tag="kind")]` enum in
  `dataset.rs` — non-breaking):
  - `refined` — `{prompt, synthesis, principles[], source, gen}` — the SFT
    target (the deliberation is scaffolding, dropped from this row).
  - `preference` — `{prompt, chosen (synthesis), rejected (thesis),
    principles[], source, gen}` — the DPO pair (DIRECTION of improvement).
  Both emitted per refinement; SFT consumed now, preference by the follow-on.
- **`[generate.refine]` config**: `enabled`, `constitution`
  (path, default `constitution.toml`), `emit = ["refined","preference"]`,
  `max_revisions` (dialectic can iterate >1 round), `judge` (which backend
  scores principle-alignment: `teacher`|`self`), `mine_overlay` (bool).
- **DPO/preference training preset** (extends the track-04 `TrainingPreset`
  trait): consumes `preference` rows. Mechanical-only in this track (loss
  computes, preferred logprob margin increases on an overfit fixture). Full
  preference-tuning quality is a follow-on, gated.
- **Cross-project merged input**: discovery feeds this stage from the union of
  the user's projects (per the merged-model decision). The stage is
  corpus-agnostic — it refines whatever drafts discovery + antagonist supply.
- CLI: `scrt-evolve refine [--config]` (run the self-refinement stage standalone
  → dataset rows) and `scrt-evolve train --preset dpo [--data]`.

## Constraints
- **No human labeling, no usage-mining.** The signal is the model's own
  constitution-judged self-critique. (Usage/transcript mining was explicitly
  NOT chosen; do not add it.)
- **Base constitution is inviolable.** Code MUST enforce that mined/overlay
  principles cannot override base; safety/correctness are not erodible by
  induction. This is a correctness property, asserted in tests.
- **Anti-collapse still applies.** Synthesis that fails the track-09 executable
  gate (for tool_call/cli kinds) is rejected before it can become a `refined`
  row. Self-critique does not exempt output from the executable gate.
- **The deliberation is NOT the training target by default.** `refined` trains
  on synthesis only (keep the model fast — fights the depth-cheapen goal to
  train verbose think-aloud). Keeping the full chain is out of scope (rejected
  option).
- ML-free `cargo build` stays green: constitution loader, dialectic
  orchestration, row schema, and gate wiring compile without candle; generation
  uses existing backends (API works with no candle); DPO loop body behind
  `--features train`.
- Reuses: track 11 antagonist (thesis source), track 10 executable gate +
  `Scorer`, track 02 `ApiEndpoint` + `Dataset` writer, track 04 `TrainingPreset`
  trait (DPO via PyO3→`trl`). Adds no new arch.
- Validation is mechanical, not quality-chasing: pipeline produces all four
  stage turns; rows validate; base-overrides-overlay invariant holds; gate
  rejects bad synthesis; DPO margin increases on overfit fixture. The "does
  self-refinement make a usefully-better model" quality experiment is OUT of
  scope (project-wide unproven-premise risk).

## Acceptance
- `constitution.toml` loads with base+overlay tiers; a fixture where an overlay
  principle contradicts a base principle resolves to base and logs the conflict
  (asserted).
- The dialectic pipeline emits, for one input, four recorded stages (thesis,
  metacognition, antithesis, synthesis) each citing principle(s) (fixture,
  mockable backend — runs without a live model).
- `refined` and `preference` rows round-trip through the JSONL
  reader/writer; adding them does not break existing-row deserialization.
- A synthesis that fails the executable gate (malformed `scrt …`) does NOT
  produce a `refined` row (anti-collapse asserted).
- DPO preset: on a seeded overfit fixture, the chosen-vs-rejected logprob margin
  increases over steps (behind `--features train`).
- `scrt-evolve refine` runs standalone producing rows; `scrt-evolve train
  --preset dpo --data …` runs standalone.
- ML-free build green with `[generate.refine]` + constitution present.
- **Styleguide gates** (code-styleguides.md): `refined`/`preference` rows stamp
  `gen=refine:*` (§2.4 provenance); the dialectic is budget-bounded
  (`max_revisions` default 1, §2.5); synthesis passes the executable gate before
  it trains (§2.1 effects gated, §2.4 quarantine-traceable). Built per §4.

## Dependencies
Track 02 (`ApiEndpoint`, `Dataset`), track 04 (`TrainingPreset` trait — DPO
preset extends it, implemented via PyO3→`trl`), track 10 (executable gate the
synthesis must pass + `Scorer` for constitution adherence), track 11 (antagonist
as thesis source; its depth-cheapen consumes the `refined` rows), track 01
revision (merged `corpus_dirs`). Paired with the DESIGN.md scope expansion
(cross-project merged model; the self-refinement outcome signal). Sequenced
after 10 and 11.
