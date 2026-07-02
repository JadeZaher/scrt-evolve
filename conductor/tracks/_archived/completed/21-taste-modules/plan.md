---
type: Implementation Plan
title: Taste — Cross-Goal Metacognitive Style Distillation
description: Implementation plan for the Taste — Cross-Goal Metacognitive Style Distillation track.
tags: [track-21, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Taste — Cross-Goal Metacognitive Style Distillation — Plan

Build in slices. Slices 1–5 are buildable NOW on shipped tracks (01 discover +
palace-search, 02 generate + `Dataset`) — config, rubric compile, palace mining,
the generation overlay, and provenance. Slices 6–7 are lane-gated: the
taste-adherence eval metric (track 10) and taste-gated rounds (track 15). This
track is the config + rubric + mining + overlay layer over those, plus a
designed (lane-gated) metric — no new ML.

Taste is the **standing-substrate sibling** of the constitution (track 12):
constitution = the values that drive processing; taste = how ideas are
represented. Goals (track 20) are the point-in-time overlay that inherits both.

## Tasks

1. [ ] `[[taste]]` config: `TasteModule { name, axis, principles: Vec<String>,
   mine?: bool, weight?: f32 }` as an additive `Vec` on `EvolveConfig`
   (`#[serde(default, skip_serializing_if = "Vec::is_empty")]`), exported from
   `lib.rs` beside `GoalConfig`. -- evidence: config round-trip test (taste parses; absent ⇒ empty = today's behavior).
2. [ ] Rubric compile (`taste/rubric.rs`): pure `compile_rubric(&[TasteModule],
   &MinedTaste) -> TasteRubric` folding base principles (+ subordinate mined
   tier) into one ordered, axis-grouped markdown rubric + a
   `Vec<TastePrinciple>`. Deterministic, no I/O; tiering invariant (mined may not
   contradict base; conflict → base, logged). Write `work_dir/taste-rubric.md`.
   -- evidence: determinism test (same input ⇒ byte-identical rubric); tiering-invariant test (overlay-contradicts-base resolves to base + logs).
3. [ ] Palace-mined tier (`mine = true`): reuse the discover palace-search seam
   (track 01) to sweep stashes tagged `taste:<name>`, distill → candidate
   opinions, dedup, append as the subordinate confidence+provenance-stamped tier.
   Best-effort: empty/absent palace ⇒ zero mined, base tier still compiles.
   -- evidence: fixture-palace mining test (tagged stashes → mined opinions with confidence); empty-palace graceful-degrade test.
4. [ ] Generation overlay: inject the compiled rubric into every goal's generate
   prompt (style overlay over the track-02 backend, NOT a new backend). Shaped
   rows stamp `gen = "<existing>+taste:<module>"` (composes with `trace:<goal>`).
   -- evidence: overlay test with a mockable backend — rubric reaches the prompt for every goal; rows stamp `taste:<module>`; rows round-trip the `Dataset` JSONL contract.
5. [ ] Provenance + lib surface: `taste` module wired into `lib.rs`; re-export
   `TasteModule`, `TasteRubric`, `TastePrinciple`, `compile_rubric`. Confirm the
   `gen` stamp composition (`trace:<goal>+taste:<module>`) is quarantine-parseable.
   -- evidence: provenance-composition test (a trace+taste row's `gen` parses to both markers).
6. [ ] Taste-adherence eval metric (LANE-GATED, track 10): design + unit-test a
   `Scorer` metric that scores reasoning-representation adherence vs.
   `Vec<TastePrinciple>` (judge-backed, like constitution-adherence); register as
   `taste_adherence` in the `[eval].metrics` menu when track 10's registry lands.
   -- evidence: mock-judge test (high-adherence output scores above low-adherence); registered into `[eval].metrics`.
7. [ ] Taste-gated rounds (LANE-GATED, track 15): a round's verdict considers
   taste-adherence beside correctness (configurable tolerance); a taste win is
   kept, a taste regression (correctness held) rolls back; catastrophe stays
   correctness-defined; taste rows quarantinable by `taste:<module>`.
   -- evidence: forced taste-regression rolls back under the track-15 txn (state restored); taste win kept; catastrophe still correctness-only.
8. [ ] Docs: README "taste" subsection (substrate sibling of constitution; goals
   inherit it); DESIGN amendment tying the constitution/taste/goals hierarchy
   together. -- evidence: docs updated; the three-layer hierarchy documented.
9. [ ] Final sweep: `cargo test` (default, ML-free), `cargo clippy --all-targets
   -D warnings`, `cargo fmt --check`. -- evidence: green; default build stays ML-free + Python-free.

## Build order note
Slices 1–5 are buildable NOW on shipped tracks (01 palace-search seam for
mining, 02 generate for the overlay). Slices 6–7 require the lane (track 10 eval
metric registry, track 15 transaction wrapper) to land first — this track is
their consumer/designer for the taste axis, not their owner. Until then, the
rubric still shapes generation deterministically; only the score + gate are
deferred (graceful degrade, logged).

## Carry-forward (deferred, lane-gated)
Slices **6–7** require tracks not yet wired for this axis:
- **Track 10** — eval `Scorer` + metric registry: hosts the `taste_adherence`
  metric. The `Vec<TastePrinciple>` from slice 2 is its reserved input.
- **Track 15** — transactional keep|rollback + quarantine: the ONLY sanctioned
  weight-mutation path; taste-gated rounds go THROUGH it. Taste rows are
  quarantinable by the `taste:<module>` provenance stamp from slices 4–5.

When the lane lands, the slice-2 `Vec<TastePrinciple>` + the slice-4/5
provenance stamps are the extension points: register the metric (6) and fold
taste-adherence into the round verdict tolerance (7).

## Coordinate-with (sibling tracks — do not subsume)
- **Track 12 (constitution).** Same base+mined tiered structure; different
  governed thing (drivers vs. representation). If both land, enforce
  base-constitution > taste on conflict (logged); taste never triggers
  catastrophe.
- **Track 20 (goals).** Goals are the point-in-time overlay that inherits this
  substrate. `taste:<module>` composes with `trace:<goal>` in the `gen` stamp.
  The overlay (slice 4) runs across every goal's generate pass.

## Sign-off
(Pending — fill in per slice as acceptance is met, mirroring tracks 01/04/20.)
