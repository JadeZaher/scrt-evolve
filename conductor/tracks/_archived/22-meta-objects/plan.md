---
type: Implementation Plan
title: Meta Objects — Config-Driven Evolution Substrate
description: Implementation plan for the Meta Objects — Config-Driven Evolution Substrate track.
tags: [track-22, in-progress]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Meta Objects — Config-Driven Evolution Substrate — Plan

Build in slices. Slices 1–7 are buildable NOW on shipped tracks (01 discover +
palace-search, 02 generate + `Dataset`, 20 goals + transcript harvest). Slices
8–9 are lane-gated: the per-kind eval hook (track 10 metric registry) and
meta-gated rounds (track 15 transaction wrapper). This track **refactors**
goals (20) and taste (21) onto one `MetaModule`/`[[meta]]` substrate — it is the
foundation 20/21/12 become instances of, not a fourth family.

Hierarchy preserved: `apply_scope = lateral` (constitution = values that drive,
taste = how ideas are represented — standing substrate) shapes
`apply_scope = scoped` (goals = point-in-time desired outcomes that inherit the
substrate). Precedence on conflict: base-constitution > taste > goal.

## Tasks

1. [ ] `MetaObject` config + `[[meta]]` schema: additive `Vec<MetaObject>` on
   `EvolveConfig` (kind/name/tier/apply_scope/data_sources + kind payload),
   exported from `lib.rs`. Unknown `kind` without a registered impl ⇒ clear load
   error. -- evidence: round-trip test (`[[meta]]` parses; absent ⇒ empty = today's behavior); unknown-kind load-error test.
2. [ ] Legacy aliases (behavior-preserving): `[[goals]]` and `[[taste]]` parse
   via serde aliases and desugar to `MetaObject`. -- evidence: an existing `[[goals]]`+`[[taste]]` config parses to `MetaObject`s and the shipped track-20 config fixtures still assert identically.
3. [ ] `MetaModule` trait + registry (`meta/mod.rs`, `meta/registry.rs`):
   `kind`/`name`/`apply_scope`/`data_sources`/`compile`/`provenance`/`eval_hook`;
   `kind → impl` registry; the loop iterates `Box<dyn MetaModule>` with NO
   `kind` branch. -- evidence: a custom-kind impl registered in a test resolves + runs end-to-end through the generic loop (no-kind-branch asserted).
4. [ ] Data-source URI layer (`meta/source.rs`): pure `parse_source(&str)`
   mapping `palace:`/`project:`/`transcript:`/`corpus:`/`url:` to existing seams;
   unknown scheme errors. -- evidence: per-scheme parse test against fixtures; unknown-scheme error test; each scheme resolves to its track 01/02/20 seam.
5. [ ] Seed impls — refactor goals: `kind="goal"` impl; `GoalConfig` becomes a
   view/adapter; `EvolveConfig::for_goal` re-expressed as scoped-meta resolution.
   `goals::run_buildable` becomes the generic scoped pass (artifacts under
   `work_dir/meta/goal/<name>/`). -- evidence: every shipped track-20 test (`config.rs`/`discover.rs`/`goals.rs`/`harvest.rs`) stays green or migrates with identical assertions.
6. [ ] Seed impls — taste: `kind="taste"` impl; track-21 `compile_rubric` is the
   taste impl's `compile`. Coordinate so track 21's config+compile land AS this
   impl. -- evidence: taste rubric compiles through the meta path; `taste:<name>` provenance stamped; rows round-trip the `Dataset` contract.
7. [ ] Lateral-over-scoped composition: every lateral object's artifact shapes
   every scoped object's generate pass (the track-21 overlay, generalized);
   precedence base-constitution > taste > goal asserted; provenance composes
   (`goal:x+taste:y`). -- evidence: lateral overlay reaches every scoped generate prompt; scoped-cannot-weaken-lateral-base test; composed-`gen` parses to all stamps.
8. [ ] Eval hook (LANE-GATED, track 10): each kind's `eval_hook()` registers its
   `MetricSpec` into `[eval].metrics` when track 10's registry lands; absent
   registry ⇒ `None`, generation still shaped (graceful degrade). -- evidence: mock-registry test (taste/constitution hooks register their metric); absent-registry degrades to shaping-only.
9. [ ] Meta-gated rounds (LANE-GATED, track 15): meta-driven weight changes go
   THROUGH the track-15 txn; rows quarantinable by `<kind>:<name>`. -- evidence: forced regression rolls back under the txn (state restored); quarantine targets a meta object's provenance stamp.
10. [ ] CLI + docs: `scrt-evolve meta list` (resolved objects: kind/scope/
    sources/provenance); legacy `--goals` still works as sugar; README
    "config-driven evolution" section (declare directions + local data sources in
    `[[meta]]`). -- evidence: `meta list` prints resolved objects; legacy flag works; docs show a user instantiating their own evolution.
11. [ ] Final sweep: `cargo test` (default, ML-free) — INCLUDING the full shipped
    track-20 suite green post-refactor — `cargo clippy --all-targets -D
    warnings`, `cargo fmt --check`. -- evidence: green; default build stays ML-free + Python-free; no track-20 regression.

## Build order note
Slices 1–7 are buildable NOW (01/02/20 shipped). Slice 5 is the load-bearing
refactor — gate it on the full track-20 suite staying green (behavior-preserving
constraint). Slices 8–9 require the lane (track 10 metric registry, track 15
txn). Until then, meta objects shape generation deterministically; only the
score + gate are deferred (graceful degrade, logged).

## Coordinate-with (sequencing — avoid double work)
- **Track 21 (taste).** Build its config (slice 1) + `compile_rubric` (slice 2)
  AS the `kind="taste"` `MetaModule` impl, not standalone-then-refactored. If 21
  already shipped standalone, its onboarding here is a bounded adapter + test
  migration (slice 6).
- **Track 12 (constitution).** Its loader lands as the `kind="constitution"`
  impl on the same trait; base-constitution holds top precedence.
- **Track 20 (goals).** The family being refactored (slice 5); the
  behavior-preserving constraint is the contract.

## Carry-forward (deferred, lane-gated)
Slices **8–9** require tracks not yet wired:
- **Track 10** — eval metric registry: hosts each kind's `eval_hook()` metric
  (`taste_adherence`, `constitution_adherence`, goal `probe_set`).
- **Track 15** — transaction wrapper + quarantine: the ONLY sanctioned
  weight-mutation path; meta-gated rounds go THROUGH it, quarantinable by the
  unified `<kind>:<name>` provenance stamp.

## Sign-off
(Pending — fill in per slice as acceptance is met, mirroring tracks 01/04/20.)
