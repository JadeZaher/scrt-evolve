# Taste — Cross-Goal Metacognitive Style Distillation — Specification

## Goal
Give scrt-evolve a second, **orthogonal** axis to goals. Where a `[[goals]]`
entry says *what* a local model should evolve toward (track 20), a `[[taste]]`
module says **how ideas are represented** — the *form* a thought takes — applied
**laterally across every goal** during generation, not scoped to one.

**Taste vs constitution — the load-bearing distinction.** A constitution
(track 12) encodes the **values you are driven by while processing information**:
what is correct, safe, humble — the *drivers* of cognition. Taste encodes the
**representation of ideas**: the form, shape, and texture a thought takes once
those values have done their work. Constitution = *what you optimize for*;
taste = *how the resulting idea is shaped and rendered*. They are siblings on
the same structural pattern (base tier + mined overlay, tiered, versioned — §2),
but they govern different things: one drives, one represents.

Concretely, taste covers axes like **metacognition** (how a self-check is
*represented* — surfaced inline, deferred, or implicit), **context management**
(how much of the working state a representation carries vs. elides), and
**reasoning-step shape** (chain length, granularity, where a step commits vs.
keeps options open). These are not goal-specific and not values — they are the
*lateral form* every idea inherits, whether the model is doing CLI fluency, Rust
idioms, or security review. So they are distilled **across the whole goal set at
generation time**, forming a `goals × taste` matrix: every goal's rows inherit
every taste module's representational style.

The taste modules compile into a single **global taste rubric** (a rubric for
*how ideas are represented*, structurally parallel to track 12's constitution
for *what drives processing*). The rubric is injected into every goal's generate
prompt and — when the lane lands — scored as a **taste-adherence** eval metric
that gates rounds.

> **Cogency-audit notes (apply throughout).**
> - **The three layers are a hierarchy, not three peers.** Constitution and
>   taste are the **standing substrate** (durable, slow-changing): the values
>   that *drive* processing and the form ideas are *represented* in. Goals are a
>   **point-in-time overlay**: transient *desired outcomes* that are naturally
>   aligned to — i.e. inherit, and are shaped by — the standing constitution +
>   taste. A goal does not pick its own values or form; it evolves toward an
>   outcome *within* the substrate. So taste applies laterally across every goal
>   precisely because goals sit downstream of it.
> - **Substrate, not a goal.** Taste is NOT another `[[goals]]` entry. Goals are
>   filtered per-tag and run independent, transient pipelines; taste is standing
>   and applies to ALL of them. A goal selects *which traces seed* and *what
>   outcome to reach*; a taste module shapes *how every resulting idea is
>   represented*. Keep the two structs separate; taste is not parameterized by
>   goal.
> - **Constitution, not THE constitution.** Track 12's `constitution.toml`
>   encodes inviolable correctness/safety + user-preference overlays — *what is
>   right*. Taste encodes *how the thinking should feel*. They are sibling
>   rubrics; taste MUST NOT be able to weaken a base safety/correctness
>   principle (if both land, base-constitution wins on conflict, logged). Taste
>   is advisory-style, not a correctness gate.
> - **Config host.** `[[taste]]` is an additive `Vec<TasteModule>` on the
>   EXISTING `EvolveConfig` (serde `#[serde(default, skip_serializing_if =
>   "Vec::is_empty")]`), mirroring `goals: Vec<GoalConfig>`. Absent ⇒ today's
>   behavior exactly. Additive + non-breaking (styleguide §1).
> - **Provenance.** Rows shaped by the taste overlay stamp `gen` with a
>   `taste:<module>` marker (composed with any existing `gen` like
>   `trace:<goal>`) so a bad taste round is quarantinable by provenance
>   (track 15, styleguide §2.4).
> - **Buildable-now vs lane-gated.** The config, rubric compile, palace mining,
>   and prompt-injection overlay are buildable on shipped tracks (01 discover +
>   palace-search, 02 generate). The taste-adherence EVAL METRIC + taste-gated
>   keep|rollback are lane-gated (tracks 10 eval, 15 regulate) — designed here,
>   wired when the lane lands. This mirrors track 20's proven posture.

## The lateral matrix (one generation pass)
```
   STANDING SUBSTRATE (durable, slow-changing)
   ┌──────────────────────────────────────────────────────────────┐
   │  constitution.toml   values that DRIVE processing  (track 12) │
   │  [[taste]] modules   form ideas are REPRESENTED in (THIS track)│
   └───────────────────────────────┬──────────────────────────────┘
                                    │ goals inherit / align to substrate
   POINT-IN-TIME OVERLAY           ▼
   [[goals]]   desired outcomes, transient     (track 20: per-tag, independent)
       │
       ▼  discover (per goal, palace_tags=<goal.tag>)            ← built (01)
       │
   [[taste]]   compile ONCE → applied across ALL goals laterally
       │
       ▼
   taste-rubric.md  (one global rubric: metacognition / context / reasoning-shape)
       │ inject into every goal's generate prompt
       ▼  generate (rows now reason in the house style)          ← built (02) + overlay
       │
   rows stamped gen=…+taste:<module>
       │
       ▼  CHECKPOINT → train → EVAL(correctness + TASTE-ADHERENCE) → keep|rollback
                                              ▲ lane-gated (10 metric, 15 gate)
```
For G goals and T taste modules the rubric is built ONCE (the modules are
goal-agnostic, standing substrate) and reused across all G generate passes — it
is a `G × T` *effect* matrix, not a `G × T` *compile* matrix. No per-goal taste
recompile. The substrate (constitution + taste) is the slow axis; goals are the
fast axis that inherits it.

## Scope

### 1. Taste config (`[[taste]]` in `evolve.toml`)
Each module declares one cross-cutting opinion axis:
- `name` — stable id (e.g. `metacognition`, `context-discipline`,
  `reasoning-shape`). Namespaces the `taste:<name>` provenance stamp.
- `axis` — which dimension it governs: `metacognition` | `context` |
  `reasoning` (open string; unknown axes are accepted + logged, applied as
  generic style). Lets the rubric group related opinions.
- `principles` — `Vec<String>`: the authored opinions, each one imperative and
  testable in spirit (e.g. "Before committing to a tool call, state the one
  fact that would change the decision."). This is the base tier.
- `mine` — optional `bool` (default false): when true, ADDITIONALLY mine
  opinions from palace stashes tagged `taste:<name>` (see §3), appended as a
  subordinate, confidence-tagged tier.
- `weight` — optional `f32` scheduler/eval hint (how strongly this axis is
  emphasized in the rubric + scored). `None` ⇒ equal weight.
Backwards compatible: no `[[taste]]` ⇒ today's single-axis behavior. Additive
`Vec` field (styleguide §1).

### 2. Rubric compile (`taste/rubric.rs`)
A pure function `compile_rubric(&[TasteModule], mined: &MinedTaste) ->
TasteRubric` that folds the modules (base principles + any mined, subordinate
tier) into one ordered, axis-grouped markdown rubric + a machine-readable
`Vec<TastePrinciple>` (for the eval metric). Deterministic, no I/O. The rubric
is the durable artifact written to `work_dir/taste-rubric.md` for inspection.
Tiering invariant (mirrors track 12): a mined opinion may NOT contradict a base
principle in the same module; conflicts resolve to base, logged.

### 3. Palace-mined taste (the `mine = true` tier)
Reuses the EXISTING palace-search seam (track 01). For a module with
`mine = true`, sweep stashes tagged `taste:<name>` (via the same
`palace_tags`/`palace_search` plumbing discover already uses), distill each into
a candidate opinion, dedup, and append as the subordinate tier stamped with
provenance + a confidence score. Mining is best-effort: an empty/absent palace
yields zero mined opinions and the base tier still compiles (graceful degrade).
No mining ⇒ fully deterministic, hand-authored rubric.

### 4. Generation overlay (buildable now)
A generation pass that injects the compiled rubric into every goal's generate
prompt so the synthesized rows reason in the house style. This is a **style
overlay over the existing generate backend** (track 02), NOT a new backend:
discover → (inject rubric) → generate → rows. Each shaped row stamps
`gen = "<existing>+taste:<module>"` (composes with `trace:<goal>` from track 20).
The overlay is the lateral-matrix mechanism: one rubric, applied across every
goal's pass.

### 5. Taste-adherence eval metric (lane-gated, track 10)
A new metric for track 10's `Scorer`: given a model output and the
`Vec<TastePrinciple>`, score how well the output's *reasoning style* follows the
rubric (judge-backed, like the constitution-adherence metric). Added to the
`[eval].metrics` menu as `taste_adherence`. Designed here, registered when track
10's metric registry exists; until then the rubric still shapes generation, the
score is just not computed (logged, graceful degrade).

### 6. Taste-gated rounds (lane-gated, track 15)
When the lane is present, a round's verdict considers taste-adherence alongside
correctness: a round that tanks taste while holding correctness can be rolled
back (configurable tolerance, like `accept_tolerance`). Catastrophe is still
correctness-defined (taste never overrides safety). No round mutates weights
outside the track-15 transaction; taste rows are quarantinable by their
`taste:<module>` provenance stamp.

## Constraints
- **Orthogonal, not a fork of goals.** Taste is a separate `Vec<TasteModule>`,
  applied across ALL goals; it does not reuse `GoalConfig` and does not run a
  per-tag pipeline. One rubric, many goals.
- **Advisory, never a safety gate.** Taste shapes *how* the model reasons; it
  can NEVER weaken a base correctness/safety principle (track 12). On conflict,
  base-constitution wins, logged. Taste-adherence gating (§6) is a *quality*
  tolerance, not a catastrophe trigger — catastrophe stays correctness-defined.
- **Builds ON, does not reinvent:** palace-search/discover (01, built), generate
  + `Dataset` (02, built), the eval `Scorer`/metric registry (10), the
  transactional keep|rollback + quarantine wrapper (15). This track is the
  config + rubric-compile + mining + generation-overlay layer, plus a designed
  (lane-gated) eval metric. No new ML, no new model logic.
- **Additive / non-breaking:** absent `[[taste]]` = today's behavior exactly.
  Default `cargo build` stays ML-free + Python-free; the overlay works with the
  API generate backend and needs no candle.
- **Bounded everything:** rubric compile folds a bounded `Vec`; mining is
  bounded by the palace sweep cap; the overlay adds one prompt section per
  goal pass — no unbounded loops (styleguide §2.5).
- **Provenance always stamped.** Every taste-shaped row carries a
  `taste:<module>` marker so a regressive taste round is quarantinable
  (styleguide §2.4).

## Acceptance
- `evolve.toml` parses `[[taste]]` (name/axis/principles/mine/weight); absent
  taste reproduces today's behavior; a round-trip test confirms parse + empty
  default. (Buildable now.)
- `compile_rubric` folds N modules into one deterministic, axis-grouped rubric +
  a `Vec<TastePrinciple>`; same input ⇒ byte-identical rubric (determinism
  asserted). A fixture where a mined opinion contradicts a base principle
  resolves to base and logs the conflict (tiering invariant asserted).
  (Buildable now.)
- Palace mining: a fixture palace with stashes tagged `taste:metacognition`
  yields mined opinions appended as the subordinate tier with confidence +
  provenance; an empty palace yields zero mined opinions and the base tier still
  compiles (graceful degrade asserted). (Buildable now.)
- Generation overlay: the compiled rubric is injected into the generate prompt
  for every goal; rows produced under the overlay stamp `gen` with a
  `taste:<module>` marker that composes with any existing `gen`; rows still
  round-trip through the `Dataset` JSONL contract. (Buildable now, mockable
  backend — runs without a live model.)
- Taste-adherence eval metric: designed + unit-tested against fixture outputs
  with a mock judge (high-adherence output scores above low-adherence);
  registered into track 10's `[eval].metrics` when the registry lands.
  (Lane-gated — track 10.)
- Taste-gated round: a forced taste regression (correctness held) rolls back the
  round under the track-15 transaction; a taste win is kept; catastrophe stays
  correctness-defined. (Lane-gated — track 15.)
- **Styleguide gates** (code-styleguides.md): taste-shaped rows stamp
  `gen=…+taste:*` (§2.4 provenance); rubric compile + mining are budget-bounded
  (§2.5); taste never weakens a base safety/correctness principle (§2.1 effects
  gated — asserted). Built per §4.

## Dependencies
- **Built/usable now:** 01 (discover + palace-search — the mining seam), 02
  (generate + `Dataset` — the overlay host).
- **Lane (must exist for full acceptance):** 10 (eval `Scorer` + metric registry
  — the `taste_adherence` metric), 15 (transactional keep|rollback + quarantine
  — taste-gated rounds).
- **Sibling, coordinate (do not subsume):** 12 (constitutional self-refine — the
  *values that drive processing*). Taste is the *representation-of-ideas* sibling
  on the same base+mined tiered structure; base-constitution wins on conflict.
  Together they are the standing substrate. 20 (learning-by-doing goals — the
  *point-in-time desired-outcome* overlay that inherits the substrate; taste
  applies laterally across the goals this track's overlay shapes, and
  `taste:<module>` composes with `trace:<goal>`).

## Honest risks
- **"Style" is fuzzy and unproven.** Whether distilling metacognitive style
  measurably improves a local model is unproven; the taste-adherence gate
  (10/15) is the backstop — a taste round that doesn't hold the probe score is
  rolled back, so bad taste costs time, not correctness. The buildable core
  (rubric + overlay) is honest regardless: it shapes generation deterministically
  whether or not the gate exists yet.
- **Taste vs constitution overlap.** Risk of two rubrics with contradictory
  pulls. Mitigated by the explicit precedence (base-constitution > taste) and by
  keeping taste advisory (never a catastrophe trigger).
- **Mining noise.** Palace-mined opinions may be low-quality; mitigated by the
  subordinate tier + confidence stamp + the same gate. `mine = false` (default)
  is the fully-deterministic, hand-authored escape hatch.
- **Scope creep into "prompt engineering framework."** Kept tractable: a config
  `Vec`, a pure compile fn, a reused palace sweep, one generate-prompt section,
  and a designed eval metric. No new ML, no new backend, weight changes THROUGH
  track 15.
