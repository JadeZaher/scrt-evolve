---
type: Track Spec
title: Meta Objects — Config-Driven Evolution Substrate
description: Collapse constitution/taste/goals into one config-driven meta-object substrate.
tags: [track-22, in-progress]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Meta Objects — Config-Driven Evolution Substrate — Specification

## Goal
Collapse the three bespoke evolution-driver families — **constitution** (values
that drive processing, track 12), **taste** (how ideas are represented, track
21), and **goals** (point-in-time desired outcomes, track 20) — into **one
config-driven abstraction**: the **meta object**. A meta object is a named,
tiered (base + mined), provenance-stamped module with declared **local data
sources** that plugs into the evolution loop. The three existing families become
*instances* of this one pattern, not three parallel code paths.

The product outcome: a user sets up their own **training directions** and **data
sources** locally — entirely in `evolve.toml` — and those `[[meta]]` declarations
*instantiate* the goal / taste / constitution-driven evolution steps. New
families (e.g. a `tone` driver, a `domain-lexicon` module, a `safety-overlay`)
become a new `kind` + a registered impl — no change to the loop. This is the
"AI as a config-driven, locally-owned asset" capstone: the evolution substrate
is declared, not hard-coded.

> **Cogency-audit notes (apply throughout).**
> - **Schema + trait, both.** A generic `[[meta]]` TOML table (declarative) is
>   resolved to a `Box<dyn MetaModule>` (behavioral). Config declares *what*;
>   the registered impl supplies *how* (compile, eval hook, apply-scope,
>   provenance). New `kind` ⇒ new impl + one registry line; the config shape and
>   the loop are unchanged. This is the open/closed seam.
> - **This is a REFACTOR, not an addition.** `[[goals]]` (track 20) and
>   `[[taste]]` (track 21) become **serde aliases that desugar to `[[meta]]`**
>   with a single underlying code path. `GoalConfig`/`TasteModule` become thin
>   views (or `From` adapters) over `MetaObject`. Blast radius is known: 6 source
>   files (`config.rs`, `goals.rs`, `harvest.rs`, `lib.rs`, `rounds.rs`,
>   CLI `main.rs`) + 4 test files reference goals; constitution lands via track
>   12 onto the same trait. The refactor is **behavior-preserving** — every
>   shipped track-20 test must stay green (or be migrated to the meta path with
>   identical assertions).
> - **Data sources are URIs.** `data_sources: Vec<String>` of scheme URIs:
>   `palace:<tag-or-search>`, `project:<path>`, `transcript:<glob>`,
>   `corpus:<dir>`, `url:<u>`. Each parses to an EXISTING discover/harvest seam
>   (track 01 palace-search, track 20 transcript harvest, track 02 corpus). The
>   URI layer is a thin parser over plumbing that already exists — it does not
>   reinvent retrieval.
> - **Substrate hierarchy preserved.** Meta objects carry an `apply_scope`:
>   `lateral` (constitution, taste — standing substrate applied across all
>   scoped objects) vs `scoped` (goals — point-in-time, per-tag, inherits the
>   lateral substrate). The abstraction must NOT flatten the hierarchy: lateral
>   objects shape scoped objects' generation, never the reverse. Precedence on
>   conflict: base-constitution > taste > goal (logged).
> - **Provenance unified.** Every meta object stamps `gen` with
>   `<kind>:<name>` (`goal:cli-fluency`, `taste:reasoning-shape`,
>   `constitution:base`); composition (`trace:<goal>+taste:<module>`) is just
>   multiple stamps joined. One quarantine grammar for all families (track 15,
>   styleguide §2.4).
> - **Additive at the config boundary, even though internally a refactor.** A
>   user's existing `evolve.toml` with `[[goals]]`/`[[taste]]` MUST still parse
>   and behave identically (the alias). `[[meta]]` is the new canonical form;
>   the old forms are sugar.

## The unified model
```
   evolve.toml
   ┌─────────────────────────────────────────────────────────────┐
   │ [[meta]] kind="constitution" apply_scope="lateral" tier=base │  ← values that DRIVE
   │ [[meta]] kind="taste"        apply_scope="lateral"           │  ← how ideas REPRESENTED
   │ [[meta]] kind="goal"         apply_scope="scoped"            │  ← point-in-time OUTCOMES
   │     data_sources = ["palace:goal-tag", "project:./app",      │
   │                     "transcript:traces/*.jsonl"]             │
   └───────────────────────────────┬─────────────────────────────┘
                                    │ load + validate
                                    ▼
   Vec<MetaObject>  ──resolve kind──▶  Box<dyn MetaModule>  (registry)
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
      data_sources()          compile(mined)          eval_hook()
      (URI → seam)         (base+mined → Artifact)   (Option<Metric>)
            │                       │                       │
            ▼                       ▼                       ▼
   discover/harvest         rubric / dataset rows     taste_adherence /
   (01/20/02 seams)         (gen=<kind>:<name>)       constitution / probe
                                    │
                                    ▼
   lateral objects shape EVERY scoped object's generate pass
   (goals inherit constitution + taste — substrate > overlay)
```

## Scope

### 1. `MetaObject` config + `[[meta]]` schema
One additive `Vec<MetaObject>` on `EvolveConfig`. Shared fields:
- `kind` — `goal` | `taste` | `constitution` | `<custom>` (open string; unknown
  kinds without a registered impl are a load error with a clear message).
- `name` — stable id; namespaces the `<kind>:<name>` provenance stamp.
- `tier` — `base` | `mined` (default `base`); mined is subordinate and
  confidence-stamped, may not contradict base (logged).
- `apply_scope` — `lateral` | `scoped` (default per kind: constitution/taste ⇒
  lateral, goal ⇒ scoped).
- `data_sources` — `Vec<String>` of scheme URIs (§3).
- `principles` / `topic` / `tag` / `probe_set` / `weight` / `cadence` — the
  kind-specific payload fields (serde-flattened or per-kind; absent when N/A).
Backwards compatible: existing `[[goals]]`/`[[taste]]` parse via serde aliases
and desugar to `MetaObject`. Absent `[[meta]]` + absent legacy ⇒ today's
single-run behavior.

### 2. `MetaModule` trait + registry (`meta/mod.rs`, `meta/registry.rs`)
```rust
trait MetaModule {
    fn kind(&self) -> &str;
    fn name(&self) -> &str;
    fn apply_scope(&self) -> ApplyScope;          // Lateral | Scoped
    fn data_sources(&self) -> &[DataSourceUri];
    fn compile(&self, mined: &MinedTier) -> MetaArtifact;  // rubric | rows | probe set
    fn provenance(&self) -> String;               // "<kind>:<name>"
    fn eval_hook(&self) -> Option<MetricSpec>;    // lane-gated metric, Option
}
```
A registry maps `kind → fn(&MetaObject) -> Box<dyn MetaModule>`. `goal`, `taste`,
`constitution` are the three seeded impls. Registration is one line per kind;
the loop iterates `Vec<Box<dyn MetaModule>>` generically. Pure where possible
(`compile` deterministic, no I/O — mirrors track 21 `compile_rubric` + track 12
constitution loader).

### 3. Data-source URI layer (`meta/source.rs`)
A pure `parse_source(&str) -> Result<DataSourceUri>` over scheme URIs, each
mapping to an existing seam:
- `palace:<tag|search>` → discover palace-search (`palace_tags`/`palace_search`,
  track 01).
- `project:<path>` → corpus scoped to a project dir (track 01/20 `project`).
- `transcript:<glob>` → the transcript harvester (track 20 `harvest`).
- `corpus:<dir>` → corpus sweep (track 02 `corpus_dir`/patterns).
- `url:<u>` → the capture-then-filter fetch seam (scrt rule; bounded).
Unknown scheme ⇒ load error. The URI layer adds NO new retrieval — it routes to
plumbing that already ships. Bounded (each seam carries its existing cap;
styleguide §2.5).

### 4. Refactor goals + taste onto the substrate (behavior-preserving)
- `GoalConfig` becomes a view/adapter over a `kind="goal"` `MetaObject`;
  `EvolveConfig::for_goal` is re-expressed as scoped-meta resolution. Every
  shipped track-20 test (`config.rs`, `discover.rs`, `goals.rs`, `harvest.rs`)
  stays green or is migrated to the meta path with identical assertions.
- `TasteModule` (track 21) becomes a `kind="taste"` impl; its `compile_rubric`
  is the `taste` impl's `compile`. (Coordinate with track 21 — ideally 21's
  config+compile slices are built AS the meta impl, not built then re-refactored.)
- `goals::run_buildable` becomes a generic meta-driven pass: iterate scoped
  objects, shaped by the lateral substrate, writing per-object artifacts under
  `work_dir/meta/<kind>/<name>/`.

### 5. Lateral-over-scoped composition (the substrate guarantee)
The loop applies every `lateral` object's compiled artifact (constitution
values + taste rubric) to every `scoped` object's generate pass — the existing
track-21 overlay, generalized. Precedence on conflict is asserted in code:
base-constitution > taste > goal. A scoped object can never weaken a lateral
base principle. Provenance composes (`goal:x + taste:y + constitution:base`).

### 6. CLI + docs surface
- `scrt-evolve evolve` iterates `[[meta]]` (legacy `--goals` flag still works,
  now sugar for "run the scoped meta objects").
- `scrt-evolve meta list` prints the resolved meta objects (kind, scope, sources,
  provenance) for inspection.
- Docs: a "config-driven evolution" section showing a user declaring directions
  + local data sources in `[[meta]]` to instantiate their own evolution.

## Constraints
- **Behavior-preserving refactor.** Existing `evolve.toml` files parse and
  behave identically; shipped track-20 tests stay green (or migrate with
  identical assertions). The user-facing config is additive even though the
  internals are unified.
- **One trait, many kinds; no special-casing in the loop.** The evolution loop
  must not branch on `kind` — it iterates `Box<dyn MetaModule>`. Kind-specific
  behavior lives behind the trait. Adding a family = a new impl + a registry
  line.
- **Substrate hierarchy is a correctness property.** Lateral shapes scoped,
  never the reverse; base-constitution > taste > goal on conflict — asserted in
  tests, not just documented.
- **URIs route to existing seams only.** No new retrieval mechanism; the URI
  layer is a parser over track 01/02/20 plumbing. Bounded by each seam's cap.
- **Builds ON, does not reinvent:** discover/palace-search (01), generate +
  `Dataset` (02), transcript harvest (20), and — when present — the eval metric
  registry (10) + transaction wrapper (15). No new ML.
- **Additive build posture.** Default `cargo build` stays ML-free + Python-free;
  the abstraction + URI layer + registry compile without candle.
- **Bounded everything.** Registry iteration over a bounded `Vec`; each
  data-source seam carries its existing cap; no unbounded loops (§2.5).

## Acceptance
- `evolve.toml` parses `[[meta]]` (kind/name/tier/apply_scope/data_sources +
  kind payload); a fixture with `kind` lacking a registered impl is a clear load
  error. Round-trip + empty-default test. (Buildable now.)
- Legacy alias: an existing `[[goals]]` + `[[taste]]` config parses, desugars to
  `MetaObject`s, and produces byte-identical downstream artifacts to the
  pre-refactor path (behavior-preserving — asserted against the shipped track-20
  fixtures). (Buildable now.)
- `MetaModule` registry resolves `goal`/`taste`/`constitution` kinds to impls;
  the loop iterates them generically with no `kind` branch (asserted by a
  custom-kind impl registered in a test working end-to-end). (Buildable now.)
- URI layer: `parse_source` maps each scheme to its seam; `palace:`/`project:`/
  `transcript:`/`corpus:` resolve against fixtures; an unknown scheme errors.
  (Buildable now.)
- Substrate guarantee: a lateral taste/constitution object shapes a scoped
  goal's generate pass (overlay reaches the prompt); a scoped object cannot
  weaken a lateral base principle (precedence asserted). Provenance composes to
  `goal:x+taste:y`. (Buildable now, mockable backend.)
- Eval hook: a kind's `eval_hook()` registers its metric into `[eval].metrics`
  when track 10's registry lands; absent registry ⇒ `None`, generation still
  shaped (graceful degrade). (Lane-gated — track 10.)
- Meta-gated round: weight changes go THROUGH track 15; a meta object's rows are
  quarantinable by `<kind>:<name>` provenance. (Lane-gated — track 15.)
- **Styleguide gates:** unified `gen=<kind>:<name>` provenance (§2.4); registry
  + sources bounded (§2.5); lateral-base-over-scoped precedence asserted (§2.1
  effects gated). Built per §4.

## Dependencies
- **Built/usable now:** 01 (discover + palace-search), 02 (generate + `Dataset`),
  20 (goals — the family being refactored + transcript harvest seam).
- **Refactors / subsumes:** 20 (`[[goals]]` → `kind="goal"`), 21 (`[[taste]]` →
  `kind="taste"`), 12 (constitution → `kind="constitution"`). Coordinate so 21
  and 12 land their config+compile AS meta impls rather than standalone then
  re-refactored.
- **Lane (for full acceptance):** 10 (eval metric registry — `eval_hook`), 15
  (transaction wrapper + quarantine — meta-gated rounds).

## Honest risks
- **Refactor regression.** Unifying three families risks breaking shipped
  track-20 behavior. Mitigated by the behavior-preserving constraint: legacy
  configs must produce byte-identical artifacts, every shipped test stays green.
  The refactor is gated on the full track-20 suite passing.
- **Over-abstraction.** A trait + registry + URI scheme is more machinery than
  three structs. Justified only because a fourth+ family is anticipated (tone,
  lexicon, safety-overlay) and the user-facing payoff is "declare your own
  evolution." If the abstraction starts leaking `kind` branches into the loop,
  that's the smell it failed — asserted against by the no-kind-branch test.
- **Hierarchy flattening.** Treating all meta objects as peers would let a goal
  override a safety value. Mitigated by `apply_scope` + asserted precedence
  (base-constitution > taste > goal).
- **Ordering vs tracks 21/12.** Building 21/12 standalone first then refactoring
  doubles the work. Mitigated by coordinating: 21's config+compile and 12's
  loader land as meta impls. If 21 already shipped standalone, its refactor is a
  bounded slice here (adapter + test migration).
