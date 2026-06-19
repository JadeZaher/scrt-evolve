# SDK Builder Interface — Trait-Powered, Two-Phase, Sandboxable — Specification

## Goal
Define the framework's **primary SDK interface**: a **trait-powered builder**
that is configured/designed and then `.execute()`d. Step parameters are
**callbacks** the engine invokes; each step is split into two phases —
**argument-generation** (`resolve_args`) and **execution** (`execute`) — so the
two can be cached, tested, and (later) sandboxed independently. **Capability
traits** select which step sets / tag types / artifact formats / training tooling
the builder exposes. The builder **lowers to the serializable typed DAG** (track
16), so a designed pipeline persists to disk and round-trips. This is THE main
way to drive scrt-evolve from Rust; the CLI is a thin shim over it.

> This supersedes the plainer `Dag::builder().discover().train()` sketch in track
> 16: that becomes the LOWERING TARGET. The trait builder is the surface; the
> typed DAG is what it compiles to + what persists.

## Scope
- **The step trait (two-phase, the core abstraction).**
  ```rust
  trait Step {
      type Args;            // what arg-generation produces
      type Out;             // what execution returns
      fn kind(&self) -> &str;                                  // serializable id
      fn resolve_args(&self, ctx: &StepCtx) -> Result<Self::Args>;  // GEN phase
      fn execute(&self, args: Self::Args) -> Result<Self::Out>;     // EXEC phase
  }
  ```
  `resolve_args` is the callback that PRODUCES what the step needs (pull inputs
  from upstream ports, query the palace, synthesize prompts); `execute` is the
  effectful run. The split is the **sandbox seam**: gen is pure-ish (no weight
  mutation), exec is the effectful part (weight-touching exec is wrapped by track
  15). Both phases are independently cacheable (content-addressed, track 16).
- **Callbacks-as-steps that persist.** A step supplied as a closure is registered
  under a **named kind** (the closure is wrapped + given a stable `kind` string)
  so the lowered DAG records the kind + cfg, not the closure body — the graph
  serializes to `dag.json` and round-trips. Anonymous closures are SDK-only
  convenience; anything saved/selected by the track-17 library uses named kinds.
  (Decision: persisting the step graph IS wanted — it's the audit trail + the
  library artifact.)
- **Capability traits select the surface (the key idea).** The builder's
  available methods, tag/node types, and output formats are determined by the
  trait(s) in scope — different traits expose different tooling:
  - `CoreEvolve` — discover / plan / generate / gate / train / export.
  - `SelfEvolve` — adds regen-antagonist, refine (dialectic), mask, expert-spawn,
    prune, self-regulate (tracks 11–15).
  - `Distill` — adds the architecture factory + library/selection ops (track 17).
  - `ToolingX` traits — expose specific training tooling (e.g. a `Peft` trait
    surfacing PyO3→peft knobs, a `Trl` trait surfacing DPO) — so a consumer opts
    into exactly the tooling they want, and unrelated steps don't clutter the API.
  Multiple traits compose; a builder typed `Builder<CoreEvolve + SelfEvolve>`
  exposes both step sets. Capability is a **typestate**: calling a step a trait
  doesn't grant is a COMPILE error, not a runtime one.
- **Design-then-execute lifecycle.** `let plan = Builder::<Caps>::new(&cfg)
  .step(..).step(..); let dag = plan.build()?  /* validate (track 16) */;
  let outs = dag.execute(&ctx)?  /* run */;`. `build()` lowers to a typed DAG and
  runs `Dag::validate()` (acyclic, typed ports, cfg valid) BEFORE anything
  executes; `execute()` runs the two-phase nodes via the track-16 scheduler.
- **Format selection via traits.** The trait set also picks artifact/export
  formats (e.g. a `Gemma` tooling trait selects Gemma-native tool rendering; an
  `OpenAiFmt` trait selects another) — the builder exposes only the formats its
  traits enable, removing invalid combinations at compile time.
- **Sandbox seam (built now as a boundary, OS isolation later).** `resolve_args`
  and `execute` are separate trait methods with serializable `Args` between them,
  so a future process/OS sandbox can wrap either phase (run arg-gen or execution
  in an isolated process, passing `Args`/`Out` across) WITHOUT changing the step
  API. This track builds the typed phase boundary + requires `Args: Serialize`;
  it does NOT build OS isolation (explicit future seam).
- CLI relationship: unchanged — the CLI constructs a builder with the right
  capability traits from argv and calls `.build()?.execute()`. No logic in the
  binary.

## Constraints
- **This is THE primary SDK surface.** The trait builder is how a Rust consumer
  drives the framework; `discover/generate/train::run` remain as thin convenience
  wrappers, and `Dag` (track 16) is the lowering target + persistence format —
  not the headline API.
- **Lowers to a serializable DAG.** Every built pipeline compiles to a typed,
  validatable, `dag.json`-serializable graph (track 16). A builder construct that
  cannot lower to a valid DAG is a compile/build error. Persisted graphs use
  named step kinds (no closure bodies on disk).
- **Two-phase, always.** Every step separates `resolve_args` from `execute`; no
  step may do effectful work in `resolve_args`. Weight-touching `execute` is
  wrapped by track 15. The phase boundary is mandatory (it's the sandbox seam).
- **Capability = typestate.** Unavailable steps/formats are COMPILE errors via
  the trait bounds, not runtime failures.
- **Args crosses the seam as data.** `Step::Args: Serialize + Deserialize` so the
  gen→exec boundary can later become a process boundary. Enforced by trait bound.
- ML-free build green: the trait, builder, capability traits, lowering, and
  phase split compile with no candle; concrete tooling-trait steps gate behind
  their features (`pyo3`/`train`/`larql`).

## Acceptance
- A `Builder::<CoreEvolve>` exposes discover/generate/train steps and builds +
  executes a pipeline that matches the canonical DAG output (fixture, no ML).
- Calling a `SelfEvolve` step on a `Builder::<CoreEvolve>` is a COMPILE error
  (typestate — asserted via a trybuild/compile-fail test or documented equivalent).
- A closure step is wrapped under a named kind; the built DAG lowers + serializes
  to `dag.json` and round-trips (named kind persisted, not the closure).
- Two-phase: a step's `resolve_args` is cached independently of `execute`
  (changing exec-only cfg reuses cached args; asserted) and `Args: Serialize`
  holds (the seam is data-crossable).
- A weight-touching step's `execute` runs through the track-15 transaction;
  `resolve_args` does not (asserted).
- Trait selects format: a `Gemma`-typed builder exposes Gemma export only; an
  invalid format method is absent at compile time.
- ML-free `cargo build` + `cargo test` green; tooling traits compile behind their
  features.
- **Styleguide gates** (code-styleguides.md): the two-phase split realizes §2.1
  (`resolve_args` pure, effects only in `execute`); `Args: Serialize` is the §2.5
  sandbox-seam requirement; weight-touching `execute` via track 15 (§2.3); built
  pipelines lower to a serializable named-kind DAG (§2.5). Built per §4.

## Dependencies
Track 16 (the typed DAG it lowers to + validates against + serializes through),
track 15 (transaction wrapping weight-touching `execute`), and it RE-EXPOSES the
steps from tracks 01/02/04/10–17 as capability-trait methods (wrapping their node
impls in the two-phase `Step`). The primary SDK interface for the whole
framework; design-stable seam for a future OS sandbox.
