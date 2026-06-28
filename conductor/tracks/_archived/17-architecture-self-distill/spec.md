---
type: Track Spec
title: Architecture Factory + Self-Distill Meta-Loop
description: Architecture factory and a self-distill meta-loop on top of the DAG engine.
tags: [track-17, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Architecture Factory + Self-Distill Meta-Loop — Specification

## Goal
Two capabilities on top of the DAG engine (track 16):
1. **Config factory / DAG generator** — the QA/interview process produces a
   high-level *intent*; a **planner LLM emits a validated DAG** (which nodes,
   wired how, with per-node cfg) + a reproducible `evolve.toml`. The pipeline is
   DESIGNED from intent, not hand-authored. This generalizes the existing
   `plan/` module (which already planks generation) to the whole graph.
2. **Architecture-artifact distillation (meta-loop)** — the system **generates
   architecture ARTIFACTS** (`DagSpec` files on disk), runs them like any other
   config, evaluates the result, and **keeps the winning artifacts in a reusable
   library** — building up distilled architecture knowledge it can SELECT and
   re-run later. It does NOT mutate a live in-memory graph, and it does NOT
   synthesize *model* architecture (no new layers/network topology / node logic);
   it composes EXISTING registered nodes into DAG artifact files.

> **Artifact-first, not mutation-first (the corrected model).** The architecture
> design is always a first-class artifact (`dag.json` / `DagSpec`). The meta-loop
> *generates those artifacts and then uses them* — generation produces a candidate
> file; the file is validated + run exactly as a hand-authored one; proven files
> are saved to a library; future runs **select and reuse** library artifacts
> instead of re-generating. A meta-generated artifact is indistinguishable from
> one the QA factory (Part 1) emits or a human writes — one format, one executor,
> three authors (human / factory / meta-loop). "Self-distillation" = accumulating
> a library of proven, re-selectable architecture files, NOT a self-rewriting
> model graph.

> Honest framing: capability 1 is a well-trodden pattern (planner → workflow
> spec → execute), already prototyped in-tree for generation. Capability 2 is
> kept tractable BECAUSE it is artifact-first: it only writes/selects/runs typed
> DAG files of REGISTERED nodes (track 16), trial runs that touch weights go
> through the transactional wrapper (track 15), and the search is bounded. No
> live graph mutation and no model-architecture synthesis are involved.

## Scope
### Part 1 — QA → planner → DAG factory
- **Intent capture**: extend the existing `interview` command to produce an
  `EvolveIntent` (goals/objectives, target modalities, constraints, budget,
  cadence) — durable `work_dir/intent.json`. Reuses the interview scaffolding.
- **DAG planner** (`arch/planner.rs`): a planner LLM consumes `EvolveIntent` +
  deterministic signals (corpus/palace shape, available nodes from the track-16
  registry, hardware budget) and emits a **`DagSpec`** = the track-16 `Dag` as
  data (nodes + edges + per-node cfg) + a rationale per node. The planner may
  ONLY use registered node kinds and must produce a graph that passes
  `Dag::validate()` (typed, acyclic) — invalid proposals are rejected and
  re-prompted, never run.
- **Materialize**: write the `DagSpec` to `dag.json` AND emit a matching
  `evolve.toml` (so a generated run is fully reproducible + hand-editable). Then
  `dag run` it via track 16.
- **Templates as rails** (safety, not the chosen "templates-only" path): the
  planner starts from / is diffed against canonical lane templates (sft-lane,
  self-evolve-lane, eval-only); a fully free-form graph requires
  `--allow-freeform`. Keeps the common case safe + deterministic-ish while
  permitting novelty behind a flag.

### Part 2 — Architecture-artifact distillation (generate → run → library → reuse)
- **Selection-FIRST (the default; "use them instead of self-generating").** On a
  new intent, the system FIRST tries to **select** a proven architecture artifact
  from the library (`arch/library/*.json`, each with its recorded eval score +
  the intent it served). A match → reuse that `DagSpec` directly (NO generation).
  Generation is the FALLBACK, run only when no library artifact fits. The library
  IS the distilled architecture knowledge; over time the system generates less
  and reuses more. `arch/match.rs` does intent↔artifact matching (signal/intent
  similarity + a fit threshold).
- **`arch/meta.rs` — the artifact generator (fallback path).** Input: the intent
  + recent `ScoreReport`s/`evolution-log` (track 10/15) + the library. Output: a
  candidate **`DagSpec` FILE** (not an in-memory mutation) built from registered
  nodes, validated by track-16 `Dag::validate()` before it is ever run. A
  candidate that fails validation is rejected + re-prompted, never run.
- **Trial run = sandboxed + transactional.** Generating a candidate is free (just
  a file). RUNNING a candidate that trains/mutates weights goes THROUGH the
  track-15 transaction: checkpoint weights → `dag run` the candidate file → eval
  (track 10) → **on pass: keep the weights AND save the artifact to the library;
  on regress: roll back the weights AND discard the artifact.** Eval-only /
  no-weight-mutation candidates may run without the txn (cheap), but anything
  touching weights MUST use it. The artifact enters the library only if its run
  passed.
- **Bounded search**: budget (max candidates, max nodes, wall-clock/token cap) +
  candidate pool (generate K, run, keep best) + stop-on-no-improvement-for-N.
  Every rejected/kept candidate logs to `arch-log.jsonl` with score delta.
- **The library + lineage are the distillation artifacts.**
  `arch/library/<id>.json` = a proven `DagSpec` + {intent served, eval score,
  parent}. `arch/lineage.json` = how artifacts descended from one another. Both
  are durable, inspectable, hand-editable, and re-runnable by hand — and become
  the SELECTION pool for future intents (a generated artifact, once proven, is
  just a reusable config like any other).
- CLI: `scrt-evolve architect --from intent.json` (Part 1+selection: reuse a
  library artifact if one fits, else generate), `scrt-evolve architect distill
  [--budget …] [--allow-freeform]` (run the generate→run→library loop),
  `scrt-evolve architect library list|show|use <id>` (browse/select proven
  artifacts), `scrt-evolve architect lineage` (descent history).

## Constraints
- **Rail 1 — typed DAG only.** The planner/meta-node emit `DagSpec`s built from
  REGISTERED node kinds; output MUST pass track-16 `Dag::validate()` before any
  execution. No free-text "code generation" of new node logic in this track —
  novelty is in WIRING + cfg, not in synthesizing untrusted Rust. (Synthesizing
  genuinely new node *implementations* is explicitly a future, separate, sandbox-
  gated concern — out of scope here.)
- **Rail 1b — artifact-first, no live mutation, no model-arch synthesis.** The
  meta-loop GENERATES `DagSpec` FILES and RUNS them; it never mutates an
  in-memory graph, and it never invents model architecture (layers/topology) or
  new node logic — only wiring + cfg over registered nodes. Selection-first:
  reuse a library artifact when one fits; generate only on a miss.
- **Rail 2 — transactional, mandatory for weight-touching trials.** Generating
  an artifact is free (a file). RUNNING a candidate that trains/mutates weights
  runs THROUGH the track-15 wrapper: checkpoint weights → run → eval →
  keep|rollback. There is NO code path that mutates weights without a restorable
  checkpoint. On pass: keep weights + admit the artifact to the library; on
  regress: roll back weights + discard the artifact; on catastrophe: roll back +
  halt + quarantine the offending artifact. (Eval-only/no-weight candidates may
  run without the txn.) This is the load-bearing safety property.
- **Rail 3 — bounded.** Artifact search is budget-capped (candidates, nodes,
  tokens/wall-clock) + stop-on-plateau; unbounded generation is a defect. Small
  default budget; large searches explicit.
- **Reproducibility + library.** Every generated artifact is serialized
  (`DagSpec` file + matching `evolve.toml`) and, if proven, saved to
  `arch/library/` with its score + lineage — re-runnable by hand and re-selectable
  by future intents. An artifact that isn't reproducible from its file is a defect.
- **No new ML; PyO3→transformers for any training the candidate DAGs do** (the
  lane directive carries through — the meta-loop schedules existing nodes, it
  doesn't introduce new training code).
- ML-free build green: planner, `DagSpec`, materialization, meta-node, bounded
  search, lineage, CLI compile without candle; the candidate runs use whatever
  features their nodes need.

## Acceptance
- `architect --from intent.json` produces a `dag.json` + matching `evolve.toml`
  that passes `Dag::validate()` and reproduces a known lane when the intent
  matches a template (Part 1 end-to-end on a fixture, mockable planner).
- A planner proposal using an unregistered node kind / producing a cyclic or
  type-mismatched graph is REJECTED (not run) and re-prompted (asserted).
- **Selection-first**: given an intent that matches a library artifact, the
  system REUSES it (no generation occurs — asserted); only a miss triggers
  generation. After a successful trial, the artifact is in the library and a
  subsequent matching intent reuses it (asserted: generate-once-then-reuse).
- Meta-loop: a generated candidate that improves the fixture `ScoreReport` is
  KEPT — weights kept AND artifact admitted to the library; one that regresses is
  ROLLED BACK — weights restored AND artifact discarded (asserted); a forced
  catastrophe rolls back + halts + quarantines the artifact — all via track 15.
- Eval-only candidate runs without invoking the txn; a weight-touching candidate
  MUST invoke it (asserted both ways).
- Bounded search stops at the budget and at no-improvement-for-N (asserted).
- `arch-log.jsonl` + `arch/library/*.json` + `arch/lineage.json` record every
  candidate with score delta; a saved library artifact round-trips and is
  selectable/re-runnable.
- `--allow-freeform` gates non-template graphs; without it, a freeform proposal
  is refused (asserted).
- ML-free `cargo build` + `cargo test` green.
- **Styleguide gates** (code-styleguides.md): generated artifacts are serializable
  named-kind DAGs (§2.5) reproducible from their file (§2.1); weight-touching
  trials run through track 15 (§2.3) and a bad trial discards the artifact +
  quarantines (§2.4); the architecture search is bounded (§2.5). Built per §4.

## Dependencies
Track 16 (the typed DAG model + registry + executor + `Dag::validate()` — the
substrate the planner/meta-node emit and the meta-loop runs), track 15 (the
transactional checkpoint→eval→rollback wrapper every meta-iteration MUST use),
track 10 (`Scorer`/`ScoreReport` the meta-loop optimizes against), the existing
`interview`/`plan` modules (intent capture + planner pattern lifted up a level).
The capstone of the whole project — turns it into a generic, self-architecting
training/model-building framework. Sequenced last.
