---
type: Implementation Plan
title: Architecture Factory + Self-Distill Meta-Loop
description: Implementation plan for the Architecture Factory + Self-Distill Meta-Loop track.
tags: [track-17, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Architecture Factory + Self-Distill Meta-Loop — Plan

## Tasks

### Part 1 — QA → planner → DAG factory
1. [ ] `EvolveIntent` (goals, modalities, constraints, budget, cadence) +
   extend `interview` to emit `work_dir/intent.json`. -- evidence: intent
   capture test.
2. [ ] `arch/planner.rs`: planner LLM (mockable) consumes intent + signals +
   track-16 registry → `DagSpec` (nodes/edges/cfg + per-node rationale). Only
   registered kinds; output must pass `Dag::validate()`; invalid → reject +
   re-prompt. -- evidence: valid-DagSpec + reject-invalid (unregistered/cyclic/
   type-mismatch) tests.
3. [ ] Materialize: write `dag.json` + matching `evolve.toml`; then `dag run`
   via track 16. -- evidence: materialize-and-run fixture test.
4. [ ] Template rails: canonical templates (sft-lane, self-evolve-lane,
   eval-only); planner diffs against one; `--allow-freeform` gates free-form.
   -- evidence: template-match + freeform-gated tests.
5. [ ] CLI `architect --from intent.json`. -- evidence: CLI test.

### Part 2 — self-distill meta-loop
6. [ ] Artifact library `arch/library/<id>.json` (proven `DagSpec` + {intent
   served, eval score, parent}) + read/write round-trip. -- evidence: library
   round-trip test.
7. [ ] `arch/match.rs` selection-FIRST: match an intent against the library
   (similarity + fit threshold); hit → reuse artifact (NO generation), miss →
   fall through to generation. -- evidence: hit-reuses + miss-generates tests.
8. [ ] `arch/meta.rs` artifact GENERATOR (fallback): intent + recent
   `ScoreReport`s/`evolution-log` + library → a candidate `DagSpec` FILE built
   from registered nodes; must pass `Dag::validate()` before any run; invalid →
   reject + re-prompt. NOT an in-memory mutation; NO model-arch synthesis.
   -- evidence: valid-candidate-file + reject-invalid tests.
9. [ ] Trial run = sandboxed + transactional: a weight-touching candidate runs
   THROUGH track 15 (checkpoint weights → `dag run` candidate → eval track 10);
   pass → keep weights + admit artifact to library; regress → rollback weights +
   discard artifact. Eval-only candidate runs without the txn. -- evidence:
   pass-admits + regress-discards + eval-only-no-txn tests.
10. [ ] Catastrophe path: forced catastrophic eval rolls back weights + halts the
    loop + quarantines the artifact (via track 15). -- evidence: catastrophe test.
11. [ ] Bounded search: budget (max candidates/nodes/tokens) + pool (generate K,
    keep best) + stop-on-plateau. -- evidence: stops-at-budget + stops-on-plateau.
12. [ ] `arch-log.jsonl` (every candidate + score delta) + `arch/lineage.json`
    (descent); a library artifact round-trips + is selectable/re-runnable + usable
    as a starting template. -- evidence: log + reuse-proven-artifact tests.
13. [ ] CLI `architect distill [--budget] [--allow-freeform]`, `architect library
    list|show|use <id>`, `architect lineage`. -- evidence: CLI tests.
14. [ ] Final sweep: `cargo build`, `cargo test`, `cargo clippy`. -- evidence: green.

## Sign-off
Pending.
