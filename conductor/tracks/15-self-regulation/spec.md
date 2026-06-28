---
type: Track Spec
title: Self-Regulation — Prune, Evaluate, Rollback, Quarantine
description: The homeostasis layer (txn keep/rollback) that makes the self-evolve lane safe to run.
tags: [track-15, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Self-Regulation — Prune, Evaluate, Rollback, Quarantine — Specification

## Goal
The **homeostasis layer** that makes the self-evolve lane (09–12) safe to run
unattended. Every evolution step (train / refine / spawn-expert / prune) becomes
**transactional**: checkpoint → apply → **evaluate** → keep or **roll back**. The
model can **prune itself smaller** (cold experts always; dense base only behind
the transaction), **detect misalignment / catastrophic regression**, and on a
catastrophe **auto-roll-back to the last good checkpoint AND quarantine the
cause** so the next round skips it. This is the recursion's base case: grow →
evaluate → keep-or-revert. Without it the loop has no convergence guarantee.

> Builds on the existing `work_dir/checkpoints/` convention (today empty — this
> track defines its format). No reverse inference; pruning uses track 13's
> attribution as a coarse prior (proposes; eval disposes).

> **Cogency-audit notes (apply throughout).** Post-renumber, the lane is
> **10** eval-harness, **11** regen, **12** refine, **13** mask, **14** experts,
> **15** this. All in-spec references to "lane (09–12)", "track 11 attribution",
> "track 12 experts", "track 09 gate", "track 10 constitution" should be read as
> 13 (attribution/mask), 14 (experts), 10 (gate via eval-harness), 12
> (constitution) respectively.
> - **Config host (A1).** `[regulate]` is a new `Option<RegulateConfig>` field on
>   the top-level `EvolveConfig` struct (serde-default → non-breaking).
> - **Shared evaluator (the audit's headline fix).** This track does NOT build
>   its own probe/scorer. It CONSUMES track 10's `ProbeSet`, `Scorer`,
>   `ScoreReport`, and `StepVerdict`. `evolve/eval.rs` here is a thin policy
>   layer (thresholds → action), not a second evaluator.
> - **Quarantine via provenance (B5).** "Quarantine the cause" keys on the
>   existing `GenExample.gen` provenance stamps (`regen:swap<N>`,
>   `refine:*`, `expert:<path_id>`) plus cluster id — the mechanism already
>   exists in `dataset.rs`; quarantine writes those identifiers to
>   `quarantine.json` and the loop filters rows whose `gen`/cluster match.
> - **Directive — PyO3→transformers.** Gated base pruning (structured
>   sparsity / layer-drop) and weight-delta checkpointing operate on the Python
>   `transformers` model via `bridge.rs` (`--features pyo3`) — torch state-dict
>   diffing + pruning utilities — rather than a hand-built candle path (optional
>   later). Expert pruning is native Rust (registry + adapter files).

## Scope
- **Transactional evolution wrapper** (`evolve/txn.rs`): wraps ANY evolution
  step. `begin()` snapshots a `Checkpoint` (base weights ref/delta + expert
  registry + constitution version + config + RNG/loop state) → step runs →
  `evaluate()` → `commit()` (keep) or `rollback()` (restore snapshot). Steps are
  idempotent + resumable so a daemon can drive them; a crashed step rolls back on
  restart.
- **Checkpoint store** (`work_dir/checkpoints/<id>/`): manifest
  {id, parent_id, step_kind, created, metrics, status (good|reverted|quarantined),
  artifacts[]}. Content-addressed where possible; base weights stored as deltas
  vs the previous good checkpoint to bound disk. A `last_good` pointer.
- **Evaluation gate** (`evolve/eval.rs`): runs after every step, reusing existing
  signals — executable correctness on a held-out probe set (track 09 gate),
  constitution-principle adherence sampling (track 10), depth/correctness metrics
  (09's `regen-metrics`). Produces a `StepVerdict`:
  - `accept` — no regression beyond tolerance.
  - `regress` — soft worse → roll back this step, keep looping.
  - `catastrophic` — hard threshold breach (correctness collapse below floor,
    safety-principle violation-rate spike, loss divergence/NaN, perplexity blowup)
    → the catastrophe path below.
  Thresholds + tolerances in config; the held-out probe set is fixed per run so
  verdicts are comparable across steps.
- **Self-pruning** (`evolve/prune.rs`), two tiers, BOTH transactional:
  - **Expert pruning (auto, low-risk):** evict experts the router rarely selects
    (usage stats from track 12) + merge near-duplicates (extends 12's merge).
    Base untouched → near-zero risk, but still checkpointed.
  - **Gated base pruning (auto, transactional only):** structured sparsity /
    layer-drop on the dense base, proposed by INVERSE attribution (track 11:
    low-contribution layers/modules). ALWAYS prune → checkpoint → eval → auto
    rollback on any regression. Never irreversible; never runs outside the
    transaction. `min_layers`/output-module floors (from 11) protect viability.
- **Catastrophe handler = rollback + quarantine + halt** (the chosen policy):
  on `catastrophic`, (1) restore `last_good`, (2) **quarantine the cause** —
  identify the offending input (dataset cluster id, expert `path_id`, or step
  config) and write it to a `quarantine.json` the loop consults to SKIP it next
  round, (3) **halt** the self-evolve loop and emit an incident report; resuming
  requires explicit re-arm. (Soft `regress` only rolls back the one step and
  continues.)
- **`[regulate]` config**: `enabled`, `probe_set` (path/size), `accept_tolerance`
  (per metric), `catastrophe` (per-metric hard floors), `prune.experts`
  (cold_threshold, merge_similarity), `prune.base` (enabled, target_sparsity,
  selector=inverse_attribution|grad|manual), `keep_checkpoints` (retention),
  `on_catastrophe = rollback_quarantine_halt`.
- **History / observability**: `work_dir/evolution-log.jsonl` — one row per step
  {step, kind, verdict, metrics, action (commit|rollback|quarantine), cause}.
  The audit trail of how the model evolved (and what it refused to keep).
- CLI: `scrt-evolve evolve step --kind <…>` (run one transactional step),
  `scrt-evolve checkpoints list|show|restore <id>`, `scrt-evolve prune
  [--experts|--base]` (transactional), `scrt-evolve quarantine list|clear`,
  `scrt-evolve evolve rearm` (resume after a halt).

## Constraints
- **Every destructive action is transactional.** Base pruning and any
  weight-mutating step MUST snapshot first; there is no code path that prunes or
  trains the base without a restorable checkpoint. Asserted.
- **The base model is never auto-pruned outside the eval-gated transaction.**
  Expert eviction is the only always-safe auto-shrink (base intact).
- **Catastrophe halts, hard.** A corrupting trend must not compound across
  rounds; auto-rollback + quarantine + halt is mandatory on a hard breach. The
  quarantine must actually prevent re-feeding the same cause (asserted).
- **Eval is mechanical + comparable.** Fixed probe set, deterministic-seeded;
  verdicts are thresholds over measured metrics, not a quality judgment call.
  Whether the *thresholds* are well-tuned is an experiment, OUT of scope; the
  *machinery* (snapshot/eval/revert/quarantine works) is what this track proves.
- **Graceful degradation.** No probe set → eval falls back to loss/NaN guards
  only (still catches catastrophic, not soft regress) and logs the reduced
  coverage. LARQL inverse-attribution for base pruning is `--features larql`
  optional; falls back to grad/manual.
- ML-free build green: txn wrapper, checkpoint store, eval scaffold, prune
  planning, quarantine, config, CLI compile without candle; weight snapshot/
  restore + base pruning + attribution behind `--features train`/`larql`.
- Reuses: 09 (executable gate, regen-metrics), 10 (constitution check), 11
  (attribution/inverse + floors), 12 (expert usage stats + merge + cluster/expert
  identity for quarantine). Adds no new arch.

## Acceptance
- A transactional step that passes eval **commits** and advances `last_good`; a
  step that soft-`regress`es **rolls back** to the prior checkpoint (model state
  restored, asserted byte/shape-equal on a fixture).
- A forced `catastrophic` verdict triggers rollback + writes the cause to
  `quarantine.json` + halts the loop; a subsequent round SKIPS the quarantined
  cause (asserted); `evolve rearm` is required to resume.
- Expert pruning evicts a cold expert and merges a near-duplicate; the router
  still serves (base + remaining experts) afterward.
- Gated base pruning on a fixture produces a smaller base when eval passes, and
  **auto-rolls-back** to the unpruned base when eval regresses (both asserted) —
  proving prune-is-never-irreversible.
- `checkpoints list/show/restore` round-trip; base weights stored as deltas vs
  parent (disk bound asserted on a fixture).
- `evolution-log.jsonl` records commit / rollback / quarantine rows with cause.
- No-probe-set degradation runs with loss/NaN-only guards and logs reduced
  coverage; no crash.
- ML-free + `--features train` + `--features "train larql"` build green.
- **Styleguide gates** (code-styleguides.md): this track IS the enforcement point
  for §2.3 (every weight-mutating step transactional; atomic, delta checkpoints;
  resume) and §2.4 (quarantine by `gen`-provenance; `evolution-log.jsonl` audit
  trail); catastrophe halts a bounded loop (§2.5). No code path mutates base
  weights without a restorable checkpoint (§2.3, asserted). Built per §4.

## Dependencies
Tracks 09 (gate + metrics), 10 (constitution eval), 11 (attribution + floors for
prune), 12 (expert usage/merge + identity for quarantine). The capstone of the
self-evolve lane; sequenced last in it (after 12). Makes the DESIGN.md daemon
north-star safe — a daemon may only auto-evolve THROUGH this transactional
wrapper.
