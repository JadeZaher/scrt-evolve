---
type: Implementation Plan
title: Self-Regulation — Prune, Evaluate, Rollback, Quarantine
description: Implementation plan for the Self-Regulation — Prune, Evaluate, Rollback, Quarantine track.
tags: [track-15, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Self-Regulation — Prune, Evaluate, Rollback, Quarantine — Plan

## Tasks

1. [x] `[regulate]` config: `enabled`, `accept_tolerance`, `catastrophe_floor`,
   `keep_checkpoints`, `on_catastrophe`. ML-free round-trip; absent = off.
   DONE: `RegulateConfig` on `EvolveConfig.regulate` (`config.rs`), with
   `tolerances()` → track-10 `VerdictTolerances`. NOTE: `prune.experts`/`prune.base`
   are a documented seam (tracks 11–14 not built) — not added as config yet.
2. [x] Checkpoint store: `work_dir/checkpoints/<id>/` manifest {id, parent_id,
   step_kind, ordinal, metrics, status, artifacts, gen_provenance}, `last_good`.
   DONE: `regulate/checkpoint.rs::CheckpointStore` — snapshot/restore the ADAPTER
   dir (the LoRA mutable artifact; base never touched ⇒ no delta complexity for
   this path, documented), atomic manifests, retention. Test:
   `checkpoints_list_and_manifest_round_trip`.
3. [x] `regulate/txn.rs` transactional wrapper: snapshot → step → evaluate →
   commit | rollback; step-error → rollback. DONE: `Regulator::run_step` with
   injected step + score closures. Tests: `accept_commits_and_advances_last_good`,
   `regress_rolls_back_to_byte_equal_state`, `step_error_rolls_back_without_verdict`.
4. [x] Eval = thin policy over **track 10**: classify via `StepVerdict` with this
   track's thresholds; does NOT build its own scorer. DONE: `run_step` calls
   track-10 `classify(baseline, candidate, tolerances)`. (The production score
   closure wraps `eval::run_eval`; tests inject deterministic reports.)
5. [x] Soft-regress rolls back the one step, loop continues (no halt). DONE:
   `regress_rolls_back_to_byte_equal_state` asserts `!halt` + state restored.
6. [x] Catastrophe handler: restore + write cause to `quarantine.json` + halt.
   DONE: `regulate/quarantine.rs` + the `Catastrophic` arm of `run_step`. Test:
   `catastrophe_rolls_back_quarantines_and_halts`.
7. [x] Quarantine enforcement: next round SKIPS the quarantined cause;
   `quarantine list|clear` (= re-arm). DONE: `Quarantine::filter` drops rows whose
   `gen` is quarantined; tests `catastrophe_..._halts` (filter) +
   `quarantine_clear_rearms`. CLI: `quarantine list|clear`.
8. [ ] Expert pruning (cold-evict + merge). DEFERRED — needs tracks 12/14 (expert
   registry/usage), not built; not required for the eval-gated training schedule.
   Documented seam in `regulate/mod.rs`.
9. [ ] Gated base pruning (inverse-attribution, transactional). DEFERRED — needs
   track 13 attribution; the transaction machinery here is exactly what it would
   run inside. Documented seam.
10. [x] `evolution-log.jsonl`: per-step {step, kind, verdict, metrics, action,
    cause}. DONE: `regulate/log.rs`; test `evolution_log_records_actions`.
11. [x] No-probe degradation: handled at the track-10 layer (`run_eval` returns an
    uncovered report when no probe exists; the score closure surfaces that). The
    txn still snapshots/commits/rolls back on whatever signal it gets.
12. [x] CLI: `checkpoints list|show|restore`, `quarantine list|clear`. (`evolve
    step` / `prune` belong to the round driver / deferred prune — not added.)
13. [x] Final sweep: `cargo build`, `cargo test`, `cargo build --features train`,
    `cargo build --features pyo3`, `cargo clippy -D warnings`, `cargo fmt --check`.
    -- GREEN.

## Sign-off
Signed off 2026-06-20 — see SIGN-OFF.md. The transactional homeostasis core
(checkpoint store, snapshot/eval/commit/rollback, catastrophe→quarantine→halt,
quarantine-skip, evolution-log) is built + verified, consuming track 10's
`StepVerdict`. Self-pruning (experts/base, tasks 8–9) is DEFERRED as a documented
seam (depends on tracks 11–14, not needed for the eval-gated schedule). Scope
simplification: checkpoints snapshot the LoRA adapter dir (base weights are never
mutated on this path), so "base weights as deltas" is not needed here.
