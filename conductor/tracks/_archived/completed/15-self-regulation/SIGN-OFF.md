# Track 15 — Self-Regulation — SIGN-OFF

Date: 2026-06-20

The transactional homeostasis layer that makes weight-mutating evolution safe to
run unattended: every step is `checkpoint → apply → eval → keep|rollback`, and a
catastrophe triggers `rollback + quarantine + halt`. The ONLY sanctioned
weight-mutation path — the scheduler/daemon (track 20/22) may auto-evolve only
THROUGH `regulate::Regulator`.

## Delivered (`src/regulate/`)

### `[regulate]` config ✅
`RegulateConfig` on `EvolveConfig.regulate` (additive, serde-default): `enabled`,
`accept_tolerance`, `catastrophe_floor`, `keep_checkpoints`, `on_catastrophe`.
`tolerances()` bridges to track-10 `VerdictTolerances`. Absent ⇒ no wrapper.

### `checkpoint.rs` — CheckpointStore ✅
`work_dir/checkpoints/<id>/{manifest.json, adapter/}` + a `last_good` pointer.
Snapshot/restore the **adapter dir** (the LoRA mutable artifact — base weights
are never touched, so the spec's "base-weights-as-deltas" complexity collapses to
an adapter copy; documented). Atomic manifests (temp+rename), retention pruning
that never removes `last_good` or `Quarantined` entries.

### `quarantine.rs` — Quarantine ✅
`work_dir/quarantine.json` keyed on the **`gen` provenance stamp** (styleguide
§2.4). `filter(dataset)` drops every row whose `gen` is quarantined so the same
catastrophic cause is never re-fed. This is the thing that stops a corrupting
trend from compounding across rounds.

### `log.rs` — evolution-log ✅
`work_dir/evolution-log.jsonl`, append-only, one row per step
{step, checkpoint_id, kind, verdict, metrics, action, cause}. The audit trail of
how the model evolved — and what it refused to keep.

### `txn.rs` — Regulator (the core) ✅
`run_step(id, kind, ordinal, baseline, step, score)`:
1. snapshot the pre-step adapter (the rollback target),
2. run `step` (mutates the adapter; returns its `gen` provenance),
3. snapshot the candidate, score it, `classify` vs baseline (track 10),
4. **Accept** → commit + advance `last_good` + retention; **Regress** → restore
   pre-step (state byte-equal) + mark Reverted, no halt; **Catastrophic** →
   restore + quarantine the provenance + mark Quarantined + **signal halt**,
5. append an evolution-log row in every case (incl. step-error → rollback).

The step + score are **injected closures** so the whole machinery is provable
ML-free; production passes a closure over `eval::run_eval`, tests pass
deterministic reports.

### CLI ✅
`checkpoints list|show|restore <id>`, `quarantine list|clear` (clear = re-arm).

## Acceptance evidence (`tests/regulate.rs`, 8/8 green)
- `accept_commits_and_advances_last_good` — pass commits, advances `last_good`,
  keeps the trained adapter.
- `regress_rolls_back_to_byte_equal_state` — soft regress restores the pre-step
  adapter **byte-equal**, does NOT halt.
- `catastrophe_rolls_back_quarantines_and_halts` — restore + quarantine + halt;
  a subsequent round's `Quarantine::filter` SKIPS the quarantined-provenance row.
- `quarantine_clear_rearms` — clear empties the quarantine (re-arm).
- `checkpoints_list_and_manifest_round_trip` — manifest carries metrics +
  provenance; round-trips.
- `step_error_rolls_back_without_verdict` — a crashing step rolls back, no verdict.
- `evolution_log_records_actions` — commit row recorded with the checkpoint id.
- `scorer_closure_can_track_calls` — confirms the injected-closure seam.

## Deferred (documented seams — not blocking the bench)
- **Expert pruning** (cold-evict + merge) — needs tracks 12/14 (expert registry).
- **Gated base pruning** (inverse-attribution, structured sparsity) — needs track
  13. The transaction machinery here is exactly what these would run inside; the
  `regulate/mod.rs` header documents the seam.

## Scope simplification (deliberate)
Checkpoints snapshot the LoRA adapter dir, not base-weight deltas, because no
path here mutates base weights (LoRA only). Full/base-weight checkpointing is a
feature-gated future addition; the manifest schema already carries everything a
delta scheme would need.

## Verification
- `cargo test` (default, ML-free): 18 suites green incl. `regulate` 8/8.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- `cargo build --features train` + `--features pyo3`: green.

## Carry-forward
Track 15 unblocks track 20 slices 6–9: the eval-gated round driver wraps each
goal's generate→train→eval round in `Regulator::run_step`, and the scheduler
loops rounds across goals, halting on catastrophe. Next track.
