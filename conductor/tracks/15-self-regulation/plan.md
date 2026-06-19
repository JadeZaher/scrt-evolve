# Self-Regulation — Prune, Evaluate, Rollback, Quarantine — Plan

## Tasks

1. [ ] `[regulate]` config: `enabled`, `probe_set`, `accept_tolerance`,
   `catastrophe` floors, `prune.experts`, `prune.base`, `keep_checkpoints`,
   `on_catastrophe`. ML-free round-trip; absent block = regulation off.
   -- evidence: config + default-off test.
2. [ ] Checkpoint store: `work_dir/checkpoints/<id>/` manifest {id, parent_id,
   step_kind, created, metrics, status, artifacts}, `last_good` pointer; base
   weights as deltas vs parent. -- evidence: store round-trip + delta-size test.
3. [ ] `evolve/txn.rs` transactional wrapper: begin(snapshot) → step →
   evaluate → commit | rollback; idempotent/resumable; crash → rollback on
   restart. -- evidence: commit-advances-last_good + rollback-restores-state test.
4. [ ] `evolve/eval.rs` = thin policy layer over **track 10**: call track 10's
   `Scorer` on the held-out `ProbeSet` → `ScoreReport`, then `StepVerdict`
   (accept|regress|catastrophic) via this track's config thresholds. Does NOT
   build its own probe/scorer. -- evidence: verdict-per-threshold test (mocked Scorer).
5. [ ] Soft-regress path: a worse-but-not-catastrophic step rolls back the one
   step and the loop continues. -- evidence: regress-rolls-back-one-step test.
6. [ ] Catastrophe handler: restore last_good + write cause (cluster id / expert
   path_id / step config) to `quarantine.json` + halt loop + incident report.
   -- evidence: catastrophe-rollback+quarantine+halt test.
7. [ ] Quarantine enforcement: next round consults `quarantine.json` and SKIPS
   the quarantined cause; `quarantine list|clear`; `evolve rearm` resumes.
   -- evidence: quarantined-cause-skipped + rearm test.
8. [ ] Expert pruning (auto, native Rust): evict cold experts (track 14 usage)
   + merge near-duplicates; router still serves after. -- evidence: evict+merge+serve test.
9. [ ] Gated base pruning: inverse-attribution (track 13 `AttributionReport`;
   grad/manual fallback) structured sparsity/layer-drop via **PyO3→torch**,
   transactional, `min_layers`/output floors. Behind `--features pyo3` (candle
   optional). -- evidence: smaller-on-pass + auto-rollback-on-
   regress test.
10. [ ] `evolution-log.jsonl`: per-step {step, kind, verdict, metrics, action,
    cause}. -- evidence: log shape test.
11. [ ] No-probe-set degradation: loss/NaN-only guards, logs reduced coverage,
    no crash. -- evidence: degraded-eval test.
12. [ ] CLI: `evolve step`, `checkpoints list|show|restore`, `prune
    [--experts|--base]`, `quarantine list|clear`, `evolve rearm`. -- evidence:
    CLI tests.
13. [ ] Final sweep: `cargo build`, `cargo test`, `cargo test --features train`,
    `cargo build --features "train larql"`, `cargo clippy --features train`.
    -- evidence: green.

## Sign-off
Pending.
