//! The transactional evolution wrapper (track 15) — the homeostasis core.
//!
//! Makes ANY weight-mutating step `checkpoint → apply → eval → keep|rollback`,
//! with `catastrophic → rollback + quarantine + halt`. No code path mutates
//! weights without a restorable checkpoint (styleguide §2.3). This is the ONLY
//! sanctioned weight-mutation path — the daemon/scheduler may auto-evolve only
//! THROUGH here.
//!
//! Design for testability + the ML-free build: the wrapper does NOT itself train
//! or score. The caller supplies:
//! - a **step**: `FnOnce() -> Result<Vec<String>>` that mutates the adapter dir
//!   and returns the `gen` provenance stamps its training rows carried (the
//!   quarantine key),
//! - a **scorer**: `Fn() -> Result<ScoreReport>` (production passes a closure
//!   over [`crate::eval::run_eval`]; tests pass a deterministic stub).
//!
//! So the transaction machinery (snapshot/eval/revert/quarantine/log/halt) is
//! provable with zero ML deps; the heavy bits plug in at the call site.

use std::path::PathBuf;

use crate::config::{EvolveConfig, RegulateConfig};
use crate::eval::{classify, ScoreReport, StepVerdict};
use crate::workdir::WorkDir;

use super::checkpoint::{CheckpointStatus, CheckpointStore};
use super::log::{self, EvolutionLogEntry, StepAction};
use super::quarantine::Quarantine;

/// The outcome of one transactional step.
#[derive(Debug, Clone)]
pub struct TxnOutcome {
    /// The checkpoint this step produced.
    pub checkpoint_id: String,
    /// The verdict reached (None if the step errored before eval).
    pub verdict: Option<StepVerdict>,
    /// The action taken.
    pub action: StepAction,
    /// Whether the loop must HALT (a catastrophe occurred).
    pub halt: bool,
    /// The metrics, if scored.
    pub metrics: Option<ScoreReport>,
}

/// The transactional driver, bound to a config + work-dir.
pub struct Regulator {
    rcfg: RegulateConfig,
    store: CheckpointStore,
    workdir: WorkDir,
    quarantine_path: PathBuf,
    log_path: PathBuf,
}

impl Regulator {
    /// Build a regulator from a config. Errors only on checkpoint-store IO.
    pub fn new(cfg: &EvolveConfig) -> anyhow::Result<Self> {
        let workdir = WorkDir::from_config(cfg);
        let store = CheckpointStore::open(workdir.checkpoints_dir())?;
        Ok(Self {
            rcfg: cfg.regulate.clone().unwrap_or_default(),
            store,
            quarantine_path: workdir.root().join("quarantine.json"),
            log_path: workdir.root().join("evolution-log.jsonl"),
            workdir,
        })
    }

    /// The checkpoint store (for the CLI `checkpoints` commands).
    pub fn store(&self) -> &CheckpointStore {
        &self.store
    }

    /// Load the current quarantine.
    pub fn quarantine(&self) -> anyhow::Result<Quarantine> {
        Quarantine::load(&self.quarantine_path)
    }

    /// The live adapter dir this transaction snapshots/restores.
    fn adapter_dir(&self) -> PathBuf {
        self.workdir.root().join("adapter")
    }

    /// Run one transactional step.
    ///
    /// 1. Score the CURRENT (pre-step) model → baseline (the `last_good`'s
    ///    metrics if present, else a fresh baseline score).
    /// 2. Snapshot the current adapter into a `Pending` checkpoint.
    /// 3. Run `step` (mutates the adapter; returns its `gen` provenance).
    /// 4. Score the candidate; classify vs baseline.
    /// 5. `Accept` → commit + advance `last_good` + enforce retention.
    ///    `Regress` → restore the snapshot (the pre-step state) + mark Reverted.
    ///    `Catastrophic` → restore + quarantine the provenance + mark Quarantined
    ///    + signal HALT.
    /// 6. Append an evolution-log row in every case.
    ///
    /// `ordinal` is the monotonic step counter (no wall-clock — determinism).
    /// `id` is the checkpoint id (the caller chooses it, e.g. `step-<ordinal>`).
    pub fn run_step<StepFn, ScoreFn>(
        &self,
        id: &str,
        step_kind: &str,
        ordinal: u64,
        baseline: &ScoreReport,
        step: StepFn,
        score: ScoreFn,
    ) -> anyhow::Result<TxnOutcome>
    where
        StepFn: FnOnce() -> anyhow::Result<Vec<String>>,
        ScoreFn: Fn() -> anyhow::Result<ScoreReport>,
    {
        let parent_id = self.store.last_good();
        let adapter = self.adapter_dir();

        // (2) Snapshot the PRE-step adapter — the rollback target.
        let pre_snapshot_id = format!("{id}-pre");
        self.store.snapshot(
            &pre_snapshot_id,
            parent_id.clone(),
            &format!("{step_kind}:pre"),
            ordinal,
            &adapter,
            Vec::new(),
        )?;

        // (3) Run the step (mutates the adapter).
        let provenance = match step() {
            Ok(p) => p,
            Err(e) => {
                // Step itself failed: restore the pre-snapshot, log, no verdict.
                self.store.restore_adapter(&pre_snapshot_id, &adapter)?;
                let entry = EvolutionLogEntry {
                    step: ordinal,
                    checkpoint_id: id.to_string(),
                    kind: step_kind.to_string(),
                    verdict: None,
                    metrics: None,
                    action: StepAction::Rollback,
                    cause: Some(format!("step error: {e}")),
                };
                log::append(&self.log_path, &entry)?;
                return Ok(TxnOutcome {
                    checkpoint_id: id.to_string(),
                    verdict: None,
                    action: StepAction::Rollback,
                    halt: false,
                    metrics: None,
                });
            }
        };

        // Snapshot the POST-step adapter as the candidate checkpoint.
        self.store.snapshot(
            id,
            Some(pre_snapshot_id.clone()),
            step_kind,
            ordinal,
            &adapter,
            provenance.clone(),
        )?;

        // (4) Score the candidate + classify.
        let candidate = score()?;
        let verdict = classify(baseline, &candidate, &self.rcfg.tolerances())?;

        // (5) Act on the verdict.
        let (action, halt) = match verdict {
            StepVerdict::Accept => {
                self.store.commit(id, candidate.clone())?;
                self.store.enforce_retention(self.rcfg.keep_checkpoints)?;
                (StepAction::Commit, false)
            }
            StepVerdict::Regress => {
                self.store.restore_adapter(&pre_snapshot_id, &adapter)?;
                self.store.mark(id, CheckpointStatus::Reverted)?;
                (StepAction::Rollback, false)
            }
            StepVerdict::Catastrophic => {
                self.store.restore_adapter(&pre_snapshot_id, &adapter)?;
                self.store.mark(id, CheckpointStatus::Quarantined)?;
                // Quarantine the cause so the next round skips it.
                let mut q = self.quarantine()?;
                q.add(provenance.iter().cloned());
                q.write(&self.quarantine_path)?;
                (StepAction::Quarantine, true)
            }
        };

        // (6) Log it.
        let cause = match action {
            StepAction::Quarantine => Some(format!("quarantined provenance: {provenance:?}")),
            _ => None,
        };
        let entry = EvolutionLogEntry {
            step: ordinal,
            checkpoint_id: id.to_string(),
            kind: step_kind.to_string(),
            verdict: Some(verdict),
            metrics: Some(candidate.clone()),
            action,
            cause,
        };
        log::append(&self.log_path, &entry)?;

        Ok(TxnOutcome {
            checkpoint_id: id.to_string(),
            verdict: Some(verdict),
            action,
            halt,
            metrics: Some(candidate),
        })
    }
}
