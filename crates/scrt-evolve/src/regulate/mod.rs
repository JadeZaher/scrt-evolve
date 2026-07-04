//! Self-regulation (track 15) — the transactional homeostasis layer.
//!
//! Makes the self-evolve lane safe to run unattended: every weight-mutating step
//! is `checkpoint → apply → eval → keep|rollback`, and a catastrophe triggers
//! `rollback + quarantine + halt`. This is the recursion's base case (grow →
//! evaluate → keep-or-revert) and the ONLY sanctioned weight-mutation path — the
//! scheduler/daemon may auto-evolve only THROUGH [`txn::Regulator`].
//!
//! - [`checkpoint::CheckpointStore`] — snapshot/restore the adapter + manifests +
//!   a `last_good` pointer (atomic, retention-bounded).
//! - [`quarantine::Quarantine`] — skip a catastrophic cause by its `gen`
//!   provenance stamp (styleguide §2.4).
//! - [`log`] — the `evolution-log.jsonl` audit trail (commit/rollback/quarantine).
//! - [`txn::Regulator`] — ties them together; consumes track 10's `ScoreReport`
//!   + `StepVerdict`.
//!
//! Self-pruning (expert eviction, gated base pruning) is a **documented seam**:
//! it depends on tracks 11–14 (attribution/experts) which are not built and are
//! not needed for the eval-gated training schedule. The transaction machinery
//! here is exactly what a future prune step would run inside.

pub mod checkpoint;
pub mod log;
pub mod quarantine;
pub mod txn;

pub use checkpoint::{CheckpointManifest, CheckpointStatus, CheckpointStore};
pub use log::{EvolutionLogEntry, StepAction};
pub use quarantine::Quarantine;
pub use txn::{Regulator, TxnOutcome};
