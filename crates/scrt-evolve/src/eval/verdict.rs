//! `StepVerdict` — the shared accept/regress/catastrophic decision (track 10).
//!
//! Given a baseline [`ScoreReport`] and a candidate one + per-metric tolerances,
//! classify the step. This is a **pure function** over two reports — the single
//! decision logic that track 11 (regen gate) and track 15 (self-regulation)
//! both call, so a "regression" means the same thing everywhere.
//!
//! Probe-version safety: a candidate is only comparable to a baseline scored on
//! the SAME probe version. A mismatch is a hard error, not a silent "accept"
//! (you'd be comparing exams from different syllabi).

use serde::{Deserialize, Serialize};

use super::score::ScoreReport;

/// The outcome of comparing a candidate round to its baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepVerdict {
    /// Candidate is at least as good (within tolerance) — keep it.
    Accept,
    /// Candidate regressed beyond tolerance but not catastrophically — roll back.
    Regress,
    /// Candidate collapsed (below the catastrophe floor, or NaN) — roll back +
    /// quarantine + halt (track 15).
    Catastrophic,
}

/// Per-metric tolerances/floors the consumer supplies (from its config). All are
/// on the **correctness** axis primarily; extend as more metrics gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VerdictTolerances {
    /// How much correctness may drop and still `Accept` (absolute, e.g. 0.02 =
    /// a 2-point drop is tolerated as noise).
    pub correctness_tolerance: f64,
    /// Absolute correctness floor: below this ⇒ `Catastrophic` regardless of
    /// the baseline (a collapse, e.g. 0.1 = 10% correct is a broken model).
    pub catastrophe_floor: f64,
}

impl Default for VerdictTolerances {
    fn default() -> Self {
        Self {
            correctness_tolerance: 0.02,
            catastrophe_floor: 0.10,
        }
    }
}

/// Errors from verdict classification.
#[derive(Debug, thiserror::Error)]
pub enum VerdictError {
    #[error(
        "probe-version mismatch: baseline scored on `{baseline}` but candidate on \
         `{candidate}` — reports are not comparable"
    )]
    ProbeVersionMismatch { baseline: String, candidate: String },
}

/// Classify a candidate against its baseline. Pure.
///
/// Rules (in order):
/// 1. Probe versions must match, else [`VerdictError::ProbeVersionMismatch`].
/// 2. NaN correctness, or correctness below `catastrophe_floor` ⇒ `Catastrophic`.
/// 3. correctness drop > `correctness_tolerance` ⇒ `Regress`.
/// 4. otherwise ⇒ `Accept`.
pub fn classify(
    baseline: &ScoreReport,
    candidate: &ScoreReport,
    tol: &VerdictTolerances,
) -> Result<StepVerdict, VerdictError> {
    if baseline.probe_version != candidate.probe_version {
        return Err(VerdictError::ProbeVersionMismatch {
            baseline: baseline.probe_version.clone(),
            candidate: candidate.probe_version.clone(),
        });
    }

    let cand = candidate.correctness;
    // Catastrophe: NaN (training blew up) or correctness collapsed below floor.
    if cand.is_nan() || cand < tol.catastrophe_floor {
        return Ok(StepVerdict::Catastrophic);
    }

    let drop = baseline.correctness - cand;
    if drop > tol.correctness_tolerance {
        return Ok(StepVerdict::Regress);
    }

    Ok(StepVerdict::Accept)
}
