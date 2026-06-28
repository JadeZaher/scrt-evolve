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
/// 0. An UNCOVERED baseline (`n == 0`, e.g. the very first version of a branch)
///    has no prior measurement to compare against — there is nothing to regress
///    from and no reference to define "collapse". The candidate is judged on its
///    own: `Catastrophic` only if its correctness is NaN (training blew up), else
///    `Accept`. The probe-version match is skipped (the sentinel baseline carries
///    no real probe).
/// 1. Otherwise probe versions must match, else [`VerdictError::ProbeVersionMismatch`].
/// 2. NaN correctness, or correctness below `catastrophe_floor` ⇒ `Catastrophic`.
/// 3. correctness drop > `correctness_tolerance` ⇒ `Regress`.
/// 4. otherwise ⇒ `Accept`.
pub fn classify(
    baseline: &ScoreReport,
    candidate: &ScoreReport,
    tol: &VerdictTolerances,
) -> Result<StepVerdict, VerdictError> {
    // Rule 0: no prior measurement ⇒ accept the first version unless it's NaN.
    if baseline.n == 0 {
        return Ok(if candidate.correctness.is_nan() {
            StepVerdict::Catastrophic
        } else {
            StepVerdict::Accept
        });
    }

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

/// The track-32 JUDGE gate verdict: degradation is primary, correctness is
/// demoted to catastrophe-only. Pure.
///
/// Rules (in order):
/// 1. `candidate_correctness` is NaN (training blew up) ⇒ `Catastrophic`.
/// 2. correctness is a hard COLLAPSE below `catastrophe_floor` (only when the
///    baseline was actually covered — an uncovered first version has no floor to
///    collapse against, mirroring [`classify`] rule 0) ⇒ `Catastrophic`.
/// 3. the judge's regressed fraction > `max_regressed_frac` ⇒ `Regress`.
/// 4. otherwise ⇒ `Accept` — "no degradation detected", the permissive default
///    that lets a weak model make progress on tiny data.
///
/// `regressed_fraction` comes from [`super::DegradationReport::regressed_fraction`]
/// (0.0 ⇒ nothing judged / nothing worse ⇒ accept). The correctness check exists
/// ONLY as the collapse backstop; it never drives accept/reject on its own here.
pub fn judge_verdict(
    baseline: &ScoreReport,
    candidate_correctness: f64,
    regressed_fraction: f64,
    tol: &VerdictTolerances,
    max_regressed_frac: f64,
) -> StepVerdict {
    if candidate_correctness.is_nan() {
        return StepVerdict::Catastrophic;
    }
    // Collapse is only meaningful against a covered baseline (rule 0 parity).
    if baseline.n > 0 && candidate_correctness < tol.catastrophe_floor {
        return StepVerdict::Catastrophic;
    }
    if regressed_fraction > max_regressed_frac {
        return StepVerdict::Regress;
    }
    StepVerdict::Accept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A covered report (`n > 0`) scored at `correctness` on probe `pv`.
    fn report(correctness: f64, pv: &str) -> ScoreReport {
        let mut r = ScoreReport::uncovered(pv, "test");
        r.correctness = correctness;
        r.n = 1;
        r
    }

    #[test]
    fn covered_same_probe_improvement_accepts() {
        let tol = VerdictTolerances::default();
        let baseline = report(0.70, "probe-vA");
        let candidate = report(0.80, "probe-vA");
        assert_eq!(
            classify(&baseline, &candidate, &tol).unwrap(),
            StepVerdict::Accept
        );
    }

    #[test]
    fn covered_same_probe_within_tolerance_accepts() {
        // A drop inside `correctness_tolerance` (0.02) is noise, not a regress.
        let tol = VerdictTolerances::default();
        let baseline = report(0.80, "probe-vA");
        let candidate = report(0.79, "probe-vA");
        assert_eq!(
            classify(&baseline, &candidate, &tol).unwrap(),
            StepVerdict::Accept
        );
    }

    #[test]
    fn covered_same_probe_regression_rolls_back() {
        let tol = VerdictTolerances::default();
        let baseline = report(0.80, "probe-vA");
        let candidate = report(0.50, "probe-vA"); // beyond tolerance, above floor
        assert_eq!(
            classify(&baseline, &candidate, &tol).unwrap(),
            StepVerdict::Regress
        );
    }

    #[test]
    fn covered_collapse_below_floor_is_catastrophic() {
        let tol = VerdictTolerances::default();
        let baseline = report(0.80, "probe-vA");
        let candidate = report(0.05, "probe-vA"); // below catastrophe_floor (0.10)
        assert_eq!(
            classify(&baseline, &candidate, &tol).unwrap(),
            StepVerdict::Catastrophic
        );
    }

    #[test]
    fn covered_different_probe_is_a_hard_error() {
        // The exact bug stable probes fix: comparing two covered reports scored
        // on different exams must NOT silently accept — it errors.
        let tol = VerdictTolerances::default();
        let baseline = report(0.80, "probe-vA");
        let candidate = report(0.90, "probe-vB");
        assert!(matches!(
            classify(&baseline, &candidate, &tol),
            Err(VerdictError::ProbeVersionMismatch { .. })
        ));
    }

    #[test]
    fn uncovered_baseline_accepts_first_version_and_skips_probe_check() {
        // First round: no prior measurement ⇒ accept unless NaN, even with a
        // mismatched probe id (the sentinel baseline carries no real probe).
        let tol = VerdictTolerances::default();
        let baseline = ScoreReport::uncovered("probe-none", "baseline");
        let candidate = report(0.30, "probe-vA");
        assert_eq!(
            classify(&baseline, &candidate, &tol).unwrap(),
            StepVerdict::Accept
        );

        let mut nan = report(f64::NAN, "probe-vA");
        nan.correctness = f64::NAN;
        assert_eq!(
            classify(&baseline, &nan, &tol).unwrap(),
            StepVerdict::Catastrophic
        );
    }

    // ───────────────── Track 32: judge_verdict (degradation gate) ─────────────────

    #[test]
    fn judge_accepts_when_no_degradation_even_if_correctness_low() {
        // The whole point: a low absolute correctness (0.30) does NOT block the
        // step under the judge gate — only degradation does.
        let tol = VerdictTolerances::default();
        let baseline = report(0.40, "probe-vA");
        assert_eq!(
            judge_verdict(&baseline, 0.30, 0.0, &tol, 0.0),
            StepVerdict::Accept
        );
    }

    #[test]
    fn judge_regresses_when_degradation_exceeds_threshold() {
        let tol = VerdictTolerances::default();
        let baseline = report(0.40, "probe-vA");
        // 20% of items got worse, threshold 0.0 ⇒ regress.
        assert_eq!(
            judge_verdict(&baseline, 0.40, 0.2, &tol, 0.0),
            StepVerdict::Regress
        );
        // Same degradation tolerated when the threshold allows it.
        assert_eq!(
            judge_verdict(&baseline, 0.40, 0.2, &tol, 0.25),
            StepVerdict::Accept
        );
    }

    #[test]
    fn judge_catastrophe_on_nan_and_collapse() {
        let tol = VerdictTolerances::default();
        let baseline = report(0.40, "probe-vA");
        // NaN always catastrophic, regardless of the judge.
        assert_eq!(
            judge_verdict(&baseline, f64::NAN, 0.0, &tol, 0.0),
            StepVerdict::Catastrophic
        );
        // Hard collapse below the floor is still catastrophic under the judge gate.
        assert_eq!(
            judge_verdict(&baseline, 0.05, 0.0, &tol, 0.0),
            StepVerdict::Catastrophic
        );
    }

    #[test]
    fn judge_uncovered_baseline_has_no_collapse_floor() {
        // First version (uncovered baseline): a low score is NOT a collapse (no
        // reference), so only NaN or actual degradation can block it.
        let tol = VerdictTolerances::default();
        let baseline = ScoreReport::uncovered("probe-none", "baseline");
        assert_eq!(
            judge_verdict(&baseline, 0.05, 0.0, &tol, 0.0),
            StepVerdict::Accept
        );
        assert_eq!(
            judge_verdict(&baseline, f64::NAN, 0.0, &tol, 0.0),
            StepVerdict::Catastrophic
        );
    }
}
