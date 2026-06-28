//! Probe-correctness trend (track 31 Q4) — turning the evolution log into a
//! "is behavior actually changing?" signal.
//!
//! Loss falling per step does NOT mean the model's behavior changed — the real
//! signal is probe correctness over committed checkpoints. The track-15
//! evolution log already records a `ScoreReport` per step; this module reads that
//! series and summarizes its direction (rising / flat / falling) so the operator
//! can see whether ambient training is moving the needle. Pure + ML-free:
//! everything here is arithmetic over [`EvolutionLogEntry`] rows. See
//! `src/AGENTS.md` §trend.rs.

use crate::regulate::{EvolutionLogEntry, StepAction};

/// One point on the correctness trend: a committed step's correctness.
#[derive(Debug, Clone, PartialEq)]
pub struct TrendPoint {
    pub step: u64,
    pub checkpoint_id: String,
    pub correctness: f64,
}

/// The summary of a correctness series.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrendSummary {
    /// The committed-checkpoint correctness series, in step order.
    pub series: Vec<TrendPoint>,
    /// Mean delta between consecutive points (the average per-step change). `0`
    /// for a series of < 2 points.
    pub mean_delta: f64,
    /// First minus last (total change across the window). `0` for < 2 points.
    pub total_change: f64,
    /// Latest correctness, if any.
    pub latest: Option<f64>,
}

impl TrendSummary {
    /// A one-glyph direction indicator for a human display.
    pub fn arrow(&self) -> &'static str {
        if self.series.len() < 2 {
            "·" // not enough data
        } else if self.total_change > 0.01 {
            "↑"
        } else if self.total_change < -0.01 {
            "↓"
        } else {
            "→" // flat
        }
    }
}

/// Build the correctness trend from evolution-log entries. Only **committed**
/// steps count — a rolled-back step didn't change the kept model, so its score
/// isn't part of the model's actual trajectory. `last` (> 0) keeps only the most
/// recent N committed points.
pub fn from_log(entries: &[EvolutionLogEntry], last: usize) -> TrendSummary {
    let mut series: Vec<TrendPoint> = entries
        .iter()
        .filter(|e| e.action == StepAction::Commit)
        .filter_map(|e| {
            e.metrics.as_ref().map(|m| TrendPoint {
                step: e.step,
                checkpoint_id: e.checkpoint_id.clone(),
                correctness: m.correctness,
            })
        })
        .collect();

    if last > 0 && series.len() > last {
        series = series.split_off(series.len() - last);
    }

    let (mean_delta, total_change) = if series.len() < 2 {
        (0.0, 0.0)
    } else {
        let deltas: Vec<f64> = series
            .windows(2)
            .map(|w| w[1].correctness - w[0].correctness)
            .collect();
        let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
        let total = series.last().unwrap().correctness - series.first().unwrap().correctness;
        (mean, total)
    };
    let latest = series.last().map(|p| p.correctness);

    TrendSummary {
        series,
        mean_delta,
        total_change,
        latest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::ScoreReport;

    fn entry(step: u64, correctness: f64, action: StepAction) -> EvolutionLogEntry {
        let mut m = ScoreReport::uncovered("probe", "stub");
        m.correctness = correctness;
        m.n = 1;
        EvolutionLogEntry {
            step,
            checkpoint_id: format!("daemon-{step}"),
            kind: "daemon:microshard".into(),
            verdict: None,
            metrics: Some(m),
            action,
            cause: None,
        }
    }

    #[test]
    fn rising_series_reports_upward() {
        let log = vec![
            entry(1, 0.1, StepAction::Commit),
            entry(2, 0.3, StepAction::Commit),
            entry(3, 0.5, StepAction::Commit),
        ];
        let t = from_log(&log, 0);
        assert_eq!(t.series.len(), 3);
        assert!(t.total_change > 0.0);
        assert!(t.mean_delta > 0.0);
        assert_eq!(t.arrow(), "↑");
        assert_eq!(t.latest, Some(0.5));
    }

    #[test]
    fn rolled_back_steps_are_excluded() {
        let log = vec![
            entry(1, 0.5, StepAction::Commit),
            entry(2, 0.9, StepAction::Rollback), // a high score that didn't stick
            entry(3, 0.6, StepAction::Commit),
        ];
        let t = from_log(&log, 0);
        assert_eq!(
            t.series.len(),
            2,
            "only committed steps form the trajectory"
        );
        assert_eq!(t.latest, Some(0.6));
    }

    #[test]
    fn flat_series_reports_flat() {
        let log = vec![
            entry(1, 0.4, StepAction::Commit),
            entry(2, 0.4, StepAction::Commit),
        ];
        assert_eq!(from_log(&log, 0).arrow(), "→");
    }

    #[test]
    fn last_n_window_truncates() {
        let log: Vec<_> = (1..=10)
            .map(|i| entry(i, i as f64 / 10.0, StepAction::Commit))
            .collect();
        let t = from_log(&log, 3);
        assert_eq!(t.series.len(), 3);
        assert_eq!(t.series.first().unwrap().step, 8);
    }

    #[test]
    fn single_point_is_neutral() {
        let log = vec![entry(1, 0.4, StepAction::Commit)];
        let t = from_log(&log, 0);
        assert_eq!(t.mean_delta, 0.0);
        assert_eq!(t.total_change, 0.0);
        assert_eq!(t.arrow(), "·");
    }

    #[test]
    fn empty_log_is_empty() {
        let t = from_log(&[], 0);
        assert!(t.series.is_empty());
        assert_eq!(t.latest, None);
    }
}
