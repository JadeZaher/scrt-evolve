//! Eval-gated round driver + scheduler (track 20 slices 6–9).
//!
//! This is the capstone: it wires together everything below it into a
//! **bounded, eval-gated, multi-goal training schedule** — the thing the DESIGN
//! daemon drives, made safe.
//!
//! One **round** for a goal (slice 6):
//! ```text
//!   discover(goal tags) → generate (quarantine-filtered) → probe build
//!     → [ TRANSACTION (track 15):  train → eval(probe) → keep | rollback ]
//! ```
//! A **catastrophe** (slice 7) rolls back, quarantines the round's `gen`
//! provenance, and HALTS the schedule (re-arm required). A soft regress rolls
//! back just that round and the schedule continues.
//!
//! The **scheduler** (slice 9) loops rounds across goals — round-robin or
//! weighted by `goal.weight` — bounded by an explicit budget (max rounds), and
//! is resumable (it reads `last_good`/quarantine/round-cursor from the work-dir,
//! so an interrupted schedule picks up where it left off).
//!
//! Weight mutation happens ONLY inside `Regulator::run_step` (track 15). The
//! heavy train + score are **injected closures**: production wires them to the
//! Python subprocess (`train --backend transformers`, `eval --scorer transformers`);
//! tests inject deterministic ones so the whole driver is provable ML-free.

use crate::config::{EvolveConfig, GoalConfig};
use crate::dataset::Dataset;
use crate::discover::DiscoveredContext;
use crate::eval::ScoreReport;
use crate::regulate::{Regulator, StepAction};
use crate::workdir::WorkDir;

/// The result of one eval-gated round for a goal.
#[derive(Debug, Clone)]
pub struct RoundReport {
    pub goal: String,
    /// The round ordinal (monotonic across the whole schedule).
    pub ordinal: u64,
    /// Passages discovered this round.
    pub passages: usize,
    /// Rows generated this round (after quarantine filtering).
    pub rows: usize,
    /// The action the transaction took (`Commit` | `Rollback` | `Quarantine`),
    /// or `None` if the round bailed before the transaction (e.g. no rows).
    pub action: Option<StepAction>,
    /// The eval metrics for the round, if it reached eval.
    pub metrics: Option<ScoreReport>,
    /// Whether this round halted the schedule (catastrophe).
    pub halt: bool,
    /// A human-facing status note.
    pub note: String,
}

/// The injected effects a round needs — kept as a trait object-free struct of
/// closures so production (subprocess) and tests (deterministic) share the
/// driver. All are goal-scoped: the driver passes the per-goal config.
pub struct RoundHooks<'a> {
    /// Discover goal-tagged context. Production: `discover::run`.
    pub discover: &'a dyn Fn(&EvolveConfig) -> anyhow::Result<DiscoveredContext>,
    /// Generate a dataset from discovered context. Production: `generate::run`.
    pub generate: &'a dyn Fn(&EvolveConfig, &DiscoveredContext) -> anyhow::Result<Dataset>,
    /// Train the adapter on the round's (probe-carved) training set — the
    /// weight-mutating step. Returns the `gen` provenance of its training rows
    /// (the quarantine key). Production: shell out to the transformers trainer.
    pub train: &'a dyn Fn(&EvolveConfig, &Dataset) -> anyhow::Result<Vec<String>>,
    /// Score the current model against the goal's probe. Production: `eval::run_eval`.
    pub score: &'a dyn Fn(&EvolveConfig) -> anyhow::Result<ScoreReport>,
}

/// Run ONE eval-gated round for a goal (slice 6 + 7). Returns its report.
///
/// `ordinal` is the schedule-wide monotonic step counter. `baseline` is the
/// score to compare the round's candidate against (the last good score, or a
/// fresh pre-round score). The round:
/// 1. discovers the goal's tagged context,
/// 2. generates a dataset, FILTERS OUT quarantined provenance (slice 7), carves
///    a held-out probe + training remainder,
/// 3. runs `train → eval → keep|rollback` inside the track-15 transaction.
pub fn run_round(
    cfg: &EvolveConfig,
    goal: &GoalConfig,
    ordinal: u64,
    baseline: &ScoreReport,
    hooks: &RoundHooks,
) -> anyhow::Result<RoundReport> {
    let per_goal = cfg.for_goal(goal);
    let reg = Regulator::new(cfg)?;

    let bail = |note: String| RoundReport {
        goal: goal.name.clone(),
        ordinal,
        passages: 0,
        rows: 0,
        action: None,
        metrics: None,
        halt: false,
        note,
    };

    // (1) Discover.
    let ctx = match (hooks.discover)(&per_goal) {
        Ok(c) => c,
        Err(e) => return Ok(bail(format!("discover failed: {e}"))),
    };
    if ctx.passages.is_empty() {
        return Ok(bail(
            "no passages discovered (no goal-tagged stashes?)".into(),
        ));
    }

    // (2) Generate, then drop any quarantined provenance.
    let dataset = match (hooks.generate)(&per_goal, &ctx) {
        Ok(d) => d,
        Err(e) => {
            return Ok(RoundReport {
                passages: ctx.passages.len(),
                ..bail(format!("generate failed: {e}"))
            })
        }
    };
    let quarantine = reg.quarantine()?;
    let (dataset, dropped) = quarantine.filter(&dataset);
    if dropped > 0 {
        eprintln!(
            "round[{}]: dropped {dropped} quarantined row(s) before training",
            goal.name
        );
    }
    if dataset.is_empty() {
        return Ok(RoundReport {
            passages: ctx.passages.len(),
            ..bail("dataset empty after quarantine filter".into())
        });
    }

    // Carve a held-out probe + training remainder (track 10). Falls back to
    // training on everything (no gate) if carving yields an empty probe.
    let frac = cfg
        .eval
        .as_ref()
        .map(|e| e.probe_holdout_frac)
        .unwrap_or(0.1);
    let (probe, train_set) = crate::eval::ProbeSet::carve(&dataset, frac)?;
    let wd = WorkDir::from_config(cfg);
    // Persist this round's probe + training set under the goal dir.
    let goal_dir = wd.root().join("goals").join(&goal.name);
    std::fs::create_dir_all(&goal_dir)?;
    let _ = probe.write(crate::eval::probe_path(cfg));
    let train_path = goal_dir.join("dataset.train.jsonl");
    let _ = train_set.write_jsonl(&train_path);

    // (3) The transaction: train (mutates adapter) → eval → keep|rollback.
    let rows = train_set.len();
    let step_kind = format!("round:{}", goal.name);
    let id = format!("round-{ordinal}-{}", goal.name);

    let outcome = reg.run_step(
        &id,
        &step_kind,
        ordinal,
        baseline,
        || (hooks.train)(&per_goal, &train_set),
        || (hooks.score)(&per_goal),
    )?;

    let note = match outcome.action {
        StepAction::Commit => "kept (eval passed)".to_string(),
        StepAction::Rollback => "rolled back (regress)".to_string(),
        StepAction::Quarantine => "CATASTROPHE — rolled back + quarantined + halt".to_string(),
    };

    Ok(RoundReport {
        goal: goal.name.clone(),
        ordinal,
        passages: ctx.passages.len(),
        rows,
        action: Some(outcome.action),
        metrics: outcome.metrics,
        halt: outcome.halt,
        note,
    })
}

/// Scheduling policy across goals (slice 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePolicy {
    /// Visit goals in declared order, one round each, repeating.
    RoundRobin,
    /// Visit goals proportionally to `goal.weight` (higher weight ⇒ more rounds).
    Weighted,
}

/// The bounded multi-goal schedule report.
#[derive(Debug, Clone, Default)]
pub struct ScheduleReport {
    pub rounds: Vec<RoundReport>,
    /// Whether the schedule halted early on a catastrophe.
    pub halted: bool,
}

impl ScheduleReport {
    /// Count rounds whose transaction action was `Commit`.
    pub fn committed(&self) -> usize {
        self.rounds
            .iter()
            .filter(|r| r.action == Some(StepAction::Commit))
            .count()
    }
}

/// Run a **bounded** multi-goal schedule (slice 9).
///
/// `max_rounds` is the hard budget (no unbounded loop — styleguide §2.5). The
/// schedule visits goals per `policy`, runs one eval-gated round each, and HALTS
/// immediately if any round catastrophes. Resumable: `start_ordinal` lets a
/// caller continue a prior schedule (the work-dir holds the durable
/// last_good/quarantine state the rounds read).
///
/// `baseline_for` supplies the comparison baseline for a goal's round (typically
/// the goal's last good score, or a fresh pre-schedule score).
pub fn run_schedule(
    cfg: &EvolveConfig,
    policy: SchedulePolicy,
    max_rounds: usize,
    start_ordinal: u64,
    hooks: &RoundHooks,
    baseline_for: &dyn Fn(&GoalConfig) -> ScoreReport,
) -> anyhow::Result<ScheduleReport> {
    if cfg.goals.is_empty() {
        anyhow::bail!("schedule: no [[goals]] declared");
    }
    let order = goal_order(&cfg.goals, policy, max_rounds);

    let mut report = ScheduleReport::default();
    let mut ordinal = start_ordinal;
    for goal in order {
        if report.rounds.len() >= max_rounds {
            break;
        }
        let baseline = baseline_for(goal);
        let round = run_round(cfg, goal, ordinal, &baseline, hooks)?;
        let halt = round.halt;
        report.rounds.push(round);
        ordinal += 1;
        if halt {
            report.halted = true;
            eprintln!("schedule: HALTED on catastrophe at ordinal {ordinal}");
            break;
        }
    }
    Ok(report)
}

/// Build the goal visitation order for a schedule, bounded by `max_rounds`.
/// Round-robin cycles goals; weighted repeats each goal ~proportional to its
/// weight. Deterministic (no RNG).
fn goal_order(goals: &[GoalConfig], policy: SchedulePolicy, max_rounds: usize) -> Vec<&GoalConfig> {
    let mut order: Vec<&GoalConfig> = Vec::new();
    match policy {
        SchedulePolicy::RoundRobin => {
            // Repeat the declared sequence until the budget is met.
            while order.len() < max_rounds {
                for g in goals {
                    if order.len() >= max_rounds {
                        break;
                    }
                    order.push(g);
                }
            }
        }
        SchedulePolicy::Weighted => {
            // Integer "quota" per goal from its weight (default 1.0), scaled so
            // the total is ~max_rounds; then interleave for fairness.
            let weights: Vec<f32> = goals
                .iter()
                .map(|g| g.weight.unwrap_or(1.0).max(0.0))
                .collect();
            let total: f32 = weights.iter().sum();
            let total = if total <= 0.0 {
                goals.len() as f32
            } else {
                total
            };
            let mut quota: Vec<usize> = weights
                .iter()
                .map(|w| ((w / total) * max_rounds as f32).round() as usize)
                .collect();
            // Ensure at least the budget is reachable: top up the heaviest goal.
            let mut assigned: usize = quota.iter().sum();
            if assigned == 0 {
                if let Some(q) = quota.first_mut() {
                    *q = max_rounds;
                    assigned = max_rounds;
                }
            }
            let _ = assigned;
            // Interleave round-robin but only from goals with remaining quota.
            while order.len() < max_rounds && quota.iter().any(|&q| q > 0) {
                for (i, g) in goals.iter().enumerate() {
                    if order.len() >= max_rounds {
                        break;
                    }
                    if quota[i] > 0 {
                        order.push(g);
                        quota[i] -= 1;
                    }
                }
            }
        }
    }
    order.truncate(max_rounds);
    order
}
