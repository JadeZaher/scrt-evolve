//! Multi-goal driver — the **buildable** half of learning-by-doing (track 20).
//!
//! This is the orchestration layer for the per-goal pipeline that runs on
//! ALREADY-SHIPPED tracks (01 discover + palace-search, 02 generate): for each
//! `[[goals]]` entry, derive a goal-scoped config ([`EvolveConfig::for_goal`])
//! and run **discover → generate**, writing per-goal artifacts under
//! `work_dir/goals/<name>/`.
//!
//! What is intentionally NOT here yet (lane-gated, slices 6–10):
//! - **eval-gating / keep|rollback** — needs track 10 (`Scorer`) + track 15
//!   (the transactional wrapper). No weights are mutated by this driver.
//! - **catastrophe/quarantine, the regen flywheel, the scheduler** — also
//!   lane-gated. This driver is their future consumer, not their owner.
//!
//! Until that lane lands, this is a bounded, non-mutating fan-out over goals:
//! safe to run, produces per-goal datasets a human can inspect, and gives the
//! eventual round driver its inputs. The loop is bounded by `cfg.goals.len()`
//! (no unbounded `while` — styleguide §2.5).

use std::path::PathBuf;

use crate::config::EvolveConfig;
use crate::dataset::Dataset;
use crate::discover::DiscoveredContext;
use crate::workdir::WorkDir;

/// Per-goal outcome of one buildable pipeline pass.
#[derive(Debug, Clone)]
pub struct GoalRun {
    /// The goal's `name`.
    pub goal: String,
    /// Passages discovered from the goal's tagged stashes.
    pub passages: usize,
    /// Rows generated for the goal (`0` if generate was skipped/failed).
    pub rows: usize,
    /// Where the goal's dataset landed (`None` if generate did not run).
    pub dataset_path: Option<PathBuf>,
    /// A per-goal status/error note for the human (generate is best-effort so
    /// one goal's API failure does not abort the others).
    pub note: String,
}

/// The result of a multi-goal buildable session.
#[derive(Debug, Clone, Default)]
pub struct GoalsReport {
    pub runs: Vec<GoalRun>,
}

impl GoalsReport {
    /// Total rows generated across all goals.
    pub fn total_rows(&self) -> usize {
        self.runs.iter().map(|r| r.rows).sum()
    }
}

/// Run the buildable per-goal pipeline (discover → generate) over every goal in
/// `cfg.goals`, writing per-goal artifacts under `work_dir/goals/<name>/`.
///
/// Bounded by the number of declared goals. Discover failures for a goal are
/// fatal to that goal only (recorded in its [`GoalRun::note`]); generate is
/// best-effort (an API/network failure for one goal does not abort the rest).
/// **No weight mutation, no eval gate** — that is lane-gated (slices 6–10).
///
/// `generate` selects a fn so tests can inject a deterministic, network-free
/// generator; production passes [`crate::generate::run`].
pub fn run_buildable(
    cfg: &EvolveConfig,
    generate: impl Fn(&EvolveConfig, &DiscoveredContext) -> anyhow::Result<Dataset>,
) -> anyhow::Result<GoalsReport> {
    if cfg.goals.is_empty() {
        anyhow::bail!(
            "evolve --goals: no `[[goals]]` declared in the config. Add a goal \
             (name/topic/tag) or run the single-run pipeline (`run`)."
        );
    }

    let root_wd = WorkDir::from_config(cfg);
    let mut report = GoalsReport::default();

    for goal in &cfg.goals {
        let per_goal_cfg = cfg.for_goal(goal);

        // Per-goal artifacts live under work_dir/goals/<name>/.
        let goal_dir = root_wd.root().join("goals").join(&goal.name);
        if let Err(e) = std::fs::create_dir_all(&goal_dir) {
            report.runs.push(GoalRun {
                goal: goal.name.clone(),
                passages: 0,
                rows: 0,
                dataset_path: None,
                note: format!("failed to create {}: {e}", goal_dir.display()),
            });
            continue;
        }

        // --- Discover (goal-tagged stashes only). ---
        let ctx = match crate::discover::run(&per_goal_cfg) {
            Ok(ctx) => ctx,
            Err(e) => {
                report.runs.push(GoalRun {
                    goal: goal.name.clone(),
                    passages: 0,
                    rows: 0,
                    dataset_path: None,
                    note: format!("discover failed: {e}"),
                });
                continue;
            }
        };

        // Persist the discovered context for inspection / a later round driver.
        let disc_path = goal_dir.join("discovered.json");
        if let Ok(json) = serde_json::to_string_pretty(&ctx) {
            let _ = crate::harvest::write_atomic(&disc_path, json.as_bytes());
        }

        // --- Generate (best-effort — network may be unavailable). ---
        match generate(&per_goal_cfg, &ctx) {
            Ok(dataset) => {
                let out = goal_dir.join("dataset.jsonl");
                if let Err(e) = dataset.write_jsonl(&out) {
                    report.runs.push(GoalRun {
                        goal: goal.name.clone(),
                        passages: ctx.passages.len(),
                        rows: dataset.len(),
                        dataset_path: None,
                        note: format!("generated {} rows but write failed: {e}", dataset.len()),
                    });
                    continue;
                }
                report.runs.push(GoalRun {
                    goal: goal.name.clone(),
                    passages: ctx.passages.len(),
                    rows: dataset.len(),
                    dataset_path: Some(out),
                    note: "ok".to_string(),
                });
            }
            Err(e) => {
                report.runs.push(GoalRun {
                    goal: goal.name.clone(),
                    passages: ctx.passages.len(),
                    rows: 0,
                    dataset_path: None,
                    note: format!("generate skipped/failed: {e}"),
                });
            }
        }
    }

    Ok(report)
}
