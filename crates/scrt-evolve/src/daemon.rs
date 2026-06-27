//! Ambient continuous-evolution daemon (track 26) — the always-on,
//! VRAM-bounded background trainer.
//!
//! Where the schedule (track 20 [`crate::rounds`]) runs a *bounded batch* of
//! eval-gated rounds, the daemon runs *continuously*: it pops the user's living
//! activity ([`crate::living_queue`]) one microshard at a time and folds each
//! into the model — but ONLY when there's free VRAM, and ONLY through the
//! track-15 transaction, so ambient training can never silently degrade the
//! model. "Training can almost always be happening as a background task, bounded
//! by VRAM; data is updated dynamically by user activity."
//!
//! Per step:
//! 1. **stop check** — explicit `daemon stop` drops a stop-file; honored at the
//!    top of every iteration (no signals — works on Windows + WSL alike).
//! 2. **VRAM gate** — if a free-VRAM probe is available and free < the budget,
//!    WAIT (self-throttle around the user's other GPU use) instead of training.
//! 3. **pop** a batch (priority lane first), drop quarantined provenance.
//! 4. **transaction** ([`Regulator::run_step`]) — checkpoint → train → eval →
//!    keep|rollback; a catastrophe rolls back + quarantines + HALTS.
//! 5. record a durable step report.
//!
//! The heavy `train`/`score`/`free_vram` effects are **injected closures** (same
//! pattern as [`crate::rounds::RoundHooks`]): production wires them to the Python
//! subprocess + an `nvidia-smi` probe; tests inject deterministic ones, so the
//! whole loop — gating, draining, transactional commit, catastrophe-halt — is
//! provable ML-free and GPU-free. The loop itself is clock-free except the
//! production-only throttle sleep, which tests never reach (they supply VRAM and
//! a bounded queue).

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::EvolveConfig;
use crate::dataset::Dataset;
use crate::eval::ScoreReport;
use crate::living_queue::{Lane, LivingQueue};
use crate::regulate::{Regulator, StepAction};
use crate::workdir::WorkDir;

/// The injected effects the daemon needs. Kept closure-based so production
/// (subprocess + GPU probe) and tests (deterministic) share one loop.
pub struct DaemonHooks<'a> {
    /// Probe free VRAM in GB. `None` ⇒ unknown (the VRAM gate is skipped — we
    /// can't throttle what we can't measure). Production: parse `nvidia-smi`.
    pub free_vram_gb: &'a dyn Fn() -> Option<f64>,
    /// Probe whether ANOTHER process is using the GPU. `Some(true)` ⇒ yield the
    /// GPU (gentle-background: don't contend with a game/video). `None` ⇒ unknown
    /// (treated as not-busy). Production: `nvidia-smi --query-compute-apps=pid`
    /// minus our own PID.
    pub gpu_busy: &'a dyn Fn() -> Option<bool>,
    /// Train on the step's queued batch — the weight-mutating effect. Returns the
    /// batch's `gen` provenance (the quarantine key). Production: the transformers
    /// trainer (one microshard, track-25 `granularity=module`).
    pub train: &'a dyn Fn(&EvolveConfig, &Dataset) -> anyhow::Result<Vec<String>>,
    /// Score the current model against the probe. Production: `eval::run_eval`.
    pub score: &'a dyn Fn(&EvolveConfig) -> anyhow::Result<ScoreReport>,
}

/// Tunables for a daemon run. `[daemon]` config + CLI flags populate these.
#[derive(Debug, Clone)]
pub struct DaemonOptions {
    /// VRAM budget in GB: the daemon trains only when at least this much is FREE
    /// (so it never OOMs the user's foreground work). `None` ⇒ ungated.
    pub max_vram_gb: Option<f64>,
    /// Queued items folded into one microshard step.
    pub batch: usize,
    /// Stop after this many transactional steps (`None` ⇒ until stopped). Bounds
    /// tests and `daemon start --max-steps N`.
    pub max_steps: Option<u64>,
    /// When the queue drains: `true` ⇒ exit (drain-once mode); `false` ⇒ wait for
    /// new activity (the long-running `daemon start`).
    pub exit_when_empty: bool,
    /// How long to wait when throttled or idle (production only; tests don't hit
    /// this path).
    pub poll_interval: Duration,
    /// The monotonic step ordinal to start at (resume point).
    pub start_ordinal: u64,
    /// Gentle-background: pause GPU training when another process is on the GPU.
    pub pause_on_gpu_process: bool,
    /// When the GPU is unavailable, fall back to a CPU step instead of pausing.
    pub cpu_fallback: bool,
    /// Train one block per step and rotate which block (`ordinal % rotation_blocks`).
    /// `0` ⇒ no rotation (train the whole adapter set each step).
    pub rotation_blocks: usize,
    /// Sleep after each executed step to cap GPU duty cycle (production only).
    pub cooldown: Duration,
}

impl Default for DaemonOptions {
    fn default() -> Self {
        Self {
            max_vram_gb: None,
            batch: 1,
            max_steps: None,
            exit_when_empty: false,
            poll_interval: Duration::from_secs(30),
            start_ordinal: 1,
            pause_on_gpu_process: true,
            cpu_fallback: true,
            rotation_blocks: 0,
            cooldown: Duration::from_secs(0),
        }
    }
}

/// Per-step placement decision — a PURE function of the resource probes, so the
/// adaptive policy is unit-testable without a GPU. `Train` carries the target
/// device + (optional) rotating block index; `Wait` carries a log reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepDecision {
    Train {
        device: &'static str,
        shard: Option<usize>,
    },
    Wait(String),
}

/// Decide a step's placement from free-VRAM + other-GPU-process probes and the
/// gentle-background options. GPU when free; CPU when the GPU is busy/starved and
/// `cpu_fallback`; otherwise wait. Block index rotates when `rotation_blocks > 0`.
pub fn decide_step(
    ordinal: u64,
    free_vram_gb: Option<f64>,
    gpu_busy: Option<bool>,
    opts: &DaemonOptions,
) -> StepDecision {
    let shard = if opts.rotation_blocks > 0 {
        Some((ordinal as usize) % opts.rotation_blocks)
    } else {
        None
    };
    let vram_ok = match (opts.max_vram_gb, free_vram_gb) {
        (Some(budget), Some(free)) => free >= budget,
        // Ungated or unmeasurable ⇒ don't block on VRAM.
        _ => true,
    };
    let other_gpu = gpu_busy.unwrap_or(false);
    let gpu_ok = vram_ok && !(opts.pause_on_gpu_process && other_gpu);
    if gpu_ok {
        StepDecision::Train {
            device: "cuda",
            shard,
        }
    } else if opts.cpu_fallback {
        StepDecision::Train {
            device: "cpu",
            shard,
        }
    } else {
        let why = if opts.pause_on_gpu_process && other_gpu {
            "another GPU process active"
        } else {
            "free VRAM below budget"
        };
        StepDecision::Wait(format!("paused: {why}"))
    }
}

/// Build a per-step config carrying the placement plan: set `[hardware].device`
/// and, when rotating, the `[train.fractional]` block index. Clone-and-mutate so
/// the daemon's shared config is untouched.
fn apply_plan(
    cfg: &EvolveConfig,
    device: &str,
    shard: Option<usize>,
    rotation_blocks: usize,
    granularity: &str,
) -> EvolveConfig {
    let mut c = cfg.clone();
    let mut hw = c.hardware.clone().unwrap_or_default();
    hw.device = device.to_string();
    c.hardware = Some(hw);
    if let Some(idx) = shard {
        let mut tr = c.train.clone().unwrap_or_default();
        let mut frac = tr.fractional.clone().unwrap_or_default();
        frac.enabled = true;
        frac.shards = Some(rotation_blocks);
        frac.shard_index = Some(idx);
        frac.granularity = granularity.to_string();
        tr.fractional = Some(frac);
        c.train = Some(tr);
    }
    c
}

/// One executed (or skipped) daemon step.
#[derive(Debug, Clone)]
pub struct DaemonStep {
    pub ordinal: u64,
    /// Items consumed this step.
    pub items: usize,
    /// The transaction action (`Commit`/`Rollback`/`Quarantine`), or `None` if
    /// the step bailed before the transaction (all rows quarantined).
    pub action: Option<StepAction>,
    pub metrics: Option<ScoreReport>,
    pub halt: bool,
    pub note: String,
}

/// The outcome of a daemon run.
#[derive(Debug, Clone, Default)]
pub struct DaemonReport {
    pub steps: Vec<DaemonStep>,
    /// A catastrophe halted the run (re-arm required).
    pub halted: bool,
    /// A stop was requested (clean shutdown).
    pub stopped: bool,
    /// The queue drained and `exit_when_empty` was set.
    pub drained: bool,
}

impl DaemonReport {
    /// How many steps committed (kept).
    pub fn committed(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.action == Some(StepAction::Commit))
            .count()
    }
}

/// The stop-file the running daemon polls. `daemon stop` creates it; the loop
/// removes it on a clean exit.
pub fn stop_file(work_dir: &Path) -> PathBuf {
    work_dir.join("daemon.stop")
}

/// The run marker written while the daemon is active (for `daemon status`).
pub fn run_file(work_dir: &Path) -> PathBuf {
    work_dir.join("daemon.run")
}

/// Request the running daemon stop (the `daemon stop` command): drop the
/// stop-file the loop checks each iteration.
pub fn request_stop(work_dir: &Path) -> anyhow::Result<()> {
    std::fs::write(stop_file(work_dir), b"stop")?;
    Ok(())
}

/// True if a stop has been requested.
pub fn stop_requested(work_dir: &Path) -> bool {
    stop_file(work_dir).exists()
}

/// Run the ambient daemon loop. `should_stop` is injected so production checks
/// the stop-file while tests terminate deterministically.
pub fn run_daemon(
    cfg: &EvolveConfig,
    opts: &DaemonOptions,
    baseline: &ScoreReport,
    hooks: &DaemonHooks,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<DaemonReport> {
    let queue = LivingQueue::from_config(cfg)?;
    let reg = Regulator::new(cfg)?;
    let wd = WorkDir::from_config(cfg);
    let log_dir = wd.root().join("logs");
    let _ = std::fs::create_dir_all(&log_dir);

    let mut report = DaemonReport::default();
    let mut ordinal = opts.start_ordinal;
    let mut taken = 0u64;

    loop {
        if let Some(max) = opts.max_steps {
            if taken >= max {
                break;
            }
        }
        if should_stop() {
            report.stopped = true;
            break;
        }

        // (2) Adaptive gate: GPU when free, CPU when the GPU is busy/starved (if
        // cpu_fallback), else wait — yielding the GPU to the user's foreground work.
        let decision = decide_step(ordinal, (hooks.free_vram_gb)(), (hooks.gpu_busy)(), opts);
        let (device, shard) = match &decision {
            StepDecision::Train { device, shard } => (*device, *shard),
            StepDecision::Wait(reason) => {
                if opts.exit_when_empty {
                    // Bounded/drain mode never blocks — report + stop.
                    report.steps.push(DaemonStep {
                        ordinal,
                        items: 0,
                        action: None,
                        metrics: None,
                        halt: false,
                        note: reason.clone(),
                    });
                    break;
                }
                std::thread::sleep(opts.poll_interval);
                continue;
            }
        };

        // (3) Pop a batch (priority-first); drop quarantined provenance.
        let items = queue.pop_batch(opts.batch)?;
        if items.is_empty() {
            report.drained = true;
            if opts.exit_when_empty {
                break;
            }
            std::thread::sleep(opts.poll_interval);
            continue;
        }
        let lane_note = if items.iter().any(|i| i.lane == Lane::Priority) {
            "priority"
        } else {
            "raw"
        };
        let dataset = Dataset::new(items.iter().map(|i| i.example.clone()).collect());
        let quarantine = reg.quarantine()?;
        let (dataset, dropped) = quarantine.filter(&dataset);
        if dataset.is_empty() {
            report.steps.push(DaemonStep {
                ordinal,
                items: items.len(),
                action: None,
                metrics: None,
                halt: false,
                note: format!("all {} row(s) quarantined — skipped", items.len()),
            });
            ordinal += 1;
            taken += 1;
            continue;
        }

        // (4) The transaction: train (microshard) → eval → keep|rollback. The
        // per-step config carries the placement plan (device + rotating block).
        let granularity = cfg
            .daemon
            .as_ref()
            .map(|d| d.granularity.as_str())
            .unwrap_or("module");
        let step_cfg = apply_plan(cfg, device, shard, opts.rotation_blocks, granularity);
        let id = format!("daemon-{ordinal}");
        let outcome = reg.run_step(
            &id,
            "daemon:microshard",
            ordinal,
            baseline,
            || (hooks.train)(&step_cfg, &dataset),
            || (hooks.score)(&step_cfg),
        )?;

        let place = match (device, shard) {
            ("cpu", _) => " on cpu".to_string(),
            (_, Some(b)) => format!(" block {b}"),
            _ => String::new(),
        };
        let note = match outcome.action {
            StepAction::Commit => format!("[{lane_note}]{place} kept (eval passed)"),
            StepAction::Rollback => format!("[{lane_note}]{place} rolled back (regress)"),
            StepAction::Quarantine => {
                format!("[{lane_note}]{place} CATASTROPHE — rolled back + quarantined + halt")
            }
        };
        let halt = outcome.halt;
        report.steps.push(DaemonStep {
            ordinal,
            items: items.len() - dropped,
            action: Some(outcome.action),
            metrics: outcome.metrics,
            halt,
            note,
        });

        ordinal += 1;
        taken += 1;
        if halt {
            report.halted = true;
            break;
        }

        // (6) Cooldown — leave the GPU idle between steps so foreground apps get
        // gaps (production only; tests use the 0 default and never sleep).
        if !opts.cooldown.is_zero() {
            std::thread::sleep(opts.cooldown);
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EvalConfig, EvolveConfig, RegulateConfig};
    use crate::dataset::GenExample;
    use crate::living_queue::Lane;
    use std::cell::Cell;

    fn qa(p: &str) -> GenExample {
        GenExample::Qa {
            prompt: p.to_string(),
            completion: format!("a:{p}"),
            source: None,
            gen: Some("teach".to_string()),
        }
    }

    fn cfg_for(dir: &Path) -> EvolveConfig {
        EvolveConfig {
            evolve: crate::config::EvolveSection {
                work_dir: Some(dir.to_path_buf()),
                ..Default::default()
            },
            eval: Some(EvalConfig::default()),
            regulate: Some(RegulateConfig::default()),
            ..Default::default()
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "scrt-evolve-daemon-{tag}-{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        base
    }

    fn good_score() -> ScoreReport {
        let mut r = ScoreReport::uncovered("probe-test", "stub");
        r.correctness = 0.9;
        r.n = 1;
        r
    }

    #[test]
    fn drains_queue_and_commits_each_step() {
        let dir = tmp("drain");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue_many(Lane::Priority, &[qa("a"), qa("b")]).unwrap();

        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, ds: &Dataset| {
            Ok(vec![ds
                .rows
                .first()
                .and_then(|r| match r {
                    GenExample::Qa { gen, .. } => gen.clone(),
                    _ => None,
                })
                .unwrap_or_default()])
        };
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            batch: 1,
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert!(report.drained);
        assert_eq!(report.committed(), 2);
    }

    #[test]
    fn stop_request_breaks_loop() {
        let dir = tmp("stop");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue_many(Lane::Raw, &[qa("a"), qa("b"), qa("c")])
            .unwrap();

        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            ..Default::default()
        };
        // Stop after the first iteration: a counter the closure flips.
        let calls = Cell::new(0u32);
        let stop = || {
            let n = calls.get();
            calls.set(n + 1);
            n >= 1
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &stop).unwrap();
        assert!(report.stopped);
        assert_eq!(report.committed(), 1);
    }

    #[test]
    fn decide_step_gpu_when_free_and_idle() {
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            ..Default::default()
        };
        // Plenty of free VRAM, no other GPU process ⇒ train on GPU.
        assert_eq!(
            decide_step(0, Some(8.0), Some(false), &opts),
            StepDecision::Train {
                device: "cuda",
                shard: None
            }
        );
    }

    #[test]
    fn decide_step_falls_back_to_cpu_when_gpu_busy() {
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            pause_on_gpu_process: true,
            cpu_fallback: true,
            ..Default::default()
        };
        // Another GPU process is active (a game) ⇒ yield the GPU, run on CPU.
        assert_eq!(
            decide_step(0, Some(8.0), Some(true), &opts),
            StepDecision::Train {
                device: "cpu",
                shard: None
            }
        );
    }

    #[test]
    fn decide_step_waits_when_busy_and_no_cpu_fallback() {
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            pause_on_gpu_process: true,
            cpu_fallback: false,
            ..Default::default()
        };
        match decide_step(0, Some(8.0), Some(true), &opts) {
            StepDecision::Wait(reason) => assert!(reason.contains("GPU process")),
            other => panic!("expected Wait, got {other:?}"),
        }
    }

    #[test]
    fn decide_step_waits_when_vram_starved_no_fallback() {
        let opts = DaemonOptions {
            max_vram_gb: Some(6.0),
            cpu_fallback: false,
            ..Default::default()
        };
        // Free VRAM below budget (a game holds it) ⇒ wait.
        assert!(matches!(
            decide_step(0, Some(1.0), Some(false), &opts),
            StepDecision::Wait(_)
        ));
    }

    #[test]
    fn decide_step_rotates_blocks() {
        let opts = DaemonOptions {
            rotation_blocks: 3,
            ..Default::default()
        };
        let shard_of = |ord| match decide_step(ord, None, Some(false), &opts) {
            StepDecision::Train { shard, .. } => shard,
            _ => None,
        };
        assert_eq!(shard_of(0), Some(0));
        assert_eq!(shard_of(1), Some(1));
        assert_eq!(shard_of(2), Some(2));
        assert_eq!(shard_of(3), Some(0)); // wraps
    }

    #[test]
    fn busy_gpu_with_cpu_fallback_still_commits() {
        // Integration: a busy GPU does NOT stall the daemon when cpu_fallback is
        // on — it trains on CPU and the queue still drains.
        let dir = tmp("cpu-fallback");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue_many(Lane::Raw, &[qa("a"), qa("b")]).unwrap();
        let free_vram = || Some(8.0);
        let gpu_busy = || Some(true); // a game is running
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_busy,
            train: &train,
            score: &score,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            cpu_fallback: true,
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(
            report.committed(),
            2,
            "CPU fallback keeps draining under a busy GPU"
        );
    }

    #[test]
    fn max_steps_bounds_the_run() {
        let dir = tmp("bound");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        for i in 0..10 {
            q.enqueue(Lane::Raw, &qa(&format!("q{i}"))).unwrap();
        }
        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            max_steps: Some(3),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.steps.len(), 3);
    }
}
