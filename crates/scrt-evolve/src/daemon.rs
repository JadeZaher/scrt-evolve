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

/// The track-32 degradation-gate hook: given the per-step config, sample the
/// probe BEFORE/AFTER and return the degradation report. Boxed-fn alias to keep
/// the [`DaemonHooks`] field legible.
pub type DegradeHook<'a> =
    &'a dyn Fn(&EvolveConfig) -> anyhow::Result<crate::eval::DegradationReport>;

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
    /// Monotonic wall-clock in seconds (Q3 budget + retry backoff timing).
    /// Production: seconds since the daemon started. Tests inject a controllable
    /// clock so the budget window is deterministic.
    pub now_secs: &'a dyn Fn() -> u64,
    /// Sleep for a duration (Q2 retry backoff). Production: `thread::sleep`; tests
    /// inject a no-op so retries don't actually wait.
    pub sleep: &'a dyn Fn(Duration),
    /// Track 32 — the OPTIONAL degradation gate. `Some` ⇒ after training, sample
    /// the probe BEFORE (base) vs AFTER (base+candidate-adapter) and judge whether
    /// AFTER degraded; the step is accepted UNLESS degradation exceeds the
    /// threshold (`crate::eval::judge_verdict`). `None` ⇒ the correctness gate
    /// (today's behavior). Production wires this to the A/B subprocess + LLM judge;
    /// tests inject a deterministic report. Runs AFTER `train`, on the mutated
    /// adapter (so the candidate is what gets sampled).
    pub degrade: Option<DegradeHook<'a>>,
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
    /// Q2: retry a TRANSIENT step error (train/score subprocess failure) this many
    /// times before recording a failed-but-non-halting step. `0` ⇒ no retry.
    pub max_retries: u32,
    /// Q2: base backoff between transient retries (doubled each attempt).
    pub backoff_base: Duration,
    /// Q2: stop the loop after this many CONSECUTIVE step failures. `0` ⇒ never.
    pub max_consecutive_failures: u32,
    /// Q3: wall-clock training budget — max minutes of training per rolling hour.
    /// `0` ⇒ unlimited. Enforced via the injected clock + a sliding window.
    pub max_minutes_per_hour: u64,
    /// Track 32: minimum genuinely-new rows to train in one step. A batch below
    /// this is skipped (rows stay queued) so we don't overfit on 1–2 rows. `0` ⇒
    /// no floor.
    pub min_train_pairs: usize,
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
            max_retries: 0,
            backoff_base: Duration::from_secs(5),
            max_consecutive_failures: 0,
            max_minutes_per_hour: 0,
            min_train_pairs: 0,
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

/// Q3 — wall-clock training budget. Pure: given the rolling window of recent
/// (timestamp, train_secs) entries, `now`, and the per-hour cap (minutes), decide
/// whether another training step is within budget. A `0` cap is unlimited. Only
/// entries within the last hour count (the window is a sliding 3600 s).
pub fn within_budget(window: &[(u64, u64)], now: u64, max_minutes_per_hour: u64) -> bool {
    if max_minutes_per_hour == 0 {
        return true;
    }
    let horizon = now.saturating_sub(3600);
    let spent_secs: u64 = window
        .iter()
        .filter(|(ts, _)| *ts >= horizon)
        .map(|(_, secs)| *secs)
        .sum();
    spent_secs < max_minutes_per_hour * 60
}

/// Drop window entries older than one hour relative to `now` (keep the slide
/// bounded). Returns the retained entries.
fn prune_window(mut window: Vec<(u64, u64)>, now: u64) -> Vec<(u64, u64)> {
    let horizon = now.saturating_sub(3600);
    window.retain(|(ts, _)| *ts >= horizon);
    window
}

/// Track 32 — the min-QA-pairs floor. Pure: may a batch of `n` trainable rows
/// train this step? `min == 0` ⇒ no floor (any non-empty batch trains). Below the
/// floor ⇒ skip + accumulate (the rows stay queued; the loop idles).
pub fn enough_to_train(n: usize, min: usize) -> bool {
    n > 0 && n >= min
}

/// Classify whether a step error is TRANSIENT (worth a retry) vs a hard problem.
/// Conservative: anything that isn't an obvious permanent misconfiguration is
/// treated as transient (OOM, subprocess blip, endpoint hiccup all recover on a
/// retry). A genuine track-15 CATASTROPHE never reaches here — it comes back as
/// `outcome.halt` from a SUCCESSFUL `run_step`, not as an `Err`.
fn is_transient(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_lowercase();
    // Permanent-looking misconfig: don't spin on these.
    let permanent = ["no such file", "not found", "is unset", "required", "parse"];
    !permanent.iter().any(|p| msg.contains(p))
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
    /// Q2: this step exhausted its retries on a TRANSIENT error (not a
    /// catastrophe-halt). The error text is in `note`.
    pub failed: bool,
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
    /// Q2: the supervisor gave up after `max_consecutive_failures` consecutive
    /// transient step failures (re-arm/restart required).
    pub gave_up: bool,
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
    let mut consecutive_failures = 0u32;
    // Q3: sliding (timestamp, train_secs) window for the per-hour wall-clock cap.
    let mut budget_window: Vec<(u64, u64)> = Vec::new();

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

        // (1b) Q3 wall-clock budget gate: if we've trained our per-hour minutes,
        // WAIT (same as the VRAM gate) — yield real time, not just VRAM.
        let now = (hooks.now_secs)();
        budget_window = prune_window(budget_window, now);
        if !within_budget(&budget_window, now, opts.max_minutes_per_hour) {
            if opts.exit_when_empty {
                report.steps.push(DaemonStep {
                    ordinal,
                    items: 0,
                    action: None,
                    metrics: None,
                    halt: false,
                    failed: false,
                    note: "paused: per-hour training budget spent".to_string(),
                });
                break;
            }
            (hooks.sleep)(opts.poll_interval);
            continue;
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
                        failed: false,
                        note: reason.clone(),
                    });
                    break;
                }
                (hooks.sleep)(opts.poll_interval);
                continue;
            }
        };

        // (2c) Track 32 min-QA-pairs floor: if there aren't yet enough pending
        // rows to train on, DON'T pop — leave them queued and accumulate (idle, or
        // stop in drain mode). Checked BEFORE popping so the cursor isn't advanced
        // (the rows wait for more to arrive). Composes with the Q5 ledger: a stale
        // corpus that yields too few new rows simply idles.
        if opts.min_train_pairs > 0 {
            let (p, r) = queue.pending();
            let pending = (p + r) as usize;
            if !enough_to_train(pending, opts.min_train_pairs) {
                if pending == 0 {
                    report.drained = true;
                }
                if opts.exit_when_empty {
                    report.steps.push(DaemonStep {
                        ordinal,
                        items: pending,
                        action: None,
                        metrics: None,
                        halt: false,
                        failed: false,
                        note: format!(
                            "{pending} pending < min_train_pairs={} — accumulating",
                            opts.min_train_pairs
                        ),
                    });
                    break;
                }
                (hooks.sleep)(opts.poll_interval);
                continue;
            }
        }

        // (3) Pop a batch (priority-first); drop quarantined provenance.
        let items = queue.pop_batch(opts.batch)?;
        if items.is_empty() {
            report.drained = true;
            if opts.exit_when_empty {
                break;
            }
            (hooks.sleep)(opts.poll_interval);
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
                failed: false,
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

        // (4) The transaction WITH Q2 retry: a TRANSIENT failure of EITHER
        // train OR score (subprocess non-zero, OOM, endpoint blip) makes
        // `run_step_strict` return `Err` — retry with exponential backoff.
        // (`run_step_strict`, unlike the lenient `run_step`, propagates a train
        // error instead of swallowing it into a silent rollback — track 31 Q2.)
        // A track-15 CATASTROPHE is NOT an `Err`; it comes back as a successful
        // outcome with `halt=true`, so it bypasses retry and halts as before. A
        // permanent-looking error (missing file, misconfig) is not retried.
        let train_start = (hooks.now_secs)();
        let mut attempt = 0u32;
        // Track 32: the verdict source. `degrade` present ⇒ the JUDGE gate
        // (accept unless degradation); else the correctness gate (run_step_strict).
        let tol = reg.tolerances();
        let max_regressed_frac = cfg
            .regulate
            .as_ref()
            .map(|r| r.max_regressed_frac)
            .unwrap_or(0.0);
        let outcome = loop {
            let txn = if let Some(degrade) = hooks.degrade {
                reg.run_step_judged(
                    &id,
                    "daemon:microshard",
                    ordinal,
                    baseline,
                    || (hooks.train)(&step_cfg, &dataset),
                    || (hooks.score)(&step_cfg),
                    |candidate| {
                        // Sample BEFORE/AFTER + judge on the just-trained adapter.
                        let report = degrade(&step_cfg)?;
                        Ok(crate::eval::judge_verdict(
                            baseline,
                            candidate.correctness,
                            report.regressed_fraction(),
                            &tol,
                            max_regressed_frac,
                        ))
                    },
                )
            } else {
                reg.run_step_strict(
                    &id,
                    "daemon:microshard",
                    ordinal,
                    baseline,
                    || (hooks.train)(&step_cfg, &dataset),
                    || (hooks.score)(&step_cfg),
                )
            };
            match txn {
                Ok(o) => break Ok(o),
                Err(e) => {
                    if attempt >= opts.max_retries || !is_transient(&e) {
                        break Err(e);
                    }
                    let backoff = opts.backoff_base * 2u32.pow(attempt);
                    attempt += 1;
                    eprintln!(
                        "daemon: step {ordinal} transient failure (attempt {attempt}/{}): {e} \
                         — retrying in {}s",
                        opts.max_retries,
                        backoff.as_secs()
                    );
                    (hooks.sleep)(backoff);
                }
            }
        };

        let place = match (device, shard) {
            ("cpu", _) => " on cpu".to_string(),
            (_, Some(b)) => format!(" block {b}"),
            _ => String::new(),
        };

        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => {
                // Retries exhausted (or permanent error): record a failed,
                // NON-halting step and let the supervisor decide whether to give
                // up. The model is untouched — `run_step` is transactional.
                consecutive_failures += 1;
                report.steps.push(DaemonStep {
                    ordinal,
                    items: items.len() - dropped,
                    action: None,
                    metrics: None,
                    halt: false,
                    failed: true,
                    note: format!("[{lane_note}]{place} step FAILED after retries: {e}"),
                });
                ordinal += 1;
                taken += 1;
                if opts.max_consecutive_failures > 0
                    && consecutive_failures >= opts.max_consecutive_failures
                {
                    report.gave_up = true;
                    break;
                }
                if opts.exit_when_empty {
                    // Bounded mode: don't spin — surface the failure and stop.
                    break;
                }
                (hooks.sleep)(opts.poll_interval);
                continue;
            }
        };

        // A successful step (committed/rolled-back/catastrophe) resets the streak.
        consecutive_failures = 0;
        // Q3: account this step's training time in the sliding budget window.
        let train_secs = (hooks.now_secs)().saturating_sub(train_start);
        budget_window.push((train_start, train_secs));

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
            failed: false,
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
            (hooks.sleep)(opts.cooldown);
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

    // A fixed clock + no-op sleep for deterministic tests (no real waiting, no
    // wall-clock dependence). Budget tests override `now` via a Cell-backed clock.
    fn zero_now() -> u64 {
        0
    }
    fn noop_sleep(_: Duration) {}

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
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
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
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
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
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
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
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
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

    // ───────────────────── Track 31 Q2/Q3 hardening ─────────────────────

    #[test]
    fn within_budget_unlimited_when_zero() {
        assert!(within_budget(&[(0, 9999)], 100, 0), "0 cap ⇒ unlimited");
    }

    #[test]
    fn within_budget_blocks_when_spent() {
        // Cap 1 min/hr = 60s. 50s spent within the last hour ⇒ still ok; 70s ⇒ over.
        assert!(within_budget(&[(100, 50)], 120, 1));
        assert!(!within_budget(&[(100, 70)], 120, 1));
    }

    #[test]
    fn within_budget_forgets_old_window() {
        // 100s of training, but it was > 1h ago ⇒ doesn't count against now.
        assert!(within_budget(&[(0, 100)], 4000, 1));
    }

    #[test]
    fn is_transient_classifies_errors() {
        assert!(is_transient(&anyhow::anyhow!("CUDA out of memory")));
        assert!(is_transient(&anyhow::anyhow!(
            "subprocess exited with code 1"
        )));
        // Permanent-looking misconfig is NOT retried.
        assert!(!is_transient(&anyhow::anyhow!(
            "model.safetensors: not found"
        )));
        assert!(!is_transient(&anyhow::anyhow!(
            "generate.api: `model` is required"
        )));
    }

    #[test]
    fn transient_score_failure_is_retried_then_survives() {
        // A flaky SCORE hook (eval subprocess blip) fails twice, then succeeds.
        // run_step surfaces a score error as Err → the daemon retries it.
        let dir = tmp("retry");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let attempts = Cell::new(0u32);
        let score = |_: &EvolveConfig| {
            let n = attempts.get();
            attempts.set(n + 1);
            if n < 2 {
                anyhow::bail!("eval subprocess timed out (transient)")
            }
            Ok(good_score())
        };
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            max_retries: 3,
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(attempts.get(), 3, "two failures + one success");
        assert_eq!(report.committed(), 1, "the step eventually committed");
        assert!(!report.gave_up);
    }

    #[test]
    fn transient_train_failure_is_retried_then_survives() {
        // A flaky TRAIN hook (train subprocess blip). Because the daemon uses
        // run_step_strict, a train error now ALSO propagates as Err and is
        // retried — it is no longer swallowed into a silent rollback (Q2 follow-up).
        let dir = tmp("retry-train");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let attempts = Cell::new(0u32);
        let train = |_: &EvolveConfig, _: &Dataset| {
            let n = attempts.get();
            attempts.set(n + 1);
            if n < 2 {
                anyhow::bail!("train subprocess exited with code 1 (transient)")
            }
            Ok(vec!["teach".to_string()])
        };
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            max_retries: 3,
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(attempts.get(), 3, "two train failures + one success");
        assert_eq!(report.committed(), 1, "the step eventually committed");
        assert!(!report.gave_up);
    }

    #[test]
    fn exhausted_retries_record_failed_step_and_supervisor_gives_up() {
        // A persistently-failing transient hook: each step exhausts retries and
        // is recorded as failed; the supervisor stops after the cap.
        let dir = tmp("giveup");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue_many(Lane::Raw, &[qa("a"), qa("b"), qa("c")])
            .unwrap();
        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| anyhow::bail!("eval keeps timing out");
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            max_retries: 1,
            max_consecutive_failures: 2,
            // NOT drain mode: the supervisor cap is what stops it.
            exit_when_empty: false,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert!(report.gave_up, "supervisor gave up after the failure cap");
        assert_eq!(report.committed(), 0);
        assert_eq!(
            report.steps.iter().filter(|s| s.failed).count(),
            2,
            "two failed steps recorded before giving up"
        );
    }

    #[test]
    fn budget_gate_pauses_in_drain_mode() {
        // A clock that jumps so the budget window shows minutes already spent.
        let dir = tmp("budget");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        // First step trains for 120s (clock advances), exceeding a 1 min/hr cap on
        // the SECOND iteration's budget check.
        let clock = Cell::new(0u64);
        let now = || {
            let t = clock.get();
            clock.set(t + 120);
            t
        };
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &now,
            sleep: &noop_sleep,
            degrade: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            max_minutes_per_hour: 1, // 60s/hr; one 120s step blows it
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        // The one queued step trains; the next budget check is over and (drain
        // mode) records a "budget spent" wait step and stops.
        assert!(report
            .steps
            .iter()
            .any(|s| s.note.contains("training budget")));
    }

    // ───────────────────── Track 32: min-pairs floor + judge gate ─────────────────────

    #[test]
    fn enough_to_train_floor() {
        assert!(!enough_to_train(0, 0), "empty never trains");
        assert!(enough_to_train(1, 0), "no floor ⇒ any non-empty trains");
        assert!(!enough_to_train(3, 4), "below floor");
        assert!(enough_to_train(4, 4), "at floor");
        assert!(enough_to_train(9, 4), "above floor");
    }

    #[test]
    fn below_min_pairs_does_not_train_and_accumulates() {
        // 3 rows pending, floor 4 ⇒ skip without popping (rows stay queued).
        let dir = tmp("minpairs");
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
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            min_train_pairs: 4,
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.committed(), 0, "nothing trained below the floor");
        // The rows are NOT consumed — still pending for a later step.
        let q2 = LivingQueue::from_config(&cfg).unwrap();
        assert_eq!(q2.pending(), (0, 3), "rows accumulate, cursor not advanced");
        assert!(report.steps.iter().any(|s| s.note.contains("accumulating")));
    }

    #[test]
    fn at_min_pairs_trains_normally() {
        let dir = tmp("minpairs-ok");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue_many(Lane::Raw, &[qa("a"), qa("b"), qa("c"), qa("d")])
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
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            min_train_pairs: 4,
            batch: 4,
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.committed(), 1, "4 pending ≥ floor 4 ⇒ trains");
    }

    #[test]
    fn judge_gate_rolls_back_on_degradation() {
        // The degrade hook reports a regression ⇒ judge_verdict ⇒ Regress, even
        // though the correctness score is good. Proves the judge is the driver.
        let dir = tmp("judge-regress");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score()); // correctness fine…
        let gpu_idle = || Some(false);
        // …but the judge says half the items degraded.
        let degrade = |_: &EvolveConfig| {
            Ok(crate::eval::DegradationReport {
                n: 2,
                regressed: 1,
                worse: vec![false, true],
            })
        };
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: Some(&degrade),
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.committed(), 0, "degradation ⇒ rolled back");
        assert!(report
            .steps
            .iter()
            .any(|s| s.action == Some(StepAction::Rollback)));
    }

    #[test]
    fn judge_gate_accepts_when_no_degradation() {
        // No degradation + a LOW correctness score (0.30, below the old gate's
        // comfort) ⇒ the judge gate still ACCEPTS. This is the whole point.
        let dir = tmp("judge-accept");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let mut low = good_score();
        low.correctness = 0.30;
        let score = move |_: &EvolveConfig| Ok(low.clone());
        let gpu_idle = || Some(false);
        let degrade = |_: &EvolveConfig| {
            Ok(crate::eval::DegradationReport {
                n: 2,
                regressed: 0,
                worse: vec![false, false],
            })
        };
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: Some(&degrade),
        };
        // Baseline at 0.40 — under the OLD gate a drop to 0.30 would Regress; the
        // judge gate accepts because no degradation was detected.
        let mut baseline = good_score();
        baseline.correctness = 0.40;
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &baseline, &hooks, &|| false).unwrap();
        assert_eq!(
            report.committed(),
            1,
            "no degradation ⇒ accept despite low/dropped correctness"
        );
    }
}
