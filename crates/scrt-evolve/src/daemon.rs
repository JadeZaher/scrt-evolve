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
//! 1. **stop check** — explicit `evolve ambient stop` drops a stop-file; honored at the
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

use crate::arbitration::ServedReady;
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

/// The track-37 Phase-D live-nudge poll hook (consume-once at the step boundary).
pub type NudgeHook<'a> = &'a dyn Fn() -> anyhow::Result<Option<crate::Nudge>>;

/// The track-37 Phase-D steering-compliance sampler hook (fraction 0–1).
pub type ComplianceHook<'a> = &'a dyn Fn(&EvolveConfig) -> anyhow::Result<f64>;

/// The injected effects the daemon needs. Kept closure-based so production
/// (subprocess + GPU probe) and tests (deterministic) share one loop.
pub struct DaemonHooks<'a> {
    /// Probe free VRAM in GB. `None` ⇒ unknown (the VRAM gate is skipped — we
    /// can't throttle what we can't measure). Production: parse `nvidia-smi`.
    pub free_vram_gb: &'a dyn Fn() -> Option<f64>,
    /// Probe free HOST RAM in GB. `None` ⇒ unknown (the RAM gate is skipped).
    /// Production: read `/proc/meminfo` `MemAvailable` (Linux/WSL). Guards against
    /// freezing the machine on a big model load.
    pub free_ram_gb: &'a dyn Fn() -> Option<f64>,
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
    /// Track 37 Phase D — the OPTIONAL live-nudge poll. `Some` ⇒ called at the top
    /// of each step (after the stop-check, before work) to fetch + consume a
    /// pending [`crate::Nudge`]; the returned nudge is merged into the live config
    /// via the safe-live allowlist. `None` ⇒ no live steering (nudges ignored).
    /// Production reads/deletes `nudge.json`; tests inject a deterministic nudge.
    pub take_nudge: Option<NudgeHook<'a>>,
    /// Track 37 Phase D — the OPTIONAL steering-compliance sampler. `Some` ⇒ after
    /// a committed step, sample K generated rows and judge them against the
    /// composed steering text, returning the compliance fraction (0–1). `None` ⇒
    /// no compliance metric (no judge call). Recorded in the step log for
    /// `watch trend`.
    pub steering_compliance: Option<ComplianceHook<'a>>,
    /// Track 33 — the OPTIONAL commit-swap emit. `Some` ⇒ called at the KEEP
    /// branch (after the merged flat adapter is the committed truth) with a
    /// [`ServedReady`] carrying the incremented version; the live server tails it
    /// to hot-swap. Rollback/catastrophe emit NOTHING. `None` ⇒ no emission
    /// (back-compat). Production appends to `<state>/served-ready.jsonl` via
    /// [`crate::arbitration::append_served_ready`]; tests assert emit count.
    pub served_ready_emit: Option<&'a dyn Fn(ServedReady)>,
}

/// Tunables for a daemon run. `[daemon]` config + CLI flags populate these.
#[derive(Debug, Clone)]
pub struct DaemonOptions {
    /// VRAM budget in GB: the daemon trains only when at least this much is FREE
    /// (so it never OOMs the user's foreground work). `None` ⇒ ungated.
    pub max_vram_gb: Option<f64>,
    /// HOST-RAM floor in GB: the daemon trains only when at least this much system
    /// RAM is FREE. Guards against freezing the machine when a large f16 model load
    /// (train, or the A/B eval that loads it TWICE) would otherwise consume all
    /// host memory. Low RAM ⇒ WAIT (never CPU fallback — CPU uses more RAM).
    /// `None` ⇒ ungated (back-compat default).
    pub min_free_ram_gb: Option<f64>,
    /// Track 33: serve-while-you-train carve-out (GB) reserved for a co-resident
    /// inference server. Subtracted from usable VRAM headroom BEFORE a block
    /// starts, so the trainer proceeds only when `free − reservation ≥ budget`.
    /// `None` ⇒ no carve-out (behavior byte-identical to today).
    pub serve_reservation_gb: Option<f64>,
    /// Queued items folded into one microshard step.
    pub batch: usize,
    /// Stop after this many transactional steps (`None` ⇒ until stopped). Bounds
    /// tests and `evolve ambient start --max-steps N`.
    pub max_steps: Option<u64>,
    /// When the queue drains: `true` ⇒ exit (drain-once mode); `false` ⇒ wait for
    /// new activity (the long-running `evolve ambient start`).
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
            min_free_ram_gb: None,
            serve_reservation_gb: None,
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
    free_ram_gb: Option<f64>,
    opts: &DaemonOptions,
) -> StepDecision {
    let shard = if opts.rotation_blocks > 0 {
        Some((ordinal as usize) % opts.rotation_blocks)
    } else {
        None
    };
    // HOST-RAM gate (system-freeze guard). Loading a large f16 model for a train
    // or eval step spikes host RAM (eval's A/B path loads the model TWICE); on a
    // small box that starves the OS and freezes the machine. Unlike the VRAM gate,
    // low RAM must NOT fall back to CPU — a CPU step uses MORE host RAM and would
    // make the freeze worse. So low RAM ⇒ WAIT unconditionally, ignoring
    // cpu_fallback. Ungated (budget None) or unmeasurable ⇒ don't block on RAM.
    let ram_ok = match (opts.min_free_ram_gb, free_ram_gb) {
        (Some(budget), Some(free)) => free >= budget,
        // `_ =>` is correct: both fields are `Option<f32>` (stdlib), not an owned
        // enum — all remaining (None, _) / (_, None) combinations mean "ungated or
        // unmeasurable", which must not block.
        _ => true,
    };
    if !ram_ok {
        return StepDecision::Wait(format!(
            "paused: free host RAM below budget ({} GB min)",
            opts.min_free_ram_gb.unwrap_or(0.0)
        ));
    }
    // Track 33: the serve carve-out shrinks usable headroom BEFORE the block —
    // the trainer sees `free − reservation`, so a co-resident inference server
    // keeps its slice. `None` ⇒ no carve-out (subtract 0.0 ⇒ byte-identical).
    let reservation = opts.serve_reservation_gb.unwrap_or(0.0);
    let vram_ok = match (opts.max_vram_gb, free_vram_gb) {
        (Some(budget), Some(free)) => (free - reservation) >= budget,
        // `_ =>` is correct: both fields are `Option<f64>` (stdlib), not an owned
        // enum. Ungated (None) or unmeasurable (Some probe failed) ⇒ don't block.
        _ => true,
    };
    // Track 33 model-A degrade: a live served inference process registers as
    // another GPU user, so the existing `pause_on_gpu_process` path already yields
    // the GPU to it as first-class foreground — no separate predicate needed.
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
    // Apply `[daemon].objective` only when sharding (materializes fractional) or
    // when fractional already exists — never create one on a non-sharded step
    // (FractionalConfig::default().enabled = true). See AGENTS.md §objective.
    let daemon_objective = c.daemon.as_ref().map(|d| d.objective.clone());
    if let Some(idx) = shard {
        let mut tr = c.train.clone().unwrap_or_default();
        let mut frac = tr.fractional.clone().unwrap_or_default();
        frac.enabled = true;
        frac.shards = Some(rotation_blocks);
        frac.shard_index = Some(idx);
        frac.granularity = granularity.to_string();
        if let Some(obj) = daemon_objective {
            frac.objective = obj;
        }
        tr.fractional = Some(frac);
        c.train = Some(tr);
    } else if let Some(obj) = daemon_objective {
        // Non-sharded step: set the objective ONLY if a fractional config already
        // exists (don't create one — that would enable fractional by default).
        if let Some(tr) = c.train.as_mut() {
            if let Some(frac) = tr.fractional.as_mut() {
                frac.objective = obj;
            }
        }
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
    /// Track 33: the final served-adapter version (bumped once per keep; 0 ⇒
    /// nothing committed). The latest signal the live server would have swapped to.
    pub served_version: u64,
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

/// The stop-file the running daemon polls. `evolve ambient stop` creates it; the loop
/// removes it on a clean exit.
pub fn stop_file(work_dir: &Path) -> PathBuf {
    work_dir.join("daemon.stop")
}

/// The run marker written while the daemon is active (for `evolve watch status`).
pub fn run_file(work_dir: &Path) -> PathBuf {
    work_dir.join("daemon.run")
}

/// Request the running daemon stop (the `evolve ambient stop` command): drop the
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
    // Track 37 Phase D: the LIVE config a nudge merges into. Ephemeral — TOML
    // (`cfg`) wins on restart because this copy dies with the loop. An active
    // focus expires at the recorded ordinal.
    let mut live_cfg = cfg.clone();
    let mut focus_expiry: Option<u64> = None;
    // Track 33: monotonic served-adapter version. Base = 0 (nothing committed
    // yet); each keep bumps it, so the first committed adapter is version 1.
    let mut served_version: u64 = 0;

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

        // Track 37 Phase D: poll + consume a pending nudge, then merge its
        // safe-live allowlist into `live_cfg`. Runs at the TOP of the step (after
        // the stop-check — the loop owns the config here). Consume-once; a
        // malformed nudge is logged and skipped, never wedges the loop.
        if let Some(take) = hooks.take_nudge {
            match take() {
                Ok(Some(nudge)) => {
                    let outcome = crate::apply_nudge(&mut live_cfg, &nudge, ordinal);
                    if let Some((_, expiry)) = outcome.focus {
                        focus_expiry = Some(expiry);
                    }
                    if !outcome.is_empty() {
                        let _ = append_nudge_log(
                            &wd,
                            ordinal,
                            &outcome.applied,
                            &outcome.rejected,
                        );
                    }
                }
                Ok(None) => {}
                Err(e) => eprintln!("daemon: nudge poll failed ({e}); ignoring"),
            }
        }
        // Expire a transient focus whose TTL has passed: clear BOTH the tracker
        // and the ephemeral steering field so base steering is restored.
        if let Some(exp) = focus_expiry {
            if ordinal >= exp {
                focus_expiry = None;
                live_cfg.evolve.focus = None;
            }
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
        let decision = decide_step(
            ordinal,
            (hooks.free_vram_gb)(),
            (hooks.gpu_busy)(),
            (hooks.free_ram_gb)(),
            opts,
        );
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
        // Track 37 Phase D: the step config is derived from `live_cfg` (base TOML
        // + any merged nudges), so a live nudge takes effect from this step on.
        let granularity = live_cfg
            .daemon
            .as_ref()
            .map(|d| d.granularity.as_str())
            .unwrap_or("module");
        let step_cfg = apply_plan(&live_cfg, device, shard, opts.rotation_blocks, granularity);
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
            // `_ =>` is correct: `device` is `&str` (not an owned enum) — any
            // non-"cpu" string with no shard index (e.g. "cuda", ungated) needs no
            // placement annotation.
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

        // Track 33 commit-swap: ONLY on a keep, AFTER the merged flat adapter is
        // the committed truth, bump the served version and emit the swap signal
        // for the live server. Rollback/catastrophe fall through here untouched,
        // so they emit nothing and the version does not advance.
        if outcome.action == StepAction::Commit {
            served_version += 1;
            if let Some(emit) = hooks.served_ready_emit {
                let adapter_path = wd.adapter_safetensors().to_string_lossy().into_owned();
                let base_path = live_cfg
                    .evolve
                    .model_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                emit(ServedReady {
                    version: served_version,
                    adapter_path,
                    base_path,
                    timestamp: format!("step-{ordinal}"),
                });
            }
        }

        // Steering-compliance metric (only when steering is set AND the sampler
        // hook is wired — no judge call otherwise). See AGENTS.md §nudge.
        if outcome.action == StepAction::Commit && live_cfg.compose_steering().is_some() {
            if let Some(sampler) = hooks.steering_compliance {
                match sampler(&step_cfg) {
                    Ok(frac) => {
                        let _ = append_compliance_log(&wd, ordinal, frac);
                    }
                    Err(e) => eprintln!("daemon: steering-compliance sample failed ({e})"),
                }
            }
        }

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

    report.served_version = served_version;
    Ok(report)
}

/// The evolution-log path the daemon + `watch` share.
fn evolution_log(wd: &WorkDir) -> PathBuf {
    wd.root().join("evolution-log.jsonl")
}

/// Append a `kind:"nudge"` evolution-log row (track 37 Phase D) recording what a
/// live nudge applied/rejected, so `watch status`/`health` can surface it. Uses
/// the benign `Commit` action (a nudge touches no weights) — readers key on
/// `kind == "nudge"`.
fn append_nudge_log(
    wd: &WorkDir,
    ordinal: u64,
    applied: &[String],
    rejected: &[String],
) -> anyhow::Result<()> {
    let mut cause = String::new();
    if !applied.is_empty() {
        cause.push_str(&format!("applied: {}", applied.join("; ")));
    }
    if !rejected.is_empty() {
        if !cause.is_empty() {
            cause.push_str(" | ");
        }
        cause.push_str(&format!("rejected: {}", rejected.join("; ")));
    }
    crate::regulate::log::append(
        &evolution_log(wd),
        &crate::regulate::log::EvolutionLogEntry {
            step: ordinal,
            checkpoint_id: format!("nudge-{ordinal}"),
            kind: "nudge".to_string(),
            verdict: None,
            metrics: None,
            action: StepAction::Commit,
            cause: Some(cause),
        },
    )
}

/// Append a `kind:"steering_compliance"` evolution-log row (track 37 Phase D):
/// the fraction of sampled generated rows the judge found steering-aligned, so
/// `watch trend` can chart compliance alongside correctness.
fn append_compliance_log(wd: &WorkDir, ordinal: u64, fraction: f64) -> anyhow::Result<()> {
    crate::regulate::log::append(
        &evolution_log(wd),
        &crate::regulate::log::EvolutionLogEntry {
            step: ordinal,
            checkpoint_id: format!("compliance-{ordinal}"),
            kind: "steering_compliance".to_string(),
            verdict: None,
            metrics: None,
            action: StepAction::Commit,
            cause: Some(format!("steering_compliance={fraction:.4}")),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EvalConfig, EvolveConfig, RegulateConfig};
    use crate::dataset::{GenExample, Outcome, Tier, Verdict};
    use crate::living_queue::Lane;
    use std::cell::Cell;

    fn qa(p: &str) -> GenExample {
        GenExample::Qa {
            prompt: p.to_string(),
            completion: format!("a:{p}"),
            source: None,
            gen: Some("teach".to_string()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
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
        let free_ram = || None::<f64>;
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
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
            decide_step(0, Some(8.0), Some(false), None, &opts),
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
            decide_step(0, Some(8.0), Some(true), None, &opts),
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
        match decide_step(0, Some(8.0), Some(true), None, &opts) {
            StepDecision::Wait(reason) => assert!(reason.contains("GPU process")),
            other => panic!("expected Wait, got {other:?}"),
        }
    }

    #[test]
    fn decide_step_waits_when_free_ram_below_floor_even_with_gpu_room() {
        // The freeze guard: plenty of VRAM + GPU idle, but host RAM is below the
        // floor ⇒ WAIT (and NOT cpu_fallback — CPU would use even more RAM).
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            min_free_ram_gb: Some(6.0),
            cpu_fallback: true,
            ..Default::default()
        };
        match decide_step(0, Some(40.0), Some(false), Some(2.0), &opts) {
            StepDecision::Wait(reason) => assert!(reason.contains("host RAM")),
            other => panic!("expected Wait (low RAM), got {other:?}"),
        }
    }

    #[test]
    fn decide_step_trains_when_ram_above_floor() {
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            min_free_ram_gb: Some(6.0),
            ..Default::default()
        };
        // RAM above floor + VRAM fine ⇒ train.
        assert_eq!(
            decide_step(0, Some(40.0), Some(false), Some(20.0), &opts),
            StepDecision::Train {
                device: "cuda",
                shard: None
            }
        );
        // RAM ungated (None floor) ⇒ never blocks on RAM even if probe is low.
        let ungated = DaemonOptions {
            max_vram_gb: Some(4.0),
            min_free_ram_gb: None,
            ..Default::default()
        };
        assert_eq!(
            decide_step(0, Some(40.0), Some(false), Some(0.5), &ungated),
            StepDecision::Train {
                device: "cuda",
                shard: None
            }
        );
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
            decide_step(0, Some(1.0), Some(false), None, &opts),
            StepDecision::Wait(_)
        ));
    }

    #[test]
    fn decide_step_rotates_blocks() {
        let opts = DaemonOptions {
            rotation_blocks: 3,
            ..Default::default()
        };
        let shard_of = |ord| match decide_step(ord, None, Some(false), None, &opts) {
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
        let free_ram = || None::<f64>;
        let gpu_busy = || Some(true); // a game is running
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_busy,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
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
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
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
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| anyhow::bail!("eval keeps timing out");
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
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
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
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
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: Some(&degrade),
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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
        let free_ram = || None::<f64>;
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
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: Some(&degrade),
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: None,
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

    // --- Track 37 Phase E: daemon objective defaults to end_task ---

    #[test]
    fn daemon_step_objective_defaults_to_end_task() {
        // apply_plan on a daemon config (no explicit objective) yields end_task on
        // the fractional config the trainer reads — the knowledge signal.
        let dir = tmp("objective");
        let mut cfg = cfg_for(&dir);
        cfg.daemon = Some(crate::config::DaemonConfig::default());
        let planned = apply_plan(&cfg, "cuda", Some(0), 4, "block");
        let obj = planned
            .train
            .and_then(|t| t.fractional)
            .map(|f| f.objective)
            .expect("fractional objective present");
        assert_eq!(obj, "end_task", "daemon steps default to the knowledge signal");
    }

    #[test]
    fn non_sharded_step_does_not_silently_enable_fractional() {
        // Safety guard: FractionalConfig::default().enabled = true, so a
        // non-sharded apply_plan must NOT create a fractional config on a config
        // that had none — else a dense daemon step silently becomes fractional.
        let dir = tmp("no-frac");
        let mut cfg = cfg_for(&dir);
        cfg.daemon = Some(crate::config::DaemonConfig::default());
        cfg.train = None; // no [train] at all
        let planned = apply_plan(&cfg, "cpu", None, 0, "module");
        assert!(
            planned.train.and_then(|t| t.fractional).is_none(),
            "non-sharded step must not materialize a fractional config"
        );
    }

    // --- Track 37 Phase D: live nudge + steering-compliance ---

    #[test]
    fn injected_nudge_is_consumed_once_and_logged() {
        let dir = tmp("nudge");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        q.enqueue(Lane::Raw, &qa("b")).unwrap();
        let free_vram = || Some(40.0);
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = move |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        // Serve a nudge exactly once (first poll), None after.
        let served = Cell::new(false);
        let take = || -> anyhow::Result<Option<crate::Nudge>> {
            if served.get() {
                Ok(None)
            } else {
                served.set(true);
                Ok(Some(crate::Nudge {
                    judge_min_score: Some(0.75),
                    ..Default::default()
                }))
            }
        };
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: Some(&take),
            steering_compliance: None,
            served_ready_emit: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        // The nudge produced a kind:"nudge" evolution-log row.
        let log = crate::regulate::log::read_all(
            &WorkDir::from_config(&cfg).root().join("evolution-log.jsonl"),
        )
        .unwrap();
        let nudge_rows: Vec<_> = log.iter().filter(|e| e.kind == "nudge").collect();
        assert_eq!(nudge_rows.len(), 1, "consumed once → exactly one nudge row");
        assert!(nudge_rows[0]
            .cause
            .as_deref()
            .unwrap()
            .contains("judge.min_score"));
    }

    #[test]
    fn steering_compliance_logged_when_steering_set() {
        let dir = tmp("compliance");
        let mut cfg = cfg_for(&dir);
        cfg.evolve.constitution = Some("Be correct and concise.".to_string());
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = move |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let compliance = |_: &EvolveConfig| Ok(0.5f64);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: Some(&compliance),
            served_ready_emit: None,
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        let log = crate::regulate::log::read_all(
            &WorkDir::from_config(&cfg).root().join("evolution-log.jsonl"),
        )
        .unwrap();
        let comp: Vec<_> = log
            .iter()
            .filter(|e| e.kind == "steering_compliance")
            .collect();
        assert_eq!(comp.len(), 1, "one compliance row for the committed step");
        assert!(comp[0]
            .cause
            .as_deref()
            .unwrap()
            .contains("steering_compliance=0.5"));
    }

    // ───────────────────── Track 33: reservation gating + swap-signal emit ─────────────────────

    #[test]
    fn reservation_subtracts_from_usable_headroom() {
        // budget 4, free 8 ⇒ 4 usable headroom. A 3 GB reservation leaves 5 free
        // ⇒ still ≥ 4 (fits); a 5 GB reservation leaves 3 free ⇒ < 4 (starved).
        let opts_fits = DaemonOptions {
            max_vram_gb: Some(4.0),
            serve_reservation_gb: Some(3.0),
            cpu_fallback: false,
            ..Default::default()
        };
        assert_eq!(
            decide_step(0, Some(8.0), Some(false), None, &opts_fits),
            StepDecision::Train {
                device: "cuda",
                shard: None
            },
            "free − reservation still clears the budget ⇒ train"
        );

        let opts_starved = DaemonOptions {
            max_vram_gb: Some(4.0),
            serve_reservation_gb: Some(5.0),
            cpu_fallback: false,
            ..Default::default()
        };
        assert!(
            matches!(
                decide_step(0, Some(8.0), Some(false), None, &opts_starved),
                StepDecision::Wait(_)
            ),
            "reservation eats the headroom below budget ⇒ wait"
        );

        // Reservation None ⇒ byte-identical to today (8 − 0 ≥ 4 ⇒ train).
        let opts_none = DaemonOptions {
            max_vram_gb: Some(4.0),
            serve_reservation_gb: None,
            cpu_fallback: false,
            ..Default::default()
        };
        assert_eq!(
            decide_step(0, Some(8.0), Some(false), None, &opts_none),
            StepDecision::Train {
                device: "cuda",
                shard: None
            }
        );
    }

    #[test]
    fn keep_emits_exactly_one_served_ready() {
        let dir = tmp("emit-keep");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let emitted = std::cell::RefCell::new(Vec::<ServedReady>::new());
        let emit = |r: ServedReady| emitted.borrow_mut().push(r);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: Some(&emit),
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.committed(), 1);
        let recs = emitted.borrow();
        assert_eq!(recs.len(), 1, "one keep ⇒ exactly one served-ready record");
        assert_eq!(recs[0].version, 1, "first commit is version 1");
        assert_eq!(report.served_version, 1);
    }

    #[test]
    fn rollback_emits_no_served_ready() {
        // The judge gate reports degradation ⇒ Rollback despite good correctness;
        // a rollback must emit NOTHING and leave the version at 0.
        let dir = tmp("emit-rollback");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let degrade = |_: &EvolveConfig| {
            Ok(crate::eval::DegradationReport {
                n: 2,
                regressed: 1,
                worse: vec![false, true],
            })
        };
        let emitted = std::cell::RefCell::new(Vec::<ServedReady>::new());
        let emit = |r: ServedReady| emitted.borrow_mut().push(r);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: Some(&degrade),
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: Some(&emit),
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.committed(), 0, "degradation ⇒ rolled back");
        assert!(emitted.borrow().is_empty(), "rollback emits no signal");
        assert_eq!(report.served_version, 0, "version does not advance on rollback");
    }

    #[test]
    fn catastrophe_emits_no_served_ready() {
        // Correctness below the catastrophe floor (0.10) ⇒ rollback + quarantine +
        // halt; the catastrophe branch must emit NOTHING.
        let dir = tmp("emit-catastrophe");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        let free_vram = || Some(40.0);
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        // Collapse below the 0.10 floor ⇒ Catastrophic.
        let mut collapsed = good_score();
        collapsed.correctness = 0.02;
        let score = move |_: &EvolveConfig| Ok(collapsed.clone());
        let gpu_idle = || Some(false);
        let emitted = std::cell::RefCell::new(Vec::<ServedReady>::new());
        let emit = |r: ServedReady| emitted.borrow_mut().push(r);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: Some(&emit),
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert!(report.halted, "a collapse halts the loop");
        assert_eq!(report.committed(), 0);
        assert!(emitted.borrow().is_empty(), "catastrophe emits no signal");
        assert_eq!(report.served_version, 0);
    }

    #[test]
    fn served_version_is_monotonic_across_keeps() {
        // Three clean keeps ⇒ versions 1, 2, 3 in order; final report carries 3.
        let dir = tmp("emit-monotonic");
        let cfg = cfg_for(&dir);
        let q = LivingQueue::from_config(&cfg).unwrap();
        q.enqueue_many(Lane::Raw, &[qa("a"), qa("b"), qa("c")])
            .unwrap();
        let free_vram = || Some(40.0);
        let free_ram = || None::<f64>;
        let train = |_: &EvolveConfig, _: &Dataset| Ok(vec!["teach".to_string()]);
        let score = |_: &EvolveConfig| Ok(good_score());
        let gpu_idle = || Some(false);
        let emitted = std::cell::RefCell::new(Vec::<ServedReady>::new());
        let emit = |r: ServedReady| emitted.borrow_mut().push(r);
        let hooks = DaemonHooks {
            free_vram_gb: &free_vram,
            free_ram_gb: &free_ram,
            gpu_busy: &gpu_idle,
            train: &train,
            score: &score,
            now_secs: &zero_now,
            sleep: &noop_sleep,
            degrade: None,
            take_nudge: None,
            steering_compliance: None,
            served_ready_emit: Some(&emit),
        };
        let opts = DaemonOptions {
            max_vram_gb: Some(4.0),
            exit_when_empty: true,
            ..Default::default()
        };
        let report = run_daemon(&cfg, &opts, &good_score(), &hooks, &|| false).unwrap();
        assert_eq!(report.committed(), 3);
        let versions: Vec<u64> = emitted.borrow().iter().map(|r| r.version).collect();
        assert_eq!(versions, vec![1, 2, 3], "versions increment once per keep");
        assert_eq!(report.served_version, 3);
    }
}
