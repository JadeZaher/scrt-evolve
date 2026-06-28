//! Evaluation harness (track 10) — the shared scoring substrate.
//!
//! The self-evolve lane (regen gate track 11, self-regulation track 15) all need
//! "score the model on a held-out probe and tell me if it got worse." This
//! module builds that ONCE:
//! - [`gate::ExecutableGate`] — the pure correctness primitive (does the emitted
//!   command/tool-call resolve against the real scrt surface),
//! - [`probe::ProbeSet`] — a fixed, versioned, held-out exam carved with asserted
//!   zero-overlap,
//! - [`score::Scorer`] / [`score::ScoreReport`] — score a model → comparable
//!   metrics (api backend = no ML; transformers backend = real forward pass via
//!   the track-19 Python subprocess),
//! - [`verdict::StepVerdict`] — the pure accept/regress/catastrophic decision the
//!   consumers share.
//!
//! The default Rust build is ML-free + Python-free: the api scorer + gate +
//! probe + verdict all compile with no candle and no Python; the heavy forward
//! pass is an external subprocess.

pub mod degrade;
pub mod gate;
pub mod probe;
pub mod score;
pub mod verdict;

use std::path::{Path, PathBuf};

use crate::config::{EvalConfig, EvolveConfig};
use crate::generate::api::ApiEndpoint;

pub use degrade::{DegradationJudge, DegradationReport, DegradationTriple, LlmDegradationJudge};
pub use gate::{ExecutableGate, GateFailure, GateVerdict};
pub use probe::ProbeSet;
pub use score::{ApiScorer, ScoreReport, Scorer};
pub use verdict::{classify, judge_verdict, StepVerdict, VerdictError, VerdictTolerances};

/// Resolve the probe path for a config: explicit `[evolve.eval].probe_path`, or
/// `work_dir/probe.jsonl` by default.
pub fn probe_path(cfg: &EvolveConfig) -> PathBuf {
    cfg.eval
        .as_ref()
        .and_then(|e| e.probe_path.clone())
        .unwrap_or_else(|| cfg.work_dir().join("probe.jsonl"))
}

/// Score the current model against the config's probe set, dispatching on the
/// configured scorer backend. Returns the [`ScoreReport`]. This is the SDK entry
/// point the CLI `eval` subcommand and the round driver (track 15) call.
///
/// Backends:
/// - `api` — generate completions via `[generate.api]`, judge with the gate.
///   No ML deps.
/// - `transformers` — shell out to `python -m scrt_evolve_score` (real forward
///   pass: perplexity / exit-depth). See [`score_transformers`].
/// - `candle` — optional native path (`--features train`); not built here.
pub fn run_eval(cfg: &EvolveConfig, python: Option<&str>) -> anyhow::Result<ScoreReport> {
    let ecfg = cfg.eval.clone().unwrap_or_default();
    let ppath = probe_path(cfg);

    if !ppath.exists() {
        eprintln!(
            "eval: no probe set at {} — returning an uncovered report (run \
             `scrt-evolve probe build` first to gate rounds)",
            ppath.display()
        );
        return Ok(ScoreReport::uncovered("probe-none", &ecfg.scorer_backend));
    }
    let probe = ProbeSet::load(&ppath)?;

    match ecfg.scorer_backend.as_str() {
        "api" => {
            // Reuse the [generate.api] endpoint as the probe-completion model.
            let gcfg = cfg.generate.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "eval: scorer_backend=\"api\" needs a [generate.api] block (the \
                     model that answers probe items)"
                )
            })?;
            let endpoint = ApiEndpoint::from_config(&gcfg)?;
            let gate = ExecutableGate::new()?;
            let scorer = ApiScorer::new(endpoint.into_transport(), gate);
            scorer.score(&probe)
        }
        "transformers" => score_transformers(cfg, &probe, &ppath, python),
        "candle" => anyhow::bail!(
            "eval: scorer_backend=\"candle\" is the optional native path (not built \
             yet) — use \"api\" or \"transformers\""
        ),
        other => anyhow::bail!(
            "eval: unknown scorer_backend \"{other}\" (expected api | transformers | candle)"
        ),
    }
}

/// The **transformers** scorer: shell out to `python -m scrt_evolve_score` for a
/// real forward pass (perplexity / exit-depth / executable correctness on
/// generated completions). Mirrors the track-19 subprocess contract: the model
/// path comes from `[evolve].model_path`, the probe is passed by path, and the
/// Python process prints a JSON [`ScoreReport`] as its final stdout line.
pub fn score_transformers(
    cfg: &EvolveConfig,
    probe: &ProbeSet,
    probe_path: &Path,
    python: Option<&str>,
) -> anyhow::Result<ScoreReport> {
    let model_path = cfg
        .evolve
        .model_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("eval(transformers): set [evolve].model_path"))?;

    let pkg_parent = crate::python_pkg_dir()
        .ok_or_else(|| anyhow::anyhow!("eval(transformers): could not locate python/ dir"))?;

    let py = python.unwrap_or("python");
    let adapter_dir = cfg.work_dir().join("adapter");

    let mut cmd = std::process::Command::new(py);
    cmd.arg("-m")
        .arg("scrt_evolve_score")
        .arg("--model")
        .arg(&model_path)
        .arg("--probe")
        .arg(probe_path)
        .env("PYTHONPATH", &pkg_parent);
    // Pass the adapter if one has been trained (score the in-training model).
    if adapter_dir.exists() {
        cmd.arg("--adapter").arg(&adapter_dir);
    }

    let output = cmd.output().map_err(|e| {
        anyhow::anyhow!("eval(transformers): launching `{py} -m scrt_evolve_score`: {e}")
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "eval(transformers): scorer exited with {}\n{}",
            output.status,
            stderr
        );
    }

    // The last non-empty stdout line is the JSON ScoreReport.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("eval(transformers): scorer produced no output"))?;
    let mut report: ScoreReport = serde_json::from_str(last.trim()).map_err(|e| {
        anyhow::anyhow!(
            "eval(transformers): scorer output was not a ScoreReport: {e}\nline: {last}"
        )
    })?;
    // Trust our own probe version over whatever the subprocess echoed, so verdict
    // comparison is anchored to the Rust-side probe content.
    report.probe_version = probe.version.clone();
    report.backend = "transformers".to_string();
    Ok(report)
}

/// Resolve verdict tolerances from a config (currently defaults; a future
/// `[evolve.eval].tolerances` block can override).
pub fn tolerances_for(_ecfg: &EvalConfig) -> VerdictTolerances {
    VerdictTolerances::default()
}
