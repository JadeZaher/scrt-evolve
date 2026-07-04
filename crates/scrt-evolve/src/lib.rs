//! # scrt-evolve
//!
//! Make a model *better at its own corpus* with no human labeling. A
//! self-contained **discover → generate → train** loop over a corpus + a scrt
//! mind-palace. The SDK is primary; the CLI (`scrt-evolve-cli`) is a thin
//! argv→SDK shim.
//!
//! ```no_run
//! use scrt_evolve::{EvolveConfig, discover, generate, train};
//!
//! let cfg = EvolveConfig::load("evolve.toml")?;
//! let ctx = discover::run(&cfg)?;
//! let dataset = generate::run(&cfg, &ctx)?;
//! let report = train::run(&cfg, &dataset)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ML is opt-in behind the `train` feature (candle); the Python bridge is
//! opt-in behind `pyo3`. A default build pulls neither.

pub mod arbitration;
pub mod branch;
pub mod config;
pub mod daemon;
pub mod dataset;
pub mod directive;
pub mod discover;
pub mod eval;
pub mod export;
pub mod generate;
pub mod goals;
pub mod harvest;
pub mod ingest;
pub mod ingest_ledger;
pub mod interview;
pub mod judge;
pub mod living_queue;
pub mod model;
pub mod model_store;
pub mod nudge;
pub mod plan;
pub mod project;
pub mod regulate;
pub mod rounds;
pub mod scaffold;
pub mod serve;
pub mod toolspec;
pub mod train;
pub mod trend;
pub mod workdir;

// The PyO3 bridge module only exists under `--features pyo3` (it needs Python
// headers); a default build does not reference it at all.
#[cfg(feature = "pyo3")]
pub mod bridge;

pub use arbitration::{
    append_served_ready, read_served_ready, select_mode, Mode, ServedReady,
};
pub use branch::{
    BranchManifest, BranchRef, BranchRegistry, BranchRouter, Lineage, LocalBranchRouter,
    RouterSignature,
};
pub use config::{
    BranchConfig, BranchEnsembleConfig, BranchRouterConfig, BranchServeConfig, ExportConfig,
    FractionalConfig, HardwareConfig, MergeShardsConfig, RegulateConfig, RuntimeConfig,
    SamplingConfig, ServeConfig, ServeMode,
};
pub use config::{
    ConfigError, DaemonConfig, EvalConfig, EvolveConfig, GoalConfig, IngestConfig, StoreConfig,
};
pub use daemon::{run_daemon, DaemonHooks, DaemonOptions, DaemonReport, DaemonStep};
pub use dataset::{Dataset, GenExample, Outcome, Tier, Verdict};
pub use directive::TrainingDirective;
pub use discover::DiscoveredContext;
pub use eval::{ProbeSet, ScoreReport, StepVerdict};
pub use export::{export_llamacpp, ExportReport, ToolFormat};
pub use goals::{GoalRun, GoalsReport};
pub use harvest::{capture_and_harvest, HarvestResult, TranscriptEntry};
pub use ingest::{
    append_rejected, doc_completion_rows, filter_outcomes, filter_relevant, interaction_log_rows,
    LlmRelevanceJudge, OutcomeFilter, RelevanceJudge, INGEST_GEN_STAMP, REJECTED_SIDECAR,
};
pub use ingest_ledger::{content_hash, FilterOutcome, IngestLedger};
pub use judge::{
    dataset_signal_stats, dataset_tier, expand_dataset, judge_rows, rejection_sample, JudgedRows,
    LlmPairJudge, OnError, PairJudge,
};
pub use living_queue::{Lane, LivingQueue, QueuedItem};
pub use nudge::{apply_nudge, take_nudge, write_nudge, Nudge, NudgeOutcome};
pub use model_store::{ModelStore, ModelVersion, ResolvedVersion, StoreManifest};
pub use regulate::{
    CheckpointStore, EvolutionLogEntry, Quarantine, Regulator, StepAction, TxnOutcome,
};
pub use rounds::{
    run_round, run_schedule, RoundHooks, RoundReport, SchedulePolicy, ScheduleReport,
};
pub use trend::{
    from_log as trend_from_log, steering_compliance_from_log, TrendPoint, TrendSummary,
};

use std::path::PathBuf;

/// Find the directory that should be on `PYTHONPATH` so the standalone Python
/// packages (`scrt_evolve_train`, `scrt_evolve_infer`, `scrt_evolve_gguf`,
/// `scrt_evolve_score`) import: the `python/` dir holding them. Walks up from the
/// current dir so the CLI works from any checkout subdir. One implementation
/// shared by the CLI and the eval subprocess scorer.
pub fn python_pkg_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("python");
        if candidate.join("scrt_evolve_train").is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}
pub use train::TrainReport;
pub use workdir::WorkDir;
