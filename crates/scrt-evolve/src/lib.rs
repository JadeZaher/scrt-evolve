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

pub mod config;
pub mod dataset;
pub mod directive;
pub mod discover;
pub mod export;
pub mod generate;
pub mod interview;
pub mod model;
pub mod plan;
pub mod project;
pub mod scaffold;
pub mod toolspec;
pub mod train;
pub mod workdir;

// The PyO3 bridge module only exists under `--features pyo3` (it needs Python
// headers); a default build does not reference it at all.
#[cfg(feature = "pyo3")]
pub mod bridge;

pub use config::{ConfigError, EvolveConfig};
pub use dataset::{Dataset, GenExample};
pub use directive::TrainingDirective;
pub use discover::DiscoveredContext;
pub use export::{export_llamacpp, ExportReport, ToolFormat};
pub use train::TrainReport;
pub use workdir::WorkDir;
