//! Work-dir layout helper.
//!
//! Every stage reads/writes durable artifacts under `work_dir` so stages are
//! independently runnable and inspectable: `discover` → `discovered.json`,
//! `generate` → `dataset.jsonl`, `train` → `adapter.safetensors` /
//! checkpoints. This resolves those paths from an [`EvolveConfig`].

use std::path::{Path, PathBuf};

use crate::config::EvolveConfig;

/// Resolves the canonical artifact paths under a config's `work_dir`.
#[derive(Debug, Clone)]
pub struct WorkDir {
    root: PathBuf,
}

impl WorkDir {
    /// Build a layout rooted at the config's resolved `work_dir`.
    pub fn from_config(cfg: &EvolveConfig) -> Self {
        Self {
            root: cfg.work_dir(),
        }
    }

    /// Build a layout rooted at an explicit directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The work-dir root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `discover` output: the retrieved-context artifact.
    pub fn discovered_json(&self) -> PathBuf {
        self.root.join("discovered.json")
    }

    /// `generate` output: the JSONL dataset (the generate↔train contract).
    pub fn dataset_jsonl(&self) -> PathBuf {
        self.root.join("dataset.jsonl")
    }

    /// `train` (lora preset) output.
    pub fn adapter_safetensors(&self) -> PathBuf {
        self.root.join("adapter.safetensors")
    }

    /// The traces root: harvested frontier transcripts live under
    /// `work_dir/traces/` (track 20 slice 4), one subdir per goal.
    pub fn traces_dir(&self) -> PathBuf {
        self.root.join("traces")
    }

    /// The per-goal traces subdir: `work_dir/traces/<goal>/`. The harvester
    /// captures raw transcripts here (`<slug>-<date>.jsonl`) before filtering.
    pub fn goal_traces_dir(&self, goal: &str) -> PathBuf {
        self.traces_dir().join(goal)
    }

    /// The checkpoints directory for longer training runs.
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    /// A named checkpoint file under [`Self::checkpoints_dir`].
    pub fn checkpoint(&self, name: &str) -> PathBuf {
        self.checkpoints_dir().join(name)
    }

    /// Create the work-dir root (and checkpoints dir) if absent.
    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.checkpoints_dir())
    }
}
