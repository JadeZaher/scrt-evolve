//! Checkpoint store (track 15) — `work_dir/checkpoints/<id>/`.
//!
//! Every weight-mutating step snapshots first so it is always reversible
//! (styleguide §2.3). For the LoRA path the mutable artifact is the **adapter
//! dir** (`work_dir/adapter`): the base weights are never touched, so a snapshot
//! is a copy of the adapter + a manifest. (The spec's "base weights as deltas"
//! complexity only applies to full/base training, which is feature-gated and not
//! exercised by the bench — documented seam.)
//!
//! Layout:
//! ```text
//! work_dir/checkpoints/
//!   last_good            # text file: the id of the last committed-good ckpt
//!   <id>/manifest.json   # CheckpointManifest
//!   <id>/adapter/        # snapshot of the adapter at checkpoint time
//! ```
//! Writes are atomic (temp + rename); a crash mid-snapshot never leaves a
//! half-written manifest.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::eval::ScoreReport;

/// Status of a checkpoint in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointStatus {
    /// Snapshot taken, step not yet evaluated.
    Pending,
    /// Committed — eval passed; this is (or was) a `last_good` candidate.
    Good,
    /// Rolled back — eval regressed; state restored from the parent.
    Reverted,
    /// Quarantined — a catastrophe traced to this step; its cause is skipped.
    Quarantined,
}

/// The manifest written to `<id>/manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub id: String,
    /// The checkpoint this step started from (the restore target on rollback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// What kind of step this snapshot wraps (e.g. `train`, `round:<goal>`).
    pub step_kind: String,
    /// A monotonically increasing creation ordinal (no wall-clock — determinism;
    /// the driver supplies it).
    pub ordinal: u64,
    /// The eval report for the step, once evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<ScoreReport>,
    pub status: CheckpointStatus,
    /// Relative artifact paths captured in this checkpoint (e.g. `adapter`).
    #[serde(default)]
    pub artifacts: Vec<String>,
    /// The provenance stamp(s) this step's training rows carried — the key by
    /// which a catastrophe quarantines the cause (styleguide §2.4).
    #[serde(default)]
    pub gen_provenance: Vec<String>,
}

/// Errors from the checkpoint store.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint io: {0}")]
    Io(#[from] std::io::Error),
    #[error("checkpoint json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("checkpoint `{id}` not found under {root}")]
    NotFound { id: String, root: String },
}

/// The on-disk checkpoint store rooted at `work_dir/checkpoints/`.
#[derive(Debug, Clone)]
pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    /// Open (creating the root if needed) a store at `checkpoints_root`.
    pub fn open(checkpoints_root: impl Into<PathBuf>) -> Result<Self, CheckpointError> {
        let root = checkpoints_root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The store root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn ckpt_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }
    fn manifest_path(&self, id: &str) -> PathBuf {
        self.ckpt_dir(id).join("manifest.json")
    }
    fn last_good_path(&self) -> PathBuf {
        self.root.join("last_good")
    }

    /// Snapshot `adapter_dir` (if present) into a new `Pending` checkpoint and
    /// write its manifest. Returns the manifest. Atomic: the manifest is the
    /// last thing written, so a half-copied snapshot is never marked complete.
    pub fn snapshot(
        &self,
        id: &str,
        parent_id: Option<String>,
        step_kind: &str,
        ordinal: u64,
        adapter_dir: &Path,
        gen_provenance: Vec<String>,
    ) -> Result<CheckpointManifest, CheckpointError> {
        let dir = self.ckpt_dir(id);
        std::fs::create_dir_all(&dir)?;

        let mut artifacts = Vec::new();
        if adapter_dir.exists() {
            let dest = dir.join("adapter");
            copy_dir_all(adapter_dir, &dest)?;
            artifacts.push("adapter".to_string());
        }

        let manifest = CheckpointManifest {
            id: id.to_string(),
            parent_id,
            step_kind: step_kind.to_string(),
            ordinal,
            metrics: None,
            status: CheckpointStatus::Pending,
            artifacts,
            gen_provenance,
        };
        self.write_manifest(&manifest)?;
        Ok(manifest)
    }

    /// Read a checkpoint's manifest.
    pub fn load_manifest(&self, id: &str) -> Result<CheckpointManifest, CheckpointError> {
        let path = self.manifest_path(id);
        if !path.exists() {
            return Err(CheckpointError::NotFound {
                id: id.to_string(),
                root: self.root.display().to_string(),
            });
        }
        let text = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Write/overwrite a manifest atomically.
    pub fn write_manifest(&self, manifest: &CheckpointManifest) -> Result<(), CheckpointError> {
        let path = self.manifest_path(&manifest.id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(manifest)?;
        crate::harvest::write_atomic(&path, json.as_bytes())?;
        Ok(())
    }

    /// Mark a checkpoint `Good`, attach its metrics, and advance `last_good`.
    pub fn commit(
        &self,
        id: &str,
        metrics: ScoreReport,
    ) -> Result<CheckpointManifest, CheckpointError> {
        let mut m = self.load_manifest(id)?;
        m.status = CheckpointStatus::Good;
        m.metrics = Some(metrics);
        self.write_manifest(&m)?;
        self.set_last_good(id)?;
        self.enforce_retention(usize::MAX)?; // retention applied explicitly elsewhere
        Ok(m)
    }

    /// Restore the adapter from checkpoint `id` back into `adapter_dir`
    /// (replacing it), and mark the rolled-back checkpoint `Reverted`. Used by
    /// both soft-regress rollback (restore parent) and catastrophe rollback.
    pub fn restore_adapter(&self, id: &str, adapter_dir: &Path) -> Result<(), CheckpointError> {
        let snap = self.ckpt_dir(id).join("adapter");
        if snap.exists() {
            // Replace the live adapter dir with the snapshot, atomically-ish:
            // remove the old, copy the snapshot in.
            if adapter_dir.exists() {
                std::fs::remove_dir_all(adapter_dir)?;
            }
            copy_dir_all(&snap, adapter_dir)?;
        } else if adapter_dir.exists() {
            // The restore target had NO adapter (e.g. base-only checkpoint):
            // remove the live adapter so state matches the snapshot.
            std::fs::remove_dir_all(adapter_dir)?;
        }
        Ok(())
    }

    /// Mark a checkpoint with a terminal status (Reverted / Quarantined).
    pub fn mark(&self, id: &str, status: CheckpointStatus) -> Result<(), CheckpointError> {
        let mut m = self.load_manifest(id)?;
        m.status = status;
        self.write_manifest(&m)?;
        Ok(())
    }

    /// The current `last_good` id, if any.
    pub fn last_good(&self) -> Option<String> {
        std::fs::read_to_string(self.last_good_path())
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Point `last_good` at `id` (atomic).
    pub fn set_last_good(&self, id: &str) -> Result<(), CheckpointError> {
        crate::harvest::write_atomic(&self.last_good_path(), id.as_bytes())?;
        Ok(())
    }

    /// List all checkpoint ids (sorted by ordinal then id for determinism).
    pub fn list(&self) -> Result<Vec<CheckpointManifest>, CheckpointError> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            if let Ok(m) = self.load_manifest(&id) {
                out.push(m);
            }
        }
        out.sort_by(|a, b| a.ordinal.cmp(&b.ordinal).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    /// Prune older `Good`/`Reverted` checkpoints, keeping the `keep` most recent
    /// good ones (by ordinal). The `last_good` checkpoint and any `Quarantined`
    /// ones are never pruned (the audit trail of refusals is kept).
    pub fn enforce_retention(&self, keep: usize) -> Result<(), CheckpointError> {
        if keep == usize::MAX {
            return Ok(());
        }
        let all = self.list()?;
        let last_good = self.last_good();
        let mut goods: Vec<&CheckpointManifest> = all
            .iter()
            .filter(|m| m.status == CheckpointStatus::Good)
            .collect();
        goods.sort_by(|a, b| b.ordinal.cmp(&a.ordinal)); // newest first
        for (i, m) in goods.iter().enumerate() {
            if i < keep {
                continue;
            }
            if Some(&m.id) == last_good.as_ref() {
                continue; // never remove last_good
            }
            let _ = std::fs::remove_dir_all(self.ckpt_dir(&m.id));
        }
        Ok(())
    }
}

/// Recursively copy a directory tree (used to snapshot/restore the adapter).
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}
