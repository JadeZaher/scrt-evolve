//! Evolution log (track 15) — `work_dir/evolution-log.jsonl`.
//!
//! One row per transactional step: what ran, the verdict, the action taken, and
//! the cause. The append-only audit trail of how the model evolved — and what it
//! refused to keep (styleguide §2.4: durable-mutation audit is not optional).

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::eval::{ScoreReport, StepVerdict};

/// The action a step's verdict produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepAction {
    /// Verdict accepted — checkpoint committed, `last_good` advanced.
    Commit,
    /// Soft regression — this step rolled back; loop continues.
    Rollback,
    /// Catastrophe — rolled back + cause quarantined + loop halted.
    Quarantine,
}

/// One evolution-log row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionLogEntry {
    /// Monotonic step ordinal.
    pub step: u64,
    /// The checkpoint id this step produced.
    pub checkpoint_id: String,
    /// What kind of step (e.g. `train`, `round:<goal>`).
    pub kind: String,
    /// The verdict (absent if the step errored before eval).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<StepVerdict>,
    /// The metrics that produced the verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<ScoreReport>,
    /// The action taken.
    pub action: StepAction,
    /// Free-text cause / note (e.g. the quarantined provenance, an error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<String>,
}

/// Append one entry to `evolution-log.jsonl` (creating it if absent). Append is
/// the durability primitive — never rewrite prior rows.
pub fn append(log_path: &Path, entry: &EvolutionLogEntry) -> anyhow::Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(entry)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

/// Read all entries from an evolution log (for `checkpoints`/observability).
pub fn read_all(log_path: &Path) -> anyhow::Result<Vec<EvolutionLogEntry>> {
    if !log_path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(log_path)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("evolution-log line {}: {e}", i + 1))?,
        );
    }
    Ok(out)
}
