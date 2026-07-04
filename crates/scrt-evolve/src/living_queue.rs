//! Living dataset queue (track 26) — the two-lane, append-only training queue
//! that feeds the ambient continuous-evolution daemon.
//!
//! Evolution stops being a scheduled batch and becomes an always-on background
//! process fed by a **living queue** that grows from the user's own activity.
//! Two lanes, both append-only JSONL under `work_dir/queue/`:
//!
//! - **`priority`** — EXPLICIT captures (`evolve ambient teach …`). Skips the
//!   relevance filter, drains FIRST. The user said "learn this," so it's trusted.
//! - **`raw`** — the PASSIVE activity tail (distilled transcripts via
//!   [`crate::harvest`]), gated by goal-relevance before it ever lands here.
//!
//! Consumption is restart-safe: a `cursor.json` records how many items each lane
//! has yielded, persisted atomically (temp + rename) after every pop. Stopping
//! and restarting the daemon resumes exactly where it left off — no lost or
//! duplicated work.
//!
//! Everything here is **ML-free** and clock-free (styleguide §1/§2.2): the queue
//! is pure file I/O over the [`GenExample`] dataset contract, so the daemon's
//! machinery is provable without a GPU. Payloads are the same rows
//! `generate`/`harvest`/`teach` already produce, so a queued item trains exactly
//! like any dataset row.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::EvolveConfig;
use crate::dataset::{Dataset, GenExample};
#[cfg(test)]
use crate::dataset::{Outcome, Tier, Verdict};
use crate::workdir::WorkDir;

/// Which lane a queued example belongs to. `priority` always drains before
/// `raw` (the explicit-capture-first policy, spec §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    /// Explicit `teach` captures — trusted, unfiltered, drained first.
    Priority,
    /// Passive activity tail — goal-filtered transcript distillations.
    Raw,
}

impl Lane {
    fn file_name(self) -> &'static str {
        match self {
            Lane::Priority => "priority.jsonl",
            Lane::Raw => "raw.jsonl",
        }
    }
}

/// One item handed back by [`LivingQueue::pop`]: the example plus where it came
/// from (the lane + its 0-based index in that lane file — the cursor coordinate).
#[derive(Debug, Clone, PartialEq)]
pub struct QueuedItem {
    /// The lane this item was popped from.
    pub lane: Lane,
    /// 0-based index within the lane file (the cursor coordinate this pop advanced past).
    pub ordinal: u64,
    /// The training example payload.
    pub example: GenExample,
}

/// Persisted consumption cursor (`queue/cursor.json`): how many items each lane
/// has already yielded. The restart-safety primitive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Cursor {
    #[serde(default)]
    priority: u64,
    #[serde(default)]
    raw: u64,
}

impl Cursor {
    fn consumed(&self, lane: Lane) -> u64 {
        match lane {
            Lane::Priority => self.priority,
            Lane::Raw => self.raw,
        }
    }
    fn advance(&mut self, lane: Lane) {
        match lane {
            Lane::Priority => self.priority += 1,
            Lane::Raw => self.raw += 1,
        }
    }
}

/// A two-lane append-only training queue rooted at `work_dir/queue/`.
#[derive(Debug, Clone)]
pub struct LivingQueue {
    dir: PathBuf,
}

impl LivingQueue {
    /// Open (creating if needed) the queue under `work_dir/queue/`.
    pub fn open(work_dir: &Path) -> anyhow::Result<Self> {
        let dir = work_dir.join("queue");
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow::anyhow!("living_queue: creating {}: {e}", dir.display()))?;
        Ok(Self { dir })
    }

    /// Open the queue for a config's work-dir.
    pub fn from_config(cfg: &EvolveConfig) -> anyhow::Result<Self> {
        Self::open(WorkDir::from_config(cfg).root())
    }

    fn lane_path(&self, lane: Lane) -> PathBuf {
        self.dir.join(lane.file_name())
    }

    fn cursor_path(&self) -> PathBuf {
        self.dir.join("cursor.json")
    }

    fn load_cursor(&self) -> Cursor {
        std::fs::read_to_string(self.cursor_path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Persist the cursor atomically (temp + rename) so a crash mid-write never
    /// corrupts the restart point (styleguide §2.3).
    fn save_cursor(&self, cursor: &Cursor) -> anyhow::Result<()> {
        let path = self.cursor_path();
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(cursor)?)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Append one example to a lane (the enqueue primitive). The line is written
    /// in a single `write_all`, which is atomic at this scale on a single host.
    pub fn enqueue(&self, lane: Lane, example: &GenExample) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(example)?;
        line.push('\n');
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.lane_path(lane))?;
        f.write_all(line.as_bytes())?;
        Ok(())
    }

    /// Append many examples to a lane; returns the count enqueued.
    pub fn enqueue_many(&self, lane: Lane, examples: &[GenExample]) -> anyhow::Result<usize> {
        if examples.is_empty() {
            return Ok(0);
        }
        let mut buf = String::new();
        for ex in examples {
            buf.push_str(&serde_json::to_string(ex)?);
            buf.push('\n');
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.lane_path(lane))?;
        f.write_all(buf.as_bytes())?;
        Ok(examples.len())
    }

    /// Enqueue a whole harvested dataset into the `raw` lane (the activity-tail
    /// path — distilled, goal-filtered transcript rows).
    pub fn enqueue_raw(&self, dataset: &Dataset) -> anyhow::Result<usize> {
        self.enqueue_many(Lane::Raw, &dataset.rows)
    }

    /// Total lines ever written to a lane (consumed + pending).
    fn lane_len(&self, lane: Lane) -> u64 {
        let path = self.lane_path(lane);
        let Ok(f) = std::fs::File::open(&path) else {
            return 0;
        };
        std::io::BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty())
            .count() as u64
    }

    /// `(priority_pending, raw_pending)` — items not yet consumed in each lane.
    pub fn pending(&self) -> (u64, u64) {
        let cursor = self.load_cursor();
        let p = self
            .lane_len(Lane::Priority)
            .saturating_sub(cursor.consumed(Lane::Priority));
        let r = self
            .lane_len(Lane::Raw)
            .saturating_sub(cursor.consumed(Lane::Raw));
        (p, r)
    }

    /// True when both lanes are fully drained.
    pub fn is_empty(&self) -> bool {
        let (p, r) = self.pending();
        p == 0 && r == 0
    }

    /// Read the example at `index` (0-based) from a lane file, skipping blanks.
    fn read_at(&self, lane: Lane, index: u64) -> anyhow::Result<Option<GenExample>> {
        let path = self.lane_path(lane);
        let Ok(f) = std::fs::File::open(&path) else {
            return Ok(None);
        };
        let mut i = 0u64;
        for line in std::io::BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if i == index {
                let ex: GenExample = serde_json::from_str(line.trim())
                    .map_err(|e| anyhow::anyhow!("living_queue: parse {}: {e}", path.display()))?;
                return Ok(Some(ex));
            }
            i += 1;
        }
        Ok(None)
    }

    /// Pop the next item — priority lane first, then raw — advancing and
    /// persisting the cursor. Returns `None` when both lanes are drained.
    pub fn pop(&self) -> anyhow::Result<Option<QueuedItem>> {
        let mut cursor = self.load_cursor();
        for lane in [Lane::Priority, Lane::Raw] {
            let idx = cursor.consumed(lane);
            if let Some(example) = self.read_at(lane, idx)? {
                cursor.advance(lane);
                self.save_cursor(&cursor)?;
                return Ok(Some(QueuedItem {
                    lane,
                    ordinal: idx,
                    example,
                }));
            }
        }
        Ok(None)
    }

    /// Pop up to `n` items (priority-first). Fewer than `n` ⇒ the queue drained.
    pub fn pop_batch(&self, n: usize) -> anyhow::Result<Vec<QueuedItem>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.pop()? {
                Some(item) => out.push(item),
                None => break,
            }
        }
        Ok(out)
    }

    /// NON-destructive read of the next `n` pending rows (priority-first) without
    /// advancing the cursor. For sampling (track 37 steering-compliance) — the
    /// rows stay queued for the real training pop.
    pub fn peek(&self, n: usize) -> anyhow::Result<Vec<GenExample>> {
        let cursor = self.load_cursor();
        let mut out = Vec::with_capacity(n);
        for lane in [Lane::Priority, Lane::Raw] {
            let mut idx = cursor.consumed(lane);
            while out.len() < n {
                match self.read_at(lane, idx)? {
                    Some(ex) => {
                        out.push(ex);
                        idx += 1;
                    }
                    None => break,
                }
            }
            if out.len() >= n {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qa(prompt: &str) -> GenExample {
        GenExample::Qa {
            prompt: prompt.to_string(),
            completion: format!("answer to {prompt}"),
            source: None,
            gen: Some("teach".to_string()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        }
    }

    fn tmp() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "scrt-evolve-queue-{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        base
    }

    #[test]
    fn enqueue_pop_round_trip() {
        let q = LivingQueue::open(&tmp()).unwrap();
        q.enqueue(Lane::Raw, &qa("a")).unwrap();
        q.enqueue(Lane::Raw, &qa("b")).unwrap();
        assert_eq!(q.pending(), (0, 2));
        let i1 = q.pop().unwrap().unwrap();
        assert_eq!(i1.lane, Lane::Raw);
        let i2 = q.pop().unwrap().unwrap();
        assert_eq!(i2.ordinal, 1);
        assert!(q.pop().unwrap().is_none());
        assert!(q.is_empty());
    }

    #[test]
    fn priority_drains_before_raw() {
        let q = LivingQueue::open(&tmp()).unwrap();
        q.enqueue(Lane::Raw, &qa("raw1")).unwrap();
        q.enqueue(Lane::Priority, &qa("prio1")).unwrap();
        // Priority comes out first despite being enqueued second.
        let first = q.pop().unwrap().unwrap();
        assert_eq!(first.lane, Lane::Priority);
        let second = q.pop().unwrap().unwrap();
        assert_eq!(second.lane, Lane::Raw);
    }

    #[test]
    fn cursor_survives_reopen() {
        let dir = tmp();
        {
            let q = LivingQueue::open(&dir).unwrap();
            q.enqueue_many(Lane::Raw, &[qa("a"), qa("b"), qa("c")])
                .unwrap();
            let _ = q.pop().unwrap().unwrap(); // consume one
        }
        // Reopen: the consumed cursor persists — we resume at index 1.
        let q = LivingQueue::open(&dir).unwrap();
        assert_eq!(q.pending(), (0, 2));
        let next = q.pop().unwrap().unwrap();
        assert_eq!(next.ordinal, 1);
    }

    #[test]
    fn pop_batch_stops_at_drain() {
        let q = LivingQueue::open(&tmp()).unwrap();
        q.enqueue_many(Lane::Priority, &[qa("a"), qa("b")]).unwrap();
        let batch = q.pop_batch(5).unwrap();
        assert_eq!(batch.len(), 2);
    }
}
