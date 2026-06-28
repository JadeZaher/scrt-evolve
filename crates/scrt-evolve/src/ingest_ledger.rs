//! Already-ingested content ledger (track 31 Q5) — the dedup memory that stops
//! the ambient self-feed from re-training rows it has already absorbed.
//!
//! The living-queue cursor ([`crate::living_queue`]) is *positional*: it records
//! how many lines each lane has yielded, so a restart resumes correctly. But
//! `auto_ingest` re-mines the SAME interaction logs on every refill, and
//! `enqueue_many` APPENDS unconditionally — so an identical row mined twice lands
//! twice, past the cursor, and trains twice. Over a stale corpus that means
//! re-training the same ~400 rows in a loop (overfitting the eval gate won't
//! catch). See `src/AGENTS.md` §ingest_ledger.rs.
//!
//! The ledger is the fix: a persistent SET of content-hashes of every row ever
//! enqueued. `run_ingest` consults it to drop already-seen rows and records the
//! new ones. ML-free, append-only, atomic — same discipline as the queue cursor.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use crate::dataset::GenExample;

/// File name under `work_dir/queue/` holding one content-hash per line.
const LEDGER_FILE: &str = "ingested.ledger";

/// A persistent set of already-ingested content hashes, rooted at
/// `work_dir/queue/ingested.ledger`. Cheap to open (one read); appends are
/// flushed immediately so a crash mid-refill never loses what was enqueued.
#[derive(Debug)]
pub struct IngestLedger {
    path: PathBuf,
    seen: HashSet<String>,
}

impl IngestLedger {
    /// Open (loading existing hashes) the ledger under `work_dir/queue/`.
    pub fn open(work_dir: &Path) -> anyhow::Result<Self> {
        let dir = work_dir.join("queue");
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow::anyhow!("ingest_ledger: creating {}: {e}", dir.display()))?;
        let path = dir.join(LEDGER_FILE);
        let mut seen = HashSet::new();
        if let Ok(f) = std::fs::File::open(&path) {
            for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
                let h = line.trim();
                if !h.is_empty() {
                    seen.insert(h.to_string());
                }
            }
        }
        Ok(Self { path, seen })
    }

    /// How many distinct rows have ever been ingested.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// True if the ledger has recorded nothing yet.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// True if this row's content has already been ingested.
    pub fn contains(&self, row: &GenExample) -> bool {
        self.seen.contains(&content_hash(row))
    }

    /// Partition `rows` into (genuinely new, already-seen-count), recording the
    /// new ones' hashes both in memory and (appended, flushed) on disk. The new
    /// rows are returned in input order; duplicates *within* this call collapse
    /// to one (the first wins). This is the single chokepoint `run_ingest` uses
    /// before enqueueing.
    pub fn filter_new(&mut self, rows: Vec<GenExample>) -> anyhow::Result<FilterOutcome> {
        let mut fresh = Vec::with_capacity(rows.len());
        let mut new_hashes = Vec::new();
        let mut skipped = 0usize;
        for row in rows {
            let h = content_hash(&row);
            if self.seen.contains(&h) {
                skipped += 1;
                continue;
            }
            // Reserve the hash so an intra-batch duplicate is also dropped.
            self.seen.insert(h.clone());
            new_hashes.push(h);
            fresh.push(row);
        }
        if !new_hashes.is_empty() {
            self.append(&new_hashes)?;
        }
        Ok(FilterOutcome {
            new: fresh,
            skipped,
        })
    }

    /// Append hashes to the on-disk ledger (atomic at this scale: one write).
    fn append(&self, hashes: &[String]) -> anyhow::Result<()> {
        let mut buf = String::with_capacity(hashes.len() * 33);
        for h in hashes {
            buf.push_str(h);
            buf.push('\n');
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(buf.as_bytes())?;
        Ok(())
    }
}

/// What [`IngestLedger::filter_new`] returns: the genuinely-new rows + how many
/// were dropped as already-ingested.
#[derive(Debug)]
pub struct FilterOutcome {
    /// Rows not previously ingested (in input order, intra-batch deduped).
    pub new: Vec<GenExample>,
    /// How many input rows were dropped because they were already ingested.
    pub skipped: usize,
}

/// A stable content hash for a row — the ledger key. Hashes the SAME compact
/// content shape ingest already dedups on (variant + the human-meaningful
/// fields), NOT the serialized JSON (which carries provenance like `gen`/`source`
/// that we deliberately ignore so the same usage re-mined from a different
/// transcript still counts as a duplicate). FNV-1a → a short hex string; we only
/// need set-membership, not cryptographic strength.
pub fn content_hash(row: &GenExample) -> String {
    let key = content_key(row);
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{hash:016x}")
}

/// The human-meaningful content of a row, ignoring provenance — mirrors the dedup
/// key shape in `ingest::content_key` (kept local so the ledger has no cross-module
/// coupling to that private fn).
fn content_key(row: &GenExample) -> String {
    match row {
        GenExample::Cli {
            prompt, command, ..
        } => format!("cli\u{1}{}\u{1}{}", prompt.trim(), command.trim()),
        GenExample::ToolCall {
            prompt,
            tool,
            arguments,
            ..
        } => format!(
            "tool\u{1}{}\u{1}{}\u{1}{}",
            prompt.trim(),
            tool,
            serde_json::to_string(arguments).unwrap_or_default()
        ),
        GenExample::Skill {
            skill_name,
            invocation,
            ..
        } => format!("skill\u{1}{}\u{1}{}", skill_name.trim(), invocation.trim()),
        GenExample::ReasoningEdit {
            prompt,
            final_action,
            ..
        } => format!("reason\u{1}{}\u{1}{}", prompt.trim(), final_action.trim()),
        GenExample::Qa {
            prompt, completion, ..
        } => format!("qa\u{1}{}\u{1}{}", prompt.trim(), completion.trim()),
        GenExample::Instruction {
            instruction,
            output,
            ..
        } => format!("instr\u{1}{}\u{1}{}", instruction.trim(), output.trim()),
        GenExample::Completion { text, .. } => format!("compl\u{1}{}", text.trim()),
        GenExample::Contrastive { query, .. } => format!("contrast\u{1}{}", query.trim()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(cmd: &str) -> GenExample {
        GenExample::Cli {
            prompt: "p".into(),
            command: cmd.into(),
            source: Some("transcript".into()),
            gen: Some("ingest".into()),
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "scrt-evolve-ledger-{tag}-{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        base
    }

    #[test]
    fn first_ingest_is_all_new_second_is_all_skipped() {
        let dir = tmp("dedup");
        let mut led = IngestLedger::open(&dir).unwrap();
        let rows = vec![cli("scrt --mp-list"), cli("scrt --effort scan")];
        let out = led.filter_new(rows.clone()).unwrap();
        assert_eq!(out.new.len(), 2);
        assert_eq!(out.skipped, 0);
        // Same rows again → nothing new.
        let out2 = led.filter_new(rows).unwrap();
        assert_eq!(out2.new.len(), 0);
        assert_eq!(out2.skipped, 2);
    }

    #[test]
    fn provenance_is_ignored_for_dedup() {
        let dir = tmp("prov");
        let mut led = IngestLedger::open(&dir).unwrap();
        led.filter_new(vec![cli("scrt --mp-stash x")]).unwrap();
        // Same usage, different source/gen (re-mined from another transcript).
        let other = GenExample::Cli {
            prompt: "p".into(),
            command: "scrt --mp-stash x".into(),
            source: Some("other-file".into()),
            gen: Some("ingest:doc".into()),
        };
        let out = led.filter_new(vec![other]).unwrap();
        assert_eq!(out.new.len(), 0, "same content, different provenance → dup");
        assert_eq!(out.skipped, 1);
    }

    #[test]
    fn intra_batch_duplicates_collapse() {
        let dir = tmp("intra");
        let mut led = IngestLedger::open(&dir).unwrap();
        let out = led
            .filter_new(vec![cli("scrt a"), cli("scrt a"), cli("scrt b")])
            .unwrap();
        assert_eq!(out.new.len(), 2);
        assert_eq!(out.skipped, 1);
    }

    #[test]
    fn ledger_survives_reopen() {
        let dir = tmp("reopen");
        {
            let mut led = IngestLedger::open(&dir).unwrap();
            led.filter_new(vec![cli("scrt persisted")]).unwrap();
            assert_eq!(led.len(), 1);
        }
        let mut led = IngestLedger::open(&dir).unwrap();
        assert_eq!(led.len(), 1, "hashes loaded from disk");
        let out = led.filter_new(vec![cli("scrt persisted")]).unwrap();
        assert_eq!(out.new.len(), 0, "still recognized after reopen");
    }

    #[test]
    fn distinct_rows_still_enqueue() {
        let dir = tmp("distinct");
        let mut led = IngestLedger::open(&dir).unwrap();
        led.filter_new(vec![cli("scrt one")]).unwrap();
        let out = led.filter_new(vec![cli("scrt two")]).unwrap();
        assert_eq!(out.new.len(), 1, "a genuinely new row is not blocked");
        assert_eq!(out.skipped, 0);
    }
}
