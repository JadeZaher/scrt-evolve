//! `ProbeSet` — the fixed, versioned, held-out evaluation set (track 10).
//!
//! A probe set is the model's exam: a set of items NEVER trained on, used to
//! score the model before/after a round so a regression is detectable. Items
//! mirror the dataset kinds ([`GenExample`]) so the same gate/scorer logic
//! applies. The set is:
//! - **held out** — a builder carves it from a dataset and asserts ZERO overlap
//!   with the training rows (a probe that leaked into training is worthless);
//! - **versioned** — every report stamps `probe_version` so a candidate report
//!   is only ever compared against a same-version baseline (track 15/verdict);
//! - **deterministic** — load order is stable; the carve is a stable, seedless
//!   hash split (no RNG — styleguide §2.2).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dataset::{Dataset, GenExample};

/// A held-out probe set: the items + a content version stamp.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProbeSet {
    /// A stable version string derived from the items' content (so identical
    /// probe content ⇒ identical version, and any change bumps it).
    pub version: String,
    /// The held-out items (a probe item is just a dataset row carrying its
    /// `gen` provenance, never trained on).
    pub items: Vec<GenExample>,
}

impl ProbeSet {
    /// Number of probe items.
    pub fn len(&self) -> usize {
        self.items.len()
    }
    /// Whether the probe set is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Load a probe set from `probe.jsonl` (one item per line) deterministically.
    /// The version is recomputed from the loaded content so a hand-edited probe
    /// file gets a fresh version automatically.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let ds = Dataset::read_jsonl(path)?;
        Ok(Self::from_items(ds.rows))
    }

    /// Write the probe set to `probe.jsonl` (atomically).
    pub fn write(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let ds = Dataset::new(self.items.clone());
        let jsonl = ds.to_jsonl()?;
        crate::harvest::write_atomic(path.as_ref(), jsonl.as_bytes())?;
        Ok(())
    }

    /// Build a probe set from items, stamping a content-derived version.
    pub fn from_items(items: Vec<GenExample>) -> Self {
        let version = content_version(&items);
        Self { version, items }
    }

    /// Carve a held-out probe set out of a dataset by a deterministic hash split.
    ///
    /// `holdout_frac` (0.0..=1.0) of the rows go to the probe; the rest are the
    /// training remainder. The split is a stable hash of each row's content
    /// (no RNG, no order dependence), so the SAME dataset always carves the SAME
    /// probe — reproducible across runs (styleguide §2.2). Returns
    /// `(probe, train_remainder)`. Zero-overlap is guaranteed by construction
    /// (a row goes to exactly one side) and re-asserted by [`Self::assert_no_overlap`].
    pub fn carve(dataset: &Dataset, holdout_frac: f32) -> anyhow::Result<(Self, Dataset)> {
        if !(0.0..=1.0).contains(&holdout_frac) {
            anyhow::bail!("probe carve: holdout_frac must be in 0.0..=1.0, got {holdout_frac}");
        }
        // Map the fraction to a hash-bucket threshold over u16 space.
        let threshold = (holdout_frac * u16::MAX as f32).round() as u32;

        let mut probe_items = Vec::new();
        let mut train_rows = Vec::new();
        for row in &dataset.rows {
            let bucket = (row_hash(row) % (u16::MAX as u64 + 1)) as u32;
            if bucket < threshold {
                probe_items.push(row.clone());
            } else {
                train_rows.push(row.clone());
            }
        }

        let probe = Self::from_items(probe_items);
        let train = Dataset::new(train_rows);
        probe.assert_no_overlap(&train)?;
        Ok((probe, train))
    }

    /// Return `dataset` with every row that matches a probe item (by content
    /// key) removed — the training remainder when REUSING a fixed probe across
    /// rounds (`[eval].stable_probe`). Unlike [`Self::carve`] (which SPLITS one
    /// dataset into probe + train), here the probe is fixed and a fresh dataset
    /// is filtered against it, so the same exam is reused round-to-round while
    /// the probe is still guaranteed never to be trained on. Deterministic.
    pub fn exclude_overlap(&self, dataset: &Dataset) -> Dataset {
        let probe_keys: std::collections::BTreeSet<String> =
            self.items.iter().map(content_key).collect();
        let rows = dataset
            .rows
            .iter()
            .filter(|r| !probe_keys.contains(&content_key(r)))
            .cloned()
            .collect();
        Dataset::new(rows)
    }

    /// Assert no probe item appears in `training` (by content). A probe that
    /// leaked into the training set is invalid; this is the spec's hard gate.
    pub fn assert_no_overlap(&self, training: &Dataset) -> anyhow::Result<()> {
        let train_keys: std::collections::BTreeSet<String> =
            training.rows.iter().map(content_key).collect();
        for item in &self.items {
            let k = content_key(item);
            if train_keys.contains(&k) {
                anyhow::bail!(
                    "probe overlap: a probe item also appears in the training set \
                     (probe must be held out). Offending content key: {k}"
                );
            }
        }
        Ok(())
    }
}

/// A stable content key for a row (excludes `source`/`gen` provenance so two
/// rows with identical content but different provenance are "the same item" for
/// overlap/dedup purposes). Deterministic.
fn content_key(row: &GenExample) -> String {
    match row {
        GenExample::Qa {
            prompt, completion, ..
        } => format!("qa\u{1}{prompt}\u{1}{completion}"),
        GenExample::Instruction {
            instruction,
            input,
            output,
            ..
        } => format!("instr\u{1}{instruction}\u{1}{input}\u{1}{output}"),
        GenExample::Completion { text, .. } => format!("compl\u{1}{text}"),
        GenExample::Contrastive {
            query,
            positive,
            negatives,
            ..
        } => format!(
            "contr\u{1}{query}\u{1}{positive}\u{1}{}",
            negatives.join("\u{2}")
        ),
        GenExample::Skill {
            skill_name,
            invocation,
            ..
        } => format!("skill\u{1}{skill_name}\u{1}{invocation}"),
        GenExample::ReasoningEdit {
            prompt,
            final_action,
            edited_steps,
            ..
        } => format!(
            "reason\u{1}{prompt}\u{1}{final_action}\u{1}{}",
            edited_steps.join("\u{2}")
        ),
        GenExample::ToolCall {
            prompt,
            tool,
            arguments,
            ..
        } => format!(
            "tool\u{1}{prompt}\u{1}{tool}\u{1}{}",
            serde_json::to_string(arguments).unwrap_or_default()
        ),
        GenExample::Cli {
            prompt, command, ..
        } => format!("cli\u{1}{prompt}\u{1}{command}"),
    }
}

/// A stable 64-bit hash of a row's content (FNV-1a over its content key). Used
/// for the reproducible carve split — no RNG.
fn row_hash(row: &GenExample) -> u64 {
    fnv1a(content_key(row).as_bytes())
}

/// A stable content version over the whole probe set: FNV-1a over the sorted
/// content keys, rendered hex + prefixed. Identical content ⇒ identical version.
fn content_version(items: &[GenExample]) -> String {
    let mut keys: Vec<String> = items.iter().map(content_key).collect();
    keys.sort();
    let mut h = FNV_OFFSET;
    for k in &keys {
        h = fnv1a_continue(h, k.as_bytes());
        h = fnv1a_continue(h, b"\n");
    }
    format!("probe-v{h:016x}-n{}", items.len())
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_continue(FNV_OFFSET, bytes)
}

fn fnv1a_continue(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
