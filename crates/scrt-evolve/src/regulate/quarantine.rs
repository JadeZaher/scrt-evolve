//! Quarantine (track 15) — `work_dir/quarantine.json`.
//!
//! On a catastrophe the offending cause is identified by the **`gen` provenance
//! stamp** the step's training rows carried (`trace:<goal>`, `regen:swap<N>`,
//! `refine:*`, `expert:<id>` — the mechanism already in `dataset.rs`/track 20's
//! harvester) and written here. The round driver consults this file and FILTERS
//! OUT any dataset row whose `gen` matches, so the same bad cause is never
//! re-fed (styleguide §2.4). This is the thing that stops a corrupting trend
//! from compounding across rounds.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dataset::{Dataset, GenExample};

/// The persisted quarantine list: a set of `gen` provenance stamps to skip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Quarantine {
    /// Provenance stamps (`gen` values) that are quarantined.
    #[serde(default)]
    pub gen_stamps: BTreeSet<String>,
}

impl Quarantine {
    /// Load from `path`, or an empty quarantine if the file is absent.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Write to `path` atomically.
    pub fn write(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        crate::harvest::write_atomic(path.as_ref(), json.as_bytes())?;
        Ok(())
    }

    /// Quarantine one or more provenance stamps (the cause of a catastrophe).
    pub fn add<I, S>(&mut self, stamps: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for s in stamps {
            self.gen_stamps.insert(s.into());
        }
    }

    /// Is this `gen` stamp quarantined?
    pub fn contains(&self, gen: &str) -> bool {
        self.gen_stamps.contains(gen)
    }

    /// Whether anything is quarantined.
    pub fn is_empty(&self) -> bool {
        self.gen_stamps.is_empty()
    }

    /// Filter a dataset, dropping every row whose `gen` provenance is
    /// quarantined. Returns `(kept, dropped_count)`. Rows without a `gen` stamp
    /// are always kept (only an explicitly-quarantined cause is removed).
    pub fn filter(&self, dataset: &Dataset) -> (Dataset, usize) {
        if self.is_empty() {
            return (dataset.clone(), 0);
        }
        let mut kept = Vec::with_capacity(dataset.rows.len());
        let mut dropped = 0usize;
        for row in &dataset.rows {
            match row_gen(row) {
                Some(g) if self.contains(g) => dropped += 1,
                _ => kept.push(row.clone()),
            }
        }
        (Dataset::new(kept), dropped)
    }
}

/// The `gen` provenance of a row, if the variant carries one.
fn row_gen(row: &GenExample) -> Option<&str> {
    match row {
        GenExample::Qa { gen, .. }
        | GenExample::Instruction { gen, .. }
        | GenExample::ToolCall { gen, .. }
        | GenExample::Cli { gen, .. } => gen.as_deref(),
        GenExample::Completion { .. } | GenExample::Contrastive { .. } => None,
    }
}
