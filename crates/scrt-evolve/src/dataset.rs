//! Dataset format — the generate↔train boundary (JSONL).
//!
//! One JSONL file is the durable contract between stages: one JSON object per
//! line, `kind` tagging which presets can consume the row. The schema is the
//! **cross-language contract** (Rust writer ↔ Python reader under `--features
//! pyo3`); changing a field is a breaking change.

use std::io::{BufRead, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// One dataset row. `kind` is the tag; the variant carries its fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum GenExample {
    Qa {
        prompt: String,
        completion: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
    },
    Instruction {
        instruction: String,
        #[serde(default)]
        input: String,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
    },
    Completion {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
    Contrastive {
        query: String,
        positive: String,
        #[serde(default)]
        negatives: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stash: Option<String>,
    },
    /// A tool-calling example: a user intent → a structured call to one of
    /// scrt's tools (name + JSON arguments matching the real tool schema).
    /// Trains the model to emit function calls, not prose.
    #[serde(rename = "tool_call")]
    ToolCall {
        /// The natural-language user request.
        prompt: String,
        /// The tool name (e.g. `scrt_stash`), from scrt-core's tool spec.
        tool: String,
        /// The call arguments as a JSON object — keys must be valid params for
        /// `tool` per the schema.
        arguments: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
    },
    /// A CLI-invocation example: a user intent → the exact runnable `scrt …`
    /// command line. Trains CLI fluency.
    Cli {
        /// The natural-language user request.
        prompt: String,
        /// The runnable command line, e.g. `scrt "auth" --mp-stash auth --mp-ttl 4h`.
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
    },
}

/// An in-memory handle over the on-disk JSONL dataset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dataset {
    pub rows: Vec<GenExample>,
}

impl Dataset {
    pub fn new(rows: Vec<GenExample>) -> Self {
        Self { rows }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Serialize the dataset to a JSONL string — one object per line.
    pub fn to_jsonl(&self) -> serde_json::Result<String> {
        let mut out = String::new();
        for row in &self.rows {
            out.push_str(&serde_json::to_string(row)?);
            out.push('\n');
        }
        Ok(out)
    }

    /// Parse a dataset from a JSONL string. Blank lines are skipped; a malformed
    /// line errors with its 1-based line number.
    pub fn from_jsonl(text: &str) -> anyhow::Result<Self> {
        let mut rows = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: GenExample = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("dataset line {}: {e}", i + 1))?;
            rows.push(row);
        }
        Ok(Self { rows })
    }

    /// Write the dataset to `path` as JSONL (creating parent dirs).
    pub fn write_jsonl(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for row in &self.rows {
            serde_json::to_writer(&mut f, row)?;
            f.write_all(b"\n")?;
        }
        f.flush()?;
        Ok(())
    }

    /// Read a dataset from a JSONL file (streaming, line by line).
    pub fn read_jsonl(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let f = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(f);
        let mut rows = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: GenExample = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("dataset line {}: {e}", i + 1))?;
            rows.push(row);
        }
        Ok(Self { rows })
    }
}
