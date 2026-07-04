//! Dataset format — the generate↔train boundary (JSONL).
//!
//! One JSONL file is the durable contract between stages: one JSON object per
//! line, `kind` tagging which presets can consume the row. The schema is the
//! **cross-language contract** (Rust writer ↔ Python reader under `--features
//! pyo3`); changing a field is a breaking change.
//!
//! ## Contract v1.1 (track 37) — additive, non-breaking
//! Every variant carries optional training-signal metadata (all
//! `#[serde(default, skip_serializing_if)]`, so absent = a byte-identical v1.0
//! line): `outcome` (execution ground truth, Phase A), `judge_score` +
//! `judge_verdict` (per-pair LLM judge, Phase B), `tier` (private|shared
//! sovereignty; most-restrictive-wins downstream), and `chosen_over` (the
//! content-key of the rejected half of a preference pair — the DPO contract,
//! *recorded not trained*). Per-variant fields (not `#[serde(flatten)]`) because
//! flatten misbehaves under an internally-tagged enum (`tag = "kind"`); see
//! `crates/scrt-evolve/src/AGENTS.md` §dataset contract v1.1.

use std::io::{BufRead, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Execution ground truth for a mined row (contract v1.1, Phase A). `Unknown`
/// is the safe default — err toward it when a `tool_result` is absent/ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Success,
    Failure,
    #[default]
    Unknown,
}

/// Per-pair judge verdict (contract v1.1, Phase B). `Unjudged` = the row never
/// passed a judge (fail-open kept it, or judging was disabled).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Keep,
    Drop,
    #[default]
    Unjudged,
}

/// Data-sovereignty tier (contract v1.1). `Private` is the safe default;
/// `most-restrictive-wins` when rolling up to a branch manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    #[default]
    Private,
    Shared,
}

impl Tier {
    /// Most-restrictive-wins fold (`Private` beats `Shared`).
    pub fn most_restrictive(self, other: Tier) -> Tier {
        match (self, other) {
            (Tier::Private, _) | (_, Tier::Private) => Tier::Private,
            (Tier::Shared, Tier::Shared) => Tier::Shared,
        }
    }

    /// Parse a config string (`"shared"` ⇒ `Shared`, anything else ⇒ the safe
    /// `Private` default). `None` ⇒ `Private`.
    pub fn from_config(s: Option<&str>) -> Tier {
        match s.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("shared") => Tier::Shared,
            // Any other string value or None → safe Private default.
            Some(_) | None => Tier::Private,
        }
    }
}

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
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
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
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
    },
    Completion {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
    },
    Contrastive {
        query: String,
        positive: String,
        #[serde(default)]
        negatives: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stash: Option<String>,
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
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
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
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
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
    },
    /// A skill-ingestion example (track 09, opt-in): teach the model to absorb a
    /// skill/capability (a `SKILL.md`-style description) and turn it into callable
    /// behavior — *when* to invoke it, *with what* inputs, *what* it produces. The
    /// completion trains the model to recognize the trigger and emit the
    /// invocation. See `crates/scrt-evolve/src/generate/AGENTS.md` §modalities.
    #[serde(rename = "skill")]
    Skill {
        /// The skill/capability name (must reference a real skill/tool).
        skill_name: String,
        /// The natural-language request that should trigger the skill.
        prompt: String,
        /// The invocation that uses the skill — a structured call or a runnable
        /// command line (validated like `cli`/`tool_call` where applicable).
        invocation: String,
        /// What success looks like (the outcome the invocation produces).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected_outcome: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
    },
    /// A reasoning-edit example (track 09, opt-in): teach the model to *evolve a
    /// reasoning trace* — given a task and a flawed chain-of-thought, produce a
    /// corrected chain that leads to a better `final_action`. Rendered so the
    /// completion carries the corrected reasoning BEFORE the action, training the
    /// model to reason internally at inference (not just emit an answer). See
    /// `crates/scrt-evolve/src/generate/AGENTS.md` §modalities.
    #[serde(rename = "reasoning_edit")]
    ReasoningEdit {
        /// The task / question the reasoning is about.
        prompt: String,
        /// The original (flawed) reasoning steps.
        #[serde(default)]
        original_steps: Vec<String>,
        /// The edit operation applied: `insert | correct | prune | reorder`.
        edit_op: String,
        /// The corrected reasoning steps (the target chain).
        #[serde(default)]
        edited_steps: Vec<String>,
        /// The final action/answer the corrected reasoning leads to.
        final_action: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gen: Option<String>,
        #[serde(default, skip_serializing_if = "Outcome::is_unknown")]
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_score: Option<f32>,
        #[serde(default, skip_serializing_if = "Verdict::is_unjudged")]
        judge_verdict: Verdict,
        #[serde(default, skip_serializing_if = "Tier::is_default")]
        tier: Tier,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_over: Option<String>,
    },
}

impl Outcome {
    fn is_unknown(&self) -> bool {
        matches!(self, Outcome::Unknown)
    }
}

impl Verdict {
    fn is_unjudged(&self) -> bool {
        matches!(self, Verdict::Unjudged)
    }
}

impl Tier {
    fn is_default(&self) -> bool {
        matches!(self, Tier::Private)
    }
}

/// Uniform accessors for the contract-v1.1 metadata across all variants — so
/// Phase-A ingest and Phase-B judging stamp rows without matching every variant.
impl GenExample {
    /// Mutable references to the five metadata fields of any variant.
    #[allow(clippy::type_complexity)]
    fn meta_mut(
        &mut self,
    ) -> (
        &mut Outcome,
        &mut Option<f32>,
        &mut Verdict,
        &mut Tier,
        &mut Option<String>,
    ) {
        match self {
            GenExample::Qa {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Instruction {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Completion {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Contrastive {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::ToolCall {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Cli {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Skill {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::ReasoningEdit {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            } => (outcome, judge_score, judge_verdict, tier, chosen_over),
        }
    }

    /// Shared references to the five metadata fields of any variant.
    #[allow(clippy::type_complexity)]
    fn meta(&self) -> (Outcome, Option<f32>, Verdict, Tier, Option<&String>) {
        match self {
            GenExample::Qa {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Instruction {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Completion {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Contrastive {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::ToolCall {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Cli {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::Skill {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            }
            | GenExample::ReasoningEdit {
                outcome,
                judge_score,
                judge_verdict,
                tier,
                chosen_over,
                ..
            } => (
                *outcome,
                *judge_score,
                *judge_verdict,
                *tier,
                chosen_over.as_ref(),
            ),
        }
    }

    /// Return the execution outcome stamped on this row.
    pub fn outcome(&self) -> Outcome {
        self.meta().0
    }

    /// Stamp an execution outcome onto this row.
    pub fn set_outcome(&mut self, o: Outcome) {
        *self.meta_mut().0 = o;
    }

    /// Return the LLM judge score (0–1) if the row has been judged.
    pub fn judge_score(&self) -> Option<f32> {
        self.meta().1
    }

    /// Return the LLM judge verdict for this row.
    pub fn judge_verdict(&self) -> Verdict {
        self.meta().2
    }

    /// Stamp the judge result (score 0–1 + verdict) onto the row.
    pub fn set_judge(&mut self, score: f32, verdict: Verdict) {
        let (_, s, v, _, _) = self.meta_mut();
        *s = Some(score);
        *v = verdict;
    }

    /// Return the data-sovereignty tier of this row.
    pub fn tier(&self) -> Tier {
        self.meta().3
    }

    /// Stamp a data-sovereignty tier onto this row.
    pub fn set_tier(&mut self, t: Tier) {
        *self.meta_mut().3 = t;
    }

    /// Return the content-key of the rejected half of a preference pair, if set.
    pub fn chosen_over(&self) -> Option<String> {
        self.meta().4.cloned()
    }

    /// Record the rejected half of a preference pair (the DPO contract; not
    /// trained here — track 37 non-goal).
    pub fn set_chosen_over(&mut self, key: String) {
        *self.meta_mut().4 = Some(key);
    }

    /// Byte length of the row's training-bearing payload — the single source of
    /// truth for length caps (track 37). Every variant is covered (no lossy
    /// catch-all): a new variant that forgets its payload is a compile break.
    pub fn payload_len(&self) -> usize {
        match self {
            GenExample::Qa { completion, .. } => completion.len(),
            GenExample::Instruction { output, .. } => output.len(),
            GenExample::Completion { text, .. } => text.len(),
            GenExample::Contrastive {
                positive, negatives, ..
            } => positive.len() + negatives.iter().map(String::len).sum::<usize>(),
            GenExample::ToolCall { arguments, .. } => {
                serde_json::to_string(arguments).map(|s| s.len()).unwrap_or(0)
            }
            GenExample::Cli { command, .. } => command.len(),
            GenExample::Skill { invocation, .. } => invocation.len(),
            GenExample::ReasoningEdit { edited_steps, .. } => {
                edited_steps.iter().map(String::len).sum()
            }
        }
    }
}

/// An in-memory handle over the on-disk JSONL dataset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dataset {
    pub rows: Vec<GenExample>,
}

impl Dataset {
    /// Construct a dataset from an already-parsed row vector.
    pub fn new(rows: Vec<GenExample>) -> Self {
        Self { rows }
    }

    /// Number of rows in the dataset.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when the dataset contains no rows.
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
