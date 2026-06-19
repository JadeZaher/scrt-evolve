//! The **training directive** — the human's stated intent for an evolution run.
//!
//! Self-routing infers *what* to generate from corpus/palace signals, but the
//! human should still be grilled on *direction* when they trigger evolution.
//! The interview (CLI) captures their answers into a durable, editable
//! [`TrainingDirective`] (`work_dir/directive.json`), which the planner then
//! consumes alongside the signals — so the plan reflects intent, not just
//! signal inference.
//!
//! The directive is loaded once and reused across runs unless re-interviewed.

use serde::{Deserialize, Serialize};

/// What the human wants this evolution run to train toward.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainingDirective {
    /// The training objective in the human's own words.
    pub goal: String,
    /// Modalities ranked by priority (first = most important). Subset of
    /// `tool_call | cli | qa | instruction`.
    #[serde(default)]
    pub priorities: Vec<String>,
    /// Tools/workflows that MUST be well-covered regardless of corpus
    /// frequency (e.g. "scrt_stash", "stash->get_stash chain").
    #[serde(default)]
    pub must_cover: Vec<String>,
    /// Who the model serves (e.g. "power users doing memory traversal").
    #[serde(default)]
    pub audience: String,
    /// Difficulty / level (e.g. "advanced flag mastery", "beginner help").
    #[serde(default)]
    pub difficulty: String,
    /// Answer style/tone (e.g. "terse, command-first").
    #[serde(default)]
    pub tone: String,
    /// Hard exclusions / things to avoid (e.g. "no destructive commands",
    /// "no prose trivia"). Enforced as guardrails.
    #[serde(default)]
    pub exclusions: Vec<String>,
    /// Optional cap on total dataset rows (a guardrail). 0 = no cap.
    #[serde(default)]
    pub max_rows: usize,
    /// Free-form extra notes the human added (incl. answers to LLM follow-up
    /// questions that don't map onto a structured field).
    #[serde(default)]
    pub notes: String,
}

impl TrainingDirective {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }
    pub fn read(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Render the directive as a compact block for the planner/critic prompt.
    /// This is the human's intent, stated as constraints the planner must honor.
    pub fn prompt_block(&self) -> String {
        let mut s = String::new();
        if !self.goal.is_empty() {
            s.push_str(&format!("GOAL: {}\n", self.goal));
        }
        if !self.priorities.is_empty() {
            s.push_str(&format!("MODALITY PRIORITY (high→low): {}\n", self.priorities.join(" > ")));
        }
        if !self.must_cover.is_empty() {
            s.push_str(&format!(
                "MUST COVER (budget specs for these even if corpus-rare): {}\n",
                self.must_cover.join(", ")
            ));
        }
        if !self.audience.is_empty() {
            s.push_str(&format!("AUDIENCE: {}\n", self.audience));
        }
        if !self.difficulty.is_empty() {
            s.push_str(&format!("DIFFICULTY/LEVEL: {}\n", self.difficulty));
        }
        if !self.tone.is_empty() {
            s.push_str(&format!("TONE/STYLE: {}\n", self.tone));
        }
        if !self.exclusions.is_empty() {
            s.push_str(&format!("HARD EXCLUSIONS (never produce): {}\n", self.exclusions.join(", ")));
        }
        if self.max_rows > 0 {
            s.push_str(&format!("MAX TOTAL ROWS: {}\n", self.max_rows));
        }
        if !self.notes.is_empty() {
            s.push_str(&format!("NOTES: {}\n", self.notes));
        }
        if s.is_empty() {
            s.push_str("(no explicit directive — infer from signals)\n");
        }
        s
    }

    /// Does any exclusion match this text (case-insensitive substring)? Used as
    /// a generation-time guardrail to drop rows the human forbade.
    pub fn excluded(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        self.exclusions
            .iter()
            .any(|ex| !ex.trim().is_empty() && lower.contains(&ex.to_lowercase()))
    }
}
