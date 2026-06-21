//! `Scorer` + `ScoreReport` — score a model against a held-out probe (track 10).
//!
//! Three signal sources, each independently optional (graceful degrade):
//! - **executable correctness** — generate a completion per probe item, then run
//!   it through the [`super::gate::ExecutableGate`] (tool_call/cli) or compare to
//!   the reference (qa/instruction). No real forward pass required when using the
//!   `api` backend — the teacher endpoint generates, the gate judges.
//! - **constitution adherence** — (judge backend) sample completions, score
//!   principle violations. Stubbed seam here (track 12 owns the constitution).
//! - **depth / perplexity** — needs a real forward pass: the `transformers`
//!   backend shells out to `python -m scrt_evolve_score` (the track-19 subprocess
//!   pattern). The default Rust build stays Python-free.
//!
//! Every report stamps `probe_version` so [`super::verdict`] only compares
//! same-version reports.

use serde::{Deserialize, Serialize};

use crate::dataset::GenExample;
use crate::generate::api::{ChatMessage, ChatTransport};

use super::gate::ExecutableGate;
use super::probe::ProbeSet;

/// The comparable metrics produced by scoring a model on a probe set.
///
/// `correctness` is always present (0.0..=1.0). The rest are `Option` — absent
/// when the active backend can't compute them (e.g. `api` can't do perplexity),
/// so a consumer knows the difference between "0.0" and "not measured".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreReport {
    /// Fraction of probe items judged correct (executable gate / reference).
    pub correctness: f64,
    /// Fraction adhering to the constitution (track 12). `None` if not judged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constitution_adherence: Option<f64>,
    /// Mean early-exit depth (lower = cheaper inference, track 11). `None`
    /// without a real forward pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_exit_depth: Option<f64>,
    /// Mean perplexity over the probe. `None` without a real forward pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<f64>,
    /// Number of probe items scored.
    pub n: usize,
    /// The probe set's version (reports compare only within a version).
    pub probe_version: String,
    /// Which backend produced this report (`api` | `transformers` | `candle`).
    pub backend: String,
}

impl ScoreReport {
    /// An "uncovered" report: no probe items, correctness 0. Returned (with a
    /// log) when there is nothing to score, so the harness degrades to a no-op
    /// instead of erroring (spec §Constraints, graceful degradation).
    pub fn uncovered(probe_version: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            correctness: 0.0,
            constitution_adherence: None,
            mean_exit_depth: None,
            perplexity: None,
            n: 0,
            probe_version: probe_version.into(),
            backend: backend.into(),
        }
    }

    /// Write the report to JSON (atomically).
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        crate::harvest::write_atomic(path.as_ref(), json.as_bytes())?;
        Ok(())
    }
}

/// A model scorer: score a probe set into a [`ScoreReport`].
pub trait Scorer {
    fn score(&self, probe: &ProbeSet) -> anyhow::Result<ScoreReport>;
}

/// The **api** scorer: generate a completion per probe item via a chat endpoint,
/// then judge it with the executable gate (tool_call/cli) or a reference match
/// (qa/instruction). No ML deps — works before any Python/candle is wired.
///
/// Generic over [`ChatTransport`] so tests inject a deterministic mock model.
pub struct ApiScorer<T: ChatTransport> {
    transport: T,
    gate: ExecutableGate,
}

impl<T: ChatTransport> ApiScorer<T> {
    /// Build with a transport + a gate.
    pub fn new(transport: T, gate: ExecutableGate) -> Self {
        Self { transport, gate }
    }
}

impl<T: ChatTransport> Scorer for ApiScorer<T> {
    fn score(&self, probe: &ProbeSet) -> anyhow::Result<ScoreReport> {
        if probe.is_empty() {
            eprintln!("eval: probe set is empty — returning an uncovered report");
            return Ok(ScoreReport::uncovered(probe.version.clone(), "api"));
        }

        let mut correct = 0usize;
        let mut scored = 0usize;

        for item in &probe.items {
            let prompt = match probe_prompt(item) {
                Some(p) => p,
                None => continue, // unscorable kind (e.g. contrastive) — skip
            };
            scored += 1;

            // Ask the model to answer the probe prompt.
            let messages = vec![
                ChatMessage::system(
                    "You answer the user with ONLY the exact scrt command, tool \
                     call, or answer requested — no prose, no code fences.",
                ),
                ChatMessage::user(prompt),
            ];
            let answer = match self.transport.complete(&messages) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("eval: scoring a probe item failed (counted incorrect): {e}");
                    continue;
                }
            };

            if self.judge(item, &answer) {
                correct += 1;
            }
        }

        let correctness = if scored == 0 {
            0.0
        } else {
            correct as f64 / scored as f64
        };

        Ok(ScoreReport {
            correctness,
            constitution_adherence: None, // judge backend (track 12) not wired here
            mean_exit_depth: None,        // needs a real forward pass
            perplexity: None,
            n: scored,
            probe_version: probe.version.clone(),
            backend: "api".to_string(),
        })
    }
}

impl<T: ChatTransport> ApiScorer<T> {
    /// Judge one model answer against the probe item. Executable kinds use the
    /// gate; prose kinds use a normalized reference match (a coarse but
    /// model-free correctness signal — the judge backend refines this later).
    fn judge(&self, item: &GenExample, answer: &str) -> bool {
        match item {
            GenExample::Cli { .. } => {
                let cmd = strip_fences(answer);
                self.gate.check_cli(&cmd).is_pass()
            }
            GenExample::ToolCall { tool, .. } => {
                // Parse the model's answer as a tool call: prefer JSON
                // {"tool":..,"arguments":..}; if it's not JSON, fall back to the
                // probe's own tool name with empty args (so a bare tool name at
                // least resolves the name check).
                match parse_tool_answer(answer) {
                    Some((t, args)) => self.gate.check_tool_call(&t, &args).is_pass(),
                    None => self
                        .gate
                        .check_tool_call(tool, &serde_json::json!({}))
                        .is_pass(),
                }
            }
            GenExample::Qa { completion, .. } => reference_match(answer, completion),
            GenExample::Instruction { output, .. } => reference_match(answer, output),
            GenExample::Completion { text, .. } => reference_match(answer, text),
            GenExample::Contrastive { .. } => false,
        }
    }
}

/// The prompt to pose for a probe item (what the model must answer).
fn probe_prompt(item: &GenExample) -> Option<String> {
    match item {
        GenExample::Qa { prompt, .. } => Some(prompt.clone()),
        GenExample::Instruction {
            instruction, input, ..
        } => Some(if input.trim().is_empty() {
            instruction.clone()
        } else {
            format!("{instruction}\n\n{input}")
        }),
        GenExample::ToolCall { prompt, .. } => Some(prompt.clone()),
        GenExample::Cli { prompt, .. } => Some(prompt.clone()),
        GenExample::Completion { .. } | GenExample::Contrastive { .. } => None,
    }
}

/// Parse a model's tool-call answer into (tool, arguments). Accepts the JSON
/// object shape `{"tool":"x","arguments":{…}}`. Returns `None` if not parseable.
fn parse_tool_answer(answer: &str) -> Option<(String, serde_json::Value)> {
    let json = strip_fences(answer);
    let v: serde_json::Value = serde_json::from_str(json.trim()).ok()?;
    let tool = v.get("tool")?.as_str()?.to_string();
    let args = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
    Some((tool, args))
}

/// Coarse reference match for prose probes: normalize whitespace + case and
/// require the reference's salient content to appear (substring after
/// normalization). Model-free; the judge backend (track 12) refines this.
fn reference_match(answer: &str, reference: &str) -> bool {
    let norm = |s: &str| {
        s.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let a = norm(answer);
    let r = norm(reference);
    if r.is_empty() {
        return false;
    }
    a == r || a.contains(&r) || r.contains(&a)
}

/// Strip a leading/trailing markdown code fence from a model answer.
fn strip_fences(s: &str) -> String {
    let t = s.trim();
    let t = t.strip_prefix("```").unwrap_or(t);
    // Drop an optional language tag on the opening fence's line.
    let t = match t.find('\n') {
        Some(nl) if t[..nl].chars().all(|c| c.is_alphanumeric()) => &t[nl + 1..],
        _ => t,
    };
    let t = t.strip_suffix("```").unwrap_or(t);
    t.trim().to_string()
}
