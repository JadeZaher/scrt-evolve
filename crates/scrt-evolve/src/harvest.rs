//! Transcript harvester — the **raw, lower-trust** trace signal (track 20).
//!
//! The goal-tagged palace stashes are the high-signal curriculum (slices 1/3).
//! But a frontier model doing real work locally also leaves **transcripts**
//! (conversation + tool-call logs). This module turns those into training rows,
//! following the scrt **capture-then-filter** rule: never ingest a raw multi-KB
//! transcript blindly — capture it to disk, filter it down to the goal-relevant
//! parts, then distill only those into [`GenExample`] rows.
//!
//! Trust posture (spec §3): transcripts are firehose, lower-trust. Rows are
//! - **filtered** to the goal topic (off-goal turns are dropped),
//! - **deduped** (the same exchange surfaced twice collapses),
//! - **provenance-stamped** `gen = "trace:<goal>"` so a bad trace round is
//!   quarantinable (track 15) and never silently mixed with stash-derived rows.
//!
//! Everything here is **pure + deterministic**: the harvest takes the parsed
//! entries + the goal and produces the same rows every time (no wall-clock, no
//! RNG — styleguide §2.1/§2.2). The capture filename's date is supplied by the
//! caller so the distill logic stays clock-free.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::GoalConfig;
use crate::dataset::{Dataset, GenExample, Outcome, Tier, Verdict};

/// One entry in a captured frontier transcript (the capture artifact schema).
///
/// This is the minimal cross-tool shape: a `role` (`user` | `assistant` |
/// `system` | `tool`), the message `text`, and optional tool-call fields. It is
/// deliberately permissive (extra fields ignored) so transcripts exported by
/// different frontends round-trip without a bespoke adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptEntry {
    /// `user` | `assistant` | `system` | `tool`.
    pub role: String,
    /// The message body (prose, or a tool result).
    #[serde(default)]
    pub text: String,
    /// For an assistant tool call: the tool name (e.g. `scrt_stash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// For an assistant tool call: the runnable command line, if the transcript
    /// recorded one (CLI-shaped traces distill into [`GenExample::Cli`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

impl TranscriptEntry {
    /// Parse a transcript from a JSONL string (one entry per line). Blank lines
    /// are skipped; a malformed line errors with its 1-based line number.
    pub fn parse_jsonl(text: &str) -> anyhow::Result<Vec<Self>> {
        let mut out = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: TranscriptEntry = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("transcript line {}: {e}", i + 1))?;
            out.push(entry);
        }
        Ok(out)
    }
}

/// The result of harvesting one transcript for one goal: the captured raw file
/// path + the distilled, filtered, provenance-stamped dataset.
#[derive(Debug, Clone)]
pub struct HarvestResult {
    /// Where the raw transcript was captured (`<slug>-<date>.jsonl`).
    pub captured_path: PathBuf,
    /// How many raw entries were captured before filtering.
    pub raw_entries: usize,
    /// How many entries survived the goal-relevance filter.
    pub kept_entries: usize,
    /// The distilled trace rows (each stamped `gen = "trace:<goal>"`).
    pub dataset: Dataset,
}

/// The provenance stamp for a goal's trace rows: `trace:<goal>` (spec §3).
/// One stable place so the round driver (slice 6, lane-gated) can quarantine a
/// bad trace round by this exact `gen` value.
pub fn trace_gen_stamp(goal: &str) -> String {
    format!("trace:{goal}")
}

/// Capture a raw transcript to `traces_dir/<slug>-<date>.jsonl` (atomically),
/// then filter + distill it into goal-stamped rows.
///
/// `traces_dir` is the per-goal directory (see [`crate::WorkDir::goal_traces_dir`]).
/// `slug` identifies the source/session; `date` (e.g. `2026-06-20`) disambiguates
/// re-captures — both supplied by the caller so this stays clock-free and
/// deterministic. `raw_jsonl` is the transcript text exactly as exported.
pub fn capture_and_harvest(
    goal: &GoalConfig,
    traces_dir: &Path,
    slug: &str,
    date: &str,
    raw_jsonl: &str,
) -> anyhow::Result<HarvestResult> {
    let entries = TranscriptEntry::parse_jsonl(raw_jsonl)?;

    // --- Capture: write the raw transcript atomically (temp + rename) so a
    // crash mid-write never leaves a half-written trace (styleguide §2.3). ---
    std::fs::create_dir_all(traces_dir)?;
    let captured_path = traces_dir.join(format!("{slug}-{date}.jsonl"));
    write_atomic(&captured_path, raw_jsonl.as_bytes())?;

    // --- Filter + distill (pure). ---
    let result = harvest_entries(goal, &entries);

    Ok(HarvestResult {
        captured_path,
        raw_entries: entries.len(),
        kept_entries: result.kept,
        dataset: result.dataset,
    })
}

/// Filter + distill an already-parsed transcript. Pure: no I/O, deterministic.
/// Public so tests (and a future in-memory caller) can harvest without writing
/// a capture file.
pub fn harvest_entries(goal: &GoalConfig, entries: &[TranscriptEntry]) -> HarvestedRows {
    let terms = topic_terms(&goal.topic);
    let stamp = trace_gen_stamp(&goal.name);

    let mut rows: Vec<GenExample> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept = 0usize;

    // Distill user→assistant exchanges. We pair each assistant entry with the
    // most recent user entry (the "prompt"), and only emit when the exchange is
    // goal-relevant (matches a topic term) — capture-then-filter.
    let mut pending_user: Option<&str> = None;

    for entry in entries {
        match entry.role.as_str() {
            "user" => {
                pending_user = Some(entry.text.as_str());
            }
            "assistant" => {
                let prompt = pending_user.unwrap_or("");
                // Goal relevance: the prompt OR the assistant body must mention
                // a topic term. Off-goal turns are dropped (spec §3).
                let haystack = format!(
                    "{} {} {} {}",
                    prompt,
                    entry.text,
                    entry.tool.as_deref().unwrap_or(""),
                    entry.command.as_deref().unwrap_or("")
                );
                if !is_relevant(&haystack, &terms) {
                    continue;
                }

                if let Some(row) = distill_exchange(prompt, entry, &stamp) {
                    let key = dedup_key(&row);
                    if seen.insert(key) {
                        rows.push(row);
                        kept += 1;
                    }
                }
            }
            // Foreign extensible role strings (e.g. "system", "tool") are context,
            // not standalone training rows.
            _ => {}
        }
    }

    HarvestedRows {
        dataset: Dataset::new(rows),
        kept,
    }
}

/// The pure harvest output: distilled rows + how many survived the filter.
#[derive(Debug, Clone)]
pub struct HarvestedRows {
    /// The distilled, goal-stamped training rows.
    pub dataset: Dataset,
    /// Number of entries that survived the goal-relevance filter.
    pub kept: usize,
}

/// Distill one (prompt, assistant-entry) exchange into a trace row, stamped.
/// A recorded command line → [`GenExample::Cli`]; otherwise a prose
/// [`GenExample::Qa`]. Returns `None` for empty/degenerate exchanges.
fn distill_exchange(prompt: &str, entry: &TranscriptEntry, stamp: &str) -> Option<GenExample> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return None;
    }
    let source = Some("transcript".to_string());

    if let Some(command) = entry.command.as_deref() {
        let command = command.trim();
        if !command.is_empty() {
            return Some(GenExample::Cli {
                prompt: prompt.to_string(),
                command: command.to_string(),
                source,
                gen: Some(stamp.to_string()),
                outcome: Outcome::Unknown,
                judge_score: None,
                judge_verdict: Verdict::Unjudged,
                tier: Tier::Private,
                chosen_over: None,
            });
        }
    }

    let completion = entry.text.trim();
    if completion.is_empty() {
        return None;
    }
    Some(GenExample::Qa {
        prompt: prompt.to_string(),
        completion: completion.to_string(),
        source,
        gen: Some(stamp.to_string()),
        outcome: Outcome::Unknown,
        judge_score: None,
        judge_verdict: Verdict::Unjudged,
        tier: Tier::Private,
        chosen_over: None,
    })
}

/// A stable dedup key over a row's content (so the same exchange seen twice in a
/// transcript collapses to one row). Deterministic.
fn dedup_key(row: &GenExample) -> String {
    match row {
        GenExample::Qa {
            prompt, completion, ..
        } => format!("qa\u{1}{}\u{1}{}", prompt.trim(), completion.trim()),
        GenExample::Cli {
            prompt, command, ..
        } => format!("cli\u{1}{}\u{1}{}", prompt.trim(), command.trim()),
        // These variants are not produced by distill_exchange, but are keyed
        // explicitly so a new variant addition is a compile break, not a silent
        // dedup collision.
        GenExample::Instruction { instruction, output, .. } => {
            format!("instruction\u{1}{}\u{1}{}", instruction.trim(), output.trim())
        }
        GenExample::Completion { text, .. } => format!("compl\u{1}{}", text.trim()),
        GenExample::Contrastive { query, positive, .. } => {
            format!("contrastive\u{1}{}\u{1}{}", query.trim(), positive.trim())
        }
        GenExample::ToolCall { tool, arguments, .. } => format!(
            "tool\u{1}{tool}\u{1}{}",
            serde_json::to_string(arguments).unwrap_or_default()
        ),
        GenExample::Skill { skill_name, invocation, .. } => {
            format!("skill\u{1}{skill_name}\u{1}{}", invocation.trim())
        }
        GenExample::ReasoningEdit { prompt, final_action, .. } => format!(
            "reasoning\u{1}{}\u{1}{}",
            prompt.trim(),
            final_action.trim()
        ),
    }
}

/// Split a goal `topic` into lowercased salient terms for relevance matching.
/// Short/stop-ish fragments (< 3 chars) are dropped; the result is deduped +
/// sorted so matching is deterministic. A topic that yields no terms (e.g. all
/// punctuation) matches everything — degrade gracefully rather than drop all.
fn topic_terms(topic: &str) -> Vec<String> {
    let mut terms: Vec<String> = topic
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_lowercase())
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

/// Is `haystack` relevant to the goal? True if it contains any topic term
/// (case-insensitive). Empty `terms` ⇒ everything is relevant (graceful
/// degrade for punctuation-only topics).
fn is_relevant(haystack: &str, terms: &[String]) -> bool {
    if terms.is_empty() {
        return true;
    }
    let hay = haystack.to_lowercase();
    terms.iter().any(|t| hay.contains(t))
}

/// Write `bytes` to `path` atomically: write to a sibling temp file, then
/// rename over the destination (styleguide §2.3 — no half-written artifacts).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.tmp",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "trace".to_string())
    ));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
