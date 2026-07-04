//! Interaction-log → training rows, generically (parsing) + LLM relevance
//! judging. Design rationale: see `src/AGENTS.md` (§`ingest.rs`).

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

use serde_json::Value;

use crate::dataset::{GenExample, Outcome, Tier, Verdict};
use crate::generate::api::{ChatMessage, ChatTransport};

/// Provenance stamp prefix for ingested rows (the quarantine key root). Kept for
/// back-compat; new rows use the per-source stamps below so a catastrophe in one
/// source doesn't quarantine all ingested data (track 31 Q2).
pub const INGEST_GEN_STAMP: &str = "ingest";

/// Per-source provenance stamp for rows mined from interaction transcripts.
pub const INGEST_GEN_TRANSCRIPT: &str = "ingest:transcript";
/// Per-source provenance stamp for rows chunked from docs.
pub const INGEST_GEN_DOC: &str = "ingest:doc";

/// Cap on a fallback intent prompt taken from a (possibly huge) user message.
const MAX_FALLBACK_PROMPT: usize = 400;
/// Drop an `unknown`-outcome command / tool-args / answer longer than this
/// (heredocs, pasted files). Outcome-verified successes get a larger cap
/// ([`MAX_ROW_CHARS_VERIFIED`]) — long successful interactions are the
/// info-dense ones the old flat cap discarded (track 37 finding 3).
const MAX_ROW_CHARS: usize = 2000;
/// Larger cap for `outcome = success` rows — a verified success is worth keeping
/// even when long.
const MAX_ROW_CHARS_VERIFIED: usize = 8000;

/// Parse one Claude Code interaction log (native JSONL) into mixed candidate rows
/// (`Bash`→`Cli`, other tool→`ToolCall`, prose turn→`Qa`), stamping each mined
/// tool row with its execution [`Outcome`] by correlating `tool_use.id` → the
/// following `tool_result` block. Pure; no relevance filtering (that's
/// [`RelevanceJudge`]) and no retry-collapse (that's [`ingest_outcomes`]). All
/// rows default to `tier = Private`; see `src/AGENTS.md` §`ingest.rs`.
pub fn interaction_log_rows(jsonl: &str) -> Vec<GenExample> {
    let mut rows = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut last_user: Option<String> = None;
    // tool_use.id → index into `rows`, for outcome stamping when the next user
    // turn carries the matching tool_result.
    let mut pending: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for line in jsonl.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("user") => {
                // A user turn may carry tool_result blocks (the outcome of the
                // prior assistant tool_use) AND/OR human text.
                stamp_tool_results(&v, &pending, &mut rows);
                if let Some(text) = user_text(&v) {
                    last_user = Some(text);
                }
            }
            Some("assistant") => {
                let Some(blocks) = v
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array)
                else {
                    continue;
                };

                let mut had_tool = false;
                let mut prose = String::new();
                for blk in blocks {
                    match blk.get("type").and_then(Value::as_str) {
                        Some("tool_use") => {
                            had_tool = true;
                            if let Some(row) = tool_use_row(blk, last_user.as_deref()) {
                                let id = blk.get("id").and_then(Value::as_str).map(str::to_string);
                                if let Some(idx) = push_unique(&mut rows, &mut seen, row) {
                                    if let Some(id) = id {
                                        pending.insert(id, idx);
                                    }
                                }
                            }
                        }
                        Some("text") => {
                            if let Some(t) = blk.get("text").and_then(Value::as_str) {
                                if !prose.is_empty() {
                                    prose.push('\n');
                                }
                                prose.push_str(t);
                            }
                        }
                        // Foreign extensible string values (e.g. "thinking", future block types).
                        _ => {}
                    }
                }

                // Pure-prose turn → Q→A; tool turns are captured as tool rows above.
                if !had_tool {
                    let prose = prose.trim();
                    if let Some(user) = last_user.as_deref() {
                        if !prose.is_empty() && prose.len() <= MAX_ROW_CHARS {
                            push_unique(
                                &mut rows,
                                &mut seen,
                                GenExample::Qa {
                                    prompt: user.to_string(),
                                    completion: prose.to_string(),
                                    source: Some("transcript".to_string()),
                                    gen: Some(INGEST_GEN_TRANSCRIPT.to_string()),
                                    outcome: Outcome::Unknown,
                                    judge_score: None,
                                    judge_verdict: Verdict::Unjudged,
                                    tier: Tier::Private,
                                    chosen_over: None,
                                },
                            );
                        }
                    }
                }
            }
            // Foreign extensible log entry types (e.g. "queue-operation", "system").
            _ => {}
        }
    }
    rows
}

/// Stamp `outcome` on the mined rows whose `tool_use.id` matches a `tool_result`
/// block in this user turn. Reads `is_error` (the primary signal) plus a text
/// heuristic for Bash results that report failure without the flag.
fn stamp_tool_results(
    user_entry: &Value,
    pending: &std::collections::HashMap<String, usize>,
    rows: &mut [GenExample],
) {
    let Some(content) = user_entry
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return;
    };
    for blk in content {
        if blk.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }
        let Some(id) = blk.get("tool_use_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(&idx) = pending.get(id) else {
            continue;
        };
        let outcome = tool_result_outcome(blk);
        if let Some(row) = rows.get_mut(idx) {
            row.set_outcome(outcome);
        }
    }
}

/// Derive an [`Outcome`] from one `tool_result` block. `is_error:true` ⇒
/// `Failure`; otherwise a conservative text scan for failure markers (non-zero
/// exit, `error:`/`fatal:`), else `Success`. Errs toward `Unknown` only when the
/// block carries no usable signal.
fn tool_result_outcome(blk: &Value) -> Outcome {
    if let Some(is_err) = blk.get("is_error").and_then(Value::as_bool) {
        return if is_err {
            Outcome::Failure
        } else {
            Outcome::Success
        };
    }
    let text = tool_result_text(blk);
    if text.is_empty() {
        return Outcome::Unknown;
    }
    let low = text.to_ascii_lowercase();
    // Bash results without an explicit flag: look for common failure markers.
    const FAIL_MARKERS: &[&str] = &[
        "command not found",
        "no such file",
        "fatal:",
        "error:",
        "traceback (most recent call last)",
        "permission denied",
        "exit code 1",
        "exit status 1",
    ];
    if FAIL_MARKERS.iter().any(|m| low.contains(m)) {
        Outcome::Failure
    } else {
        Outcome::Success
    }
}

/// Extract the text payload from a `tool_result` block (string or content-array).
fn tool_result_text(blk: &Value) -> String {
    match blk.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|it| it.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        // Foreign serde_json::Value variants (Null, Bool, Number, Object) or None
        // carry no extractable text.
        _ => String::new(),
    }
}

/// The result of applying outcome-based filtering to a parsed log: `kept` rows
/// are training-worthy; `rejected` rows (bare failures) are excluded from
/// training but preserved for the audit sidecar (`rejected.jsonl`).
#[derive(Debug, Default)]
pub struct OutcomeFilter {
    pub kept: Vec<GenExample>,
    pub rejected: Vec<GenExample>,
}

/// Apply Phase-A outcome filtering to stamped rows (from [`interaction_log_rows`]):
/// 1. **Retry-collapse** — a run of ~same-command `Failure`s immediately followed
///    by a `Success` collapses to the ONE success row, which records the last
///    failed variant's content key in `chosen_over` (the DPO preference pair,
///    recorded not trained). The collapsed failures move to `rejected`.
/// 2. **Bare-failure exclusion** — any remaining `Failure` row is excluded from
///    training (moved to `rejected` for audit).
/// 3. **Length enforcement** — a non-`Success` row over [`MAX_ROW_CHARS`] is
///    dropped (rejected); `Success` rows survive up to [`MAX_ROW_CHARS_VERIFIED`].
/// 4. **Tier stamp** — every kept row inherits `tier`.
pub fn filter_outcomes(rows: Vec<GenExample>, tier: Tier) -> OutcomeFilter {
    let mut out = OutcomeFilter::default();
    let mut i = 0;
    while i < rows.len() {
        let row = &rows[i];
        // A failure that is followed (later in the log) by a ~same-command
        // success: fold the whole failed run into that success.
        if row.outcome() == Outcome::Failure {
            if let Some(succ_idx) = find_following_success(&rows, i) {
                // Collect the failed run [i, succ_idx).
                let mut last_failed_key: Option<String> = None;
                for f in &rows[i..succ_idx] {
                    if f.outcome() == Outcome::Failure && similar_command(f, &rows[succ_idx]) {
                        last_failed_key = Some(content_key(f));
                    }
                    out.rejected.push(f.clone());
                }
                let mut success = rows[succ_idx].clone();
                if let Some(k) = last_failed_key {
                    success.set_chosen_over(k);
                }
                emit_kept(&mut out, success, tier);
                i = succ_idx + 1;
                continue;
            }
            // Bare failure with no recovering success: exclude from training.
            out.rejected.push(row.clone());
            i += 1;
            continue;
        }
        emit_kept(&mut out, row.clone(), tier);
        i += 1;
    }
    out
}

/// Push a kept row after enforcing the outcome-aware length cap and stamping tier.
fn emit_kept(out: &mut OutcomeFilter, mut row: GenExample, tier: Tier) {
    let cap = if row.outcome() == Outcome::Success {
        MAX_ROW_CHARS_VERIFIED
    } else {
        MAX_ROW_CHARS
    };
    if row.payload_len() > cap {
        out.rejected.push(row);
        return;
    }
    row.set_tier(tier);
    out.kept.push(row);
}

/// The next `Success` row at/after `from+1` whose command is ~similar to the
/// failure at `from` — the recovering attempt. `None` if none before a break.
fn find_following_success(rows: &[GenExample], from: usize) -> Option<usize> {
    let base = &rows[from];
    for (offset, r) in rows.iter().enumerate().skip(from + 1) {
        if !similar_command(base, r) {
            // A different command breaks the retry chain.
            continue;
        }
        match r.outcome() {
            Outcome::Success => return Some(offset),
            Outcome::Failure => {} // another attempt at the same command
            Outcome::Unknown => {}
        }
    }
    None
}

/// Normalized-prefix similarity for retry detection — NOT equality. Two `Cli`
/// rows are "the same command" if their normalized command prefixes match; two
/// `ToolCall` rows are similar if they invoke the same tool. All other variant
/// combinations are not considered similar (errs toward NOT-similar).
fn similar_command(a: &GenExample, b: &GenExample) -> bool {
    match (a, b) {
        (GenExample::Cli { command: ca, .. }, GenExample::Cli { command: cb, .. }) => {
            normalized_prefix(ca) == normalized_prefix(cb)
        }
        (GenExample::ToolCall { tool: ta, .. }, GenExample::ToolCall { tool: tb, .. }) => ta == tb,
        // All other variant combinations are structurally distinct — not a retry.
        (GenExample::Qa { .. }, _)
        | (GenExample::Instruction { .. }, _)
        | (GenExample::Completion { .. }, _)
        | (GenExample::Contrastive { .. }, _)
        | (GenExample::Skill { .. }, _)
        | (GenExample::ReasoningEdit { .. }, _)
        | (GenExample::Cli { .. }, _)
        | (GenExample::ToolCall { .. }, _) => false,
    }
}

/// The normalized command prefix used for retry similarity: the first two
/// whitespace-separated tokens (program + subcommand/first-arg), lowercased.
/// Captures "same command, tweaked args" without collapsing unrelated commands.
fn normalized_prefix(command: &str) -> String {
    command
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

/// File name (under `work_dir/queue/`) of the excluded-rows audit sidecar.
pub const REJECTED_SIDECAR: &str = "rejected.jsonl";

/// Append rejected (excluded-from-training) rows to the audit sidecar under
/// `work_dir/queue/`. Provenance-preserving: the rows keep their `outcome`/
/// `chosen_over` stamps so the sidecar is a full audit trail. No-op on empty.
pub fn append_rejected(work_dir: &Path, rejected: &[GenExample]) -> anyhow::Result<()> {
    if rejected.is_empty() {
        return Ok(());
    }
    let dir = work_dir.join("queue");
    std::fs::create_dir_all(&dir)
        .map_err(|e| anyhow::anyhow!("ingest sidecar: creating {}: {e}", dir.display()))?;
    let path = dir.join(REJECTED_SIDECAR);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| anyhow::anyhow!("ingest sidecar: opening {}: {e}", path.display()))?;
    for row in rejected {
        let line = serde_json::to_string(row)?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

/// One `tool_use` block → a row (`Bash`→`Cli`, else `ToolCall`). `None` if empty
/// or over-long.
fn tool_use_row(blk: &Value, last_user: Option<&str>) -> Option<GenExample> {
    let name = blk.get("name").and_then(Value::as_str).unwrap_or("");
    if name.is_empty() {
        return None;
    }
    let input = blk.get("input").cloned().unwrap_or(Value::Null);
    let desc = input.get("description").and_then(Value::as_str);
    let prompt = pick_intent(desc, last_user);

    if name.eq_ignore_ascii_case("Bash") {
        let command = input.get("command").and_then(Value::as_str)?.trim();
        // Keep up to the verified cap here; the post-pass ([`retry_collapse`])
        // drops any row still over MAX_ROW_CHARS once its outcome is known.
        if command.is_empty() || command.len() > MAX_ROW_CHARS_VERIFIED {
            return None;
        }
        return Some(GenExample::Cli {
            prompt,
            command: command.to_string(),
            source: Some("transcript".to_string()),
            gen: Some(INGEST_GEN_TRANSCRIPT.to_string()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        });
    }

    // Keep the real tool arguments, drop the harness-only `description`.
    let mut args = input;
    if let Some(obj) = args.as_object_mut() {
        obj.remove("description");
    }
    if serde_json::to_string(&args).map(|s| s.len()).unwrap_or(0) > MAX_ROW_CHARS_VERIFIED {
        return None;
    }
    Some(GenExample::ToolCall {
        prompt,
        tool: name.to_string(),
        arguments: args,
        source: Some("transcript".to_string()),
        gen: Some(INGEST_GEN_TRANSCRIPT.to_string()),
        outcome: Outcome::Unknown,
        judge_score: None,
        judge_verdict: Verdict::Unjudged,
        tier: Tier::Private,
        chosen_over: None,
    })
}

/// Chunk a doc (markdown/plain text) into [`GenExample::Completion`] rows —
/// paragraph-ish blocks split on blank lines, short/degenerate blocks dropped,
/// capped at `max_rows`. For absorbing project/domain text. Generic + pure.
pub fn doc_completion_rows(text: &str, source: &str, max_rows: usize) -> Vec<GenExample> {
    let mut rows = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for block in text.split("\n\n") {
        if rows.len() >= max_rows {
            break;
        }
        let block = block.trim();
        if block.len() < 80 || block.len() > MAX_ROW_CHARS {
            continue;
        }
        if seen.insert(block.to_string()) {
            rows.push(GenExample::Completion {
                text: block.to_string(),
                source: Some(source.to_string()),
                gen: None,
                outcome: Outcome::Unknown,
                judge_score: None,
                judge_verdict: Verdict::Unjudged,
                tier: Tier::Private,
                chosen_over: None,
            });
        }
    }
    rows
}

// --- Relevance judging (LLM, injected) — see src/AGENTS.md §ingest.rs ---

/// One relevance verdict per row, against a free-text `criterion`. LLM in
/// production ([`LlmRelevanceJudge`]); mock in tests.
pub trait RelevanceJudge {
    fn relevant(&self, criterion: &str, rows: &[GenExample]) -> anyhow::Result<Vec<bool>>;
}

/// LLM relevance judge over any [`ChatTransport`]. Batches, parses a JSON array of
/// relevant item numbers, and errs toward inclusion on failure.
pub struct LlmRelevanceJudge<T: ChatTransport> {
    transport: T,
    batch: usize,
}

impl<T: ChatTransport> LlmRelevanceJudge<T> {
    /// Construct a judge with the given transport and batch size (minimum 1).
    pub fn new(transport: T, batch: usize) -> Self {
        Self {
            transport,
            batch: batch.max(1),
        }
    }
}

impl<T: ChatTransport> RelevanceJudge for LlmRelevanceJudge<T> {
    fn relevant(&self, criterion: &str, rows: &[GenExample]) -> anyhow::Result<Vec<bool>> {
        let mut keep = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(self.batch) {
            let messages = [
                ChatMessage::system(
                    "You curate training data. Given a numbered list of agent \
                     interactions and a relevance criterion, reply with ONLY a JSON \
                     array of the numbers that are relevant (e.g. [1,3,4]). No prose.",
                ),
                ChatMessage::user(build_judge_prompt(criterion, chunk)),
            ];
            match self.transport.complete(&messages) {
                Ok(answer) => {
                    let relevant = parse_relevant_indices(&answer, chunk.len());
                    for i in 0..chunk.len() {
                        keep.push(relevant.contains(&i));
                    }
                }
                Err(e) => {
                    eprintln!("ingest judge: batch failed, keeping its rows ({e})");
                    keep.extend(std::iter::repeat(true).take(chunk.len()));
                }
            }
        }
        Ok(keep)
    }
}

/// Keep only the rows the judge rates relevant. Empty input short-circuits (no
/// LLM call). A convenience over [`RelevanceJudge::relevant`].
pub fn filter_relevant(
    judge: &dyn RelevanceJudge,
    criterion: &str,
    rows: Vec<GenExample>,
) -> anyhow::Result<Vec<GenExample>> {
    if rows.is_empty() {
        return Ok(rows);
    }
    let keep = judge.relevant(criterion, &rows)?;
    Ok(rows
        .into_iter()
        .zip(keep)
        .filter_map(|(row, k)| k.then_some(row))
        .collect())
}

/// Render the judge prompt: the criterion + a 1-based numbered list of compact
/// candidate renderings.
fn build_judge_prompt(criterion: &str, rows: &[GenExample]) -> String {
    let mut s = format!("Relevance criterion: {criterion}\n\nInteractions:\n");
    for (i, row) in rows.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, render_candidate(row)));
    }
    s.push_str("\nReply with a JSON array of the relevant numbers only.");
    s
}

/// A one-line, length-capped rendering of a candidate for the judge to read.
fn render_candidate(row: &GenExample) -> String {
    match row {
        GenExample::Cli { command, .. } => format!("shell: {}", truncate(command, 200)),
        GenExample::ToolCall {
            tool, arguments, ..
        } => format!(
            "tool {tool}: {}",
            truncate(&serde_json::to_string(arguments).unwrap_or_default(), 200)
        ),
        GenExample::Qa {
            prompt, completion, ..
        } => {
            format!(
                "Q: {} / A: {}",
                truncate(prompt, 120),
                truncate(completion, 200)
            )
        }
        GenExample::Instruction {
            instruction,
            output,
            ..
        } => format!(
            "instruction: {} -> {}",
            truncate(instruction, 120),
            truncate(output, 160)
        ),
        GenExample::Completion { text, .. } => format!("text: {}", truncate(text, 200)),
        GenExample::Contrastive { query, .. } => format!("contrastive: {}", truncate(query, 160)),
        GenExample::Skill {
            skill_name,
            invocation,
            ..
        } => format!("skill {skill_name}: {}", truncate(invocation, 200)),
        GenExample::ReasoningEdit {
            prompt,
            final_action,
            ..
        } => format!(
            "reasoning: {} -> {}",
            truncate(prompt, 120),
            truncate(final_action, 160)
        ),
    }
}

/// 0-based relevant indices from the reply: a JSON array of 1-based numbers
/// anywhere in the text (prose/fences tolerated); out-of-range ignored.
fn parse_relevant_indices(answer: &str, n: usize) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let (Some(start), Some(end)) = (answer.find('['), answer.rfind(']')) else {
        return out;
    };
    if end <= start {
        return out;
    }
    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&answer[start..=end]) {
        for it in items {
            if let Some(num) = it.as_u64() {
                let idx = num as usize;
                if idx >= 1 && idx <= n {
                    out.insert(idx - 1);
                }
            }
        }
    }
    out
}

// --- Shared helpers (pure) ---

/// Human text from a `user` entry (string or `text` blocks), `<…>` wrappers stripped.
fn user_text(v: &Value) -> Option<String> {
    let content = v.get("message")?.get("content")?;
    let raw = match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return None,
    };
    let cleaned = strip_noise(&raw);
    let trimmed = cleaned.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Drop `<…>` harness wrappers (IDE/system reminders, tool envelopes) so the
/// fallback intent is the human's words.
fn strip_noise(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            // char is exhaustive; these two guards cover all remaining code points.
            _ if !in_tag => out.push(ch),
            _ => {} // inside a tag — discard
        }
    }
    out
}

/// Choose a tool call's training prompt: its own description (best), else the
/// recent user message (capped), else a generic instruction.
fn pick_intent(desc: Option<&str>, last_user: Option<&str>) -> String {
    if let Some(d) = desc {
        let d = d.trim();
        if !d.is_empty() {
            return d.to_string();
        }
    }
    if let Some(u) = last_user {
        let u = u.trim();
        if !u.is_empty() {
            return truncate(u, MAX_FALLBACK_PROMPT);
        }
    }
    "Perform the appropriate action for the task.".to_string()
}

/// Char-safe truncation to at most `max` chars, trimmed.
fn truncate(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    t.chars().take(max).collect::<String>().trim().to_string()
}

/// Push a row only if its content key is new (dedup within a log). Returns the
/// index of the pushed row, or `None` if it was a duplicate.
fn push_unique(
    rows: &mut Vec<GenExample>,
    seen: &mut BTreeSet<String>,
    row: GenExample,
) -> Option<usize> {
    if seen.insert(content_key(&row)) {
        rows.push(row);
        Some(rows.len() - 1)
    } else {
        None
    }
}

/// A stable content key for dedup.
fn content_key(row: &GenExample) -> String {
    match row {
        GenExample::Cli {
            prompt, command, ..
        } => {
            format!("cli\u{1}{}\u{1}{}", prompt.trim(), command.trim())
        }
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
        GenExample::Qa {
            prompt, completion, ..
        } => {
            format!("qa\u{1}{}\u{1}{}", prompt.trim(), completion.trim())
        }
        GenExample::Completion { text, .. } => format!("compl\u{1}{}", text.trim()),
        other => format!("other\u{1}{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(json: &str) -> String {
        json.to_string()
    }

    #[test]
    fn bash_tool_use_becomes_cli_row_with_description_prompt() {
        let log = [
            line(r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"do the thing"}]}}"#),
            line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"git status","description":"Show working tree status"}}]}}"#),
            line(r#"{"type":"queue-operation","operation":"enqueue"}"#),
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 1);
        match &rows[0] {
            GenExample::Cli {
                prompt,
                command,
                gen,
                ..
            } => {
                assert_eq!(prompt, "Show working tree status");
                assert_eq!(command, "git status");
                // Per-source stamp now (track 31 Q2): transcript rows are tagged
                // `ingest:transcript` so a catastrophe quarantines only that source.
                assert_eq!(gen.as_deref(), Some(INGEST_GEN_TRANSCRIPT));
            }
            other => panic!("expected Cli, got {other:?}"),
        }
    }

    #[test]
    fn arbitrary_tool_use_becomes_tool_call_row() {
        let log = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Grep","input":{"pattern":"TODO","path":"src","description":"find TODOs"}}]}}"#;
        let rows = interaction_log_rows(log);
        assert_eq!(rows.len(), 1);
        match &rows[0] {
            GenExample::ToolCall {
                tool,
                arguments,
                prompt,
                ..
            } => {
                assert_eq!(tool, "Grep");
                assert_eq!(prompt, "find TODOs");
                // The harness-only `description` is stripped from training args.
                assert!(arguments.get("description").is_none());
                assert_eq!(
                    arguments.get("pattern").and_then(|v| v.as_str()),
                    Some("TODO")
                );
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn prose_only_assistant_turn_becomes_qa_row() {
        let log = [
            line(r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"what is 2+2?"}]}}"#),
            line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"It is 4."}]}}"#),
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 1);
        match &rows[0] {
            GenExample::Qa {
                prompt, completion, ..
            } => {
                assert_eq!(prompt, "what is 2+2?");
                assert_eq!(completion, "It is 4.");
            }
            other => panic!("expected Qa, got {other:?}"),
        }
    }

    #[test]
    fn tool_turn_does_not_also_emit_prose_qa() {
        // An assistant turn with BOTH reasoning text and a tool call yields only
        // the tool row (the prose is reasoning, not the answer).
        let log = [
            line(r#"{"type":"user","message":{"role":"user","content":"list files"}}"#),
            line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Let me list them."},{"type":"tool_use","name":"Bash","input":{"command":"ls -la","description":"list files"}}]}}"#),
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 1);
        assert!(matches!(rows[0], GenExample::Cli { .. }));
    }

    #[test]
    fn noise_lines_skipped_and_rows_deduped() {
        let log = [
            line(r#"not json at all"#),
            line(r#"{"type":"queue-operation","operation":"dequeue"}"#),
            line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"ls","description":"list"}}]}}"#),
            line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"ls","description":"list"}}]}}"#),
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 1, "garbage skipped, duplicate collapsed");
    }

    #[test]
    fn over_long_payloads_are_dropped() {
        // Beyond the VERIFIED cap, parsing itself drops the row.
        let huge = "x".repeat(MAX_ROW_CHARS_VERIFIED + 10);
        let log = format!(
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"Bash","input":{{"command":"{huge}"}}}}]}}}}"#
        );
        assert!(interaction_log_rows(&log).is_empty());

        // Between MAX_ROW_CHARS and the VERIFIED cap: parsing KEEPS it (track 37),
        // but an unknown-outcome row over MAX_ROW_CHARS is dropped by the outcome
        // filter (only verified successes earn the larger cap).
        let mid = "x".repeat(MAX_ROW_CHARS + 10);
        let log = format!(
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"Bash","input":{{"command":"{mid}"}}}}]}}}}"#
        );
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 1, "parse keeps it (below verified cap)");
        let filtered = filter_outcomes(rows, Tier::Private);
        assert!(filtered.kept.is_empty(), "unknown over MAX_ROW_CHARS dropped");
    }

    #[test]
    fn doc_chunks_into_completion_rows() {
        let doc = "# Title\n\nThis is a sufficiently long paragraph that comfortably clears the eighty character minimum threshold used for a row.\n\nshort\n\nAnother long paragraph that also clears the eighty character minimum threshold for a chunk row.";
        let rows = doc_completion_rows(doc, "notes.md", 10);
        assert_eq!(rows.len(), 2);
        assert!(matches!(rows[0], GenExample::Completion { .. }));
    }

    // --- relevance judge (mock transport) ---

    struct MockTransport {
        reply: String,
    }
    impl ChatTransport for MockTransport {
        fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(self.reply.clone())
        }
    }

    fn cli(cmd: &str) -> GenExample {
        GenExample::Cli {
            prompt: "p".into(),
            command: cmd.into(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        }
    }

    #[test]
    fn llm_judge_keeps_only_relevant_rows() {
        // The model says items 1 and 3 are relevant (1-based) → keep rows 0 and 2.
        let judge = LlmRelevanceJudge::new(
            MockTransport {
                reply: "[1, 3]".into(),
            },
            10,
        );
        let rows = vec![cli("a"), cli("b"), cli("c")];
        let kept = filter_relevant(&judge, "criterion", rows).unwrap();
        let cmds: Vec<&str> = kept
            .iter()
            .filter_map(|r| match r {
                GenExample::Cli { command, .. } => Some(command.as_str()),
                GenExample::Qa { .. }
                | GenExample::Instruction { .. }
                | GenExample::Completion { .. }
                | GenExample::Contrastive { .. }
                | GenExample::ToolCall { .. }
                | GenExample::Skill { .. }
                | GenExample::ReasoningEdit { .. } => None,
            })
            .collect();
        assert_eq!(cmds, vec!["a", "c"]);
    }

    #[test]
    fn llm_judge_parses_array_amid_prose_and_drops_all_on_empty() {
        let judge = LlmRelevanceJudge::new(
            MockTransport {
                reply: "Sure! The relevant ones are [2].".into(),
            },
            10,
        );
        let kept = filter_relevant(&judge, "c", vec![cli("a"), cli("b")]).unwrap();
        assert_eq!(kept.len(), 1);

        let none = LlmRelevanceJudge::new(MockTransport { reply: "[]".into() }, 10);
        assert!(filter_relevant(&none, "c", vec![cli("a")])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn empty_input_makes_no_llm_call() {
        // reply would keep nothing, but empty input must short-circuit to empty.
        let judge = LlmRelevanceJudge::new(
            MockTransport {
                reply: "[1]".into(),
            },
            10,
        );
        assert!(filter_relevant(&judge, "c", vec![]).unwrap().is_empty());
    }

    // --- Phase A: outcome stamping + retry-collapse ---

    /// A tool_use turn + a following user turn carrying its tool_result.
    fn tool_and_result(cmd: &str, id: &str, is_error: bool) -> String {
        [
            format!(
                r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","id":"{id}","name":"Bash","input":{{"command":"{cmd}","description":"run {cmd}"}}}}]}}}}"#
            ),
            format!(
                r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"{id}","is_error":{is_error},"content":"out"}}]}}}}"#
            ),
        ]
        .join("\n")
    }

    #[test]
    fn tool_result_stamps_outcome_on_rows() {
        let log = [
            tool_and_result("cargo build", "t1", true),
            tool_and_result("cargo test", "t2", false),
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].outcome(), Outcome::Failure);
        assert_eq!(rows[1].outcome(), Outcome::Success);
    }

    #[test]
    fn retry_chain_collapses_to_one_success_with_chosen_over() {
        // fail × 3 then a success of the ~same command → exactly 1 kept row.
        let log = [
            tool_and_result("cargo build --release", "a", true),
            tool_and_result("cargo build --release --verbose", "b", true),
            tool_and_result("cargo build --release --locked", "c", true),
            tool_and_result("cargo build --release --frozen", "d", false),
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows.len(), 4, "four distinct commands parsed pre-collapse");

        let filtered = filter_outcomes(rows, Tier::Private);
        assert_eq!(filtered.kept.len(), 1, "collapsed to one success");
        assert_eq!(filtered.rejected.len(), 3, "three failures in the sidecar");
        assert_eq!(filtered.kept[0].outcome(), Outcome::Success);
        assert!(
            filtered.kept[0].chosen_over().is_some(),
            "success records the rejected preference pair"
        );
    }

    #[test]
    fn bare_failure_is_excluded_from_training() {
        let log = tool_and_result("rm -rf /nope", "x", true);
        let rows = interaction_log_rows(&log);
        let filtered = filter_outcomes(rows, Tier::Private);
        assert!(filtered.kept.is_empty(), "bare failure not trained");
        assert_eq!(filtered.rejected.len(), 1, "kept for audit");
    }

    #[test]
    fn long_success_kept_but_long_unknown_dropped() {
        // A 5000-char command: as a verified success it survives; as unknown it
        // exceeds MAX_ROW_CHARS and is dropped.
        let long = "echo ".to_string() + &"x".repeat(5000);
        let succ = GenExample::Cli {
            prompt: "p".into(),
            command: long.clone(),
            source: None,
            gen: None,
            outcome: Outcome::Success,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        };
        let unk = GenExample::Cli {
            prompt: "p".into(),
            command: long,
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        };
        let f1 = filter_outcomes(vec![succ], Tier::Private);
        assert_eq!(f1.kept.len(), 1, "long verified success kept");
        let f2 = filter_outcomes(vec![unk], Tier::Private);
        assert_eq!(f2.kept.len(), 0, "long unknown dropped");
    }

    #[test]
    fn tier_stamped_on_kept_rows() {
        let log = tool_and_result("ls", "t", false);
        let rows = interaction_log_rows(&log);
        let filtered = filter_outcomes(rows, Tier::Shared);
        assert_eq!(filtered.kept.len(), 1);
        assert_eq!(filtered.kept[0].tier(), Tier::Shared);
    }

    #[test]
    fn text_heuristic_marks_failure_without_is_error_flag() {
        let log = [
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"z","name":"Bash","input":{"command":"foo","description":"run foo"}}]}}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"z","content":"foo: command not found"}]}}"#,
        ]
        .join("\n");
        let rows = interaction_log_rows(&log);
        assert_eq!(rows[0].outcome(), Outcome::Failure);
    }
}
