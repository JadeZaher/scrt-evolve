//! Per-pair data judge (track 37 Phase B). Scores each candidate training row
//! 0–1 on correctness / quality / steering-alignment BEFORE it enters the queue
//! — the AlpaGasus/Deita "judge-scored selection beats train-on-everything"
//! lever. Mirrors the injected-`ChatTransport` shape of `ingest::LlmRelevanceJudge`
//! and `eval::degrade::LlmDegradationJudge`. Design: see `src/AGENTS.md` §judge.

use crate::dataset::{GenExample, Verdict};
use crate::generate::api::{ChatMessage, ChatTransport};

/// On-judge-error policy. `Keep` (fail-open, default) matches the existing
/// relevance-judge precedent + the track-31 preflight backstop: a flaky judge
/// must not stall an unattended daemon. `Drop` (fail-closed) is the documented
/// flip for users publishing branches P2P (unjudged data never ships).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnError {
    #[default]
    Keep,
    Drop,
}

impl OnError {
    /// Parse a config string; unknown ⇒ the safe `Keep` default.
    pub fn from_config(s: Option<&str>) -> OnError {
        match s.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("drop") => OnError::Drop,
            _ => OnError::Keep,
        }
    }
}

/// One 0–1 quality score per row. LLM in production ([`LlmPairJudge`]); mock in
/// tests.
pub trait PairJudge {
    /// Score each row 0.0–1.0. `steering` is the composed constitution/taste
    /// block (when set) so the judge can weigh steering-alignment.
    fn score(&self, rows: &[GenExample], steering: Option<&str>) -> anyhow::Result<Vec<f32>>;
}

/// A judged row: the (mutated, stamped) row plus whether it cleared the
/// threshold. Sub-threshold rows are dropped from training but retained by the
/// caller for the audit sidecar.
#[derive(Debug)]
pub struct JudgedRows {
    pub kept: Vec<GenExample>,
    pub dropped: Vec<GenExample>,
}

/// LLM pair judge over any [`ChatTransport`]. Batches, parses a JSON array of
/// per-row scores, applies the `on_error` policy on transport failure.
pub struct LlmPairJudge<T: ChatTransport> {
    transport: T,
    batch: usize,
    on_error: OnError,
}

impl<T: ChatTransport> LlmPairJudge<T> {
    /// Create a judge with the given transport, batch size (clamped to ≥1), and error policy.
    pub fn new(transport: T, batch: usize, on_error: OnError) -> Self {
        Self {
            transport,
            batch: batch.max(1),
            on_error,
        }
    }
}

impl<T: ChatTransport> PairJudge for LlmPairJudge<T> {
    fn score(&self, rows: &[GenExample], steering: Option<&str>) -> anyhow::Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(self.batch) {
            let messages = [
                ChatMessage::system(
                    "You rate candidate supervised-fine-tuning rows. For each numbered \
                     row give a single quality score from 0.0 (useless/incorrect) to \
                     1.0 (correct, well-formed, useful), weighing correctness, quality, \
                     and — when a STEERING block is given — alignment with it. Reply \
                     with ONLY a JSON array of the scores in order (e.g. [0.9,0.2,0.7]). \
                     No prose.",
                ),
                ChatMessage::user(build_score_prompt(chunk, steering)),
            ];
            match self.transport.complete(&messages) {
                Ok(answer) => {
                    let parsed = parse_scores(&answer, chunk.len());
                    scores.extend(parsed);
                }
                Err(e) => {
                    // Fail-open ⇒ 1.0 (keep); fail-closed ⇒ 0.0 (drop).
                    let fill = match self.on_error {
                        OnError::Keep => 1.0,
                        OnError::Drop => 0.0,
                    };
                    eprintln!(
                        "pair judge: batch failed, applying on_error={:?} ({e})",
                        self.on_error
                    );
                    scores.extend(std::iter::repeat(fill).take(chunk.len()));
                }
            }
        }
        Ok(scores)
    }
}

/// Score `rows` with `judge`, stamp each with its score + verdict, and split into
/// kept (≥ `min_score`) / dropped. Empty input short-circuits (no LLM call).
pub fn judge_rows(
    judge: &dyn PairJudge,
    rows: Vec<GenExample>,
    min_score: f32,
    steering: Option<&str>,
) -> anyhow::Result<JudgedRows> {
    if rows.is_empty() {
        return Ok(JudgedRows {
            kept: Vec::new(),
            dropped: Vec::new(),
        });
    }
    let scores = judge.score(&rows, steering)?;
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for (mut row, score) in rows.into_iter().zip(scores) {
        let score = score.clamp(0.0, 1.0);
        if score >= min_score {
            row.set_judge(score, Verdict::Keep);
            kept.push(row);
        } else {
            row.set_judge(score, Verdict::Drop);
            dropped.push(row);
        }
    }
    Ok(JudgedRows { kept, dropped })
}

/// Render the score prompt: optional steering block + a 1-based numbered list of
/// compact candidate renderings.
fn build_score_prompt(rows: &[GenExample], steering: Option<&str>) -> String {
    let mut s = String::new();
    if let Some(steer) = steering {
        let steer = steer.trim();
        if !steer.is_empty() {
            s.push_str("STEERING (weigh alignment with this):\n");
            s.push_str(steer);
            s.push_str("\n\n");
        }
    }
    s.push_str("Rows to score:\n");
    for (i, row) in rows.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, render_row(row)));
    }
    s.push_str("\nReply with a JSON array of scores 0.0–1.0, one per row, in order.");
    s
}

/// A one-line, length-capped rendering of a candidate for the judge.
fn render_row(row: &GenExample) -> String {
    match row {
        GenExample::Cli {
            prompt, command, ..
        } => format!("[cli] {} => {}", cap(prompt, 120), cap(command, 200)),
        GenExample::ToolCall {
            prompt,
            tool,
            arguments,
            ..
        } => format!(
            "[tool_call] {} => {tool}({})",
            cap(prompt, 120),
            cap(&serde_json::to_string(arguments).unwrap_or_default(), 200)
        ),
        GenExample::Qa {
            prompt, completion, ..
        } => format!("[qa] Q: {} / A: {}", cap(prompt, 120), cap(completion, 200)),
        GenExample::Instruction {
            instruction,
            output,
            ..
        } => format!(
            "[instruction] {} => {}",
            cap(instruction, 120),
            cap(output, 200)
        ),
        GenExample::Completion { text, .. } => format!("[completion] {}", cap(text, 200)),
        GenExample::Contrastive { query, .. } => format!("[contrastive] {}", cap(query, 160)),
        GenExample::Skill {
            skill_name,
            invocation,
            ..
        } => format!("[skill] {skill_name}: {}", cap(invocation, 200)),
        GenExample::ReasoningEdit {
            prompt,
            final_action,
            ..
        } => format!(
            "[reasoning_edit] {} => {}",
            cap(prompt, 120),
            cap(final_action, 160)
        ),
    }
}

/// Char-safe cap for a rendered field.
fn cap(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    t.chars().take(max).collect()
}

/// Parse a JSON array of `n` scores from the reply (prose/fences tolerated).
/// Missing/garbled entries default to 1.0 (fail-open per-item — the batch-level
/// `on_error` policy governs whole-batch failure; a partially-garbled reply
/// should not silently drop rows).
fn parse_scores(answer: &str, n: usize) -> Vec<f32> {
    let mut out = vec![1.0f32; n];
    let (Some(start), Some(end)) = (answer.find('['), answer.rfind(']')) else {
        return out;
    };
    if end <= start {
        return out;
    }
    if let Ok(serde_json::Value::Array(items)) =
        serde_json::from_str::<serde_json::Value>(&answer[start..=end])
    {
        for (i, it) in items.iter().enumerate() {
            if i >= n {
                break;
            }
            if let Some(f) = it.as_f64() {
                out[i] = (f as f32).clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// Aggregate training-signal stats over a dataset, for the branch manifest's
/// `eval_report` (the lexame "marked expertise" fields). Additive keys only.
pub fn dataset_signal_stats(rows: &[GenExample]) -> std::collections::BTreeMap<String, f64> {
    use crate::dataset::Outcome;
    let mut m = std::collections::BTreeMap::new();
    let total = rows.len();
    if total == 0 {
        return m;
    }
    let judged: Vec<f32> = rows.iter().filter_map(|r| r.judge_score()).collect();
    let judged_fraction = judged.len() as f64 / total as f64;
    let judge_mean_score = if judged.is_empty() {
        0.0
    } else {
        judged.iter().map(|s| *s as f64).sum::<f64>() / judged.len() as f64
    };
    let verified = rows
        .iter()
        .filter(|r| r.outcome() == Outcome::Success)
        .count();
    let outcome_verified_fraction = verified as f64 / total as f64;
    m.insert("judged_fraction".to_string(), judged_fraction);
    m.insert("judge_mean_score".to_string(), judge_mean_score);
    m.insert(
        "outcome_verified_fraction".to_string(),
        outcome_verified_fraction,
    );
    m
}

/// Most-restrictive tier over a dataset (Private wins). `Private` on empty.
pub fn dataset_tier(rows: &[GenExample]) -> crate::dataset::Tier {
    use crate::dataset::Tier;
    rows.iter()
        .fold(Tier::Shared, |acc, r| acc.most_restrictive(r.tier()))
}

/// Rejection sampling / best-of-N (track 37 Phase C, RAFT-style): given `n`
/// candidate rows per seed, judge them, keep the top-`keep_k` scoring ≥
/// `min_score` (stamped `gen=rsample:<n>`). Returns the kept rows across all
/// seeds. `candidates` is a flat list; `n` is the group size (candidates per
/// seed passage, contiguous). Fewer than `n` in the last group is tolerated.
pub fn rejection_sample(
    judge: &dyn PairJudge,
    candidates: Vec<GenExample>,
    n: usize,
    keep_k: usize,
    min_score: f32,
    steering: Option<&str>,
) -> anyhow::Result<Vec<GenExample>> {
    if candidates.is_empty() || n == 0 {
        return Ok(candidates);
    }
    let scores = judge.score(&candidates, steering)?;
    let mut scored: Vec<(GenExample, f32)> = candidates.into_iter().zip(scores).collect();
    let mut kept = Vec::new();
    for group in scored.chunks_mut(n) {
        // Rank this seed's candidates by score, keep top-k above threshold.
        group.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (mut row, score) in group.iter().take(keep_k).cloned() {
            if score >= min_score {
                let score = score.clamp(0.0, 1.0);
                row.set_judge(score, Verdict::Keep);
                stamp_rsample(&mut row, n);
                kept.push(row);
            }
        }
    }
    Ok(kept)
}

/// Stamp `gen=rsample:<n>` on a rejection-sampled row (preserving any prior gen
/// suffix would over-complicate; the rsample stamp is the provenance of record).
fn stamp_rsample(row: &mut GenExample, n: usize) {
    let stamp = format!("rsample:{n}");
    match row {
        GenExample::Qa { gen, .. }
        | GenExample::Instruction { gen, .. }
        | GenExample::Completion { gen, .. }
        | GenExample::ToolCall { gen, .. }
        | GenExample::Cli { gen, .. }
        | GenExample::Skill { gen, .. }
        | GenExample::ReasoningEdit { gen, .. } => *gen = Some(stamp),
        GenExample::Contrastive { .. } => {}
    }
}

// --- Phase C: Evol/Self-Instruct expansion (judge-gated volume synthesis) ---

/// Evol-Instruct/Self-Instruct expansion (track 37 Phase C): for each seed row
/// and each evolution `op` (deepen/broaden/concretize/…), ask the teacher for an
/// evolved variant, judge it, and admit only rows ≥ `min_score` (stamped
/// `gen=expand:<op>`). Stage-independent — operates on in-memory rows so the CLI
/// can run it against a `dataset.jsonl` with no daemon. See `src/AGENTS.md` §judge.
pub fn expand_dataset<T: ChatTransport>(
    transport: &T,
    seeds: &[GenExample],
    ops: &[String],
    judge: &dyn PairJudge,
    min_score: f32,
    steering: Option<&str>,
) -> anyhow::Result<Vec<GenExample>> {
    let mut candidates: Vec<GenExample> = Vec::new();
    for seed in seeds {
        for op in ops {
            let messages = [
                ChatMessage::system(
                    "You EVOLVE one supervised-fine-tuning row into a NEW, related row \
                     that is more useful for training. Apply the requested evolution \
                     operation. Keep the SAME kind/shape (a cli stays a cli, a qa stays \
                     a qa). Reply with ONLY the new row's core fields as a compact JSON \
                     object matching the input's shape — no prose, no markdown.",
                ),
                ChatMessage::user(build_evolve_prompt(seed, op)),
            ];
            let Ok(answer) = transport.complete(&messages) else {
                continue; // a failed evolution just yields no candidate
            };
            if let Some(row) = parse_evolved(seed, op, &answer) {
                candidates.push(row);
            }
        }
    }
    // Judge every candidate; admit the kept ones.
    let judged = judge_rows(judge, candidates, min_score, steering)?;
    Ok(judged.kept)
}

/// Render the evolve instruction: the op + the seed's current content.
fn build_evolve_prompt(seed: &GenExample, op: &str) -> String {
    let (shape, body) = match seed {
        GenExample::Cli {
            prompt, command, ..
        } => ("cli", format!("prompt: {prompt}\ncommand: {command}")),
        GenExample::Qa {
            prompt, completion, ..
        } => ("qa", format!("prompt: {prompt}\ncompletion: {completion}")),
        GenExample::Instruction {
            instruction,
            output,
            ..
        } => (
            "instruction",
            format!("instruction: {instruction}\noutput: {output}"),
        ),
        GenExample::ToolCall {
            prompt,
            tool,
            arguments,
            ..
        } => (
            "tool_call",
            format!(
                "prompt: {prompt}\ntool: {tool}\narguments: {}",
                serde_json::to_string(arguments).unwrap_or_default()
            ),
        ),
        GenExample::Completion { text, .. } => ("completion", format!("text: {text}")),
        _ => ("qa", String::new()),
    };
    format!(
        "Evolution operation: {op}\nInput row (shape={shape}):\n{body}\n\n\
         Produce the evolved row as JSON with the same fields as the input shape."
    )
}

/// Parse the teacher's evolved row, preserving the seed's shape; stamp
/// `gen=expand:<op>`. Best-effort — returns `None` on unparseable/empty output.
fn parse_evolved(seed: &GenExample, op: &str, answer: &str) -> Option<GenExample> {
    let start = answer.find('{')?;
    let end = answer.rfind('}')?;
    if end <= start {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&answer[start..=end]).ok()?;
    let gen_stamp = format!("expand:{op}");
    let get = |k: &str| v.get(k).and_then(serde_json::Value::as_str).map(str::to_string);
    let row = match seed {
        GenExample::Cli { .. } => {
            let prompt = get("prompt")?;
            let command = get("command")?;
            if prompt.trim().is_empty() || command.trim().is_empty() {
                return None;
            }
            GenExample::Cli {
                prompt,
                command,
                source: Some("synth".to_string()),
                gen: Some(gen_stamp),
                outcome: crate::dataset::Outcome::Unknown,
                judge_score: None,
                judge_verdict: Verdict::Unjudged,
                tier: seed.tier(),
                chosen_over: None,
            }
        }
        GenExample::Instruction { .. } => {
            let instruction = get("instruction")?;
            let output = get("output")?;
            if instruction.trim().is_empty() || output.trim().is_empty() {
                return None;
            }
            GenExample::Instruction {
                instruction,
                input: get("input").unwrap_or_default(),
                output,
                source: Some("synth".to_string()),
                gen: Some(gen_stamp),
                outcome: crate::dataset::Outcome::Unknown,
                judge_score: None,
                judge_verdict: Verdict::Unjudged,
                tier: seed.tier(),
                chosen_over: None,
            }
        }
        // Default (qa, completion, tool_call, etc.): synthesize a Qa row from
        // prompt/completion when present — the safe, universally-trainable shape.
        _ => {
            let prompt = get("prompt").or_else(|| get("instruction"))?;
            let completion = get("completion").or_else(|| get("output")).or_else(|| get("text"))?;
            if prompt.trim().is_empty() || completion.trim().is_empty() {
                return None;
            }
            GenExample::Qa {
                prompt,
                completion,
                source: Some("synth".to_string()),
                gen: Some(gen_stamp),
                outcome: crate::dataset::Outcome::Unknown,
                judge_score: None,
                judge_verdict: Verdict::Unjudged,
                tier: seed.tier(),
                chosen_over: None,
            }
        }
    };
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Outcome, Tier};

    struct MockJudge {
        reply: String,
    }
    impl ChatTransport for MockJudge {
        fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(self.reply.clone())
        }
    }
    struct ErrJudge;
    impl ChatTransport for ErrJudge {
        fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            anyhow::bail!("boom")
        }
    }

    fn qa(a: &str) -> GenExample {
        GenExample::Qa {
            prompt: "q".into(),
            completion: a.into(),
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
    fn scores_parsed_and_thresholded() {
        let judge = LlmPairJudge::new(
            MockJudge {
                reply: "[0.9, 0.2, 0.7]".into(),
            },
            10,
            OnError::Keep,
        );
        let rows = vec![qa("a"), qa("b"), qa("c")];
        let out = judge_rows(&judge, rows, 0.5, None).unwrap();
        assert_eq!(out.kept.len(), 2, "0.9 and 0.7 clear 0.5");
        assert_eq!(out.dropped.len(), 1, "0.2 dropped");
        assert_eq!(out.kept[0].judge_score(), Some(0.9));
        assert_eq!(out.kept[0].judge_verdict(), Verdict::Keep);
        assert_eq!(out.dropped[0].judge_verdict(), Verdict::Drop);
    }

    #[test]
    fn fail_open_keeps_all_fail_closed_drops_all() {
        let rows = vec![qa("a"), qa("b")];
        let open = LlmPairJudge::new(ErrJudge, 10, OnError::Keep);
        let out = judge_rows(&open, rows.clone(), 0.5, None).unwrap();
        assert_eq!(out.kept.len(), 2, "fail-open keeps");

        let closed = LlmPairJudge::new(ErrJudge, 10, OnError::Drop);
        let out = judge_rows(&closed, rows, 0.5, None).unwrap();
        assert_eq!(out.dropped.len(), 2, "fail-closed drops");
    }

    #[test]
    fn empty_input_no_call() {
        let judge = LlmPairJudge::new(MockJudge { reply: "[1]".into() }, 10, OnError::Keep);
        let out = judge_rows(&judge, vec![], 0.5, None).unwrap();
        assert!(out.kept.is_empty() && out.dropped.is_empty());
    }

    #[test]
    fn garbled_reply_defaults_to_keep_per_item() {
        let judge = LlmPairJudge::new(
            MockJudge {
                reply: "no json here".into(),
            },
            10,
            OnError::Keep,
        );
        let out = judge_rows(&judge, vec![qa("a")], 0.5, None).unwrap();
        assert_eq!(out.kept.len(), 1, "unparseable ⇒ 1.0 ⇒ kept");
    }

    struct SeqTransport {
        // returns an evolved qa row regardless of prompt
        reply: String,
    }
    impl ChatTransport for SeqTransport {
        fn complete(&self, _m: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(self.reply.clone())
        }
    }

    #[test]
    fn expand_produces_judged_stamped_rows() {
        // Teacher returns an evolved qa row; judge keeps it.
        let teacher = SeqTransport {
            reply: r#"{"prompt":"evolved q","completion":"evolved a"}"#.into(),
        };
        let judge = LlmPairJudge::new(MockJudge { reply: "[0.9]".into() }, 10, OnError::Keep);
        let seeds = vec![qa("seed a"), qa("seed b")];
        let ops = vec!["deepen".to_string(), "broaden".to_string()];
        let out = expand_dataset(&teacher, &seeds, &ops, &judge, 0.5, None).unwrap();
        // 2 seeds × 2 ops = 4 candidates, all judged-kept.
        assert_eq!(out.len(), 4, "all evolved rows admitted");
        for r in &out {
            assert!(matches!(r, GenExample::Qa { .. }));
            assert_eq!(r.judge_verdict(), Verdict::Keep);
            match r {
                GenExample::Qa { gen, .. } => {
                    assert!(gen.as_deref().unwrap().starts_with("expand:"));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn expand_drops_low_scoring_rows() {
        let teacher = SeqTransport {
            reply: r#"{"prompt":"x","completion":"y"}"#.into(),
        };
        let judge = LlmPairJudge::new(MockJudge { reply: "[0.1]".into() }, 10, OnError::Keep);
        let out = expand_dataset(&teacher, &[qa("s")], &["deepen".to_string()], &judge, 0.5, None)
            .unwrap();
        assert!(out.is_empty(), "0.1 < 0.5 ⇒ dropped");
    }

    #[test]
    fn rejection_sample_keeps_top_k_ranked() {
        // 8 candidates for one seed; scores make 5 clear 0.5; keep_k=3 → top 3.
        let judge = LlmPairJudge::new(
            MockJudge {
                reply: "[0.1,0.9,0.3,0.8,0.7,0.2,0.95,0.6]".into(),
            },
            10,
            OnError::Keep,
        );
        let cands: Vec<GenExample> = (0..8).map(|i| qa(&format!("c{i}"))).collect();
        let kept = rejection_sample(&judge, cands, 8, 3, 0.5, None).unwrap();
        assert_eq!(kept.len(), 3, "keep_k=3");
        // Ranked descending: 0.95, 0.9, 0.8.
        assert_eq!(kept[0].judge_score(), Some(0.95));
        assert_eq!(kept[1].judge_score(), Some(0.9));
        assert_eq!(kept[2].judge_score(), Some(0.8));
        for r in &kept {
            match r {
                GenExample::Qa { gen, .. } => {
                    assert_eq!(gen.as_deref(), Some("rsample:8"))
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn signal_stats_and_tier_rollup() {
        let mut r1 = qa("a");
        r1.set_judge(0.8, Verdict::Keep);
        r1.set_outcome(Outcome::Success);
        r1.set_tier(Tier::Shared);
        let mut r2 = qa("b");
        r2.set_judge(0.6, Verdict::Keep);
        r2.set_tier(Tier::Private); // most-restrictive-wins ⇒ Private
        let rows = vec![r1, r2];
        let stats = dataset_signal_stats(&rows);
        assert_eq!(stats["judged_fraction"], 1.0);
        assert!((stats["judge_mean_score"] - 0.7).abs() < 1e-6);
        assert!((stats["outcome_verified_fraction"] - 0.5).abs() < 1e-6);
        assert_eq!(dataset_tier(&rows), Tier::Private);
    }
}
