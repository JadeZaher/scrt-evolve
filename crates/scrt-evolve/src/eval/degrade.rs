//! Degradation judge (track 32) — the "did it get WORSE?" LLM gate.
//!
//! The correctness gate ([`super::verdict::classify`]) accepts a step only if the
//! ABSOLUTE probe score didn't drop — too noisy to move a weak model. This is the
//! complementary, more permissive gate: sample each probe prompt on the model
//! BEFORE the step (base) and AFTER (base + candidate adapter), and ask an LLM
//! judge whether the AFTER answer **degraded**. The step is accepted UNLESS
//! degradation is detected — flipping "prove it improved" into "prove it didn't
//! get worse", which is achievable on tiny QA-pair counts. See `src/AGENTS.md`
//! §degrade.rs.
//!
//! Mirrors [`crate::ingest::LlmRelevanceJudge`]: generic over [`ChatTransport`]
//! (so it's unit-tested with a mock, ML-free), batches items, parses a JSON array
//! of the item numbers that got WORSE, and **errs toward `same-or-better`**
//! (accept) on a judge failure/garble — a flaky judge must never stall progress
//! (the track-15 catastrophe floor is the backstop, and `doctor`'s track-31 judge
//! preflight detects a down/missing judge model).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::generate::api::{ChatMessage, ChatTransport};

/// One A/B sample: the probe prompt and the BEFORE (base) / AFTER (base+adapter)
/// completions the judge compares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradationTriple {
    pub prompt: String,
    pub before: String,
    pub after: String,
}

/// The outcome of judging a batch of A/B samples.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DegradationReport {
    /// Total triples judged.
    pub n: usize,
    /// How many AFTER answers the judge rated WORSE than BEFORE.
    pub regressed: usize,
    /// Per-item: `true` ⇒ this item regressed (worse AFTER).
    pub worse: Vec<bool>,
}

impl DegradationReport {
    /// Fraction of judged items that regressed (`0.0` when nothing was judged —
    /// "no evidence of degradation", the permissive default).
    pub fn regressed_fraction(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.regressed as f64 / self.n as f64
        }
    }
}

/// Judge whether each AFTER completion degraded vs its BEFORE. LLM in production
/// ([`LlmDegradationJudge`]); mock in tests.
pub trait DegradationJudge {
    /// Score each triple: returns a report marking which AFTER answers got worse.
    fn judge(&self, triples: &[DegradationTriple]) -> anyhow::Result<DegradationReport>;
}

/// LLM degradation judge over any [`ChatTransport`]. Batches, parses a JSON array
/// of the 1-based item numbers that got WORSE, and errs toward `same-or-better`
/// (not-worse) on a failed/garbled batch.
pub struct LlmDegradationJudge<T: ChatTransport> {
    transport: T,
    batch: usize,
}

impl<T: ChatTransport> LlmDegradationJudge<T> {
    /// Build a judge with the given transport and batch size (clamped to ≥ 1).
    pub fn new(transport: T, batch: usize) -> Self {
        Self {
            transport,
            batch: batch.max(1),
        }
    }
}

impl<T: ChatTransport> DegradationJudge for LlmDegradationJudge<T> {
    fn judge(&self, triples: &[DegradationTriple]) -> anyhow::Result<DegradationReport> {
        let mut worse = Vec::with_capacity(triples.len());
        for chunk in triples.chunks(self.batch) {
            let messages = [
                ChatMessage::system(
                    "You compare two answers (BEFORE and AFTER) to the same request \
                     and decide ONLY whether AFTER is WORSE than BEFORE (a regression: \
                     less correct, less relevant, malformed, or empty when BEFORE was \
                     fine). If AFTER is the same or better, it is NOT worse. Reply with \
                     ONLY a JSON array of the item numbers whose AFTER got WORSE (e.g. \
                     [2,5]); reply [] if none got worse. No prose.",
                ),
                ChatMessage::user(build_degrade_prompt(chunk)),
            ];
            match self.transport.complete(&messages) {
                Ok(answer) => {
                    let worse_idx = parse_worse_indices(&answer, chunk.len());
                    for i in 0..chunk.len() {
                        worse.push(worse_idx.contains(&i));
                    }
                }
                Err(e) => {
                    // Err toward NOT-worse: a flaky judge must not block progress.
                    eprintln!("degrade judge: batch failed, treating as not-worse ({e})");
                    worse.extend(std::iter::repeat(false).take(chunk.len()));
                }
            }
        }
        let regressed = worse.iter().filter(|w| **w).count();
        Ok(DegradationReport {
            n: worse.len(),
            regressed,
            worse,
        })
    }
}

/// Render the judge prompt: a 1-based numbered list of (request, BEFORE, AFTER).
fn build_degrade_prompt(triples: &[DegradationTriple]) -> String {
    let mut s = String::from(
        "For each item, is the AFTER answer WORSE than the BEFORE answer for that \
         request?\n\n",
    );
    for (i, t) in triples.iter().enumerate() {
        s.push_str(&format!(
            "{}. REQUEST: {}\n   BEFORE: {}\n   AFTER: {}\n",
            i + 1,
            truncate(&t.prompt, 200),
            truncate(&t.before, 300),
            truncate(&t.after, 300),
        ));
    }
    s.push_str("\nReply with a JSON array of the item numbers whose AFTER got worse.");
    s
}

/// 0-based indices of items the judge marked WORSE: a JSON array of 1-based
/// numbers anywhere in the reply (prose/fences tolerated); out-of-range ignored.
/// Identical shape to the relevance judge's parser.
fn parse_worse_indices(answer: &str, n: usize) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let (Some(start), Some(end)) = (answer.find('['), answer.rfind(']')) else {
        return out;
    };
    if end <= start {
        return out;
    }
    if let Ok(serde_json::Value::Array(items)) =
        serde_json::from_str::<serde_json::Value>(&answer[start..=end])
    {
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

/// Char-safe truncation to at most `max` chars, trimmed.
fn truncate(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    t.chars().take(max).collect::<String>().trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTransport {
        reply: String,
    }
    impl ChatTransport for MockTransport {
        fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(self.reply.clone())
        }
    }

    struct FailTransport;
    impl ChatTransport for FailTransport {
        fn complete(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            anyhow::bail!("endpoint down")
        }
    }

    fn triple(p: &str, b: &str, a: &str) -> DegradationTriple {
        DegradationTriple {
            prompt: p.into(),
            before: b.into(),
            after: a.into(),
        }
    }

    #[test]
    fn flags_the_worse_item() {
        // Judge says item 2 got worse (1-based) → index 1.
        let judge = LlmDegradationJudge::new(
            MockTransport {
                reply: "[2]".into(),
            },
            10,
        );
        let report = judge
            .judge(&[
                triple("q1", "good", "good"),
                triple("q2", "good", "garbage"),
            ])
            .unwrap();
        assert_eq!(report.n, 2);
        assert_eq!(report.regressed, 1);
        assert_eq!(report.worse, vec![false, true]);
        assert!((report.regressed_fraction() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_array_means_nothing_regressed() {
        let judge = LlmDegradationJudge::new(MockTransport { reply: "[]".into() }, 10);
        let report = judge
            .judge(&[triple("q", "a", "a"), triple("q2", "b", "b2")])
            .unwrap();
        assert_eq!(report.regressed, 0);
        assert_eq!(report.regressed_fraction(), 0.0);
    }

    #[test]
    fn parses_array_amid_prose() {
        let judge = LlmDegradationJudge::new(
            MockTransport {
                reply: "The worse ones are [1, 3].".into(),
            },
            10,
        );
        let report = judge
            .judge(&[
                triple("a", "x", "y"),
                triple("b", "x", "x"),
                triple("c", "x", "z"),
            ])
            .unwrap();
        assert_eq!(report.worse, vec![true, false, true]);
    }

    #[test]
    fn judge_failure_errs_toward_not_worse() {
        // A down judge must NOT block progress: every item is treated as not-worse.
        let judge = LlmDegradationJudge::new(FailTransport, 10);
        let report = judge
            .judge(&[triple("a", "x", "y"), triple("b", "x", "y")])
            .unwrap();
        assert_eq!(report.regressed, 0, "flaky judge => accept (not-worse)");
        assert_eq!(report.worse, vec![false, false]);
    }

    #[test]
    fn empty_input_is_zero_regressed() {
        let judge = LlmDegradationJudge::new(
            MockTransport {
                reply: "[1]".into(),
            },
            10,
        );
        let report = judge.judge(&[]).unwrap();
        assert_eq!(report.n, 0);
        assert_eq!(report.regressed_fraction(), 0.0);
    }

    #[test]
    fn batching_spans_chunks() {
        // batch=1 forces 3 separate calls; the mock says item "1" worse each call,
        // so every item is flagged worse.
        let judge = LlmDegradationJudge::new(
            MockTransport {
                reply: "[1]".into(),
            },
            1,
        );
        let report = judge
            .judge(&[
                triple("a", "x", "y"),
                triple("b", "x", "y"),
                triple("c", "x", "y"),
            ])
            .unwrap();
        assert_eq!(report.worse, vec![true, true, true]);
    }
}
