//! The evolution interview — grill the human on training direction.
//!
//! Two layers (per the design choice "fixed core + LLM follow-ups"):
//! 1. A **fixed core** question set, always asked, mapping onto the structured
//!    [`TrainingDirective`] fields.
//! 2. **LLM-generated follow-up** questions, produced from the corpus/palace
//!    signals — corpus-specific clarifications the model wants answered,
//!    including at least one grounded in the mind-palace (mpg) state.
//!
//! The SDK is headless: it *produces* the questions and *assembles* a directive
//! from answers. The CLI does the actual interactive asking (terminal stdin) or
//! accepts answers via flags/file for headless use.

use serde::{Deserialize, Serialize};

use crate::config::EvolveConfig;
use crate::discover::DiscoveredContext;
use crate::generate::api::{ChatMessage, ChatTransport, HttpTransport};
use crate::plan::signals::{self, Signals};

/// One interview question. `field` names the directive field the answer maps to
/// (or `notes` for free-form / follow-up answers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// Stable id (e.g. "goal", "priorities", "followup_1").
    #[serde(default)]
    pub id: String,
    /// The question text shown to the human.
    pub text: String,
    /// Which directive field this maps onto: goal | priorities | must_cover |
    /// audience | difficulty | tone | exclusions | max_rows | notes.
    pub field: String,
    /// Suggested options (for choice-style questions); empty = free text.
    #[serde(default)]
    pub options: Vec<String>,
    /// True if multiple options may be selected.
    #[serde(default)]
    pub multi: bool,
}

/// The fixed-core interview — always asked. Maps onto the structured directive
/// fields. The CLI renders these; selecting/typing produces answers.
pub fn core_questions() -> Vec<Question> {
    vec![
        Question {
            id: "goal".into(),
            text: "What is the primary training goal — what should the model get better at?".into(),
            field: "goal".into(),
            options: vec![],
            multi: false,
        },
        Question {
            id: "priorities".into(),
            text: "Which output modalities matter most? (rank by selecting in priority order)".into(),
            field: "priorities".into(),
            options: vec![
                "tool_call".into(),
                "cli".into(),
                "qa".into(),
                "instruction".into(),
            ],
            multi: true,
        },
        Question {
            id: "must_cover".into(),
            text: "Which tools or workflows MUST be well-covered (even if rare in the corpus)? (comma-separated)".into(),
            field: "must_cover".into(),
            options: vec![],
            multi: false,
        },
        Question {
            id: "audience".into(),
            text: "Who is the model for, and at what level? (e.g. 'power users, advanced flag mastery')".into(),
            field: "audience".into(),
            options: vec![],
            multi: false,
        },
        Question {
            id: "tone".into(),
            text: "What answer style/tone? (e.g. 'terse, command-first')".into(),
            field: "tone".into(),
            options: vec![],
            multi: false,
        },
        Question {
            id: "exclusions".into(),
            text: "Any hard exclusions — what should NEVER be generated? (comma-separated, e.g. 'destructive commands, prose trivia')".into(),
            field: "exclusions".into(),
            options: vec![],
            multi: false,
        },
        Question {
            id: "max_rows".into(),
            text: "Cap on total dataset rows? (0 = no cap)".into(),
            field: "max_rows".into(),
            options: vec![],
            multi: false,
        },
    ]
}

/// Build the LLM prompt that generates corpus/palace-specific follow-up
/// questions. The prompt explicitly demands at least one question grounded in
/// the mind-palace (mpg) state.
fn followup_system_prompt() -> &'static str {
    "You design a SHORT clarifying interview to steer training-data generation \
for the `scrt` tool. Given usage signals (palace/mpg state, tool/flag usage, \
corpus shape), produce 2-3 SPECIFIC follow-up questions whose answers would \
materially change what data to generate. At least ONE question MUST be grounded \
in the mind-palace (mpg) state (its stashes, tags, links, or emptiness).\n\n\
Output ONLY a JSON array of objects: \
[{\"id\":\"followup_1\",\"text\":\"...\",\"field\":\"notes\",\"options\":[],\"multi\":false}]\n\
Rules: questions must be answerable in one short sentence; field is always \
\"notes\" (their answers become directive notes); no format/prose preamble."
}

fn followup_user_prompt(signals: &Signals) -> String {
    let mpg = if signals.palace.stash_count == 0 {
        "The mind-palace is EMPTY (no stashes). Consider asking whether the human \
intends to seed it, or whether discovery should rely on the corpus only."
            .to_string()
    } else {
        format!(
            "The mind-palace has {} stashes, {} links. Top tags drive topic density.",
            signals.palace.stash_count, signals.palace.total_links
        )
    };
    format!(
        "USAGE SIGNALS:\n{}\n\nMPG STATE NOTE: {}\n\nGenerate the follow-up \
interview questions now (JSON array only).",
        signals::summary(signals),
        mpg
    )
}

/// Parse the LLM's follow-up questions array, dropping malformed entries.
pub fn parse_followups(raw: &str) -> Vec<Question> {
    let json = {
        let t = raw.trim();
        match (t.find('['), t.rfind(']')) {
            (Some(a), Some(b)) if a <= b => &t[a..=b],
            _ => t,
        }
    };
    let values: Vec<serde_json::Value> = serde_json::from_str(json).unwrap_or_default();
    let mut out = Vec::new();
    for (i, v) in values.into_iter().enumerate() {
        if let Ok(mut q) = serde_json::from_value::<Question>(v) {
            if q.text.trim().is_empty() {
                continue;
            }
            if q.id.trim().is_empty() {
                q.id = format!("followup_{}", i + 1);
            }
            // Follow-up answers always land in notes.
            q.field = "notes".into();
            out.push(q);
        }
    }
    out
}

/// Generate follow-up questions from signals via a chat transport (mockable).
pub fn followups_with_transport<T: ChatTransport>(
    transport: &T,
    signals: &Signals,
) -> anyhow::Result<Vec<Question>> {
    let messages = vec![
        ChatMessage::system(followup_system_prompt().to_string()),
        ChatMessage::user(followup_user_prompt(signals)),
    ];
    let raw = transport.complete(&messages)?;
    Ok(parse_followups(&raw))
}

/// Build the full interview (core + signal-derived follow-ups) for a config +
/// discovered context, using the configured API backend for the follow-ups.
/// Follow-up generation failure is non-fatal — the core questions still stand.
pub fn build(cfg: &EvolveConfig, ctx: &DiscoveredContext) -> Vec<Question> {
    let mut qs = core_questions();
    let signals = signals::extract(cfg, ctx);
    let gcfg = cfg.generate.clone().unwrap_or_default();
    if let Ok(transport) = HttpTransport::from_api_config(&gcfg) {
        if let Ok(followups) = followups_with_transport(&transport, &signals) {
            qs.extend(followups);
        }
    }
    qs
}

/// Assemble a [`TrainingDirective`](crate::directive::TrainingDirective) from
/// (question, answer) pairs. Answers for list fields are split on commas;
/// `max_rows` parses as a number; unknown/`notes` answers accumulate into notes.
pub fn assemble_directive(answers: &[(Question, String)]) -> crate::directive::TrainingDirective {
    use crate::directive::TrainingDirective;
    let mut d = TrainingDirective::default();
    let mut notes = Vec::new();
    for (q, ans) in answers {
        let a = ans.trim();
        if a.is_empty() {
            continue;
        }
        match q.field.as_str() {
            "goal" => d.goal = a.to_string(),
            "audience" => d.audience = a.to_string(),
            "difficulty" => d.difficulty = a.to_string(),
            "tone" => d.tone = a.to_string(),
            "priorities" => d.priorities = split_list(a),
            "must_cover" => d.must_cover = split_list(a),
            "exclusions" => d.exclusions = split_list(a),
            "max_rows" => d.max_rows = a.parse().unwrap_or(0),
            _ => notes.push(format!("Q[{}]: {} → {}", q.id, q.text, a)),
        }
    }
    if !notes.is_empty() {
        d.notes = notes.join("\n");
    }
    d
}

fn split_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}
