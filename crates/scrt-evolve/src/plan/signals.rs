//! Usage-signal extraction (deterministic — no LLM).
//!
//! Three signals feed the planner's decision about what to generate:
//! 1. **Palace structure** — stash density, tags, relations/links, and
//!    simhash clusters (topics with dense, linked stashes are high-value).
//! 2. **Tool/flag co-occurrence** — which `scrt_*` tools and `--mp-*` flags
//!    appear near each other in the corpus (real workflows, not rare flags).
//! 3. **Corpus shape** — per-passage content classification (code / CLI-ref /
//!    conceptual / config) so modality routing can match content type.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use scrt_core::palace::{ops::SystemClock, FilePalace, Palace};

use crate::config::{DomainConfig, EvolveConfig};
use crate::discover::DiscoveredContext;

/// The full signal summary handed to the planner.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Signals {
    pub palace: PalaceSignal,
    pub cooccurrence: CooccurrenceSignal,
    pub corpus_shape: CorpusShapeSignal,
}

/// Palace-structure signal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PalaceSignal {
    pub stash_count: usize,
    /// Per-stash: name, note, tag count, relation/link count, node count.
    pub stashes: Vec<StashStat>,
    /// Tag frequency across the palace (high-frequency tags = dense topics).
    pub tag_frequency: BTreeMap<String, usize>,
    pub total_links: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StashStat {
    pub name: String,
    pub note: String,
    pub tags: Vec<String>,
    pub links: usize,
    pub nodes: usize,
}

/// Tool/flag co-occurrence signal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CooccurrenceSignal {
    /// How often each scrt tool name appears in the corpus passages.
    pub tool_frequency: BTreeMap<String, usize>,
    /// How often each `--mp-*` / `--flag` appears.
    pub flag_frequency: BTreeMap<String, usize>,
    /// Tool/flag pairs seen in the same passage (workflow signal). Key is
    /// "a+b" (sorted), value is co-occurrence count.
    pub pairs: BTreeMap<String, usize>,
}

/// Corpus-shape signal: how many passages of each content shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusShapeSignal {
    /// shape -> count (code | cli_ref | conceptual | config).
    pub shape_counts: BTreeMap<String, usize>,
    /// Per-passage shape, index-aligned with the discovered passages.
    pub per_passage: Vec<String>,
}

/// Extract all signals from the config + discovered context. The `[domain]`
/// block (track 37 Phase C) drives which tool names + flag prefixes count as
/// co-occurrence signal; absent ⇒ [`DomainConfig::default`] ⇒ the built-in
/// `scrt` tool set + `--mp-` flags (behavior-identical to before).
pub fn extract(cfg: &EvolveConfig, ctx: &DiscoveredContext) -> Signals {
    let domain = cfg.domain.clone().unwrap_or_default();
    Signals {
        palace: extract_palace(cfg),
        cooccurrence: extract_cooccurrence(ctx, &domain),
        corpus_shape: extract_corpus_shape(ctx),
    }
}

/// Palace structure: density, tags, links. Empty if no palace configured.
fn extract_palace(cfg: &EvolveConfig) -> PalaceSignal {
    let Some(path) = &cfg.evolve.palace_path else {
        return PalaceSignal::default();
    };
    if !path.exists() {
        return PalaceSignal::default();
    }
    let palace = FilePalace::load(path, &SystemClock);
    let data = palace.data();

    let mut stashes = Vec::new();
    let mut tag_frequency: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_links = 0;

    for (name, stash) in &data.stashes {
        for t in &stash.tags {
            *tag_frequency.entry(t.clone()).or_default() += 1;
        }
        total_links += stash.relations.len();
        stashes.push(StashStat {
            name: name.clone(),
            note: stash.note.clone(),
            tags: stash.tags.clone(),
            links: stash.relations.len(),
            nodes: stash.nodes.len(),
        });
    }

    PalaceSignal {
        stash_count: stashes.len(),
        stashes,
        tag_frequency,
        total_links,
    }
}

/// Tool/flag co-occurrence across passages. `domain` supplies the tool names
/// counted and the flag prefixes recognized (track 37 Phase C). With
/// [`DomainConfig::default`] this is the built-in `scrt` tool set + the generic
/// `--` flag recognizer — behavior-identical to before parameterization.
fn extract_cooccurrence(ctx: &DiscoveredContext, domain: &DomainConfig) -> CooccurrenceSignal {
    let mut tool_frequency: BTreeMap<String, usize> = BTreeMap::new();
    let mut flag_frequency: BTreeMap<String, usize> = BTreeMap::new();
    let mut pairs: BTreeMap<String, usize> = BTreeMap::new();

    for p in &ctx.passages {
        let text = &p.text;

        // Tools present in this passage (domain-configured tool names).
        let mut present: Vec<String> = Vec::new();
        for t in &domain.tools {
            let n = text.matches(t.as_str()).count();
            if n > 0 {
                *tool_frequency.entry(t.clone()).or_default() += n;
                present.push(t.clone());
            }
        }

        // Flags present (`--mp-foo` / `--foo` style tokens, plus any domain flag
        // prefixes). The generic `--` recognizer is always on (back-compat);
        // `domain.flag_patterns` UNION-adds any extra prefixes to accept.
        let mut flags_here: Vec<String> = Vec::new();
        for tok in text.split(|c: char| c.is_whitespace() || c == '`' || c == '"') {
            if let Some(flag) = parse_flag(tok, &domain.flag_patterns) {
                *flag_frequency.entry(flag.clone()).or_default() += 1;
                if !flags_here.contains(&flag) {
                    flags_here.push(flag);
                }
            }
        }

        // Co-occurrence: all tool×tool and tool×flag pairs in this passage.
        let mut terms = present.clone();
        terms.extend(flags_here.iter().cloned());
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                let (a, b) = (&terms[i], &terms[j]);
                let key = if a <= b {
                    format!("{a}+{b}")
                } else {
                    format!("{b}+{a}")
                };
                *pairs.entry(key).or_default() += 1;
            }
        }
    }

    CooccurrenceSignal {
        tool_frequency,
        flag_frequency,
        pairs,
    }
}

/// Recognize a CLI flag token (`--mp-stash`, `--effort`), trimming trailing
/// punctuation. Returns the normalized flag or `None`.
///
/// The generic `--<name>` (len>3) recognizer is always on for back-compat; any
/// prefix in `flag_patterns` is UNION-added as an additional accept rule, so a
/// custom domain can count non-`--` flags (e.g. `-v`, `/flag`) without changing
/// the default (`--mp-`) behavior, which already matches under the generic rule.
fn parse_flag(tok: &str, flag_patterns: &[String]) -> Option<String> {
    let t = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '/');
    if t.starts_with("--") && t.len() > 3 {
        return Some(t.to_string());
    }
    for pat in flag_patterns {
        if !pat.is_empty() && t.starts_with(pat.as_str()) && t.len() > pat.len() {
            return Some(t.to_string());
        }
    }
    None
}

/// Classify each passage's content shape.
fn extract_corpus_shape(ctx: &DiscoveredContext) -> CorpusShapeSignal {
    let mut shape_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_passage = Vec::with_capacity(ctx.passages.len());
    for p in &ctx.passages {
        let shape = classify_shape(&p.text, &p.source);
        *shape_counts.entry(shape.clone()).or_default() += 1;
        per_passage.push(shape);
    }
    CorpusShapeSignal {
        shape_counts,
        per_passage,
    }
}

/// Heuristic content-shape classifier. Deterministic.
pub fn classify_shape(text: &str, source: &str) -> String {
    let lower = source.to_lowercase();
    // Config files.
    if lower.ends_with(".toml")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".json")
    {
        return "config".to_string();
    }
    // CLI reference: shows command/flag usage.
    let cli_markers = text.matches("--").count() + text.matches("scrt ").count();
    let code_markers = text.matches("fn ").count()
        + text.matches("pub ").count()
        + text.matches("struct ").count()
        + text.matches("impl ").count()
        + text.matches("=>").count();
    let is_code_file = lower.ends_with(".rs")
        || lower.ends_with(".ts")
        || lower.ends_with(".js")
        || lower.ends_with(".py");

    if cli_markers >= 2 && cli_markers >= code_markers {
        return "cli_ref".to_string();
    }
    if is_code_file || code_markers >= 2 {
        return "code".to_string();
    }
    "conceptual".to_string()
}

/// Render the signals as a compact, model-readable summary for the planner.
/// Caps lists so the prompt stays within context.
pub fn summary(signals: &Signals) -> String {
    let mut s = String::new();

    s.push_str("## Palace structure\n");
    s.push_str(&format!(
        "stashes: {}, total links: {}\n",
        signals.palace.stash_count, signals.palace.total_links
    ));
    if !signals.palace.tag_frequency.is_empty() {
        let mut tags: Vec<_> = signals.palace.tag_frequency.iter().collect();
        tags.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> = tags
            .iter()
            .take(12)
            .map(|(k, v)| format!("{k}({v})"))
            .collect();
        s.push_str(&format!("top tags: {}\n", top.join(", ")));
    }
    for st in signals.palace.stashes.iter().take(15) {
        s.push_str(&format!(
            "- {} [{} tags, {} links, {} nodes]: {}\n",
            st.name,
            st.tags.len(),
            st.links,
            st.nodes,
            st.note.chars().take(80).collect::<String>()
        ));
    }

    s.push_str("\n## Tool/flag usage (corpus co-occurrence)\n");
    let mut tools: Vec<_> = signals.cooccurrence.tool_frequency.iter().collect();
    tools.sort_by(|a, b| b.1.cmp(a.1));
    s.push_str(&format!(
        "tools: {}\n",
        tools
            .iter()
            .map(|(k, v)| format!("{k}({v})"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let mut flags: Vec<_> = signals.cooccurrence.flag_frequency.iter().collect();
    flags.sort_by(|a, b| b.1.cmp(a.1));
    let topflags: Vec<String> = flags
        .iter()
        .take(20)
        .map(|(k, v)| format!("{k}({v})"))
        .collect();
    s.push_str(&format!("top flags: {}\n", topflags.join(", ")));
    let mut pairs: Vec<_> = signals.cooccurrence.pairs.iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(a.1));
    let toppairs: Vec<String> = pairs
        .iter()
        .take(15)
        .map(|(k, v)| format!("{k}({v})"))
        .collect();
    s.push_str(&format!(
        "top co-occurring workflows: {}\n",
        toppairs.join(", ")
    ));

    s.push_str("\n## Corpus shape\n");
    let mut shapes: Vec<_> = signals.corpus_shape.shape_counts.iter().collect();
    shapes.sort_by(|a, b| b.1.cmp(a.1));
    s.push_str(&format!(
        "passage shapes: {}\n",
        shapes
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    s
}
