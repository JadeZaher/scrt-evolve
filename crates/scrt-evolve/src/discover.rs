//! Discover stage — corpus + palace → [`DiscoveredContext`].
//!
//! Consumes **scrt-core** in-process (no subprocess, no PyO3): walks palace
//! stashes and/or sweeps the corpus as seed queries, scrt-searches the corpus
//! via [`scrt_core::search_with_meta`], dedups near-duplicate passages with
//! `palace::simhash`, ranks, optionally clusters across topics, and caps by
//! `max_passages`. The result is written to `work_dir/discovered.json` and is
//! deterministic (no unseeded RNG) so it is diffable.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use scrt_core::palace::{ops::SystemClock, simhash, FilePalace, Palace};
use scrt_core::types::{Effort, Node, SearchOptions, SortMode, Strategy, WindowCurve};
use scrt_core::{search_with_meta, SearchConfig, SourceInput};

use crate::config::{DiscoverConfig, EvolveConfig};

/// One retrieved context chunk worth generating data about, with provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub text: String,
    pub source: String,
    pub score: f32,
    /// Which seed (a stash name, or `corpus:<pattern>`) surfaced this passage.
    /// Lets generation attribute coverage and lets discovery cluster by seed.
    #[serde(default)]
    pub seed: String,
}

/// A reference to a palace stash that seeded/anchored discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StashRef {
    pub name: String,
}

/// The discover stage's durable output (written to `work_dir/discovered.json`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveredContext {
    pub passages: Vec<Passage>,
    pub anchors: Vec<StashRef>,
}

/// A seed query: a pattern to search the corpus with, tagged by its origin.
struct Seed {
    /// Origin label: a stash name, or `corpus:<pattern>` for corpus sweeps.
    origin: String,
    /// The regex/literal handed to scrt-search.
    pattern: String,
}

/// Run discovery against the corpus + palace per the config's `[discover]`
/// block. Requires `corpus_dir` to be set; `palace_path` is needed only when
/// `seed` includes `palace`.
pub fn run(cfg: &EvolveConfig) -> anyhow::Result<DiscoveredContext> {
    let dcfg = cfg.discover.clone().unwrap_or_default();
    let corpus_dir = cfg
        .evolve
        .corpus_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("discover: `[evolve].corpus_dir` is required"))?;
    if !corpus_dir.exists() {
        anyhow::bail!("discover: corpus_dir does not exist: {}", corpus_dir.display());
    }

    // Load the palace if discovery seeds from it.
    let seeds_from_palace = matches!(dcfg.seed.as_str(), "palace" | "both");
    let palace = if seeds_from_palace {
        let path = cfg.evolve.palace_path.clone().ok_or_else(|| {
            anyhow::anyhow!("discover: seed=\"{}\" needs `[evolve].palace_path`", dcfg.seed)
        })?;
        Some(FilePalace::load(&path, &SystemClock))
    } else {
        None
    };

    let anchors: Vec<StashRef> = palace
        .as_ref()
        .map(|p| {
            p.data()
                .stashes
                .keys()
                .map(|name| StashRef { name: name.clone() })
                .collect()
        })
        .unwrap_or_default();

    let seeds = build_seeds(&dcfg, palace.as_ref());
    if seeds.is_empty() {
        anyhow::bail!(
            "discover: no seed queries produced (seed=\"{}\"). For seed=palace the \
             palace must have stashes with notes/patterns; for seed=corpus set \
             `[discover].corpus_patterns` or rely on the defaults.",
            dcfg.seed
        );
    }

    let corpus_input = SourceInput::Path(corpus_dir.to_string_lossy().into_owned());

    // Collect raw passages across all seeds, paired with their match-line core
    // (used as the dedup key).
    let mut raw: Vec<(Passage, String)> = Vec::new();
    for seed in &seeds {
        let config = search_config(&seed.pattern, corpus_input.clone(), &dcfg);
        // A search miss for one seed is not fatal — other seeds may hit.
        let (result, _meta) = match search_with_meta(&config) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for node in &result.nodes {
            raw.push(passage_from_node(node, &seed.origin));
        }
    }

    // Dedup near-duplicate passages via simhash, rank, optionally spread across
    // seeds (clusters), then cap.
    let deduped = dedup_passages(raw);
    let ordered = if dcfg.cluster {
        cluster_round_robin(deduped)
    } else {
        rank_by_score(deduped)
    };
    let passages = ordered.into_iter().take(dcfg.max_passages).collect();

    Ok(DiscoveredContext { passages, anchors })
}

/// Build the seed queries from config + palace.
fn build_seeds(dcfg: &DiscoverConfig, palace: Option<&FilePalace>) -> Vec<Seed> {
    let mut seeds = Vec::new();

    if matches!(dcfg.seed.as_str(), "palace" | "both") {
        if let Some(p) = palace {
            for (name, stash) in &p.data().stashes {
                // Prefer the stash's own search pattern; fall back to the note.
                let pat = if !stash.search.pattern.trim().is_empty() {
                    stash.search.pattern.clone()
                } else {
                    note_to_pattern(&stash.note)
                };
                if !pat.trim().is_empty() {
                    seeds.push(Seed {
                        origin: name.clone(),
                        pattern: pat,
                    });
                }
            }
        }
    }

    if matches!(dcfg.seed.as_str(), "corpus" | "both") {
        for pat in &dcfg.corpus_patterns {
            seeds.push(Seed {
                origin: format!("corpus:{pat}"),
                pattern: pat.clone(),
            });
        }
    }

    seeds
}

/// Turn a free-text stash note into an alternation regex over its salient
/// words (deterministic: lowercased, deduped, length-filtered, sorted).
fn note_to_pattern(note: &str) -> String {
    let mut words: Vec<String> = note
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(|w| regex_escape(&w.to_lowercase()))
        .collect();
    words.sort();
    words.dedup();
    words.join("|")
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\.+*?()|[]{}^$".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// A search config tuned for discovery: scan effort (broad, small windows),
/// deterministic default sort, capped node count per seed.
fn search_config(pattern: &str, input: SourceInput, dcfg: &DiscoverConfig) -> SearchConfig {
    let mut cfg = SearchConfig::from_effort(pattern, vec![input], Effort::Normal);
    cfg.strategy = Strategy::Fill;
    cfg.sort = SortMode::Default; // deterministic
    cfg.window_curve = WindowCurve::Flat;
    // Per-seed node cap: keep generous but bounded; the global cap is applied
    // after dedup across all seeds.
    cfg.max_nodes = dcfg.max_passages.max(10);
    cfg.rg_options = SearchOptions {
        case_insensitive: true,
        word_match: false,
        fixed_strings: false,
        multiline: false,
        hidden: false,
        no_ignore: false,
        include_globs: vec![],
        // Don't mine our own work-dir artifacts back into the corpus.
        exclude_globs: vec!["**/.scrt-evolve/**".to_string()],
        type_filter: None,
        glob_case_insensitive: false,
        max_columns: None,
    };
    cfg
}

/// Build a `Passage` from a scrt-core `Node`, stitching the context window into
/// the passage text and keeping the source path + a score for ranking. The
/// match line is carried separately so dedup can key on the matched content
/// (the "what" the passage is about) rather than the surrounding context.
fn passage_from_node(node: &Node, origin: &str) -> (Passage, String) {
    let mut text = String::new();
    for line in &node.context_before {
        text.push_str(line);
        text.push('\n');
    }
    text.push_str(&node.match_text);
    text.push('\n');
    for line in &node.context_after {
        text.push_str(line);
        text.push('\n');
    }
    // Score: more context tokens ⇒ richer passage. Stable, no RNG.
    let score = node.tokens as f32;
    let passage = Passage {
        text: text.trim_end().to_string(),
        source: node.source.id.clone(),
        score,
        seed: origin.to_string(),
    };
    (passage, node.match_text.clone())
}

/// Drop near-duplicate passages using simhash over the **match-line core** (the
/// matched content, not the surrounding context). Two passages whose match
/// lines simhash within a small Hamming distance are duplicates — the same
/// content surfaced from different files. The first (after a stable sort) wins.
/// Deterministic.
fn dedup_passages(mut passages: Vec<(Passage, String)>) -> Vec<Passage> {
    // Stable order before dedup: by source then by text, so the survivor is
    // deterministic regardless of seed iteration order.
    passages.sort_by(|(a, _), (b, _)| {
        a.source.cmp(&b.source).then_with(|| a.text.cmp(&b.text))
    });

    const HAMMING_DUP_THRESHOLD: u32 = 3; // ≤3 bits differ ⇒ near-identical

    let mut kept: Vec<(u64, Passage)> = Vec::new();
    for (p, match_core) in passages {
        let features = simhash_features(match_core.trim());
        let h = simhash::simhash(&features);
        let is_dup = kept
            .iter()
            .any(|(kh, _)| simhash::hamming(*kh, h) <= HAMMING_DUP_THRESHOLD);
        if !is_dup {
            kept.push((h, p));
        }
    }
    kept.into_iter().map(|(_, p)| p).collect()
}

/// Tokenize text into simhash features (lowercased word shingles). Mirrors the
/// spirit of scrt-core's prose projection without needing a full Stash.
fn simhash_features(text: &str) -> Vec<String> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    if words.len() < 2 {
        return words;
    }
    // 2-shingles capture local structure better than bag-of-words.
    words.windows(2).map(|w| w.join(" ")).collect()
}

/// Rank passages by descending score, deterministic tie-break by source/text.
fn rank_by_score(mut passages: Vec<Passage>) -> Vec<Passage> {
    passages.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.text.cmp(&b.text))
    });
    passages
}

/// Spread coverage across seeds: round-robin the per-seed score-ranked lists so
/// the output doesn't re-mine one topic before touching others. Deterministic.
fn cluster_round_robin(passages: Vec<Passage>) -> Vec<Passage> {
    // Group by seed (BTreeSet of seed names gives deterministic group order).
    let seed_names: BTreeSet<String> = passages.iter().map(|p| p.seed.clone()).collect();
    let mut groups: Vec<Vec<Passage>> = seed_names
        .iter()
        .map(|name| {
            let mut g: Vec<Passage> =
                passages.iter().filter(|p| &p.seed == name).cloned().collect();
            g = rank_by_score(g);
            g
        })
        .collect();

    let mut out = Vec::with_capacity(passages.len());
    let mut idx = 0;
    loop {
        let mut progressed = false;
        for g in groups.iter_mut() {
            if let Some(p) = g.get(idx).cloned() {
                out.push(p);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
        idx += 1;
    }
    out
}
