//! Discover stage tests against a fixture corpus on disk. No ML, no network.

use scrt_evolve::discover;
use scrt_evolve::EvolveConfig;

/// Create a fixture corpus dir with a couple of files, including a deliberate
/// near-duplicate line so dedup has something to collapse.
fn fixture_corpus(suffix: &str) -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "scrt-evolve-corpus-{}-{suffix}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(
        dir.join("memory.md"),
        "# Memory traversal\n\
         The mp-stash command stores a named stash.\n\
         The mp-compose command unions two stashes.\n\
         The mp-graph command reconstructs the investigation topology.\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("dup.md"),
        "# Other doc\n\
         The mp-stash command stores a named stash.\n",
    )
    .unwrap();
    dir
}

fn config_for(corpus: &std::path::Path, max_passages: usize, cluster: bool) -> EvolveConfig {
    config_for_patterns(
        corpus,
        max_passages,
        cluster,
        &["mp-stash", "mp-compose", "mp-graph"],
    )
}

fn config_for_patterns(
    corpus: &std::path::Path,
    max_passages: usize,
    cluster: bool,
    patterns: &[&str],
) -> EvolveConfig {
    let pats = patterns
        .iter()
        .map(|p| format!("{p:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let toml = format!(
        r#"
[evolve]
corpus_dir = {corpus:?}

[discover]
seed = "corpus"
max_passages = {max_passages}
cluster = {cluster}
corpus_patterns = [{pats}]
"#,
    );
    EvolveConfig::from_toml_str(&toml).unwrap()
}

#[test]
fn discovers_passages_with_provenance() {
    let corpus = fixture_corpus("basic");
    let cfg = config_for(&corpus, 100, false);
    let ctx = discover::run(&cfg).expect("discover runs");

    assert!(!ctx.passages.is_empty(), "should find passages");
    // Provenance points at real corpus files.
    for p in &ctx.passages {
        assert!(
            p.source.contains("memory.md") || p.source.contains("dup.md"),
            "source should be a real corpus file, got {}",
            p.source
        );
    }
    let _ = std::fs::remove_dir_all(&corpus);
}

#[test]
fn near_duplicate_passages_collapse() {
    // Two files whose ONLY content is the same matched line. Searched by a
    // single pattern, they produce two passages with an identical match core,
    // which dedup must collapse to one.
    let mut dir = std::env::temp_dir();
    dir.push(format!("scrt-evolve-dedup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("a.md"),
        "The mp-stash command stores a named stash.\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("b.md"),
        "The mp-stash command stores a named stash.\n",
    )
    .unwrap();

    let cfg = config_for_patterns(&dir, 100, false, &["mp-stash"]);
    let ctx = discover::run(&cfg).unwrap();

    let stash_passages = ctx
        .passages
        .iter()
        .filter(|p| p.text.contains("mp-stash command stores a named stash"))
        .count();
    assert_eq!(
        stash_passages, 1,
        "near-duplicate match lines must collapse"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn max_passages_is_honored() {
    let corpus = fixture_corpus("cap");
    let cfg = config_for(&corpus, 1, false);
    let ctx = discover::run(&cfg).unwrap();
    assert!(
        ctx.passages.len() <= 1,
        "output must not exceed max_passages"
    );
    let _ = std::fs::remove_dir_all(&corpus);
}

#[test]
fn missing_corpus_dir_errors_clearly() {
    let cfg = EvolveConfig::from_toml_str(
        "[evolve]\ncorpus_dir = \"/no/such/corpus/dir/exists\"\n[discover]\nseed = \"corpus\"",
    )
    .unwrap();
    let err = discover::run(&cfg).unwrap_err();
    assert!(err.to_string().contains("corpus_dir"));
}

#[test]
fn discovered_context_round_trips_json() {
    let corpus = fixture_corpus("json");
    let cfg = config_for(&corpus, 100, true);
    let ctx = discover::run(&cfg).unwrap();
    let json = serde_json::to_string(&ctx).unwrap();
    let back: scrt_evolve::DiscoveredContext = serde_json::from_str(&json).unwrap();
    assert_eq!(ctx.passages.len(), back.passages.len());
    let _ = std::fs::remove_dir_all(&corpus);
}
