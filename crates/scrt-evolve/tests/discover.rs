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

/// Build a fixture palace JSON with two stashes whose search patterns target
/// two distinct corpus topics, so `palace_search` can be shown to narrow which
/// stash seeds discovery.
fn write_palace(dir: &std::path::Path) -> std::path::PathBuf {
    let stash = |name: &str, note: &str, pattern: &str, tag: &str| {
        format!(
            r#""{name}":{{"name":"{name}","note":"{note}","tags":["{tag}"],
            "created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z",
            "expires_at":null,"search":{{"pattern":"{pattern}","effort":"normal","sources_count":0}},
            "sources":[],"nodes":[],"file_paths":[],"relations":[]}}"#
        )
    };
    let json = format!(
        r#"{{"version":2,"stashes":{{{},{}}}}}"#,
        stash(
            "auth-flow",
            "the login authentication path",
            "authenticate",
            "security"
        ),
        stash(
            "cache-layer",
            "the redis cache subsystem",
            "cacheget",
            "perf"
        ),
    );
    let path = dir.join("mind-palace.json");
    std::fs::write(&path, json).unwrap();
    path
}

#[test]
fn palace_search_narrows_which_stashes_seed() {
    // Corpus with two clearly-separated topics, one matched by each stash's
    // search pattern.
    let mut base = std::env::temp_dir();
    base.push(format!("scrt-evolve-palsearch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    // Keep the palace OUTSIDE the corpus dir so the corpus sweep can't match
    // the palace JSON file itself.
    let corpus = base.join("corpus");
    let palace_dir = base.join("palace");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::create_dir_all(&palace_dir).unwrap();
    std::fs::write(
        corpus.join("auth.md"),
        "fn authenticate(user) checks the password.\n",
    )
    .unwrap();
    std::fs::write(
        corpus.join("cache.md"),
        "fn cacheget(key) reads from redis.\n",
    )
    .unwrap();
    let palace = write_palace(&palace_dir);

    let toml = format!(
        r#"
[evolve]
corpus_dir = {corpus:?}
palace_path = {palace:?}

[discover]
seed = "palace"
cluster = false
palace_search = "auth"
"#,
    );
    let cfg = EvolveConfig::from_toml_str(&toml).unwrap();
    let ctx = discover::run(&cfg).expect("palace-seeded discover runs");

    // Only the auth-flow stash should have seeded (its "authenticate" pattern),
    // so the auth passage surfaces and the cache passage does not.
    assert!(
        !ctx.passages.is_empty(),
        "auth stash should surface a passage"
    );
    assert!(
        ctx.passages.iter().any(|p| p.source.contains("auth.md")),
        "auth passage should be present"
    );
    assert!(
        !ctx.passages.iter().any(|p| p.source.contains("cache.md")),
        "palace_search=\"auth\" must NOT seed the cache stash, got sources: {:?}",
        ctx.passages.iter().map(|p| &p.source).collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Track 20 slice 3 — goal→discover wiring.
//
// `EvolveConfig::for_goal(goal)` sets discover.palace_search = goal.topic and
// discover.palace_tags = [goal.tag] and forces seed = "palace". The contract
// is one goal ⇄ one tag: only stashes carrying the goal's tag (AND matching its
// topic) should seed. Reuses the `write_palace` fixture (tags: security/perf).
// ---------------------------------------------------------------------------

#[test]
fn goal_scoped_discover_seeds_only_goal_tagged_stashes() {
    use scrt_evolve::GoalConfig;

    let mut base = std::env::temp_dir();
    base.push(format!("scrt-evolve-goal-discover-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let corpus = base.join("corpus");
    let palace_dir = base.join("palace");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::create_dir_all(&palace_dir).unwrap();
    std::fs::write(
        corpus.join("auth.md"),
        "fn authenticate(user) checks the password.\n",
    )
    .unwrap();
    std::fs::write(
        corpus.join("cache.md"),
        "fn cacheget(key) reads from redis.\n",
    )
    .unwrap();
    let palace = write_palace(&palace_dir);

    // A base config with the corpus + palace, plus a seed-everything default
    // discover block — for_goal must override it to the goal's tag/topic.
    let toml = format!(
        r#"
[evolve]
corpus_dir = {corpus:?}
palace_path = {palace:?}

[discover]
seed = "both"
cluster = false

[[goals]]
name = "security-mastery"
topic = "authenticate"
tag = "security"
"#,
    );
    let cfg = scrt_evolve::EvolveConfig::from_toml_str(&toml).unwrap();
    let goal: &GoalConfig = &cfg.goals[0];
    let per_goal = cfg.for_goal(goal);

    let ctx = discover::run(&per_goal).expect("goal-scoped discover runs");

    // The "security"-tagged auth-flow stash seeds (its "authenticate" pattern);
    // the "perf"-tagged cache-layer stash is filtered OUT by palace_tags.
    assert!(
        ctx.passages.iter().any(|p| p.source.contains("auth.md")),
        "the security-tagged stash should surface the auth passage"
    );
    assert!(
        !ctx.passages.iter().any(|p| p.source.contains("cache.md")),
        "palace_tags=[\"security\"] must NOT seed the perf-tagged cache stash, \
         got sources: {:?}",
        ctx.passages.iter().map(|p| &p.source).collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(&base);
}
