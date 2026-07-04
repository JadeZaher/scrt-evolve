//! Multi-goal buildable driver tests (track 20 slice 5). No ML, no network.
//!
//! Exercises `goals::run_buildable` with an injected, deterministic generator
//! so the loop is tested without an API call: for each `[[goals]]`, discover the
//! goal's tagged stashes → generate → write per-goal artifacts. Proves the loop
//! is bounded by goal count, scopes discover by tag, and writes
//! `work_dir/goals/<name>/{discovered.json,dataset.jsonl}`.

use scrt_evolve::dataset::{Dataset, GenExample, Outcome, Tier, Verdict};
use scrt_evolve::{DiscoveredContext, EvolveConfig};

/// Build a fixture palace with two tagged stashes (security / perf) targeting
/// two corpus topics — mirrors the discover.rs palace fixture.
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
        stash("auth-flow", "the login path", "authenticate", "security"),
        stash("cache-layer", "the redis cache", "cacheget", "perf"),
    );
    let path = dir.join("mind-palace.json");
    std::fs::write(&path, json).unwrap();
    path
}

/// A deterministic, network-free generator: one Qa row per discovered passage,
/// echoing the passage source. Stands in for `generate::run` in tests.
fn fake_generate(_cfg: &EvolveConfig, ctx: &DiscoveredContext) -> anyhow::Result<Dataset> {
    let rows = ctx
        .passages
        .iter()
        .map(|p| GenExample::Qa {
            prompt: format!("about {}", p.source),
            completion: p.text.clone(),
            source: Some(p.source.clone()),
            gen: Some("test".to_string()),
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        })
        .collect();
    Ok(Dataset::new(rows))
}

#[test]
fn run_buildable_loops_goals_and_scopes_by_tag() {
    let mut base = std::env::temp_dir();
    base.push(format!("scrt-evolve-goals-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let corpus = base.join("corpus");
    let palace_dir = base.join("palace");
    let work = base.join("work");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::create_dir_all(&palace_dir).unwrap();
    std::fs::write(
        corpus.join("auth.md"),
        "fn authenticate(user) checks the password.\n",
    )
    .unwrap();
    std::fs::write(corpus.join("cache.md"), "fn cacheget(key) reads redis.\n").unwrap();
    let palace = write_palace(&palace_dir);

    let toml = format!(
        r#"
[evolve]
corpus_dir = {corpus:?}
palace_path = {palace:?}
work_dir = {work:?}

[[goals]]
name = "security-mastery"
topic = "authenticate"
tag = "security"

[[goals]]
name = "perf-mastery"
topic = "cacheget"
tag = "perf"
"#,
    );
    let cfg = EvolveConfig::from_toml_str(&toml).unwrap();

    let report = scrt_evolve::goals::run_buildable(&cfg, fake_generate).expect("runs");

    // One run per goal (loop bounded by goal count).
    assert_eq!(report.runs.len(), 2);

    // The security goal seeds only the security-tagged auth stash → auth.md.
    let sec = report
        .runs
        .iter()
        .find(|r| r.goal == "security-mastery")
        .unwrap();
    assert_eq!(sec.note, "ok");
    assert!(sec.rows >= 1, "security goal produced rows");
    let sec_data = Dataset::read_jsonl(sec.dataset_path.as_ref().unwrap()).unwrap();
    assert!(
        sec_data.rows.iter().all(|r| match r {
            GenExample::Qa { source, .. } => source.as_deref().unwrap().contains("auth.md"),
            _ => false,
        }),
        "security goal must only see the auth (security-tagged) stash, not cache"
    );

    // The perf goal seeds only the perf-tagged cache stash → cache.md.
    let perf = report
        .runs
        .iter()
        .find(|r| r.goal == "perf-mastery")
        .unwrap();
    let perf_data = Dataset::read_jsonl(perf.dataset_path.as_ref().unwrap()).unwrap();
    assert!(
        perf_data.rows.iter().all(|r| match r {
            GenExample::Qa { source, .. } => source.as_deref().unwrap().contains("cache.md"),
            _ => false,
        }),
        "perf goal must only see the cache (perf-tagged) stash, not auth"
    );

    // Per-goal artifacts were written under work_dir/goals/<name>/.
    assert!(work
        .join("goals")
        .join("security-mastery")
        .join("discovered.json")
        .exists());
    assert!(work
        .join("goals")
        .join("perf-mastery")
        .join("dataset.jsonl")
        .exists());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn run_buildable_errors_without_goals() {
    let cfg = EvolveConfig::from_toml_str("[evolve]\ncorpus_dir = \".\"\n").unwrap();
    let err = scrt_evolve::goals::run_buildable(&cfg, fake_generate).unwrap_err();
    assert!(err.to_string().contains("no `[[goals]]`"));
}

#[test]
fn run_buildable_generate_failure_is_per_goal() {
    // A failing generator must NOT abort the loop — each goal records its note.
    let mut base = std::env::temp_dir();
    base.push(format!("scrt-evolve-goals-genfail-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let corpus = base.join("corpus");
    let palace_dir = base.join("palace");
    let work = base.join("work");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::create_dir_all(&palace_dir).unwrap();
    std::fs::write(
        corpus.join("auth.md"),
        "fn authenticate(user) checks the password.\n",
    )
    .unwrap();
    let palace = write_palace(&palace_dir);

    let toml = format!(
        r#"
[evolve]
corpus_dir = {corpus:?}
palace_path = {palace:?}
work_dir = {work:?}

[[goals]]
name = "security-mastery"
topic = "authenticate"
tag = "security"
"#,
    );
    let cfg = EvolveConfig::from_toml_str(&toml).unwrap();

    let report =
        scrt_evolve::goals::run_buildable(&cfg, |_c, _ctx| anyhow::bail!("simulated API outage"))
            .expect("loop still completes");

    assert_eq!(report.runs.len(), 1);
    let run = &report.runs[0];
    assert_eq!(run.rows, 0);
    assert!(run.dataset_path.is_none());
    assert!(
        run.note.contains("generate skipped/failed"),
        "got note: {}",
        run.note
    );
    // Discover still ran, so passages were found before the generate failure.
    assert!(run.passages >= 1);

    let _ = std::fs::remove_dir_all(&base);
}
