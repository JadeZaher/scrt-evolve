//! Branch factory CLI (track 29, Phase 3) — drives the compiled `scrt-evolve`
//! binary against a fixture registry. The routing DECISION logic is unit-tested in
//! the SDK (`scrt-evolve/tests/branch.rs`); this asserts the CLI shim wires
//! list / route / serve-fallback correctly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use scrt_evolve::branch::router::corpus_signature;
use scrt_evolve::branch::{BranchManifest, BranchRegistry, Lineage, MANIFEST_VERSION};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_evolve")
}

fn manifest(name: &str, domain: &str, texts: &[&str]) -> BranchManifest {
    let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    let mut eval = BTreeMap::new();
    eval.insert("correctness".to_string(), 0.88);
    BranchManifest {
        name: name.to_string(),
        base_model: "granite-eval-0.5b".to_string(),
        domain: domain.to_string(),
        corpus_descriptor: "fixture".to_string(),
        router_signature: corpus_signature("simhash", &texts),
        eval_report: eval,
        lineage: Lineage::default(),
        version: MANIFEST_VERSION.to_string(),
        gguf_sha: "deadbeef".to_string(),
        created: "2026-06-26T00:00:00Z".to_string(),
        tier: scrt_evolve::dataset::Tier::Private,
    }
}

/// A temp work-dir with a 2-branch fixture registry + an evolve.toml pointing at it.
/// `floor` is the router `confidence_floor` (high ⇒ only strong matches resolve).
fn fixture_floor(tag: &str, floor: f32) -> (PathBuf, PathBuf) {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "scrt-evolve-branchcli-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut reg = BranchRegistry::empty();
    reg.upsert(manifest(
        "legal",
        "legal/tool-calling",
        &["contract law clause arbitration tort liability damages statute"],
    ));
    reg.upsert(manifest(
        "cooking",
        "food/recipes",
        &["saute onions preheat oven bake simmer roast garlic"],
    ));
    reg.write(dir.join("branches").join("registry.json"))
        .unwrap();

    let cfg_path = dir.join("evolve.toml");
    let toml = format!(
        r#"
[evolve]
model_path = "/m"
work_dir = {dir:?}

[branch]
  [branch.router]
  kind = "simhash"
  confidence_floor = {floor}
  top_k = 1
"#
    );
    std::fs::write(&cfg_path, toml).unwrap();
    (dir, cfg_path)
}

/// Default fixture: a permissive floor so a strong in-domain query resolves.
fn fixture(tag: &str) -> (PathBuf, PathBuf) {
    fixture_floor(tag, 0.5)
}

fn run(cfg: &Path, args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(bin());
    cmd.arg("branch");
    cmd.args(args);
    cmd.arg("--config").arg(cfg);
    let out = cmd.output().expect("run scrt-evolve");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), s)
}

#[test]
fn branch_list_shows_registered_branches() {
    let (_dir, cfg) = fixture("list");
    let (ok, out) = run(&cfg, &["list"]);
    assert!(ok, "branch list should succeed: {out}");
    assert!(out.contains("legal"), "list shows legal: {out}");
    assert!(out.contains("cooking"), "list shows cooking: {out}");
}

#[test]
fn branch_route_resolves_matching_query() {
    let (_dir, cfg) = fixture("route");
    let (ok, out) = run(
        &cfg,
        &[
            "route",
            "what statute governs this contract arbitration clause",
        ],
    );
    assert!(ok, "branch route should succeed: {out}");
    assert!(out.contains("legal"), "route resolves to legal: {out}");
}

#[test]
fn branch_serve_route_falls_back_to_base_when_no_match() {
    // A high floor ⇒ gibberish resolves to no branch ⇒ base-only fallback.
    // No --prompt, so the base path is a no-op (we only assert the fallback notice).
    let (_dir, cfg) = fixture_floor("serve-fallback", 0.95);
    let (ok, out) = run(&cfg, &["serve", "--route", "zzzzz qqqqq wwwww"]);
    assert!(ok, "serve --route fallback should succeed: {out}");
    assert!(
        out.to_lowercase().contains("base-only"),
        "no-match routes to base-only fallback: {out}"
    );
}

#[test]
fn branch_list_empty_registry_is_clean() {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "scrt-evolve-branchcli-empty-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("evolve.toml");
    std::fs::write(
        &cfg_path,
        format!("[evolve]\nmodel_path = \"/m\"\nwork_dir = {dir:?}\n"),
    )
    .unwrap();
    let (ok, out) = run(&cfg_path, &["list"]);
    assert!(ok, "empty list ok: {out}");
    assert!(
        out.contains("no branches registered"),
        "empty notice: {out}"
    );
}
