//! `init` scaffold tests: writes a valid, round-trippable file and flags a
//! missing model_path as a warning (not an error).

use scrt_evolve::EvolveConfig;

/// A unique temp path under the OS temp dir (no external dep; the process id
/// + a per-test suffix is enough isolation for these tests).
fn temp_path(suffix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("scrt-evolve-test-{}-{suffix}.toml", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn init_writes_a_valid_round_trippable_scaffold() {
    let path = temp_path("valid");
    let report = scrt_evolve::scaffold::init(&path).expect("init writes scaffold");

    // File exists and loads through the real config loader.
    assert!(path.exists());
    let cfg = EvolveConfig::load(&path).expect("scaffold must be a valid config");
    assert!(cfg.evolve.model_path.is_some());
    assert!(cfg.discover.is_some());
    assert!(cfg.generate.is_some());
    assert!(cfg.train.is_some());

    // The scaffold points at a placeholder model that doesn't exist -> warn.
    assert!(
        report.model_path_missing,
        "scaffold's placeholder model_path should be flagged missing"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn init_refuses_to_overwrite() {
    let path = temp_path("overwrite");
    scrt_evolve::scaffold::init(&path).expect("first init ok");
    let err = scrt_evolve::scaffold::init(&path).expect_err("second init must error");
    assert!(err.to_string().contains("already exists"));
    let _ = std::fs::remove_file(&path);
}
