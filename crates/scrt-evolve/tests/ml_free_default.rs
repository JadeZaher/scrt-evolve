//! Acceptance guard: a DEFAULT build must NOT pull candle or pyo3 into the
//! dependency tree. Asserted via `cargo tree` so a stray non-optional dep or a
//! mis-gated feature is caught here rather than in a slow surprise build.

use std::process::Command;

fn cargo_tree_default() -> String {
    let output = Command::new(env!("CARGO"))
        .args(["tree", "-p", "scrt-evolve", "--edges", "normal"])
        .output()
        .expect("run cargo tree");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn default_build_has_no_candle() {
    let tree = cargo_tree_default();
    assert!(
        !tree.contains("candle-core") && !tree.contains("candle-nn"),
        "candle must not be in the default dependency tree:\n{tree}"
    );
}

#[test]
fn default_build_has_no_pyo3() {
    let tree = cargo_tree_default();
    assert!(
        !tree.contains("pyo3"),
        "pyo3 must not be in the default dependency tree:\n{tree}"
    );
}

#[test]
fn default_build_has_no_safetensors() {
    let tree = cargo_tree_default();
    assert!(
        !tree.contains("safetensors"),
        "safetensors (train-only) must not be in the default tree:\n{tree}"
    );
}
