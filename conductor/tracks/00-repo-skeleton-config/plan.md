# Repo Skeleton + Config — Plan

## Tasks

1. [x] Create the Cargo workspace (`Cargo.toml`, `default-members`, shared
   `[workspace.dependencies]`, `release` profile) per DESIGN.md §Crate layout +
   scrt's profile conventions. -- evidence: `cargo metadata` lists both crates.
2. [x] Add `crates/scrt-evolve` (lib) with the public module skeleton:
   `lib.rs`, `config.rs`, `discover.rs`, `generate/`, `dataset.rs`, `train/`,
   `model.rs` (empty/`todo!()` seams that compile). -- evidence: `lib.rs` re-exports compile.
3. [x] Declare scrt-core as an interim git dep (pinned tag/commit) + a
   `// -> later: crates.io` note tracked for track 08. -- evidence: `Cargo.toml` dep line.
4. [x] Define feature flags: `train` (gates candle), `pyo3` (gates the bridge
   crate). Both off by default. -- evidence: `cargo tree` shows neither on default build.
5. [x] Implement `EvolveConfig` + per-stage/per-preset structs matching
   DESIGN.md §Config schema exactly (serde + toml). -- evidence: struct fields vs DESIGN table.
6. [x] Implement `EvolveConfig::load(path)` with validation: required
   `model_path` per active stage, partial-config support, `api_key_env`
   literal-secret rejection. -- evidence: tests below.
7. [x] Implement the work-dir layout helper (resolves `discovered.json`,
   `dataset.jsonl`, `adapter.safetensors`, checkpoints under `work_dir`). -- evidence: path-resolution test.
8. [x] Add `crates/scrt-evolve-cli` with clap subcommands
   `init|discover|generate|train|run` (only `init` implemented; others stubbed
   to call SDK fns). -- evidence: `scrt-evolve --help` lists all five.
9. [x] Implement `scrt-evolve init`: write commented scaffold, warn on missing
   `model_path`. -- evidence: init test asserts file written + warning on bad path.
10. [x] Add the `pyo3` stub crate/module seam (compiles under `--features pyo3`,
    exposes nothing real yet). -- evidence: `cargo build --features pyo3` green.
11. [x] Tests: config round-trip, partial configs, invalid-config rejection,
    work-dir paths, init scaffold, `cargo tree` ML-free assertion. -- evidence: test names.
12. [x] Final sweep: `cargo build`, `cargo test`, `cargo build --features pyo3`,
    `cargo build -p scrt-evolve --features train`, `cargo clippy`. -- evidence: all green.

## Sign-off
Complete — see `SIGN-OFF.md`. All acceptance criteria met (2026-06-17).
