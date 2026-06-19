# Repo Skeleton + Config — Specification

## Goal
DESIGN.md phase 1. Stand up the workspace, the `EvolveConfig` schema (toml
load + validate), and the work-dir layout. Establish the **PyO3 feature stub**
seam so the Python training-tooling bridge has a home from day one. No ML in
this track.

## Scope
- Cargo workspace per DESIGN.md §Crate layout: lib crate
  `crates/scrt-evolve` (SDK) + binary `crates/scrt-evolve-cli`.
- `config.rs`: `EvolveConfig` + per-stage (`[discover]`, `[generate]`,
  `[train]`) + per-preset sub-blocks, loaded from `evolve.toml`. Partial
  configs must work (generate-only, train-only) — each stage reads only what
  it needs.
- `evolve init` CLI subcommand that scaffolds a commented `evolve.toml` and
  warns (not errors) if `model_path` doesn't exist yet.
- Work-dir layout helper (`work_dir` defaults to `.scrt-evolve`; resolves
  `discovered.json`, `dataset.jsonl`, adapter/checkpoint paths).
- Feature flags wired but inert: `train` (candle, off by default), `pyo3`
  (off by default). Default `cargo build` pulls neither.
- Secret handling: `api_key_env` is a **var name**, never an inline key; a
  validator rejects anything that looks like a literal secret in that field.

## Constraints
- Default build/test must NOT compile candle or require Python headers
  (mirror the scrt workspace's `scrt-py`/`scrt-napi` gating and the in-tree
  spike's proven ML-opt-in pattern).
- `evolve.toml` schema must match DESIGN.md §Config schema field-for-field
  (don't invent fields).
- scrt-core is an interim **git dep**; pin it, don't vendor.

## Acceptance
- `cargo build` and `cargo test` succeed with NO candle and NO pyo3 in the
  dependency tree (assert via `cargo tree`).
- `cargo build --features pyo3` and `cargo build -p scrt-evolve --features train`
  both compile (stubs are enough; no real ML/Python logic yet).
- `EvolveConfig::load` round-trips the DESIGN.md example `evolve.toml`;
  partial configs (only `[generate]`, only `[train]`) load without error.
- Invalid config (missing `model_path` where required, inline secret in
  `api_key_env`) is rejected with a clear error.
- `scrt-evolve init` writes a valid scaffold and warns on a missing model path.

## Dependencies
None (root track). Consumes scrt-core only as a declared git dep; no scrt API
is called yet.
