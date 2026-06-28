---
type: Implementation Plan
title: Generate / API Backend
description: Implementation plan for the Generate / API Backend track.
tags: [track-02, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Generate / API Backend — Plan

## Tasks

1. [x] Define `GenBackend` trait, `GenContext`, `GenExample` (DESIGN.md
   trait #2). -- evidence: `src/generate/mod.rs` lines 38–56.
2. [x] Define the `Dataset` JSONL row enum (`qa|instruction|completion|
   contrastive|tool_call|cli`) + `source`/`gen` provenance per DESIGN.md §Dataset format. -- evidence: `src/dataset.rs` lines 14–77 serde-tagged enum.
3. [x] Implement `Dataset` writer + reader (one object per line; streaming
   reader). -- evidence: test `dataset_round_trips_through_jsonl` (line 120–149).
4. [x] `prompts.rs`: QA + instruction synthesis templates; `per_passage`
   fan-out. -- evidence: `src/generate/prompts.rs` lines 12–46 (system/user/refine prompts).
5. [x] Implement `ApiEndpoint` (reqwest/rustls): `base_url`/`model`/`turns`,
   `api_key_env` lookup, request/response mapping for OpenAI + Anthropic
   shapes. -- evidence: `src/generate/api.rs` struct (lines 41–99), trait impl (lines 101–141).
6. [x] Multi-turn refine loop when `turns > 1`. -- evidence: test `turns_greater_than_one_issues_refine_turns` (lines 105–117).
7. [x] `generate::run(&cfg, &ctx) -> Dataset` driver (passage → N examples →
   rows). -- evidence: `src/generate/mod.rs` lines 81–139 (run + run_with_backend); test `mocked_backend_produces_qa_and_instruction_rows`.
8. [x] `scrt-evolve generate [--in discovered.json] [--backend api]` →
   `dataset.jsonl`; runs standalone from disk. -- evidence: CLI `crates/scrt-evolve-cli/src/main.rs` lines 228–238 (cmd_generate).
9. [ ] PyO3 bridge: `read_dataset(path)` (+ iterator) under `--features pyo3`. -- (carry-forward: deferred; `bridge.rs` stubs only `version()` function; real read_dataset body is pending).
10. [ ] Cross-language schema-parity test: Rust writes `dataset.jsonl`, Python
    (via bridge) reads identical rows. -- (carry-forward: deferred; no .py test fixtures; dataset schema is final, Python harness pending).
11. [x] Error path: unset `api_key_env` → clear error. -- evidence: test `missing_api_key_env_is_a_clear_error` (lines 152–168); also test `no_api_key_env_means_unauthenticated_local_endpoint_ok` (lines 211–224).
12. [x] Final sweep: `cargo test`, `cargo build --features pyo3` + validation, `cargo clippy`. -- evidence: 12/12 tests pass (generate 9 + export 3); clippy clean; pyo3 build succeeds.

## Sign-off

✅ **Core API path complete (2026-06-18).** All discovery→dataset→export flows implemented and tested. PyO3 bridge + Python schema-parity carried forward — see SIGN-OFF.md.
