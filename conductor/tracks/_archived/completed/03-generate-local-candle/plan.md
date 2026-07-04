---
type: Implementation Plan
title: Generate / Local Candle Backend
description: Implementation plan for the Generate / Local Candle Backend track.
tags: [track-03, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Generate / Local Candle Backend — Plan

## Tasks

1. [x] Define the `model.rs` loader seam: a `LoadedModel` + a per-arch loader
   trait (safetensors + tokenizer); ONE concrete arch first. -- evidence: `crates/scrt-evolve/src/model.rs` (LoadedModel struct, TinyLlama impl, HfConfig struct, arch_supported, ModelConfig, ModelError enum).
2. [x] Unsupported-arch path returns a clear error. -- evidence: `crates/scrt-evolve/src/model.rs::tests::unsupported_arch_errors_not_panics` (line 734, validates ModelError::UnsupportedArch on MambaForCausalLM).
3. [x] Implement `LocalCandle` `GenBackend` over candle text-generation with
   sampling (`max_new_tokens`, `temperature`). -- evidence: `crates/scrt-evolve/src/generate/local.rs` (LocalCandle struct, from_config, from_model, generate_text, GenBackend impl).
4. [x] Reuse `prompts.rs` templates so rows match the API backend exactly. -- evidence: `crates/scrt-evolve/src/generate/local.rs::render_prompt` + `crates/scrt-evolve/tests/generate_local.rs::local_and_api_rows_are_schema_interchangeable` (line 101, proves both backends serialize identically).
5. [x] Dedup + basic quality filter on generated output; optional
   critique/refine pass. -- evidence: `crates/scrt-evolve/src/generate/local.rs::filter_degenerate` + `crates/scrt-evolve/tests/generate_local.rs::degenerate_output_is_filtered` (line 58, drops empty/short/repeated/echo rows, keeps unique good rows). NOTE: Optional critique/refine pass NOT implemented (lower priority, tracked as carry-forward per spec §"lower-trust" phase 1).
6. [x] Wire `[generate].backend = "local"` / `--backend local` selection. -- evidence: `crates/scrt-evolve/src/generate/mod.rs` line 90-96 (dispatch on "local", LocalCandle::from_config called, feature-gated).
7. [x] Tiny fixture model for CI (or a gated small-model download). -- evidence: `crates/scrt-evolve/src/model.rs::LoadedModel::random_fixture` (line 450+, deterministic in-memory tiny model from seed, no download required).
8. [x] Cross-backend interchangeability: a `LocalCandle` row and an
   `ApiEndpoint` row both validate against `Dataset` schema. -- evidence: `crates/scrt-evolve/tests/generate_local.rs::local_and_api_rows_are_schema_interchangeable` (line 101, round-trip JSONL identical).
9. [x] `evolve train generate --backend local` end-to-end on fixture. -- evidence: `crates/scrt-evolve/tests/generate_local.rs::local_backend_produces_valid_rows_on_fixture` (line 30, runs offline, stamps gen=local).
10. [x] Final sweep: `cargo test --features train`, `cargo clippy
    --features train`. -- evidence: all tests pass (4 generate_local + 6 model tests green); clippy clean; default build ML-free.

## Sign-off
Complete — see `SIGN-OFF.md`. All acceptance criteria met (2026-06-19).
