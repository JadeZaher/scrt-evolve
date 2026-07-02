---
type: Track Spec
title: Generate / API Backend
description: The GenBackend trait and the ApiEndpoint backend — DESIGN.md phase 3.
tags: [track-02, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Generate / API Backend — Specification

## Goal
DESIGN.md phase 3. Define the `GenBackend` trait, implement the `ApiEndpoint`
backend (configurable OpenAI/Anthropic/OpenAI-compatible), the QA/instruction
prompt templates, and the JSONL `Dataset` writer/reader. Deliver end-to-end
**discover → dataset** with NO local model. Introduce the **PyO3 dataset
export** so Python training tooling can consume the dataset.

## Scope
- `GenBackend` trait: `fn generate(&self, ctx: &GenContext) -> Result<Vec<GenExample>>`
  (DESIGN.md §The three core traits #2).
- `ApiEndpoint` impl: configurable `base_url`, `model`, `api_key_env`, `turns`
  (multi-turn refine/critique if >1). Auth read from the named env var only.
- `prompts.rs`: QA + instruction synthesis templates, `per_passage` examples.
- `dataset.rs`: the JSONL contract from DESIGN.md §Dataset format — rows
  `kind = qa | instruction | completion | contrastive` with `source` + `gen`
  provenance. Writer + reader, one object per line.
- `generate::run(&cfg, &ctx) -> Dataset` (SDK) + `scrt-evolve generate
  [--in discovered.json] [--backend api]` (CLI) → `dataset.jsonl`.
- **PyO3 bridge (first real surface):** under `--features pyo3`, expose a
  `read_dataset(path) -> list[dict]` (and an iterator) so `transformers`/`peft`
  pipelines can load a scrt-evolve dataset directly. The Rust JSONL writer and
  the Python reader must agree byte-for-byte on the row schema.

## Constraints
- **No secrets inline** — `ApiEndpoint` reads the key from `api_key_env`;
  fail clearly if the env var is unset.
- API calls go through `reqwest` (rustls). Network calls are mocked in tests
  (no live API in CI).
- The dataset is the durable, inspectable artifact — generate once, train many
  presets from it. Row schema is the **cross-language contract**; changing it
  is a breaking change for the Python side.
- Stage independence: `generate` must run from an on-disk `discovered.json`
  without re-running discover.

## Acceptance
- `generate::run` over a fixture `DiscoveredContext` with a **mocked**
  `ApiEndpoint` produces a valid `dataset.jsonl` (qa + instruction rows, with
  `source`/`gen` provenance, `per_passage` honored).
- `Dataset` round-trips (write → read → equal); one JSON object per line.
- Missing `api_key_env` env var → clear error, no panic.
- `turns > 1` issues the refine turn(s) (assert on the mock).
- Under `--features pyo3`, a Python test loads `dataset.jsonl` via the bridge
  and sees the same rows (schema parity test, Rust-written ↔ Python-read).
- `scrt-evolve generate --in discovered.json --backend api` writes the dataset
  without invoking discover.

## Dependencies
Track 01 (`DiscoveredContext`/`discovered.json`), track 00 (config + pyo3
feature stub). No candle.
