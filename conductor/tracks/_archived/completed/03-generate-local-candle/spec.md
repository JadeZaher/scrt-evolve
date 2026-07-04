---
type: Track Spec
title: Generate / Local Candle Backend
description: LocalCandle, a GenBackend that loads the model locally via candle — DESIGN.md phase 4.
tags: [track-03, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Generate / Local Candle Backend — Specification

## Goal
DESIGN.md phase 4. Implement `LocalCandle`, a `GenBackend` that loads the
user's model and runs candle text generation to synthesize the same
`GenExample` rows as the API backend — fully offline. Behind the `train`
feature.

## Scope
- `LocalCandle` impl of `GenBackend`: load `model_path` (safetensors +
  tokenizer) via the shared model loader seam (`model.rs`), run candle
  text-generation with sampling (`[generate.local]`: `max_new_tokens`,
  `temperature`) over the same `prompts.rs` templates.
- Selected via `[generate].backend = "local"` (or `--backend local`).
- Produces identical `GenExample`/`Dataset` rows as `ApiEndpoint` so the
  dataset stays backend-agnostic.
- This track establishes the **first use of `model.rs`** (the per-arch loader
  seam). Start with ONE well-supported architecture (e.g. Llama/Qwen family);
  `model.rs` is a clean trait so more arches are backlog, not blockers.

## Constraints
- Behind `--features train` (candle). Default build unaffected.
- Local generation is **lower-trust** (model-collapse / echo-chamber risk,
  DESIGN.md §Honest risks): apply dedup + a basic quality filter to output,
  and support an optional critique/refine pass.
- The model loader must be a **seam**: loading an unsupported arch fails with a
  clear "arch not yet supported" error, not a panic.
- Same `GenExample` schema as track 02 — no divergent rows.

## Acceptance
- With `--features train` and a tiny fixture model (or a CI-downloadable
  small model), `LocalCandle::generate` produces valid `GenExample` rows for a
  fixture passage.
- Output rows validate against the same `Dataset` schema as the API backend
  (a row from each backend is interchangeable).
- Dedup + quality filter drop a deliberately-degenerate (repeated/empty)
  generation in a unit test.
- An unsupported architecture yields a clear error (no panic).
- `evolve train generate --backend local` runs end-to-end on the fixture model.

## Dependencies
Track 02 (`GenBackend`, `prompts.rs`, `Dataset`), track 00 (`train` feature +
`model.rs` seam). First consumer of `model.rs` (shared with track 04).
