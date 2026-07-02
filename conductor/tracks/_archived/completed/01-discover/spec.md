---
type: Track Spec
title: Discover
description: discover.rs consumes scrt-core in-process to surface candidate context — DESIGN.md phase 2.
tags: [track-01, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Discover — Specification

## Goal
DESIGN.md phase 2. Implement `discover.rs`: consume **scrt-core** in-process to
turn the corpus + palace into a `DiscoveredContext` (ranked, deduped,
budget-capped passages with provenance), written to `work_dir/discovered.json`.
No ML.

> **Revision note (self-evolve lane, B4 cogency fix).** The product goal
> "evolve across ALL the user's projects, merged into ONE model" requires
> discovery to union **multiple** project directories. Add
> `corpus_dirs: Vec<PathBuf>` to `EvolveSection` alongside the existing single
> `corpus_dir` (a convenience that maps to a one-element `corpus_dirs`); when
> set, discovery sweeps each dir and unions/dedups the passages into one
> `DiscoveredContext`, carrying per-passage `source` (already present) so the
> originating project is traceable. Backward-compatible: `corpus_dir` alone
> behaves exactly as today. Consumed by the lane (esp. track 12 self-refine).
> This is a small additive change tracked here; implement when the lane lands.

## Scope
- `DiscoveredContext { passages: Vec<Passage>, anchors: Vec<StashRef> }` where
  `Passage { text, source, score }` and `StashRef` identifies the seeding
  palace stash (per DESIGN.md §The three core traits #1).
- Discovery strategy, config-driven by `[discover]`:
  - `seed = palace | corpus | both` — walk palace stashes (their notes) as
    seed queries, OR sweep the corpus, OR both.
  - scrt-search the corpus via `scrt_core::search_with_meta(&SearchConfig)`.
  - `dedup = "simhash"` — drop near-duplicate passages using
    `scrt_core::palace::simhash` (the chunked best-pair / Jaccard signals
    already shipped in scrt-core).
  - rank, then `cluster = true` to spread coverage across distinct topics
    (cluster stashes by similarity, sample across clusters) so generation
    doesn't re-mine one topic.
  - cap by `max_passages` (and/or a token budget).
- Reads palace via `scrt_core::palace::FilePalace` / `palace::ops`.
- `discover::run(&cfg) -> DiscoveredContext` (SDK) + `scrt-evolve discover`
  (CLI) writing `discovered.json`.

## Constraints
- **scrt-core is called as a Rust crate** — direct fn calls, NOT a subprocess,
  NOT PyO3 (DESIGN.md §How it consumes scrt).
- No new ML deps; this track stays on the default (ML-free) build.
- Deterministic given the same palace + corpus + config (dedup/cluster/sampling
  reproducible — no unseeded RNG), so `discovered.json` is diffable.

## Acceptance
- `discover::run` against a **fixture palace + fixture corpus** produces a
  non-empty `DiscoveredContext` with correct provenance (`source` points at
  real corpus paths; `anchors` reference real stashes).
- Near-duplicate passages are collapsed (assert two seeded-identical passages
  yield one).
- `cluster = true` yields passages from ≥2 distinct stash clusters when the
  fixture has them.
- `max_passages` is honored (output length ≤ cap).
- CLI writes a valid `discovered.json` that round-trips back into
  `DiscoveredContext`.

## Dependencies
Track 00 (workspace + config + work-dir). Requires scrt-core's
`search_with_meta`, `palace::FilePalace`/`ops`, and `palace::simhash`.
