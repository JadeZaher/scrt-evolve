---
type: Implementation Plan
title: Discover
description: Implementation plan for the Discover track.
tags: [track-01, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Discover — Plan

## Tasks

1. [x] Define `DiscoveredContext`, `Passage { text, source, score }`,
   `StashRef` with serde (the on-disk `discovered.json` shape). -- evidence: discover.rs:20-43
2. [x] Wire `scrt_core::palace::FilePalace`/`ops` to load the palace at
   `cfg.palace_path` and enumerate stashes (note + nodes). -- evidence: discover.rs:68-87
3. [x] Build seed queries from `[discover].seed` (palace stash notes / corpus
   sweep / both). -- evidence: discover.rs:129-162, `build_seeds` deterministic patterns
4. [x] Run corpus retrieval via `scrt_core::search_with_meta(&SearchConfig)`
   for each seed; collect passages with `source` + `score`. -- evidence: discover.rs:104-114
5. [x] Dedup near-duplicates via `scrt_core::palace::simhash` (chunked
   best-pair / Jaccard); collapse to representatives. -- evidence: `near_duplicate_passages_collapse` test, discover.rs:247-268
6. [x] Rank passages; implement `cluster = true` (cluster stashes by simhash,
   round-robin sample across clusters for topic spread). -- evidence: discover.rs:285-328 `cluster_round_robin`
7. [x] Apply `max_passages` (+ optional token budget) cap. -- evidence: `max_passages_is_honored` test, discover.rs:124
8. [x] `discover::run(&cfg) -> DiscoveredContext` SDK entry. -- evidence: `discovers_passages_with_provenance` test, discover.rs:56-127
9. [x] `evolve train discover --config evolve.toml` writes
   `work_dir/discovered.json`; round-trips back. -- evidence: `discovered_context_round_trips_json` test, main.rs:140-143, 202-215
10. [ ] Build a small fixture palace + corpus under `tests/fixtures/`. -- (carry-forward: not yet present; tests use temp directories instead)
11. [x] Final sweep: `cargo test` (default, ML-free), `cargo clippy`. -- evidence: all 5 discover tests pass, 51-test suite green, clippy clean

## Sign-off
Complete — see `SIGN-OFF.md`. All acceptance criteria met (2026-06-18).
