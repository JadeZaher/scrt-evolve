---
type: Implementation Plan
title: "Expert Spawn & Router (Adapter-Experts)"
description: "Implementation plan for the Expert Spawn & Router (Adapter-Experts) track."
tags: [track-14, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Expert Spawn & Router (Adapter-Experts, grow-on-demand) — Plan

## Tasks

1. [ ] `[experts]` config: `enabled`, `min_cluster_size`,
   `recurrence_threshold`, `max_experts`, `planner`, `router`, `top_k`,
   `route_confidence_floor`. ML-free round-trip; absent block = experts off.
   -- evidence: config + default-off test.
2. [ ] Training-path detector: cluster dataset rows by kind/tool/source +
   simhash similarity (reuse track 01); flag uncovered clusters over threshold
   as spawn candidates. -- evidence: clusters fixture into ≥2 paths; covered
   cluster not re-flagged.
3. [ ] `ExpertBlueprint` {path_id, target_modules[], seed_source,
   attribution_source} built by CONSUMING track 13's `AttributionReport` (no
   second attribution pass; grad/manual fallback inherited from 13). -- evidence:
   blueprint-from-report + no-LARQL-fallback test.
4. [ ] Expert registry `experts/registry.json` {path_id, blueprint,
   adapter_path, router_signature, created, parent_path, stats}; read/write
   round-trip. -- evidence: registry round-trip test.
5. [ ] Expert training: LoRA via **PyO3→`peft`** (track 04 preset) on a cluster
   targeting blueprint modules (mask from track 13) → `experts/<path_id>.safetensors`.
   Behind `--features pyo3` (candle optional). -- evidence: spawn-trains-and-registers test.
6. [ ] Router scaffold: input-descriptor similarity → top-k experts;
   confidence floor → base-only. Degrades to no-op with empty registry /
   `router=off`. -- evidence: routes-to-match + low-conf-base-only +
   empty-registry-noop tests.
7. [ ] Composition: apply ≤`top_k` LoRA adapters additively, bounded.
   -- evidence: bounded-compose test.
8. [ ] Bounded growth: `max_experts` cap + merge near-duplicate router
   signatures (no twin experts) + eviction policy. -- evidence: merge-not-twin +
   cap test.
9. [ ] Anti-collapse: expert training data passes track-10 gate / track-12
   constitution before training. -- evidence: gated-data-only test.
10. [ ] Spawn pipeline end to end (detect→plan→train→register), non-interactive
    + resumable. -- evidence: full-spawn fixture test.
11. [ ] CLI: `experts plan` (no train), `experts spawn`, `experts list`,
    `run --experts` (load registry + router). -- evidence: CLI tests.
12. [ ] Final sweep: `cargo build`, `cargo test`, `cargo test --features train`,
    `cargo build --features "train larql"`, `cargo clippy --features train`.
    -- evidence: green.

## Sign-off
Pending.
