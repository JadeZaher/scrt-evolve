---
type: Implementation Plan
title: DAG Engine — Typed Node Registry + Executor
description: Implementation plan for the DAG Engine — Typed Node Registry + Executor track.
tags: [track-16, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# DAG Engine — Typed Node Registry + Executor — Plan

## Tasks

1. [ ] `dag/node.rs`: `PortSpec`/`Port`/`PortMap`, the artifact-type enum (`Ty`),
   and the `NodeImpl` trait (kind, input/output ports, validate_cfg, run).
   -- evidence: trait + port-type tests.
2. [ ] `dag/registry.rs`: `NodeRegistry` mapping `kind` → `NodeImpl`; register
   the existing stages by WRAPPING their `run` fns (discover, plan, generate,
   gate, train, eval). -- evidence: registry holds + dispatches stages test.
3. [ ] `dag/graph.rs`: `Dag`/`Node`/`Edge` types + `Dag::validate()` (ports
   exist, types match, acyclic via topo-sort, required ports fed, per-node
   `validate_cfg`). -- evidence: 4 rejection cases + 1 valid-graph test.
4. [ ] `dag.json` serde round-trip (Dag → file → identical Dag). -- evidence:
   round-trip test.
5. [ ] `dag/exec.rs`: topo-sort scheduler, typed `PortMap` passing, content-
   addressed artifact cache (key = node kind + cfg + upstream input hashes).
   -- evidence: 3-node run + stale-recompute (cache hit upstream) test.
6. [ ] Canonical-DAG builder: construct the equivalent linear DAG from existing
   stage config (the current `run()` as a graph). -- evidence: canonical-DAG
   reproduces current run() output (back-compat).
7. [ ] `[dag]` config as `Option<DagConfig>` on `EvolveConfig` (serde-default):
   `dag_path`, `cache`, `parallel`, `on_node_error`. Absent → canonical DAG.
   -- evidence: config + absent-is-canonical test.
8. [ ] DAG-centric SDK: public `Dag::builder(&cfg)` (programmatic graph),
   `Dag::canonical(&cfg)` (the default lane as a DAG), `Dag::load`/`save`, and
   `Dag::run` — all public. Keep `discover/generate/train::run` as convenience
   wrappers over the canonical DAG. -- evidence: SDK-builds-and-runs-a-DAG test
   (no CLI) + convenience-wrappers-still-work test.
9. [ ] Pure-shim refactor: MOVE `main.rs:run()`'s `Run` orchestration
   (cmd_discover→cmd_plan→cmd_generate→train) into the SDK canonical DAG; the
   `Run` arm becomes `Dag::canonical(&cfg).run()`. CLI holds no pipeline logic.
   -- evidence: canonical-DAG reproduces prior Run output; binary has no inline chain.
10. [ ] Back-compat shim: per-stage subcommands construct + run their node via the
    SDK; behavior unchanged. -- evidence: existing CLI tests still green.
11. [ ] CLI: `dag run --dag`, `dag validate`, `dag show` (one-line SDK shims).
    -- evidence: CLI tests.
12. [ ] Resumability: a partial run resumes from completed node artifacts
    (reuse track 15 checkpoint conventions). -- evidence: resume-from-partial test.
13. [ ] Final sweep: `cargo build`, `cargo test`, `cargo clippy`. -- evidence: green.

## Sign-off
Pending.
