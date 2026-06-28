---
type: Track Spec
title: DAG Engine — Typed Node Registry + Executor
description: Re-express the pipeline as a typed DAG with a node registry and executor.
tags: [track-16, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# DAG Engine — Typed Node Registry + Executor — Specification

## Goal
Re-express the framework's pipeline as a **typed DAG** instead of a hardcoded
chain. Every step (discover, plan, generate, gate, train, eval, prune,
spawn-expert, …) becomes a registered **Node** with typed input/output **ports**;
a run is a **Dag** = nodes + edges, validated at build time (acyclic, port types
match) and executed by a topological scheduler with artifact caching. The
existing linear `main.rs:run()` becomes ONE generated, serializable DAG — the
foundation that makes the QA→DAG factory + self-architecting meta-loop (track 17)
possible.

> This generalizes the EXISTING `plan/` module pattern (a planner LLM emits a
> durable, inspectable `GenPlan`/`GenSpec` artifact instead of a human
> hardcoding generation) up one level: from "generation is planned" to "the
> whole pipeline is a planned, serializable DAG." The DAG is DATA (`dag.json`),
> which is the precondition for a node that emits/edits a DAG (track 17).

## Scope
- **`dag/node.rs` — the Node abstraction.** A `Node` has: stable `id`, a `kind`
  (registry key), typed `inputs: Vec<Port>` / `outputs: Vec<Port>`, and an opaque
  per-node `cfg` (serde value validated by the node impl). A `NodeImpl` trait:
  ```text
  trait NodeImpl {
      fn kind(&self) -> &str;
      fn input_ports(&self) -> &[PortSpec];   // typed: DiscoveredContext, Dataset, …
      fn output_ports(&self) -> &[PortSpec];
      fn validate_cfg(&self, cfg: &Value) -> Result<()>;
      fn run(&self, ctx: &NodeCtx, inputs: PortMap) -> Result<PortMap>;
  }
  ```
- **`dag/registry.rs` — node registry.** Maps `kind` → `NodeImpl`. The existing
  stages are registered as the first nodes by WRAPPING their current `run`
  functions (no rewrite): `discover` (→`DiscoveredContext`), `plan`
  (`DiscoveredContext`→`GenPlan`), `generate` (`GenPlan|DiscoveredContext`→
  `Dataset`), `gate` (`Dataset`→`Dataset`, track 10), `train` (`Dataset`→
  `TrainReport`/artifact), `eval` (model→`ScoreReport`, track 10), and the lane
  nodes (regen/refine/mask/expert/prune/txn) as they land. Third-party/custom
  nodes register via the same trait.
- **Port type system.** `PortSpec { name, ty }` where `ty` is a closed enum over
  the framework's artifact types (`DiscoveredContext`, `GenPlan`, `Dataset`,
  `LoadedModel`, `Adapter`, `ScoreReport`, `AttributionReport`, `Checkpoint`,
  `ExpertRegistry`, …) — extensible but checked. Edge validation: source port
  type == dest port type, or a registered coercion.
- **`dag/graph.rs` — `Dag` + validation.** `Dag { nodes: Vec<Node>, edges:
  Vec<Edge> }`; `Edge { from: (node_id, port), to: (node_id, port) }`. Build-time
  validation: every edge's ports exist and types match; no cycles (topo-sort
  succeeds); every required input port is fed; each node's `cfg` passes
  `validate_cfg`. Serializable to/from `work_dir/dag.json` (the durable,
  inspectable, editable artifact).
- **`dag/exec.rs` — executor.** Topo-sort → run nodes; pass outputs along edges
  via a typed `PortMap`; **content-address artifacts by input hash** so an
  unchanged subgraph is cached (re-run only what's stale — the "move fast"
  property at the orchestration level). Parallel-ready (independent nodes may run
  concurrently) but v1 may execute sequentially; resumable from a partial run
  (reuse track 15 checkpoint conventions for node-level state). Per-node failure
  surfaces with the node id + does not corrupt completed artifacts.
- **DAG-centric SDK substrate (the lowering target; primary surface is track 18).**
  This track exposes building / loading / running a `Dag` programmatically —
  `Dag::canonical(&cfg)`, `Dag::load("dag.json")`, `dag.run(&ctx)`. This is the
  layer the **trait-powered builder (track 18) LOWERS TO** — track 18 is the
  headline SDK interface; the `Dag` API here is what it compiles into and what
  persists. The original `discover::run` / `generate::run` / `train::run` stay as
  thin convenience wrappers over the canonical DAG. Every node, the registry,
  validator, executor, and `dag.json` load/save are PUBLIC SDK items the CLI
  calls, never reimplements.
- **CLI = pure argv→SDK shim (enforced).** The binary parses args, calls one SDK
  function, and prints — it holds NO orchestration of its own. Concretely this
  track FIXES the current violation: `main.rs:run()`'s `Run` arm chains
  `cmd_discover → cmd_plan → cmd_generate → train` inline; that pipeline logic
  MOVES into the SDK as the canonical DAG (`Dag::canonical(&cfg).run()`), and the
  `Run` subcommand becomes a one-line call to it. New `scrt-evolve dag
  run --dag dag.json` / `dag validate` / `dag show` are likewise one-line shims
  over the SDK's DAG API.
- **`[dag]` config block** — new `Option<DagConfig>` field on `EvolveConfig`
  (serde-default, non-breaking): `dag_path` (load a graph), `cache` (on|off),
  `parallel` (bool), `on_node_error` (halt|skip-downstream). Absent → the
  canonical linear DAG built from the existing stage config (today's behavior).

## Constraints
- **Wrap, don't rewrite.** Existing stage `run` fns are wrapped as nodes; their
  signatures/tests stay green. The DAG is additive — with no `[dag]` block and no
  `dag.json`, the framework behaves exactly as today (a generated canonical DAG).
- **Typed + acyclic, enforced at build time.** An invalid graph (type mismatch,
  cycle, unfed required port, bad node cfg) fails `Dag::validate()` BEFORE any
  node runs — never a runtime surprise mid-train.
- **DAG is pure data.** `dag.json` fully describes a run and round-trips; no node
  may depend on ambient state not expressed as an input port (so a DAG is
  reproducible and a track-17 planner can author one).
- **No new ML.** This track is pure orchestration — it moves NO model code; it
  schedules the nodes that other tracks build. ML-free `cargo build` green.
- **Caching is sound.** A cached artifact is reused ONLY if the node kind, cfg,
  and all upstream input hashes match (cache key = hash of those). Stale-on-edit.
- **SDK-first, all of it (cross-track rule for 10–17).** Everything in the
  self-evolve + architecture lanes is a PUBLIC SDK function/type the CLI merely
  shims; no behavior may live only in the binary. The DAG-centric API + the
  convenience wrappers are both first-class. (DESIGN.md:187 contract, now
  explicitly extended to the new lane.)

## Acceptance
- A node registry holds the existing stages; building the canonical DAG and
  running it reproduces today's `run()` output on a fixture (back-compat).
- `Dag::validate()` rejects: a type-mismatched edge, a cycle, an unfed required
  port, and a node whose `cfg` fails `validate_cfg` — each with a clear error
  naming the node/edge (four asserted cases).
- `dag.json` round-trips (serialize → load → identical `Dag`).
- Executor topo-sorts + runs a 3-node fixture DAG; re-running with one node's cfg
  changed re-executes only that node + its descendants (cache hit upstream,
  asserted).
- `dag run --dag`, `dag validate`, `dag show` work on a fixture graph.
- `[dag]` absent → canonical DAG = current behavior (asserted identical).
- **SDK-primary**: a library test builds + runs a DAG via `Dag::builder`/`Dag::load`
  with NO CLI involved; the three convenience fns (`discover/generate/train::run`)
  still work and produce the same artifacts (asserted).
- **Pure-shim**: the `Run` subcommand contains no pipeline logic — it calls
  `Dag::canonical(&cfg).run()`; asserted that the canonical DAG (SDK) reproduces
  the prior inline `cmd_*` chain output, i.e. the orchestration moved out of the
  binary.
- ML-free `cargo build` + `cargo test` green.
- **Styleguide gates** (code-styleguides.md): this track IS the enforcement point
  for §2.1 (no-ambient-state — a node depends only on declared ports + cfg), §2.3
  (content-addressed artifacts, atomic writes, resume-from-partial, finished-DAG
  re-run is a no-op), and §2.5 (`dag.json` serializable round-trip). The cache key
  = hash(kind, cfg, upstream inputs) realizes §2.1/§2.2. Built per §4.

## Dependencies
Wraps stages from tracks 01 (discover), the `plan/` module (plan), 02
(generate/dataset), 10 (gate/eval), and registers lane nodes (11–15) as they
land. No ML. Foundation for track 17 (QA→DAG factory + meta-loop). Best built
after the self-evolve lane's node shapes are known (10–15) so their ports are
stable, but the engine itself depends on none of their internals.
