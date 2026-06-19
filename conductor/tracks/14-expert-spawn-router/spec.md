# Expert Spawn & Router (Adapter-Experts, grow-on-demand) — Specification

## Goal
Turn the dense model into a **modular adapter-expert system**: each recurring
**training path** gets its own LoRA **expert**, planned by LARQL attribution
("which layers does this path route through?"), and a small learned **router**
dispatches inputs to the right expert(s) at runtime. The expert set **grows on
demand** — when the self-evolve loop detects a new recurring path cluster, it
plans and spawns a new expert — so the model's capability topology evolves with
the user's projects/goals. The base model stays dense and always-runnable; the
router degrades to a no-op.

> This is the buildable form of the "topology shift / expert node per path" idea.
> It is **adapter-experts + routing** (≈ MoLE / AdapterFusion), NOT a Mixtral
> FFN-MoE conversion and NOT carving a standalone sub-model (both considered,
> not chosen). No reverse inference. LARQL plans; training disposes.

> **Cogency-audit notes (apply throughout).**
> - **Config host (A1).** `[experts]` is a new `Option<ExpertsConfig>` field on
>   the top-level `EvolveConfig` struct (it is its own stage, beside
>   discover/generate/train; serde-default → non-breaking). NOT an unparsed
>   floating section.
> - **Single attribution owner (C6).** Attribution is OWNED by track 13. This
>   track CONSUMES track 13's `AttributionReport` to build an `ExpertBlueprint`;
>   it does NOT run its own `TRACE`/attribution pass. (Spec text below saying
>   "run TRACE" means "read track 13's report".) All "track 11 attribution"
>   references mean **track 13** post-renumber.
> - **Directive — PyO3→peft.** Each expert is a LoRA adapter trained via the
>   PyO3→`peft`/`transformers` bridge (peft adapter save/load + multi-adapter
>   composition is native there), not a hand-built candle LoRA. Router is native
>   Rust (it only selects + scales adapters); candle LoRA is optional later.
> - **Provenance (B5).** Expert-spawned training rows stamp `GenExample.gen`
>   (`expert:<path_id>`) so track 15 quarantine can isolate a bad expert's data.

## Scope
- **Training-path detector** (the grow-on-demand trigger): cluster the dataset
  (and incoming self-evolve rows) into recurring task types — by `kind`, by
  scrt tool used, by source/project, and by lexical similarity (reuse scrt-core
  simhash / discover clustering). A cluster that exceeds a size/recurrence
  threshold and is NOT already covered by an expert is a **spawn candidate**.
- **Expert blueprint planner** (reuses track 11's attribution selector): for a
  spawn candidate, run `TRACE`/attribution over the cluster's target tokens →
  the layers/modules the path routes through → a `ExpertBlueprint`
  {path_id, target_modules[], seed_source, attribution_source}. With no LARQL,
  fall back to track 11's `grad`/`manual` selection — the planner still works.
- **Expert = a LoRA adapter** trained for that path on its cluster, targeting the
  blueprint's modules (reuses the track-04 LoRA preset + track-11 mask).
  Artifact: `experts/<path_id>.safetensors` + an entry in the expert registry.
- **Expert registry** (`experts/registry.json`): {path_id, blueprint,
  adapter_path, router_signature (cluster centroid / descriptor), created,
  parent_path (if split from another), stats}. The durable record of the grown
  topology.
- **Router** (the one new runtime piece): a small learned/heuristic dispatcher
  mapping an input to top-k experts to apply. v1 router is **input-descriptor
  similarity** (match input to each expert's `router_signature`) with an optional
  learned head later. MUST degrade safely:
  - no experts / no router → base model only (today's behavior).
  - low confidence → apply nothing (base) rather than a wrong expert.
  - composition: apply ≤k adapters additively (LoRA sums), bounded.
- **Spawn pipeline** (grow-on-demand, daemon-triggerable): detect cluster →
  plan blueprint → train expert (LoRA, masked) → register → no model surgery.
  Non-interactive + resumable so the future daemon can drive it; on-demand via
  CLI now.
- **`[experts]` config**: `enabled`, `min_cluster_size`, `recurrence_threshold`,
  `max_experts` (roster cap + eviction policy), `planner`
  (attribution|grad|manual), `router` (similarity|learned|off), `top_k`,
  `route_confidence_floor`.
- CLI: `scrt-evolve experts plan` (detect + blueprint report, NO training),
  `scrt-evolve experts spawn [--path-id]` (plan→train→register),
  `scrt-evolve experts list`, and `--experts` on `run`/inference to load the
  registry + router.

## Constraints
- **Base model stays dense and standalone.** Experts and router are strictly
  additive; with `router = off` or an empty registry, behavior is identical to
  pre-track-12. This is the safety floor that makes growth non-regressive.
- **LARQL plans, training disposes** (same as track 11): the blueprint is a
  static prior; the LoRA training is the ground truth. No per-token graph search.
  `--features larql` is hard-optional; planner falls back to grad/manual.
- **Bounded growth.** `max_experts` + an eviction/merge policy (least-used or
  merge near-duplicate router signatures) prevents unbounded expert sprawl.
  Two near-identical clusters MUST merge, not spawn twins (asserted).
- **Routing safety > coverage.** A wrong expert is worse than none; the
  confidence floor means "apply base only" is always the safe default. Tested.
- **Anti-collapse carries through.** Expert training data is still gated (track
  09 executable gate / track 10 constitution) before it trains an expert.
- ML-free build green: detector, blueprint type, registry, router scaffold,
  config, CLI compile without candle; LoRA training + attribution behind
  `--features train` / `--features larql`.
- Reuses tracks 01 (clustering/simhash), 04 (LoRA preset), 11 (attribution +
  mask). Adds the router as the only net-new runtime concept.

## Acceptance
- Path detector clusters a fixture dataset into ≥2 recurring paths and flags an
  uncovered cluster as a spawn candidate; a cluster already in the registry is
  NOT re-flagged (dedup/merge asserted).
- `ExpertBlueprint` planner produces target modules from attribution
  (`--features larql`) and from `grad` with no LARQL (fallback asserted).
- `experts spawn` trains a LoRA expert on a cluster targeting the blueprint
  modules and registers it; `experts/registry.json` round-trips.
- Router: an input matching an expert's signature routes to it; a low-confidence
  input routes to base-only (no adapter applied) — asserted both ways.
- Safety floor: with `router = off` / empty registry, generation is identical to
  base (back-compat asserted).
- `max_experts` cap + merge: two near-duplicate clusters merge into one expert,
  not two (asserted).
- CLI: `experts plan` (no training), `experts spawn`, `experts list`, and
  `run --experts` all run on a fixture standalone.
- ML-free + `--features train` + `--features "train larql"` build green.
- **Styleguide gates** (code-styleguides.md): expert-spawned rows stamp
  `gen=expert:<path_id>` (§2.4 provenance); the registry + expert artifacts are
  written atomically + content-addressed (§2.3); empty registry / `router=off` is
  byte-identical to base (§2.1 no-ambient-state); spawn is bounded by `max_experts`
  (§2.5). Built per §4.

## Dependencies
Track 11 (attribution selector — shared, produces the blueprint instead of a
mask), track 04 (LoRA preset trains each expert; mask honored), track 01
(clustering for path detection). Consumes gated data from 09/10. The router is
new. Sequenced after 11. Grow-on-demand spawn is daemon-ready (daemon itself
remains the DESIGN.md north-star, not a build track).
