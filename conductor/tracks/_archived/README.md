---
type: Track Archive
title: scrt-evolve — Archived Tracks
description: Unbuilt tracks for the speculative self-evolving / self-architecting vision, archived 2026-06-28.
timestamp: 2026-06-28T00:00:00Z
---

# Archived tracks

These tracks are **archived, not roadmap**. They were specced (spec + plan
exist) but **never shipped a module** — the 2026-06-28 source audit
(`conductor/RETRO.md` §Standing roadmap) confirmed zero implementation. They are
kept here for provenance, not because they are queued.

They are archived because they belong to the **speculative self-evolving /
self-architecting (lexame) vision**, which the project deliberately did *not*
pursue. The shipped product is the **config-driven daemon + branch factory** —
standalone Branch-Train-Merge experts, per-request routing, eval-gated through
the track-15 transaction. That product does not need an in-model self-evolve
lane or a self-architecting DAG/SDK framework; the steering substrate it *does*
need (constitution + taste as prompt-constants) already ships as
`[evolve].constitution`/`taste` + per-`[[goals]]` overrides, composed into
generation via `EvolveConfig::compose_steering()` (the `custom_prompt` seam).

Dir numbers are preserved (no renumbering) — the gaps they leave in the active
`tracks/` spine are intentional and point here.

## What's archived

**Non-LoRA / distributed training presets (superseded):**
- `05-train-contrastive` — InfoNCE embedding-adapter mode. The product trains
  LoRA adapters (the compounding-adapter / branch design); an embedding adapter
  is an orthogonal mode it doesn't use.
- `06-train-full-pretrain` — full-finetune + continued-pretraining. The shipped
  Python trainer is prompt-masked-CE LoRA; full-weight modes don't serve the
  adapter-over-immutable-base product.
- `07-train-shard` — multi-node distributed training. Superseded for the VRAM
  goal by shipped **fractional/microshard** training (track 25), and
  decentralization was contracted out of this repo to hivemind. The candle
  `train/{contrastive,full,pretrain,shard}.rs` modules remain as doc-stubs only
  (`train::run` bails on every non-`lora` preset).

**Meta-objects (superseded by shipped steering):**
- `22-meta-objects` — the constitution/taste meta-object substrate. The steering
  it describes already ships as `[evolve].constitution`/`taste` (+ per-`[[goals]]`
  overrides) composed via `EvolveConfig::compose_steering()` (track 21). No
  separate meta-object module is needed.

**Self-evolve lane (in-model capability growth):**
- `11-regen-antagonist` — model-as-its-own-antagonist regen flywheel + early-exit depth-cheapen.
- `12-self-refine-constitutional` — thesis→shadow-antithesis→synthesis dialectic emitting SFT/DPO rows.
- `13-attribution-training-mask` — `TrainingMask` + reusable `AttributionReport` (which params to update).
- `14-expert-spawn-router` — grow-on-demand MoLE adapter-experts + in-model router. (Superseded by the *standalone* BTM branch factory, track 29.)

**Architecture / SDK lane (self-architecting framework):**
- `16-dag-engine` — typed-DAG substrate; every stage a node, a run a validated `Dag`.
- `17-architecture-self-distill` — QA→planner→DagSpec factory + artifact-distillation library.
- `18-sdk-builder-interface` — trait-powered typestate builder as the primary SDK surface.

## If revived

Any of these can be un-archived by `git mv`-ing the dir back to
`conductor/tracks/<NN-name>/` and re-listing it in `tracks.md`. The specs remain
valid design records; re-validate them against the then-current shipped seams
first (the daemon + branch factory have moved on since these were written).
