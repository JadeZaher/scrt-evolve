---
type: Workflow
title: scrt-evolve — Workflow
description: Development methodology and track workflow for scrt-evolve.
timestamp: 2026-06-28T00:00:00Z
---

# scrt-evolve — Workflow

## Methodology
TDD where it pays (config validation, discover, dataset round-trip, the
Python/Rust dataset-contract parity). ML training loops are validated by
**shape + overfit-a-tiny-batch + artifact-loads** checks, not by chasing a
quality number (quality is the deferred, gated experiment — see DESIGN.md
§Honest risks).

## Track conventions
- Each track is `conductor/tracks/<NN-slug>/` with `spec.md` (goal / scope /
  constraints / acceptance / dependencies) and `plan.md` (numbered tasks,
  each closed with **evidence** — `file:line` + what proves it). On completion
  add `SIGN-OFF.md`.
- Tracks are **review-gated**: each maps to a DESIGN.md build-order phase and
  must compile + pass its own tests before the next track starts. The phase
  gating in DESIGN.md §Build order is the source of truth for ordering.

## Test policy
Apply all changes in a track, then run the full sweep ONCE at the end
(`cargo test`, `cargo test --features train`, `cargo clippy`). Don't loop
test→fix→test per change.

## Stage independence (load-bearing)
The work-dir artifacts (`discovered.json` → `dataset.jsonl` →
adapter/weights) are the contract between stages. Every stage must be
runnable standalone against an on-disk input, so you can stop, inspect, edit,
and resume. Don't couple stages through in-memory-only state.
