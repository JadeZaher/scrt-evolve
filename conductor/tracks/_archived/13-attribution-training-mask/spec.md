---
type: Track Spec
title: Attribution-Guided Training Mask
description: A cross-cutting pre-step deciding which slices of the model to train.
tags: [track-13, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Attribution-Guided Training Mask (Tier-1, all paths) — Specification

## Goal
A **cross-cutting pre-training-step** that decides **which slices of the model to
update** (layers / modules), so every training path trains *fewer* parameters
and therefore *faster*, without losing the target behaviors. The selection
signal is **attribution**: which layers/FFN modules actually carry the behaviors
the dataset + constitution are trying to improve. LARQL's `TRACE … FOR <token>`
supplies a cheap static attribution prior (optional); a gradient/heuristic
fallback works without it. This is the "faster training" the daemon triggers
before any preset runs.

> **Naming guard.** This is NOT distributed sharding (that is track 07
> `shard`, = nodes). This is *parameter selection* — a **training mask** over
> the model's own layers/modules. The research framed it as "what shards to
> train"; the buildable meaning is "what layers to update."

> **Honest scope (research due-diligence,
> `.omc/research/larql-regen-swap-2026-06-17.md`).** LARQL attribution is
> *static* (the Logit Lens limitation): it ignores the live, attention-shaped
> residual stream, so it is a **coarse prior, not ground truth**. It must
> PROPOSE a mask that gradient signal can refine — never dictate it. No
> per-token graph search. Absence of a vindex degrades to the gradient/heuristic
> selector, never an error.

> **Cogency-audit notes (apply throughout).**
> - **Config host (A1).** `[train.mask]` is a new `Option<MaskConfig>` field on
>   the EXISTING `TrainConfig` struct (serde-default → non-breaking).
> - **Single attribution owner (C6).** This track OWNS the attribution pass and
>   emits a reusable `AttributionReport`. Track 14 (experts) CONSUMES that report
>   to build expert blueprints — it does NOT run its own `TRACE`. One attribution
>   pass, two consumers (mask here, blueprint there).
> - **Mutable-weight prerequisite (A2) + directive.** Honoring a mask (freeze
>   out-of-mask params) requires a real trainable model, which `LoadedModel` does
>   not yet provide. Per the user directive, masked training is applied through
>   the **PyO3→`peft`/`transformers` bridge** (peft already supports
>   target-module / layer freezing natively), so the mask maps onto peft's
>   `target_modules` + `modules_to_save` rather than a hand-built candle freeze.
>   The candle freeze is an optional later path.
> - **Eval.** Any "does the mask preserve quality" check uses track 10's
>   `Scorer`, not a bespoke evaluator (out of scope here regardless).

## Scope
- **`train/mask.rs` — `TrainingMask`**: a set of {layer, module} entries marking
  what is trainable; the complement is frozen. A `TrainingMask::full()` (train
  everything) is the always-valid default and the no-attribution fallback.
- **Mask selectors** (pluggable; pick via config):
  - `full` — no masking (today's behavior; the safe default).
  - `attribution` (`--features larql`) — run LARQL `TRACE … FOR <target>` over a
    sample of the dataset's target tokens (tool-call tokens, synthesis tokens),
    aggregate per-layer/module attribution, select the top-k contributors.
  - `grad` — a one-pass gradient/Fisher-magnitude proxy over a dataset sample
    (no LARQL): select modules with the largest update signal. The fallback that
    makes the feature work without a vindex.
  - `manual` — explicit layer/module list from config.
- **Tier-1, all-paths integration**: the `train::run` driver computes a
  `TrainingMask` ONCE per run (selector from config) and passes it to whichever
  `TrainingPreset` is active. Presets honor the mask by **freezing** out-of-mask
  parameters (LoRA: only inject adapters on in-mask modules; full/pretrain:
  zero/skip grads outside the mask; regen depth-cheapen: mask composes with the
  early-exit head; dpo: same). The `TrainingPreset` trait gains a `&TrainingMask`
  argument (additive; `full()` preserves current behavior).
- **`[train.mask]` config**: `selector` (full|attribution|grad|manual),
  `top_k` / `coverage` (how aggressive), `min_layers` (floor so the mask is never
  degenerate), `modules` (for `manual`), `sample_size` (dataset rows to attribute
  over), `refine_with_grad` (let one grad pass adjust an attribution mask).
- **Mask provenance + cost report**: emit `work_dir/training-mask.json`
  {selector, trainable_params, total_params, frozen_fraction, selected[],
  attribution_source}. The frozen_fraction is the "faster training" evidence.
- **Daemon-triggerable**: the mask step is a pure function of (dataset, model,
  config) — no interactivity — so an on-demand/daemon run computes it
  automatically before training. (Daemon itself is the DESIGN.md north-star, not
  built here; this track just guarantees the step is non-interactive + resumable.)
- **Safety/correctness floor**: `min_layers` and a mandatory inclusion of the
  output/embedding-adjacent modules prevent a pathological mask that can't learn.
  Validated.

## Constraints
- **Attribution PROPOSES, gradient/learning DISPOSES.** The mask is a prior;
  `refine_with_grad` lets actual training signal correct it. Never treat static
  LARQL attribution as authoritative.
- **`full()` is always valid and is the default.** Masking must be strictly
  opt-in; with no `[train.mask]` block, behavior is identical to pre-track-11.
  This is what makes it safe to bundle into ALL paths.
- **LARQL is a hard-optional `--features larql`** (shared with tracks 09/10):
  fully removable; `grad`/`manual`/`full` selectors cover the no-LARQL case.
- ML-free build stays green: trait change, mask type, config, and report
  compile without candle; selector bodies (grad/attribution) behind their
  features.
- Mechanical validation only: a non-trivial mask freezes a measurable fraction
  of params; a masked LoRA run injects adapters ONLY on in-mask modules; a
  masked full-finetune leaves out-of-mask grads at zero; `min_layers` floor
  holds; `full()` run is byte-identical in selected-set to "train everything".
  Whether masking *preserves quality* is a follow-on experiment, OUT of scope.

## Acceptance
- `TrainingMask::full()` selects all trainable modules; a `manual` mask selects
  exactly the configured set; `min_layers` floor rejects/expands a degenerate
  mask (asserted).
- `grad` selector produces a non-trivial mask over a seeded fixture (frozen
  fraction > 0, < 1) with NO LARQL feature. -- behind `--features train`.
- `attribution` selector (`--features larql`) produces a mask from `TRACE`
  attribution over a fixture; with no vindex it falls back to `grad`/`full`
  without error.
- The `TrainingPreset` trait carries `&TrainingMask`; LoRA injects adapters only
  on in-mask modules (asserted count); a `full()` mask reproduces current
  injection exactly (back-compat).
- `training-mask.json` is emitted with frozen_fraction and selected modules.
- The mask step runs non-interactively as part of `train::run` for every preset
  (full|lora|pretrain|dpo|regen) on a fixture.
- ML-free + `--features train` + `--features "train larql"` all build green.
- **Styleguide gates** (code-styleguides.md): the `AttributionReport` is a
  reproducible function of (dataset sample, model, cfg) under a fixed seed (§2.1
  idempotency, §2.2 determinism); `full()` is byte-identical to current behavior
  (§2.1 no-ambient-state); attribution is a prior, gradient disposes (no
  authoritative static state). Built per §4.

## Dependencies
Track 04 (`TrainingPreset` trait + LoRA injection it amends; model loader).
Composes with track 06 (full/pretrain honor the mask), track 09 (regen
depth-cheapen composes with the mask), track 10 (dpo honors the mask). Shares the
optional `larql` feature with 09/10. Tier-1: sequenced right after 04 so later
presets are built mask-aware from the start; revisit ordering vs 06/09/10 at
implementation.
