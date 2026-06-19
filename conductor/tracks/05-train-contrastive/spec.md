# Train / Contrastive — Specification

## Goal
DESIGN.md phase 6. Port the **existing InfoNCE embedding-adapter seam** (the
in-tree `crates/scrt-evolve` spike) into this repo as the `contrastive`
`TrainingPreset`. Unlike the instruction presets it consumes **palace
structure directly** (stash note = query, captured nodes = positives, other
stashes' nodes = negatives), and it improves **scrt's own retrieval** rather
than the generative model.

## Scope
- `contrastive.rs`: `TrainingPreset` impl running an InfoNCE loop
  (`[train.contrastive]`: `negatives_per_row`, `temperature`) → an embedding
  adapter (`.safetensors`).
- Input is the `contrastive`-kind dataset rows OR palace structure directly:
  port the in-tree `corpus.rs` export (`{query, positive, negatives[], stash}`)
  + the candle InfoNCE `train.rs` loop from SPIKE-NOTES §1–3.
- The embedding backbone is a per-arch seam (the spike targeted nomic-embed-text
  BERT); reuse `model.rs`'s loader contract where it fits, or a parallel
  embedding loader.
- `scrt-evolve train --preset contrastive` consumes palace structure directly
  (no generate stage needed), and the contrastive adapter feeds scrt's hybrid
  retriever signal 3 (SPIKE-NOTES §A).

## Constraints
- Behind `--features train`.
- This is a **port + lift**, not a redesign: preserve the verified corpus-export
  row shape and the note-as-query / cross-stash-negatives semantics already
  tested in the in-tree crate. The in-tree crate becomes a thin re-export or is
  retired at extraction (track 08).
- The contrastive path's `kind` routing in the `train()` driver must NOT
  collide with the instruction presets (different dataset shape).

## Acceptance
- `train::run` with `preset = "contrastive"` over a fixture palace produces an
  embedding adapter `.safetensors` that re-loads.
- Corpus export reproduces the verified row shape: note-as-query, positives
  from the stash, `negatives_per_row` sampled from OTHER stashes
  (deterministic), empty-note stashes skipped (the in-tree tests carry over).
- Overfit smoke: InfoNCE loss decreases on a tiny fixture.
- `scrt-evolve train --preset contrastive` runs from palace structure with no
  `generate` step.

## Dependencies
Track 04 (`TrainingPreset` trait + `train()` driver), track 01 (palace access).
Lifts code from the in-tree `crates/scrt-evolve` (`config.rs`/`corpus.rs`/
`train.rs`).
