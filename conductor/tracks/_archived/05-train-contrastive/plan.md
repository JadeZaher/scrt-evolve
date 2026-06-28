---
type: Implementation Plan
title: Train / Contrastive
description: Implementation plan for the Train / Contrastive track.
tags: [track-05, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Train / Contrastive — Plan

## Tasks

1. [ ] Lift the in-tree `corpus.rs` export (palace → `{query, positive,
   negatives[], stash}` JSONL) into this repo's contrastive path; reuse the
   `contrastive`-kind `Dataset` row. -- evidence: corpus export module + carried-over tests.
2. [ ] Port the verified semantics: note-as-query, positives from own stash,
   `negatives_per_row` from OTHER stashes (deterministic, index-strided),
   empty-note skip. -- evidence: the 5 in-tree tests pass here.
3. [ ] Embedding backbone loader (per-arch seam; nomic-embed-text BERT first). -- evidence: embed-load test.
4. [ ] `contrastive.rs` `TrainingPreset`: InfoNCE loop (`negatives_per_row`,
   `temperature`) → embedding adapter `.safetensors`. -- evidence: contrastive.rs.
5. [ ] Route `preset = "contrastive"` in the `train()` driver to consume
   palace structure / `contrastive` rows (no collision with instruction
   presets). -- evidence: driver routing test.
6. [ ] Save + reload the embedding adapter (shape-check). -- evidence: save/reload test.
7. [ ] Overfit smoke: InfoNCE loss decreases on a tiny fixture. -- evidence: loss-down test.
8. [ ] `scrt-evolve train --preset contrastive` from palace structure,
   no generate stage. -- evidence: CLI test.
9. [ ] Final sweep: `cargo test --features train`, `cargo clippy
   --features train`. -- evidence: green.

## Sign-off
Pending.
