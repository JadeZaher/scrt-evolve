# Attribution-Guided Training Mask (Tier-1, all paths) — Plan

## Tasks

1. [ ] `train/mask.rs`: `TrainingMask` ({layer,module} set), `TrainingMask::full()`
   default, `min_layers` floor + mandatory output/embedding inclusion.
   -- evidence: full()/manual()/floor tests.
2. [ ] `[train.mask]` config: `selector`, `top_k`/`coverage`, `min_layers`,
   `modules`, `sample_size`, `refine_with_grad`. ML-free round-trip; absent block
   = `full()`. -- evidence: config + default-is-full test.
3. [ ] Trait change: `TrainingPreset::train(&self, model, data, &TrainingMask,
   cfg)` (additive arg; `full()` preserves behavior). Update existing presets'
   signatures. -- evidence: workspace compiles; full() back-compat test.
4. [ ] `grad` selector: one-pass gradient/Fisher-magnitude proxy over a dataset
   sample → non-trivial mask, NO LARQL. Behind `--features train`. -- evidence:
   frozen-fraction in (0,1) test.
5. [ ] `manual` selector: mask from config module list. -- evidence: exact-set test.
6. [ ] LoRA honors mask: inject adapters ONLY on in-mask modules; `full()`
   reproduces current injection. -- evidence: injected-count + back-compat test.
7. [ ] full/pretrain honor mask: out-of-mask grads zeroed/skipped. -- evidence:
   masked-grad test.
8. [ ] `train::run` computes the mask ONCE per run (selector from config) and
   passes it to the active preset — tier-1, all paths, non-interactive.
   -- evidence: run-with-mask test across presets.
9. [ ] `attribution` selector (`--features larql`): `TRACE … FOR <target>` over
   sampled target tokens → per-layer/module aggregation → top-k mask; degrade to
   grad/full with no vindex. -- evidence: attribution-mask + no-vindex-fallback test.
10. [ ] `refine_with_grad`: one grad pass adjusts an attribution mask (prior →
    refined). -- evidence: refinement-changes-mask test.
11. [ ] `training-mask.json` report {selector, trainable/total params,
    frozen_fraction, selected[], attribution_source}. -- evidence: report shape test.
12. [ ] Final sweep: `cargo build`, `cargo test`, `cargo test --features train`,
    `cargo build --features "train larql"`, `cargo clippy --features train`.
    -- evidence: green.

## Sign-off
Pending.
