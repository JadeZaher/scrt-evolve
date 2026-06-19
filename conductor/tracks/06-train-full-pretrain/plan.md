# Train / Full + Pretrain — Plan

## Tasks

1. [ ] Factor shared training plumbing out of `lora.rs` (batch iter, optimizer
   step, grad_accum, save/reload) so `full`/`pretrain` reuse it. -- evidence: shared module + lora still green.
2. [ ] `full.rs`: full-weight finetune (`lr`/`epochs`/`grad_accum`) over
   `qa`/`instruction` rows → full-weights artifact. -- evidence: full.rs.
3. [ ] `grad_accum` accumulation across micro-batches. -- evidence: step-vs-optstep-count test.
4. [ ] `pretrain.rs`: causal-LM continued pretraining over `completion`/raw
   passages with `block_size` chunking → weights/adapter. -- evidence: pretrain.rs.
5. [ ] Emit `completion`-kind rows from raw corpus passages (in generate or a
   direct corpus reader) for the pretrain input. -- evidence: completion-row source.
6. [ ] Driver routing: `completion` → `pretrain`, `qa`/`instruction` →
   `full`/`lora`; mixed dataset partitioned correctly. -- evidence: routing test.
7. [ ] Overfit smokes for both presets (loss down, seeded); artifacts reload. -- evidence: two loss-down tests.
8. [ ] `scrt-evolve train --preset full` and `--preset pretrain` standalone. -- evidence: CLI tests.
9. [ ] Final sweep: `cargo test --features train`, `cargo clippy
   --features train`. -- evidence: green.

## Sign-off
Pending.
