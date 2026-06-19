# Train / LoRA — Plan

## Tasks

1. [ ] Define `TrainingPreset` trait (assoc `Config`, `train()`),
   `TrainReport`, and the `train::run` driver that routes dataset `kind` →
   preset. -- evidence: train/mod.rs.
2. [ ] Dataset batch iterator over `qa`/`instruction` rows (tokenize via the
   model tokenizer, mask prompt tokens for loss). -- evidence: batching test.
3. [ ] LoRA injection: wrap `target_modules` projections with rank-`r`
   adapters (`alpha` scaling). -- evidence: injected-layer count/shape test.
4. [ ] Training loop: forward, loss, backward, optimizer step (candle),
   `epochs`/`lr`. -- evidence: loop runs N steps.
5. [ ] Save `adapter.safetensors`; reload + shape-check. -- evidence: save/reload test.
6. [ ] Overfit-tiny-batch smoke: loss decreases over steps (seeded). -- evidence: loss-down test.
7. [ ] PyO3 training-step seam: dataset + batch iterator + step/save hook
   exposed under `--features pyo3`. -- evidence: pyo3 surface compiles.
8. [ ] Python parity: a `peft`/`trl`-style script trains one step on the same
   dataset and saves a compatible adapter. -- evidence: python bridge test.
9. [ ] `scrt-evolve train --preset lora [--data dataset.jsonl]` standalone. -- evidence: CLI test.
10. [ ] Final sweep: `cargo test --features train`, pyo3 bridge test,
    `cargo clippy --features train`. -- evidence: green.

## Sign-off
Pending.
