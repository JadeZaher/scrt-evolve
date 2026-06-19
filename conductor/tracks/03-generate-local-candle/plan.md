# Generate / Local Candle Backend — Plan

## Tasks

1. [ ] Define the `model.rs` loader seam: a `LoadedModel` + a per-arch loader
   trait (safetensors + tokenizer); ONE concrete arch first. -- evidence: model.rs trait + impl.
2. [ ] Unsupported-arch path returns a clear error. -- evidence: error test.
3. [ ] Implement `LocalCandle` `GenBackend` over candle text-generation with
   sampling (`max_new_tokens`, `temperature`). -- evidence: local.rs.
4. [ ] Reuse `prompts.rs` templates so rows match the API backend exactly. -- evidence: shared-template test.
5. [ ] Dedup + basic quality filter on generated output; optional
   critique/refine pass. -- evidence: degenerate-output filtered test.
6. [ ] Wire `[generate].backend = "local"` / `--backend local` selection. -- evidence: backend-dispatch test.
7. [ ] Tiny fixture model for CI (or a gated small-model download). -- evidence: fixture/test config.
8. [ ] Cross-backend interchangeability: a `LocalCandle` row and an
   `ApiEndpoint` row both validate against `Dataset` schema. -- evidence: parity test.
9. [ ] `scrt-evolve generate --backend local` end-to-end on fixture. -- evidence: CLI test (feature-gated).
10. [ ] Final sweep: `cargo test --features train`, `cargo clippy
    --features train`. -- evidence: green.

## Sign-off
Pending.
