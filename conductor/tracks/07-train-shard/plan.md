# Train / Shard — Plan

## Tasks

1. [ ] Port/replicate the hivemind tensor wire format
   (`[shape_len:u32][shape…][dtype_len:u32][dtype][data]`) in Rust; golden-vector
   parity test against `shard_server.py`'s packer. -- evidence: wire-parity test.
2. [ ] Define `[train.shard]` config (`role`, `peers`, `shard_strategy`,
   `base_preset`) + `shard.rs` `TrainingPreset` skeleton. -- evidence: shard.rs.
3. [ ] Coordinator↔worker transport (tokio + reqwest/WS): `/register`,
   `/health`, forward/delta endpoints mirroring the hivemind relay. -- evidence: transport test.
4. [ ] **Data sharding:** coordinator splits the dataset across peers; each
   worker runs `base_preset` locally; coordinator averages deltas → merged
   adapter. -- evidence: 2-worker merge test (merged adapter reloads).
5. [ ] **Layer sharding (smoke):** route a forward pass across ≥2 layer-shard
   servers; compare to co-located output within tolerance. -- evidence: distributed-vs-colocated test.
6. [ ] PyO3 bridge: drive the Python `shard_server.py`/coordinator from the
   Rust preset (or accept a Python worker speaking the wire format). -- evidence: rust↔python interop test.
7. [ ] Straggler policy: timeout + drop-the-laggard (no hang). -- evidence: straggler test.
8. [ ] `scrt-evolve train --preset shard` brings up the local fixture cluster
   and completes. -- evidence: CLI end-to-end test.
9. [ ] Document the small-trusted-cluster v1 bar + the byte-parity contract
   with hivemind. -- evidence: CATALOG.md / track notes.
10. [ ] Final sweep: `cargo test --features train`, pyo3 interop test,
    `cargo clippy --features train`. -- evidence: green.

## Sign-off
Pending.
