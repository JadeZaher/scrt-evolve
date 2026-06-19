# Train / Shard — Specification

## Goal
DESIGN.md phase 8 — **deliberately last and deliberately scoped**. Add the
`shard` `TrainingPreset`: decentralized training split across a small trusted
set of nodes (coordinator + workers, gradient/param exchange), producing merged
weights. Reuse the **existing hivemind sharding pipeline + tensor wire format +
coordinator/worker topology** (the Python stack in
`hivemind-models/scripts/`) via the PyO3 bridge rather than reinventing it.

## Scope
- `shard.rs`: `TrainingPreset` impl driven by `[train.shard]`:
  - `role = coordinator | worker`,
  - `peers = ["host:port", …]`,
  - `shard_strategy = data | layer`,
  - `base_preset = lora` (what each shard runs locally — reuse tracks 04/06).
- **Reuse the hivemind machinery** (the merge point the user asked for):
  - the **tensor wire format** from `shard_server.py`
    (`[shape_len:u32][shape…][dtype_len:u32][dtype][data]`) — match it
    byte-for-byte so Rust coordinator ↔ Python workers interop.
  - the **coordinator/worker topology** from `expert_coordinator.py` /
    `moe_coordinator.py` / `shard_server.py` (HTTP/WS forward-pass dispatch,
    `/register`/`/health`, region/peer registry).
  - **data sharding** = split the dataset across workers, each runs
    `base_preset` locally, coordinator averages deltas. **layer sharding** =
    map model layers to shard servers (the `shard_layers_a_b.safetensors`
    split), forward/backward routed across them.
- The PyO3 bridge lets the Rust `shard` preset stand up / drive the Python
  shard servers + coordinator, or a worker can be a Python process speaking the
  shared wire format.
- Transport: `tokio` + `reqwest`/WS for coordinator↔worker (mirrors hivemind's
  relay).

## Constraints
- Behind `--features train` (and `--features pyo3` for the Python-stack drive).
- **v1 bar is a small trusted cluster, NOT Byzantine fault tolerance** and NOT
  a robust distributed system (DESIGN.md §Non-goals + §Honest risks).
  Stragglers/failures get a basic timeout + drop-the-laggard policy, not
  consensus.
- Wire-format **byte parity** with the hivemind Python stack is load-bearing —
  a Rust-packed tensor must unpack identically in `shard_server.py` and vice
  versa (this is the same byte-parity discipline scrt-core holds with
  Node-mpg).
- Don't fork the Python scripts into this repo; **drive them across the
  bridge**. Reference them; the hivemind repo stays the source of truth for the
  serving topology.

## Acceptance
- **Tensor wire parity:** a Rust-packed tensor round-trips through
  `shard_server.py`'s unpack (and back) bit-identically (golden-vector test
  against the Python packer).
- **Data sharding:** a `shard` run with `role=coordinator` + ≥2 local worker
  processes (Rust and/or Python) splits a fixture dataset, each worker trains
  `base_preset=lora`, the coordinator merges deltas → merged adapter that
  reloads.
- **Layer sharding (smoke):** a forward pass routed across ≥2 layer-shard
  servers reproduces a co-located run within tolerance (the hivemind
  "bit-identical co-located vs distributed" check, applied to one slice).
- Straggler policy: a worker that times out is dropped without hanging the run.
- `scrt-evolve train --preset shard --config evolve.toml` brings up the
  topology and completes on the local fixture cluster.

## Dependencies
Track 06 (`full`/`pretrain` as `base_preset` options) + track 04 (`lora`
base_preset, `TrainingPreset` trait). PyO3 bridge from tracks 02/04. External:
the `hivemind-models` sharding scripts (`shard_server.py`,
`expert_coordinator.py`, `moe_coordinator.py`, the tensor wire format).
