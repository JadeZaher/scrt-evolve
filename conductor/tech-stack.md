---
type: Tech Stack
title: scrt-evolve — Tech Stack
description: Languages, build, and dependencies for scrt-evolve.
timestamp: 2026-06-28T00:00:00Z
---

# scrt-evolve — Tech Stack

## Language / build
- **Rust** (edition 2021), Cargo workspace: lib crate `scrt-evolve` (the SDK) +
  thin binary `scrt-evolve-cli`. SDK is primary; CLI is an argv→SDK shim.
- ML is **opt-in behind a `train` feature flag**. A default
  `cargo build`/`cargo test` does NOT compile candle (the spike already proved
  this pattern in the in-tree `scrt-evolve` crate).

## Core dependencies
- **scrt-core** — retrieval engine, consumed as a Rust crate (NOT CLI
  subprocess, NOT PyO3). Direct in-process calls to `search_with_meta`, the
  `palace` module, and `palace::simhash`. Interim git dep until scrt is
  published; swap to crates.io is a one-line change (track 08).
- **candle** (0.8) — text generation (local GenBackend) + training loops, all
  behind the `train` feature.
- **safetensors** — model weights load + adapter output.
- **serde / serde_json** — config (toml) and the JSONL dataset contract.
- **anyhow / thiserror** — error ergonomics.
- **tokio / reqwest** (rustls) — async + HTTP for the API GenBackend and the
  shard coordinator/worker transport.

## PyO3 bridge (training-tooling interop)
- A **`pyo3` feature** exposes the dataset + a training-step seam to Python so
  conventional tooling (`transformers`, `peft`, `trl`, `torch`) can consume
  scrt-evolve datasets and drive presets. This is the merge point with the
  existing Python training stack (hivemind-models scripts: sharding,
  coordinator/worker, trainable tensors). Default build does NOT require Python
  headers; built explicitly with `-p` / `--features pyo3` (mirrors how the
  scrt workspace gates `scrt-py`/`scrt-napi`).

## External Python training tooling merged via the bridge
- **hivemind-models** sharding pipeline: `shard_server.py` (serves layer
  forward passes over HTTP, safetensors shard + tensor wire format),
  `expert_coordinator.py` / `moe_coordinator.py` (dispatch expert/MoE compute
  to remote workers), `convert_model.py` / `extract_experts.py` / sharding
  utilities. The shard training preset (track 07) reuses this coordinator+worker
  topology and tensor wire format rather than reinventing it.

## Profiles
- `release`: `lto = "thin"`, `codegen-units = 1`, `strip = true` (matches scrt
  workspace).

## Code styleguides (enforced)
- **[code-styleguides.md](code-styleguides.md)** — Rust style + **durable MPP
  execution** rules (idempotency, determinism, atomic/content-addressed
  artifacts, transactional weight mutation, provenance/quarantine,
  serializable-graph persistence) + **durable mind-palace (scrt)** usage. `[MECH]`
  rules run in CI; `[REVIEW]` rules are referenced by each track's Acceptance
  criteria. New durability-relevant work must satisfy the relevant rule before
  sign-off.
