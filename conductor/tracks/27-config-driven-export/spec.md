# Track 27 — Config-Driven Export Pipeline — Spec

## Goal
Make the whole post-training export chain — **merge (sharded) adapter → convert
to GGUF → quantize → place** — run from `evolve.toml`, so sharding-merge rules,
format-conversion targets, and source/target/scratch weight paths are
configuration, not ad-hoc scripts. Retro track: BUILT this session after the
manual GGUF export proved the steps (see the granite-gguf bring-up).

## Background
Exporting the evolved Granite by hand exposed every gap: the per-shard adapters
the fractional trainer emits had to be merged by hand; the export loaded base in
float32 (OOM); intermediates written to a 9p `/mnt/c` mount OOM'd/IO-errored;
nothing placed the GGUF where LM Studio looks. All of that is now config.

## Scope (delivered)
- **`[export]` config block** (`ExportConfig`, additive top-level on
  `EvolveConfig`): `quant` (format target), `dtype` (merge-load; bf16 default
  avoids the fp32 OOM), `llama_cpp_path` (conversion SOURCE tooling),
  `work_path` (fast native-fs scratch — NOT 9p), `out_path` (TARGET weight
  file), `place_dir` (deploy copy, e.g. LM Studio), `max_shard_size`,
  `keep_intermediates`, and `[export.merge_shards]` (`enabled` + `pattern`) —
  the sharding-merge rule.
- **`scrt_evolve_gguf.merge_shards`** — generic union of per-shard adapter files
  (global-layer-indexed keys ⇒ collision-free) into one `adapter.safetensors` +
  `adapter_config.json`. Idempotent; rejects duplicate keys; accepts an existing
  single-file adapter as a no-op.
- **`export_gguf()` upgraded**: configurable merge-load `dtype`, `max_shard_size`
  on `save_pretrained`, optional stage-0 shard-merge, a configurable native-fs
  `work_dir` for intermediates, and a stage-4 `place_dir` copy (non-fatal on
  failure so a full disk doesn't lose the export). New CLI flags mirror these.
- **Rust `cmd_export_gguf` reads `[export]`**: config supplies defaults; explicit
  CLI flags still override. Passes dtype/shard-size/work/place/merge through.

## Generic / architecture-level
Nothing model-specific: merge is by key-union, dtype/quant are strings, the
converter is whatever llama.cpp arch support exists. Works for any model the
fractional trainer + a current llama.cpp handle.

## Acceptance (met)
- `EvolveConfig` round-trips `[export]` incl. `[export.merge_shards]`; defaults
  when omitted (quant Q4_K_M, dtype bfloat16, shard 3GB); absent ⇒ None.
- `merge_shard_adapters` unions disjoint shards, rejects collisions, no-ops on a
  single-file adapter (3 Python tests).
- Full sweep green: cargo test (incl.
  `export_config_round_trips_with_merge_shards_and_defaults`) + clippy
  `-D warnings` + fmt; Python merge_shards 3/3 + shard 8/8 + track23 6/6; bench
  `evolve.toml` `[export]` block parses.
- The manual pipeline it replaces is proven end-to-end (4.03GB Q4_K_M GGUF built
  from the evolved Granite this session).

## Out of scope (→ track 28)
Packaging the Python half as an installable artifact, CLI↔package interpreter
binding, env preflight/`doctor`. Track 27 makes the export config-driven; 28
makes it shippable.
