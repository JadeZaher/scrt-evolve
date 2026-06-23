# Track 27 — Config-Driven Export Pipeline — Plan

## Tasks
1. [x] `ExportConfig` + `MergeShardsConfig` in config.rs (additive); `export`
   field on `EvolveConfig`; exported from lib.rs. Defaults: quant Q4_K_M, dtype
   bfloat16, max_shard_size 3GB; everything else Option/None.
2. [x] `scrt_evolve_gguf/merge_shards.py` — `merge_shard_adapters(dir, pattern)`:
   union per-shard safetensors (assert disjoint global-indexed keys) → single
   `adapter.safetensors` + `adapter_config.json` (rank/alpha/targets/base from
   shard jsons). Idempotent; CLI entrypoint too.
3. [x] `export_gguf()` upgraded: `dtype` (bf16 load), `max_shard_size`,
   `merge_shards_pattern` (stage 0), `work_dir` (native-fs scratch), `place_dir`
   (stage 4 copy, non-fatal). `__main__` gains --dtype/--max-shard-size/
   --merge-shards/--work-dir/--place-dir.
4. [x] Rust `cmd_export_gguf` reads `[export]` for defaults (CLI flags override):
   quant/dtype/shard-size/llama_cpp/work/place/merge-shards/keep all plumbed.
5. [x] Tests: Rust `export_config_round_trips_with_merge_shards_and_defaults`;
   Python `test_merge_shards.py` (union / duplicate-key / single-file no-op).
6. [x] bench/evolve.toml `[export]` block (work_path native fs, place_dir = LM
   Studio models dir, merge_shards enabled).
7. [x] Full verification sweep GREEN (cargo test + clippy -D warnings + fmt;
   Python merge_shards + shard + track23; bench toml parses).

## Status
COMPLETE this session. The export chain is config-driven and tested. Sign-off in
this dir. Shipping it (installable Python pkg + CLI binding) is track 28.
