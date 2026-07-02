# Track 27 — Config-Driven Export Pipeline — SIGN-OFF

Date: 2026-06-21

The post-training export chain (merge sharded adapter → convert → quantize →
place) is now driven entirely by the `[export]` block in `evolve.toml`. The
ad-hoc scripts used to hand-export the evolved Granite this session are replaced
by first-class, tested, config-driven code.

## Delivered
- `[export]` config: quant / dtype / llama_cpp_path / work_path / out_path /
  place_dir / max_shard_size / keep_intermediates + `[export.merge_shards]`
  (enabled, pattern). Additive; absent ⇒ CLI-flag defaults.
- `merge_shards.py`: generic, idempotent union of per-shard adapters (the exact
  manual step that joined the 5 fractional shards into one adapter.safetensors).
- `export_gguf()`: bf16-by-default merge load (kills the float32 OOM), shard-size
  cap, native-fs scratch dir (kills the 9p OOM/IO-error), optional shard-merge
  stage, and a place-into-deploy-dir stage (the LM Studio handoff).
- Rust `cmd_export_gguf` wired to `[export]`.

## Honest limit
This makes the export config-driven and tested in unit form; the heavy
end-to-end (real model load + 14GB convert) was validated MANUALLY this session
(4.03GB Q4_K_M GGUF produced). The config path was exercised by unit tests +
bench-toml parse, not yet by a fresh full run through the Rust CLI (that needs
the venv interpreter binding from track 28 to be ergonomic outside a checkout).

## Verification
- cargo test (all suites + export_config round-trip) ✓; clippy -D warnings ✓;
  fmt ✓.
- Python: test_merge_shards 3/3; regression test_shard 8/8 + test_track23 6/6.
- bench/evolve.toml `[export]` parses.

## Next
Track 28 (packaging & portability) makes this shippable: installable
scrt-evolve-ml package + CLI↔package interpreter binding (retire the
`python_pkg_dir()` checkout hack) + `doctor` preflight.
