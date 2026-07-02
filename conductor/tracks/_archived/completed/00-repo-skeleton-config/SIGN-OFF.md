# Track 00 — Repo Skeleton + Config — SIGN-OFF

**Status:** ✅ Complete. All acceptance criteria in `spec.md` met.

## Acceptance evidence

| Criterion | Evidence |
| :--- | :--- |
| `cargo build` + `cargo test` succeed with NO candle / NO pyo3 in the dep tree | Default build green (27.7s); `tests/ml_free_default.rs` asserts via `cargo tree` that `candle-core`, `candle-nn`, `safetensors`, and `pyo3` are absent — all 3 guard tests pass. |
| `cargo build --features pyo3` compiles | Green (12.7s) — `crates/scrt-evolve/src/bridge.rs` `#[pymodule]` stub builds against Python headers. |
| `cargo build -p scrt-evolve --features train` compiles | Green (35.3s) — candle 0.8 pulled only under the feature. |
| `EvolveConfig::load` round-trips the DESIGN example `evolve.toml` | `tests/config.rs::design_example_round_trips` — loads the §Config-schema example field-for-field + serialize→re-parse round-trip. |
| Partial configs (generate-only, train-only) load | `tests/config.rs::partial_generate_only_loads`, `partial_train_only_loads`, `empty_config_loads_with_defaults`. |
| Invalid config rejected (missing `model_path` where required; inline secret in `api_key_env`) | `require_model_path_errors_when_absent`; `inline_secret_with_sk_prefix_is_rejected`, `inline_secret_with_spaces_is_rejected`, `inline_secret_long_token_is_rejected`; `valid_env_var_name_is_accepted` guards against false positives. |
| `scrt-evolve init` writes a valid scaffold + warns on missing model path | `tests/scaffold.rs` + live run: `wrote scaffold to evolve.toml` followed by the missing-`model_path` warning; scaffold re-loads through `EvolveConfig::load`. |

## Full sweep result
- `cargo test` (default): **18 passed** (config 10, ml_free 3, scaffold 2, workdir 2, doctest 1), 0 failed.
- `cargo clippy --all-targets -- -D warnings` (default), `--features train`, `--features pyo3`: all clean.
- `scrt-evolve --help` lists all five subcommands (`init|discover|generate|train|run`).
- `cargo metadata --no-deps` lists both crates (`scrt-evolve`, `scrt-evolve-cli`).

## Notes / carry-forward
- `scrt-core` is pinned as an interim git dep to `JadeZaher/scrt-cli` rev `c768549`. The crates.io swap is a one-line change tracked for **track 08** (noted in root `Cargo.toml`).
- Stage drivers (`discover::run`, `generate::run`, `train::run`) are compiling seams that `bail!` "not implemented yet (track NN)" — filled in by tracks 01–07.
- No ML / Python logic implemented (correct for this track); the `train` and `pyo3` seams are inert stubs that prove the gating.
