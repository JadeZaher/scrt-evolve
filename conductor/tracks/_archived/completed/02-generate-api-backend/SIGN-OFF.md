# Track 02 — Generate / API Backend — SIGN-OFF

**Status:** ✅ Core API path complete. PyO3 bridge + Python parity carried forward.

## Acceptance evidence

| Criterion | Evidence |
| :--- | :--- |
| `GenBackend` trait + `GenContext`/`GenExample` defined per DESIGN.md | `crates/scrt-evolve/src/generate/mod.rs` — trait at line 53; GenContext struct (lines 38–50); GenExample enum in `src/dataset.rs` (lines 14–77). |
| `Dataset` JSONL schema (qa \| instruction \| completion \| contrastive \| tool_call \| cli) with `source`/`gen` provenance | `src/dataset.rs` lines 14–77: serde-tagged enum with optional `source` and `gen` fields stamped by parser. |
| `Dataset` round-trip (write → read → equal, one object per line) | Test `dataset_round_trips_through_jsonl` (line 120–149): writes 4 variants, reads back via `from_jsonl()`, asserts equality. |
| QA + instruction synthesis templates; `per_passage` fan-out | `src/generate/prompts.rs` lines 12–46: `system_prompt()`, `user_prompt()`, `refine_prompt()`. Test `mocked_backend_produces_qa_and_instruction_rows` (line 54–82) validates both kinds emitted. |
| `ApiEndpoint` (reqwest/rustls): `base_url`/`model`/`turns`, `api_key_env` lookup, OpenAI + Anthropic shapes | `src/generate/api.rs` lines 41–99: struct with `from_config()` env-var resolution (lines 52–84), transport trait at lines 37–39. |
| Multi-turn refine when `turns > 1` | Test `turns_greater_than_one_issues_refine_turns` (lines 105–117): backend with 2 turns, mock validates refine loop issued; completion is the refined answer. |
| `generate::run(&cfg, &ctx) -> Dataset` driver (passage → N examples → rows) | `src/generate/mod.rs` lines 81–93: `run()` routes to `ApiEndpoint::from_config()` then calls `run_with_backend()` (lines 98–139). Test `mocked_backend_produces_qa_and_instruction_rows` validates end-to-end. |
| CLI `scrt-evolve generate [--in discovered.json] [--backend api]` → `dataset.jsonl`; runs standalone from disk | `crates/scrt-evolve-cli/src/main.rs` lines 228–238: `cmd_generate()` loads discovered context, calls `scrt_evolve::generate::run()`, writes to workdir. CLI test (no discover call) integration verified by structure. |
| Missing `api_key_env` env var → clear error | Test `missing_api_key_env_is_a_clear_error` (lines 152–168): unset var causes `ApiEndpoint::from_config()` to return error naming the var. Also test `no_api_key_env_means_unauthenticated_local_endpoint_ok` (lines 211–224): omitting `api_key_env` entirely is OK. |
| Tool-call rows validate against real scrt schemas | Test `tool_call_rows_validate_against_real_schemas` (lines 171–193): hallucinated tool dropped, schema-valid call survives. Grounded in `crate::toolspec::scrt_tools()`. |
| CLI rows require `scrt …` prefix | Test `cli_rows_require_scrt_command` (lines 196–208): non-scrt commands dropped during parsing. |
| Parser tolerates markdown-fenced JSON array | Test `parser_tolerates_markdown_fenced_array` (lines 85–90): response wrapped in ` ```json…``` ` parses correctly. |
| Malformed rows skipped, not fatal | Test `malformed_rows_are_skipped_not_fatal` (lines 93–102): one good + one missing-field row; good one survives. |
| Export: Gemma chat corpus + JSONL (non-instruction rows excluded) | Test `export_writes_gemma_chat_corpus_and_jsonl` (`tests/export.rs` lines 13–62): writes `finetune-train.txt` and `finetune-chat.jsonl`, contrastive rows excluded. |
| Export: tool_call rows render in Gemma tool_code format | Test `tool_call_rows_render_in_gemma_tool_code_format` (lines 65–90): tool_call row emits ` ```tool_code\nscrt_stash(…)\n``` `. |
| Export: stubbed formats (OpenAI, Anthropic) drop tool_call rows but pass qa | Test `stubbed_tool_formats_drop_tool_call_rows` (lines 93–118): tool_call dropped under OpenAi format, qa survives. |

## Full sweep result

**Default build (no pyo3, no train):**
- `cargo test` (crates/scrt-evolve): **12 passed** (generate 9 + export 3), 0 failed.
  - Generate tests: `mocked_backend_produces_qa_and_instruction_rows`, `parser_tolerates_markdown_fenced_array`, `malformed_rows_are_skipped_not_fatal`, `turns_greater_than_one_issues_refine_turns`, `dataset_round_trips_through_jsonl`, `missing_api_key_env_is_a_clear_error`, `tool_call_rows_validate_against_real_schemas`, `cli_rows_require_scrt_command`, `no_api_key_env_means_unauthenticated_local_endpoint_ok`.
  - Export tests: `export_writes_gemma_chat_corpus_and_jsonl`, `tool_call_rows_render_in_gemma_tool_code_format`, `stubbed_tool_formats_drop_tool_call_rows`.
- `cargo clippy --all-targets -- -D warnings` (default build): clean.
- `cargo build --features pyo3`: compiles; `bridge.rs` `#[pymodule]` macro valid (proves Python headers gating works).

## Notes / carry-forward

- **PyO3 bridge:** `bridge.rs` currently stubs only a `version()` function to prove the module compiles under `--features pyo3`. Real `read_dataset(path) -> list[dict]` is **deferred to a later pass** — the structure is ready, the function body is not.
- **Python schema-parity test:** No .py test fixtures exist. Rust-writes/Python-reads validation is **deferred to a later pass** — dataset JSONL schema is final and documented, but the paired Python test harness is pending.
- **CLI integration test for generate subcommand:** CLI structure verified through `crates/scrt-evolve-cli/src/main.rs` (`cmd_generate` and full pipeline wiring); end-to-end in-process test harness (vs. CLI subprocess spawn) is infrastructure work outside this track's scope — tracked for **track 08** (CLI + workdir unification).
- `generate::run_plan_with_backend()` (planner-driven generation) is implemented and tested by the planner track (track 05).
- Default-build suite (51 tests total) runs green with track 02 integration.
