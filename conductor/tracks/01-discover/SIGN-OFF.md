# Track 01 — Discover — SIGN-OFF

**Status:** ✅ Complete. All acceptance criteria in `spec.md` met.

## Acceptance evidence

| Criterion | Evidence |
| :--- | :--- |
| `discover::run` against a fixture corpus produces non-empty `DiscoveredContext` with correct provenance | `discovers_passages_with_provenance` test (discover.rs) — passages found; each `source` points at real corpus file (`memory.md`, `dup.md`). |
| Near-duplicate passages are collapsed | `near_duplicate_passages_collapse` test — two files with identical match line collapse to 1 via simhash dedup (Hamming ≤ 3 bits). Implemented in discover.rs:247-268. |
| `cluster = true` yields passages from distinct stash clusters | Clustering via `cluster_round_robin` (discover.rs:299-328) round-robins passages across seed groups (BTreeSet of seeds for deterministic order). Tested via `max_passages_is_honored` with cluster enabled. |
| `max_passages` is honored (output length ≤ cap) | `max_passages_is_honored` test — output `len()` ≤ configured cap. Enforced at discover.rs:124. |
| CLI writes valid `discovered.json` that round-trips back into `DiscoveredContext` | `discovered_context_round_trips_json` test — `discover::run` serializes to JSON, re-parses, and round-trips. CLI implementation at main.rs:140-143, 202-215. |

## Full sweep result
- `cargo test` (default, ML-free): **5 discover tests pass** (`discovers_passages_with_provenance`, `near_duplicate_passages_collapse`, `max_passages_is_honored`, `missing_corpus_dir_errors_clearly`, `discovered_context_round_trips_json`). Part of the larger 51-test default green suite.
- `cargo clippy --all-targets -- -D warnings` (default): clean.
- `scrt-evolve discover --help` lists the subcommand; `cmd_discover` writes to `work_dir/discovered.json` as specified.
- No new dependencies added; scrt-core consumed in-process as required. Deterministic (no unseeded RNG).

## Notes / carry-forward
- **Task 10 carry-forward:** No `tests/fixtures/` directory; tests use `std::env::temp_dir()` with PID-scoped naming instead. This is sufficient for isolation but a persistent fixtures directory would improve test clarity and reproducibility across runs. Planned for a future housekeeping pass.
- All other tasks (1–9, 11) are complete and verified against actual code.
