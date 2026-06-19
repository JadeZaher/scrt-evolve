# Evaluation Harness — Probe Set + Scorer (shared) — Plan

## Tasks

1. [ ] `evolve/gate.rs` — the **executable gate** primitive (parse/dry-run a
   `scrt …` command + schema-validate vs `toolspec.rs`). Owned HERE (foundation)
   so track 11 consumes it rather than 10 depending on 11 — breaks the apparent
   cycle. -- evidence: gate accepts valid / rejects malformed test.
2. [ ] `[evolve.eval]` config as a new `Option<EvalConfig>` field on
   `EvolveConfig` (serde-default → non-breaking): `probe_path`,
   `probe_holdout_frac`, `scorer_backend`, `judge`, `metrics`. ML-free
   round-trip; absent → no-eval. -- evidence: config + absent-no-eval test.
3. [ ] `evolve/probe.rs` `ProbeSet`: load `probe.jsonl` deterministically;
   carry `gen` provenance; builder carves a held-out split from a dataset with
   asserted zero-overlap. -- evidence: build + zero-overlap + deterministic test.
4. [ ] `ScoreReport` {correctness, constitution_adherence, mean_exit_depth,
   perplexity?, n, probe_version} + version stamping. -- evidence: report shape test.
5. [ ] `Scorer` `api` backend: generate probe completions via `ApiEndpoint`;
   correctness (via gate) + constitution_adherence (judge endpoint). No ML deps.
   -- evidence: api-scorer report test (mocked endpoint).
6. [ ] `StepVerdict` pure function: (baseline, candidate, tolerances/floors) →
   accept|regress|catastrophic; refuse probe-version mismatch. -- evidence:
   three-outcome + version-mismatch test.
7. [ ] `Scorer` `pyo3` backend: drive HF `transformers` through `bridge.rs` for
   perplexity / exit-depth / logprobs. Behind `--features pyo3`. -- evidence:
   compiles + computes on a tiny model (or seam-surface assertion in CI).
8. [ ] `Scorer` `candle` backend (optional, lower priority), `--features train`.
   -- evidence: compiles behind feature.
9. [ ] Graceful degradation: api backend omits depth/perplexity with a log; no
   probe → "uncovered". -- evidence: degraded-coverage test.
10. [ ] CLI: `eval --probe`, `probe build --from --holdout`. -- evidence: CLI tests.
11. [ ] Final sweep: `cargo build`, `cargo test`, `cargo build --features pyo3`,
    `cargo build --features train`, `cargo clippy`. -- evidence: green.

## Sign-off
Pending.
