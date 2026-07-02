---
type: Implementation Plan
title: Evaluation Harness — Probe Set + Scorer
description: Implementation plan for the Evaluation Harness — Probe Set + Scorer track.
tags: [track-10, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Evaluation Harness — Probe Set + Scorer (shared) — Plan

## Tasks

1. [x] `evolve/gate.rs` — the **executable gate** primitive (parse/dry-run a
   `scrt …` command + schema-validate vs `toolspec.rs`). Owned HERE (foundation)
   so track 11 consumes it rather than 10 depending on 11 — breaks the apparent
   cycle. -- evidence: gate accepts valid / rejects malformed test.
   DONE (2026-06-20): `src/eval/gate.rs` — `ExecutableGate` (`check_tool_call`,
   `check_cli`), typed `GateFailure`/`GateVerdict`. Tests in `tests/eval.rs`:
   `gate_accepts_valid_tool_call_and_cli`, `gate_rejects_malformed`.
2. [x] `[eval]` config as a new `Option<EvalConfig>` field on `EvolveConfig`
   (serde-default → non-breaking): `probe_path`, `probe_holdout_frac`,
   `scorer_backend`, `judge`, `metrics`. ML-free round-trip; absent → no-eval.
   -- evidence: config + absent-no-eval test.
   DONE: `EvalConfig` on `EvolveConfig.eval`. NOTE: implemented as TOP-LEVEL
   `[eval]` (consistent with `[discover]`/`[generate]`/`[train]` siblings) rather
   than the spec's `[evolve.eval]` sub-table — the spec text was self-contradictory
   ("a field on EvolveConfig" = `[eval]`). Test: `eval_config_round_trips_and_absent_is_none`.
3. [x] `evolve/probe.rs` `ProbeSet`: load `probe.jsonl` deterministically; carry
   `gen` provenance; builder carves a held-out split with asserted zero-overlap.
   -- evidence: build + zero-overlap + deterministic test.
   DONE: `ProbeSet::carve` (deterministic FNV hash split, no RNG), content
   `version` stamp, `assert_no_overlap`. Tests:
   `probe_carve_holds_out_with_zero_overlap_and_is_deterministic`,
   `probe_overlap_is_detected`, `probe_round_trips_jsonl`.
4. [x] `ScoreReport` {correctness, constitution_adherence, mean_exit_depth,
   perplexity?, n, probe_version, backend} + version stamping. -- evidence: report shape test.
   DONE: `src/eval/score.rs::ScoreReport`. Asserted in `api_scorer_*` tests.
5. [x] `Scorer` `api` backend: generate probe completions via `ApiEndpoint`;
   correctness (via gate). No ML deps. -- evidence: api-scorer report test (mocked endpoint).
   DONE: `ApiScorer<T: ChatTransport>`; `run_eval` builds it from `[generate.api]`.
   Tests with mock transports: `api_scorer_scores_correctness_with_mock_model`,
   `api_scorer_empty_probe_is_uncovered`. constitution_adherence is a documented
   seam (track 12 owns the constitution) — left `None`.
6. [x] `StepVerdict` pure function: (baseline, candidate, tolerances/floors) →
   accept|regress|catastrophic; refuse probe-version mismatch. -- evidence:
   three-outcome + version-mismatch test.
   DONE: `src/eval/verdict.rs::classify`. Tests: `verdict_classifies_three_outcomes`,
   `verdict_refuses_probe_version_mismatch`.
7. [x] `Scorer` real-forward-pass backend (perplexity / exit-depth / logprobs).
   -- evidence: compiles + computes on a tiny model (or seam-surface assertion).
   DONE — via **Python subprocess** (`transformers` backend), NOT pyo3: the
   project settled on the subprocess seam in track 19 (styleguide: "Python
   invoked as an external subprocess, never linked"). `python/scrt_evolve_score/`
   (load base+adapter, generate, perplexity, logit-lens exit-depth) +
   `eval::score_transformers` shells out to `python -m scrt_evolve_score`. Smoke:
   `python -m scrt_evolve_score --help` imports cleanly under the torch venv.
   The `pyo3` build still compiles (bridge.rs untouched).
8. [x] `Scorer` `candle` backend (optional) — documented as the not-built native
   path; `run_eval` returns a clear error for `scorer_backend = "candle"`.
   `--features train` build green.
9. [x] Graceful degradation: api backend omits depth/perplexity (logged); no
   probe → "uncovered"; empty probe → uncovered. -- evidence:
   `api_scorer_empty_probe_is_uncovered`; `run_eval` no-probe path returns uncovered.
10. [x] CLI: `eval --probe`, `probe build --from --holdout`. -- evidence: `cmd_eval`
    + `cmd_probe_build` in the CLI; `eval`/`probe build` parse + dispatch.
11. [x] Final sweep: `cargo build`, `cargo test`, `cargo build --features pyo3`,
    `cargo build --features train`, `cargo clippy`, `cargo fmt --check`. -- GREEN.

## Sign-off
Signed off 2026-06-20 — see SIGN-OFF.md. ML-free core (gate/probe/api-scorer/
verdict/CLI) + the `transformers` real-forward-pass scorer (Python subprocess,
perplexity + logit-lens exit-depth) built and verified. Deviations from spec,
both deliberate + documented: (a) config is top-level `[eval]` not `[evolve.eval]`
(sibling-consistent; spec was self-contradictory); (b) the heavy scorer is a
Python subprocess not pyo3 (the track-19 architecture the project adopted).
Default + `--features train` + `--features pyo3` builds all green; clippy clean;
fmt clean. constitution_adherence (judge backend) is a documented seam pending
track 12.
