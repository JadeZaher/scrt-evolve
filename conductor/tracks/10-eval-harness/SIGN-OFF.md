# Track 10 — Evaluation Harness — SIGN-OFF

Date: 2026-06-20

The shared evaluation substrate the self-evolve lane (tracks 11/12/15) depends
on: a held-out probe set, a model scorer with comparable metrics, and a pure
accept/regress/catastrophic verdict. Built ONCE, with one owner.

## Delivered

### `eval/gate.rs` — executable gate ✅
`ExecutableGate` (`check_tool_call`, `check_cli`) — the pure correctness
primitive: does an emitted tool call resolve against scrt's real tool schemas
(name + required + no-unknown params), and does a CLI command start with `scrt`
and use only real flags. Typed `GateFailure`/`GateVerdict`. The real scrt flag
surface is lifted into the SDK (was duplicated in `demo/benchmark.py`). Owned
here to break the apparent 10⇄11 cycle. Tests: `gate_accepts_valid_tool_call_and_cli`,
`gate_rejects_malformed`.

### `[eval]` config ✅
`EvalConfig` as `EvolveConfig.eval` (serde-default, `skip_serializing_if`):
`probe_path`, `probe_holdout_frac`, `scorer_backend`, `judge`, `metrics`. Absent
⇒ no eval (lane runs unguarded with a logged warning). **Deviation (deliberate):**
top-level `[eval]` not the spec's `[evolve.eval]` — consistent with the existing
top-level `[discover]`/`[generate]`/`[train]` blocks; the spec text was
self-contradictory ("a field on EvolveConfig" ⟹ `[eval]`). Test:
`eval_config_round_trips_and_absent_is_none`.

### `eval/probe.rs` — ProbeSet ✅
A fixed, versioned, held-out exam. `carve(dataset, holdout_frac)` does a
**deterministic** content-hash split (FNV-1a, no RNG — styleguide §2.2) and
`assert_no_overlap` re-asserts zero training leakage. Content-derived `version`
so any probe change bumps the version (verdict compares only within a version).
Round-trips through `probe.jsonl`. Tests:
`probe_carve_holds_out_with_zero_overlap_and_is_deterministic`,
`probe_overlap_is_detected`, `probe_round_trips_jsonl`.

### `eval/score.rs` — Scorer + ScoreReport ✅
`ScoreReport {correctness, constitution_adherence?, mean_exit_depth?,
perplexity?, n, probe_version, backend}` — `Option` metrics distinguish "0.0"
from "not measured". Two scorers:
- **`ApiScorer<T: ChatTransport>`** (no ML): generate a completion per probe item
  via a chat endpoint, judge with the gate (tool_call/cli) or a normalized
  reference match (qa/instruction). Generic over the transport ⇒ unit-testable
  with a mock model. Tests: `api_scorer_scores_correctness_with_mock_model`,
  `api_scorer_empty_probe_is_uncovered`.
- **`transformers` backend** (real forward pass): `eval::score_transformers`
  shells out to `python -m scrt_evolve_score` — load base+adapter (reusing the
  track-19 infer path), generate, compute **perplexity** + a **logit-lens
  early-exit-depth** proxy. **Deviation (deliberate):** a Python *subprocess*,
  not pyo3 — the architecture the project adopted in track 19 (styleguide:
  "Python invoked as an external subprocess, never linked into the binary"). The
  `pyo3` feature build still compiles (bridge.rs untouched). Smoke-verified:
  `python -m scrt_evolve_score --help` imports cleanly under the torch venv.

### `eval/verdict.rs` — StepVerdict ✅
`classify(baseline, candidate, tolerances) -> StepVerdict` — pure: NaN/below-floor
⇒ `Catastrophic`, drop > tolerance ⇒ `Regress`, else `Accept`. Refuses a
probe-version mismatch (`VerdictError::ProbeVersionMismatch`). The single shared
decision tracks 11/15 both call. Tests: `verdict_classifies_three_outcomes`,
`verdict_refuses_probe_version_mismatch`.

### CLI ✅
`scrt-evolve eval [--probe] [--python]` (score → `work_dir/score.json`) and
`scrt-evolve probe build [--from --holdout --out --remainder]` (carve a probe).

### Python: `python/scrt_evolve_score/` ✅
`score.py` (gate mirror kept in sync with `eval/gate.rs`, perplexity, logit-lens
exit-depth) + `__main__.py` emitting a `ScoreReport` JSON line. Reuses
`scrt_evolve_infer` for loading/generation.

## Deferred / seams (documented, not blocking)
- **constitution_adherence** (judge backend) — left `None`; track 12 owns the
  constitution. The `[eval].judge` field is parsed and reserved.
- **`candle` scorer backend** — `run_eval` returns a clear "not built" error;
  the `--features train` build compiles.
- Real model-on-disk perplexity numbers — exercised by track 22 (the bench), not
  in CI (no model in the unit-test env).

## Verification (one integrated sweep)
- `cargo test` (default, ML-free): green — `tests/eval.rs` 10/10 + all prior suites.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- `cargo build --features train`: green.
- `cargo build --features pyo3`: green.
- `python -m scrt_evolve_score --help` under the torch venv: imports clean.

## Carry-forward
Track 10 unblocks track 15 (self-regulation): the transactional wrapper consumes
`ProbeSet`, `run_eval`/`ScoreReport`, and `StepVerdict::classify` as its
keep|rollback decision. Next track.
