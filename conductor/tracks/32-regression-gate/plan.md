---
type: plan
track: 32
title: Regression Gate + min-QA-pairs floor
status: completed
created: 2026-06-28
---

# Track 32 — Regression Gate + min-QA-pairs floor — Plan

> **DONE (2026-06-28).** All 5 phases shipped + tested; full sweep green
> (`cargo test` 0 failures · `clippy --all-targets` clean · `fmt --check` ok ·
> Python compiles). Files: `scrt_evolve_score` (`sample_ab` + `--ab` entry),
> `eval/degrade.rs` (DegradationJudge/LlmDegradationJudge), `eval/verdict.rs`
> (`judge_verdict`), `regulate/txn.rs` (`run_step_judged` + injected `decide`),
> `daemon.rs` (`degrade` hook + `enough_to_train` + min-pairs skip), CLI
> `run_ab_degrade` + `daemon_serve` wiring, config (`[regulate].gate/
> degrade_judge/max_regressed_frac`, `[daemon].min_train_pairs`),
> `bench/ambient.toml` opt-in, RUNBOOK sweep recipe, AGENTS.md. Opt-in: set
> `[regulate].gate = "judge"`. The running daemon must restart to adopt it.

Per the test-once-at-end policy: ALL code lands first, then ONE full sweep
(`cargo test` + `clippy` + `fmt` + touched Python).

## Phase 1 — A/B completion sampler (Python, track-19 lane)
1. [ ] Add `sample_ab(model_path, probe_path, adapter_dir)` to `scrt_evolve_score`
   (or a new `scrt_evolve_ab` module) reusing `load_base_model` / `apply_adapter`
   / `generate`: load base once → BEFORE completions; apply adapter → AFTER
   completions; emit JSON `[{prompt, before, after}]`. Greedy (temp 0) for
   determinism.
2. [ ] A console entry / `-m` runnable so the Rust side can shell to it like
   `scrt_evolve_score`. Input: model + probe + adapter dir; output: JSON triples
   on stdout.

## Phase 2 — Degradation judge (Rust, mirrors LlmRelevanceJudge)
1. [ ] `eval/degrade.rs`: `DegradationTriple { prompt, before, after }`,
   `DegradationReport { n, regressed, items: Vec<bool> }`, trait
   `DegradationJudge`, impl `LlmDegradationJudge<T: ChatTransport>` (batched;
   JSON verdict per item; **errs toward same-or-better** on failure/garble).
2. [ ] Prompt: numbered list of (prompt, BEFORE, AFTER); ask for ONLY a JSON
   array of the item numbers that got WORSE. Parse reuses the relevance-judge
   index-parsing shape.
3. [ ] Unit tests (mock transport): detects worse; passes same/better; err→accept;
   batch boundary.

## Phase 3 — Gate policy wiring (regulate)
1. [ ] `[regulate].gate` enum (`"correctness"` default = today; `"judge"` = new),
   `[regulate].degrade_judge: Option<GenerateApiConfig>`,
   `[regulate].max_regressed_frac` (default 0.0). Config structs + defaults +
   back-compat (`gate` absent ⇒ correctness).
2. [ ] A pure mapper `judge_verdict(report, candidate_correctness, tol,
   max_regressed_frac) -> StepVerdict`: NaN/collapse→Catastrophic;
   regressed_frac > max→Regress; else Accept. Unit-tested.
3. [ ] Thread it into `run_step` (or a sibling `run_step_judged`) WITHOUT touching
   the correctness path: when gate=judge, the daemon supplies the A/B-sampler +
   judge closures; `run_step` uses `judge_verdict` instead of `classify`.
   Catastrophe/quarantine/halt/log unchanged.

## Phase 4 — Min-QA-pairs floor
1. [ ] `[daemon].min_train_pairs` (default 4). Pure `enough_to_train(batch_len,
   min) -> bool`.
2. [ ] In `daemon.rs`: a popped batch below the floor → record a skipped step
   (note: "below min_train_pairs — accumulating") and DON'T train; non-drain mode
   idles. Composes with Q5 (the rows stay queued).
3. [ ] Tests: skip below floor; train at/above floor.

## Phase 5 — Empirical min-N derivation
1. [ ] Document the default + reasoning in spec (DONE in spec).
2. [ ] `bench/` sweep recipe: vary `min_train_pairs ∈ {1,2,4,8}`, compare the Q4
   trend slope + judge regress rate; pick the smallest non-degrading N. Add as a
   short RUNBOOK note (no code — a documented procedure).

## Final
- [ ] ONE sweep: `cargo test` + `clippy --all-targets` + `fmt --check` + Python.
- [ ] `tracks.md` Build status + Tracks row for 32.
- [ ] `src/AGENTS.md` notes for `eval/degrade.rs` + the gate policy.
- [ ] Update `bench/ambient.toml` with the new `[regulate].gate`/`min_train_pairs`
  (commented, opt-in) so the running daemon can adopt it on restart.
