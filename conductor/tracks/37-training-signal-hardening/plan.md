---
type: plan
track: 37
title: Training-signal hardening — implementation plan
status: planned
created: 2026-07-02
depends_on: [35, 32, 31, 29, 26, 15, 9]
---

# Track 37 — Training-signal hardening — Plan

Per the test-once-at-end policy (workflow.md §Test policy): all phases land
first, then ONE full sweep (`cargo test`, `cargo test --features train`,
`cargo clippy`, Python contract tests) at the end. Every task closes with
**evidence** (`file:line` + what proves it). Track-15 keep|rollback semantics
and the stage-independence rule (every stage runnable standalone against
on-disk artifacts) are invariants throughout.

## Overview

| Phase | Delivers | Closes finding |
|:--|:--|:--|
| 0 | Dataset contract v1.1 (RowMeta) | foundation for A/B |
| A | Outcome signal at ingest + retry-collapse | 1, part of 3 |
| B | Per-pair data judge + manifest rollup + tier | 2, lexame trust |
| C | High-throughput synthesis + domain params + planner fixes | 3, 4, 5 |
| D | Steerable loop (absorbs track 35) + compliance metric | 7, track 35 |
| E | Training-signal fixes (end_task default, live calib, rotary) | 6 |

Ordering: 0 → A → B strictly (B stamps fields 0 defines onto rows A produces).
C depends on B (the judge ranks candidates). D depends on B+C (the knobs it
exposes). E is independent after B (calib sourcing reads the queue) and can
land in parallel with C/D.

---

## Phase 0 — Dataset contract v1.1

Goal: additive optional row metadata, cross-language safe.

1. [ ] **Task:** Decide flatten vs per-variant. Prototype `#[serde(flatten)]
   meta: RowMeta` on the internally-tagged `GenExample`
   (`crates/scrt-evolve/src/dataset.rs:16`); if round-trip of legacy rows (no
   meta) and new rows (full meta) is lossless under the `kind` tag, use
   flatten; else add per-variant optional fields mirroring the existing
   `source`/`gen` pattern (`dataset.rs:20-23`). Fields: `outcome`
   (`success|failure|unknown`), `judge_score: Option<f32>`, `judge_verdict`
   (`keep|drop|unjudged`), `tier` (`private|shared`), `chosen_over:
   Option<String>` — all `#[serde(default, skip_serializing_if = ...)]`.
   Evidence: `dataset.rs` round-trip test — a v1.0 JSONL line parses
   unchanged and re-serializes byte-identical; a v1.1 line round-trips all
   five fields.
2. [ ] **Task:** Python reader parity: extend the dataset contract test so
   `scrt_evolve_train`'s loader tolerates + preserves the new optional keys
   (they must not break tokenization/collation).
   Evidence: Python contract test reads a v1.1 fixture JSONL and trains a
   fixture step (existing pattern) without error.
3. [ ] **Task:** Document contract v1.1 in `dataset.rs`'s module doc + the
   dataset section of `crates/scrt-evolve/src/AGENTS.md` (additive-optional =
   non-breaking; `chosen_over` is the recorded-not-trained DPO contract).
   Evidence: doc sections present; `git diff` shows module doc updated at
   `dataset.rs:1-6`.

---

## Phase A — Outcome signal at ingest

Goal: mined rows carry ground truth; failures stop minting training rows.

1. [ ] **Task:** Parse `tool_result` correlation in `interaction_log_rows`
   (`crates/scrt-evolve/src/ingest.rs:29`): index `tool_use.id` → the
   following user-entry `tool_result` block; derive `outcome` from `is_error`,
   plus Bash heuristics (exit-code text, error-adjacency) — err toward
   `unknown`. Stamp rows.
   Evidence: unit test with a fixture JSONL containing a failing + succeeding
   tool pair asserts `outcome` on both rows; `ingest.rs` test module.
2. [ ] **Task:** Retry-collapse: within a transcript, N failed attempts of a
   ~same command (normalized-prefix similarity, NOT equality) followed by a
   success collapse to ONE `outcome=success` row with `chosen_over` set to the
   failed variant's content key (`content_key`, `ingest.rs:392`). Bare
   failures are excluded from the returned rows and written to a
   `rejected.jsonl` sidecar in the work dir (audit trail, provenance-stamped).
   Evidence: test plants fail×3+success → exactly 1 row returned,
   `chosen_over` populated, 3 rows in the sidecar.
3. [ ] **Task:** Raise `MAX_ROW_CHARS` (`ingest.rs:24`) for outcome-verified
   rows (proposal: 2000 → 8000 when `outcome=success`; unchanged otherwise) so
   long info-dense successes survive. Keep `MAX_FALLBACK_PROMPT` as-is.
   Evidence: test with a 5000-char successful command row kept; same-length
   `unknown` row still dropped.
4. [ ] **Task:** Tier stamping: ingest source config gains an optional `tier`
   (default `private`); rows inherit it. CLI ingest path
   (`crates/scrt-evolve-cli/src/main.rs` ingest cmd) threads it through.
   Evidence: ingest test asserts `tier` on emitted rows; config-reference
   lists the new key.
5. [ ] **Task:** Dedup-ledger compatibility: outcome/meta fields must NOT
   change `IngestLedger` content hashing (a re-mined identical interaction
   with a newly-parsed outcome is still a duplicate).
   Evidence: ledger test — same row ± meta hashes identical
   (`ingest_ledger.rs` test module).

---

## Phase B — Per-pair data judge + manifest rollup

Goal: every pair scored before it trains; scores roll up to the branch
manifest (the lexame "marked expertise" fields).

1. [ ] **Task:** `LlmPairJudge` in `crates/scrt-evolve/src/ingest.rs` (or a
   sibling `judge.rs`): generic over `ChatTransport`, batched, mirroring
   `LlmRelevanceJudge` (`ingest.rs:184`) / `LlmDegradationJudge`
   (`eval/degrade.rs:66`). Prompt: score each numbered row 0–1 on
   correctness/quality/steering-alignment (steering text passed in when
   `compose_steering()` is Some, `config.rs:1704`); reply = JSON array of
   scores; garble tolerance per the existing index-parse pattern.
   Evidence: unit tests with a mock transport — scores parsed, garbled reply
   handled per policy, batching respected.
2. [ ] **Task:** `[judge]` config: `min_score` (default 0.5), `on_error =
   "keep"|"drop"` (default `keep`, documented rationale: fail-open matches
   `ingest.rs:212-215` precedent + track-31 preflight backstop; flip to `drop`
   before publishing branches), `batch`, `sample_k` (Phase D compliance).
   Validate + surface in `config-reference`.
   Evidence: `config.rs` tests for defaults + validation; config-reference
   snapshot includes `[judge]`.
3. [ ] **Task:** Persist verdicts: judged rows get `judge_score` +
   `judge_verdict`; sub-threshold rows dropped pre-queue (daemon) /
   pre-dataset (CLI), recorded in the sidecar. Standalone command
   `evolve dataset judge --in dataset.jsonl` (stage independence).
   Evidence: CLI test — a fixture dataset judged on-disk, kept rows carry
   scores, dropped rows in sidecar; command runs with no daemon.
4. [ ] **Task:** Daemon wiring: judge post-mine/pre-enqueue in the ambient
   ingest path so the living queue holds only judged rows; count judged/dropped
   in the step log. Track-15 txn untouched (judge is strictly upstream of the
   weight-touching span).
   Evidence: daemon test via injected hooks (`daemon.rs:365` pattern) — mock
   judge drops a row, queue length reflects it; existing txn tests unchanged.
5. [ ] **Task:** Rollup + manifest: a `dataset_signal_stats()` producing
   `judged_fraction`, `judge_mean_score`, `outcome_verified_fraction`; branch
   create (`branch/create.rs:203`) merges them into `eval_report`
   (`manifest.rs:74`, additive BTreeMap keys). Add optional `tier` to
   `BranchManifest` (`manifest.rs:62`, `#[serde(default)]` — additive),
   computed most-restrictive-wins over the branch dataset's row tiers.
   Evidence: manifest round-trip test with the new keys + tier; a legacy
   manifest without `tier` still parses; branch-create fixture test asserts
   the three rollup keys present.

---

## Phase C — High-throughput synthesis

Goal: more rows, judged, in any domain; planner routes every real modality.

1. [ ] **Task:** Planner modality fixes: add `skill` + `reasoning_edit` to
   `is_valid_modality` (`plan/planner.rs:123-125`) AND to the system prompt's
   modality list (`planner.rs:35-40`); resolve `completion` — map it to a real
   completion `GenMode` if one exists, else REJECT at plan-parse with a named
   error (kill the silent Prose degrade at `generate/mod.rs:187-195`).
   Evidence: parse test — a plan spec with `skill`/`reasoning_edit` validates
   and routes to `GenMode::Skill`/`GenMode::ReasoningEdit`; `completion`
   behavior asserted per the decision.
2. [ ] **Task:** `[domain]` config: `name`, `description`, `command_prefixes`
   (default `["scrt"]`), `flag_patterns` (default `["--mp-"]`), tool-schema
   source. Defaults reproduce today's scrt values exactly.
   Evidence: config tests; absent `[domain]` ⇒ resolved struct equals the scrt
   defaults.
3. [ ] **Task:** Thread `[domain]` through: planner system prompt
   (`planner.rs:20-24` — job description templated from
   `domain.name`/`description`), signal extraction (`plan/signals.rs:51-54` —
   tool/flag counting driven by prefixes/patterns), cli validation
   (`generate/api.rs:234-242` — accept any configured prefix).
   Evidence: snapshot test — default `[domain]` produces a byte-identical
   planner prompt + identical signal counts on a fixture corpus vs the current
   hardcoded path; a custom domain fixture counts its own prefixes and
   validates its own commands.
4. [ ] **Task:** Rejection sampling: `[generate].candidates_per_seed`
   (default 1 = today) — generate N candidate batches per seed passage, score
   with `LlmPairJudge`, keep top-k above `min_score`; stamp
   `gen=rsample:<n>`.
   Evidence: generate test with mock backend + mock judge — 8 candidates in,
   ≥3 kept out, ranked by score, all stamped.
5. [ ] **Task:** `evolve dataset expand --in dataset.jsonl`: Evol/Self-Instruct
   pass over seed rows (mined + taught) — per-row evolution ops
   (deepen/broaden/concretize) via the chat transport, every expanded row
   judged before admission, stamped `gen=expand:<op>`; standalone
   (stage-independent) + optional daemon step hook behind a config flag.
   Evidence: CLI test with mock transport — 5 seeds → >5 admitted rows, all
   judged + stamped; runs against on-disk JSONL with no daemon.

---

## Phase D — Steerable loop (absorbs track 35)

Goal: track 35's nudge delivered in full, plus the data-layer knobs that make
steering meaningful, plus compliance measurement.

1. [ ] **Task:** `evolve ambient nudge` command writing `nudge.json` into the
   work dir (atomic tmp→rename, mirroring `daemon::stop_file` at
   `daemon.rs:342-344`). Flags per track 35 spec (`--goal --weight`,
   `--focus --steps N`, throttle knobs, gate mode) + new: `--judge-min-score`,
   `--modality-mix`, `--candidates-per-seed`, `--synthesis-rate`.
   Evidence: CLI test — file written atomically with the expected schema;
   `--json` summary line emitted.
2. [ ] **Task:** Daemon-side poll + consume: at the top of each step, after
   the stop-check, read + DELETE `nudge.json` (once-only), validate against
   the safe-live allowlist, merge accepted fields into the live
   `EvolveConfig`/`DaemonOptions`. Allowlist: goal weights, focus (with
   step-count TTL), throttle knobs, gate mode, judge `min_score`, modality
   mix, `candidates_per_seed`, synthesis rate. Reject-with-reason:
   `model_path`, `[train.fractional]` shape, `rotation_blocks`, work_dir.
   Sticky vs expiring modeled per-field (weights sticky, focus expires).
   Nudges are ephemeral — TOML wins on restart (document).
   Evidence: daemon tests via injected hooks — accepted nudge visible in the
   next step's effective config; rejected nudge logged with reason; nudge file
   consumed exactly once; track-15 txn tests untouched.
3. [ ] **Task:** Evolution-log `kind:"nudge"` row + surface in
   `watch status`/`watch health` ("focus changed at step N", rejected-fields
   note).
   Evidence: log-row test; `watch status` fixture output includes the nudge
   line.
4. [ ] **Task:** Steering-compliance metric: each step, sample
   `[judge].sample_k` generated rows, judge them against the
   `compose_steering()` text (`config.rs:1704`), record
   `steering_compliance` (fraction) in the step log; surface in
   `watch trend` alongside the track-31 Q4 correctness trend.
   Evidence: daemon test with mock judge — compliance fraction computed +
   logged; `watch trend` fixture shows the series; steering unset ⇒ metric
   absent (no judge call).
5. [ ] **Task:** Close out track 35: mark 35 delivered-by-37 in
   `conductor/tracks.md` (Build status + track row), pointer from
   `tracks/35-nudge-live-retuning/` to this track.
   Evidence: `tracks.md` diff; 35's dir carries the delivered-by note.

---

## Phase E — Training-signal fixes (kept in 37; deeper trainer science = candidate track 38)

Goal: the three surgical fixes that make the judged data actually teach.

1. [ ] **Task:** Daemon default objective → `end_task`: `[daemon]`-scoped
   default (mirroring how `[daemon].granularity` overrides
   `[train.fractional].granularity` — the known gotcha) so daemon steps get
   the knowledge signal by default; `distill` remains the non-daemon default;
   document the asymmetry at `config.rs:993-1000`.
   Evidence: config test — daemon-resolved config yields `end_task` absent an
   explicit setting; non-daemon default unchanged; doc comment updated.
2. [ ] **Task:** Live calib batches: source calibration inputs from the
   current step's queue rows instead of the 8 fixed recycled batches
   (`config.rs:1003-1005`; recycling at
   `python/scrt_evolve_train/shard.py:327`) — pass through the existing
   dataset contract, fall back to the fixed batches when the queue is thin
   (< `calib_batches`).
   Evidence: Python test — with a 12-row fixture dataset, calib inputs differ
   per batch (no `step % len` recycling of a fixed 8); thin-queue fallback
   asserted.
3. [ ] **Task:** Rotary kwargs on the same-model path: replace the empty
   `layer_kwargs` (`shard.py:236`) with `_rotary_kwargs`
   (`shard.py:723-739`) so bare block calls work on RoPE arches ≥ transformers
   ~4.41; no-op (returns `{}`) for arches without model-level rotary
   (Mamba-hybrid unaffected).
   Evidence: Python test on a tiny RoPE fixture arch — `_run_block` succeeds
   where the bare call raised "cannot unpack non-iterable NoneType"; existing
   Mamba/fixture shard tests still pass.

---

## Final sweep (once, at the end)

1. [ ] `cargo test` (ML-free) — 0 failures, including all new
   ingest/judge/planner/domain/nudge/manifest tests.
2. [ ] `cargo test --features train` — green (fixture paths).
3. [ ] `cargo clippy --all-targets` — clean.
4. [ ] Python: dataset-contract parity + shard tests (calib sourcing, rotary
   fixture) — green.
5. [ ] End-to-end acceptance fixture (spec §Acceptance 1–5): planted
   fail/success corpus → ingest excludes+collapses; rejection sampling ≥3
   kept/seed; nudge changes modality mix without restart; manifest carries
   rollup + tier; back-compat asserted.
   Evidence: one integration test (or scripted fixture run) exercising the
   chain ML-free via mock `ChatTransport` + daemon hooks.
6. [ ] `conductor/tracks.md` updated: track 37 row + Build status; track 35
   marked delivered-by-37.
7. [ ] `AGENTS.md` files touched: `crates/scrt-evolve/src/AGENTS.md`
   (§dataset v1.1, §judge, §domain, §nudge), Python AGENTS/notes for the
   calib/rotary seams. `SIGN-OFF.md` on completion.
