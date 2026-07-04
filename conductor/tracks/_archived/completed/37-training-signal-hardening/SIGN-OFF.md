---
type: sign-off
track: 37
title: Training-signal hardening — SIGN-OFF
status: completed
created: 2026-07-02
---

# Track 37 — Training-signal hardening — SIGN-OFF

Date: 2026-07-02

The push from **"safely not-worse"** (track 32's degradation gate + track 31's
dedup/idle) to **"measurably better, fast, and steerable"**: mined rows now carry
real execution outcomes, every candidate pair is judged before it trains, volume
is multiplied by judge-ranked synthesis, the loop's data mix is a live knob, and
the resulting quality evidence rolls into the branch manifest — the field a peer
reads to trust a shared branch's claimed expertise (the lexame P2P prerequisite).
**Delivers track 35 (nudge) in full as Phase D.**

## Delivered (all six phases, strict order 0→A→B then C/D/E)

- **Phase 0 — dataset contract v1.1** (`dataset.rs`): five additive per-variant
  optionals (`outcome`, `judge_score`, `judge_verdict`, `tier`, `chosen_over`),
  all `skip_serializing_if` so a v1.0 line round-trips **byte-identically**.
  Per-variant (not `#[serde(flatten)]` — flatten misbehaves under the
  internally-tagged `kind` enum, as the spec anticipated). Uniform accessors
  (`.set_outcome`/`.set_judge`/`.set_tier`/`.set_chosen_over`). Ledger
  hash-invariant (`content_hash` keys on content only — a re-mined row with a
  fresh stamp is still a duplicate). `chosen_over` = the recorded-not-trained DPO
  pairing.
- **Phase A — outcome signal at ingest** (`ingest.rs`): `tool_use.id`→
  `tool_result` correlation (`is_error` + Bash text heuristic) stamps `outcome`;
  `filter_outcomes` retry-collapses a ~same-command failure run + success into
  ONE success row carrying `chosen_over`, excludes bare failures to a
  `rejected.jsonl` sidecar, keeps verified successes up to 8000 chars, and stamps
  `[ingest].tier`. "Same command" = normalized-prefix (not equality).
- **Phase B — per-pair data judge** (`judge.rs`): `LlmPairJudge` (`ChatTransport`
  mirror of the relevance/degradation judges) scores 0–1; `[judge]`
  (`min_score`/`on_error keep|drop`/`batch`/`sample_k`, validated). Wired
  post-mine/pre-enqueue in `run_ingest` (living queue holds only judged rows;
  track-15 txn strictly downstream, untouched). Standalone `evolve dataset judge`.
  Manifest rollup: `dataset_signal_stats` (judged_fraction / judge_mean_score /
  outcome_verified_fraction) merged into `eval_report` + new additive `tier`
  field (most-restrictive-wins) on `BranchManifest` — in both `branch create` and
  `branch register`.
- **Phase C — high-throughput synthesis** (`plan/`, `generate/`, `judge.rs`):
  planner routes `skill`/`reasoning_edit` (were unreachable) and no longer
  silently degrades `completion`→Prose; `[domain]` parameterization (name /
  description / command_prefixes / flag_patterns / tools) threaded through the
  planner prompt, signal extraction, and cli validation with a **byte-identical**
  scrt default; `rejection_sample` (RAFT best-of-N, `[generate].candidates_per_seed`,
  `gen=rsample:<n>`); `evolve dataset expand` (Evol/Self-Instruct, `gen=expand:<op>`,
  each judged before admission).
- **Phase D — steerable loop = track 35** (`nudge.rs`, `daemon.rs`, CLI):
  `evolve ambient nudge` → atomic `nudge.json`; daemon polls + DELETES it
  (consume-once) at the top of a step after the stop-check; `apply_nudge` merges
  only the safe-live allowlist (goal weights, judge min_score, modality mix,
  candidates_per_seed, synthesis rate, gate mode, focus-with-TTL) into a
  `live_cfg` clone; restart-required knobs rejected with a reason; nudges
  ephemeral (TOML wins on restart). `kind:"nudge"` + `kind:"steering_compliance"`
  evolution-log rows (surfaced by `watch status`/`health`/`trend`); compliance =
  fraction of `sample_k` queue rows the judge finds steering-aligned (only when
  steering set + sampler wired — no judge call otherwise).
- **Phase E — surgical training fixes**: `[daemon].objective` defaults to
  `end_task` (the knowledge signal), overriding `[train.fractional].objective`
  for daemon steps exactly as `granularity` does (`daemon::apply_plan`); non-daemon
  default stays `distill` with rationale intact. Python (`shard.py`): live-calib
  sourcing (draw calibration inputs from the step's rows, fall back to fixed
  batches when the queue is thin) + same-model path reuses `_rotary_kwargs`
  instead of empty `layer_kwargs` (fixes the RoPE "cannot unpack non-iterable
  NoneType" crash on transformers ≥ ~4.41; no-op for Mamba-hybrid).

## Verification

- **`cargo test` (ML-free): 23 suites, 248 tests — all green.** Includes the new
  dataset v1.1 round-trip/byte-identity, ingest outcome+retry-collapse+sidecar,
  ledger meta-invariance, judge (parse/threshold/fail-open-closed/garble),
  rejection-sample ranking, expand, config (`[judge]`/`[domain]`/`[ingest].tier`)
  defaults+validation, nudge (consume-once/allowlist/reject/malformed), daemon
  (injected-nudge log + steering-compliance log), and branch manifest rollup+tier.
- **`cargo test --features train`: 21 suites green** (fixture paths; ML-heavy
  paths gated on runtime deps). Fixed one pre-existing, unrelated `train_lora.rs`
  drift (missing `LoraConfig.init_adapter`) to unblock the gate.
- **`cargo clippy --all-targets`: clean** (type-complexity aliases added:
  `NudgeHook`/`ComplianceHook`; test `.get().is_none()`→`!contains_key`).
- **Python:** `shard.py` + the new `test_shard.py` (live-calib + rotary fixtures)
  are **syntax-verified (UTF-8)**. `pytest` could NOT execute here: the installed
  torch is incompatible with the installed transformers
  (`ImportError: TransformGetItemToIndex` — transformers moved it), which breaks
  collection of the **pre-existing** `test_track23.py` identically. This is an
  environment version-skew, not track-37 code — it needs a torch/transformers
  bump in the venv (out of scope). The Python changes are minimal and
  self-contained; run `python -m pytest python/tests/` after aligning the deps.

## Back-compat (asserted)

Absent `[domain]`/`[judge]`/nudge, existing configs behave identically: v1.0
dataset lines round-trip byte-identically; the default-domain planner prompt is
byte-identical (snapshot test); scrt cli validation unchanged; `candidates_per_seed`
defaults to 1 (no rejection sampling). The track-15 keep|rollback/quarantine/halt
transaction is untouched — the judge sits strictly BEFORE data enters the queue.

## Non-goals honored

No P2P transport/trust-scoring, no desktop shell, **no DPO training** (the
(chosen, rejected) contract is recorded via `chosen_over`, not trained — deeper
trainer science, incl. DPO + loss curricula, is candidate **track 38**), no
track-33 serve-while-train, no track-15 semantics change.

## Post-review hardening (2026-07-03)

An adversarial validation pass (opus architect) caught three places where Phase
C/D plumbing was landed but **not connected end-to-end** — fixed before sign-off:

1. **Rejection sampling was dead code.** `rejection_sample` existed + was unit-
   tested but no generation path called it. Now `generate::run` builds a judge
   from `[generate.api]` and calls the new `run_with_backend_sampled` when
   `candidates_per_seed > 1` AND `[judge]` is set (fan-out N, judge-rank, keep
   top-`per_passage`, stamp `gen=rsample:<n>`). Tests:
   `rejection_sampling_generates_n_and_keeps_top_k` +
   `candidates_per_seed_one_is_single_pass` (byte-identical single pass).
2. **`--modality-mix` (and synthesis_rate/gate_mode) were log-only no-ops.**
   `apply_nudge` now genuinely mutates the live config: `modality_mix` replaces
   `[generate].kinds`, `synthesis_rate` sets `[generate].synthesis_rate`,
   `gate_mode` sets `[regulate].gate`, `candidates_per_seed` sets
   `[generate].candidates_per_seed` (creating the block if absent). Regression
   test: `modality_mix_actually_replaces_generate_kinds`.
3. **steering-compliance never reached `watch trend`.** `trend::from_log`
   filters on `metrics` (correctness only), so the `metrics:None` compliance rows
   were invisible. Added `trend::steering_compliance_from_log` (parses the
   fraction from the `kind:"steering_compliance"` row's `cause`) and wired
   `cmd_daemon_trend` to display + emit it. Test:
   `steering_compliance_series_parsed_from_cause`.

Also added: config-reference entries for `[judge]`/`[domain]`/
`[generate].candidates_per_seed`/`[daemon].objective`/`[ingest].tier`;
`daemon_step_objective_defaults_to_end_task` (Phase E task-1 evidence) +
`non_sharded_step_does_not_silently_enable_fractional` (pins the
`FractionalConfig::default().enabled = true` safety guard — a non-sharded daemon
step must NOT silently materialize a fractional config); `[generate].synthesis_rate`
config field. A second adversarial architect pass confirmed all three fixes are
connected end-to-end with no new regression. Post-fix sweep: **254 ML-free tests /
23 suites green, 21 train-feature suites green, clippy clean.**

## Status

Track 37 is **shipped (code + tests)**. Track **35 (nudge) is delivered by this
track's Phase D** — see `conductor/tracks.md` §Build status (updated). The
`conductor/tracks.md` Build-status table is the authoritative status of record.

## Program tracks
35 (delivered-by-37), 32, 31, 29, 26, 15, 9 — built on. Candidate follow-on: 38
(trainer science: DPO training, loss curricula, per-block LR schedules).
