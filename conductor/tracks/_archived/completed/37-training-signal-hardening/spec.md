---
type: spec
track: 37
title: Training-signal hardening — data judge, high-throughput synthesis, steerable loop
status: completed
created: 2026-07-02
depends_on: [35, 32, 31, 29, 26, 15, 9]
---

# Track 37 — Training-signal hardening — Spec

## Goal
Push the training signal from **"safely not-worse"** (the honest current state:
track-32's degradation gate + track-31's dedup/idle) to **"measurably better,
fast, and steerable"**: stamp mined rows with real outcomes, judge every pair
before it trains, multiply row volume with judge-ranked synthesis, make the
loop's data mix a live knob, and roll the resulting quality evidence into the
branch manifest — the field a peer reads to trust a shared branch's claimed
expertise.

This is the prerequisite for the product's next stage: scrt-evolve embedded in
a desktop app where a community of users each train their own branch models and
share them P2P with **marked expertise** (the lexame/hivemind pairing —
`SCRT-EVOLVE-INTEGRATION.md`). "Marked expertise" = the manifest's
`router_signature` + `eval_report` (`crates/scrt-evolve/src/branch/manifest.rs:72-74`);
this track makes those fields trustworthy.

## Why a track (grounded findings, 2026-07-02 review)
Seven verified weaknesses; each phase below closes one or more:

1. **No outcome signal on mined data.** `interaction_log_rows`
   (`crates/scrt-evolve/src/ingest.rs:29`) parses `tool_use` and prose turns but
   never reads `tool_result` — a Bash command that failed and was retried 3×
   mints 3 training rows indistinguishable from the one that worked.
   Prompt→completion alignment is heuristic: tool `description`, else a 400-char
   truncation of the last user message (`ingest.rs:359`, `MAX_FALLBACK_PROMPT`
   at `ingest.rs:22`).
2. **No per-pair quality judging.** The only LLM verdicts are (a) binary batch
   relevance, fail-open on judge error (`ingest.rs:212-215` keeps the whole
   batch; no chat endpoint = no filter at all,
   `crates/scrt-evolve-cli/src/main.rs:3974`) and (b) the post-training adapter
   degradation gate (track 32, `eval/degrade.rs`). Nothing rates
   correctness/quality/usefulness of an individual pair before it trains.
3. **Row volume too low for a strong signal.** Steady-state is ~dozens of novel
   (ledger-passing) rows/day; `min_train_pairs` floor is 4
   (`config.rs:553-557`); `MAX_ROW_CHARS=2000` discards long info-dense
   interactions (`ingest.rs:24`); doc rows are capped at 20/file
   (`main.rs:3951`).
4. **Synthesis is scrt-domain-hardcoded.** The planner's job description is
   "better at USING the `scrt` tool" (`src/plan/planner.rs:22-24`); signals
   count `scrt_*` tools and `--mp-*` flags (`src/plan/signals.rs:51-54`); cli
   validation drops any command not starting with `scrt`
   (`src/generate/api.rs:234-242`). A user tuning on their own non-scrt work
   gets only generic prose QA.
5. **Planner can't route the track-09 modalities.** `is_valid_modality`
   (`planner.rs:123-125`) accepts only qa|instruction|tool_call|cli|completion —
   `skill`/`reasoning_edit` are unreachable via the planner, and `completion`
   validates but silently degrades to Prose (`src/generate/mod.rs:187-195`).
6. **Training-side signal thinness.** Default fractional objective `distill` is
   a self-referential near-no-op ("does NOT impart new knowledge",
   `config.rs:993-998`); `calib_batches` defaults to 8 fixed recycled batches
   (`config.rs:1003-1005`, recycled at `python/scrt_evolve_train/shard.py:327`);
   the same-model shard path passes empty `layer_kwargs` (`shard.py:236`) while
   the cross-model path builds `_rotary_kwargs` (`shard.py:723-739`) — bare
   layer calls crash on mainstream RoPE arches ≥ transformers ~4.41.
7. **No steering-compliance measurement.** `compose_steering`
   (`config.rs:1704`) injects constitution/taste into every generation batch;
   nothing measures whether generated rows actually reflect it.

## Relationship to track 35 (nudge) — recommendation: ABSORB
**Track 37 delivers track 35's implementation in full as Phase D**, then extends
its allowlist with the new data-layer knobs (judge threshold, modality mix,
synthesis rate, goal weights). Rationale: nudge without these knobs is a thin
direction lever (goal weights only); this track's steerability acceptance needs
the control channel anyway; and building the control-file/live-merge machinery
twice guarantees divergence. Track 35's spec + PASSOFF remain the design source
for the mechanism (control-file poll at the step boundary mirroring
`daemon::stop_file` at `daemon.rs:342`, consume-once, safe-live allowlist,
reject-with-reason, TOML-wins-on-restart); on 37 sign-off, mark 35
**delivered-by-37** in `tracks.md` — do not build it separately first.

## Research basis (technique → gap it closes)
- **LLM-as-judge data selection** — AlpaGasus (arXiv 2307.08701: judge-scored
  filtering beats training on all data), Deita (2312.15685: complexity+quality
  scoring for selection). Basis for the Phase-B per-pair judge (finding 2).
- **Rejection sampling / best-of-N** — RAFT (2304.06767) and Llama 2/3
  post-training practice: generate N candidates per seed, judge, keep top-k.
  Multiplies volume AND quality with one teacher (finding 3).
- **Self-Instruct (2212.10560) / Evol-Instruct (2304.12244)** — seed-instruction
  expansion + complexity evolution: turns few mined rows into many diverse ones
  (finding 3).
- **Magpie (2406.08464)** — instruction extraction from an aligned model with
  near-zero seed data; a cheap volume lever when a local teacher is configured
  (finding 3; optional mode, not load-bearing).
- **Outcome-supervised filtering** — execution-feedback practice from
  code-model training (CodeRL-style): exit codes / `is_error` / retry-collapse
  are free ground truth on mined tool rows (finding 1).
- **Preference pairs from outcomes** — DPO (2305.18290): failed-then-succeeded
  sequences yield natural (rejected, chosen) pairs. This track records the
  pairing in the dataset contract; DPO training itself is a non-goal.

## Design

### Dataset contract v1.1 (additive, versioned)
`GenExample` is an internally-tagged enum whose variants already carry optional
`source`/`gen` (`dataset.rs:16`, e.g. `dataset.rs:20-23`). Add **optional,
default-skipped** metadata to every variant (shared `RowMeta` via
`#[serde(flatten)]` if serde's tagged-enum flatten behaves; per-variant fields
as the fallback — decide in the plan):
- `outcome: success|failure|unknown` (Phase A),
- `judge_score: f32 0–1` + `judge_verdict: keep|drop|unjudged` (Phase B),
- `tier: private|shared` (sovereignty; most-restrictive-wins downstream),
- `chosen_over: Option<String>` (content-key of the rejected half of a
  preference pair — the DPO contract, recorded not trained).
Additive optionals keep old JSONL readable by both readers; assert Rust↔Python
parity per the existing contract test pattern. Document as contract v1.1.

### Phase A — outcome signal at ingest
Correlate `tool_use.id` → the following `tool_result` block (`is_error`,
content text; exit-code / error-text heuristics for Bash) in
`interaction_log_rows`. Stamp `outcome`. **Retry-collapse**: N failed attempts
of the ~same command followed by a success collapse to 1 chosen row
(`outcome=success`, `chosen_over` pointing at the failed variant); bare
failures default to **excluded from training** (kept in a quarantine-style
sidecar for audit, consistent with provenance rules). Raise `MAX_ROW_CHARS`
for outcome-verified rows (long successful interactions are the info-dense
ones being discarded today). Tier stamps flow from the ingest source config.

### Phase B — per-pair data judge
`LlmPairJudge` mirroring the `ChatTransport`-injected pattern of
`LlmRelevanceJudge` (`ingest.rs:184`) and `LlmDegradationJudge`
(`eval/degrade.rs:66`): scores each row 0–1 on correctness/quality/steering
alignment. New `[judge]` config: `min_score` threshold, and `on_error =
"keep"|"drop"` — **configurable fail-open vs fail-closed, default `keep`**
(documented; matches the existing fail-open precedent and the track-31 judge
preflight backstop — a flaky judge must not stall an unattended daemon; users
publishing branches P2P should flip to `drop`). Verdicts persist on rows
(auditable). Daemon wiring: judge runs post-mine/pre-enqueue so the living
queue holds judged rows. **Rollup**: judged-fraction, mean judge score,
outcome-verified fraction computed per dataset and written as additive keys
into the branch manifest's `eval_report` (already a `BTreeMap<String,f64>`,
`manifest.rs:74` — no schema break); `tier` added to the manifest as an
optional field (additive), most-restrictive row tier wins.

### Phase C — high-throughput synthesis
- **Planner routing fixes**: admit `skill`/`reasoning_edit` in
  `is_valid_modality` + the system-prompt modality list; make `completion`
  either parse as a real completion mode or be rejected at plan-parse (no
  silent Prose degrade at `generate/mod.rs:193`).
- **Domain parameterization**: a `[domain]` config (name, description,
  command_prefixes, flag_patterns, tool source) drives the planner system
  prompt (`planner.rs:22`), signal extraction (`signals.rs:51-54`), and cli
  validation (`api.rs:238`). Default = today's scrt values ⇒ absent `[domain]`
  is behavior-identical (back-compat asserted).
- **Rejection sampling**: `[generate].candidates_per_seed = N` (default 1 =
  today), judge-ranked keep-k via the Phase-B judge.
- **Expansion**: an Evol/Self-Instruct pass over mined+taught seed rows
  (`evolve dataset expand`), each expanded row judged before admission and
  stamped `gen=expand:<op>`.

### Phase D — steerable loop (absorbs track 35)
Track 35's deliverables in full: `evolve ambient nudge` writing an atomic
`nudge.json`; daemon poll+consume at the step boundary (top of step, after the
stop-check — the loop owns the config then; `should_stop` injection at
`daemon.rs:365` shows the test seam); safe-live allowlist with
reject-with-reason; `kind:"nudge"` evolution-log row surfaced in
`watch status`/`health`. **Extended allowlist**: judge `min_score`, modality
mix override, `candidates_per_seed`/synthesis rate, goal weights, throttle
knobs, gate mode. Restart-required knobs (model_path, fractional shape,
rotation_blocks, work_dir) rejected. **Steering compliance**: sample K
generated rows per step, judge against the `compose_steering()` text, report
compliance fraction in `watch trend`.

### Phase E — training-signal fixes gated on the new data (kept in 37)
The smallest defensible slice — three surgical fixes, each bounded:
1. daemon steps default to `objective = "end_task"` (the knowledge signal;
   `distill` stays the non-daemon default with the rationale documented),
2. `calib_batches` sourced from live-queue rows instead of 8 fixed recycled
   batches (`shard.py:327`),
3. same-model shard path reuses `_rotary_kwargs` (`shard.py:723-739`) instead
   of empty `layer_kwargs` (`shard.py:236`).
**Recommendation: keep in 37** — these are days, not weeks, and (1)+(2) only
make sense atop the judged queue. Deeper trainer science (loss curricula, DPO
training, per-block LR schedules) is explicitly a candidate **track 38**, not
here.

## Deliverables
1. Dataset contract v1.1: `outcome`/`judge_score`/`judge_verdict`/`tier`/
   `chosen_over` (additive), Rust↔Python parity test.
2. Outcome-stamped ingest + retry-collapse + preference-pair recording.
3. `LlmPairJudge` + `[judge]` config (threshold, fail-open/closed) + persisted
   verdicts + daemon wiring.
4. Manifest rollup: judge/outcome stats in `eval_report`, `tier` field,
   most-restrictive-wins propagation.
5. Planner modality fixes + `[domain]` parameterization + rejection sampling +
   `dataset expand`.
6. Nudge (track 35 complete) + extended data-layer allowlist + steering-
   compliance metric in `watch trend`.
7. Phase-E training fixes (daemon end_task default, live calib batches,
   rotary kwargs).

## Non-goals
- P2P transport/serving/trust-scoring — hivemind's side; this track only makes
  the manifest fields trustworthy.
- Desktop app shell.
- DPO **training** — the (chosen, rejected) contract is recorded; the trl lane
  is future work.
- Track 33 serve-while-train; hot model reload.
- Changing track-15 keep|rollback/quarantine/halt semantics — the judge sits
  BEFORE data enters the queue; the txn is untouched.

## Risks / open questions
- **Judge cost/latency per pair** — batch like the existing judges; the
  threshold + `on_error` policy must keep the unattended daemon from stalling.
- **serde flatten on an internally-tagged enum** — known quirks; per-variant
  optional fields are the accepted fallback (verbose but contract-safe).
- **Retry-collapse false positives** — "same command" needs a similarity
  heuristic (normalized command prefix), not equality; err toward `unknown`
  outcome when uncertain.
- **Domain parameterization regressions** — the scrt default must be
  byte-identical for existing configs; assert with fixture snapshots.
- **Live-merge safety (from 35)** — apply nudges ONLY at the top of a step;
  nudges are ephemeral (TOML wins on restart); allowlist errs toward
  restart-required.
- **Stage independence** — judge/expand must be runnable standalone against an
  on-disk dataset.jsonl (per workflow.md), not only inside the daemon.

## Acceptance
On a fixture corpus with planted failed+succeeded command pairs, ML-free (mock
`ChatTransport` + injected daemon hooks, per the track 26/31/32 pattern):
1. Ingest **excludes** the planted failures from training rows, collapses the
   retry chain to one `outcome=success` row carrying `chosen_over`, and stamps
   `tier` from the source config.
2. With `candidates_per_seed = 8` and a mock judge, generation yields **≥3
   judged-kept rows per seed passage**, each persisting `judge_score`; rows
   below `min_score` never reach the queue; `on_error = drop` drops the batch
   a failing judge would have kept.
3. With a daemon running (bounded test mode), `evolve ambient nudge
   --modality-mix ...` **changes the next step's generation mix without
   restart**; a restart-required nudge is rejected with a clear reason; the
   nudge appears as an evolution-log row; steering-compliance fraction appears
   in `watch trend`.
4. `branch create` on the fixture writes a manifest whose `eval_report`
   carries `judged_fraction` / `judge_mean_score` / `outcome_verified_fraction`
   and whose `tier` is the most restrictive row tier; manifest round-trips.
5. Absent `[domain]`/`[judge]`/nudge, existing configs behave identically
   (back-compat asserted); track-15 transaction tests untouched and green.
6. Phase E: same-model shard smoke passes on a RoPE fixture arch (no bare-call
   crash); daemon-config default objective is `end_task` and documented; calib
   batches provably drawn from queue rows on a fixture.
