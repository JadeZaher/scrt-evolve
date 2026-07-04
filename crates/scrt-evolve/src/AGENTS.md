# `crates/scrt-evolve/src` — module design notes

Directory-level design rationale for the SDK modules. **Prefer this file over
long inline comment blocks**: code carries terse one-line doc-comments (the
"what"); the "why" and the cross-module reasoning live here. Add a section when a
module's intent isn't obvious from its signatures.

## `ingest.rs` — interaction-log → training rows

Feeds the ambient daemon's living queue from real agent activity, **generically**
(no domain hardcoding — the same path serves CLI training, tool training, prose,
docs). Two cleanly split layers:

- **Parsing** (`interaction_log_rows`, `doc_completion_rows`) is pure, ML-free,
  deterministic — the testable surface. A Claude Code transcript distills into
  mixed rows: a `Bash` tool call → `Cli`; any other tool call → `ToolCall`
  (arguments minus the harness-only `description`); a prose-only assistant turn →
  `Qa`; a doc chunks into `Completion`. A tool-using turn emits only its tool
  row(s) — the surrounding prose there is reasoning, not an answer. Over-long
  payloads (heredocs, pasted files) are dropped; rows dedupe within a log.
- **Relevance** (`RelevanceJudge` / `LlmRelevanceJudge`) is an injected LLM step
  over any `ChatTransport`, so the SDK stays ML-free and the judge is unit-tested
  with a mock; the CLI wires the real chat endpoint. Relevance is a *model*
  decision against a free-text criterion, not a keyword rule — so ingestion works
  for any project. It batches, parses a JSON array of relevant item numbers, and
  **errs toward inclusion** (a failed/garbled batch keeps its rows) so a flaky
  endpoint degrades to "ingest more", never silent data loss — the eval gate is
  the real safety net.

Rows are stamped **per source** (track 31 Q2): transcript-derived rows carry
`gen = "ingest:transcript"` (`INGEST_GEN_TRANSCRIPT`); doc rows are `Completion`
(which has no `gen` field, so they're a separate, un-quarantinable class).
`INGEST_GEN_DOC` is reserved for when `Completion` gains a `gen` field. This means
a catastrophe in one source quarantines only that source, not all ingested data
(the old single `INGEST_GEN_STAMP = "ingest"` blanket is kept for back-compat).

The CLI layer (`ambient ingest` in `scrt-evolve-cli`) adds a cheap, generic
`--match` substring pre-filter (bounds the candidate set / LLM cost before any
call) and `--relevance` (the judge criterion). The intent prompt for a tool row
is the call's own `description` when present, else the recent user text.

## `ingest_ledger.rs` — already-ingested dedup memory (track 31 Q5)

The reason the self-feed doesn't overfit on stale data. The living-queue cursor is
*positional* (resume-safe), but `auto_ingest` re-mines the SAME transcripts every
refill and `enqueue_many` appends unconditionally — so an identical row mined
twice would train twice. The ledger is a persistent SET of content-hashes
(`work_dir/queue/ingested.ledger`, one FNV-1a hex per line, append-only) that
`run_ingest` consults right before enqueue: only genuinely-new rows go in. The
hash ignores provenance (`gen`/`source`) so the same usage re-mined from a
different transcript is still a duplicate. When a refill yields **zero** new rows,
`cmd_ambient` falls through to its existing idle path (poll + wait) — idling on a
stale corpus instead of re-training it (user-locked 2026-06-28; replay/
consolidation is a future track). Pairs with, doesn't replace, the cursor: cursor
= "don't re-consume the queue file", ledger = "don't re-enqueue mined activity".

## `trend.rs` — probe-correctness trend (track 31 Q4)

Answers "is behavior actually changing?" — because loss falling per step does NOT
mean the kept model changed. Pure arithmetic over the track-15 evolution log:
takes only **committed** steps (a rolled-back step didn't move the kept model),
reads each one's `ScoreReport.correctness`, and reports the series + mean-delta +
total-change + a direction arrow. The CLI surfaces it in `watch status`
(latest + arrow), `watch health`, and `watch trend` (full series). Over a small
data pool expect overfitting (rising probe score) before broad change — which is
exactly why this is read alongside the Q5 ledger's "nothing new" signal.

## daemon hardening (track 31 Q2/Q3) — see `daemon.rs`

The track-26 loop gained four resilience seams, all additive and all leaving the
track-15 transaction's keep|rollback/catastrophe semantics untouched:

- **Retries (Q2).** The daemon calls `Regulator::run_step_strict` (not the
  lenient `run_step`), so a failure of EITHER train OR score surfaces as `Err`.
  (`run_step` swallows a train error into a logged rollback — the historical
  contract the scheduler / `branch create` still rely on; `run_step_strict`
  restores the adapter the same way but then *propagates* the error. The txn
  guarantee is identical in both — only the return differs.) The loop retries an
  `Err` with exponential backoff (`max_retries`, `backoff_base`) when
  `is_transient` (anything not obviously a permanent misconfig). Exhausted ⇒ a
  `failed: true`, non-halting `DaemonStep`; the model is untouched (the txn is
  transactional). A real CATASTROPHE is never an `Err` — it's a successful
  outcome with `halt: true`, so it bypasses retry and halts as before.
- **Supervisor (Q2).** A running count of consecutive failed steps; exceeding
  `max_consecutive_failures` sets `report.gave_up` and stops (the CLI re-entering
  is the "restart"). A successful step resets the streak.
- **Budget (Q3).** `within_budget` (pure) checks a sliding 1-hour window of
  (timestamp, train_secs) against `max_minutes_per_hour`; over budget ⇒ `Wait`
  like the VRAM gate. The clock is injected (`now_secs` hook) so it's testable.
- **Health/observability (Q2).** `watch health` reads the evolution log for
  run-state, last step+verdict, committed count, last cause/error, and a halt
  flag. Caveat: only *transactional* steps land in the evolution log, so a
  transient subprocess failure (which never completes a txn) shows in
  `daemon.log`/stderr, not in `health`'s "last error".

The two new hooks on `DaemonHooks` — `now_secs` (monotonic seconds) and `sleep`
(injected so tests don't actually wait) — keep the whole loop clock-free and
GPU-free in tests, preserving track 26's ML-free testability.

## `eval/degrade.rs` + the judge gate (track 32) — progress on tiny data

The correctness gate (`eval::classify`) accepts a step only if the ABSOLUTE probe
score didn't drop — too noisy to move a weak model (it sits at 0.0–0.5 and bounces;
track 31 Q4 confirmed the flat/noisy pattern). The **judge gate** flips the question
to "did it get WORSE?": sample each probe prompt on the model BEFORE (base) and
AFTER (base+candidate-adapter), and `eval::LlmDegradationJudge` (a `ChatTransport`
mirror of `ingest::LlmRelevanceJudge`) decides per item. `eval::judge_verdict` maps
the result: NaN/collapse → Catastrophic, regressed-fraction > `max_regressed_frac`
→ Regress, else **Accept** ("no degradation detected"). The judge **errs toward
not-worse** on a failure/garble — a flaky judge must never stall progress; the
catastrophe floor is the backstop and `doctor`'s track-31 preflight catches a
down/missing judge model.

Wiring (no track-15 semantics changed):
- `regulate::txn` gained `run_step_judged` (+ a private `decide` closure param on
  the shared body): same snapshot/commit/rollback/quarantine/log/halt, but the
  verdict comes from the injected `decide` instead of `classify`. `run_step` /
  `run_step_strict` pass `classify`; the daemon passes a closure that runs the A/B
  degradation report + `judge_verdict`.
- `DaemonHooks.degrade: Option<…>` selects the gate: `Some` ⇒ judge gate, `None` ⇒
  correctness gate (today's behavior). Production (`run_ab_degrade` in the CLI)
  shells `python -m scrt_evolve_score --ab` (the `sample_ab` path: base vs
  base+adapter completions) and runs the LLM judge over `[regulate.degrade_judge]`
  (or `[generate.api]`). Selected by `[regulate].gate = "judge"`.
- **Correctness is still computed** under the judge gate (the trend, Q4) — it's the
  catastrophe backstop, not the accept driver.

## min-QA-pairs floor (track 32) — see `daemon.rs` `enough_to_train`

`[daemon].min_train_pairs` (default 4): a step won't train on fewer than N pending
rows. Checked BEFORE popping (so the cursor isn't advanced — the rows accumulate,
not get consumed-and-dropped); below the floor the loop idles (or stops in drain
mode). Composes with the Q5 ledger: a stale corpus that yields too few genuinely-new
rows simply waits. The default is conservative ("at least half a `batch=8` of new
signal"); the right N is **empirical** — tune via the `bench/` sweep (vary
`min_train_pairs ∈ {1,2,4,8}`, watch the Q4 trend slope + the judge regress rate;
pick the smallest non-degrading N). The number is the deliverable's *output*, not
an assertion.

## dataset contract v1.1 (track 37) — additive training-signal metadata

`GenExample` carries five OPTIONAL fields on every variant (per-variant, NOT
`#[serde(flatten)]` — flatten misbehaves under the internally-tagged `kind` enum):
`outcome` (success|failure|unknown), `judge_score` (0–1), `judge_verdict`
(keep|drop|unjudged), `tier` (private|shared), `chosen_over` (content-key of the
rejected half of a preference pair). All are `#[serde(default, skip_serializing_if)]`
so a v1.0 line round-trips **byte-identically** (a defaulted field never
serializes). Uniform accessors (`.outcome()/.set_outcome()`, `.set_judge()`,
`.tier()`, `.set_chosen_over()`) stamp rows without matching every variant. The
dedup ledger is UNCHANGED — `content_hash` keys on content only (`content_key`),
so a re-mined row with a fresh outcome stamp is still a duplicate. `chosen_over` is
the **recorded-not-trained** DPO contract (track 37 non-goal: no trl lane here).

## ingest.rs — outcome signal + retry-collapse (track 37 Phase A)

`interaction_log_rows` now correlates `tool_use.id` → the following `tool_result`
block (`is_error` primary; a Bash text heuristic — "command not found"/"fatal:"/…
— secondary) and stamps `outcome`; err toward `Unknown` when no signal. Parse-time
length cap is raised to `MAX_ROW_CHARS_VERIFIED` (8000) so a long SUCCESS survives;
`filter_outcomes` then (1) **retry-collapses** a run of ~same-command failures
followed by a success into the ONE success row (recording the failed variant's key
in `chosen_over`), (2) **excludes bare failures** from training (they go to the
`rejected.jsonl` audit sidecar under `work_dir/queue/`), (3) drops a non-success row
still over `MAX_ROW_CHARS` (2000), and (4) stamps `tier` from `[ingest].tier`.
"Same command" is a normalized-PREFIX match (first two tokens, lowercased) — NOT
equality — to catch "same command, tweaked args" while erring toward not-similar.

## judge.rs — per-pair data judge + synthesis (track 37 Phase B/C)

`LlmPairJudge` (a `ChatTransport` mirror of the relevance/degradation judges) scores
each row 0–1 on correctness/quality/steering-alignment; `judge_rows` stamps
`judge_score`+`judge_verdict` and splits kept (≥ `[judge].min_score`, default 0.5) /
dropped. `[judge].on_error` (default `keep` = fail-open, matching the relevance-judge
precedent + track-31 preflight backstop; flip to `drop` = fail-closed before
publishing branches P2P). Wired into the ingest path (`run_ingest`, post-mine /
pre-enqueue) so the living queue holds only judged rows — strictly UPSTREAM of the
track-15 weight-touching txn (untouched). Standalone `evolve dataset judge
--in dataset.jsonl` for stage independence. `dataset_signal_stats` (judged_fraction,
judge_mean_score, outcome_verified_fraction) + `dataset_tier` (most-restrictive-wins)
roll up into the branch manifest's `eval_report` + new `tier` field — the lexame
"marked expertise" a peer reads to trust a shared branch. `rejection_sample`
(RAFT-style best-of-N: `[generate].candidates_per_seed`, judge-rank, keep top-k,
stamp `gen=rsample:<n>`) and `expand_dataset` (Evol/Self-Instruct, stamp
`gen=expand:<op>`, each judged before admission — `evolve dataset expand`).

## [domain] parameterization (track 37 Phase C)

`[domain]` (name, description, command_prefixes, flag_patterns, tools) de-hardcodes
`scrt` from the planner job prompt (`planner::system_prompt`), signal extraction
(`signals::extract` tool/flag counting), and cli validation (`generate::api`
`cli_command_allowed`). Defaults reproduce the scrt values **byte-identically** —
absent `[domain]` is behavior-identical (asserted by a snapshot test). The planner
also now routes `skill`/`reasoning_edit` (was unreachable) and no longer silently
degrades `completion` to Prose (rejected at plan-parse).

## nudge.rs + steerable loop (track 37 Phase D = track 35 delivered)

`evolve ambient nudge` writes an atomic `nudge.json` (tmp→rename, mirroring
`daemon::stop_file`). The daemon polls + DELETES it (consume-once) at the TOP of a
step, after the stop-check — the loop owns the config then (`take_nudge` hook is the
test seam, like `should_stop`). `apply_nudge` merges only the SAFE-LIVE allowlist
into a `live_cfg` clone: goal weights (sticky), judge `min_score`, modality mix,
`candidates_per_seed`, synthesis rate, gate mode, and a `focus` with a step-count
TTL. Restart-required knobs (model_path, fractional shape, rotation_blocks, work_dir)
are REJECTED with a reason. Nudges are **ephemeral** — the TOML `cfg` wins on restart
(`live_cfg` dies with the loop). Each accepted nudge writes a `kind:"nudge"`
evolution-log row (surfaced by `watch status`/`health`); each committed step samples
`[judge].sample_k` queue rows against `compose_steering()` and logs a
`kind:"steering_compliance"` row (charted by `watch trend`) — only when steering is
set AND the sampler hook is wired (no judge call otherwise).

## objective asymmetry (track 37 Phase E) — see `config.rs` `default_daemon_objective`

`[daemon].objective` defaults to `end_task` (the KNOWLEDGE signal) — the ambient loop
exists to actually teach from judged live data. It overrides
`[train.fractional].objective` for daemon steps EXACTLY as `[daemon].granularity`
overrides its fractional twin (applied in `daemon::apply_plan`). The NON-daemon
fractional default stays `distill` (a representation-only near-no-op) with its
rationale intact. The Python trainer's live-calib sourcing + same-model rotary-kwargs
fix are documented in `python/scrt_evolve_train/AGENTS.md`.
