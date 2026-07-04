---
type: Implementation Plan
title: Ambient Daemon Hardening
description: Implementation plan for the Ambient Daemon Hardening track.
tags: [track-31, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 31 — Ambient Daemon Hardening — Plan

> **DONE (2026-06-28).** All five phases shipped + tested; full sweep green
> (`cargo test` 0 failures · `clippy --all-targets` clean · `fmt --check` ok).
> Live-verified: `doctor` (judge_model PASS), `watch health`, `watch trend`,
> `watch status` all run against the live work dir. Files: `generate/api.rs`
> (`list_models`/`preflight`/`classify_models`), `ingest_ledger.rs`, `trend.rs`,
> `daemon.rs` (retries/supervisor/budget/clock hooks), `ingest.rs` (per-source
> stamps), CLI `watch health`/`trend` + `doctor` judge check + `run_ingest`
> ledger wiring + `cmd_ambient` idle-on-empty. **The running daemon must be
> restarted to pick up the new binary** (the live process predates this code).

Five phases, one per gap. Phases are independent; per the test-once-at-end
policy, ALL code lands first, then ONE full sweep (`cargo test` + `clippy` +
`fmt`). The daemon stays running throughout (these are additive seams; the
running binary picks them up on its next restart).

Phase ordering is by correctness risk, matching the handoff: Q1 (fail-loud) and
Q5 (stop re-training stale data) first, then Q2 (resilience), then Q3 (budget),
then Q4 (measurement).

## Phase 1 — Q1: judge model + preflight
1. [ ] Repoint `bench/ambient.toml` `[generate.api].model` →
   `ibm/granite-4-h-tiny` (available, small, leaves VRAM for training).
2. [ ] Add `list_models()` to `generate::api` (GET `{base_url}/v1/models`,
   reuse the blocking reqwest client + bearer auth). Returns `Vec<String>` ids.
3. [ ] Add a `judge_preflight(cfg)` helper: when `[ingest].relevance` is set and
   `backend=api`, fetch models and check the configured model is present.
4. [ ] Wire it into `validate_ambient` (warn) and `cmd_doctor` (a `judge_model`
   check). Endpoint-down ⇒ a soft note (not a hard fail): the daemon already
   degrades to keep-all.
5. [ ] Tests: mock-transport `list_models` parse; preflight present/absent/down.

## Phase 2 — Q5: dedup ledger + idle-on-empty
1. [ ] New module `ingest_ledger.rs` (ML-free): a persistent set of
   content-hashes under `work_dir/queue/ingested.ledger` (one hash per line,
   append-only, atomic). Reuse `ingest`'s existing content-key shape.
2. [ ] `run_ingest`: after the relevance/cap passes, filter rows whose content
   hash is already in the ledger; enqueue only the new ones; record their hashes.
   Report `enqueued` vs `skipped (already ingested)`.
3. [ ] `cmd_ambient`: the refill already gates on `pending < refill_below`; with
   the ledger, a refill that yields 0 new rows leaves the queue empty → the
   existing `is_empty()` idle path fires (poll + wait). Add a log line so the
   "idle because nothing new" state is legible.
4. [ ] Tests: re-ingesting the same rows enqueues 0 the second time; ledger
   survives reopen; new rows still enqueue.

## Phase 3 — Q2: retries / health / supervisor / per-source stamps
1. [ ] In `daemon.rs`, classify a step's `train`/`score` error: **catastrophe**
   (the existing track-15 halt path — NaN/broken training) is untouched and never
   retried; a **transient** error (subprocess non-zero, OOM, endpoint blip) is
   retried with bounded exponential backoff (`max_retries`, `backoff_base_secs`
   in `[daemon]`), then surfaced as a failed-but-non-halting step if exhausted.
2. [ ] Track consecutive transient failures; exceed a cap ⇒ stop with a clear
   "supervisor giving up" report (auto-restart is the CLI loop re-entering).
3. [ ] `watch health`: read run/stop files + the evolution log tail → run-state,
   last ordinal, last verdict, last error, consecutive failures, committed count.
   JSON + human.
4. [ ] Per-source gen stamps: ingest stamps rows by source slug (e.g.
   `ingest:transcript`, `ingest:doc`) instead of a single `ingest`, so a
   catastrophe quarantines only the offending source.
5. [ ] Tests: injected transient hook → retried then survives; injected
   catastrophe → still halts (no retry); health report shape; per-source stamp.

## Phase 4 — Q3: wall-clock training budget
1. [ ] `[daemon]` fields: `max_minutes_per_hour` (0 = unlimited) and/or
   `active_hours` (e.g. "off 22:00-08:00"). Defaults preserve today's behavior.
2. [ ] Pure `within_budget(now, recent_train_secs, opts) -> bool` decision
   function (clock + accumulated-train-time injected), mirroring `decide_step`.
3. [ ] Wire into the daemon loop: when over budget, `Wait` (sleep, don't train),
   same as the VRAM gate. Production passes a real clock; tests inject one.
4. [ ] Tests: over/under budget; active-hours window; default = unlimited.

## Phase 5 — Q4: probe-correctness trend
1. [ ] `trend.rs` (pure): given the evolution log (or checkpoint manifests),
   compute the committed-checkpoint correctness series + a simple
   slope/delta-over-last-N summary.
2. [ ] Surface in `watch status` / `health` (latest correctness + trend arrow)
   and a small `watch trend` view (series + summary, JSON + human).
3. [ ] (Optional, behind a flag) A/B infer on a held-out prompt set for a
   behavioral spot-check; defer if it needs the ML subprocess.
4. [ ] Tests: trend from a synthetic log (rising / flat / noisy).

## Final
- [ ] ONE sweep: `cargo test` + `cargo clippy --all-targets` + `cargo fmt
  --check` + any Python touched. Fix all, re-run once.
- [ ] Update `conductor/tracks.md` Build status + the Tracks table row for 31.
- [x] Retire `conductor/HANDOFF.md`; what shipped is recorded in `conductor/RETRO.md` (§Ambient hardening lane).
- [ ] `src/AGENTS.md`: directory-level notes for the new modules (per the
  docs-in-directory convention), one-line pointers in code.
