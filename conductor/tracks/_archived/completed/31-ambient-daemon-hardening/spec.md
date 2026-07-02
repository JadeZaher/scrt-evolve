---
type: Track Spec
title: Ambient Daemon Hardening
description: Make the shipped track-26 ambient daemon production-robust for unattended runs.
tags: [track-31, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 31 — Ambient Daemon Hardening — Spec

## Goal
Make the shipped track-26 ambient daemon **production-robust for an unattended,
multi-day run**. Track 26 delivered the machinery (two-lane living queue,
VRAM-gated loop, every step through the track-15 transaction) and proved it
ML-free. Running it for real against a *living* corpus surfaced five gaps — all
correctness or resilience, none in the core loop's transactional safety. This
track closes them.

The driving scenario: `scrt-evolve --ambient --dir bench` left running for days,
feeding on real `~/.claude/projects` activity, with LM Studio teachers that come
and go and a corpus that is sometimes *stale* (nothing genuinely new to learn).

## Builds on
- **Track 26** (ambient daemon) — the subsystem being hardened: `daemon.rs`,
  `living_queue.rs`, `ingest.rs`, and the CLI `cmd_ambient` / `run_ingest` /
  `daemon_serve` wiring.
- **Track 15** (self-regulation) — the keep|rollback transaction the daemon
  already routes every step through; untouched, but Q2's retry logic must NOT
  swallow a genuine catastrophe-halt.
- **Track 10** (eval harness) — the stable-probe gate; Q4 reads its per-checkpoint
  `ScoreReport` to track a behavioral-change trend.

## The five gaps (user-prioritized; from the retired 2026-06-28 daemon-run handoff, folded here)

### Q1 — Teacher (judge) model availability
**Problem.** The relevance judge points at `[generate.api].model`, a model that
may not be loaded in LM Studio (the configured `meta-llama-3-8b-instruct` was
removed when disk was freed). On a missing model the judge errs toward inclusion
(keep-all) — silently unfiltered, not loud.
**Fix.** (a) Repoint `[generate.api].model` at an available, small judge
(`ibm/granite-4-h-tiny`). (b) Add a **judge preflight**: query the endpoint's
`/v1/models`, and warn (ambient) / fail (doctor) early when the configured model
isn't loadable. "Listed in config" ≠ "loadable on 8 GB".

### Q2 — Error handling / retries / health / supervision
**Problem.** A subprocess failure (OOM, missing model, disk-full) propagates via
`?` and exits the process — no retry, no transient-vs-catastrophe distinction, no
health command, no auto-restart. A single shared `ingest` gen-stamp means one
catastrophe quarantines *all* ingested data.
**Fix.** Distinguish **transient** failures (retry with bounded backoff) from
**catastrophe** (the existing track-15 halt — never retried). Add a `daemon
health` view (last step, last error, consecutive failures, uptime). Add a
supervisor/auto-restart around the loop bounded by a max-consecutive-failures
cap. Per-source gen stamps so a bad batch quarantines only its source.

### Q3 — Training-time budget
**Problem.** Only `cooldown_secs` / `poll_interval` / `max_steps` / the VRAM gate
exist — no wall-clock budget, so the daemon can train at full duty cycle around
the clock.
**Fix.** A `[daemon]` time window: max-minutes-per-period and/or active-hours, so
the daemon self-limits its share of the machine over real time. Pure, testable
decision function (clock injected) like `decide_step`.

### Q4 — Time-to-behavioral-change measurement
**Problem.** Loss falls per step, but loss ≠ behavior. The real signal is probe
correctness (already stored per checkpoint, currently low/noisy). It is not
surfaced as a trend.
**Fix.** Track the **probe-correctness trend** across committed checkpoints (read
the evolution log / checkpoint manifests) and expose it (`daemon status` /
`health` and/or a small `trend` view). Optionally an A/B infer on held-out
prompts. Expect overfitting before broad change given the small pool — ties to Q5.

### Q5 — Stale-corpus / re-training the same rows (TOP correctness risk)
**Problem.** Auto-ingest re-mines the same logs → the same ~400 deduped rows.
`enqueue_raw`/`enqueue_many` **append unconditionally**; the queue cursor is
positional per-lane and does NOT dedup against already-trained history, so the
same content gets appended past the cursor and re-trained — overfitting the gate
won't catch.
**Fix.** A persistent **content-hash ledger** of already-ingested rows: only
genuinely new rows enqueue. When nothing new exists, the ambient loop **goes
idle** (poll, wait for new activity) rather than re-training stale data
(user-locked decision 2026-06-28; replay/consolidation mode is explicitly out of
scope for this track).

## Out of scope
- Replay / consolidation mode for the "nothing new" regime (future track).
- Live GPU validation of the daemon (carried from track 26 as deferred).
- Any change to the track-15 transaction's keep|rollback/catastrophe semantics.

## Acceptance
- Judge preflight catches a missing model before the first refill (covered by a
  test against a mock `/v1/models`); `doctor` reports it.
- A transient train/score error retries with backoff and the loop survives; a
  catastrophe still halts (track-15 semantics preserved). Both covered by daemon
  loop tests with injected failing hooks.
- `daemon health` reports run-state, last step, last error, consecutive failures.
- A wall-clock budget pauses training when the period allotment is spent (pure
  decision-function test, clock injected).
- The dedup ledger prevents re-enqueue of an already-ingested row (ingest test);
  the ambient loop idles when ingest yields nothing new.
- A probe-correctness trend is computed from the evolution log (pure test).
- Full sweep green: `cargo test` + `clippy` + `fmt`. ML-free per styleguide §1.
