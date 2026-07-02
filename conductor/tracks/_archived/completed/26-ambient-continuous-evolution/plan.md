---
type: Implementation Plan
title: Ambient Continuous Evolution
description: Implementation plan for the Ambient Continuous Evolution track.
tags: [track-26, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 26 — Ambient Continuous Evolution — Plan

## Tasks
1. [ ] `living_queue` (Rust SDK module): two-lane (raw / priority) append-only
   JSONL queue under work_dir; atomic enqueue + pop; restart-safe cursor.
   Additive; default build stays ML-free (no ML deps in the queue itself).
2. [ ] Activity tail → enqueue:
   - passive watcher over `~/.claude/projects/*.jsonl` reusing the track-20
     harvest adapter → `raw` lane;
   - explicit `scrt-evolve teach <…>` / scrt-capture hook → `priority` lane.
3. [ ] Constitution/taste-driven synthesis: when the queue needs generated QA
   (not raw activity), shape the generate prompt with the active constitution +
   taste meta-modules (tracks 21/22).
4. [ ] `daemon` subcommand: VRAM-gated loop (`[hardware]` + `--max-vram`), per
   step → one microshard (track 25 `granularity=module`) over the next queued
   item(s) → track-15 transaction (keep|rollback) → durable per-step log.
   Explicit `daemon start` / `daemon stop`.
5. [ ] `[daemon]` config block: `max_vram`, `poll_interval`, lane weights,
   granularity (default `module`), eval cadence. Additive.
6. [ ] Tests: queue enqueue/pop/restart round-trip (Rust); VRAM-gate skip logic;
   transactional commit on a stubbed trainer (machinery testable ML-free, same
   injected-closure pattern as track 20 rounds).
7. [ ] Real verification: run `daemon start --max-vram 4G` on Granite (WSL2) with
   a live tail of actual CLI activity; confirm bounded VRAM, eval-gated commits,
   resume-after-stop.

## Future axes (designed, sequenced after the daemon — from the user's ask)
- [ ] Curriculum refinement loop: score generated QA each round, drop/repair weak
  pairs, re-teach from the refined set (data quality compounds over time).
- [ ] Memory consolidation: every K rounds merge shard adapters → base + replay/
  distill older QA (anti-catastrophic-forgetting across a long ambient run).
- [ ] Teaching-as-evolution: the evolving student informs the teacher to
  regenerate better QA mid-run (iterated teaching / self-distillation).

## Status
**MACHINERY SHIPPED (2026-06-26)** — the ML-free, testable core is built + green;
the live GPU run is the only deferred piece.

- [x] **Task 1 — `living_queue`** (`src/living_queue.rs`): two-lane (raw/priority)
  append-only JSONL queue under `work_dir/queue/`, atomic enqueue + cursor-based
  pop (priority drains first), restart-safe `cursor.json` (temp+rename). 4 unit
  tests (round-trip, priority ordering, cursor-survives-reopen, batch drain).
- [~] **Task 2 — activity tail → enqueue**: explicit `teach` → PRIORITY lane is
  DONE (CLI `teach --prompt --completion`); raw-lane ingestion helper
  `LivingQueue::enqueue_raw(Dataset)` is present for distilled harvest rows. The
  always-on filesystem *watcher* over `~/.claude/projects` is the deferred
  production wiring (reuses `harvest::harvest_entries`).
- [ ] **Task 3 — constitution/taste synthesis**: deferred — depends on tracks
  21/22 (taste/meta-objects), which are not built.
- [x] **Task 4 — `daemon` subcommand** (`src/daemon.rs` + CLI `daemon start/stop/
  status`): VRAM-gated loop (`[hardware]`/`[daemon]` + `--max-vram`), per step →
  track-15 transaction (keep|rollback), catastrophe → halt, durable
  `logs/daemon.log`, explicit stop-file control, resume from the queue cursor.
- [x] **Task 5 — `[daemon]` config block**: `max_vram_gb`, `poll_interval_secs`,
  `batch`, `granularity` (default `module`), `eval_cadence`. Additive.
- [x] **Task 6 — tests**: queue round-trip + VRAM-gate/stop/max-steps + stubbed-
  trainer transactional commit (3 daemon tests, injected-closure pattern).
- [ ] **Task 7 — real GPU verification** on Granite (WSL2): DEFERRED (needs the
  GPU box; the machinery is exercised ML-free).

The three future axes (curriculum refinement, memory consolidation, teaching-as-
evolution) remain backlog. Build order delivered: queue → daemon → eval-gated
commit; tail (explicit) + synthesis are the remaining wiring.
