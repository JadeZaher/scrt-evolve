# Track 26 — Ambient Continuous Evolution — Spec

## Goal
Turn evolution from a scheduled multi-day BATCH into an **always-on, VRAM-bounded
background process** fed by a **living dataset queue** that updates from the
user's own CLI / work activity, with **constitution + taste** driving generation
fundamentals. The model drifts toward the user's real usage, continuously,
eval-gated. "Training can almost always be happening as a background task,
bounded by VRAM; data is updated dynamically by user activity."

## Builds on
- Track 25 (fractional / microshard training) — the bounded-VRAM training
  PRIMITIVE the daemon consumes (one shard/microshard per step → fits any GPU).
- Tracks 10/15 (eval gate + keep|rollback transaction) — the backstop so ambient
  training can never silently degrade the model.
- Tracks 21/22 (taste / meta-objects) — constitution (values that drive
  processing) + taste (representational form) become the generate-prompt drivers.
- Track 20 slice 4 harvest.rs — the transcript adapter the activity tail reuses.

## Architecture (user-locked decisions, 2026-06-21)
1. **Living dataset queue** — append-only JSONL under work_dir, TWO lanes:
   - `raw` lane: filled by a PASSIVE TAIL of `~/.claude/projects` (and scrt CLI
     usage) as the user works; gated by constitution/taste filters before
     becoming training examples.
   - `priority` lane: filled by EXPLICIT captures (`scrt stash …`, an
     `scrt-evolve teach …` call); skips the filter, weighted higher.
   Both streams feed the daemon; priority drains first.
2. **VRAM-bounded daemon, EXPLICIT start/stop** — `scrt-evolve daemon start`
   runs until stopped (NOT idle-triggered, NOT always-auto). Per step: check free
   VRAM (`[hardware]` + `--max-vram`); if a microshard fits, pop the next
   training example(s), run ONE microshard step (track-25 `granularity=module`),
   commit through the track-15 transaction (keep|rollback), yield. Self-throttles
   around the user's other GPU use by simply waiting when VRAM is tight.
3. **Constitution/taste-driven generation** — when the queue needs synthesized
   QA (vs. raw activity), the generate prompts are shaped by the active
   constitution + taste meta-modules, so the curriculum reflects the user's
   values/representational preferences, not just raw transcripts.

## Scope (to build)
- `living_queue` module (Rust): two-lane append-only queue + atomic pop, under
  work_dir; survives restarts.
- Activity tail: a watcher over `~/.claude/projects` reusing the harvest adapter
  → enqueue raw; an explicit `teach`/scrt-capture path → enqueue priority.
- `daemon` subcommand: VRAM-gated loop, per-step microshard via track 25,
  transactional commit via track 15, durable per-step log to work_dir/logs/.
- `[daemon]` config: `max_vram`, `poll_interval`, `lanes`/weights, granularity
  (default `module`), eval cadence.
- Constitution/taste wiring into the generate prompt for queued synthesis.

## Acceptance (when built)
- `scrt-evolve daemon start` runs bounded by `--max-vram`, consumes queued
  activity microshard-by-microshard, and every committed step passes the eval
  gate (or rolls back) — verified on Granite with a live tail of real activity.
- Stopping/restarting the daemon resumes from the queue (no lost/dup work).
- Constitution/taste changes visibly alter what synthesized QA is generated.

## Status
DESIGNED, NOT YET BUILT. Track 25 (the training primitive) is complete; this
track is the next build. Recorded so the direction is durable. See memory
[[ambient-continuous-evolution]], [[micro-sub-block-training]],
[[sharded-fractional-training]].
