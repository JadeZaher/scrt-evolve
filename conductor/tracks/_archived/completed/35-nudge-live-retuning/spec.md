---
type: spec
track: 35
title: Nudge — live retuning of a running ambient daemon
status: planned
created: 2026-06-30
depends_on: [26, 31, 15]
---

# Track 35 — Nudge (live retuning) — Spec

## Goal
Let a user **steer a RUNNING ambient daemon's direction** without stopping it —
adjust which goals it favors, bump/drop a goal's weight, or shift the active
focus — and have the loop pick up the change at the **next step boundary**. The
branded component name for this steering capability is **"nudge"**.

Today the only way to change direction is: `ambient stop` → edit the TOML →
`ambient start`. That's a full restart (model reload, cache cold). Nudge makes
direction a *live* knob.

## Why a track (not a CLI tweak)
The CLI surface is easy (`evolve ambient nudge --goal scrt-cli --weight 2.0`);
the real work is the **daemon-side control channel + safe live-apply**:
- a control-file the loop polls at the step boundary (mirroring the `daemon.run`
  stop-file pattern already in `daemon.rs`),
- merging a nudge into the in-memory `EvolveConfig`/`DaemonOptions` mid-run
  WITHOUT breaking the track-15 transaction or the resume invariants,
- deciding which knobs are safe to change live vs. which require a restart.

That's design + invariants work, so it's a track with a passoff, not an inline
edit. `steer`/`teach` (priority-lane enqueue) stays as-is; nudge is the *direction*
knob, teach is the *content* knob.

## What "nudge" can change (proposed; refine in plan)
Safe-live (re-read each step, no model reload):
- **goal weights** — bump/drop a `[[goals]]` weight so the weighted policy favors it.
- **active focus** — restrict the next N steps to one goal/tag (a temporary filter).
- **throttle knobs** — `max_vram_gb`, `min_free_ram_gb`, `cooldown_secs`,
  `max_minutes_per_hour`, `poll_interval_secs` (pure gate inputs — trivially live).
- **gate mode** — `[regulate].gate` correctness↔judge (affects only the next
  decision; safe).

Restart-required (nudge should REJECT with a clear message):
- `model_path`, `[train.fractional]` block shape, `rotation_blocks` (changing the
  block plan mid-adapter corrupts the shard set), work_dir.

## Mechanism (proposed)
1. `evolve ambient nudge <flags>` writes a `nudge.json` into work_dir (atomic
   tmp→rename, same as the stop-file).
2. The daemon loop, at the top of each step (right after the stop-check), reads +
   CONSUMES `nudge.json` (delete-after-apply so a nudge applies once), validates it
   against the safe-live allowlist, and merges accepted fields into the live
   `EvolveConfig`/`DaemonOptions`. Rejected fields are logged (and surfaced to the
   next `watch status`).
3. The applied nudge is recorded in the evolution log (a `kind:"nudge"` row) so
   `watch trend`/`health` can show "focus changed at step N".

## Deliverables
1. `evolve ambient nudge` command (writes the control-file).
2. Daemon-side poll + consume + validated live-merge at the step boundary.
3. Safe-live allowlist + reject-with-reason for restart-required knobs.
4. `nudge` row in the evolution log + surfaced in `watch status`/`health`.
5. Unit tests: allowlist enforcement, weight-merge math, reject path, once-only apply.

## Non-goals
- Changing the training algorithm or the track-15 keep|rollback semantics.
- Hot model/adapter reload (that's track 33's territory).
- A daemon RPC/socket — the control-file poll matches the existing stop-file idiom
  and needs no new transport.

## Risks / open questions (resolve in plan)
- **Live-merge safety**: mutating `EvolveConfig` mid-run must not race the step in
  flight. Apply ONLY at the top of a step (loop owns the config then), never async.
- **Which knobs are truly safe** — the allowlist is the crux; err toward
  restart-required when unsure (reject-with-reason is cheap; a corrupted run is not).
- **Once vs. sticky** — a weight nudge should persist (sticky); a "focus next N
  steps" nudge should expire. Model this explicitly (per-field TTL or a count).
- **Interaction with resume** — a nudge changes in-memory state, not the TOML. On
  restart the TOML wins (nudges are ephemeral). Document this so it isn't surprising.

## Acceptance
- With a daemon running, `evolve ambient nudge --goal scrt-cli --weight 2.0` shifts
  the next step's goal weighting (visible in the log + `watch status`), no restart,
  no broken transaction. A restart-required nudge is rejected with a clear reason.
