---
type: note
track: 35
title: Inline passoff — nudge / live retuning
status: planned
created: 2026-06-30
---

# Track 35 — Inline passoff (nudge / live retuning)

## One-line
`evolve ambient nudge <flags>` steers a RUNNING daemon (goal weights / focus /
throttle knobs) — applied at the next step boundary via a polled control-file, no
restart. "nudge" = the direction knob; `teach` stays the content knob.

## Why deferred
The CLI verb is trivial; the daemon-side control channel + safe live-merge
(without breaking the track-15 transaction or resume invariants) is real design
work. Per the "complex → track + passoff" pattern.

## The mechanism to build (mirror the stop-file)
The daemon ALREADY polls a `daemon.run` stop-file at the top of each step
(`crates/scrt-evolve/src/daemon.rs`, `run_file`/`stop_requested`). Build nudge the
same way:
1. `evolve ambient nudge` writes `work_dir/nudge.json` (atomic tmp→rename).
2. The loop, at the step top (right after the stop-check), reads + DELETES
   `nudge.json` (once-only apply), validates against a **safe-live allowlist**,
   merges accepted fields into the live `EvolveConfig`/`DaemonOptions`.
3. Log a `kind:"nudge"` row; surface in `watch status`/`health`.

## Safe-live allowlist (the crux — err toward reject)
- SAFE (pure gate/policy inputs, re-read each step): goal weights, active focus
  (next-N filter), `max_vram_gb`, `min_free_ram_gb`, `cooldown_secs`,
  `max_minutes_per_hour`, `poll_interval_secs`, `[regulate].gate`.
- REJECT-with-reason (would corrupt an in-flight adapter/plan): `model_path`,
  `[train.fractional]` shape, `rotation_blocks`, `work_dir`.

## Landmarks
- `crates/scrt-evolve/src/daemon.rs` — `run_file`/`stop_requested` (the pattern to
  mirror), the step loop (apply point = top of step), `DaemonOptions` (the live knobs).
- `crates/scrt-evolve/src/config.rs` — `EvolveConfig`, `[[goals]]`, `[regulate]`.
- `crates/scrt-evolve-cli/src/main.rs` — the new `ambient` subcommand group (added in
  the CLI-rebrand work); add `nudge` there next to `start/stop/teach`.
- Weighted goal policy — where goal weights feed the pop/selection (search the
  living-queue + rounds policy).

## Watch-outs
- Apply ONLY at the step top (loop owns the config then) — never async, or you race
  the in-flight step.
- Sticky vs. expiring nudges: a weight change persists; a "focus next N steps"
  expires. Model per-field (TTL or count).
- Nudges are EPHEMERAL (in-memory). On `ambient start` the TOML wins. Document it.
- Don't reintroduce a steer CLI — nudge IS the steer verb; teach stays for content.
