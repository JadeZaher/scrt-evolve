---
type: Implementation Plan
title: "Closeout & Polish"
description: "Implementation plan for the Closeout & Polish track."
tags: [track-30, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 30 — Closeout & Polish — Plan

The capstone "make it finished + legible" track. Doc/audit/ergonomics heavy; no new product ML.
Phases are mostly independent → parallelizable (P1–P5 can fan out); P0 finishes the live branch,
P6 is the final gate. Working rhythm: batch edits per phase, then **one** verification sweep.

## Phase 0 — Finish the live branch (functional proof)
1. [x] Complete the in-flight TinyLlama→scrt-CLI run: eval (`scrt_evolve_score` on the probe) →
   Q4_K_M GGUF export (`scrt_evolve_gguf`) → `branch register` (manifest + registry) → serve a
   real prompt (`llama-completion`) and capture the output. Free GPU first (`lms unload --all`).
   **Done when:** `branch list` shows `scrt-cli`, and a served prompt returns a domain answer.
2. [x] Record the result (loss, eval correctness, a sample prompt+completion) in the track-29 plan
   §Live validation and in a README example.

## Phase 1 — Track pruning + status truth
3. [x] Audit every track 00–29: does each "Done" map to a shipped module + a passing test? Build a
   table `{ track, status (Done|Not-started), module(s), test file(s) }`. Downgrade any unverifiable
   green.
4. [x] Rewrite `tracks.md` as an **authoritative status map**: the table up top, the dependency
   graph reconciled to reality, the verbose per-track design prose trimmed (detail lives in the
   per-track dirs). Keep the phase-gate list but mark which gates are met.
5. [x] Confirm the only open build tracks are **26 (ambient)** + **28 (packaging)**; flag them as the
   roadmap. Note track 08 (publish) status.

## Phase 2 — Retros
6. [x] `conductor/RETRO.md`: per-lane retrospective — **core** (00–08), **self-evolve** (10–15),
   **architecture** (16–18), **bench/training** (21–27), **product/BTM** (29). For each: what
   shipped, what diverged from DESIGN.md, the load-bearing decisions, one "do differently." Honest +
   decision-focused, not a changelog.

## Phase 3 — Test + architecture audit
7. [x] Full sweep + coverage read: `cargo test`, `clippy -D warnings`, `cargo fmt --check`, Python
   tests. List any **critical** gaps (untested public surface, contract types); fill those only.
8. [x] Readability/architecture pass (delegate to `code-reviewer` + `code-simplifier`): module
   boundaries, naming, dead code, over-long functions, stale doc-comments. Apply low-risk legibility
   cleanups (behavior-preserving); re-run the sweep once after.

## Phase 4 — Documentation
9. [x] **README.md** rewrite: one-paragraph what-it-is; install; copy-pasteable quickstart; the
   `discover → generate → train → eval → export → branch` flow with real commands; the `[branch]`
   factory; honest constraints (transformers = real ML path, candle = fixture; WSL/GPU notes from
   the bench RUNBOOK). Verify every command.
10. [x] **AGENTS.md**: the SDK + CLI surface an agent uses, the dataset.jsonl + manifest/registry
    contracts, the `BranchRouter` seam, and how the paired `scrt-evolve` skill steers a frontier
    agent. Point to `SCRT-EVOLVE-INTEGRATION.md` for the hivemind contract.
11. [x] **COMPLETED.md** (or a README section): what's built (by lane) + what's deferred (26/28),
    with pointers to tracks + retros.

## Phase 5 — DevUX + AIUX critical analysis + refinement
12. [x] `conductor/UX-REVIEW.md`: critique **DevUX** (CLI help/usage, error messages, defaults,
    discoverability, config-schema reference, common-path friction) and **AIUX** (command
    affordances, structured/JSON outputs, failure signals, skill instructions, self-describing
    contracts). Rank findings by value/effort.
13. [x] Apply the high-value fixes: clearer `--help`/usage text, actionable error messages (not raw
    anyhow chains where a hint helps), saner defaults, `--json` where an agent needs it, skill/doc
    tweaks. Keep each change small; re-run the sweep after.

## Phase 6 — Final verification + sign-off
14. [x] Final full sweep GREEN (cargo test + clippy + fmt + Python). Spot-check that README/AGENTS
    commands actually run (or are clearly env-gated). Confirm docs match code.
15. [x] Update `tracks.md`: add the track-30 row + "After 30" gate; mark the status table current.
    Write the closeout sign-off (what "completed" means + the 26/28 roadmap). Update memory.

## Status
DONE (2026-06-26) — driven to completion via orchestrated agents + the main loop. Final sweep GREEN
(`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` all pass).

Delivered:
- **P0 — functional proof.** TinyLlama-1.1B → scrt-CLI branch trained on the RTX 4060 (loss
  3.70→0.05), exported to a 667 MB Q4_K_M GGUF, **registered**, **routed** (`branch route` → scrt-cli
  @0.53), and **served** a real domain completion. End-to-end factory proven. (Eval correctness 0.0 —
  honest: tiny base + tiny data; the gate correctly wouldn't auto-admit it. Recorded in track-29 plan.)
- **P1 — status truth.** `tracks.md` now leads with an AUTHORITATIVE verified build-status map. The
  audit corrected an over-optimistic earlier read: genuinely shipped = 00–04, 10, 15, 19, 20, 21,
  23, 24, 25, 27, 29 (+ the bench lane); designed-not-built = 05–09, 11–14, 16–18, 22(partial), 26, 28.
- **P2 — retros.** `conductor/RETRO.md`: per-lane retrospective (core / self-evolve / architecture /
  bench / product) with honest "diverged / load-bearing / do-differently", incl. the candle→
  transformers amendment, the track-10 shared-evaluator audit fix, and the gates-met-on-paper-no-code
  finding.
- **P3 — test + arch audit.** Full sweep green; filled the critical contract gaps (new
  `tests/dataset.rs`: ToolCall/Cli JSONL round-trip + malformed-line error); applied safe readability
  fixes (stale scheduler doc-comments, PYTHONPATH separator bug, `provenance_of` exhaustiveness).
- **P4 — docs.** Rewrote `README.md` (usage-first quickstart), added `AGENTS.md` (SDK/CLI/contracts
  operator map) + `COMPLETED.md` (capability→command→test + roadmap).
- **P5 — DevUX/AIUX.** `conductor/UX-REVIEW.md` (both lenses, ranked). Applied: candle-fixture
  warning, `export-gguf --quant` Option (kill silent precedence), `scorer_backend` doc-default fix.
  **Documented-not-applied** (recommended follow-ups, in UX-REVIEW): actionable subprocess-failure
  hints, a global `--json` summary, a `doctor` preflight command, and the larger refactors (run()
  split, params structs, subprocess-launch helper).
- **P6 — sign-off.** Sweep green; docs ground-checked; `tracks.md` "After 30" gate + this sign-off.

**Honest scope of "completed":** the shipped lane is finished, tested, documented, and ergonomically
refined; the advanced self-evolve (11–14), architecture/SDK (16–18), several train presets (05–07),
and ambient (26) + packaging (28) lanes remain the named roadmap (see the tracks.md status map). A
few larger UX refinements are documented in UX-REVIEW.md as the next ergonomics pass.
