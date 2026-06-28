---
type: Track Spec
title: "Closeout & Polish"
description: Bring scrt-evolve to a finished, functional, well-documented state (audit/retro/docs).
tags: [track-30, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 30 — Closeout & Polish — bring scrt-evolve to a finished, functional, well-documented state — Specification

## Goal
Take scrt-evolve from "the lane is built (tracks 00–29 landed)" to **picture-perfect completed**:
a functional, well-tested, well-architected, easy-to-read/use/modify codebase with honest
documentation of what's done, a strong usage README, good agents documentation, and a critical
pass over **DevUX** (how a human drives it) and **AIUX** (how an agent drives it) with concrete
refinements applied. This is the capstone "make it shippable + legible" track — **no new product
ML**; it consolidates, audits, documents, and ergonomically refines what already exists.

## Why now
The spine grew to 30 tracks across five lanes (core, self-evolve, architecture, bench, product).
Almost everything is built (only 26 + 28 remain unstarted), track 29 is live-validated, but the
repo reads like a build log, not a finished product: the status of each track is scattered, there
are no retros, the README/AGENTS docs lag the code, and nobody has critically asked "is this
pleasant to use — for a person or for an agent?" This track closes that gap.

## Scope
- **P0 — Finish the live branch** (functional proof). Complete the in-flight TinyLlama→scrt-CLI
  run: eval gate → Q4_K_M GGUF export → `branch register` → serve a real prompt and capture the
  output. The repo must contain at least one **demonstrably working branch** end-to-end.
- **P1 — Track pruning + status truth.** Collapse `tracks.md` into a single **authoritative build
  status** (one row per track: Done / Not-started, the shipped module(s), the test file(s)).
  Verify every "Done" claim maps to real code + a passing test (no aspirational greens). Move the
  verbose per-track narrative behind the per-track dirs; the spine doc becomes a legible map, not a
  design dump. Reconcile the dependency graph with reality.
- **P2 — Retros.** A consolidated `conductor/RETRO.md`: per-lane retrospective (what shipped, what
  diverged from DESIGN.md, the load-bearing decisions, what to do differently). Short, honest,
  decision-focused — not a changelog.
- **P3 — Test + architecture audit.** Full sweep green (`cargo test`, `clippy -D warnings`,
  `cargo fmt --check`; Python tests). Assess coverage; fill **critical** gaps only (no coverage
  theater). One readability/architecture pass: module boundaries, naming, dead code, over-long
  functions, doc-comment accuracy — so the code is **easy to read and modify**. Apply
  `code-simplifier`-style cleanups where they reduce cognitive load without changing behavior.
- **P4 — Documentation.** A **strong README**: what scrt-evolve is (one paragraph), install, a
  copy-pasteable quickstart, the `discover → generate → train → eval → export → branch` flow with
  real commands, the `[branch]` factory, and the honest constraints (Python/transformers is the
  real ML path; candle is fixture; WSL/GPU notes). A good **AGENTS.md**: the SDK + CLI surface an
  agent uses, the dataset/manifest contracts, and how the paired `scrt-evolve` skill steers it. A
  **COMPLETED.md** (or a section): what's built, what's deferred, with pointers.
- **P5 — DevUX + AIUX critical analysis + refinement.** Write a critique (`conductor/UX-REVIEW.md`)
  then **apply** the high-value fixes:
  - **DevUX**: CLI help/usage clarity, error messages (actionable, not stack-traces), sane
    defaults, discoverability (`--help` completeness), the config schema reference, friction in the
    common path. Fix the worst offenders.
  - **AIUX**: can an agent drive this confidently? Clear command affordances, structured/parseable
    outputs (JSON where it matters), unambiguous failure signals, the skill's instructions, the
    dataset/manifest contracts being self-describing. Fix the worst offenders.
- **P6 — Final verification + sign-off.** Everything green; docs match code (spot-checked); the
  repo presents as a finished, legible product. Update `tracks.md` status for 30; write the sign-off.

## Out of scope
- New product features or ML (tracks 26/28 stay separate, listed as the only open build work).
- Publishing/release (track 08) unless trivially in reach; note it, don't force it.
- Rewriting working subsystems — refine for legibility/ergonomics, don't re-architect.

## Constraints
- **Honest greens only.** A "Done" status must map to shipped code + a passing test; if a claim
  can't be verified, it's downgraded, not left aspirational.
- **No behavior regressions.** Readability/ergonomic refactors keep tests green; run the full sweep
  after each phase's edits (batch edits, then one sweep — the working rhythm).
- **Refine, don't rewrite.** Smallest change that improves legibility/UX; preserve the architecture.
- **Docs are load-bearing artifacts**, written from the real code, not from memory — verify commands.
- **Two open build tracks (26, 28) are acknowledged, not closed** — the "completed" claim is scoped
  to the shipped lane + this polish, with 26/28 clearly flagged as the remaining roadmap.

## Acceptance
- A real branch is served end-to-end and its output captured (P0).
- `tracks.md` is a single authoritative status table; every "Done" verified against code + tests;
  the dependency graph matches reality (P1).
- `conductor/RETRO.md` exists with a per-lane retrospective (P2).
- Full sweep green: `cargo test` + `clippy -D warnings` + `cargo fmt --check` + Python tests;
  critical coverage gaps filled; one readability pass applied (P3).
- README quickstart is copy-pasteable and accurate; AGENTS.md documents the SDK/CLI/contracts;
  completed-work doc present (P4).
- `conductor/UX-REVIEW.md` critiques DevUX + AIUX, and the high-value fixes are applied + verified
  (P5).
- Final sweep green; docs spot-checked against code; track 30 signed off in `tracks.md` (P6).

## Dependencies
- All shipped tracks (00–25, 27, 29) — this audits/documents them.
- The live branch run (track 29 §Live validation) — P0 finishes it.
- Independent of the two open build tracks (26 ambient, 28 packaging) — those are the roadmap, not
  blockers; this track documents them as "next."

## Cogency-audit notes (apply throughout)
- **Status truth over status optimism** — verify, don't assert. A green that can't be reproduced is
  a defect of the doc, not a feature.
- **Refine-not-rewrite** — the architecture is sound; this is legibility + ergonomics, not redesign.
- **DevUX and AIUX are distinct lenses** — a tool pleasant for a human can still be opaque to an
  agent (and vice-versa); critique both, fix both.
- **Scope the "completed" claim honestly** — shipped lane + polish complete; 26/28 are the named
  remaining roadmap, not hidden.
