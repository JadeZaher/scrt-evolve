# Learning-by-Doing — Incremental Multi-Goal Evolution — Plan

Build in slices; each slice is independently runnable and leans on what already
ships (discover+palace-search, generate, Python train/infer/GGUF). The
eval-gated round + scheduler depend on the lane (tracks 10, 15) existing.

## Tasks

1. [ ] `[[goals]]` config: `GoalConfig { name, topic, tag, project?,
   probe_set?, weight?, cadence? }` as an additive `Vec` on `EvolveConfig`
   (serde defaults; absent ⇒ single-run). -- evidence: config round-trip test (goals parse; empty = today).
2. [ ] Author the `scrt-evolve` SKILL.md (frontmatter + body) that pairs with
   `scrt-context`: instructs goal-tagged stashing, cross-references scrt-context,
   worked example (stash with `--mp-tag <goal.tag>` → discover via `palace_tags`).
   -- evidence: SKILL.md committed; example commands valid against the built CLI.
3. [ ] Goal→discover wiring: for a goal, set `discover.palace_search=topic` +
   `discover.palace_tags=[tag]` and run discover scoped to `project`. -- evidence: goal-scoped discover test (only goal-tagged stashes seed).
4. [ ] Transcript harvester: capture a frontier transcript to
   `work_dir/traces/<goal>/<slug>-<date>.jsonl`, filter (capture-then-filter),
   distill goal-relevant parts → rows stamped `gen=trace:<goal>`. -- evidence: fixture-transcript → trace rows round-trip; off-goal noise dropped.
5. [ ] Medium-round generate: produce ~100+ rows per round from stashes (+ trace
   rows), deduped, provenance-stamped. -- evidence: round dataset size + provenance test.
6. [ ] Eval-gated round driver: checkpoint → train (track 19) → eval (track 10
   `Scorer` vs goal `probe_set`) → keep|rollback via track 15 txn; append verdict
   to `evolution-log.jsonl`. -- evidence: pass commits + advances last_good; forced regress rolls back (state restored).
7. [ ] Catastrophe handling: forced collapse/NaN → auto-rollback + quarantine the
   round's `gen` provenance + halt; next round skips it. -- evidence: catastrophe test (halt + quarantine + skip).
8. [ ] Generation-improves-itself: allow a kept round's model to be the next
   round's generator (regen, track 11) with antagonist-ratio decay + teacher
   anchor. -- evidence: ≥2 swaps; held-out score holds; un-gated self-output never admitted.
9. [ ] Scheduler: bounded driver over goals (round-robin / weighted), cadence or
   on-demand, budget/stop-condition, non-interactive + resumable. -- evidence: ≥2 rounds across ≥2 goals, budget-bounded, resumes after interrupt.
10. [ ] `scrt-evolve evolve --goals` / daemon-mode entrypoint (CLI surface) that
    runs the scheduler; daemon may ONLY auto-evolve THROUGH this (→ track 15).
    -- evidence: CLI runs a bounded multi-goal session; weight changes go through the txn.
11. [ ] Docs: README "learning by doing" section; DESIGN amendment tying the
    daemon north-star to this track; install scrt-evolve as a skill. -- evidence: docs updated; skill install documented.
12. [ ] Final sweep: `cargo test` (default), `cargo clippy`, the Python harvest +
    round tests, skill-example smoke. -- evidence: green.

## Build order note
Slices 1–3 + the skill (2) + transcript harvest (4) + medium-round generate (5)
are buildable NOW on shipped tracks (01/02/19). Slices 6–10 (eval-gated round,
catastrophe, regen flywheel, scheduler) require the lane (tracks 10, 15, 11) to
land first — this track is their consumer/orchestrator, not their owner.

## Sign-off
Pending — design only (2026-06-20). No code yet; this is the spec for the
learning-by-doing capstone. Slices 1–5 + skill can start on shipped tracks;
6–10 gated on the self-evolve lane (10/15/11).
