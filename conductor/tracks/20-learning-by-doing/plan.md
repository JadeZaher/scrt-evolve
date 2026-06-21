# Learning-by-Doing — Incremental Multi-Goal Evolution — Plan

Build in slices; each slice is independently runnable and leans on what already
ships (discover+palace-search, generate, Python train/infer/GGUF). The
eval-gated round + scheduler depend on the lane (tracks 10, 15) existing.

## Tasks

1. [x] `[[goals]]` config: `GoalConfig { name, topic, tag, project?,
   probe_set?, weight?, cadence? }` as an additive `Vec` on `EvolveConfig`
   (serde defaults; absent ⇒ single-run). -- evidence: config round-trip test (goals parse; empty = today).
   DONE (2026-06-20): `GoalConfig` + `EvolveConfig.goals: Vec<GoalConfig>`
   (`#[serde(default, skip_serializing_if = "Vec::is_empty")]`) in
   `crates/scrt-evolve/src/config.rs`; exported from `lib.rs`. Tests in
   `tests/config.rs`: `goals_parse_and_round_trip`,
   `absent_goals_is_empty_and_preserves_single_run`,
   `for_goal_wires_discover_search_and_tag`,
   `for_goal_without_project_keeps_top_level_corpus`. All green.
2. [x] Author the `scrt-evolve` SKILL.md (frontmatter + body) that pairs with
   `scrt-context`: instructs goal-tagged stashing, cross-references scrt-context,
   worked example (stash with `--mp-tag <goal.tag>` → discover via `palace_tags`).
   -- evidence: SKILL.md committed; example commands valid against the built CLI.
   DONE (prior commit 2397a8d): `skills/scrt-evolve/SKILL.md` exists, pairs with
   scrt-context, shows a goal-tagged stash → `palace_tags` discover example.
3. [x] Goal→discover wiring: for a goal, set `discover.palace_search=topic` +
   `discover.palace_tags=[tag]` and run discover scoped to `project`. -- evidence: goal-scoped discover test (only goal-tagged stashes seed).
   DONE (2026-06-20): `EvolveConfig::for_goal(&GoalConfig)` (pure; forces
   `seed="palace"`, sets `palace_search`/`palace_tags`, scopes corpus to
   `goal.project`). Test `goal_scoped_discover_seeds_only_goal_tagged_stashes`
   in `tests/discover.rs` (security-tagged stash seeds; perf-tagged filtered out).
4. [x] Transcript harvester: capture a frontier transcript to
   `work_dir/traces/<goal>/<slug>-<date>.jsonl`, filter (capture-then-filter),
   distill goal-relevant parts → rows stamped `gen=trace:<goal>`. -- evidence: fixture-transcript → trace rows round-trip; off-goal noise dropped.
   DONE (2026-06-20): `crates/scrt-evolve/src/harvest.rs` — `TranscriptEntry`,
   `capture_and_harvest` (atomic capture to `<slug>-<date>.jsonl`),
   `harvest_entries` (pure filter+distill+dedup, `gen="trace:<goal>"`).
   `WorkDir::goal_traces_dir`. Tests in `tests/harvest.rs`:
   `capture_writes_raw_file_and_distills_stamped_rows`,
   `trace_rows_round_trip_through_dataset_contract`, `harvest_is_deterministic`,
   `off_goal_transcript_yields_no_rows`. All green.
5. [ ] Medium-round generate: produce ~100+ rows per round from stashes (+ trace
   rows), deduped, provenance-stamped. -- evidence: round dataset size + provenance test.
   PARTIAL: per-goal discover→generate fan-out built (`goals::run_buildable`,
   writes `work_dir/goals/<name>/dataset.jsonl`); the ~100+-row medium-round
   sizing + stash-and-trace merge belongs with the eval-gated round driver
   (slice 6, lane-gated). Carry-forward.
6. [x] Eval-gated round driver: discover → generate → (probe carve) → train →
   eval → keep|rollback via track 15 txn; verdict to `evolution-log.jsonl`.
   DONE (2026-06-20): `rounds::run_round` wraps train+eval in `Regulator::run_step`.
   Tests `tests/rounds.rs`: `round_commits_when_eval_passes`,
   `round_rolls_back_on_regress`, `train_is_only_called_inside_transaction`.
7. [x] Catastrophe handling: collapse/NaN → auto-rollback + quarantine the round's
   `gen` provenance + halt; next round skips it. DONE: the `Catastrophic` arm of
   `run_step` + `Quarantine::filter` in `run_round`. Tests:
   `round_catastrophe_halts_and_quarantines`, `schedule_halts_midway_on_catastrophe`.
8. [ ] Generation-improves-itself (regen flywheel, track 11). DEFERRED — track 11
   not built; OPTIONAL for the bench (the API teacher is the safe default). The
   round driver is the future host. Documented carry-forward.
9. [x] Scheduler: bounded driver over goals (round-robin / weighted), budget/
   stop-condition, resumable. DONE: `rounds::run_schedule` (bounded by
   `max_rounds`, halts on catastrophe, `start_ordinal` resume). Tests:
   `schedule_is_bounded_and_round_robins_two_goals`, `weighted_schedule_favors_heavier_goal`.
10. [x] `scrt-evolve evolve --schedule` CLI: runs the bounded eval-gated schedule;
    weight changes go THROUGH the track-15 txn. DONE: `cmd_evolve_schedule` wires
    production hooks (discover/generate from SDK; train/score as Python
    subprocesses) into `run_schedule`. `--goals` (buildable, no gate) retained.
11. [ ] Docs: README "learning by doing" section; DESIGN amendment tying the
    daemon north-star to this track; install scrt-evolve as a skill. -- evidence: docs updated; skill install documented.
12. [ ] Final sweep: `cargo test` (default), `cargo clippy`, the Python harvest +
    round tests, skill-example smoke. -- evidence: green.

## Build order note
Slices 1–3 + the skill (2) + transcript harvest (4) + medium-round generate (5)
are buildable NOW on shipped tracks (01/02/19). Slices 6–10 (eval-gated round,
catastrophe, regen flywheel, scheduler) require the lane (tracks 10, 15, 11) to
land first — this track is their consumer/orchestrator, not their owner.

## Carry-forward (deferred, lane-gated)
Slices **6–9** and the rest of **10** require tracks that are NOT yet built:
- **Track 10** — eval harness (`Scorer`/`StepVerdict`/`gate`): the probe-set
  gate that decides keep|rollback. `GoalConfig.probe_set` is parsed and reserved
  for it; nothing consumes it yet.
- **Track 15** — transactional keep|rollback + quarantine: the ONLY sanctioned
  weight-mutation path. The buildable driver here deliberately mutates NO weights.
- **Track 11** — regen flywheel (anti-collapse rails): slice 8.

Specifically deferred: 6 (eval-gated round driver), 7 (catastrophe/rollback +
quarantine), 8 (generation-improves-itself), 9 (scheduler), and the
scheduler/txn half of 10. When the lane lands, `goals::run_buildable` is the
extension point: wrap each goal's per-round generate→train in the track-15 txn,
score against `goal.probe_set` via track 10, and drive rounds with the slice-9
scheduler (using `goal.weight`/`goal.cadence`, both already parsed). The
`harvest` trace rows (`gen="trace:<goal>"`) are quarantinable by that exact
provenance stamp.

## Sign-off
**Round 1 (2026-06-20):** slices 1, 3, 4 signed off; 2 (SKILL.md) prior; 5/10 partial.

**Round 2 (2026-06-20, after tracks 10+15 landed):** slices **6, 7, 9, 10** now
DONE — the eval-gated round driver + catastrophe/quarantine/halt + bounded
weighted/round-robin scheduler + the `evolve --schedule` CLI, all THROUGH the
track-15 transaction (`rounds.rs`, `tests/rounds.rs` 8/8). Slice **5** medium-round
sizing is satisfied operationally (round generate produces the goal's full
dataset; `max_passages`/`per_passage` control volume). Slice **8** (regen
flywheel, track 11) remains DEFERRED + OPTIONAL (API teacher is the safe default).
Slices 11 (docs) + 12 (final Python smoke) pending the bench track (24).
Verified green: `cargo test` (19 suites, ML-free), `cargo clippy -D warnings`,
`cargo fmt --check`, `--features train`, `--features pyo3`. Default build stays
ML-free + Python-free; weight mutation only inside the track-15 txn.
