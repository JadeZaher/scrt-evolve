# Track 20 — Learning-by-Doing — SIGN-OFF (partial)

Date: 2026-06-20

This signs off the **buildable slices** of track 20 — the ones that stand on
already-shipped tracks (01 discover + palace-search, 02 generate, 19 Python
real-model path). The eval-gated lane (slices 6–9 + the scheduler/txn half of
10) is **deferred**: it requires tracks 10 (eval harness), 15 (transactional
keep|rollback + quarantine), and 11 (regen flywheel), which are not yet built.
No weights are mutated by anything signed off here.

## Signed off (full acceptance)

### Slice 1 — `[[goals]]` config ✅
- `GoalConfig { name, topic, tag, project?, probe_set?, weight?, cadence? }`
  added as an **additive** `goals: Vec<GoalConfig>` on `EvolveConfig`
  (`#[serde(default, skip_serializing_if = "Vec::is_empty")]`,
  `crates/scrt-evolve/src/config.rs`). Exported from `lib.rs`.
- **Non-breaking:** absent `[[goals]]` ⇒ empty vec ⇒ today's single-run
  behavior (styleguide §1 additive `Option`/`Vec` rule).
- Acceptance — *"`evolve.toml` parses `[[goals]]` … absent goals reproduces
  single-run behavior"*: met by `tests/config.rs`:
  `goals_parse_and_round_trip` (all seven fields parse + serialize round-trip),
  `absent_goals_is_empty_and_preserves_single_run`.

### Slice 3 — goal→discover wiring ✅
- `EvolveConfig::for_goal(&GoalConfig) -> EvolveConfig` — **pure** (clones,
  no I/O, leaves `self` untouched): forces `discover.seed = "palace"`, sets
  `palace_search = goal.topic` and `palace_tags = [goal.tag]`, scopes
  `corpus_dir` to `goal.project` when set.
- Acceptance — *only goal-tagged stashes seed*: met by
  `tests/discover.rs::goal_scoped_discover_seeds_only_goal_tagged_stashes`
  (security-tagged stash seeds the auth passage; perf-tagged cache stash is
  filtered out by `palace_tags`). Reuses the existing `write_palace` fixture
  pattern, as specified.

### Slice 4 — transcript harvester ✅
- `crates/scrt-evolve/src/harvest.rs`: `TranscriptEntry` (permissive JSONL
  schema), `capture_and_harvest` (capture-then-filter: **atomic** write of the
  raw transcript to `work_dir/traces/<goal>/<slug>-<date>.jsonl`, then filter +
  distill), and `harvest_entries` (the pure core).
- Trust posture (spec §3): off-goal turns **filtered** out (topic-term match),
  duplicate exchanges **deduped**, every row **provenance-stamped**
  `gen = "trace:<goal.name>"` (quarantinable by track 15 later).
- **Deterministic + clock-free**: the capture date is a caller argument; the
  distill logic uses no wall-clock/RNG (styleguide §2.1/§2.2). Atomic write
  via temp+rename (§2.3). `WorkDir::goal_traces_dir` added.
- Acceptance — *"captures a fixture transcript, filters it, and produces
  `trace:<goal>`-stamped rows that round-trip through the dataset contract"*:
  met by `tests/harvest.rs`:
  `capture_writes_raw_file_and_distills_stamped_rows`,
  `trace_rows_round_trip_through_dataset_contract`, `harvest_is_deterministic`,
  `off_goal_transcript_yields_no_rows`.

## Partial (not signed off — buildable portion done, acceptance lane-gated)

- **Slice 5** (medium-round generate): the per-goal **discover→generate** fan-out
  is built (`goals::run_buildable`, writes `work_dir/goals/<name>/dataset.jsonl`,
  tested in `tests/goals.rs`). The ~100+-row medium-round sizing + stash/trace
  merge belongs with the eval-gated round driver (slice 6). Carry-forward.
- **Slice 10** (CLI surface): `evolve train auto --goals` exists and runs the
  bounded buildable loop (no weight mutation). The scheduler + through-the-txn
  weight path are slices 9/6. Carry-forward.

## Deferred (lane-gated — see plan.md "Carry-forward")
Slices 6 (eval-gated round driver), 7 (catastrophe/rollback + quarantine),
8 (regen flywheel), 9 (scheduler). Gated on tracks 10 / 15 / 11.

## Verification (one integrated sweep, per the test policy)
- `cargo test` (default, **ML-free + Python-free**): all suites green —
  including the new `config` (14), `discover` (7), `harvest` (4), `goals` (3).
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- Default build pulls **no candle, no Python** (slice work is all pure Rust /
  std; the heavy ML path remains the track-19 subprocess).
