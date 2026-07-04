---
type: Implementation Plan
title: Extract / Publish
description: Implementation plan for the Extract / Publish track.
tags: [track-08, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Extract / Publish — Plan

## Tasks

1. [ ] Confirm published scrt-core exists + its API matches call sites
   (`search_with_meta`, `palace::FilePalace`/`ops`, `palace::simhash`). -- evidence: API diff vs git-dep.
   -- BLOCKED (user): needs the published crate. Prep landed: full consumed surface
   documented in `API-SURFACE.md` (broader than the shorthand — also `types::*`,
   `SourceInput`, `Palace` trait, `ops::list_stashes`); task 1 must verify ALL of it.
2. [~] Swap `scrt-core = { git … }` → `scrt-core = "0.1"`; fix any drifted
   call sites (called out, not silent). -- evidence: Cargo.toml + any patch.
   -- PREP DONE (swap itself is the user's, post-publish): confirmed truly one line —
   root `Cargo.toml:46` in `[workspace.dependencies]`; members already use
   `scrt-core.workspace = true`. Ready FROM/TO diff in `API-SURFACE.md`.
3. [x] Resolve the in-tree `crates/scrt-evolve`: retire or thin re-export;
   record the decision. -- evidence: `DECISION-in-tree-crate.md` — decided RETIRE
   the UPSTREAM scrt-cli in-tree crate (not this repo's SDK; a re-export would make
   scrt-cli→scrt-evolve→scrt-core near-cyclic). Executed upstream w/ the 0.1 publish.
4. [x] Release docs: README (install, 3 feature builds, full flow), CHANGELOG
   (tracks 00–08), license, feature-flag matrix. -- evidence: README already
   acceptance-complete (install/3-builds/flow/MIT); added `CHANGELOG.md` (honest
   0.1.0 shipped-set + roadmap split) + a "Feature flags" matrix in README.
5. [x] CI builds default + `--features train` + `--features pyo3`. -- evidence:
   `.github/workflows/ci.yml` — `lint` (fmt+clippy) + `build-and-test` 3-leg matrix
   (default/train/pyo3, `--locked`, setup-python for pyo3). pyo3 leg is BUILD-ONLY:
   `extension-module` omits libpython linkage on Linux, so test executables can't
   link (pyo3 FAQ; caught in the 2026-07-03 code review).
6. [x] Full sweep across all feature combos. -- evidence: green matrix 2026-07-03 —
   default build+test all green; `--features train` build green; `--features pyo3`
   initially FAILED (bridge.rs E0004: track-37 `GenExample::Skill`/`ReasoningEdit`
   arms missing at 2 match sites — the known field-add ripple), fixed exhaustively
   (no wildcard) mirroring export.rs/serde conventions; pyo3 build+test green after.
7. [ ] Confirm version/tag with the user, then tag the first release
   (no surprise registry push). -- evidence: tag. -- USER handles (no outward push here).

## Sign-off
Pending — this is the terminal track; sign-off here closes the spine.
Remaining before sign-off: task 6 green sweep + the user's post-publish steps
(tasks 1 verify / 2 commit the one-line swap / 7 tag).
