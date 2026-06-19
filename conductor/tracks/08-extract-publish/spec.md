# Extract / Publish — Specification

## Goal
DESIGN.md phase 9 — the finalization track. Swap the scrt-core **git dep** for
the **published crate**, retire or thin-re-export the in-tree
`crates/scrt-evolve`, and cut scrt-evolve's own first release.

## Scope
- Replace `scrt-core = { git = … }` with `scrt-core = "0.1"` (crates.io) — the
  one-line change DESIGN.md flagged as its own task. Verify the published API
  matches what `discover`/`contrastive` call (`search_with_meta`,
  `palace::FilePalace`/`ops`, `palace::simhash`).
- Resolve the in-tree `crates/scrt-evolve` relationship (DESIGN.md
  §Relationship): now that its corpus/InfoNCE seam is lifted (track 05), either
  retire it or leave a thin re-export. Decide and execute here.
- Release hygiene: version, CHANGELOG, README install/usage, license, the
  feature-flag matrix (`default` / `train` / `pyo3`) documented, CI building all
  three.
- Tag the first release.

## Constraints
- The crates.io swap must be **truly one-line** at the dep level — if the
  published API drifted, that's a fix in the call sites, called out explicitly,
  not a silent rewrite.
- Don't break the byte-parity / dataset-contract guarantees other consumers
  (the Python bridge, hivemind workers) depend on across a version bump.
- Releasing publishes outward — confirm the version, tag, and any registry push
  before doing it (no surprise publish).

## Acceptance
- `cargo build` / `cargo test` (and `--features train`, `--features pyo3`) all
  green against the **published** scrt-core, no git dep remaining.
- The in-tree `crates/scrt-evolve` is retired or reduced to a documented thin
  re-export (decision recorded).
- README documents install, the three feature builds, and the
  discover→generate→train flow; CHANGELOG covers tracks 00–08.
- A tagged release exists; CI builds the default + `train` + `pyo3` matrices.

## Dependencies
All prior tracks (00–07). External: a **published** scrt-core (this track is
blocked until that release exists; until then the git-dep pin stands).
