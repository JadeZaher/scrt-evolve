# Extract / Publish — Plan

## Tasks

1. [ ] Confirm published scrt-core exists + its API matches call sites
   (`search_with_meta`, `palace::FilePalace`/`ops`, `palace::simhash`). -- evidence: API diff vs git-dep.
2. [ ] Swap `scrt-core = { git … }` → `scrt-core = "0.1"`; fix any drifted
   call sites (called out, not silent). -- evidence: Cargo.toml + any patch.
3. [ ] Resolve the in-tree `crates/scrt-evolve`: retire or thin re-export;
   record the decision. -- evidence: decision note + repo state.
4. [ ] Release docs: README (install, 3 feature builds, full flow), CHANGELOG
   (tracks 00–08), license, feature-flag matrix. -- evidence: docs present.
5. [ ] CI builds default + `--features train` + `--features pyo3`. -- evidence: CI config.
6. [ ] Full sweep across all feature combos. -- evidence: green matrix.
7. [ ] Confirm version/tag with the user, then tag the first release
   (no surprise registry push). -- evidence: tag.

## Sign-off
Pending — this is the terminal track; sign-off here closes the spine.
