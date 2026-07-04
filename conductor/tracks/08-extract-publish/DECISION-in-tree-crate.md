# Decision — retire the upstream in-tree `scrt-evolve` crate (task 3)

## Context

See `DESIGN.md` Relationship to the in-tree scrt-evolve crate (~L547-553). The
original scrt-cli workspace contained a `crates/scrt-evolve` (corpus export +
InfoNCE seam behind a `train` feature). That code was the **seed** for THIS
standalone repo (the contrastive preset + corpus/dataset plumbing), lifted in the
design's phase 6.

> **Note:** this decision is about the **OLD** crate still living in the
> scrt-cli workspace, **not** about this repo's own `crates/scrt-evolve` SDK.

## Decision

**RETIRE it** — delete from the scrt-cli workspace — rather than leave a thin
re-export shim.

## Rationale

- A thin re-export would make scrt-cli depend on scrt-evolve, which itself
  depends on scrt-core (also in scrt-cli) — a confusing near-cyclic dependency.
- There are no known external consumers of the old crate's public surface.
- Its functionality is fully superseded by this standalone repo.

## Execution

Performed **upstream** in the scrt-cli repo, coupled with the scrt-core 0.1
crates.io publish (the user's release step). If a consumer of the old surface
ever surfaces, add a thin `pub use` shim then.