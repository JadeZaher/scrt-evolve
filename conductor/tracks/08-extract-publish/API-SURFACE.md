# scrt-core API surface consumed by scrt-evolve (task 1/2 prep)

The exact published-crate surface that `scrt-core` 0.1 (crates.io) **MUST** expose for
the git→crates.io dependency swap to be truly one-line. Verified by grep against
the current git-dep build.

---

## Top-level

| Symbol | Call site |
|---|---|
| `scrt_core::search_with_meta(&SearchConfig) -> (SearchResult, meta)` | `crates/scrt-evolve/src/discover.rs:16` (import), `:113` (call) |
| `scrt_core::SearchConfig` | `discover.rs:16` (import), `:205` (construction) |
| `scrt_core::SourceInput` | `discover.rs:16` (import) |

## `scrt_core::tool_spec`

| Symbol | Call site |
|---|---|
| `tool_spec::build_tool_spec("openai") -> Result<...>` | `crates/scrt-evolve/src/toolspec.rs:26` (inline-qualified — no `use`, easy to miss in import-only greps) |

## `scrt_core::types`

| Symbol | Call site |
|---|---|
| `types::Effort` | `crates/scrt-evolve/src/discover.rs:15` |
| `types::Node` | `crates/scrt-evolve/src/discover.rs:15` |
| `types::SearchOptions` | `crates/scrt-evolve/src/discover.rs:15` |
| `types::SortMode` | `crates/scrt-evolve/src/discover.rs:15` |
| `types::Strategy` | `crates/scrt-evolve/src/discover.rs:15` |
| `types::WindowCurve` | `crates/scrt-evolve/src/discover.rs:15` |

## `scrt_core::palace`

| Symbol | Call site |
|---|---|
| `palace::FilePalace` | `discover.rs:14`, `crates/scrt-evolve/src/plan/signals.rs:15` |
| `palace::Palace` (trait) | `discover.rs:14`, `crates/scrt-evolve/src/plan/signals.rs:15` |
| `palace::simhash` (module) | `discover.rs:14`, `crates/scrt-evolve/src/branch/router.rs:13` |
| `palace::ops::SystemClock` | `discover.rs:14,144`, `signals.rs:15` |
| `palace::ops::list_stashes(...)` | `discover.rs:14,144`, `signals.rs:15` |

---

## Notes

- **This is the FULL surface** — broader than the plan.md shorthand of just
  `search_with_meta` / `FilePalace` / `ops` / `simhash`. Verified by a bare
  `scrt_core` grep over `crates/` (9 hits, 4 files) — an import-only grep MISSES
  the inline-qualified `tool_spec` call (caught in review).
- **Task 1 (verify published API matches)** must check ALL of the above,
  including `types::*`, `SourceInput`, the `Palace` trait, `ops::list_stashes`,
  and `tool_spec::build_tool_spec`.
- **The one-line swap (task 2):** only ONE line changes — root `Cargo.toml`
  line 46, in `[workspace.dependencies]`:

  ```
  # FROM:
  scrt-core = { git = "https://github.com/JadeZaher/scrt-cli.git", package = "scrt-core", rev = "b22139d" }
  # TO:
  scrt-core = "0.1"
  ```

  Member crates already use `scrt-core.workspace = true`, so **no other file
  changes**. Do this only **after** the published crate exists and the above
  surface is confirmed present. If any symbol drifted, fix the call site
  **explicitly** (called out), never silently.