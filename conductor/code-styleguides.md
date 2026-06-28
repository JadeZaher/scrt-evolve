---
type: Code Styleguide
title: scrt-evolve — Code Styleguides
description: Enforceable code style rules for the scrt-evolve codebase.
timestamp: 2026-06-28T00:00:00Z
---

# scrt-evolve — Code Styleguides

Enforceable rules for this codebase. Sections marked **[MECH]** are mechanizable
(clippy/test/CI); **[REVIEW]** are checked in code review and referenced by track
Acceptance criteria. Rules exist to make the framework's **durable, resumable,
massively-parallel (MPP) execution** correct — not for taste alone.

> Cross-refs: the DAG substrate (track 16), two-phase builder (track 18), the
> transactional wrapper (track 15), and the eval harness (track 10). Durability
> rules below are the contract those tracks must satisfy; their Acceptance
> criteria cite this file.

## 1. Rust style (match the existing crate)

- **[MECH]** `cargo clippy` clean (warnings denied in CI); `cargo fmt` enforced.
- **[REVIEW]** Errors: `thiserror` for library error *types*, `anyhow` only at
  driver/CLI boundaries — mirror `config.rs::ConfigError`. Never `unwrap()` /
  `expect()` / `panic!` on a path reachable from SDK input; return a typed error.
- **[REVIEW]** Config: every new block is an `Option<…>` field on its host struct
  (`GenerateConfig` / `TrainConfig` / `EvolveConfig`) with `#[serde(default)]` +
  per-field default fns — additions stay non-breaking (see `config.rs`). Secrets
  are env-var NAMES, never inline (the `looks_like_inline_secret` rule).
- **[REVIEW]** Public SDK surface is the product (DESIGN.md:187). Anything the CLI
  does is a public SDK item; the binary holds **no orchestration of its own**.
- **[MECH]** ML/Python stay behind features: a default `cargo build` + `cargo
  test` compiles with NO candle and NO Python. Feature-gated code (`train`,
  `pyo3`, `larql`) must not leak into the default build.
- **[REVIEW]** Heavy-ML real-model path is Python (transformers), driven via subprocess over the dataset.jsonl contract (track 19). Candle code stays a fixture behind `--features train`. The default Rust build remains ML-free AND Python-free — the Python path is invoked as an external subprocess, never linked into the binary.
- **[REVIEW]** Doc-comment every public item; module headers explain the *why*
  (match the existing `//!` headers).

## 2. Durable execution (DAG + steps + MPP) — the load-bearing rules

These make a run **resumable, parallel-safe, and reproducible**. A step or node
that violates one is a defect, not a style nit.

### 2.1 Idempotency & purity
- **[REVIEW]** A step's `resolve_args` (arg-generation phase, track 18) is
  **pure-ish**: it reads inputs + config + retrieval, and MUST NOT mutate weights,
  the registry, or any durable artifact. All effects live in `execute`.
- **[REVIEW]** `execute` is **idempotent under its cache key**: running it twice
  with the same (kind, cfg, upstream input hashes) yields the same artifact. No
  reliance on wall-clock, PID, ordering, or `Math.random`-style entropy.
- **[MECH]** **No ambient state.** A node may depend ONLY on declared input ports
  + its cfg — never on globals, env (beyond declared secret names), or files not
  named as inputs. (Enforced by the port-typed `NodeCtx`; reviewed for escapes.)

### 2.2 Determinism
- **[REVIEW]** Every stochastic step takes an explicit **seed** from config/ctx;
  no implicit RNG. Same seed + same inputs → same output (the basis of the
  overfit/smoke tests across tracks).
- **[REVIEW]** Parallel execution MUST NOT change results: independent nodes may
  run concurrently (track 16), so a node's output cannot depend on which sibling
  ran first. Merges/reductions are order-independent or explicitly sorted.

### 2.3 Crash-safety & atomic artifacts
- **[REVIEW]** Artifacts are written **atomically**: write to a temp path in the
  same dir, then rename. A crash mid-write never leaves a half-written
  `dataset.jsonl` / `adapter.safetensors` / `dag.json` / checkpoint.
- **[REVIEW]** Artifacts are **content-addressed** by their cache key (track 16);
  a partial run resumes by reusing completed-node artifacts and recomputing only
  stale descendants. Re-running a finished DAG is a no-op (all cache hits).
- **[REVIEW]** **Weight-mutating `execute` runs ONLY inside the track-15
  transaction** (checkpoint → run → eval → keep|rollback). There is no code path
  that mutates base weights or prunes without a restorable checkpoint.

### 2.4 Provenance & quarantine
- **[REVIEW]** Generated rows stamp the existing `GenExample.gen` provenance
  (`regen:swap<N>`, `refine:*`, `expert:<path_id>`) so a bad artifact is traceable
  and quarantinable (track 15). Dropping provenance is a defect.
- **[REVIEW]** Every durable mutation appends to its log (`evolution-log.jsonl`,
  `arch-log.jsonl`) with {what, verdict, cause} — the audit trail is not optional.

### 2.5 Serializability (so the graph persists)
- **[MECH]** A `Step::Args` type is `Serialize + Deserialize` (track 18) — the
  gen→exec boundary is data-crossable (the sandbox seam). A built pipeline MUST
  lower to a `dag.json`-serializable DAG; persisted graphs use **named step
  kinds** (no closure bodies on disk).
- **[REVIEW]** Bounded everything: self-loops (regen swaps, meta-search, prune
  rounds) carry an explicit budget/stop-condition. No unbounded `while`.

## 3. Durable mind-palace (scrt) usage — long-horizon resilience

For multi-turn / multi-phase work (the self-evolve + architecture lanes), the
scrt mind-palace is **token-budgeted working memory**, not an archive.

- **[REVIEW]** Stash findings as you go; **recall** (`--mp-get` / `--mp-from`)
  instead of re-searching — recall is ~3× cheaper than recompute.
- **[REVIEW]** **Always TTL + tag** an exploratory stash: `--mp-ttl 4h` +
  `--mp-stash-tag scan` for scratch; `--mp-ttl 24h` + `finding` for findings; no
  TTL only for canonical context. Untagged/un-TTL'd scratch is a defect.
- **[REVIEW]** Capture large external/tool output to
  `.omc/research/<slug>-<date>.md` first, then filter through scrt — never pass a
  raw multi-KB fetch/log straight into reasoning context.
- **[REVIEW]** **Budget: ≤20 active stashes** per palace; prune scratch at
  session close (`--mp-prune-tag scan`). Compose/intersect at synthesis instead
  of re-reading source.
- **[REVIEW]** Link stashes you'll traverse (`--mp-link … depends-on|see-also`);
  an unused edge is noise. The palace should shrink between sessions, not grow.

## 4. Dev ergonomics (how we build, in what order)

The working rhythm. The goal is **focused output cycles** — scaffold thoroughly,
verify minimally and locally during the build, then one integrated correctness
sweep at the end. This keeps signal high and noise (and cold-start cost) low.

- **[REVIEW]** **Scaffold thoroughly before verifying.** Write the full set of
  types/seams/signatures/stubs for a unit of work FIRST (it compiles, it's wired,
  tests may be stubbed). Don't interleave a build-test cycle into every
  half-written function.
- **[REVIEW]** **Minimal verification during the build.** While scaffolding, run
  the *narrowest* check that proves the seam holds — a single `cargo check -p
  <crate>`, or one focused `cargo test <name>` — not the whole suite. One target,
  one question per run.
- **[REVIEW]** **Batch the heavy sweep to the END.** Apply ALL changes for a
  unit/track first, THEN run the full `build → test → lint → typecheck` sweep
  ONCE. Do NOT loop test→fix→test→fix per individual change; a single integrated
  sweep is faster (one cold-start) and surfaces the whole remaining picture at
  once. (Matches the project-wide test-execution policy.)
- **[REVIEW]** **Then a dedicated review + fix cycle.** After the end sweep,
  review the full diff against §1–§3 and fix everything in one focused pass —
  authoring and reviewing are separate passes, not interleaved.
- **[REVIEW]** **Minimal scripts, focused output.** Prefer the smallest command
  that answers the current question; scope to a crate/test/feature
  (`-p`, a test filter, `--features X`) rather than building the world. Avoid
  noisy multi-step shell chains when one targeted command suffices.
- **[REVIEW]** **Exception — harness-touching changes.** A change to the test
  harness itself (fixtures, mocks, CI config) may be run inline once to confirm
  the harness still works, before continuing to scaffold.
- **[MECH]** Each track's final task is the integrated sweep (the existing
  "Final sweep: `cargo build` / `cargo test` / `cargo clippy`" task) — that IS
  the end-of-cycle gate; per-change re-runs are not expected in the plan.

## 5. How these are enforced

- **[MECH]** rules → CI: `cargo fmt --check`, `cargo clippy -D warnings`, the
  default-build no-ML/no-Python check, and the `Args: Serialize` / serializable-
  DAG round-trip tests.
- **[REVIEW]** rules → each track's **Acceptance** criteria reference the relevant
  §2/§3 rule (e.g. "execute is idempotent under cache key", "weight-mutating
  exec via track 15", "rows stamp `gen` provenance"). A track is not signed off
  until its durability rules are demonstrated by a test or explicit evidence.
- **[REVIEW]** §4 governs the *build process* of every track: scaffold →
  minimal local checks → ONE end sweep → review+fix pass.
