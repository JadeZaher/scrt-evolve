---
type: Implementation Plan
title: SDK Builder Interface
description: Implementation plan for the SDK Builder Interface track.
tags: [track-18, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# SDK Builder Interface — Trait-Powered, Two-Phase, Sandboxable — Plan

## Tasks

1. [ ] The `Step` trait: assoc `Args`/`Out` (with `Args: Serialize +
   Deserialize`), `kind()`, `resolve_args(ctx) -> Args`, `execute(args) -> Out`.
   -- evidence: trait + two-phase fixture step test.
2. [ ] Closure-as-step wrapper: register a closure under a stable named `kind`
   so it lowers to a serializable node (closure body NOT persisted). -- evidence:
   closure-step lowers + dag.json round-trips by kind.
3. [ ] Capability traits: `CoreEvolve`, `SelfEvolve`, `Distill`, + tooling traits
   (`Peft`/`Trl`/`Gemma`/…) that expose distinct step sets / tags / formats. The
   builder is generic over its capability set (typestate). -- evidence:
   builder-exposes-only-trait-steps test.
4. [ ] Typestate enforcement: calling an out-of-capability step/format is a
   COMPILE error. -- evidence: trybuild/compile-fail (or documented equivalent).
5. [ ] `build()` lowers the builder to a typed track-16 `Dag` + runs
   `Dag::validate()` before execution. -- evidence: build-lowers-and-validates +
   reject-invalid test.
6. [ ] `execute(&ctx)` runs the two-phase nodes via the track-16 scheduler; both
   phases content-addressed/cached independently. -- evidence: exec-runs +
   resolve_args-cached-independently test.
7. [ ] Track-15 wrapping: weight-touching `execute` goes through the transaction;
   `resolve_args` does not. -- evidence: exec-wrapped + gen-not-wrapped test.
8. [ ] Sandbox seam: enforce `Args: Serialize` across gen→exec so the boundary is
   data-crossable; document the future process/OS-isolation hook (NOT built).
   -- evidence: Args-serializes test + seam doc.
9. [ ] Re-expose existing steps (01/02/04/10–17 node impls) as capability-trait
   `Step`s wrapping their node logic. -- evidence: CoreEvolve canonical pipeline
   matches today's output.
10. [ ] CLI shim: construct a builder with capability traits from argv →
    `build()?.execute()`; no logic in the binary. -- evidence: CLI parity test.
11. [ ] Final sweep: `cargo build`, `cargo test`, `cargo clippy`. -- evidence: green.

## Sign-off
Pending.
