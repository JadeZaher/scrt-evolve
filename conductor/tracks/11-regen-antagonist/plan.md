# Regen-swap Antagonist & Topology Shift — Plan

## Tasks

1. [ ] `[generate.regen]` config: `enabled`, `swap_every_steps`, `temperature`,
   `top_p`, `antagonist_ratio`, `ratio_decay` (none|linear), `gate`
   (schema|execute), `dropout`, plus a stubbed `sparsity_lambda` (default 0).
   Load + validate + defaults; ML-free build green with the block present.
   -- evidence: config round-trip test.
2. [ ] `generate/regen.rs`: `RegenAntagonist` struct holding
   `Arc<RwLock<LoadedModel>>` + sampling cfg; `refresh()` swaps the weight
   handle with no disk reload. Implements `GenBackend`. ML-free seam compiles;
   candle sampling body behind `--features train`. -- evidence: refresh test
   (post-refresh output reflects a mutated weight).
3. [ ] Divergence knobs: high-T / wide top-p sampling, inference-time activation
   dropout, and **prompt perturbation** (pair two unrelated palace stashes →
   prompt a bridging CLI workflow). -- evidence: perturbation produces distinct
   prompts from the same passage set (seeded).
4. [ ] Executable acceptance gate: `schema` (validate vs `toolspec.rs`) and
   `execute` (parse/dry-run the `scrt …` command, args resolve). Reject path
   discards the sample. -- evidence: gate rejects malformed + accepts valid, both
   modes.
5. [ ] Anti-collapse wiring: only gate-passing antagonist rows enter the dataset;
   teacher (API/frozen) mixed in per `antagonist_ratio` with `ratio_decay`
   applied across swaps. -- evidence: forced gate-failing sample absent from
   output dataset; ratio anneals over N swaps.
6. [ ] Self-distilled targets + grounding nodes: per accepted example record
   full-depth soft label, used FFN features/layers, early-exit attempt; build the
   neighbor-adjacency set from used features + seeding stashes. -- evidence:
   record carries {soft_label, used_layers, neighbor_nodes}.
7. [ ] Depth-first `regen` `TrainingPreset`: self-distilled **early-exit head** +
   two-term loss `CE(full_depth_target) + λ·exit_depth`. Sparsity term present
   but inert (default λ=0, asserted no-op). Reuses track 04 loop. -- evidence:
   early-exit head trains (loss down) on seeded fixture.
8. [ ] Regen loop driver: generate→gate→train→`refresh()` every
   `swap_every_steps`. -- evidence: loop runs ≥2 swaps on a tiny fixture+model.
9. [ ] Topology-shift smoke (the core claim): across swaps, **mean exit depth
   decreases while held-out correctness does not regress** (seeded, deterministic).
   -- evidence: monotone-exit-depth + stable-correctness test.
10. [ ] `regen-metrics.jsonl` emitter: per-swap {swap, correctness,
    mean_exit_depth}. -- evidence: metrics file shape test.
11. [ ] LARQL sidecar behind `--features larql` (isolated, removable): `TRACE …
    FOR <tool>` → add FFN/attn attribution to metrics rows; `ROUTE VERIFY` arg
    guard; `INSERT INTO EDGES` CLI-graph seeding. Degrade gracefully with no
    vindex. -- evidence: `--features larql` builds; no-vindex path runs with
    depth-only metrics.
12. [ ] CLI: `scrt-evolve generate --backend regen` and `scrt-evolve run
    --regen`, standalone on a fixture. -- evidence: CLI test.
13. [ ] Final sweep: `cargo build` (ML-free), `cargo test --features train`,
    `cargo build --features train` (no larql), `cargo build --features
    "train larql"`, `cargo clippy --features train`. -- evidence: green.

## Sign-off
Pending.
