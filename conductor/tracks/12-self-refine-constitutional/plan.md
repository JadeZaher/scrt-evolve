# Self-Refinement (Constitutional, Dialectic) — Plan

## Tasks

1. [ ] `constitution.toml` schema + loader: base (non-negotiable) + overlay
   (user/mined, with provenance+confidence) tiers. -- evidence: load+default test.
2. [ ] Tiering invariant: overlay cannot override base; contradictions resolve
   to base and log. -- evidence: conflict-fixture test (resolves to base).
3. [ ] `[generate.refine]` config: `enabled`, `constitution`, `emit`,
   `max_revisions`, `judge`, `mine_overlay`. ML-free round-trip. -- evidence:
   config test.
4. [ ] Two new `GenExample` variants — `refined` and `preference` — additive to
   the `kind`-tagged enum; JSONL round-trip; existing rows still deserialize.
   -- evidence: schema round-trip + back-compat test.
5. [ ] `generate/refine.rs` dialectic orchestration: thesis → metacognition →
   antithesis(shadow) → synthesis, each a recorded turn citing principles, over
   a pluggable/mockable backend. -- evidence: four-stage emit test (no live model).
6. [ ] Overlay mining (optional, `mine_overlay`): induce user-preference
   principles from corpus/palace; tag provenance+confidence; subordinate to base.
   -- evidence: mined-overlay test (tagged, subordinate).
7. [ ] Executable-gate integration: synthesis for tool_call/cli kinds must pass
   the track-10 executable gate before becoming a `refined` row. -- evidence: bad-synthesis
   rejected test.
8. [ ] Emit `refined` (SFT) + `preference` (DPO) rows per refinement per `emit`.
   -- evidence: both-kinds-emitted test.
9. [ ] DPO `TrainingPreset`: consume `preference` rows; chosen/rejected logprob
   margin loss. Behind `--features train`. -- evidence: margin-increases overfit test.
10. [ ] CLI: `scrt-evolve refine [--config]` and `scrt-evolve train --preset dpo
    [--data]`, standalone. -- evidence: CLI test.
11. [ ] Final sweep: `cargo build` (ML-free), `cargo test`, `cargo test
    --features train`, `cargo clippy`. -- evidence: green.

## Sign-off
Pending.
