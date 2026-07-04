---
type: spec
track: 40
title: Delegation contract (evolve ⇄ lexame) — capability card + daisy-chain
status: planned
created: 2026-07-03
updated: 2026-07-03
depends_on: [29, 37]
---

# Track 40 — Delegation contract — Spec

## Goal

Define and version the **cross-repo delegation contract** between scrt-evolve
and laxame-hivemind: the **capability card** (the inbound routing surface —
what a model is measurably good at) and the **daisy-chain** semantics that
let a request hop peer→peer→peer across the inference network. Evolve builds
the artifacts and guarantees their meaning; lexame builds the network that
consumes them.

The contract document is canonical in BOTH repos (sibling of
`SCRT-EVOLVE-INTEGRATION.md`, the existing manifest/registry contract):

- `scrt-evolve/.omc/DELEGATION-CONTRACT.md`
- `laxame-hivemind/.omc/DELEGATION-CONTRACT.md`

Schema changes bump `card_version`/`contract_version` and must land in both
copies in the same change set.

## The capability card (inbound — "route INTO me")

An abstract, *measured* representation of what a model is optimized and
capable of reasoning on — a model card another model can route into. Extends
the manifest's marked-expertise lineage — track 29 `router_signature`
(lexical) → track 37 `eval_report`/`tier` (judged) → this track's
**`competence_profile`** (measured): per-domain confidence, each entry backed
by a probe-set hash + judged score from the track-10/32 harness.
**Non-gameable-by-construction**: every claim derives from a judged probe
run, never self-report. Emitted as a manifest extension + standalone
`card.json`.

An **outbound scalar** (a sequence-level calibrated confidence a peer emits
when it wants help — the delegation trigger) is contract-reserved but
**deferred**: its semantics are pinned in the contract (§3) so the schema
doesn't churn later, but no implementation ships in this track. Until one is
de-risked, delegation triggering is consumer-side policy over card + request.

## Ownership split (the feature/policy boundary, contract-ized)

| Side | Owns |
| :-- | :-- |
| **evolve** | Card schema + emission (`competence_profile` on the manifest; `evolve branch card` to (re)generate from probe runs); the measurement itself (judged probe runs through the track-10/32 harness); shard packaging; conformance fixtures. |
| **lexame** | Transport + discovery; **hop budget / TTL + loop prevention** (visited-set); per-hop **provenance** records; **trust weighting** (the open hard problem); **end-to-end confidence composition** — confidence MULTIPLIES down a chain (three 0.9 hops ≠ 0.9; forwarding a raw upstream score is a contract violation); delegation *policy* against the live network. |

Evolve never trains the delegation policy (no network to supervise against);
lexame never re-derives card claims (no probe harness).

## Deliverables (evolve side)

- **D0** — `DELEGATION-CONTRACT.md` v0.1 drafted + mirrored to both repos
  *(done with this spec — the draft IS the design artifact)*.
- **D1** — `competence_profile` manifest extension + `card.json` emission:
  per-domain scores aggregated over judged probe runs (track-32 judge),
  probe-set hashes recorded. Additive field, byte-identical manifests when
  absent (the track-37 Phase 0 discipline).
- **D2** — `evolve branch card` — (re)generate a branch's **measured
  competence data** (JSON) from a probe run, for lexame's card assembly;
  refuses if probes are stale vs. the adapter's `last_good`.
- **D3** — ~~delegation-shard packaging~~ **MOVED TO LEXAME (2026-07-03):
  the card/shard is a lexame-side artifact** — assembled, published, and
  versioned as a network object in laxame-hivemind. Evolve's deliverable
  ends at the measured data (D1/D2) + exported probe packs (D5) it is
  assembled from.
- **D4** — contract conformance fixtures: a lexame-side consumer can validate
  a shard/card against the schema without evolve installed (JSON schema +
  golden files).
- **D5** — **probe-pack export**: export `{probe set, judging rubric,
  probe_set_hash}` so any third party can re-run the evidence behind a claim
  (publication/distribution of the pack is lexame-side, like the card). This is what makes routing *empirically
  verifiable* (contract §4.5): a routing peer challenge-verifies a card by
  running sampled probes over the same channel real work arrives on —
  behavior under challenge is the unfakeable "piece of the model". Challenge
  scheduling/tolerance/trust-ledger are lexame-side; evolve's job is that
  every card claim is re-runnable.

## Depends / gates

- 29 (manifest/registry — the surface being extended), 37 (eval_report/tier —
  the judged layer beneath the measured one).
- Lexame-side daisy-chain runtime is tracked in laxame-hivemind, not here;
  this track's acceptance is contract + artifacts, not network behavior.

## Risks

- **Measurement drift** — a card claiming yesterday's competence after
  today's keep-commit. Mitigation: cards stamp the checkpoint ordinal +
  `last_good` id they were measured against; D2 refuses stale probe runs.
- **Schema churn across repos** — mitigated by same-change-set mirroring +
  versioned schema + D4 fixtures.
- **Over-claiming domains** — profile domains come from the probe-set
  taxonomy (track 10/20 goals), not free text; no probes, no claim.

## Acceptance

- Contract doc v0.1 present + identical in both repos' `.omc/`.
- A branch manifest gains `competence_profile` (additive; absent = byte-
  identical); `evolve branch card` emits `card.json` from a real probe run.
- D4 fixtures validate a golden card + shard layout; a mutated card fails.
- ML-free build green; card emission needs only the existing probe/judge
  harness.
