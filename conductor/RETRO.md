# scrt-evolve — Per-Lane Retrospective

A decision-focused retro, not a changelog. For each lane: what actually
shipped, how it diverged from `DESIGN.md` intent, the call(s) that mattered,
and one honest lesson. Grounded in the source tree
(`crates/scrt-evolve/src/lib.rs`), `conductor/tracks.md`, and the per-track
`plan.md`/`SIGN-OFF.md` files — not from memory. Roadmap-vs-shipped claims
re-verified against the source tree on 2026-06-28 (see §Standing roadmap).

The single most honest finding up front: **the shipped reality is the
discover→generate→train→eval→regulate→export→branch chain, driven by a Python
ML backend. The grand "self-architecting DAG framework" of the architecture
lane (16–18) never shipped a line of code, and large parts of the self-evolve
lane (11–14, 21–22) are specs without modules.** The project was disciplined
about building the runnable core and the product (the branch factory) and
honest about deferring the speculative top of the design.

---

## Cross-cutting lessons

- **The 2026-06-20 amendment (candle is fixture-only; real ML is Python) was
  the project-defining call.** `DESIGN.md:53-86` retroactively demoted the
  entire candle `train`/`local` path to "overfit-tiny-batch fixture" once
  empirical loading of TinyLlama/Llama proved candle's `model.rs` couldn't
  handle RoPE/GQA/BF16. Every real-model result after that flows through
  `python/scrt_evolve_train` over the `dataset.jsonl` contract. This is the
  difference between a Rust-ML aspiration and a thing that trained a model.
- **Track 10 (eval-harness) as the shared foundation was the audit fix that
  saved the self-evolve lane.** The original specs for 11/12/15 each assumed a
  private evaluator nobody owned (`tracks.md:162`). Collapsing them onto ONE
  `ProbeSet`/`Scorer`/`StepVerdict`/`gate.rs` (`src/eval/`) is why track 15
  (regulate) and track 20 (rounds) could be built at all — they consume a real
  gate instead of reinventing one. Build-the-shared-thing-first paid off.
- **Compose-not-fork (track 29 branch factory).** The product wasn't new ML;
  it was an orchestrator (`src/branch/create.rs`) that wires shipped stages
  (01 discover → 02 generate → 19 train → 10 eval → 15 txn → 27 export) into
  `branch create`, plus a thin net-new manifest/registry/router layer. The
  highest-leverage track shipped *zero* new ML.
- **Fractional/microshard training (track 25) was the VRAM-bounding primitive
  that made everything else real on an 8GB box.** Train one contiguous
  layer-block (or sub-layer module group) at a time, distill at layer
  boundaries. Without it, "evolve a real Granite model locally" was a slide;
  with it, it ran. This is the single most reused enabling decision in the
  bench lane.
- **The BTM/c-BTM topology pivot reframed the product.** The design imagined
  in-model MoLE adapter-experts + router (track 14). Reality settled on
  *standalone* Branch-Train-Merge expert LMs (track 29), per-request routing,
  with the decentralized Merge fabric pushed out of this repo entirely into
  hivemind via a cross-repo contract (`SCRT-EVOLVE-INTEGRATION.md`). Track 14
  itself never shipped; its *patterns* (clustering/registry/router/merge) were
  harvested by 29.

---

## Core lane (00–08) — workspace / config / discover / generate / train

**Shipped.** The full backbone: `config.rs` (`EvolveConfig`, toml load+validate),
`workdir.rs`, `discover.rs` (scrt-core search + palace + simhash dedup/cluster →
`DiscoveredContext`), `generate/{api,local,prompts}.rs` behind a `GenBackend`
trait, `dataset.rs` (the jsonl contract), and the `train/` preset family
(`lora`, `full`, `pretrain`, `contrastive`, `shard` modules all present). Tracks
00–04 + 19 are signed off; the discover→generate→export flow is tested ML-free.

**Diverged.** The DESIGN positioned candle LoRA as "the primary path among
candle presets" (`DESIGN.md:317-330`). Reality: candle is a fixture, the
*primary* path is `--backend transformers` (track 19, `python/scrt_evolve_train`).
Tracks 05/06/07 (contrastive, full+pretrain, shard) have source modules but
they are 1–3-line *shells* — the real-model versions were never built. **All
three are now ARCHIVED** (2026-06-28): they are orthogonal to the LoRA-adapter
product, and `shard`'s VRAM goal is already met by shipped fractional training
(track 25) with decentralization contracted out to hivemind.

**Load-bearing decision(s).** (1) Making `dataset.jsonl` the durable
generate↔train boundary — it's what let the Python backend slot in as a
subprocess without rewriting the Rust pipeline. (2) API-first generation
(track 02 before 03) to sidestep local-model echo-chamber collapse.

**Do differently.** Don't ship five `TrainingPreset` modules when only one
(`lora` via Python) is real. The `full`/`pretrain`/`contrastive`/`shard` modules
were scaffolding that read as "done" in the tree but was "Pending" in sign-off —
that gap is exactly the kind of thing this retro exists to flag, and the
2026-06-28 sweep resolved it by archiving them (the stub headers now say so).

---

## Self-evolve lane (10–15) — eval / regen / constitutional / mask / experts / regulation

**Shipped.** The two *foundational* tracks, fully: track 10 `src/eval/`
(`gate.rs`, `probe.rs`, `score.rs`, `verdict.rs`) and track 15 `src/regulate/`
(`checkpoint.rs`, `txn.rs`, `quarantine.rs`, `log.rs`) — checkpoint → eval →
keep|rollback, catastrophe → quarantine-by-`gen`-provenance → halt, all consuming
track 10's `StepVerdict`. Track 20 (`rounds.rs`, `goals.rs`, `harvest.rs`) sits
on top: eval-gated multi-goal rounds through the track-15 txn. These are the
real, tested heart of "safe unattended evolution."

**Diverged.** Tracks **11 (regen-antagonist), 12 (constitutional dialectic), 13
(attribution mask), 14 (expert-spawn-router) never shipped code** — no
`RegenAntagonist`, `AttributionReport`, `TrainingMask`, or router module exists
(`grep` finds zero references in `src/`). All four sign-offs read "Pending." The
lane's most distinctive design ideas — the Jungian-shadow dialectic, attribution
masking, grow-on-demand MoLE experts, the regen flywheel — are unbuilt. Track 15
itself shipped a *simplified* scope: it snapshots the LoRA adapter dir (base is
never mutated on this path), so the design's "base weights as deltas" and
self-pruning (tasks 8–9) are documented seams, not code.

**Load-bearing decision(s).** (1) The audit that forced 11/12/15 onto track 10's
single evaluator instead of three private ones — without it, none of them were
buildable. (2) Scoping track 15 to adapter-snapshot transactions, which made
homeostasis real *now* by dropping the base-delta machinery nobody needed yet.

**Do differently.** Be honest in the dependency graph that 11–14 were
*aspirational* from the start. They were drawn as first-class tracks with phase
gates ("After 11…", "After 14…") that were never met. The eval+regulate
foundation was the right 20% to build; the other 80% of this lane was
prematurely specified at gate-level detail.

---

## Architecture lane (16–18) — dag-engine / self-distill / sdk-builder

**Shipped.** Nothing. There is no `dag` module, no `Dag`/`DagSpec` type, no
`Builder`, no `arch/library/` (`grep -rilE "struct Dag|trait .*Builder|DagSpec"`
over `crates/` returns empty). All three sign-offs are "Pending."

**Diverged.** This is the largest design-vs-reality gap in the project.
`DESIGN.md:134-167` and `tracks.md:57-86` describe an ambitious endgame: the
linear pipeline re-expressed as a typed, validated, content-addressed DAG (16);
a planner-LLM that designs the DAG from an `EvolveIntent` and distills proven
architectures into a reusable, selection-first library (17); and a
typestate-driven trait builder as THE primary SDK surface (18). The `interview.rs`
and `plan/` modules (planner/critic/signals) are the only fragments that gesture
at this — config *generation* exists in seed form, but the DAG substrate it was
meant to lower to was never built. The "primary SDK interface" in `lib.rs`
remains the original three convenience functions, not the builder.

**Load-bearing decision(s).** Implicitly: *not* building this lane. Given an 8GB
box and a finite budget, the team spent its time on the bench and the branch
factory instead of a self-architecting workflow engine. That was almost
certainly the right call, even though the design front-loaded these tracks with
elaborate rails (typed-DAG-only, transactional, bounded).

**Do differently.** Don't write phase gates and honest-risk paragraphs for a
lane you may never build. The "DAG engine becomes a workflow product" scope-creep
risk (`tracks.md:168`) was self-aware — but the cure was to *not spec it to this
depth*, not to spec the mitigations. This lane is the clearest case of design
running ahead of build.

---

## Bench / training lane (21–27) — taste / meta-objects / QAT / benchmarks / fractional / export

**Shipped.** The *infrastructure* tracks, solidly: track 23 (`python/scrt_evolve_dequant`,
registry-driven GGUF→dequant + QAT/STE), track 25 (fractional/microshard
training — the VRAM-bounding primitive, COMPLETE + verified on real Granite/GPU),
track 27 (`src/export.rs` + `python/scrt_evolve_gguf`, config-driven
merge→f16→Q4_K_M GGUF export), and track 24 (the assembled bench, bring-up
validated end-to-end on cached Granite + the user's Claude Code transcripts + an
LM Studio teacher). The runnable bench lives in `bench/` (RUNBOOK, evolve.toml,
corpus, harvest script).

**Diverged.** Tracks **21 (taste-modules) and 22 (meta-objects) did not ship** —
sign-offs "Pending," and the only `taste`/`constitution` references in `src/` are
config fields and comments, not a taste/constitution-driven generation engine.
Per project memory, constitution + taste are still "the missing generation
drivers" (the `custom_prompt` seam). The design's bet that taste/constitution
would shape generation is unrealized; what actually moved the needle was the
training *objective* (`end_task`), not the generation driver. Also: the lane
quietly pivoted twice mid-flight (whole-model → shard-at-a-time → per-module
microshard) as VRAM reality bit.

**Load-bearing decision(s).** (1) Shard/microshard granularity (track 25) as the
floor that trades time for VRAM — the enabling primitive for the whole 8GB
program. (2) WSL2+CUDA to fix Granite's Mamba backward segfault (per memory),
without which the bench simply didn't run.

**Do differently.** Validate the *generation-quality* hypothesis (taste,
constitution) before specifying tracks for it. The lane learned the hard way
that the real data-sensitivity lever was the objective, not rank/data/taste —
that learning should have come from a cheap experiment, not from two
never-built tracks.

---

## Product / BTM lane (29) — branch factory

**Shipped.** The most complete net-new track after the foundations. `src/branch/`
(`manifest.rs`, `registry.rs`/`router.rs`, `create.rs`) + CLI `branch
{create,list,register,route,serve}`. `branch create` composes the whole shipped
chain scoped to a per-branch corpus, *inside* the track-15 transaction (Accept →
register; Regress → no register; Catastrophe → quarantine + halt). Manifest +
`branches/registry.json` + `BranchRouter` are the cross-repo contract to hivemind.
DONE 2026-06-26, all green (18 SDK + 4 CLI tests, clippy/fmt clean, ML-free build).
Live-validated: a TinyLlama-1.1B→scrt-CLI branch trained on an RTX 4060.

**Diverged.** The design's capability-growth story was in-model MoLE
adapter-experts + per-token router (track 14). The product became *standalone*
BTM expert LMs with per-request routing (sibling to 14, not 14). The
decentralized Merge half was cut from this repo entirely and contracted out to
hivemind — a clean scope-narrowing the original DESIGN didn't anticipate.
"Smaller-by-base in v1" (teacher→smaller-student compression deferred to the
`bench/seam_distill` precursor, de-risk PASSED) is another honest descope.

**Load-bearing decision(s).** (1) Compose shipped stages instead of building new
ML — turned the flagship product into an orchestration + thin-manifest track.
(2) Putting the weight-touching span through the track-15 txn, so a branch that
fails its eval gate is never registered — the safety property is inherited, not
re-implemented.

**Do differently.** The branch factory is what the project *should* have aimed at
sooner. It validates the core thesis (self-generated data finetunes a usefully
specialized model) more directly than the unbuilt self-evolve/architecture
lanes ever would have. Lead with the product; let the speculative lanes be
backlog, not gated tracks.

---

## Ambient hardening lane (31) — production-robustness for the daemon

**Shipped.** The five-question hardening pass surfaced by the first real
living-corpus daemon run (the now-retired `HANDOFF.md`, 2026-06-28). All five
addressed with code + tests, ML-free, the track-15 transaction untouched:
**Q1** judge-model preflight (`generate::api::{list_models, preflight}` over GET
`/v1/models`, wired into `doctor` as a FAIL + `--ambient` as a warn;
`bench/ambient.toml` repointed off the vanished `meta-llama-3-8b-instruct` to
`ibm/granite-4-h-tiny`); **Q5** content-hash **dedup ledger**
(`ingest_ledger.rs` + `queue/ingested.ledger`) so re-mined-but-identical rows
don't retrain, with idle-on-nothing-new (the top correctness risk — stale-data
recycling); **Q2** transient-vs-catastrophe **retries** + supervisor cap +
`watch health` + per-source `gen` stamps; **Q3** wall-clock training budget
(`[daemon].max_minutes_per_hour`); **Q4** probe-correctness **trend**
(`trend.rs`) in `watch status/health/trend`. Live-verified against the running
work dir.

**Diverged.** Nothing of substance — this lane was scoped *from* the live run's
real failures, so design and build matched. The one honest finding is empirical,
not architectural: the Q4 trend showed correctness bouncing 0.4–0.6 with
Δtotal≈0 over ~10 committed checkpoints — the predicted noisy/overfit pattern of
a ~400-row recycled corpus, which is *why* Q5's dedup ledger is the load-bearing
fix.

**Load-bearing decision(s).** Hardening was driven by observed failure modes
(missing judge model, stale-data recycling, subprocess-exit-on-blip), not
imagined ones — the opposite of the architecture lane. The single train-vs-score
error asymmetry is the subtle one: only a *score/eval* `Err` is retried; a
*train* failure is caught by the txn → a (non-fatal) rollback, so only
transactional steps reach the evolution log.

**Do differently.** This is the model the *whole* project should have followed:
spec the hardening track *after* the live run exposed what actually breaks. Q1–Q5
are all real because a real daemon hit them.

---

## Closeout lane (30) — UX review applied

The track-30 DevUX/AIUX critique (`UX-REVIEW.md`) is not just a document — its
top fixes **shipped into the CLI** and that is the closeout's most concrete
output. Verified in `crates/scrt-evolve-cli/src/main.rs`: the **candle-fixture
warning** ("the candle backend is a mechanical FIXTURE — it cannot load real
[checkpoints]") now fires on every `cmd_train` run (D1/A3); a **`doctor`
preflight** subcommand (`cmd_doctor`) validates config/model/python/llama.cpp
before a long run (D3); a **process-global `--json`** flag (`emit_json`) emits
machine-readable summary lines across the artifact-producing commands, with
`is_fixture`/resolved-value echo for agents (A1/A2/A3). The remaining UX-REVIEW
findings (overridable-flag `Option`s, `dataset-reference`, the SKILL "Driving the
CLI" section) are tracked as polish, not blockers.

---

## Standing roadmap (named, not built) — re-verified 2026-06-28

Audited against the source tree (not sign-off prose). The roadmap tracks are
**genuinely pending, not merely missing status updates**:

- **05/06/07** (contrastive / full+pretrain / shard): the preset files exist but
  are **1–3-line doc-comment stubs** (`contrastive.rs` 3 lines, `full.rs` 1,
  `pretrain.rs` 2, `shard.rs` 3) — shells, not implementations.
- **11/12/13/14** (regen-antagonist / constitutional / attribution-mask /
  expert-router): **zero modules**. The only `regen` hits in `src/` are
  doc-comments and the `gen=regen` provenance enum value — no `RegenAntagonist`,
  `AttributionReport`, `TrainingMask`, or router type exists. **→ ARCHIVED.**
- **16/17/18** (dag-engine / arch-self-distill / sdk-builder): **zero modules**
  (`grep` for `struct Dag`/`DagSpec`/`trait …Builder` is empty). **→ ARCHIVED.**
- **05/06/07** (non-LoRA / distributed presets — stubs only), **22**
  (meta-objects — config field only): unbuilt **and superseded**. **→ ARCHIVED.**
- **08** (extract/publish — still on the git dep `scrt-core = { git = …, rev =
  b22139d }`) and **09** (modalities — no `SkillIngestion`/`ReasoningEdit`):
  unbuilt but **live roadmap** — 09 is the one dynamic-pipeline improvement worth
  keeping; 08 is a release swap blocked on scrt-core hitting crates.io.

**Archived 2026-06-28 (`tracks/_archived/`).** Three groups left the active
spine: (a) the speculative **lexame** vision — in-model self-evolve lane 11–14 +
self-architecting DAG/SDK lane 16–18, which the standalone-BTM-branch-factory
product does not need; (b) **non-LoRA / distributed presets** 05/06/07 —
orthogonal to the LoRA-adapter-over-immutable-base product (07 also superseded by
fractional training, track 25, + the out-of-repo hivemind merge); (c) **22
meta-objects**, superseded by the shipped `compose_steering()` constitution/taste
seam (21). The one piece of those lanes the product genuinely wanted —
constitution + taste as prompt-constants steering generation/judging — already
ships as `[evolve].constitution`/`taste` (+ per-`[[goals]]` overrides) composed
via `EvolveConfig::compose_steering()` into the `custom_prompt` seam, so
archiving all of this costs the product nothing. Dir numbers are preserved (the
gaps in the live spine point to `_archived/`). See `tracks/_archived/README.md`.

**What actually remains.** After the sweep, the product is shipped and meets its
goal. The only genuine completion item is the **live ambient GPU run (track
26)** — a *validation* gap (the daemon machinery is shipped + tested ML-free), not
code. Beyond that: **08** (external-blocked release) and **09** (optional new
modalities). The posture is harden, not expand — the candle `train` feature is
now vestigial (fixture-only, LoRA-only).
