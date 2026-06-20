# scrt-evolve — Tracks (the spine)

This is the comprehensive spine of `/new-track`s implementing
[DESIGN.md](../DESIGN.md). Tracks map 1:1 to the DESIGN.md §Build order phases
and are **review-gated**: each must compile + pass its own tests before the
next begins. Ordering is a near-strict chain (a few parallel opportunities
noted below).

## Dependency graph

```
00 ─► 01 ─► 02 ─► 03                 (discover → API gen → local gen)
        │     │
        │     ├────► 04 ─► 06        (LoRA → full+pretrain)
        │     │        │
        │     │        └────► 07     (shard — reuses hivemind topology)
        │     ├────► 05              (contrastive — port of in-tree seam; parallel to 04)
        │     └────► 09              (new modalities — skill ingestion + reasoning edit)
        └────────────────────► 08    (extract/publish — last)

Self-evolve lane (the product goal — locally tune ONE model that evolves with a
user's goals across ALL their projects, MERGED into one model, on-demand):

  10  eval-harness  ── FOUNDATION: ProbeSet + Scorer + StepVerdict + the shared
      │                executable gate. Built FIRST; 11/12/15 consume it (the
      │                audit found they each assumed an evaluator nobody built).
      ▼
  04 ─► 13 (mask: which params to update → faster training, all paths) ─┐
  04 ─► 11 (regen self-distill + depth-cheapen "topology shift") ───────┤
  10 ─► 12 (constitutional dialectic self-refine — the outcome signal) ─┤
  13 ─► 14 (expert spawn + router — grow capability on demand) ◄────────┘
  {11,12,13,14} ─► 15 (self-regulation): HOMEOSTASIS capstone — wraps EVERY step
            in checkpoint → evaluate(via 10) → keep|rollback; self-prunes (cold
            experts always; gated base pruning transactionally); catastrophe →
            auto-rollback + quarantine(by gen-provenance) + halt. Base case of
            the recursion (grow → evaluate → revert).

  13 ─► 14 : ONE attribution pass, two consumers — 13 emits a training MASK
            (which params to update) AND a reusable `AttributionReport`; 14
            consumes that report as an expert BLUEPRINT (which layers a new
            adapter-expert targets). Base model stays dense; experts additive.

DIRECTIVE — heavy ML via PyO3→transformers: the candle-thin workflows in this
lane (masked LoRA, DPO/preference, early-exit depth-cheapen, base pruning,
perplexity/exit-depth scoring) are implemented by driving HF
`transformers`/`peft`/`trl`/`torch` through `bridge.rs` (`--features pyo3`), NOT
hand-built in candle. peft gives target-module freezing + multi-adapter compose
natively; trl gives DPO; torch gives state-dict diffing for checkpoints/pruning.
Candle remains an OPTIONAL later path. The `api` paths + native Rust (router,
registry, txn, clustering) work with no Python at all.

A future local daemon (DESIGN.md north-star, NOT a build track) triggers
discover→refine→mask→train→swap (+ detect→plan→spawn-expert) on a cadence;
tracks 10–15 are built non-interactive + resumable, and a daemon may ONLY
auto-evolve THROUGH track 15's transactional wrapper.

Architecture lane (turns the project into a generic, self-architecting
training/model-building framework):
  16  dag-engine : EVERY step above becomes a typed Node; a run is a validated
      │            (acyclic, typed-ports) `Dag` serialized to dag.json. The
      │            current linear run() becomes ONE generated canonical DAG.
      │            Generalizes the in-tree `plan/` pattern (planner emits gen
      │            specs) up a level: the whole pipeline is planned DATA.
      ▼
  17  architecture-self-distill :
        Part 1 (factory): QA/interview → intent.json → planner LLM → a validated
          DagSpec + reproducible evolve.toml → run. The pipeline is DESIGNED from
          intent, not hand-authored.
  18  sdk-builder-interface : THE primary SDK surface — a trait-powered builder
      designed-then-executed. Capability TRAITS select which step sets / tag types
      / formats / training tooling the builder exposes (CoreEvolve, SelfEvolve,
      Distill, Peft/Trl/Gemma…); steps are two-phase callbacks (resolve_args →
      execute) with the phase boundary as the sandbox seam (OS isolation later).
      Lowers to the track-16 serializable DAG (persists to dag.json). CLI = thin
      shim that builds with the right traits and .execute()s.

        Part 2 (artifact distillation): ARTIFACT-FIRST, not mutation-first. The
          system GENERATES DagSpec FILES, runs them like any config, and keeps
          winners in a reusable LIBRARY. Selection-first: on a new intent it
          REUSES a proven library artifact when one fits; it only GENERATES on a
          miss (→ "generate those artifacts and use them instead of self-
          generating"). It never mutates a live graph and never synthesizes MODEL
          architecture — only wiring/cfg over registered nodes. Weight-touching
          trial runs go THROUGH track 15 (pass → keep + library; regress →
          rollback + discard). Bounded search. The library + lineage ARE the
          distilled architecture knowledge.

PyO3 bridge: introduced in 02 (dataset export to Python), deepened in 04
(training-step seam), load-bearing in 07 (Python sharding stack) and in the
self-evolve lane (10–15: transformers/peft/trl drive the heavy ML workflows).
```

## Tracks

| # | Track | DESIGN phase | What it delivers | Depends on |
| :- | :--- | :--- | :--- | :--- |
| **00** | [repo-skeleton-config](tracks/00-repo-skeleton-config/) | 1 | Workspace, `EvolveConfig` (toml load+validate), work-dir layout, **PyO3 feature stub**. No ML. | — |
| **01** | [discover](tracks/01-discover/) | 2 | `discover.rs` over scrt-core (search + palace + simhash dedup/cluster) → `DiscoveredContext` → `discovered.json`. No ML. | 00 |
| **02** | [generate-api-backend](tracks/02-generate-api-backend/) | 3 | `GenBackend` trait, `ApiEndpoint` impl, prompt templates, `Dataset` JSONL writer/reader, **dataset→Python export over the PyO3 bridge**. End-to-end discover→dataset, no local model. | 01 |
| **03** | [generate-local-candle](tracks/03-generate-local-candle/) | 4 | `LocalCandle` GenBackend (candle inference) behind the same trait + `train` feature. | 02 |
| **04** | [train-lora](tracks/04-train-lora/) | 5 | `TrainingPreset` trait, model loader (`model.rs` per-arch seam, ONE arch first), LoRA injection + training loop → `adapter.safetensors`. **PyO3 training-step seam** so `peft`/`trl` can drive it. | 02 (data), 03 (model loader shared) |
| **19** | [python-train-infer](tracks/19-python-train-infer/) | — (core validation) | Standalone Python `scrt_evolve_train` (transformers LoRA) + `scrt_evolve_infer` (base+adapter A/B), driven from the Rust CLI via subprocess. The **PRIMARY real-model training/inference path** (candle = fixture). dataset.jsonl is the contract. | 02 (dataset), 03/04 (candle fixture it validates) |
| **20** | [learning-by-doing](tracks/20-learning-by-doing/) | — (product capstone) | **Incremental multi-goal LEARNING-BY-DOING evolution**: `[[goals]]` (name/topic/tag) in `evolve.toml`; a paired **`scrt-evolve` SKILL** steers a frontier agent to stash goal-tagged findings as it works → the palace + transcripts become the curriculum. Per-goal eval-gated rounds (discover→generate→train→eval→keep\|rollback via track 15), generation-improves-itself (track 11 flywheel), and a bounded **scheduler** across goals. Orchestration over shipped tracks — no new ML. Makes the DESIGN daemon buildable + safe. | 01 (palace-search), 02, 19 (train), 10 (eval), 15 (txn), 11 (regen) |
| **05** | [train-contrastive](tracks/05-train-contrastive/) | 6 | Port the in-tree InfoNCE embedding-adapter seam → `contrastive` preset (consumes palace structure directly). | 04 (trait), 01 (palace access) |
| **06** | [train-full-pretrain](tracks/06-train-full-pretrain/) | 7 | `full` finetune + `pretrain` (continued causal-LM on raw corpus) presets. | 04 |
| **07** | [train-shard](tracks/07-train-shard/) | 8 | Decentralized `shard` preset (coordinator + worker) reusing the **hivemind tensor wire format + coordinator/worker topology** via the PyO3 bridge. Small trusted cluster only. | 06, 04 |
| **09** | [modalities-skill-reasoning](tracks/09-modalities-skill-reasoning/) | — (new) | New generation modalities: **skill ingestion** (`SkillIngestion` rows — absorb a SKILL.md into callable behavior) + **reasoning-step modification** (`ReasoningEdit` rows — insert/correct/prune/reorder CoT). Flows through the existing planner→generate→dataset→export pipeline. | 02 (gen/dataset) |
| **10** | [eval-harness](tracks/10-eval-harness/) | — (lane foundation) | **Shared** `ProbeSet` + `Scorer` (`ScoreReport`) + `StepVerdict` + the executable `gate.rs`. Scoring backends: `api` (no ML), **`pyo3`→`transformers`** (perplexity/exit-depth), `candle` (optional). Built FIRST so 11/12/15 stop assuming an evaluator nobody built. | 02 (`ApiEndpoint`), 00 (`pyo3`) |
| **11** | [regen-antagonist](tracks/11-regen-antagonist/) | — (new) | `RegenAntagonist` GenBackend (model's own refreshed checkpoint, hot-swapped via `refresh()`) + depth-first **early-exit cheapness** training (the "topology shift", a loss not a graph search) + self-distilled grounding nodes. Consumes track 10's gate + `Scorer`. Depth-cheapen via **PyO3→transformers**. Optional `larql` `TRACE` sidecar for measurement. CLI-first. | 10 (gate+scorer), 04, 03 |
| **12** | [self-refine-constitutional](tracks/12-self-refine-constitutional/) | — (new) | Constitutional **sequential dialectic** (thesis → metacognition → Jungian shadow antithesis → synthesis) vs authored-base + mined-overlay `constitution.toml`. Emits `refined` (SFT) + `preference` (DPO) rows; DPO via **PyO3→`trl`**. `max_revisions`=1 default. No human labeling. The cross-project outcome signal. | 10 (gate+scorer), 11 (thesis), 04, 01 (merged corpus) |
| **13** | [attribution-training-mask](tracks/13-attribution-training-mask/) | — (new) | **Tier-1, all-paths** `TrainingMask` (which layers/modules to update → faster training) + the single reusable `AttributionReport`. Selectors: `full` (default), `grad` (no-LARQL fallback), `attribution` (`larql`), `manual`. Mask honored via **PyO3→`peft`** target-module freezing. NOT distributed sharding (that's 07). `full()` = current behavior. | 04 (composes with 06/11/12/14) |
| **14** | [expert-spawn-router](tracks/14-expert-spawn-router/) | — (new) | **Grow-on-demand adapter-experts**: path-detector clusters recurring paths; **consumes track 13's `AttributionReport`** as an `ExpertBlueprint` (no duplicate attribution pass); each path → a **PyO3→`peft`** LoRA expert + registry; a native-Rust **router** dispatches top-k. Base stays dense; router → no-op when off. ≈MoLE, NOT FFN-MoE/carve. | 13 (attribution), 04 (LoRA), 01 (clustering) |
| **15** | [self-regulation](tracks/15-self-regulation/) | — (capstone) | **Homeostasis**: transactional evolution (checkpoint → evaluate via **track 10** → keep\|rollback); **self-pruning** (auto expert eviction/merge native-Rust + gated base sparsity via **PyO3→torch**); catastrophe → **auto-rollback + quarantine by `gen`-provenance + halt**. Makes the daemon safe. | 10 (eval), 11, 12, 13, 14 |
| **16** | [dag-engine](tracks/16-dag-engine/) | — (arch) | **Typed DAG substrate**: every step becomes a registered `Node` with typed input/output ports; a run is a build-time-validated (acyclic, types match) `Dag` serialized to `dag.json`, executed by a topo scheduler with content-addressed artifact caching. Existing `run()` becomes one canonical generated DAG (wrap, don't rewrite). No ML. | 01, plan/, 02, 10 (+ wraps 11–15 as they land) |
| **18** | [sdk-builder-interface](tracks/18-sdk-builder-interface/) | — (SDK surface) | **THE primary SDK interface**: a trait-powered builder, designed-then-`.execute()`d. **Capability traits** select exposed step sets / tag types / formats / training tooling (`CoreEvolve`/`SelfEvolve`/`Distill`/`Peft`/`Trl`/`Gemma`…); unavailable steps are COMPILE errors (typestate). Steps are **two-phase callbacks** (`resolve_args`→`execute`) — the phase boundary is the **sandbox seam** (`Args: Serialize`; OS isolation later). Lowers to the track-16 serializable DAG (persists). CLI = thin shim. | 16 (lowers to), 15 (wraps exec), re-exposes 01/02/04/10–17 |
| **17** | [architecture-self-distill](tracks/17-architecture-self-distill/) | — (arch capstone) | **QA→planner→DAG factory** (intent.json → validated `DagSpec` + reproducible `evolve.toml`) + **artifact distillation** (ARTIFACT-FIRST: generate `DagSpec` FILES, run them, keep winners in a reusable `arch/library/`; **selection-first** — reuse a proven artifact when one fits, generate only on a miss). Weight-touching trials go THROUGH track 15 (pass→keep+library, regress→rollback+discard). No live mutation, no model-arch synthesis. Rails: typed-DAG-only, transactional, bounded. | 16, 15, 10, interview/plan |
| **08** | [extract-publish](tracks/08-extract-publish/) | 9 | Swap scrt git dep → published crate; retire/re-export in-tree crate; cut first release. | all |

## Phase gates (from DESIGN.md)
- After **00**: compiles, `config` tested, PyO3 stub builds with `--features pyo3`.
- After **01**: discover tested against a fixture palace, no ML.
- After **02**: discover→dataset runs with no local model; dataset readable from Python.
- After **04**: LoRA produces a loadable `adapter.safetensors`; overfit-tiny-batch smoke passes.
- After **07**: shard run produces merged weights across ≥2 local worker processes.
- After **09**: `SkillIngestion` + `ReasoningEdit` rows generate, round-trip, and export (Gemma-native); planner can target both modalities.
- After **10**: `probe build` carves a zero-overlap held-out set; `api`-backend `Scorer` produces correctness + constitution_adherence with no ML deps; `StepVerdict` classifies accept/regress/catastrophic; `--features pyo3` computes perplexity/exit-depth via `transformers`; probe-version mismatch refused.
- After **11**: regen loop runs ≥2 swaps; mean exit depth decreases while held-out correctness (via track 10 `Scorer`) holds; gate-failing antagonist samples never enter the dataset; rows stamp `gen=regen:swap<N>`; `--features larql` builds and is removable.
- After **12**: dialectic emits all four stages; `refined`/`preference` rows round-trip without breaking existing rows; overlay cannot override base constitution; gate-failing synthesis never becomes a `refined` row; `max_revisions` defaults to 1; DPO margin increases on a fixture (PyO3→`trl`).
- After **13**: a `grad`/`manual` mask freezes a measurable param fraction with NO LARQL; masked training touches only in-mask modules (via PyO3→`peft`); `full()` reproduces current behavior; a reusable `AttributionReport` is emitted; `training-mask.json` reports frozen_fraction.
- After **14**: the detector clusters a fixture into ≥2 paths and flags an uncovered one; `experts spawn` trains+registers a `peft` expert from track 13's `AttributionReport` (no second attribution pass); router routes a matching input to its expert and a low-confidence input to base-only; empty registry / `router=off` is byte-identical to base; near-duplicate clusters merge (no twins).
- After **15**: a passing step commits + advances `last_good`; a regressing step rolls back (state restored); a forced catastrophe auto-rolls-back + quarantines the cause (by `gen`-provenance/cluster) + halts, and the next round skips it; gated base pruning shrinks on pass and auto-rolls-back on regress (prune never irreversible); checkpoints store base as deltas; `evolution-log.jsonl` records commit/rollback/quarantine. Eval is via track 10.
- After **16**: the registry holds existing stages; the canonical DAG reproduces current `run()` (back-compat); `Dag::validate()` rejects cyclic/type-mismatched/unfed/bad-cfg graphs; `dag.json` round-trips; the executor caches an unchanged subgraph and recomputes only stale descendants; `[dag]` absent = today's behavior. The `Run` orchestration has MOVED out of the binary into the SDK canonical DAG (CLI = pure shim).
- After **18**: a `Builder::<CoreEvolve>` builds + executes a pipeline matching the canonical DAG; an out-of-capability step is a COMPILE error (typestate); a closure step lowers under a named kind and `dag.json` round-trips; `resolve_args` caches independently of `execute` and `Args: Serialize` (the sandbox seam is data-crossable); a weight-touching `execute` runs through track 15 while `resolve_args` does not; a tooling trait gates the exposed format.
- After **17**: `architect --from intent.json` REUSES a fitting library artifact (no generation) or, on a miss, emits a validated `dag.json` + matching `evolve.toml`; an invalid generated candidate is rejected + re-prompted; a weight-touching trial that improves is kept (weights + artifact admitted to `arch/library/`) and one that regresses is rolled back (weights restored, artifact discarded) via track 15; an eval-only candidate runs without the txn; catastrophe rolls back + halts + quarantines the artifact; bounded search stops at budget/plateau; a saved library artifact round-trips and a later matching intent reuses it (generate-once-then-reuse).
- After **08**: builds against published scrt-core; first tagged release.

## Honest risks carried across tracks (DESIGN.md §Honest risks)

**NOTE (Amendment 2026-06-20):** The candle `train`/`local` backends (tracks 03/04) are confirmed **fixture/mechanical paths only** and cannot load real pretrained models (RoPE/GQA/BF16). The real-model training/inference path is **Python/transformers** (track 19), driven via subprocess over the dataset.jsonl contract — primary, fully validated, and consistent with the lane directive. See DESIGN.md §Amendment 2026-06-20 and track 19 spec for details.

- candle's finetuning ecosystem is thin — per-arch model loaders are hand-built; start with ONE arch (track 04), expand as backlog. **Candle paths are fixture-only; real-model training via track 19 (Python backend).**
- Local-gen quality can collapse (echo chamber) — ship API-first (02 before 03), treat local-gen as lower-trust.
- Shard training is genuinely hard — v1 bar is a small trusted cluster, deliberately last (07).
- The quality premise is unproven — the gated LongMemEval-style measurement is out of scope for these build tracks.
- **Heavy ML via PyO3→transformers, not candle (lane directive).** candle's training ecosystem is thin, so the self-evolve lane's heavy workflows (masked LoRA, DPO, early-exit depth-cheapen, base pruning, perplexity/exit-depth scoring) are driven through HF `transformers`/`peft`/`trl`/`torch` via `bridge.rs` (`--features pyo3`). The `api` paths + native Rust (router, registry, txn, clustering, gate) need no Python; candle is an optional later path. This makes the lane buildable without first maturing candle finetuning.
- **Shared evaluator (audit fix).** Tracks 11/12/15 do NOT each build a probe/scorer — track **10** owns `ProbeSet`/`Scorer`/`ScoreReport`/`StepVerdict`/`gate.rs`; the others consume it. (Earlier specs that assumed a private evaluator were the headline cogency gap.)
- Self-distillation can collapse (track 11) — the executable gate (from 10) + decaying `antagonist_ratio` + teacher anchor are mandatory; training on un-gated self-output is a defect. The "topology shift" (depth-first early-exit) is validated mechanically (exit depth ↓, correctness holds via track 10's `Scorer`), not as a quality claim. LARQL stays an optional, removable sidecar — never a runtime; its reverse-inference/speed premise was evaluated and rejected (see `.omc/research/larql-regen-swap-2026-06-17.md`).
- Constitutional self-critique can rationalize (track 12) — base constitution principles (safety/correctness) are inviolable by code; mined overlay is subordinate; synthesis still passes track 10's executable gate. `max_revisions`=1 default; `refined` trains on synthesis only (not the verbose chain) to protect the move-fast/depth-cheapen goal.
- Attribution-guided masking is a coarse static prior (track 13) — it PROPOSES, gradient/peft training DISPOSES. `full()` (no masking) is the always-valid default; masking is strictly opt-in so bundling it into all paths can't regress existing presets. Track 13 is the SINGLE attribution owner; track 14 consumes its `AttributionReport`. Distinct from distributed `shard` (07).
- Expert sprawl + mis-routing (track 14) — growth bounded by `max_experts` + near-duplicate merge (no twins); routing safety beats coverage (low confidence → base-only). Base model stays dense + standalone; experts + router are strictly additive, so empty registry / `router=off` is byte-identical to base. Adapter-experts (≈MoLE), deliberately NOT FFN-MoE or a carved sub-model.
- Self-corruption / runaway shrinkage (track 15) — every weight-mutating step (train, base-prune) is transactional: snapshot → eval (via track 10) → keep|rollback, so no step is irreversible and base pruning is always revertible. Catastrophe (correctness collapse / safety-violation spike / loss-NaN) auto-rolls-back + quarantines the cause (by `gen`-provenance/cluster) + HALTS; resuming needs explicit re-arm. The base is never auto-pruned outside the eval-gated transaction. Threshold *tuning* is an experiment; the snapshot/eval/revert/quarantine *machinery* is what the track proves.
- "DAG engine becomes a workflow product" scope creep (track 16) — kept tractable by WRAPPING existing stages (no rewrite), a CLOSED artifact-type port enum (not arbitrary types), and additivity (`[dag]` absent = today's behavior). It schedules nodes; it introduces no ML and no new step logic.
- SDK builder over-abstraction (track 18) — the trait-builder is the PRIMARY interface but is constrained by: it must LOWER to the track-16 serializable DAG (no construct that can't); capability = typestate (compile errors, not runtime); two-phase (`resolve_args`/`execute`) is mandatory with `Args: Serialize` so the gen→exec boundary is the sandbox seam (OS isolation is a FUTURE seam, not built now); persisted graphs use named step kinds (no closure bodies on disk). The CLI stays a pure shim that builds-with-traits and `.execute()`s.
- Self-architecting runs away or corrupts (track 17) — kept tractable by being **artifact-first** (it generates `DagSpec` FILES and runs them; it never mutates a live graph and never synthesizes MODEL architecture/node logic — only wiring+cfg over registered nodes) and **selection-first** (reuse a proven library artifact when one fits; generate only on a miss). Three rails: (1) generated artifacts are typed DAGs of REGISTERED nodes that must pass `Dag::validate()` before any run; (2) weight-touching trial runs go THROUGH track 15 (checkpoint→eval→keep+library | rollback+discard; catastrophe halts+quarantines); (3) BUDGET-bounded search, stop-on-plateau. Every artifact is reproducible from its file + re-selectable from `arch/library/`. Synthesizing new node *implementations* is explicitly OUT of scope (a future sandbox-gated concern).
