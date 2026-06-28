---
type: Track Spec
title: Branch Factory — standalone domain branches (BTM Expert LMs)
description: "First-class branch op: turn a base model + corpus into a standalone domain branch, routed per-request."
tags: [track-29, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Branch Factory — create, serve & route standalone domain branches (BTM Expert LMs) — Specification

## Goal
Add a first-class **branch** operation: turn a (small) base model, optionally with a
**selected domain corpus**, into a standalone, domain-specialized **smaller** model —
a **Branch-Train-Merge Expert LM (ELM)** [Li & Gururangan, arXiv 2208.03306] — that is
independently created, eval-gated, packaged (**GGUF + manifest**), registered, and
locally **served + routed per-request**. The base/source model stays untouched; branches
are strictly additive artifacts.

Two creation modes (the operation the owner specified):
```
base model               → smaller domain expert                      (specialize / carve)
base model + domain data  → smaller domain expert (trained + enhanced)  (selected corpus →
                            discover → teacher QA → distill the smaller domain model)
```
"Smaller" comes from `[branch].base` (a small base, specialized) in v1 — NOT teacher→
smaller-student compression (a later mode; the `bench/seam_distill` de-risk is its precursor).

This is the **Branch + Train (+ local Merge)** half of the product. The **distributed Merge**
(P2P serve + ensemble across peers) is the **hivemind** repo's job, which consumes this
track's branch artifact + registry + the `BranchRouter` seam. The cross-repo contract is
`SCRT-EVOLVE-INTEGRATION.md` (authored 2026-06-25, git-ignored in the hivemind repo).

## Positioning vs track 14 (sibling, different artifact)
Track 14 (expert-spawn-router) is **adapter-experts inside ONE dense model** (≈ MoLE), and
explicitly chose *"NOT carving a standalone sub-model."* Track 29 is the **standalone-branch
(BTM) alternative** track 14 set aside: it produces a **separate small GGUF model per domain**,
routed **per-request** (the c-BTM property that keeps P2P traffic sparse), not per-token.
Why now: Stage-3 research (`.omc/research/hybrid-mamba-moe-synthesis-2026-06-25.md`) showed
learned per-token MoE routing specializes on token-surface not domain (Mixtral 2401.04088)
and upcycled experts collapse (2502.19261) — so "one expert = one node" breaks; standalone
BTM branches + request-level routing is the rescue. **29 reuses 14's clustering / registry /
router / merge PATTERNS** but emits standalone branches rather than in-model adapters.

## Scope
- **`[branch]` config** — a new `Option<BranchConfig>` on `EvolveConfig` (serde-default →
  non-breaking, exactly like `[export]` / `[runtime]`), its own stage beside
  discover/generate/train/export. Fields: `enabled`, `base` (the small base path/id — the
  "smaller" lever), `name`, `domain`, `corpus` (per-branch corpus dir/selector — overrides
  `[evolve].corpus_dir` for this branch), `objective` (default `end_task`), `max_branches`
  (roster cap + near-dup merge), `[branch.router]` (`kind` = simhash|embedding|tfidf,
  `confidence_floor`, `top_k`), `[branch.ensemble]` (`single_best` | `average_topk`),
  `[branch.serve]` (reuse `[runtime]`: port / n_gpu_layers).
- **Branch-create pipeline** — `scrt-evolve branch create --name <n> --base <m> [--corpus
  <dir>] [--domain <d>]`: scope a per-branch `EvolveConfig` (override base + corpus) and
  **compose the SHIPPED stages** — `discover::run` (01) → `generate::run` teacher QA (02) →
  track-19 transformers train (`objective=end_task`, optional fractional/QAT) → track-10
  eval gate → track-27 GGUF export → write **manifest.json** + register in
  **`branches/registry.json`**. The weight-touching span runs **inside the track-15
  transaction** (checkpoint → eval → keep | rollback; catastrophe → quarantine by
  `gen=branch:<name>` + halt). **No new ML** — composition + the new packaging/registry/
  router/serve layer only.
- **Branch artifact** — `{ <name>.gguf (Q4_K_M, llama.cpp-servable — handles the hybrid
  Mamba SSM state the HF forward OOMs on), manifest.json }`. Manifest schema (the hivemind
  contract): `{ name, base_model, domain, corpus_descriptor, router_signature, eval_report,
  lineage(parent), version, gguf_sha, created }`. Written **atomically + content-addressed**
  (§2.3).
- **Branch registry** — `branches/registry.json` `{ schema_version, branches:[manifest…] }`.
  Round-trips; atomic writes; the durable fleet record and the file hivemind reads to
  discover branches + their `router_signature`s.
- **`router_signature` computation** — at create time, derive the branch's domain descriptor
  from its corpus/dataset (reuse scrt-core simhash/clustering from track 01 for the ML-free
  path; an embedding descriptor optional behind a feature — `all-MiniLM-L6-v2` / `bge-small`
  are already in the HF cache). Stored in the manifest.
- **`BranchRouter` (the one net-new runtime concept)** — trait
  `resolve(&self, req: &str) -> Vec<(BranchRef, f32)>`. v1 impl = **`LocalBranchRouter`**:
  descriptor-similarity of the request against each branch's `router_signature` → top-k
  **local** branches, with a `confidence_floor` (low confidence → empty → base-only). The
  trait is the **shared seam**: hivemind implements `RemoteBranchRouter` returning
  `(peer, branch)` (documented extension point; NOT built here). MUST degrade safely:
  no branches / `router=off` → base-only, byte-identical (§2.1).
- **Serve + route CLI** (subcommand group, mirroring `Probe`/`Checkpoints`/`Quarantine`):
  - `scrt-evolve branch create […]` — the factory.
  - `scrt-evolve branch list` — read the registry.
  - `scrt-evolve branch route "<query>"` — resolve only: print the chosen branch(es) + scores,
    no serve.
  - `scrt-evolve branch serve <name>` — serve ONE named branch via the existing runtime
    (reuse `RunModel` / `[runtime]` on the branch GGUF).
  - `scrt-evolve serve --branches` — route the request → serve the resolved branch; `ensemble
    = average_topk` blends top-k branch outputs (the BTM inference **Merge**), `single_best`
    (default) serves top-1. v1 may be one-shot (`--prompt`) like `run-model`; a persistent
    server is a later extension.
- **Provenance + safety** — branch-training rows stamp `GenExample.gen = branch:<name>` so
  track-15 quarantine can isolate a bad branch's data (§2.4). `max_branches` cap + near-
  duplicate merge (two near-identical domains MUST merge, not spawn twins — reuse track 14's
  merge/eviction). Empty registry / `router=off` → base behavior (§2.1).
- **ML-free build green** — config, manifest, registry, router scaffold, CLI compile with no
  candle/torch; the create pipeline's ML (train/eval/export) is behind `--features` and runs
  via **PyO3→transformers** (lane directive). The `api` + native-Rust paths (router, registry,
  manifest, merge bookkeeping) need no Python.

## Constraints
- **Base/source stays untouched + standalone.** Branches are additive; with no branches /
  `router=off`, behavior is identical to today's single-model path. Safety floor (§2.1).
- **Compose, don't re-implement.** `branch create` ORCHESTRATES shipped stages (01/02/19/10/27)
  scoped to a per-branch corpus; the only net-new is the manifest/registry/router/serve layer
  + the `[branch]` config. **No new ML preset, no new training math.** If you start re-writing
  a stage, stop and scope the existing one.
- **Weight-touching goes through track 15.** `branch create` trains → it MUST run inside the
  transactional wrapper (15 owns keep/rollback/quarantine). A branch that fails its eval gate
  is rolled back + NOT registered. This track does not re-implement transactions.
- **Per-request routing, not per-token.** The router resolves whole requests to whole branches
  (the c-BTM property). Distinct from track 14's per-token adapter routing.
- **`BranchRouter` is the shared seam with hivemind.** Local resolver here; the remote resolver
  (P2P) is hivemind's. The manifest + registry + trait are the contract — schema changes are
  coordinated via `SCRT-EVOLVE-INTEGRATION.md`. Don't fork the routing model.
- **Bounded fleet.** `max_branches` + near-duplicate merge (no twins, asserted). Reuse track 14.
- **Smaller-by-base, not compression (v1).** "Smaller" comes from `[branch].base`; teacher→
  smaller-student distillation is a later mode (precursor: `bench/seam_distill`), noted not built.

## Acceptance
- `branch create --name <n> --base <fixture> --corpus <fixture>` produces a GGUF + manifest +
  registry entry on a fixture; manifest + registry round-trip (schema asserted).
- The create pipeline runs discover→generate→train→eval→export scoped to the branch corpus
  (composition asserted); the branch's dataset rows stamp `gen=branch:<n>`.
- A branch that FAILS the eval gate is rolled back via track 15 and is NOT registered
  (transaction asserted); a forced catastrophe quarantines by `gen=branch:<n>` + halts.
- `router_signature` is computed from the branch corpus and stored in the manifest.
- `BranchRouter`: a query matching a branch's signature resolves to it; a low-confidence query
  resolves to base-only (no branch) — asserted both ways.
- `branch route "<q>"` prints the resolved branch(es) + scores with no serving; `branch list`
  lists the registry; `branch serve <name>` serves the named branch via the runtime.
- `serve --branches` routes a request and serves the resolved branch (one-shot v1, mock runtime
  in test); `ensemble=average_topk` blends top-k outputs; `single_best` (default) serves top-1.
- Empty registry / `router=off` → generation identical to base (back-compat asserted).
- `max_branches` cap + near-duplicate merge: two near-identical domains merge into one branch,
  not two (asserted).
- ML-free build green; `--features train` (create pipeline) build green.
- The serialized manifest/registry schema **matches `SCRT-EVOLVE-INTEGRATION.md`** (the hivemind
  contract) — asserted by a schema test; the brief is updated if the schema changes.
- **Styleguide gates** (code-styleguides.md): branch rows stamp `gen=branch:<n>` (§2.4);
  manifest + registry written atomically + content-addressed (§2.3); empty registry /
  `router=off` byte-identical to base (§2.1); fleet bounded by `max_branches` (§2.5). Built per §4.

## Dependencies
- **01** (discover + simhash/clustering → passages + `router_signature`s), **02** (teacher QA —
  the `[generate]` ApiEndpoint), **19** (Python transformers train/infer — the real-model
  specialize path; `objective=end_task`), **10** (eval gate — admits/rejects a branch), **15**
  (transactional create: checkpoint→eval→keep|rollback + quarantine), **27** (config-driven GGUF
  export — the branch artifact), **runtime** (`[runtime]` / `RunModel` — serve a branch).
- Reuses **14**'s clustering + registry + router + merge/eviction PATTERNS (sibling: 29 =
  standalone BTM branches, 14 = MoLE adapter-experts). `BranchRouter` is the net-new runtime
  (local resolver; remote = hivemind).
- Optional later: teacher→smaller-student compression mode (precursor: `bench/seam_distill`).

## Cogency-audit notes (apply throughout)
- **Config host.** `[branch]` is a new `Option<BranchConfig>` on `EvolveConfig` (serde-default →
  non-breaking), its own stage. NOT an unparsed floating section.
- **Compose-not-fork.** `branch create` calls the shipped `discover::run` / `generate::run` /
  track-19 train / track-10 eval / track-27 export with a per-branch-scoped config; it adds NO
  new ML. Re-implementing a stage is a defect.
- **Transaction owner.** Branch creation is weight-touching → it runs THROUGH track 15's wrapper
  (15 owns keep/rollback/quarantine). This track does not re-implement transactions.
- **Router seam single-owner.** The `BranchRouter` trait + the LOCAL resolver live here; the
  REMOTE resolver is hivemind's (documented extension point). The manifest + registry schema are
  the cross-repo contract — changes coordinated via `SCRT-EVOLVE-INTEGRATION.md`.
- **Provenance.** Branch-spawned training rows stamp `GenExample.gen = branch:<name>` so track-15
  quarantine can isolate a bad branch's data.
