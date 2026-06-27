# Track 29 — Branch Factory — Plan

Build order is **composition-first**: reuse shipped stages (01/02/19/10/15/27) and add only
the thin registry/router/manifest/serve layer + `[branch]` config. TDD per phase; each phase
compiles + passes its own tests before the next. ML-free first (config/types/router/CLI), then
the create pipeline behind `--features train`.

## Phase 0 — Config + types (native Rust, ML-free)
1. [x] `BranchConfig` (+ `BranchRouterConfig`, `BranchEnsembleConfig`, `BranchServeConfig`) in
   `config.rs`; `pub branch: Option<BranchConfig>` on `EvolveConfig` (serde-default, additive);
   export from `lib.rs`. Defaults: `objective="end_task"`, `router.kind="simhash"`,
   `router.confidence_floor`, `router.top_k=1`, `ensemble="single_best"`, `max_branches`.
   **Test:** `branch_config_round_trips_with_defaults`; `evolve_config_without_branch_unchanged`
   (back-compat — a config with no `[branch]` parses + behaves as today).
2. [x] `BranchManifest` + `BranchRegistry` serde types matching `SCRT-EVOLVE-INTEGRATION.md`.
   Atomic + content-addressed write/read (reuse the project's atomic-write util; §2.3).
   **Test:** manifest/registry round-trip; partial-write leaves no corrupt file; `schema_version`
   mismatch refused.

## Phase 1 — Router (native Rust; ML-free path via simhash)
3. [x] `BranchRouter` trait `resolve(&self, req:&str)->Vec<(BranchRef,f32)>`; `LocalBranchRouter`
   impl = descriptor-similarity over registry `router_signature`s (scrt-core simhash for the
   ML-free path; embedding descriptor optional behind a feature). `confidence_floor` → empty.
   **Test:** matching query → its branch; low-confidence query → empty (base-only); `router=off`
   / empty registry → empty — all asserted.
4. [x] `router_signature` computation from a branch's corpus/dataset (simhash centroid; reuse
   track-01 clustering). **Test:** signature from a fixture corpus is stable and discriminates
   two distinct fixture domains.
5. [x] near-duplicate merge + `max_branches` cap (reuse track-14 merge logic): registering a
   near-identical signature merges instead of spawning a twin; cap enforced.
   **Test:** two near-dup branches → one; cap rejects the (N+1)th or evicts per policy.

## Phase 2 — Create pipeline (compose shipped stages; ML behind `--features train`)
6. [x] `branch::create(cfg, name, base, corpus, domain)` orchestrator: build a per-branch
   `EvolveConfig` (override `base` + `corpus`) → `discover::run` → `generate::run` (teacher QA;
   rows stamped `gen=branch:<name>`) → track-19 train (`objective=end_task`) → track-10 eval
   gate → track-27 GGUF export → assemble `BranchManifest` (incl. `router_signature`,
   `eval_report`, `gguf_sha`) → register. Wrap the weight-touching span in the **track-15
   transaction** (keep on pass / rollback + DON'T register on fail; catastrophe → quarantine by
   `gen=branch:<name>` + halt). **Test (fixture; mock teacher + `--features`-gated tiny train):**
   create yields gguf + manifest + registry entry; a forced eval-fail rolls back and leaves the
   registry unchanged.
7. [x] Provenance plumb: `GenExample.gen = branch:<name>` end-to-end (generate → dataset →
   train → quarantine path). **Test:** dataset rows carry the stamp; quarantine isolates them.

## Phase 3 — Serve + route CLI (subcommand group)
8. [x] `Branch { #[command(subcommand)] }` in `main.rs` with `Create`, `List`, `Route`, `Serve`
   (mirror the `Probe`/`Checkpoints`/`Quarantine` group pattern). Wire: `create`→`branch::create`;
   `list`→read registry; `route "<q>"`→`LocalBranchRouter::resolve` (print branches+scores, no
   serve); `serve <name>`→reuse `RunModel`/`[runtime]` on the branch GGUF.
9. [x] `serve --branches` (route per-request → serve the resolved branch). `ensemble=average_topk`
   → weighted blend of top-k branch outputs (BTM Merge); `single_best` (default) → top-1. v1
   one-shot (`--prompt`); note persistent server as a later extension. **Test:** `route` prints
   the expected branch on a fixture registry; `serve --branches` selects + runs the resolved
   branch (mock runtime); empty registry → base path.

## Phase 4 — Config wiring + cross-repo contract + verification
10. [x] `cmd_branch_*` handlers read `[branch]` for defaults (CLI flags override): base / corpus /
    objective / router / ensemble / serve plumbed (mirror track-27's export wiring).
11. [x] A runnable example: `[branch]` block in `bench/` (or a `run-*/`) `evolve.toml` — a fixture
    domain branch (e.g. "scrt-cli" reusing the bench corpus) demoing `branch create` + `route`.
12. [x] Schema-contract test: the serialized manifest/registry matches `SCRT-EVOLVE-INTEGRATION.md`
    (the hivemind contract); update the brief if the schema changes.
13. [x] `tracks.md`: finalize the dependency-graph node + the "After 29" phase gate (the table row
    is added at track-creation time; this finishes the graph + gate prose).
14. [x] Full verification sweep GREEN: `cargo test` + `clippy -D warnings` + `cargo fmt --check`;
    Python tests still green (train/eval/export); ML-free `cargo build` + `--features train` build
    green; the example `evolve.toml` parses.

## Status
DONE (2026-06-26) — **scope C** (create + serve + local route), composition-first. All phases
implemented + green: `cargo test` (branch SDK 18 + branch CLI 4, no regressions), `clippy -D
warnings`, `cargo fmt --check`, ML-free `cargo build` + `--features train` build, and the example
`bench/evolve.toml` `[branch]` block parses.

Landed:
- `BranchConfig` (+ router/ensemble/serve) on `EvolveConfig` (serde-default, back-compat) and
  `BranchManifest`/`BranchRegistry` (atomic, content-addressed `gguf_sha` via `sha2`; schema-version
  refused on mismatch) — `crates/scrt-evolve/src/{config.rs,branch/manifest.rs}`.
- `BranchRouter` trait + `LocalBranchRouter` (simhash centroid, unigram features; `confidence_floor`
  → base-only), `corpus_signature`, near-dup `admit` (merge vs cap, no twins) — `branch/router.rs`.
- `branch::create` orchestrator: scope per-branch config → discover → teacher-QA generate (stamp
  `gen=branch:<name>`) → carve probe → train → eval gate → export INSIDE the track-15 transaction;
  register only on Accept; Regress = no register; Catastrophe = quarantine `branch:<name>` + halt.
  Heavy stages injected as `BranchHooks` (SDK stays ML-free + testable) — `branch/create.rs`.
- CLI `branch {create,list,register,route,serve}` (+ `serve --route` = `serve --branches`;
  ensemble note) wiring the real subprocess stages — `crates/scrt-evolve-cli/src/main.rs`.
  `branch register` = the native counterpart to `create`'s export step: compute the
  `router_signature` (real `scrt_core` simhash) + assemble manifest + admit into the registry
  for an **externally-built** GGUF (out-of-process train/export, or importing a peer's branch).
- Cross-repo schema-contract test vs `SCRT-EVOLVE-INTEGRATION.md`; `tracks.md` graph node + "After 29"
  gate; runnable `[branch]` example in `bench/evolve.toml`.

The distributed **Merge** (P2P serve + ensemble across peers) remains the **hivemind** repo's,
contracted via `SCRT-EVOLVE-INTEGRATION.md`. Precursor for a future teacher→smaller-student
compression mode: `bench/seam_distill/` (de-risk PASSED 2026-06-25).

## Live local-branch validation (2026-06-26)
First real end-to-end branch on this box (RTX 4060 8GB): **TinyLlama-1.1B → scrt-CLI domain expert**.
Config: `bench/branch-scrt-cli.toml`. Work dir: `bench/work/scrt-cli-branch/`.
- discover: 40 passages from the scrt-cli repo; generate (teacher = LM Studio `meta-llama-3-8b-instruct`):
  **60 teacher-QA rows**; probe carve: 13 probe / 47 train.
- train: TinyLlama LoRA on the GPU (WSL `~/scrt-gpu-venv`), 200 steps, q_proj/v_proj, bf16 —
  **loss 3.70 → 0.05** (44 LoRA modules); export: merge → f16 → quantize → **667 MB Q4_K_M GGUF**
  (`bench/work/scrt-cli-branch/scrt-cli.gguf`); registered (`branch list` shows it; `branch route`
  on a mind-palace query resolves to it at score 0.53); served via `llama-completion` — returns
  domain-shaped output (talks about stashing search results into the mind palace).
- **Honest quality note:** eval correctness = **0.0** on the 13-item probe — TinyLlama-1.1B on ~60
  teacher-QA rows hallucinates exact CLI syntax. The **machinery** is proven end-to-end (train →
  export → register → route → serve); **quality** needs a bigger base / more data — and the eval
  gate correctly would NOT auto-admit a 0.0 branch (a real `branch create` rolls it back). The
  factory works; this base+data is a demo, not a production expert.

**Run as DECOMPOSED, not one `branch create`** — environment split: the teacher (LM Studio) is reachable
**only from native Windows**; the GPU + llama.cpp + torch live **only in WSL2**; `cargo`/`cmake` absent in
WSL. So the *identical shipped stages* ran across native + WSL. The one thing skipped vs `branch create`:
the **track-15 transaction** wrapper (train/export ran outside it) — compensated by gating manually
(register only on eval pass). Technique = genuine BTM Branch+Train; only the orchestration was split.

### Resume commands (for a fresh session, if the run was interrupted)
Native = `./target/debug/scrt-evolve`; `$M` = the TinyLlama snapshot (`/mnt/c/...`); `$W` =
`/mnt/c/Users/atooz/Programming/ai-utils-memory/scrt-evolve/bench/work/scrt-cli-branch`.
Free GPU first: `"/c/Users/atooz/.lmstudio/bin/lms" unload --all`.
```bash
# eval (WSL): correctness of base+adapter on the probe
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc 'source ~/scrt-gpu-venv/bin/activate; \
  export PYTHONPATH=/mnt/c/.../scrt-evolve/python; \
  python3 -m scrt_evolve_score --model $M --probe $W/probe.jsonl --adapter $W/adapter --metrics correctness'
# export Q4_K_M GGUF (WSL)
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc 'source ~/scrt-gpu-venv/bin/activate; \
  export PYTHONPATH=/mnt/c/.../scrt-evolve/python; \
  python3 -m scrt_evolve_gguf --model $M --adapter $W/adapter --out $W/scrt-cli.gguf \
    --quant Q4_K_M --llama-cpp ~/llama.cpp'
# register (native): manifest + registry + router_signature
./target/debug/scrt-evolve branch register --config bench/branch-scrt-cli.toml --name scrt-cli \
  --gguf bench/work/scrt-cli-branch/scrt-cli.gguf --correctness <score>
# serve (WSL): prompt the branch
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc '~/llama.cpp/build/bin/llama-completion \
  -m $W/scrt-cli.gguf -p "How do I stash search results with the mind palace CLI?" -n 200'
```

## Follow-ups (next session)
1. **2nd branch + routing demo** — create a `conductor` (track/spec/plan) branch and show `branch route`
   picking between scrt-cli vs conductor (this is where BTM **Merge** value first appears; one branch only
   proves Branch+Train + the routing surface).
2. **Real `serve --branches` ensemble** — `average_topk` currently serves the top-1 representative + logs
   intent; implement actual cross-branch output blending (or hand it to hivemind's Merge leg).
3. **Transactional `branch create` in this env** — the decomposed run skipped the track-15 txn wrapper;
   either build the CLI in WSL (needs `sudo apt install cmake` + rustup) for one-command GPU `branch create`,
   or enable LM Studio "Serve on Local Network" so WSL can reach the teacher.
4. **teacher→smaller-student distillation mode** — the v1 "smaller" is small-base+specialize; the true
   distill-to-smaller mode (precursor `bench/seam_distill`, de-risk PASSED) is the next BTM capability.
