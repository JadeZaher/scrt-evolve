# AGENTS.md — operating scrt-evolve

A precise operator's map for an AI agent driving this repo. For *why* and the
narrative, see [README.md](README.md); for status see
[conductor/tracks.md](conductor/tracks.md). This file is the *verbs and
contracts*.

## Code commentary & docs convention

**Prefer directory-level docs over long inline comment blocks.** Code carries
terse one-line doc-comments (the "what"); module/cross-cutting rationale (the
"why") lives in an `AGENTS.md` in that source directory (e.g.
[crates/scrt-evolve/src/AGENTS.md](crates/scrt-evolve/src/AGENTS.md)). When you'd
reach for a multi-paragraph `//!`/`///` block, add or extend that file's section
and point to it. Keep inline comments for the non-obvious local line only.

## Ground truth (do not over-claim)

- **Real ML is Python.** Training/inference/scoring/export run as **subprocesses**
  out of the `python/` packages. The Rust crate is ML-free by default.
- **candle `train`/`local` = fixtures.** `train` defaults to `--backend candle`,
  which **cannot load real checkpoints**. For a real model always pass
  `--backend transformers --python <interp>`.
- **Machine-readable output: pass global `--json`.** The artifact-producing
  commands then print a final JSON summary line on stdout (`generate`, `eval`,
  `plan`, `train`, `export-gguf`, `branch list|route|create`, `discover`,
  `probe build`, `daemon *`, `teach`, `doctor`). Exit codes are clean (`0` ok,
  `1` error). Schemas are introspectable: `config-reference` (evolve.toml),
  `dataset-reference` (rows + manifest), `commands [--json]` (the subcommand
  surface), `config-show` (resolved config for this run).
- **Preflight with `doctor`** before a long run — it reports torch/cuda/
  transformers/mamba, model path, llama.cpp, work_dir, each PASS/FAIL + a fix.
- **Interpreter binding (track 28):** the Python verbs run `<python> -m
  scrt_evolve_*` against the installed `scrt-evolve-ml`. Resolution: `--python` >
  `$SCRT_EVOLVE_PYTHON` > `[hardware].python` > bare `python`. A repo checkout's
  `python/` dir is only a PYTHONPATH fallback. See [PORTABILITY.md](PORTABILITY.md).
- GPU/Mamba/llama.cpp run in WSL2+CUDA; teacher endpoint is OpenAI-compatible
  (LM Studio). See [bench/RUNBOOK.md](bench/RUNBOOK.md).

## SDK entry points (crate `scrt_evolve`, see `src/lib.rs`)

The CLI is a thin argv→SDK shim. Primary functions:

| Function | Signature (abridged) | Produces |
|---|---|---|
| `EvolveConfig::load` | `(path) -> Result<EvolveConfig, ConfigError>` | parsed+validated config |
| `discover::run` | `(&cfg) -> Result<DiscoveredContext>` | passages + anchors |
| `generate::run` | `(&cfg, &ctx) -> Result<Dataset>` | the JSONL dataset (API teacher backend) |
| `train::run` | `(&cfg, &Dataset) -> Result<TrainReport>` | candle FIXTURE adapter (not real) |
| `eval::run_eval` | `(&cfg, python: Option<&str>) -> Result<ScoreReport>` | probe score |
| `eval::ProbeSet::carve` | `(&Dataset, holdout_frac) -> (ProbeSet, Dataset)` | held-out probe + train remainder |
| `Regulator::new` | `(&cfg) -> Result<Regulator>` | checkpoint store + quarantine |
| `rounds::run_schedule` | `(&cfg, policy, max_rounds, start_ordinal, &hooks, &baseline)` | eval-gated multi-goal rounds |
| `branch::create` | `(&cfg, name, base, corpus, domain, &baseline, created, &hooks) -> CreateReport` | a branch (transactional) |
| `LocalBranchRouter::from_config` | `(&cfg, &BranchRegistry)` → impl `BranchRouter` | request→branch resolver |

Real ML in the SDK path is delivered by **hooks** (closures the CLI supplies):
`rounds::RoundHooks { discover, generate, train, score }` and
`branch::BranchHooks` wrap the Python subprocesses so the pure-Rust round/branch
drivers stay ML-free and testable.

`python_pkg_dir()` walks up from cwd to find the `python/` dir holding
`scrt_evolve_train` and puts it on `PYTHONPATH`.

## Consumption contract (SDK-first — the CLI is one consumer, not the API)

scrt-evolve is built to be driven **two ways from the same library**: in-process
(a Rust caller — e.g. a desktop client — links `scrt_evolve` and calls the entry
points above) and out-of-process (shell out to the binary, read the `--json`
summary line). Both consume the same orchestration; neither owns it. The pattern,
so new commands stay portable to both:

1. **The SDK owns orchestration and returns a `Serialize` report.** A capability
   is a library function that runs the whole flow and returns a structured value
   — `CreateReport`, `TrainReport`, `ExportReport`, `DaemonReport`, `GoalsReport`,
   `ScoreReport`, `TxnOutcome`. No capability prints; it *returns*. A desktop
   client gets the typed struct; the CLI renders it.
2. **Heavy/impure work is injected as hooks, not embedded.** The Python
   subprocesses, GPU handoff, and llama.cpp calls live in closures the caller
   supplies (`BranchHooks`, `rounds::RoundHooks`, `DaemonHooks`). The SDK driver
   stays ML-free and unit-testable with deterministic mocks; each consumer wires
   its own real stages. **Never call a subprocess from inside an SDK driver.**
3. **The CLI handler is parse → build hooks → call SDK → render.** A `cmd_*`
   should resolve config + flags, construct the production hooks, call one SDK
   function, then `println!`/`emit_json` the returned report. Logic beyond that
   (loops, transactions, store mutation, keep/rollback decisions) belongs in the
   SDK so the desktop client inherits it for free.
4. **`--json` is the out-of-process contract.** Every artifact-producing command
   emits one machine-readable summary line; that line is the IPC surface for a
   client that shells out instead of linking. Keep it stable + flat.

**Conformance + known debt.** `branch::create` / `rounds::run_schedule` /
`run_daemon` follow this (orchestration in the SDK, hooks injected, report
returned). The one outlier is **`branch evolve`**: its orchestration (resume from
the live adapter, build the cross-round baseline, commit to the `[store]` ring,
deploy the GGUF) still lives in `cmd_branch_evolve` in `main.rs`, not behind a
`branch::evolve(...) -> EvolveReport` SDK entry point. **Migration target:** lift
that span into `src/branch/evolve.rs` mirroring `create` (CLI keeps only hook
wiring + render), add an ML-free mock-hook test, and **re-validate the live GPU
round on the CUDA box** (the path it touches can't be exercised ML-free). Until
then a desktop client cannot drive `evolve` in-process — only via the CLI.

## CLI surface an agent drives

Config flag defaults to `evolve.toml` everywhere. Binary:
`target/release/scrt-evolve`.

| Command | Purpose | Key flags |
|---|---|---|
| `init` | scaffold a commented `evolve.toml` | `--path` |
| `config-reference` | print annotated schema (queryable) | `--toml` (copy-pasteable template) |
| `discover` | corpus+palace → `discovered.json` | `--config` |
| `interview` | human directive → `directive.json` | `--answer id=value` (repeatable), `--core-only` |
| `plan` | planner LLM → `plan.json` | `--in` |
| `generate` | dataset → `dataset.jsonl` | `--backend local\|api`, `--self-route`, `--gap-rounds N` |
| `run` | discover → generate (→ export) | `--export` |
| `probe build` | carve held-out probe | `--from`, `--holdout`, `--out`, `--remainder` |
| `train` | LoRA adapter → `work_dir/adapter` | `--backend candle\|transformers`, `--python`, `--data`, `--preset`, `--out`, `--steps`, `--max-seq-len` |
| `eval` | score vs probe → `score.json` | `--probe`, `--python` |
| `infer` | HF base-vs-adapter A/B | `--prompt` (req), `--adapter`, `--ab`, `--max-new-tokens`, `--temperature`, `--chat`, `--python` |
| `run-model` | serve via `[runtime]` (llamacpp/transformers) | `--prompt` (req), `--python` |
| `export-gguf` | merge adapter → f16 → quantize | `--adapter`, `--out`, `--quant`, `--llama-cpp`, `--keep-intermediates`, `--python` |
| `dequant` | GGUF → HF safetensors (so it can be trained) | `--gguf`, `--out`, `--dtype`, `--tokenizer`, `--python` |
| `checkpoints {list,show <id>,restore <id>}` | checkpoint store inspection / manual rollback | `--config` |
| `quarantine {list,clear}` | provenance the loop skips after catastrophe | `--config` |
| `evolve <project>` | auto-detect palace+corpus, self-route pipeline | `--config`, `--gap-rounds`, `--export` |
| `evolve --goals` | multi-goal discover→generate (no eval gate) | `--config` |
| `evolve --schedule` | EVAL-GATED multi-goal rounds | `--max-rounds`, `--policy round-robin\|weighted`, `--python` |
| `branch create` | build+gate+export+register a branch | `--name` (req), `--base`, `--corpus`, `--domain`, `--distill`, `--teacher <path>`, `--steps`, `--python` |
| `branch list` | registered fleet (`branches/registry.json`) | `--config` |
| `branch register` | admit an externally-built GGUF (ML-free) | `--name` (req), `--gguf` (req), `--base`, `--domain`, `--dataset`, `--correctness`, `--parent` |
| `branch route <query>` | resolve request → branch(es) + scores | `--config` |
| `branch serve [<name>]` | serve a branch (one-shot) | `--route <query>`, `--prompt`, `--python` |
| `daemon {start,stop,status}` | ambient continuous-evolution loop (track 26): living-queue → microshard → track-15 txn. **Gentle-background** (default): yields the GPU to other processes (`pause_on_gpu_process`), CPU-fallback when busy, optional block rotation + cooldown — coexists with gaming/video | `start`: `--max-vram`, `--max-steps`, `--drain`, `--python`; `[daemon]`: `pause_on_gpu_process`/`cpu_fallback`/`rotation_blocks`/`cooldown_secs`; `status`: queue pending |
| `teach` | enqueue a prompt→completion on the daemon PRIORITY lane | `--prompt` (req), `--completion` (req) |
| `doctor` | preflight env (python/cuda/mamba/model/llama.cpp/work_dir) | `--python`, `--json` |
| `config-show` / `dataset-reference` / `commands` | resolved config / data-contract schema / subcommand manifest | `commands --json` |

**`evolve` is overloaded:** positional `<project>` is required for the plain
form, optional under `--goals`/`--schedule`. Pick the form explicitly.

## Contracts

### `dataset.jsonl` — the generate↔train boundary (`src/dataset.rs`)

One JSON object per line; `kind` tags the variant. Cross-language contract
(Rust writer ↔ Python reader). Variants and their fields:

- `qa`: `prompt`, `completion`, `source?`, `gen?`
- `instruction`: `instruction`, `input` (default ""), `output`, `source?`, `gen?`
- `completion`: `text`, `source?`
- `contrastive`: `query`, `positive`, `negatives[]`, `stash?`
- `tool_call`: `prompt`, `tool`, `arguments` (JSON object), `source?`, `gen?`
- `cli`: `prompt`, `command`, `source?`, `gen?`

`gen` is the **provenance stamp** (e.g. `trace:<goal>`); it is the quarantine
key — a catastrophic round quarantines the distinct `gen` stamps in its training
rows so the loop skips that cause.

### Branch manifest + registry — cross-repo contract (`src/branch/manifest.rs`)

`branches/registry.json` = `{ schema_version: 1, branches: [BranchManifest] }`.
A branch artifact is `{ <name>.gguf, manifest.json }`, GGUF content-addressed by
SHA-256. `BranchManifest` fields:

- `name` (registry/router key), `base_model`, `domain`, `corpus_descriptor`
- `router_signature`: `{ kind: "simhash"|"embedding"|"tfidf", vector: [f64] }`
  (simhash expands its 64 bits to a 64-dim {0,1} vector; cosine ≈ Hamming)
- `eval_report`: `{ metric → score }` (the gate that admitted it)
- `lineage`: `{ parent? }`, `version` (`MANIFEST_VERSION = "1"`), `gguf_sha`, `created`

All writes are atomic (temp+rename). `REGISTRY_SCHEMA_VERSION` / `MANIFEST_VERSION`
are version gates; a mismatched registry is **refused**, not guessed. **Changing
a field here is a coordinated cross-repo change** — the consumer is hivemind's
`SCRT-EVOLVE-INTEGRATION.md` (that doc lives in the hivemind repo, not here; the
contract is mirrored by these serde types).

### The `BranchRouter` seam (`src/branch/router.rs`)

```rust
pub trait BranchRouter { fn resolve(&self, req: &str) -> Vec<(BranchRef, f32)>; }
```

Routing is **per-request, not per-token**. `LocalBranchRouter` is the v1 local
resolver (simhash similarity vs each branch's `router_signature`, filtered by
`confidence_floor`, top-`k`). An **empty** result = "no branch matched" → serve
base-only (the safety floor). hivemind implements `RemoteBranchRouter` over the
**same trait** returning `(peer, branch)` — do not fork the trait.

## Config blocks (full schema: `scrt-evolve config-reference`)

`[evolve]` (model_path/corpus_dir/palace_path/work_dir/constitution/taste),
`[discover]`, `[generate]` (+`.api`/`.local`), `[train]` (+`.lora`/`.qat`/
`.fractional`/`.distill`/`full`/`pretrain`/`contrastive`/`shard`), `[eval]`,
`[regulate]`, `[hardware]`, `[export]` (+`.merge_shards`), `[runtime]`
(+`.sampling`), `[branch]` (+`.router`/`.ensemble`/`.serve`, `mode=distill` for
cross-model seam compression), and `[[goals]]` (name/topic/tag,
optional project/weight/cadence/constitution/taste). All stage blocks are
optional; partial configs work (the per-stage `model_path` requirement is
enforced only when a stage runs).

## Paired skill

`skills/scrt-evolve/SKILL.md` — the **learning-by-doing** companion: as you do
real work in a project that declares `[[goals]]`, stash goal-relevant findings
to the scrt mind-palace **tagged by the goal's `tag`** (one goal ⇄ one tag).
Those tagged stashes are what `discover` (with `palace_tags=[tag]`) pulls as the
high-signal curriculum. It pairs with the `scrt-context`/`mpg-context` skill
(the "how to stash/search"); this skill is the "what to capture, and why."
