# COMPLETED.md — what's done vs roadmap

Honest scoping: **the runnable core (discover/generate), the eval + regulate
foundation, the Python/transformers bench training + export path, and the
branch factory are shipped and tested. The advanced self-evolve and architecture
lanes are designed-not-built.**

For the authoritative per-track table see
[`conductor/tracks.md`](conductor/tracks.md); for lessons + what diverged from
`DESIGN.md` see [`conductor/RETRO.md`](conductor/RETRO.md).

## Shipped (capability → CLI command → proving test)

| Capability | CLI command | Test |
|---|---|---|
| Config load/validate, partial configs, inline-secret guard | `config-reference`, all `--config` loads | `crates/scrt-evolve/tests/config.rs` |
| Work-dir / artifact layout | (all stages) | `crates/scrt-evolve/tests/workdir.rs` |
| Config scaffold | `init` | `crates/scrt-evolve/tests/scaffold.rs` |
| Discover (scrt-core search + palace + simhash dedup/cluster) | `discover` | `crates/scrt-evolve/tests/discover.rs` |
| Generate dataset (API teacher backend; JSONL contract) | `generate`, `run` | `crates/scrt-evolve/tests/generate.rs`, `generate_local.rs`, `dataset.rs` |
| Self-routed plan + gap-critic generation | `generate --self-route`, `plan` | `crates/scrt-evolve/tests/plan.rs` |
| Project auto-detect (palace+corpus) | `evolve <project>` | `crates/scrt-evolve/tests/project.rs` |
| Learning-by-doing multi-goal discover→generate | `evolve --goals` | `crates/scrt-evolve/tests/goals.rs` |
| Transcript harvest → corpus | (bench harvester) | `crates/scrt-evolve/tests/harvest.rs` |
| Candle training (FIXTURE — not real models) | `train` (`--backend candle`) | `crates/scrt-evolve/tests/train_lora.rs` |
| Eval harness: probe carve + scorer + gate + verdict | `probe build`, `eval` | `crates/scrt-evolve/tests/eval.rs` |
| Self-regulation: transactional checkpoint→eval→keep/rollback, quarantine, halt | `checkpoints {list,show,restore}`, `quarantine {list,clear}` | `crates/scrt-evolve/tests/regulate.rs` |
| Eval-gated multi-goal schedule (rounds through the txn) | `evolve --schedule` | `crates/scrt-evolve/tests/rounds.rs` |
| Config-driven GGUF export (merge → f16 → quantize → place) | `export-gguf` | `crates/scrt-evolve/tests/export.rs` |
| Branch factory: create/list/register/route/serve + manifest/registry/router | `branch {create,list,register,route,serve}` | `crates/scrt-evolve/tests/branch.rs`, `crates/scrt-evolve-cli/tests/branch_cli.rs` |
| Ambient daemon: two-lane living queue + VRAM-gated loop, every step through the track-15 txn (track 26) | `daemon {start,stop,status}`, `teach` | `living_queue` + `daemon` unit tests (queue round-trip, priority/cursor, VRAM-gate/stop/max-steps, txn commit) |
| Packaging + interpreter binding: `scrt-evolve-ml` pyproject, `--python`>`$SCRT_EVOLVE_PYTHON`>`[hardware].python` resolver preferring the installed package (track 28) | (all Python verbs), `doctor` | `python_resolution_precedence` (main.rs), `python/pyproject.toml`, `PORTABILITY.md` |
| Preflight + machine-readable surface (UX-review) | `doctor`, `config-show`, `dataset-reference`, `commands`, global `--json` | exercised via the CLI smoke + clap introspection |
| ML-free default build (no torch/candle to build the CLI) | `cargo build` | `crates/scrt-evolve/tests/ml_free_default.rs` |

### Real ML path (Python subprocesses, `python/`)

The validated real-model path is `--backend transformers` (and the export /
inference / scoring siblings), shelling out to the standalone packages:
`scrt_evolve_train`, `scrt_evolve_infer`, `scrt_evolve_score`,
`scrt_evolve_gguf`, `scrt_evolve_dequant`. CLI verbs: `train --backend
transformers`, `infer`, `run-model`, `eval --python`, `export-gguf`, `dequant`.
These need a torch+transformers venv (and llama.cpp for GGUF); GPU/Mamba run in
WSL2+CUDA — see [`bench/RUNBOOK.md`](bench/RUNBOOK.md).

### Bench / VRAM-bounding primitives (shipped)

- **Fractional / microshard training** (`[train.fractional]`): train one
  contiguous layer-block (or per-module sub-layer group) at a time via
  block-local distillation, bounding peak VRAM to a single block — verified on
  real Granite-4.0-h-tiny on an RTX 4060 (~3.1–3.4 GB at `block_size=8`).
- **QAT** (`[train.qat]`) and **dequant** (`scrt_evolve_dequant`): GGUF→HF +
  quantization-aware training toward the deployment quant.

## Live branch result

A real branch was built end-to-end and registered:

- **TinyLlama-1.1B → `scrt-cli` domain expert**
- Config: [`bench/branch-scrt-cli.toml`](bench/branch-scrt-cli.toml)
- Trained on the RTX 4060 (loss 3.70 → 0.05)
- Exported: a 667 MB **Q4_K_M GGUF** at
  `bench/work/scrt-cli-branch/scrt-cli.gguf`
- Registered: `bench/work/scrt-cli-branch/branches/registry.json`
  (domain `scrt/cli`, simhash `router_signature`, base = TinyLlama)
- Held-out eval **correctness is ~0** (a 1.1B model on a few dozen examples).
  This is the point being proven: the **end-to-end factory works**, and the
  **eval gate correctly would not auto-admit** a weak branch — `branch create`'s
  transaction registers only eval-passing branches; this one was registered via
  the explicit `branch register` path for fleet/serve demonstration.

## Roadmap (designed, not built)

Authoritative status in [`conductor/tracks.md`](conductor/tracks.md); lessons in
[`conductor/RETRO.md`](conductor/RETRO.md). The intentionally-open build work:

- **Track 26 — ambient daemon: SHIPPED** (machinery + tests; the live Granite/WSL
  GPU run is the only deferred piece — the loop is exercised ML-free).
- **Track 28 — packaging + venv binding: SHIPPED** (pyproject + resolver + doctor
  + `PORTABILITY.md`; per-platform binary CI + index publishing remain un-automated).
- **Track 08 — extract/publish against a published scrt-core**; **Track 30 — closeout**.

Designed but **no module shipped** (specs/stubs only — do not treat as roadmap
without re-scoping):

- Extra train presets: `contrastive` / `full` / `pretrain` / `shard` have source
  shells but unvalidated real-model versions; decentralized `shard` is the least
  finished.
- New modalities (skill ingestion / reasoning edits).
- **Self-evolve lane (11–14):** regen-antagonist, constitutional dialectic,
  attribution mask, in-model expert-spawn router — **no code exists**.
- **Architecture lane (16–18):** dag-engine / self-distill / sdk-builder —
  **nothing shipped** (largest design-vs-reality gap).
- **Bench lane 21–22:** taste-modules / meta-objects — not shipped; constitution
  + taste exist as config fields / the `custom_prompt` seam, but the
  taste-driven generation engine is unbuilt. (RETRO's finding: the real
  data-sensitivity lever was the training **objective** (`end_task`), not the
  generation driver.)
