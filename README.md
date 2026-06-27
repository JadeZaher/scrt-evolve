# scrt-evolve

**Make a model better at its own corpus — with no human labeling.** Point
scrt-evolve at a base model, a directory of work, and (optionally) a
[scrt](../scrt-cli) mind-palace, and it runs a self-contained loop:
**discover → generate → train → eval → export**. The corpus and palace *are*
the training signal; scrt discovers the relevant context, scrt-evolve turns it
into supervised data by generating its own QA / instruction pairs (via a local
model **or** a teacher API), trains a LoRA adapter on it, gates the result on a
held-out probe, and can export a quantized GGUF. On top of that loop it ships a
Branch-Train-Merge **branch factory**: take a small base + a domain corpus,
specialize it into a standalone domain-expert model (a "branch"), eval-gate it,
GGUF-package it, register it, and route requests to it locally.

> **Honest status.** The runnable core (discover/generate/eval/regulate), the
> Python/transformers training + export path, and the branch factory are
> **shipped and tested**. Several training presets, modalities, and the
> advanced self-evolve / architecture lanes are **specs/stubs, not built** — see
> [`conductor/tracks.md`](conductor/tracks.md) for authoritative per-track
> status and [`conductor/RETRO.md`](conductor/RETRO.md) for what diverged from
> the design.

---

## Constraints box (read this first)

- **Real ML training/inference runs through Python, not Rust.** The validated
  real-model path is `--backend transformers`, which shells out to the
  standalone Python packages under `python/` (`scrt_evolve_train`,
  `scrt_evolve_infer`, `scrt_evolve_score`, `scrt_evolve_gguf`,
  `scrt_evolve_dequant`). These load real HuggingFace causal-LMs (RoPE/GQA/BF16)
  via `transformers`.
- **The candle `train` / `local` backends are FIXTURES.** The in-tree candle
  path is a tiny hand-built arch for mechanical validation. It **cannot load
  real pretrained checkpoints.** `scrt-evolve train` defaults to `candle`, so for
  a real model you **must** pass `--backend transformers`.
- **GPU training + llama.cpp (export/serve) run in WSL2 + CUDA** on the dev box;
  the LM Studio teacher endpoint is reachable from native Windows. A hybrid-SSM
  model (e.g. Granite/Mamba) **segfaults on a CPU-only torch backward** — train
  it under a CUDA torch with the `mamba-ssm`/`causal-conv1d` kernels. See
  [`bench/RUNBOOK.md`](bench/RUNBOOK.md) for the verified environment.
- **Generation needs a teacher.** The `api` generate backend points at any
  OpenAI-compatible chat endpoint (LM Studio, a hosted API); the key is passed
  by **env-var NAME**, never inline.

---

## Install / build

```bash
# Rust CLI (ML-free by default — no torch/candle needed to build):
cargo build --release -p scrt-evolve-cli
# binary → target/release/scrt-evolve (.exe on Windows)

# Optional features (off by default):
#   train  → the in-tree candle FIXTURE backend (mechanical only)
#   pyo3   → the Python dataset/training-step bridge (needs Python headers)
cargo build --release -p scrt-evolve --features train
```

For **real training / inference / export** you need a Python venv with
`torch + transformers + peft + accelerate + safetensors` (plus `gguf` for
export, `sentencepiece`/`bitsandbytes` as needed). The CLI auto-locates the
`python/` package dir and puts it on `PYTHONPATH`; pass your interpreter with
`--python /path/to/venv/python`. For GPU/Mamba/llama.cpp specifics, follow
[`bench/RUNBOOK.md`](bench/RUNBOOK.md).

---

## Quickstart

A minimal `evolve.toml` (only `model_path` is strictly required; secrets are
env-var names):

```toml
[evolve]
model_path = "/models/TinyLlama-1.1B-Chat-v1.0"   # an HF model dir (safetensors + tokenizer)
corpus_dir = "./src"
work_dir   = ".scrt-evolve"                        # artifacts land here (default)

[discover]
seed = "corpus"                                    # palace | corpus | both
max_passages = 120

[generate]
backend = "api"                                    # local (candle fixture) | api (teacher)
kinds   = ["qa", "instruction"]
per_passage = 3
  [generate.api]
  base_url = "http://localhost:1234/v1"            # e.g. LM Studio
  model    = "meta-llama-3-8b-instruct"
  # api_key_env = "SCRT_EVOLVE_API_KEY"            # env var NAME, omit for local
  turns = 1

[train]
preset = "lora"
  [train.lora]
  rank = 16
  alpha = 32
  target_modules = ["q_proj", "v_proj"]            # ["auto"] to auto-detect
  lr = 2e-4
  epochs = 1

[eval]
probe_holdout_frac = 0.15
scorer_backend = "api"                             # api (no ML) | transformers (real forward pass)
metrics = ["correctness"]
```

Run the stages (each writes an inspectable artifact under `work_dir`):

```bash
EVOLVE=target/release/scrt-evolve
PY=/path/to/venv/python                            # interpreter for real ML

# 0. Scaffold a commented config (optional)
$EVOLVE init                                       # → evolve.toml

# 1. Discover context from corpus + palace          → discovered.json
$EVOLVE discover --config evolve.toml

# 2. Generate a dataset via the teacher              → dataset.jsonl
$EVOLVE generate --config evolve.toml
#    (1+2 in one shot:  $EVOLVE run --config evolve.toml)

# 3. Carve a held-out probe                          → probe.jsonl + dataset.train.jsonl
$EVOLVE probe build --config evolve.toml

# 4. Train a REAL model (LoRA via transformers)      → work_dir/adapter/
$EVOLVE train --config evolve.toml --backend transformers \
  --data .scrt-evolve/dataset.train.jsonl --python $PY
#    NOTE: omitting --backend gives the candle FIXTURE — not your model.

# 5. Score against the probe                          → score.json
$EVOLVE eval --config evolve.toml --python $PY

# 6. A/B the adapter vs base on a prompt (HF, via Python)
$EVOLVE infer --config evolve.toml \
  --prompt "What does scrt --mp-stash do?" --ab --python $PY

# 7. Merge + export a quantized GGUF for LM Studio   → work_dir/<model>-Q4_K_M.gguf
$EVOLVE export-gguf --config evolve.toml --quant Q4_K_M --python $PY
```

### Eval-gated multi-goal schedule

For unattended evolution across several `[[goals]]`, the `evolve --schedule`
umbrella runs bounded rounds of `discover → generate → train → eval →
keep|rollback` through the transactional regulator (halts on catastrophe,
resumable across runs):

```bash
$EVOLVE evolve --schedule --config evolve.toml \
  --max-rounds 4 --policy weighted --python $PY

$EVOLVE checkpoints list --config evolve.toml     # inspect kept/rolled-back rounds
$EVOLVE quarantine list  --config evolve.toml     # provenance the loop is skipping
```

### Branch factory (Branch-Train-Merge)

A **branch** is a standalone domain-specialized model (a BTM Expert LM). Two
ways to build one:

**A) One-shot** — `branch create` composes discover → teacher-QA generate →
train (`objective=end_task`) → eval gate → GGUF export inside the regulator
transaction; an eval-passing branch is registered, an eval-failing one is rolled
back and **not** registered:

```bash
$EVOLVE branch create --config evolve.toml --name scrt-cli \
  --base /models/TinyLlama-1.1B-Chat-v1.0 \
  --corpus ./scrt-cli --domain "scrt/cli" --python $PY
```

**B) Decomposed** — when train/export must run out-of-process (e.g. a WSL GPU
box) or you're importing a peer's artifact, run the stages separately and
register the finished GGUF natively (ML-free):

```bash
# native: discover + generate
$EVOLVE run --config evolve.toml
# WSL/GPU: train + export (Python) ... produces scrt-cli.gguf
# native: register the GGUF into the fleet (computes router_signature, manifest)
$EVOLVE branch register --config evolve.toml --name scrt-cli \
  --gguf work/scrt-cli-branch/scrt-cli.gguf --domain "scrt/cli"
```

Then route and serve:

```bash
$EVOLVE branch list  --config evolve.toml                 # registered fleet
$EVOLVE branch route --config evolve.toml "how do I stash with mp"   # resolve → branch + score
$EVOLVE branch serve --config evolve.toml --route "stash with mp" \
  --prompt "How do I stash a search result?" --python $PY
```

A real branch built this way ships in the repo: **TinyLlama-1.1B →
`scrt-cli` domain expert**, config
[`bench/branch-scrt-cli.toml`](bench/branch-scrt-cli.toml), trained on an
RTX 4060, exported to a 667 MB Q4_K_M GGUF, registered in the branch registry.
(Its held-out correctness is low — a 1.1B model on a few dozen examples — which
is the eval gate working as designed: it proves the end-to-end factory without
auto-admitting a weak branch.)

---

## SDK

Everything the CLI does is a library call — scrt-evolve is an SDK first:

```rust
use scrt_evolve::{EvolveConfig, discover, generate, train};

let cfg     = EvolveConfig::load("evolve.toml")?;
let ctx     = discover::run(&cfg)?;        // corpus + palace -> DiscoveredContext
let dataset = generate::run(&cfg, &ctx)?;  // -> Dataset (jsonl-backed)
let report  = train::run(&cfg, &dataset)?; // -> TrainReport (candle fixture path)
```

See [`AGENTS.md`](AGENTS.md) for the full operator map (SDK entry points, the
CLI surface, and the dataset / manifest / registry contracts).

---

## Pointers

- [`conductor/tracks.md`](conductor/tracks.md) — authoritative per-track status
  (what's shipped vs roadmap).
- [`COMPLETED.md`](COMPLETED.md) — what's done (capability → CLI command → test)
  vs the roadmap.
- [`conductor/RETRO.md`](conductor/RETRO.md) — honest per-lane retrospective.
- [`conductor/UX-REVIEW.md`](conductor/UX-REVIEW.md) — known DevUX/AIUX rough edges.
- [`bench/RUNBOOK.md`](bench/RUNBOOK.md) — the verified WSL2/GPU run procedure.
- [`AGENTS.md`](AGENTS.md) — for an AI agent driving this repo.
- `scrt-evolve config-reference [--toml]` — the full annotated `evolve.toml`
  schema, queryable from the CLI.
- `scrt-evolve dataset-reference` — the `dataset.jsonl` row + branch
  manifest/registry schemas (the cross-language / cross-repo contracts).
- `scrt-evolve doctor` — preflight your env (Python deps, model path, llama.cpp,
  work_dir) with PASS/FAIL + a fix for each, before a long run.
- `scrt-evolve config-show` — the fully-resolved config for THIS run (defaults
  applied); `commands [--json]` — the machine-readable subcommand surface.
- `--json` (global) — a machine-readable summary line on the artifact-producing
  commands, for coding agents driving the CLI.
- Per-track specs/plans live under `conductor/tracks/<NN>-<name>/`.

## License

MIT — see [LICENSE](./LICENSE).
