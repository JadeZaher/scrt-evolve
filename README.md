# scrt-evolve

**scrt-evolve** makes a model *better at its own corpus* — with no human
labeling. Point it at a model, a directory of work, and a
[scrt](../scrt-cli) mind-palace, and it runs a self-contained loop:

```
   discover            generate                 train
 ┌──────────┐      ┌───────────────┐      ┌──────────────────┐
 │ use scrt │  ->  │ synth QA /     │  ->  │ finetune via a   │
 │ to find  │      │ finetuning     │      │ preset (lora /   │
 │ context  │      │ pairs from it  │      │ full / pretrain /│
 │ from the │      │ (local model   │      │ contrastive /    │
 │ corpus + │      │  OR API turns) │      │ shard)           │
 │ palace   │      └───────────────┘      └──────────────────┘
 └──────────┘
```

The corpus and palace **are** the training signal. scrt discovers the
relevant context; scrt-evolve turns it into supervised data by generating
its own QA / instruction pairs (via a local model *or* an API endpoint of
your choice), and a training preset finetunes on it. The "labels" are
model-generated from discovered context — self-supervised in the labeling
sense, not hand-written.

> **Status: design draft.** The architecture is fixed in
> [DESIGN.md](./DESIGN.md); no implementation yet. This README is the
> intent; the design doc is the contract. Built in review-gated phases.

## Why it exists

An agent's accumulated work — a corpus plus the mind-palace it built while
working — is a ready-made, unlabeled training signal. A stash's **note** is
a natural-language query; its captured **nodes** are context the agent
judged relevant. That structure is enough to:

1. **discover** what's worth learning (retrieve + dedup + cluster context),
2. **generate** supervised pairs about it (no human in the loop), and
3. **train** a model or adapter to internalize it.

It's a self-directed, post-deployment shaping loop, scoped to one
directory of work, over unstructured data.

## How it relates to scrt

scrt-evolve **consumes** [scrt](../scrt-cli) — the Rust retrieval engine —
as a library (`scrt-core`), in-process. scrt does the retrieval (search,
the mind palace, lexical similarity); scrt-evolve does the generation and
training scrt deliberately leaves out.

The two are complementary by design. scrt ships three **cheap, lexical**
similarity signals (SimHash, chunked best-pair/Jaccard, random-projection
cosine) that match *surface form, not meaning* — "dog Rex" and "my pet's
name" never match. scrt-evolve is the **semantic** tier: a trained model is
the only thing that crosses that gap. scrt finds and structures the
context; scrt-evolve learns from it.

> Interim dependency: until scrt is published to crates.io, `scrt-core` is
> a pinned git dependency. (Swapping to the published crate is a one-line
> change tracked for the first release.)

## What it does, in three stages

Each stage is independently runnable from the CLI or the SDK, and writes an
inspectable artifact, so you can stop, read, edit, and resume between them.

### 1. Discover — `scrt-evolve discover`
Uses scrt to retrieve context from the corpus, seeded by the palace
stashes, deduped and clustered via scrt's own similarity so generation
covers distinct topics instead of re-mining one. → `discovered.json`

### 2. Generate — `scrt-evolve generate`
Turns discovered context into supervised examples — QA pairs, instruction
data, or raw completions — via a **pluggable backend**:

- **local model** — candle inference on your model, fully offline; or
- **API endpoint** — turns against a configurable endpoint (OpenAI /
  Anthropic / any OpenAI-compatible), for a stronger teacher.

Both emit the same JSONL, so the dataset is backend-agnostic. → `dataset.jsonl`

### 3. Train — `scrt-evolve train`
Finetunes via a **preset**, each with its own config:

| Preset | What it does |
| :--- | :--- |
| **lora** | PEFT LoRA adapters (the practical default) |
| **full** | update all weights |
| **pretrain** | continued causal-LM pretraining on the raw corpus (domain adaptation) |
| **contrastive** | InfoNCE embedding adapter from palace structure — improves *scrt's own retrieval* |
| **shard** | decentralized training across a small trusted cluster |

→ `adapter.safetensors` (or full weights)

## Usage

```bash
# one config drives everything
scrt-evolve run --config evolve.toml

# …or each stage on its own, inspecting artifacts between them
scrt-evolve discover --config evolve.toml      # -> discovered.json
scrt-evolve generate --config evolve.toml      # -> dataset.jsonl
scrt-evolve train    --config evolve.toml      # -> adapter / weights

# override the configured backend / preset inline
scrt-evolve generate --backend api
scrt-evolve train    --preset lora --data dataset.jsonl
```

Everything the CLI does is also a library call — scrt-evolve is an SDK
first, a CLI second:

```rust
use scrt_evolve::{EvolveConfig, discover, generate, train};

let cfg     = EvolveConfig::load("evolve.toml")?;
let ctx     = discover::run(&cfg)?;        // corpus + palace -> DiscoveredContext
let dataset = generate::run(&cfg, &ctx)?;  // -> Dataset (jsonl-backed)
let report  = train::run(&cfg, &dataset)?; // -> TrainReport
```

## Configuration

One `evolve.toml`; the only required field is the model path. Secrets are
passed by env-var name, never inline. See [DESIGN.md](./DESIGN.md#config-schema)
for the full schema.

```toml
[evolve]
model_path  = "/models/my-model"      # the one required thing
corpus_dir  = "./src"
palace_path = ".mpg/mind-palace.json"

[generate]
backend = "api"                        # local | api
  [generate.api]
  base_url    = "https://api.…/v1"
  model       = "…"
  api_key_env = "SCRT_EVOLVE_API_KEY"  # env var NAME, not the key

[train]
preset = "lora"
  [train.lora]
  rank = 16
  alpha = 32
  lr = 2e-4
```

## Honest caveats

- **candle's finetuning ecosystem is thin.** No turnkey PEFT/trl
  equivalent — LoRA injection, the training loop, and per-architecture
  model loaders are largely hand-built. v1 starts with one well-supported
  architecture and grows coverage; "load any safetensors and finetune" is
  the goal, not a day-one guarantee.
- **Self-generated data can echo-chamber.** A small local model generating
  its own training data risks amplifying its own errors. The API backend
  sidesteps this with a stronger teacher; local-gen output is treated as
  lower-trust (deduped, filtered, optionally critiqued).
- **The premise is unproven at quality.** "Self-generated data finetunes a
  usefully-better model" is plausible, not guaranteed. The pipeline being
  *wired and inspectable* — you can read the dataset and swap the teacher —
  is what makes the bet measurable rather than blind.

## License

MIT.
