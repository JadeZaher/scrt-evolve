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
its own QA / instruction pairs (via a local model *or* an API endpoint), and
a training preset finetunes on it. The "labels" are model-generated from
discovered context — self-supervised, not hand-written. A stash's **note** is
a natural-language query; its captured **nodes** are context the agent judged
relevant — a self-directed, post-deployment shaping loop over one directory of
work.

> **Status: design draft.** The architecture is fixed in
> [DESIGN.md](./DESIGN.md). This README is the intent; the design doc is the
> contract. Built in review-gated phases.

## How it relates to scrt

scrt-evolve **consumes** [scrt](../scrt-cli) — the Rust retrieval engine — as
a library (`scrt-core`), in-process. scrt does the retrieval (search, mind
palace, lexical similarity); scrt-evolve does the generation and training scrt
leaves out. scrt's similarity is **cheap and lexical** — it matches surface
form, not meaning ("dog Rex" and "my pet's name" never match). scrt-evolve is
the **semantic** tier: a trained model is the only thing that crosses that gap.

> Until scrt is published to crates.io, `scrt-core` is a pinned git dependency
> (swapping to the published crate is a one-line change tracked for release).

## What it does, in three stages

Each stage is independently runnable from the CLI or the SDK, and writes an
inspectable artifact, so you can stop, read, edit, and resume between them.

### 1. Discover — `scrt-evolve discover`
Uses scrt to retrieve context from the corpus, seeded by the palace
stashes, deduped and clustered via scrt's own similarity so generation
covers distinct topics instead of re-mining one. → `discovered.json`

### 2. Generate — `scrt-evolve generate`
Turns discovered context into supervised examples — QA pairs, instruction
data, or raw completions — via a **pluggable backend**: a **local model**
(candle inference, fully offline) or an **API endpoint** (OpenAI / Anthropic /
any OpenAI-compatible, for a stronger teacher). Both emit the same JSONL, so
the dataset is backend-agnostic. → `dataset.jsonl`

### 3. Train — `scrt-evolve train`
Finetunes a model into a reusable **adapter artifact**. Two backends:

- **`--backend transformers`** (the real-model path): shells out to the
  Python trainer (`python/scrt_evolve_train`), which loads a real HuggingFace
  causal-LM (RoPE / GQA / BF16) and trains LoRA on it. It runs as a subprocess
  over the `dataset.jsonl` contract, so the Rust build itself stays ML-free.
- **`--backend candle`** (the in-tree fixture): a small hand-built arch for
  mechanical validation; it does **not** load real checkpoints (see *Honest
  caveats*). Candle presets (`lora`, `full`, `pretrain`, `contrastive`,
  `shard`) are the Rust-native north star, deferred until candle's training
  ecosystem matures.

→ `adapter.safetensors` + `adapter_config.json`

### Run the artifact — `scrt-evolve infer`
Load the base model with the trained adapter and generate — `--ab` runs base
vs base+adapter side by side so you can see what the tuning changed.

## Constitution + taste: steering what gets learned

Discovery decides *what context* to learn from; **constitution and taste decide
what the model learns to do with it.** Two plain-text config fields compose into
the generation system prompt (the `custom_prompt` seam), so they shape the
*dataset* — and therefore the trained model:

- **`constitution`** — the **values** driving *how* the model answers (e.g.
  "cite file:line for every claim", "never invent an API not in the context").
- **`taste`** — the **representational form** ideas take: style, structure,
  conventions (e.g. "lead with the one-line takeaway, then imperative steps").

Both are optional (neither set ⇒ the built-in template). Because the steering
shapes the generated pairs, the model internalizes it through ordinary
finetuning — no reward model, no human labeling. The eval harness closes the
loop on the values half: point `[eval].judge` at an endpoint and it scores
**constitution-adherence**, so a round can gate on "did this move toward the
constitution?" rather than just "does it parse?"

## Model orchestration: many goals, one evolving model

The product shape is **one locally-tuned model that evolves with a user's goals
across all their projects.** A config declares any number of `[[goals]]`, and
the multi-goal driver fans **discover → generate** out over them, writing
inspectable per-goal artifacts under `work_dir/goals/<name>/`:

- Each goal carries a **`topic`** (scopes the corpus sweep + `palace_search`)
  and a **`tag`** (marks goal-relevant stashes → `palace_tags`), so the palace
  seeds *only* that goal's curated context; an optional **`project`** scopes it
  to one project's corpus.
- Each goal can **layer its own `constitution` / `taste`** on the global ones —
  so one goal tunes for terse code answers, another for cited prose, all feeding
  the same base model.
- It's a **bounded, non-mutating fan-out**: one goal's API failure is recorded
  against that goal alone, and no weights move until you've read the datasets.
  Per-goal scheduler hints (`weight`, `cadence`) are carried for the eventual
  round driver.

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
scrt-evolve train    --backend transformers --python /path/to/venv/python  # real model
scrt-evolve train    --backend candle --preset lora                        # fixture

# run the resulting adapter, base vs tuned side by side
scrt-evolve infer --prompt "What does scrt --mp-stash do?" --ab

# merge the adapter into the base and export a quantized GGUF for LM Studio
scrt-evolve export-gguf --quant Q4_K_M     # Q2_K…Q8_0 | f16 | none
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

# global steering — composed into the generate system prompt (see
# "Constitution + taste" above). both optional.
constitution = "Cite file:line for every claim. Prefer the smallest correct change."
taste        = "Lead with the one-line takeaway, then imperative steps."

[discover]
seed = "palace"                        # palace | corpus | both
palace_search = "auth"                 # only seed from stashes matching this
                                       # (scrt's --mp-list-search; name/note/
                                       # pattern/tag substring). omit ⇒ all stashes
palace_tags = ["security"]             # and/or restrict to stashes with these tags

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

[eval]
  [eval.judge]                          # optional: score constitution-adherence
  base_url    = "https://api.…/v1"
  model       = "…"
  api_key_env = "SCRT_EVOLVE_JUDGE_KEY"

# orchestrate many goals against the one model; each fans out
# discover → generate into work_dir/goals/<name>/
[[goals]]
name  = "auth-hardening"
topic = "authentication"               # scopes corpus sweep + palace_search
tag   = "security"                     # palace tag marking this goal's stashes
constitution = "Flag every place input crosses a trust boundary."  # layered on global

[[goals]]
name    = "api-docs"
topic   = "public api"
tag     = "docs"
project = "./packages/sdk"             # scope this goal to one project's corpus
taste   = "Answer as reference-doc prose with a runnable example."  # layered on global
```

## Honest caveats

- **Real training runs through Python.** candle has no turnkey PEFT equivalent
  and its in-tree model is a *fixture* that can't load real RoPE/GQA/BF16
  checkpoints, so the validated real-model path is `--backend transformers`.
  "100% Rust-native training" is the north star, not a day-one guarantee (see
  the dated amendment in [DESIGN.md](./DESIGN.md)).
- **Self-generated data can echo-chamber.** A small local model risks
  amplifying its own errors; the API backend sidesteps this with a stronger
  teacher, and local-gen output is treated as lower-trust (deduped, filtered).
- **The premise is unproven at quality.** "Self-generated data finetunes a
  usefully-better model" is plausible, not guaranteed — the pipeline being
  *wired and inspectable* is what makes the bet measurable rather than blind.

## License

MIT — see [LICENSE](./LICENSE).
