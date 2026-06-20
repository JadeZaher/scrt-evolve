# scrt_evolve_train

Real-model LoRA training path for scrt-evolve. The Rust/Candle path (`crates/`) is a fixture that cannot load RoPE/GQA models in production; this Python module is the actual training implementation. It uses `transformers` + hand-rolled LoRA (no `peft` dependency).

Ported and adapted from **lexame hivemind-models** `src/moe/expert_trainer.py`.

---

## What it does

1. Reads a `dataset.jsonl` (scrt-evolve format, see schema below).
2. Loads a local HuggingFace causal-LM snapshot (e.g. TinyLlama-1.1B) via `transformers`.
3. Attaches `LoRALinear` adapters to configured target linear modules (`q_proj`, `v_proj` by default).
4. Trains for a configurable number of steps with **prompt-masked cross-entropy** (loss only on completion tokens, mirroring the Rust LoRA preset).
5. Saves `adapter.safetensors` + `adapter_config.json` to the output directory (adapter-only; base weights are not modified).

---

## Installation / dependencies

Requires (all must be in the Python environment — no extras):

```
torch>=2.0
transformers>=4.35
safetensors>=0.4
```

No `peft`, no `accelerate`, no `datasets`.

---

## Invocation

```bash
# From the python/ directory (or set PYTHONPATH=python/)
python -m scrt_evolve_train \
  --dataset /path/to/dataset.jsonl \
  --model   /path/to/TinyLlama-1.1B \
  --out     /path/to/adapter_output \
  --steps   40 \
  --batch-size 1 \
  --max-seq-len 256 \
  --lr      2e-4 \
  --rank    16 \
  --alpha   32.0 \
  --dropout 0.05 \
  --target-modules q_proj,v_proj \
  --seed    0 \
  --log-every 5
```

If `--out` is omitted, the adapter is saved to `<dataset_dir>/adapter/`.

**PYTHONPATH note:** run from inside `python/`, or set `PYTHONPATH=/path/to/scrt-evolve/python`.

---

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dataset PATH` | (required) | Path to `dataset.jsonl` |
| `--model PATH` | (required) | Local HuggingFace model snapshot |
| `--out DIR` | `<dataset_dir>/adapter` | Adapter output directory |
| `--steps N` | 40 | Gradient steps |
| `--batch-size N` | 1 | Batch size (1 = safe for CPU) |
| `--max-seq-len N` | 256 | Max sequence length in tokens |
| `--lr F` | 2e-4 | AdamW learning rate |
| `--rank N` | 16 | LoRA rank |
| `--alpha F` | 32.0 | LoRA alpha (scaling = alpha/rank) |
| `--dropout F` | 0.05 | LoRA dropout probability |
| `--target-modules LIST` | `q_proj,v_proj` | Comma-separated Linear leaf names to wrap |
| `--seed N` | 0 | Random seed |
| `--log-every N` | 5 | Loss logging interval (stderr) |

---

## Dataset schema (`dataset.jsonl`)

One JSON object per line. The trainer processes two kinds:

### `qa`

```json
{"kind":"qa","prompt":"<str>","completion":"<str>","source":"<optional>","gen":"<optional>"}
```

- `prompt_text` = `prompt`
- `completion_text` = `completion`

### `instruction`

```json
{"kind":"instruction","instruction":"<str>","input":"<str>","output":"<str>","source":"<optional>","gen":"<optional>"}
```

- `prompt_text` = `instruction` (+ `"\n\n" + input` if `input` is non-empty)
- `completion_text` = `output`

All other kinds (`completion`, `contrastive`, `tool_call`, `cli`, etc.) are **skipped**; the count is logged to stderr.

---

## Output artifacts

```
<out>/
  adapter.safetensors   # lora_A / lora_B tensors keyed by module path
  adapter_config.json   # rank, alpha, target_modules, base_model_path
```

### Final-summary JSON (last stdout line)

The last line printed to **stdout** is a machine-readable JSON summary parseable by the Rust CLI:

```json
{"first_loss": 2.891234, "final_loss": 1.432100, "steps": 40, "adapters": 14, "out": "/abs/path/to/adapter"}
```

All other diagnostic output goes to **stderr**.

---

## Attribution

`LoRALinear` and the LoRA attachment/optimizer pattern are ported and adapted from:

> **lexame hivemind-models** — `src/moe/expert_trainer.py`

Used with adaptation for the scrt-evolve dataset schema and HuggingFace integration.
