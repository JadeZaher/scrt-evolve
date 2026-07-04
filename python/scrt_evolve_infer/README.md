# scrt_evolve_infer

Standalone inference module for scrt-evolve LoRA adapters.

Loads a HuggingFace causal-LM (e.g. TinyLlama-1.1B) and optionally applies a
LoRA adapter produced by `scrt_evolve_train`. Supports A/B comparison between
the base model and the adapter-patched model.

## Key design

- Reuses `scrt_evolve_train.trainer.LoRALinear` directly — the adapter tensors
  are loaded into the same class that saved them, so rank / alpha / scaling are
  guaranteed to match.
- Import path: `from scrt_evolve_train.trainer import LoRALinear`
  (both packages live under `python/`; add `python/` to `PYTHONPATH`).
- CPU-safe, float32, no CUDA assumed.

## Requirements

Same as `scrt_evolve_train`: `torch`, `transformers`, `safetensors`.

## Usage

Set `PYTHONPATH` to the `python/` directory of the repo, then:

```
# Base model only
python -m scrt_evolve_infer \
    --model /path/to/TinyLlama-1.1B \
    --prompt "What is scrt?"

# Adapter only (model path read from adapter_config.json)
python -m scrt_evolve_infer \
    --adapter /path/to/adapter \
    --prompt "What is scrt?"

# A/B: base vs adapter side-by-side
python -m scrt_evolve_infer \
    --adapter /path/to/adapter \
    --prompt "What is scrt?" \
    --ab

# Multiple prompts from a file, with sampling
python -m scrt_evolve_infer \
    --adapter /path/to/adapter \
    --prompts-file prompts.txt \
    --ab \
    --temperature 0.7 \
    --max-new-tokens 200
```

Or via the Rust CLI shim (reads model_path from evolve.toml):

```
evolve model infer --prompt "What is scrt?" --ab
evolve model infer --prompt "What is scrt?" --adapter ./my-adapter --ab --temperature 0.7
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | (from adapter_config.json) | Base HuggingFace model snapshot |
| `--adapter DIR` | none | Adapter dir with adapter.safetensors + adapter_config.json |
| `--prompt TEXT` | required* | Single prompt |
| `--prompts-file FILE` | required* | Newline-delimited file of prompts |
| `--max-new-tokens N` | 128 | Max tokens to generate |
| `--temperature F` | 0.0 | 0 = greedy; >0 = sampling |
| `--chat` | off | Wrap prompt in tokenizer chat template |
| `--ab` | off | Show base and adapter outputs side-by-side |

*`--prompt` and `--prompts-file` are mutually exclusive; one is required.

## Output format

```
=== PROMPT: <prompt> ===
[base]    <base generation>
[adapter] <adapter generation>
```

When `--ab` is not set and `--adapter` is given, only `[adapter]` is shown.
When no `--adapter` is given, only `[base]` is shown.

## Adapter format

Produced by `scrt_evolve_train`. Layout in `adapter_dir/`:

```
adapter_config.json   {"rank":16,"alpha":32.0,"target_modules":["q_proj","v_proj"],
                        "base_model_path":"<abs>","format":"safetensors"}
adapter.safetensors   tensors named like:
                        model.layers.0.self_attn.q_proj.lora_A  [rank, in]
                        model.layers.0.self_attn.q_proj.lora_B  [out, rank]
```
