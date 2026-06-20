# scrt_evolve_gguf

Merge a LoRA adapter into a HuggingFace base model and export a quantized GGUF
file for use in **LM Studio** or **llama.cpp**.

Reuses `scrt_evolve_train.LoRALinear` and `attach_lora` verbatim.
Shells out to llama.cpp tooling for the conversion and quantization steps.

---

## 3-stage pipeline

```
base model (HF dir)
    +
adapter dir (adapter.safetensors + adapter_config.json)
          |
          v
[Stage 1] MERGE (Python)
  - Load base model (float32, CPU-safe)
  - attach_lora() wraps target nn.Linear modules with LoRALinear
  - Copy lora_A / lora_B tensors from adapter.safetensors
  - LoRALinear.merge_and_unload() folds delta into base weights
  - Swap plain nn.Linear back in (model is a clean CausalLM again)
  - Save merged HF model to <out_dir>/_merged_hf/
          |
          v
[Stage 2] CONVERT (subprocess)
  convert_hf_to_gguf.py <merged_hf_dir>
      --outfile <stem>-f16.gguf --outtype f16
  cwd = <llama_cpp_dir>  (so vendored gguf-py resolves)
          |
          v
[Stage 3] QUANTIZE (subprocess)
  llama-quantize <stem>-f16.gguf <out.gguf> Q4_K_M
  (skipped for quant=f16 or quant=none — f16 GGUF is the final output)
          |
          v
final GGUF  (default Q4_K_M — drop into LM Studio or llama.cpp)
```

---

## Requirements

The interpreter running this needs `torch`, `transformers`, `safetensors` (for
the merge), and — for SentencePiece-tokenizer models like Llama/TinyLlama —
`sentencepiece` (the llama.cpp converter imports it to write the vocab):

```bash
python -m pip install sentencepiece
```

A llama.cpp checkout (with `convert_hf_to_gguf.py` + a built `llama-quantize`)
is located automatically (`~/.unsloth/llama.cpp`, `~/llama.cpp`, `$LLAMA_CPP`)
or via `--llama-cpp`.

---

## Quick start

```bash
# From the python/ directory (so scrt_evolve_train is importable):
cd <repo>/python

python -m scrt_evolve_gguf \
    --adapter /path/to/adapter \
    --out     /path/to/output.gguf \
    --quant   Q4_K_M
```

`--model` is optional when `adapter_config.json` contains `base_model_path`.

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | from adapter_config.json | Base HF model directory |
| `--adapter DIR` | (none) | Adapter dir; omit for base-only export |
| `--out FILE` | `<adapter_dir>/../model-<quant>.gguf` | Output GGUF path |
| `--quant TYPE` | `Q4_K_M` | Quantization type (see below) |
| `--llama-cpp DIR` | auto-detected | llama.cpp checkout dir |
| `--keep-merged` | false | Keep `_merged_hf/` after conversion |
| `--keep-f16` | false | Keep intermediate f16 GGUF after quantization |

### Supported quant types

`Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_M`, `Q5_K_M`,
`Q6_K`, `Q8_0`, `f16`, `none`

`f16` and `none` skip the quantize step — the f16 GGUF is the final output.

---

## llama.cpp auto-detection

When `--llama-cpp` is not provided, the following locations are tried in order:

1. `$LLAMA_CPP` environment variable
2. `~/.unsloth/llama.cpp`
3. `~/llama.cpp`
4. `~/Documents/llama.cpp`

A valid location must contain `convert_hf_to_gguf.py`.
The quantize binary is found under `<llama_cpp_dir>/build/bin/Release/llama-quantize(.exe)`.

---

## Final JSON summary

The last line printed to stdout is machine-readable and consumed by the
Rust CLI:

```json
{"out": "/abs/path/to/output.gguf", "quant": "Q4_K_M", "size_bytes": 637534208, "base_model": "/abs/path/to/base", "adapter": "/abs/path/to/adapter"}
```

All progress and info messages go to stderr.

---

## Attribution

- `LoRALinear` and `attach_lora` reused from
  `scrt_evolve_train.trainer` (originally ported from lexame
  hivemind-models `src/moe/expert_trainer.py`).
- GGUF conversion via
  [llama.cpp `convert_hf_to_gguf.py`](https://github.com/ggerganov/llama.cpp).
- Quantization via `llama-quantize` (built from llama.cpp).
