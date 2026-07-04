# scrt-evolve-ml

The Python ML backend for [scrt-evolve](../README.md) — the heavy half that the
native Rust CLI drives as subprocesses. Five packages, each runnable as
`python -m <module>`:

| Module | Does | CLI command |
| :-- | :-- | :-- |
| `scrt_evolve_train` | Real-model LoRA training (transformers, prompt-masked CE; QAT + fractional/microshard) | `evolve train fit --backend transformers` |
| `scrt_evolve_infer` | Base vs. base+adapter A/B inference | `evolve model infer` |
| `scrt_evolve_gguf` | Merge adapter → f16 → quantized GGUF | `evolve train export-gguf` |
| `scrt_evolve_score` | Forward-pass scoring (perplexity / exit-depth) against a probe set | `evolve train eval` (transformers backend) |
| `scrt_evolve_dequant` | GGUF → HF safetensors (registry-driven, streaming) | `evolve train dequant` |

## Install

```bash
pip install scrt-evolve-ml[cpu]     # CPU torch — eval/api + small-model LoRA
pip install scrt-evolve-ml[cuda]    # CUDA torch (see ../PORTABILITY.md first)
pip install -e .[cpu]               # editable, for development
```

Then bind the CLI to this interpreter:

```bash
export SCRT_EVOLVE_PYTHON=/path/to/venv/bin/python   # or [hardware].python in evolve.toml
evolve doctor                                        # confirms torch/cuda/mamba/etc.
```

The CLI runs `<python> -m scrt_evolve_*` against the **installed** package; a repo
checkout's `python/` dir is only a `PYTHONPATH` fallback. For the full OS ×
accelerator matrix, the verified WSL2 + CUDA recipe, and the known ecosystem gaps
(no Windows mamba wheels, llama.cpp arch lag), see [PORTABILITY.md](../PORTABILITY.md).

## Tests

```bash
pip install -e .[cpu,test]
pytest          # python/tests
```
