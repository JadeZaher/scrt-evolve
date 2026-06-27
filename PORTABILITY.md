# scrt-evolve — Portability & Environment Contract

scrt-evolve ships as **two halves that bind at runtime**, not one frozen binary:

1. **The Rust CLI** (`scrt-evolve`) — a native binary. The default build is
   ML-free and Python-free (styleguide §1), so it cross-compiles and ships
   cleanly per platform.
2. **The Python ML backend** (`scrt-evolve-ml`) — a pip/uv-installable package
   (`python/pyproject.toml`) holding the heavy modules
   (`scrt_evolve_train/_infer/_gguf/_score/_dequant`).

The CLI invokes the backend as `<python> -m scrt_evolve_*`. Why two halves and
not one exe: the ML subprocesses need torch + transformers and, for hybrid-SSM
(Mamba) models, **compiled CUDA kernels** (`mamba-ssm`, `causal-conv1d`) that have
no universal wheels. PyInstaller/Nuitka cannot freeze a multi-GB, platform-specific
CUDA stack. "Portable" here means **the framework tells you exactly how to stand up
its environment and fails loudly (`doctor`) when it can't** — not "runs from one
binary anywhere."

## The binding: how the CLI finds the interpreter

One resolver, precedence high→low:

1. `--python /path/to/venv/python` (per-command flag)
2. `$SCRT_EVOLVE_PYTHON` (environment)
3. `[hardware].python` in `evolve.toml`
4. bare `python` on `PATH`

The CLI runs `-m <module>` against the **installed** `scrt-evolve-ml`. A repo
checkout's `python/` dir is added to `PYTHONPATH` only as a dev fallback, so you do
NOT need the source tree once the package is installed. Verify the whole chain with
`scrt-evolve doctor`.

## OS × accelerator matrix

| OS | Accelerator | Train (LoRA, non-SSM) | Train (hybrid-SSM / Mamba) | Eval / export | Notes |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Linux | CUDA | ✅ `[cuda]` | ✅ `[cuda]` + built mamba kernels | ✅ | The reference platform. |
| **WSL2** (Win11) | CUDA | ✅ | ✅ (the **only** verified Win+hybrid route) | ✅ | See the recipe below. |
| Windows native | CUDA | ✅ `[cuda]` | ❌ no mamba wheels; CPU backward segfaults | ✅ | Teacher (LM Studio) + llama.cpp serving work; hybrid *training* does not. |
| macOS | MPS / CPU | ⚠️ MPS partial | ❌ | ✅ (CPU) | CPU eval/api fine; GPU training unvalidated. |
| any | CPU only | ⚠️ small models | ❌ | ✅ | `[cpu]` extra; good for eval/api + tiny LoRA. |

## The verified WSL2 + CUDA recipe (Windows + hybrid-SSM)

This is the **only** route found to train a hybrid-SSM (Granite/Mamba) model on a
Windows box this project ran on. cargo/cmake are NOT assumed in WSL — install what
you need there explicitly.

```bash
# 1. A venv inside WSL2 (Ubuntu), Python 3.10+.
python3 -m venv ~/scrt-gpu-venv && source ~/scrt-gpu-venv/bin/activate

# 2. CUDA torch from the CUDA index (NOT the default PyPI CPU wheel).
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. The package + CPU-listed deps (transformers/safetensors/numpy/gguf).
pip install /mnt/c/.../scrt-evolve/python[cuda]

# 4. Hybrid-SSM kernels — BUILD FROM SOURCE (no universal wheels). Needs nvcc +
#    a torch-matching CUDA toolkit. This is the genuinely hard, version-coupled step.
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm   --no-build-isolation

# 5. A fresh llama.cpp checkout for export/dequant (its arch converters lag; a
#    current checkout is required for newer model families).
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build && cmake --build ~/llama.cpp/build -j

# 6. Bind + preflight.
export SCRT_EVOLVE_PYTHON=~/scrt-gpu-venv/bin/python
scrt-evolve doctor
```

### WSL gotchas that cost real time

- **Use the native ext4 fs for scratch, not a `/mnt/c` 9p mount.** Set
  `[export].work_path` to a fast native path; a 9p mount makes merge/convert
  crawl. (`out_path`/`place_dir` can still land on `/mnt/c` for LM Studio.)
- **WSL RAM is capped** (`.wslconfig`). Merge/export of a multi-B model can OOM at
  the default cap — raise it or use fractional/microshard training (track 25).
- **Disk**: a merge→f16→quantize export needs several × the model size free on the
  scratch target. `doctor` flags a missing model/llama.cpp; size the disk yourself.
- **The teacher (LM Studio :1234) is reachable from native Windows**, while GPU +
  torch + llama.cpp live in WSL2 — generation (teacher) and training (GPU) run on
  opposite sides of that boundary.

## Known ecosystem gaps (made legible, not solved)

- **No Windows-native `mamba-ssm`/`causal-conv1d` wheels.** Hybrid-SSM training on
  Windows requires WSL2 + a from-source kernel build. `doctor`'s `mamba_kernels`
  check reports importability so you find out in 2 seconds, not at minute 9.
- **llama.cpp architecture lag.** `convert_hf_to_gguf.py` trails new model
  families; a stale checkout fails to convert. Keep llama.cpp current.
- **CUDA torch is version-coupled** to the kernel builds. Match the torch CUDA
  version (e.g. cu121) to the toolkit you build `mamba-ssm` against.

## The CLI ↔ package version contract

The Rust CLI and `scrt-evolve-ml` version **in lockstep** (both `0.1.0` today).
The durable interface between them is **stable, not the versions**:

- the `dataset.jsonl` row schema (`scrt-evolve dataset-reference`),
- the subprocess CLI flags the Rust side passes to each `-m scrt_evolve_*` module,
- the final-line JSON the score/dequant modules print back.

A breaking change to any of those is a minor-version bump on **both** halves. The
CLI does not pin an exact `scrt-evolve-ml` version (the user controls the venv); it
relies on these contracts and on `doctor` to catch a mismatch early.
