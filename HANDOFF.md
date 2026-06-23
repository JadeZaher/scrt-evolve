# scrt-evolve — Session Pass-Off (2026-06-21)

Inline handoff so a future session (or coding agent) can resume without
re-deriving this session's hard-won state. Pairs with the auto-memory under
`~/.claude/projects/.../memory/` (MEMORY.md index + per-topic files).

## What scrt-evolve is now
An **opinionated, config-driven LLM-training + local-model-tooling framework**.
The whole loop is driven by `evolve.toml`:

  discover → generate (constitution/taste-steered) → train (dense | FRACTIONAL/
  sharded GPU, objective=distill|end_task) → export (merge → GGUF → quantize →
  place) → run-model (llama.cpp | transformers)

Run `scrt-evolve config-reference` for the full annotated schema (the entry
point for configuring it). `config-reference --toml` prints a template.

## Architecture (two halves — NOT a single exe)
- **Rust CLI** (`crates/scrt-evolve` SDK + `crates/scrt-evolve-cli`): native
  binary, default build is ML-free + Python-free (styleguide).
- **Python ML subprocesses** (`python/scrt_evolve_{train,gguf,score,dequant,infer}`):
  torch/transformers/(CUDA kernels). Shipped via pip/uv (planned track 28), bound
  to the CLI by a configured venv interpreter. Cannot be frozen into an exe
  (torch+CUDA+compiled mamba kernels). See memory [[portability-packaging-direction]].

## The working GPU environment (WSL2 — the ONLY path that trains Granite here)
Windows native (Py 3.13) can't build the mamba kernels. Everything ML runs in
**WSL2 Ubuntu 24.04**, venv `~/scrt-gpu-venv`:
- torch **2.5.1+cu121**, causal-conv1d 1.6.2, mamba-ssm 2.3.2 (built from source
  against that torch; see [[granite-gguf-export-pipeline]] + [[sharded-fractional-training]]).
- transformers 5.12, peft 0.19, bitsandbytes, cmake (pip).
- **WSL RAM raised to 30GB** in `%USERPROFILE%\.wslconfig` (memory=30GB,swap=12GB)
  — the bf16 model + serialization OOMs at lower caps.
- Fresh **llama.cpp at `~/llama.cpp`** (supports granitemoehybrid; built
  `llama-quantize` + `llama-completion`). The old `~/.unsloth/llama.cpp` is too
  old (no granite converter).
- Invoke WSL via **PowerShell** (`wsl.exe -d Ubuntu -- bash /mnt/c/.../script.sh`)
  with LF-only scripts; the Git-Bash tool mangles `/mnt/c` paths.

### ⚠ FRAGILITY: torch keeps getting clobbered to CPU
Installing Python deps (llama.cpp converter reqs, etc.) pulls a fresh **CPU**
torch (2.11.0+cpu), overwriting the cu121 build → "Torch not compiled with CUDA".
FIX: `pip install --force-reinstall --no-deps torch==2.5.1 --index-url
https://download.pytorch.org/whl/cu121`, then verify mamba kernels still import.
Always `--no-deps` when touching torch so the kernel ABI (built vs 2.5.1) holds.

## What's BUILT + verified this session
- **Fractional/sharded training** (track 25): layer-block shards bound VRAM to
  ~one block (3.3GB blk / 0.9GB per-module on 8GB RTX 4060); Granite backward
  works (no segfault). `[train.fractional]` (block_size/shards/granularity/
  objective).
- **objective = end_task** (THE data-sensitivity fix): final shard learns real
  CE on completions via the LM head (vs distill = MSE-vs-self, which imparts NO
  knowledge — that's why the first GGUF confabulated "scrt = secure computation").
  shard.py `_train_final_shard_end_task` / `_find_final_norm`.
- **Config-driven export** (track 27): `[export]` (quant/dtype/llama_cpp_path/
  work_path/out_path/place_dir/merge_shards). merge_shards.py unions per-shard
  adapters. Proven: produced a 4.03GB Q4_K_M GGUF of the evolved Granite.
- **Inference runtime**: `[runtime]` + `scrt-evolve run-model` (llamacpp via
  llama-completion | transformers). Reads config correctly.
- **config-reference** command: full schema for agents.
- **constitution/taste → generation**: `[evolve].constitution/taste` + per-goal
  overrides → `for_goal` layers → `compose_steering` → generate system prompt.
  End-to-end test passes. See [[signal-chain-integration]].

## Verified state of the signal chain
goals→discover→generate→train is a WORKING chain (end_task made training learn).
constitution/taste now steer generation (minimal slice via custom_prompt seam).
Full meta-object engine (tracks 21/22) still spec-only.

## end_task RESULT (the data-sensitivity verdict)
**The objective fix WORKS — confirmed by the loss curve.** A real end_task train
on Granite's final shard (shard 4, layers 32-40, rank 32, 80 steps, lr 5e-4,
59 pairs):
  CE loss **30.35 → 0.49** (first→last), a genuine 62× learning curve.
Contrast the old distill objective: 0.00002 → 0.0001 (flat noise — "reproduce
self", imparts nothing). So `objective=end_task` IS the lever that makes the
model sensitive to training data. Adapter saved: `~/scrt-export/adapter-endtask`
(shard-merged → adapter.safetensors), peak VRAM 3.78GB.

**The GGUF generation re-test is NOT yet done** — blocked by a WSL crash (below),
not by the pipeline. The loss curve is strong evidence the model learned; whether
it now correctly *answers* "what does scrt --mp-stash do" (vs the distill model's
"secure computation" confabulation) still needs the merge→GGUF→prompt once WSL is
back. NOTE: 59 pairs is still smoke-grade — a low final-loss on tiny data can be
memorization; a convincing knowledge test wants more pairs + a held-out probe.

## ⚠ WSL CRASHED at end of session (needs recovery before any ML re-run)
After many heavy model loads + OOM-kills, the WSL2 VM hit **catastrophic failure**
(`Wsl/Service/E_UNEXPECTED`, then `CreateInstance/E_FAIL` code 6) — the distro
won't launch; OS-level I/O errors (`getpwnam failed`, `I/O error @util.cpp`). The
export Python also died on `OSError: [Errno 5] Input/output error` during temp
cleanup — same FS instability.
RECOVERY (in order):
  1. `wsl --shutdown`; wait; retry. (tried — still failed)
  2. Restart LxssManager service **as admin** (Restart-Service needs elevation).
  3. **Windows reboot** — the reliable fix for E_UNEXPECTED.
  4. If the VHD is corrupt: `wsl --list -v`; worst case the ext4 VHD under
     `%LOCALAPPDATA%\…\Ubuntu` may need repair. The venv + ~/llama.cpp + GGUFs
     live INSIDE that VHD — if it's lost, the env (torch cu121 + built mamba
     kernels + llama.cpp build) must be rebuilt per the "working GPU env" recipe
     above. The adapter + datasets are reproducible from the repo.
After recovery, re-verify GPU torch (the clobber note) THEN resume the GGUF test.

## Original open items
1. **End_task knowledge GENERATION test**: resume after WSL recovery —
   `~/scrt-gpu-venv`, merge `~/scrt-export/adapter-endtask` → GGUF →
   `llama-completion` prompt "What does scrt --mp-stash do?".
2. **Smoke-grade only**: 59 training pairs is tiny. Real knowledge needs many
   phrasings + epochs. Data sensitivity = end_task + amplify (rank/epochs/data).
3. **C: drive is FULL (~1.4GB free)** — blocked placing the GGUF into LM Studio.
   GGUF lives at `~/granite-scrt-Q4_K_M.gguf` (WSL, 848GB free). Free C: or point
   LM Studio at the WSL path.
4. **Not committed/pushed**: this session's work (end_task, export config,
   runtime, config-reference, constitution/taste) is UNCOMMITTED on `main` at
   88d8eb9. Tracks 25/26/27/28 dirs written.

## How to resume the knowledge test (the immediate next step)
```
# 1. Ensure GPU torch (see fragility note)
# 2. end_task train the final shard:
wsl.exe -d Ubuntu -- bash /mnt/c/.../wsl_endtask.sh   # or via CLI once [train.fractional].objective="end_task"
# 3. merge + export (config-driven):
scrt-evolve export-gguf --config bench/evolve.toml    # uses [export] (merge_shards, dtype bf16, place_dir)
# 4. prompt it:
scrt-evolve run-model --config bench/evolve.toml --prompt "What does scrt --mp-stash do?"
```

## Verification baseline (all green as of this session)
Rust: all suites + config round-trips (fractional/objective, export, runtime,
constitution/taste steering), clippy -D warnings, fmt. Python: shard 8/8,
merge_shards 3/3, track23 6/6.
