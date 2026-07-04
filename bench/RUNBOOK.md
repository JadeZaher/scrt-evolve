# scrt-evolve Benchmark Runbook (track 24)

Evolve **IBM Granite-4.0-h-tiny** toward goals distilled from your own Claude
Code work, via the **eval-gated multi-goal schedule** (tracks 10/15/20) with
**quantization-aware training** (track 23), then export a Q4_K_M GGUF for LM
Studio.

This is the FINAL assembly: it wires together everything the lane built. The
multi-day run is **operator-launched** (you run it) — the steps below are bounded
and resumable, so you can run a short smoke first, then the long schedule.

## Granite TRAINING — the GPU path (VERIFIED) + the segfault to avoid
Granite-4.0-h-tiny is a hybrid **Mamba-2 (SSM)** model.
- On a **CPU-only torch**, `loss.backward()` **SEGFAULTS** (exit 139) in the
  naive Mamba kernel. Do not train Granite on the native Windows CPU venv.
- On a **CUDA torch WITH the `mamba-ssm`/`causal-conv1d` kernels**, training
  **works** — verified on this box's RTX 4060 (8GB) under **WSL2**: forward +
  backward run on the GPU, no segfault, across all 40 layers (incl. every SSM
  layer). See "GPU setup (WSL2)" below.

Granite is 6.9B, so a **dense** bf16 fine-tune (~22GB) OOMs on 8GB. The bench
therefore trains **fractionally** (`[train.fractional]`): one contiguous
**layer block** at a time, via block-local distillation (frozen block = teacher,
LoRA'd block = student). **Peak VRAM ≈ one block — measured ~3.1–3.4 GB for
`block_size=8`, flat across all 5 shards on the RTX 4060.** This is what lets a
large model train on a small GPU; smaller `block_size` ⇒ less VRAM, more
streaming. Shards are independent — you can train `--shard-index N` per machine.

EVAL + the teacher are forward-only (Granite = `transformers` eval scorer + the
LM Studio teacher); both run on CPU. To run the whole thing on plain CPU now,
switch `[evolve].model_path` to a non-Mamba student (e.g. TinyLlama) — fractional
mode still applies and bounds VRAM/RAM the same way.

### GPU setup (WSL2) — the verified environment
The native Windows Python 3.13 has no `mamba-ssm` wheels and no compiler; WSL2
Ubuntu is the path that works. One-time:
```bash
# In WSL2 Ubuntu (CUDA passthrough is automatic; nvidia-smi must see the GPU):
python3 -m venv ~/scrt-gpu-venv && source ~/scrt-gpu-venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy transformers peft accelerate safetensors sentencepiece bitsandbytes
# Build the Mamba kernels against THIS torch (CUDA_HOME points at the apt toolkit):
#  - patch torch's strict CUDA minor-version check (nvcc 12.0 vs torch 12.1 is ABI-safe)
#  - build with --no-build-isolation --no-deps so pip can't swap torch out
CUDA_HOME=/usr MAX_JOBS=4 pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm
# (mamba-ssm 2.3.2's __init__ eagerly imports Mamba3, which needs triton>=3.2;
#  make that import optional — Granite uses the Mamba2 selective-scan path.)
```
Raise the WSL RAM cap so the bf16 model fits (host has 32GB) — in
`%USERPROFILE%\.wslconfig`: `[wsl2]` / `memory=26GB`, then `wsl --shutdown`.
Run training from this venv with `[hardware].device = "cuda"`. The repo's
`python/` is reachable from WSL at `/mnt/c/.../scrt-evolve/python` (set
`PYTHONPATH` to it).

The track-15 transaction still **rolls a failed step back cleanly** (captured in
`work/evolution-log.jsonl` + `work/logs/round-<n>.log`).

## Prerequisites (verified present on this machine)
- Cached f16 HF Granite: `~/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-tiny/snapshots/<rev>/`
  (the bench `evolve.toml` `model_path` points here — NOT the GGUF).
- Python venv with torch + transformers (5.8, has `GraniteMoeHybridForCausalLM`):
  `C:/Users/atooz/Documents/Escherbridge/laxame-hivemind/hivemind-models/.venv313/Scripts/python.exe`
- LM Studio serving a teacher at `http://localhost:1234/v1` (the granite GGUF, or
  liquid/lfm2.5-1.2b). Edit `[generate.api].model` to match the served model id.
  **IMPORTANT: load the teacher with a context length ≥ 8192** (Settings →
  Context Length). Transcript passages are long; at the default 4096 the teacher
  rejects many passages with `n_keep >= n_ctx` (verified during bring-up). With
  a small context, also lower the adapter `--max-chars` (e.g. 1500) so passages
  fit.
- A llama.cpp checkout (for the final GGUF export): `~/.unsloth/llama.cpp` (built).

Set once per shell:
```powershell
$PY = "C:/Users/atooz/Documents/Escherbridge/laxame-hivemind/hivemind-models/.venv313/Scripts/python.exe"
cd C:/Users/atooz/Programming/ai-utils-memory/scrt-evolve
```

## Step 1 — Ingest Claude Code transcripts → corpus
Claude Code stores sessions in its native format; the shipped `evolve` ingest
path flattens them to the generic transcript shape under `bench/corpus/`.
```bash
$EVOLVE ingest --from "C:/Users/atooz/.claude/projects" --out "bench/corpus"
```
Output: per-session `<project>__<uuid>.jsonl` of `{role,text,command?}` rows.
(Under WSL, point `--from` at `/mnt/c/Users/atooz/.claude/projects`.)

## Step 2 — Build the binary
The binary is `evolve`. **For Granite, build + run under WSL** (the native
Windows venv lacks the mamba kernels — Granite's backward segfaults on it):
```bash
# in WSL: source ~/.cargo/env first
cargo build --release -p scrt-evolve-cli   # → target/release/evolve
EVOLVE=./target/release/evolve
```
(A CPU-only non-Mamba student can build/run the Windows `evolve.exe` the same way.)

## Step 3 — SMOKE (bounded) — prove the pipeline end-to-end
A 2-round schedule confirms discover→generate→train→eval→keep|rollback works on
Granite before the long run.

**Budget the smoke down first** — generate calls the teacher once per passage
per goal, so 120 passages × 3 goals is many calls before training even starts.
For a fast smoke, set `[discover].max_passages = 8` in a copy of the config (or
edit it), and adapt only a few sessions (`--limit-sessions 5`). Restore
`max_passages = 120` for the real run. Expect the smoke to still take minutes
(CPU teacher + CPU Granite train step).
```powershell
& $EVOLVE train auto --schedule `
  --config bench/evolve.toml `
  --max-rounds 2 --policy weighted `
  --python $PY
```
Watch for: per-round lines with `correctness=` and `[kept|rolled back]`; a
`work/score.json`; `work/checkpoints/` populated; `work/evolution-log.jsonl`
rows. If a round catastrophes it halts — inspect with
`& $EVOLVE watch quarantine list --config bench/evolve.toml` and re-arm with
`watch quarantine clear`.

## Step 4 — THE BENCH (long, operator-launched) — multi-day schedule
Scale the budget up. The schedule is bounded by `--max-rounds` and RESUMABLE
(re-running continues ordinals after the last checkpoint, reads last_good +
quarantine), so you can stop/restart across days.
```powershell
# e.g. 60 rounds, weighted across the 3 goals. Run in a durable shell.
& $EVOLVE train auto --schedule `
  --config bench/evolve.toml `
  --max-rounds 60 --policy weighted `
  --python $PY
```
Resume after an interruption: just re-run the same command — it picks up where
it left off. Track progress:
```powershell
& $EVOLVE watch checkpoints list --config bench/evolve.toml
Get-Content bench/work/evolution-log.jsonl -Tail 20
```

## Step 5 — Export the evolved model to GGUF (for LM Studio)
After the schedule, merge the kept adapter + export a Q4_K_M GGUF.
```powershell
& $EVOLVE train export-gguf `
  --config bench/evolve.toml `
  --quant Q4_K_M `
  --python $PY
# → bench/work/<model>-Q4_K_M.gguf  (load in LM Studio)
```

## Step 6 — Measure the lift (optional)
Compare base vs evolved on the held-out probe / the demo benchmark:
```powershell
& $EVOLVE train eval --config bench/evolve.toml --python $PY   # evolved (adapter applied)
& $PY demo/benchmark.py demo/baseline-static/dataset.jsonl bench/work/goals/scrt-cli-fluency/dataset.jsonl
```

## Step 7 — Track 32: tune `min_train_pairs` + the judge gate (empirical)

The min-QA-pairs floor and the judge gate both have a knob whose right value is
empirical, not assertable. Procedure to find the floor:

```powershell
# Sweep min_train_pairs over a fixed corpus; after each run read the trend.
foreach ($n in 1,2,4,8) {
  # edit bench/ambient.toml -> [daemon].min_train_pairs = $n  (or copy per-N configs)
  & $EVOLVE --ambient --dir bench            # let it run a bounded while, then stop
  & $EVOLVE ambient stop  --config bench/ambient.toml
  & $EVOLVE watch trend   --config bench/ambient.toml   # record Δtotal / arrow
  & $EVOLVE watch health  --config bench/ambient.toml   # record committed / last error
}
```

Pick the **smallest** `N` whose `watch trend` slope is non-negative and whose
degradation-judge regress rate is bounded — that's "enough signal per step to not
overfit" without starving the loop. Default is 4 (half a `batch=8`).

For the **judge gate** (`[regulate].gate = "judge"`): turn it on, run the same
loop, and compare the commit rate + `watch trend` against the correctness gate.
The judge gate should commit MORE steps (it accepts "no worse" instead of
requiring "measurably better") while `trend` stays flat-or-up — that's the gate
doing its job (progress without regression). If the judge endpoint is down, the
gate degrades to accept-all (catastrophe floor still applies) — `doctor` flags a
missing judge model first.

## Honest caveats
- **CPU training is slow.** Granite-4.0-h-tiny on CPU + QAT fake-quant overhead
  means each round's train step is the bottleneck. Budget steps accordingly
  (`--steps` default 40 inside the schedule's train hook). A GPU box would be
  far faster; this bench is correctness-of-machinery first, throughput second.
- **Teacher quality bounds curriculum quality.** The generate teacher (LM Studio)
  writes the rows; a stronger teacher → better curriculum. The eval gate is the
  backstop: a round that doesn't improve the probe is rolled back.
- **QAT is approximate.** The fake-quant is group-wise affine, not bit-exact
  llama.cpp Q4_K_M; it captures the dominant quant-error effect, not the exact
  block format. Whether it improves the *exported* GGUF is the experiment.
