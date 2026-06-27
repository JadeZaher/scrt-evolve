# Cross-model seam distillation — `branch create --distill` (track 29 v1.1)

Productizes the `bench/seam_distill` precursor (see [RESULTS.md](RESULTS.md)) into
the shipped pipeline: compress a **larger teacher** into a **smaller student
branch** by matching the student's per-block hidden state to the teacher's hidden
state at a mapped layer **seam**. This is genuine teacher→smaller-student
compression — *not* "smaller-by-base" (v1's only lever).

## The gap it closes

`shard.py`'s block-local loop distilled a block against **its own** frozen output
(`teacher = block(x, lora_off); student = block(x, lora_on)`) — a
representation/regularization signal that imparts no new capability. The teacher
and student were the **same** weights. This feature makes the teacher a
**distinct, larger model**.

## Mechanism — two decoupled phases (the VRAM crux)

An 8 GB box cannot hold a 7B teacher + a student at once (let alone 32B + 3B). So
the teacher and student are **never co-resident**:

- **Phase A — capture** (`--distill-phase capture`): load the teacher *alone*,
  stream it **one layer at a time** (peak VRAM = one teacher layer), capture its
  residual-stream hidden states at the mapped seam boundaries, write them to a
  disk cache as **fp32**, then **free the teacher**. The cache also stores the
  exact token ids so Phase B feeds the student an identical sequence.
- **Phase B — train** (`--distill-phase train`): load the student *alone*; for
  each student block, stream its boundary input and train the block's LoRA + a
  discard-after **read-out projection** (bridges width) to match the cached
  teacher seam (`cosine_mse`, fp32 master weights).

`--distill-phase both` (default) runs A then B in one process — the teacher is
freed before the student loads. The phases can also run on **different machines**
(capture on a big box, train on a small one).

### Layer correspondence

The teacher is deeper than the student. A student block ending at student-layer
`b` (of `L_s`) maps to teacher seam `round(b · L_t / L_s)`:

- `stride` (default) — one nearest teacher seam per student boundary.
- `block_avg` — average the teacher layers spanning the student block.

### Width bridge

When teacher/student hidden sizes differ (e.g. 4096 vs 2048), a per-block linear
**lifts the student output up to teacher width** so the loss is computed in
teacher space (the target is never down-sampled). The projection is a
**distill-time scaffold** — trained jointly, then **discarded** (only the LoRA is
saved), so the exported model is byte-for-byte a normal student + LoRA.

### Hard requirement: a shared tokenizer

Hidden states are matched **position-by-position**, so teacher and student must
tokenize identically. The Python side **guards** on `vocab_size` and aborts with a
clear message on a mismatch. (For cross-*tokenizer* pairs, sequence-level data
distillation — teacher pregenerates completions, student SFTs — is the right tool;
that is the existing `generate`→`train(end_task)` branch path, not this one.)

## Findings carried from the precursor ([RESULTS.md](RESULTS.md))

- **fp32 master weights.** bf16 AdamW updates for the small per-block delta round
  away and the student stalls; the student block trains in fp32.
- **Capture targets in fp32.** Avoids precision loss in the cached targets.
- Unlike the same-model delta case, the teacher target is a *different* model's
  state (not ≈ the student input), so it cannot trivially collapse to identity —
  the full (projected) hidden state is matched directly with `cosine + mse`.

## Usage

```toml
# evolve.toml
[branch]
base = "/models/TinyLlama-1.1B"   # the SMALL student
mode = "distill"

[train.distill]
teacher_model = "/models/Wizard-Vicuna-7B"   # the LARGER teacher (shared tokenizer)
layer_map = "stride"
loss      = "cosine_mse"
projection = "auto"

[train.fractional]
block_size = 2        # VRAM streaming knob (reused by distill)
calib_batches = 8
```

```bash
# CLI flags override config; the weight-touching span runs inside the track-15 txn.
scrt-evolve branch create --name scrt-distill \
  --base /models/TinyLlama-1.1B --teacher /models/Wizard-Vicuna-7B \
  --distill --steps 300 --python ~/scrt-gpu-venv/bin/python
```

The branch is **eval-gated**: a passing student is exported to a GGUF + registered;
a regress rolls back and is not registered; a catastrophe quarantines + halts —
exactly like a standard branch create (`src/branch/create.rs` is unchanged).

## Env realities

GPU + torch live in **WSL2** (`~/scrt-gpu-venv`, cu121). The teacher must be a
**safetensors** snapshot (transformers on torch < 2.6 refuses to load legacy
`.bin`; see `convert_teacher_safetensors.py` for the one-time convert). Both
models load via the standard transformers path — no SSM kernels needed for a
transformer↔transformer pair.

## Local validation run (2026-06-26, RTX 4060 8GB, WSL2)

**Pair:** Wizard-Vicuna-7B-Uncensored (teacher, LLaMA, 32 layers, d=4096) →
TinyLlama-1.1B (student, 22 layers, d=2048). Shared Llama SentencePiece tokenizer
(vocab 32000). `block_size=2` ⇒ 11 student blocks; stride seam map
`[3,6,9,12,15,17,20,23,26,29,32]`; `cosine_mse`, `projection=student_up`,
300 steps/block. (Mistral-7B was the first choice but is tokenizer-only in this
cache; the teacher `.bin` was converted once to safetensors — torch<2.6 won't load
legacy `.bin` — see `convert_teacher_safetensors.py`.)

| Stage | Result |
| :-- | :-- |
| **Phase A — capture** | 291 s · 353 MB seam cache · teacher then freed |
| **Phase B — train** | 184 s · 11 blocks × 300 steps · 132 LoRA adapters · **peak VRAM 0.93 GB/block** |
| **Phase C — export** | 42 s · f16 2.2 GB → **Q4_K_M 637 MB GGUF** |

Per-block final loss fell from ~31 (init) to **0.19–2.49** on **10/11 blocks**;
shard 2 (layers 4–6) initially **diverged** (stuck at ~31 after a mid-run spike).
The microshard VRAM bound held at **<1 GB** throughout — the two models were never
co-resident (the teacher pass is one-time).

### Stability fix — grad clip + dynamic per-block LR (re-run, same cache)

The divergence was an LR/stability issue, fixed with three additive knobs in the
Phase B loop (`lr_mode="auto"`, default):

- **Gradient clipping** (`grad_clip`, default 1.0) — a hard ceiling on step size,
  killing the spikes that wreck a block.
- **Dynamic per-block LR** (`block_lr_scale`) — each block's base LR is scaled by
  its teacher-seam magnitude relative to the shallowest block (`ref_rms /
  target_rms`, clamped), so larger-magnitude deep blocks take gentler steps. No
  hand-tuning — derived from the cached targets.
- **Warmup → cosine decay** (`lr_at_step`) within each block — warmup prevents the
  early spike, decay settles convergence (the precursor used this).

Re-running **Phase B only against the cached seams** (the decoupling win — no
teacher reload) eliminated the divergence: **every block converged, shard 2 went
31.26 → 0.355.**

| block | before | after | | block | before | after |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 0.19 | 0.21 | | 6 | 1.03 | 0.96 |
| 1 | 0.29 | 0.28 | | 7 | 1.36 | 1.43 |
| **2** | **31.26 ✗** | **0.355 ✓** | | 8 | 1.77 | 1.84 |
| 3 | 1.21 | 0.48 | | 9 | 2.17 | 2.27 |
| 4 | 0.55 | 0.50 | | 10 | 2.49 | 2.96 |
| 5 | 1.17 | 0.64 | | | | |

Deep blocks (8–10) carry higher *absolute* loss — a magnitude artifact of the MSE
term (their teacher hidden states are larger), not instability; the directional
(cosine) component fits well. Normalizing the MSE by target variance is the next
refinement if comparable per-block loss is wanted.

**Reproduce:** `bench/seam_distill/run_distill_branch.sh` (WSL2); re-train only with
`SKIP_CAPTURE=1 SKIP_EXPORT=1`.

## Running it as a gentle background task (coexist with gaming / video)

The two-phase split is what makes distillation background-friendly: the heavy
teacher pass is **one-time** (291 s here) and produces a reusable cache; the
ongoing student training is **one block at a time, <1 GB VRAM** (0.93 GB measured).
Fold that into the shipped **ambient daemon** (track 26) and tune it to yield:

```toml
[daemon]
max_vram_gb          = 7.0    # only train when the GPU is basically idle
pause_on_gpu_process = true   # yield instantly when a game/video uses the GPU
cpu_fallback         = true   # when the GPU is busy, do a light CPU step instead of stalling
rotation_blocks      = 22     # train one layer-block/step, rotate — VRAM stays at one block
cooldown_secs        = 5      # idle gap between steps so foreground apps don't stutter
```

The daemon probes `nvidia-smi` each step: GPU free → train on GPU; another process
on the GPU (a game) → fall back to a CPU block or pause; VRAM starved → wait. Every
step still runs through the track-15 transaction (eval-gate → keep|rollback), so
ambient training can never silently degrade the model. **Recipe:** run the teacher
capture once when idle (or overnight), then `daemon start` trains from the cache
forever, politely.

## Honest status

- The precursor proved the **mechanism** (a from-scratch block learns a layer's
  contribution by hidden-state distillation, and generalizes). This feature proves
  the mechanism **across two distinct models** end-to-end into a smaller GGUF.
- It does **NOT** prove that a small student *retains a large teacher's quality*.
  LoRA on a frozen student is a deliberately weak lever (it nudges, it cannot
  relocate capacity). Compression **quality is unproven** — only the pipeline is.
- **32B → 3B is the target scale**, runnable with this exact two-phase
  teacher-streaming approach on a larger GPU. It is documented, not yet run.
