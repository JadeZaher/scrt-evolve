# Seam-distillation de-risk — RESULTS

**Date:** 2026-06-25 · **Box:** RTX 4060 8GB, WSL2 Ubuntu, `~/scrt-gpu-venv`
(torch 2.5.1+cu121, mamba-ssm 2.3.2) · **Script:** `seam_distill_tinyllama.py`

## Question
Can a single Mamba2 (SSM) block, synthesized from scratch, learn to reproduce a
TinyLlama **transformer** decoder layer's contribution (`in_L → out_L`) via
hidden-state distillation on the 8GB box? This is the capability that gates
evolve's Mamba **linker** head for the Branch-Train-Merge P2P fabric.

## Verdict: YES — capability confirmed, data-limited, scales cleanly
A from-scratch Mamba2 block learns the layer's map and **generalizes** to
held-out sequences. The headline metric is **delta-cosine** =
`cos(mixer(norm(x)), out−x)` (how well the SSM reproduces the layer's actual
*contribution*; full-output cosine is a near-trivial bar because the residual
stream is self-similar across one layer).

### Data-scaling curve (layer 11, student = `x + Mamba2(RMSNorm(x))`, 25.8M params)
| calibration | tokens | val delta-cos | val full-cos | train/val gap |
|---|---|---|---|---|
| 16 seqs  | 3k   | 0.51  | 0.91 | 0.37 (overfit) |
| 256 seqs | 65k  | 0.645 | 0.93 | 0.09 |
| 512 seqs | 131k | **0.739** | **0.95** | **0.033** (generalizing) |

Monotonic with data; gap collapses to ~0. The literature's alignment stage
(MOHAWK, arXiv 2408.10189) uses ~240M tokens — **1,800× more** than the 131k
here — so this extrapolates to high fidelity. Each run: ~150s, peak VRAM 2.3 GB.

## Two implementation findings that de-risk the real build
1. **Distill the DELTA, not the full output.** `out_L = in_L + small_delta`, so a
   full-output MSE is minimized by driving the mixer to ~0 (the SSM collapses to
   identity — looked like a 131× MSE "win" but was only 1.03× better than
   predicting `out=in`). Train the mixer on `out − in`.
2. **fp32 master weights.** With bf16 params, AdamW updates for the small delta
   (`mean|delta|≈0.05`) round away below bf16 resolution and the mixer stays
   stuck at ~0. fp32 params unblocked learning entirely (train delta-cos
   0.42 → 0.88). Capture activations in fp32 too (else `out−in` loses the delta
   to catastrophic cancellation). The Mamba2 *kernel* still runs in bf16.

## What this proves for the project
- evolve's `shard.py` seam loop (`capture_boundaries` → block teacher/student →
  MSE) extends to a **cross-architecture** student. The one missing piece (a
  different-arch teacher target + a real trainable SSM student) works on 8GB.
- The next step is the **linker-specific** experiment: distill a small Mamba head
  whose target is a routing/handoff decision (not a layer's hidden state) — the
  switch-vs-state-transfer supervision fork.

## Reproduce
```bash
wsl -d Ubuntu
source ~/scrt-gpu-venv/bin/activate
cd /mnt/c/Users/atooz/Programming/ai-utils-memory/scrt-evolve
python3 bench/seam_distill/probe_env.py            # env + Mamba2 CUDA fwd/bwd
python3 bench/seam_distill/seam_distill_tinyllama.py
```
