"""Env probe for the cross-arch seam-distillation experiment.

Confirms the three things the full experiment depends on:
  1. CUDA torch is live (the Mamba backward segfaults on CPU torch).
  2. mamba-ssm's Mamba2 imports (2.3.2's __init__ eagerly imports Mamba3 which
     needs triton>=3.2; we fall back to the submodule path if the top-level fails).
  3. A real Mamba2 forward+backward runs on the GPU at TinyLlama's width (d=2048).

Run (WSL2):
  source ~/scrt-gpu-venv/bin/activate
  python3 bench/seam_distill/probe_env.py
"""
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "| capability", torch.cuda.get_device_capability(0))

Mamba2 = None
for path in ("mamba_ssm", "mamba_ssm.modules.mamba2"):
    try:
        mod = __import__(path, fromlist=["Mamba2"])
        Mamba2 = getattr(mod, "Mamba2")
        print(f"Mamba2 import OK via: {path}")
        break
    except Exception as e:  # noqa: BLE001 - we want the reason
        print(f"  import {path} failed: {repr(e)[:160]}")
if Mamba2 is None:
    sys.exit("FATAL: cannot import Mamba2 from mamba_ssm")

import transformers  # noqa: E402
print("transformers", transformers.__version__)

# Real fwd+bwd on CUDA at d_model=2048 (TinyLlama residual width).
d = 2048
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
m = Mamba2(d_model=d).to(dev).to(dt)
n_params = sum(p.numel() for p in m.parameters())
print(f"Mamba2(d_model={d}) params: {n_params:,} ({n_params * 2 / 1e6:.1f} MB bf16)")

x = torch.randn(1, 16, d, device=dev, dtype=dt, requires_grad=True)
y = m(x)
loss = y.float().pow(2).mean()
loss.backward()
print("fwd out", tuple(y.shape),
      "| input grad:", x.grad is not None,
      "| any param grad:", any(p.grad is not None for p in m.parameters()))
if dev == "cuda":
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
print("PROBE_OK")
