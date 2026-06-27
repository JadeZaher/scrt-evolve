"""convert_teacher_safetensors.py — one-time legacy .bin → safetensors convert.

transformers (with torch < 2.6) refuses to load pytorch_model*.bin via
torch.load (CVE-2025-32434); it only loads safetensors. The seam-distill teacher
on this box (Wizard-Vicuna-7B) ships as .bin, so convert it ONCE to a safetensors
snapshot on fast native ext4. torch.load(weights_only=True) is safe to call
directly here — the restriction is a transformers-side guard, not torch itself.

Usage (WSL):
  python3 convert_teacher_safetensors.py <src_snapshot_dir> <dst_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

# Gather the .bin shards (single or sharded via the index).
index = src / "pytorch_model.bin.index.json"
if index.exists():
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted({src / f for f in weight_map.values()})
else:
    shards = [src / "pytorch_model.bin"]

state: dict[str, torch.Tensor] = {}
for sh in shards:
    print(f"loading {sh.name} ...", file=sys.stderr, flush=True)
    part = torch.load(str(sh), map_location="cpu", weights_only=True)
    for k, v in part.items():
        # bf16 to halve disk + match the capture dtype; clone to break any shared
        # storage (safetensors forbids overlapping tensors).
        state[k] = v.to(torch.bfloat16).contiguous().clone()
    del part

out = dst / "model.safetensors"
print(f"saving {len(state)} tensors -> {out}", file=sys.stderr, flush=True)
save_file(state, str(out), metadata={"format": "pt"})

# Copy the non-weight files transformers needs (config + tokenizer).
for name in (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
):
    f = src / name
    if f.exists():
        shutil.copy2(f, dst / name)
print("DONE", file=sys.stderr, flush=True)
