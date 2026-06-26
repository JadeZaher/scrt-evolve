"""Cross-arch seam-distillation de-risk experiment (canonical version).

QUESTION: can a single Mamba2 (SSM) block, synthesized from scratch, learn to
reproduce a TinyLlama *transformer* decoder layer's contribution (the map
in_L -> out_L)? This is the capability that gates evolve's Mamba "linker" head
for the Branch-Train-Merge P2P fabric (and, later, Mamba-fying branch bodies).

If yes, evolve can SYNTHESIZE an SSM block by distillation against a teacher's
per-layer boundary activations -- exactly MOHAWK Stage-2 hidden-state alignment
(arXiv 2408.10189), the mechanism shard.py already implements as
self-distillation (the only missing piece being a different-architecture
student, which this builds).

--- What the iteration taught (kept here as the record) --------------------
v1  full-output MSE, bf16     -> SSM COLLAPSED TO IDENTITY. out_L = in_L + small
                                 delta, so full-output MSE is minimized by
                                 driving the mixer to ~0 (1.03x better than
                                 predicting out=in; delta-cos 0.34).
   FIX: target the DELTA (out-in) directly, and capture in fp32 (out~=in in
        bf16 loses the delta to catastrophic cancellation).
v2  delta target, bf16 params -> still stalled: AdamW updates for the small
                                 delta (mean|delta|~0.05) round away below bf16
                                 resolution. train delta-cos stuck ~0.42.
   FIX: fp32 student params (master weights).
v3  fp32 params, 16 seqs      -> LEARNS: train delta-cos 0.88, full-cos 0.98.
                                 But overfits (val delta-cos 0.51) -- 3k tokens
                                 for a 25M mixer.
   FIX: more calibration data.
v4  fp32, 256 seqs (65k tok)  -> val delta-cos 0.645, train/val gap 0.37->0.09.
                                 Generalizes; purely data-limited.
This run scales to 512 seqs for the third point on the data-scaling curve.

STUDENT: residual Mamba block  out = x + Mamba2(RMSNorm(x))  -- the mixer
predicts the layer's delta; the residual carries identity for free.

METRICS (held-out val): delta-cos = cos(mixer(norm(x)), out-x) is load-bearing
(full-output cosine is a near-trivial bar since the residual stream is
self-similar across one layer). delta-relMSE = ||pred-tgt||/||tgt||.

Run (WSL2):
  source ~/scrt-gpu-venv/bin/activate
  python3 bench/seam_distill/seam_distill_tinyllama.py
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- config -----------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = ("/mnt/c/Users/atooz/.cache/huggingface/hub/"
              "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
              "fe8a4ea1ffedaf415f4da2f062534de366a451e6")
LAYER = 11
SEQ_LEN = 256
N_SEQS = 512
N_VAL = 64
STEPS = 8000
LR = 1e-3
SEED = 0
REPO = "/mnt/c/Users/atooz/Programming/ai-utils-memory/scrt-evolve"

torch.manual_seed(SEED)
assert torch.cuda.is_available(), "need CUDA — the Mamba backward segfaults on CPU torch"
dev = torch.device("cuda")
bf16 = torch.bfloat16

from mamba_ssm import Mamba2  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def load_corpus_text() -> str:
    skip = ("/.git", "/target", "/node_modules", "/.venv", "/__pycache__", "/.mpg", "/work")
    files = []
    for dp, _, fns in os.walk(REPO):
        if any(s in dp.replace("\\", "/") for s in skip):
            continue
        for fn in fns:
            if fn.endswith((".md", ".py", ".toml", ".rs", ".txt")):
                files.append(os.path.join(dp, fn))
    chunks = []
    for f in sorted(files)[:400]:
        try:
            chunks.append(open(f, encoding="utf-8", errors="ignore").read())
        except OSError:
            pass
    return "\n\n".join(chunks)


print("loading tokenizer + corpus ...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
ids = tok(load_corpus_text(), return_tensors="pt").input_ids[0]
need = N_SEQS * SEQ_LEN
avail = ids.numel()
tiled = avail < need
if tiled:
    ids = ids.repeat((need // avail) + 1)
seqs = ids[:need].view(N_SEQS, SEQ_LEN)
print(f"corpus tokens: available={avail}, need={need}, tiled={tiled} | "
      f"{N_SEQS} seqs ({N_SEQS - N_VAL} train / {N_VAL} val)", file=sys.stderr)

# ------------------------- capture teacher seam (fp32, on CPU) -------------------------
print(f"loading {MODEL_ID} (frozen teacher) ...", file=sys.stderr)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=bf16, local_files_only=True
).to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
d_model = model.config.hidden_size

ins, outs = [], []
with torch.no_grad():
    for i in range(N_SEQS):
        o = model(seqs[i:i + 1].to(dev), output_hidden_states=True, use_cache=False)
        hs = o.hidden_states
        ins.append(hs[LAYER].float().cpu())       # fp32, parked on CPU (VRAM-safe at scale)
        outs.append(hs[LAYER + 1].float().cpu())
IN = torch.cat(ins, 0)     # [N, S, D] fp32 CPU
OUT = torch.cat(outs, 0)
DELTA = OUT - IN           # fp32 target (computed before any bf16 cast -> no cancellation)
del model, ins, outs
torch.cuda.empty_cache()
print(f"seam at layer {LAYER}: IN/OUT {tuple(IN.shape)} fp32 | "
      f"mean|delta|={DELTA.abs().mean():.4f} mean|in|={IN.abs().mean():.4f}", file=sys.stderr)

tr = list(range(N_SEQS - N_VAL))
va = list(range(N_SEQS - N_VAL, N_SEQS))


# ------------------------- student -------------------------
class SeamStudent(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mixer = Mamba2(d_model=d)

    def delta(self, x):  # fp32 in/out; the layer's contribution
        return self.mixer(self.norm(x))


student = SeamStudent(d_model).to(dev)  # fp32 master weights (bf16 stalled in v2)
n_student = sum(p.numel() for p in student.parameters())
print(f"student params: {n_student:,} ({n_student * 4 / 1e6:.1f} MB fp32)", file=sys.stderr)


@torch.no_grad()
def metrics(idxs):
    fmse = fcos = dcos = drel = 0.0
    for j in idxs:
        x = IN[j:j + 1].to(dev)
        y = OUT[j:j + 1].to(dev)
        td = DELTA[j:j + 1].to(dev)
        pd = student.delta(x)
        pf = x + pd
        fmse += F.mse_loss(pf, y).item()
        fcos += F.cosine_similarity(pf.reshape(-1, d_model), y.reshape(-1, d_model), dim=-1).mean().item()
        a, b = pd.reshape(-1, d_model), td.reshape(-1, d_model)
        dcos += F.cosine_similarity(a, b, dim=-1).mean().item()
        drel += (a - b).norm().item() / (b.norm().item() + 1e-9)
    n = len(idxs)
    return fmse / n, fcos / n, dcos / n, drel / n


init = metrics(va)
print(f"untrained val: delta-cos={init[2]:.3f}", file=sys.stderr)

# ------------------------- train (delta target, fp32, cosine LR) -------------------------
opt = torch.optim.AdamW(student.parameters(), lr=LR)
t0 = time.time()
for step in range(STEPS):
    for g in opt.param_groups:
        g["lr"] = 0.5 * LR * (1 + math.cos(math.pi * step / STEPS))
    j = tr[step % len(tr)]
    pd = student.delta(IN[j:j + 1].to(dev))
    loss = F.mse_loss(pd, DELTA[j:j + 1].to(dev))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 1000 == 0 or step == 0:
        print(f"step {step + 1}/{STEPS} train_delta_mse={loss.item():.6f}", file=sys.stderr)
elapsed = time.time() - t0

tr_m, va_m = metrics(tr), metrics(va)

print("\n================= RESULTS (fp32 params, delta target, scaled data) =================")
print(f"layer {LAYER} of {MODEL_ID}  |  student delta = Mamba2(RMSNorm(x))  ({n_student/1e6:.1f}M params)")
print(f"data: {N_SEQS - N_VAL} train / {N_VAL} val seqs x {SEQ_LEN} tok (~{(N_SEQS*SEQ_LEN)//1000}k tokens, tiled={tiled})")
print(f"trained {STEPS} steps in {elapsed:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1e6:.0f} MB\n")
print("                         full-MSE   full-cos   DELTA-cos   delta-relMSE")
print(f"  untrained (val):       {init[0]:8.5f}   {init[1]:7.4f}   {init[2]:8.4f}   {init[3]:9.4f}")
print(f"  trained   (train):     {tr_m[0]:8.5f}   {tr_m[1]:7.4f}   {tr_m[2]:8.4f}   {tr_m[3]:9.4f}")
print(f"  trained   (val):       {va_m[0]:8.5f}   {va_m[1]:7.4f}   {va_m[2]:8.4f}   {va_m[3]:9.4f}\n")
print(f"  train/val delta-cos gap: {tr_m[2] - va_m[2]:.3f}  (small gap => generalizing, not memorizing)")
ok = (va_m[2] > 0.85) and (va_m[3] < 0.4)
strong = (va_m[2] > 0.95) and (va_m[3] < 0.25)
verdict = "STRONG PASS" if strong else ("PASS" if ok else "PARTIAL (capability shown, data-limited)")
print(f"  VERDICT: {verdict}")
print(f"    val delta-cos={va_m[2]:.3f}, val full-cos={va_m[1]:.3f}, val delta-relMSE={va_m[3]:.3f}")
print("====================================================================================")
