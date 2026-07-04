"""
qat.py — quantization-aware training primitives (track 23).

Simulates the quantization the model will be DEPLOYED under (e.g. Q4_K_M GGUF)
during LoRA training, so the adapter learns to compensate for quant error and
the exported GGUF degrades less.

Two pieces:
  - `fake_quantize` — a group-wise affine quant→dequant with a straight-through
    estimator (STE): the forward quantizes+dequantizes the effective weight; the
    backward is the identity (gradients flow as if no quantization happened).
  - `Calibrator` — a bounded pass that observes weights/activations and picks
    per-group scales (absmax) instead of static defaults.

CPU-safe (float32 sim), bounded (calibration batch count is explicit), and
deterministic given a seed (no RNG in the quant path).

This is QAT-lite by design: a group-wise affine simulation of the K-quant
family, NOT a bit-exact reproduction of llama.cpp's Q4_K_M block format. It
captures the dominant effect (limited mantissa within per-group scales) which is
what the adapter needs to adapt to; bit-exactness is out of scope (and would not
change the gradient signal materially).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# Number of representable levels for the quant families we simulate. Q4 → 4-bit
# (16 levels), Q6 → 6-bit, Q8 → 8-bit. We map a GGUF quant name onto a bit width.
_QUANT_BITS = {
    "Q2_K": 2,
    "Q3_K_S": 3, "Q3_K_M": 3, "Q3_K_L": 3,
    "Q4_0": 4, "Q4_K_M": 4, "Q4_K_S": 4, "Q4_K": 4,
    "Q5_K_M": 5, "Q5_K_S": 5, "Q5_K": 5,
    "Q6_K": 6,
    "Q8_0": 8,
}

# Default group size for the per-group affine sim (Q4_K uses 32-wide super-block
# sub-groups; 32 is a faithful default).
DEFAULT_GROUP_SIZE = 32


def quant_bits(quant: str) -> int:
    """Bit width for a GGUF quant name. Unknown → 4 (Q4-class), with no error so
    a novel quant name degrades to a reasonable default."""
    return _QUANT_BITS.get(quant.strip(), 4)


class _FakeQuantSTE(torch.autograd.Function):
    """Straight-through fake-quantize: forward quant→dequant, backward identity."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, scale: torch.Tensor, levels: int) -> torch.Tensor:
        # Symmetric affine: q = round(w / scale) clamped to +-(levels//2),
        # dequant = q * scale. scale is per-group (broadcast over the group dim).
        half = levels // 2
        q = torch.clamp(torch.round(w / scale), -half, half - 1)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass the gradient straight through w; none to scale/levels.
        return grad_output, None, None


def fake_quantize(
    w: torch.Tensor,
    quant: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Group-wise affine fake-quant of a 2-D weight `w` ([out, in]).

    Splits the LAST dim into groups of `group_size`, computes a per-group absmax
    scale (or uses a provided calibrated `scale`), and quant→dequants with an STE
    so gradients flow to `w` unchanged. Returns a tensor the same shape as `w`.
    """
    levels = 2 ** quant_bits(quant)
    if w.dim() != 2:
        # Only matmul weights are fake-quantized; pass others through.
        return w
    out_f, in_f = w.shape
    g = min(group_size, in_f) if in_f > 0 else group_size
    if g <= 0 or in_f % g != 0:
        # Fall back to a single group over the row when not evenly divisible.
        g = in_f if in_f > 0 else 1
    n_groups = max(1, in_f // g)

    wv = w.view(out_f, n_groups, g)
    if scale is None:
        absmax = wv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = absmax / (levels // 2)
    else:
        scale = scale.view(out_f, n_groups, 1)

    q = _FakeQuantSTE.apply(wv, scale, levels)
    return q.view(out_f, in_f)


@dataclass
class CalibConfig:
    """QAT settings resolved from CLI/config."""
    enabled: bool = False
    quant: str = "Q4_K_M"
    group_size: int = DEFAULT_GROUP_SIZE
    calibrate_batches: int = 0  # 0 ⇒ static absmax (no calibration pass)


@dataclass
class Calibrator:
    """Collects per-module calibrated scales over a bounded number of batches.

    Calibration here = observe the effective weight's running absmax across the
    first N training batches and freeze a per-group scale from it, instead of
    recomputing absmax every step. Bounded by `cfg.calibrate_batches`.
    """
    cfg: CalibConfig
    seen_batches: int = 0
    scales: dict[str, torch.Tensor] = field(default_factory=dict)

    def active(self) -> bool:
        return self.cfg.enabled and self.cfg.calibrate_batches > 0

    def still_calibrating(self) -> bool:
        return self.active() and self.seen_batches < self.cfg.calibrate_batches

    def observe(self, name: str, w: torch.Tensor) -> None:
        """Update the running per-group absmax for module `name` from weight `w`."""
        if not self.still_calibrating() or w.dim() != 2:
            return
        out_f, in_f = w.shape
        g = self.cfg.group_size
        if g <= 0 or in_f % g != 0:
            g = in_f if in_f > 0 else 1
        n_groups = max(1, in_f // g)
        levels = 2 ** quant_bits(self.cfg.quant)
        absmax = w.detach().view(out_f, n_groups, g).abs().amax(dim=-1, keepdim=True)
        cur = absmax / (levels // 2)
        if name in self.scales:
            self.scales[name] = torch.maximum(self.scales[name], cur)
        else:
            self.scales[name] = cur

    def tick(self) -> None:
        if self.active():
            self.seen_batches += 1

    def scale_for(self, name: str) -> torch.Tensor | None:
        """The frozen calibrated scale for a module, or None (use dynamic absmax)."""
        if self.cfg.calibrate_batches > 0 and not self.still_calibrating():
            return self.scales.get(name)
        return None
