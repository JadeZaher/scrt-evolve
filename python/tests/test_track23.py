"""Track 23 unit tests — generic, no real model/GGUF required.

Run: PYTHONPATH=python <py> python/tests/test_track23.py
Covers: ArchSpec rule engine (name mapping, registry, unknown-arch), QAT STE
(forward quantizes / backward identity), auto-detect targets on a toy module.
"""
import sys

import torch
import torch.nn as nn

from scrt_evolve_dequant import archspec
from scrt_evolve_train.qat import fake_quantize, quant_bits, Calibrator, CalibConfig
from scrt_evolve_train.trainer import auto_detect_targets


def test_archspec_name_rules():
    spec = archspec.get("llama")
    assert spec is not None
    # Layer-indexed rule substitutes {n}.
    assert spec.map_tensor_name("blk.7.attn_q.weight") == "model.layers.7.self_attn.q_proj.weight"
    assert spec.map_tensor_name("blk.0.ffn_down.weight") == "model.layers.0.mlp.down_proj.weight"
    # Fixed (non-layer) rule.
    assert spec.map_tensor_name("token_embd.weight") == "model.embed_tokens.weight"
    # Unmatched name → None.
    assert spec.map_tensor_name("blk.0.totally_unknown.weight") is None
    # Dropped pattern.
    assert spec.is_dropped("rope_freqs.weight")
    print("OK archspec_name_rules")


def test_registry_unknown_arch():
    assert archspec.get("no_such_arch_xyz") is None
    assert "llama" in archspec.supported()
    print("OK registry_unknown_arch")


def test_qat_ste():
    torch.manual_seed(0)
    w = torch.randn(8, 32, requires_grad=True)
    q = fake_quantize(w, "Q4_K_M", group_size=32)
    assert (q.detach() - w.detach()).abs().max() > 0, "forward must quantize"
    q.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w)), "backward must be STE identity"
    assert quant_bits("Q4_K_M") == 4 and quant_bits("Q6_K") == 6 and quant_bits("Q8_0") == 8
    print("OK qat_ste")


def test_qat_calibration_bounded():
    cfg = CalibConfig(enabled=True, quant="Q4_K_M", group_size=32, calibrate_batches=3)
    cal = Calibrator(cfg=cfg)
    w = torch.randn(8, 32)
    # Calibrating for the first 3 ticks; scale frozen after.
    for _ in range(3):
        assert cal.still_calibrating()
        cal.observe("m", w)
        cal.tick()
    assert not cal.still_calibrating(), "calibration is bounded by calibrate_batches"
    assert cal.scale_for("m") is not None, "a frozen scale is available post-calibration"
    print("OK qat_calibration_bounded")


def test_auto_detect_targets():
    # A toy 'hybrid' model: most layers have a 'shared' proj; few have 'attn_q'.
    class Layer(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.shared = nn.Linear(16, 16)
            if attn:
                self.attn_q = nn.Linear(16, 16)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(attn=(i % 5 == 0)) for i in range(20)])
            self.lm_head = nn.Linear(16, 100)

    targets = auto_detect_targets(Model())
    # 'shared' is on every layer (20) → must rank above the rare 'attn_q' (4).
    assert "shared" in targets, targets
    assert targets.index("shared") < targets.index("attn_q") if "attn_q" in targets else True
    # lm_head is excluded.
    assert "lm_head" not in targets
    print("OK auto_detect_targets")


def test_auto_detect_excludes_ssm():
    # A hybrid model: a mamba block (in_proj/out_proj) + attention (q_proj).
    # in_proj/out_proj appear ONLY under .mamba → must be excluded (they segfault
    # the naive CPU SSM backward). q_proj (attention) must be kept.
    class Mamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(16, 32)
            self.out_proj = nn.Linear(32, 16)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba = Mamba()
            self.self_attn = Attn()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(8)])

    targets = auto_detect_targets(Model())
    assert "in_proj" not in targets, f"SSM in_proj must be excluded: {targets}"
    assert "out_proj" not in targets, f"SSM out_proj must be excluded: {targets}"
    assert "q_proj" in targets, f"attention q_proj must be kept: {targets}"
    print("OK auto_detect_excludes_ssm")


if __name__ == "__main__":
    test_archspec_name_rules()
    test_registry_unknown_arch()
    test_qat_ste()
    test_qat_calibration_bounded()
    test_auto_detect_targets()
    test_auto_detect_excludes_ssm()
    print("\nALL TRACK 23 PYTHON TESTS PASSED")
    sys.exit(0)
