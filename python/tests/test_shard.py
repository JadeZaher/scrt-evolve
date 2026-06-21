"""Unit tests for sharded / fractional training (shard.py) and the router /
dtype guards added to trainer.py. CPU-only, model-free — they exercise the
generic layer-discovery, shard planning, target hygiene, and LoRA dtype
behavior without loading any real model.

Run: PYTHONPATH=python python python/tests/test_shard.py
"""

import torch
import torch.nn as nn

from scrt_evolve_train.shard import (
    discover_groups,
    find_decoder_layers,
    plan_shards,
    _set_group_student,
)
from scrt_evolve_train.trainer import (
    LoRALinear,
    attach_lora,
    auto_detect_targets,
)


def test_plan_shards_block_size_and_count():
    # block_size wins over shards and covers every layer with no gaps/overlap.
    assert plan_shards(40, None, 8) == [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]
    assert plan_shards(40, 5, None) == [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]
    # uneven split: last block is the remainder, still covers all layers.
    sh = plan_shards(10, None, 4)
    assert sh == [(0, 4), (4, 8), (8, 10)]
    assert sh[0][0] == 0 and sh[-1][1] == 10
    # neither set ⇒ one dense shard.
    assert plan_shards(7, None, None) == [(0, 7)]
    print("OK plan_shards")


def test_find_decoder_layers_generic():
    # The longest ModuleList of structured children is the decoder stack,
    # regardless of attribute path (here nested under .model.layers).
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(6)])
            self.embed = nn.Embedding(10, 8)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    layers, name = find_decoder_layers(Model())
    assert len(layers) == 6, f"expected 6 layers, got {len(layers)}"
    assert name == "model.layers", f"unexpected stack name {name}"
    print("OK find_decoder_layers")


def test_auto_detect_excludes_router():
    # A MoE-ish layer: real content projections + a router/gate linear named
    # 'layer' (a poor generic name). The router must be excluded; the content
    # projections kept.
    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_linear = nn.Linear(16, 32)
            self.output_linear = nn.Linear(32, 16)
            self.router = nn.Module()
            self.router.layer = nn.Linear(16, 8)  # gate classifier

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_sparse_moe = MoE()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(4)])

    targets = auto_detect_targets(Model())
    assert "layer" not in targets, f"router 'layer' must be excluded: {targets}"
    assert "input_linear" in targets and "output_linear" in targets, targets
    print("OK auto_detect_excludes_router")


def test_attach_lora_skips_router():
    # Even if 'layer' is explicitly requested, attach_lora must not wrap a
    # module living under a router/gate path.
    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Module()
            self.router.layer = nn.Linear(16, 8)
            self.input_linear = nn.Linear(16, 32)

    moe = MoE()
    n = attach_lora(moe, ["layer", "input_linear"], rank=4, alpha=8, dropout=0.0)
    # input_linear wrapped (1), router.layer skipped → exactly 1.
    assert n == 1, f"expected 1 adapter (router skipped), got {n}"
    assert isinstance(moe.input_linear, LoRALinear)
    assert isinstance(moe.router.layer, nn.Linear)  # untouched
    print("OK attach_lora_skips_router")


def test_lora_dtype_matches_wrapped_layer():
    # LoRA params adopt the wrapped layer's dtype so a bf16 block never hits a
    # dtype-mismatch matmul. (CPU bf16 matmul is supported for this shape.)
    base = nn.Linear(16, 16).to(torch.bfloat16)
    lora = LoRALinear(base, rank=4, alpha=8, dropout=0.0)
    assert lora.lora_A.dtype == torch.bfloat16
    assert lora.lora_B.dtype == torch.bfloat16
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    out = lora(x)  # must not raise a dtype mismatch
    assert out.dtype == torch.bfloat16
    print("OK lora_dtype_matches_wrapped_layer")


def test_lora_disabled_returns_base():
    # With lora_disabled the module reproduces the frozen base exactly (the
    # teacher path for block-local distillation).
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8, dropout=0.0)
    # perturb LoRA so a difference would show if it leaked.
    with torch.no_grad():
        lora.lora_B.add_(1.0)
    x = torch.randn(3, 8)
    lora.lora_disabled = True
    assert torch.allclose(lora(x), base(x)), "disabled LoRA must equal base"
    lora.lora_disabled = False
    assert not torch.allclose(lora(x), base(x)), "enabled LoRA must differ from base"
    print("OK lora_disabled_returns_base")


def test_discover_groups_skips_linear_free_children():
    # A layer with two linear-bearing groups + a layernorm (no linear) → only
    # the two groups are discovered (generic: "has a Linear", not by name).
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(8, 16)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(8)  # no Linear → skipped
            self.self_attn = Attn()
            self.mlp = MLP()

    groups = dict(discover_groups(Layer()))
    assert set(groups.keys()) == {"self_attn", "mlp"}, groups.keys()
    print("OK discover_groups_skips_linear_free_children")


def test_set_group_student_isolates_one_group():
    # Wrapping both groups with LoRA, _set_group_student should enable ONLY the
    # active group's LoRA (lora_disabled False) and disable the rest.
    from scrt_evolve_train.trainer import attach_lora

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(8, 8)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = MLP()

    layer = Layer()
    attach_lora(layer, ["q_proj", "up"], rank=4, alpha=8, dropout=0.0)
    groups = dict(discover_groups(layer))

    _set_group_student(layer, groups["self_attn"])
    assert layer.self_attn.q_proj.lora_disabled is False, "active group must be enabled"
    assert layer.mlp.up.lora_disabled is True, "inactive group must be disabled"

    _set_group_student(layer, None)  # pure teacher: all disabled
    assert layer.self_attn.q_proj.lora_disabled is True
    assert layer.mlp.up.lora_disabled is True
    print("OK set_group_student_isolates_one_group")


if __name__ == "__main__":
    test_plan_shards_block_size_and_count()
    test_find_decoder_layers_generic()
    test_auto_detect_excludes_router()
    test_attach_lora_skips_router()
    test_lora_dtype_matches_wrapped_layer()
    test_lora_disabled_returns_base()
    test_discover_groups_skips_linear_free_children()
    test_set_group_student_isolates_one_group()
    print("\nALL SHARD/FRACTIONAL PYTHON TESTS PASSED")
