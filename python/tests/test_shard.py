"""Unit tests for sharded / fractional training (shard.py) and the router /
dtype guards added to trainer.py. CPU-only, model-free — they exercise the
generic layer-discovery, shard planning, target hygiene, and LoRA dtype
behavior without loading any real model.

Run: PYTHONPATH=python python python/tests/test_shard.py
"""

import torch
import torch.nn as nn

from scrt_evolve_train.shard import (
    block_lr_scale,
    build_projection,
    discover_groups,
    distill_loss,
    find_decoder_layers,
    lr_at_step,
    plan_seam_map,
    plan_shards,
    _seam_indices,
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


# ─────────────── Cross-model seam distillation (track 29 v1.1) ───────────────


def test_seam_map_stride_maps_deeper_teacher_to_student_blocks():
    # Student: 4 layers in 2 blocks; teacher: 8 layers. Stride maps each student
    # boundary to the proportional teacher seam (b * L_t / L_s).
    blocks = plan_shards(4, None, 2)  # [(0,2),(2,4)]
    smap = plan_seam_map(blocks, 8, "stride")
    assert smap == {2: [4], 4: [8]}, smap
    # The final student boundary always lands on the teacher's final seam.
    assert smap[4] == [8]
    # Per-layer student (block_size=1) over a 22→student... sanity: monotonic.
    blocks2 = plan_shards(4, None, 1)  # [(0,1),(1,2),(2,3),(3,4)]
    smap2 = plan_seam_map(blocks2, 8, "stride")
    assert smap2 == {1: [2], 2: [4], 3: [6], 4: [8]}, smap2
    print("OK seam_map_stride")


def test_seam_map_block_avg_spans_teacher_layers():
    # block_avg maps a student block to the SPAN of teacher layers it covers, so
    # Phase A can average them into a smoother target.
    blocks = plan_shards(2, None, 1)  # [(0,1),(1,2)]
    smap = plan_seam_map(blocks, 6, "block_avg")
    # student boundary 1 -> teacher seam 3; block covers teacher seams 1..3.
    assert smap[1] == [1, 2, 3], smap
    assert smap[2] == [4, 5, 6], smap
    # _seam_indices collects the full distinct set Phase A must capture.
    assert _seam_indices(smap) == [1, 2, 3, 4, 5, 6]
    print("OK seam_map_block_avg")


def test_build_projection_identity_when_equal_width():
    # Equal widths (or mode='none') ⇒ identity (no width bridge, nothing to drop).
    p = build_projection(16, 16, "auto")
    assert isinstance(p, nn.Identity)
    x = torch.randn(2, 5, 16)
    assert torch.equal(p(x), x)
    p_none = build_projection(8, 12, "none")
    assert isinstance(p_none, nn.Identity)
    print("OK build_projection_identity")


def test_build_projection_lifts_student_to_teacher_width():
    # Differing widths ⇒ a linear lift student(d_s) → teacher(d_t), so the loss
    # is computed in teacher space (the target is never down-sampled).
    p = build_projection(8, 12, "auto")
    assert isinstance(p, nn.Linear)
    assert p.weight.shape == (12, 8)
    out = p(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 12)
    print("OK build_projection_lift")


def test_distill_loss_variants_and_zero_at_match():
    pred = torch.randn(3, 7, 12)
    # Perfect match ⇒ mse 0 and cosine 0 (1 - 1).
    assert distill_loss(pred, pred.clone(), "mse").item() < 1e-6
    assert distill_loss(pred, pred.clone(), "cosine").item() < 1e-6
    assert distill_loss(pred, pred.clone(), "cosine_mse").item() < 1e-6
    # cosine_mse = cosine + mse (both non-negative); strictly ≥ each alone here.
    tgt = torch.randn(3, 7, 12)
    cm = distill_loss(pred, tgt, "cosine_mse").item()
    c = distill_loss(pred, tgt, "cosine").item()
    m = distill_loss(pred, tgt, "mse").item()
    assert abs(cm - (c + m)) < 1e-4, (cm, c, m)
    print("OK distill_loss_variants")


def test_distill_training_step_reduces_loss_on_toy_pair():
    # End-to-end mechanism on a TINY cross-width pair (no model load, CPU): a
    # student "block" (Linear 8→8) + read-out projection (8→12) learns to match a
    # fixed teacher target in 12-dim. Loss must fall — proves the projection +
    # loss + optimizer wire up and the cross-width gradient flows.
    torch.manual_seed(0)
    student_block = nn.Linear(8, 8)
    proj = build_projection(8, 12, "auto")
    x = torch.randn(4, 6, 8)
    teacher_target = torch.randn(4, 6, 12)  # a distinct-width "teacher seam"
    opt = torch.optim.AdamW(
        list(student_block.parameters()) + list(proj.parameters()), lr=1e-2
    )
    first = None
    last = 0.0
    for _ in range(200):
        pred = proj(student_block(x))
        loss = distill_loss(pred, teacher_target, "cosine_mse")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first, f"distill loss did not fall: {first:.4f} -> {last:.4f}"
    assert last < first * 0.7, f"weak fit: {first:.4f} -> {last:.4f}"
    print(f"OK distill_training_step ({first:.3f} -> {last:.3f})")


def test_block_lr_scale_gentler_for_larger_targets():
    # Reference block (target_rms == ref) keeps the full base LR.
    assert block_lr_scale(10.0, 10.0) == 1.0
    # A deeper block with 4× the magnitude gets ~1/4 the LR (clamped at lo).
    assert abs(block_lr_scale(10.0, 40.0) - 0.25) < 1e-9
    # Even larger ⇒ clamped to the lo floor, never zero.
    assert block_lr_scale(10.0, 1000.0) == 0.25
    # A smaller-magnitude block never exceeds the base LR (clamped at hi).
    assert block_lr_scale(10.0, 2.0) == 1.0
    # Degenerate target ⇒ safe hi default.
    assert block_lr_scale(10.0, 0.0) == 1.0
    print("OK block_lr_scale")


def test_lr_at_step_warmup_then_cosine_decay():
    base, total, warmup = 1e-3, 100, 10
    # Warmup ramps up linearly and ends at ~base.
    assert lr_at_step(base, 0, total, warmup) < lr_at_step(base, 5, total, warmup)
    assert abs(lr_at_step(base, warmup - 1, total, warmup) - base) < 1e-12
    # Just after warmup it is near the peak; by the end it decays to ~0.
    assert lr_at_step(base, warmup, total, warmup) > 0.9 * base
    assert lr_at_step(base, total - 1, total, warmup) < 0.05 * base
    # No warmup ⇒ pure cosine from the first step.
    assert abs(lr_at_step(base, 0, total, 0) - base) < 1e-6
    print("OK lr_at_step")


if __name__ == "__main__":
    test_plan_shards_block_size_and_count()
    test_find_decoder_layers_generic()
    test_auto_detect_excludes_router()
    test_attach_lora_skips_router()
    test_lora_dtype_matches_wrapped_layer()
    test_lora_disabled_returns_base()
    test_discover_groups_skips_linear_free_children()
    test_set_group_student_isolates_one_group()
    test_seam_map_stride_maps_deeper_teacher_to_student_blocks()
    test_seam_map_block_avg_spans_teacher_layers()
    test_build_projection_identity_when_equal_width()
    test_build_projection_lifts_student_to_teacher_width()
    test_distill_loss_variants_and_zero_at_match()
    test_distill_training_step_reduces_loss_on_toy_pair()
    test_block_lr_scale_gentler_for_larger_targets()
    test_lr_at_step_warmup_then_cosine_decay()
    print("\nALL SHARD/FRACTIONAL PYTHON TESTS PASSED")
