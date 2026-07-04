"""Track 33 Phase 4 tests — adapter hot-swap (reload_adapter) + safetensors→GGUF-LoRA name mapping.

Run: PYTHONPATH=python <py> python/tests/test_track33_swap.py
The name-mapping tests are gguf-free (pure). The end-to-end GGUF write is skipped
when the `gguf` package is absent (pytest.importorskip). The reload_adapter tests
need torch/transformers-free scaffolding (toy nn.Module only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure name-mapping (no gguf, no torch needed beyond archspec import)
# ---------------------------------------------------------------------------

def test_map_lora_name_roundtrips_save_adapter_contract():
    from scrt_evolve_infer.lora_to_gguf import map_lora_name

    # save_adapter() emits HF-style '<module-path>.lora_A/.lora_B'.
    assert (
        map_lora_name("model.layers.0.self_attn.q_proj.lora_A", "llama")
        == "blk.0.attn_q.weight.lora_a"
    )
    assert (
        map_lora_name("model.layers.7.self_attn.q_proj.lora_B", "llama")
        == "blk.7.attn_q.weight.lora_b"
    )
    assert (
        map_lora_name("model.layers.3.mlp.down_proj.lora_A", "llama")
        == "blk.3.ffn_down.weight.lora_a"
    )
    print("OK map_lora_name_roundtrips")


def test_map_lora_state_names_bulk():
    from scrt_evolve_infer.lora_to_gguf import map_lora_state_names

    keys = [
        "model.layers.0.self_attn.q_proj.lora_A",
        "model.layers.0.self_attn.q_proj.lora_B",
        "model.layers.1.mlp.up_proj.lora_A",
        "model.layers.1.mlp.up_proj.lora_B",
    ]
    m = map_lora_state_names(keys, "llama")
    assert m["model.layers.0.self_attn.q_proj.lora_A"] == "blk.0.attn_q.weight.lora_a"
    assert m["model.layers.1.mlp.up_proj.lora_B"] == "blk.1.ffn_up.weight.lora_b"
    assert len(m) == 4
    print("OK map_lora_state_names_bulk")


def test_map_lora_name_rejects_bad_suffix():
    from scrt_evolve_infer.lora_to_gguf import map_lora_name

    with pytest.raises(ValueError):
        map_lora_name("model.layers.0.self_attn.q_proj.weight", "llama")
    print("OK map_lora_name_rejects_bad_suffix")


def test_map_lora_name_rejects_unknown_arch():
    from scrt_evolve_infer.lora_to_gguf import map_lora_name

    with pytest.raises(ValueError):
        map_lora_name("model.layers.0.self_attn.q_proj.lora_A", "no_such_arch")
    print("OK map_lora_name_rejects_unknown_arch")


def test_map_lora_name_rejects_uncovered_module():
    from scrt_evolve_infer.lora_to_gguf import map_lora_name

    # A module the llama arch rules don't cover → loud failure, no silent drop.
    with pytest.raises(ValueError):
        map_lora_name("model.layers.0.mamba.in_proj.lora_A", "llama")
    print("OK map_lora_name_rejects_uncovered_module")


# ---------------------------------------------------------------------------
# End-to-end GGUF write — SKIPPED when gguf is not installed.
# ---------------------------------------------------------------------------

def test_convert_writes_gguf(tmp_path):
    pytest.importorskip("gguf")
    torch = pytest.importorskip("torch")
    from scrt_evolve_train.trainer import save_adapter
    from scrt_evolve_infer.lora_to_gguf import convert

    model = _toy_model(torch)
    _init_lora(torch, model)
    save_adapter(
        model, tmp_path, rank=4, alpha=8.0,
        target_modules=["q_proj"], base_model_path="fake",
    )
    out = tmp_path / "adapter.gguf"
    convert(tmp_path, out, arch="llama")
    assert out.exists() and out.stat().st_size > 0
    print("OK convert_writes_gguf")


# ---------------------------------------------------------------------------
# reload_adapter — hot-swap over an already-loaded model (needs torch only).
# ---------------------------------------------------------------------------

def _toy_model(torch):
    """A minimal llama-shaped model: model.layers.N.self_attn.q_proj (nn.Linear)."""
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(2)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    return Model()


def _init_lora(torch, model):
    """Attach LoRALinear wrappers and give lora_B a non-zero value (save emits it)."""
    from scrt_evolve_train.trainer import attach_lora

    n = attach_lora(model, target_modules=["q_proj"], rank=4, alpha=8.0, dropout=0.0)
    assert n == 2, n
    from scrt_evolve_train.trainer import LoRALinear

    with torch.no_grad():
        for _, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                mod.lora_B.copy_(torch.randn_like(mod.lora_B))


def _lora_snapshot(torch, model):
    from scrt_evolve_train.trainer import LoRALinear

    snap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            snap[name] = (mod.lora_A.detach().clone(), mod.lora_B.detach().clone())
    return snap


def test_reload_swaps_weights_without_reinstantiating_base(tmp_path):
    torch = pytest.importorskip("torch")
    from scrt_evolve_train.trainer import save_adapter
    from scrt_evolve_infer.infer import apply_adapter, reload_adapter

    torch.manual_seed(0)

    # Build a base and apply adapter v1.
    model = _toy_model(torch)
    src_v1 = _toy_model(torch)
    _init_lora(torch, src_v1)
    dir_v1 = tmp_path / "v1"
    save_adapter(src_v1, dir_v1, rank=4, alpha=8.0, target_modules=["q_proj"], base_model_path="fake")
    apply_adapter(model, dir_v1)

    base_id = id(model)
    inner_id = id(model.model)
    before = _lora_snapshot(torch, model)

    # A different adapter v2.
    src_v2 = _toy_model(torch)
    _init_lora(torch, src_v2)
    with torch.no_grad():
        for _, mod in src_v2.named_modules():
            if hasattr(mod, "lora_A"):
                mod.lora_A.copy_(torch.randn_like(mod.lora_A) + 5.0)
    dir_v2 = tmp_path / "v2"
    save_adapter(src_v2, dir_v2, rank=4, alpha=8.0, target_modules=["q_proj"], base_model_path="fake")

    reload_adapter(model, dir_v2)

    # Base object identity preserved — no re-instantiation.
    assert id(model) == base_id
    assert id(model.model) == inner_id

    after = _lora_snapshot(torch, model)
    assert set(after) == set(before)
    changed = any(
        not torch.allclose(before[k][0], after[k][0]) for k in before
    )
    assert changed, "reload_adapter must overwrite lora weights"
    print("OK reload_swaps_weights")


def test_reload_missing_tensor_raises_before_mutation(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from scrt_evolve_train.trainer import save_adapter
    from scrt_evolve_infer.infer import apply_adapter, reload_adapter

    model = _toy_model(torch)
    src = _toy_model(torch)
    _init_lora(torch, src)
    good = tmp_path / "good"
    save_adapter(src, good, rank=4, alpha=8.0, target_modules=["q_proj"], base_model_path="fake")
    apply_adapter(model, good)
    before = _lora_snapshot(torch, model)

    # A partial adapter missing one module's tensors.
    from safetensors.torch import load_file

    state = load_file(str(good / "adapter.safetensors"))
    partial = {k: v for k, v in state.items() if "layers.1" not in k}
    bad = tmp_path / "bad"
    bad.mkdir()
    save_file(partial, str(bad / "adapter.safetensors"))

    with pytest.raises(RuntimeError):
        reload_adapter(model, bad)

    # Model must be untouched (all-or-nothing).
    after = _lora_snapshot(torch, model)
    for k in before:
        assert torch.allclose(before[k][0], after[k][0])
        assert torch.allclose(before[k][1], after[k][1])
    print("OK reload_missing_tensor_raises")


def test_reload_without_prior_apply_raises(tmp_path):
    torch = pytest.importorskip("torch")
    from scrt_evolve_train.trainer import save_adapter
    from scrt_evolve_infer.infer import reload_adapter

    model = _toy_model(torch)  # no LoRALinear wrappers yet
    src = _toy_model(torch)
    _init_lora(torch, src)
    d = tmp_path / "a"
    save_adapter(src, d, rank=4, alpha=8.0, target_modules=["q_proj"], base_model_path="fake")

    with pytest.raises(RuntimeError):
        reload_adapter(model, d)
    print("OK reload_without_prior_apply_raises")


if __name__ == "__main__":
    # Pure mapping tests run without torch/gguf.
    test_map_lora_name_roundtrips_save_adapter_contract()
    test_map_lora_state_names_bulk()
    test_map_lora_name_rejects_bad_suffix()
    test_map_lora_name_rejects_unknown_arch()
    test_map_lora_name_rejects_uncovered_module()
    print("\nPURE MAPPING TESTS PASSED (run torch/gguf tests via pytest)")
    sys.exit(0)
