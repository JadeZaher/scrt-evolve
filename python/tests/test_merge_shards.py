"""Unit tests for scrt_evolve_gguf.merge_shards — the config-driven sharding
merge rule that unions per-shard adapter files into one adapter.safetensors.

Run: PYTHONPATH=python python python/tests/test_merge_shards.py
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scrt_evolve_gguf.merge_shards import merge_shard_adapters


def _write_shard(d: Path, idx: int, layer_keys: list[str]):
    sd = {}
    for lk in layer_keys:
        sd[f"{lk}.lora_A"] = torch.zeros(4, 8)
        sd[f"{lk}.lora_B"] = torch.zeros(8, 4)
    save_file(sd, str(d / f"adapter-shard-{idx:03d}.safetensors"))
    (d / f"adapter-shard-{idx:03d}.json").write_text(json.dumps({
        "rank": 4, "alpha": 8.0, "target_modules": ["q_proj"],
        "base_model_path": "/base", "shard": idx, "layer_offset": idx,
    }))


def test_merge_unions_disjoint_shards():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_shard(d, 0, ["model.layers.0.self_attn.q_proj"])
        _write_shard(d, 1, ["model.layers.1.self_attn.q_proj"])
        res = merge_shard_adapters(d)
        assert res["merged"] is True
        assert res["n_shards"] == 2
        assert res["n_tensors"] == 4  # 2 layers × (A+B)
        # single-file outputs exist
        assert (d / "adapter.safetensors").exists()
        cfg = json.loads((d / "adapter_config.json").read_text())
        assert cfg["rank"] == 4 and cfg["alpha"] == 8.0
        assert cfg["merged_from_shards"] == 2
        # union has both layers' keys
        merged = load_file(str(d / "adapter.safetensors"))
        assert "model.layers.0.self_attn.q_proj.lora_A" in merged
        assert "model.layers.1.self_attn.q_proj.lora_B" in merged
    print("OK merge_unions_disjoint_shards")


def test_merge_detects_duplicate_keys():
    # Two shards writing the SAME global key must be rejected (keys must be
    # global-layer-indexed and disjoint).
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_shard(d, 0, ["model.layers.0.self_attn.q_proj"])
        _write_shard(d, 1, ["model.layers.0.self_attn.q_proj"])  # collision
        try:
            merge_shard_adapters(d)
        except SystemExit as e:
            assert "duplicate key" in str(e)
            print("OK merge_detects_duplicate_keys")
            return
        raise AssertionError("expected SystemExit on duplicate keys")


def test_no_shards_accepts_existing_single_file():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        save_file({"x": torch.zeros(2, 2)}, str(d / "adapter.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps({"rank": 4, "alpha": 8.0}))
        res = merge_shard_adapters(d)
        assert res["merged"] is False
        print("OK no_shards_accepts_existing_single_file")


if __name__ == "__main__":
    test_merge_unions_disjoint_shards()
    test_merge_detects_duplicate_keys()
    test_no_shards_accepts_existing_single_file()
    print("\nALL MERGE-SHARDS PYTHON TESTS PASSED")
