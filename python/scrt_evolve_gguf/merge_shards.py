"""
merge_shards.py — combine the per-shard adapter files that fractional /
sharded training emits into the single `adapter.safetensors` +
`adapter_config.json` that the export/infer merge stage consumes.

The sharded trainer (scrt_evolve_train.shard) writes one file per shard:
    adapter-shard-000.safetensors / .json
    adapter-shard-001.safetensors / .json
    ...
Each weight key is GLOBAL-layer-indexed (e.g.
`model.layers.7.self_attn.q_proj.lora_A`), so the union of all shard state
dicts is order-independent and collision-free — exactly the single-file adapter
the rest of the pipeline expects.

This is the config-driven (`[export.merge_shards]`) replacement for the manual
merge step. Idempotent: if a single-file `adapter.safetensors` already exists
and no shard files match, it is a no-op.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file


def merge_shard_adapters(
    adapter_dir: str | Path,
    pattern: str = "adapter-shard-*.safetensors",
) -> dict[str, Any]:
    """Union per-shard adapter files in *adapter_dir* into a single
    `adapter.safetensors` + `adapter_config.json`.

    Returns a summary dict. Raises SystemExit on hard errors (no shards AND no
    existing single-file adapter).
    """
    ad = Path(adapter_dir)
    if not ad.is_dir():
        sys.exit(f"ERROR: adapter dir not found: {ad}")

    shard_files = sorted(ad.glob(pattern))
    single = ad / "adapter.safetensors"

    if not shard_files:
        # Nothing to merge — accept an already-single-file adapter as-is.
        if single.exists() and (ad / "adapter_config.json").exists():
            return {
                "merged": False,
                "reason": "single-file adapter already present; no shards matched",
                "adapter": str(single),
            }
        sys.exit(
            f"ERROR: no shard files matched '{pattern}' in {ad} and no existing "
            f"adapter.safetensors — nothing to merge."
        )

    merged: dict[str, Any] = {}
    rank = alpha = base = None
    targets: set[str] = set()
    per_shard: list[dict[str, Any]] = []

    for sf in shard_files:
        jf = sf.with_suffix(".json")
        if jf.exists():
            cfg = json.loads(jf.read_text(encoding="utf-8"))
            rank = cfg.get("rank", rank)
            alpha = cfg.get("alpha", alpha)
            base = cfg.get("base_model_path", base)
            for t in cfg.get("target_modules", []):
                targets.add(t)
        sd = load_file(str(sf))
        # Global-layer-indexed keys ⇒ union is collision-free; assert that.
        for k in sd:
            if k in merged:
                sys.exit(
                    f"ERROR: duplicate key '{k}' across shards — shard keys must "
                    f"be global-layer-indexed and disjoint."
                )
        merged.update(sd)
        per_shard.append({"file": sf.name, "tensors": len(sd)})

    if rank is None or alpha is None:
        sys.exit(
            "ERROR: could not read rank/alpha from any shard json "
            f"(*.json next to {pattern}). Cannot write adapter_config.json."
        )

    # Atomic single-file write.
    tmp = ad / "adapter.safetensors.tmp"
    save_file(merged, str(tmp))
    tmp.replace(single)

    cfg_out = {
        "rank": rank,
        "alpha": alpha,
        "target_modules": sorted(targets),
        "base_model_path": base,
        "format": "safetensors",
        "merged_from_shards": len(shard_files),
    }
    (ad / "adapter_config.json").write_text(
        json.dumps(cfg_out, indent=2), encoding="utf-8"
    )

    return {
        "merged": True,
        "shards": per_shard,
        "n_shards": len(shard_files),
        "n_tensors": len(merged),
        "adapter": str(single),
        "target_modules": sorted(targets),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m scrt_evolve_gguf.merge_shards",
        description="Union per-shard adapter files into one adapter.safetensors.",
    )
    ap.add_argument("adapter_dir", help="Directory containing adapter-shard-*.safetensors")
    ap.add_argument(
        "--pattern",
        default="adapter-shard-*.safetensors",
        help="Glob for the shard files. Default: %(default)s",
    )
    a = ap.parse_args()
    result = merge_shard_adapters(a.adapter_dir, a.pattern)
    print(json.dumps(result))
