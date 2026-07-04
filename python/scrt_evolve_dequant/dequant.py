"""
dequant.py — generic GGUF → HF safetensors converter (track 23).

Architecture-AGNOSTIC: reads the GGUF `general.architecture`, looks up an
`ArchSpec` in the registry (archspec.py), and applies its rule-based name/config
maps. No model/brand-specific logic here. Add a new architecture by registering
a spec — never by editing this file.

STREAMING: tensors are dequantized and written ONE AT A TIME (the user's
"dequantize in parts" idea), so peak memory is ~one tensor, not the whole model.
Output is a sharded HF model dir (`model-0000N-of-...safetensors` +
`model.safetensors.index.json`) that
`AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` loads.

Lossiness is honest: dequantizing a Q4 GGUF recovers the QUANTIZED weights
upcast to f16/f32, not the original full-precision weights. The output config is
stamped `_dequantized_from_gguf: true` so downstream knows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scrt_evolve_dequant import archspec


def _gguf():
    """Import the vendored gguf reader (llama.cpp gguf-py on PYTHONPATH)."""
    try:
        import gguf  # noqa: F401
        return gguf
    except ImportError as e:
        sys.exit(
            "ERROR: the `gguf` package is required to read GGUF files. It ships "
            "vendored in a llama.cpp checkout under `gguf-py/`. Add it to "
            "PYTHONPATH (the Rust `dequant`/`export-gguf` shims do this "
            "automatically), or `pip install gguf`.\n"
            f"import error: {e}"
        )


def _field_value(reader, key: str) -> Any:
    f = reader.get_field(key)
    if f is None:
        return None
    try:
        return f.contents()
    except Exception:
        return None


def _read_arch(reader) -> str:
    arch = _field_value(reader, "general.architecture")
    if not isinstance(arch, str) or not arch:
        sys.exit("ERROR: GGUF has no `general.architecture` metadata — cannot select an ArchSpec.")
    return arch


def build_hf_config(reader, arch: str, spec: archspec.ArchSpec) -> dict[str, Any]:
    """Reconstruct an HF config.json dict from GGUF metadata via the spec."""
    cfg: dict[str, Any] = {
        "model_type": spec.hf_model_type,
        "architectures": list(spec.hf_architectures),
        "torch_dtype": "float16",
        "_dequantized_from_gguf": True,
    }
    for ck in spec.config_keys:
        gkey = ck.gguf_key.replace("{arch}", arch)
        val = _field_value(reader, gkey)
        if val is None:
            continue
        if ck.transform is not None:
            val = ck.transform(val)
        # GGUF sometimes stores per-layer arrays (e.g. head_count_kv); collapse a
        # uniform array to a scalar, else keep the max (HF wants a scalar here).
        if isinstance(val, (list, tuple, np.ndarray)):
            uniq = set(int(x) for x in val if x is not None)
            val = (uniq.pop() if len(uniq) == 1 else max(int(x) for x in val))
        cfg[ck.hf_key] = val
    return cfg


def dequantize_to_hf(
    gguf_path: str,
    out_dir: str,
    dtype: str = "f16",
    shard_max_bytes: int = 2_000_000_000,
) -> dict[str, Any]:
    """Convert a GGUF to a sharded HF safetensors dir. Streaming. Returns a
    summary dict (also printed as the final stdout line by __main__)."""
    from safetensors.numpy import save_file

    gguf = _gguf()
    src = Path(gguf_path)
    if not src.exists():
        sys.exit(f"ERROR: GGUF not found: {gguf_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype in ("f16", "float16") else np.float32

    print(f"INFO: reading GGUF {src}", file=sys.stderr)
    reader = gguf.GGUFReader(str(src))
    arch = _read_arch(reader)
    spec = archspec.get(arch)
    if spec is None:
        sys.exit(
            f"ERROR: no ArchSpec registered for architecture '{arch}'.\n"
            f"Supported: {archspec.supported()}\n"
            f"Register an ArchSpec for '{arch}' in scrt_evolve_dequant/archspec.py "
            "(rule-based GGUF→HF name + config maps) — the converter itself is generic."
        )
    print(f"INFO: architecture '{arch}' → HF model_type '{spec.hf_model_type}'", file=sys.stderr)

    # --- Stream tensors: dequant one at a time, accumulate into shards. ---
    index_map: dict[str, str] = {}
    unmapped: list[str] = []
    shard_idx = 1
    shard: dict[str, np.ndarray] = {}
    shard_bytes = 0
    n_written = 0
    shard_paths: list[Path] = []

    def flush_shard() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        name = f"model-{shard_idx:05d}.safetensors"  # renamed with total at end
        path = out / name
        save_file(shard, str(path))
        shard_paths.append(path)
        for k in shard:
            index_map[k] = name
        shard = {}
        shard_bytes = 0
        shard_idx += 1

    for tensor in reader.tensors:
        gname = str(tensor.name)
        if spec.is_dropped(gname):
            continue
        hf_name = spec.map_tensor_name(gname)
        if hf_name is None:
            unmapped.append(gname)
            continue
        # Dequantize to float, then cast to the requested storage dtype.
        deq = gguf.dequantize(tensor.data, tensor.tensor_type).astype(np_dtype)
        shard[hf_name] = deq
        shard_bytes += deq.nbytes
        n_written += 1
        if shard_bytes >= shard_max_bytes:
            flush_shard()
    flush_shard()

    if n_written == 0:
        sys.exit(
            f"ERROR: no tensors mapped for arch '{arch}'. The ArchSpec name rules "
            f"matched nothing — the GGUF layout may differ from the registered spec."
        )

    # Rename shards with the final total + build the index.
    total = len(shard_paths)
    final_index: dict[str, str] = {}
    renamed: dict[Path, str] = {}
    for i, p in enumerate(shard_paths, 1):
        final_name = f"model-{i:05d}-of-{total:05d}.safetensors"
        renamed[p] = final_name
        p.rename(out / final_name)
    for k, old_name in index_map.items():
        # old_name was model-{idx:05d}.safetensors; find its final name by order.
        idx = int(old_name.split("-")[1].split(".")[0])
        final_index[k] = f"model-{idx:05d}-of-{total:05d}.safetensors"

    index = {"metadata": {"total_size": 0}, "weight_map": final_index}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # --- Config ---
    config = build_hf_config(reader, arch, spec)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if unmapped:
        print(
            f"WARN: {len(unmapped)} GGUF tensor(s) had no name rule and were skipped "
            f"(arch '{arch}' spec may be incomplete for this model):",
            file=sys.stderr,
        )
        for u in unmapped[:20]:
            print(f"  unmapped: {u}", file=sys.stderr)
        if len(unmapped) > 20:
            print(f"  … and {len(unmapped) - 20} more", file=sys.stderr)

    summary = {
        "out": str(out.resolve()),
        "arch": arch,
        "hf_model_type": spec.hf_model_type,
        "n_tensors": n_written,
        "n_unmapped": len(unmapped),
        "shards": total,
        "dtype": dtype,
    }
    return summary
