"""lora_to_gguf.py — safetensors flat-LoRA → GGUF-LoRA converter for llama.cpp --lora.

Reads adapter_config.json + adapter.safetensors (save_adapter() contract) and
writes a GGUF-LoRA file. The pure safetensors→GGUF name mapping is importable and
unit-testable WITHOUT the `gguf` package; only the actual writer needs it.

See scrt_evolve_infer/AGENTS.md for the naming contract + why gguf is guarded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scrt_evolve_dequant import archspec

# save_adapter() emits HF-style names; strip these to recover the base weight.
_LORA_A = ".lora_A"
_LORA_B = ".lora_B"

# GGUF-LoRA suffixes llama.cpp expects on each base tensor name.
_GGUF_A = ".lora_a"
_GGUF_B = ".lora_b"

_GGUF_LORA_ARCH_ID = "arch"

_PIP_HINT = "install it from llama.cpp: pip install gguf"


# The regex fragment ArchSpec uses to capture a layer index in a GGUF name.
_LAYER_CAPTURE = r"(?P<n>\d+)"


def _rule_gguf_literal(pattern: str, n: str | None) -> str:
    """Turn a NameRule GGUF *pattern* back into a concrete tensor name.

    The builtin patterns are literal regexes (escaped dots) plus at most one
    named layer-index group. We unescape the literals and substitute *n*.
    """
    literal = pattern.replace(_LAYER_CAPTURE, n if n is not None else "")
    return literal.replace("\\.", ".").replace("\\", "")


def _hf_to_gguf_base(hf_weight_name: str, spec: "archspec.ArchSpec") -> str | None:
    """Invert an ArchSpec name rule: HF '...q_proj.weight' → GGUF 'blk.N.attn_q.weight'.

    Uses each rule's HF *template* as the inverse matcher (with `{n}` → index
    capture), then rebuilds the concrete GGUF name from the rule's *pattern*.
    """
    for rule in spec.name_rules:
        if "{n}" in rule.template:
            inv = "^" + re.escape(rule.template).replace(re.escape("{n}"), r"(?P<n>\d+)") + "$"
            m = re.match(inv, hf_weight_name)
            if m is None:
                continue
            return _rule_gguf_literal(rule.pattern, m.group("n"))
        if rule.template == hf_weight_name:
            return _rule_gguf_literal(rule.pattern, None)
    return None


def map_lora_name(safetensors_key: str, arch: str) -> str:
    """
    Map one save_adapter() safetensors key to its GGUF-LoRA tensor name.

    'model.layers.0.self_attn.q_proj.lora_A' → 'blk.0.attn_q.weight.lora_a'.
    Pure (no gguf dependency); raises ValueError on an unmappable key so the
    caller can fail loudly instead of silently dropping a tensor.
    """
    if safetensors_key.endswith(_LORA_A):
        base, suffix = safetensors_key[: -len(_LORA_A)], _GGUF_A
    elif safetensors_key.endswith(_LORA_B):
        base, suffix = safetensors_key[: -len(_LORA_B)], _GGUF_B
    else:
        raise ValueError(
            f"map_lora_name: key {safetensors_key!r} is neither *.lora_A nor *.lora_B"
        )

    spec = archspec.get(arch)
    if spec is None:
        raise ValueError(
            f"map_lora_name: unknown architecture {arch!r}; "
            f"supported: {archspec.supported()}"
        )

    hf_weight = f"{base}.weight"
    gguf_base = _hf_to_gguf_base(hf_weight, spec)
    if gguf_base is None:
        raise ValueError(
            f"map_lora_name: no GGUF base name for HF weight {hf_weight!r} "
            f"under arch {arch!r} — adapter targets a module the arch rules don't cover"
        )
    # gguf_base ends in '.weight'; llama.cpp keys the LoRA on the base tensor name.
    return f"{gguf_base}{suffix}"


def map_lora_state_names(
    safetensors_keys: list[str], arch: str
) -> dict[str, str]:
    """Map every safetensors LoRA key → GGUF-LoRA name (pure, gguf-free)."""
    return {k: map_lora_name(k, arch) for k in safetensors_keys}


def _resolve_arch(adapter_dir: Path, arch: str | None) -> str:
    """Pick the architecture: explicit flag wins, else adapter_config, else 'llama'."""
    if arch:
        return arch
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("gguf_arch", "arch", "architecture"):
            if cfg.get(key):
                return str(cfg[key])
    return "llama"


def convert(
    adapter_dir: str | Path,
    out_path: str | Path,
    arch: str | None = None,
) -> Path:
    """
    Convert adapter.safetensors in *adapter_dir* to a GGUF-LoRA at *out_path*.

    Requires the `gguf` package (llama.cpp's writer); raises ImportError with a
    pip hint if it is unavailable. The name-mapping itself is done via the pure
    map_lora_name(), so correctness is testable without gguf installed.
    """
    try:
        import gguf  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(f"gguf package not installed — {_PIP_HINT}") from exc

    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    out_path = Path(out_path)
    weights_path = adapter_dir / "adapter.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter.safetensors not found in {adapter_dir}")

    resolved_arch = _resolve_arch(adapter_dir, arch)
    state = load_file(str(weights_path))
    name_map = map_lora_state_names(list(state.keys()), resolved_arch)

    writer = gguf.GGUFWriter(str(out_path), arch=_GGUF_LORA_ARCH_ID)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    for st_key, gguf_name in name_map.items():
        tensor = state[st_key].to(dtype=state[st_key].dtype).cpu().contiguous()
        writer.add_tensor(gguf_name, tensor.numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        f"INFO: wrote GGUF-LoRA {out_path} ({len(name_map)} tensors, arch={resolved_arch})",
        file=sys.stderr,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scrt_evolve_infer.lora_to_gguf",
        description="Convert a flat safetensors LoRA adapter to a GGUF-LoRA for llama.cpp --lora.",
    )
    parser.add_argument("--adapter", required=True, help="adapter directory (adapter.safetensors + config)")
    parser.add_argument("--out", required=True, help="output GGUF-LoRA path")
    parser.add_argument("--arch", default=None, help="GGUF architecture id (default: from config or 'llama')")
    args = parser.parse_args(argv)

    try:
        convert(args.adapter, args.out, arch=args.arch)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
