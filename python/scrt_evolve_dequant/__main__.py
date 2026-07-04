"""
__main__.py — CLI for scrt_evolve_dequant (track 23).

    python -m scrt_evolve_dequant --gguf <path> --out <dir> [--dtype f16] [--tokenizer <hf-dir-or-id>]

Generic GGUF → HF safetensors conversion via the ArchSpec registry. Emits a JSON
summary as the final stdout line (parsed by the Rust `dequant` shim). A
`--tokenizer` HF dir/id is copied in as the fallback tokenizer (GGUF tokenizer
extraction is a documented seam).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from scrt_evolve_dequant import archspec
from scrt_evolve_dequant.dequant import dequantize_to_hf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scrt_evolve_dequant",
        description="Generic GGUF → HF safetensors converter (architecture registry).",
    )
    # Not required so `--list-arch` works alone; validated in main().
    p.add_argument("--gguf", help="Source .gguf path.")
    p.add_argument("--out", help="Output HF model dir.")
    p.add_argument("--dtype", default="f16", choices=["f16", "f32"], help="Storage dtype.")
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer dir to copy in as the fallback (recommended: the HF "
        "repo/dir for the same base). GGUF tokenizer extraction is a seam.",
    )
    p.add_argument("--list-arch", action="store_true", help="List registered architectures and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_arch:
        print(json.dumps({"supported": archspec.supported()}))
        return 0
    if not args.gguf or not args.out:
        print("ERROR: --gguf and --out are required (unless --list-arch).", file=sys.stderr)
        return 2

    try:
        summary = dequantize_to_hf(args.gguf, args.out, dtype=args.dtype)
    except SystemExit:
        raise
    except Exception as e:  # pragma: no cover - surfaced clearly to the Rust side
        print(f"ERROR: dequant failed: {e}", file=sys.stderr)
        return 1

    # Copy a fallback tokenizer if provided.
    if args.tokenizer:
        tok_src = Path(args.tokenizer)
        if tok_src.is_dir():
            for fn in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                       "special_tokens_map.json", "vocab.json", "merges.txt"):
                f = tok_src / fn
                if f.exists():
                    shutil.copy2(f, Path(args.out) / fn)
            summary["tokenizer"] = "copied"
        else:
            print(
                f"WARN: --tokenizer '{args.tokenizer}' is not a local dir; skipping "
                "copy (pass an HF tokenizer dir).",
                file=sys.stderr,
            )

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
