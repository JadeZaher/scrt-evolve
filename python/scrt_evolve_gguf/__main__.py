"""
__main__.py — CLI entry point for scrt_evolve_gguf.

Usage:
    python -m scrt_evolve_gguf [options]

Run from the python/ directory (or with PYTHONPATH=<repo>/python) so that
scrt_evolve_train is importable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .export import VALID_QUANTS, export_gguf, find_llama_cpp


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scrt_evolve_gguf",
        description=(
            "Merge a LoRA adapter into a HuggingFace base model and export "
            "a quantized GGUF for use in LM Studio / llama.cpp.\n\n"
            "3-stage pipeline:\n"
            "  1. MERGE   — attach + merge LoRALinear adapters, save HF model dir\n"
            "  2. CONVERT — shell out to convert_hf_to_gguf.py -> f16 GGUF\n"
            "  3. QUANTIZE— shell out to llama-quantize -> final GGUF\n\n"
            "Final stdout line is JSON: "
            '{"out":"...", "quant":"...", "size_bytes":N, '
            '"base_model":"...", "adapter":"..."}'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "Path to the base HuggingFace model directory. "
            "If omitted, base_model_path is read from --adapter's adapter_config.json."
        ),
    )
    p.add_argument(
        "--adapter",
        metavar="DIR",
        default=None,
        help=(
            "Path to the adapter directory containing adapter.safetensors and "
            "adapter_config.json. Optional — omit for a base-only GGUF export."
        ),
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help=(
            "Output path for the final .gguf file. "
            "Defaults to <adapter_dir>/../model-<quant>.gguf, "
            "or ./model-<quant>.gguf if no adapter is given."
        ),
    )
    p.add_argument(
        "--quant",
        metavar="TYPE",
        default="Q4_K_M",
        choices=sorted(VALID_QUANTS),
        help=(
            "Quantization type. "
            "Choices: %(choices)s. "
            "Use 'f16' or 'none' to skip quantization and keep the f16 GGUF. "
            "Default: %(default)s."
        ),
    )
    p.add_argument(
        "--llama-cpp",
        metavar="DIR",
        default=None,
        dest="llama_cpp",
        help=(
            "Path to the llama.cpp checkout containing convert_hf_to_gguf.py. "
            "Auto-detected from $LLAMA_CPP, ~/.unsloth/llama.cpp, ~/llama.cpp, "
            "~/Documents/llama.cpp if not provided."
        ),
    )
    p.add_argument(
        "--keep-merged",
        action="store_true",
        default=False,
        help="Keep the intermediate _merged_hf/ HuggingFace directory after conversion.",
    )
    p.add_argument(
        "--keep-f16",
        action="store_true",
        default=False,
        help="Keep the intermediate f16 GGUF after quantization.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Resolve model path ---
    model_path: str | None = args.model

    if model_path is None:
        # Try to pull base_model_path from adapter_config.json
        if args.adapter is None:
            parser.error(
                "--model is required when --adapter is not provided "
                "(no adapter_config.json to infer base_model_path from)."
            )
        config_path = Path(args.adapter) / "adapter_config.json"
        if not config_path.exists():
            parser.error(
                f"--model not provided and adapter_config.json not found at "
                f"'{config_path}'. Provide --model explicitly."
            )
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            model_path = cfg.get("base_model_path")
        except Exception as exc:
            parser.error(f"Failed to read adapter_config.json: {exc}")
        if not model_path:
            parser.error(
                "base_model_path is missing or empty in adapter_config.json. "
                "Provide --model explicitly."
            )
        print(
            f"INFO: inferred --model from adapter_config.json: {model_path}",
            file=sys.stderr,
        )

    # --- Resolve output path ---
    out_path: str = args.out or ""
    if not out_path:
        quant_safe = args.quant.replace("_", "-").lower()
        if args.adapter:
            out_path = str(Path(args.adapter).parent / f"model-{quant_safe}.gguf")
        else:
            out_path = str(Path.cwd() / f"model-{quant_safe}.gguf")
        print(f"INFO: --out not provided, defaulting to '{out_path}'", file=sys.stderr)

    # --- Locate llama.cpp ---
    llama_cpp_dir = find_llama_cpp(args.llama_cpp)
    print(f"INFO: using llama.cpp at '{llama_cpp_dir}'", file=sys.stderr)

    # --- Run the pipeline ---
    export_gguf(
        model_path=model_path,
        adapter_dir=args.adapter,
        out_path=out_path,
        quant=args.quant,
        llama_cpp_dir=llama_cpp_dir,
        keep_merged=args.keep_merged,
        keep_f16=args.keep_f16,
    )


if __name__ == "__main__":
    main()
