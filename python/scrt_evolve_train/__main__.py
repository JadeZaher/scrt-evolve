"""
__main__.py — CLI entry point for scrt_evolve_train.

Usage:
    python -m scrt_evolve_train --dataset path/to/dataset.jsonl --model path/to/model [options]
"""

import argparse
import sys

from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scrt_evolve_train",
        description=(
            "scrt-evolve real-model LoRA training (transformers-based, no peft). "
            "Reads a scrt-evolve dataset.jsonl, loads a HuggingFace causal-LM, "
            "attaches LoRA adapters, trains with prompt-masked CE, saves adapter."
        ),
    )

    p.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to dataset.jsonl (scrt-evolve format). Required.",
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Local path to a HuggingFace model snapshot (e.g. TinyLlama-1.1B). Required.",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Output directory for the adapter. Default: <dataset_dir>/adapter.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=40,
        metavar="N",
        help="Number of gradient steps. Default: 40.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size. Default: 1 (safe for CPU).",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        metavar="N",
        help="Maximum sequence length (tokens). Default: 256.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        metavar="F",
        help="AdamW learning rate. Default: 2e-4.",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=16,
        metavar="N",
        help="LoRA rank. Default: 16.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=32.0,
        metavar="F",
        help="LoRA alpha (scaling = alpha/rank). Default: 32.0.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        metavar="F",
        help="LoRA dropout probability. Default: 0.05.",
    )
    p.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        metavar="LIST",
        help=(
            "Comma-separated list of nn.Linear leaf names to wrap with LoRA. "
            "Default: 'q_proj,v_proj' (Llama/TinyLlama). "
            "Example for GPT-2: 'c_attn,c_proj'."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Random seed for reproducibility. Default: 0.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=5,
        metavar="N",
        help="Print loss to stderr every N steps. Default: 5.",
    )

    return p


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
