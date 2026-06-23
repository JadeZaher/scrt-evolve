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
            "Comma-separated nn.Linear leaf names to wrap with LoRA, or 'auto' to "
            "auto-detect generically (recommended for hybrid/MoE arches like "
            "granitemoehybrid where q_proj/v_proj don't cover the compute path). "
            "Default: 'q_proj,v_proj' (Llama/TinyLlama)."
        ),
    )
    # --- QAT (track 23): quantization-aware training ---
    p.add_argument(
        "--qat",
        default=None,
        metavar="QUANT",
        help=(
            "Enable quantization-aware training simulating this GGUF quant during "
            "the LoRA forward (e.g. 'Q4_K_M'), so the adapter compensates for the "
            "deployment quantization. Absent ⇒ plain LoRA."
        ),
    )
    p.add_argument(
        "--qat-group-size",
        type=int,
        default=32,
        metavar="N",
        help="QAT per-group affine group size. Default: 32 (Q4_K-faithful).",
    )
    p.add_argument(
        "--qat-calibrate",
        type=int,
        default=0,
        metavar="N",
        help=(
            "QAT calibration: observe per-group scales over the first N batches, "
            "then freeze them. 0 ⇒ dynamic per-step absmax (no calibration pass)."
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

    # --- Hardware (config-driven GPU usage) ---
    p.add_argument(
        "--device",
        default="auto",
        metavar="DEV",
        help="Accelerator: 'auto' (cuda if available, else cpu), 'cuda', or 'cpu'. Default: auto.",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        metavar="DT",
        help=(
            "Compute dtype: 'auto' (bf16 on cuda, fp32 on cpu), 'float32', "
            "'bfloat16', or 'float16'. Default: auto."
        ),
    )

    # --- Sharded / fractional training (decentralized; bounds VRAM to 1 block) ---
    p.add_argument(
        "--shard-mode",
        action="store_true",
        help=(
            "Train by contiguous layer-block shards via block-local distillation, "
            "keeping only ONE block resident on the accelerator at a time. Lets a "
            "large model be LoRA-trained on a small GPU (and shards run in parallel "
            "across machines). Absent => dense training (default)."
        ),
    )
    p.add_argument(
        "--shards",
        type=int,
        default=None,
        metavar="N",
        help="Number of contiguous layer-block shards to split the model into. "
        "Mutually exclusive with --block-size (block-size wins if both set).",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        metavar="N",
        help="Layers per shard block. Overrides --shards. The hard VRAM knob: "
        "smaller => less peak VRAM, more streaming.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        metavar="N",
        help="Train ONLY this shard index (0-based) and exit. For decentralized "
        "runs: one process/machine per shard. Absent => all shards in sequence.",
    )
    p.add_argument(
        "--calib-batches",
        type=int,
        default=8,
        metavar="N",
        help="Number of token batches to distill each shard over (boundary "
        "activations captured from these). Default: 8.",
    )
    p.add_argument(
        "--granularity",
        default="block",
        choices=["block", "module"],
        help=(
            "Sharded-mode training granularity. 'block' (default): train all of a "
            "layer-block's LoRA together. 'module': PER-MODULE sub-layer floor — "
            "train one submodule group (attention / MoE / MLP) at a time within "
            "each layer, against the layer's frozen-output teacher. Lowest VRAM, "
            "most passes (trade time for memory). Pair with --block-size 1."
        ),
    )
    p.add_argument(
        "--objective",
        default="distill",
        choices=["distill", "end_task"],
        help=(
            "Sharded learning objective. 'distill' (default): block-local MSE vs "
            "the frozen block's own output — a representation/regularization signal "
            "(does NOT impart new knowledge). 'end_task': the FINAL shard trains "
            "real cross-entropy against the completion tokens via the LM head (the "
            "actual knowledge signal — use this to teach the model new content). "
            "Non-final shards still distill under end_task."
        ),
    )

    return p


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if getattr(args, "shard_mode", False):
        from .shard import train_sharded

        train_sharded(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
