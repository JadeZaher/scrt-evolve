"""
__main__.py — CLI entry point for scrt_evolve_score.

Usage:
    python -m scrt_evolve_score --model <path> --probe <probe.jsonl> [--adapter <dir>]

Loads the model (+ optional adapter), scores it against the probe set, and prints
a ScoreReport JSON object as the final stdout line (parsed by the Rust eval
harness). Progress/info go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys

from scrt_evolve_score.score import sample_ab, score_probe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scrt_evolve_score",
        description="Score a HuggingFace causal-LM (+ optional LoRA adapter) "
        "against a held-out probe set; emit a ScoreReport JSON line.",
    )
    p.add_argument("--model", metavar="PATH", required=True, help="Base HF model dir.")
    p.add_argument("--probe", metavar="PATH", required=True, help="probe.jsonl path.")
    p.add_argument("--adapter", metavar="DIR", default=None,
                   help="Optional LoRA adapter dir (adapter.safetensors + config).")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max tokens to generate per correctness probe (default 64).")
    p.add_argument("--metrics", default="correctness,perplexity,mean_exit_depth",
                   help="Comma-separated metrics to compute.")
    # Track 32: A/B sampling mode for the LLM-judge regression gate. Emits
    # [{prompt, before, after}] (base vs base+adapter) instead of a ScoreReport.
    p.add_argument("--ab", action="store_true",
                   help="A/B mode: emit (prompt, before, after) triples for the "
                        "regression gate (requires --adapter).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.ab:
        if not args.adapter:
            print("ERROR: --ab requires --adapter (the candidate to compare vs base)",
                  file=sys.stderr)
            return 1
        try:
            triples = sample_ab(
                model_path=args.model,
                probe_path=args.probe,
                adapter_dir=args.adapter,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"ERROR: A/B sampling failed: {e}", file=sys.stderr)
            return 1
        # Final stdout line: the JSON triples array (parsed by the Rust gate).
        print(json.dumps(triples))
        return 0

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    try:
        report = score_probe(
            model_path=args.model,
            probe_path=args.probe,
            adapter_dir=args.adapter,
            max_new_tokens=args.max_new_tokens,
            metrics=metrics,
        )
    except Exception as e:  # surface a clear error; non-zero exit for the Rust side
        print(f"ERROR: scoring failed: {e}", file=sys.stderr)
        return 1
    # Final stdout line: the JSON ScoreReport.
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
