"""
__main__.py — CLI entry point for scrt_evolve_infer.

Usage:
    python -m scrt_evolve_infer --model <path> --prompt "..." [options]
    python -m scrt_evolve_infer --adapter <dir> --prompt "..." --ab [options]

Run `python -m scrt_evolve_infer --help` for full flag reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scrt_evolve_infer",
        description=(
            "scrt-evolve inference: run a HuggingFace causal-LM with an optional "
            "LoRA adapter (produced by scrt_evolve_train). Supports A/B comparison "
            "between base model and adapter-patched model. CPU-safe, float32."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base model only
  python -m scrt_evolve_infer --model /path/to/model --prompt "What is scrt?"

  # Adapter only (model path read from adapter_config.json)
  python -m scrt_evolve_infer --adapter ./adapter --prompt "What is scrt?"

  # A/B: base vs adapter side-by-side
  python -m scrt_evolve_infer --adapter ./adapter --prompt "What is scrt?" --ab

  # From a file of prompts with sampling
  python -m scrt_evolve_infer --adapter ./adapter --prompts-file prompts.txt \\
      --ab --temperature 0.7 --max-new-tokens 200
""",
    )

    # --- Model / adapter ---
    p.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "Path to the base HuggingFace model snapshot (e.g. TinyLlama-1.1B). "
            "If omitted, base_model_path is read from adapter_config.json under "
            "--adapter (which must then be provided)."
        ),
    )
    p.add_argument(
        "--adapter",
        metavar="DIR",
        default=None,
        help=(
            "Directory containing adapter.safetensors + adapter_config.json "
            "(output of scrt_evolve_train). Optional — if omitted, only the base "
            "model is run."
        ),
    )

    # --- Prompts ---
    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        metavar="TEXT",
        default=None,
        help="Single prompt string to generate from.",
    )
    prompt_group.add_argument(
        "--prompts-file",
        metavar="FILE",
        default=None,
        help="Path to a newline-delimited file of prompts (one per line, blank lines skipped).",
    )

    # --- Generation ---
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        metavar="N",
        help="Maximum number of new tokens to generate per prompt. Default: 128.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        metavar="F",
        help=(
            "Sampling temperature. 0 (default) = greedy; >0 = sampling with that "
            "temperature."
        ),
    )

    # --- Mode flags ---
    p.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help=(
            "Wrap the prompt in the tokenizer's chat template "
            "(via tokenizer.apply_chat_template) before generating. "
            "Default OFF — prompts are fed plain, matching the training format."
        ),
    )
    p.add_argument(
        "--ab",
        action="store_true",
        default=False,
        help=(
            "A/B mode: when --adapter is given, run BOTH the base model and the "
            "adapter-patched model for each prompt and display them side by side. "
            "Without --ab, only the adapter result is shown when --adapter is set."
        ),
    )

    return p


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        return [args.prompt]
    path = Path(args.prompts_file)
    if not path.exists():
        sys.exit(f"ERROR: prompts file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [l.strip() for l in lines if l.strip()]
    if not prompts:
        sys.exit(f"ERROR: no prompts found in {path}")
    return prompts


def resolve_model_path(args: argparse.Namespace) -> str:
    """
    Return the base model path: either --model or base_model_path from
    adapter_config.json. Exits with a clear message if neither is resolvable.
    """
    if args.model:
        return args.model
    if args.adapter:
        config_path = Path(args.adapter) / "adapter_config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            mp = cfg.get("base_model_path")
            if mp:
                print(
                    f"INFO: base model path from adapter_config.json: {mp}",
                    file=sys.stderr,
                )
                return mp
    sys.exit(
        "ERROR: --model is required unless --adapter is given and adapter_config.json "
        "contains base_model_path."
    )


def print_result_block(
    prompt: str,
    base_text: str | None,
    adapter_text: str | None,
    show_base: bool,
    show_adapter: bool,
) -> None:
    """Print one prompt's results in a clear labelled block."""
    print(f"\n{'='*3} PROMPT: {prompt} {'='*3}")
    if show_base and base_text is not None:
        print(f"[base]    {base_text}")
    if show_adapter and adapter_text is not None:
        print(f"[adapter] {adapter_text}")


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    # Validate: need at least one of --model or --adapter.
    if args.model is None and args.adapter is None:
        parser.error("provide at least --model or --adapter (or both)")

    # Determine what we need to run.
    run_adapter = args.adapter is not None
    run_base = (not run_adapter) or args.ab  # base-only OR A/B mode

    model_path = resolve_model_path(args)
    prompts = load_prompts(args)

    # Late import so --help is instant (no torch import cost).
    from .infer import apply_adapter, generate, load_base_model

    # Load base model once.
    model_base, tokenizer = load_base_model(model_path)

    # Optionally build adapter model (a separate copy with LoRA applied).
    model_adapter = None
    if run_adapter:
        import copy
        print("INFO: building adapter model (deep-copying base)...", file=sys.stderr)
        model_adapter = copy.deepcopy(model_base)
        apply_adapter(model_adapter, args.adapter)
        model_adapter.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        chat=args.chat,
    )

    for prompt in prompts:
        base_text: str | None = None
        adapter_text: str | None = None

        if run_base:
            base_text = generate(model_base, tokenizer, prompt, **gen_kwargs)

        if run_adapter and model_adapter is not None:
            adapter_text = generate(model_adapter, tokenizer, prompt, **gen_kwargs)

        print_result_block(
            prompt=prompt,
            base_text=base_text,
            adapter_text=adapter_text,
            show_base=run_base,
            show_adapter=run_adapter,
        )


if __name__ == "__main__":
    main()
