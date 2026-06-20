"""
export.py — 3-stage merge + convert + quantize pipeline.

Stage 1 (MERGE, Python):
    Load base HF model, optionally apply a LoRA adapter via the same
    logic as scrt_evolve_infer.infer.apply_adapter, call
    LoRALinear.merge_and_unload() on every wrapper, swap the plain
    nn.Linear back in, then save a full HF model dir.

Stage 2 (CONVERT, subprocess):
    Shell out to <llama_cpp_dir>/convert_hf_to_gguf.py to produce an
    intermediate f16 GGUF.

Stage 3 (QUANTIZE, subprocess):
    Shell out to llama-quantize(.exe) to quantize to the requested type,
    or skip and rename the f16 GGUF when quant is "f16"/"none".

Final stdout line is a JSON summary parseable by the Rust CLI:
    {"out": "<abs>", "quant": "...", "size_bytes": N, "base_model": "...", "adapter": "..."}

Progress and info go to stderr.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse verbatim from scrt_evolve_train — same class, same tensor layout.
from scrt_evolve_train.trainer import LoRALinear, attach_lora


# ---------------------------------------------------------------------------
# Quant type validation
# ---------------------------------------------------------------------------

VALID_QUANTS = {
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "f16", "F16",
    "none",
}

# Quants that bypass llama-quantize (the f16 GGUF is the final output).
F16_PASSTHROUGH = {"f16", "F16", "none"}


# ---------------------------------------------------------------------------
# llama.cpp auto-detection
# ---------------------------------------------------------------------------

def find_llama_cpp(explicit: str | None) -> str:
    """
    Locate a llama.cpp checkout that contains convert_hf_to_gguf.py.

    Search order:
      1. explicit argument (if provided)
      2. $LLAMA_CPP environment variable
      3. ~/.unsloth/llama.cpp
      4. ~/llama.cpp
      5. ~/Documents/llama.cpp

    Returns the first valid directory path (as str).
    Raises SystemExit with a clear message listing every location tried.
    """
    SENTINEL = "convert_hf_to_gguf.py"
    tried: list[str] = []

    def _valid(p: str) -> bool:
        candidate = Path(p) / SENTINEL
        tried.append(p)
        return candidate.is_file()

    # 1. Explicit override
    if explicit:
        if _valid(explicit):
            return explicit
        sys.exit(
            f"ERROR: --llama-cpp '{explicit}' does not contain {SENTINEL}.\n"
            f"Tried: {tried}"
        )

    # 2. Environment variable
    env_val = os.environ.get("LLAMA_CPP", "")
    if env_val and _valid(env_val):
        return env_val

    # 3-5. Well-known paths
    home = Path.home()
    candidates = [
        str(home / ".unsloth" / "llama.cpp"),
        str(home / "llama.cpp"),
        str(home / "Documents" / "llama.cpp"),
    ]
    for c in candidates:
        if _valid(c):
            return c

    sys.exit(
        "ERROR: could not find a llama.cpp checkout containing "
        f"{SENTINEL}.\n"
        "Tried (in order):\n"
        + "\n".join(f"  {t}" for t in tried)
        + "\n\nFix options:\n"
        "  --llama-cpp <dir>          pass the path explicitly\n"
        "  export LLAMA_CPP=<dir>     set the environment variable\n"
        "  git clone https://github.com/ggerganov/llama.cpp ~/.unsloth/llama.cpp"
    )


# ---------------------------------------------------------------------------
# Helper: locate llama-quantize binary
# ---------------------------------------------------------------------------

def _find_quantize_exe(llama_cpp_dir: str) -> str:
    """
    Look for llama-quantize(.exe) under *llama_cpp_dir* in the standard
    CMake build output locations.  Returns the path as str or raises
    SystemExit with a build hint.
    """
    base = Path(llama_cpp_dir)
    candidates = [
        base / "build" / "bin" / "Release" / "llama-quantize.exe",
        base / "build" / "bin" / "Release" / "llama-quantize",
        base / "build" / "bin" / "llama-quantize.exe",
        base / "build" / "bin" / "llama-quantize",
        base / "llama-quantize.exe",
        base / "llama-quantize",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    sys.exit(
        "ERROR: llama-quantize binary not found under "
        f"'{llama_cpp_dir}'.\n"
        "Checked:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\n\nBuild llama.cpp first:\n"
        "  cd " + llama_cpp_dir + "\n"
        "  cmake -B build && cmake --build build --config Release -j\n"
        "Then re-run this command."
    )


# ---------------------------------------------------------------------------
# Stage 1 — MERGE
# ---------------------------------------------------------------------------

def _apply_and_merge_adapter(model: nn.Module, adapter_dir: Path) -> None:
    """
    Apply the LoRA adapter from *adapter_dir* to *model* in-place, then
    merge every LoRALinear wrapper back into its underlying nn.Linear and
    swap the plain Linear back into the model graph.

    This is the same apply-adapter logic as scrt_evolve_infer.infer.apply_adapter,
    followed by merge_and_unload() on every wrapper and an in-place replacement
    of the LoRALinear node with the merged nn.Linear.
    """
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter.safetensors"

    if not config_path.exists():
        sys.exit(f"ERROR: adapter_config.json not found in '{adapter_dir}'")
    if not weights_path.exists():
        sys.exit(f"ERROR: adapter.safetensors not found in '{adapter_dir}'")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rank: int = cfg["rank"]
    alpha: float = float(cfg["alpha"])
    target_modules: list[str] = cfg["target_modules"]

    print(
        f"INFO: applying adapter rank={rank} alpha={alpha} "
        f"target_modules={target_modules}",
        file=sys.stderr,
    )

    # Attach LoRALinear wrappers (dropout=0.0 — we're merging, not training).
    n_attached = attach_lora(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
    )
    if n_attached == 0:
        sys.exit(
            f"ERROR: zero LoRA adapters attached — "
            f"target_modules={target_modules} matched no nn.Linear leaves.\n"
            "Ensure the base model matches the one used for training."
        )
    print(f"INFO: attached {n_attached} LoRALinear modules", file=sys.stderr)

    # Load adapter weights.
    state = load_file(str(weights_path))

    # Build module-path → LoRALinear map.
    lora_modules: dict[str, LoRALinear] = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, LoRALinear)
    }

    consumed: set[str] = set()
    modules_loaded: set[str] = set()

    for tensor_key, tensor in state.items():
        if tensor_key.endswith(".lora_A"):
            module_path = tensor_key[: -len(".lora_A")]
            suffix = "lora_A"
        elif tensor_key.endswith(".lora_B"):
            module_path = tensor_key[: -len(".lora_B")]
            suffix = "lora_B"
        else:
            continue

        if module_path not in lora_modules:
            continue

        param = getattr(lora_modules[module_path], suffix)
        with torch.no_grad():
            param.copy_(tensor)
        consumed.add(tensor_key)
        modules_loaded.add(module_path)

    # Validate — same checks as infer.py.
    unmatched = set(state.keys()) - consumed
    if unmatched:
        sys.exit(
            f"ERROR: {len(unmatched)} adapter tensor(s) had no matching "
            "LoRALinear:\n" + "\n".join(f"  {k}" for k in sorted(unmatched))
        )
    missing = set(lora_modules.keys()) - modules_loaded
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} LoRALinear module(s) received no weights:\n"
            + "\n".join(f"  {k}" for k in sorted(missing))
        )

    print(
        f"INFO: adapter loaded — {len(consumed)} tensors into "
        f"{len(modules_loaded)} LoRALinear modules",
        file=sys.stderr,
    )

    # --- MERGE: call merge_and_unload() and swap plain Linear back in ---
    # Walk the model and collect (parent, attr_name, LoRALinear) triples.
    # We must collect first, then replace, to avoid mutating the iterator.
    to_replace: list[tuple[nn.Module, str, LoRALinear]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        parts = full_name.split(".")
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        to_replace.append((parent, parts[-1], module))

    for parent, attr, lora_mod in to_replace:
        merged_linear = lora_mod.merge_and_unload()  # returns self.original (nn.Linear)
        setattr(parent, attr, merged_linear)

    print(f"INFO: merged {len(to_replace)} LoRA adapters into base weights", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 2 — CONVERT to f16 GGUF
# ---------------------------------------------------------------------------

def _convert_to_f16_gguf(
    merged_hf_dir: Path,
    f16_gguf_path: Path,
    llama_cpp_dir: str,
) -> None:
    """
    Run convert_hf_to_gguf.py to produce an f16 GGUF from the merged HF dir.
    cwd is set to llama_cpp_dir so the vendored gguf-py package resolves.
    """
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        sys.exit(f"ERROR: convert_hf_to_gguf.py not found at '{convert_script}'")

    # Build PYTHONPATH so gguf-py vendored inside llama.cpp is importable.
    gguf_py_dir = str(Path(llama_cpp_dir) / "gguf-py")
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        gguf_py_dir + os.pathsep + existing_pp if existing_pp else gguf_py_dir
    )

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_hf_dir),
        "--outfile", str(f16_gguf_path),
        "--outtype", "f16",
    ]

    print(f"INFO: [stage 2] running convert_hf_to_gguf.py -> {f16_gguf_path}", file=sys.stderr)
    print(f"INFO:   cwd={llama_cpp_dir}", file=sys.stderr)
    print(f"INFO:   cmd={' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(
        cmd,
        cwd=llama_cpp_dir,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            f"ERROR: convert_hf_to_gguf.py failed with exit code {result.returncode}.\n"
            "Check stderr above for details. A common cause for Llama/SentencePiece\n"
            "models is a missing `sentencepiece` package — install it into the same\n"
            "interpreter with:  python -m pip install sentencepiece"
        )
    print("INFO: [stage 2] conversion to f16 GGUF complete", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 3 — QUANTIZE
# ---------------------------------------------------------------------------

def _quantize_gguf(
    f16_gguf_path: Path,
    out_path: Path,
    quant: str,
    llama_cpp_dir: str,
) -> None:
    """
    Run llama-quantize to convert *f16_gguf_path* to *out_path* at *quant*.
    """
    quantize_exe = _find_quantize_exe(llama_cpp_dir)

    cmd = [quantize_exe, str(f16_gguf_path), str(out_path), quant]

    print(
        f"INFO: [stage 3] running llama-quantize {quant} -> {out_path}",
        file=sys.stderr,
    )
    print(f"INFO:   cmd={' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(
            f"ERROR: llama-quantize failed with exit code {result.returncode}.\n"
            "Check stderr above for details."
        )
    print("INFO: [stage 3] quantization complete", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_gguf(
    model_path: str,
    adapter_dir: str | None,
    out_path: str,
    quant: str,
    llama_cpp_dir: str,
    keep_merged: bool = False,
    keep_f16: bool = False,
) -> dict[str, Any]:
    """
    Run the full 3-stage merge → convert → quantize pipeline.

    Parameters
    ----------
    model_path:
        Path to the base HuggingFace model directory.
    adapter_dir:
        Path to the adapter directory (adapter.safetensors + adapter_config.json).
        Pass None for a base-only export (merge is a no-op).
    out_path:
        Destination path for the final GGUF file.
    quant:
        Quantization type (e.g. "Q4_K_M", "Q8_0", "f16", "none").
        Must be a member of VALID_QUANTS.
    llama_cpp_dir:
        Path to the llama.cpp checkout.  Auto-detected if empty string — call
        find_llama_cpp() before this function for explicit control.
    keep_merged:
        If True, do not delete the intermediate _merged_hf/ directory.
    keep_f16:
        If True, do not delete the intermediate f16 GGUF.

    Returns
    -------
    dict with keys: out, quant, size_bytes, base_model, adapter.
    The same dict is also printed as a JSON line on stdout (last line).
    """
    # --- Validate inputs ---
    quant_norm = quant.strip()
    if quant_norm not in VALID_QUANTS:
        sys.exit(
            f"ERROR: unknown quant type '{quant_norm}'.\n"
            f"Valid types: {', '.join(sorted(VALID_QUANTS))}"
        )

    model_path_p = Path(model_path)
    if not model_path_p.exists():
        sys.exit(f"ERROR: base model path not found: '{model_path}'")

    out_path_p = Path(out_path).resolve()
    out_dir = out_path_p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_path_p.stem  # used for intermediate file names

    # Intermediate paths inside out_dir
    merged_hf_dir = out_dir / "_merged_hf"
    f16_gguf_path = out_dir / f"{stem}-f16.gguf"

    adapter_str = str(Path(adapter_dir).resolve()) if adapter_dir else "none"

    # -----------------------------------------------------------------------
    # Stage 1 — MERGE
    # -----------------------------------------------------------------------
    print(f"INFO: [stage 1] loading base model from '{model_path}'", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path_p), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path_p),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()

    if adapter_dir is not None:
        adapter_dir_p = Path(adapter_dir)
        if not adapter_dir_p.exists():
            sys.exit(f"ERROR: adapter directory not found: '{adapter_dir}'")
        print(f"INFO: [stage 1] applying + merging adapter from '{adapter_dir}'", file=sys.stderr)
        _apply_and_merge_adapter(model, adapter_dir_p)
    else:
        print("INFO: [stage 1] no adapter — exporting base model as-is", file=sys.stderr)

    print(f"INFO: [stage 1] saving merged HF model to '{merged_hf_dir}'", file=sys.stderr)
    merged_hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_hf_dir))
    tokenizer.save_pretrained(str(merged_hf_dir))
    print("INFO: [stage 1] merge complete", file=sys.stderr)

    # Free model RAM before shelling out
    del model

    # -----------------------------------------------------------------------
    # Stage 2 — CONVERT to f16 GGUF
    # -----------------------------------------------------------------------
    _convert_to_f16_gguf(merged_hf_dir, f16_gguf_path, llama_cpp_dir)

    # -----------------------------------------------------------------------
    # Stage 3 — QUANTIZE (or pass-through for f16/none)
    # -----------------------------------------------------------------------
    if quant_norm in F16_PASSTHROUGH:
        print(
            f"INFO: [stage 3] quant={quant_norm} — renaming f16 GGUF to '{out_path_p}'",
            file=sys.stderr,
        )
        shutil.move(str(f16_gguf_path), str(out_path_p))
        # f16 gguf IS the output — keep_f16 flag is moot
        keep_f16 = False  # already moved, nothing to clean
    else:
        _quantize_gguf(f16_gguf_path, out_path_p, quant_norm, llama_cpp_dir)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    if not keep_merged and merged_hf_dir.exists():
        shutil.rmtree(str(merged_hf_dir), ignore_errors=True)
        print("INFO: removed intermediate _merged_hf/ dir", file=sys.stderr)

    if not keep_f16 and f16_gguf_path.exists():
        f16_gguf_path.unlink(missing_ok=True)
        print("INFO: removed intermediate f16 GGUF", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    size_bytes = out_path_p.stat().st_size if out_path_p.exists() else 0
    summary: dict[str, Any] = {
        "out": str(out_path_p),
        "quant": quant_norm,
        "size_bytes": size_bytes,
        "base_model": str(model_path_p.resolve()),
        "adapter": adapter_str,
    }
    # Final stdout line — parseable by Rust CLI
    print(json.dumps(summary))
    return summary
