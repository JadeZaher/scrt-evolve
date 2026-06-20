"""
infer.py — loader + generation logic for scrt-evolve LoRA adapters.

Reuses scrt_evolve_train.trainer.LoRALinear and its attach logic verbatim
so the adapter tensor naming contract is kept identical between save and load.

Adapter tensor names follow the pattern produced by save_adapter():
    model.layers.0.self_attn.q_proj.lora_A   shape [rank, in_features]
    model.layers.0.self_attn.q_proj.lora_B   shape [out_features, rank]

The module-path prefix before ".lora_A"/".lora_B" is the full named_modules()
key of the LoRALinear wrapper, which matches the key used during training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the exact LoRALinear from the trainer — same class, same tensor layout.
from scrt_evolve_train.trainer import LoRALinear, attach_lora


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(
    model_path: str,
) -> tuple[Any, Any]:
    """
    Load a HuggingFace causal-LM and its tokenizer from *model_path*.

    Uses float32 and low_cpu_mem_usage — safe for CPU inference.
    Returns (model, tokenizer) with model in eval mode.
    """
    model_path = str(model_path)
    print(f"INFO: loading tokenizer from {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"INFO: loading model from {model_path}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Adapter application
# ---------------------------------------------------------------------------

def apply_adapter(model: nn.Module, adapter_dir: str | Path) -> None:
    """
    Load adapter_config.json + adapter.safetensors from *adapter_dir* and
    patch *model* in-place.

    Steps:
    1. Read adapter_config.json for rank, alpha, target_modules.
    2. Attach LoRALinear wrappers using the same attach_lora() from the trainer
       (dropout=0.0 for inference — no stochastic masking).
    3. Load adapter.safetensors; copy each lora_A/lora_B tensor into the
       matching LoRALinear by module path prefix.
    4. Assert every adapter tensor found a home AND every LoRALinear got
       weights — mismatches produce a clear error listing unmatched names.
    """
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter.safetensors not found in {adapter_dir}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rank: int = cfg["rank"]
    alpha: float = float(cfg["alpha"])
    target_modules: list[str] = cfg["target_modules"]

    print(
        f"INFO: applying adapter rank={rank} alpha={alpha} "
        f"target_modules={target_modules}",
        file=sys.stderr,
    )

    # Attach LoRALinear wrappers (dropout=0.0 — inference mode).
    n_attached = attach_lora(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
    )
    if n_attached == 0:
        raise RuntimeError(
            f"apply_adapter: zero LoRA adapters attached — "
            f"target_modules={target_modules} matched no nn.Linear leaves. "
            "Check that the base model matches the one used for training."
        )
    print(f"INFO: attached {n_attached} LoRALinear modules", file=sys.stderr)

    # Load adapter weights.
    state = load_file(str(weights_path))  # dict[str, Tensor]

    # Build a map from module path → LoRALinear for all attached wrappers.
    # named_modules() returns the SAME keys used by save_adapter() in trainer.py.
    lora_modules: dict[str, LoRALinear] = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }

    # Track which adapter tensors were consumed and which modules got weights.
    consumed: set[str] = set()
    modules_loaded: set[str] = set()

    for tensor_key, tensor in state.items():
        # tensor_key looks like: "model.layers.0.self_attn.q_proj.lora_A"
        if tensor_key.endswith(".lora_A"):
            module_path = tensor_key[: -len(".lora_A")]
            suffix = "lora_A"
        elif tensor_key.endswith(".lora_B"):
            module_path = tensor_key[: -len(".lora_B")]
            suffix = "lora_B"
        else:
            continue  # unexpected key — skip silently; checked below

        if module_path not in lora_modules:
            # Will surface in the unmatched check below.
            continue

        lora_mod = lora_modules[module_path]
        param = getattr(lora_mod, suffix)  # nn.Parameter
        with torch.no_grad():
            param.copy_(tensor)
        consumed.add(tensor_key)
        modules_loaded.add(module_path)

    # Validate: every adapter tensor must have been consumed.
    all_adapter_keys = set(state.keys())
    unmatched_tensors = all_adapter_keys - consumed
    if unmatched_tensors:
        raise RuntimeError(
            f"apply_adapter: {len(unmatched_tensors)} adapter tensor(s) had no "
            f"matching LoRALinear in the model:\n"
            + "\n".join(f"  {k}" for k in sorted(unmatched_tensors))
        )

    # Validate: every attached LoRALinear must have received weights.
    modules_missing = set(lora_modules.keys()) - modules_loaded
    if modules_missing:
        raise RuntimeError(
            f"apply_adapter: {len(modules_missing)} LoRALinear module(s) received "
            f"no weights from the adapter file:\n"
            + "\n".join(f"  {k}" for k in sorted(modules_missing))
        )

    print(
        f"INFO: adapter loaded — {len(consumed)} tensors into "
        f"{len(modules_loaded)} LoRALinear modules",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    chat: bool = False,
) -> str:
    """
    Generate text for *prompt* and return only the generated portion
    (the prompt itself is stripped from the output).

    When *chat* is True and the tokenizer has ``apply_chat_template``, the
    prompt is wrapped in TinyLlama-style chat markup before encoding.
    When *temperature* <= 0, greedy decoding is used (do_sample=False);
    otherwise sampling with the given temperature.

    Training concatenated prompt + completion plainly, so by default the
    prompt is fed as-is (no chat template) to match the training distribution.
    """
    if chat and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        encoded_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        encoded_prompt = prompt

    inputs = tokenizer(encoded_prompt, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # Strip prompt tokens — return only the generated continuation.
    generated_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
