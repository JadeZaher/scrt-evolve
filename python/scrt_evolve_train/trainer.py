"""
trainer.py — scrt-evolve real-model LoRA training path (transformers-based).

Loads a HuggingFace causal-LM (e.g. TinyLlama-1.1B), attaches hand-rolled
LoRA adapters to configured linear modules, trains on a scrt-evolve
dataset.jsonl with prompt-masked cross-entropy, and saves the adapter
artifact (adapter.safetensors + adapter_config.json).

Ported/adapted from lexame hivemind-models src/moe/expert_trainer.py.
No peft dependency. CPU-safe (float32).
"""

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# LoRA layer — ported verbatim-ish from lexame expert_trainer.py
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge_and_unload(self) -> nn.Linear:
        with torch.no_grad():
            self.original.weight.add_((self.lora_B @ self.lora_A) * self.scaling)
        return self.original


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------

def attach_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """
    Walk the model and replace any nn.Linear whose name ends with one of
    target_modules with a LoRALinear. Returns the number of adapters attached.
    """
    count = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split(".")[-1]
        if leaf not in target_modules:
            continue
        # Walk to parent
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[tuple[str, str]]:
    """
    Read dataset.jsonl. Render qa and instruction rows to (prompt, completion)
    pairs. Skip and count other kinds. Exits on fatal errors.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        sys.exit(f"ERROR: dataset not found: {path}")

    pairs: list[tuple[str, str]] = []
    skipped = 0
    total = 0

    with dataset_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                row: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed JSON at line {lineno}: {e}", file=sys.stderr)
                skipped += 1
                continue

            kind = row.get("kind", "")
            if kind == "qa":
                prompt_text = row.get("prompt", "")
                completion_text = row.get("completion", "")
                if not prompt_text or not completion_text:
                    skipped += 1
                    continue
                pairs.append((prompt_text, completion_text))
            elif kind == "instruction":
                instruction = row.get("instruction", "")
                inp = row.get("input", "")
                output = row.get("output", "")
                if not instruction or not output:
                    skipped += 1
                    continue
                if inp:
                    prompt_text = instruction + "\n\n" + inp
                else:
                    prompt_text = instruction
                pairs.append((prompt_text, output))
            else:
                skipped += 1

    if skipped:
        print(f"INFO: skipped {skipped}/{total} rows (unsupported kind or missing fields)", file=sys.stderr)

    if not pairs:
        sys.exit(
            f"ERROR: no qa/instruction rows found in {path} "
            f"(total lines: {total}, skipped: {skipped})"
        )

    print(f"INFO: loaded {len(pairs)} training pairs from {total} rows", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Tokenization / batching
# ---------------------------------------------------------------------------

def build_batch(
    pairs: list[tuple[str, str]],
    tokenizer: Any,
    step: int,
    batch_size: int,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    """
    Build a batch of input_ids / attention_mask / labels tensors.
    Labels are -100 on prompt tokens and pad tokens; loss only on completions.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for b in range(batch_size):
        idx = (step * batch_size + b) % len(pairs)
        prompt_text, completion_text = pairs[idx]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        # Completion: no BOS; add EOS
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + completion_ids

        # Truncate to max_seq_len
        if len(full_ids) > max_seq_len:
            # Prefer to keep as much completion as possible
            prompt_len = max(1, max_seq_len - len(completion_ids))
            prompt_ids = prompt_ids[:prompt_len]
            completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]
            full_ids = prompt_ids + completion_ids

        seq_len = len(full_ids)
        pad_len = max_seq_len - seq_len

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        ids = full_ids + [pad_id] * pad_len
        mask = [1] * seq_len + [0] * pad_len

        # Labels: -100 on prompt and pad positions, actual ids on completion
        prompt_actual_len = len(prompt_ids)
        lbl = [-100] * prompt_actual_len + completion_ids + [-100] * pad_len

        assert len(ids) == max_seq_len
        assert len(mask) == max_seq_len
        assert len(lbl) == max_seq_len

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(lbl)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Adapter save
# ---------------------------------------------------------------------------

def save_adapter(
    model: nn.Module,
    out_dir: Path,
    rank: int,
    alpha: float,
    target_modules: list[str],
    base_model_path: str,
) -> None:
    """
    Save lora_A / lora_B params as adapter.safetensors and write
    adapter_config.json. Uses atomic write (tmp -> replace).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, torch.Tensor] = {}
    for full_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{full_name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{full_name}.lora_B"] = module.lora_B.detach().cpu()

    # Atomic write
    tmp_path = out_dir / "adapter.safetensors.tmp"
    final_path = out_dir / "adapter.safetensors"
    save_file(state, str(tmp_path))
    os.replace(str(tmp_path), str(final_path))

    config = {
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "base_model_path": base_model_path,
        "format": "safetensors",
    }
    config_path = out_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(args: Any) -> None:
    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    # Dataset
    pairs = load_dataset(args.dataset)

    # Model path check
    model_path = args.model
    if not Path(model_path).exists():
        sys.exit(f"ERROR: model path not found: {model_path}")

    print(f"INFO: loading tokenizer from {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"INFO: loading model from {model_path}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.train()

    # Attach LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    n_adapters = attach_lora(model, target_modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    if n_adapters == 0:
        sys.exit(
            f"ERROR: zero LoRA adapters attached. target_modules={target_modules} did not match "
            f"any nn.Linear leaf names in the model. For Llama-family models, "
            f"'q_proj,v_proj' should match — check the model architecture."
        )
    print(f"INFO: attached {n_adapters} LoRA adapters", file=sys.stderr)

    # Optimizer — only lora params
    lora_params = [
        p for m in model.modules()
        if isinstance(m, LoRALinear)
        for p in [m.lora_A, m.lora_B]
    ]
    assert len(lora_params) > 0, "No LoRA parameters collected — logic error."
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    # Output dir
    dataset_dir = Path(args.dataset).parent
    out_dir = Path(args.out) if args.out else dataset_dir / "adapter"

    # Train loop
    first_loss: float | None = None
    last_loss: float = 0.0

    for step in range(args.steps):
        batch = build_batch(pairs, tokenizer, step, args.batch_size, args.max_seq_len)

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss: torch.Tensor = outputs.loss
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if first_loss is None:
            first_loss = loss_val
        last_loss = loss_val

        if (step + 1) % args.log_every == 0 or step == 0:
            print(f"step {step+1}/{args.steps}  loss={loss_val:.4f}", file=sys.stderr)

    # Save adapter
    save_adapter(
        model=model,
        out_dir=out_dir,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=target_modules,
        base_model_path=str(Path(model_path).resolve()),
    )

    summary = {
        "first_loss": round(first_loss or 0.0, 6),
        "final_loss": round(last_loss, 6),
        "steps": args.steps,
        "adapters": n_adapters,
        "out": str(out_dir.resolve()),
    }
    # Final JSON summary on stdout — parseable by Rust CLI
    print(json.dumps(summary))
