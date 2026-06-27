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
        # Match the wrapped layer's dtype/device so the LoRA matmul never hits a
        # dtype mismatch (e.g. a bf16 block) and the params land on the right
        # device. Falls back to default (fp32/cpu) for layers without a weight.
        ref = getattr(original, "weight", None)
        ref_dtype = ref.dtype if ref is not None else torch.float32
        ref_device = ref.device if ref is not None else None
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=ref_dtype, device=ref_device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=ref_dtype, device=ref_device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.original.parameters():
            p.requires_grad = False
        # QAT (track 23): when set, the effective weight (base + LoRA delta) is
        # passed through a fake-quant of `qat_quant` before the matmul, so the
        # adapter learns to compensate for deployment quantization. None ⇒ plain
        # LoRA (today's behavior). `qat_name` keys calibrated scales.
        self.qat_quant: str | None = None
        self.qat_group_size: int = 32
        self.qat_name: str = ""
        self.qat_calibrator = None  # set to a qat.Calibrator when calibrating
        # When True, behave as the frozen base layer (no LoRA delta, no QAT).
        # Used by the sharded distillation trainer to get the teacher output
        # from the same module. Default False ⇒ standard LoRA behavior.
        self.lora_disabled: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_disabled:
            return self.original(x)
        if self.qat_quant is None:
            base = self.original(x)
            # Dtype-safe LoRA delta: some modules (e.g. MoE routers) run in fp32
            # while the rest of the block is bf16/fp16. Match the base output's
            # dtype so the delta add never raises a dtype mismatch.
            delta = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base + delta.to(base.dtype)
        # QAT path: build the effective weight, fake-quantize it (STE), and apply.
        from scrt_evolve_train import qat as _qat

        eff_w = self.original.weight + (self.lora_B @ self.lora_A) * self.scaling
        if self.qat_calibrator is not None:
            self.qat_calibrator.observe(self.qat_name, eff_w)
        scale = self.qat_calibrator.scale_for(self.qat_name) if self.qat_calibrator else None
        q_w = _qat.fake_quantize(eff_w, self.qat_quant, self.qat_group_size, scale)
        out = torch.nn.functional.linear(self.dropout(x), q_w, self.original.bias)
        return out

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
        # Never wrap a MoE router/gate, even if its leaf name matches a target
        # (routers are routing classifiers, not content projections).
        if _is_router_path(full_name):
            continue
        # Walk to parent
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        count += 1
    return count


# Path substrings that mark a module as a state-space / convolution (Mamba/SSM)
# block. Generic (not model-specific): any arch whose modules live under these
# names is treated as SSM. The naive CPU SSM backward segfaults in current
# torch/transformers, so we EXCLUDE these from LoRA by default and leave them
# frozen (LoRA freezes the base, so no grad flows through them).
SSM_PATH_MARKERS = ("mamba", "ssm", "conv1d", "conv_1d", ".conv")

# Path substrings that mark a Linear as a MoE ROUTER / GATE (a tiny routing
# classifier, not a content projection). Generic: adapting the router is both
# unhelpful and often dtype-fragile (routers frequently run in fp32 while the
# block is bf16). Excluded from LoRA auto-detection.
ROUTER_PATH_MARKERS = ("router", "gate", "gating")


def _is_ssm_path(full_name: str) -> bool:
    low = full_name.lower()
    return any(m in low for m in SSM_PATH_MARKERS)


def _is_router_path(full_name: str) -> bool:
    low = full_name.lower()
    return any(m in low for m in ROUTER_PATH_MARKERS)


def auto_detect_targets(
    model: nn.Module, top_k: int = 6, exclude_ssm: bool = True
) -> list[str]:
    """Generic, architecture-agnostic LoRA target selection.

    Enumerate the model's nn.Linear leaves and pick the most common leaf names
    (a projection repeated across N layers is the signal), ranked by frequency.
    Model-agnostic: any arch's real linear projections are discovered, not
    hardcoded.

    `exclude_ssm` (default True): a leaf name is dropped if it ONLY ever appears
    inside a state-space/convolution (Mamba) block. This is correct hygiene — you
    don't want LoRA on layers that won't train well — but NOTE it does NOT by
    itself make a hybrid Mamba model trainable on CPU: autograd still traverses
    the naive Mamba forward op during `loss.backward()`, which segfaults on CPU
    without the `causal_conv1d`/`mamba-ssm` CUDA kernels (verified on
    granitemoehybrid). Hybrid-SSM TRAINING needs CUDA regardless; forward-only
    eval/inference works on CPU. A leaf appearing in BOTH ssm and non-ssm
    contexts is kept (it's a general projection).
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    ssm_ctx: Counter[str] = Counter()
    nonssm_ctx: Counter[str] = Counter()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split(".")[-1]
        if leaf in ("lm_head", "embed_tokens"):
            continue
        # Skip MoE routers/gates — tiny routing classifiers, not content
        # projections, and frequently fp32 (dtype-fragile under LoRA).
        if _is_router_path(full_name):
            continue
        counts[leaf] += 1
        if _is_ssm_path(full_name):
            ssm_ctx[leaf] += 1
        else:
            nonssm_ctx[leaf] += 1

    if not counts:
        return []

    candidates = []
    for leaf, n in counts.items():
        # Drop leaves that live ONLY in SSM blocks.
        if exclude_ssm and ssm_ctx.get(leaf, 0) > 0 and nonssm_ctx.get(leaf, 0) == 0:
            continue
        candidates.append((leaf, n))

    ranked = sorted(candidates, key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:top_k]]


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

def _resume_adapter_weights(model: nn.Module, adapter_dir: str, device: Any) -> None:
    """Load an existing adapter's lora_A/lora_B weights into the model's already
    -attached LoRALinear modules, so training CONTINUES from it. Mirrors the
    naming contract in :func:`save_adapter` / ``scrt_evolve_infer.apply_adapter``.
    Exits if the file is missing; warns (does not fail) on shape/target drift."""
    from safetensors.torch import load_file

    weights = Path(adapter_dir) / "adapter.safetensors"
    if not weights.exists():
        sys.exit(f"ERROR: --resume-adapter: {weights} not found")
    state = load_file(str(weights))
    mods = dict(model.named_modules())
    loaded = 0
    for key, tensor in state.items():
        if not (key.endswith(".lora_A") or key.endswith(".lora_B")):
            continue
        mod_path, attr = key.rsplit(".", 1)
        mod = mods.get(mod_path)
        if not isinstance(mod, LoRALinear):
            print(f"WARN: resume-adapter: no LoRALinear at '{mod_path}' — skipped", file=sys.stderr)
            continue
        param = getattr(mod, attr)
        if tuple(param.shape) != tuple(tensor.shape):
            print(
                f"WARN: resume-adapter: shape mismatch at {key} "
                f"({tuple(param.shape)} vs {tuple(tensor.shape)}) — skipped",
                file=sys.stderr,
            )
            continue
        with torch.no_grad():
            param.copy_(tensor.to(param.device, param.dtype))
        loaded += 1
    print(f"INFO: resumed {loaded} adapter tensors from {adapter_dir}", file=sys.stderr)


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

    # Resolve the accelerator. `--device auto` (default) uses CUDA when available
    # else CPU; explicit `cuda`/`cpu` force it. The dense trainer historically ran
    # CPU-only (it ignored --device) — honoring it here makes GPU training ~10x
    # faster for small models that fit (TinyLlama-class). fp32 master weights are
    # kept (LoRA needs them); the base in fp32 fits a small model on the GPU.
    dev_spec = (getattr(args, "device", "auto") or "auto").strip()
    if dev_spec in ("", "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_spec)

    print(f"INFO: loading model from {model_path} (device={device})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.to(device)
    model.train()

    # Resolve LoRA targets. `--target-modules auto` (or an empty value) triggers
    # generic, architecture-agnostic auto-detection so hybrid/MoE arches work
    # without hardcoded names.
    raw_targets = (args.target_modules or "").strip()
    if raw_targets in ("", "auto"):
        target_modules = auto_detect_targets(model)
        print(f"INFO: auto-detected LoRA targets: {target_modules}", file=sys.stderr)
    else:
        target_modules = [m.strip() for m in raw_targets.split(",") if m.strip()]

    n_adapters = attach_lora(model, target_modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    if n_adapters == 0:
        # Auto-detect fallback: if explicit targets matched nothing, try auto.
        auto = auto_detect_targets(model)
        if auto and auto != target_modules:
            print(
                f"WARN: explicit targets {target_modules} matched nothing; "
                f"falling back to auto-detected {auto}",
                file=sys.stderr,
            )
            target_modules = auto
            n_adapters = attach_lora(model, target_modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    if n_adapters == 0:
        sys.exit(
            f"ERROR: zero LoRA adapters attached. No nn.Linear leaves matched "
            f"{target_modules}. Use `--target-modules auto` to auto-detect, or "
            "inspect the model's module names."
        )
    print(f"INFO: attached {n_adapters} LoRA adapters", file=sys.stderr)

    # CONTINUE training from an existing adapter (config `[train.lora].init_adapter`
    # / `--resume-adapter`). Loads its weights into the freshly-attached LoRALinears
    # so a branch keeps evolving instead of restarting from scratch each round —
    # the config-driven "further training" path (no merge/file-shuffling needed).
    resume = getattr(args, "resume_adapter", None)
    if resume:
        _resume_adapter_weights(model, resume, device)

    # QAT setup (track 23): if --qat <quant> is set, configure every LoRALinear to
    # fake-quantize its effective weight during the forward pass. Optional
    # calibration picks per-group scales over the first N batches.
    qat_quant = getattr(args, "qat", None)
    calibrator = None
    if qat_quant:
        from scrt_evolve_train import qat as _qat

        calib_cfg = _qat.CalibConfig(
            enabled=True,
            quant=qat_quant,
            group_size=getattr(args, "qat_group_size", 32),
            calibrate_batches=getattr(args, "qat_calibrate", 0),
        )
        calibrator = _qat.Calibrator(cfg=calib_cfg)
        for name, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.qat_quant = qat_quant
                m.qat_group_size = calib_cfg.group_size
                m.qat_name = name
                m.qat_calibrator = calibrator
        print(
            f"INFO: QAT enabled - quant={qat_quant} group_size={calib_cfg.group_size} "
            f"calibrate_batches={calib_cfg.calibrate_batches}",
            file=sys.stderr,
        )

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
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss: torch.Tensor = outputs.loss
        loss.backward()
        optimizer.step()

        # Advance QAT calibration window (bounded by calibrate_batches).
        if calibrator is not None:
            calibrator.tick()

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
