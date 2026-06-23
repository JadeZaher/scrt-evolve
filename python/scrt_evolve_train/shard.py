"""
shard.py — sharded / fractional LoRA training via block-local distillation.

Goal: train adapters for an arbitrarily large model while keeping peak VRAM
bounded to ONE contiguous block of decoder layers, so the framework runs on
anyone's GPU. This is the "decentralized fractional training" path: each shard
is trained INDEPENDENTLY (embarrassingly parallel), only one block is ever
resident on the accelerator, and the rest of the model stays on CPU/disk.

Approach — block-local distillation (model-agnostic):

  The frozen, full-precision model is the teacher. For a contiguous block of
  layers [a, b):
    1. boundary activation in_k  = hidden states entering layer a
       (captured once by streaming the frozen prefix forward, one block on the
        accelerator at a time — never the whole model).
    2. teacher target            = block_k(in_k)            (frozen, no grad)
    3. student output            = block_k + LoRA (in_k)    (LoRA trainable)
    4. loss                      = MSE(student, teacher)    [+ optional QAT]

  Only block k's weights + its activations occupy VRAM. No suffix is needed,
  no LM head, no global backward through the network — so the bound is exactly
  one block regardless of model depth, and shards can be trained in any order
  or fully in parallel on separate machines.

  Pairs naturally with QAT (track 23): when --qat is set, the student block's
  LoRALinear fake-quantizes its effective weight, so the adapter learns to make
  the QUANTIZED block reproduce the full-precision block's behavior — exactly
  what you want before a Q4_K_M GGUF deployment.

This module is additive: it reuses LoRALinear / auto_detect_targets /
load_dataset / build_batch / save_adapter from trainer.py and is selected by
`scrt-evolve` only when sharded mode is requested. The dense trainer is
untouched.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .trainer import (
    LoRALinear,
    attach_lora,
    auto_detect_targets,
    build_batch,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Generic decoder-layer discovery (model-agnostic, no hardcoded paths)
# ---------------------------------------------------------------------------

def find_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Find the model's stack of repeated decoder layers, generically.

    Heuristic: the decoder stack is the longest ``nn.ModuleList`` of structurally
    similar children (same submodule type). Works for any HF causal-LM
    (``model.model.layers``, ``transformer.h``, ``gpt_neox.layers``, …) without
    hardcoding the attribute path. Returns (module_list, dotted_name).
    """
    best: tuple[int, nn.ModuleList | None, str] = (0, None, "")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best[0]:
            # require the entries to be modules (decoder layers), not leaves
            if all(isinstance(c, nn.Module) and any(True for _ in c.children()) for c in mod):
                best = (len(mod), mod, name)
    if best[1] is None:
        sys.exit(
            "ERROR: could not locate a decoder-layer ModuleList for sharding. "
            "This model's layer stack was not found generically."
        )
    return best[1], best[2]


def plan_shards(n_layers: int, n_shards: int | None, block_size: int | None) -> list[tuple[int, int]]:
    """Split ``n_layers`` into contiguous [start, end) blocks.

    Exactly one of ``n_shards`` / ``block_size`` drives the split (block_size
    wins if both are given). Returns the list of (start, end) ranges covering
    every layer with no gaps or overlap.
    """
    if block_size and block_size > 0:
        step = block_size
    elif n_shards and n_shards > 0:
        step = max(1, (n_layers + n_shards - 1) // n_shards)
    else:
        step = n_layers  # one shard = dense fallback
    shards: list[tuple[int, int]] = []
    a = 0
    while a < n_layers:
        b = min(n_layers, a + step)
        shards.append((a, b))
        a = b
    return shards


# ---------------------------------------------------------------------------
# Activation capture — stream the frozen prefix one block at a time
# ---------------------------------------------------------------------------

def _layer_call(layer: nn.Module, hidden: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Call one decoder layer and normalize its (possibly tuple) output to the
    hidden-state tensor. HF decoder layers return either a Tensor or a tuple
    whose first element is the hidden state."""
    out = layer(hidden, **kw)
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def capture_boundaries(
    model: nn.Module,
    layers: nn.ModuleList,
    boundaries: list[int],
    embeds: torch.Tensor,
    device: torch.device,
    layer_kwargs: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Capture the hidden state entering each layer index in ``boundaries`` by
    streaming the frozen layer stack forward, moving ONE layer to ``device`` at a
    time and evicting it back to CPU afterwards. Returns {layer_idx: activation}
    on CPU. Peak VRAM = one layer + one activation.

    The prefix must reflect the PURE frozen base — so any LoRA adapters left on
    earlier layers from a previously-trained shard are disabled for the duration
    of the capture (and restored afterwards), preventing cross-shard
    contamination of boundary activations during a sequential multi-shard run.
    """
    target_boundary = max(boundaries)  # we only need to stream up to here
    want = set(boundaries)

    # Disable any LoRA on the layers we will stream, remembering prior state.
    prior: list[tuple[LoRALinear, bool]] = []
    for layer in layers:
        for m in layer.modules():
            if isinstance(m, LoRALinear):
                prior.append((m, m.lora_disabled))
                m.lora_disabled = True

    captured: dict[int, torch.Tensor] = {}
    hidden = embeds.to(device)
    try:
        for i, layer in enumerate(layers):
            if i in want:
                captured[i] = hidden.detach().to("cpu")
            if i >= target_boundary:
                break  # no need to run layers at/after the deepest boundary
            layer.to(device)
            hidden = _layer_call(layer, hidden, **layer_kwargs)
            layer.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for m, was in prior:
            m.lora_disabled = was
    return captured


# ---------------------------------------------------------------------------
# Sharded training
# ---------------------------------------------------------------------------

def train_sharded(args: Any) -> None:
    import random

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(_resolve_device(getattr(args, "device", "auto")))
    dtype = _resolve_dtype(getattr(args, "dtype", "auto"), device)
    print(f"INFO[shard]: device={device} dtype={dtype}", file=sys.stderr)

    model_path = args.model
    if not Path(model_path).exists():
        sys.exit(f"ERROR: model path not found: {model_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"INFO[shard]: loading model to CPU ({dtype})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.eval()  # base is frozen everywhere; LoRA carries the only grads
    for p in model.parameters():
        p.requires_grad_(False)

    layers, layers_name = find_decoder_layers(model)
    n_layers = len(layers)
    shards = plan_shards(n_layers, getattr(args, "shards", None), getattr(args, "block_size", None))
    print(
        f"INFO[shard]: decoder stack '{layers_name}' has {n_layers} layers; "
        f"{len(shards)} shard(s): {shards}",
        file=sys.stderr,
    )

    # Embedding lookup (small) stays resident on device for activation capture.
    embed = model.get_input_embeddings().to(device)

    # Build a small set of token batches to distill over (reuse dense batcher).
    pairs = load_dataset(args.dataset)
    n_batches = max(1, getattr(args, "calib_batches", 8))
    batches = [
        build_batch(pairs, tokenizer, step, args.batch_size, args.max_seq_len)
        for step in range(n_batches)
    ]

    # Optional shard selection (train one shard per process for true
    # decentralization). --shard-index N trains only shard N.
    only = getattr(args, "shard_index", None)
    selected = list(enumerate(shards))
    if only is not None:
        if only < 0 or only >= len(shards):
            sys.exit(f"ERROR: --shard-index {only} out of range (0..{len(shards)-1})")
        selected = [(only, shards[only])]
        print(f"INFO[shard]: training ONLY shard {only} = layers {shards[only]}", file=sys.stderr)

    # Layer kwargs: decoder layers need at least position info on some arches.
    # We keep it minimal/generic — most HF layers accept just hidden_states.
    layer_kwargs: dict[str, Any] = {}

    out_dir = Path(args.out) if args.out else Path(args.dataset).parent / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_modules = _resolve_targets(args, model)
    all_summaries: list[dict[str, Any]] = []
    total_adapters = 0

    # Objective: `distill` (block-local MSE-vs-self — a representation/regularize
    # signal) or `end_task` (the FINAL shard learns real cross-entropy against the
    # completion tokens via the LM head — the actual KNOWLEDGE signal). Under
    # end_task, non-final shards still distill; the final shard does CE.
    objective = (getattr(args, "objective", "distill") or "distill").strip()
    final_norm = _find_final_norm(model)
    lm_head = model.get_output_embeddings()

    for shard_id, (a, b) in selected:
        is_final = b >= n_layers
        # The activations we need: the input to layer `a` (boundary). Capture by
        # streaming the frozen prefix [0, a) — peak VRAM stays at one layer.
        per_batch_in: list[torch.Tensor] = []
        per_batch_labels: list[torch.Tensor] = []
        for batch in batches:
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                emb = embed(ids).to(dtype)
            caps = capture_boundaries(model, layers, [a], emb, device, layer_kwargs)
            per_batch_in.append(caps[a])
            per_batch_labels.append(batch["labels"])

        # Build this shard's block, attach LoRA (params init on CPU), THEN move
        # the whole block — base + freshly created LoRA params — to the device
        # together so everything lives on one device.
        block = nn.ModuleList([layers[i] for i in range(a, b)])
        n_added = 0
        for li in range(len(block)):
            n_added += attach_lora(
                block[li], target_modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout
            )
        if n_added == 0:
            auto = auto_detect_targets(block)
            for li in range(len(block)):
                n_added += attach_lora(block[li], auto, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
            target_modules = auto
        if n_added == 0:
            sys.exit(f"ERROR: shard {shard_id}: zero LoRA adapters attached on layers {a}:{b}")
        block.to(device)
        total_adapters += n_added
        print(f"INFO[shard {shard_id}]: layers {a}:{b}  adapters={n_added}", file=sys.stderr)

        # Optional QAT on this shard's LoRALinears.
        _maybe_enable_qat(args, block)

        granularity = getattr(args, "granularity", "block") or "block"
        if objective == "end_task" and is_final:
            # The FINAL shard learns the real end-task signal: block → norm →
            # head → CE on completions. Real knowledge gradient, bounded VRAM.
            first_loss, last_loss = _train_final_shard_end_task(
                block, final_norm, lm_head, per_batch_in, per_batch_labels,
                device, dtype, layer_kwargs, args, shard_id,
            )
            print(
                f"INFO[shard {shard_id}]: end-task CE on final block "
                f"(layers {a}:{b}) — first={first_loss:.4f} last={last_loss:.4f}",
                file=sys.stderr,
            )
        elif granularity == "module":
            # PER-MODULE SUB-LAYER microsharding: train ONE submodule group at a
            # time within each layer, against that LAYER's frozen-output teacher.
            # Only the active group's LoRA gets gradients; the rest of the layer
            # is frozen base. Footprint floor = one layer + one group's optimizer
            # state. We still distill at the LAYER boundary (robust target), so
            # the per-layer input must be streamed for each layer in the block.
            first_loss, last_loss, n_groups = _train_block_by_module(
                block, a, per_batch_in, model, layers, device, dtype, layer_kwargs, args, shard_id
            )
            print(
                f"INFO[shard {shard_id}]: per-module mode trained {n_groups} group(s) "
                f"across layers {a}:{b}",
                file=sys.stderr,
            )
        else:
            lora_params = [
                p for m in block.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)
            ]
            optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

            first_loss = None
            last_loss = 0.0
            for step in range(args.steps):
                in_k = per_batch_in[step % len(per_batch_in)].to(device).to(dtype)

                # Teacher: frozen block (LoRA delta off) — adapters disabled.
                with torch.no_grad():
                    teacher = _run_block(block, in_k, layer_kwargs, lora_enabled=False)

                # Student: same block with LoRA (and QAT) active.
                student = _run_block(block, in_k, layer_kwargs, lora_enabled=True)
                loss = torch.nn.functional.mse_loss(student.float(), teacher.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lv = loss.item()
                if first_loss is None:
                    first_loss = lv
                last_loss = lv
                if (step + 1) % args.log_every == 0 or step == 0:
                    print(f"shard {shard_id} step {step+1}/{args.steps}  loss={lv:.6f}", file=sys.stderr)

        # Save this shard's adapter (namespaced by global layer index so shards
        # trained on different machines merge cleanly).
        _save_shard_adapter(block, a, layers_name, out_dir, shard_id, args, target_modules,
                            str(Path(model_path).resolve()))

        # Evict the shard from the device before moving on.
        block.to("cpu")
        if device.type == "cuda":
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        else:
            peak = None
        all_summaries.append({
            "shard": shard_id,
            "layers": [a, b],
            "adapters": n_added,
            "first_loss": round(first_loss or 0.0, 6),
            "final_loss": round(last_loss, 6),
            "peak_vram_gb": peak,
        })

    summary = {
        "mode": "sharded",
        "granularity": getattr(args, "granularity", "block") or "block",
        "objective": objective,
        "shards": all_summaries,
        "n_shards_trained": len(all_summaries),
        "total_adapters": total_adapters,
        "out": str(out_dir.resolve()),
    }
    print(json.dumps(summary))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_block(block: nn.ModuleList, hidden: torch.Tensor, layer_kwargs: dict[str, Any],
               lora_enabled: bool) -> torch.Tensor:
    """Forward through a contiguous block. When ``lora_enabled`` is False, the
    LoRALinear modules fall back to their frozen base (delta scaled to zero)."""
    for m in block.modules():
        if isinstance(m, LoRALinear):
            m.lora_disabled = not lora_enabled  # read by patched forward
    h = hidden
    for layer in block:
        h = _layer_call(layer, h, **layer_kwargs)
    return h


def _find_final_norm(model: nn.Module) -> nn.Module | None:
    """Locate the model's final norm (applied after the last decoder layer,
    before the LM head). Generic: the `.norm`/`.final_layernorm`/`.ln_f` child of
    the decoder backbone (`model.model` on most HF causal-LMs)."""
    backbone = getattr(model, "model", model)
    for name in ("norm", "final_layernorm", "ln_f", "final_norm"):
        m = getattr(backbone, name, None)
        if isinstance(m, nn.Module):
            return m
    return None


def _train_final_shard_end_task(
    block: nn.ModuleList,
    final_norm: nn.Module | None,
    lm_head: nn.Module,
    per_batch_in: list[torch.Tensor],
    per_batch_labels: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    layer_kwargs: dict[str, Any],
    args: Any,
    shard_id: int,
) -> tuple[float, float]:
    """END-TASK objective for the FINAL shard: run the block (LoRA on) → final
    norm → LM head → cross-entropy against the completion labels. The boundary
    input is the cached frozen-prefix activation, so gradient flows ONLY through
    the final block's LoRA (the norm + head run frozen). This is the real
    knowledge signal — unlike block-local distillation, the target is the DESIRED
    tokens, not the block's own output. Footprint ≈ one block + head + logits.

    Labels follow the dataset convention: -100 on prompt/pad, completion ids
    elsewhere (loss only on completions). Shapes are [batch, seq].
    """
    # final norm + head resident, frozen (no grad on their params).
    if final_norm is not None:
        final_norm.to(device)
        for p in final_norm.parameters():
            p.requires_grad_(False)
    lm_head.to(device)
    for p in lm_head.parameters():
        p.requires_grad_(False)

    lora_params = [
        p for m in block.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)
    ]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    first_loss: float | None = None
    last_loss = 0.0
    n = len(per_batch_in)
    for step in range(args.steps):
        in_k = per_batch_in[step % n].to(device).to(dtype)
        labels = per_batch_labels[step % n].to(device)

        # Final block with LoRA active → norm → head → logits.
        for m in block.modules():
            if isinstance(m, LoRALinear):
                m.lora_disabled = False
        h = in_k
        for layer in block:
            h = _layer_call(layer, h, **layer_kwargs)
        if final_norm is not None:
            h = final_norm(h)
        logits = lm_head(h)

        # Causal LM shift: predict token t+1 from position t; CE on completions.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lv = loss.item()
        if first_loss is None:
            first_loss = lv
        last_loss = lv
        if (step + 1) % args.log_every == 0 or step == 0:
            print(
                f"shard {shard_id} [end_task] step {step+1}/{args.steps} ce_loss={lv:.6f}",
                file=sys.stderr,
            )

    return (first_loss or 0.0, last_loss)


# ---------------------------------------------------------------------------
# Per-module (sub-layer) helpers — the microsharding floor
# ---------------------------------------------------------------------------

def discover_groups(layer: nn.Module) -> list[tuple[str, nn.Module]]:
    """Generic submodule-GROUP discovery within one decoder layer.

    A group is a direct child module of the layer that contains at least one
    nn.Linear (e.g. self_attn / block_sparse_moe / shared_mlp / mamba on a
    GraniteMoeHybrid layer; q/k/v/o or mlp on a vanilla layer). Layernorms and
    other linear-free children are skipped. Model-agnostic — keyed on "has a
    trainable Linear", not on any name.
    """
    groups: list[tuple[str, nn.Module]] = []
    for child_name, child in layer.named_children():
        if any(isinstance(m, nn.Linear) for m in child.modules()):
            groups.append((child_name, child))
    return groups


def _set_group_student(layer: nn.Module, active_group: nn.Module | None) -> None:
    """Enable LoRA ONLY on the modules under ``active_group`` (the student);
    disable LoRA everywhere else in the layer (frozen base). ``active_group=None``
    disables all LoRA in the layer (pure frozen teacher)."""
    active_ids = set(id(m) for m in active_group.modules()) if active_group is not None else set()
    for m in layer.modules():
        if isinstance(m, LoRALinear):
            m.lora_disabled = id(m) not in active_ids


def _run_layer(layer: nn.Module, hidden: torch.Tensor, layer_kwargs: dict[str, Any]) -> torch.Tensor:
    """Forward a single decoder layer (LoRA state is set by the caller)."""
    return _layer_call(layer, hidden, **layer_kwargs)


def _train_block_by_module(
    block: nn.ModuleList,
    block_start: int,
    per_batch_in: list[torch.Tensor],
    model: nn.Module,
    layers: nn.ModuleList,
    device: torch.device,
    dtype: torch.dtype,
    layer_kwargs: dict[str, Any],
    args: Any,
    shard_id: int,
) -> tuple[float, float, int]:
    """Per-module sub-layer training within one block.

    For each layer in the block, for each submodule GROUP in that layer, train
    that group's LoRA in isolation (rest of the layer = frozen base) against the
    LAYER's own frozen output (teacher). Footprint = one layer + one group's
    optimizer state. Returns (first_loss, last_loss, n_groups_trained).

    Each layer's INPUT is the previous layer's output within the block, computed
    with all LoRA disabled (pure frozen prefix inside the block) — so groups
    train against a stable base activation, not a moving one.
    """
    first_loss: float | None = None
    last_loss = 0.0
    n_groups = 0

    # per_batch_in holds the input to the FIRST layer of the block (layer
    # block_start), one tensor per calib batch.
    n_batches = len(per_batch_in)
    # Running per-batch input to the current layer; starts at the block input.
    layer_inputs = [t.to(device).to(dtype) for t in per_batch_in]

    for li, layer in enumerate(block):
        gidx = block_start + li
        groups = discover_groups(layer)
        for gname, gmod in groups:
            # Does this group actually carry LoRA adapters? (Some children — e.g.
            # an SSM block whose linears were excluded — have none.)
            if not any(isinstance(m, LoRALinear) for m in gmod.modules()):
                continue
            params = [
                p for m in gmod.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)
            ]
            optimizer = torch.optim.AdamW(params, lr=args.lr)
            n_groups += 1
            for step in range(args.steps):
                x = layer_inputs[step % n_batches]
                # Teacher: this layer fully frozen (all LoRA off).
                _set_group_student(layer, None)
                with torch.no_grad():
                    teacher = _run_layer(layer, x, layer_kwargs)
                # Student: only THIS group's LoRA active.
                _set_group_student(layer, gmod)
                student = _run_layer(layer, x, layer_kwargs)
                loss = torch.nn.functional.mse_loss(student.float(), teacher.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lv = loss.item()
                if first_loss is None:
                    first_loss = lv
                last_loss = lv
                if (step + 1) % args.log_every == 0 or step == 0:
                    print(
                        f"shard {shard_id} L{gidx}/{gname} step {step+1}/{args.steps} loss={lv:.6f}",
                        file=sys.stderr,
                    )

        # Advance each batch's activation to the NEXT layer's input, with the
        # whole layer frozen (pure base prefix) so downstream groups see a stable
        # input. Done once per layer, no grad.
        _set_group_student(layer, None)
        with torch.no_grad():
            layer_inputs = [_run_layer(layer, x, layer_kwargs) for x in layer_inputs]

    return (first_loss or 0.0, last_loss, n_groups)


def _resolve_device(spec: str) -> str:
    if spec in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def _resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec in ("", "auto"):
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[spec]


def _resolve_targets(args: Any, model: nn.Module) -> list[str]:
    raw = (getattr(args, "target_modules", "") or "").strip()
    if raw in ("", "auto"):
        t = auto_detect_targets(model)
        print(f"INFO[shard]: auto-detected LoRA targets: {t}", file=sys.stderr)
        return t
    return [m.strip() for m in raw.split(",") if m.strip()]


def _maybe_enable_qat(args: Any, block: nn.ModuleList) -> None:
    qat_quant = getattr(args, "qat", None)
    if not qat_quant:
        return
    from scrt_evolve_train import qat as _qat

    calib = _qat.Calibrator(cfg=_qat.CalibConfig(
        enabled=True, quant=qat_quant,
        group_size=getattr(args, "qat_group_size", 32),
        calibrate_batches=getattr(args, "qat_calibrate", 0),
    ))
    for name, m in block.named_modules():
        if isinstance(m, LoRALinear):
            m.qat_quant = qat_quant
            m.qat_group_size = calib.cfg.group_size
            m.qat_name = name
            m.qat_calibrator = calib
    print(f"INFO[shard]: QAT enabled quant={qat_quant}", file=sys.stderr)


def _save_shard_adapter(block: nn.ModuleList, layer_offset: int, layers_name: str,
                        out_dir: Path, shard_id: int, args: Any,
                        target_modules: list[str], base_model_path: str) -> None:
    """Persist this shard's LoRA params, keyed by GLOBAL layer index so shards
    trained independently can be merged into one adapter set."""
    state: dict[str, torch.Tensor] = {}
    for li, layer in enumerate(block):
        gidx = layer_offset + li
        for full_name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                key = f"{layers_name}.{gidx}.{full_name}"
                state[f"{key}.lora_A"] = module.lora_A.detach().cpu()
                state[f"{key}.lora_B"] = module.lora_B.detach().cpu()
    fname = out_dir / f"adapter-shard-{shard_id:03d}.safetensors"
    tmp = out_dir / f"adapter-shard-{shard_id:03d}.safetensors.tmp"
    save_file(state, str(tmp))
    os.replace(str(tmp), str(fname))
    cfg = {
        "rank": args.rank, "alpha": args.alpha, "target_modules": target_modules,
        "base_model_path": base_model_path, "format": "safetensors",
        "shard": shard_id, "layer_offset": layer_offset, "layers_name": layers_name,
    }
    (out_dir / f"adapter-shard-{shard_id:03d}.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")
