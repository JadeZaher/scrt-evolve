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
import math
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


# ---------------------------------------------------------------------------
# Cross-MODEL seam distillation — compress a larger TEACHER into a smaller
# STUDENT branch (track 29 v1.1). The same-model path above distills a block
# against ITS OWN frozen output (a regularization signal). This path distills
# the student's per-block output against a DISTINCT, larger teacher's
# hidden-state at a mapped seam — genuine cross-model compression.
#
# Two fully-DECOUPLED phases so teacher + student are NEVER co-resident (the
# VRAM crux for a 32B→3B on a small box):
#   Phase A (capture): load the teacher ALONE, stream it one layer at a time,
#     write its per-seam hidden states to a disk cache, then UNLOAD it. Peak
#     VRAM = one teacher layer. The cache also stores the exact token ids so
#     Phase B feeds the student an identical sequence (positional alignment).
#   Phase B (train): load the student ALONE; for each student block, stream its
#     boundary input and train the block's LoRA (+ a discard-after read-out
#     projection that bridges width) to match the cached teacher seam.
#
# Requires a SHARED tokenizer (hidden states are matched position-by-position,
# so teacher + student must tokenize identically) — guarded at runtime.
#
# Findings carried from bench/seam_distill/RESULTS.md (the de-risked precursor):
#   * capture teacher targets in fp32 and train the student block in fp32
#     (bf16 AdamW updates for a small delta round away → the student stalls);
#   * match the full (projected) hidden state with cosine+MSE. Unlike the
#     same-model delta case, the teacher target is a DIFFERENT model's state
#     (not ≈ the student input), so it cannot trivially collapse to identity.
# ---------------------------------------------------------------------------

# Cache layout written by Phase A and read by Phase B (under --teacher-cache).
_SEAM_MANIFEST = "seam_manifest.json"


def plan_seam_map(
    student_blocks: list[tuple[int, int]], teacher_layers: int, strategy: str = "stride"
) -> dict[int, list[int]]:
    """Map each student block's OUTPUT boundary onto teacher seam layer(s).

    A student block ``[a, b)`` produces the hidden state entering student layer
    ``b`` (HF ``hidden_states[b]``: 0 = embeddings, k = output of layer k-1). The
    teacher (deeper) is sampled at the proportionally-corresponding depth.

    Returns ``{student_b: [teacher_seam_idx, ...]}`` where each teacher index is
    into the teacher's ``hidden_states`` (0..=teacher_layers). ``stride`` maps to
    a single nearest teacher seam (uniform depth ratio); ``block_avg`` maps to
    the span of teacher layers covered by this student block (averaged in
    Phase A) — a smoother target when the depth ratio is large.
    """
    n_student = max(b for _, b in student_blocks)
    out: dict[int, list[int]] = {}
    prev_t = 0
    for _, b in student_blocks:
        # Proportional teacher depth for this student boundary.
        t_hi = round(b * teacher_layers / n_student)
        t_hi = max(1, min(teacher_layers, t_hi))
        if strategy == "block_avg":
            lo = max(prev_t + 1, 1)
            out[b] = list(range(lo, t_hi + 1)) or [t_hi]
        else:  # "stride" (default): single nearest teacher seam
            out[b] = [t_hi]
        prev_t = t_hi
    return out


def build_projection(d_student: int, d_teacher: int, mode: str = "auto") -> nn.Module:
    """Read-out projection bridging the student's output width to the teacher's.

    The loss is computed in TEACHER space (project the student UP), so the
    teacher target is never lossily down-sampled. Identity when widths already
    match or ``mode='none'``. This module is a distill-time SCAFFOLD — trained
    jointly with the student LoRA and DISCARDED afterwards (only the LoRA is
    saved), so it never enters the exported model.
    """
    if mode == "none" or d_student == d_teacher:
        return nn.Identity()
    # "auto" / "student_up": a plain linear lift d_student → d_teacher.
    proj = nn.Linear(d_student, d_teacher, bias=False)
    if min(d_student, d_teacher) > 1:
        nn.init.orthogonal_(proj.weight)
    return proj


def distill_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "cosine_mse") -> torch.Tensor:
    """Hidden-state distillation loss between (projected) student output and the
    teacher seam target, both ``[*, d_teacher]``. Computed in fp32.

    * ``mse``        — plain mean-squared error (scale-sensitive).
    * ``cosine``     — 1 − mean cosine similarity (direction only; scale-free).
    * ``cosine_mse`` — their sum (default): direction + magnitude. Robust across
      models with different residual-stream scales.
    """
    p = pred.float().reshape(-1, pred.shape[-1])
    t = target.float().reshape(-1, target.shape[-1])
    mse = torch.nn.functional.mse_loss(p, t)
    if kind == "mse":
        return mse
    cos = 1.0 - torch.nn.functional.cosine_similarity(p, t, dim=-1).mean()
    if kind == "cosine":
        return cos
    return cos + mse


def _rotary_kwargs(model: nn.Module, hidden: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Build the kwargs a decoder layer needs to be called DIRECTLY on recent
    transformers. As of ~4.41 the rotary embeddings moved to the model level and
    are passed DOWN to each layer as ``position_embeddings``; a bare
    ``layer(hidden)`` then crashes ("cannot unpack non-iterable NoneType"). We
    recompute them from the backbone's ``rotary_emb`` for the given sequence
    length. (Attention defaults to causal when no mask is given.) Arches without a
    model-level rotary return ``{}`` — the bare call already works there."""
    backbone = getattr(model, "model", model)
    rotary = getattr(backbone, "rotary_emb", None)
    seq = hidden.shape[1]
    pos = torch.arange(seq, device=device).unsqueeze(0)
    if rotary is None:
        return {}
    rotary = rotary.to(device)
    pe = rotary(hidden.to(device), pos)
    return {"position_ids": pos, "position_embeddings": pe}


def block_lr_scale(ref_rms: float, target_rms: float, lo: float = 0.25, hi: float = 1.0) -> float:
    """DYNAMIC per-block LR multiplier from teacher-target magnitudes.

    A student block whose teacher seam has a larger residual-stream magnitude than
    the reference (shallowest) block produces proportionally larger MSE gradients,
    so a single global LR overshoots there. Scale the base LR by
    ``ref_rms / target_rms`` (block at the reference magnitude ⇒ 1.0; larger ⇒
    gentler), clamped to ``[lo, hi]`` so it never explodes or vanishes."""
    if target_rms <= 0:
        return hi
    return max(lo, min(hi, ref_rms / target_rms))


def lr_at_step(base_lr: float, step: int, total_steps: int, warmup: int) -> float:
    """Warmup→cosine-decay schedule within a block: linear ramp for the first
    ``warmup`` steps (prevents the early spike that diverges a block), then a
    cosine decay to ~0 (settles convergence). Pure for unit testing."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    denom = max(1, total_steps - warmup)
    prog = (step - warmup) / denom
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))


def _seam_indices(seam_map: dict[int, list[int]]) -> list[int]:
    """All distinct teacher seam indices Phase A must capture, sorted."""
    idxs: set[int] = set()
    for v in seam_map.values():
        idxs.update(v)
    return sorted(idxs)


def _read_model_depth_width(model_path: str) -> tuple[int, int, int]:
    """Read (num_layers, hidden_size, vocab_size) from a model's config WITHOUT
    loading weights (cheap — Phase A needs the student's depth to plan seams)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    d = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    v = getattr(cfg, "vocab_size", 0)
    if n is None or d is None:
        sys.exit(f"ERROR: could not read num_hidden_layers/hidden_size from {model_path}")
    return int(n), int(d), int(v)


@torch.no_grad()
def _capture_teacher_seams(
    model: nn.Module,
    layers: nn.ModuleList,
    seam_idxs: list[int],
    embeds: torch.Tensor,
    device: torch.device,
    layer_kwargs: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Stream the teacher one layer at a time, capturing residual-stream hidden
    states at the requested ``hidden_states`` indices (0 = embeddings). One
    layer is resident on ``device`` at a time → peak VRAM = one teacher layer.
    Targets are returned in fp32 on CPU (RESULTS.md: fp32 avoids the small-delta
    cancellation). Mirrors :func:`capture_boundaries` but indexes by hidden-state
    position (after-layer-k) rather than by layer input."""
    want = set(seam_idxs)
    deepest = max(seam_idxs)
    captured: dict[int, torch.Tensor] = {}
    hidden = embeds.to(device)
    if 0 in want:
        captured[0] = hidden.detach().float().to("cpu")
    for i, layer in enumerate(layers):
        layer.to(device)
        hidden = _layer_call(layer, hidden, **layer_kwargs)
        layer.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        hs_idx = i + 1  # hidden state AFTER layer i
        if hs_idx in want:
            captured[hs_idx] = hidden.detach().float().to("cpu")
        if hs_idx >= deepest:
            break
    return captured


def _distill_capture(args: Any, seam_map: dict[int, list[int]], cache_dir: Path) -> None:
    """Phase A: capture the teacher's per-seam hidden states to ``cache_dir``.

    Loads the teacher ALONE, streams it over the calibration batches, writes one
    safetensors per calib batch (``teacher_b{batch}.safetensors`` keyed by seam
    index) plus a manifest with token ids + dims, then frees the teacher.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(_resolve_device(getattr(args, "device", "auto")))
    dtype = _resolve_dtype(getattr(args, "dtype", "auto"), device)
    teacher_path = args.teacher_model
    if not Path(teacher_path).exists():
        sys.exit(f"ERROR: teacher model path not found: {teacher_path}")

    print(f"INFO[distill/capture]: teacher={teacher_path} device={device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers, _ = find_decoder_layers(model)
    l_teacher = len(layers)
    d_teacher = model.config.hidden_size
    embed = model.get_input_embeddings().to(device)
    seam_idxs = _seam_indices(seam_map)
    print(
        f"INFO[distill/capture]: teacher {l_teacher} layers, d={d_teacher}; "
        f"capturing seams {seam_idxs}",
        file=sys.stderr,
    )

    pairs = load_dataset(args.dataset)
    n_batches = max(1, getattr(args, "calib_batches", 8))
    # Rotary kwargs for direct layer calls (constant across batches — positions
    # 0..max_seq_len-1). Computed in the teacher's compute dtype.
    _dummy = torch.zeros(1, args.max_seq_len, d_teacher, device=device, dtype=dtype)
    layer_kwargs = _rotary_kwargs(model, _dummy, device)

    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_token_ids: list[list[list[int]]] = []
    for step in range(n_batches):
        batch = build_batch(pairs, tok, step, args.batch_size, args.max_seq_len)
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            emb = embed(ids).to(dtype)
        caps = _capture_teacher_seams(model, layers, seam_idxs, emb, device, layer_kwargs)
        # Persist this batch's seam tensors (fp32, CPU) keyed by seam index.
        state = {f"seam_{k}": v.contiguous() for k, v in caps.items()}
        save_file(state, str(cache_dir / f"teacher_b{step}.safetensors"))
        batch_token_ids.append(batch["input_ids"].tolist())
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"INFO[distill/capture]: batch {step+1}/{n_batches} cached", file=sys.stderr)

    manifest = {
        "teacher_model": str(Path(teacher_path).resolve()),
        "teacher_layers": l_teacher,
        "teacher_width": d_teacher,
        "teacher_vocab": int(getattr(model.config, "vocab_size", 0)),
        "seam_map": {str(k): v for k, v in seam_map.items()},
        "n_batches": n_batches,
        "max_seq_len": args.max_seq_len,
        "batch_token_ids": batch_token_ids,
    }
    (cache_dir / _SEAM_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    print(f"INFO[distill/capture]: wrote {n_batches} batch(es) + manifest → {cache_dir}", file=sys.stderr)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _load_seam_manifest(cache_dir: Path) -> dict[str, Any]:
    mpath = cache_dir / _SEAM_MANIFEST
    if not mpath.exists():
        sys.exit(
            f"ERROR[distill/train]: teacher seam cache not found at {mpath}. "
            "Run the capture phase first (--distill-phase capture)."
        )
    return json.loads(mpath.read_text(encoding="utf-8"))


def _distill_train(args: Any, cache_dir: Path) -> dict[str, Any]:
    """Phase B: train the student branch against the cached teacher seams.

    Loads the student ALONE (teacher is gone). For each student block, streams
    its boundary input over the SAME token ids the teacher saw, then trains the
    block's LoRA + a discard-after read-out projection to match the cached
    teacher seam (cosine+MSE in fp32). Saves the LoRA adapter (projection
    dropped) keyed by global layer index — so it merges/exports unchanged.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(_resolve_device(getattr(args, "device", "auto")))
    manifest = _load_seam_manifest(cache_dir)
    l_teacher = manifest["teacher_layers"]
    d_teacher = manifest["teacher_width"]
    seam_map = {int(k): v for k, v in manifest["seam_map"].items()}

    model_path = args.model
    if not Path(model_path).exists():
        sys.exit(f"ERROR: student model path not found: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    student_vocab = int(getattr(tok, "vocab_size", 0) or 0)
    teacher_vocab = int(manifest.get("teacher_vocab", 0))
    if teacher_vocab and student_vocab and teacher_vocab != student_vocab:
        sys.exit(
            f"ERROR[distill]: teacher/student tokenizer mismatch "
            f"(teacher vocab={teacher_vocab}, student vocab={student_vocab}). "
            "Seam distillation matches hidden states position-by-position and "
            "requires a SHARED tokenizer. Pick a same-family teacher/student "
            "pair, or use sequence-level (data) distillation instead."
        )

    # fp32 student (master weights — bf16 stalls small-delta learning per RESULTS.md).
    print(f"INFO[distill/train]: student={model_path} device={device} (fp32 master weights)", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers, layers_name = find_decoder_layers(model)
    n_layers = len(layers)
    d_student = model.config.hidden_size
    shards = plan_shards(n_layers, getattr(args, "shards", None), getattr(args, "block_size", None))
    print(
        f"INFO[distill/train]: student {n_layers} layers d={d_student} → teacher "
        f"{l_teacher} layers d={d_teacher}; {len(shards)} block(s): {shards}",
        file=sys.stderr,
    )

    embed = model.get_input_embeddings().to(device)
    target_modules = _resolve_targets(args, model)
    proj_mode = getattr(args, "projection", "auto") or "auto"
    loss_kind = getattr(args, "distill_loss", "cosine_mse") or "cosine_mse"
    # Stability: gradient clipping + dynamic per-block LR (auto) vs constant (fixed).
    grad_clip = float(getattr(args, "grad_clip", 1.0) or 0.0)
    lr_mode = getattr(args, "lr_mode", "auto") or "auto"
    ref_rms: float | None = None  # set from the first (shallowest) block
    # Rotary kwargs for direct layer calls (fp32 student; constant positions).
    _dummy = torch.zeros(1, args.max_seq_len, d_student, device=device, dtype=torch.float32)
    lk_student = _rotary_kwargs(model, _dummy, device)

    # Reconstruct the token-id batches the teacher saw (identical sequences).
    token_batches = [torch.tensor(b, dtype=torch.long) for b in manifest["batch_token_ids"]]
    n_batches = len(token_batches)

    out_dir = Path(args.out) if args.out else Path(args.dataset).parent / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    total_adapters = 0
    for shard_id, (a, b) in enumerate(shards):
        seam_targets_for_block = seam_map.get(b, [l_teacher])

        # Student boundary input (input to layer a), streamed per calib batch.
        per_batch_in: list[torch.Tensor] = []
        for ids in token_batches:
            ids_dev = ids.to(device)
            with torch.no_grad():
                emb = embed(ids_dev).float()
            caps = capture_boundaries(model, layers, [a], emb, device, lk_student)
            per_batch_in.append(caps[a])

        # Build this block + attach LoRA (all-linear auto targets recommended).
        block = nn.ModuleList([layers[i] for i in range(a, b)]).float()
        n_added = 0
        for li in range(len(block)):
            n_added += attach_lora(block[li], target_modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
        if n_added == 0:
            auto = auto_detect_targets(block)
            for li in range(len(block)):
                n_added += attach_lora(block[li], auto, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
            target_modules = auto
        if n_added == 0:
            sys.exit(f"ERROR[distill]: shard {shard_id}: zero LoRA adapters on layers {a}:{b}")
        block.to(device)
        total_adapters += n_added

        # Read-out projection bridging student→teacher width (scaffold, dropped).
        proj = build_projection(d_student, d_teacher, proj_mode).to(device).float()

        lora_params = [
            p for m in block.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)
        ]
        params = lora_params + list(proj.parameters())

        # DYNAMIC per-block LR (auto): scale the base LR by this block's teacher
        # seam magnitude relative to the shallowest block (which trains well at the
        # base LR). Deep, larger-magnitude seams ⇒ gentler steps ⇒ no divergence.
        tgt0 = _load_block_target(cache_dir, 0, seam_targets_for_block, device)
        target_rms = float(tgt0.float().pow(2).mean().sqrt().item())
        if ref_rms is None:
            ref_rms = target_rms
        if lr_mode == "auto":
            block_lr = args.lr * block_lr_scale(ref_rms, target_rms)
        else:
            block_lr = args.lr
        optimizer = torch.optim.AdamW(params, lr=block_lr)
        warmup = max(1, args.steps // 10) if lr_mode == "auto" else 0

        first_loss: float | None = None
        last_loss = 0.0
        for step in range(args.steps):
            if lr_mode == "auto":
                lr_t = lr_at_step(block_lr, step, args.steps, warmup)
                for g in optimizer.param_groups:
                    g["lr"] = lr_t
            bi = step % n_batches
            in_k = per_batch_in[bi].to(device).float()
            # Teacher target for this block = the cached seam(s), averaged.
            tgt = _load_block_target(cache_dir, bi, seam_targets_for_block, device)

            student_out = _run_block(block, in_k, lk_student, lora_enabled=True)
            pred = proj(student_out)
            loss = distill_loss(pred, tgt, loss_kind)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            lv = loss.item()
            if first_loss is None:
                first_loss = lv
            last_loss = lv
            if (step + 1) % args.log_every == 0 or step == 0:
                print(
                    f"distill shard {shard_id} (L{a}:{b}→seam{seam_targets_for_block}) "
                    f"step {step+1}/{args.steps} loss={lv:.6f} lr={block_lr:.2e}",
                    file=sys.stderr,
                )

        _save_shard_adapter(block, a, layers_name, out_dir, shard_id, args, target_modules,
                            str(Path(model_path).resolve()))
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
            "teacher_seams": seam_targets_for_block,
            "adapters": n_added,
            "target_rms": round(target_rms, 4),
            "block_lr": round(block_lr, 8),
            "first_loss": round(first_loss or 0.0, 6),
            "final_loss": round(last_loss, 6),
            "peak_vram_gb": peak,
        })

    return {
        "mode": "distill",
        "teacher_model": manifest["teacher_model"],
        "teacher_layers": l_teacher,
        "teacher_width": d_teacher,
        "student_width": d_student,
        "loss": loss_kind,
        "projection": "identity" if d_student == d_teacher or proj_mode == "none" else "student_up",
        "shards": all_summaries,
        "total_adapters": total_adapters,
        "out": str(out_dir.resolve()),
    }


def _load_block_target(
    cache_dir: Path, batch_idx: int, seam_idxs: list[int], device: torch.device
) -> torch.Tensor:
    """Load (and average, for block_avg) the cached teacher seam target(s) for
    one calib batch. Returns fp32 on ``device``."""
    from safetensors.torch import load_file

    state = load_file(str(cache_dir / f"teacher_b{batch_idx}.safetensors"))
    tensors = [state[f"seam_{i}"] for i in seam_idxs if f"seam_{i}" in state]
    if not tensors:
        sys.exit(f"ERROR[distill]: no cached seams {seam_idxs} in batch {batch_idx}")
    stacked = torch.stack(tensors, 0).mean(0) if len(tensors) > 1 else tensors[0]
    return stacked.to(device).float()


def train_distill(args: Any) -> None:
    """Entry for cross-model seam distillation (``--distill-mode``). Runs the
    requested phase(s): ``capture`` (teacher → cache), ``train`` (student ←
    cache), or ``both`` (default — capture then train sequentially; the teacher
    is freed before the student loads, so they are never co-resident)."""
    import random

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not getattr(args, "teacher_model", None):
        sys.exit("ERROR[distill]: --teacher-model is required for --distill-mode")

    cache_dir = Path(
        getattr(args, "teacher_cache", None)
        or (Path(args.out) if args.out else Path(args.dataset).parent / "adapter") / "distill_cache"
    )
    phase = getattr(args, "distill_phase", "both") or "both"

    if phase in ("capture", "both"):
        # Plan seams from the STUDENT's depth (read cheaply from its config).
        n_student, _, _ = _read_model_depth_width(args.model)
        l_teacher, _, _ = _read_model_depth_width(args.teacher_model)
        student_blocks = plan_shards(
            n_student, getattr(args, "shards", None), getattr(args, "block_size", None)
        )
        seam_map = plan_seam_map(student_blocks, l_teacher, getattr(args, "layer_map", "stride"))
        _distill_capture(args, seam_map, cache_dir)

    if phase in ("train", "both"):
        summary = _distill_train(args, cache_dir)
        print(json.dumps(summary))
    elif phase == "capture":
        print(json.dumps({"mode": "distill", "phase": "capture", "cache": str(cache_dir.resolve())}))


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
