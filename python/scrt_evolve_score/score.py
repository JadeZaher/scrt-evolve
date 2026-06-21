"""score.py — probe scoring against a real model forward pass.

The Rust eval harness (track 10) shells out here for the `transformers` scorer
backend. We load the base model (+ optional adapter, reusing the infer path),
generate a completion per probe item, and compute three signals:

  correctness    — executable gate (tool_call/cli) + reference match (qa/instr),
  perplexity     — mean per-token perplexity over each item's reference text,
  mean_exit_depth— logit-lens early-exit proxy: the mean fraction of layers
                   after which the model's top-1 next-token prediction already
                   matches its final-layer prediction (lower = cheaper path).

Output contract: the LAST stdout line is a JSON ScoreReport matching the Rust
`eval::score::ScoreReport` struct. Progress/info go to stderr.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from scrt_evolve_infer.infer import apply_adapter, generate, load_base_model


# ---------------------------------------------------------------------------
# Executable gate (Python mirror of eval/gate.rs — kept minimal + in sync)
# ---------------------------------------------------------------------------

# Real scrt tool schemas (name -> (required, all-props)). Mirror of toolspec.
TOOL_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    "scrt_search": (
        {"pattern"},
        {"pattern", "in", "cmd", "url", "effort", "max_tokens", "max_nodes",
         "clip_chars", "sort", "window_curve", "retriever", "mp_from",
         "mp_stash", "mp_tag", "mp_ttl", "page", "page_size"},
    ),
    "scrt_stash": ({"name", "note"}, {"name", "note", "tags", "replace", "ttl", "palace_path"}),
    "scrt_list_stashes": (set(), {"tag_filter", "palace_path"}),
    "scrt_get_stash": ({"name"}, {"name", "with_nodes", "palace_path"}),
    "scrt_drop_stash": ({"name"}, {"name", "palace_path"}),
    "scrt_similar": (set(), {"name", "term", "match", "score", "top", "palace_path"}),
}

# Real scrt CLI long flags (mirror of eval/gate.rs real_scrt_flags()).
REAL_FLAGS = {
    "--in", "--cmd", "--url", "--effort", "--max-tokens", "--max-nodes", "--clip",
    "--sort", "--window-curve", "--format", "--retriever", "--page", "--page-size",
    "--mp-stash", "--mp-ttl", "--mp-tag", "--mp-stash-tag", "--mp-from", "--mp-compose",
    "--mp-intersect", "--mp-except", "--mp-graph", "--mp-link", "--mp-similar",
    "--mp-prune", "--mp-prune-keep", "--mp-prune-tag", "--mp-prune-expired",
    "--mp-prune-older-than", "--mp-prune-dry-run", "--mp-list", "--mp-list-search",
    "--mp-list-tag", "--mp-find", "--mp-get", "--mp-drop", "--term", "--match",
    "--score", "--top", "--all", "--fuzzy", "--json", "--help", "--version",
    "--no-ignore", "--hidden", "--ignore-case", "--serve",
}


def _strip_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t[3:]
        nl = t.find("\n")
        if nl != -1 and t[:nl].strip().isalnum():
            t = t[nl + 1:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _gate_cli(command: str) -> bool:
    import re
    cmd = command.strip()
    if not cmd:
        return False
    first = cmd.split()[0]
    bin_ = first.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if bin_.endswith(".exe"):
        bin_ = bin_[:-4]
    if bin_ != "scrt":
        return False
    flags = {f.split("=")[0] for f in re.findall(r"--[a-z][a-z0-9-]*", cmd)}
    return flags <= REAL_FLAGS


def _gate_tool_call(answer: str, fallback_tool: str) -> bool:
    txt = _strip_fences(answer)
    tool = fallback_tool
    args: dict[str, Any] = {}
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            tool = obj.get("tool", fallback_tool)
            a = obj.get("arguments", {})
            if isinstance(a, dict):
                args = a
    except (json.JSONDecodeError, ValueError):
        pass
    if tool not in TOOL_SCHEMA:
        return False
    required, props = TOOL_SCHEMA[tool]
    if not required <= set(args.keys()):
        return False
    return set(args.keys()) <= props


def _reference_match(answer: str, reference: str) -> bool:
    norm = lambda s: " ".join(s.split()).lower()
    a, r = norm(answer), norm(reference)
    if not r:
        return False
    return a == r or r in a or a in r


def _probe_prompt(item: dict) -> str | None:
    kind = item.get("kind")
    if kind == "qa":
        return item.get("prompt")
    if kind == "instruction":
        instr, inp = item.get("instruction", ""), item.get("input", "")
        return instr if not inp.strip() else f"{instr}\n\n{inp}"
    if kind in ("tool_call", "cli"):
        return item.get("prompt")
    return None  # completion / contrastive — unscorable for correctness


def _judge(item: dict, answer: str) -> bool:
    kind = item.get("kind")
    if kind == "cli":
        return _gate_cli(_strip_fences(answer))
    if kind == "tool_call":
        return _gate_tool_call(answer, item.get("tool", ""))
    if kind == "qa":
        return _reference_match(answer, item.get("completion", ""))
    if kind == "instruction":
        return _reference_match(answer, item.get("output", ""))
    return False


# ---------------------------------------------------------------------------
# Perplexity + exit-depth (real forward pass)
# ---------------------------------------------------------------------------

def _reference_text(item: dict) -> str | None:
    """The text whose perplexity/exit-depth we measure for this item."""
    kind = item.get("kind")
    if kind == "qa":
        return item.get("completion")
    if kind == "instruction":
        return item.get("output")
    if kind == "cli":
        return item.get("command")
    if kind == "completion":
        return item.get("text")
    return None


def _perplexity_and_exit_depth(model, tokenizer, text: str) -> tuple[float | None, float | None]:
    """Mean per-token perplexity + a logit-lens early-exit fraction for *text*.

    Exit-depth: for each position, find the earliest transformer layer whose
    logit-lens top-1 next-token already equals the final-layer top-1. The mean
    (earliest_layer / n_layers) across positions is a cheap "how deep did it
    need to go" proxy. Requires output_hidden_states + a tied/′lm_head′ unembed.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return None, None

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    logits = out.logits  # [1, T, V]

    # --- Perplexity: shift so token t predicts token t+1 ---
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    tok_lp = log_probs[range(shift_labels.shape[0]), shift_labels]
    mean_nll = -tok_lp.mean().item()
    ppl = math.exp(mean_nll) if mean_nll < 30 else float("inf")

    # --- Exit-depth via logit-lens over hidden_states ---
    exit_frac = None
    hs = out.hidden_states  # tuple(len = n_layers+1) of [1, T, H]
    unembed = _get_unembed(model)
    if hs is not None and unembed is not None and len(hs) > 2:
        n_layers = len(hs) - 1  # hs[0] is embeddings
        final_top1 = logits[0, :-1, :].argmax(dim=-1)  # [T-1]
        positions = final_top1.shape[0]
        # earliest layer whose logit-lens top1 matches the final top1, per position
        earliest = torch.full((positions,), n_layers, dtype=torch.long)
        matched = torch.zeros(positions, dtype=torch.bool)
        for layer in range(1, len(hs)):
            h = hs[layer][0, :-1, :]
            layer_top1 = (h @ unembed.T).argmax(dim=-1)
            now = (layer_top1 == final_top1) & (~matched)
            earliest[now] = layer
            matched |= now
        depth_sum = (earliest.float() / float(n_layers)).sum().item()
        exit_frac = depth_sum / positions if positions else None

    return ppl, exit_frac


def _get_unembed(model) -> torch.Tensor | None:
    """The unembedding matrix [V, H] for logit-lens (lm_head weight, often tied)."""
    head = getattr(model, "lm_head", None)
    if head is not None and hasattr(head, "weight"):
        return head.weight.detach()
    return None


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def _load_probe(probe_path: str) -> list[dict]:
    items = []
    for line in Path(probe_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def score_probe(
    model_path: str,
    probe_path: str,
    adapter_dir: str | None = None,
    max_new_tokens: int = 64,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Score a model against a probe set. Returns a ScoreReport dict."""
    metrics = metrics or ["correctness", "perplexity", "mean_exit_depth"]
    items = _load_probe(probe_path)

    model, tokenizer = load_base_model(model_path)
    if adapter_dir:
        print(f"INFO: applying adapter from {adapter_dir}", file=sys.stderr)
        apply_adapter(model, adapter_dir)

    correct = 0
    scored = 0
    ppl_vals: list[float] = []
    depth_vals: list[float] = []

    for i, item in enumerate(items):
        prompt = _probe_prompt(item)
        if prompt is not None and "correctness" in metrics:
            scored += 1
            try:
                answer = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
                if _judge(item, answer):
                    correct += 1
            except Exception as e:  # one bad item must not abort the run
                print(f"WARN: item {i} generation failed (counted incorrect): {e}", file=sys.stderr)

        ref = _reference_text(item)
        if ref and ("perplexity" in metrics or "mean_exit_depth" in metrics):
            try:
                ppl, depth = _perplexity_and_exit_depth(model, tokenizer, ref)
                if ppl is not None and math.isfinite(ppl):
                    ppl_vals.append(ppl)
                if depth is not None:
                    depth_vals.append(depth)
            except Exception as e:
                print(f"WARN: item {i} forward-pass scoring failed: {e}", file=sys.stderr)

    correctness = (correct / scored) if scored else 0.0
    report = {
        "correctness": correctness,
        "n": scored,
        "probe_version": "subprocess",  # Rust side overwrites with its own version
        "backend": "transformers",
    }
    if ppl_vals:
        report["perplexity"] = sum(ppl_vals) / len(ppl_vals)
    if depth_vals:
        report["mean_exit_depth"] = sum(depth_vals) / len(depth_vals)

    return report
