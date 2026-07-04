"""scrt_evolve_score — real-forward-pass probe scorer (track 10).

Loads a HuggingFace causal-LM (+ optional LoRA adapter), scores it against a
held-out probe set (`probe.jsonl`, the scrt-evolve dataset schema), and prints a
ScoreReport JSON line on stdout for the Rust CLI to parse. Computes:
  - correctness   : executable gate over tool_call/cli items + reference match
                    for qa/instruction (generated completions).
  - perplexity    : mean token perplexity over the probe completions.
  - mean_exit_depth: logit-lens early-exit estimate (fraction of layers needed
                    to commit to the final token), a cheap exit-depth proxy.

Mirrors scrt_evolve_infer's loader/generate path; subprocess-driven (no pyo3).
"""

from scrt_evolve_score.score import score_probe

__all__ = ["score_probe"]
