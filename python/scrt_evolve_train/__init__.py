"""
scrt_evolve_train — real-model LoRA training path for scrt-evolve.

Loads a HuggingFace causal-LM, attaches hand-rolled LoRA adapters, trains
on a scrt-evolve dataset.jsonl with prompt-masked cross-entropy, and saves
the adapter as safetensors. No peft dependency.

Ported/adapted from lexame hivemind-models src/moe/expert_trainer.py.
"""
