"""
scrt_evolve_infer — inference module for scrt-evolve LoRA adapters.

Loads a HuggingFace causal-LM base model and optionally applies a trained
LoRA adapter (produced by scrt_evolve_train) for comparison / A-B evaluation.
Reuses scrt_evolve_train.trainer.LoRALinear for weight loading so adapter
tensors are applied with the exact same parameterisation used during training.
"""
