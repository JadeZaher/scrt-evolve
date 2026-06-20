"""
scrt_evolve_gguf — merge a LoRA adapter into a HuggingFace base model and
export a quantized GGUF file for use in LM Studio / llama.cpp.

3-stage pipeline:
  1. MERGE  — attach LoRALinear adapters, call merge_and_unload(), save merged
              HF model to a temp dir.
  2. CONVERT — shell out to llama.cpp/convert_hf_to_gguf.py to produce an f16
               GGUF.
  3. QUANTIZE — shell out to llama-quantize(.exe) to produce a quantized GGUF
                at the requested quant type (Q4_K_M, Q8_0, …).

Reuses scrt_evolve_train.trainer.LoRALinear and attach_lora verbatim.
"""

from .export import export_gguf, find_llama_cpp

__all__ = ["export_gguf", "find_llama_cpp"]
