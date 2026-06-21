"""scrt_evolve_dequant — generic GGUF → HF safetensors converter (track 23).

Architecture-agnostic: an `ArchSpec` registry (archspec.py) describes each
architecture family's GGUF→HF tensor-name + config-key maps via rules; the
converter (dequant.py) drives the registry. Add a new architecture by
registering a spec, never by editing the converter. Streaming (one tensor at a
time → bounded memory). NO model/brand-specific logic.
"""

from scrt_evolve_dequant import archspec
from scrt_evolve_dequant.dequant import dequantize_to_hf

__all__ = ["dequantize_to_hf", "archspec"]
