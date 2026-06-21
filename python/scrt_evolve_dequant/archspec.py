"""
archspec.py — the architecture registry for GGUF→HF conversion (track 23).

Generic, SDK-style, NO model/brand-specific logic in the converter. An
`ArchSpec` describes ONE architecture family: how its GGUF tensor names map to
HF names, and how GGUF metadata keys map to an HF `config.json`. The converter
(dequant.py) is arch-agnostic — it reads the GGUF's `general.architecture`,
looks the spec up in `REGISTRY`, and applies its RULES. Add support for a new
architecture by REGISTERING a spec, never by editing the converter.

Name mapping is RULE-BASED (regex with a `{N}` layer-index capture + a template
substitution), not a table of per-tensor literals — so a 40-layer and a 4-layer
model of the same family share one rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class NameRule:
    """One GGUF→HF tensor-name rule.

    `pattern` is a regex over the GGUF tensor name; it may capture a layer index
    as a group named `n`. `template` is the HF name with `{n}` substituted from
    that capture. A rule with no `{n}` is a fixed (non-layer) tensor.
    """
    pattern: str
    template: str

    def apply(self, gguf_name: str) -> str | None:
        m = re.fullmatch(self.pattern, gguf_name)
        if m is None:
            return None
        groups = m.groupdict()
        out = self.template
        if "n" in groups and groups["n"] is not None:
            out = out.replace("{n}", groups["n"])
        return out


@dataclass(frozen=True)
class ConfigKey:
    """Map a GGUF metadata key (with `{arch}` placeholder) to an HF config key,
    optionally transforming the value."""
    gguf_key: str
    hf_key: str
    transform: Callable[[object], object] | None = None


@dataclass(frozen=True)
class ArchSpec:
    """A registered architecture family."""
    # The HF `model_type` / architectures entry this maps to (informational +
    # written into the reconstructed config).
    hf_model_type: str
    hf_architectures: list[str]
    # Ordered tensor-name rules; first match wins.
    name_rules: list[NameRule]
    # GGUF metadata → HF config mappings.
    config_keys: list[ConfigKey] = field(default_factory=list)
    # Tensors (by GGUF name regex) intentionally dropped (e.g. rope freqs that HF
    # recomputes). Empty ⇒ keep everything that maps.
    drop_patterns: list[str] = field(default_factory=list)

    def map_tensor_name(self, gguf_name: str) -> str | None:
        """HF name for a GGUF tensor, or None if no rule matches."""
        for rule in self.name_rules:
            mapped = rule.apply(gguf_name)
            if mapped is not None:
                return mapped
        return None

    def is_dropped(self, gguf_name: str) -> bool:
        return any(re.fullmatch(p, gguf_name) for p in self.drop_patterns)


# ---------------------------------------------------------------------------
# The registry. Keyed on the GGUF `general.architecture` string.
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ArchSpec] = {}


def register(arch: str, spec: ArchSpec) -> None:
    """Register (or override) the spec for a GGUF architecture id."""
    REGISTRY[arch] = spec


def get(arch: str) -> ArchSpec | None:
    return REGISTRY.get(arch)


def supported() -> list[str]:
    return sorted(REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in specs. These are the GENERIC, reusable building blocks — a new model
# of an existing family needs NO new code; a new family is one register() call.
# ---------------------------------------------------------------------------

# The standard decoder-only attention block shared by llama/mistral/qwen2 and
# many others. Lifted as a reusable rule list so families compose it.
_LLAMA_LIKE_RULES = [
    NameRule(r"token_embd\.weight", "model.embed_tokens.weight"),
    NameRule(r"output_norm\.weight", "model.norm.weight"),
    NameRule(r"output\.weight", "lm_head.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.attn_norm\.weight", "model.layers.{n}.input_layernorm.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.attn_q\.weight", "model.layers.{n}.self_attn.q_proj.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.attn_k\.weight", "model.layers.{n}.self_attn.k_proj.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.attn_v\.weight", "model.layers.{n}.self_attn.v_proj.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.attn_output\.weight", "model.layers.{n}.self_attn.o_proj.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.ffn_norm\.weight", "model.layers.{n}.post_attention_layernorm.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.ffn_gate\.weight", "model.layers.{n}.mlp.gate_proj.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.ffn_up\.weight", "model.layers.{n}.mlp.up_proj.weight"),
    NameRule(r"blk\.(?P<n>\d+)\.ffn_down\.weight", "model.layers.{n}.mlp.down_proj.weight"),
]

_LLAMA_LIKE_CONFIG = [
    ConfigKey("{arch}.block_count", "num_hidden_layers"),
    ConfigKey("{arch}.embedding_length", "hidden_size"),
    ConfigKey("{arch}.feed_forward_length", "intermediate_size"),
    ConfigKey("{arch}.attention.head_count", "num_attention_heads"),
    ConfigKey("{arch}.attention.head_count_kv", "num_key_value_heads"),
    ConfigKey("{arch}.context_length", "max_position_embeddings"),
    ConfigKey("{arch}.vocab_size", "vocab_size"),
    ConfigKey("{arch}.attention.layer_norm_rms_epsilon", "rms_norm_eps"),
    ConfigKey("{arch}.rope.freq_base", "rope_theta"),
]


def _register_builtins() -> None:
    # llama family (llama / mistral share this GGUF layout).
    for arch in ("llama", "mistral"):
        register(
            arch,
            ArchSpec(
                hf_model_type=arch,
                hf_architectures=["LlamaForCausalLM" if arch == "llama" else "MistralForCausalLM"],
                name_rules=list(_LLAMA_LIKE_RULES),
                config_keys=list(_LLAMA_LIKE_CONFIG),
                drop_patterns=[r"rope_freqs\.weight"],
            ),
        )
    # qwen2 — same tensor layout, different model_type.
    register(
        "qwen2",
        ArchSpec(
            hf_model_type="qwen2",
            hf_architectures=["Qwen2ForCausalLM"],
            name_rules=list(_LLAMA_LIKE_RULES),
            config_keys=list(_LLAMA_LIKE_CONFIG),
            drop_patterns=[r"rope_freqs\.weight"],
        ),
    )


_register_builtins()
