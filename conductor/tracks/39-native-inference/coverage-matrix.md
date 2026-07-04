---
type: artifact
track: 39
title: Native inference coverage matrix — every registry arch family
produced_by: P0-5 (Lane 3)
status: complete
created: 2026-07-04
updated: 2026-07-04
---

# Track 39 — Architecture Coverage Matrix

> **Purpose:** enumerate every architecture family present in registry use,
> classify its native-candle servability, and name op gaps concretely.  This
> is the go/no-go input for P0-6 (retirement schedule).

## Sources consulted

- `crates/scrt-evolve/src/model.rs` — `arch_supported()`, `TinyLlama`,
  `LoadedModel`, `DecoderLayer`; the existing parity fixture test at line 739
  (`MambaForCausalLM` rejected).
- `crates/scrt-evolve/src/branch/manifest.rs` — `BranchManifest`,
  `RouterSignature`; the registry lists branches by `base_model` string, not
  an arch enum, so family membership is inferred from the HF architecture tag
  in the model's `config.json`.
- `spec.md` §Model-generic design (family matrix paragraph).
- candle-transformers upstream: known loaders as of 2026-07.

---

## Family classification

| # | Architecture family | HF `architectures` tag(s) | candle support | Status | Notes |
|---|---|---|---|---|---|
| 1 | **llama-family** | `LlamaForCausalLM`, `LlamaModel` | ✅ Loader + forward in `model.rs` (`TinyLlama`) | **`native`** | Day-1. `ScrtEvolveTinyCausalLM` is the internal fixture arch (same class). Covers TinyLlama, Llama-2/3, Mistral, Mistral-Instruct. |
| 2 | **Qwen2 / Qwen2.5** | `Qwen2ForCausalLM` | ✅ candle-transformers has a Qwen2 loader (GQA + rotary; llama-derived) | **`native`** | Architecture is close enough to llama-family that the same `ArchAdapter` wraps it; verify RoPE scaling variant at P0-4. |
| 3 | **Gemma / Gemma-2** | `GemmaForCausalLM`, `Gemma2ForCausalLM` | ✅ candle-transformers Gemma loader | **`native`** | Pre-norm + GeGLU variant; confirm logit-softcapping behaviour in parity harness (P0-4). |
| 4 | **Mistral / Mixtral** | `MistralForCausalLM`, `MixtralForCausalLM` | ✅ candle-transformers supports both (SWA + sliding window) | **`native`** (dense); **`needs-op-work`** (MoE/Mixtral) | Dense Mistral = llama-family variant, day-1. Mixtral MoE router (`num_experts`, sparse dispatch) is not yet wired in `ArchAdapter`; see row 7. |
| 5 | **Phi-2 / Phi-3** | `PhiForCausalLM`, `Phi3ForCausalLM` | ✅ candle-transformers loaders present | **`native`** | Phi-3 uses grouped-query attention + partial rotary; confirm head-count handling. |
| 6 | **Granite-4-h (hybrid)** | `GraniteHybridForCausalLM` *(IBM)* | ❌ Missing: Mamba2/SSD selective-scan op + hybrid block wiring | **`needs-op-work`** | **See §Granite gap below** — the load-bearing risk of the track. Preflight-**refused** until Phase B (PB-5) flips this to `native`. |
| 7 | **MoE families** (Mixtral, Qwen-MoE, DeepSeek-MoE) | `MixtralForCausalLM`, `Qwen2MoeForCausalLM`, `DeepseekV2ForCausalLM` | ❌ Sparse expert dispatch not in `ArchAdapter` v1 | **`refused`** | Out of v1 scope per spec ("MoE families = out of v1, listed for the trait's sake"). `branch create` preflight refuses these bases until a future phase wires the router boundary. |
| 8 | **Mamba pure-SSM** | `MambaForCausalLM` | ❌ Already rejected in `model.rs` line 739 fixture | **`refused`** | Existing test confirms rejection. Pure-SSM Mamba-1 lacks the transformer attention path; different kernel work than Granite's hybrid. Separate track if ever needed. |
| 9 | **Falcon** | `FalconForCausalLM` | ⚠️ candle-transformers has a loader but multi-query + alibi PE variant needs verification | **`needs-op-work`** | Not in current registry; listed for completeness. Alibi positional encoding differs from RoPE — needs parity test before clearing. |

---

## Retirement schedule input (for P0-6)

| Family | Ready for llama.cpp retirement? | Condition |
|---|---|---|
| llama-family (rows 1) | **Yes** — pending P0-4 parity gate | P0-4 green = cleared |
| Qwen2/Qwen2.5 (row 2) | **Yes** — pending RoPE scaling check at P0-4 | Verify then clear |
| Gemma / Gemma-2 (row 3) | **Yes** — pending softcapping check at P0-4 | Verify then clear |
| Dense Mistral (row 4) | **Yes** — pending P0-4 | Verify then clear |
| Phi-2/3 (row 5) | **Conditional** — head-count verification needed | Targeted parity check |
| Granite-4-h (row 6) | **No** — blocked on Phase B (PB-1→PB-3) | Stays `refused` until PB-5 |
| MoE families (row 7) | **No** — out of v1 | Future phase |
| Pure Mamba (row 8) | **No** | Separate effort |
| Falcon (row 9) | **No** — not in registry, alibi PE unverified | Not scheduled |

---

## Granite-4-h op gap (named concretely)

Granite-4-h is IBM's hybrid architecture interleaving **Mamba2 (SSD, structured
state-space, selective scan)** blocks with standard transformer attention blocks.
The two missing pieces for native serving:

### 1. Mamba2 / SSD selective-scan kernel (`PB-1`)

The core of Mamba2 is the **structured state-space duality (SSD) selective
scan**: a recurrent inner loop over the sequence where the state-transition
matrix is input-dependent (hence "selective"). The operation:

```
h_t = A_t ⊙ h_{t-1} + B_t x_t
y_t = C_t h_t + D x_t
```

where `A_t`, `B_t`, `C_t` are linear projections of the input (the
"selective" part). In Mamba2 the state is rearranged from Mamba1's 1-D state
into a multi-head structured block (semiseparable matrix / SSD form), enabling
a parallel scan over the chunked sequence. **This parallel scan has no candle
primitive** — it requires either:
- A custom CPU kernel (the Phase B correctness reference), or
- A CUDA kernel (Phase B fast path, `ssd_cuda.rs`).

The naive forward (loop over sequence) is correct but O(L²) in memory. The
parallel chunked-scan is O(L log L) and is what the Python `mamba-ssm` C++/CUDA
library implements. **A CPU O(L) sequential reference is the correctness
oracle**; the CUDA fast path follows later.

### 2. Hybrid block wiring (`PB-2`)

Granite-4-h interleaves SSM blocks and attention blocks at a fixed stride (the
exact interleaving pattern is in the `config.json` `layer_types` field). The
`GraniteAdapter` must:
- Read `layer_types` from the HF config to build the correct `layers()` sequence
  (`Ssm | Attn | Mlp`).
- Route `forward_layer(i, x)` through the Mamba2 scan path (PB-1) for `Ssm`
  layers and through the existing attention path for `Attn` layers.
- Expose the Mamba→attention boundary as a `SeamPoint` (the structural joint
  used by `bench/seam_distill`).

Until Phase B: Granite `branch create` is **preflight-refused** with the message
`"arch GraniteHybridForCausalLM: Mamba2/SSD selective-scan not yet implemented
in the native engine (track 39 Phase B)"`. No silent fallback to llama.cpp.

---

## Branch registry snapshot

Current `branches/registry.json` (ML-free inspection via `evolve branch list`)
lists only branches with a `base_model` in the **llama-family** (TinyLlama
variants used by the track-19/29 test corpus). No Granite or MoE branches exist
in the registry at the time of this matrix authorship — the preflight gate (PC-4)
will enforce the invariant before any new family enters the registry.

---

*TODO (Lane 1 / P0-4)*: After the parity harness runs, annotate each `native`
row with the measured greedy-token match rate and any divergence cause. That
data populates `parity-report.md`.
