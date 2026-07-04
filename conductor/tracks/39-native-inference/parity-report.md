---
type: artifact
track: 39
title: Native inference parity report
produced_by: P0-6 (scaffolded by Lane 3; parity numbers filled by Lane 1 / P0-4)
status: scaffold — awaiting P0-4 parity harness results
created: 2026-07-04
updated: 2026-07-04
---

# Track 39 — Parity Report

> **Purpose:** record whether the native candle forward matches the Python
> `scrt_evolve_infer` reference (greedy, fixed seed) on the probe set, per
> architecture family and per leg (safetensors base+LoRA, Q4_K_M GGUF).  This
> is the go/no-go gate that decides the llama.cpp retirement schedule.

---

## How to read this report

- **Match rate:** fraction of probe positions where the native greedy token
  equals the Python greedy token (fixed seed, `temperature = 0`).
- **Cleared:** `yes` means this family's match rate is ≥ the acceptance
  threshold AND no systematic divergence was found. A cleared family proceeds
  to llama.cpp retirement in Phase A.
- **Divergence log:** any token mismatch is logged with the prompt prefix,
  expected token, and actual token. Systematic patterns (tokenizer delta,
  RoPE scaling, sampling) are root-caused here.

Acceptance threshold: **100% greedy token match** on the probe set (fixed seed,
greedy argmax). Any divergence must be root-caused and resolved or documented as
an accepted delta (with justification) before a family is cleared.

---

## Probe set

<!-- TODO (Lane 1 / P0-4): fill in probe details -->
- **Probe file:** `crates/scrt-evolve/tests/fixtures/probe_prompts.jsonl`
  *(to be committed by P0-4)*
- **Prompt count:** TODO
- **Max new tokens:** TODO (fixed across all families)
- **Seed:** TODO (SplitMix64 seed used in `LocalCandle`)
- **Python reference:** `python -m scrt_evolve_infer --ab --temperature 0`
  output captured to `crates/scrt-evolve/tests/fixtures/probe_expected.jsonl`
  *(to be committed by P0-4)*

---

## Results by family

### llama-family (TinyLlama — safetensors base + LoRA leg)

| Metric | Value |
|---|---|
| Match rate (base, no adapter) | TODO — fill after `cargo test --features train serve_parity` |
| Match rate (base + adapter) | TODO |
| Divergence count | TODO |
| Root cause (if any) | TODO |
| **Cleared for retirement?** | TODO |

### llama-family (TinyLlama — Q4_K_M GGUF leg, P0-3)

| Metric | Value |
|---|---|
| Match rate vs Python `--ab` base | TODO |
| Divergence count | TODO |
| Root cause (if any) | TODO |
| **Cleared for retirement?** | TODO |

### Qwen2 / Qwen2.5

| Metric | Value |
|---|---|
| Match rate | TODO — pending RoPE scaling variant check |
| **Cleared for retirement?** | TODO |

### Gemma / Gemma-2

| Metric | Value |
|---|---|
| Match rate | TODO — pending logit-softcapping check |
| **Cleared for retirement?** | TODO |

### Dense Mistral

| Metric | Value |
|---|---|
| Match rate | TODO |
| **Cleared for retirement?** | TODO |

---

## Known divergence patterns (to-be-filled)

<!-- TODO (Lane 1 / P0-4): document any divergences here with root cause -->

| Pattern | Affected families | Root cause | Resolution |
|---|---|---|---|
| (none yet) | — | — | — |

---

## Go/no-go decision (P0-6 gate)

<!-- TODO (reviewer / P0-6): complete after P0-4 parity harness is green -->

**Decision date:** TODO

**Families cleared for Phase A retirement:**
- TODO (list each cleared family + evidence commit SHA)

**Families staying preflight-refused:**
- Granite-4-h — blocked on Phase B (PB-1 Mamba2/SSD scan). See
  `coverage-matrix.md §Granite gap`.
- MoE families — out of v1 scope.
- Pure Mamba — separate effort.

**Retirement schedule:**
- Phase A: retire llama.cpp for [TODO: cleared families] — execute
  `retirement-checklist.md`.
- Phase B: Granite cleared once PB-3 parity passes — flip
  `coverage-matrix.md` row 6 to `native`.

**Reviewer sign-off:** TODO

---

*Scaffold authored by Lane 3 (P0-5). Parity numbers, divergence log, and
go/no-go decision to be filled by Lane 1 after P0-4 (`serve_parity.rs`) runs
green.*
