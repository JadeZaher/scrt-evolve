# Track 23 — Quantized-Base Training (GGUF→HF dequant + QAT) — SIGN-OFF

Date: 2026-06-20

Lets evolve train fit models that exist on disk as quantized GGUFs, and adds
quantization-aware training so the LoRA adapter is learned against the
deployment quant. Built **generic at the architecture level** (no model/brand
logic), SDK-style, with an internal registry — per the user mandate.

## Delivered

### Generic GGUF→HF dequant converter (`python/scrt_evolve_dequant/`)
- **`archspec.py` — the registry.** `ArchSpec` describes one architecture family
  via RULE-BASED maps: `NameRule` (regex with a `{n}` layer capture → HF name
  template) + `ConfigKey` (GGUF metadata → HF config). `REGISTRY` is keyed on the
  GGUF `general.architecture`. Add an architecture by `register()`-ing a spec —
  **never by editing the converter**. Shipped specs: `llama`, `mistral`, `qwen2`
  (sharing a reusable llama-like rule set).
- **`dequant.py` — the generic converter.** Reads the arch id, looks up the spec,
  applies its rules. **Streaming**: dequantizes + writes tensors ONE AT A TIME
  into size-bounded shards (the user's "dequantize in parts" — peak memory ≈ one
  tensor). Emits a sharded HF dir + index + reconstructed `config.json` stamped
  `_dequantized_from_gguf` (honest about lossiness). Unknown arch / unmapped
  tensors → clear errors, not silent breakage.
- **`__main__.py`** — `--gguf/--out/--dtype/--tokenizer/--list-arch`.
- **`evolve train dequant`** Rust shim — subprocess, auto-detects the vendored
  `gguf-py` on PYTHONPATH (mirrors export-gguf). No model logic in Rust.

### QAT — quantization-aware training (`python/scrt_evolve_train/qat.py`)
- **`fake_quantize` + `_FakeQuantSTE`**: group-wise affine quant→dequant of the
  effective LoRA weight with a straight-through estimator (backward = identity),
  so the adapter compensates for deployment quant. Generic quant-family math
  (`quant_bits` maps any GGUF quant name → bit width); no model specifics.
- **`Calibrator`**: bounded per-group absmax calibration over N batches, then
  frozen scales. `CalibConfig` carries the settings.
- Wired into `LoRALinear.forward` (QAT path when `qat_quant` set) + the train
  loop (`--qat`, `--qat-group-size`, `--qat-calibrate`); calibrator ticks per step.

### Generic LoRA targeting (`auto_detect_targets`)
- `--target-modules auto` enumerates the model's `nn.Linear` leaves and ranks
  projection names by cross-layer frequency — so hybrid/MoE arches
  (granitemoehybrid: Mamba SSM + 64-expert MoE, where q_proj/v_proj cover almost
  nothing) train without hardcoded names. Falls back to auto if explicit targets
  match nothing.

### `[train.qat]` config
- `QatConfig` on `TrainConfig.qat` (additive, serde-default); `train --backend
  transformers` forwards the flags. Absent ⇒ plain LoRA (non-breaking).

## Acceptance evidence
- **Python (`python/tests/test_track23.py`, 5/5):** archspec name rules (layer
  substitution, fixed tensors, unmatched→None, drop patterns), registry
  unknown-arch, QAT STE (forward quantizes / backward identity), calibration
  bounded + frozen, auto-detect targets (frequent proj ranks above rare;
  lm_head excluded).
- **Rust:** `qat_config_round_trips_and_absent_is_none`; full suite 19/19;
  `dequant` CLI parses; `clippy -D warnings` + `fmt --check` clean;
  `--features train` + `--features pyo3` build green.
- **Smoke:** `python -m scrt_evolve_dequant --list-arch` → `[llama, mistral,
  qwen2]`; QAT/trainer modules import under the torch venv.

## Major scope correction (recorded)
The bench target Granite-4.0-h-tiny is `granitemoehybrid` and its full-precision
HF safetensors are ALREADY cached locally. So the BENCH (track 24) trains the
cached f16 HF directly — no lossy dequant needed for Granite. The dequant
converter is a general capability for HF-less models; the granitehybrid GGUF→HF
spec is a documented seam (not required, and the hybrid MoE/SSM tensor layout is
non-trivial to map — out of scope until a real HF-less hybrid GGUF needs it).

## Deferred / seams (documented)
- Full real-model GGUF→HF→train→merge→GGUF round-trip — exercised in track 24,
  not CI (no small vanilla LLM GGUF on disk).
- `granitehybrid` (and other hybrid/MoE) ArchSpec — register when an HF-less
  hybrid GGUF actually needs dequant.
- GGUF tokenizer extraction — `--tokenizer <hf-dir>` fallback is the documented
  path (copy the matching HF tokenizer).

## Carry-forward
Track 23 + the completed evolve lane (10/15/20-gated) mean track 24 (the bench)
can now: point at cached f16 HF Granite, harvest `.claude/projects/` transcripts,
generate goals, and run the bounded eval-gated schedule with QAT toward a Q4_K_M
GGUF export. Final track.
