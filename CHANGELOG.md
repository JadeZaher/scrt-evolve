# Changelog

All notable changes to scrt-evolve will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- **Discover stage** (track 01): search a corpus + scrt mind-palace and produce
  a structured `DiscoveredContext` with simhash dedup and clustering. Exposed as
  `evolve train discover` and the `discover::run` SDK entry point.

- **Generate stage — API teacher backend** (track 02): turn discovered context
  into a QA/instruction dataset via any OpenAI-compatible chat endpoint. The
  `GenBackend` trait + `ApiEndpoint` impl drive prompt templates and output the
  cross-language `dataset.jsonl` contract. Exposed as `evolve train generate` /
  `evolve train run` and the `generate::run` SDK entry point.

- **Generate stage — local candle fixture backend** (track 03): a `LocalCandle`
  implementation of the `GenBackend` trait for mechanical validation (cannot
  load real pretrained models). Gated by the `train` Cargo feature.

- **Training stage — LoRA via candle fixture** (track 04): a `TrainingPreset`
  trait and LoRA injection + training loop producing `adapter.safetensors` for
  mechanical validation. Gated by the `train` feature.

- **Real ML path — transformers training, inference, scoring, and GGUF export**
  (track 19): standalone Python packages (`scrt_evolve_train`,
  `scrt_evolve_infer`, `scrt_evolve_score`, `scrt_evolve_gguf`,
  `scrt_evolve_dequant`) driven from the Rust CLI via subprocess over the
  `dataset.jsonl` contract. The validated production path. Exposed as
  `--backend transformers` on `train fit`, `model infer`, `train eval`,
  `train export-gguf`, and `dequant`.

- **Eval harness** (track 10): `ProbeSet` carve from the dataset, `Scorer` with
  `api`/`transformers` backends, `StepVerdict` (accept/regress/catastrophic),
  and the executable eval gate. Exposed as `evolve train probe build`,
  `evolve train eval`, and `ProbeSet::carve` + `eval::run_eval` in the SDK.

- **Self-regulation — transactional homeostasis** (track 15): every
  weight-mutating step runs inside a checkpoint → evaluate → keep|rollback
  transaction. Catastrophe triggers auto-rollback + quarantine by
  `gen`-provenance + halt. Exposed as `evolve watch checkpoints
  {list,show,restore}`, `evolve watch quarantine {list,clear}`, and the
  `Regulator` SDK type.

- **Learning-by-doing — multi-goal eval-gated rounds** (track 20): `[[goals]]`
  in `evolve.toml`, a paired `scrt-evolve` skill that stashes goal-tagged
  findings, and a bounded `round-robin`/`weighted` scheduler that runs
  discover→generate→train→eval→keep|rollback through the track-15 regulator.
  Exposed as `evolve train auto --schedule`.

- **Branch factory — Branch-Train-Merge** (track 29): `branch create` composes
  discover → teacher-QA generate → train → eval gate → GGUF export inside the
  regulator transaction; `branch {list,register,route,serve}` manage a local
  fleet with a `BranchRouter` resolving requests per-request. Manifest +
  `branches/registry.json` form the cross-repo contract with hivemind. Live-
  validated on TinyLlama-1.1B.

- **Config-driven GGUF export** (track 27): merge adapter → f16 → quantize →
  place. Exposed as `evolve train export-gguf`.

- **Fractional / microshard training** (track 25): train one contiguous
  layer-block at a time via block-local distillation, bounding peak VRAM to a
  single block. Verified on Granite-4.0-h-tiny at ~3.1–3.4 GB.

- **Quantized-base / QAT** (track 23): GGUF→HF dequant + quantization-aware
  training toward the deployment quant. Exposed as `evolve train dequant` and
  `[train.qat]`.

- **Ambient daemon** (track 26): a two-lane living queue + VRAM-gated background
  loop that consumes the corpus, trains through the track-15 transaction, and
  idles on empty. Exposed as `evolve ambient {start,stop}`, `evolve watch
  status`, and `evolve ambient teach`.

- **Ambient daemon hardening** (track 31): judge-model preflight, content-hash
  dedup ledger, transient-vs-catastrophe retries, wall-clock training budget,
  probe-correctness trend. Exposed as `evolve watch health`, `evolve watch
  trend`, and `evolve doctor --ambient`.

- **Regression gate — LLM-judge no-degradation** (track 32): opt-in
  `[regulate].gate = "judge"` that samples base BEFORE vs base+adapter AFTER on
  probe prompts and accepts unless the judge sees degradation; correctness
  demoted to catastrophe-only. A `min_train_pairs` floor
  (`[daemon].min_train_pairs`, default 4) skips+accumulates below N.

- **Generation modalities — skill ingestion + reasoning edits** (track 09):
  `Skill` and `ReasoningEdit` dataset variants threaded through generate →
  dataset → probe/score → branch/train and export, with planner routing for
  both (via track 37 phase C).

- **Training-signal hardening** (track 37): dataset contract v1.1
  (`outcome`/`judge_score`/`tier`/`chosen_over` additive fields), outcome-
  stamped ingest with retry-collapse + `rejected.jsonl` sidecar, `LlmPairJudge`
  + `evolve dataset judge`, planner routes `skill`/`reasoning_edit`, `[domain]`
  parameterization, `evolve dataset expand` (Evol/Self-Instruct). Absorbs track
  35 nudge (steerable loop — `evolve ambient nudge`). All Rust tested ML-free.

- **Packaging + interpreter binding** (track 28): `scrt-evolve-ml` pyproject
  with `cpu`/`cuda` extras + console scripts; `--python > $SCRT_EVOLVE_PYTHON >
  [hardware].python` resolver preferring the installed package; `evolve doctor`
  preflight (torch/cuda/mamba/model/llama.cpp/work_dir PASS/FAIL + fix);
  `PORTABILITY.md`.

- **Benchmarks + taste config** (tracks 21, 24): `[evolve].constitution`/`taste`
  steering fields composed via `compose_steering()`; benchmark suite in `bench/`
  with a documented WSL2+CUDA runbook (`bench/RUNBOOK.md`).

- **Project auto-detect** (`evolve <project>`): auto-locates palace + corpus
  from a project name.

- **Self-routed plan + gap-critic generation**: `evolve train plan` + `generate
  --self-route` for planner-driven dataset creation with gap detection.

- **CLI surface**: `evolve init`, `config reference`, `config show`, `config
  dataset`, `commands`, global `--json` for machine-readable summary lines,
  `doctor`, `dequant`, `model infer`, `model run`.

- **SDK-first architecture**: every capability is a library function returning a
  `Serialize` report; the CLI is a thin argv→SDK shim. Heavy/impure work
  (Python subprocesses, GPU) is injected as hooks so the Rust orchestration
  stays ML-free and unit-testable.

### Changed

- **Initial release.** No prior version to diff against.

### Not yet shipped (roadmap)

These tracks are designed but not built as of 0.1.0:

- **Track 08 extract-publish** — swap scrt-core git dep → published crate; cut
  the tagged release (this changelog is a step toward it).
- **Track 36 install-ux** — CI builds, install scripts, `evolve setup`.
- **Track 39 native-inference** — candle-native serving engine, retiring the
  llama.cpp sidecar dependency.
- **Track 40 delegation-contract** — cross-repo evolve⇄lexame capability-card +
  daisy-chain contract.

Tracks 05/06/07/11–14/16–18/22 are **archived** (superseded or speculative; see
`conductor/tracks/_archived/`). They were never built and are not planned for
future releases.