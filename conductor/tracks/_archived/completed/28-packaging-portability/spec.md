---
type: Track Spec
title: "Packaging & Portability"
description: Ship scrt-evolve as a portable LLM-training + local-inference package.
tags: [track-28, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 28 — Packaging & Portability — Spec

## Goal
Make scrt-evolve shippable and portable as the opinionated LLM-training + local-
model-tooling framework it has become — WITHOUT pretending it can be a single
frozen exe. Ship the **Rust CLI as a native binary** + the **Python ML half as a
pip/uv-installable package**, bound at runtime by a **configured venv
interpreter**. Make environment failures legible via a `doctor` preflight.

## The constraint (why not one exe)
The CLI is a real native binary (default build is ML-free + Python-free by
styleguide — that part already ships cleanly). But the ML subprocesses
(`scrt_evolve_train/_gguf/_score/_dequant`) require torch + transformers and,
for hybrid-SSM models, COMPILED CUDA kernels (mamba-ssm/causal-conv1d — no
Windows wheels; built from source this session). PyInstaller/Nuitka cannot freeze
a multi-GB, platform-specific CUDA stack. So the two halves must ship separately
and bind at runtime. (Decision: pip/uv package, not Docker.)

## Scope (to build)
1. **Python package `scrt-evolve-ml`** — pyproject.toml packaging the `python/`
   modules (train/gguf/score/dequant) with extras:
   - `[cpu]` → torch CPU + transformers + peft + safetensors + gguf
   - `[cuda]` → CUDA torch (index-url note) + the above; mamba kernels documented
     as a build-from-source step (the ecosystem gap — they have no universal
     wheels). Console entry points: `scrt-evolve-train` etc.
2. **CLI↔package binding = installed module via venv.** Resolve the interpreter
   from (a) `[hardware].python` / `$SCRT_EVOLVE_PYTHON`, then run
   `python -m scrt_evolve_train` against the INSTALLED package. RETIRE
   `python_pkg_dir()` (the walk-up-to-find-`python/` dev hack) as the primary
   path — keep it only as a checkout fallback. Add `[hardware].python` config.
3. **`scrt-evolve doctor`** — preflight that checks: interpreter resolvable;
   torch import + `cuda.is_available()` + version; transformers/peft/gguf present;
   mamba/causal-conv1d importable (for hybrid training); a current llama.cpp with
   the needed arch converter; free disk on scratch + place targets. Prints
   exactly what's missing and the fix command. Generic (kernel/arch-keyed).
4. **Env contract doc** — `PORTABILITY.md`: the matrix of OS × accelerator ×
   what's needed, the WSL2 path for Windows+hybrid (the only working route found
   this session), and the known ecosystem gaps (mamba wheels, llama.cpp arch lag).
5. **Release wiring** — Rust binary artifacts per platform; the Python package to
   an index (or git+pip). Version the contract between them.

## Acceptance
- `pip install scrt-evolve-ml[cuda]` into a venv, set `[hardware].python` (or
  `$SCRT_EVOLVE_PYTHON`), and `scrt-evolve train/export-gguf` work from an
  INSTALLED package (no checkout `python/` dir needed).
- `scrt-evolve doctor` correctly reports a green env and, on a broken one, names
  the missing piece + fix (verified by temporarily hiding a dep).
- Default Rust build still ML-free; nothing here regresses the existing sweep.

## Honest note
The genuinely hard, only-partly-solvable piece is the **CUDA kernel build**
(mamba-ssm/causal-conv1d): no Windows wheels, version-coupled to torch. This
track makes that gap LEGIBLE (doctor + contract) and scripts the known-good WSL2
path; it does not magically produce Windows kernel wheels. "Portable" here means
"the framework tells you exactly how to stand up its env and fails loudly when it
can't," not "runs from one binary anywhere."
