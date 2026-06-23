# Track 28 — Packaging & Portability — Plan

## Tasks
1. [ ] `python/pyproject.toml` packaging scrt-evolve-ml: declare the modules
   (scrt_evolve_train/_gguf/_score/_dequant) as a distributable package, with
   extras `[cpu]` / `[cuda]`, console_scripts, and pinned-but-ranged deps.
   Verify `pip install -e .` then `pip install .[cpu]` both work.
2. [ ] CLI interpreter resolution: add `[hardware].python` config + honor
   `$SCRT_EVOLVE_PYTHON`. Make `cmd_train_transformers`/`cmd_export_gguf`/
   `cmd_eval`/infer resolve the interpreter from config-or-env FIRST, and run
   `-m <module>` against the installed package; fall back to `python_pkg_dir()`
   (PYTHONPATH=checkout) only when no installed package is found. One shared
   resolver fn.
3. [ ] `scrt-evolve doctor` subcommand (Rust) → shells a small Python probe that
   reports torch/cuda/transformers/peft/gguf/mamba/causal-conv1d + checks
   llama.cpp arch converter + free disk on work/place dirs. Pretty-print
   pass/fail + fix hints. Reuse `HardwareConfig.can_train_state_space()` logic.
4. [ ] `PORTABILITY.md`: OS×accelerator matrix; the verified WSL2 recipe (torch
   cu121 + built mamba kernels + cmake-via-pip + fresh llama.cpp); ecosystem gaps
   (no Windows mamba wheels; llama.cpp arch lag); the "native fs not 9p" + WSL RAM
   cap gotchas; disk-space needs for export.
5. [ ] Release wiring: cargo release profile + per-platform binary artifacts;
   publish/tag scrt-evolve-ml; record the CLI↔package version contract.
6. [ ] Tests: interpreter-resolution unit test (env var > config > fallback);
   doctor probe parses on a known-good venv; pyproject builds (`python -m build`).

## Sequencing
pyproject (1) → interpreter binding (2) → doctor (3) in parallel with docs (4) →
release (5). Keep every change additive; default Rust build stays ML-free; the
existing sweep must stay green.

## Status
NOT STARTED (design locked: pip/uv package + venv-interpreter binding). Prereq
track 27 (config-driven export) is COMPLETE. This is the track that makes the
framework distributable. See memory [[portability-packaging-direction]].
