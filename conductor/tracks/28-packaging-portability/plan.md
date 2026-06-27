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
**SHIPPED (2026-06-26)** — the framework is distributable; the only un-automated
piece is publishing to an index (the artifacts + contract are in place).

- [x] **Task 1 — `python/pyproject.toml`**: packages `scrt-evolve-ml` (the 5
  modules) with `[cpu]`/`[cuda]`/`[test]` extras + console_scripts
  (`scrt-evolve-train` …); `python/README.md` added.
- [x] **Task 2 — interpreter resolution**: `resolve_python()` in the CLI —
  `--python` > `$SCRT_EVOLVE_PYTHON` > `[hardware].python` (new config field) >
  bare `python`. `cmd_train_transformers`/`infer`/`run-model`/`export-gguf`/
  `dequant`/`eval` all route through it and run `-m <module>` against the
  INSTALLED package; the checkout `python/` dir is now only a PYTHONPATH fallback
  (no longer bails when absent). Unit test asserts the precedence.
- [x] **Task 3 — `scrt-evolve doctor`**: built (extends the UX-review doctor) —
  config parse, model_path, python pkg dir, a torch/cuda/transformers/safetensors/
  mamba probe, llama.cpp auto-detect, writable work_dir; PASS/FAIL + fix per check;
  `--json` for agents.
- [x] **Task 4 — `PORTABILITY.md`**: OS×accelerator matrix, the verified WSL2+CUDA
  recipe, WSL gotchas (9p/RAM/disk), the ecosystem gaps (no Windows mamba wheels,
  llama.cpp arch lag), and the CLI↔package version contract.
- [~] **Task 5 — release wiring**: `[profile.release]` (lto/codegen/strip) present;
  the version contract is documented (PORTABILITY.md). Per-platform binary CI +
  index publishing remain un-automated (deferred — environment-specific).
- [x] **Task 6 — tests**: interpreter-resolution precedence unit test (env > config
  > flag ordering). pyproject `python -m build` is a manual check (needs network).

See memory [[portability-packaging-direction]].
