---
type: Track Plan
title: Install UX — end-to-end installation experience (Windows native + Linux)
description: Phased implementation plan for CI release artifacts, install scripts, evolve setup subcommand, and README quickstart.
tags: [track-36, pending]
timestamp: 2026-07-02T00:00:00Z
resource: ./metadata.json
---

# Track 36 — Install UX — Plan

Per the test-once-at-end policy: all code and scripts land first, then one
full sweep (`cargo test --workspace`, `cargo clippy --all-targets`, shell
script smoke tests) at the end.

## Overview

Four phases, each an independently shippable artifact:

| Phase | Artifact | Independently testable? |
|:--|:--|:--|
| 1 | `.github/workflows/release.yml` | Yes — dry-run on PR, real upload on tag |
| 2 | `install.sh` + `install.ps1` | Yes — CPU-only smoke test in CI |
| 3 | `evolve setup` subcommand (Rust) | Yes — ML-free unit tests |
| 4 | README quickstart update | Yes — doc-only, no compile step |

Phases 2 and 3 are loosely ordered (setup subcommand can land before scripts
or after). Phase 4 is last (references the real URLs from Phase 1).

---

## Phase 1 — CI release artifacts

**Goal:** A `v*` tag push produces `evolve-linux-x86_64` and
`evolve-windows-x86_64.exe` attached to a GitHub Release; a non-tag push
(PR / main) runs the build+test matrix without uploading.

### Tasks

- [ ] **Task:** Add `x86_64-unknown-linux-musl` target to the CI matrix.
  Install `musl-tools` on the ubuntu runner and `rustup target add
  x86_64-unknown-linux-musl`. Confirm `cargo build --release --target
  x86_64-unknown-linux-musl` produces a static binary (`ldd` returns "not a
  dynamic executable").
  Evidence: CI log shows `ldd evolve-linux-x86_64` → "statically linked".

- [ ] **Task:** Add `x86_64-pc-windows-msvc` target to the CI matrix on a
  `windows-latest` runner. `cargo build --release` (default target on that
  runner) produces `evolve.exe`. Rename artifact to `evolve-windows-x86_64.exe`.
  Evidence: artifact file appears in the build output with correct name.

- [ ] **Task:** Write `.github/workflows/release.yml`. Trigger: `push` on
  `tags: ["v*"]` (upload) and `pull_request` (build+test only, no upload).
  Steps: checkout, `rustup toolchain install stable`, install musl-tools (Linux
  job), `cargo test --workspace` (ML-free; no Python needed), `cargo build
  --release`, rename artifacts, upload to GitHub Releases via
  `softprops/action-gh-release` (only on tag push).
  Evidence: a dry-run PR run completes without uploading; a tag push triggers
  the release job and assets appear on the Releases page.

- [ ] **Verification [checkpoint]:** Create a test tag `v0.0.0-rc1` on a
  branch, push it, confirm both assets appear on the Releases page with the
  correct filenames. Delete the test release after verification.

---

## Phase 2 — Install scripts
## Phase 2 — Install scripts

**Goal:** One shell command bootstraps the full install (binary + Python backend) on Linux and Windows.

### Tasks

- [x] **Task:** Write `scripts/install.sh`.
  Logic:
  1. Detect OS/arch (error-out if not x86_64 Linux).
  2. Fetch latest release tag from GitHub API (`curl …/releases/latest`).
  3. Download `evolve-linux-x86_64` to `~/.local/bin/evolve`, `chmod +x`.
  4. Warn if `~/.local/bin` is not on `$PATH`; emit `export PATH` line.
  5. Locate Python 3.9+ (`python3` or `python`); error if absent.
  6. Create venv at `~/.local/share/scrt-evolve/venv` (skip if already exists).
  7. Detect CUDA: `nvidia-smi` presence → install `scrt-evolve-ml[cuda]`;
     else `scrt-evolve-ml[cpu]`.
  8. Print: `Run: evolve doctor` to verify.
  Exit non-zero on any fatal step.
  Evidence: `bash -n scripts/install.sh` (syntax check) passes; a mock-binary
  smoke test (replace the download step with a local copy) runs end-to-end
  without error.

- [x] **Task:** Write `scripts/install.ps1`.
  Logic mirrors install.sh but PowerShell idioms:
  1. Detect arch (error if not AMD64).
  2. Fetch latest release tag via `Invoke-RestMethod`.
  3. Download `evolve-windows-x86_64.exe` to
     `$env:USERPROFILE\.local\bin\evolve.exe`; create dir if needed.
  4. Warn if the dir is not in `$env:PATH`; emit `$PROFILE` snippet to add it.
  5. Locate Python 3.9+ (`Get-Command python`); error if absent.
  6. Create venv at `$env:LOCALAPPDATA\scrt-evolve\venv`.
  7. Detect CUDA: `Get-Command nvidia-smi -ErrorAction SilentlyContinue`.
  8. Install `scrt-evolve-ml[cuda]` or `[cpu]` via `pip install`.
  9. Note: `mamba-ssm` is not available on Windows native; for hybrid-SSM
     training, see PORTABILITY.md §WSL2.
  10. Print: `Run: evolve doctor` to verify.
  Exit on error (`$ErrorActionPreference = "Stop"`).
  Evidence: `pwsh -NoProfile -File scripts/install.ps1 -WhatIf` (if feasible)
  or `pwsh -Command "& { . ./scripts/install.ps1 }"` with mocked download
  passes in CI on a `windows-latest` runner.

- [ ] **Task:** Add a CI job `test-install-scripts` (on `ubuntu-latest` and
  `windows-latest`) that runs the install scripts with download replaced by
  a local build artifact (CPU path only, no real GPU). Validates the full
  script flow in a clean temp directory.
  Evidence: CI job green on both runners; `evolve --version` succeeds after
  the script runs.

- [ ] **Verification [checkpoint]:** Manual test on a clean Ubuntu 22.04 VM
  (or Docker container: `ubuntu:22.04`) and Windows 11: run the one-liner,
  confirm `evolve doctor` passes (no model weights needed for the binary/Python
  chain check).

---

## Phase 3 — CLI bootstrap subcommand (`evolve setup`)

**Goal:** Once the binary is on PATH, `evolve setup` closes the last manual
step (Python venv + config binding) from within the CLI itself.

### Tasks

- [ ] **Task (TDD — Red):** Add a failing test in `src/cmd/setup.rs` (new
  file) for the venv-location resolution logic: given no `--venv` flag and no
  existing config, `resolve_venv_path()` returns the platform default path
  (`~/.local/share/scrt-evolve/venv` on Linux,
  `%LOCALAPPDATA%\scrt-evolve\venv` on Windows). Test panics/fails without
  implementation.
  Evidence: `cargo test cmd::setup` fails with "cannot find function".

- [ ] **Task (TDD — Green):** Implement `src/cmd/setup.rs`:
  - `resolve_venv_path(override: Option<PathBuf>) -> PathBuf` — platform dirs
    via `dirs` crate (already a dep via existing code, or add it).
  - `detect_accelerator() -> Accelerator { Cuda, Cpu }` — shells to
    `nvidia-smi` (stdout capture; exit 0 = CUDA).
  - `create_venv(python: &Path, venv: &Path) -> Result<()>` — runs `python -m
    venv <path>` via `std::process::Command`.
  - `pip_install(venv: &Path, extra: &str) -> Result<()>` — runs
    `<venv>/bin/pip install scrt-evolve-ml[<extra>]` (Linux) or
    `<venv>\Scripts\pip.exe install …` (Windows).
  - `write_python_config(venv: &Path, config_path: &Path) -> Result<()>` —
    reads/writes `[hardware].python` in `evolve.toml` using the existing
    config layer (track-28 seam).
  - Orchestrating `run_setup(args: SetupArgs) -> Result<()>` that calls the
    above in order, prints progress, runs `evolve doctor` as a child process.
  All pure-logic functions (path resolution, accelerator detection result
  mapping) are unit-tested ML-free with mocked `Command` outputs where
  needed.
  Evidence: `cargo test cmd::setup` passes; `cargo clippy` clean.

- [ ] **Task (TDD — Refactor):** Wire `evolve setup` into the CLI arg parser
  (`src/cli.rs` or equivalent). Add `--cpu`, `--cuda`, `--venv <path>`,
  `--python <path>` flags consistent with existing CLI conventions. Ensure the
  subcommand appears in `evolve --help`.
  Evidence: `evolve setup --help` shows all flags; `evolve --help` lists
  `setup` in the subcommand list.

- [ ] **Task:** On Windows, after writing config, print:
  ```
  Set SCRT_EVOLVE_PYTHON in your shell:
    $env:SCRT_EVOLVE_PYTHON = "<venv_path>\Scripts\python.exe"
  Add to $PROFILE to persist.
  ```
  On Linux, print:
  ```
  export SCRT_EVOLVE_PYTHON=<venv_path>/bin/python
  # Add to ~/.bashrc or ~/.zshrc to persist.
  ```
  Evidence: unit test asserts the emitted string matches the platform pattern.

- [ ] **Task:** Idempotency: if `[hardware].python` already points to a
  working interpreter (one that imports `scrt_evolve_train` successfully),
  `evolve setup` prints "Already configured — running doctor" and skips
  venv creation and pip install.
  Evidence: test with a pre-written config referencing a mock interpreter path
  that passes the import check; setup exits 0 without calling `create_venv`.

- [ ] **Verification [checkpoint]:** On Linux (or WSL2): delete
  `[hardware].python` from config, run `evolve setup`, confirm venv is created,
  `evolve doctor` passes the Python/pip chain checks.

---

## Phase 4 — README quickstart

**Goal:** The first thing a new user reads is a working copy-pasteable block,
not a pointer to a large reference doc.

### Tasks

- [ ] **Task:** Add a "Quick install" section near the top of `README.md`
  (after the project blurb, before any deep-dive sections). Content:

  **Linux (x86_64):**
  ```sh
  curl -fsSL https://raw.githubusercontent.com/<org>/scrt-evolve/main/scripts/install.sh | sh
  evolve setup     # configure Python backend
  evolve doctor    # verify
  ```

  **Windows (PowerShell, x86_64):**
  ```powershell
  iwr https://raw.githubusercontent.com/<org>/scrt-evolve/main/scripts/install.ps1 | iex
  evolve setup     # configure Python backend
  evolve doctor    # verify
  ```

  Each block ≤10 lines. Include one sentence on the non-SSM Windows
  limitation with a link to PORTABILITY.md for hybrid-SSM/Mamba.

- [ ] **Task:** Add a brief "Verify your install" subsection:
  ```
  evolve doctor prints a checklist. Green lines = ready. A red "python"
  line means run `evolve setup`. A red "model" line means point
  [model].path in evolve.toml at your weights.
  ```
  Evidence: README renders correctly (check headings, code fences, link
  targets are valid).

- [ ] **Task:** Ensure PORTABILITY.md is NOT modified by this track (it
  remains the deep reference; the quickstart links into it). Add a one-line
  note in PORTABILITY.md's intro paragraph pointing back to the README
  quickstart for first-time users.
  Evidence: `git diff PORTABILITY.md` shows only the one-line addition.

- [ ] **Verification [checkpoint]:** README preview renders cleanly on GitHub
  (check via `gh browse` or a PR preview). No broken links, no duplicate
  content with PORTABILITY.md.

---

## Final sweep

- [ ] `cargo test --workspace` — 0 failures.
- [ ] `cargo clippy --all-targets -- -D warnings` — clean.
- [ ] `cargo fmt --check` — clean.
- [ ] Shell script syntax checks: `bash -n scripts/install.sh`,
  `pwsh -Command "Get-Content scripts/install.ps1 | Set-Content /dev/null"` (or equivalent parse check).
- [ ] `tracks.md` Build status updated: add row for track 36.
- [ ] `src/cmd/AGENTS.md` (or nearest AGENTS.md) updated with a note on the
  `setup` subcommand's seams (`resolve_venv_path`, `detect_accelerator`,
  `write_python_config`) and the track-28 config dep.
