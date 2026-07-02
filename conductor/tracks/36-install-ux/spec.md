---
type: Track Spec
title: Install UX — end-to-end installation experience (Windows native + Linux)
description: CI release artifacts, install scripts, evolve setup subcommand, and README quickstart for Windows and Linux.
tags: [track-36, pending]
timestamp: 2026-07-02T00:00:00Z
resource: ./metadata.json
---

# Track 36 — Install UX — Spec

## Goal

Give a first-time user a single copy-pasteable command that produces a working
`evolve` binary on PATH, a configured Python ML backend, and a passing
`evolve doctor` — without reading PORTABILITY.md first.

Today the install story is documentation only. There is no release artifact,
no installer script, and no CLI bootstrap subcommand. This track closes that
gap end-to-end across Windows native (x86_64-pc-windows-msvc) and Linux
(x86_64-unknown-linux-gnu).

## Scope

Four independently deliverable phases, each a testable artifact:

1. **CI release artifacts** — GitHub Actions workflow that builds the Rust CLI
   for both targets, runs tests, and uploads `evolve-linux-x86_64` and
   `evolve-windows-x86_64.exe` to GitHub Releases on tag push.

2. **Install scripts** — `install.sh` (Linux/bash) and `install.ps1`
   (Windows/PowerShell) that download the latest release binary, place it on
   PATH, create a venv, install `scrt-evolve-ml[cuda]` or `[cpu]` based on
   detected GPU, and print a `evolve doctor` prompt.

3. **CLI bootstrap subcommand** — `evolve setup [--cpu|--cuda]`: once the
   binary is installed, creates/targets a venv, installs `scrt-evolve-ml`,
   sets `[hardware].python` in `evolve.toml` (or prints the env binding), and
   runs `evolve doctor`.

4. **README quickstart** — replace the current "see PORTABILITY.md" pointer
   with a 3-step copy-pasteable block for each platform (≤10 lines each),
   linking to PORTABILITY.md for advanced config.

## Constraints

- The Rust CLI binary stays ML-free. `evolve doctor`, `evolve discover`, and
  `evolve setup` must all run without Python available. `setup` shells out to
  Python only after it has located or created the venv.
- Linux target uses musl static linking (`x86_64-unknown-linux-musl`) so the
  binary has no glibc dependency on older distros (Ubuntu 20.04+, RHEL 8+).
- Windows target links against the MSVC runtime. No MinGW.
- No PyInstaller / Nuitka bundled exe. PORTABILITY.md §"Why not a bundled exe"
  explains: CUDA kernels cannot be frozen.
- Windows native install covers non-SSM LoRA training only. The installer
  detects absence of `mamba-ssm` wheels and emits a note pointing to the
  WSL2 path; it does NOT attempt to install WSL.
- The CI workflow uses the existing `scrt-core` git dep as-is. It must not
  block on track 08 (scrt-core crates.io publish).
- `evolve setup` must be idempotent: re-running on an already-configured
  install is a no-op (or upgrades) rather than an error.
- Install scripts must be hermetic about PATH mutation: on Linux, writes to
  `~/.local/bin` and checks/emits a `$PATH` warning if needed; on Windows,
  uses `$env:USERPROFILE\.local\bin` and emits a `$PROFILE` snippet.

## Dependencies

- Track 28 (packaging/portability) — `pyproject.toml` with `[cpu]`/`[cuda]`
  extras and the `[hardware].python` resolver. Already shipped.
- Track 08 (extract-publish) — release tagging convention; the CI workflow
  fires on `v*` tag pushes. Mechanically independent; CI uses git dep until
  08 ships.

## Acceptance criteria

### Phase 1 — CI release artifacts
- A `v*` tag push triggers the Actions workflow and produces two release
  assets: `evolve-linux-x86_64` (musl static) and `evolve-windows-x86_64.exe`
  (MSVC). Both assets are downloadable from the GitHub Releases page.
- The workflow runs `cargo test --workspace` before uploading; a failing test
  blocks the upload.
- The workflow does NOT require Python or the ML backend to succeed.

### Phase 2 — Install scripts
- `curl -fsSL https://<repo>/install.sh | sh` on a clean Ubuntu 22.04 VM:
  places `evolve` on PATH, creates a venv, installs `scrt-evolve-ml[cuda]`
  (CUDA detected) or `[cpu]` (CPU-only), and prints `evolve doctor` output
  without errors (minus model weights, which the user provides separately).
- `iwr https://<repo>/install.ps1 | iex` on Windows 11: same outcome via
  PowerShell, placing the exe in `$env:USERPROFILE\.local\bin`.
- Both scripts are idempotent: re-running does not fail or duplicate PATH
  entries.
- Both scripts exit non-zero on any fatal error (binary download failure,
  Python not found, pip install failure).
- Both scripts are smoke-testable in CI without a real GPU (CPU path only).

### Phase 3 — CLI bootstrap subcommand
- `evolve setup` on a fresh binary install (no `$SCRT_EVOLVE_PYTHON`, no
  `[hardware].python` in config) creates a venv in a documented default
  location, installs `scrt-evolve-ml`, writes `[hardware].python` to
  `evolve.toml`, then runs `evolve doctor`.
- `evolve setup --cpu` forces the cpu extra; `--cuda` forces the cuda extra.
  Without a flag, auto-detects via `nvidia-smi` / `nvcc` presence.
- On Windows, emits the PowerShell env-set syntax for `$SCRT_EVOLVE_PYTHON`
  in addition to writing config.
- Re-running `evolve setup` on an already-configured install completes without
  error (idempotent).
- The subcommand is testable ML-free: a mock venv path + mock pip call are
  sufficient to exercise the Rust-side logic.

### Phase 4 — README quickstart
- README contains a "Quick install" section with a Linux block and a Windows
  block, each ≤10 lines.
- Both blocks reference `evolve doctor` as the verification step.
- A link to PORTABILITY.md is present for advanced config (WSL2+CUDA,
  Mamba/SSM, custom venv paths).
- The section does not duplicate PORTABILITY.md content.

## Out of scope

- macOS support (MPS/CPU path is partial per PORTABILITY.md; no macOS CI
  runner in scope for this track).
- WSL2 automated installer (the WSL2+CUDA path is documented in PORTABILITY.md
  and requires manual steps; the installer notes this).
- Chocolatey / Homebrew / apt packaging (future track).
- GUI installer or NSIS/WiX packaging.
- Bundled model weights in the release artifact.
- Changing the `--python` resolver logic (already shipped in track 28).

## Open questions

1. Default venv location for `evolve setup`: `~/.local/share/scrt-evolve/venv`
   (XDG on Linux, `%LOCALAPPDATA%\scrt-evolve\venv` on Windows) vs.
   project-local `.venv`? Recommendation: use the platform-appropriate user
   data dir so a global install of `evolve` has one shared venv, but allow
   `--venv <path>` to override.

2. GitHub Releases hosting vs. a CDN mirror: the install scripts point at the
   `github.com/…/releases/latest` redirect. This is fine for v1; a CDN alias
   can be added later.

3. `cargo-zigbuild` vs. native musl cross-compile for the Linux musl target
   in CI: `cargo-zigbuild` is simpler in the Action; native musl requires the
   `musl-tools` apt package. Either works; recommendation is musl-tools (no
   extra toolchain).
