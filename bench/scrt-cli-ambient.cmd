@echo off
REM ============================================================================
REM scrt-cli-ambient.cmd — config-driven ambient evolution loop for a branch.
REM
REM Each round, the binary runs ONE config-driven pipeline:
REM   discover -> generate (LM Studio teacher) -> [free GPU] -> train (continue
REM   from current version) -> eval-gate -> on KEEP: commit a bounded version +
REM   deploy the GGUF in place (reversible via `branch rollback`).
REM
REM Prereqs (one-time): native CUDA venv bound via SCRT_EVOLVE_PYTHON, native
REM   llama.cpp for export, scrt-evolve on PATH (or set SCRT below). See
REM   bench/branch-scrt-cli.toml for the [store]/[hardware] knobs.
REM
REM Usage:  scrt-cli-ambient.cmd [branch-name] [config.toml] [interval-seconds]
REM ============================================================================
setlocal enabledelayedexpansion

set "NAME=%~1"
if "%NAME%"=="" set "NAME=scrt-cli"
set "CONFIG=%~2"
if "%CONFIG%"=="" set "CONFIG=bench\branch-scrt-cli.toml"
set "INTERVAL=%~3"
if "%INTERVAL%"=="" set "INTERVAL=1800"
if "%SCRT%"=="" set "SCRT=scrt-evolve"
set "TEACHER=meta-llama-3-8b-instruct"

echo [ambient] branch=%NAME% config=%CONFIG% interval=%INTERVAL%s
echo [ambient] Ctrl-C to stop.

:loop
  echo.
  echo [ambient] %DATE% %TIME% — loading teacher %TEACHER%
  call lms load %TEACHER% --gpu max -y
  echo [ambient] evolving %NAME% (one config-driven round)
  call %SCRT% branch evolve --name "%NAME%" --config "%CONFIG%" --steps 120
  echo [ambient] round done; current versions:
  call %SCRT% branch versions --name "%NAME%" --config "%CONFIG%"
  echo [ambient] sleeping %INTERVAL%s (Ctrl-C to stop)
  timeout /t %INTERVAL% /nobreak >nul
goto loop
