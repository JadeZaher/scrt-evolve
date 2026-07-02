---
type: Track Spec
title: Benchmark (Granite eval-gated evolution)
description: Assemble the lane into a runnable Granite eval-gated evolution benchmark.
tags: [track-24, completed]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# Track 24 — Benchmark (Granite eval-gated evolution) — Spec

## Goal
The FINAL track: assemble everything the lane built into a runnable **benchmark**
that evolves **IBM Granite-4.0-h-tiny** toward goals distilled from the user's own
Claude Code work (`~/.claude/projects` transcripts), via the **eval-gated
multi-goal schedule** (tracks 10/15/20) with **quantization-aware training**
(track 23), and exports a Q4_K_M GGUF. Proves the whole system end-to-end on a
real model + real corpus.

## Scope
- **`bench/` scaffold** — a self-contained bench directory: `evolve.toml`
  (Granite cached-HF model_path, 3 goals, generate/eval/regulate/QAT blocks),
  the corpus dir, work dir, and a `RUNBOOK.md`.
- **Claude Code transcript adapter** (`bench/harvest_claude_projects.py`) —
  converts CC's native session format (`type`/`message{role,content:[blocks]}`)
  into the GENERIC scrt-evolve `TranscriptEntry` shape (`{role,text,command?}`)
  the SDK harvester consumes. Streaming (never loads the 376MB tree into memory).
  Bench-specific (CC-input-aware) so nothing CC-specific leaks into the SDK.
- **Bench config** — `evolve.toml` pointing `model_path` at the CACHED
  full-precision HF Granite (not the GGUF), corpus at the adapted transcripts,
  with `[eval]` (transformers scorer), `[regulate]` (keep|rollback), `[train.qat]`
  (Q4_K_M QAT + calibration), and `[[goals]]` (scrt-cli-fluency, conductor-workflow,
  tool-calling) weighted.
- **Runbook** — operator steps: adapt → build → SMOKE (bounded) → long schedule
  (resumable, multi-day) → export GGUF → measure. The multi-day run is
  operator-launched (not held in an agent session).
- **Bring-up validation** — confirm each stage runs on real data: adapter
  produces valid entries, discover yields passages, the schedule starts and
  reaches live generation, a tiny end-to-end round completes a
  train→eval→keep|rollback cycle on Granite.

## Constraints
- **No new SDK capability** — track 24 is assembly + config + a bench-specific
  input adapter + a runbook. All evolution logic is the already-built+tested lane.
- **Generic SDK preserved** — the CC adapter is bench-local; the SDK's
  `TranscriptEntry`/harvester stays generic (track 23 mandate).
- **Honest about cost** — CPU training of a 13GB hybrid MoE model + QAT overhead
  is slow; the bench is correctness-of-machinery first. The runbook states this.
- **Operator-launched long run** — the agent builds + smoke-tests; the multi-day
  schedule is kicked off by the user (it's resumable, so that's safe).

## Acceptance
- The CC adapter converts real `~/.claude/projects` sessions to valid generic
  transcript JSONL (verified: 5 sessions → 876 entries; 1 session → 48).
- `scrt-evolve discover --config bench/evolve.toml` yields passages from the
  adapted corpus (verified: 120 passages).
- `scrt-evolve evolve --schedule --config bench/evolve.toml` starts, runs
  discover, and reaches live generation against the LM Studio teacher (verified
  during bring-up).
- A bounded end-to-end smoke completes a full discover→generate→train→eval→
  keep|rollback round on Granite, producing a checkpoint + evolution-log row +
  score.json. (Evidence: SIGN-OFF.md records the smoke result.)
- The RUNBOOK documents the long run + GGUF export + the LM Studio context
  requirement surfaced during bring-up.

## Dependencies
The whole lane: 10 (eval), 15 (regulate), 20 (goals/harvest/round/schedule),
19 (Python train/infer/gguf), 23 (QAT + auto-detect targets). The cached HF
Granite + LM Studio teacher + llama.cpp (for export).
