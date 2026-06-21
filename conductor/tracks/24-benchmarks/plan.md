# Track 24 — Benchmark (Granite eval-gated evolution) — Plan

## Tasks
1. [x] `bench/` scaffold: `evolve.toml` (Granite cached-HF model_path, 3 weighted
   goals, generate/eval/regulate/QAT blocks), corpus/work dirs, `RUNBOOK.md`.
2. [x] Claude Code transcript adapter `bench/harvest_claude_projects.py`:
   CC native session format → generic `{role,text,command?}` JSONL (streaming).
   Verified: 5 sessions → 876 entries; 1 session → 48.
3. [x] Bench config points at the CACHED f16 HF Granite (granitemoehybrid), not
   the GGUF. `[train.lora].target_modules = ["auto"]` (hybrid arch). QAT Q4_K_M +
   calibration. `[eval]` transformers scorer. `[regulate]` keep|rollback.
4. [x] Bring-up validation on real data:
   - adapter produces valid entries ✓
   - `discover` yields 120 passages from adapted transcripts ✓
   - `evolve --schedule` starts, runs discover, reaches LIVE generation against
     the LM Studio teacher ✓
   - generate-path robustness: added `salvage_objects` so a small teacher's
     truncated/loose JSON still yields rows (the bench's real failure mode) —
     unit-tested `parser_salvages_truncated_array` ✓
   - tiny end-to-end smoke (1 round / few passages) on Granite → see SIGN-OFF.
5. [x] RUNBOOK documents: adapt → build → SMOKE (budgeted) → long resumable
   schedule → GGUF export → measure; plus the LM Studio context-length
   requirement (≥8192) surfaced during bring-up.
6. [operator] The multi-day schedule run + final GGUF export — launched by the
   user per the RUNBOOK (resumable; not held in an agent session).

## Sign-off
See SIGN-OFF.md. The bench is assembled + bring-up-validated end-to-end on real
data (cached Granite + the user's Claude Code transcripts + LM Studio teacher).
The long run is operator-launched.
