---
type: spec
track: 32
title: Regression Gate (LLM-judge no-degradation) + min-QA-pairs floor
status: completed
created: 2026-06-28
depends_on: [31, 26, 15, 10, 19]
---

# Track 32 — Regression Gate + min-QA-pairs floor — Spec

## Goal
Let ambient/branch training **make progress on tiny QA-pair counts** by changing
what "accept this step" means. Today the track-15 gate accepts a step only if the
**absolute `correctness`** on a fixed probe didn't drop beyond tolerance. On a
weak model (TinyLlama-1.1B) that score is noisy at 0.0–0.5 (track 29: 60 rows →
0.0; later → 0.2; track 31 live: bouncing 0.4–0.6 with ≈0 trend), so the gate
can't distinguish "small genuine improvement" from "noise" and effectively stalls.

Flip the question from **"prove it improved"** (hard on a weak model) to **"prove
it didn't get worse"** (achievable): sample the probe prompts on the model BEFORE
and AFTER the step, ask an LLM judge whether the AFTER answers **degraded**, and
**accept the step UNLESS degradation is detected.** Plus a **minimum training
QA-pair floor**: don't train on too-few rows — accumulate until there are enough.

## Builds on
- **Track 31** (ambient hardening) — composes with the Q5 dedup ledger's
  idle-on-empty: the min-pairs floor is the same "hold and wait" shape.
- **Track 15** (self-regulation) — the gate is a new `classify`-policy plugged
  into `run_step`; catastrophe/quarantine/halt semantics are UNTOUCHED.
- **Track 10** (eval) — reuses `ScoreReport`/probe; correctness is still computed
  (for the Q4 trend) but is no longer the accept driver under the judge gate.
- **Track 19** (Python train/infer/score) — the real forward pass for true A/B
  sampling (base vs base+adapter), reusing `score.py`'s load/apply/generate.
- **`ingest::LlmRelevanceJudge`** — the batched-LLM-judge-over-`ChatTransport`
  pattern the degradation judge mirrors (incl. err-toward-permissive on failure).

## User-locked decisions (2026-06-28)
1. **Sampling = true A/B.** BEFORE = base model; AFTER = base + the just-trained
   candidate adapter. Real forward pass (transformers), most faithful "did THIS
   step degrade?" signal. (A cached-BEFORE variant is a possible later
   optimization; the seam is built to allow it but v1 is true A/B.)
2. **Judge is primary.** Accept the step unless the judge detects degradation.
   The correctness check is demoted to the **catastrophe floor only** (NaN /
   collapse). This is what unblocks progression on small data.
3. **Min pairs = skip + accumulate.** A batch below the floor does NOT train —
   the rows stay queued and the loop idles (composes with Q5 idle-on-empty),
   so we never train on 1–2 rows.

## Behavior

### Degradation gate
- After training a candidate adapter, run the A/B sampler over the probe prompts:
  `(prompt, before, after)` per item (base completion vs base+adapter completion).
- `LlmDegradationJudge` judges each triple → `worse` | `same-or-better`.
  **Errs toward `same-or-better`** (accept) on a judge failure/garble — symmetry
  with the relevance judge; a flaky judge must not stall progress (the catastrophe
  floor remains the backstop, and `doctor`'s track-31 judge preflight detects a
  down/missing judge model).
- Verdict mapping: `regressed_fraction > max_regressed_frac` ⇒ **Regress**
  (rollback); NaN/collapse on the correctness probe ⇒ **Catastrophic**; else
  **Accept**. Default `max_regressed_frac = 0.0` (ANY degraded item rolls back).

### Min-QA-pairs floor
- `[daemon].min_train_pairs = N`: if a popped batch has `< N` rows, don't train —
  leave them queued and idle. Pure, testable decision (`enough_to_train`).

## Minimum-N: methodology, not a magic number
The exact floor is **empirical** and this track delivers the *knob + method*, not
a hardcoded guess. Reasoning behind the default:
- The probe is 10–13 items; a step that trains on fewer rows than the probe can
  meaningfully cover is almost pure overfitting (track 31 Q4 confirmed the
  overfit-before-broad-change pattern on this pool).
- LoRA rank-16 on q/v (track 29) has enough capacity to memorize a handful of
  rows verbatim — exactly the failure the gate + floor guard against.
- Default **`min_train_pairs = 4`** (one micro-batch is `batch=8`, so 4 is "at
  least half a batch of genuinely-new signal"), conservative and overridable.
- **Sweep recipe (in `bench/`)**: vary `min_train_pairs ∈ {1,2,4,8}`, run the
  ambient loop on a fixed corpus, and compare (a) the Q4 probe-correctness trend
  slope and (b) the degradation-judge regress rate. Pick the smallest N whose
  trend is non-negative and whose regress rate is bounded. The number is TUNED by
  running this, not asserted here.

## Out of scope
- Cached-BEFORE sampling optimization (seam allowed; v1 is true A/B).
- Changing track-15 catastrophe/quarantine/halt semantics.
- A learned (non-LLM) degradation classifier.

## Acceptance
- `LlmDegradationJudge` flags a worse AFTER and passes a same/better AFTER
  (mock-transport tests); a judge error errs toward accept.
- Gate policy maps: any-degradation→Regress, NaN→Catastrophic, clean→Accept
  (pure test).
- `[regulate].gate = "judge"` selects the new policy; `"correctness"` (default)
  preserves today's behavior exactly (back-compat test).
- `enough_to_train(batch_len, min)` skips below the floor (pure test); the ambient
  loop idles instead of training a sub-floor batch.
- A/B sampler emits `(prompt, before, after)` from base vs base+adapter (Python).
- Full sweep green: `cargo test` + `clippy` + `fmt` + touched Python. ML-free Rust
  core (the judge is `ChatTransport`-injected; the A/B forward pass is the only ML
  and lives in the Python subprocess, like every other real-model path).
