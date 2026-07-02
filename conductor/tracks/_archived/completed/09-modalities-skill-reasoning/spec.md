---
type: Track Spec
title: New Modalities — Skill Ingestion + Reasoning-Step Modification
description: Extend the generation modality set with skill ingestion and reasoning-step edits.
tags: [track-09, pending]
timestamp: 2026-06-28T00:00:00Z
resource: ./metadata.json
---

# New Modalities — Skill Ingestion + Reasoning-Step Modification — Specification

## Goal
Extend the generation modality set beyond `qa | instruction | tool_call | cli |
completion` with two human-requested training targets:
1. **skill ingestion** — teach a model to absorb a *skill* (a `SKILL.md` /
   capability description) and turn it into callable behavior (when to invoke
   it, with what inputs, what it produces).
2. **reasoning-step modification** — teach a model to *edit* a reasoning trace:
   insert, correct, prune, or reorder chain-of-thought steps toward a better
   final action (not just produce an answer, but improve a reasoning path).

Both must flow through the existing self-routing pipeline: the planner can
target them, the generator produces them, the dataset carries them, and the
exporters render them (Gemma-native first; others stubbed).

## Origin
Requested in the evolution interview (directive follow-up): the human ranked
`skill ingestion` and `reasoning step modification` as desired modalities. They
are recorded in `demo/work/directive.json` notes as backlog. This track is that
backlog item.

## Scope
- `dataset.rs`: two new `GenExample` variants:
  - `SkillIngestion { skill_name, skill_doc_excerpt, prompt, invocation,
    expected_outcome, source, gen }` — `invocation` is a structured call or CLI
    that uses the skill; `expected_outcome` is what success looks like.
  - `ReasoningEdit { prompt, original_steps: Vec<String>, edit_op:
    insert|correct|prune|reorder, edited_steps: Vec<String>, final_action,
    source, gen }`.
  Confirm exact field shapes against DESIGN.md §Dataset format and keep the
  cross-language (PyO3) contract intact.
- `generate/prompts.rs`: synthesis prompts for each, grounded in real skill
  docs / real scrt workflows (no invented capabilities).
- `generate/mod.rs`: extend `GenMode` + `plan_modes` + `mode_for_modality` to
  route the two new modalities.
- `generate/api.rs`: parse + **validate** the new rows (skill must name a real
  skill/tool; reasoning edits must yield a valid final action — reuse the tool
  schema validator for `final_action` when it is a tool call).
- `plan` stage: planner + critic prompts learn the two modalities exist and
  when to choose them (skill ingestion ← skill docs in corpus; reasoning edits
  ← multi-step workflows / chained tool use).
- `export.rs`: Gemma-native rendering for both (skill ingestion → an example
  turn invoking the skill; reasoning edit → a `tool_code`/CoT turn showing the
  corrected reasoning). `openai`/`anthropic` stay stubbed.
- `directive`/`interview`: add the two to the recognized modality vocabulary so
  the directive's `priorities` can include them without being dropped.

## Constraints
- Validation is mechanical, not quality-chasing: a skill-ingestion row must
  reference a known skill/tool and a runnable invocation; a reasoning-edit row
  must have non-empty original+edited steps and a valid final action.
- Same self-routing rule that the rest of the system follows: the **framework
  owns the output-format contract**; the planner only steers content. Do NOT
  let planner-authored prompts redefine the row schema.
- Exclusions/guardrails from the directive apply (e.g. "no destructive
  commands" must reject a skill invocation or final action that is destructive).
- No new ML deps; this is a data-generation track (default build stays ML-free).

## Acceptance
- `SkillIngestion` and `ReasoningEdit` round-trip through the JSONL dataset
  (write → read → equal), with `kind` tags `skill_ingestion` /
  `reasoning_edit`.
- A mocked backend produces valid rows of each modality from a fixture passage;
  malformed rows (unknown skill, empty step lists, invalid final action) are
  dropped, not fatal.
- The planner, given a directive that prioritizes the new modalities, emits
  specs targeting them (assert on a mocked planner response).
- Gemma export renders both modalities into the training corpus; stubbed
  formats drop them with a clear notice.
- `directive.priorities = ["skill_ingestion", ...]` is preserved (not dropped
  as an unknown modality).
- Full sweep green: `cargo test`, `cargo clippy -D warnings`, and the default
  build remains candle/pyo3-free.

## Dependencies
Tracks 00–02 (config, discover, generate + dataset + ApiEndpoint), the `plan`
stage (planner/critic/signals), `directive`/`interview`, `toolspec`, and
`export`. Establishes a pattern for adding further modalities later.

## Open design questions (resolve in plan.md before building)
- Skill source: where do skill docs come from — the corpus (a `skills/`
  dir / `SKILL.md` files discovered by `discover`), or a separate input?
- Reasoning-edit ground truth: synthesize the "before" trace and the edit, or
  derive edits from real multi-step tool-call chains in the corpus?
- Do these belong behind the same `tool_format` exporter, or do skill/reasoning
  rows need their own render target?
