# New Modalities — Skill Ingestion + Reasoning-Step Modification — Plan

## Tasks

1. [ ] Resolve the three open design questions in spec.md §Open design
   questions; record the decisions at the top of this file before coding.
   -- evidence: decisions written here.
2. [ ] `dataset.rs`: add `SkillIngestion` (`#[serde(rename="skill_ingestion")]`)
   and `ReasoningEdit` (`#[serde(rename="reasoning_edit")]`) variants with the
   fields from spec §Scope; keep `PartialEq`. -- evidence: variants compile + serde rename.
3. [ ] Dataset round-trip test for both new kinds (write→read→equal, one object
   per line). -- evidence: test name.
4. [ ] `generate/prompts.rs`: `skill_ingestion_*` and `reasoning_edit_*` system
   + user prompts, format-contract-owned (strict JSON array). -- evidence: prompt fns.
5. [ ] `generate/mod.rs`: extend `GenMode`, `plan_modes`, `mode_for_modality`
   for `skill_ingestion` / `reasoning_edit`. -- evidence: plan_modes test.
6. [ ] `generate/api.rs`: parse + validate the new rows (skill names a known
   skill/tool; reasoning edits non-empty + valid final action via the tool
   validator). Drop invalids, keep the rest. -- evidence: validation test.
7. [ ] `plan` planner + critic prompts: teach the two modalities + when to pick
   them; ensure planner can target them. -- evidence: planner-targets-modality test.
8. [ ] `directive`/`interview`: add both to the recognized modality vocabulary
   so `priorities` keeps them. -- evidence: directive priorities test.
9. [ ] `export.rs`: Gemma rendering for both; stubbed formats drop with notice.
   -- evidence: gemma export test + stub-drop test.
10. [ ] Guardrails: directive exclusions reject destructive skill invocations /
    final actions. -- evidence: exclusion test.
11. [ ] Final sweep: `cargo test`, `cargo clippy -D warnings`, default build
    candle/pyo3-free, `cargo build --features pyo3` + `--features train` green.
    -- evidence: all green.
12. [ ] Demo: add a `skills/` fixture (or use scrt's own skills) and show the
    planner choosing skill ingestion when the directive prioritizes it; export
    the rows. -- evidence: demo output captured.

## Sign-off
Pending — add `SIGN-OFF.md` when acceptance criteria in spec.md are met.
