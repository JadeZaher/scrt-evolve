# `generate/` — synthetic-data generation

The teacher-driven synthesis stage: discovered passages → supervised rows
(`Dataset`). Backends (`api`, `local`) are dumb executors behind `GenBackend`;
the prompt engineering lives in `prompts.rs` and is shared across both. Modes are
planned from `[generate].kinds` (`plan_modes`) and routed by `GenMode`.

## §modalities — the row kinds and how they train

Each `GenExample` variant (`dataset.rs`) is one modality. A row is rendered to a
prompt→completion pair at TWO places that MUST agree (the cross-language
contract): `export.rs::row_to_pair` (Rust, GGUF/llama.cpp export) and
`python/scrt_evolve_train/trainer.py` (the real training path). Changing a
rendering means editing both.

| kind | trains the model to… | render (user → model) |
| :-- | :-- | :-- |
| `qa` / `instruction` | answer questions / follow instructions | prompt → completion |
| `cli` | emit a runnable `scrt …` command | request → command |
| `tool_call` | emit a structured function call | request → `{tool, arguments}` |
| `completion` | language-model raw text (pretrain-style) | "" → text |
| `skill` | recognize a skill's trigger and invoke it | request → invocation (+ `# expected: …`) |
| `reasoning_edit` | **evolve a reasoning trace** | task + flawed steps → corrected chain + `=> action` |

### Opt-in modalities (track 09): `skill`, `reasoning_edit`

Both are **opt-in via `[generate].kinds`** — absent from `kinds` ⇒ never
planned, so the pipeline is byte-identical to today unless requested. Add them
explicitly:

```toml
[generate]
kinds = ["qa", "cli", "skill", "reasoning_edit"]
```

**Why a flag, not spontaneous.** Reasoning-edit rows are a different supervision
shape (a corrected chain, not a prompt→completion pair). *How* the edit renders
into a training pair is a deliberate policy choice, so it can't happen on its
own — it must be requested. We chose the **"produce corrected chain"** rendering:
the completion carries the corrected reasoning steps BEFORE the final action
(`1. … 2. … => action`). That is what trains the model to **reason internally at
inference** — it learns to emit the reasoning chain itself, not just the answer.
No separate inference-time scaffold is needed; the behavior is trained in.

### Validation (mechanical, in `api.rs`)

- `skill`: non-empty `skill_name` AND `invocation` (must be referenceable, not
  invented — `non_empty_str`).
- `reasoning_edit`: non-empty `final_action`, a known `edit_op`
  (`insert|correct|prune|reorder`), and ≥1 non-empty `edited_steps`
  (`valid_reasoning_edit`).

Invalid candidates are dropped (the rest of the batch is kept), same as the
`tool_call`/`cli` validators.

### Touch-points when adding a modality (the compile-forced checklist)

A new `GenExample` variant forces an arm at every exhaustive match — the
compiler is the checklist: `dataset.rs` (the variant), `generate/mod.rs`
(`GenMode` + `plan_modes` + `mode_for_modality`), `prompts.rs` (system+user),
`api.rs`/`local.rs` (`render_prompt` + parse/validate + `default_kind_for`),
`export.rs` (`row_to_pair`), `eval/{probe,score}.rs` (content key, judge, probe
prompt), `ingest.rs` + `ingest_ledger.rs` (candidate render + dedup key),
`plan/{critic,mod}.rs` (modality count + row text), `regulate/quarantine.rs`
(`row_gen`), `branch/create.rs` (`stamp_gen`), `main.rs` (gen-set + domain text),
and `trainer.py` (the row→pair render, mirroring `export.rs`).
