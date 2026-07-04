---
type: artifact
track: 39
title: llama.cpp retirement checklist
produced_by: PA-6 (scaffolded by Lane 3; items evidenced as PA-2/PA-3/PA-5 complete)
status: scaffold — blocked on P0-6 go/no-go gate + PA-2/PA-3/PA-5
created: 2026-07-04
updated: 2026-07-04
---

# Track 39 — llama.cpp Retirement Checklist

> **Purpose:** a per-family, per-item checklist for retiring the llama.cpp
> sidecar serving path.  Executed once for each family cleared by the P0-6
> parity gate.  No item is marked done until its evidence (commit SHA or test
> run) is recorded here.

**Pre-condition:** this checklist applies only to families cleared by the
go/no-go gate in `parity-report.md`. Granite-4-h and MoE families are
**NOT** retired here — they remain preflight-refused until Phase B/future.

---

## Retirement scope

The retirement covers:
1. **`evolve model infer`** — rewired from `python -m scrt_evolve_infer` to the
   native candle engine (`serve::infer`, PA-2).
2. **`evolve model run`** — rewired from the `[runtime]` llama.cpp subprocess to
   the native candle engine (`serve::run`, PA-3).
3. **Config deprecation** — `[runtime]` llama.cpp keys emit a load-time warning
   pointing at `[serve.placement]` (PA-5, done in `config.rs`).
4. **Docs updated** — `PORTABILITY.md`, `README.md`, `DESIGN.md`, `AGENTS.md`
   no longer present llama.cpp as a serving dependency for cleared families.
   GGUF remains documented as an **export/interop format** (track 27 unchanged).

---

## Checklist — per cleared family

### Template (copy per family)

```
#### <Family name> (e.g. llama-family / TinyLlama)

Cleared by parity-report.md on: TODO (date + commit SHA)

- [ ] PA-2 `evolve model infer --adapter … --prompt … --ab` uses native engine
      for this family.
      Evidence: _commit SHA_
- [ ] PA-3 `evolve model run --prompt …` uses native engine for this family.
      Evidence: _commit SHA_
- [ ] Config deprecation warning fires when a `[runtime]` block with llama.cpp
      keys is present (PA-5, already in config.rs).
      Evidence: `cargo test -p scrt-evolve config::placement_tests` green
- [ ] `evolve doctor` preflight reports this family as natively-servable
      (no llama.cpp required).
      Evidence: _test / run output_
- [ ] `branch create` preflight does NOT refuse this family (PC-4 invariant).
      Evidence: _test output_
- [ ] `PORTABILITY.md` updated: no llama.cpp serving dependency for this family.
      Evidence: _commit SHA_
- [ ] `README.md` updated: serving section reflects native engine.
      Evidence: _commit SHA_
- [ ] `DESIGN.md` updated: architecture section reflects native engine.
      Evidence: _commit SHA_
- [ ] `AGENTS.md` updated: CLI surface table reflects native engine (no
      `run-model`/`serve` llama.cpp note for cleared families).
      Evidence: _commit SHA_
```

---

## llama-family (TinyLlama)

Cleared by parity-report.md on: **TODO** (fill after P0-6 gate)

- [ ] PA-2 `evolve model infer --adapter … --prompt … --ab` uses native engine.
      Evidence: TODO
- [ ] PA-3 `evolve model run --prompt …` uses native engine.
      Evidence: TODO
- [ ] Config deprecation warning fires for `[runtime]` llama.cpp keys.
      Evidence: `cargo test -p scrt-evolve config::placement_tests::runtime_llamacpp_keys_still_parse_and_emit_warning` green
- [ ] `evolve doctor` reports llama-family as natively-servable.
      Evidence: TODO
- [ ] `branch create` does not refuse llama-family bases.
      Evidence: TODO
- [ ] `PORTABILITY.md` updated.
      Evidence: TODO
- [ ] `README.md` updated.
      Evidence: TODO
- [ ] `DESIGN.md` updated.
      Evidence: TODO
- [ ] `AGENTS.md` updated.
      Evidence: TODO

---

## Qwen2 / Qwen2.5

Cleared by parity-report.md on: **TODO**

*(items same as template above — copy when cleared)*

---

## Gemma / Gemma-2

Cleared by parity-report.md on: **TODO**

*(items same as template above — copy when cleared)*

---

## Dense Mistral

Cleared by parity-report.md on: **TODO**

*(items same as template above — copy when cleared)*

---

## NOT retiring (blocker noted)

| Family | Reason not retired here |
|---|---|
| Granite-4-h | Phase B (PB-1→PB-3) required first; stays preflight-refused |
| MoE families | Out of v1 scope |
| Pure Mamba | Out of v1 scope |
| Mixtral (sparse MoE) | Out of v1 scope |

---

## Final retirement gate (PA-V)

Executed after all cleared families complete their checklists above:

- [ ] `cargo build` (ML-free) green.
      Evidence: TODO
- [ ] `cargo test -p scrt-evolve` green.
      Evidence: TODO
- [ ] `cargo test --features train` green (parity + adapter tests).
      Evidence: TODO
- [ ] `cargo clippy` green.
      Evidence: TODO
- [ ] This checklist committed with all cleared-family items evidenced.
      Evidence: _this file commit SHA_

**Reviewer sign-off (PA-V gate):** TODO

---

*Scaffold authored by Lane 3 (PA-5/PA-6 doc side). Item evidence filled by
Lane 1 (PA-2/PA-3) and Lane 2 as those tasks complete. The checklist is the
retirement trace — done = evidenced, not just ticked.*
