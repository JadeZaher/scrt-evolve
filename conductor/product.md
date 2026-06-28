---
type: Product Overview
title: scrt-evolve — Product
description: What scrt-evolve is and the product it builds toward.
timestamp: 2026-06-28T00:00:00Z
---

# scrt-evolve — Product

## What it is
A standalone Rust framework that makes a model **better at its own corpus**
with no human labeling. Given a model path, a corpus directory, and a scrt
mind-palace, it runs a self-contained loop: **discover** relevant context (via
scrt retrieval) → **generate** synthetic supervised data from it (local model
or API) → **train** the model with a selectable preset (lora / full / pretrain
/ contrastive / shard).

The corpus + palace **are** the training signal. Labels are model-generated
from discovered context, not hand-written — self-supervised in the labeling
sense.

## Who it's for
Agents/operators who accumulate a curated scrt palace during daily work and
want to distill that signal into an adapter or weights, fully locally if
desired (API backend optional for higher-quality teacher generation).

## North star
A pipeline that is **wired and inspectable** end to end — you can read the
dataset, swap the teacher, stop/resume between stages — so the bet
("self-generated data finetunes a usefully-better model") becomes measurable
rather than blind.

## Experience: operators and agents are both first-class drivers
scrt-evolve is driven two ways and the CLI is designed for both. The full
critique + finding ledger lives in [UX-REVIEW.md](UX-REVIEW.md); the product
stance it produced:

**DevUX (human operator).** The CLI should fail *before* a long run, not at
minute nine, and never silently do the wrong thing. Concretely: defaults are
sane and partial configs are a documented feature; `config-reference [--toml]`
is a queryable, annotated schema; config-resolution errors name the exact fix
(`set [evolve].model_path`). The closeout hardened the two sharpest edges — a
**`doctor` preflight** (`cmd_doctor`) validates config/model/python/llama.cpp up
front, and `train` now prints a loud **candle-fixture warning** so the obvious
first run can't quietly train a toy arch instead of your model.

**AIUX (LLM agent).** An agent driving the CLI should never have to regex prose
to learn what happened. A process-global **`--json`** flag emits a
machine-readable summary line for the artifact-producing commands (counts,
resolved paths, and effective resolved values like `is_fixture` and quant
source) so an agent can branch on outcomes programmatically; exit codes are
clean and uniform. The cross-language contracts (`dataset.jsonl`,
`manifest.json`, `registry.json`) are versioned and self-describing by design.

**Ambient / DevEx (unattended evolution).** The ambient daemon is the
hands-off mode of the same product: `scrt-evolve --ambient --dir <project>`
trains an expert on a project's living corpus, eval-gated every step through the
track-15 transaction, managed by `daemon status/stop/health/trend`. Hardening
(track 31) made it production-robust — judge-model preflight, a content-hash
dedup ledger that idles instead of recycling stale data, transient-vs-catastrophe
retries with a supervisor cap, and a wall-clock training budget — so it can run
unattended without drifting or wedging.

## Authoritative design
[DESIGN.md](../DESIGN.md) is the locked architecture (crate shape, the three
core traits, config schema, CLI/SDK surface, 9-phase build order, honest
risks). Prior art: the InfoNCE embedding-adapter spike (the in-tree
`corpus.rs` export + candle `train.rs` loop, now folded into track 05) and
the SimHash similarity work that shipped in scrt-cli (`--mp-similar`). Tracks
here implement DESIGN.md; they do not re-decide it.
