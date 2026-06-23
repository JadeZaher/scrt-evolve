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

## Authoritative design
[DESIGN.md](../DESIGN.md) is the locked architecture (crate shape, the three
core traits, config schema, CLI/SDK surface, 9-phase build order, honest
risks). Prior art: the InfoNCE embedding-adapter spike (the in-tree
`corpus.rs` export + candle `train.rs` loop, now folded into track 05) and
the SimHash similarity work that shipped in scrt-cli (`--mp-similar`). Tracks
here implement DESIGN.md; they do not re-decide it.
