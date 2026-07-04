# scrt_evolve_train — module notes

## shard.py

### live calib batches (Phase E task 2 — track 37)
`_use_live_calib` is set when `len(pairs) >= n_batches` (default 8). In that
case the main distill training loop calls `build_batch(pairs, tokenizer, step, ...)`
fresh each step (step index is unbounded — no recycling). When the dataset is
thin (< n_batches pairs) it falls back to `per_batch_in[step % len(per_batch_in)]`,
the original fixed-batch recycling path. Both paths produce the same tensor
shape; the switch is purely on available data width.

New optional row fields (judge_score, judge_verdict, tier, chosen_over, outcome)
are silently ignored by `load_dataset` — `row.get(...)` with no strict schema.

### rotary kwargs — same-model path (Phase E task 3 — track 37)
`layer_kwargs` on the same-model `train_sharded` path now calls `_rotary_kwargs`
instead of defaulting to `{}`. `_rotary_kwargs` builds `position_embeddings` for
RoPE arches (transformers >= 4.41 pass rotary embeddings DOWN to each layer);
returns `{}` for Mamba/hybrid arches without `model.rotary_emb` — safe no-op.
A dummy tensor sized to `max_seq_len` is used for the initial computation; the
same positions [0..seq-1] are valid across all calib inputs of that length.
