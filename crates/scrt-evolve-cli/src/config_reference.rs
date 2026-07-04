//! Annotated `evolve.toml` config reference + a copy-pasteable template, surfaced
//! via `evolve config reference [--toml]`. Kept here so the schema docs live
//! in one place a coding agent can read in full.

/// Human/agent-readable reference: every config block, its fields, defaults, and
/// purpose. Printed by `evolve config reference`.
pub const CONFIG_REFERENCE: &str = r#"scrt-evolve — evolve.toml CONFIG REFERENCE
================================================================================
Everything scrt-evolve does is driven by an evolve.toml. Blocks are ADDITIVE:
omit a block to get its defaults (or to disable that stage). Below: every block,
field, default, and what it controls. (`evolve config reference --toml`
prints a copy-pasteable template.)

[evolve]                          # core paths
  model_path   = "<dir>"          # base HF model dir (safetensors + tokenizer)
  corpus_dir   = "<dir>"          # corpus to adapt to
  palace_path  = "<file>"         # scrt mind-palace (retrieval signal)
  work_dir     = "<dir>"          # artifacts land here (default: .scrt-evolve)
  constitution = "<text>"         # GLOBAL values driving HOW the model answers;
                                  # composed into the generate system prompt ->
                                  # shapes the dataset -> shapes training
  taste        = "<text>"         # GLOBAL representational form (style/structure);
                                  # composed alongside constitution

[discover]                        # corpus/palace -> passages
  seed          = "corpus"        # corpus | palace | both
  max_passages  = 120
  dedup         = "simhash"
  cluster       = true
  corpus_patterns = ["...", ...]  # regex topics to pull from the corpus

[generate]                        # passages -> dataset.jsonl (teacher-distilled)
  backend     = "api"             # api (LM Studio/OpenAI-compatible) | local
  kinds       = ["qa","instruction","cli","tool_call"]
  per_passage = 3
  tool_format = "gemma"
  candidates_per_seed = 1         # track 37: >1 + [judge] => rejection sampling
                                  # (best-of-N: gen N, judge-rank, keep top-per_passage)
  # synthesis_rate    = 0.25      # track 37: fraction of steps that run Evol expansion (nudge-settable)
  [generate.api]
    base_url    = "http://localhost:1234/v1"
    model       = "<served-model-id>"
    api_key_env = "<ENV_VAR>"     # optional; local endpoints ignore auth

[judge]                           # track 37: per-pair DATA judge (pre-queue quality gate)
  min_score = 0.5                 # keep rows scoring >= this (0.0-1.0)
  on_error  = "keep"              # keep (fail-open, default) | drop (fail-closed; flip before P2P publish)
  batch     = 15                  # rows per LLM judge call
  sample_k  = 4                   # rows sampled/step for the steering-compliance metric (0 = off)

[domain]                          # track 37: what the planner tunes FOR (absent => scrt defaults, byte-identical)
  name             = "scrt"
  description      = "the `scrt` tool ..."
  command_prefixes = ["scrt"]     # a `cli` row must start with one of these
  flag_patterns    = ["--mp-"]    # flag prefixes counted as signal
  tools            = ["scrt_search","scrt_stash","scrt_list_stashes","scrt_get_stash","scrt_drop_stash","scrt_similar"]

[train]                           # dataset.jsonl -> adapter
  preset = "lora"                 # lora | full | pretrain | contrastive | shard
  [train.lora]
    rank = 16
    alpha = 32
    target_modules = ["auto"]     # "auto" => generic nn.Linear auto-detect
    lr = 2e-4
    epochs = 1
    # init_adapter = "<dir>"      # CONTINUE from an existing adapter (further
                                  # training); absent => fresh adapter
  [train.qat]                     # quantization-aware training (toward deploy quant)
    enabled = true
    quant = "Q4_K_M"
    group_size = 32
    calibrate_batches = 8
  [train.fractional]              # FRACTIONAL/sharded training (bounds VRAM)
    enabled = true
    block_size = 8                # layers per shard (the hard VRAM knob)
    # shards = 5                  # alt: N equal blocks (block_size wins)
    calib_batches = 8
    granularity = "block"         # block | module (per-submodule sub-layer floor)
    objective = "distill"         # distill (MSE-vs-self; no new knowledge) |
                                  # end_task (final shard learns CE on completions
                                  # via LM head — THE knowledge signal; use to
                                  # teach new content). Pair w/ rank up + epochs.
  [train.distill]                 # cross-MODEL seam distillation (compress a
                                  # larger teacher -> smaller student branch)
    enabled = true
    teacher_model = "/models/Mistral-7B"  # REQUIRED: the larger teacher (shared
                                  # tokenizer w/ student). Two decoupled phases:
                                  # teacher pre-captures seam targets to disk,
                                  # student trains against the cache (never
                                  # co-resident). Reuses [train.fractional] knobs.
    layer_map = "stride"          # stride (nearest teacher seam) | block_avg
    loss = "cosine_mse"           # cosine_mse | mse | cosine
    projection = "auto"           # auto (lift student->teacher width if differ) |
                                  # none (require equal widths). Scaffold, dropped.
    grad_clip = 1.0               # gradient-clip max-norm (caps spike steps; 0 => off)
    lr_mode = "auto"              # auto: DYNAMIC per-block LR (from seam magnitude)
                                  # + warmup->cosine schedule | fixed: constant --lr

[eval]                            # held-out probe gate (keep|rollback)
  scorer_backend = "api"          # api (default, no ML) | transformers (real forward pass)
  stable_probe   = false          # true => reuse a fixed probe across rounds (REAL
                                  # cross-round gate for `branch evolve`); false re-carves each round

[regulate]                        # transactional homeostasis (checkpoint->apply->eval->keep|rollback)
  # present => every weight-mutating step is guarded + catastrophe halts

[hardware]                        # compute environment (generic/arch-level)
  device      = "cuda"            # auto | cpu | cuda | mps
  vram_gb     = 8.0
  ram_gb      = 26.0
  kernels     = ["mamba-ssm","causal-conv1d"]  # accel kernels present
  python      = "<venv python>"   # interpreter for ML subprocesses (scrt-evolve-ml
                                  # venv); precedence: --python > $SCRT_EVOLVE_PYTHON > here
  machine     = "<provenance string>"
  free_gpu_command = "lms unload --all"  # evict a GPU teacher before training
                                  # (single-GPU box; run before each train step)

[store]                           # bounded model-weight VERSION ring (storage+loading)
  # dir         = "<work_dir>/store"  # version ring + store.json
  keep_versions = 2               # current + N-1 prior (older pruned on commit)
  deploy_to     = "<live .gguf>"  # swap the kept GGUF in place here (e.g. LM Studio)
  # A version = the tiny adapter (reverse trace) + optional GGUF over the shared
  # immutable base; `branch evolve` commits + deploys, `branch rollback` reverts.

[export]                          # merge (sharded) adapter -> GGUF -> quantize -> place
  quant          = "Q4_K_M"       # format/quant target (f16|none skip quantize)
  dtype          = "bfloat16"     # merge-load dtype (bf16 avoids fp32 OOM)
  llama_cpp_path = "<dir>"        # llama.cpp build w/ convert + llama-quantize
  work_path      = "<dir>"        # FAST native-fs scratch (NOT a 9p mount)
  out_path       = "<file.gguf>"  # final GGUF target
  place_dir      = "<dir>"        # copy finished GGUF here (e.g. LM Studio models)
  max_shard_size = "3GB"
  keep_intermediates = false
  [export.merge_shards]           # sharding-merge rule (fractional -> single adapter)
    enabled = true
    pattern = "adapter-shard-*.safetensors"

[serve.placement]                 # TRACK 39: per-layer GPU placement for the native candle engine
  mode       = "auto"             # auto: probe free VRAM at load and fill greedily
                                  # manual: honor gpu_shards exactly; refuses fast on impossible maps
  gpu_shards = [0,1,2,8,9,16]    # (manual only) explicit layer indices on GPU — interleaved OK
                                  # the rest reside on CPU/RAM
                                  # Replaces llama.cpp's contiguous n_gpu_layers prefix limit.

[runtime]                         # DEPRECATED (track 39): load + run a model via llama.cpp sidecar
                                  # llama.cpp keys (backend="llamacpp", llama_cpp_path, n_gpu_layers)
                                  # are deprecated. Migrate to [serve.placement] for GPU placement;
                                  # use `evolve model infer` (native candle) for serving.
                                  # These keys still parse during the retirement window but warn.
  backend      = "llamacpp"       # DEPRECATED: llamacpp | transformers (HF)
  model_path   = "<file.gguf|dir>"  # weights to serve (default: [export].out_path or [evolve].model_path)
  llama_cpp_path = "<dir>"        # DEPRECATED: llama.cpp build (shared w/ [export] if unset)
  n_ctx        = 8192
  n_gpu_layers = 0                # DEPRECATED: use [serve.placement].gpu_shards instead
  n_threads    = 0                # 0 => engine default
  [runtime.sampling]
    temperature = 0.0             # 0 => greedy
    top_p       = 1.0
    max_tokens  = 256

[daemon]                          # ambient continuous-evolution daemon (track 26)
  max_vram_gb       = 4.0          # train only when >= this much VRAM is FREE (0 => ungated)
  poll_interval_secs = 30          # wait this long when throttled / queue idle
  batch             = 1            # queued items folded into one microshard step
  granularity       = "module"     # track-25 microshard granularity (module | block)
  objective         = "end_task"   # track 37: daemon learning objective (end_task = knowledge signal;
                                   #   overrides [train.fractional].objective for daemon steps, like granularity)
  eval_cadence      = 1            # reserved; v1 eval-gates EVERY step (safe default)
  # --- gentle background (coexist with gaming / video) ---
  pause_on_gpu_process = true      # yield the GPU when ANOTHER process uses it
  cpu_fallback      = true         # when GPU busy/starved, train a light step on CPU (else pause)
  rotation_blocks   = 0            # >0: train one block/step, rotate (ordinal % N) — bounds VRAM, spreads coverage
  cooldown_secs     = 0            # sleep after each step to cap GPU duty cycle (0 => none)
  # --- ambient self-feed (used by `scrt-evolve --ambient`) ---
  seed_adapter      = "<adapter dir>"  # seed work_dir/adapter from here if absent (continue an expert)
  auto_ingest       = false        # when queue runs low, re-run [ingest] to mine fresh activity
  refill_below      = 1            # pending-rows threshold that triggers auto_ingest

[ingest]                          # what `evolve ambient ingest` / `--ambient` mine into the queue
  sources   = ["~/.claude/projects"]  # interaction-log dirs (empty => Claude Code projects)
  docs      = ["conductor"]        # doc dirs -> completion rows (*.md/*.txt)
  match     = ["--mp-","--effort"] # cheap substring prefilter (bounds LLM-judge cost)
  relevance = "<criterion>"        # LLM judges each candidate against this (via [generate.api])
  lane      = "raw"                # raw (passive tail) | priority (drains first)
  max       = 600                  # cap rows enqueued per ingest (0 => no cap)
  tier      = "private"            # track 37: sovereignty tier stamped on mined rows (private | shared)

[[goals]]                         # learning-by-doing goals (repeatable)
  name          = "<goal>"
  topic         = "<subject>"     # feeds discover palace-search + corpus scope
  tag           = "<tag>"         # one goal <-> one palace tag
  weight        = 1.0
  constitution  = "<text>"        # goal-specific values, LAYERED on [evolve].constitution
  taste         = "<text>"        # goal-specific form, layered on [evolve].taste

Umbrella commands:
  evolve train auto --schedule --max-rounds N    # eval-gated multi-goal loop
  evolve ambient start [--max-vram 4G]           # ambient continuous-evolution loop
  evolve ambient teach --prompt "..." --completion "..."  # explicit priority-lane capture
  evolve train export-gguf                        # the [export] pipeline
  evolve model run --prompt "..."                 # the [runtime] serving lane
"#;

/// Copy-pasteable commented template. Printed by `config-reference --toml`.
pub const CONFIG_TEMPLATE: &str = r#"# evolve.toml — scrt-evolve config (generated template). Edit paths to taste.
# Run `evolve config reference` for the full annotated schema.

[evolve]
model_path  = "/path/to/hf-model"
corpus_dir  = "/path/to/corpus"
work_dir    = "./.scrt-evolve"

[discover]
seed = "corpus"
max_passages = 120

[generate]
backend = "api"
  [generate.api]
  base_url = "http://localhost:1234/v1"
  model    = "your-served-model-id"

[train]
preset = "lora"
  [train.lora]
  rank = 16
  alpha = 32
  target_modules = ["auto"]
  [train.fractional]   # bounds VRAM; train one layer-block at a time
  enabled = true
  block_size = 8
  granularity = "block"   # or "module" for the sub-layer VRAM floor
  objective = "end_task"  # CE on completions (teaches knowledge); "distill" = regularize only

[hardware]
device = "cuda"
kernels = ["mamba-ssm", "causal-conv1d"]   # for hybrid-SSM training

[export]
quant = "Q4_K_M"
dtype = "bfloat16"
llama_cpp_path = "/path/to/llama.cpp"
work_path = "/fast/native/scratch"          # NOT a 9p mount
out_path  = "/fast/native/scratch/model-Q4_K_M.gguf"
place_dir = "/path/to/lmstudio/models/your-model"
  [export.merge_shards]
  enabled = true
  pattern = "adapter-shard-*.safetensors"

# [runtime] is DEPRECATED for serving (track 39 — native candle engine).
# Migrate GPU placement to [serve.placement]; use `evolve model infer` natively.
# These keys still parse during the retirement window but emit a warning.
# [runtime]
# backend = "llamacpp"
# n_ctx = 8192
# n_gpu_layers = 99
#   [runtime.sampling]
#   temperature = 0.0
#   max_tokens = 256

# [serve.placement]               # track 39: native per-layer GPU placement
# mode = "auto"                   # auto (probe VRAM) | manual (honor gpu_shards)
# gpu_shards = [0, 1, 2, 8, 9, 16]  # manual: interleaved layer indices on GPU
"#;

/// Dataset (`dataset.jsonl`) + branch manifest/registry schema reference.
/// Printed by `evolve config dataset`. These are the cross-language
/// (Rust writer ↔ Python reader) and cross-repo (hivemind Merge) contracts;
/// changing a field is a breaking change.
pub const DATASET_REFERENCE: &str = r#"scrt-evolve — DATA CONTRACTS REFERENCE
================================================================================
The cross-language / cross-repo data shapes. `dataset.jsonl` is the
generate↔train boundary (Rust writer ↔ Python reader); the branch manifest +
registry are the contract feeding hivemind's Merge fabric. Changing a field is
a breaking change.

dataset.jsonl — one JSON object per line, tagged by `kind`
--------------------------------------------------------------------------------
Common optional fields on most rows:
  source : string   # provenance (which passage/stash the row came from)
  gen    : string   # generation stamp (e.g. "regen:swap2", "branch:<name>") —
                    # the quarantine key (track 15). Absent on completion/contrastive.

kind = "qa"                 # a prompt → answer pair (the workhorse SFT row)
  prompt     : string  (req)
  completion : string  (req)
  source?, gen?

kind = "instruction"        # instruction-tuning triple
  instruction : string  (req)
  input       : string  (default "")
  output      : string  (req)
  source?, gen?

kind = "completion"         # raw continued-pretraining text (no prompt/response split)
  text   : string  (req)
  source?

kind = "contrastive"        # embedding-adapter row (InfoNCE)
  query     : string        (req)
  positive  : string        (req)
  negatives : [string]      (default [])
  stash?    : string

kind = "tool_call"          # user intent → structured tool call (function-calling)
  prompt    : string        (req)  # the NL request
  tool      : string        (req)  # tool name, e.g. "scrt_stash"
  arguments : object        (req)  # JSON args; keys must be valid params for `tool`
  source?, gen?

kind = "cli"                # user intent → exact runnable command line (CLI fluency)
  prompt  : string  (req)
  command : string  (req)   # e.g.  scrt "auth" --mp-stash auth --mp-ttl 4h
  source?, gen?

Example line:
  {"kind":"qa","prompt":"What does --mp-ttl do?","completion":"Sets a stash TTL.","gen":"branch:scrt-cli"}

branch manifest.json — one per branch (work_dir/branches/<name>/manifest.json)
--------------------------------------------------------------------------------
  version           : string   # MANIFEST_VERSION ("1")
  name              : string   # registry/router key
  base_model        : string   # the base this branch specialized
  domain            : string   # human label, e.g. "legal/tool-calling"
  corpus_descriptor : string   # e.g. "128 dataset rows"
  router_signature  : object   # { kind, ... } — how the router matches queries
  eval_report       : { string: float }   # e.g. {"correctness": 0.83}
  lineage           : { parent: string|null }
  gguf_sha          : string   # SHA-256 of the branch GGUF (content address)
  created           : string   # ISO-8601 UTC

branches/registry.json — the fleet (work_dir/branches/registry.json)
--------------------------------------------------------------------------------
  schema_version : int                  # REGISTRY_SCHEMA_VERSION (1); mismatch refused
  branches       : [ manifest, ... ]    # the registered BranchManifest entries

Produced by: `generate` (dataset.jsonl), `branch create` / `branch register`
(manifest + registry). Print the config schema with `config-reference`.
"#;
