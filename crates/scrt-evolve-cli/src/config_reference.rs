//! Annotated `evolve.toml` config reference + a copy-pasteable template, surfaced
//! via `scrt-evolve config-reference [--toml]`. Kept here so the schema docs live
//! in one place a coding agent can read in full.

/// Human/agent-readable reference: every config block, its fields, defaults, and
/// purpose. Printed by `scrt-evolve config-reference`.
pub const CONFIG_REFERENCE: &str = r#"scrt-evolve — evolve.toml CONFIG REFERENCE
================================================================================
Everything scrt-evolve does is driven by an evolve.toml. Blocks are ADDITIVE:
omit a block to get its defaults (or to disable that stage). Below: every block,
field, default, and what it controls. (`scrt-evolve config-reference --toml`
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
  [generate.api]
    base_url    = "http://localhost:1234/v1"
    model       = "<served-model-id>"
    api_key_env = "<ENV_VAR>"     # optional; local endpoints ignore auth

[train]                           # dataset.jsonl -> adapter
  preset = "lora"                 # lora | full | pretrain | contrastive | shard
  [train.lora]
    rank = 16
    alpha = 32
    target_modules = ["auto"]     # "auto" => generic nn.Linear auto-detect
    lr = 2e-4
    epochs = 1
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

[eval]                            # held-out probe gate (keep|rollback)
  scorer_backend = "transformers" # api (no ML) | transformers (real forward pass)

[regulate]                        # transactional homeostasis (checkpoint->apply->eval->keep|rollback)
  # present => every weight-mutating step is guarded + catastrophe halts

[hardware]                        # compute environment (generic/arch-level)
  device      = "cuda"            # auto | cpu | cuda | mps
  vram_gb     = 8.0
  ram_gb      = 26.0
  kernels     = ["mamba-ssm","causal-conv1d"]  # accel kernels present
  python      = "<venv python>"   # interpreter for ML subprocesses (planned binding)
  machine     = "<provenance string>"

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

[runtime]                         # load + run a model for generation
  backend      = "llamacpp"       # llamacpp (GGUF via llama-completion) | transformers (HF)
  model_path   = "<file.gguf|dir>"  # weights to serve (default: [export].out_path or [evolve].model_path)
  llama_cpp_path = "<dir>"        # llama.cpp build (shared w/ [export] if unset)
  n_ctx        = 8192
  n_gpu_layers = 0                # llama.cpp -ngl: 0=CPU, 99=offload all that fit
  n_threads    = 0                # 0 => engine default
  [runtime.sampling]
    temperature = 0.0             # 0 => greedy
    top_p       = 1.0
    max_tokens  = 256

[[goals]]                         # learning-by-doing goals (repeatable)
  name          = "<goal>"
  topic         = "<subject>"     # feeds discover palace-search + corpus scope
  tag           = "<tag>"         # one goal <-> one palace tag
  weight        = 1.0
  constitution  = "<text>"        # goal-specific values, LAYERED on [evolve].constitution
  taste         = "<text>"        # goal-specific form, layered on [evolve].taste

Umbrella commands:
  scrt-evolve evolve --schedule --max-rounds N   # eval-gated multi-goal loop
  scrt-evolve export-gguf                         # the [export] pipeline
  scrt-evolve run-model --prompt "..."            # the [runtime] serving lane
"#;

/// Copy-pasteable commented template. Printed by `config-reference --toml`.
pub const CONFIG_TEMPLATE: &str = r#"# evolve.toml — scrt-evolve config (generated template). Edit paths to taste.
# Run `scrt-evolve config-reference` for the full annotated schema.

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

[runtime]
backend = "llamacpp"
n_ctx = 8192
n_gpu_layers = 99
  [runtime.sampling]
  temperature = 0.0
  max_tokens = 256
"#;
