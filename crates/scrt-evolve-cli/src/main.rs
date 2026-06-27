//! `scrt-evolve` — thin CLI shim over the scrt-evolve SDK.
//!
//! Subcommands: `init | discover | generate | export | train | run`. Each
//! reads/writes the work-dir artifacts so stages are independently runnable:
//! `discover` → `discovered.json`, `generate` → `dataset.jsonl`, `export` →
//! llama.cpp fine-tune files, `train` → `adapter.safetensors` (candle, behind
//! the `train` feature).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use scrt_evolve::{EvolveConfig, WorkDir};

mod config_reference;
use config_reference::{CONFIG_REFERENCE, CONFIG_TEMPLATE};

#[derive(Parser)]
#[command(
    name = "scrt-evolve",
    version,
    about = "Make a model better at its own corpus — discover → generate → train → export → run.",
    long_about = "scrt-evolve — opinionated local LLM training + model tooling.\n\n\
        Everything is driven by an `evolve.toml` config. Run `scrt-evolve config-reference`\n\
        for the FULL annotated schema of every config block (the recommended starting\n\
        point for coding agents configuring this for a user), or `scrt-evolve init` to\n\
        scaffold a commented evolve.toml.\n\n\
        Pipeline: discover → generate → train (dense | fractional/sharded GPU) → export\n\
        (merge → GGUF → quantize → place) → run-model (llama.cpp / transformers). The\n\
        `evolve --schedule` umbrella runs the eval-gated multi-goal loop."
)]
struct Cli {
    /// Emit a machine-readable JSON summary line for artifact-producing
    /// commands (`generate`, `eval`, `plan`, `train`, `export-gguf`,
    /// `branch list|route|create`, `discover`, `probe build`, …), in addition
    /// to the human-readable output. The JSON object is always the LAST line on
    /// stdout. Intended for coding agents driving the CLI. Accepted in either
    /// position (`scrt-evolve --json generate` or `scrt-evolve generate --json`).
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

/// Process-global `--json` flag. Set once from the parsed CLI in `run()`; read
/// by `emit_json`. A binary-wide toggle avoids threading a bool through every
/// `cmd_*` signature for what is purely an output-formatting concern.
static JSON_OUTPUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Print `value` as a single compact JSON line on stdout when `--json` is set;
/// a no-op otherwise. The artifact-producing commands call this AFTER their
/// human output so an agent can parse the final line.
fn emit_json(value: serde_json::Value) {
    if JSON_OUTPUT.load(std::sync::atomic::Ordering::Relaxed) {
        println!(
            "{}",
            serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
        );
    }
}

/// Build an actionable error for a failed ML subprocess. The real-model paths
/// (train/infer/score/export/dequant/run-model) are all subprocess-driven, so a
/// bare "exited with N" leaves an operator staring at a raw Python traceback with
/// no scrt-evolve guidance. This names the failed module, the most likely cause
/// (missing deps in the chosen interpreter), the `--python` remediation, and the
/// captured log path when `SCRT_EVOLVE_LOG_FILE` is set.
fn subprocess_failure(module: &str, py: &str, status: std::process::ExitStatus) -> anyhow::Error {
    let log_hint = std::env::var("SCRT_EVOLVE_LOG_FILE")
        .ok()
        .map(|p| format!("\n  full output captured at: {p}"))
        .unwrap_or_default();
    anyhow::anyhow!(
        "the `{module}` subprocess (`{py} -m {module}`) failed ({status}).\n  \
         → Most often the interpreter is missing deps: ensure `{py}` has \
         torch + transformers + safetensors (plus `gguf` for export/dequant). \
         Pass `--python /path/to/venv/python` to select the right interpreter. \
         The subprocess output is above.{log_hint}"
    )
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a commented evolve.toml in the current directory.
    Init {
        #[arg(long, default_value = "evolve.toml")]
        path: PathBuf,
    },
    /// Discover context from the corpus + palace → work_dir/discovered.json.
    Discover {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
    /// Interview the human on training direction → work_dir/directive.json.
    /// Prints the question set; answers are supplied via --answer id=value
    /// (repeatable) for non-interactive use.
    Interview {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        #[arg(long = "in")]
        input: Option<PathBuf>,
        /// Answer a question: --answer goal="tool-calling fluency" (repeatable).
        #[arg(long = "answer", value_name = "ID=VALUE")]
        answers: Vec<String>,
        /// Skip the LLM follow-up questions (core questions only).
        #[arg(long)]
        core_only: bool,
    },
    /// Plan generation: planner LLM analyzes signals + directive → plan.json.
    Plan {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Override the discovered-context input (default: work_dir/discovered.json).
        #[arg(long = "in")]
        input: Option<PathBuf>,
    },
    /// Generate a dataset from discovered context → work_dir/dataset.jsonl.
    Generate {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Override the discovered-context input (default: work_dir/discovered.json).
        #[arg(long = "in")]
        input: Option<PathBuf>,
        /// Override the configured backend (local | api).
        #[arg(long)]
        backend: Option<String>,
        /// Self-route: let the planner decide modalities + write its own prompts
        /// (uses work_dir/plan.json if present, else plans first). Runs the
        /// gap-critic loop for this many follow-up rounds.
        #[arg(long)]
        self_route: bool,
        /// Number of gap-critic follow-up rounds when --self-route is set.
        #[arg(long, default_value_t = 1)]
        gap_rounds: usize,
    },
    /// Export the dataset to llama.cpp fine-tune format (for GGUF models).
    Export {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Override the dataset input (default: work_dir/dataset.jsonl).
        #[arg(long)]
        data: Option<PathBuf>,
        /// The base GGUF model to adapt (default: [evolve].model_path).
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Train a model from a dataset → adapter.
    ///
    /// `--backend candle` (default) uses the in-tree candle preset — a
    /// fixture/mechanical path (tiny hand-built arch; overfit-a-tiny-batch).
    /// `--backend transformers` shells out to the standalone Python trainer
    /// (`python/scrt_evolve_train`) which loads a REAL HuggingFace causal-LM
    /// (RoPE/GQA/BF16) via `transformers` and LoRA-trains it — the real-model
    /// path. The dataset.jsonl is the shared contract between both.
    Train {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long)]
        preset: Option<String>,
        /// `candle` (fixture) | `transformers` (real model via Python).
        #[arg(long, default_value = "candle")]
        backend: String,
        /// Python interpreter for `--backend transformers` (must have torch +
        /// transformers + safetensors). Defaults to `python`.
        #[arg(long)]
        python: Option<String>,
        /// Override the adapter output dir (default: work_dir/adapter).
        #[arg(long)]
        out: Option<PathBuf>,
        /// `transformers` backend: number of training steps.
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// `transformers` backend: max sequence length.
        #[arg(long, default_value_t = 256)]
        max_seq_len: usize,
    },
    /// Run discover → generate (→ export) in one shot.
    Run {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Also export to llama.cpp fine-tune format after generating.
        #[arg(long)]
        export: bool,
    },
    /// Run inference with an optional LoRA adapter; compare base vs adapter (--ab).
    ///
    /// Shells out to `python -m scrt_evolve_infer`. Reads the base model path
    /// from [evolve].model_path in the config. When --adapter is omitted, defaults
    /// to work_dir/adapter (the standard output location of `train --backend
    /// transformers`). Use --ab to see base and adapter outputs side-by-side.
    Infer {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Directory containing adapter.safetensors + adapter_config.json.
        /// Default: work_dir/adapter.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// The prompt to generate from. Required.
        #[arg(long)]
        prompt: String,
        /// Show base model and adapter outputs side-by-side.
        #[arg(long)]
        ab: bool,
        /// Maximum number of new tokens to generate. Default: 128.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Sampling temperature. 0 = greedy (default). >0 = sampling.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Wrap the prompt in the tokenizer's chat template before generating.
        #[arg(long)]
        chat: bool,
        /// Python interpreter (must have torch + transformers + safetensors).
        /// Default: python.
        #[arg(long)]
        python: Option<String>,
    },
    /// Run a model for generation through the config-driven inference RUNTIME.
    ///
    /// Driven by the `[runtime]` block in evolve.toml (backend / model_path /
    /// n_ctx / n_gpu_layers / n_threads / [runtime.sampling]). `backend =
    /// "llamacpp"` serves a GGUF efficiently via llama.cpp's `llama-completion`
    /// (the right path for hybrid-SSM models whose naive transformers forward
    /// OOMs); `backend = "transformers"` runs a HF dir via Python. This is the
    /// serving lane; use `infer` for HF base-vs-adapter A/B comparison.
    RunModel {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// The prompt to generate from. Required.
        #[arg(long)]
        prompt: String,
        /// Python interpreter for the `transformers` backend. Default: python.
        #[arg(long)]
        python: Option<String>,
    },
    /// Print the full annotated `evolve.toml` config schema — every block, field,
    /// default, and purpose. The recommended reference for coding agents
    /// configuring scrt-evolve for a user. `--toml` prints a copy-pasteable
    /// commented template; default prints the reference doc.
    ConfigReference {
        /// Print a copy-pasteable commented evolve.toml template instead of the
        /// reference doc.
        #[arg(long)]
        toml: bool,
    },
    /// Print the fully-resolved `EvolveConfig` (defaults applied) as JSON.
    /// Answers "what will actually run?" without launching anything — the loaded
    /// config after parsing, including defaults for blocks you omitted.
    ConfigShow {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
    /// Print the `dataset.jsonl` row schema + the branch `manifest.json` /
    /// `registry.json` schema — the cross-language/cross-repo contracts. The
    /// command-surface analogue of `config-reference` for data shapes.
    DatasetReference,
    /// Print a machine-readable manifest of every subcommand (name, summary,
    /// flags) — the command-surface analogue of `config-reference`. Use `--json`
    /// for a parseable object; default prints a readable list.
    Commands,
    /// Preflight check: validate config parse, model_path, the python/ package
    /// dir, the `--python` interpreter's ML deps, llama.cpp auto-detect, and a
    /// writable work_dir. Prints PASS/FAIL + a fix for each — turns "long run
    /// dies at minute 9" into "told you in 2 seconds". `--json` for agents.
    Doctor {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Python interpreter to check for torch/transformers (default: python).
        #[arg(long)]
        python: Option<String>,
    },
    /// Merge a LoRA adapter into the base model and export a quantized GGUF
    /// (for LM Studio / llama.cpp).
    ///
    /// Shells out to `python -m scrt_evolve_gguf`: merge (reusing the trainer's
    /// LoRALinear) → `convert_hf_to_gguf.py` (f16) → `llama-quantize` (the
    /// `--quant` type). Needs a llama.cpp checkout (auto-detected, or
    /// `--llama-cpp`). Base model is read from [evolve].model_path.
    ExportGguf {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Adapter dir (adapter.safetensors + adapter_config.json).
        /// Default: work_dir/adapter. Omit-adapter is not supported here —
        /// pass the dir explicitly to override the default.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Output .gguf path. Default: work_dir/<model>-<quant>.gguf.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Quantization type: Q2_K | Q3_K_S | Q3_K_M | Q3_K_L | Q4_0 | Q4_K_M |
        /// Q5_K_M | Q6_K | Q8_0 | f16 | none. Omitted ⇒ use `[export].quant`
        /// (default Q4_K_M); passing this is an explicit override.
        #[arg(long)]
        quant: Option<String>,
        /// Path to a llama.cpp checkout (with convert_hf_to_gguf.py +
        /// llama-quantize). Auto-detected if omitted (~/.unsloth/llama.cpp,
        /// ~/llama.cpp, $LLAMA_CPP).
        #[arg(long)]
        llama_cpp: Option<PathBuf>,
        /// Keep the intermediate merged HF dir and f16 GGUF.
        #[arg(long)]
        keep_intermediates: bool,
        /// Python interpreter (torch + transformers + safetensors + gguf).
        #[arg(long)]
        python: Option<String>,
    },
    /// Score the current model against a held-out probe set (track 10 eval
    /// harness). Prints a `ScoreReport` and writes it to `work_dir/score.json`.
    ///
    /// Backend from `[evolve.eval].scorer_backend`: `api` (no ML — generate
    /// completions via `[generate.api]`, judge with the executable gate) or
    /// `transformers` (real forward pass via `python -m scrt_evolve_score`).
    Eval {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Override the probe path (default: `[evolve.eval].probe_path` or
        /// `work_dir/probe.jsonl`).
        #[arg(long)]
        probe: Option<PathBuf>,
        /// Python interpreter for the `transformers` backend.
        #[arg(long)]
        python: Option<String>,
    },
    /// Convert a GGUF model to an HF safetensors dir (track 23) so it can be
    /// LoRA-trained. Generic + architecture-registry-driven (no model-specific
    /// logic): shells out to `python -m scrt_evolve_dequant`. Streaming
    /// (bounded memory). Lossy for quantized sources (recovers the quantized
    /// weights upcast). Use `--tokenizer` to copy in a fallback HF tokenizer.
    Dequant {
        /// Source `.gguf` path.
        #[arg(long)]
        gguf: PathBuf,
        /// Output HF model dir.
        #[arg(long)]
        out: PathBuf,
        /// Storage dtype: `f16` | `f32`. Default f16.
        #[arg(long, default_value = "f16")]
        dtype: String,
        /// HF tokenizer dir to copy in as the fallback (GGUF tokenizer
        /// extraction is a documented seam).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Python interpreter (needs `gguf` + `safetensors` + `numpy`).
        #[arg(long)]
        python: Option<String>,
    },
    /// Probe-set management (track 10).
    Probe {
        #[command(subcommand)]
        command: ProbeCommand,
    },
    /// Checkpoint store inspection (track 15 — the transactional homeostasis
    /// layer). Checkpoints are produced by eval-gated steps.
    Checkpoints {
        #[command(subcommand)]
        command: CheckpointCommand,
    },
    /// Quarantine management (track 15): the provenance stamps the loop skips
    /// after a catastrophe.
    Quarantine {
        #[command(subcommand)]
        command: QuarantineCommand,
    },
    /// Branch-Train-Merge **branch factory** (track 29): create, list, route +
    /// serve standalone domain-specialized branches (BTM Expert LMs).
    Branch {
        #[command(subcommand)]
        command: BranchCommand,
    },
    /// Ambient continuous-evolution **daemon** (track 26): an always-on,
    /// VRAM-bounded background trainer fed by the living activity queue. Every
    /// step is eval-gated through the track-15 transaction (keep|rollback), so
    /// ambient training can never silently degrade the model.
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
    /// Teach the daemon explicitly: enqueue a prompt→completion pair onto the
    /// PRIORITY lane of the living queue (skips the relevance filter, trains
    /// before passive activity). The cheap "learn this" capture.
    Teach {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// The prompt / user intent the model should learn to handle.
        #[arg(long)]
        prompt: String,
        /// The desired completion for that prompt.
        #[arg(long)]
        completion: String,
    },
    /// Point at a PROJECT directory: auto-detect its mpg palace + corpus and run
    /// the whole self-routing pipeline (discover → plan → generate → export).
    ///
    /// With `--goals`, run the **learning-by-doing multi-goal** pipeline instead
    /// (track 20): for each `[[goals]]` in the config, discover the goal's
    /// tagged stashes → generate a per-goal dataset under
    /// `work_dir/goals/<name>/`. In `--goals` mode the `project` positional is
    /// optional (goals carry their own `project` scoping). For the EVAL-GATED
    /// schedule (train → eval → keep|rollback across goals, halt on catastrophe),
    /// use `--schedule`. The regen flywheel (track 11) remains optional/un-wired.
    ///
    /// Three distinct invocations (the flags pick the mode):
    ///   scrt-evolve evolve ./my-project          # single-project self-route
    ///   scrt-evolve evolve --goals               # multi-goal generate (no gate)
    ///   scrt-evolve evolve --schedule --max-rounds 4   # eval-gated multi-goal loop
    /// In `--goals` / `--schedule` mode the `project` positional is ignored
    /// (goals carry their own scoping); without either, `project` is REQUIRED.
    Evolve {
        /// The project directory to evolve a model against. Optional in
        /// `--goals` mode.
        project: Option<PathBuf>,
        /// Optional base evolve.toml supplying [generate]/[train] settings
        /// (corpus_dir/palace_path are auto-detected and override the base).
        /// In `--goals` mode this is the config the `[[goals]]` are read from
        /// (default: `evolve.toml`).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Gap-critic follow-up rounds.
        #[arg(long, default_value_t = 1)]
        gap_rounds: usize,
        /// Also export to llama.cpp format after generating.
        #[arg(long)]
        export: bool,
        /// Run the multi-goal learning-by-doing pipeline over the config's
        /// `[[goals]]` (discover → generate per goal). No eval-gating yet.
        #[arg(long)]
        goals: bool,
        /// Run the EVAL-GATED multi-goal SCHEDULE (track 20 slices 6–9): bounded
        /// rounds across goals, each `discover → generate → train → eval →
        /// keep|rollback` through the track-15 transaction; halts on catastrophe.
        /// Implies `--goals`. Weight + round count control the budget.
        #[arg(long)]
        schedule: bool,
        /// Max rounds for `--schedule` (the hard budget; no unbounded loop).
        #[arg(long, default_value_t = 4)]
        max_rounds: usize,
        /// Schedule policy for `--schedule`: `round-robin` | `weighted`.
        #[arg(long, default_value = "weighted")]
        policy: String,
        /// Python interpreter for the schedule's train + score subprocesses.
        #[arg(long)]
        python: Option<String>,
    },
}

#[derive(Subcommand)]
enum CheckpointCommand {
    /// List all checkpoints (id, status, kind, metrics) + the `last_good` pointer.
    List {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
    /// Show one checkpoint's manifest.
    Show {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        id: String,
    },
    /// Restore the adapter from a checkpoint into `work_dir/adapter` (manual
    /// rollback). Transactional steps do this automatically on regress.
    Restore {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        id: String,
    },
}

#[derive(Subcommand)]
enum QuarantineCommand {
    /// List the quarantined `gen` provenance stamps.
    List {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
    /// Clear the quarantine (re-arm: the loop will no longer skip those causes).
    Clear {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
}

#[derive(Subcommand)]
enum BranchCommand {
    /// Create a branch: scope a per-branch config (override base + corpus) and
    /// compose discover → teacher-QA generate → train → eval gate → GGUF export
    /// inside the track-15 transaction; eval-passing branches are registered.
    Create {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Branch name (the registry/router key). Overrides `[branch].name`.
        #[arg(long)]
        name: String,
        /// The small base model to specialize. Overrides `[branch].base` /
        /// `[evolve].model_path`.
        #[arg(long)]
        base: Option<String>,
        /// Per-branch corpus dir (overrides `[branch].corpus` / `[evolve].corpus_dir`).
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Human-readable domain label for the manifest (e.g. `legal/tool-calling`).
        #[arg(long)]
        domain: Option<String>,
        /// Cross-MODEL seam distillation mode: a DISTINCT larger `--teacher`
        /// supervises this branch's smaller `--base` at layer seams (genuine
        /// compression). Overrides `[branch].mode = "distill"`.
        #[arg(long)]
        distill: bool,
        /// The larger TEACHER model (path/id) for distill mode. Overrides
        /// `[train.distill].teacher_model`. Implies `--distill`.
        #[arg(long)]
        teacher: Option<String>,
        /// Training steps per block (distill) / total (standard). Default 40.
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Python interpreter for the train / eval / export subprocesses.
        #[arg(long)]
        python: Option<String>,
    },
    /// FURTHER-train an existing branch (config-driven): load the branch's
    /// persisted config, continue from its current stored adapter, run an
    /// eval-gated round vs the live version, and on KEEP commit a new version to
    /// the bounded `[store]` ring + deploy the GGUF. The self-describing,
    /// repeatable evolution step a `.cmd` loops.
    Evolve {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Branch name to evolve (its `branches/<name>/branch.toml` drives the run).
        #[arg(long)]
        name: String,
        /// Training steps this round. Default 120.
        #[arg(long, default_value_t = 120)]
        steps: usize,
        /// Python interpreter for the train / eval / export subprocesses.
        #[arg(long)]
        python: Option<String>,
    },
    /// Show a branch's bounded weight-version ring (`[store]`): each version's
    /// adapter, optional GGUF, score, and which is `current` (live).
    Versions {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        #[arg(long)]
        name: String,
    },
    /// Roll a branch's live model back to the current version's PARENT (the
    /// reverse trace) and re-deploy that GGUF. Reversible — the rolled-back
    /// version stays in the ring until pruned.
    Rollback {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        #[arg(long)]
        name: String,
    },
    /// List the registered branches (reads `branches/registry.json`).
    List {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
    /// Register an EXTERNALLY-built branch GGUF into the fleet: compute its
    /// `router_signature` from the branch dataset, assemble the manifest, and admit
    /// it into `branches/registry.json`. The native-Rust counterpart to the export
    /// step of `create` — used when train/export ran out-of-process (e.g. a WSL GPU
    /// box) or to import a peer's branch artifact. ML-free.
    Register {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Branch name (registry/router key).
        #[arg(long)]
        name: String,
        /// The finished branch GGUF to register (content-addressed by SHA-256).
        #[arg(long)]
        gguf: PathBuf,
        /// Base model id/path recorded in the manifest (overrides `[branch].base`).
        #[arg(long)]
        base: Option<String>,
        /// Domain label (overrides `[branch].domain`).
        #[arg(long)]
        domain: Option<String>,
        /// Branch dataset the signature is computed from (default:
        /// `work_dir/branches/<name>/dataset.jsonl`).
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Eval correctness recorded in the manifest's `eval_report`.
        #[arg(long)]
        correctness: Option<f64>,
        /// Parent branch (lineage), if this forks from another.
        #[arg(long)]
        parent: Option<String>,
    },
    /// Resolve a query to branch(es) + scores WITHOUT serving (the router).
    Route {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// The request to route.
        query: String,
    },
    /// Serve a branch. Pass a `<name>` to serve a specific branch, or `--route
    /// <query>` to route the request to the best branch(es) and serve those
    /// (`[branch.ensemble]`: `single_best` ⇒ top-1, `average_topk` ⇒ blend top-k).
    Serve {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// The branch to serve (omit when using `--route`).
        name: Option<String>,
        /// Route this query to the best branch(es) and serve, instead of `<name>`.
        #[arg(long)]
        route: Option<String>,
        /// The prompt to run (one-shot v1, mirrors `run-model`).
        #[arg(long)]
        prompt: Option<String>,
        /// Python interpreter (for a `transformers` runtime backend).
        #[arg(long)]
        python: Option<String>,
    },
}

#[derive(Subcommand)]
enum DaemonCommand {
    /// Start the ambient loop — runs until `daemon stop`. Pops the living queue
    /// microshard-by-microshard, training ONLY when free VRAM ≥ the budget, every
    /// step through the track-15 transaction. Resumes from the queue cursor.
    Start {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Python interpreter for the train + score subprocesses.
        #[arg(long)]
        python: Option<String>,
        /// VRAM budget in GB (override `[daemon].max_vram_gb`): train only when at
        /// least this much is free. 0 ⇒ ungated.
        #[arg(long)]
        max_vram: Option<f64>,
        /// Stop after N committed/attempted steps (a bound for a supervised run).
        /// Omit for an unbounded daemon (until `daemon stop`).
        #[arg(long)]
        max_steps: Option<u64>,
        /// Drain mode: process the queue once and exit when it empties (instead
        /// of waiting for new activity). Good for a one-shot catch-up.
        #[arg(long)]
        drain: bool,
    },
    /// Signal a running daemon to stop (drops the stop-file it polls each step).
    Stop {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
    /// Show the daemon run-state + the living queue's pending counts per lane.
    Status {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
    },
}

#[derive(Subcommand)]
enum ProbeCommand {
    /// Carve a held-out probe set out of a dataset (asserted zero-overlap with
    /// the training remainder). Writes `probe.jsonl` + the training remainder.
    Build {
        #[arg(long, default_value = "evolve.toml")]
        config: PathBuf,
        /// Dataset to carve from (default: `work_dir/dataset.jsonl`).
        #[arg(long = "from")]
        from: Option<PathBuf>,
        /// Fraction held out into the probe (default: `[evolve.eval]` value or 0.1).
        #[arg(long)]
        holdout: Option<f32>,
        /// Probe output path (default: `[evolve.eval].probe_path` or
        /// `work_dir/probe.jsonl`).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Where to write the training remainder (default:
        /// `work_dir/dataset.train.jsonl`).
        #[arg(long)]
        remainder: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    // clap's derive-generated command tree (now ~26 subcommands, several with
    // nested subcommands + many args) builds one large stack frame during parse.
    // In DEBUG builds that frame can exceed the 1 MB Windows main-thread stack
    // and overflow before `run()` is even entered (release is fine — smaller
    // frames). Run the whole program on a thread with a generous stack so the
    // debug binary (which the test suite exercises) parses reliably.
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run_main)
        .expect("spawn main worker thread")
        .join()
        .unwrap_or(ExitCode::FAILURE)
}

fn run_main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    JSON_OUTPUT.store(cli.json, std::sync::atomic::Ordering::Relaxed);
    match cli.command {
        Command::Init { path } => cmd_init(&path),
        Command::Discover { config } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_discover(&cfg)
        }
        Command::Interview {
            config,
            input,
            answers,
            core_only,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_interview(&cfg, input, answers, core_only)
        }
        Command::Plan { config, input } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_plan(&cfg, input)
        }
        Command::Generate {
            config,
            input,
            backend,
            self_route,
            gap_rounds,
        } => {
            let mut cfg = EvolveConfig::load(&config)?;
            if let Some(b) = backend {
                cfg.generate.get_or_insert_with(Default::default).backend = b;
            }
            if self_route {
                cmd_generate_self_routed(&cfg, input, gap_rounds)
            } else {
                cmd_generate(&cfg, input)
            }
        }
        Command::Export {
            config,
            data,
            model,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_export(&cfg, data, model)
        }
        Command::Train {
            config,
            data,
            preset,
            backend,
            python,
            out,
            steps,
            max_seq_len,
        } => {
            let mut cfg = EvolveConfig::load(&config)?;
            if let Some(p) = preset {
                cfg.train.get_or_insert_with(Default::default).preset = p;
            }
            match backend.as_str() {
                "candle" => cmd_train(&cfg, data),
                "transformers" => {
                    cmd_train_transformers(&cfg, data, python, out, steps, max_seq_len)
                }
                other => anyhow::bail!(
                    "train: unknown backend \"{other}\" (expected candle | transformers)"
                ),
            }
        }
        Command::Infer {
            config,
            adapter,
            prompt,
            ab,
            max_new_tokens,
            temperature,
            chat,
            python,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_infer(
                &cfg,
                adapter,
                &prompt,
                ab,
                max_new_tokens,
                temperature,
                chat,
                python,
            )
        }
        Command::RunModel {
            config,
            prompt,
            python,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_run_model(&cfg, &prompt, python)
        }
        Command::ConfigReference { toml } => {
            print!(
                "{}",
                if toml {
                    CONFIG_TEMPLATE
                } else {
                    CONFIG_REFERENCE
                }
            );
            Ok(())
        }
        Command::ConfigShow { config } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_config_show(&cfg)
        }
        Command::DatasetReference => {
            print!("{}", config_reference::DATASET_REFERENCE);
            Ok(())
        }
        Command::Commands => cmd_commands(),
        Command::Doctor { config, python } => cmd_doctor(&config, python),
        Command::ExportGguf {
            config,
            adapter,
            out,
            quant,
            llama_cpp,
            keep_intermediates,
            python,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_export_gguf(
                &cfg,
                adapter,
                out,
                quant,
                llama_cpp,
                keep_intermediates,
                python,
            )
        }
        Command::Eval {
            config,
            probe,
            python,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_eval(&cfg, probe, python)
        }
        Command::Dequant {
            gguf,
            out,
            dtype,
            tokenizer,
            python,
        } => cmd_dequant(&gguf, &out, &dtype, tokenizer, python),
        Command::Probe { command } => match command {
            ProbeCommand::Build {
                config,
                from,
                holdout,
                out,
                remainder,
            } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_probe_build(&cfg, from, holdout, out, remainder)
            }
        },
        Command::Checkpoints { command } => match command {
            CheckpointCommand::List { config } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_checkpoints_list(&cfg)
            }
            CheckpointCommand::Show { config, id } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_checkpoints_show(&cfg, &id)
            }
            CheckpointCommand::Restore { config, id } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_checkpoints_restore(&cfg, &id)
            }
        },
        Command::Quarantine { command } => match command {
            QuarantineCommand::List { config } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_quarantine_list(&cfg)
            }
            QuarantineCommand::Clear { config } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_quarantine_clear(&cfg)
            }
        },
        Command::Branch { command } => match command {
            BranchCommand::Create {
                config,
                name,
                base,
                corpus,
                domain,
                distill,
                teacher,
                steps,
                python,
            } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_create(
                    &cfg, &name, base, corpus, domain, distill, teacher, steps, python,
                )
            }
            BranchCommand::Evolve {
                config,
                name,
                steps,
                python,
            } => cmd_branch_evolve(&config, &name, steps, python),
            BranchCommand::Versions { config, name } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_versions(&cfg, &name)
            }
            BranchCommand::Rollback { config, name } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_rollback(&cfg, &name)
            }
            BranchCommand::List { config } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_list(&cfg)
            }
            BranchCommand::Register {
                config,
                name,
                gguf,
                base,
                domain,
                dataset,
                correctness,
                parent,
            } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_register(
                    &cfg,
                    &name,
                    &gguf,
                    base,
                    domain,
                    dataset,
                    correctness,
                    parent,
                )
            }
            BranchCommand::Route { config, query } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_route(&cfg, &query)
            }
            BranchCommand::Serve {
                config,
                name,
                route,
                prompt,
                python,
            } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_branch_serve(&cfg, name, route, prompt, python)
            }
        },
        Command::Daemon { command } => match command {
            DaemonCommand::Start {
                config,
                python,
                max_vram,
                max_steps,
                drain,
            } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_daemon_start(&cfg, python, max_vram, max_steps, drain)
            }
            DaemonCommand::Stop { config } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_daemon_stop(&cfg)
            }
            DaemonCommand::Status { config } => {
                let cfg = EvolveConfig::load(&config)?;
                cmd_daemon_status(&cfg)
            }
        },
        Command::Teach {
            config,
            prompt,
            completion,
        } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_teach(&cfg, &prompt, &completion)
        }
        Command::Run { config, export } => {
            let cfg = EvolveConfig::load(&config)?;
            cmd_discover(&cfg)?;
            cmd_generate(&cfg, None)?;
            if export {
                cmd_export(&cfg, None, None)?;
            }
            Ok(())
        }
        Command::Evolve {
            project,
            config,
            gap_rounds,
            export,
            goals,
            schedule,
            max_rounds,
            policy,
            python,
        } => {
            if schedule {
                let config = config.unwrap_or_else(|| PathBuf::from("evolve.toml"));
                let cfg = EvolveConfig::load(&config)?;
                cmd_evolve_schedule(&cfg, &policy, max_rounds, python)
            } else if goals {
                let config = config.unwrap_or_else(|| PathBuf::from("evolve.toml"));
                let cfg = EvolveConfig::load(&config)?;
                cmd_evolve_goals(&cfg)
            } else {
                let project = project.ok_or_else(|| {
                    anyhow::anyhow!(
                        "evolve: pass a PROJECT directory, or use --goals / --schedule \
                         to run the multi-goal pipeline over the config's [[goals]]"
                    )
                })?;
                cmd_evolve(&project, config, gap_rounds, export)
            }
        }
    }
}

fn cmd_init(path: &PathBuf) -> Result<()> {
    let report = scrt_evolve::scaffold::init(path)?;
    println!("wrote scaffold to {}", path.display());
    if report.model_path_missing {
        // Instruct rather than warn: on a fresh scaffold the placeholder path is
        // EXPECTED to be absent — the user has done nothing wrong yet. Frame it
        // as the next step, not an error (D7). `doctor` validates it for real.
        println!(
            "next: edit [evolve].model_path in {} to point at your HF model dir, \
             then run `scrt-evolve doctor` to preflight your environment.",
            path.display()
        );
    }
    Ok(())
}

fn cmd_discover(cfg: &EvolveConfig) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let ctx = scrt_evolve::discover::run(cfg)?;
    let path = wd.discovered_json();
    let json = serde_json::to_string_pretty(&ctx)?;
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    println!(
        "discover: {} passages, {} anchors → {}",
        ctx.passages.len(),
        ctx.anchors.len(),
        path.display()
    );
    emit_json(serde_json::json!({
        "command": "discover",
        "passages": ctx.passages.len(),
        "anchors": ctx.anchors.len(),
        "out": path.display().to_string(),
        "status": "ok",
    }));
    Ok(())
}

fn load_discovered(wd: &WorkDir, input: Option<PathBuf>) -> Result<scrt_evolve::DiscoveredContext> {
    let in_path = input.unwrap_or_else(|| wd.discovered_json());
    if !in_path.exists() {
        anyhow::bail!(
            "discovered context not found at {} — run `scrt-evolve discover` first",
            in_path.display()
        );
    }
    let text = std::fs::read_to_string(&in_path)
        .with_context(|| format!("reading discovered context {}", in_path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", in_path.display()))
}

fn cmd_generate(cfg: &EvolveConfig, input: Option<PathBuf>) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let ctx = load_discovered(&wd, input)?;

    let dataset = scrt_evolve::generate::run(cfg, &ctx)?;
    let out = wd.dataset_jsonl();
    dataset.write_jsonl(&out)?;
    println!("generate: {} rows → {}", dataset.len(), out.display());
    emit_json(serde_json::json!({
        "command": "generate",
        "rows": dataset.len(),
        "out": out.display().to_string(),
        "status": "ok",
    }));
    Ok(())
}

/// Load the directive from work_dir/directive.json if present; else an empty
/// directive (pure signal-driven planning). Warns when none is found so the
/// human knows they ran without stating direction.
fn load_directive(wd: &WorkDir) -> scrt_evolve::TrainingDirective {
    let path = wd.root().join("directive.json");
    match scrt_evolve::TrainingDirective::read(&path) {
        Ok(d) => {
            println!("using training directive: {}", path.display());
            d
        }
        Err(_) => {
            eprintln!(
                "note: no directive.json — planning from signals only. Run \
                 `scrt-evolve interview` to state training direction."
            );
            scrt_evolve::TrainingDirective::default()
        }
    }
}

fn cmd_interview(
    cfg: &EvolveConfig,
    input: Option<PathBuf>,
    answers: Vec<String>,
    core_only: bool,
) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let ctx = load_discovered(&wd, input)?;

    let questions = if core_only {
        scrt_evolve::interview::core_questions()
    } else {
        scrt_evolve::interview::build(cfg, &ctx)
    };

    // Parse --answer id=value pairs.
    let mut answer_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for a in &answers {
        if let Some((id, val)) = a.split_once('=') {
            answer_map.insert(id.trim().to_string(), val.trim().to_string());
        }
    }

    // If no answers were supplied, print the questions so the human can answer
    // them (interactive terminals can pipe answers back via --answer).
    if answer_map.is_empty() {
        println!("# Evolution interview — answer with: --answer <id>=<value>\n");
        for q in &questions {
            let opts = if q.options.is_empty() {
                String::new()
            } else {
                format!(
                    "  options: [{}]{}",
                    q.options.join(", "),
                    if q.multi {
                        " (multi, comma-separated)"
                    } else {
                        ""
                    }
                )
            };
            println!("[{}] {}\n{}\n", q.id, q.text, opts);
        }
        println!("(no --answer given; not writing directive.json)");
        emit_json(serde_json::json!({
            "command": "interview",
            "status": "questions_only",
            "wrote_directive": false,
            "questions": questions.iter().map(|q| serde_json::json!({
                "id": q.id,
                "text": q.text,
                "options": q.options,
                "multi": q.multi,
            })).collect::<Vec<_>>(),
        }));
        return Ok(());
    }

    // Assemble the directive from whatever answers we got.
    let qa: Vec<(scrt_evolve::interview::Question, String)> = questions
        .into_iter()
        .filter_map(|q| answer_map.get(&q.id).map(|v| (q.clone(), v.clone())))
        .collect();
    let directive = scrt_evolve::interview::assemble_directive(&qa);

    let path = wd.root().join("directive.json");
    directive.write(&path)?;
    println!("wrote training directive → {}", path.display());
    println!("{}", directive.prompt_block());
    emit_json(serde_json::json!({
        "command": "interview",
        "status": "ok",
        "wrote_directive": true,
        "directive": path.display().to_string(),
        "answers": qa.len(),
    }));
    Ok(())
}

fn cmd_plan(cfg: &EvolveConfig, input: Option<PathBuf>) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let ctx = load_discovered(&wd, input)?;
    let directive = load_directive(&wd);

    let plan = scrt_evolve::plan::planner::run(cfg, &ctx, &directive)?;
    let out = wd.root().join("plan.json");
    plan.write(&out)?;
    println!(
        "plan: {} specs ({} examples planned) → {}",
        plan.specs.len(),
        plan.total_count(),
        out.display()
    );
    println!("strategy: {}", plan.strategy);
    for s in &plan.specs {
        println!(
            "  - {:11} x{:<4} {}{}",
            s.modality,
            s.count,
            if s.target_tools.is_empty() {
                String::new()
            } else {
                format!("[{}] ", s.target_tools.join(","))
            },
            s.rationale.chars().take(80).collect::<String>()
        );
    }
    emit_json(serde_json::json!({
        "command": "plan",
        "specs": plan.specs.len(),
        "examples_planned": plan.total_count(),
        "strategy": plan.strategy,
        "out": out.display().to_string(),
        "status": "ok",
    }));
    Ok(())
}

fn cmd_generate_self_routed(
    cfg: &EvolveConfig,
    input: Option<PathBuf>,
    gap_rounds: usize,
) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let ctx = load_discovered(&wd, input)?;
    let directive = load_directive(&wd);

    println!("self-route: planning + generating ({gap_rounds} gap round(s))…");
    let result = scrt_evolve::plan::generate_self_routed(cfg, &ctx, &directive, gap_rounds)?;

    // Persist the merged plan + dataset.
    let plan_path = wd.root().join("plan.json");
    result.plan.write(&plan_path)?;
    let out = wd.dataset_jsonl();
    result.dataset.write_jsonl(&out)?;

    println!(
        "self-route: {} specs across {} round(s), {} rows → {}",
        result.plan.specs.len(),
        result.plan.round + 1,
        result.dataset.len(),
        out.display()
    );
    println!("plan → {}", plan_path.display());
    Ok(())
}

fn cmd_evolve(
    project: &PathBuf,
    config: Option<PathBuf>,
    gap_rounds: usize,
    export: bool,
) -> Result<()> {
    // 1. Resolve the project: auto-detect its mpg palace + corpus.
    let layout = scrt_evolve::project::resolve(project)?;
    println!("evolve: project = {}", layout.root.display());
    println!("evolve: {}", layout.palace_note);

    // 2. Build a config: corpus/palace auto-set, base supplies generate/train.
    let base = match &config {
        Some(p) => Some(EvolveConfig::load(p)?),
        None => None,
    };
    let cfg = scrt_evolve::project::config_for_project(&layout, base);
    let wd = WorkDir::from_config(&cfg);
    wd.ensure()?;

    // 3. Discover from the project corpus (+ palace if detected).
    cmd_discover(&cfg)?;

    // 4. Directive: reuse if present, else tell the human to interview.
    let dir_path = wd.root().join("directive.json");
    let directive = if dir_path.exists() {
        scrt_evolve::TrainingDirective::read(&dir_path)?
    } else {
        eprintln!(
            "evolve: no directive.json in {} — proceeding from signals only. \
             Run `scrt-evolve interview` first to state training direction.",
            wd.root().display()
        );
        scrt_evolve::TrainingDirective::default()
    };

    // 5. Self-route: plan (directive-driven) → generate → gap-critic loop.
    let ctx = load_discovered(&wd, None)?;
    println!("evolve: self-routing ({gap_rounds} gap round(s))…");
    let result = scrt_evolve::plan::generate_self_routed(&cfg, &ctx, &directive, gap_rounds)?;
    result.plan.write(wd.root().join("plan.json"))?;
    let out = wd.dataset_jsonl();
    result.dataset.write_jsonl(&out)?;
    println!(
        "evolve: {} specs / {} round(s), {} rows → {}",
        result.plan.specs.len(),
        result.plan.round + 1,
        result.dataset.len(),
        out.display()
    );

    // 6. Optional export.
    if export {
        cmd_export(&cfg, None, None)?;
    }
    Ok(())
}

/// Learning-by-doing multi-goal pipeline (track 20 slice 5). Thin shim — the
/// orchestration lives in `scrt_evolve::goals::run_buildable` (styleguide §1).
/// Runs discover → generate per goal; NO eval-gating / keep|rollback yet (those
/// are lane-gated on tracks 10/15 — carry-forward).
fn cmd_evolve_goals(cfg: &EvolveConfig) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;

    println!(
        "evolve --goals: {} goal(s) — discover → generate per goal (no eval gate yet)",
        cfg.goals.len()
    );

    let report = scrt_evolve::goals::run_buildable(cfg, scrt_evolve::generate::run)?;

    for run in &report.runs {
        println!(
            "  - {:24} {:>4} passages, {:>4} rows  [{}]",
            run.goal, run.passages, run.rows, run.note
        );
        if let Some(p) = &run.dataset_path {
            println!("      dataset → {}", p.display());
        }
    }
    println!(
        "evolve --goals: {} total rows across {} goal(s)",
        report.total_rows(),
        report.runs.len()
    );
    Ok(())
}

/// The EVAL-GATED multi-goal SCHEDULE (track 20 slices 6–9). Wires production
/// hooks (discover/generate from the SDK; train/score as Python subprocesses)
/// into `scrt_evolve::rounds::run_schedule`. Every weight change goes through the
/// track-15 transaction; the schedule halts on catastrophe.
fn cmd_evolve_schedule(
    cfg: &EvolveConfig,
    policy: &str,
    max_rounds: usize,
    python: Option<String>,
) -> Result<()> {
    use scrt_evolve::rounds::{run_schedule, RoundHooks, SchedulePolicy};

    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    if cfg.goals.is_empty() {
        anyhow::bail!("evolve --schedule: no [[goals]] in the config");
    }

    let policy = match policy {
        "round-robin" | "roundrobin" | "rr" => SchedulePolicy::RoundRobin,
        "weighted" | "w" => SchedulePolicy::Weighted,
        other => {
            anyhow::bail!("evolve --schedule: unknown policy \"{other}\" (round-robin|weighted)")
        }
    };
    let py = python.clone();

    // Hardware pre-flight: if [hardware] is declared, surface whether this box
    // can train a state-space (Mamba) model — a heads-up before a long run that
    // would otherwise segfault on the first backward. Generic (kernel-keyed), not
    // model-specific; advisory only (the operator chooses the student model).
    if let Some(hw) = &cfg.hardware {
        match hw.can_train_state_space() {
            Ok(()) => println!(
                "hardware: device={} vram={}GB, mamba kernels present — state-space \
                 training enabled",
                hw.device, hw.vram_gb
            ),
            Err(reason) => eprintln!(
                "hardware WARNING: {reason}.\n  → A HYBRID-SSM student (e.g. Granite) \
                 will SEGFAULT on the training backward here. Use a CUDA torch + \
                 mamba kernels, or set model_path to a non-Mamba model. (Eval + \
                 teacher are forward-only and run fine.)"
            ),
        }
    }

    println!(
        "evolve --schedule: {} goal(s), max_rounds={max_rounds}, policy={policy:?} \
         (each round: discover → generate → train → eval → keep|rollback)",
        cfg.goals.len()
    );

    // Regulator (checkpoint store / quarantine) + resume ordinal — computed up
    // front so the per-round log index and baseline can use them.
    let reg = scrt_evolve::Regulator::new(cfg)?;
    let start_ordinal = reg
        .store()
        .list()?
        .iter()
        .map(|m| m.ordinal)
        .max()
        .map(|o| o + 1)
        .unwrap_or(1);

    // --- Production hooks ---
    let discover = |c: &EvolveConfig| scrt_evolve::discover::run(c);
    let generate =
        |c: &EvolveConfig, ctx: &scrt_evolve::DiscoveredContext| scrt_evolve::generate::run(c, ctx);
    // Per-round log index (each train/score call writes to work_dir/logs/).
    let log_seq = std::cell::Cell::new(start_ordinal);
    // The weight-mutating step: write the round's train set, shell out to the
    // transformers trainer, and return the rows' `gen` provenance (quarantine key).
    let py_train = py.clone();
    let train = |c: &EvolveConfig, train_set: &scrt_evolve::Dataset| -> Result<Vec<String>> {
        let wd = WorkDir::from_config(c);
        let data_path = wd.root().join("dataset.round-train.jsonl");
        train_set.write_jsonl(&data_path)?;
        // Per-round durable log: work_dir/logs/round-<n>.log (train + score tee here).
        let n = log_seq.get();
        let log_path = wd.root().join("logs").join(format!("round-{n}.log"));
        std::env::set_var("SCRT_EVOLVE_LOG_FILE", &log_path);
        println!("  round log → {}", log_path.display());
        // Reuse the existing transformers trainer shim (defaults for steps/seq).
        cmd_train_transformers(c, Some(data_path), py_train.clone(), None, 40, 256)?;
        // Provenance = the distinct `gen` stamps in this round's training rows.
        Ok(provenance_of(train_set))
    };
    let py_score = py.clone();
    let score = |c: &EvolveConfig| -> Result<scrt_evolve::ScoreReport> {
        let r = scrt_evolve::eval::run_eval(c, py_score.as_deref());
        // Advance the per-round log index AFTER score, so train+score of one
        // round share round-<n>.log, then the next round increments.
        log_seq.set(log_seq.get() + 1);
        std::env::remove_var("SCRT_EVOLVE_LOG_FILE");
        r
    };

    let hooks = RoundHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
    };

    // Baseline: the last good score if present (read from the last_good
    // checkpoint's metrics), else a conservative zero baseline so the first
    // round can only improve.
    let last_good_score = reg
        .store()
        .last_good()
        .and_then(|id| reg.store().load_manifest(&id).ok())
        .and_then(|m| m.metrics);
    let baseline = move |_g: &scrt_evolve::GoalConfig| -> scrt_evolve::ScoreReport {
        last_good_score
            .clone()
            .unwrap_or_else(|| scrt_evolve::ScoreReport::uncovered("probe-none", "baseline"))
    };

    // start_ordinal (resume point) was computed up front.
    let report = run_schedule(cfg, policy, max_rounds, start_ordinal, &hooks, &baseline)?;

    println!(
        "\nevolve --schedule: {} round(s), {} committed{}",
        report.rounds.len(),
        report.committed(),
        if report.halted {
            " — HALTED on catastrophe"
        } else {
            ""
        }
    );
    for r in &report.rounds {
        let corr = r
            .metrics
            .as_ref()
            .map(|m| format!("{:.3}", m.correctness))
            .unwrap_or_else(|| "-".into());
        println!(
            "  #{:<3} {:20} rows={:<4} correctness={:<6} [{}]",
            r.ordinal, r.goal, r.rows, corr, r.note
        );
    }
    if report.halted {
        eprintln!(
            "\nschedule halted. Inspect: `scrt-evolve quarantine list`, \
             `scrt-evolve checkpoints list`. Re-arm with `scrt-evolve quarantine clear`."
        );
    }
    Ok(())
}

/// The distinct `gen` provenance stamps present in a dataset (the quarantine key
/// for a round's training rows).
fn provenance_of(ds: &scrt_evolve::Dataset) -> Vec<String> {
    use scrt_evolve::GenExample::*;
    let mut set = std::collections::BTreeSet::new();
    for row in &ds.rows {
        let g = match row {
            Qa { gen, .. } | Instruction { gen, .. } | ToolCall { gen, .. } | Cli { gen, .. } => {
                gen.clone()
            }
            // Variants without a `gen` field carry no provenance. Listed
            // explicitly so a new variant forces a compile-time decision here.
            Completion { .. } | Contrastive { .. } => None,
        };
        if let Some(g) = g {
            set.insert(g);
        }
    }
    set.into_iter().collect()
}

/// Score the current model against the probe set (track 10). Thin shim — the
/// scoring + backend dispatch live in `scrt_evolve::eval::run_eval`.
fn cmd_eval(cfg: &EvolveConfig, probe: Option<PathBuf>, python: Option<String>) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;

    // Apply a --probe override by patching the config's eval block.
    let mut cfg = cfg.clone();
    if let Some(p) = probe {
        cfg.eval.get_or_insert_with(Default::default).probe_path = Some(p);
    }
    if cfg.eval.is_none() {
        eprintln!(
            "eval: no [evolve.eval] block — scoring with defaults (api backend). \
             Add [evolve.eval] to configure the probe/backend."
        );
    }

    // Resolve the interpreter (flag > $SCRT_EVOLVE_PYTHON > [hardware].python) so
    // the transformers scorer honors the same binding as train/export (track 28).
    let py = resolve_python(Some(&cfg), python);
    let report = scrt_evolve::eval::run_eval(&cfg, Some(py.as_str()))?;
    let out = wd.root().join("score.json");
    report.write(&out)?;

    println!(
        "eval: correctness={:.3} n={} backend={} probe={}",
        report.correctness, report.n, report.backend, report.probe_version
    );
    if let Some(c) = report.constitution_adherence {
        println!("  constitution_adherence={c:.3}");
    }
    if let Some(d) = report.mean_exit_depth {
        println!("  mean_exit_depth={d:.3}");
    }
    if let Some(p) = report.perplexity {
        println!("  perplexity={p:.3}");
    }
    println!("  report → {}", out.display());
    emit_json(serde_json::json!({
        "command": "eval",
        "correctness": report.correctness,
        "n": report.n,
        "backend": report.backend,
        "probe_version": report.probe_version,
        "constitution_adherence": report.constitution_adherence,
        "mean_exit_depth": report.mean_exit_depth,
        "perplexity": report.perplexity,
        "out": out.display().to_string(),
        "status": "ok",
    }));
    Ok(())
}

/// Carve a held-out probe set from a dataset (track 10). Thin shim over
/// `ProbeSet::carve`, which guarantees + re-asserts zero training overlap.
fn cmd_probe_build(
    cfg: &EvolveConfig,
    from: Option<PathBuf>,
    holdout: Option<f32>,
    out: Option<PathBuf>,
    remainder: Option<PathBuf>,
) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;

    let from_path = from.unwrap_or_else(|| wd.dataset_jsonl());
    if !from_path.exists() {
        anyhow::bail!(
            "probe build: dataset not found at {} — run `scrt-evolve generate` first",
            from_path.display()
        );
    }
    let dataset = scrt_evolve::Dataset::read_jsonl(&from_path)
        .with_context(|| format!("reading dataset {}", from_path.display()))?;

    let frac = holdout
        .or_else(|| cfg.eval.as_ref().map(|e| e.probe_holdout_frac))
        .unwrap_or(0.1);

    let (probe, train) = scrt_evolve::ProbeSet::carve(&dataset, frac)?;

    let probe_out = out.unwrap_or_else(|| scrt_evolve::eval::probe_path(cfg));
    let train_out = remainder.unwrap_or_else(|| wd.root().join("dataset.train.jsonl"));

    probe.write(&probe_out)?;
    train.write_jsonl(&train_out)?;

    println!(
        "probe build: {} probe items (holdout={:.0}%), {} train rows  [{}]",
        probe.len(),
        frac * 100.0,
        train.len(),
        probe.version,
    );
    println!("  probe → {}", probe_out.display());
    println!("  train → {}", train_out.display());
    emit_json(serde_json::json!({
        "command": "probe-build",
        "probe_items": probe.len(),
        "train_rows": train.len(),
        "holdout_frac": frac,
        "probe_version": probe.version,
        "probe": probe_out.display().to_string(),
        "train": train_out.display().to_string(),
        "status": "ok",
    }));
    Ok(())
}

/// List checkpoints + the `last_good` pointer (track 15).
fn cmd_checkpoints_list(cfg: &EvolveConfig) -> Result<()> {
    let reg = scrt_evolve::Regulator::new(cfg)?;
    let store = reg.store();
    let last_good = store.last_good();
    let all = store.list()?;
    if all.is_empty() {
        println!("checkpoints: none yet (run an eval-gated step to produce one)");
        return Ok(());
    }
    println!(
        "checkpoints (last_good = {}):",
        last_good.as_deref().unwrap_or("none")
    );
    for m in &all {
        let corr = m
            .metrics
            .as_ref()
            .map(|s| format!("{:.3}", s.correctness))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "  {:20} {:11} {:12} correctness={}",
            m.id,
            format!("{:?}", m.status),
            m.step_kind,
            corr
        );
    }
    Ok(())
}

/// Show one checkpoint manifest (track 15).
fn cmd_checkpoints_show(cfg: &EvolveConfig, id: &str) -> Result<()> {
    let reg = scrt_evolve::Regulator::new(cfg)?;
    let m = reg.store().load_manifest(id)?;
    println!("{}", serde_json::to_string_pretty(&m)?);
    Ok(())
}

/// Restore the adapter from a checkpoint (manual rollback, track 15).
fn cmd_checkpoints_restore(cfg: &EvolveConfig, id: &str) -> Result<()> {
    let reg = scrt_evolve::Regulator::new(cfg)?;
    let wd = WorkDir::from_config(cfg);
    let adapter = wd.root().join("adapter");
    reg.store().restore_adapter(id, &adapter)?;
    println!(
        "restored adapter from checkpoint {id} → {}",
        adapter.display()
    );
    Ok(())
}

/// List quarantined provenance stamps (track 15).
fn cmd_quarantine_list(cfg: &EvolveConfig) -> Result<()> {
    let reg = scrt_evolve::Regulator::new(cfg)?;
    let q = reg.quarantine()?;
    if q.is_empty() {
        println!("quarantine: empty");
    } else {
        println!("quarantine ({} stamp(s)):", q.gen_stamps.len());
        for s in &q.gen_stamps {
            println!("  {s}");
        }
    }
    Ok(())
}

/// Clear the quarantine = re-arm (track 15).
fn cmd_quarantine_clear(cfg: &EvolveConfig) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    let path = wd.root().join("quarantine.json");
    let empty = scrt_evolve::Quarantine::default();
    empty.write(&path)?;
    println!("quarantine cleared (re-armed) → {}", path.display());
    Ok(())
}

/// Load the dataset and run training, printing the report. Thin shim — all
/// orchestration lives in `scrt_evolve::train::run` (styleguide §1).
fn cmd_train(cfg: &EvolveConfig, data: Option<PathBuf>) -> Result<()> {
    eprintln!(
        "train: the candle backend is a mechanical FIXTURE — it cannot load real \
         pretrained models. For real training use the Python/transformers path \
         (`train-transformers`) or the eval-gated schedule (`evolve --schedule`)."
    );
    let wd = WorkDir::from_config(cfg);
    let data_path = data.unwrap_or_else(|| wd.dataset_jsonl());
    if !data_path.exists() {
        anyhow::bail!(
            "train(candle): dataset not found at {} — run `scrt-evolve generate` first",
            data_path.display()
        );
    }
    let dataset = scrt_evolve::Dataset::read_jsonl(&data_path)
        .with_context(|| format!("reading dataset {}", data_path.display()))?;

    let report = scrt_evolve::train::run(cfg, &dataset)?;
    println!(
        "train: preset={} steps={} final_loss={}",
        report.preset,
        report.steps,
        report
            .final_loss
            .map(|l| format!("{l:.4}"))
            .unwrap_or_else(|| "n/a".to_string()),
    );
    if let Some(artifact) = &report.artifact {
        println!("train: artifact → {}", artifact.display());
    }
    emit_json(serde_json::json!({
        "command": "train",
        "backend": "candle",
        "is_fixture": true,
        "preset": report.preset,
        "steps": report.steps,
        "final_loss": report.final_loss,
        "artifact": report.artifact.as_ref().map(|a| a.display().to_string()),
        "status": "ok",
    }));
    Ok(())
}

/// Real-model training path: shell out to the standalone Python trainer
/// (`python/scrt_evolve_train`). Loads a real HuggingFace causal-LM via
/// `transformers` and LoRA-trains it on the dataset.jsonl. The candle path
/// (`cmd_train`) is the fixture; this is the one that handles RoPE/GQA models.
// Args mirror the `train` subcommand's flags 1:1; bundling them buys nothing.
#[allow(clippy::too_many_arguments)]
fn cmd_train_transformers(
    cfg: &EvolveConfig,
    data: Option<PathBuf>,
    python: Option<String>,
    out: Option<PathBuf>,
    steps: usize,
    max_seq_len: usize,
) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    let data_path = data.unwrap_or_else(|| wd.dataset_jsonl());
    if !data_path.exists() {
        anyhow::bail!(
            "train(transformers): dataset not found at {} — run `generate` first",
            data_path.display()
        );
    }
    let model_path = cfg
        .evolve
        .model_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("train(transformers): set [evolve].model_path"))?;
    let out_dir = out.unwrap_or_else(|| wd.root().join("adapter"));

    // LoRA hyperparameters from [train.lora] (or defaults).
    let lora = cfg
        .train
        .as_ref()
        .and_then(|t| t.lora.clone())
        .unwrap_or_default();
    let targets = lora.target_modules.join(",");

    // Interpreter from config/env/flag (track 28 binding); the installed
    // `scrt-evolve-ml` package is preferred. A repo checkout's `python/` dir is
    // added to PYTHONPATH only as a fallback (dev mode).
    let py = resolve_python(Some(cfg), python);
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("-m")
        .arg("scrt_evolve_train")
        .arg("--dataset")
        .arg(&data_path)
        .arg("--model")
        .arg(&model_path)
        .arg("--out")
        .arg(&out_dir)
        .arg("--steps")
        .arg(steps.to_string())
        .arg("--max-seq-len")
        .arg(max_seq_len.to_string())
        .arg("--lr")
        .arg(lora.lr.to_string())
        .arg("--rank")
        .arg(lora.rank.to_string())
        .arg("--alpha")
        .arg(lora.alpha.to_string())
        .arg("--target-modules")
        .arg(&targets);
    // CONTINUE from an existing adapter (config-driven "further training") so a
    // branch keeps evolving across rounds instead of restarting fresh.
    if let Some(init) = lora.init_adapter.as_ref() {
        cmd.arg("--resume-adapter").arg(init);
    }
    if let Some(pkg_parent) = find_python_pkg_dir() {
        cmd.env("PYTHONPATH", &pkg_parent);
    }

    // Hardware pass-through: forward [hardware].device so GPU usage is fully
    // config-driven (default "auto" lets the Python side pick cuda-if-available).
    if let Some(hw) = cfg.hardware.as_ref() {
        cmd.arg("--device").arg(&hw.device);
    }

    // QAT pass-through (track 23): if [train.qat] is set + enabled, forward the
    // fake-quant flags to the Python trainer.
    if let Some(qat) = cfg.train.as_ref().and_then(|t| t.qat.clone()) {
        if qat.enabled {
            cmd.arg("--qat")
                .arg(&qat.quant)
                .arg("--qat-group-size")
                .arg(qat.group_size.to_string())
                .arg("--qat-calibrate")
                .arg(qat.calibrate_batches.to_string());
        }
    }

    // Cross-MODEL seam distillation pass-through: if [train.distill] is set +
    // enabled + has a teacher, switch the trainer to two-phase distillation
    // (teacher pre-captures seam targets → student trains against the cache).
    // Takes precedence over plain fractional (it IS the fractional streaming,
    // re-targeted at a distinct teacher). Reuses [train.fractional] for the
    // block-size / calib-batches VRAM knobs.
    let frac = cfg.train.as_ref().and_then(|t| t.fractional.clone());
    let distill = cfg.train.as_ref().and_then(|t| t.distill.clone());
    let distill_active = distill
        .as_ref()
        .is_some_and(|d| d.enabled && d.teacher_model.is_some());
    if distill_active {
        for a in distill_args(distill.as_ref().unwrap(), frac.as_ref()) {
            cmd.arg(a);
        }
    } else if let Some(frac) = frac.as_ref() {
        // Fractional / sharded layer-block training pass-through: if
        // [train.fractional] is set + enabled, switch the trainer to block-local
        // distillation (bounds peak VRAM to one block). Config-driven so the same
        // pipeline runs on a small GPU without code changes.
        if frac.enabled {
            cmd.arg("--shard-mode");
            if let Some(bs) = frac.block_size {
                cmd.arg("--block-size").arg(bs.to_string());
            } else if let Some(n) = frac.shards {
                cmd.arg("--shards").arg(n.to_string());
            }
            // Block ROTATION (ambient daemon): train only this block index.
            if let Some(idx) = frac.shard_index {
                cmd.arg("--shard-index").arg(idx.to_string());
            }
            cmd.arg("--calib-batches")
                .arg(frac.calib_batches.to_string());
            cmd.arg("--granularity").arg(&frac.granularity);
            cmd.arg("--objective").arg(&frac.objective);
        }
    }

    println!(
        "train(transformers): {} -m scrt_evolve_train  (model={}, {} steps)",
        py,
        model_path.display(),
        steps
    );

    // Log capture: when SCRT_EVOLVE_LOG_FILE is set (the schedule sets it per
    // round), tee the trainer's stdout+stderr to that file so a multi-day run
    // has a durable, inspectable trail — and a crash (e.g. a CPU SSM segfault)
    // is captured automatically instead of vanishing with the console.
    let status = run_subprocess_logged(&mut cmd, &py, "scrt_evolve_train")?;
    if !status.success() {
        return Err(subprocess_failure("scrt_evolve_train", &py, status));
    }
    println!("train(transformers): adapter → {}", out_dir.display());
    emit_json(serde_json::json!({
        "command": "train",
        "backend": "transformers",
        "is_fixture": false,
        "model": model_path.display().to_string(),
        "adapter": out_dir.display().to_string(),
        "steps": steps,
        "status": "ok",
    }));
    Ok(())
}

/// Build the `scrt_evolve_train` flags for cross-MODEL seam distillation from
/// `[train.distill]` (+ the `[train.fractional]` VRAM knobs it reuses). Pure +
/// total so it is unit-testable without spawning a subprocess. Caller guarantees
/// `distill.teacher_model` is `Some` (checked at the call site).
fn distill_args(
    distill: &scrt_evolve::config::DistillConfig,
    frac: Option<&scrt_evolve::config::FractionalConfig>,
) -> Vec<String> {
    let mut args = vec![
        "--distill-mode".to_string(),
        "--teacher-model".to_string(),
        distill.teacher_model.clone().unwrap_or_default(),
        "--layer-map".to_string(),
        distill.layer_map.clone(),
        "--distill-loss".to_string(),
        distill.loss.clone(),
        "--projection".to_string(),
        distill.projection.clone(),
        "--grad-clip".to_string(),
        distill.grad_clip.to_string(),
        "--lr-mode".to_string(),
        distill.lr_mode.clone(),
    ];
    if let Some(cache) = distill.teacher_cache.as_ref() {
        args.push("--teacher-cache".to_string());
        args.push(cache.clone());
    }
    // VRAM streaming: reuse [train.fractional]'s block-size / calib-batches so a
    // large teacher streams one block at a time (absent ⇒ Python defaults).
    if let Some(frac) = frac {
        if let Some(bs) = frac.block_size {
            args.push("--block-size".to_string());
            args.push(bs.to_string());
        } else if let Some(n) = frac.shards {
            args.push("--shards".to_string());
            args.push(n.to_string());
        }
        args.push("--calib-batches".to_string());
        args.push(frac.calib_batches.to_string());
    }
    args
}

#[cfg(test)]
mod distill_args_tests {
    use super::distill_args;
    use scrt_evolve::config::{DistillConfig, FractionalConfig};

    #[test]
    fn distill_args_emits_teacher_and_map_defaults() {
        let d = DistillConfig {
            teacher_model: Some("/models/mistral-7b".to_string()),
            ..DistillConfig::default()
        };
        let args = distill_args(&d, None);
        // Mode + teacher are always present; defaults flow through.
        assert!(args.contains(&"--distill-mode".to_string()));
        let ti = args.iter().position(|a| a == "--teacher-model").unwrap();
        assert_eq!(args[ti + 1], "/models/mistral-7b");
        let li = args.iter().position(|a| a == "--layer-map").unwrap();
        assert_eq!(args[li + 1], "stride");
        let lo = args.iter().position(|a| a == "--distill-loss").unwrap();
        assert_eq!(args[lo + 1], "cosine_mse");
        // Stability defaults: grad clip 1.0, auto LR.
        let gc = args.iter().position(|a| a == "--grad-clip").unwrap();
        assert_eq!(args[gc + 1], "1");
        let lm = args.iter().position(|a| a == "--lr-mode").unwrap();
        assert_eq!(args[lm + 1], "auto");
        // No fractional ⇒ no streaming knobs.
        assert!(!args.contains(&"--block-size".to_string()));
        assert!(!args.contains(&"--calib-batches".to_string()));
    }

    #[test]
    fn distill_args_reuses_fractional_block_size() {
        let d = DistillConfig {
            teacher_model: Some("/t".to_string()),
            teacher_cache: Some("/tmp/seams".to_string()),
            ..DistillConfig::default()
        };
        let frac = FractionalConfig {
            block_size: Some(2),
            calib_batches: 16,
            ..FractionalConfig::default()
        };
        let args = distill_args(&d, Some(&frac));
        let bi = args.iter().position(|a| a == "--block-size").unwrap();
        assert_eq!(args[bi + 1], "2");
        let ci = args.iter().position(|a| a == "--calib-batches").unwrap();
        assert_eq!(args[ci + 1], "16");
        let tc = args.iter().position(|a| a == "--teacher-cache").unwrap();
        assert_eq!(args[tc + 1], "/tmp/seams");
    }
}

/// Run a subprocess, optionally teeing its combined output to the file named by
/// `SCRT_EVOLVE_LOG_FILE`. Without that env var, output inherits the console
/// (today's behavior). The captured file is appended to (one round may run
/// train then score into the same file).
fn run_subprocess_logged(
    cmd: &mut std::process::Command,
    py: &str,
    module: &str,
) -> Result<std::process::ExitStatus> {
    let log_file = std::env::var("SCRT_EVOLVE_LOG_FILE").ok();
    match log_file {
        None => cmd
            .status()
            .with_context(|| format!("launching `{py} -m {module}`")),
        Some(path) => {
            use std::io::Write;
            let out = cmd
                .stderr(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .output()
                .with_context(|| format!("launching `{py} -m {module}`"))?;
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = writeln!(f, "===== {module} =====");
                let _ = f.write_all(&out.stdout);
                let _ = f.write_all(&out.stderr);
            }
            // Mirror stdout to the console too (so the JSON summary line the
            // caller may parse, and progress, are still visible live-ish).
            std::io::stdout().write_all(&out.stdout).ok();
            std::io::stderr().write_all(&out.stderr).ok();
            Ok(out.status)
        }
    }
}

/// Inference shim: shell out to `python -m scrt_evolve_infer`.
///
/// Reads base model path from cfg.evolve.model_path; adapter dir defaults to
/// work_dir/adapter. All generation flags are passed through transparently.
/// Mirrors the structure of cmd_train_transformers exactly.
// Args mirror the `infer` subcommand's flags 1:1; bundling them buys nothing.
#[allow(clippy::too_many_arguments)]
fn cmd_infer(
    cfg: &EvolveConfig,
    adapter: Option<PathBuf>,
    prompt: &str,
    ab: bool,
    max_new_tokens: usize,
    temperature: f32,
    chat: bool,
    python: Option<String>,
) -> Result<()> {
    let model_path = cfg
        .evolve
        .model_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("infer: set [evolve].model_path in evolve.toml"))?;

    let wd = WorkDir::from_config(cfg);
    let adapter_dir = adapter.unwrap_or_else(|| wd.root().join("adapter"));

    let py = resolve_python(Some(cfg), python);
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("-m")
        .arg("scrt_evolve_infer")
        .arg("--model")
        .arg(&model_path)
        .arg("--adapter")
        .arg(&adapter_dir)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-new-tokens")
        .arg(max_new_tokens.to_string())
        .arg("--temperature")
        .arg(temperature.to_string());
    if let Some(pkg_parent) = find_python_pkg_dir() {
        cmd.env("PYTHONPATH", &pkg_parent);
    }

    if ab {
        cmd.arg("--ab");
    }
    if chat {
        cmd.arg("--chat");
    }

    println!(
        "infer: {} -m scrt_evolve_infer  (model={}, adapter={}{})",
        py,
        model_path.display(),
        adapter_dir.display(),
        if ab { ", --ab" } else { "" },
    );

    let status = cmd
        .status()
        .with_context(|| format!("launching `{py} -m scrt_evolve_infer`"))?;
    if !status.success() {
        return Err(subprocess_failure("scrt_evolve_infer", &py, status));
    }
    Ok(())
}

/// Config-driven inference RUNTIME (`run-model`): load + run a model for
/// generation per `[runtime]`. Backend-generic — `llamacpp` serves a GGUF via
/// the llama.cpp `llama-cli` runner (efficient quantized inference; the right
/// path for hybrid-SSM models whose naive transformers forward OOMs), and
/// `transformers` falls through to the Python HF path. This is the dedicated
/// serving lane (vs. `infer`, which is the HF base-vs-adapter A/B comparison).
fn cmd_run_model(cfg: &EvolveConfig, prompt: &str, python: Option<String>) -> Result<()> {
    let rt = cfg.runtime.clone().unwrap_or_default();
    let sampling = rt.sampling.clone().unwrap_or_default();

    match rt.backend.as_str() {
        "llamacpp" => {
            // Resolve the GGUF: [runtime].model_path > [export].out_path.
            let gguf = rt
                .model_path
                .clone()
                .or_else(|| cfg.export.as_ref().and_then(|e| e.out_path.clone()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "run-model(llamacpp): set [runtime].model_path (a .gguf) \
                         or [export].out_path"
                    )
                })?;
            // Resolve the llama-cli runner from [runtime] or [export] llama.cpp path.
            let llama_root = rt
                .llama_cpp_path
                .clone()
                .or_else(|| cfg.export.as_ref().and_then(|e| e.llama_cpp_path.clone()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "run-model(llamacpp): set [runtime].llama_cpp_path (or \
                         [export].llama_cpp_path) to a llama.cpp build with llama-cli"
                    )
                })?;
            // `llama-completion` is the non-interactive runner in current
            // llama.cpp (the old `llama-cli` no longer supports `-no-cnv`).
            let root = llama_root.trim_end_matches('/');
            let cli = format!("{root}/build/bin/llama-completion");

            let mut cmd = std::process::Command::new(&cli);
            cmd.arg("-m")
                .arg(&gguf)
                .arg("-p")
                .arg(prompt)
                .arg("-n")
                .arg(sampling.max_tokens.to_string())
                .arg("--temp")
                .arg(sampling.temperature.to_string())
                .arg("--top-p")
                .arg(sampling.top_p.to_string())
                .arg("-c")
                .arg(rt.n_ctx.to_string())
                .arg("-ngl")
                .arg(rt.n_gpu_layers.to_string());
            if rt.n_threads > 0 {
                cmd.arg("-t").arg(rt.n_threads.to_string());
            }
            println!(
                "run-model(llamacpp): {cli}  (gguf={gguf}, ctx={}, ngl={}, temp={})",
                rt.n_ctx, rt.n_gpu_layers, sampling.temperature
            );
            let status = cmd
                .status()
                .with_context(|| format!("launching llama-completion at {cli}"))?;
            if !status.success() {
                anyhow::bail!(
                    "run-model(llamacpp): `{cli}` exited with {status}.\n  \
                     → Check that the llama.cpp build at [runtime].llama_cpp_path has \
                     `build/bin/llama-completion` (run its build), and that the GGUF \
                     ({gguf}) is a valid model for this llama.cpp version."
                );
            }
            Ok(())
        }
        "transformers" => {
            // Fall through to the Python HF inference path; model from
            // [runtime].model_path or [evolve].model_path.
            let model = rt
                .model_path
                .clone()
                .map(PathBuf::from)
                .or_else(|| cfg.evolve.model_path.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "run-model(transformers): set [runtime].model_path or \
                         [evolve].model_path"
                    )
                })?;
            let py = resolve_python(Some(cfg), python);
            let mut cmd = std::process::Command::new(&py);
            cmd.arg("-m")
                .arg("scrt_evolve_infer")
                .arg("--model")
                .arg(&model)
                .arg("--prompt")
                .arg(prompt)
                .arg("--max-new-tokens")
                .arg(sampling.max_tokens.to_string())
                .arg("--temperature")
                .arg(sampling.temperature.to_string());
            if let Some(pkg_parent) = find_python_pkg_dir() {
                cmd.env("PYTHONPATH", &pkg_parent);
            }
            println!(
                "run-model(transformers): {py} -m scrt_evolve_infer  (model={})",
                model.display()
            );
            let status = cmd
                .status()
                .with_context(|| format!("launching `{py} -m scrt_evolve_infer`"))?;
            if !status.success() {
                return Err(subprocess_failure("scrt_evolve_infer", &py, status));
            }
            Ok(())
        }
        other => anyhow::bail!(
            "run-model: unknown [runtime].backend '{other}' \
             (expected 'llamacpp' or 'transformers')"
        ),
    }
}

/// Convert a GGUF to an HF safetensors dir (track 23) by shelling out to
/// `python -m scrt_evolve_dequant`. Generic + registry-driven; the Rust side is
/// a thin shim (gguf-py on PYTHONPATH, like export-gguf).
fn cmd_dequant(
    gguf: &PathBuf,
    out: &PathBuf,
    dtype: &str,
    tokenizer: Option<PathBuf>,
    python: Option<String>,
) -> Result<()> {
    if !gguf.exists() {
        anyhow::bail!("dequant: GGUF not found: {}", gguf.display());
    }
    // PYTHONPATH is assembled from whatever's present: the checkout `python/` dir
    // (dev fallback — the installed package needs none) and the vendored gguf-py
    // from a llama.cpp checkout. Either may be absent; an installed `scrt-evolve-ml`
    // + `gguf` pip package covers both. Platform separator: `;` Windows, `:` else.
    let sep = if cfg!(windows) { ";" } else { ":" };
    let mut path_parts: Vec<String> = Vec::new();
    if let Some(p) = find_python_pkg_dir() {
        path_parts.push(p.display().to_string());
    }
    if let Some(p) = find_llama_gguf_py() {
        path_parts.push(p.display().to_string());
    }

    let py = resolve_python(None, python);
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("-m")
        .arg("scrt_evolve_dequant")
        .arg("--gguf")
        .arg(gguf)
        .arg("--out")
        .arg(out)
        .arg("--dtype")
        .arg(dtype);
    if !path_parts.is_empty() {
        cmd.env("PYTHONPATH", path_parts.join(sep));
    }
    if let Some(tok) = &tokenizer {
        cmd.arg("--tokenizer").arg(tok);
    }

    println!(
        "dequant: {} -m scrt_evolve_dequant  (gguf={}, out={}, dtype={})",
        py,
        gguf.display(),
        out.display(),
        dtype
    );
    let status = cmd
        .status()
        .with_context(|| format!("launching `{py} -m scrt_evolve_dequant`"))?;
    if !status.success() {
        return Err(subprocess_failure("scrt_evolve_dequant", &py, status));
    }
    println!("dequant: HF model dir → {}", out.display());
    Ok(())
}

/// Best-effort locate a vendored `gguf-py` dir in a llama.cpp checkout, mirroring
/// the auto-detect in the export path so `dequant` can read GGUFs without the
/// `gguf` pip package installed.
fn find_llama_gguf_py() -> Option<PathBuf> {
    let home = dirs_home()?;
    for base in [
        home.join(".unsloth").join("llama.cpp"),
        home.join("llama.cpp"),
        home.join("Documents").join("llama.cpp"),
    ] {
        let candidate = base.join("gguf-py");
        if candidate.join("gguf").is_dir() {
            return Some(candidate);
        }
    }
    std::env::var("LLAMA_CPP")
        .ok()
        .map(|p| PathBuf::from(p).join("gguf-py"))
        .filter(|p| p.join("gguf").is_dir())
}

/// Home dir without pulling in the `dirs` crate (HOME / USERPROFILE).
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

/// Merge a LoRA adapter into the base model and export a quantized GGUF by
/// shelling out to `python -m scrt_evolve_gguf` (merge → convert → quantize).
#[allow(clippy::too_many_arguments)]
fn cmd_export_gguf(
    cfg: &EvolveConfig,
    adapter: Option<PathBuf>,
    out: Option<PathBuf>,
    quant: Option<String>,
    llama_cpp: Option<PathBuf>,
    keep_intermediates: bool,
    python: Option<String>,
) -> Result<()> {
    let model_path =
        cfg.evolve.model_path.clone().ok_or_else(|| {
            anyhow::anyhow!("export-gguf: set [evolve].model_path in evolve.toml")
        })?;

    let wd = WorkDir::from_config(cfg);
    let adapter_dir = adapter.unwrap_or_else(|| wd.root().join("adapter"));

    // `[export]` config is the source of defaults; explicit CLI flags override.
    let exp = cfg.export.clone().unwrap_or_default();

    // quant: explicit CLI flag wins; otherwise fall back to `[export].quant`
    // (whose own default is Q4_K_M). Track the source so the --json summary can
    // tell an agent whether its flag was honored or the config value won (A2).
    let quant_was_explicit = quant.is_some();
    let quant_eff: String = quant.unwrap_or_else(|| exp.quant.clone());

    let py = resolve_python(Some(cfg), python);
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("-m")
        .arg("scrt_evolve_gguf")
        .arg("--model")
        .arg(&model_path)
        .arg("--adapter")
        .arg(&adapter_dir)
        .arg("--quant")
        .arg(&quant_eff)
        .arg("--dtype")
        .arg(&exp.dtype)
        .arg("--max-shard-size")
        .arg(&exp.max_shard_size);
    if let Some(pkg_parent) = find_python_pkg_dir() {
        cmd.env("PYTHONPATH", &pkg_parent);
    }

    // out: CLI flag > [export].out_path > python default.
    if let Some(o) = &out {
        cmd.arg("--out").arg(o);
    } else if let Some(o) = &exp.out_path {
        cmd.arg("--out").arg(o);
    }
    // llama.cpp: CLI flag > [export].llama_cpp_path > python auto-detect.
    if let Some(lc) = &llama_cpp {
        cmd.arg("--llama-cpp").arg(lc);
    } else if let Some(lc) = &exp.llama_cpp_path {
        cmd.arg("--llama-cpp").arg(lc);
    }
    // scratch + placement from config.
    if let Some(w) = &exp.work_path {
        cmd.arg("--work-dir").arg(w);
    }
    if let Some(p) = &exp.place_dir {
        cmd.arg("--place-dir").arg(p);
    }
    // sharding-merge rule: union per-shard adapters first when enabled.
    if let Some(ms) = &exp.merge_shards {
        if ms.enabled {
            cmd.arg("--merge-shards").arg(&ms.pattern);
        }
    }
    if keep_intermediates || exp.keep_intermediates {
        cmd.arg("--keep-merged").arg("--keep-f16");
    }

    println!(
        "export-gguf: {} -m scrt_evolve_gguf  (model={}, adapter={}, quant={}, dtype={})",
        py,
        model_path.display(),
        adapter_dir.display(),
        quant_eff,
        exp.dtype,
    );

    let status = cmd
        .status()
        .with_context(|| format!("launching `{py} -m scrt_evolve_gguf`"))?;
    if !status.success() {
        return Err(subprocess_failure("scrt_evolve_gguf", &py, status));
    }
    emit_json(serde_json::json!({
        "command": "export-gguf",
        "model": model_path.display().to_string(),
        "adapter": adapter_dir.display().to_string(),
        "quant": quant_eff,
        "quant_source": if quant_was_explicit { "flag" } else { "config" },
        "out": out.as_ref().map(|o| o.display().to_string())
            .or_else(|| exp.out_path.clone()),
        "status": "ok",
    }));
    Ok(())
}

/// Find the `python/` dir for `PYTHONPATH`. Thin re-export of the shared SDK
/// helper so the CLI and the eval subprocess scorer agree on resolution.
fn find_python_pkg_dir() -> Option<PathBuf> {
    scrt_evolve::python_pkg_dir()
}

/// Resolve the Python interpreter for the ML subprocesses (track 28 binding).
/// Precedence: `--python` flag > `$SCRT_EVOLVE_PYTHON` > `[hardware].python` >
/// bare `python`. This is the ONE place the interpreter is chosen, so the
/// installed-package path (`<venv>/python -m scrt_evolve_*`) and the dev checkout
/// agree. The `python/` checkout dir is only a PYTHONPATH fallback (see callers).
fn resolve_python(cfg: Option<&EvolveConfig>, cli: Option<String>) -> String {
    cli.or_else(|| {
        std::env::var("SCRT_EVOLVE_PYTHON")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .or_else(|| {
        cfg.and_then(|c| c.hardware.as_ref())
            .and_then(|h| h.python.clone())
    })
    .unwrap_or_else(|| "python".to_string())
}

/// `config-show` — print the fully-resolved config as JSON (defaults applied).
/// The dry-run analogue of `config-reference`: the schema vs. THIS run's values.
fn cmd_config_show(cfg: &EvolveConfig) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(cfg)?);
    Ok(())
}

/// `commands` — a machine-readable manifest of every subcommand, derived from
/// clap so it never drifts from the real surface. Human list by default; a JSON
/// array with `--json` (the command-surface analogue of `config-reference`).
fn cmd_commands() -> Result<()> {
    use clap::CommandFactory;
    let cli = Cli::command();
    let mut items: Vec<serde_json::Value> = Vec::new();
    for sub in cli.get_subcommands() {
        let flags: Vec<serde_json::Value> = sub
            .get_arguments()
            .filter_map(|a| {
                a.get_long().map(|long| {
                    serde_json::json!({
                        "flag": format!("--{long}"),
                        "help": a.get_help().map(|h| h.to_string()).unwrap_or_default(),
                        "required": a.is_required_set(),
                    })
                })
            })
            .collect();
        items.push(serde_json::json!({
            "name": sub.get_name(),
            "about": sub.get_about().map(|a| a.to_string()).unwrap_or_default(),
            "flags": flags,
        }));
    }
    if JSON_OUTPUT.load(std::sync::atomic::Ordering::Relaxed) {
        println!("{}", serde_json::to_string(&items)?);
    } else {
        println!("scrt-evolve subcommands ({}):", items.len());
        for sub in cli.get_subcommands() {
            let about = sub
                .get_about()
                .map(|a| a.to_string())
                .unwrap_or_default()
                .lines()
                .next()
                .unwrap_or("")
                .to_string();
            println!("  {:<18} {}", sub.get_name(), about);
        }
        println!(
            "\nRun `scrt-evolve <cmd> --help` for flags, or `commands --json` for a manifest."
        );
    }
    Ok(())
}

/// `doctor` — preflight the environment so a long real-model run fails in 2
/// seconds with a fix, not at minute 9 with a traceback. Each check prints
/// PASS/FAIL + a remediation; `--json` emits a structured report. Reuses the
/// same finder helpers the real commands use, so it checks what they check.
fn cmd_doctor(config: &std::path::Path, python: Option<String>) -> Result<()> {
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let mut check = |name: &str, ok: bool, detail: String| {
        checks.push((name.to_string(), ok, detail));
    };

    // 1. Config parses.
    let cfg = match EvolveConfig::load(config) {
        Ok(c) => {
            check("config_parse", true, format!("{} parsed", config.display()));
            Some(c)
        }
        Err(e) => {
            check(
                "config_parse",
                false,
                format!("{} — {e:#}. Fix the toml or run `init`.", config.display()),
            );
            None
        }
    };

    // 2. model_path exists (only when set).
    match cfg.as_ref().and_then(|c| c.evolve.model_path.clone()) {
        Some(p) if p.exists() => check("model_path", true, format!("{} exists", p.display())),
        Some(p) => check(
            "model_path",
            false,
            format!(
                "{} does not exist — set [evolve].model_path to your HF model dir",
                p.display()
            ),
        ),
        None => check(
            "model_path",
            false,
            "[evolve].model_path is unset — required for train/infer/export".to_string(),
        ),
    }

    // 3. python/ package dir located (shared helper the real commands use).
    match find_python_pkg_dir() {
        Some(p) => check(
            "python_pkg_dir",
            true,
            format!("python/ at {}", p.display()),
        ),
        None => check(
            "python_pkg_dir",
            false,
            "could not locate the python/ package dir — run from the repo checkout \
             or install scrt-evolve-ml on PYTHONPATH"
                .to_string(),
        ),
    }

    // 4. Interpreter ML deps (track 28): one probe reports torch/cuda/
    //    transformers/safetensors/mamba so `doctor` covers the packaging binding.
    //    The interpreter is resolved the SAME way the real commands resolve it
    //    (flag > $SCRT_EVOLVE_PYTHON > [hardware].python > python).
    let py = resolve_python(cfg.as_ref(), python);
    let probe = "import json\nr={}\n\
        try:\n import torch; r['torch']=torch.__version__; r['cuda']=bool(torch.cuda.is_available())\n\
        except Exception as e: r['torch_err']=str(e)\n\
        try:\n import transformers; r['transformers']=transformers.__version__\n\
        except Exception as e: r['transformers_err']=str(e)\n\
        try:\n import safetensors; r['safetensors']=True\n\
        except Exception as e: r['safetensors_err']=str(e)\n\
        try:\n import mamba_ssm, causal_conv1d; r['mamba']=True\n\
        except Exception: r['mamba']=False\n\
        print(json.dumps(r))";
    let probe_out = std::process::Command::new(&py)
        .arg("-c")
        .arg(probe)
        .output();
    match probe_out {
        Ok(o) if o.status.success() => {
            let txt = String::from_utf8_lossy(&o.stdout);
            let j: serde_json::Value = serde_json::from_str(txt.trim()).unwrap_or_default();
            let has = |k: &str| j.get(k).map(|v| !v.is_null()).unwrap_or(false);
            let core_ok = has("torch") && has("transformers") && has("safetensors");
            if core_ok {
                check(
                    "python_deps",
                    true,
                    format!(
                        "`{py}` torch={} transformers={}",
                        j.get("torch").and_then(|v| v.as_str()).unwrap_or("?"),
                        j.get("transformers")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?"),
                    ),
                );
            } else {
                check(
                    "python_deps",
                    false,
                    format!(
                        "`{py}` is missing torch/transformers/safetensors — \
                         `pip install scrt-evolve-ml[cuda]` into the venv, then set \
                         [hardware].python / $SCRT_EVOLVE_PYTHON to it"
                    ),
                );
            }
            // Capability notes (not hard failures: CPU-only / non-SSM is valid).
            let cuda = j.get("cuda").and_then(|v| v.as_bool()).unwrap_or(false);
            check(
                "cuda",
                true,
                if cuda {
                    "torch.cuda.is_available() = true".to_string()
                } else {
                    "CUDA not available (CPU-only — fine for eval/api; real GPU training needs CUDA torch)".to_string()
                },
            );
            let mamba = j.get("mamba").and_then(|v| v.as_bool()).unwrap_or(false);
            check(
                "mamba_kernels",
                true,
                if mamba {
                    "mamba-ssm + causal-conv1d importable (hybrid-SSM training OK)".to_string()
                } else {
                    "mamba-ssm/causal-conv1d absent (hybrid-SSM TRAINING will segfault; non-SSM models fine). See PORTABILITY.md".to_string()
                },
            );
        }
        Ok(_) | Err(_) => check(
            "python_deps",
            false,
            format!(
                "could not run the dep probe with `{py}` — install scrt-evolve-ml \
                 and set [hardware].python / $SCRT_EVOLVE_PYTHON, or pass --python"
            ),
        ),
    }

    // 5. llama.cpp auto-detect (gguf-py — needed by export/dequant).
    match find_llama_gguf_py() {
        Some(p) => check("llama_cpp", true, format!("gguf-py at {}", p.display())),
        None => check(
            "llama_cpp",
            false,
            "no llama.cpp checkout auto-detected (~/.unsloth/llama.cpp, ~/llama.cpp, \
             $LLAMA_CPP) — needed for export-gguf/dequant; clone llama.cpp or set \
             [export].llama_cpp_path"
                .to_string(),
        ),
    }

    // 6. work_dir writable.
    if let Some(c) = cfg.as_ref() {
        let wd = WorkDir::from_config(c);
        let writable =
            wd.ensure().is_ok() && std::fs::write(wd.root().join(".doctor-probe"), b"ok").is_ok();
        let _ = std::fs::remove_file(wd.root().join(".doctor-probe"));
        if writable {
            check(
                "work_dir",
                true,
                format!("{} writable", wd.root().display()),
            );
        } else {
            check(
                "work_dir",
                false,
                format!("{} is not writable", wd.root().display()),
            );
        }
    }

    let failed = checks.iter().filter(|(_, ok, _)| !ok).count();

    if JSON_OUTPUT.load(std::sync::atomic::Ordering::Relaxed) {
        let arr: Vec<serde_json::Value> = checks
            .iter()
            .map(|(name, ok, detail)| {
                serde_json::json!({"check": name, "pass": ok, "detail": detail})
            })
            .collect();
        emit_json(serde_json::json!({
            "command": "doctor",
            "checks": arr,
            "failed": failed,
            "status": if failed == 0 { "ok" } else { "fail" },
        }));
    } else {
        println!("scrt-evolve doctor:");
        for (name, ok, detail) in &checks {
            println!(
                "  [{}] {:<16} {detail}",
                if *ok { "PASS" } else { "FAIL" },
                name
            );
        }
        if failed == 0 {
            println!("\nall checks passed.");
        } else {
            println!("\n{failed} check(s) failed — fix the FAILs above before a real run.");
        }
    }
    Ok(())
}

// ───────────────────────── Ambient daemon (track 26) ─────────────────────────

/// Best-effort free-VRAM probe for the daemon's throttle gate (parses
/// `nvidia-smi`). Returns GB free on the first GPU, or `None` when unavailable —
/// then the gate is skipped (we can't throttle what we can't measure).
fn probe_free_vram_gb() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mb: f64 = text.lines().next()?.trim().parse().ok()?;
    Some(mb / 1024.0)
}

/// Probe whether ANOTHER process is using the GPU (gentle-background gate): list
/// the GPU's compute apps and report `true` if any PID other than ours is present.
/// `None` when `nvidia-smi` is unavailable (then the gate treats the GPU as free).
fn probe_gpu_busy() -> Option<bool> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let me = std::process::id();
    let text = String::from_utf8_lossy(&out.stdout);
    let others = text
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .any(|pid| pid != me);
    Some(others)
}

/// `daemon start` — run the ambient continuous-evolution loop (track 26).
/// Mirrors the `evolve --schedule` production hooks (train via the transformers
/// subprocess, score via the eval harness) but is driven by the living activity
/// queue instead of bounded goal rounds. Runs until `daemon stop`, or bounded by
/// `--max-steps` / `--drain`.
fn cmd_daemon_start(
    cfg: &EvolveConfig,
    python: Option<String>,
    max_vram: Option<f64>,
    max_steps: Option<u64>,
    drain: bool,
) -> Result<()> {
    use scrt_evolve::daemon::{self, DaemonHooks, DaemonOptions};

    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;

    let dcfg = cfg.daemon.clone().unwrap_or_default();
    // Budget: --max-vram > [daemon].max_vram_gb; 0 ⇒ ungated.
    let max_vram_gb = max_vram.or(if dcfg.max_vram_gb > 0.0 {
        Some(dcfg.max_vram_gb)
    } else {
        None
    });

    // Resume the monotonic ordinal from the checkpoint store, and take the last
    // good score as the baseline (else a conservative uncovered baseline).
    let reg = scrt_evolve::Regulator::new(cfg)?;
    let start_ordinal = reg
        .store()
        .list()?
        .iter()
        .map(|m| m.ordinal)
        .max()
        .map(|o| o + 1)
        .unwrap_or(1);
    let baseline = reg
        .store()
        .last_good()
        .and_then(|id| reg.store().load_manifest(&id).ok())
        .and_then(|m| m.metrics)
        .unwrap_or_else(|| scrt_evolve::ScoreReport::uncovered("probe-none", "baseline"));

    // Capture train/score subprocess output to one durable daemon log.
    let _ = std::fs::create_dir_all(wd.root().join("logs"));
    let log_path = wd.root().join("logs").join("daemon.log");
    std::env::set_var("SCRT_EVOLVE_LOG_FILE", &log_path);

    // Clear any stale stop-file; write a run marker for `daemon status`.
    let _ = std::fs::remove_file(daemon::stop_file(wd.root()));
    let _ = std::fs::write(daemon::run_file(wd.root()), b"running");

    // --- Production hooks (mirror cmd_evolve_schedule) ---
    let py_train = python.clone();
    let train = |c: &EvolveConfig, ds: &scrt_evolve::Dataset| -> Result<Vec<String>> {
        let w = WorkDir::from_config(c);
        let data_path = w.root().join("dataset.daemon-step.jsonl");
        ds.write_jsonl(&data_path)?;
        cmd_train_transformers(c, Some(data_path), py_train.clone(), None, 40, 256)?;
        Ok(provenance_of(ds))
    };
    let py_score = python.clone();
    let score = |c: &EvolveConfig| -> Result<scrt_evolve::ScoreReport> {
        scrt_evolve::eval::run_eval(c, py_score.as_deref())
    };
    let free_vram = probe_free_vram_gb;
    let gpu_busy = probe_gpu_busy;
    let hooks = DaemonHooks {
        free_vram_gb: &free_vram,
        gpu_busy: &gpu_busy,
        train: &train,
        score: &score,
    };

    let opts = DaemonOptions {
        max_vram_gb,
        batch: dcfg.batch.max(1),
        max_steps,
        exit_when_empty: drain,
        poll_interval: std::time::Duration::from_secs(dcfg.poll_interval_secs.max(1)),
        start_ordinal,
        pause_on_gpu_process: dcfg.pause_on_gpu_process,
        cpu_fallback: dcfg.cpu_fallback,
        rotation_blocks: dcfg.rotation_blocks,
        cooldown: std::time::Duration::from_secs(dcfg.cooldown_secs),
    };

    println!(
        "daemon start: budget={} batch={} {}{}{}{} (stop with `scrt-evolve daemon stop`)",
        max_vram_gb
            .map(|v| format!("{v:.1}G"))
            .unwrap_or_else(|| "ungated".into()),
        opts.batch,
        if drain { "drain-once" } else { "continuous" },
        if opts.pause_on_gpu_process {
            " yield-to-gpu"
        } else {
            ""
        },
        if opts.cpu_fallback {
            " cpu-fallback"
        } else {
            ""
        },
        if opts.rotation_blocks > 0 {
            format!(" rotate/{}", opts.rotation_blocks)
        } else {
            String::new()
        },
    );

    let stop_root = wd.root().to_path_buf();
    let report = daemon::run_daemon(cfg, &opts, &baseline, &hooks, &|| {
        daemon::stop_requested(&stop_root)
    })?;

    std::env::remove_var("SCRT_EVOLVE_LOG_FILE");
    let _ = std::fs::remove_file(daemon::run_file(wd.root()));
    let _ = std::fs::remove_file(daemon::stop_file(wd.root()));

    let reason = if report.halted {
        "HALTED on catastrophe"
    } else if report.stopped {
        "stopped"
    } else if report.drained {
        "queue drained"
    } else {
        "done"
    };
    println!(
        "daemon: {} step(s), {} committed — {reason}",
        report.steps.len(),
        report.committed()
    );
    for s in &report.steps {
        println!("  #{:<3} items={:<3} {}", s.ordinal, s.items, s.note);
    }
    emit_json(serde_json::json!({
        "command": "daemon-start",
        "steps": report.steps.len(),
        "committed": report.committed(),
        "halted": report.halted,
        "stopped": report.stopped,
        "drained": report.drained,
        "status": if report.halted { "halted" } else { "ok" },
    }));
    if report.halted {
        anyhow::bail!(
            "daemon halted (catastrophe) — see `quarantine list` + evolution-log; \
             re-arm with `quarantine clear`"
        );
    }
    Ok(())
}

/// `daemon stop` — signal a running daemon to exit after its current step.
fn cmd_daemon_stop(cfg: &EvolveConfig) -> Result<()> {
    use scrt_evolve::daemon;
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    daemon::request_stop(wd.root())?;
    println!("daemon stop: requested (the running daemon exits after its current step)");
    emit_json(serde_json::json!({"command": "daemon-stop", "status": "ok"}));
    Ok(())
}

/// `daemon status` — run-state + living-queue pending counts.
fn cmd_daemon_status(cfg: &EvolveConfig) -> Result<()> {
    use scrt_evolve::daemon;
    use scrt_evolve::LivingQueue;
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let queue = LivingQueue::from_config(cfg)?;
    let (prio, raw) = queue.pending();
    let running = daemon::run_file(wd.root()).exists();
    let stopping = daemon::stop_requested(wd.root());
    let state = if stopping {
        "stopping"
    } else if running {
        "running"
    } else {
        "stopped"
    };
    println!("daemon status: {state}");
    println!("  queue pending: priority={prio} raw={raw}");
    emit_json(serde_json::json!({
        "command": "daemon-status",
        "state": state,
        "pending_priority": prio,
        "pending_raw": raw,
        "status": "ok",
    }));
    Ok(())
}

/// `teach` — enqueue an explicit prompt→completion onto the PRIORITY lane.
fn cmd_teach(cfg: &EvolveConfig, prompt: &str, completion: &str) -> Result<()> {
    use scrt_evolve::{GenExample, Lane, LivingQueue};
    let wd = WorkDir::from_config(cfg);
    wd.ensure()?;
    let queue = LivingQueue::from_config(cfg)?;
    let example = GenExample::Qa {
        prompt: prompt.to_string(),
        completion: completion.to_string(),
        source: Some("teach".to_string()),
        gen: Some("teach".to_string()),
    };
    queue.enqueue(Lane::Priority, &example)?;
    let (prio, raw) = queue.pending();
    println!("teach: enqueued to PRIORITY lane (pending: priority={prio} raw={raw})");
    emit_json(serde_json::json!({
        "command": "teach",
        "lane": "priority",
        "pending_priority": prio,
        "pending_raw": raw,
        "status": "ok",
    }));
    Ok(())
}

// ───────────────────────── Branch factory (track 29) ─────────────────────────

/// `branch create` — compose discover → teacher-QA generate → train → eval gate →
/// GGUF export inside the track-15 transaction; register an eval-passing branch.
/// Mirrors the `evolve --schedule` production hooks (no new ML).
#[allow(clippy::too_many_arguments)]
fn cmd_branch_create(
    cfg: &EvolveConfig,
    name: &str,
    base: Option<String>,
    corpus: Option<PathBuf>,
    domain: Option<String>,
    distill: bool,
    teacher: Option<String>,
    steps: usize,
    python: Option<String>,
) -> Result<()> {
    use scrt_evolve::branch::{self, BranchHooks};

    // CLI flags override `[branch]` config defaults.
    let bcfg = cfg.branch.clone().unwrap_or_default();
    let base = base.or_else(|| bcfg.base.clone());
    let corpus = corpus.or_else(|| bcfg.corpus.clone());
    let domain = domain.or_else(|| bcfg.domain.clone());
    let py = python;

    // Distill mode: `--distill`/`--teacher` or `[branch].mode = "distill"`. When
    // active, inject `[train.distill]` into the working config so the branch's
    // train hook runs cross-MODEL seam distillation. The teacher comes from
    // `--teacher` (highest), else the existing `[train.distill].teacher_model`.
    let distill_mode = distill || teacher.is_some() || bcfg.mode == "distill";
    let cfg_owned;
    let cfg: &EvolveConfig = if distill_mode {
        let mut c = cfg.clone();
        let mut train_cfg = c.train.clone().unwrap_or_default();
        let mut dcfg = train_cfg.distill.clone().unwrap_or_default();
        dcfg.enabled = true;
        if let Some(t) = teacher.clone() {
            dcfg.teacher_model = Some(t);
        }
        if dcfg.teacher_model.is_none() {
            anyhow::bail!(
                "branch create --distill: no teacher model — pass --teacher <path> \
                 or set [train.distill].teacher_model"
            );
        }
        train_cfg.distill = Some(dcfg);
        c.train = Some(train_cfg);
        cfg_owned = c;
        &cfg_owned
    } else {
        cfg
    };

    // --- Production hooks (mirror cmd_evolve_schedule) ---
    let discover = |c: &EvolveConfig| scrt_evolve::discover::run(c);
    let generate =
        |c: &EvolveConfig, ctx: &scrt_evolve::DiscoveredContext| scrt_evolve::generate::run(c, ctx);
    let py_train = py.clone();
    let train = |c: &EvolveConfig, train_set: &scrt_evolve::Dataset| -> Result<Vec<String>> {
        let wd = WorkDir::from_config(c);
        let data_path = wd.root().join("dataset.branch-train.jsonl");
        train_set.write_jsonl(&data_path)?;
        cmd_train_transformers(c, Some(data_path), py_train.clone(), None, steps, 256)?;
        Ok(provenance_of(train_set))
    };
    let py_score = py.clone();
    let score = |c: &EvolveConfig| -> Result<scrt_evolve::ScoreReport> {
        scrt_evolve::eval::run_eval(c, py_score.as_deref())
    };
    let py_export = py.clone();
    let export = |c: &EvolveConfig, path: &std::path::Path| -> Result<PathBuf> {
        let adapter = WorkDir::from_config(c).root().join("adapter");
        // quant=None ⇒ cmd_export_gguf resolves it from `[export].quant`
        // (default Q4_K_M), the same value this site used to compute inline.
        cmd_export_gguf(
            c,
            Some(adapter),
            Some(path.to_path_buf()),
            None,
            None,
            false,
            py_export.clone(),
        )?;
        Ok(path.to_path_buf())
    };

    let hooks = BranchHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
        export: &export,
    };

    // Conservative baseline: an uncovered start (the branch can only improve); a
    // catastrophic collapse still trips the catastrophe floor.
    let baseline = scrt_evolve::ScoreReport::uncovered("probe-none", "baseline");
    let created = now_iso8601();

    let report = branch::create(
        cfg,
        name,
        base.as_deref(),
        corpus.as_deref(),
        domain.as_deref(),
        &baseline,
        &created,
        &hooks,
    )?;

    let action = report
        .action
        .map(|a| format!("{a:?}"))
        .unwrap_or_else(|| "bailed".to_string());
    println!("branch create [{name}]: {action} — {}", report.note);
    if let Some(m) = &report.manifest {
        let sha = &m.gguf_sha[..m.gguf_sha.len().min(12)];
        println!(
            "  base={} domain={} gguf_sha={sha}…",
            m.base_model, m.domain
        );
        // Persist the branch's config so it is SELF-DESCRIBING: `branch evolve
        // <name>` reloads this to re-run the same recipe (model paths, train,
        // teacher, eval) with no flags. Best-effort (a write failure is non-fatal).
        persist_branch_config(cfg, name);
    }
    emit_json(serde_json::json!({
        "command": "branch-create",
        "name": name,
        "action": action,
        "registered": report.manifest.is_some() && !report.halt,
        "halt": report.halt,
        "base": report.manifest.as_ref().map(|m| m.base_model.clone()),
        "domain": report.manifest.as_ref().map(|m| m.domain.clone()),
        "gguf_sha": report.manifest.as_ref().map(|m| m.gguf_sha.clone()),
        "note": report.note,
        "status": if report.halt { "halted" } else { "ok" },
    }));
    if report.halt {
        anyhow::bail!("branch create halted (catastrophe) — see quarantine + evolution-log");
    }
    Ok(())
}

/// Persist a branch's config to `branches/<name>/branch.toml` so the branch is
/// SELF-DESCRIBING — `branch evolve <name>` reloads it to re-run the same recipe
/// with no flags. Best-effort (a write failure is non-fatal).
fn persist_branch_config(cfg: &EvolveConfig, name: &str) {
    let dir = cfg.work_dir().join("branches").join(name);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(text) = cfg.to_toml() {
        let _ = std::fs::write(dir.join("branch.toml"), text);
    }
}

/// Run the configured `[hardware].free_gpu_command` (e.g. `lms unload --all`) to
/// evict a GPU-resident teacher before training — on a single-GPU box the teacher
/// and trainer can't both hold VRAM. Best-effort; a failure is logged, not fatal.
fn maybe_free_gpu(cfg: &EvolveConfig) {
    let cmd = match cfg
        .hardware
        .as_ref()
        .and_then(|h| h.free_gpu_command.clone())
    {
        Some(c) if !c.trim().is_empty() => c,
        _ => return,
    };
    println!("free-gpu: {cmd}");
    let result = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", &cmd])
            .status()
    } else {
        std::process::Command::new("sh").args(["-c", &cmd]).status()
    };
    if let Err(e) = result {
        eprintln!("free-gpu: command failed (non-fatal): {e}");
    }
}

/// Build the eval baseline for the next `branch evolve` round from the live
/// stored version: its recorded `correctness` scored on its recorded
/// `probe_version`. This is what makes the cross-round gate REAL — with
/// `[eval].stable_probe`, the candidate is scored on the SAME probe, so both
/// reports carry the same `probe_version` and [`scrt_evolve::eval::classify`]
/// does a genuine same-exam Accept/Regress/Catastrophic.
///
/// `stable_pv` is the version of the on-disk stable probe, if one exists. A
/// version committed before `probe_version` tracking carries `None`; under a
/// stable probe it WAS scored on that same fixed exam, so we anchor its baseline
/// to `stable_pv` rather than dropping to the uncovered path — this makes the
/// very next round a real gate without a manual migration. A stored
/// `probe_version` always wins (a version committed under the feature). With no
/// probe version from either source, the baseline is uncovered (the `n == 0`
/// accept-unless-NaN path) rather than fabricating an id that can never match.
///
/// The sole place a baseline `ScoreReport` is hand-built (Goal 3 centralization).
fn baseline_from_version(
    v: &scrt_evolve::model_store::ModelVersion,
    stable_pv: Option<&str>,
) -> scrt_evolve::ScoreReport {
    let pv = v.probe_version.as_deref().or(stable_pv);
    match (v.metrics.get("correctness"), pv) {
        (Some(c), Some(pv)) => {
            let mut r = scrt_evolve::ScoreReport::uncovered(pv, "live");
            r.correctness = *c;
            r.n = 1;
            r
        }
        _ => scrt_evolve::ScoreReport::uncovered("probe-none", "baseline"),
    }
}

/// Flatten a `ScoreReport` into the model-store's per-version metric map.
fn score_to_metrics(r: &scrt_evolve::ScoreReport) -> std::collections::BTreeMap<String, f64> {
    let mut m = std::collections::BTreeMap::new();
    m.insert("correctness".to_string(), r.correctness);
    if let Some(v) = r.constitution_adherence {
        m.insert("constitution_adherence".to_string(), v);
    }
    if let Some(v) = r.perplexity {
        m.insert("perplexity".to_string(), v);
    }
    m
}

/// `branch evolve <name>` — config-driven FURTHER training of a branch. Loads the
/// branch's persisted config, continues from its current stored adapter, runs an
/// eval-gated round vs the live version, and on KEEP commits a new version to the
/// bounded `[store]` ring + deploys the GGUF. The repeatable step a `.cmd` loops.
fn cmd_branch_evolve(
    config: &std::path::Path,
    name: &str,
    steps: usize,
    python: Option<String>,
) -> Result<()> {
    use scrt_evolve::branch::{self, BranchHooks};
    use scrt_evolve::model_store::ModelStore;
    use scrt_evolve::regulate::StepAction;

    // 1. Prefer the branch's persisted (self-describing) config.
    let base_cfg = EvolveConfig::load(config)?;
    let branch_dir = base_cfg.work_dir().join("branches").join(name);
    let persisted = branch_dir.join("branch.toml");
    let mut cfg = if persisted.exists() {
        println!(
            "branch evolve [{name}]: using persisted config {}",
            persisted.display()
        );
        EvolveConfig::load(&persisted)?
    } else {
        println!("branch evolve [{name}]: no persisted branch.toml — using {config:?}");
        base_cfg
    };

    // 2. Open the bounded per-branch model store.
    let store_dir = branch_dir.join("store");
    let base_model = cfg
        .branch
        .as_ref()
        .and_then(|b| b.base.clone())
        .or_else(|| {
            cfg.evolve
                .model_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .unwrap_or_default();
    let keep = ModelStore::keep_from_config(&cfg);
    let mut store = ModelStore::open(&store_dir, &base_model, keep)
        .map_err(|e| anyhow::anyhow!("open model store: {e}"))?;

    // 3. CONTINUE from the current version's adapter (config-driven resume).
    if let Some(cur) = store.resolve_current() {
        let mut train_cfg = cfg.train.clone().unwrap_or_default();
        let mut lora = train_cfg.lora.clone().unwrap_or_default();
        lora.init_adapter = Some(cur.adapter_dir.to_string_lossy().into_owned());
        train_cfg.lora = Some(lora);
        cfg.train = Some(train_cfg);
        println!("branch evolve [{name}]: continuing from version {}", cur.id);
    } else {
        println!("branch evolve [{name}]: no stored version yet — training fresh");
    }
    let cfg = cfg;

    // 4. Baseline = the live version's score (this round must IMPROVE on it),
    //    rebuilt from its stored correctness + probe_version so the candidate
    //    (scored on the same stable probe) is genuinely comparable. Under a
    //    stable probe, anchor a pre-feature version's baseline to the on-disk
    //    exam (the branch's `probe.jsonl`) so even the next round is a real gate.
    let stable_pv = if cfg.eval.as_ref().map(|e| e.stable_probe).unwrap_or(false) {
        let ppath = branch_dir.join("probe.jsonl");
        ppath
            .exists()
            .then(|| {
                scrt_evolve::eval::ProbeSet::load(&ppath)
                    .ok()
                    .map(|p| p.version)
            })
            .flatten()
    } else {
        None
    };
    let baseline = store
        .current()
        .map(|v| baseline_from_version(v, stable_pv.as_deref()))
        .unwrap_or_else(|| scrt_evolve::ScoreReport::uncovered("probe-none", "baseline"));

    // 5. Production hooks (mirror create; free the GPU before training).
    let py = python;
    let discover = |c: &EvolveConfig| scrt_evolve::discover::run(c);
    let generate =
        |c: &EvolveConfig, ctx: &scrt_evolve::DiscoveredContext| scrt_evolve::generate::run(c, ctx);
    let py_train = py.clone();
    let train = |c: &EvolveConfig, train_set: &scrt_evolve::Dataset| -> Result<Vec<String>> {
        maybe_free_gpu(c);
        let wd = WorkDir::from_config(c);
        let data_path = wd.root().join("dataset.branch-train.jsonl");
        train_set.write_jsonl(&data_path)?;
        cmd_train_transformers(c, Some(data_path), py_train.clone(), None, steps, 256)?;
        Ok(provenance_of(train_set))
    };
    let py_score = py.clone();
    let score = |c: &EvolveConfig| -> Result<scrt_evolve::ScoreReport> {
        scrt_evolve::eval::run_eval(c, py_score.as_deref())
    };
    let py_export = py.clone();
    let export = |c: &EvolveConfig, path: &std::path::Path| -> Result<PathBuf> {
        let adapter = WorkDir::from_config(c).root().join("adapter");
        cmd_export_gguf(
            c,
            Some(adapter),
            Some(path.to_path_buf()),
            None,
            None,
            false,
            py_export.clone(),
        )?;
        Ok(path.to_path_buf())
    };
    let hooks = BranchHooks {
        discover: &discover,
        generate: &generate,
        train: &train,
        score: &score,
        export: &export,
    };

    let created = now_iso8601();
    let domain = cfg.branch.as_ref().and_then(|b| b.domain.clone());
    let report = branch::create(
        &cfg,
        name,
        None,
        None,
        domain.as_deref(),
        &baseline,
        &created,
        &hooks,
    )?;

    let action = report
        .action
        .map(|a| format!("{a:?}"))
        .unwrap_or_else(|| "bailed".to_string());

    // 6. On KEEP: commit a new version to the bounded ring + deploy the GGUF.
    if matches!(report.action, Some(StepAction::Commit)) {
        let adapter_dir = branch_dir.join("adapter");
        let gguf = scrt_evolve::branch::create::gguf_path(&cfg, name);
        let metrics = report
            .metrics
            .as_ref()
            .map(score_to_metrics)
            .unwrap_or_default();
        // Carry the exam this candidate was scored on, so the NEXT round's
        // baseline (via `baseline_from_version`) is comparable to its candidate.
        let probe_version = report.metrics.as_ref().map(|m| m.probe_version.clone());
        let gguf_arg = if gguf.exists() {
            Some(gguf.as_path())
        } else {
            None
        };
        let vid = store
            .commit(&adapter_dir, gguf_arg, metrics, probe_version, &created)
            .map_err(|e| anyhow::anyhow!("store commit: {e}"))?;
        println!(
            "branch evolve [{name}]: KEEP — committed {vid} ({} version(s) retained)",
            store.versions().len()
        );
        if let Some(dst) = cfg.store.as_ref().and_then(|s| s.deploy_to.clone()) {
            if let Some(g) = store.resolve(&vid).and_then(|r| r.gguf) {
                if let Some(parent) = std::path::Path::new(&dst).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::copy(&g, &dst)?;
                println!("  deployed → {dst}");
            }
        }
    } else {
        println!(
            "branch evolve [{name}]: {action} — live model unchanged ({})",
            report.note
        );
    }

    emit_json(serde_json::json!({
        "command": "branch-evolve",
        "name": name,
        "action": action,
        "halt": report.halt,
        "versions": store.versions().iter().map(|v| v.id.clone()).collect::<Vec<_>>(),
        "current": store.manifest().current,
        "status": if report.halt { "halted" } else { "ok" },
    }));
    if report.halt {
        anyhow::bail!("branch evolve halted (catastrophe) — see quarantine + evolution-log");
    }
    Ok(())
}

/// Open a branch's per-branch model store.
fn open_branch_store(
    cfg: &EvolveConfig,
    name: &str,
) -> Result<scrt_evolve::model_store::ModelStore> {
    use scrt_evolve::model_store::ModelStore;
    let branch_dir = cfg.work_dir().join("branches").join(name);
    let base_model = cfg
        .branch
        .as_ref()
        .and_then(|b| b.base.clone())
        .or_else(|| {
            cfg.evolve
                .model_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .unwrap_or_default();
    ModelStore::open(
        branch_dir.join("store"),
        &base_model,
        ModelStore::keep_from_config(cfg),
    )
    .map_err(|e| anyhow::anyhow!("open model store: {e}"))
}

/// `branch versions <name>` — show the bounded weight-version ring.
fn cmd_branch_versions(cfg: &EvolveConfig, name: &str) -> Result<()> {
    let store = open_branch_store(cfg, name)?;
    let current = store.manifest().current.clone();
    if store.versions().is_empty() {
        println!("branch '{name}': no stored versions yet");
    } else {
        println!(
            "branch '{name}' versions (keep={}, base={}):",
            store.manifest().keep_versions,
            store.manifest().base_model
        );
        for v in store.versions() {
            let live = if Some(&v.id) == current.as_ref() {
                " *current*"
            } else {
                ""
            };
            let corr = v
                .metrics
                .get("correctness")
                .map(|c| format!("correctness={c:.4}"))
                .unwrap_or_default();
            let gguf = if v.gguf.is_some() { " +gguf" } else { "" };
            println!(
                "  {}{live}  parent={}  {corr}{gguf}  {}",
                v.id,
                v.parent.as_deref().unwrap_or("-"),
                v.created
            );
        }
    }
    emit_json(serde_json::json!({
        "command": "branch-versions",
        "name": name,
        "current": current,
        "versions": store.versions().iter().map(|v| v.id.clone()).collect::<Vec<_>>(),
        "status": "ok",
    }));
    Ok(())
}

/// `branch rollback <name>` — revert the live model to the current version's
/// parent and re-deploy that GGUF.
fn cmd_branch_rollback(cfg: &EvolveConfig, name: &str) -> Result<()> {
    let mut store = open_branch_store(cfg, name)?;
    let reverted = store
        .rollback()
        .map_err(|e| anyhow::anyhow!("rollback: {e}"))?;
    match reverted {
        Some(id) => {
            println!("branch '{name}': rolled back → version {id} (live)");
            // Re-deploy the reverted version's GGUF if a target is configured.
            if let Some(dst) = cfg.store.as_ref().and_then(|s| s.deploy_to.clone()) {
                if let Some(g) = store.resolve(&id).and_then(|r| r.gguf) {
                    if let Some(parent) = std::path::Path::new(&dst).parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    std::fs::copy(&g, &dst)?;
                    println!("  redeployed → {dst}");
                }
            }
            emit_json(serde_json::json!({
                "command": "branch-rollback", "name": name, "current": id, "status": "ok",
            }));
        }
        None => {
            println!("branch '{name}': nothing to roll back to (no parent version)");
            emit_json(serde_json::json!({
                "command": "branch-rollback", "name": name, "current": store.manifest().current,
                "status": "noop",
            }));
        }
    }
    Ok(())
}

/// `branch list` — print the registered fleet.
fn cmd_branch_list(cfg: &EvolveConfig) -> Result<()> {
    let reg_path = scrt_evolve::branch::create::registry_path(cfg);
    let reg = scrt_evolve::branch::BranchRegistry::load(&reg_path)?;
    if reg.branches.is_empty() {
        println!("no branches registered (registry: {})", reg_path.display());
        return Ok(());
    }
    println!(
        "{} branch(es) [{}]:",
        reg.branches.len(),
        reg_path.display()
    );
    for b in &reg.branches {
        let corr = b.eval_report.get("correctness").copied();
        let corr_str = corr
            .map(|c| format!("{c:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "  {:<20} base={:<24} domain={:<20} correctness={corr_str}",
            b.name, b.base_model, b.domain
        );
    }
    emit_json(serde_json::json!({
        "command": "branch-list",
        "registry": reg_path.display().to_string(),
        "branches": reg.branches.iter().map(|b| serde_json::json!({
            "name": b.name,
            "base_model": b.base_model,
            "domain": b.domain,
            // null (not NaN) when a branch carries no correctness metric (A7).
            "correctness": b.eval_report.get("correctness").copied(),
            "gguf_sha": b.gguf_sha,
        })).collect::<Vec<_>>(),
        "status": "ok",
    }));
    Ok(())
}

/// `branch register` — admit an externally-built branch GGUF into the fleet. The
/// native counterpart to `create`'s export step: compute the `router_signature`
/// from the branch dataset (so it matches what the router uses), assemble the
/// manifest (content-addressed `gguf_sha`), and admit it into the registry.
#[allow(clippy::too_many_arguments)]
fn cmd_branch_register(
    cfg: &EvolveConfig,
    name: &str,
    gguf: &std::path::Path,
    base: Option<String>,
    domain: Option<String>,
    dataset: Option<PathBuf>,
    correctness: Option<f64>,
    parent: Option<String>,
) -> Result<()> {
    use scrt_evolve::branch::router::corpus_signature;
    use scrt_evolve::branch::{admit, AdmitOutcome};
    use scrt_evolve::branch::{
        sha256_file, BranchManifest, BranchRegistry, Lineage, MANIFEST_VERSION,
    };

    if !gguf.exists() {
        anyhow::bail!("register: gguf not found: {}", gguf.display());
    }
    let bcfg = cfg.branch.clone().unwrap_or_default();

    // Signature from the branch dataset (the domain content the branch learned).
    let ds_path = dataset.unwrap_or_else(|| {
        cfg.work_dir()
            .join("branches")
            .join(name)
            .join("dataset.jsonl")
    });
    let ds = scrt_evolve::Dataset::read_jsonl(&ds_path)
        .with_context(|| format!("register: reading branch dataset {}", ds_path.display()))?;
    let texts: Vec<String> = ds.rows.iter().map(row_domain_text).collect();
    let kind = bcfg
        .router
        .as_ref()
        .map(|r| r.kind.clone())
        .unwrap_or_else(|| "simhash".to_string());
    let router_signature = corpus_signature(&kind, &texts);

    let base_model = base
        .or(bcfg.base)
        .or_else(|| {
            cfg.evolve
                .model_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let mut eval_report = std::collections::BTreeMap::new();
    if let Some(c) = correctness {
        eval_report.insert("correctness".to_string(), c);
    }

    let manifest = BranchManifest {
        name: name.to_string(),
        base_model,
        domain: domain.or(bcfg.domain).unwrap_or_default(),
        corpus_descriptor: format!("{} dataset rows", ds.rows.len()),
        router_signature,
        eval_report,
        lineage: Lineage { parent },
        version: MANIFEST_VERSION.to_string(),
        gguf_sha: sha256_file(gguf)?,
        created: now_iso8601(),
    };

    // Persist the manifest alongside the GGUF (branch dir).
    let branch_dir = cfg.work_dir().join("branches").join(name);
    std::fs::create_dir_all(&branch_dir)?;
    manifest.write(branch_dir.join("manifest.json"))?;

    // Admit into the shared registry (bounded fleet; near-dup merges, no twins).
    let reg_path = scrt_evolve::branch::create::registry_path(cfg);
    let mut registry = BranchRegistry::load(&reg_path)?;
    let max_branches = bcfg.max_branches.max(1);
    match admit(&mut registry, manifest, max_branches, 0.85) {
        AdmitOutcome::Added => {
            registry.write(&reg_path)?;
            println!("registered branch '{name}' → {}", reg_path.display());
        }
        AdmitOutcome::Merged { into } => {
            println!("branch '{name}' is a near-duplicate of '{into}' — merged (not registered as a twin)");
        }
        AdmitOutcome::Rejected { reason } => {
            anyhow::bail!("register: {reason}");
        }
    }
    Ok(())
}

/// The domain text of a dataset row (prompt + completion / instruction etc.) used
/// to compute a branch's `router_signature`.
fn row_domain_text(row: &scrt_evolve::GenExample) -> String {
    use scrt_evolve::GenExample::*;
    match row {
        Qa {
            prompt, completion, ..
        } => format!("{prompt} {completion}"),
        Instruction {
            instruction,
            input,
            output,
            ..
        } => format!("{instruction} {input} {output}"),
        Completion { text, .. } => text.clone(),
        Contrastive {
            query, positive, ..
        } => format!("{query} {positive}"),
        ToolCall { prompt, tool, .. } => format!("{prompt} {tool}"),
        Cli {
            prompt, command, ..
        } => format!("{prompt} {command}"),
    }
}

/// `branch route "<q>"` — resolve a query to branch(es) + scores; no serving.
fn cmd_branch_route(cfg: &EvolveConfig, query: &str) -> Result<()> {
    use scrt_evolve::{BranchRouter, LocalBranchRouter};
    let reg =
        scrt_evolve::branch::BranchRegistry::load(scrt_evolve::branch::create::registry_path(cfg))?;
    let router = LocalBranchRouter::from_config(cfg, &reg);
    let hits = router.resolve(query);
    if hits.is_empty() {
        println!("route: no branch matched {query:?} (base-only)");
        emit_json(serde_json::json!({
            "command": "branch-route",
            "query": query,
            "matches": [],
            "resolved": serde_json::Value::Null,
            "status": "base_only",
        }));
        return Ok(());
    }
    println!("route: {} match(es) for {query:?}:", hits.len());
    for (r, score) in &hits {
        println!("  {:<20} domain={:<20} score={score:.3}", r.name, r.domain);
    }
    emit_json(serde_json::json!({
        "command": "branch-route",
        "query": query,
        "matches": hits.iter().map(|(r, score)| serde_json::json!({
            "name": r.name, "domain": r.domain, "score": score,
        })).collect::<Vec<_>>(),
        "resolved": hits.first().map(|(r, _)| r.name.clone()),
        "status": "ok",
    }));
    Ok(())
}

/// `branch serve <name>` / `branch serve --route "<q>"` — serve a branch GGUF via
/// the runtime. `--route` resolves the request to the best branch(es); empty ⇒
/// base-only fallback (`serve --branches` semantics).
fn cmd_branch_serve(
    cfg: &EvolveConfig,
    name: Option<String>,
    route: Option<String>,
    prompt: Option<String>,
    python: Option<String>,
) -> Result<()> {
    use scrt_evolve::branch::create::gguf_path;
    use scrt_evolve::{BranchRouter, LocalBranchRouter};

    let reg =
        scrt_evolve::branch::BranchRegistry::load(scrt_evolve::branch::create::registry_path(cfg))?;

    let target = if let Some(q) = &route {
        let router = LocalBranchRouter::from_config(cfg, &reg);
        let hits = router.resolve(q);
        let ensemble = cfg
            .branch
            .as_ref()
            .and_then(|b| b.ensemble.as_ref())
            .map(|e| e.mode.clone())
            .unwrap_or_else(|| "single_best".to_string());
        match hits.into_iter().next() {
            Some((r, score)) => {
                if ensemble == "average_topk" {
                    println!(
                        "ensemble=average_topk: v1 serves the top-1 representative \
                         (cross-branch output blend is the hivemind Merge)"
                    );
                }
                println!("serve --route: resolved {:?} (score {score:.3})", r.name);
                r.name
            }
            None => {
                println!("serve --route: no branch matched — base-only fallback");
                return match prompt {
                    Some(p) => cmd_run_model(cfg, &p, python),
                    None => Ok(()),
                };
            }
        }
    } else if let Some(n) = name {
        if reg.get(&n).is_none() {
            anyhow::bail!("serve: branch '{n}' not in the registry");
        }
        n
    } else {
        anyhow::bail!("serve: pass a branch <name> or --route \"<query>\"");
    };

    let gguf = gguf_path(cfg, &target);
    if !gguf.exists() {
        anyhow::bail!("serve: gguf for '{target}' missing: {}", gguf.display());
    }

    // Point the runtime at the branch GGUF (overlay `[branch.serve]` on `[runtime]`).
    let mut scoped = cfg.clone();
    let mut rt = scoped.runtime.clone().unwrap_or_default();
    rt.model_path = Some(gguf.to_string_lossy().into_owned());
    if let Some(bs) = cfg.branch.as_ref().and_then(|b| b.serve.as_ref()) {
        if let Some(ngl) = bs.n_gpu_layers {
            rt.n_gpu_layers = ngl;
        }
    }
    scoped.runtime = Some(rt);

    println!("serve[{target}]: gguf={}", gguf.display());
    cmd_run_model(&scoped, &prompt.unwrap_or_default(), python)
}

/// ISO-8601 UTC timestamp (`YYYY-MM-DDThh:mm:ssZ`) for manifest `created`. Uses
/// Hinnant's civil-from-days so no chrono dependency is needed.
fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil_from_days (Howard Hinnant).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn cmd_export(cfg: &EvolveConfig, data: Option<PathBuf>, model: Option<PathBuf>) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    let data_path = data.unwrap_or_else(|| wd.dataset_jsonl());
    if !data_path.exists() {
        anyhow::bail!(
            "export: dataset not found at {} — run `scrt-evolve generate` first",
            data_path.display()
        );
    }
    let dataset = scrt_evolve::Dataset::read_jsonl(&data_path)
        .with_context(|| format!("reading dataset {}", data_path.display()))?;
    let model_gguf = model
        .or_else(|| cfg.evolve.model_path.clone())
        .ok_or_else(|| anyhow::anyhow!("export: pass --model or set [evolve].model_path"))?;

    let tool_format = scrt_evolve::ToolFormat::parse(
        &cfg.generate
            .as_ref()
            .map(|g| g.tool_format.clone())
            .unwrap_or_else(|| "gemma".into()),
    );
    let report = scrt_evolve::export_llamacpp(&dataset, wd.root(), &model_gguf, tool_format)?;
    println!(
        "export: {} examples → {} and {}",
        report.example_count,
        report.train_txt.display(),
        report.chat_jsonl.display()
    );
    println!(
        "\nllama.cpp fine-tune command:\n{}",
        report.suggested_command
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_python(p: Option<&str>) -> EvolveConfig {
        EvolveConfig {
            hardware: p.map(|p| scrt_evolve::HardwareConfig {
                python: Some(p.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Interpreter resolution precedence (track 28): flag > $SCRT_EVOLVE_PYTHON >
    /// [hardware].python > bare `python`. All assertions in one test so the env
    /// var isn't raced by parallel tests.
    #[test]
    fn python_resolution_precedence() {
        std::env::remove_var("SCRT_EVOLVE_PYTHON");
        let cfg = cfg_with_python(Some("/cfg/python"));

        // The --python flag wins over env + config.
        std::env::set_var("SCRT_EVOLVE_PYTHON", "/env/python");
        assert_eq!(
            resolve_python(Some(&cfg), Some("/flag/python".to_string())),
            "/flag/python"
        );
        // Env beats config when no flag.
        assert_eq!(resolve_python(Some(&cfg), None), "/env/python");
        // Config beats the default when no flag + no env.
        std::env::remove_var("SCRT_EVOLVE_PYTHON");
        assert_eq!(resolve_python(Some(&cfg), None), "/cfg/python");
        // Bare `python` when nothing is set.
        let bare = EvolveConfig::default();
        assert_eq!(resolve_python(Some(&bare), None), "python");
    }
}
