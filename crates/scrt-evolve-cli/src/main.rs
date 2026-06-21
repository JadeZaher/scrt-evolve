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

#[derive(Parser)]
#[command(
    name = "scrt-evolve",
    version,
    about = "Make a model better at its own corpus — discover → generate → train."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
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
        /// Q5_K_M | Q6_K | Q8_0 | f16 | none. Default Q4_K_M.
        #[arg(long, default_value = "Q4_K_M")]
        quant: String,
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
                &quant,
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
        eprintln!(
            "warning: the scaffolded `model_path` does not exist yet — \
             edit [evolve].model_path in {} to point at your model directory \
             before running.",
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
    Ok(())
}

fn load_discovered(wd: &WorkDir, input: Option<PathBuf>) -> Result<scrt_evolve::DiscoveredContext> {
    let in_path = input.unwrap_or_else(|| wd.discovered_json());
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
            _ => None,
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

    let report = scrt_evolve::eval::run_eval(&cfg, python.as_deref())?;
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
    let wd = WorkDir::from_config(cfg);
    let data_path = data.unwrap_or_else(|| wd.dataset_jsonl());
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

    // Locate the python/ package dir: it ships next to the repo root. Search
    // upward from the current dir for a `python/scrt_evolve_train` so the CLI
    // works from a checkout regardless of cwd.
    let pkg_parent = find_python_pkg_dir()
        .ok_or_else(|| anyhow::anyhow!("train(transformers): could not locate python/ dir"))?;

    let py = python.unwrap_or_else(|| "python".to_string());
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
        .arg(&targets)
        .env("PYTHONPATH", &pkg_parent);

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

    // Fractional / sharded layer-block training pass-through: if
    // [train.fractional] is set + enabled, switch the trainer to block-local
    // distillation (bounds peak VRAM to one block). Config-driven so the same
    // pipeline runs on a small GPU without code changes.
    if let Some(frac) = cfg.train.as_ref().and_then(|t| t.fractional.clone()) {
        if frac.enabled {
            cmd.arg("--shard-mode");
            if let Some(bs) = frac.block_size {
                cmd.arg("--block-size").arg(bs.to_string());
            } else if let Some(n) = frac.shards {
                cmd.arg("--shards").arg(n.to_string());
            }
            cmd.arg("--calib-batches")
                .arg(frac.calib_batches.to_string());
            cmd.arg("--granularity").arg(&frac.granularity);
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
        anyhow::bail!("train(transformers): trainer exited with {status}");
    }
    println!("train(transformers): adapter → {}", out_dir.display());
    Ok(())
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

    let pkg_parent = find_python_pkg_dir()
        .ok_or_else(|| anyhow::anyhow!("infer: could not locate python/ dir"))?;

    let py = python.unwrap_or_else(|| "python".to_string());
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
        .arg(temperature.to_string())
        .env("PYTHONPATH", &pkg_parent);

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
        anyhow::bail!("infer: python process exited with {status}");
    }
    Ok(())
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
    let pkg_parent = find_python_pkg_dir()
        .ok_or_else(|| anyhow::anyhow!("dequant: could not locate python/ dir"))?;

    // Put the vendored gguf-py on PYTHONPATH (alongside the package parent), the
    // same way export-gguf relies on it.
    let gguf_py = find_llama_gguf_py();
    let pythonpath = match &gguf_py {
        Some(p) => format!("{};{}", pkg_parent.display(), p.display()),
        None => pkg_parent.display().to_string(),
    };

    let py = python.unwrap_or_else(|| "python".to_string());
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("-m")
        .arg("scrt_evolve_dequant")
        .arg("--gguf")
        .arg(gguf)
        .arg("--out")
        .arg(out)
        .arg("--dtype")
        .arg(dtype)
        .env("PYTHONPATH", &pythonpath);
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
        anyhow::bail!("dequant: python process exited with {status}");
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
    quant: &str,
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

    let pkg_parent = find_python_pkg_dir()
        .ok_or_else(|| anyhow::anyhow!("export-gguf: could not locate python/ dir"))?;

    let py = python.unwrap_or_else(|| "python".to_string());
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("-m")
        .arg("scrt_evolve_gguf")
        .arg("--model")
        .arg(&model_path)
        .arg("--adapter")
        .arg(&adapter_dir)
        .arg("--quant")
        .arg(quant)
        .env("PYTHONPATH", &pkg_parent);

    if let Some(o) = &out {
        cmd.arg("--out").arg(o);
    }
    if let Some(lc) = &llama_cpp {
        cmd.arg("--llama-cpp").arg(lc);
    }
    if keep_intermediates {
        cmd.arg("--keep-merged").arg("--keep-f16");
    }

    println!(
        "export-gguf: {} -m scrt_evolve_gguf  (model={}, adapter={}, quant={})",
        py,
        model_path.display(),
        adapter_dir.display(),
        quant,
    );

    let status = cmd
        .status()
        .with_context(|| format!("launching `{py} -m scrt_evolve_gguf`"))?;
    if !status.success() {
        anyhow::bail!("export-gguf: python process exited with {status}");
    }
    Ok(())
}

/// Find the `python/` dir for `PYTHONPATH`. Thin re-export of the shared SDK
/// helper so the CLI and the eval subprocess scorer agree on resolution.
fn find_python_pkg_dir() -> Option<PathBuf> {
    scrt_evolve::python_pkg_dir()
}

fn cmd_export(cfg: &EvolveConfig, data: Option<PathBuf>, model: Option<PathBuf>) -> Result<()> {
    let wd = WorkDir::from_config(cfg);
    let data_path = data.unwrap_or_else(|| wd.dataset_jsonl());
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
