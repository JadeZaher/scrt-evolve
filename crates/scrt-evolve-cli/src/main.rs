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
    /// Point at a PROJECT directory: auto-detect its mpg palace + corpus and run
    /// the whole self-routing pipeline (discover → plan → generate → export).
    Evolve {
        /// The project directory to evolve a model against.
        project: PathBuf,
        /// Optional base evolve.toml supplying [generate]/[train] settings
        /// (corpus_dir/palace_path are auto-detected and override the base).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Gap-critic follow-up rounds.
        #[arg(long, default_value_t = 1)]
        gap_rounds: usize,
        /// Also export to llama.cpp format after generating.
        #[arg(long)]
        export: bool,
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
        } => cmd_evolve(&project, config, gap_rounds, export),
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

    println!(
        "train(transformers): {} -m scrt_evolve_train  (model={}, {} steps)",
        py,
        model_path.display(),
        steps
    );
    let status = cmd
        .status()
        .with_context(|| format!("launching `{py} -m scrt_evolve_train`"))?;
    if !status.success() {
        anyhow::bail!("train(transformers): trainer exited with {status}");
    }
    println!("train(transformers): adapter → {}", out_dir.display());
    Ok(())
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

/// Find the directory that should be on `PYTHONPATH` so `scrt_evolve_train`
/// imports: the `python/` dir holding the package. Walks up from cwd.
fn find_python_pkg_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("python");
        if candidate.join("scrt_evolve_train").is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
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
