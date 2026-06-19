//! llama.cpp fine-tune export.
//!
//! scrt-evolve's candle-native LoRA loop (track 04) trains safetensors models.
//! A GGUF (llama.cpp quantized) model can't be trained by candle directly, so
//! this module is the bridge for the GGUF path: it converts a [`Dataset`] of
//! QA/instruction rows into the formats llama.cpp's fine-tune tooling consumes,
//! and emits a ready-to-run command.
//!
//! Two artifacts are written under the work-dir:
//! - `finetune-train.txt` — the training corpus as Gemma-chat-formatted turns,
//!   one example per block (llama.cpp `finetune` reads a raw text corpus).
//! - `finetune-chat.jsonl` — the same examples as OpenAI-style chat records
//!   (`{"messages":[…]}`), for tooling that prefers structured chat input.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::dataset::{Dataset, GenExample};

/// What `export_llamacpp` produced.
#[derive(Debug, Clone)]
pub struct ExportReport {
    pub train_txt: PathBuf,
    pub chat_jsonl: PathBuf,
    pub example_count: usize,
    /// A ready-to-run llama.cpp finetune command (informational).
    pub suggested_command: String,
}

#[derive(Serialize)]
struct ChatRecord {
    messages: Vec<ChatTurn>,
}

#[derive(Serialize)]
struct ChatTurn {
    role: &'static str,
    content: String,
}

/// How `tool_call` rows are rendered into the model turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// Gemma's native tool-calling: a ```tool_code block with a Python-style
    /// call, e.g. ```tool_code\nscrt_stash(name="auth", note="…")\n```.
    Gemma,
    /// OpenAI assistant `tool_calls` shape — STUBBED (not yet implemented).
    OpenAi,
    /// Anthropic tool-use blocks — STUBBED (not yet implemented).
    Anthropic,
}

impl ToolFormat {
    /// Parse from the config string. Unknown / non-gemma formats are accepted
    /// but their renderers are stubs that error at render time.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "openai" => ToolFormat::OpenAi,
            "anthropic" => ToolFormat::Anthropic,
            _ => ToolFormat::Gemma,
        }
    }
}

/// Render a tool call into the model-turn text for a given format.
fn render_tool_call(
    tool: &str,
    arguments: &serde_json::Value,
    fmt: ToolFormat,
) -> anyhow::Result<String> {
    match fmt {
        ToolFormat::Gemma => {
            // Gemma native: ```tool_code\n<tool>(<k>=<v>, …)\n```
            let args = arguments.as_object().cloned().unwrap_or_default();
            let mut parts = Vec::new();
            for (k, v) in &args {
                // Render values as Python literals (strings quoted, others JSON).
                let rendered = match v {
                    serde_json::Value::String(s) => format!("{s:?}"),
                    other => other.to_string(),
                };
                parts.push(format!("{k}={rendered}"));
            }
            Ok(format!("```tool_code\n{tool}({})\n```", parts.join(", ")))
        }
        ToolFormat::OpenAi => anyhow::bail!(
            "export: tool_format=\"openai\" is not implemented yet (only \"gemma\")"
        ),
        ToolFormat::Anthropic => anyhow::bail!(
            "export: tool_format=\"anthropic\" is not implemented yet (only \"gemma\")"
        ),
    }
}

/// Render one dataset row as an (instruction, response) pair, or `None` for
/// rows that aren't instruction-shaped (e.g. `contrastive`). Tool-call rows are
/// rendered in `fmt`; a render error (stubbed format) drops the row.
fn row_to_pair(row: &GenExample, fmt: ToolFormat) -> Option<(String, String)> {
    match row {
        GenExample::Qa { prompt, completion, .. } => {
            Some((prompt.clone(), completion.clone()))
        }
        GenExample::Instruction { instruction, input, output, .. } => {
            let user = if input.trim().is_empty() {
                instruction.clone()
            } else {
                format!("{instruction}\n\n{input}")
            };
            Some((user, output.clone()))
        }
        GenExample::ToolCall { prompt, tool, arguments, .. } => {
            render_tool_call(tool, arguments, fmt)
                .ok()
                .map(|model_turn| (prompt.clone(), model_turn))
        }
        GenExample::Cli { prompt, command, .. } => {
            Some((prompt.clone(), command.clone()))
        }
        // Raw completions train as-is; pretrain-style, not chat. Skip for the
        // instruction-tuning export.
        GenExample::Completion { .. } => None,
        GenExample::Contrastive { .. } => None,
    }
}

/// Gemma chat template for a single training turn. Gemma uses
/// `<start_of_turn>user … <end_of_turn>\n<start_of_turn>model … <end_of_turn>`.
fn gemma_block(user: &str, model: &str) -> String {
    format!(
        "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{model}<end_of_turn>\n"
    )
}

/// Convert a dataset into llama.cpp fine-tune artifacts under `work_dir`.
///
/// `model_gguf` is the path to the base GGUF being adapted (used to build the
/// suggested command). `out_dir` is where the artifacts are written.
pub fn export_llamacpp(
    dataset: &Dataset,
    out_dir: impl AsRef<Path>,
    model_gguf: &Path,
    tool_format: ToolFormat,
) -> anyhow::Result<ExportReport> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let pairs: Vec<(String, String)> = dataset
        .rows
        .iter()
        .filter_map(|r| row_to_pair(r, tool_format))
        .collect();

    // 1) Gemma-chat text corpus, blocks separated by a blank line.
    let mut corpus = String::new();
    for (user, model) in &pairs {
        corpus.push_str(&gemma_block(user, model));
        corpus.push('\n');
    }
    let train_txt = out_dir.join("finetune-train.txt");
    std::fs::write(&train_txt, &corpus)?;

    // 2) OpenAI-style chat JSONL.
    let chat_jsonl = out_dir.join("finetune-chat.jsonl");
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&chat_jsonl)?);
        for (user, model) in &pairs {
            let rec = ChatRecord {
                messages: vec![
                    ChatTurn { role: "user", content: user.clone() },
                    ChatTurn { role: "assistant", content: model.clone() },
                ],
            };
            serde_json::to_writer(&mut f, &rec)?;
            f.write_all(b"\n")?;
        }
        f.flush()?;
    }

    let suggested_command = format!(
        "llama-finetune \\\n  --model-base {model} \\\n  --train-data {train} \\\n  \
--lora-out {out}/gemma-scrt-lora.gguf \\\n  --lora-r 16 --lora-alpha 32 \\\n  \
--adam-iter 64 --batch 4 --ctx 4096",
        model = model_gguf.display(),
        train = train_txt.display(),
        out = out_dir.display(),
    );

    Ok(ExportReport {
        train_txt,
        chat_jsonl,
        example_count: pairs.len(),
        suggested_command,
    })
}
