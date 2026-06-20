//! llama.cpp export tests.

use scrt_evolve::dataset::{Dataset, GenExample};

fn temp_dir(suffix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "scrt-evolve-export-{}-{suffix}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&p);
    p
}

#[test]
fn export_writes_gemma_chat_corpus_and_jsonl() {
    let dir = temp_dir("basic");
    let ds = Dataset::new(vec![
        GenExample::Qa {
            prompt: "How do I stash a search?".into(),
            completion: "Use scrt --mp-stash NAME.".into(),
            source: Some("README.md".into()),
            gen: Some("api".into()),
        },
        GenExample::Instruction {
            instruction: "Compose two stashes".into(),
            input: "stash-a stash-b".into(),
            output: "scrt PATTERN --mp-compose stash-a stash-b".into(),
            source: None,
            gen: None,
        },
        // Non-instruction rows are skipped by the export.
        GenExample::Contrastive {
            query: "x".into(),
            positive: "y".into(),
            negatives: vec![],
            stash: None,
        },
    ]);

    let model = std::path::Path::new("/models/gemma.gguf");
    let report =
        scrt_evolve::export_llamacpp(&ds, &dir, model, scrt_evolve::ToolFormat::Gemma).unwrap();

    assert_eq!(report.example_count, 2, "contrastive row excluded");

    let corpus = std::fs::read_to_string(&report.train_txt).unwrap();
    assert!(corpus.contains("<start_of_turn>user"));
    assert!(corpus.contains("<start_of_turn>model"));
    assert!(corpus.contains("scrt --mp-stash NAME"));
    // Instruction with input is joined into the user turn.
    assert!(corpus.contains("Compose two stashes"));
    assert!(corpus.contains("stash-a stash-b"));

    let chat = std::fs::read_to_string(&report.chat_jsonl).unwrap();
    assert_eq!(chat.lines().count(), 2);
    assert!(chat.contains("\"role\":\"user\""));
    assert!(chat.contains("\"role\":\"assistant\""));

    // Command references the base GGUF + the train corpus.
    assert!(report.suggested_command.contains("gemma.gguf"));
    assert!(report.suggested_command.contains("finetune-train.txt"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tool_call_rows_render_in_gemma_tool_code_format() {
    let dir = temp_dir("toolcall");
    let ds = Dataset::new(vec![GenExample::ToolCall {
        prompt: "Save the current search as a stash called auth with note Auth findings".into(),
        tool: "scrt_stash".into(),
        arguments: serde_json::json!({"name": "auth", "note": "Auth findings"}),
        source: Some("README.md".into()),
        gen: Some("api".into()),
    }]);

    let model = std::path::Path::new("/models/gemma.gguf");
    let report =
        scrt_evolve::export_llamacpp(&ds, &dir, model, scrt_evolve::ToolFormat::Gemma).unwrap();
    assert_eq!(report.example_count, 1);

    let corpus = std::fs::read_to_string(&report.train_txt).unwrap();
    // Gemma native tool-call block with a Python-style call.
    assert!(
        corpus.contains("```tool_code"),
        "expected tool_code block:\n{corpus}"
    );
    assert!(corpus.contains("scrt_stash("));
    assert!(corpus.contains("name=\"auth\""));
    assert!(corpus.contains("note=\"Auth findings\""));
    // Still wrapped in the Gemma chat turns.
    assert!(corpus.contains("<start_of_turn>model"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stubbed_tool_formats_drop_tool_call_rows() {
    // openai/anthropic renderers are stubs — tool_call rows are dropped, but
    // qa rows still export fine.
    let dir = temp_dir("stubfmt");
    let ds = Dataset::new(vec![
        GenExample::Qa {
            prompt: "q".into(),
            completion: "a".into(),
            source: None,
            gen: None,
        },
        GenExample::ToolCall {
            prompt: "p".into(),
            tool: "scrt_stash".into(),
            arguments: serde_json::json!({"name": "x", "note": "y"}),
            source: None,
            gen: None,
        },
    ]);
    let model = std::path::Path::new("/m.gguf");
    let report =
        scrt_evolve::export_llamacpp(&ds, &dir, model, scrt_evolve::ToolFormat::OpenAi).unwrap();
    // The qa row survives; the tool_call row is dropped (stub renderer).
    assert_eq!(report.example_count, 1);
    let _ = std::fs::remove_dir_all(&dir);
}
