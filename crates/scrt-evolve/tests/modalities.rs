//! Track 09 — new generation modalities (`skill`, `reasoning_edit`).
//!
//! Covers the contract that matters: the new rows round-trip through the JSONL
//! boundary, they are opt-in (absent from `kinds` ⇒ never planned), and the
//! reasoning-edit row renders so the corrected chain precedes the final action
//! (the property that trains internal reasoning at inference).

use scrt_evolve::dataset::{Dataset, GenExample};
use scrt_evolve::generate::{plan_modes, GenMode};

#[test]
fn skill_and_reasoning_rows_round_trip() {
    let rows = vec![
        GenExample::Skill {
            skill_name: "scrt".to_string(),
            prompt: "stash the auth findings".to_string(),
            invocation: "scrt \"auth\" --mp-stash auth --mp-ttl 4h".to_string(),
            expected_outcome: Some("a named stash `auth` exists".to_string()),
            source: Some("AGENTS.md".to_string()),
            gen: Some("trace:scrt-cli".to_string()),
        },
        GenExample::ReasoningEdit {
            prompt: "find files matching A that also reference B".to_string(),
            original_steps: vec![
                "grep for A".to_string(),
                "grep for B separately".to_string(),
            ],
            edit_op: "correct".to_string(),
            edited_steps: vec![
                "scrt search A → stash a".to_string(),
                "scrt --mp-intersect a b".to_string(),
            ],
            final_action: "scrt \"X\" --mp-intersect a b".to_string(),
            source: None,
            gen: None,
        },
    ];
    let ds = Dataset::new(rows.clone());
    let jsonl = ds.to_jsonl().expect("serialize");
    // The `kind` tags must be the snake_case contract names.
    assert!(jsonl.contains("\"kind\":\"skill\""));
    assert!(jsonl.contains("\"kind\":\"reasoning_edit\""));
    let parsed = Dataset::from_jsonl(&jsonl).expect("round-trip");
    assert_eq!(parsed.rows, rows);
}

#[test]
fn new_modalities_are_opt_in() {
    // Absent from kinds ⇒ never planned (pipeline byte-identical to today).
    let modes = plan_modes(&["qa".to_string(), "cli".to_string()]);
    assert!(!modes.contains(&GenMode::Skill));
    assert!(!modes.contains(&GenMode::ReasoningEdit));

    // Present ⇒ each gets its own pass.
    let modes = plan_modes(&[
        "qa".to_string(),
        "skill".to_string(),
        "reasoning_edit".to_string(),
    ]);
    assert!(modes.contains(&GenMode::Prose));
    assert!(modes.contains(&GenMode::Skill));
    assert!(modes.contains(&GenMode::ReasoningEdit));
}

#[test]
fn reasoning_edit_renders_corrected_chain_before_action() {
    // The training artifact must put the corrected steps BEFORE `=> action`,
    // so the model learns to emit the reasoning then the action.
    let ds = Dataset::new(vec![GenExample::ReasoningEdit {
        prompt: "intersect two stashes".to_string(),
        original_steps: vec!["read both files".to_string()],
        edit_op: "correct".to_string(),
        edited_steps: vec![
            "stash a from search A".to_string(),
            "stash b from search B".to_string(),
        ],
        final_action: "scrt --mp-intersect a b".to_string(),
        source: None,
        gen: None,
    }]);

    let dir = {
        let mut p = std::env::temp_dir();
        p.push(format!("scrt-evolve-modalities-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    let model = std::path::Path::new("base.gguf");
    scrt_evolve::export_llamacpp(&ds, &dir, model, scrt_evolve::ToolFormat::Gemma).unwrap();

    let txt = std::fs::read_to_string(dir.join("finetune-train.txt")).unwrap();
    // Corrected steps appear, numbered, and the action marker follows them.
    let action_at = txt
        .find("=> scrt --mp-intersect a b")
        .expect("action present");
    let step_at = txt.find("1. stash a from search A").expect("chain present");
    assert!(
        step_at < action_at,
        "corrected reasoning must precede the final action; got:\n{txt}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
