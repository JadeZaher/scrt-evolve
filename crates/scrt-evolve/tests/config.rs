//! Config load + validation tests (the heart of track 00 acceptance).

use scrt_evolve::config::ConfigError;
use scrt_evolve::EvolveConfig;

/// The DESIGN.md §Config schema example, verbatim (sans inline comments,
/// which TOML tolerates anyway — kept here to assert a real round-trip).
const DESIGN_EXAMPLE: &str = r#"
[evolve]
model_path  = "/models/my-model"
corpus_dir  = "./src"
palace_path = ".mpg/mind-palace.json"
work_dir    = ".scrt-evolve"

[discover]
seed = "palace"
max_passages = 500
dedup = "simhash"
cluster = true

[generate]
backend = "api"
kinds = ["qa", "instruction"]
per_passage = 3
  [generate.local]
  max_new_tokens = 512
  temperature = 0.7
  [generate.api]
  base_url = "https://api.example.com/v1"
  model = "gpt-x"
  api_key_env = "SCRT_EVOLVE_API_KEY"
  turns = 1

[train]
preset = "lora"
  [train.lora]
  rank = 16
  alpha = 32
  target_modules = ["q_proj","v_proj"]
  lr = 2e-4
  epochs = 1
  [train.full]
  lr = 1e-5
  epochs = 1
  grad_accum = 8
  [train.pretrain]
  lr = 1e-5
  block_size = 1024
  [train.contrastive]
  negatives_per_row = 4
  temperature = 0.05
  [train.shard]
  role = "coordinator"
  peers = ["host:port"]
  shard_strategy = "data"
  base_preset = "lora"
"#;

#[test]
fn design_example_round_trips() {
    let cfg = EvolveConfig::from_toml_str(DESIGN_EXAMPLE).expect("DESIGN example must load");

    // Top section.
    assert_eq!(
        cfg.evolve.model_path.as_deref(),
        Some(std::path::Path::new("/models/my-model"))
    );
    assert_eq!(cfg.work_dir(), std::path::PathBuf::from(".scrt-evolve"));

    // Discover.
    let d = cfg.discover.as_ref().expect("[discover]");
    assert_eq!(d.seed, "palace");
    assert_eq!(d.max_passages, 500);
    assert_eq!(d.dedup, "simhash");
    assert!(d.cluster);

    // Generate + sub-blocks.
    let g = cfg.generate.as_ref().expect("[generate]");
    assert_eq!(g.backend, "api");
    assert_eq!(g.kinds, vec!["qa".to_string(), "instruction".to_string()]);
    assert_eq!(g.per_passage, 3);
    assert_eq!(g.local.as_ref().unwrap().max_new_tokens, 512);
    assert_eq!(g.api.as_ref().unwrap().turns, 1);
    assert_eq!(
        g.api.as_ref().unwrap().api_key_env.as_deref(),
        Some("SCRT_EVOLVE_API_KEY")
    );

    // Train + every preset sub-block.
    let t = cfg.train.as_ref().expect("[train]");
    assert_eq!(t.preset, "lora");
    assert_eq!(t.lora.as_ref().unwrap().rank, 16);
    assert_eq!(t.lora.as_ref().unwrap().alpha, 32);
    assert_eq!(t.full.as_ref().unwrap().grad_accum, 8);
    assert_eq!(t.pretrain.as_ref().unwrap().block_size, 1024);
    assert_eq!(t.contrastive.as_ref().unwrap().negatives_per_row, 4);
    assert_eq!(t.shard.as_ref().unwrap().role, "coordinator");
    assert_eq!(t.shard.as_ref().unwrap().base_preset, "lora");

    // Serialize back out and re-parse — proves the schema round-trips.
    let serialized = toml::to_string(&cfg).expect("serialize");
    let reparsed = EvolveConfig::from_toml_str(&serialized).expect("re-parse");
    assert_eq!(
        reparsed.evolve.model_path, cfg.evolve.model_path,
        "round-trip preserves model_path"
    );
}

#[test]
fn partial_generate_only_loads() {
    // A generate-only config: no [train], no [discover], and model_path absent.
    let toml = r#"
[generate]
backend = "api"
kinds = ["qa"]
per_passage = 2
  [generate.api]
  base_url = "https://api.example.com/v1"
  model = "x"
  api_key_env = "MY_KEY"
"#;
    let cfg = EvolveConfig::from_toml_str(toml).expect("generate-only must load");
    assert!(cfg.train.is_none());
    assert!(cfg.discover.is_none());
    assert!(cfg.evolve.model_path.is_none());
    assert_eq!(cfg.generate.unwrap().backend, "api");
}

#[test]
fn partial_train_only_loads() {
    let toml = r#"
[train]
preset = "contrastive"
  [train.contrastive]
  negatives_per_row = 8
"#;
    let cfg = EvolveConfig::from_toml_str(toml).expect("train-only must load");
    assert!(cfg.generate.is_none());
    let t = cfg.train.unwrap();
    assert_eq!(t.preset, "contrastive");
    assert_eq!(t.contrastive.unwrap().negatives_per_row, 8);
}

#[test]
fn empty_config_loads_with_defaults() {
    // Even an empty file is valid — every block is optional.
    let cfg = EvolveConfig::from_toml_str("").expect("empty config must load");
    assert!(cfg.evolve.model_path.is_none());
    assert_eq!(cfg.work_dir(), std::path::PathBuf::from(".scrt-evolve"));
}

#[test]
fn require_model_path_errors_when_absent() {
    let cfg = EvolveConfig::from_toml_str("[train]\npreset = \"lora\"").unwrap();
    let err = cfg.require_model_path("train").unwrap_err();
    assert!(matches!(
        err,
        ConfigError::MissingModelPath { stage: "train" }
    ));
}

#[test]
fn require_model_path_ok_when_present() {
    let cfg = EvolveConfig::from_toml_str("[evolve]\nmodel_path = \"/models/m\"").unwrap();
    assert_eq!(
        cfg.require_model_path("train").unwrap(),
        std::path::Path::new("/models/m")
    );
}

#[test]
fn inline_secret_with_sk_prefix_is_rejected() {
    let toml = r#"
[generate]
  [generate.api]
  api_key_env = "sk-ant-abc123definitelyarealkeyshapedstring"
"#;
    let err = EvolveConfig::from_toml_str(toml).unwrap_err();
    assert!(
        matches!(err, ConfigError::InlineSecret { field } if field == "generate.api.api_key_env"),
        "got {err:?}"
    );
}

#[test]
fn inline_secret_with_spaces_is_rejected() {
    let toml = r#"
[generate]
  [generate.api]
  api_key_env = "my literal key value"
"#;
    let err = EvolveConfig::from_toml_str(toml).unwrap_err();
    assert!(
        matches!(err, ConfigError::InlineSecret { .. }),
        "got {err:?}"
    );
}

#[test]
fn inline_secret_long_token_is_rejected() {
    // 40+ char single token reads as a key, not an env var name.
    let long = "A".repeat(48);
    let toml = format!("[generate]\n  [generate.api]\n  api_key_env = \"{long}\"\n");
    let err = EvolveConfig::from_toml_str(&toml).unwrap_err();
    assert!(
        matches!(err, ConfigError::InlineSecret { .. }),
        "got {err:?}"
    );
}

#[test]
fn valid_env_var_name_is_accepted() {
    for name in ["SCRT_EVOLVE_API_KEY", "MY_KEY", "_X", "OPENAI_API_KEY"] {
        let toml = format!("[generate]\n  [generate.api]\n  api_key_env = \"{name}\"\n");
        EvolveConfig::from_toml_str(&toml)
            .unwrap_or_else(|e| panic!("`{name}` should be a valid env var name, got {e:?}"));
    }
}
