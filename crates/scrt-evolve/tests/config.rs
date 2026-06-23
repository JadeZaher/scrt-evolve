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

// ---------------------------------------------------------------------------
// Track 20 slice 1 — `[[goals]]` config (additive, non-breaking).
// ---------------------------------------------------------------------------

#[test]
fn absent_goals_is_empty_and_preserves_single_run() {
    // No [[goals]] table at all ⇒ goals is empty ⇒ today's single-run behavior.
    let cfg = EvolveConfig::from_toml_str(DESIGN_EXAMPLE).expect("loads");
    assert!(cfg.goals.is_empty(), "absent [[goals]] must yield no goals");

    // And an entirely empty config too.
    let empty = EvolveConfig::from_toml_str("").expect("empty loads");
    assert!(empty.goals.is_empty());
}

#[test]
fn goals_parse_and_round_trip() {
    let toml = r#"
[evolve]
corpus_dir = "./src"
palace_path = ".mpg/mind-palace.json"

[[goals]]
name = "scrt-cli-fluency"
topic = "mp-* commands"
tag = "scrt-cli"

[[goals]]
name = "auth-mastery"
topic = "authentication flow"
tag = "security"
project = "./services/auth"
probe_set = "./probes/auth.jsonl"
weight = 2.0
cadence = "daily"
"#;
    let cfg = EvolveConfig::from_toml_str(toml).expect("goals must parse");
    assert_eq!(cfg.goals.len(), 2);

    // First goal: only the three required identifiers; optionals default to None.
    let g0 = &cfg.goals[0];
    assert_eq!(g0.name, "scrt-cli-fluency");
    assert_eq!(g0.topic, "mp-* commands");
    assert_eq!(g0.tag, "scrt-cli");
    assert!(g0.project.is_none());
    assert!(g0.probe_set.is_none());
    assert!(g0.weight.is_none());
    assert!(g0.cadence.is_none());

    // Second goal: all fields populated.
    let g1 = &cfg.goals[1];
    assert_eq!(g1.name, "auth-mastery");
    assert_eq!(g1.tag, "security");
    assert_eq!(
        g1.project.as_deref(),
        Some(std::path::Path::new("./services/auth"))
    );
    assert_eq!(
        g1.probe_set.as_deref(),
        Some(std::path::Path::new("./probes/auth.jsonl"))
    );
    assert_eq!(g1.weight, Some(2.0));
    assert_eq!(g1.cadence.as_deref(), Some("daily"));

    // Serialize back out and re-parse — the [[goals]] array round-trips.
    let serialized = toml::to_string(&cfg).expect("serialize");
    let reparsed = EvolveConfig::from_toml_str(&serialized).expect("re-parse");
    assert_eq!(reparsed.goals, cfg.goals, "goals round-trip exactly");
}

#[test]
fn for_goal_wires_discover_search_and_tag() {
    let toml = r#"
[evolve]
corpus_dir = "./src"
palace_path = ".mpg/mind-palace.json"

[discover]
seed = "both"
max_passages = 250

[[goals]]
name = "scrt-cli-fluency"
topic = "mp-* commands"
tag = "scrt-cli"
project = "./other-project"
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let goal = &cfg.goals[0];
    let per_goal = cfg.for_goal(goal);

    let d = per_goal.discover.as_ref().expect("[discover] derived");
    // base seed was "both" → preserved as "both" (corpus dimension kept).
    assert_eq!(d.seed, "both", "for_goal preserves corpus seeding as both");
    assert_eq!(d.palace_search.as_deref(), Some("mp-* commands"));
    assert_eq!(d.palace_tags, vec!["scrt-cli".to_string()]);
    // max_passages (and other inherited discover settings) are preserved.
    assert_eq!(d.max_passages, 250);
    // project scopes the corpus.
    assert_eq!(
        per_goal.evolve.corpus_dir.as_deref(),
        Some(std::path::Path::new("./other-project"))
    );
    // The original config is untouched (purity).
    assert_eq!(cfg.discover.as_ref().unwrap().seed, "both");
}

#[test]
fn for_goal_without_project_keeps_top_level_corpus() {
    let toml = r#"
[evolve]
corpus_dir = "./src"
palace_path = ".mpg/mind-palace.json"

[[goals]]
name = "g"
topic = "t"
tag = "tg"
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let per_goal = cfg.for_goal(&cfg.goals[0]);
    assert_eq!(
        per_goal.evolve.corpus_dir.as_deref(),
        Some(std::path::Path::new("./src")),
        "no goal.project ⇒ inherit the top-level corpus"
    );
}

// ---------------------------------------------------------------------------
// Track 23 — `[train.qat]` quantization-aware training config (additive).
// ---------------------------------------------------------------------------

#[test]
fn qat_config_round_trips_and_absent_is_none() {
    let toml = r#"
[train]
preset = "lora"
  [train.qat]
  enabled = true
  quant = "Q4_K_M"
  group_size = 32
  calibrate_batches = 8
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let q = cfg
        .train
        .as_ref()
        .unwrap()
        .qat
        .as_ref()
        .expect("[train.qat]");
    assert!(q.enabled);
    assert_eq!(q.quant, "Q4_K_M");
    assert_eq!(q.group_size, 32);
    assert_eq!(q.calibrate_batches, 8);

    // Round-trips.
    let ser = toml::to_string(&cfg).unwrap();
    let back = EvolveConfig::from_toml_str(&ser).unwrap();
    assert_eq!(back.train.unwrap().qat.unwrap().quant, "Q4_K_M");

    // Absent ⇒ None (plain LoRA; non-breaking).
    let plain = EvolveConfig::from_toml_str("[train]\npreset = \"lora\"").unwrap();
    assert!(plain.train.unwrap().qat.is_none());
}

// ---------------------------------------------------------------------------
// Track 24 — `[hardware]` config + state-space trainability pre-flight.
// ---------------------------------------------------------------------------

#[test]
fn hardware_config_round_trips_and_absent_is_none() {
    let toml = r#"
[hardware]
device = "cuda"
vram_gb = 8.0
ram_gb = 32.0
kernels = ["mamba-ssm", "causal-conv1d"]
machine = "test box"
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let h = cfg.hardware.as_ref().expect("[hardware]");
    assert_eq!(h.device, "cuda");
    assert_eq!(h.vram_gb, 8.0);
    assert!(h.has_kernel("mamba-ssm") && h.has_kernel("CAUSAL-CONV1D"));
    assert_eq!(h.machine.as_deref(), Some("test box"));

    let ser = toml::to_string(&cfg).unwrap();
    let back = EvolveConfig::from_toml_str(&ser).unwrap();
    assert_eq!(back.hardware.unwrap().vram_gb, 8.0);

    assert!(EvolveConfig::from_toml_str("[evolve]\nmodel_path=\"/m\"")
        .unwrap()
        .hardware
        .is_none());
}

#[test]
fn can_train_state_space_gates_on_device_and_kernels() {
    use scrt_evolve::HardwareConfig;
    // CPU / no kernels → cannot train a Mamba model (the segfault case).
    let cpu = HardwareConfig::default();
    assert!(cpu.can_train_state_space().is_err());

    // CUDA but missing kernels → still can't.
    let cuda_no_kernels = HardwareConfig {
        device: "cuda".to_string(),
        ..Default::default()
    };
    assert!(cuda_no_kernels.can_train_state_space().is_err());

    // CUDA + both kernels → OK.
    let ready = HardwareConfig {
        device: "cuda".to_string(),
        kernels: vec!["mamba-ssm".to_string(), "causal-conv1d".to_string()],
        ..Default::default()
    };
    assert!(ready.can_train_state_space().is_ok());
}

// ---------------------------------------------------------------------------
// Track 24 — `[train.fractional]` sharded layer-block training (additive).
// ---------------------------------------------------------------------------

#[test]
fn fractional_config_round_trips_and_absent_is_none() {
    let toml = r#"
[train]
preset = "lora"
  [train.fractional]
  enabled = true
  block_size = 8
  calib_batches = 8
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let f = cfg
        .train
        .as_ref()
        .unwrap()
        .fractional
        .as_ref()
        .expect("[train.fractional]");
    assert!(f.enabled);
    assert_eq!(f.block_size, Some(8));
    assert_eq!(f.shards, None);
    assert_eq!(f.calib_batches, 8);
    // granularity + objective default when omitted (non-breaking).
    assert_eq!(f.granularity, "block");
    assert_eq!(f.objective, "distill");

    // explicit end_task objective round-trips.
    let et = EvolveConfig::from_toml_str(
        "[train]\npreset=\"lora\"\n  [train.fractional]\n  objective=\"end_task\"\n",
    )
    .unwrap();
    assert_eq!(et.train.unwrap().fractional.unwrap().objective, "end_task");

    // Round-trips.
    let ser = toml::to_string(&cfg).unwrap();
    let back = EvolveConfig::from_toml_str(&ser).unwrap();
    assert_eq!(back.train.unwrap().fractional.unwrap().block_size, Some(8));

    // Explicit per-module granularity round-trips.
    let micro = EvolveConfig::from_toml_str(
        "[train]\npreset=\"lora\"\n  [train.fractional]\n  block_size=1\n  granularity=\"module\"\n",
    )
    .unwrap();
    assert_eq!(
        micro.train.unwrap().fractional.unwrap().granularity,
        "module"
    );

    // Absent ⇒ None (dense training; non-breaking).
    let plain = EvolveConfig::from_toml_str("[train]\npreset = \"lora\"").unwrap();
    assert!(plain.train.unwrap().fractional.is_none());
}

// ---------------------------------------------------------------------------
// `[export]` config-driven export pipeline (additive).
// ---------------------------------------------------------------------------

#[test]
fn export_config_round_trips_with_merge_shards_and_defaults() {
    let toml = r#"
[export]
quant = "Q5_K_M"
dtype = "bfloat16"
llama_cpp_path = "~/llama.cpp"
work_path = "~/scrt-export"
place_dir = "/models/lmstudio"
max_shard_size = "3GB"
  [export.merge_shards]
  enabled = true
  pattern = "adapter-shard-*.safetensors"
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let e = cfg.export.as_ref().expect("[export]");
    assert_eq!(e.quant, "Q5_K_M");
    assert_eq!(e.dtype, "bfloat16");
    assert_eq!(e.llama_cpp_path.as_deref(), Some("~/llama.cpp"));
    assert_eq!(e.work_path.as_deref(), Some("~/scrt-export"));
    assert_eq!(e.place_dir.as_deref(), Some("/models/lmstudio"));
    let ms = e.merge_shards.as_ref().expect("[export.merge_shards]");
    assert!(ms.enabled);
    assert_eq!(ms.pattern, "adapter-shard-*.safetensors");

    // Round-trips.
    let ser = toml::to_string(&cfg).unwrap();
    let back = EvolveConfig::from_toml_str(&ser).unwrap();
    assert_eq!(back.export.unwrap().quant, "Q5_K_M");

    // Defaults when fields omitted: quant Q4_K_M, dtype bfloat16, shard 3GB.
    let minimal = EvolveConfig::from_toml_str("[export]\n").unwrap();
    let e2 = minimal.export.unwrap();
    assert_eq!(e2.quant, "Q4_K_M");
    assert_eq!(e2.dtype, "bfloat16");
    assert_eq!(e2.max_shard_size, "3GB");
    assert!(e2.merge_shards.is_none());

    // Absent ⇒ None (CLI flag defaults; non-breaking).
    let none = EvolveConfig::from_toml_str("[evolve]\nmodel_path=\"/m\"").unwrap();
    assert!(none.export.is_none());
}

// ---------------------------------------------------------------------------
// `[runtime]` inference runtime config (additive).
// ---------------------------------------------------------------------------

#[test]
fn runtime_config_round_trips_with_sampling_and_defaults() {
    let toml = r#"
[runtime]
backend = "llamacpp"
model_path = "/m/model-Q4_K_M.gguf"
llama_cpp_path = "~/llama.cpp"
n_ctx = 8192
n_gpu_layers = 99
n_threads = 8
  [runtime.sampling]
  temperature = 0.2
  top_p = 0.95
  max_tokens = 128
"#;
    let cfg = EvolveConfig::from_toml_str(toml).unwrap();
    let r = cfg.runtime.as_ref().expect("[runtime]");
    assert_eq!(r.backend, "llamacpp");
    assert_eq!(r.model_path.as_deref(), Some("/m/model-Q4_K_M.gguf"));
    assert_eq!(r.n_ctx, 8192);
    assert_eq!(r.n_gpu_layers, 99);
    let s = r.sampling.as_ref().expect("[runtime.sampling]");
    assert_eq!(s.max_tokens, 128);
    assert!((s.top_p - 0.95).abs() < 1e-6);

    // Round-trips.
    let ser = toml::to_string(&cfg).unwrap();
    let back = EvolveConfig::from_toml_str(&ser).unwrap();
    assert_eq!(back.runtime.unwrap().n_gpu_layers, 99);

    // Defaults when omitted: backend llamacpp, n_ctx 8192, ngl 0.
    let minimal = EvolveConfig::from_toml_str("[runtime]\n").unwrap();
    let r2 = minimal.runtime.unwrap();
    assert_eq!(r2.backend, "llamacpp");
    assert_eq!(r2.n_ctx, 8192);
    assert_eq!(r2.n_gpu_layers, 0);
    assert!(r2.sampling.is_none());

    // Absent ⇒ None (transformers fallback; non-breaking).
    let none = EvolveConfig::from_toml_str("[evolve]\nmodel_path=\"/m\"").unwrap();
    assert!(none.runtime.is_none());
}

#[test]
fn valid_env_var_name_is_accepted() {
    for name in ["SCRT_EVOLVE_API_KEY", "MY_KEY", "_X", "OPENAI_API_KEY"] {
        let toml = format!("[generate]\n  [generate.api]\n  api_key_env = \"{name}\"\n");
        EvolveConfig::from_toml_str(&toml)
            .unwrap_or_else(|e| panic!("`{name}` should be a valid env var name, got {e:?}"));
    }
}
