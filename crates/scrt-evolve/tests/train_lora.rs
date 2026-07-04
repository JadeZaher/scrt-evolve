//! Track 04 — LoRA training acceptance tests.
//!
//! Mechanical bar (spec §Constraints, not a quality claim): injected adapter
//! count/shape reflects config, a tiny fixed batch overfits (loss goes down,
//! deterministically seeded), the adapter saves + reloads shape-checked, and
//! the `train::run` driver routes the `lora` preset (+ surfaces a load error
//! rather than panicking on a missing model).

#![cfg(feature = "train")]

use scrt_evolve::config::{EvolveConfig, LoraConfig, TrainConfig};
use scrt_evolve::dataset::{Dataset, GenExample, Outcome, Tier, Verdict};
use scrt_evolve::model::{LoadedModel, ModelConfig};
use scrt_evolve::train::lora::{inject_adapters, load_adapter, LoraPreset};

/// A handful of qa rows the tiny model can overfit.
fn tiny_dataset() -> Dataset {
    Dataset::new(vec![
        GenExample::Qa {
            prompt: "ping".to_string(),
            completion: "pong".to_string(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Qa {
            prompt: "hi".to_string(),
            completion: "yo".to_string(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
        GenExample::Instruction {
            instruction: "echo".to_string(),
            input: String::new(),
            output: "ok".to_string(),
            source: None,
            gen: None,
            outcome: Outcome::Unknown,
            judge_score: None,
            judge_verdict: Verdict::Unjudged,
            tier: Tier::Private,
            chosen_over: None,
        },
    ])
}

fn lora_cfg(rank: usize, alpha: usize, epochs: usize) -> LoraConfig {
    LoraConfig {
        rank,
        alpha,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        lr: 1e-2,
        epochs,
        init_adapter: None,
    }
}

#[test]
fn adapter_injection_reflects_config() {
    let cfg = ModelConfig::tiny();
    let model = LoadedModel::random_fixture(cfg.clone(), 7).expect("fixture");
    let lcfg = lora_cfg(4, 8, 1);

    let adapters = inject_adapters(&model, &lcfg, 7).expect("inject");

    // q_proj + v_proj per layer => 2 * num_layers injected pairs.
    assert_eq!(
        adapters.adapters.len(),
        cfg.num_layers * 2,
        "injected pair count must equal 2*num_layers"
    );
    // scaling = alpha / rank.
    assert!((adapters.scaling - 2.0).abs() < 1e-9, "alpha/rank scaling");

    for ad in &adapters.adapters {
        // A: [rank, in], B: [out, rank].
        assert_eq!(ad.a.as_tensor().dims(), &[lcfg.rank, ad.in_dim], "A shape");
        assert_eq!(ad.b.as_tensor().dims(), &[ad.out_dim, lcfg.rank], "B shape");
    }
}

#[test]
fn overfit_tiny_batch_loss_decreases() {
    let model = LoadedModel::random_fixture(ModelConfig::tiny(), 7).expect("fixture");
    let data = tiny_dataset();
    // Enough steps for the seeded adapter to drive loss down on a fixed batch.
    let lcfg = lora_cfg(4, 8, 25);
    let preset = LoraPreset::new(11);

    let (report, _adapters, first_loss) = preset
        .train_detailed(&model, &data, &lcfg, None)
        .expect("train");

    let first = first_loss.expect("a first-step loss");
    let final_loss = report.final_loss.expect("a final loss");
    assert!(
        report.steps > 1,
        "expected multiple steps, got {}",
        report.steps
    );
    assert!(
        final_loss < first,
        "loss must decrease: first={first} final={final_loss}"
    );

    // Determinism: same seeds + inputs => identical trajectory.
    let model2 = LoadedModel::random_fixture(ModelConfig::tiny(), 7).expect("fixture");
    let (report2, _a2, first2) = LoraPreset::new(11)
        .train_detailed(&model2, &tiny_dataset(), &lcfg, None)
        .expect("train2");
    assert_eq!(first2, first_loss, "first loss must be deterministic");
    assert_eq!(
        report2.final_loss, report.final_loss,
        "final loss deterministic"
    );
}

#[test]
fn adapter_saves_and_reloads() {
    let model = LoadedModel::random_fixture(ModelConfig::tiny(), 7).expect("fixture");
    let data = tiny_dataset();
    let lcfg = lora_cfg(4, 8, 2);

    let dir = std::env::temp_dir().join(format!("scrt_evolve_lora_rt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapter.safetensors");

    let (report, adapters, _first) = LoraPreset::new(3)
        .train_detailed(&model, &data, &lcfg, Some(&path))
        .expect("train");
    assert_eq!(report.artifact.as_deref(), Some(path.as_path()));
    assert!(path.exists(), "adapter.safetensors must exist");

    // Reload + shape-check one injected A and B.
    let loaded = load_adapter(&path, &model.device).expect("reload");
    let first = &adapters.adapters[0];
    let a_name = format!("lora.{}.a", first.target);
    let b_name = format!("lora.{}.b", first.target);
    let a = loaded.get(&a_name).expect("A present after reload");
    let b = loaded.get(&b_name).expect("B present after reload");
    assert_eq!(a.dims(), &[lcfg.rank, first.in_dim], "reloaded A shape");
    assert_eq!(b.dims(), &[first.out_dim, lcfg.rank], "reloaded B shape");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn train_run_driver_routes_lora() {
    // Driver-test choice (per task option): rather than synthesize a full
    // on-disk model (config.json + tokenizer.json + safetensors), we prove
    // routing + the error path. A `lora` preset config pointing at a
    // nonexistent model_path must route to the lora arm and surface a clear
    // load *error* (not a panic). An unknown preset must bail with a
    // later-track message. This exercises train::run end-to-end.
    let missing = std::env::temp_dir().join("scrt_evolve_no_such_model_dir_xyz");

    let mut cfg = EvolveConfig::default();
    cfg.evolve.model_path = Some(missing.clone());
    cfg.evolve.work_dir =
        Some(std::env::temp_dir().join(format!("scrt_evolve_driver_wd_{}", std::process::id())));
    cfg.train = Some(TrainConfig {
        preset: "lora".to_string(),
        lora: Some(lora_cfg(4, 8, 1)),
        ..Default::default()
    });

    let data = tiny_dataset();
    let err = scrt_evolve::train::run(&cfg, &data).expect_err("missing model must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("loading model") || msg.contains("model load failed"),
        "expected a model-load error, got: {msg}"
    );

    // Unknown preset routes to the not-implemented bail.
    cfg.train = Some(TrainConfig {
        preset: "contrastive".to_string(),
        ..Default::default()
    });
    let err2 = scrt_evolve::train::run(&cfg, &data).expect_err("unknown preset must bail");
    let msg2 = format!("{err2:#}");
    assert!(
        msg2.contains("not implemented yet"),
        "expected later-track bail, got: {msg2}"
    );
}
