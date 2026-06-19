//! Work-dir layout path-resolution tests.

use scrt_evolve::{EvolveConfig, WorkDir};

#[test]
fn resolves_artifact_paths_under_default_work_dir() {
    let cfg = EvolveConfig::from_toml_str("[evolve]\nmodel_path = \"/m\"").unwrap();
    let wd = WorkDir::from_config(&cfg);

    assert_eq!(wd.root(), std::path::Path::new(".scrt-evolve"));
    assert_eq!(wd.discovered_json(), wd.root().join("discovered.json"));
    assert_eq!(wd.dataset_jsonl(), wd.root().join("dataset.jsonl"));
    assert_eq!(
        wd.adapter_safetensors(),
        wd.root().join("adapter.safetensors")
    );
    assert_eq!(wd.checkpoints_dir(), wd.root().join("checkpoints"));
    assert_eq!(
        wd.checkpoint("step-100.safetensors"),
        wd.root().join("checkpoints").join("step-100.safetensors")
    );
}

#[test]
fn honors_custom_work_dir() {
    let cfg =
        EvolveConfig::from_toml_str("[evolve]\nwork_dir = \"/tmp/run-42\"").unwrap();
    let wd = WorkDir::from_config(&cfg);
    assert_eq!(wd.root(), std::path::Path::new("/tmp/run-42"));
    assert_eq!(
        wd.dataset_jsonl(),
        std::path::Path::new("/tmp/run-42").join("dataset.jsonl")
    );
}
