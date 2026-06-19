//! Point-at-a-project resolution tests.

use scrt_evolve::project::{config_for_project, resolve};

fn temp_project(suffix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("scrt-evolve-proj-{}-{suffix}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn detects_mpg_palace_in_project() {
    let proj = temp_project("withpalace");
    std::fs::create_dir_all(proj.join(".mpg")).unwrap();
    std::fs::write(
        proj.join(".mpg/mind-palace.json"),
        r#"{"version":2,"stashes":{}}"#,
    )
    .unwrap();

    let layout = resolve(&proj).unwrap();
    assert!(layout.palace.is_some());
    assert!(layout.palace.as_ref().unwrap().ends_with("mind-palace.json"));

    let cfg = config_for_project(&layout, None);
    assert_eq!(cfg.evolve.corpus_dir.as_deref(), Some(proj.as_path()));
    assert!(cfg.evolve.palace_path.is_some());
    // work_dir defaults under the project.
    assert!(cfg.evolve.work_dir.as_ref().unwrap().ends_with(".scrt-evolve"));

    let _ = std::fs::remove_dir_all(&proj);
}

#[test]
fn corpus_only_when_no_palace() {
    let proj = temp_project("nopalace");
    std::fs::write(proj.join("README.md"), "# project\nsome docs\n").unwrap();

    let layout = resolve(&proj).unwrap();
    assert!(layout.palace.is_none());
    assert!(layout.palace_note.contains("corpus-only"));

    let cfg = config_for_project(&layout, None);
    assert_eq!(cfg.evolve.corpus_dir.as_deref(), Some(proj.as_path()));
    assert!(cfg.evolve.palace_path.is_none());

    let _ = std::fs::remove_dir_all(&proj);
}

#[test]
fn base_config_generate_settings_are_preserved() {
    let proj = temp_project("withbase");
    std::fs::write(proj.join("README.md"), "docs").unwrap();

    let base = scrt_evolve::EvolveConfig::from_toml_str(
        "[generate]\nbackend = \"api\"\nper_passage = 7",
    )
    .unwrap();
    let layout = resolve(&proj).unwrap();
    let cfg = config_for_project(&layout, Some(base));

    // Auto-detected corpus overrides, but base [generate] is preserved.
    assert_eq!(cfg.evolve.corpus_dir.as_deref(), Some(proj.as_path()));
    assert_eq!(cfg.generate.as_ref().unwrap().per_passage, 7);

    let _ = std::fs::remove_dir_all(&proj);
}

#[test]
fn errors_on_non_directory() {
    let err = resolve("/no/such/project/dir/xyz").unwrap_err();
    assert!(err.to_string().contains("not a directory"));
}
