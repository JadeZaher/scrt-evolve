//! Point-at-a-project resolution.
//!
//! `evolve train auto <project-dir>` should "just work": take a directory,
//! auto-detect an initialized mpg/scrt mind-palace inside it, and build an
//! [`EvolveConfig`] whose corpus is the project and whose palace is the
//! detected one — no hand-configuration. This is the project-aware entry point
//! the mpg follow-up asked for.

use std::path::{Path, PathBuf};

use crate::config::{EvolveConfig, EvolveSection};

/// Common locations an mpg/scrt palace lives at, relative to a project root.
/// Mirrors the `MPG_MIND_PALACE` default + the in-repo `.palacegen` the spike
/// used. First existing match wins.
const PALACE_CANDIDATES: &[&str] = &[
    ".mpg/mind-palace.json",
    ".mpg/palace.json",
    ".palacegen/palace.json",
    "mind-palace.json",
    "palace.json",
];

/// What [`resolve`] found for a project.
#[derive(Debug, Clone)]
pub struct ProjectLayout {
    pub root: PathBuf,
    /// The detected palace file, if any (mpg state).
    pub palace: Option<PathBuf>,
    /// Where the palace was found (for reporting), or a note if none.
    pub palace_note: String,
}

/// Detect the project layout: resolve the root and look for an initialized
/// palace. Honors `MPG_MIND_PALACE` (absolute or project-relative) first.
pub fn resolve(project_dir: impl AsRef<Path>) -> anyhow::Result<ProjectLayout> {
    let root = project_dir.as_ref();
    if !root.is_dir() {
        anyhow::bail!("evolve: not a directory: {}", root.display());
    }
    let root = root.to_path_buf();

    // 1. Explicit env override.
    if let Ok(env_path) = std::env::var("MPG_MIND_PALACE") {
        let p = PathBuf::from(&env_path);
        let abs = if p.is_absolute() { p } else { root.join(p) };
        if abs.exists() {
            return Ok(ProjectLayout {
                root,
                palace_note: format!("palace from MPG_MIND_PALACE: {}", abs.display()),
                palace: Some(abs),
            });
        }
    }

    // 2. Conventional locations under the project root.
    for cand in PALACE_CANDIDATES {
        let p = root.join(cand);
        if p.exists() {
            return Ok(ProjectLayout {
                root: root.clone(),
                palace_note: format!("detected palace: {}", p.display()),
                palace: Some(p),
            });
        }
    }

    // 3. None found — corpus-only is still valid.
    Ok(ProjectLayout {
        root,
        palace: None,
        palace_note: "no mpg palace detected — discovery will be corpus-only".into(),
    })
}

/// Build an [`EvolveConfig`] for a project, layering the auto-detected
/// corpus/palace onto an optional base config (so `[generate]`/`[train]` from a
/// base `evolve.toml` are preserved). The work-dir defaults to
/// `<project>/.scrt-evolve` unless the base config sets one.
pub fn config_for_project(layout: &ProjectLayout, base: Option<EvolveConfig>) -> EvolveConfig {
    let mut cfg = base.unwrap_or_default();
    let seed_from_palace = layout.palace.is_some();

    cfg.evolve.corpus_dir = Some(layout.root.clone());
    if cfg.evolve.palace_path.is_none() {
        cfg.evolve.palace_path = layout.palace.clone();
    }
    if cfg.evolve.work_dir.is_none() {
        cfg.evolve.work_dir = Some(layout.root.join(EvolveSection::DEFAULT_WORK_DIR));
    }
    // If a palace was detected and the discover seed wasn't set to use it,
    // default discovery to draw on both palace + corpus.
    if seed_from_palace {
        let d = cfg.discover.get_or_insert_with(Default::default);
        if d.seed == "palace" || d.seed == "corpus" {
            // keep an explicit single-source choice; only upgrade the default.
        }
        if d.seed.is_empty() {
            d.seed = "both".into();
        }
    }
    cfg
}
