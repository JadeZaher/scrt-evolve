//! model_store.rs — bounded, config-driven model-weight VERSION store.
//!
//! The base model is **immutable and shared**; a "version" of an evolved branch
//! is just its **adapter** (kilobytes–megabytes) plus an optional exported
//! **GGUF** (the deploy artifact). So keeping a small rollback history costs
//! almost nothing — `keep_versions` (default 2) bounds the ring and older
//! versions are pruned on commit.
//!
//! This is the "swap in place + reversible" mechanism done at the right
//! granularity: the GGUF is a *regenerated, gated deploy artifact*, while the
//! **adapter lineage is the reverse trace**. A kept evolve round [`commit`]s a new
//! version (set `current`, prune to bound); [`rollback`] repoints `current` to its
//! parent (the live model is reconstructable as base + that version's adapter).
//!
//! Pure storage/serde — no ML. Mirrors the atomic-write + schema-versioned
//! pattern of [`crate::branch::manifest`].

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::EvolveConfig;

/// Schema version of `store.json` (bump on a breaking layout change).
pub const STORE_SCHEMA_VERSION: u32 = 1;

/// Errors from the model store.
#[derive(Debug)]
pub enum StoreError {
    Io(std::io::Error),
    Json(serde_json::Error),
    SchemaMismatch { found: u32, expected: u32 },
    NotFound(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "model store io error: {e}"),
            StoreError::Json(e) => write!(f, "model store json error: {e}"),
            StoreError::SchemaMismatch { found, expected } => write!(
                f,
                "model store schema mismatch: found {found}, expected {expected}"
            ),
            StoreError::NotFound(id) => write!(f, "model store version not found: {id}"),
        }
    }
}

impl std::error::Error for StoreError {}
impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}
impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        StoreError::Json(e)
    }
}

/// One stored model version: an adapter (the reverse trace) + optional GGUF over
/// the shared immutable base. Paths are RELATIVE to the store dir so the ring is
/// relocatable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Monotonic id, e.g. `v3`.
    pub id: String,
    /// The version this one evolved from (the rollback target). `None` for the
    /// first version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Adapter dir relative to the store dir (e.g. `v3/adapter`).
    pub adapter: String,
    /// Exported GGUF relative to the store dir, if one was committed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf: Option<String>,
    /// The eval metrics that admitted this version (flattened `ScoreReport`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    /// The probe version this version's `correctness` was measured on. The next
    /// round rebuilds its baseline from this + `correctness`, so the candidate
    /// (scored on the same stable probe) and the baseline carry the SAME
    /// `probe_version` → the verdict is a real same-exam comparison. `None` for
    /// versions committed before stable probes, or with no probe (uncovered).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_version: Option<String>,
    /// ISO-8601 creation timestamp (passed in for determinism).
    pub created: String,
}

/// The on-disk `store.json` manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreManifest {
    pub schema_version: u32,
    /// The immutable base every version's adapter applies to.
    pub base_model: String,
    /// The currently-deployed version id (the live model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<String>,
    /// Retention bound (current + prior). Pruned on commit.
    pub keep_versions: usize,
    /// Versions in commit order (oldest first).
    pub versions: Vec<ModelVersion>,
}

/// An absolute-path view of a version, ready for loading.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedVersion {
    pub id: String,
    pub base_model: String,
    pub adapter_dir: PathBuf,
    pub gguf: Option<PathBuf>,
}

/// The bounded version store rooted at a directory.
#[derive(Debug, Clone)]
pub struct ModelStore {
    dir: PathBuf,
    manifest: StoreManifest,
}

impl ModelStore {
    /// The store directory from config: `[store].dir` or `<work_dir>/store`.
    pub fn dir_from_config(cfg: &EvolveConfig) -> PathBuf {
        cfg.store
            .as_ref()
            .and_then(|s| s.dir.as_ref())
            .map(PathBuf::from)
            .unwrap_or_else(|| cfg.work_dir().join("store"))
    }

    /// `keep_versions` from config (min 1), default 2.
    pub fn keep_from_config(cfg: &EvolveConfig) -> usize {
        cfg.store
            .as_ref()
            .map(|s| s.keep_versions)
            .unwrap_or(2)
            .max(1)
    }

    /// Open the store at `dir`, loading `store.json` if present or initializing an
    /// empty ring keyed to `base_model`. Creates the dir.
    pub fn open(
        dir: impl AsRef<Path>,
        base_model: &str,
        keep_versions: usize,
    ) -> Result<Self, StoreError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("store.json");
        let manifest = if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let m: StoreManifest = serde_json::from_str(&text)?;
            if m.schema_version != STORE_SCHEMA_VERSION {
                return Err(StoreError::SchemaMismatch {
                    found: m.schema_version,
                    expected: STORE_SCHEMA_VERSION,
                });
            }
            m
        } else {
            StoreManifest {
                schema_version: STORE_SCHEMA_VERSION,
                base_model: base_model.to_string(),
                current: None,
                keep_versions: keep_versions.max(1),
                versions: Vec::new(),
            }
        };
        Ok(Self { dir, manifest })
    }

    /// Open from an `EvolveConfig` (resolving dir + keep_versions + base from it).
    pub fn from_config(cfg: &EvolveConfig) -> Result<Self, StoreError> {
        let base = cfg
            .evolve
            .model_path
            .as_deref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self::open(
            Self::dir_from_config(cfg),
            &base,
            Self::keep_from_config(cfg),
        )
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
    pub fn manifest(&self) -> &StoreManifest {
        &self.manifest
    }
    pub fn versions(&self) -> &[ModelVersion] {
        &self.manifest.versions
    }

    /// The next monotonic id (`v1`, `v2`, …) — one past the highest existing.
    pub fn next_id(&self) -> String {
        let n = self
            .manifest
            .versions
            .iter()
            .filter_map(|v| v.id.strip_prefix('v').and_then(|s| s.parse::<u64>().ok()))
            .max()
            .unwrap_or(0);
        format!("v{}", n + 1)
    }

    /// Commit a new version: copy `adapter_src` (and `gguf_src` if given) into the
    /// ring under a fresh id, set it `current` (parent = the previous current),
    /// then prune to `keep_versions`. Returns the committed id. `probe_version` is
    /// the exam this version's `correctness` was scored on (carried so the next
    /// round's baseline is comparable). `created` is an ISO-8601 timestamp (passed
    /// for determinism). Atomically rewrites `store.json`.
    pub fn commit(
        &mut self,
        adapter_src: &Path,
        gguf_src: Option<&Path>,
        metrics: BTreeMap<String, f64>,
        probe_version: Option<String>,
        created: &str,
    ) -> Result<String, StoreError> {
        let id = self.next_id();
        let vdir = self.dir.join(&id);
        std::fs::create_dir_all(&vdir)?;

        let adapter_rel = format!("{id}/adapter");
        copy_dir(adapter_src, &self.dir.join(&adapter_rel))?;

        let gguf_rel = match gguf_src {
            Some(src) => {
                let rel = format!("{id}/model.gguf");
                std::fs::copy(src, self.dir.join(&rel))?;
                Some(rel)
            }
            None => None,
        };

        let parent = self.manifest.current.clone();
        self.manifest.versions.push(ModelVersion {
            id: id.clone(),
            parent,
            adapter: adapter_rel,
            gguf: gguf_rel,
            metrics,
            probe_version,
            created: created.to_string(),
        });
        self.manifest.current = Some(id.clone());
        self.prune()?;
        self.save()?;
        Ok(id)
    }

    /// The current (live) version, if any.
    pub fn current(&self) -> Option<&ModelVersion> {
        let cur = self.manifest.current.as_deref()?;
        self.get(cur)
    }

    /// Look up a version by id.
    pub fn get(&self, id: &str) -> Option<&ModelVersion> {
        self.manifest.versions.iter().find(|v| v.id == id)
    }

    /// Resolve a version to absolute load paths (base + adapter dir + GGUF).
    pub fn resolve(&self, id: &str) -> Option<ResolvedVersion> {
        let v = self.get(id)?;
        Some(ResolvedVersion {
            id: v.id.clone(),
            base_model: self.manifest.base_model.clone(),
            adapter_dir: self.dir.join(&v.adapter),
            gguf: v.gguf.as_ref().map(|g| self.dir.join(g)),
        })
    }

    /// Resolve the current version for loading.
    pub fn resolve_current(&self) -> Option<ResolvedVersion> {
        let cur = self.manifest.current.clone()?;
        self.resolve(&cur)
    }

    /// Roll the live model back to the current version's PARENT (the reverse
    /// trace). Returns the new current id, or `None` if there is no parent to
    /// revert to. The reverted-from version stays in the ring (until pruned) so a
    /// re-promote is possible. Atomically rewrites `store.json`.
    pub fn rollback(&mut self) -> Result<Option<String>, StoreError> {
        let cur = match self.current() {
            Some(c) => c.clone(),
            None => return Ok(None),
        };
        let parent = match cur.parent {
            Some(p) => p,
            None => return Ok(None),
        };
        self.manifest.current = Some(parent.clone());
        self.save()?;
        Ok(Some(parent))
    }

    /// Promote an arbitrary stored version to `current` (manual redeploy).
    pub fn promote(&mut self, id: &str) -> Result<(), StoreError> {
        if self.get(id).is_none() {
            return Err(StoreError::NotFound(id.to_string()));
        }
        self.manifest.current = Some(id.to_string());
        self.save()
    }

    /// Prune the ring to `keep_versions`: retain the most-recently-committed N
    /// PLUS the current version (and its parent, so a rollback target survives);
    /// delete the rest from disk + manifest. Never touches the shared base.
    fn prune(&mut self) -> Result<(), StoreError> {
        let keep = self.manifest.keep_versions.max(1);
        let mut keep_ids: HashSet<String> = self
            .manifest
            .versions
            .iter()
            .rev()
            .take(keep)
            .map(|v| v.id.clone())
            .collect();
        if let Some(cur) = &self.manifest.current {
            keep_ids.insert(cur.clone());
            // Keep the current's parent too, so rollback always has a target.
            if let Some(parent) = self.get(cur).and_then(|v| v.parent.clone()) {
                keep_ids.insert(parent);
            }
        }

        let (kept, dropped): (Vec<_>, Vec<_>) = self
            .manifest
            .versions
            .drain(..)
            .partition(|v| keep_ids.contains(&v.id));
        for v in &dropped {
            let _ = std::fs::remove_dir_all(self.dir.join(&v.id));
        }
        self.manifest.versions = kept;
        Ok(())
    }

    /// Atomic `store.json` write (tmp + rename).
    fn save(&self) -> Result<(), StoreError> {
        let path = self.dir.join("store.json");
        let tmp = self.dir.join("store.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.manifest)?)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

/// Recursively copy a directory tree (std has no built-in). Used to snapshot a
/// version's adapter into the ring.
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("scrt-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Make a fake adapter dir with one file.
    fn fake_adapter(root: &Path, tag: &str) -> PathBuf {
        let d = root.join(format!("src-{tag}"));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("adapter.safetensors"), tag.as_bytes()).unwrap();
        d
    }

    fn metrics(c: f64) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("correctness".to_string(), c);
        m
    }

    #[test]
    fn commit_sets_current_and_copies_adapter() {
        let root = tmp("commit");
        let store_dir = root.join("store");
        let mut store = ModelStore::open(&store_dir, "/base", 2).unwrap();
        let a = fake_adapter(&root, "a");
        let id = store
            .commit(&a, None, metrics(0.9), None, "2026-06-26T00:00:00Z")
            .unwrap();
        assert_eq!(id, "v1");
        assert_eq!(store.current().unwrap().id, "v1");
        // Adapter snapshot copied into the ring (independent of the source).
        assert!(store_dir.join("v1/adapter/adapter.safetensors").exists());
        // store.json round-trips with no tmp left behind.
        let reload = ModelStore::open(&store_dir, "/base", 2).unwrap();
        assert_eq!(reload.current().unwrap().id, "v1");
        assert!(!store_dir.join("store.json.tmp").exists());
    }

    #[test]
    fn ring_is_bounded_by_keep_versions() {
        let root = tmp("bound");
        let store_dir = root.join("store");
        let mut store = ModelStore::open(&store_dir, "/base", 2).unwrap();
        for i in 0..4 {
            let a = fake_adapter(&root, &format!("a{i}"));
            store
                .commit(&a, None, metrics(0.9), None, "2026-06-26T00:00:00Z")
                .unwrap();
        }
        // keep_versions=2 ⇒ only the last two survive (plus parent-of-current,
        // which IS one of the last two here).
        let ids: Vec<&str> = store.versions().iter().map(|v| v.id.as_str()).collect();
        assert_eq!(ids, vec!["v3", "v4"], "older versions pruned");
        assert_eq!(store.current().unwrap().id, "v4");
        // Pruned version dirs are gone from disk.
        assert!(!store_dir.join("v1").exists());
        assert!(!store_dir.join("v2").exists());
        assert!(store_dir.join("v3/adapter/adapter.safetensors").exists());
    }

    #[test]
    fn rollback_repoints_current_to_parent() {
        let root = tmp("rollback");
        let store_dir = root.join("store");
        let mut store = ModelStore::open(&store_dir, "/base", 3).unwrap();
        let a = fake_adapter(&root, "a");
        store.commit(&a, None, metrics(0.8), None, "t").unwrap(); // v1
        let b = fake_adapter(&root, "b");
        store.commit(&b, None, metrics(0.9), None, "t").unwrap(); // v2 (parent v1)
        assert_eq!(store.current().unwrap().id, "v2");
        let reverted = store.rollback().unwrap();
        assert_eq!(reverted, Some("v1".to_string()));
        assert_eq!(store.current().unwrap().id, "v1");
        // v2 still in the ring (re-promotable).
        assert!(store.get("v2").is_some());
        // First version has no parent ⇒ no further rollback.
        assert_eq!(store.rollback().unwrap(), None);
    }

    #[test]
    fn resolve_returns_absolute_load_paths() {
        let root = tmp("resolve");
        let store_dir = root.join("store");
        let mut store = ModelStore::open(&store_dir, "/models/base", 2).unwrap();
        let a = fake_adapter(&root, "a");
        let gguf = root.join("m.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        store
            .commit(&a, Some(&gguf), metrics(0.9), None, "t")
            .unwrap();
        let r = store.resolve_current().unwrap();
        assert_eq!(r.base_model, "/models/base");
        assert_eq!(r.adapter_dir, store_dir.join("v1/adapter"));
        assert_eq!(r.gguf, Some(store_dir.join("v1/model.gguf")));
        assert!(r.gguf.unwrap().exists());
    }

    #[test]
    fn probe_version_round_trips_for_the_cross_round_gate() {
        let root = tmp("probever");
        let store_dir = root.join("store");
        let mut store = ModelStore::open(&store_dir, "/base", 2).unwrap();
        let a = fake_adapter(&root, "a");
        store
            .commit(
                &a,
                None,
                metrics(0.7),
                Some("probe-vabc123-n10".to_string()),
                "t",
            )
            .unwrap();
        // The committed version carries the exam it was scored on...
        assert_eq!(
            store.current().unwrap().probe_version.as_deref(),
            Some("probe-vabc123-n10")
        );
        // ...and it survives a reload (so the next round's baseline is comparable).
        let reload = ModelStore::open(&store_dir, "/base", 2).unwrap();
        assert_eq!(
            reload.current().unwrap().probe_version.as_deref(),
            Some("probe-vabc123-n10")
        );
        assert_eq!(
            reload.current().unwrap().metrics.get("correctness"),
            Some(&0.7)
        );
    }

    #[test]
    fn missing_store_is_empty_and_schema_is_guarded() {
        let root = tmp("schema");
        let store_dir = root.join("store");
        // Empty store: no current.
        let store = ModelStore::open(&store_dir, "/base", 2).unwrap();
        assert!(store.current().is_none());
        assert_eq!(store.next_id(), "v1");
        // A future schema is refused.
        std::fs::write(
            store_dir.join("store.json"),
            r#"{"schema_version":999,"base_model":"/b","keep_versions":2,"versions":[]}"#,
        )
        .unwrap();
        let err = ModelStore::open(&store_dir, "/base", 2).unwrap_err();
        assert!(matches!(err, StoreError::SchemaMismatch { found: 999, .. }));
    }
}
