//! Branch artifact metadata — the **cross-repo contract** with hivemind.
//!
//! A branch (a BTM Expert LM, arXiv 2208.03306) is packaged as `{ <name>.gguf,
//! manifest.json }` and recorded in `branches/registry.json`. These serde types
//! ARE the contract documented in `SCRT-EVOLVE-INTEGRATION.md` (§3a/§3b): hivemind
//! reads the registry to discover branches + their `router_signature`s and ensemble
//! them across peers. Changing a field here is a coordinated cross-repo change.
//!
//! All writes are **atomic** (temp + rename via [`crate::harvest::write_atomic`])
//! and the GGUF is **content-addressed** by its SHA-256 (`gguf_sha`) so the artifact
//! is verifiable across repos (styleguide §2.3).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dataset::Tier;

/// The registry schema version. A registry with a different version is refused on
/// load (forward/backward-incompatible schemas must be migrated, not guessed).
pub const REGISTRY_SCHEMA_VERSION: u32 = 1;

/// The manifest format version (`version` field). Bumped on a breaking manifest
/// schema change; coordinated via `SCRT-EVOLVE-INTEGRATION.md`.
pub const MANIFEST_VERSION: &str = "1";

/// Errors specific to branch registry/manifest (de)serialization + schema checks.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("branch registry schema mismatch: file is v{found}, this build expects v{expected}")]
    SchemaMismatch { found: u32, expected: u32 },
    #[error("branch registry/manifest I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("branch registry/manifest JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// The routing descriptor a request is matched against (contract §3a). One uniform
/// shape across descriptor kinds: `simhash` (ML-free default) expands its 64-bit
/// hash to a 64-dim {0,1} vector; `embedding`/`tfidf` store a centroid directly.
/// hivemind's remote router matches the SAME shape — do not fork it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterSignature {
    /// `simhash` | `embedding` | `tfidf`.
    pub kind: String,
    /// The domain descriptor vector (a centroid). For `simhash`, the 64 hash bits
    /// as `0.0`/`1.0` so cosine similarity tracks Hamming distance.
    pub vector: Vec<f64>,
}

/// A branch's provenance — which branch (if any) it forked from (contract §3a).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Lineage {
    /// The parent branch name, if this branch was forked/derived from one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

/// One branch's `manifest.json` (contract §3a). The durable record of what the
/// branch is, what produced it, how it routes, and what gate admitted it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchManifest {
    /// The branch name — the registry + router key (e.g. `legal-tools`).
    pub name: String,
    /// The base model this branch specialized (path or HF id).
    pub base_model: String,
    /// Human-readable domain label (e.g. `legal/tool-calling`).
    pub domain: String,
    /// A description of the corpus that produced it (for provenance/audit).
    pub corpus_descriptor: String,
    /// The domain descriptor used to route requests to this branch.
    pub router_signature: RouterSignature,
    /// The eval gates that admitted the branch (metric → score).
    pub eval_report: BTreeMap<String, f64>,
    /// Fork provenance.
    #[serde(default)]
    pub lineage: Lineage,
    /// Manifest format version ([`MANIFEST_VERSION`]).
    pub version: String,
    /// SHA-256 of the GGUF artifact (content address; verifiable cross-repo).
    pub gguf_sha: String,
    /// ISO-8601 creation timestamp.
    pub created: String,
    /// Data-sovereignty tier (track 37): most-restrictive row tier in the branch corpus.
    /// Additive v1.1 field — legacy manifests without `tier` default to Private. // see AGENTS.md §manifest tier
    #[serde(default, skip_serializing_if = "tier_is_private")]
    pub tier: Tier,
}

impl BranchManifest {
    /// Serialize to pretty JSON (stable: `BTreeMap` keeps `eval_report` ordered).
    pub fn to_json(&self) -> Result<String, RegistryError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parse from JSON.
    pub fn from_json(text: &str) -> Result<Self, RegistryError> {
        Ok(serde_json::from_str(text)?)
    }

    /// Write `manifest.json` atomically (temp + rename; no half-written file).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), RegistryError> {
        let bytes = self.to_json()?.into_bytes();
        ensure_parent(path.as_ref())?;
        crate::harvest::write_atomic(path.as_ref(), &bytes)?;
        Ok(())
    }

    /// Read `manifest.json`.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&text)
    }
}

/// The branch fleet record — `branches/registry.json` (contract §3b). hivemind
/// reads this to discover branches + their `router_signature`s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchRegistry {
    /// Schema version — [`REGISTRY_SCHEMA_VERSION`]; mismatched files are refused.
    pub schema_version: u32,
    /// The branch manifests, in insertion order.
    #[serde(default)]
    pub branches: Vec<BranchManifest>,
}

impl Default for BranchRegistry {
    fn default() -> Self {
        Self {
            schema_version: REGISTRY_SCHEMA_VERSION,
            branches: Vec::new(),
        }
    }
}

impl BranchRegistry {
    /// A fresh, empty registry at the current schema version.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Load a registry, refusing a schema-version mismatch. A missing file is an
    /// empty registry (first-run ergonomics, not an error).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::empty());
        }
        let text = std::fs::read_to_string(path)?;
        let reg: Self = serde_json::from_str(&text)?;
        if reg.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err(RegistryError::SchemaMismatch {
                found: reg.schema_version,
                expected: REGISTRY_SCHEMA_VERSION,
            });
        }
        Ok(reg)
    }

    /// Write the registry atomically (temp + rename; crash-safe — §2.3).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), RegistryError> {
        let bytes = serde_json::to_string_pretty(self)?.into_bytes();
        ensure_parent(path.as_ref())?;
        crate::harvest::write_atomic(path.as_ref(), &bytes)?;
        Ok(())
    }

    /// Look up a branch by name.
    pub fn get(&self, name: &str) -> Option<&BranchManifest> {
        self.branches.iter().find(|b| b.name == name)
    }

    /// Insert or replace a branch by name (idempotent by `name`). Returns `true`
    /// when an existing entry was replaced rather than appended.
    pub fn upsert(&mut self, manifest: BranchManifest) -> bool {
        if let Some(slot) = self.branches.iter_mut().find(|b| b.name == manifest.name) {
            *slot = manifest;
            true
        } else {
            self.branches.push(manifest);
            false
        }
    }

    /// Remove a branch by name; returns the removed manifest if present.
    pub fn remove(&mut self, name: &str) -> Option<BranchManifest> {
        let idx = self.branches.iter().position(|b| b.name == name)?;
        Some(self.branches.remove(idx))
    }
}

/// SHA-256 of `bytes`, lowercase hex — the content address for a branch GGUF.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_encode(&hasher.finalize())
}

/// SHA-256 of a file (streamed in 64 KiB chunks so a multi-GB GGUF never loads
/// whole into RAM), lowercase hex.
pub fn sha256_file(path: impl AsRef<Path>) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path.as_ref())?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

/// Returns `true` when `t` is the default `Private` tier — used by serde
/// `skip_serializing_if` to keep Private off the wire (legacy manifests parse as Private via
/// `#[serde(default)]`; Shared is always serialized so peers can see the permission level).
fn tier_is_private(t: &Tier) -> bool {
    matches!(t, Tier::Private)
}

/// Create `path`'s parent directory if it doesn't exist (the atomic writer needs
/// the dir to place its sibling temp file).
fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
