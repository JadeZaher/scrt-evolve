//! # Branch factory (track 29)
//!
//! The **Branch + Train** half of Branch-Train-Merge [Li & Gururangan, arXiv
//! 2208.03306; c-BTM 2303.14177]: turn a (small) base model — optionally with a
//! selected domain corpus — into a standalone, domain-specialized **branch** (a
//! BTM Expert LM), eval-gated + GGUF-packaged + registered + locally routed. The
//! base stays untouched; branches are strictly additive.
//!
//! Build is **composition-first**: [`create`] orchestrates the shipped stages
//! (discover → teacher-QA generate → train → eval gate → GGUF export) inside the
//! track-15 transaction — no new ML lives here. The net-new is the
//! manifest/registry/[`router`] layer.
//!
//! The [`manifest`] types ARE the cross-repo contract with hivemind
//! (`SCRT-EVOLVE-INTEGRATION.md`); the [`router::BranchRouter`] trait is the shared
//! routing seam (local resolver here; the remote/P2P resolver is hivemind's).

pub mod create;
pub mod manifest;
pub mod router;

pub use create::{create, BranchHooks, CreateReport};
pub use manifest::{
    sha256_file, sha256_hex, BranchManifest, BranchRegistry, Lineage, RegistryError,
    RouterSignature, MANIFEST_VERSION, REGISTRY_SCHEMA_VERSION,
};
pub use router::{
    admit, corpus_signature, AdmitOutcome, BranchRef, BranchRouter, LocalBranchRouter,
};
