//! `BranchRouter` — the shared request→branch routing seam (track 29).
//!
//! Routing is **per-request, not per-token** (the c-BTM property that keeps P2P
//! traffic sparse): a whole request resolves to whole branch(es) by matching its
//! domain descriptor against each branch's `router_signature`. This module ships
//! the **v1 LOCAL** resolver; hivemind implements `RemoteBranchRouter` returning
//! `(peer, branch)` over the SAME trait — one routing model, two resolvers
//! (`SCRT-EVOLVE-INTEGRATION.md` §3c). **Do not fork the trait.**
//!
//! The ML-free default descriptor is **simhash** (reuses scrt-core's simhash); an
//! `embedding`/`tfidf` centroid is an optional later kind matched by cosine.

use scrt_core::palace::simhash;

use crate::config::{BranchRouterConfig, EvolveConfig};

use super::manifest::{BranchManifest, BranchRegistry, RouterSignature};

/// A resolved routing target. v1 (local) carries the branch `name`; the remote
/// resolver (hivemind) extends the same shape with the hosting peer.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchRef {
    /// The branch name (the registry/manifest key).
    pub name: String,
    /// The branch's human-readable domain (for display / logging).
    pub domain: String,
}

/// The routing seam: resolve a request to ranked branch candidates. Returning an
/// **empty** vec means "no branch matched" → the caller serves base-only (the
/// safety floor: `router=off` / empty registry / all-below-floor ⇒ base-only).
pub trait BranchRouter {
    /// Rank branches for `req`, best first. Empty ⇒ base-only.
    fn resolve(&self, req: &str) -> Vec<(BranchRef, f32)>;
}

/// v1 local resolver: descriptor-similarity of the request against each branch's
/// `router_signature`, filtered by `confidence_floor`, top-`k`.
#[derive(Debug, Clone)]
pub struct LocalBranchRouter {
    branches: Vec<BranchManifest>,
    kind: String,
    confidence_floor: f32,
    top_k: usize,
    /// Master off-switch: when `false`, [`resolve`](Self::resolve) always returns
    /// empty (base-only, byte-identical to today's single-model path).
    enabled: bool,
}

impl LocalBranchRouter {
    /// Build a router over a registry snapshot + the `[branch.router]` config.
    pub fn new(registry: &BranchRegistry, cfg: &BranchRouterConfig) -> Self {
        Self {
            branches: registry.branches.clone(),
            kind: cfg.kind.clone(),
            confidence_floor: cfg.confidence_floor,
            top_k: cfg.top_k.max(1),
            enabled: true,
        }
    }

    /// Build a router from an `EvolveConfig`'s `[branch.router]` (defaults if absent)
    /// over a registry snapshot — the CLI convenience.
    pub fn from_config(cfg: &EvolveConfig, registry: &BranchRegistry) -> Self {
        let rcfg = cfg
            .branch
            .as_ref()
            .and_then(|b| b.router.clone())
            .unwrap_or_default();
        Self::new(registry, &rcfg)
    }

    /// An explicitly-off router: always resolves to base-only. Models `router=off`.
    pub fn off() -> Self {
        Self {
            branches: Vec::new(),
            kind: "simhash".to_string(),
            confidence_floor: 1.0,
            top_k: 1,
            enabled: false,
        }
    }
}

impl BranchRouter for LocalBranchRouter {
    fn resolve(&self, req: &str) -> Vec<(BranchRef, f32)> {
        if !self.enabled || self.branches.is_empty() {
            return Vec::new();
        }
        let req_sig = request_signature(&self.kind, req);
        let mut scored: Vec<(BranchRef, f32)> = self
            .branches
            .iter()
            .filter_map(|b| {
                let sim = signature_similarity(&req_sig, &b.router_signature)?;
                (sim >= self.confidence_floor).then_some((
                    BranchRef {
                        name: b.name.clone(),
                        domain: b.domain.clone(),
                    },
                    sim,
                ))
            })
            .collect();
        // Best first; deterministic tie-break by name.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.name.cmp(&b.0.name))
        });
        scored.truncate(self.top_k);
        scored
    }
}

/// The outcome of trying to admit a candidate branch into a registry under the
/// `max_branches` cap + near-duplicate merge policy (no twins).
#[derive(Debug, Clone, PartialEq)]
pub enum AdmitOutcome {
    /// The candidate was appended to the registry.
    Added,
    /// The candidate is a near-duplicate of an existing branch — folded in rather
    /// than spawned as a twin (the existing branch already covers the domain).
    Merged { into: String },
    /// The roster is full and the candidate is not a near-duplicate — rejected.
    Rejected { reason: String },
}

/// Admit a candidate branch into `registry` under the bounded-fleet policy
/// (styleguide §2.5). Two near-identical domains MUST collapse to one branch (no
/// twins); past `max_branches` a novel domain is rejected. Reuses track-14's
/// merge/eviction shape, specialized for standalone branches.
pub fn admit(
    registry: &mut BranchRegistry,
    candidate: BranchManifest,
    max_branches: usize,
    merge_threshold: f32,
) -> AdmitOutcome {
    // 1. Same name ⇒ a deliberate update, not a twin: replace in place.
    if registry.get(&candidate.name).is_some() {
        registry.upsert(candidate);
        return AdmitOutcome::Added;
    }
    // 2. Near-duplicate domain ⇒ merge (keep the existing branch; drop the twin).
    if let Some(existing) = registry.branches.iter().find(|b| {
        signature_similarity(&candidate.router_signature, &b.router_signature)
            .map(|s| s >= merge_threshold)
            .unwrap_or(false)
    }) {
        return AdmitOutcome::Merged {
            into: existing.name.clone(),
        };
    }
    // 3. Cap: a novel domain past the roster cap is rejected (reject the N+1th).
    if registry.branches.len() >= max_branches {
        return AdmitOutcome::Rejected {
            reason: format!("roster full ({max_branches}); novel branch not admitted"),
        };
    }
    registry.upsert(candidate);
    AdmitOutcome::Added
}

/// Compute a branch's `router_signature` from its corpus/dataset texts, using the
/// given descriptor `kind`. simhash (ML-free default) folds every text's features
/// into one corpus centroid hash; `embedding`/`tfidf` are reserved later kinds and
/// currently fall back to simhash so the build stays ML-free.
pub fn corpus_signature(kind: &str, texts: &[String]) -> RouterSignature {
    // `embedding`/`tfidf` are reserved later kinds; until they land, every kind
    // folds to the ML-free simhash centroid so the default build needs no ML.
    let _ = kind;
    let mut features: Vec<String> = Vec::new();
    for t in texts {
        features.extend(features_of(t));
    }
    RouterSignature {
        kind: "simhash".to_string(),
        vector: bits_to_vec(simhash::simhash(&features)),
    }
}

/// The request's descriptor under `kind` (mirrors [`corpus_signature`] for one text).
fn request_signature(kind: &str, req: &str) -> RouterSignature {
    corpus_signature(kind, std::slice::from_ref(&req.to_string()))
}

/// Similarity in `[0,1]` between two same-kind signatures. simhash ⇒ bit-agreement
/// (= `1 − Hamming/64`); other kinds ⇒ cosine. Mismatched kind/shape ⇒ `None`
/// (incomparable — skipped by the router).
fn signature_similarity(a: &RouterSignature, b: &RouterSignature) -> Option<f32> {
    if a.kind != b.kind || a.vector.len() != b.vector.len() || a.vector.is_empty() {
        return None;
    }
    if a.kind == "simhash" {
        let agree = a
            .vector
            .iter()
            .zip(&b.vector)
            .filter(|(x, y)| (*x - *y).abs() < f64::EPSILON)
            .count();
        Some(agree as f32 / a.vector.len() as f32)
    } else {
        cosine(&a.vector, &b.vector)
    }
}

fn cosine(a: &[f64], b: &[f64]) -> Option<f32> {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some((dot / (na * nb)) as f32)
}

/// Expand a 64-bit simhash to a 64-dim `{0.0, 1.0}` vector (MSB→LSB) so cosine /
/// bit-agreement similarity works uniformly with the centroid kinds.
fn bits_to_vec(h: u64) -> Vec<f64> {
    (0..64).map(|i| ((h >> (63 - i)) & 1) as f64).collect()
}

/// Lowercased **unigram** (bag-of-words) features. Routing is a TOPICAL match (does
/// this request's vocabulary overlap the branch's domain?), so unigrams are the right
/// granularity — unlike `discover`'s 2-shingles, which target near-identical passages
/// and almost never overlap between a short query and a corpus centroid.
fn features_of(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}
