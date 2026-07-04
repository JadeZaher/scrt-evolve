//! `EvolveConfig` — the `evolve.toml` schema.
//!
//! One top `[evolve]` section + per-stage (`[discover]`, `[generate]`,
//! `[train]`) + per-preset sub-blocks. Every stage reads only what it needs,
//! so **partial configs work** (generate-only, train-only): each stage block
//! is `Option`, and the per-stage `model_path` requirement is enforced only
//! when that stage actually runs (see [`EvolveConfig::require_model_path`]).
//!
//! The schema mirrors DESIGN.md §Config schema field-for-field.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Errors surfaced while loading/validating an `evolve.toml`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error(
        "`{field}` looks like an inline secret. It must be the NAME of an \
         environment variable (e.g. \"SCRT_EVOLVE_API_KEY\"), never the key \
         itself — the framework reads the named env var at runtime."
    )]
    InlineSecret { field: &'static str },
    #[error("`model_path` is required for the `{stage}` stage but was not set in [evolve]")]
    MissingModelPath { stage: &'static str },
    #[error("`{field}` out of range: {detail}")]
    OutOfRange {
        field: &'static str,
        detail: &'static str,
    },
}

/// Top-level config: `[evolve]` + the three optional stage blocks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolveConfig {
    #[serde(default)]
    pub evolve: EvolveSection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discover: Option<DiscoverConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<GenerateConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train: Option<TrainConfig>,
    /// `[eval]` — the shared evaluation harness (track 10). Top-level, like the
    /// other stage blocks (`[discover]`/`[generate]`/`[train]`). Absent ⇒ the
    /// lane runs **unguarded** (a logged warning); present ⇒ rounds are scored
    /// against a held-out probe set and gated by [`crate::eval::StepVerdict`].
    /// Additive + non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<EvalConfig>,
    /// `[regulate]` — the self-regulation / transactional homeostasis layer
    /// (track 15). Makes every weight-mutating step `checkpoint → apply → eval →
    /// keep|rollback`, with catastrophe → rollback+quarantine+halt. Absent ⇒ no
    /// transaction wrapper (steps run unguarded — a logged warning). Additive +
    /// non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regulate: Option<RegulateConfig>,
    /// `[hardware]` — the compute environment for the heavy ML subprocesses.
    /// Generic + architecture-level: declares the target device, available VRAM,
    /// and which acceleration kernels are present, so the pipeline can route /
    /// warn appropriately (e.g. a hybrid-SSM model needs CUDA + mamba kernels to
    /// train; CPU forward-only is fine for eval/teacher). Absent ⇒ auto/CPU
    /// defaults. Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware: Option<HardwareConfig>,
    /// `[export]` — the config-driven model-export pipeline (merge sharded
    /// adapter → convert → quantize → place). Absent ⇒ `export-gguf` uses its
    /// CLI-flag defaults. Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportConfig>,
    /// `[runtime]` — the inference runtime (load + run a model for generation,
    /// backend-generic: GGUF via llama.cpp, or HF via transformers). Absent ⇒
    /// `infer` uses the transformers fallback. Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeConfig>,
    /// `[branch]` — the Branch-Train-Merge **branch factory** (track 29). Turns a
    /// (small) base + optional domain corpus into a standalone domain-specialized
    /// branch (a BTM Expert LM), eval-gated + GGUF-packaged + registered + locally
    /// routed. Absent ⇒ no branch operations (today's single-model path is
    /// byte-identical). Additive + non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<BranchConfig>,
    /// Learning-by-doing **goals** (track 20). Each `[[goals]]` table declares
    /// something a local model should evolve toward and how its traces are
    /// captured (topic ⇄ palace search, tag ⇄ palace tag). Additive + non-
    /// breaking: an absent/empty `goals` reproduces today's single-run
    /// behavior (styleguide §1). Goals drive the per-goal discover→generate
    /// pipeline; the eval-gated schedule across goals is shipped — run it with
    /// `evolve --schedule` (rounds::run_schedule).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub goals: Vec<GoalConfig>,
    /// `[daemon]` — the ambient continuous-evolution daemon (track 26). Turns
    /// evolution into an always-on, VRAM-bounded background process fed by the
    /// living activity queue, every step eval-gated through track 15. Absent ⇒
    /// `evolve ambient start` uses its flag/default values. Additive + non-breaking
    /// (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub daemon: Option<DaemonConfig>,
    /// `[store]` — the bounded, config-driven model-weight VERSION store. A branch
    /// keeps a small ring of versions (current + a few prior); each version is the
    /// tiny adapter (the reverse trace) + optional GGUF (the deploy artifact) over
    /// the shared immutable base. On a kept evolve round a new version is committed
    /// and the ring is pruned to `keep_versions`; rollback repoints `current` to
    /// its parent. Absent ⇒ no version store (today's single-adapter path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<StoreConfig>,
    /// `[ingest]` — config-driven activity ingestion for the ambient daemon
    /// (sources, relevance criterion, prefilters). Absent ⇒ `evolve ambient ingest` needs
    /// explicit flags. See `crates/scrt-evolve/src/AGENTS.md` §ingest.rs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ingest: Option<IngestConfig>,
    /// `[judge]` — per-pair data judge (track 37 Phase B): scores each candidate
    /// row before it trains. Absent ⇒ no pre-queue judging (today's behavior).
    /// See `crates/scrt-evolve/src/AGENTS.md` §judge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeConfig>,
    /// `[domain]` — domain parameterization (track 37 Phase C): what the planner
    /// is tuning FOR. Absent ⇒ the built-in `scrt` defaults (behavior-identical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<DomainConfig>,
    /// `[serve]` — the live serve-while-you-train server (track 33). Turns on a
    /// long-lived inference server that hot-swaps the adapter at each keep-commit,
    /// VRAM-arbitrated against the fractional trainer. Absent ⇒ no live serving
    /// (today's one-shot `run-model` path is unchanged). Additive + non-breaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serve: Option<ServeConfig>,
}

/// `[serve.placement]` — per-shard GPU placement (track 39). Controls which
/// layer indices reside on GPU vs CPU/RAM; supports arbitrary interleaving
/// (e.g. `gpu_shards = [0,1,2,8,9,16]`) which llama.cpp's contiguous
/// `n_gpu_layers` prefix could not express.
///
/// - `mode = "auto"` (default): measure free VRAM at load and fill greedily.
/// - `mode = "manual"`: honor `gpu_shards` exactly; refuses fast on impossible maps.
///
/// When absent the engine uses its own placement heuristic (all-CPU for the
/// candle reference backend; VRAM-measured auto for live serving).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementConfig {
    /// Placement resolve mode: `auto` | `manual`.
    #[serde(default = "default_placement_mode")]
    pub mode: String,
    /// `mode = "manual"`: explicit layer indices to reside on GPU. Indices are
    /// zero-based, may be non-contiguous (interleaved). Ignored in `auto` mode.
    #[serde(default)]
    pub gpu_shards: Vec<usize>,
}

fn default_placement_mode() -> String {
    "auto".to_string()
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            mode: default_placement_mode(),
            gpu_shards: Vec::new(),
        }
    }
}

/// `[serve]` — the live inference server (track 33). All fields default so an
/// absent/partial block behaves like today (no live serving). See
/// `src/arbitration.rs` and the track 33 `AGENTS.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServeConfig {
    /// Run a long-lived server that hot-swaps the adapter at each keep-commit.
    /// `false` (default) ⇒ no live serving.
    #[serde(default)]
    pub live: bool,
    /// Coalesce swaps: only hot-swap every N `served-ready` records (`0` = swap
    /// on every keep). Bounds re-quantization cost on the GGUF path.
    #[serde(default)]
    pub swap_debounce_commits: u32,
    /// Residency mode: `auto` (doctor measures + decides), `b` (force co-resident),
    /// `a` (force strict-alternate). Default `auto`.
    #[serde(default)]
    pub mode: ServeMode,
    /// `[serve.placement]` — per-layer GPU placement for the native candle engine
    /// (track 39). `mode = "auto"` (default) probes free VRAM; `mode = "manual"`
    /// honors `gpu_shards` exactly. Absent ⇒ engine default heuristic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<PlacementConfig>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            live: false,
            swap_debounce_commits: 0,
            mode: ServeMode::Auto,
            placement: None,
        }
    }
}

/// Serve residency mode selector (track 33). `auto` lets `doctor` pick B-or-A
/// from a live VRAM measurement; `a`/`b` force the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServeMode {
    /// Measure co-resident footprint and select B (fits) or degrade to A.
    #[default]
    Auto,
    /// Force strict-alternate (mode A).
    A,
    /// Force co-resident (mode B).
    B,
}

/// `[judge]` — the per-pair data judge (track 37 Phase B). Scores every candidate
/// row 0–1 on correctness/quality/steering-alignment before it enters the living
/// queue (daemon) or the on-disk dataset (`evolve dataset judge`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeConfig {
    /// Keep only rows scoring ≥ this (0.0–1.0). Default 0.5.
    #[serde(default = "default_judge_min_score")]
    pub min_score: f32,
    /// On judge/transport error: `keep` (fail-open, default — matches the
    /// relevance-judge precedent + track-31 preflight backstop; a flaky judge
    /// must not stall an unattended daemon) or `drop` (fail-closed — the
    /// documented flip before publishing branches P2P).
    #[serde(default = "default_judge_on_error")]
    pub on_error: String,
    /// Rows per LLM call. Default 15 (matches the relevance judge).
    #[serde(default = "default_judge_batch")]
    pub batch: usize,
    /// How many generated rows to sample per daemon step for the steering-
    /// compliance metric (track 37 Phase D). Default 4. `0` ⇒ disabled.
    #[serde(default = "default_judge_sample_k")]
    pub sample_k: usize,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            min_score: default_judge_min_score(),
            on_error: default_judge_on_error(),
            batch: default_judge_batch(),
            sample_k: default_judge_sample_k(),
        }
    }
}

fn default_judge_min_score() -> f32 {
    0.5
}
fn default_judge_on_error() -> String {
    "keep".to_string()
}
fn default_judge_batch() -> usize {
    15
}
fn default_judge_sample_k() -> usize {
    4
}

/// `[domain]` — what the planner tunes FOR (track 37 Phase C). Absent, or any
/// unset field, resolves to the built-in `scrt` value so existing configs are
/// byte-identical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Short domain name used in the planner's job description (default `scrt`).
    #[serde(default = "default_domain_name")]
    pub name: String,
    /// One-line domain description for the planner prompt.
    #[serde(default = "default_domain_description")]
    pub description: String,
    /// Command prefixes that a `cli` row must start with (default `["scrt"]`).
    #[serde(default = "default_domain_command_prefixes")]
    pub command_prefixes: Vec<String>,
    /// Flag token prefixes counted as signal (default `["--mp-"]`).
    #[serde(default = "default_domain_flag_patterns")]
    pub flag_patterns: Vec<String>,
    /// Tool names counted as signal (default = the scrt tool set).
    #[serde(default = "default_domain_tools")]
    pub tools: Vec<String>,
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            name: default_domain_name(),
            description: default_domain_description(),
            command_prefixes: default_domain_command_prefixes(),
            flag_patterns: default_domain_flag_patterns(),
            tools: default_domain_tools(),
        }
    }
}

fn default_domain_name() -> String {
    "scrt".to_string()
}
fn default_domain_description() -> String {
    "the `scrt` tool — both as structured tool calls and as a CLI — plus \
     understanding its concepts"
        .to_string()
}
fn default_domain_command_prefixes() -> Vec<String> {
    vec!["scrt".to_string()]
}
fn default_domain_flag_patterns() -> Vec<String> {
    vec!["--mp-".to_string()]
}
fn default_domain_tools() -> Vec<String> {
    vec![
        "scrt_search".to_string(),
        "scrt_stash".to_string(),
        "scrt_list_stashes".to_string(),
        "scrt_get_stash".to_string(),
        "scrt_drop_stash".to_string(),
        "scrt_similar".to_string(),
    ]
}

/// `[ingest]` — what the ambient daemon mines into its living queue. Makes
/// `evolve ambient ingest` / `evolve --ambient` flagless. See `src/AGENTS.md`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Interaction-log dirs (scanned recursively for `*.jsonl`). Empty ⇒ the
    /// Claude Code projects dir (`~/.claude/projects`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<PathBuf>,
    /// Doc dirs/files chunked into `completion` rows (`*.md`/`*.txt`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub docs: Vec<PathBuf>,
    /// Cheap case-insensitive substring prefilter (bounds the LLM-judge cost).
    #[serde(default, rename = "match", skip_serializing_if = "Vec::is_empty")]
    pub match_: Vec<String>,
    /// LLM relevance criterion. Set ⇒ judge candidates via `[generate.api]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relevance: Option<String>,
    /// Target lane: `raw` (default) or `priority`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<String>,
    /// Cap rows enqueued per ingest (0 = no cap).
    #[serde(default)]
    pub max: usize,
    /// Data-sovereignty tier stamped on mined rows: `private` (default) or
    /// `shared` (track 37). Flows to the branch manifest, most-restrictive-wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
}

/// `[store]` — bounded model-weight version management (storage + loading).
///
/// The base model is immutable and shared; a "version" is just its adapter
/// (kilobytes–megabytes) plus an optional exported GGUF. So a full rollback
/// history of N versions costs almost nothing. `keep_versions` bounds the ring
/// (default 2 = current + one prior); `deploy_to` is where the live GGUF is
/// placed/swapped (e.g. an LM Studio models path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// Directory holding the version ring + `store.json` manifest. Absent ⇒
    /// `<work_dir>/store`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
    /// How many versions to retain (current + prior). Older versions are pruned
    /// on commit. Minimum 1. Default 2.
    #[serde(default = "default_keep_versions")]
    pub keep_versions: usize,
    /// Where to place/swap the live deployable GGUF on a kept commit (e.g. an LM
    /// Studio models file). Absent ⇒ no auto-deploy (the GGUF stays in the ring).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deploy_to: Option<String>,
}

fn default_keep_versions() -> usize {
    2
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            dir: None,
            keep_versions: default_keep_versions(),
            deploy_to: None,
        }
    }
}

/// One learning-by-doing goal (`[[goals]]` in `evolve.toml`).
///
/// A goal declares *what to evolve toward* and *how its traces are captured*.
/// The contract is **one goal ⇄ one tag**: the paired `scrt-evolve` skill
/// stamps goal-relevant palace stashes with `tag`, and discovery pulls exactly
/// those (`palace_tags = [tag]`, `palace_search = topic`). All fields beyond
/// the three identifiers are optional scheduler/eval hints, consumed by the
/// lane-gated round driver (tracks 10/15) when it lands.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalConfig {
    /// Stable id, e.g. `scrt-cli-fluency`. Used to namespace per-goal
    /// artifacts (`work_dir/traces/<name>/`, the `gen=trace:<name>` stamp).
    pub name: String,
    /// The subject to evolve toward — feeds `discover.palace_search` and scopes
    /// the corpus sweep.
    pub topic: String,
    /// The palace tag the skill stamps on goal-relevant stashes — feeds
    /// `discover.palace_tags`. One goal ⇄ one tag.
    pub tag: String,
    /// Optional project directory scoping the corpus/transcripts to one project.
    /// When set, discover runs against this project's corpus instead of the
    /// top-level `[evolve].corpus_dir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<PathBuf>,
    /// Optional path to this goal's held-out eval probes (consumed by track
    /// 10's `Scorer` to gate the goal's rounds). Lane-gated; unused until the
    /// eval harness lands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_set: Option<PathBuf>,
    /// Optional scheduler weight (priority hint for round-robin vs weighted
    /// scheduling). `None` ⇒ equal weight. Consumed by the shipped scheduler
    /// (`evolve --schedule`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    /// Optional scheduler cadence hint (e.g. `"1h"`, `"daily"`). `None` ⇒
    /// on-demand. Reserved hint for the scheduler (`evolve --schedule`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cadence: Option<String>,
    /// Per-goal constitution override/addition — values specific to this goal,
    /// layered on top of the global `[evolve].constitution`. Composed into the
    /// goal's generate system prompt (the steering seam).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constitution: Option<String>,
    /// Per-goal taste override/addition — representational form specific to this
    /// goal, layered on the global `[evolve].taste`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taste: Option<String>,
}

/// `[evolve]` — the top section. `model_path` is the one thing most stages
/// need, but it is kept optional at parse time so a stage that doesn't need a
/// model (e.g. a future inspect-only command) can load a config without it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolveSection {
    /// Local model directory (safetensors weights + tokenizer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// The corpus to adapt to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_dir: Option<PathBuf>,
    /// The scrt mind-palace providing the retrieval signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palace_path: Option<PathBuf>,
    /// Where datasets, adapters, and checkpoints land. Defaults to
    /// `.scrt-evolve` (see [`EvolveSection::work_dir_or_default`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_dir: Option<PathBuf>,
    /// GLOBAL constitution — values that drive HOW the model should process /
    /// answer (applied to every goal's generation). Composed into the generate
    /// system prompt (the `custom_prompt` steering seam). Minimal slice of the
    /// taste/meta-object substrate (tracks 21/22); a plain string for now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constitution: Option<String>,
    /// GLOBAL taste — the representational FORM ideas should take (style,
    /// structure, conventions). Composed into the generate system prompt
    /// alongside `constitution`. Minimal slice; a plain string for now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taste: Option<String>,
    /// EPHEMERAL live-focus (track 37 nudge). `#[serde(skip)]` — never read from
    /// or written to TOML, so it exists only on the daemon's live config clone
    /// and dies on restart (nudges are ephemeral by contract). Composed into
    /// steering for its TTL window, then cleared by the daemon.
    #[serde(skip)]
    pub focus: Option<String>,
}

impl EvolveSection {
    /// The default work-dir name when `work_dir` is unset.
    pub const DEFAULT_WORK_DIR: &'static str = ".scrt-evolve";

    /// Resolve the work-dir, falling back to the default.
    pub fn work_dir_or_default(&self) -> PathBuf {
        self.work_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(Self::DEFAULT_WORK_DIR))
    }
}

/// `[eval]` — the shared evaluation harness config (track 10).
///
/// The harness scores the current model against a held-out probe set and gates
/// evolution rounds. Every field is defaulted so an empty `[eval]` block is
/// valid; an absent block means **no eval** (the lane runs unguarded with a
/// logged warning — graceful degradation, spec §Constraints).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Path to the held-out probe set (`probe.jsonl`). Defaults to
    /// `work_dir/probe.jsonl`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_path: Option<PathBuf>,
    /// Fraction of a dataset to carve into the probe when building one
    /// (`probe build --holdout`). 0.0..=1.0.
    #[serde(default = "default_probe_holdout_frac")]
    pub probe_holdout_frac: f32,
    /// REUSE a fixed probe across rounds instead of re-carving one from each
    /// round's fresh dataset. This is what makes a multi-round branch evolve
    /// gate REAL: candidate and the stored baseline are scored on the SAME exam,
    /// so [`crate::eval::classify`] does genuine Accept/Regress/Catastrophic
    /// (re-carving per round gives each round a different probe → not comparable).
    /// First round carves once (none exists yet); later rounds load it and filter
    /// the new dataset against it (the probe is never trained on). Default
    /// `false` ⇒ the carve-each-round behavior (fine for one-shot `branch create`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stable_probe: bool,
    /// Which scorer backend to use: `api` (no ML deps — correctness +
    /// constitution only) | `transformers` (Python subprocess, real forward
    /// pass for perplexity/exit-depth) | `candle` (optional, `--features train`).
    #[serde(default = "default_scorer_backend")]
    pub scorer_backend: String,
    /// Optional judge endpoint for constitution-adherence scoring (an
    /// OpenAI-compatible chat endpoint). Reuses the `[generate.api]` shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<GenerateApiConfig>,
    /// Which metrics to compute. Unknown/unsupported metrics for the active
    /// backend are skipped with a log (graceful degrade). Defaults to the
    /// always-available `correctness`.
    #[serde(default = "default_eval_metrics")]
    pub metrics: Vec<String>,
}

fn default_probe_holdout_frac() -> f32 {
    0.1
}
fn default_scorer_backend() -> String {
    "api".to_string()
}
fn default_eval_metrics() -> Vec<String> {
    vec!["correctness".to_string()]
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            probe_path: None,
            probe_holdout_frac: default_probe_holdout_frac(),
            stable_probe: false,
            scorer_backend: default_scorer_backend(),
            judge: None,
            metrics: default_eval_metrics(),
        }
    }
}

/// `[regulate]` — the self-regulation / transactional homeostasis config
/// (track 15).
///
/// Defaults make an empty `[regulate]` block a safe, working transaction
/// wrapper: enabled, keep a few checkpoints, rollback+quarantine+halt on
/// catastrophe. Pruning (experts/base) is a documented seam (tracks 11–14) —
/// `prune` is reserved here and unused until those land.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulateConfig {
    /// Master switch. `false` ⇒ steps run unguarded (no checkpoint/eval/rollback).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// How much correctness may drop and still `accept` (absolute). Mirrors
    /// [`crate::eval::VerdictTolerances::correctness_tolerance`].
    #[serde(default = "default_accept_tolerance")]
    pub accept_tolerance: f64,
    /// Absolute correctness floor: below ⇒ `catastrophic`.
    #[serde(default = "default_catastrophe_floor")]
    pub catastrophe_floor: f64,
    /// How many checkpoints to retain (older good ones beyond this are pruned).
    #[serde(default = "default_keep_checkpoints")]
    pub keep_checkpoints: usize,
    /// Catastrophe policy. Only `rollback_quarantine_halt` is implemented; other
    /// values are accepted but treated as the default with a log.
    #[serde(default = "default_on_catastrophe")]
    pub on_catastrophe: String,
    /// Track 32 — which accept/reject GATE policy drives a step:
    /// - `"correctness"` (default, back-compat): accept unless the absolute probe
    ///   correctness dropped beyond `accept_tolerance` ([`crate::eval::classify`]).
    /// - `"judge"`: accept UNLESS an LLM judge detects DEGRADATION (sample BEFORE
    ///   base vs AFTER base+adapter on the probe prompts), with correctness demoted
    ///   to the catastrophe floor only ([`crate::eval::judge_verdict`]). Unblocks
    ///   progression on tiny QA-pair counts where the absolute score is too noisy.
    #[serde(default = "default_gate")]
    pub gate: String,
    /// `gate="judge"`: the LLM degradation-judge endpoint (OpenAI-compatible chat,
    /// same shape as `[generate.api]`). Absent ⇒ reuse `[generate.api]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_judge: Option<GenerateApiConfig>,
    /// `gate="judge"`: the fraction of probe items that may regress and still
    /// `accept`. `0.0` (default) ⇒ ANY degraded item rolls the step back.
    #[serde(default = "default_max_regressed_frac")]
    pub max_regressed_frac: f64,
}

fn default_true() -> bool {
    true
}
fn default_accept_tolerance() -> f64 {
    0.02
}
fn default_catastrophe_floor() -> f64 {
    0.10
}
fn default_keep_checkpoints() -> usize {
    5
}
fn default_on_catastrophe() -> String {
    "rollback_quarantine_halt".to_string()
}
fn default_gate() -> String {
    "correctness".to_string()
}
fn default_max_regressed_frac() -> f64 {
    0.0
}

impl Default for RegulateConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            accept_tolerance: default_accept_tolerance(),
            catastrophe_floor: default_catastrophe_floor(),
            keep_checkpoints: default_keep_checkpoints(),
            on_catastrophe: default_on_catastrophe(),
            gate: default_gate(),
            degrade_judge: None,
            max_regressed_frac: default_max_regressed_frac(),
        }
    }
}

impl RegulateConfig {
    /// The verdict tolerances implied by this config.
    pub fn tolerances(&self) -> crate::eval::VerdictTolerances {
        crate::eval::VerdictTolerances {
            correctness_tolerance: self.accept_tolerance,
            catastrophe_floor: self.catastrophe_floor,
        }
    }
}

/// `[daemon]` — the ambient continuous-evolution daemon (track 26). All fields
/// have sane defaults; CLI flags (`--max-vram`, `--max-steps`) override at
/// invocation. The daemon trains continuously but only when free VRAM ≥
/// `max_vram_gb`, and every step goes through the track-15 transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// VRAM budget in GB: the daemon trains only when at least this much is FREE,
    /// so it self-throttles around the user's foreground GPU use. `0` ⇒ ungated
    /// (train whenever the queue is non-empty).
    #[serde(default = "default_daemon_max_vram")]
    pub max_vram_gb: f64,
    /// HOST-RAM floor in GB (system-freeze guard). Train only when at least this
    /// much system RAM is FREE (`MemAvailable`). A big f16 model load — especially
    /// the A/B eval that loads the model TWICE — can otherwise consume all host
    /// memory and freeze the machine. Low RAM ⇒ WAIT (never CPU fallback, which
    /// uses more RAM). `None`/absent ⇒ ungated (back-compat).
    #[serde(default)]
    pub min_free_ram_gb: Option<f64>,
    /// Serve carve-out in GB (track 33): VRAM reserved for a co-resident live
    /// inference server. The trainer subtracts this from available headroom
    /// before starting a block (`free − reservation ≥ block_need`). `None`/absent
    /// ⇒ no carve-out (behave exactly as today). See `src/arbitration.rs`.
    #[serde(default)]
    pub serve_reservation_gb: Option<f64>,
    /// Seconds to wait when throttled or the queue is idle, before re-checking.
    #[serde(default = "default_daemon_poll")]
    pub poll_interval_secs: u64,
    /// Queued items folded into one microshard training step.
    #[serde(default = "default_daemon_batch")]
    pub batch: usize,
    /// Microshard granularity (track 25). `module` is the per-submodule VRAM
    /// floor — the default for the ambient daemon. `block` trains a layer-block.
    #[serde(default = "default_daemon_granularity")]
    pub granularity: String,
    /// Fractional learning objective for DAEMON steps (track 37 Phase E).
    /// Defaults to `end_task` — the KNOWLEDGE signal — because the ambient loop
    /// exists to actually teach the model from judged live data (unlike the
    /// non-daemon `[train.fractional].objective` default of `distill`, a
    /// representation-only near-no-op). Overrides `[train.fractional].objective`
    /// for daemon steps exactly as `granularity` overrides its fractional twin.
    /// See `crates/scrt-evolve/src/config.rs` §objective asymmetry.
    #[serde(default = "default_daemon_objective")]
    pub objective: String,
    /// Eval cadence (reserved): v1 gates EVERY step for safety; a value > 1 is
    /// accepted but does not yet skip evals (documented seam).
    #[serde(default = "default_daemon_eval_cadence")]
    pub eval_cadence: u64,
    /// **Gentle background** (coexist with gaming/video): pause GPU training
    /// whenever ANOTHER process is using the GPU, not just when VRAM is starved.
    /// A compute-heavy app with low VRAM use still stutters under contention; this
    /// yields the GPU to it entirely. Default `true`.
    #[serde(default = "default_true")]
    pub pause_on_gpu_process: bool,
    /// When the GPU is unavailable (busy or VRAM-starved), fall back to a light
    /// CPU training step instead of fully pausing — the "hybrid/adaptive" lane
    /// (GPU when free, CPU when you're gaming, pause only if even the CPU is
    /// loaded). `false` ⇒ pause (wait) instead. Default `true`.
    #[serde(default = "default_true")]
    pub cpu_fallback: bool,
    /// Train ONE layer-block per step and ROTATE which block each step
    /// (`shard_index = ordinal % rotation_blocks`), so peak VRAM stays at one
    /// block while coverage spreads over time. `0` ⇒ no rotation (train the whole
    /// model's adapters each step, today's behavior). Set to the student's layer
    /// count (or fewer, larger blocks).
    #[serde(default = "default_daemon_rotation_blocks")]
    pub rotation_blocks: usize,
    /// Seconds to sleep AFTER each executed step, capping GPU duty cycle so
    /// foreground apps get idle gaps (no sustained contention). `0` ⇒ no cooldown.
    #[serde(default = "default_daemon_cooldown")]
    pub cooldown_secs: u64,
    /// Adapter dir to SEED `work_dir/adapter` from if absent (continue an existing
    /// expert, e.g. a branch's current version). Absent ⇒ train fresh from base.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_adapter: Option<String>,
    /// Self-feed: when queue pending falls below `refill_below`, re-run `[ingest]`
    /// to mine fresh activity, then keep training. The "just goes on" switch.
    #[serde(default)]
    pub auto_ingest: bool,
    /// Refill threshold for `auto_ingest` (pending rows below which to re-ingest).
    #[serde(default = "default_daemon_refill_below")]
    pub refill_below: u64,
    /// Track 31 Q2 — resilience. Retry a TRANSIENT step failure (subprocess
    /// non-zero, OOM, endpoint blip — NOT a track-15 catastrophe) this many times
    /// with exponential backoff before recording it as a failed-but-non-halting
    /// step. `0` ⇒ no retry (fail the step immediately, old behavior).
    #[serde(default = "default_daemon_max_retries")]
    pub max_retries: u32,
    /// Base backoff seconds between transient retries (doubles each attempt).
    #[serde(default = "default_daemon_backoff")]
    pub backoff_base_secs: u64,
    /// Supervisor cap: stop the loop after this many CONSECUTIVE step failures
    /// (each already exhausted its retries). `0` ⇒ never give up on this count.
    #[serde(default = "default_daemon_max_consecutive_failures")]
    pub max_consecutive_failures: u32,
    /// Track 31 Q3 — wall-clock budget. Cap training to this many minutes per
    /// rolling hour (the daemon `Wait`s once spent, like the VRAM gate). `0` ⇒
    /// unlimited (old behavior).
    #[serde(default)]
    pub max_minutes_per_hour: u64,
    /// Track 32 — minimum genuinely-new QA pairs to train on in one step. A popped
    /// batch with fewer than this many rows is NOT trained — the rows stay queued
    /// and the loop idles (composes with the Q5 ledger's idle-on-empty), so we
    /// never overfit on 1–2 rows. `0` ⇒ no floor (train any non-empty batch). The
    /// default is conservative; tune via the bench sweep (see track 32 spec).
    #[serde(default = "default_daemon_min_train_pairs")]
    pub min_train_pairs: usize,
}

fn default_daemon_max_vram() -> f64 {
    4.0
}
fn default_daemon_poll() -> u64 {
    30
}
fn default_daemon_batch() -> usize {
    1
}
fn default_daemon_granularity() -> String {
    "module".to_string()
}
fn default_daemon_objective() -> String {
    // Track 37 Phase E: the KNOWLEDGE signal by default for the ambient loop.
    "end_task".to_string()
}
fn default_daemon_eval_cadence() -> u64 {
    1
}
fn default_daemon_rotation_blocks() -> usize {
    0
}
fn default_daemon_cooldown() -> u64 {
    0
}
fn default_daemon_refill_below() -> u64 {
    1
}
fn default_daemon_max_retries() -> u32 {
    3
}
fn default_daemon_backoff() -> u64 {
    5
}
fn default_daemon_max_consecutive_failures() -> u32 {
    5
}
fn default_daemon_min_train_pairs() -> usize {
    // Conservative default: at least half a default micro-batch (batch=8) of
    // genuinely-new signal before a step trains. Tune via the track-32 sweep.
    4
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            max_vram_gb: default_daemon_max_vram(),
            min_free_ram_gb: None,
            serve_reservation_gb: None,
            poll_interval_secs: default_daemon_poll(),
            batch: default_daemon_batch(),
            granularity: default_daemon_granularity(),
            objective: default_daemon_objective(),
            eval_cadence: default_daemon_eval_cadence(),
            pause_on_gpu_process: true,
            cpu_fallback: true,
            rotation_blocks: default_daemon_rotation_blocks(),
            cooldown_secs: default_daemon_cooldown(),
            seed_adapter: None,
            auto_ingest: false,
            refill_below: default_daemon_refill_below(),
            max_retries: default_daemon_max_retries(),
            backoff_base_secs: default_daemon_backoff(),
            max_consecutive_failures: default_daemon_max_consecutive_failures(),
            max_minutes_per_hour: 0,
            min_train_pairs: default_daemon_min_train_pairs(),
        }
    }
}

/// `[hardware]` — the compute environment for the heavy ML subprocesses
/// (track 24). Generic + architecture-level: nothing model-specific. Lets the
/// pipeline reason about whether a given model can TRAIN here (e.g. a hybrid-SSM
/// model's backward needs CUDA + the mamba kernels) vs only run forward
/// (eval/teacher), and record the machine a run happened on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Target device for training: `auto` | `cpu` | `cuda` | `mps`. `auto` lets
    /// the Python side pick (cuda if available, else cpu).
    #[serde(default = "default_device")]
    pub device: String,
    /// Approximate usable VRAM in GB (0 ⇒ unknown / CPU). Used to warn before
    /// loading a model that won't fit.
    #[serde(default)]
    pub vram_gb: f32,
    /// System RAM in GB (0 ⇒ unknown). For CPU/offload sizing.
    #[serde(default)]
    pub ram_gb: f32,
    /// Acceleration kernels available in the environment, e.g.
    /// `["mamba-ssm", "causal-conv1d", "flash-attn"]`. A hybrid-SSM model needs
    /// `mamba-ssm` + `causal-conv1d` to TRAIN (their absence ⇒ the naive CPU path
    /// that segfaults on backward). Empty ⇒ none / naive fallbacks.
    #[serde(default)]
    pub kernels: Vec<String>,
    /// Free-form description of the machine (CPU/GPU/OS) for provenance — what
    /// hardware a benchmark run was actually executed on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
    /// The Python interpreter that drives the heavy ML subprocesses (track 28
    /// packaging binding). Point this at the venv where `scrt-evolve-ml` is
    /// installed (`pip install scrt-evolve-ml[cuda]`). Resolution precedence is
    /// `--python` flag > `$SCRT_EVOLVE_PYTHON` > this field > `python` on PATH.
    /// `None` ⇒ bare `python`. The CLI runs `<python> -m scrt_evolve_*` against
    /// the INSTALLED package; a repo checkout's `python/` dir is only a fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python: Option<String>,
    /// Optional shell command run to FREE the GPU before a training step (on a
    /// single-GPU box the teacher and the trainer can't both hold VRAM). E.g.
    /// `lms unload --all` to evict the LM Studio teacher after generation. Run
    /// best-effort (failure is non-fatal). Absent ⇒ no-op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_gpu_command: Option<String>,
}

fn default_device() -> String {
    "auto".to_string()
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            vram_gb: 0.0,
            ram_gb: 0.0,
            kernels: Vec::new(),
            machine: None,
            python: None,
            free_gpu_command: None,
        }
    }
}

impl HardwareConfig {
    /// Does this environment have a given acceleration kernel?
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernels.iter().any(|k| k.eq_ignore_ascii_case(name))
    }

    /// Whether a hybrid state-space (Mamba) model can TRAIN here: needs a non-CPU
    /// device AND the mamba/conv kernels (else the naive backward segfaults).
    /// Returns `Ok(())` if trainable, else `Err(reason)` for a clear pre-flight
    /// warning. Generic — keyed on kernels, not on any model name.
    pub fn can_train_state_space(&self) -> Result<(), String> {
        let on_gpu = matches!(self.device.as_str(), "cuda" | "mps")
            || (self.device == "auto" && self.has_kernel("mamba-ssm"));
        if !on_gpu {
            return Err(
                "state-space (Mamba) training needs a CUDA/MPS device; device is \
                 cpu/auto with no GPU kernels (the naive CPU SSM backward segfaults)"
                    .to_string(),
            );
        }
        if !(self.has_kernel("mamba-ssm") && self.has_kernel("causal-conv1d")) {
            return Err(
                "state-space (Mamba) training needs the `mamba-ssm` + `causal-conv1d` \
                 kernels installed (declare them in [hardware].kernels once installed)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// `[discover]` — corpus + palace discovery strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoverConfig {
    /// `palace | corpus | both` — where discovery starts.
    #[serde(default = "default_seed")]
    pub seed: String,
    #[serde(default = "default_max_passages")]
    pub max_passages: usize,
    /// e.g. `simhash` — use scrt's similarity to drop near-dup context.
    #[serde(default = "default_dedup")]
    pub dedup: String,
    /// Spread generation across distinct topics.
    #[serde(default = "default_cluster")]
    pub cluster: bool,
    /// Patterns to sweep the corpus with when `seed` includes `corpus`. Each
    /// becomes a scrt-search query; defaults cover common doc/comment markers
    /// so a corpus sweep finds something without configuration.
    #[serde(default = "default_corpus_patterns")]
    pub corpus_patterns: Vec<String>,
    /// When `seed` includes `palace`, restrict seeding to stashes whose name,
    /// note, search pattern, or any tag contains this case-insensitive
    /// substring (scrt's `--mp-list-search`). `None` ⇒ all stashes seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palace_search: Option<String>,
    /// When `seed` includes `palace`, restrict seeding to stashes carrying ALL
    /// of these tags. Composes with `palace_search`. Empty ⇒ no tag filter.
    #[serde(default)]
    pub palace_tags: Vec<String>,
}

fn default_seed() -> String {
    "palace".to_string()
}
fn default_max_passages() -> usize {
    500
}
fn default_dedup() -> String {
    "simhash".to_string()
}
fn default_cluster() -> bool {
    true
}
fn default_corpus_patterns() -> Vec<String> {
    // Generic markers: doc comments, headings, public items. Override in
    // `[discover].corpus_patterns` to target a specific topic.
    vec![
        r"///".to_string(),
        r"^#+\s".to_string(),
        r"pub fn".to_string(),
    ]
}

impl Default for DiscoverConfig {
    fn default() -> Self {
        Self {
            seed: default_seed(),
            max_passages: default_max_passages(),
            dedup: default_dedup(),
            cluster: default_cluster(),
            corpus_patterns: default_corpus_patterns(),
            palace_search: None,
            palace_tags: Vec::new(),
        }
    }
}

/// `[generate]` — synthetic-data generation, with per-backend sub-blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConfig {
    /// `local | api`.
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_kinds")]
    pub kinds: Vec<String>,
    /// Examples synthesized per passage.
    #[serde(default = "default_per_passage")]
    pub per_passage: usize,
    /// How `tool_call` rows are rendered at export time: `gemma` (native
    /// tool_code block) is implemented; `openai` / `anthropic` are stubbed.
    #[serde(default = "default_tool_format")]
    pub tool_format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<GenerateLocalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<GenerateApiConfig>,
    /// Rejection sampling (track 37 Phase C): generate N candidate rows per seed
    /// passage, judge-rank, keep top-k above `[judge].min_score`. Default 1 =
    /// today's behavior (no rejection sampling). See `src/AGENTS.md` §judge.
    #[serde(default = "default_candidates_per_seed")]
    pub candidates_per_seed: usize,
    /// Synthesis rate (track 37 Phase C/D): fraction 0.0–1.0 of steps that run an
    /// Evol/Self-Instruct expansion pass. `None` ⇒ off. Set live via a nudge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_rate: Option<f32>,
}

fn default_candidates_per_seed() -> usize {
    1
}

fn default_tool_format() -> String {
    "gemma".to_string()
}

fn default_backend() -> String {
    "api".to_string()
}
fn default_kinds() -> Vec<String> {
    vec!["qa".to_string(), "instruction".to_string()]
}
fn default_per_passage() -> usize {
    3
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            kinds: default_kinds(),
            per_passage: default_per_passage(),
            tool_format: default_tool_format(),
            local: None,
            api: None,
            candidates_per_seed: default_candidates_per_seed(),
            synthesis_rate: None,
        }
    }
}

/// `[generate.local]` — the local candle backend knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateLocalConfig {
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_new_tokens() -> usize {
    512
}
fn default_temperature() -> f32 {
    0.7
}

impl Default for GenerateLocalConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            temperature: default_temperature(),
        }
    }
}

/// `[generate.api]` — the API backend knobs. `api_key_env` is a var NAME.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateApiConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The NAME of the env var holding the key — never the key itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Multi-turn refine if > 1.
    #[serde(default = "default_turns")]
    pub turns: usize,
}

fn default_turns() -> usize {
    1
}

/// `[train]` — preset selection + per-preset sub-blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// `lora | full | pretrain | contrastive | shard`.
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full: Option<FullConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pretrain: Option<PretrainConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrastive: Option<ContrastiveConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<ShardConfig>,
    /// `[train.qat]` — quantization-aware training (track 23). Absent ⇒ plain
    /// LoRA. Additive + non-breaking (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qat: Option<QatConfig>,
    /// `[train.fractional]` — single-node FRACTIONAL training: split the model
    /// into contiguous layer-block shards and train one block at a time via
    /// block-local distillation, bounding peak VRAM to a single block so a large
    /// model trains on a small GPU. Distinct from `[train.shard]` (which is
    /// multi-node distributed). Absent ⇒ dense training. Additive (styleguide §1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fractional: Option<FractionalConfig>,
    /// `[train.distill]` — cross-MODEL seam distillation: compress a DISTINCT,
    /// larger teacher into the (smaller) student by matching the student's
    /// per-block output to the teacher's hidden state at a mapped seam. Unlike
    /// `[train.fractional]` (which distills a block against ITS OWN frozen
    /// output — a regularization signal), this imparts the teacher's
    /// representations into a genuinely smaller model. Runs two decoupled phases
    /// (teacher pre-captures seam targets to disk → student trains against the
    /// cache) so the two models are never co-resident. Absent ⇒ no distillation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill: Option<DistillConfig>,
}

/// `[train.distill]` — cross-model seam (hidden-state) distillation.
///
/// Productizes the `bench/seam_distill` precursor: a larger TEACHER supervises a
/// smaller STUDENT at layer/seam boundaries. The student's block output is
/// matched (cosine+MSE) to the teacher's hidden state at the proportionally
/// corresponding depth. Requires a SHARED tokenizer (hidden states are matched
/// position-by-position). Pairs with `[train.fractional]` for the VRAM-bounded
/// per-block streaming (`block_size` / `calib_batches`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Master switch. `false` ⇒ ignored even if present (toggle without deleting).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// The larger TEACHER model (path or HF id). REQUIRED to activate; absent ⇒
    /// the mode no-ops (falls back to standard training).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_model: Option<String>,
    /// Teacher→student seam correspondence. `stride` (default — each student
    /// block maps to the nearest teacher seam by uniform depth ratio) or
    /// `block_avg` (average the teacher layers spanning the student block).
    #[serde(default = "default_layer_map")]
    pub layer_map: String,
    /// Hidden-state distillation loss: `cosine_mse` (default — direction +
    /// magnitude, robust across differing residual-stream scales), `mse`, or
    /// `cosine`.
    #[serde(default = "default_distill_loss")]
    pub loss: String,
    /// Width bridge when teacher/student hidden sizes differ: `auto` (default —
    /// identity if equal, else lift student→teacher width), `none` (require equal
    /// widths), or `student_up`. The projection is a distill-time scaffold,
    /// discarded after training (only the LoRA is exported).
    #[serde(default = "default_projection")]
    pub projection: String,
    /// Directory for the teacher seam cache (Phase A writes, Phase B reads).
    /// Absent ⇒ `<adapter_out>/distill_cache`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_cache: Option<String>,
    /// Gradient-clipping max-norm for the student (caps spike steps that diverge a
    /// block — the higher-magnitude deep seams are prone to this). `0` ⇒ off.
    /// Default `1.0`.
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f64,
    /// LR adaptivity. `auto` (default): a DYNAMIC per-block learning rate computed
    /// from each block's teacher-target magnitude (deep, large-magnitude blocks
    /// get a gentler rate) PLUS a warmup→cosine-decay schedule within each block.
    /// `fixed`: a constant `--lr` everywhere (the original behavior).
    #[serde(default = "default_lr_mode")]
    pub lr_mode: String,
}

fn default_grad_clip() -> f64 {
    1.0
}
fn default_lr_mode() -> String {
    "auto".to_string()
}

fn default_layer_map() -> String {
    "stride".to_string()
}
fn default_distill_loss() -> String {
    "cosine_mse".to_string()
}
fn default_projection() -> String {
    "auto".to_string()
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            teacher_model: None,
            layer_map: default_layer_map(),
            loss: default_distill_loss(),
            projection: default_projection(),
            teacher_cache: None,
            grad_clip: default_grad_clip(),
            lr_mode: default_lr_mode(),
        }
    }
}

/// `[train.fractional]` — fractional / sharded layer-block training.
///
/// Generic and model-agnostic: the Python side discovers the decoder-layer
/// stack, splits it into contiguous blocks, and trains each block's LoRA
/// adapters by distilling the frozen full-precision block (teacher) into the
/// adapted block (student). Only one block is ever resident on the accelerator,
/// so peak VRAM is bounded regardless of model depth. Pairs with `[train.qat]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalConfig {
    /// Master switch. `false` ⇒ behave as dense training even if this table is
    /// present (lets you keep the config but toggle the mode off).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Layers per block — the hard VRAM knob (smaller ⇒ less peak VRAM, more
    /// streaming). Takes precedence over `shards` when both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// Alternatively, split the model into this many equal blocks. Ignored if
    /// `block_size` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shards: Option<usize>,
    /// Train ONLY this block index (0-based) and exit — for decentralized runs
    /// (one process per shard) and for the ambient daemon's block ROTATION (train
    /// a different block each step). Absent ⇒ all blocks in sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_index: Option<usize>,
    /// Token batches each block is distilled over (boundary activations are
    /// captured from these). More ⇒ a stronger local signal.
    #[serde(default = "default_calib_batches")]
    pub calib_batches: usize,
    /// Training granularity: `block` (default — train a whole layer-block's LoRA
    /// together) or `module` (PER-MODULE sub-layer floor — train one submodule
    /// group, e.g. attention / MoE / MLP, at a time within each layer, against
    /// the layer's frozen-output teacher). `module` is the lowest-VRAM, most-
    /// passes setting (trade time for memory); pair with `block_size = 1`.
    #[serde(default = "default_granularity")]
    pub granularity: String,
    /// Learning objective: `distill` (default — block-local MSE vs the frozen
    /// block's own output; a representation/regularization signal that does NOT
    /// impart new knowledge) or `end_task` (the FINAL shard learns real
    /// cross-entropy against the completion tokens via the LM head — the actual
    /// KNOWLEDGE signal; use this to teach the model new content). Non-final
    /// shards still distill under `end_task`.
    #[serde(default = "default_objective")]
    pub objective: String,
}

fn default_calib_batches() -> usize {
    8
}
fn default_granularity() -> String {
    "block".to_string()
}
fn default_objective() -> String {
    "distill".to_string()
}

impl Default for FractionalConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            block_size: None,
            shards: None,
            shard_index: None,
            calib_batches: default_calib_batches(),
            granularity: default_granularity(),
            objective: default_objective(),
        }
    }
}

/// `[export]` — config-driven model-export pipeline: merge (sharded) adapter →
/// convert to GGUF → quantize → place. Every knob the manual pipeline needed —
/// sharding-merge rules, the merge-load dtype/device, the format conversion
/// target, source (llama.cpp) + scratch + target weight paths — lives here so
/// `evolve train export-gguf` runs the whole chain from config. Absent ⇒ the CLI
/// falls back to its flag defaults (non-breaking). Generic + architecture-level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Target quantization / output format: `Q4_K_M` | `Q5_K_M` | `Q6_K` |
    /// `Q8_0` | `f16` | `none` | … (any llama.cpp quant; `f16`/`none` skip the
    /// quantize step). The "format conversion" target.
    #[serde(default = "default_export_quant")]
    pub quant: String,
    /// dtype to load the base model in during the MERGE stage. `bfloat16`
    /// (default) avoids the float32 OOM on large/hybrid models; `float32` for
    /// max fidelity on small models.
    #[serde(default = "default_export_dtype")]
    pub dtype: String,
    /// Path to a llama.cpp checkout providing `convert_hf_to_gguf.py` +
    /// `llama-quantize` (the conversion SOURCE tooling). Auto-detected if unset
    /// (`$LLAMA_CPP`, `~/llama.cpp`, `~/.unsloth/llama.cpp`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_path: Option<String>,
    /// Scratch directory for intermediates (merged HF dir + f16 GGUF). Point
    /// this at a FAST native filesystem — on WSL, a `~/…` path, NOT a `/mnt/c`
    /// 9p mount (large writes there OOM / I/O-error). Default: alongside `out`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_path: Option<String>,
    /// Final GGUF output path (the TARGET weight file). Default:
    /// `work_dir/<model>-<quant>.gguf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_path: Option<String>,
    /// Optional directory to PLACE (copy) the finished GGUF into — e.g. an LM
    /// Studio models dir. Absent ⇒ leave it at `out_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_dir: Option<String>,
    /// `save_pretrained` shard size for the merged HF dir (caps the per-file
    /// write so a big model doesn't spike RAM). Default `3GB`.
    #[serde(default = "default_max_shard_size")]
    pub max_shard_size: String,
    /// Keep the intermediate merged-HF dir + f16 GGUF (default false ⇒ cleaned).
    #[serde(default)]
    pub keep_intermediates: bool,
    /// `[export.merge_shards]` — how to combine the per-shard adapter files that
    /// fractional training emits into the single `adapter.safetensors` the merge
    /// stage consumes. Absent ⇒ assume a single-file adapter already.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_shards: Option<MergeShardsConfig>,
}

/// `[export.merge_shards]` — sharding-merge rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeShardsConfig {
    /// Master switch. `false` ⇒ skip the merge (adapter is already single-file).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Glob (relative to the adapter dir) matching the per-shard weight files.
    /// Their keys are global-layer-indexed, so the union is order-independent.
    #[serde(default = "default_shard_glob")]
    pub pattern: String,
}

fn default_export_quant() -> String {
    "Q4_K_M".to_string()
}
fn default_export_dtype() -> String {
    "bfloat16".to_string()
}
fn default_max_shard_size() -> String {
    "3GB".to_string()
}
fn default_shard_glob() -> String {
    "adapter-shard-*.safetensors".to_string()
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            quant: default_export_quant(),
            dtype: default_export_dtype(),
            llama_cpp_path: None,
            work_path: None,
            out_path: None,
            place_dir: None,
            max_shard_size: default_max_shard_size(),
            keep_intermediates: false,
            merge_shards: None,
        }
    }
}

impl Default for MergeShardsConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            pattern: default_shard_glob(),
        }
    }
}

/// `[runtime]` — the inference runtime: how to LOAD + RUN a model efficiently for
/// generation, config-driven and backend-generic. `evolve model infer` / `evolve model run`
/// use this to serve the evolved model (or any model). Absent ⇒ infer falls back
/// to the transformers HF path against `[evolve].model_path`. Additive.
///
/// **DEPRECATED (track 39):** The llama.cpp-specific keys in this block
/// (`backend = "llamacpp"`, `llama_cpp_path`, `n_gpu_layers`) are deprecated now
/// that evolve owns its inference runtime natively (candle, in-process). Use
/// `[serve.placement]` for GPU placement control (supports interleaved shards),
/// and `[evolve].model_path` / the native `evolve model infer` command instead.
/// These keys still parse and function during the retirement window but will emit
/// a warning at config load.
///
/// `backend` selects the engine by an internal registry (no brand logic):
///   - `llamacpp`  → **DEPRECATED** — GGUF served via the llama.cpp `llama-cli`
///     runner. Migrate to the native candle engine (`evolve model infer`).
///   - `transformers` → a HuggingFace model via the Python `scrt_evolve_infer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Inference engine: `llamacpp` (GGUF) | `transformers` (HF dir).
    #[serde(default = "default_runtime_backend")]
    pub backend: String,
    /// Weights to serve. For `llamacpp` a `.gguf` file; for `transformers` an HF
    /// model dir. Absent ⇒ fall back to `[export].out_path` (llamacpp) or
    /// `[evolve].model_path` (transformers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// Path to the llama.cpp checkout/build providing the `llama-cli` runner
    /// (llamacpp backend). Auto-detected if unset (shared with `[export]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_path: Option<String>,
    /// Context window (tokens). Transcript-derived prompts are long — keep ≥ 8192.
    #[serde(default = "default_n_ctx")]
    pub n_ctx: usize,
    /// Layers to offload to the GPU (llamacpp `-ngl`). 0 ⇒ pure CPU; a high
    /// value (e.g. 99) ⇒ offload all that fit. Generic VRAM/speed knob.
    #[serde(default)]
    pub n_gpu_layers: usize,
    /// CPU threads for generation. 0 ⇒ let the engine choose.
    #[serde(default)]
    pub n_threads: usize,
    /// Sampling controls for generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingConfig>,
}

/// `[runtime.sampling]` — decoding controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// 0.0 ⇒ greedy (deterministic); >0 ⇒ sampled.
    #[serde(default)]
    pub temperature: f32,
    /// Nucleus sampling cutoff (1.0 ⇒ off).
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Max new tokens to generate per prompt.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_runtime_backend() -> String {
    "llamacpp".to_string()
}
fn default_n_ctx() -> usize {
    8192
}
fn default_top_p() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    256
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: default_runtime_backend(),
            model_path: None,
            llama_cpp_path: None,
            n_ctx: default_n_ctx(),
            n_gpu_layers: 0,
            n_threads: 0,
            sampling: None,
        }
    }
}

/// `[branch]` — the Branch-Train-Merge **branch factory** (track 29). A branch is
/// a standalone domain-specialized model (a BTM Expert LM, arXiv 2208.03306):
/// `branch create` scopes a per-branch `EvolveConfig` (override `base` + `corpus`)
/// and composes the shipped stages (discover → teacher-QA generate → train
/// `objective=end_task` → eval gate → GGUF export) inside the track-15 transaction,
/// then writes a manifest + registers it. "Smaller" comes from `base` (a small base,
/// specialized) in v1. Generic + architecture-level — no new ML lives here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchConfig {
    /// Master switch. `false` ⇒ the branch stage is skipped (back-compat).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// The small base model to specialize (path or HF id) — the "smaller" lever.
    /// Overrides `[evolve].model_path` for this branch. Absent ⇒ inherit `[evolve]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    /// Default branch name (CLI `--name` overrides). The registry/router key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Human-readable domain label, e.g. `legal/tool-calling`. Stored in the
    /// manifest; informational (routing uses `router_signature`, not this).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    /// Per-branch corpus dir/selector — overrides `[evolve].corpus_dir` for this
    /// branch so each branch trains on its own domain slice. Absent ⇒ inherit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus: Option<PathBuf>,
    /// Training objective for the branch. `end_task` (default) makes the branch
    /// learn the real downstream task (the data-sensitivity lever); other values
    /// pass through to the train preset.
    #[serde(default = "default_branch_objective")]
    pub objective: String,
    /// Branch construction mode. `standard` (default) trains the small `base` on
    /// teacher-QA data (smaller-by-base). `distill` runs cross-MODEL seam
    /// distillation: a DISTINCT larger teacher (`[train.distill].teacher_model`)
    /// supervises this branch's smaller `base` at layer seams, producing a
    /// genuinely compressed model. The weight-touching span still runs inside the
    /// track-15 transaction (eval-gate → keep|rollback).
    #[serde(default = "default_branch_mode")]
    pub mode: String,
    /// Roster cap: at most `max_branches` live branches; registering past the cap
    /// merges near-duplicates / evicts per policy (no twins). Bounded fleet.
    #[serde(default = "default_max_branches")]
    pub max_branches: usize,
    /// `[branch.router]` — request→branch routing knobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<BranchRouterConfig>,
    /// `[branch.ensemble]` — top-k blend policy for `serve --branches`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble: Option<BranchEnsembleConfig>,
    /// `[branch.serve]` — how to serve a branch GGUF (reuses `[runtime]` knobs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serve: Option<BranchServeConfig>,
}

/// `[branch.router]` — descriptor-similarity routing of a request to branch(es).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchRouterConfig {
    /// Descriptor kind: `simhash` (ML-free default) | `embedding` | `tfidf`. The
    /// `router_signature` stored in each manifest is computed with this kind.
    #[serde(default = "default_router_kind")]
    pub kind: String,
    /// Minimum similarity for a branch to be a candidate. Below it ⇒ no branch
    /// (base-only). The routing safety floor.
    #[serde(default = "default_confidence_floor")]
    pub confidence_floor: f32,
    /// How many top branches `resolve` returns (1 ⇒ single-best routing).
    #[serde(default = "default_router_top_k")]
    pub top_k: usize,
}

/// `[branch.ensemble]` — how `serve --branches` combines top-k branch outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchEnsembleConfig {
    /// `single_best` (default — serve top-1) | `average_topk` (blend top-k, the
    /// BTM inference Merge — output-average weighted by domain posterior).
    #[serde(default = "default_ensemble_mode")]
    pub mode: String,
}

/// `[branch.serve]` — branch serving knobs (a thin overlay on `[runtime]`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchServeConfig {
    /// Port for a persistent server (a later extension; v1 serve is one-shot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// GPU layers to offload when serving the branch (llama.cpp `-ngl`). Absent ⇒
    /// inherit `[runtime].n_gpu_layers`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<usize>,
}

fn default_branch_objective() -> String {
    "end_task".to_string()
}
fn default_branch_mode() -> String {
    "standard".to_string()
}
fn default_max_branches() -> usize {
    16
}
fn default_router_kind() -> String {
    "simhash".to_string()
}
fn default_confidence_floor() -> f32 {
    0.5
}
fn default_router_top_k() -> usize {
    1
}
fn default_ensemble_mode() -> String {
    "single_best".to_string()
}

impl Default for BranchConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            base: None,
            name: None,
            domain: None,
            corpus: None,
            objective: default_branch_objective(),
            mode: default_branch_mode(),
            max_branches: default_max_branches(),
            router: None,
            ensemble: None,
            serve: None,
        }
    }
}

impl Default for BranchRouterConfig {
    fn default() -> Self {
        Self {
            kind: default_router_kind(),
            confidence_floor: default_confidence_floor(),
            top_k: default_router_top_k(),
        }
    }
}

impl Default for BranchEnsembleConfig {
    fn default() -> Self {
        Self {
            mode: default_ensemble_mode(),
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
        }
    }
}

/// `[train.qat]` — quantization-aware training settings (track 23).
///
/// Simulates the deployment quant during the LoRA forward so the adapter adapts
/// to it. Generic: `quant` is any GGUF quant name (the Python side maps it to a
/// bit width); nothing here is model-specific.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatConfig {
    /// Master switch.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Target GGUF quant to simulate (e.g. `Q4_K_M`).
    #[serde(default = "default_qat_quant")]
    pub quant: String,
    /// Per-group affine group size for the fake-quant.
    #[serde(default = "default_qat_group_size")]
    pub group_size: usize,
    /// Calibration batches (0 ⇒ dynamic per-step absmax, no calibration pass).
    #[serde(default)]
    pub calibrate_batches: usize,
}

fn default_qat_quant() -> String {
    "Q4_K_M".to_string()
}
fn default_qat_group_size() -> usize {
    32
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            quant: default_qat_quant(),
            group_size: default_qat_group_size(),
            calibrate_batches: 0,
        }
    }
}

fn default_preset() -> String {
    "lora".to_string()
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            preset: default_preset(),
            lora: None,
            full: None,
            pretrain: None,
            contrastive: None,
            shard: None,
            qat: None,
            fractional: None,
            distill: None,
        }
    }
}

/// `[train.lora]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    #[serde(default = "default_lora_rank")]
    pub rank: usize,
    #[serde(default = "default_lora_alpha")]
    pub alpha: usize,
    #[serde(default = "default_lora_targets")]
    pub target_modules: Vec<String>,
    #[serde(default = "default_lora_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// CONTINUE training from an existing adapter dir (its `adapter.safetensors`
    /// is loaded into the LoRA before training). The config-driven "further
    /// training" path — a branch keeps evolving across rounds instead of
    /// restarting from a fresh adapter. Absent ⇒ fresh adapter (today's behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_adapter: Option<String>,
}

fn default_lora_rank() -> usize {
    16
}
fn default_lora_alpha() -> usize {
    32
}
fn default_lora_targets() -> Vec<String> {
    vec!["q_proj".to_string(), "v_proj".to_string()]
}
fn default_lora_lr() -> f64 {
    2e-4
}
fn default_epochs() -> usize {
    1
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: default_lora_rank(),
            alpha: default_lora_alpha(),
            target_modules: default_lora_targets(),
            lr: default_lora_lr(),
            epochs: default_epochs(),
            init_adapter: None,
        }
    }
}

/// `[train.full]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConfig {
    #[serde(default = "default_full_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_grad_accum")]
    pub grad_accum: usize,
}

fn default_full_lr() -> f64 {
    1e-5
}
fn default_grad_accum() -> usize {
    8
}

impl Default for FullConfig {
    fn default() -> Self {
        Self {
            lr: default_full_lr(),
            epochs: default_epochs(),
            grad_accum: default_grad_accum(),
        }
    }
}

/// `[train.pretrain]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainConfig {
    #[serde(default = "default_full_lr")]
    pub lr: f64,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
}

fn default_block_size() -> usize {
    1024
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self {
            lr: default_full_lr(),
            block_size: default_block_size(),
        }
    }
}

/// `[train.contrastive]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    #[serde(default = "default_negatives_per_row")]
    pub negatives_per_row: usize,
    #[serde(default = "default_contrastive_temperature")]
    pub temperature: f32,
}

fn default_negatives_per_row() -> usize {
    4
}
fn default_contrastive_temperature() -> f32 {
    0.05
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            negatives_per_row: default_negatives_per_row(),
            temperature: default_contrastive_temperature(),
        }
    }
}

/// `[train.shard]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// `coordinator | worker`.
    #[serde(default = "default_shard_role")]
    pub role: String,
    #[serde(default)]
    pub peers: Vec<String>,
    /// `data | layer`.
    #[serde(default = "default_shard_strategy")]
    pub shard_strategy: String,
    /// What each shard runs locally.
    #[serde(default = "default_base_preset")]
    pub base_preset: String,
}

fn default_shard_role() -> String {
    "coordinator".to_string()
}
fn default_shard_strategy() -> String {
    "data".to_string()
}
fn default_base_preset() -> String {
    "lora".to_string()
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            role: default_shard_role(),
            peers: Vec::new(),
            shard_strategy: default_shard_strategy(),
            base_preset: default_base_preset(),
        }
    }
}

impl EvolveConfig {
    /// Load + validate an `evolve.toml` from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&text)
    }

    /// Parse + validate from an in-memory TOML string (the testable core).
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let cfg: EvolveConfig = toml::from_str(text)?;
        cfg.validate()?;
        cfg.emit_deprecation_warnings();
        Ok(cfg)
    }

    /// Emit load-time deprecation warnings for legacy config keys.
    ///
    /// `[runtime]` llama.cpp keys (`backend = "llamacpp"`, `llama_cpp_path`,
    /// `n_gpu_layers`) are deprecated since track 39 (native candle engine).
    /// They still parse and the config is usable during the retirement window,
    /// but users should migrate to `[serve.placement]` for GPU placement and
    /// use `evolve model infer` (native) instead of the llama.cpp sidecar.
    fn emit_deprecation_warnings(&self) {
        if let Some(runtime) = &self.runtime {
            let has_llamacpp_keys = runtime.backend == "llamacpp"
                || runtime.llama_cpp_path.is_some()
                || runtime.n_gpu_layers > 0;
            if has_llamacpp_keys {
                eprintln!(
                    "[scrt-evolve] DEPRECATION WARNING: `[runtime]` llama.cpp keys \
                     (`backend = \"llamacpp\"`, `llama_cpp_path`, `n_gpu_layers`) are \
                     deprecated (track 39 — native candle inference). Migrate GPU \
                     placement to `[serve.placement]` and use `evolve model infer` \
                     for native serving. These keys will be removed after the \
                     retirement window completes. See PORTABILITY.md for details."
                );
            }
        }
    }

    /// Serialize to a pretty TOML string — used to PERSIST a branch's
    /// self-describing config (`branches/<name>/branch.toml`) so it can be
    /// reloaded and re-run with no flags.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Stage-independent validation. Stage-specific `model_path` requirements
    /// are enforced lazily by [`Self::require_model_path`] when a stage runs,
    /// so partial configs (generate-only / train-only) load fine here.
    fn validate(&self) -> Result<(), ConfigError> {
        if let Some(generate) = &self.generate {
            if let Some(api) = &generate.api {
                if let Some(key_env) = &api.api_key_env {
                    if looks_like_inline_secret(key_env) {
                        return Err(ConfigError::InlineSecret {
                            field: "generate.api.api_key_env",
                        });
                    }
                }
            }
        }
        if let Some(judge) = &self.judge {
            if !(0.0..=1.0).contains(&judge.min_score) {
                return Err(ConfigError::OutOfRange {
                    field: "judge.min_score",
                    detail: "must be between 0.0 and 1.0",
                });
            }
            let oe = judge.on_error.trim().to_ascii_lowercase();
            if oe != "keep" && oe != "drop" {
                return Err(ConfigError::OutOfRange {
                    field: "judge.on_error",
                    detail: "must be \"keep\" or \"drop\"",
                });
            }
        }
        Ok(())
    }

    /// Assert `model_path` is present, for a stage that needs it. Callers in
    /// the stage drivers (discover/generate-local/train) invoke this so the
    /// requirement is enforced only when the stage actually runs.
    pub fn require_model_path(&self, stage: &'static str) -> Result<&Path, ConfigError> {
        self.evolve
            .model_path
            .as_deref()
            .ok_or(ConfigError::MissingModelPath { stage })
    }

    /// The resolved work-dir for this config.
    pub fn work_dir(&self) -> PathBuf {
        self.evolve.work_dir_or_default()
    }

    /// Derive a per-goal [`EvolveConfig`] for the buildable discover→generate
    /// pipeline (track 20 slice 3). The returned config:
    /// - sets `discover.palace_search = goal.topic` and
    ///   `discover.palace_tags = [goal.tag]` so only the goal's tagged stashes
    ///   seed (the one-goal ⇄ one-tag contract),
    /// - forces `discover.seed = "palace"` (the curated, goal-tagged traces are
    ///   the high-signal source),
    /// - scopes `corpus_dir` to `goal.project` when set (else keeps the
    ///   top-level corpus),
    ///
    /// All other settings (`[generate]`, `[train]`, `palace_path`, `work_dir`)
    /// are inherited unchanged. This is pure (no I/O, no mutation of `self`) so
    /// it is safe to call inside a bounded scheduler loop.
    pub fn for_goal(&self, goal: &GoalConfig) -> EvolveConfig {
        let mut cfg = self.clone();

        if let Some(project) = &goal.project {
            cfg.evolve.corpus_dir = Some(project.clone());
        }

        let mut dcfg = cfg.discover.unwrap_or_default();
        // Always apply the goal's palace narrowing (topic→search, tag→tags) so a
        // populated palace seeds ONLY this goal's curated stashes. Preserve the
        // corpus dimension: if the base config already sweeps the corpus
        // (seed = corpus|both), keep it as "both" so a transcript/code corpus
        // still contributes when the palace is empty (the bench case). Only when
        // the base was palace-only do we stay palace-only.
        // `seed` is a free-form string field ("palace" | "corpus" | "both");
        // `_` is correct here — any non-corpus/both value falls back to "palace".
        dcfg.seed = match dcfg.seed.as_str() {
            "corpus" | "both" => "both".to_string(),
            _ => "palace".to_string(),
        };
        dcfg.palace_search = Some(goal.topic.clone());
        dcfg.palace_tags = vec![goal.tag.clone()];
        cfg.discover = Some(dcfg);

        // Layer the goal's constitution/taste ON TOP of the global ones so the
        // composed config carries the goal-specific steering (global base +
        // goal addition). The composer (`compose_steering`) renders both.
        if let Some(gc) = &goal.constitution {
            cfg.evolve.constitution = Some(match &cfg.evolve.constitution {
                Some(base) => format!("{base}\n{gc}"),
                None => gc.clone(),
            });
        }
        if let Some(gt) = &goal.taste {
            cfg.evolve.taste = Some(match &cfg.evolve.taste {
                Some(base) => format!("{base}\n{gt}"),
                None => gt.clone(),
            });
        }

        cfg
    }

    /// Compose the active constitution + taste into a single steering block to be
    /// injected as the generate `custom_prompt` (the seam that lets values +
    /// representational form shape the dataset, and thus training). Returns
    /// `None` when neither is set (preserves today's built-in-template behavior).
    pub fn compose_steering(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(c) = self.evolve.constitution.as_ref().map(|s| s.trim()) {
            if !c.is_empty() {
                parts.push(format!(
                    "## Constitution (values that drive how you answer)\n{c}"
                ));
            }
        }
        if let Some(t) = self.evolve.taste.as_ref().map(|s| s.trim()) {
            if !t.is_empty() {
                parts.push(format!("## Taste (the form ideas should take)\n{t}"));
            }
        }
        // Ephemeral live-focus (track 37 nudge) — a transient emphasis for its
        // TTL window; steers generation while active, then the daemon clears it.
        if let Some(f) = self.evolve.focus.as_ref().map(|s| s.trim()) {
            if !f.is_empty() {
                parts.push(format!("## Focus (emphasize this right now)\n{f}"));
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }
}

/// Heuristic: does this `api_key_env` value look like a literal secret rather
/// than an env-var NAME? Env var names are conventionally UPPER_SNAKE and
/// short; real API keys are long, mixed-case, and often carry provider
/// prefixes (`sk-`, `sk-ant-`) or non-identifier characters.
fn looks_like_inline_secret(value: &str) -> bool {
    let v = value.trim();
    // Known provider key prefixes are an immediate tell.
    if v.starts_with("sk-") || v.starts_with("sk_") {
        return true;
    }
    // Env var names are valid identifiers: [A-Za-z_][A-Za-z0-9_]*.
    let is_identifier = !v.is_empty()
        && v.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !is_identifier {
        // Contains spaces, dashes, slashes, etc. — not an env var name.
        return true;
    }
    // A very long all-caps-or-mixed token is suspicious; real env var names
    // are short. 40+ chars in a single identifier reads as a key, not a name.
    v.len() >= 40
}

#[cfg(test)]
mod placement_tests {
    use super::*;

    #[test]
    fn placement_config_round_trips() {
        // auto mode — default
        let text = "[serve.placement]\nmode = \"auto\"\n";
        let cfg = EvolveConfig::from_toml_str(text).unwrap();
        let serve = cfg.serve.expect("serve present");
        let placement = serve.placement.expect("placement present");
        assert_eq!(placement.mode, "auto");
        assert!(placement.gpu_shards.is_empty());

        // manual mode with interleaved shards
        let text = "[serve.placement]\nmode = \"manual\"\ngpu_shards = [0, 1, 2, 8, 9, 16]\n";
        let cfg = EvolveConfig::from_toml_str(text).unwrap();
        let serve = cfg.serve.expect("serve present");
        let placement = serve.placement.expect("placement present");
        assert_eq!(placement.mode, "manual");
        assert_eq!(placement.gpu_shards, vec![0, 1, 2, 8, 9, 16]);
    }

    #[test]
    fn placement_absent_is_none() {
        let cfg = EvolveConfig::from_toml_str("[serve]\nlive = true\n").unwrap();
        let serve = cfg.serve.expect("serve present");
        assert!(serve.placement.is_none());
    }

    #[test]
    fn runtime_llamacpp_keys_still_parse_and_emit_warning() {
        // A config with llama.cpp keys must parse successfully (retirement window).
        // We can't easily capture eprintln! in a test, but we can verify it doesn't
        // error and that the fields round-trip correctly.
        let text = r#"
[runtime]
backend = "llamacpp"
n_gpu_layers = 32
llama_cpp_path = "/opt/llama.cpp"
"#;
        let cfg = EvolveConfig::from_toml_str(text).unwrap();
        let runtime = cfg.runtime.expect("runtime present");
        assert_eq!(runtime.backend, "llamacpp");
        assert_eq!(runtime.n_gpu_layers, 32);
        assert_eq!(runtime.llama_cpp_path.as_deref(), Some("/opt/llama.cpp"));
    }

    #[test]
    fn runtime_and_placement_coexist() {
        // A config may have both [runtime] (legacy) and [serve.placement] (new).
        let text = r#"
[runtime]
backend = "llamacpp"
n_gpu_layers = 16

[serve.placement]
mode = "manual"
gpu_shards = [0, 1, 2]
"#;
        let cfg = EvolveConfig::from_toml_str(text).unwrap();
        let serve = cfg.serve.expect("serve present");
        let placement = serve.placement.expect("placement present");
        assert_eq!(placement.mode, "manual");
        assert_eq!(placement.gpu_shards, vec![0, 1, 2]);
        let runtime = cfg.runtime.expect("runtime present");
        assert_eq!(runtime.n_gpu_layers, 16);
    }
}

#[cfg(test)]
mod serve_tests {
    use super::*;

    #[test]
    fn serve_and_reservation_parse_from_toml() {
        let text = r#"
[daemon]
max_vram_gb = 3.5
serve_reservation_gb = 4.0

[serve]
live = true
swap_debounce_commits = 3
mode = "b"
"#;
        let cfg = EvolveConfig::from_toml_str(text).unwrap();
        let daemon = cfg.daemon.expect("daemon present");
        assert_eq!(daemon.serve_reservation_gb, Some(4.0));
        let serve = cfg.serve.expect("serve present");
        assert!(serve.live);
        assert_eq!(serve.swap_debounce_commits, 3);
        assert_eq!(serve.mode, ServeMode::B);
    }

    #[test]
    fn serve_defaults_when_omitted() {
        // No [serve] block, no serve_reservation_gb ⇒ back-compat defaults.
        let cfg = EvolveConfig::from_toml_str("[daemon]\nmax_vram_gb = 3.5\n").unwrap();
        assert!(cfg.serve.is_none());
        assert_eq!(cfg.daemon.unwrap().serve_reservation_gb, None);

        // An empty [serve] block resolves every field to its default.
        let cfg = EvolveConfig::from_toml_str("[serve]\n").unwrap();
        let serve = cfg.serve.expect("serve present");
        assert!(!serve.live);
        assert_eq!(serve.swap_debounce_commits, 0);
        assert_eq!(serve.mode, ServeMode::Auto);
    }
}
