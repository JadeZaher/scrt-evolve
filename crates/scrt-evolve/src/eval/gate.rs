//! The **executable gate** — the shared correctness primitive (track 10).
//!
//! "Does the model's emitted command actually parse and resolve against the
//! real tool surface?" is the cheapest, hardest-to-fake correctness signal for
//! a tool/CLI-trained model. This module owns that check so the eval harness
//! (this track) and the regen gate (track 11) and self-regulation (track 15)
//! all call ONE implementation and get consistent verdicts.
//!
//! It is **pure** (no model forward pass, no I/O beyond loading the static tool
//! spec): given an already-emitted `tool_call` or `cli` string, it validates
//! structure against [`crate::toolspec`] (for tool calls) or the scrt CLI flag
//! surface (for command lines). Owned HERE — not in track 11 — to break the
//! apparent 10⇄11 dependency cycle (the spec's resolution).

use std::collections::BTreeSet;

use crate::toolspec::{self, ToolSchema};

/// Why a gate check failed. A typed reason so consumers can log/aggregate the
/// failure mode, not just a bool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateFailure {
    /// The tool name is not one of scrt's real tools.
    UnknownTool { tool: String },
    /// A required parameter was missing.
    MissingRequired { tool: String, param: String },
    /// A parameter name is not in the tool's schema.
    UnknownParam { tool: String, param: String },
    /// The arguments were not a JSON object.
    ArgumentsNotObject,
    /// A CLI command did not start with `scrt`.
    NotScrtCommand,
    /// A CLI flag is not a real scrt flag.
    UnknownFlag { flag: String },
    /// The input was empty / unparseable.
    Empty,
}

impl std::fmt::Display for GateFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateFailure::UnknownTool { tool } => write!(f, "unknown tool `{tool}`"),
            GateFailure::MissingRequired { tool, param } => {
                write!(f, "`{tool}` missing required param `{param}`")
            }
            GateFailure::UnknownParam { tool, param } => {
                write!(f, "`{tool}` has unknown param `{param}`")
            }
            GateFailure::ArgumentsNotObject => write!(f, "arguments are not a JSON object"),
            GateFailure::NotScrtCommand => write!(f, "command does not start with `scrt`"),
            GateFailure::UnknownFlag { flag } => write!(f, "unknown flag `{flag}`"),
            GateFailure::Empty => write!(f, "empty / unparseable input"),
        }
    }
}

/// The verdict of an executable gate check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateVerdict {
    /// The command/tool-call parses + resolves against the real surface.
    Pass,
    /// It does not, with the reason(s) why.
    Fail(Vec<GateFailure>),
}

impl GateVerdict {
    /// Did the check pass?
    pub fn is_pass(&self) -> bool {
        matches!(self, GateVerdict::Pass)
    }
}

/// The executable gate over the scrt tool + CLI surface. Holds the loaded tool
/// schemas once so repeated checks (a whole probe set) don't re-build the spec.
#[derive(Debug, Clone)]
pub struct ExecutableGate {
    tools: Vec<ToolSchema>,
    /// The set of real scrt CLI flags, for `cli`-row validation.
    flags: BTreeSet<String>,
}

impl ExecutableGate {
    /// Build a gate from scrt's real tool schemas (loaded once).
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            tools: toolspec::scrt_tools()?,
            flags: real_scrt_flags(),
        })
    }

    /// Construct from explicit schemas + flags — for tests/mocks that don't want
    /// to depend on the live scrt-core spec.
    pub fn from_parts(tools: Vec<ToolSchema>, flags: BTreeSet<String>) -> Self {
        Self { tools, flags }
    }

    /// Validate a structured tool call: the tool exists, all required params are
    /// present, and no unknown params appear. Pure.
    pub fn check_tool_call(&self, tool: &str, arguments: &serde_json::Value) -> GateVerdict {
        let schema = match self.tools.iter().find(|t| t.name == tool) {
            Some(s) => s,
            None => {
                return GateVerdict::Fail(vec![GateFailure::UnknownTool {
                    tool: tool.to_string(),
                }])
            }
        };
        let obj = match arguments.as_object() {
            Some(o) => o,
            None => return GateVerdict::Fail(vec![GateFailure::ArgumentsNotObject]),
        };

        let mut failures = Vec::new();
        let present: BTreeSet<&str> = obj.keys().map(String::as_str).collect();

        for req in &schema.required {
            if !present.contains(req.as_str()) {
                failures.push(GateFailure::MissingRequired {
                    tool: tool.to_string(),
                    param: req.clone(),
                });
            }
        }
        let allowed: BTreeSet<&str> = schema.properties.iter().map(String::as_str).collect();
        for key in present {
            if !allowed.contains(key) {
                failures.push(GateFailure::UnknownParam {
                    tool: tool.to_string(),
                    param: key.to_string(),
                });
            }
        }

        if failures.is_empty() {
            GateVerdict::Pass
        } else {
            GateVerdict::Fail(failures)
        }
    }

    /// Validate a runnable CLI command: starts with `scrt`, and every long flag
    /// it uses is a real scrt flag. Bare commands (no flags) pass. Pure.
    pub fn check_cli(&self, command: &str) -> GateVerdict {
        let cmd = command.trim();
        if cmd.is_empty() {
            return GateVerdict::Fail(vec![GateFailure::Empty]);
        }
        // First whitespace-separated token must be the `scrt` binary (allow a
        // path like `./scrt` or `scrt.exe`).
        let first = cmd.split_whitespace().next().unwrap_or_default();
        let bin = first.rsplit(['/', '\\']).next().unwrap_or(first);
        let bin = bin.strip_suffix(".exe").unwrap_or(bin);
        if bin != "scrt" {
            return GateVerdict::Fail(vec![GateFailure::NotScrtCommand]);
        }

        let mut failures = Vec::new();
        for flag in extract_long_flags(cmd) {
            if !self.flags.contains(&flag) {
                failures.push(GateFailure::UnknownFlag { flag });
            }
        }

        if failures.is_empty() {
            GateVerdict::Pass
        } else {
            GateVerdict::Fail(failures)
        }
    }
}

/// Extract the distinct long flags (`--foo`) from a command line. `--foo=bar`
/// yields `--foo`. Deterministic (sorted, deduped).
fn extract_long_flags(cmd: &str) -> Vec<String> {
    let mut flags: BTreeSet<String> = BTreeSet::new();
    for tok in cmd.split_whitespace() {
        if let Some(rest) = tok.strip_prefix("--") {
            // Strip an `=value` suffix and any trailing punctuation.
            let name = rest.split('=').next().unwrap_or(rest);
            let name = name.trim_end_matches([',', ';', '"', '\'']);
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                flags.insert(format!("--{name}"));
            }
        }
    }
    flags.into_iter().collect()
}

/// The real scrt CLI long-flag surface (mirrors scrt-cli `args.rs`). Kept in one
/// place so the gate and the bench validator agree on what's "real". This is the
/// same set the demo benchmark.py hard-codes — lifted into the SDK so it is not
/// duplicated divergently.
fn real_scrt_flags() -> BTreeSet<String> {
    [
        "--in",
        "--cmd",
        "--url",
        "--effort",
        "--max-tokens",
        "--max-nodes",
        "--clip",
        "--sort",
        "--window-curve",
        "--format",
        "--retriever",
        "--page",
        "--page-size",
        "--mp-stash",
        "--mp-ttl",
        "--mp-tag",
        "--mp-stash-tag",
        "--mp-from",
        "--mp-compose",
        "--mp-intersect",
        "--mp-except",
        "--mp-graph",
        "--mp-link",
        "--mp-similar",
        "--mp-prune",
        "--mp-prune-keep",
        "--mp-prune-tag",
        "--mp-prune-expired",
        "--mp-prune-older-than",
        "--mp-prune-dry-run",
        "--mp-list",
        "--mp-list-search",
        "--mp-list-tag",
        "--mp-find",
        "--mp-get",
        "--mp-drop",
        "--term",
        "--match",
        "--score",
        "--top",
        "--all",
        "--fuzzy",
        "--json",
        "--help",
        "--version",
        "--no-ignore",
        "--hidden",
        "--ignore-case",
        "--serve",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}
