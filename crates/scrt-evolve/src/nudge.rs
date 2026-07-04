//! Live steering nudges (track 35, delivered by track 37 Phase D). A `nudge.json`
//! written by `evolve ambient nudge` is polled + consumed ONCE at the top of a
//! daemon step (after the stop-check — the loop owns the config then, mirroring
//! `daemon::stop_file`). Accepted fields merge into the live config; rejected
//! fields are reported with a reason. Nudges are ephemeral — the TOML wins on
//! restart. Design + allowlist: see `src/AGENTS.md` §nudge.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::EvolveConfig;

/// The nudge control file (atomic tmp→rename), polled at the step boundary.
pub fn nudge_file(work_dir: &Path) -> PathBuf {
    work_dir.join("nudge.json")
}

/// A live steering nudge. All fields optional; only the SAFE-LIVE allowlist is
/// applied at a step boundary. Restart-required knobs (model_path, fractional
/// shape, rotation_blocks, work_dir) are intentionally ABSENT here — a nudge
/// carrying them is rejected with a reason. `focus_steps` gives a focus a TTL.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Nudge {
    /// Bump a goal's weight (sticky until the next nudge/restart).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// A transient focus topic, expiring after `focus_steps` steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_steps: Option<u64>,
    /// Data-layer knobs (track 37): judge threshold, modality mix, synthesis rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_min_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modality_mix: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates_per_seed: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_rate: Option<f32>,
    /// Gate mode override (e.g. `strict`/`degrade`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_mode: Option<String>,
    /// Restart-required fields a caller might mistakenly set — captured so we can
    /// REJECT them with a clear reason rather than silently ignore.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected_probe: Vec<String>,
}

/// The outcome of applying a nudge: what changed (for the evolution-log row) and
/// what was rejected + why (surfaced in `watch status`/`health`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NudgeOutcome {
    pub applied: Vec<String>,
    pub rejected: Vec<String>,
    /// The focus topic + its expiry step (absolute ordinal), when set.
    pub focus: Option<(String, u64)>,
}

impl NudgeOutcome {
    /// True when no knobs were applied, rejected, or focused.
    pub fn is_empty(&self) -> bool {
        self.applied.is_empty() && self.rejected.is_empty() && self.focus.is_none()
    }
}

/// Restart-required knob names — set in the config file, never live. A nudge
/// naming any of these (via `rejected_probe`) is reported as rejected.
const RESTART_REQUIRED: &[&str] = &[
    "model_path",
    "fractional",
    "rotation_blocks",
    "work_dir",
];

/// Read + DELETE the nudge file (consume-once), returning the parsed nudge if
/// present. A malformed file is consumed and reported as `Err` so it can't wedge
/// the loop. `Ok(None)` when no nudge is pending.
pub fn take_nudge(work_dir: &Path) -> anyhow::Result<Option<Nudge>> {
    let path = nudge_file(work_dir);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path);
    // Consume regardless of parse result — a bad nudge must not persist.
    let _ = std::fs::remove_file(&path);
    let text = text.map_err(|e| anyhow::anyhow!("nudge: reading {}: {e}", path.display()))?;
    let nudge: Nudge =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("nudge: parsing: {e}"))?;
    Ok(Some(nudge))
}

/// Write a nudge atomically (tmp→rename), mirroring the stop-file discipline.
pub fn write_nudge(work_dir: &Path, nudge: &Nudge) -> anyhow::Result<()> {
    std::fs::create_dir_all(work_dir).ok();
    let path = nudge_file(work_dir);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(nudge)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Merge the safe-live allowlist of a nudge into `cfg`, at absolute step
/// `ordinal` (for focus TTL). Returns what was applied/rejected. Restart-required
/// fields (in `rejected_probe`) are reported, never applied. Sticky vs expiring
/// is modeled per-field: weights are sticky; focus expires after `focus_steps`.
pub fn apply_nudge(cfg: &mut EvolveConfig, nudge: &Nudge, ordinal: u64) -> NudgeOutcome {
    let mut out = NudgeOutcome::default();

    // Goal weight (sticky).
    if let (Some(goal), Some(weight)) = (&nudge.goal, nudge.weight) {
        set_goal_weight(cfg, goal, weight as f32);
        out.applied.push(format!("goal[{goal}].weight={weight}"));
    }

    // Judge threshold (data-layer, sticky).
    if let Some(ms) = nudge.judge_min_score {
        let ms = ms.clamp(0.0, 1.0);
        let mut jc = cfg.judge.clone().unwrap_or_default();
        jc.min_score = ms;
        cfg.judge = Some(jc);
        out.applied.push(format!("judge.min_score={ms}"));
    }

    // All `[generate]` data-layer knobs merge through ONE clone-mutate-writeback
    // (candidates_per_seed / synthesis_rate / modality_mix), so multiple generate
    // nudges in one message compose instead of clobbering each other.
    if nudge.candidates_per_seed.is_some()
        || nudge.synthesis_rate.is_some()
        || !nudge.modality_mix.is_empty()
    {
        let mut g = cfg.generate.clone().unwrap_or_default();
        if let Some(n) = nudge.candidates_per_seed {
            g.candidates_per_seed = n;
            out.applied.push(format!("candidates_per_seed={n}"));
        }
        if let Some(r) = nudge.synthesis_rate {
            g.synthesis_rate = Some(r.clamp(0.0, 1.0));
            out.applied.push(format!("synthesis_rate={r}"));
        }
        if !nudge.modality_mix.is_empty() {
            g.kinds = nudge.modality_mix.clone();
            out.applied
                .push(format!("modality_mix={}", nudge.modality_mix.join(",")));
        }
        cfg.generate = Some(g);
    }

    // Gate mode (sticky) — merges into [regulate].gate.
    if let Some(gm) = &nudge.gate_mode {
        let mut r = cfg.regulate.clone().unwrap_or_default();
        r.gate = gm.clone();
        cfg.regulate = Some(r);
        out.applied.push(format!("gate_mode={gm}"));
    }

    // Focus (expiring): merge into the ephemeral steering field AND record the
    // absolute expiry ordinal (the daemon clears the field when `ordinal >= exp`).
    if let Some(focus) = &nudge.focus {
        let steps = nudge.focus_steps.unwrap_or(1).max(1);
        cfg.evolve.focus = Some(focus.clone());
        out.focus = Some((focus.clone(), ordinal + steps));
        out.applied.push(format!("focus={focus} for {steps} step(s)"));
    }

    // Restart-required probe → reject with reason.
    for field in &nudge.rejected_probe {
        if RESTART_REQUIRED.contains(&field.as_str()) {
            out.rejected
                .push(format!("{field}: restart-required (edit evolve.toml + restart)"));
        } else {
            out.rejected.push(format!("{field}: unknown nudge field"));
        }
    }

    out
}

/// Set a goal's weight in `[[goals]]` if the goal exists (sticky live merge).
/// Matches by `name`, `tag`, or `topic`.
fn set_goal_weight(cfg: &mut EvolveConfig, goal: &str, weight: f32) {
    for g in cfg.goals.iter_mut() {
        if g.name == goal || g.tag == goal || g.topic == goal {
            g.weight = Some(weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "scrt-evolve-nudge-{tag}-{:?}",
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn write_then_take_consumes_once() {
        let d = tmp("consume");
        let n = Nudge {
            judge_min_score: Some(0.7),
            ..Default::default()
        };
        write_nudge(&d, &n).unwrap();
        assert!(nudge_file(&d).exists());
        let got = take_nudge(&d).unwrap().expect("a nudge");
        assert_eq!(got.judge_min_score, Some(0.7));
        assert!(!nudge_file(&d).exists(), "consumed");
        assert!(take_nudge(&d).unwrap().is_none(), "gone on second poll");
    }

    #[test]
    fn apply_merges_allowlisted_fields() {
        let mut cfg = EvolveConfig::default();
        let n = Nudge {
            judge_min_score: Some(0.8),
            focus: Some("auth".into()),
            focus_steps: Some(3),
            ..Default::default()
        };
        let out = apply_nudge(&mut cfg, &n, 10);
        assert_eq!(cfg.judge.as_ref().unwrap().min_score, 0.8);
        assert_eq!(out.focus, Some(("auth".to_string(), 13)));
        // Focus must actually reach steering (not just be logged).
        assert_eq!(cfg.evolve.focus.as_deref(), Some("auth"));
        assert!(cfg
            .compose_steering()
            .unwrap()
            .contains("Focus (emphasize this right now)"));
        assert!(out.applied.iter().any(|a| a.contains("judge.min_score")));
        assert!(out.rejected.is_empty());
    }

    #[test]
    fn modality_mix_actually_replaces_generate_kinds() {
        // Regression guard: modality_mix must MUTATE config, not just log.
        let mut cfg = EvolveConfig::default();
        let n = Nudge {
            modality_mix: vec!["tool_call".into(), "cli".into()],
            candidates_per_seed: Some(8),
            synthesis_rate: Some(0.25),
            gate_mode: Some("judge".into()),
            ..Default::default()
        };
        let out = apply_nudge(&mut cfg, &n, 0);
        let g = cfg.generate.expect("generate block created");
        assert_eq!(g.kinds, vec!["tool_call".to_string(), "cli".to_string()]);
        assert_eq!(g.candidates_per_seed, 8);
        assert_eq!(g.synthesis_rate, Some(0.25));
        assert_eq!(cfg.regulate.unwrap().gate, "judge");
        assert!(out.applied.iter().any(|a| a.contains("modality_mix")));
    }

    #[test]
    fn restart_required_field_is_rejected() {
        let mut cfg = EvolveConfig::default();
        let n = Nudge {
            rejected_probe: vec!["model_path".into()],
            ..Default::default()
        };
        let out = apply_nudge(&mut cfg, &n, 0);
        assert_eq!(out.rejected.len(), 1);
        assert!(out.rejected[0].contains("restart-required"));
        assert!(out.applied.is_empty());
    }

    #[test]
    fn malformed_nudge_is_consumed_and_errors() {
        let d = tmp("bad");
        std::fs::write(nudge_file(&d), b"{not json").unwrap();
        assert!(take_nudge(&d).is_err());
        assert!(!nudge_file(&d).exists(), "bad nudge consumed, not wedged");
    }
}
