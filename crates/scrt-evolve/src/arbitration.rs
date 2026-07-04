//! VRAM arbitration + commit-swap signal contract (track 33 Phase 0). The shared
//! vocabulary every serve-while-you-train phase compiles against. Design +
//! schema: see `conductor/tracks/33-concurrent-inference-during-training/AGENTS.md`.

use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Runtime residency decision: co-resident serve+train (mode B) or strict
/// alternation (mode A degrade path). Chosen by [`select_mode`] from measured
/// footprints; persisted with its reason so `doctor`/`watch` can report it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum Mode {
    /// Serve + train share the GPU. The three summed footprints fit under the
    /// (reservation-adjusted) ceiling.
    Coresident {
        /// Served GGUF footprint (GB) at the configured `n_gpu_layers`.
        serve_gb: f64,
        /// Peak VRAM (GB) of one training block/microshard.
        block_gb: f64,
        /// CUDA context overhead (GB).
        ctx_gb: f64,
    },
    /// Strict alternation (mode A). The co-resident set overflowed the ceiling;
    /// `reason` names the overflow amount in GB.
    Alternate {
        /// Human-readable reason naming the overflow (GB).
        reason: String,
    },
}

/// Decide co-residence vs alternation from measured footprints. PURE.
///
/// `reservation_gb` (when `Some`) is the trainer's serve carve-out: the block's
/// usable ceiling becomes `ceiling_gb - reservation`, so the block must fit in
/// that reduced headroom alongside the served model + CUDA context. `None` ⇒ no
/// carve-out (the full `ceiling_gb` is usable). Returns [`Mode::Coresident`] iff
/// `serve + block + ctx <= usable_ceiling`, else [`Mode::Alternate`].
pub fn select_mode(
    serve_footprint_gb: f64,
    block_peak_gb: f64,
    cuda_ctx_gb: f64,
    ceiling_gb: f64,
    reservation_gb: Option<f64>,
) -> Mode {
    let usable_ceiling = ceiling_gb - reservation_gb.unwrap_or(0.0);
    let needed = serve_footprint_gb + block_peak_gb + cuda_ctx_gb;
    if needed <= usable_ceiling {
        Mode::Coresident {
            serve_gb: serve_footprint_gb,
            block_gb: block_peak_gb,
            ctx_gb: cuda_ctx_gb,
        }
    } else {
        let overflow = needed - usable_ceiling;
        let reason = match reservation_gb {
            Some(r) => format!(
                "co-resident footprint {needed:.2} GB exceeds usable ceiling {usable_ceiling:.2} GB \
                 (ceiling {ceiling_gb:.2} GB − reservation {r:.2} GB) by {overflow:.2} GB; \
                 degrading to alternate (mode A)"
            ),
            None => format!(
                "co-resident footprint {needed:.2} GB exceeds ceiling {ceiling_gb:.2} GB \
                 by {overflow:.2} GB; degrading to alternate (mode A)"
            ),
        };
        Mode::Alternate { reason }
    }
}

/// The commit-swap signal record: emitted on a `keep` commit, consumed by the
/// live server to hot-swap the served adapter. One JSON object per line in the
/// append-only `<state>/served-ready.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServedReady {
    /// Monotonic committed-adapter version.
    pub version: u64,
    /// Path to the merged flat adapter to serve.
    pub adapter_path: String,
    /// Path to the frozen base the adapter overlays.
    pub base_path: String,
    /// RFC3339-ish commit timestamp.
    pub timestamp: String,
}

/// Append one [`ServedReady`] as a JSON line to the append-only signal file.
pub fn append_served_ready(path: &Path, rec: &ServedReady) -> io::Result<()> {
    let mut line = serde_json::to_string(rec)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(line.as_bytes())
}

/// Read all [`ServedReady`] records from the signal file. Missing file ⇒ empty
/// vec; blank lines are skipped.
pub fn read_served_ready(path: &Path) -> io::Result<Vec<ServedReady>> {
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: ServedReady = serde_json::from_str(&line)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        out.push(rec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coresident_when_exactly_at_ceiling() {
        // 3.5 + 3.3 + 1.2 = 8.0, ceiling 8.0, no reservation ⇒ fits (<=).
        let m = select_mode(3.5, 3.3, 1.2, 8.0, None);
        assert_eq!(
            m,
            Mode::Coresident {
                serve_gb: 3.5,
                block_gb: 3.3,
                ctx_gb: 1.2,
            }
        );
    }

    #[test]
    fn alternate_when_overflow_by_epsilon() {
        let m = select_mode(3.5, 3.3, 1.2001, 8.0, None);
        match m {
            Mode::Alternate { reason } => {
                assert!(reason.contains("exceeds ceiling"), "reason: {reason}");
                assert!(reason.contains("mode A"), "reason: {reason}");
            }
            other => panic!("expected Alternate, got {other:?}"),
        }
    }

    #[test]
    fn reservation_shrinks_headroom_forcing_alternate() {
        // Without reservation this fits (7.0 <= 8.0); a 2.0 GB reservation drops
        // the usable ceiling to 6.0, so 7.0 now overflows.
        let fits = select_mode(3.0, 3.0, 1.0, 8.0, None);
        assert!(matches!(fits, Mode::Coresident { .. }));

        let m = select_mode(3.0, 3.0, 1.0, 8.0, Some(2.0));
        match m {
            Mode::Alternate { reason } => {
                assert!(reason.contains("reservation"), "reason: {reason}");
                assert!(reason.contains("usable ceiling"), "reason: {reason}");
            }
            other => panic!("expected Alternate, got {other:?}"),
        }
    }

    #[test]
    fn reservation_that_still_fits_stays_coresident() {
        // usable = 8.0 - 1.0 = 7.0; needed = 6.9 ⇒ fits.
        let m = select_mode(3.4, 2.5, 1.0, 8.0, Some(1.0));
        assert!(matches!(m, Mode::Coresident { .. }));
    }

    #[test]
    fn none_reservation_uses_full_ceiling() {
        let m = select_mode(2.0, 2.0, 1.0, 8.0, None);
        assert_eq!(
            m,
            Mode::Coresident {
                serve_gb: 2.0,
                block_gb: 2.0,
                ctx_gb: 1.0,
            }
        );
    }

    #[test]
    fn served_ready_append_read_roundtrip() {
        let dir = std::env::temp_dir().join(format!("scrt-arb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("served-ready.jsonl");
        let _ = std::fs::remove_file(&path);

        let r1 = ServedReady {
            version: 1,
            adapter_path: "work/adapter".into(),
            base_path: "base".into(),
            timestamp: "2026-07-03T00:00:00Z".into(),
        };
        let r2 = ServedReady {
            version: 2,
            adapter_path: "work/adapter".into(),
            base_path: "base".into(),
            timestamp: "2026-07-03T00:01:00Z".into(),
        };
        append_served_ready(&path, &r1).unwrap();
        append_served_ready(&path, &r2).unwrap();

        let read = read_served_ready(&path).unwrap();
        assert_eq!(read, vec![r1, r2]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_missing_file_is_empty() {
        let path = std::env::temp_dir().join("scrt-arb-does-not-exist-xyz.jsonl");
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_served_ready(&path).unwrap(), Vec::new());
    }
}
