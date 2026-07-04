//! Custom candle ops for arch coverage (track 39, Phase B).
//! See `AGENTS.md` in this directory for the math + design.

pub mod ssd;

pub use ssd::{ssd_scan, ScanError};
