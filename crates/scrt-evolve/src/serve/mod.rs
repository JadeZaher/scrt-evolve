//! Native inference engine (track 39). Currently a stub owned by Lane 1;
//! Lane 4 has landed the SSD scan under `ops/` only.
//!
//! When Lane 1 arrives it will replace this file with the real serve module
//! (loader/engine/etc.) and must keep `pub mod ops;` present.

#[cfg(feature = "train")]
pub mod ops;

#[cfg(feature = "train")]
pub mod loader;
