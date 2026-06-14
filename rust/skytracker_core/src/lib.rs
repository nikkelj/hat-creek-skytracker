//! skytracker_core — Rust core for the Hat Creek Skytracker.
//!
//! Step 1 of the Rust-core / Python-shell experiment: a byte-faithful port of
//! the NexStar mount protocol (`protocol`) plus a byte-level simulated mount
//! (`sim`) so the full encode -> transact -> parse loop can be exercised with no
//! hardware. The `extension-module` feature adds the PyO3 bindings.

pub mod protocol;
pub mod sim;

#[cfg(feature = "extension-module")]
mod python_bindings;
