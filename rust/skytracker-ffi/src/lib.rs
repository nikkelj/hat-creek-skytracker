//! skytracker-ffi — the PyO3 extension module.
//!
//! The Python import name remains `skytracker_core` (see pyproject.toml
//! `module-name` and the `#[pymodule]` fn in `bindings.rs`), so
//! `rust_loop_adapter.py` and every parity test import exactly as before.
//! All engine logic lives in the pure-Rust sibling crates; this crate is
//! glue only, and is deleted at the end of the Rust port.

#[cfg(feature = "extension-module")]
mod adsb_bindings;
#[cfg(feature = "extension-module")]
mod astro_bindings;
#[cfg(feature = "extension-module")]
mod bindings;
#[cfg(feature = "extension-module")]
mod platesolve_bindings;
#[cfg(feature = "extension-module")]
mod imaging_bindings;
#[cfg(feature = "extension-module")]
mod pointing_bindings;
