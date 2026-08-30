//! skytracker_core — pure-Rust engine core for the Hat Creek Skytracker.
//!
//! Byte-faithful port of the NexStar mount protocol (`protocol`) plus a
//! byte-level simulated mount (`sim`), the fixed-rate control loop
//! (`core_loop`), mode decision logic (`controller`), PID, hotspot detection,
//! and coordinate transforms. This crate is deliberately Python-free: the
//! PyO3 bindings live in the sibling `skytracker-ffi` crate, which is the
//! strangler seam and is deleted once the Rust app fully replaces Python.

pub mod autotune;
pub mod controller;
pub mod core_loop;
pub mod hotspot;
pub mod pid;
pub mod protocol;
pub mod rate;
#[cfg(feature = "serial")]
pub mod serial;
pub mod sim;
pub mod transforms;
