//! skytracker-platesolve — pure-Rust port of the ESA tetra3 lost-in-space
//! plate solver (Phase 2 of the Rust port).
//!
//! Reads the app's existing tetra3 .npz pattern databases unchanged; the
//! pattern hash is bit-exact against Python tetra3 (golden-tested on the
//! live db_cam1_tyc geometry), centroids match within 0.3 px, and
//! solutions match within 30 arcsec / 0.1 deg roll on identical images.

pub mod centroid;
pub mod db;
pub mod hash;
pub mod npy;
pub mod solve;
