//! skytracker-pointing — TPOINT-style pointing models (Phase 2b of the
//! Rust port): 7-term alt-az and equatorial fits with partial (seeded)
//! and MAD-robust modes, plus the polar-axis plane fit. Ports
//! pointing_model.py / eq_pointing_model.py / polar_align.py math with
//! coefficient parity at 1e-6 (see test_rust_pointing_parity.py).

pub mod altaz;
pub mod eq;
pub mod fit;
pub mod lstsq;
pub mod polar;
