//! skytracker-astro — the astro engine (Phase 1 of the Rust port).
//!
//! Replaces the skyfield/sgp4 Python stack: timescales + GAST (`time`),
//! IAU 2006/2000B precession-nutation (`frames`), SGP4 satellite passes in
//! the app's canonical 8-column format (`sgp4_pass`), DE421 solar-system
//! bodies (`ephemeris`, anise as the SPK reader), and Hipparcos star
//! apparent places (`stars`).
//!
//! Accuracy contract (validated against golden vectors recorded from
//! skyfield in ../../tests/golden/): satellites within 20 arcsec,
//! bodies/stars within 60 arcsec, GAST within 5 ms. The app closes the
//! remaining error optically (pointing model + closed-loop tracking).

pub mod apparent;
pub mod ephemeris;
pub mod frames;
mod generated_tables;
pub mod sgp4_pass;
pub mod spk;
pub mod stars;
pub mod time;
