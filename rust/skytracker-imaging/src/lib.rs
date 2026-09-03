//! skytracker-imaging — the imaging engine (Phase 3 of the Rust port).
//!
//! Ports the OpenCV numeric cluster the stacking / stabilization /
//! sharpening pipeline is built from: parity-tested filters and warps,
//! phase correlation with OpenCV's subpixel peak, Shi-Tomasi + pyramidal
//! Lucas-Kanade tracking, and RANSAC similarity estimation. Validated
//! against cv2 golden vectors (tests/golden/cv2_ops.npz); the composed
//! pipelines gate on end-to-end quality (PSNR/transform accuracy), not
//! bit parity.

pub mod enhance;
pub mod features;
pub mod filters;
pub mod gridshift;
pub mod image;
pub mod metrics;
pub mod phasecorr;
pub mod ransac;
pub mod stabilize;
#[cfg(feature = "mp4-export")]
pub mod video;
pub mod warp;
