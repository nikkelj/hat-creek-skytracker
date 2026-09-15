//! Flow-method stabilization estimate (port of stabilizer.Stabilizer's
//! "flow" path): good-features on the reference, pyramidal LK into the
//! current frame, RANSAC similarity fit, and the plausibility validation
//! bounds. The visual warp itself stays with the caller.

use crate::features::{calc_optical_flow_pyr_lk, good_features_to_track};
use crate::image::ImageF32;
use crate::ransac::{estimate_affine_partial_2d, Similarity};

pub struct StabilizeParams {
    pub max_features: usize,
    pub ransac_threshold: f64,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub scale_tol: f64,
    pub max_rotation_deg: f64,
    pub max_translation_frac: f64,
}

impl Default for StabilizeParams {
    fn default() -> Self {
        StabilizeParams {
            max_features: 600,
            ransac_threshold: 3.0,
            min_inliers: 8,
            min_inlier_ratio: 0.4,
            scale_tol: 0.12,
            max_rotation_deg: 20.0,
            max_translation_frac: 0.6,
        }
    }
}

pub struct FlowEstimate {
    /// 2x3 affine mapping current -> reference, None when rejected.
    pub m: Option<[[f64; 3]; 2]>,
    pub num_inliers: usize,
    pub reject_reason: Option<String>,
}

/// Detect reference features (call once per anchor frame).
pub fn detect_reference_points(ref_gray: &ImageF32, max_features: usize) -> Vec<[f32; 2]> {
    good_features_to_track(ref_gray, max_features, 0.01, 8.0)
}

fn validate(m: &Similarity, w: usize, h: usize, p: &StabilizeParams) -> Option<String> {
    // det of the linear part: a^2 + b^2 for a similarity (always >= 0), but
    // keep the check for parity with the Python bounds.
    let det = m.a * m.a + m.b * m.b;
    if det <= 0.0 {
        return Some("reflection (det<=0)".into());
    }
    let scale = m.scale();
    if !(1.0 - p.scale_tol..=1.0 + p.scale_tol).contains(&scale) {
        return Some(format!("scale {scale:.3}"));
    }
    let rot = m.rotation_deg().abs();
    if rot > p.max_rotation_deg {
        return Some(format!("rotation {rot:.1}deg"));
    }
    if m.tx.abs() > p.max_translation_frac * w as f64
        || m.ty.abs() > p.max_translation_frac * h as f64
    {
        return Some(format!("translation ({:.0},{:.0})", m.tx, m.ty));
    }
    None
}

/// Estimate the frame -> reference similarity from tracked flow.
pub fn estimate_flow(
    ref_gray: &ImageF32,
    ref_points: &[[f32; 2]],
    cur_gray: &ImageF32,
    params: &StabilizeParams,
) -> FlowEstimate {
    if ref_points.len() < 3 {
        return FlowEstimate {
            m: None,
            num_inliers: 0,
            reject_reason: Some("too few reference points".into()),
        };
    }
    let (tracked, status) = calc_optical_flow_pyr_lk(ref_gray, cur_gray, ref_points, 21, 3);
    let mut src = Vec::new(); // points in the current frame
    let mut dst = Vec::new(); // where they sit in the reference
    for i in 0..ref_points.len() {
        if status[i] {
            src.push([tracked[i][0] as f64, tracked[i][1] as f64]);
            dst.push([ref_points[i][0] as f64, ref_points[i][1] as f64]);
        }
    }
    if src.len() < 3 {
        return FlowEstimate {
            m: None,
            num_inliers: 0,
            reject_reason: Some("too few tracked points".into()),
        };
    }
    let Some(res) = estimate_affine_partial_2d(&src, &dst, params.ransac_threshold, 2000, 42)
    else {
        return FlowEstimate {
            m: None,
            num_inliers: 0,
            reject_reason: Some("ransac failed".into()),
        };
    };
    let n = res.inliers.iter().filter(|&&k| k).count();
    let total = src.len();
    if n < params.min_inliers || (total > 0 && (n as f64 / total as f64) < params.min_inlier_ratio)
    {
        return FlowEstimate {
            m: None,
            num_inliers: n,
            reject_reason: Some(format!("inliers {n}/{total}")),
        };
    }
    if let Some(reason) = validate(&res.model, cur_gray.w, cur_gray.h, params) {
        return FlowEstimate {
            m: None,
            num_inliers: n,
            reject_reason: Some(reason),
        };
    }
    let m = res.model;
    FlowEstimate {
        m: Some([[m.a, -m.b, m.tx], [m.b, m.a, m.ty]]),
        num_inliers: n,
        reject_reason: None,
    }
}
