//! Finishing stage (port of sharpen.py): multi-scale unsharp mask +
//! deterministic auto-stretch on float [0, 1] images.

use crate::filters::{gaussian_kernel, sep_filter};
use crate::image::ImageF32;

/// cv2.GaussianBlur with ksize=(0,0): kernel size derived from sigma —
/// for float depths, cvRound(sigma * 4 * 2 + 1) | 1.
fn auto_ksize(sigma: f64) -> usize {
    let k = (sigma * 4.0 * 2.0 + 1.0).round() as usize;
    k | 1
}

/// Multi-scale unsharp: for each (sigma, amount), boost (img - blur(sigma))
/// by amount. Input/output float [0, 1].
pub fn unsharp_layers(img: &ImageF32, layers: &[(f64, f64)]) -> ImageF32 {
    let mut out = img.clone();
    for &(sigma, amount) in layers {
        if amount <= 0.0 || sigma <= 0.0 {
            continue;
        }
        let blurred = sep_filter(&out, &gaussian_kernel(auto_ksize(sigma), sigma));
        for (o, b) in out.data.iter_mut().zip(blurred.data.iter()) {
            *o += amount as f32 * (*o - b);
        }
    }
    for v in out.data.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    out
}

/// numpy percentile (linear interpolation) on a copy.
fn percentile(values: &[f32], pct: f64) -> f64 {
    let mut v: Vec<f32> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let idx = pct / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    v[lo] as f64 * (1.0 - frac) + v[hi.min(n - 1)] as f64 * frac
}

/// Deterministic display stretch (port of sharpen.auto_stretch, gray path):
/// percentile black/white points then a median-targeting gamma.
pub fn auto_stretch(
    img: &ImageF32,
    black_pct: f64,
    white_pct: f64,
    target_median: f64,
    max_gamma: f64,
) -> ImageF32 {
    let lo = percentile(&img.data, black_pct);
    let hi = percentile(&img.data, white_pct);
    let mut out = img.clone();
    if hi <= lo {
        return out;
    }
    let span = (hi - lo) as f32;
    for v in out.data.iter_mut() {
        *v = ((*v - lo as f32) / span).clamp(0.0, 1.0);
    }

    let fg: Vec<f32> = out.data.iter().copied().filter(|&v| v > 0.001).collect();
    if !fg.is_empty() {
        let med = percentile(&fg, 50.0);
        if med > 0.0 && med < target_median {
            let g = (target_median.ln() / med.ln()).max(1.0 / max_gamma);
            for v in out.data.iter_mut() {
                *v = v.powf(g as f32);
            }
        }
    }
    for v in out.data.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    out
}
