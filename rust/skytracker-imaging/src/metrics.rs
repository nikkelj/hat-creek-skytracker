//! Frame-quality metrics and target centroiding (ports of stacking.py's
//! sharpness / brightness_centroid numeric kernels).

use crate::filters::{laplacian, sobel_x, sobel_y};
use crate::image::ImageF32;

/// cv2.resize INTER_AREA for downscale: exact fractional box average.
pub fn resize_area(img: &ImageF32, nw: usize, nh: usize) -> ImageF32 {
    let (w, h) = (img.w, img.h);
    let sx = w as f64 / nw as f64;
    let sy = h as f64 / nh as f64;
    let mut out = ImageF32::new(nw, nh);
    for oy in 0..nh {
        let y0 = oy as f64 * sy;
        let y1 = y0 + sy;
        for ox in 0..nw {
            let x0 = ox as f64 * sx;
            let x1 = x0 + sx;
            let mut acc = 0.0f64;
            let mut area = 0.0f64;
            let iy0 = y0.floor() as usize;
            let iy1 = (y1.ceil() as usize).min(h);
            let ix0 = x0.floor() as usize;
            let ix1 = (x1.ceil() as usize).min(w);
            for yy in iy0..iy1 {
                let wy = (y1.min((yy + 1) as f64) - y0.max(yy as f64)).max(0.0);
                for xx in ix0..ix1 {
                    let wx = (x1.min((xx + 1) as f64) - x0.max(xx as f64)).max(0.0);
                    acc += img.at(yy, xx) as f64 * wy * wx;
                    area += wy * wx;
                }
            }
            *out.at_mut(oy, ox) = (acc / area) as f32;
        }
    }
    out
}

/// Variance of the Laplacian (stacking.sharpness method="laplacian").
pub fn sharpness_laplacian(gray: &ImageF32) -> f64 {
    let lap = laplacian(gray);
    let n = lap.data.len() as f64;
    let mean: f64 = lap.data.iter().map(|&v| v as f64).sum::<f64>() / n;
    lap.data
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

/// Mean squared Sobel gradient magnitude (method="tenengrad").
pub fn sharpness_tenengrad(gray: &ImageF32) -> f64 {
    let gx = sobel_x(gray);
    let gy = sobel_y(gray);
    let n = gx.data.len() as f64;
    gx.data
        .iter()
        .zip(gy.data.iter())
        .map(|(&x, &y)| (x as f64) * (x as f64) + (y as f64) * (y as f64))
        .sum::<f64>()
        / n
}

/// Intensity-weighted centroid of the bright target, or None. Threshold
/// defaults to mean + 2*std (stacking._default_threshold); weights are
/// clip(gray - threshold, 0).
pub fn brightness_centroid(gray: &ImageF32, threshold: Option<f64>) -> Option<(f64, f64)> {
    let n = gray.data.len() as f64;
    let threshold = threshold.unwrap_or_else(|| {
        let mean: f64 = gray.data.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = gray
            .data
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        mean + 2.0 * var.sqrt()
    });
    let mut total = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for y in 0..gray.h {
        for x in 0..gray.w {
            let wgt = (gray.at(y, x) as f64 - threshold).max(0.0);
            total += wgt;
            sx += x as f64 * wgt;
            sy += y as f64 * wgt;
        }
    }
    if total <= 0.0 {
        None
    } else {
        Some((sx / total, sy / total))
    }
}
