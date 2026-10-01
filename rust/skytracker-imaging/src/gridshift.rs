//! Alignment-point local shifts + dense grid warp (ports of
//! stacking.measure_local_shifts / warp_by_grid).

use crate::image::ImageF32;
use crate::phasecorr::{hanning_window, phase_correlate};
use crate::warp::{remap, Border};

/// Per-point sub-pixel shift via patchwise phase correlation. `points` are
/// (x, y); returns (dx, dy) rows. Zero shift where the response is below
/// `min_response` or the shift exceeds `max_shift`.
pub fn measure_local_shifts(
    ref_gray: &ImageF32,
    cur_gray: &ImageF32,
    points: &[[f64; 2]],
    patch: usize,
    min_response: f64,
    max_shift: Option<f64>,
) -> Vec<[f64; 2]> {
    let (w, h) = (ref_gray.w, ref_gray.h);
    let half = patch / 2;
    let side = 2 * half;
    let mut shifts = vec![[0.0f64; 2]; points.len()];
    if side > w || side > h {
        return shifts;
    }
    let win = hanning_window(side, side);

    let cut = |img: &ImageF32, x0: usize, y0: usize| -> ImageF32 {
        let mut out = ImageF32::new(side, side);
        for y in 0..side {
            for x in 0..side {
                *out.at_mut(y, x) = img.at(y0 + y, x0 + x);
            }
        }
        out
    };

    for (i, p) in points.iter().enumerate() {
        let x0 = ((p[0].round() as isize - half as isize).max(0) as usize).min(w - side);
        let y0 = ((p[1].round() as isize - half as isize).max(0) as usize).min(h - side);
        let a = cut(ref_gray, x0, y0);
        let b = cut(cur_gray, x0, y0);
        let (dx, dy, response) = phase_correlate(&a, &b, Some(&win));
        if response < min_response {
            continue;
        }
        if let Some(ms) = max_shift {
            if dx.abs() > ms || dy.abs() > ms {
                continue;
            }
        }
        shifts[i] = [dx, dy];
    }
    shifts
}

/// cv2.resize INTER_LINEAR (pixel-centre aligned: src = (dst+0.5)*s - 0.5,
/// clamped) — used to upsample the coarse shift grid.
pub fn resize_linear(img: &ImageF32, nw: usize, nh: usize) -> ImageF32 {
    let sx = img.w as f32 / nw as f32;
    let sy = img.h as f32 / nh as f32;
    let mut out = ImageF32::new(nw, nh);
    for oy in 0..nh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0).min(img.h as f32 - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(img.h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..nw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0).min(img.w as f32 - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(img.w - 1);
            let wx = fx - x0 as f32;
            *out.at_mut(oy, ox) = img.at(y0, x0) * (1.0 - wx) * (1.0 - wy)
                + img.at(y0, x1) * wx * (1.0 - wy)
                + img.at(y1, x0) * (1.0 - wx) * wy
                + img.at(y1, x1) * wx * wy;
        }
    }
    out
}

/// Remap `frame` by the dense displacement field interpolated from the
/// (rows x cols) shift grid: output (x, y) samples input (x + dx, y + dy).
pub fn warp_by_grid(
    frame: &ImageF32,
    rows: usize,
    cols: usize,
    shifts: &[[f64; 2]],
    border: Border,
) -> ImageF32 {
    let (w, h) = (frame.w, frame.h);
    let dx_grid = ImageF32::from_vec(shifts.iter().map(|s| s[0] as f32).collect(), cols, rows);
    let dy_grid = ImageF32::from_vec(shifts.iter().map(|s| s[1] as f32).collect(), cols, rows);
    let dx_full = resize_linear(&dx_grid, w, h);
    let dy_full = resize_linear(&dy_grid, w, h);
    let mut map_x = ImageF32::new(w, h);
    let mut map_y = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            *map_x.at_mut(y, x) = x as f32 + dx_full.at(y, x);
            *map_y.at_mut(y, x) = y as f32 + dy_full.at(y, x);
        }
    }
    remap(frame, &map_x, &map_y, border)
}
