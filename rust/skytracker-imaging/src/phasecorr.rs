//! cv2.phaseCorrelate parity: Hanning window, cross-power spectrum,
//! fftShift, peak, then OpenCV's 5x5 weighted-centroid subpixel estimate.
//! Gate: 0.05 px vs the cv2 goldens.

use crate::image::ImageF32;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// cv2.createHanningWindow((w, h), CV_32F).
pub fn hanning_window(w: usize, h: usize) -> ImageF32 {
    let mut out = ImageF32::new(w, h);
    let tau = std::f64::consts::TAU;
    for y in 0..h {
        let wy = 0.5 * (1.0 - (tau * y as f64 / (h - 1) as f64).cos());
        for x in 0..w {
            let wx = 0.5 * (1.0 - (tau * x as f64 / (w - 1) as f64).cos());
            *out.at_mut(y, x) = (wy * wx) as f32;
        }
    }
    out
}

fn fft2(data: &mut [Complex<f64>], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::new();
    let fft_row = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    for y in 0..h {
        fft_row.process(&mut data[y * w..(y + 1) * w]);
    }
    let fft_col = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col = vec![Complex::default(); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = data[y * w + x];
        }
        fft_col.process(&mut col);
        for y in 0..h {
            data[y * w + x] = col[y];
        }
    }
}

/// (shift_x, shift_y, response): the translation of `moved` relative to
/// `base` and the peak response, matching cv2.phaseCorrelate(base, moved,
/// window). cv2 returns the shift such that shifting `base` by it aligns
/// with `moved` (positive = moved content displaced by +shift).
pub fn phase_correlate(base: &ImageF32, moved: &ImageF32, window: Option<&ImageF32>) -> (f64, f64, f64) {
    let (w, h) = (base.w, base.h);
    let n = w * h;
    let apply = |img: &ImageF32| -> Vec<Complex<f64>> {
        (0..n)
            .map(|i| {
                let v = img.data[i] as f64
                    * window.map_or(1.0, |win| win.data[i] as f64);
                Complex::new(v, 0.0)
            })
            .collect()
    };
    let mut fa = apply(base);
    let mut fb = apply(moved);
    fft2(&mut fa, w, h, false);
    fft2(&mut fb, w, h, false);

    // Cross-power spectrum: FFT1 . conj(FFT2) / |...| (OpenCV divides by
    // magnitude; zero-magnitude bins pass through as zero).
    let mut cross: Vec<Complex<f64>> = fa
        .iter()
        .zip(fb.iter())
        .map(|(a, b)| {
            let p = a * b.conj();
            let mag = p.norm();
            if mag > 0.0 {
                p / mag
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();
    fft2(&mut cross, w, h, true);
    // rustfft inverse is unnormalized; scale by 1/n.
    let corr: Vec<f64> = cross.iter().map(|c| c.re / n as f64).collect();

    // Materialize the fftShifted surface, then scan row-major with strict
    // greater-than — exactly cv2's fftShift + minMaxLoc, so half-pixel
    // peak ties resolve to the same cell cv2 picks.
    let mut surface = vec![0.0f64; w * h];
    for y in 0..h {
        let sy = (y + h / 2) % h;
        for x in 0..w {
            let sx = (x + w / 2) % w;
            surface[y * w + x] = corr[sy * w + sx];
        }
    }
    let mut peak = (0usize, 0usize);
    let mut peak_v = f64::MIN;
    for y in 0..h {
        for x in 0..w {
            let v = surface[y * w + x];
            if v > peak_v {
                peak_v = v;
                peak = (y, x);
            }
        }
    }

    // OpenCV weightedCentroid: 5x5 region centred on the peak, clamped to
    // the image; weights = raw correlation values; response = their sum.
    let (py, px) = (peak.0 as isize, peak.1 as isize);
    let mut sum = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for dy in -2..=2isize {
        for dx in -2..=2isize {
            let yy = py + dy;
            let xx = px + dx;
            if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                continue;
            }
            let v = surface[yy as usize * w + xx as usize];
            sum += v;
            sum_x += xx as f64 * v;
            sum_y += yy as f64 * v;
        }
    }
    let (cx, cy) = if sum != 0.0 {
        (sum_x / sum, sum_y / sum)
    } else {
        (px as f64, py as f64)
    };

    let shift_x = (w / 2) as f64 - cx;
    let shift_y = (h / 2) as f64 - cy;
    (shift_x, shift_y, sum)
}
