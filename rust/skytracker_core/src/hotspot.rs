//! Hot-spot detection + pixel->angle geometry — port of `hotspot.py`.
//!
//! Pure functions over a 2D intensity map (`ndarray::ArrayView2<f64>`), so they
//! unit-test with synthetic frames and carry no Python/numpy dependency. The
//! PyO3 layer (in `python_bindings.rs`) adapts a numpy frame to this core.
//!
//! Matches the Python detection model: the target is the brightest compact
//! source in the search region. Validated against `test_hotspot.py` (mirrored in
//! `tests/hotspot.rs`) and `test_rust_hotspot_parity.py`.

use ndarray::ArrayView2;

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub cx: f64,
    pub cy: f64,
    pub peak: f64,
    pub background: f64,
    pub noise: f64,
    pub snr: f64,
    pub n_pixels: usize,
}

/// Median (numpy convention: mean of the two central values for even length),
/// computed in O(n) via selection and IN PLACE (the buffer is reordered). We
/// own the scratch buffers at the call sites, so reordering is fine — and this
/// avoids the O(n log n) full sort that made the naive port slower than numpy.
fn median_inplace(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        v[mid]
    } else {
        // After selection, v[..mid] all <= v[mid]; the (mid-1)-th order
        // statistic is the max of that lower partition.
        let hi = v[mid];
        let lo = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

/// Population standard deviation (numpy default ddof=0).
fn std_pop(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

/// 3x3 box blur with edge replication, matching `_smooth3` (np.pad mode="edge"
/// then 3x3 sum / 9). Edge replication == clamping the sample indices.
fn smooth3(region: &ArrayView2<f32>) -> Vec<f64> {
    let h = region.nrows() as isize;
    let w = region.ncols() as isize;
    let clamp = |i: isize, n: isize| -> usize { i.max(0).min(n - 1) as usize };
    let mut out = vec![0.0_f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    s += region[[clamp(y + dy, h), clamp(x + dx, w)]] as f64;
                }
            }
            out[(y * w + x) as usize] = s / 9.0;
        }
    }
    out
}

/// Detect the brightest compact hot spot. Mirrors `hotspot.detect_hotspot`.
///
/// `gate`: optional (cx, cy, radius) in full-image pixels to restrict the search.
#[allow(clippy::too_many_arguments)]
pub fn detect_hotspot(
    img: &ArrayView2<f32>,
    gate: Option<(f64, f64, f64)>,
    snr_threshold: f64,
    centroid_radius: usize,
    half_max_fraction: f64,
    min_pixels: usize,
) -> Option<Detection> {
    let h = img.nrows();
    let w = img.ncols();
    if h == 0 || w == 0 {
        return None;
    }

    // Tracking gate window (full frame otherwise).
    let (x0, y0, x1, y1) = match gate {
        Some((gx, gy, gr)) => {
            let x0 = ((gx - gr) as isize).max(0) as usize;
            let x1 = (((gx + gr) as isize) + 1).min(w as isize).max(0) as usize;
            let y0 = ((gy - gr) as isize).max(0) as usize;
            let y1 = (((gy + gr) as isize) + 1).min(h as isize).max(0) as usize;
            (x0, y0, x1, y1)
        }
        None => (0, 0, w, h),
    };
    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    let region = img.slice(ndarray::s![y0..y1, x0..x1]);
    let rh = region.nrows();
    let rw = region.ncols();
    // Scratch buffer (f32 -> f64). Reordered in place by the median; we don't
    // need its order afterwards (abs-dev and std are order-independent).
    let mut flat: Vec<f64> = region.iter().map(|&v| v as f64).collect();
    if flat.is_empty() {
        return None;
    }

    // Robust background / noise (median / MAD) over the region. O(n) selection.
    let background = median_inplace(&mut flat);
    let mut abs_dev: Vec<f64> = flat.iter().map(|x| (x - background).abs()).collect();
    let mad = median_inplace(&mut abs_dev);
    let noise = if mad > 0.0 {
        1.4826 * mad
    } else {
        let s = std_pop(&flat);
        if s != 0.0 {
            s
        } else {
            1.0
        }
    };

    // Peak on the smoothed copy (first max in C order), value taken from region.
    let smooth = smooth3(&region);
    let mut arg = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &v) in smooth.iter().enumerate() {
        if v > best {
            best = v;
            arg = i;
        }
    }
    let ry = arg / rw;
    let rx = arg % rw;
    let peak = region[[ry, rx]] as f64;

    let snr = (peak - background) / noise;
    if snr < snr_threshold {
        return None;
    }

    // Intensity-weighted centroid in a window around the peak.
    let cr = centroid_radius as isize;
    let wy0 = ((ry as isize) - cr).max(0) as usize;
    let wy1 = (((ry as isize) + cr + 1).min(rh as isize)) as usize;
    let wx0 = ((rx as isize) - cr).max(0) as usize;
    let wx1 = (((rx as isize) + cr + 1).min(rw as isize)) as usize;

    let thresh = background + half_max_fraction * (peak - background);

    let mut n = 0usize;
    let mut wsum = 0.0_f64;
    let mut sum_xw = 0.0_f64;
    let mut sum_yw = 0.0_f64;
    for yy in wy0..wy1 {
        for xx in wx0..wx1 {
            let val = region[[yy, xx]] as f64;
            if val >= thresh {
                n += 1;
                let weight = val - background;
                wsum += weight;
                // local coords within the window (xx - wx0, yy - wy0)
                sum_xw += (xx - wx0) as f64 * weight;
                sum_yw += (yy - wy0) as f64 * weight;
            }
        }
    }

    if n < min_pixels || wsum <= 0.0 {
        return None;
    }

    let cx = sum_xw / wsum + wx0 as f64 + x0 as f64;
    let cy = sum_yw / wsum + wy0 as f64 + y0 as f64;

    Some(Detection {
        cx,
        cy,
        peak,
        background,
        noise,
        snr,
        n_pixels: n,
    })
}

/// Convert a pixel offset from boresight into the mount-coordinate angular
/// correction (az_error, el_error) in degrees. Mirrors
/// `hotspot.pixel_offset_to_angles`.
#[allow(clippy::too_many_arguments)]
pub fn pixel_offset_to_angles(
    dx_pix: f64,
    dy_pix: f64,
    pixel_size_um: f64,
    focal_length_mm: f64,
    rotation_deg: f64,
    el_deg: f64,
    x_sign: f64,
    y_sign: f64,
    apply_cos_el: bool,
) -> (f64, f64) {
    if focal_length_mm == 0.0 {
        return (0.0, 0.0);
    }
    let ifov_deg = ((pixel_size_um * 1e-3) / focal_length_mm).to_degrees();

    let ax = dx_pix * ifov_deg * x_sign;
    let ay = dy_pix * ifov_deg * y_sign;

    let th = rotation_deg.to_radians();
    let (cos_t, sin_t) = (th.cos(), th.sin());
    let cross_el = ax * cos_t - ay * sin_t;
    let el_err = ax * sin_t + ay * cos_t;

    let az_err = if apply_cos_el {
        let c = el_deg.to_radians().cos();
        if c.abs() > 1e-3 {
            cross_el / c
        } else {
            cross_el
        }
    } else {
        cross_el
    };

    (az_err, el_err)
}
