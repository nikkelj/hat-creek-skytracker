//! Shi-Tomasi corners (cv2.goodFeaturesToTrack) and pyramidal
//! Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK, winSize 21,
//! maxLevel 3) — the stabilizer's tracking primitives. Float-precision
//! port (OpenCV's 8-bit fixed-point interpolation differs at ~1e-3 px,
//! inside the 0.1 px gate).

use crate::filters::{filter3x3, sobel_x, sobel_y};
use crate::image::{reflect101, ImageF32};

/// cv2.pyrDown: 5-tap Gaussian [1,4,6,4,1]/16 separable smoothing
/// (BORDER_REFLECT_101) then take even rows/cols.
pub fn pyr_down(img: &ImageF32) -> ImageF32 {
    let k = [1.0f32 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (w, h) = (img.w, img.h);
    let mut tmp = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, &kv) in k.iter().enumerate() {
                acc += img.at(y, reflect101(x as isize + i as isize - 2, w)) * kv;
            }
            *tmp.at_mut(y, x) = acc;
        }
    }
    let mut sm = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, &kv) in k.iter().enumerate() {
                acc += tmp.at(reflect101(y as isize + i as isize - 2, h), x) * kv;
            }
            *sm.at_mut(y, x) = acc;
        }
    }
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = ImageF32::new(w2, h2);
    for y in 0..h2 {
        for x in 0..w2 {
            *out.at_mut(y, x) = sm.at((y * 2).min(h - 1), (x * 2).min(w - 1));
        }
    }
    out
}

/// cv2.goodFeaturesToTrack(maxCorners, qualityLevel, minDistance,
/// blockSize=3): min-eigenvalue corner response, quality threshold, 3x3
/// NMS, greedy min-distance selection sorted by response.
pub fn good_features_to_track(
    img: &ImageF32,
    max_corners: usize,
    quality_level: f32,
    min_distance: f32,
) -> Vec<[f32; 2]> {
    let (w, h) = (img.w, img.h);
    // cornerMinEigenVal scaling for 8-bit-range input, aperture 3, block 3:
    // scale = 1 / ((1 << 2) * 3 * 255).
    let scale = 1.0f32 / ((1 << 2) as f32 * 3.0 * 255.0);
    let dx0 = sobel_x(img);
    let dy0 = sobel_y(img);
    let dx: Vec<f32> = dx0.data.iter().map(|v| v * scale).collect();
    let dy: Vec<f32> = dy0.data.iter().map(|v| v * scale).collect();

    // Box-sum (unnormalized 3x3) of dx^2, dy^2, dx*dy, then min eigenvalue.
    let box3 = |src: &dyn Fn(usize) -> f32| -> Vec<f32> {
        let img_src = ImageF32::from_vec((0..w * h).map(src).collect(), w, h);
        let mut o = filter3x3(&img_src, &[[1.0; 3]; 3]);
        std::mem::take(&mut o.data)
    };
    let sxx = box3(&|i| dx[i] * dx[i]);
    let syy = box3(&|i| dy[i] * dy[i]);
    let sxy = box3(&|i| dx[i] * dy[i]);
    let mut eig = vec![0.0f32; w * h];
    for i in 0..w * h {
        let a = sxx[i] * 0.5;
        let c = syy[i] * 0.5;
        let b = sxy[i];
        eig[i] = (a + c) - ((a - c) * (a - c) + b * b).sqrt();
    }

    let max_eig = eig.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_eig * quality_level;

    // 3x3 non-max suppression, collect candidates above threshold.
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = eig[y * w + x];
            if v < threshold || v <= 0.0 {
                continue;
            }
            let mut is_max = true;
            'nms: for dy in -1..=1isize {
                for dx2 in -1..=1isize {
                    if dy == 0 && dx2 == 0 {
                        continue;
                    }
                    let yy = y as isize + dy;
                    let xx = x as isize + dx2;
                    if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                        continue;
                    }
                    if eig[yy as usize * w + xx as usize] > v {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                candidates.push((v, y, x));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Greedy min-distance selection.
    let min_d2 = min_distance * min_distance;
    let mut out: Vec<[f32; 2]> = Vec::new();
    for (_, y, x) in candidates {
        let p = [x as f32, y as f32];
        if out
            .iter()
            .all(|q| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) >= min_d2)
        {
            out.push(p);
            if out.len() >= max_corners {
                break;
            }
        }
    }
    out
}

fn bilinear(img: &ImageF32, x: f32, y: f32) -> f32 {
    crate::image::sample_bilinear_const(img, x, y, 0.0)
}

/// Pyramidal LK: track `points` ((x, y) like cv2) from `prev` to `next`.
/// Returns (tracked points, status). winSize square, iterations/eps match
/// cv2 defaults (30, 0.01).
pub fn calc_optical_flow_pyr_lk(
    prev: &ImageF32,
    next: &ImageF32,
    points: &[[f32; 2]],
    win_size: usize,
    max_level: usize,
) -> (Vec<[f32; 2]>, Vec<bool>) {
    // Build pyramids.
    let mut pyr_prev = vec![prev.clone()];
    let mut pyr_next = vec![next.clone()];
    for l in 0..max_level {
        pyr_prev.push(pyr_down(&pyr_prev[l]));
        pyr_next.push(pyr_down(&pyr_next[l]));
    }

    let half = (win_size / 2) as f32;
    let mut out = Vec::with_capacity(points.len());
    let mut status = Vec::with_capacity(points.len());

    for &p in points {
        let scale = (1 << max_level) as f32;
        let mut g = [0.0f32, 0.0f32]; // guess at current level
        let mut ok = true;

        for level in (0..=max_level).rev() {
            let lp = [p[0] / (1 << level) as f32, p[1] / (1 << level) as f32];
            let ip = &pyr_prev[level];
            let jp = &pyr_next[level];
            let _ = scale;

            // Spatial gradient over the window (Scharr, like OpenCV LK).
            let n = win_size * win_size;
            let mut ix = vec![0.0f32; n];
            let mut iy = vec![0.0f32; n];
            let mut iv = vec![0.0f32; n];
            let mut g00 = 0.0f64;
            let mut g01 = 0.0f64;
            let mut g11 = 0.0f64;
            for wy in 0..win_size {
                for wx in 0..win_size {
                    let sx = lp[0] - half + wx as f32;
                    let sy = lp[1] - half + wy as f32;
                    let v = bilinear(ip, sx, sy);
                    // Scharr derivative /32 like OpenCV.
                    let dx = ((bilinear(ip, sx + 1.0, sy - 1.0) - bilinear(ip, sx - 1.0, sy - 1.0)) * 3.0
                        + (bilinear(ip, sx + 1.0, sy) - bilinear(ip, sx - 1.0, sy)) * 10.0
                        + (bilinear(ip, sx + 1.0, sy + 1.0) - bilinear(ip, sx - 1.0, sy + 1.0)) * 3.0)
                        / 32.0;
                    let dy = ((bilinear(ip, sx - 1.0, sy + 1.0) - bilinear(ip, sx - 1.0, sy - 1.0)) * 3.0
                        + (bilinear(ip, sx, sy + 1.0) - bilinear(ip, sx, sy - 1.0)) * 10.0
                        + (bilinear(ip, sx + 1.0, sy + 1.0) - bilinear(ip, sx + 1.0, sy - 1.0)) * 3.0)
                        / 32.0;
                    let idx = wy * win_size + wx;
                    ix[idx] = dx;
                    iy[idx] = dy;
                    iv[idx] = v;
                    g00 += (dx * dx) as f64;
                    g01 += (dx * dy) as f64;
                    g11 += (dy * dy) as f64;
                }
            }
            let det = g00 * g11 - g01 * g01;
            let min_eig = 0.5 * ((g00 + g11) - ((g00 - g11).powi(2) + 4.0 * g01 * g01).sqrt())
                / n as f64;
            if det < 1e-12 || min_eig < 1e-4 {
                ok = false;
                break;
            }

            // Iterate the flow at this level.
            let mut nu = g;
            for _ in 0..30 {
                let mut b0 = 0.0f64;
                let mut b1 = 0.0f64;
                for wy in 0..win_size {
                    for wx in 0..win_size {
                        let idx = wy * win_size + wx;
                        let jx = lp[0] + nu[0] - half + wx as f32;
                        let jy = lp[1] + nu[1] - half + wy as f32;
                        let d = (bilinear(jp, jx, jy) - iv[idx]) as f64;
                        b0 += d * ix[idx] as f64;
                        b1 += d * iy[idx] as f64;
                    }
                }
                let dx = ((g11 * b0 - g01 * b1) / det) as f32;
                let dy = ((g00 * b1 - g01 * b0) / det) as f32;
                nu[0] -= dx;
                nu[1] -= dy;
                if (dx * dx + dy * dy).sqrt() < 0.01 {
                    break;
                }
            }
            g = nu;
            if level > 0 {
                g = [g[0] * 2.0, g[1] * 2.0];
            }
        }

        let tracked = [p[0] + g[0], p[1] + g[1]];
        let inside = tracked[0] >= 0.0
            && tracked[1] >= 0.0
            && tracked[0] < next.w as f32
            && tracked[1] < next.h as f32;
        out.push(tracked);
        status.push(ok && inside);
    }
    (out, status)
}
