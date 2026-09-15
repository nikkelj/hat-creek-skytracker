//! Feature/template tracker for extended targets at close range — grab an
//! operator-selected patch of the vehicle pre-launch and follow that PATCH
//! through ascent. The hotspot centroid tracker is the wrong tool here: a
//! close rocket is a large structured object and the exhaust plume outshines
//! it at ignition, so a brightest-blob tracker slides down onto the plume.
//! This one matches the grabbed template by zero-mean normalized cross
//! correlation (coarse-to-fine, small scale search), so it stays on the
//! airframe feature the operator picked.

/// Sub-pixel match result.
#[derive(Clone, Copy, Debug)]
pub struct FeatureMatch {
    pub cx: f64,
    pub cy: f64,
    /// ZNCC peak score in [-1, 1]; 1 = perfect match.
    pub score: f64,
}

pub struct FeatureTracker {
    template: Vec<f32>,
    half: usize,
    pub cx: f64,
    pub cy: f64,
    /// Accumulated scale relative to the grab (for display).
    pub scale: f64,
}

fn crop(img: &[f32], w: usize, h: usize, cx: f64, cy: f64, half: usize) -> Option<Vec<f32>> {
    let side = 2 * half + 1;
    let x0 = cx.round() as isize - half as isize;
    let y0 = cy.round() as isize - half as isize;
    if x0 < 0 || y0 < 0 || x0 + side as isize > w as isize || y0 + side as isize > h as isize {
        return None;
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let mut out = Vec::with_capacity(side * side);
    for y in 0..side {
        let row = (y0 + y) * w + x0;
        out.extend_from_slice(&img[row..row + side]);
    }
    Some(out)
}

/// Bilinear resample of a square patch to the same side length under `scale`
/// (scale > 1 zooms in: samples a smaller source region).
fn resample(patch: &[f32], side: usize, scale: f64) -> Vec<f32> {
    let c = (side as f64 - 1.0) / 2.0;
    let mut out = vec![0.0f32; side * side];
    for y in 0..side {
        for x in 0..side {
            let sx = c + (x as f64 - c) / scale;
            let sy = c + (y as f64 - c) / scale;
            let (ix, iy) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - ix, sy - iy);
            let (ix, iy) = (ix as isize, iy as isize);
            let get = |xx: isize, yy: isize| -> f64 {
                let xx = xx.clamp(0, side as isize - 1) as usize;
                let yy = yy.clamp(0, side as isize - 1) as usize;
                patch[yy * side + xx] as f64
            };
            let v = get(ix, iy) * (1.0 - fx) * (1.0 - fy)
                + get(ix + 1, iy) * fx * (1.0 - fy)
                + get(ix, iy + 1) * (1.0 - fx) * fy
                + get(ix + 1, iy + 1) * fx * fy;
            out[y * side + x] = v as f32;
        }
    }
    out
}

/// Zero-mean NCC of `tpl` against the same-size patch of `img` at (x0, y0)
/// (top-left). Returns -1 when the patch leaves the frame or is flat.
fn zncc_at(img: &[f32], w: usize, h: usize, tpl: &[f32], side: usize, x0: isize, y0: isize) -> f64 {
    if x0 < 0 || y0 < 0 || x0 as usize + side > w || y0 as usize + side > h {
        return -1.0;
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let n = (side * side) as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for y in 0..side {
        let row = (y0 + y) * w + x0;
        let trow = y * side;
        for x in 0..side {
            let a = img[row + x] as f64;
            let b = tpl[trow + x] as f64;
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    if va <= 1e-9 || vb <= 1e-9 {
        return -1.0;
    }
    cov / (va * vb).sqrt()
}

/// Downsample an image by `k` (box mean) — the coarse pyramid level.
fn downsample(img: &[f32], w: usize, h: usize, k: usize) -> (Vec<f32>, usize, usize) {
    let (dw, dh) = (w / k, h / k);
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let mut s = 0.0f32;
            for dy in 0..k {
                let row = (y * k + dy) * w + x * k;
                for dx in 0..k {
                    s += img[row + dx];
                }
            }
            out[y * dw + x] = s / (k * k) as f32;
        }
    }
    (out, dw, dh)
}

impl FeatureTracker {
    /// Grab a (2·half+1)² template centered on (cx, cy). None when the box
    /// leaves the frame or the patch has no texture to track.
    pub fn init(img: &[f32], w: usize, h: usize, cx: f64, cy: f64, half: usize) -> Option<Self> {
        let half = half.clamp(12, 96);
        let template = crop(img, w, h, cx, cy, half)?;
        // Reject textureless grabs (flat sky): they match everywhere.
        let n = template.len() as f64;
        let mean = template.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = template.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        if var < 4.0 {
            return None;
        }
        Some(FeatureTracker { template, half, cx, cy, scale: 1.0 })
    }

    pub fn half(&self) -> usize {
        self.half
    }

    /// Track one frame: coarse pass on a 4× pyramid over ±search_r, fine ZNCC
    /// refine with a small scale search (close targets shrink as they climb),
    /// parabolic sub-pixel peak. `min_score` gates the match (~0.45).
    pub fn track(&mut self, img: &[f32], w: usize, h: usize, search_r: usize, min_score: f64) -> Option<FeatureMatch> {
        let side = 2 * self.half + 1;
        const K: usize = 4;
        // Coarse: downsampled template over a downsampled search window.
        let (dimg, dw, dh) = downsample(img, w, h, K);
        let dtpl = {
            let (t, _, _) = downsample(&self.template, side, side, K);
            t
        };
        let dside = side / K;
        if dside < 4 || dw < dside || dh < dside {
            return None;
        }
        let dr = (search_r / K).max(2) as isize;
        let (pcx, pcy) = ((self.cx / K as f64).round() as isize, (self.cy / K as f64).round() as isize);
        let mut best = (-2.0f64, 0isize, 0isize);
        for dy in -dr..=dr {
            for dx in -dr..=dr {
                let x0 = pcx + dx - dside as isize / 2;
                let y0 = pcy + dy - dside as isize / 2;
                let s = zncc_at(&dimg, dw, dh, &dtpl, dside, x0, y0);
                if s > best.0 {
                    best = (s, dx, dy);
                }
            }
        }
        if best.0 < -1.5 {
            return None;
        }
        let (gx, gy) = (self.cx + (best.1 * K as isize) as f64, self.cy + (best.2 * K as isize) as f64);

        // Fine: full-res ZNCC around the coarse hit, over a few scales.
        let mut fine = (-2.0f64, 0isize, 0isize, 1.0f64);
        let mut score_grid = [[-1.0f64; 11]; 11]; // for sub-pixel, scale-best only
        for &sc in &[0.95f64, 1.0, 1.05] {
            let tpl = if (sc - 1.0).abs() < 1e-9 { self.template.clone() } else { resample(&self.template, side, sc) };
            let mut local = [[-1.0f64; 11]; 11];
            let mut lbest = (-2.0f64, 0isize, 0isize);
            for dy in -5isize..=5 {
                for dx in -5isize..=5 {
                    let x0 = gx.round() as isize + dx - self.half as isize;
                    let y0 = gy.round() as isize + dy - self.half as isize;
                    let s = zncc_at(img, w, h, &tpl, side, x0, y0);
                    local[(dy + 5) as usize][(dx + 5) as usize] = s;
                    if s > lbest.0 {
                        lbest = (s, dx, dy);
                    }
                }
            }
            if lbest.0 > fine.0 {
                fine = (lbest.0, lbest.1, lbest.2, sc);
                score_grid = local;
            }
        }
        if fine.0 < min_score {
            return None;
        }
        // Parabolic sub-pixel on the winning scale's 3×3 neighborhood.
        let (bi, bj) = ((fine.2 + 5) as usize, (fine.1 + 5) as usize);
        let sub = |m1: f64, c: f64, p1: f64| -> f64 {
            let d = m1 - 2.0 * c + p1;
            if d.abs() < 1e-12 { 0.0 } else { (0.5 * (m1 - p1) / d).clamp(-0.5, 0.5) }
        };
        let (mut sx, mut sy) = (0.0, 0.0);
        if bj > 0 && bj < 10 {
            sx = sub(score_grid[bi][bj - 1], score_grid[bi][bj], score_grid[bi][bj + 1]);
        }
        if bi > 0 && bi < 10 {
            sy = sub(score_grid[bi - 1][bj], score_grid[bi][bj], score_grid[bi + 1][bj]);
        }
        let cx = gx + fine.1 as f64 + sx;
        let cy = gy + fine.2 as f64 + sy;
        self.cx = cx;
        self.cy = cy;

        // Scale adaptation: a shrinking (receding) target re-bakes the
        // template at the winning scale so drift stays continuous.
        if (fine.3 - 1.0).abs() > 1e-9 {
            self.template = resample(&self.template, side, fine.3);
            self.scale *= fine.3;
        }
        // Slow template update on strong matches only: absorbs lighting/roll
        // drift without the classic template-drift walk-off.
        if fine.0 > 0.85 {
            if let Some(patch) = crop(img, w, h, cx, cy, self.half) {
                const ALPHA: f32 = 0.06;
                for (t, p) in self.template.iter_mut().zip(patch.iter()) {
                    *t = (1.0 - ALPHA) * *t + ALPHA * *p;
                }
            }
        }
        Some(FeatureMatch { cx, cy, score: fine.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize, ox: f64, oy: f64) -> Vec<f32> {
        // Smooth multi-frequency texture translated by (ox, oy) — enough
        // structure for ZNCC, no pathological periodicity.
        let mut img = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64 - ox, y as f64 - oy);
                img[y * w + x] = (60.0
                    + 40.0 * (fx * 0.11).sin() * (fy * 0.07).cos()
                    + 25.0 * (fx * 0.031 + fy * 0.023).sin()
                    + 12.0 * (fx * 0.19).cos()) as f32;
            }
        }
        img
    }

    #[test]
    fn tracks_subpixel_translation() {
        let (w, h) = (200, 160);
        let img0 = textured(w, h, 0.0, 0.0);
        let mut tr = FeatureTracker::init(&img0, w, h, 100.0, 80.0, 24).unwrap();
        let img1 = textured(w, h, 7.4, -3.6); // scene shifts by (+7.4, -3.6)
        let m = tr.track(&img1, w, h, 32, 0.45).unwrap();
        assert!(m.score > 0.9, "score {}", m.score);
        assert!((m.cx - 107.4).abs() < 0.35 && (m.cy - 76.4).abs() < 0.35, "({:.2},{:.2})", m.cx, m.cy);
    }

    #[test]
    fn rejects_textureless_grab_and_bad_match() {
        let (w, h) = (128, 128);
        let flat = vec![50.0f32; w * h];
        assert!(FeatureTracker::init(&flat, w, h, 64.0, 64.0, 24).is_none());
        let img0 = textured(w, h, 0.0, 0.0);
        let mut tr = FeatureTracker::init(&img0, w, h, 64.0, 64.0, 20).unwrap();
        // A completely different scene must not fake a lock.
        let noise: Vec<f32> = (0..w * h).map(|i| ((i * 2654435761usize) % 255) as f32).collect();
        assert!(tr.track(&noise, w, h, 24, 0.45).is_none());
    }
}
