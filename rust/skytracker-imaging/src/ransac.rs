//! cv2.estimateAffinePartial2D equivalent: RANSAC over 2-point similarity
//! hypotheses, then exact least squares on the inlier set (the similarity
//! model is linear, so LSQ is the same optimum cv2's LM refinement finds).
//! RNG differs from cv2's, but on real data both converge to the same
//! inlier set; the gate is the recovered transform (0.1 px / 0.05 deg).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 2x3 similarity [[a, -b, tx], [b, a, ty]].
#[derive(Clone, Copy, Debug)]
pub struct Similarity {
    pub a: f64,
    pub b: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Similarity {
    pub fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        [
            self.a * p[0] - self.b * p[1] + self.tx,
            self.b * p[0] + self.a * p[1] + self.ty,
        ]
    }

    pub fn rotation_deg(&self) -> f64 {
        self.b.atan2(self.a).to_degrees()
    }

    pub fn scale(&self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }
}

fn from_two_pairs(src: &[[f64; 2]; 2], dst: &[[f64; 2]; 2]) -> Option<Similarity> {
    let dx = src[1][0] - src[0][0];
    let dy = src[1][1] - src[0][1];
    let ex = dst[1][0] - dst[0][0];
    let ey = dst[1][1] - dst[0][1];
    let d2 = dx * dx + dy * dy;
    if d2 < 1e-12 {
        return None;
    }
    let a = (dx * ex + dy * ey) / d2;
    let b = (dx * ey - dy * ex) / d2;
    let tx = dst[0][0] - (a * src[0][0] - b * src[0][1]);
    let ty = dst[0][1] - (b * src[0][0] + a * src[0][1]);
    Some(Similarity { a, b, tx, ty })
}

/// Exact LSQ similarity over point pairs (normal equations; the model is
/// linear in (a, b, tx, ty)).
pub fn fit_similarity_lsq(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Similarity> {
    let n = src.len() as f64;
    if src.len() < 2 {
        return None;
    }
    let (mut sx, mut sy, mut px, mut py) = (0.0, 0.0, 0.0, 0.0);
    let (mut sxx, mut sxp, mut syp, mut sxq, mut syq) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (s, d) in src.iter().zip(dst.iter()) {
        sx += s[0];
        sy += s[1];
        px += d[0];
        py += d[1];
        sxx += s[0] * s[0] + s[1] * s[1];
        sxp += s[0] * d[0];
        syp += s[1] * d[1];
        sxq += s[0] * d[1];
        syq += s[1] * d[0];
    }
    // Solve the 4x4 normal system (closed form via centered coordinates).
    let mx = sx / n;
    let my = sy / n;
    let ux = px / n;
    let uy = py / n;
    let var = sxx / n - (mx * mx + my * my);
    if var < 1e-12 {
        return None;
    }
    let cov_a = (sxp + syp) / n - (mx * ux + my * uy);
    let cov_b = (sxq - syq) / n - (mx * uy - my * ux);
    let a = cov_a / var;
    let b = cov_b / var;
    Some(Similarity {
        a,
        b,
        tx: ux - (a * mx - b * my),
        ty: uy - (b * mx + a * my),
    })
}

pub struct RansacResult {
    pub model: Similarity,
    pub inliers: Vec<bool>,
}

/// RANSAC similarity estimation (threshold in the same pixel units as the
/// points; cv2 default 2000 iterations).
pub fn estimate_affine_partial_2d(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    threshold: f64,
    max_iters: usize,
    seed: u64,
) -> Option<RansacResult> {
    let n = src.len();
    if n < 2 {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let t2 = threshold * threshold;
    let mut best: Option<(usize, Similarity)> = None;

    for _ in 0..max_iters {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n);
        while j == i {
            j = rng.gen_range(0..n);
        }
        let Some(model) = from_two_pairs(&[src[i], src[j]], &[dst[i], dst[j]]) else {
            continue;
        };
        let count = src
            .iter()
            .zip(dst.iter())
            .filter(|(s, d)| {
                let p = model.apply(**s);
                (p[0] - d[0]).powi(2) + (p[1] - d[1]).powi(2) < t2
            })
            .count();
        if best.as_ref().map_or(true, |(bc, _)| count > *bc) {
            best = Some((count, model));
        }
    }

    let (_, model) = best?;
    let inliers: Vec<bool> = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| {
            let p = model.apply(*s);
            (p[0] - d[0]).powi(2) + (p[1] - d[1]).powi(2) < t2
        })
        .collect();

    // Refine with exact LSQ over the inliers.
    let in_src: Vec<[f64; 2]> = src
        .iter()
        .zip(inliers.iter())
        .filter(|(_, &k)| k)
        .map(|(s, _)| *s)
        .collect();
    let in_dst: Vec<[f64; 2]> = dst
        .iter()
        .zip(inliers.iter())
        .filter(|(_, &k)| k)
        .map(|(d, _)| *d)
        .collect();
    let refined = fit_similarity_lsq(&in_src, &in_dst).unwrap_or(model);
    Some(RansacResult {
        model: refined,
        inliers,
    })
}
