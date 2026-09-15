//! Port of tetra3.Tetra3.solve_from_centroids for the app's parameter
//! space: known scalar distortion (the app passes 0), fov_estimate given,
//! edge_ratio 4-star patterns. Structure and iteration order mirror the
//! Python line-by-line so the FIRST accepted solution is the same one.

use crate::db::SolverDatabase;
use crate::hash::{key_to_index, probe_indices};
use nalgebra::{Matrix3, Vector3};

pub struct SolveParams {
    pub fov_estimate_deg: f64,
    pub fov_max_error_deg: Option<f64>,
    pub pattern_checking_stars: usize,
    pub match_radius: f64,
    pub match_threshold: f64,
    /// Known distortion k (the app uses 0.0).
    pub distortion: f64,
}

pub struct Solution {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    pub fov_deg: f64,
    pub distortion: f64,
    pub rmse_arcsec: f64,
    pub matches: usize,
    pub prob: f64,
    /// (y, x) image centroids of the matched stars, brightest first.
    pub matched_centroids: Vec<[f64; 2]>,
}

/// `_compute_vectors`: pinhole unit vectors from (y, x) centroids.
/// Centroids pass through f32 like the Python (np.float32 cast).
fn compute_vectors(centroids: &[[f64; 2]], h: f64, w: f64, fov: f64) -> Vec<Vector3<f64>> {
    let scale = (fov / 2.0).tan() / w * 2.0;
    centroids
        .iter()
        .map(|c| {
            let y = c[0] as f32 as f64;
            let x = c[1] as f32 as f64;
            let v = Vector3::new(1.0, (w / 2.0 - x) * scale, (h / 2.0 - y) * scale);
            v / v.norm()
        })
        .collect()
}

/// `_compute_centroids`: derotated unit vectors -> (y, x) centroids;
/// returns only those inside the frame, with their source indices.
fn compute_centroids(
    vectors: &[Vector3<f64>],
    h: f64,
    w: f64,
    fov: f64,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let scale = -w / 2.0 / (fov / 2.0).tan();
    let mut cents = Vec::new();
    let mut kept = Vec::new();
    for (i, v) in vectors.iter().enumerate() {
        let y = scale * v[2] / v[0] + h / 2.0;
        let x = scale * v[1] / v[0] + w / 2.0;
        if y > 0.0 && x > 0.0 && y < h && x < w {
            cents.push([y, x]);
            kept.push(i);
        }
    }
    (cents, kept)
}

/// `_undistort_centroids` with scalar k (identity when k = 0).
fn undistort(centroids: &[[f64; 2]], h: f64, w: f64, k: f64) -> Vec<[f64; 2]> {
    centroids
        .iter()
        .map(|c| {
            let y = c[0] as f32 as f64 - h / 2.0;
            let x = c[1] as f32 as f64 - w / 2.0;
            let r = (y * y + x * x).sqrt() / w * 2.0;
            let scale = (1.0 - k * r * r) / (1.0 - k);
            [y * scale + h / 2.0, x * scale + w / 2.0]
        })
        .collect()
}

/// Angles between all pairs in scipy `pdist` order: (0,1),(0,2),...,(2,3).
fn pairwise_angles(vectors: &[Vector3<f64>]) -> Vec<f64> {
    let n = vectors.len();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let d = (vectors[i] - vectors[j]).norm();
            out.push(2.0 * (0.5 * d).asin());
        }
    }
    out
}

/// Kabsch as tetra3 does it: H = img^T . cat, R = U . V^T (no reflection fix).
fn find_rotation_matrix(image: &[Vector3<f64>], catalog: &[Vector3<f64>]) -> Matrix3<f64> {
    let mut h = Matrix3::zeros();
    for (a, b) in image.iter().zip(catalog.iter()) {
        h += a * b.transpose();
    }
    let svd = h.svd(true, true);
    svd.u.unwrap() * svd.v_t.unwrap()
}

/// Sort vectors by distance from their mean (tetra3's unique ordering).
fn sort_by_radius(vectors: &mut Vec<Vector3<f64>>) {
    let mean = vectors.iter().sum::<Vector3<f64>>() / vectors.len() as f64;
    let mut order: Vec<usize> = (0..vectors.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = (vectors[a] - mean).norm();
        let rb = (vectors[b] - mean).norm();
        ra.partial_cmp(&rb).unwrap()
    });
    *vectors = order.iter().map(|&i| vectors[i]).collect();
}

/// Binomial CDF P(X <= k) for small n (exact sum).
fn binom_cdf(k: f64, n: usize, p: f64) -> f64 {
    if k < 0.0 {
        return 0.0;
    }
    let k = k.floor() as usize;
    if k >= n {
        return 1.0;
    }
    let q = 1.0 - p;
    let mut sum = 0.0;
    let mut coef = 1.0f64;
    for i in 0..=k {
        if i > 0 {
            coef = coef * (n - i + 1) as f64 / i as f64;
        }
        sum += coef * p.powi(i as i32) * q.powi((n - i) as i32);
    }
    sum.min(1.0)
}

/// `_get_nearby_stars`: cube prefilter then dot-product cone test.
fn nearby_stars(db: &SolverDatabase, center: &Vector3<f64>, radius: f64) -> Vec<usize> {
    let max_dist = 2.0 * (radius / 2.0).sin();
    let cos_r = radius.cos();
    let mut out = Vec::new();
    for (i, row) in db.star_table.iter().enumerate() {
        let (x, y, z) = (row[2] as f64, row[3] as f64, row[4] as f64);
        if (x - center[0]).abs() < max_dist
            && (y - center[1]).abs() < max_dist
            && (z - center[2]).abs() < max_dist
            && center[0] * x + center[1] * y + center[2] * z > cos_r
        {
            out.push(i);
        }
    }
    out
}

/// `_find_centroid_matches` incl. numpy's unique-by-first-occurrence
/// semantics (final result ordered by ascending catalog index).
fn find_centroid_matches(
    image: &[[f64; 2]],
    catalog: &[[f64; 2]],
    r: f64,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, ic) in image.iter().enumerate() {
        for (j, cc) in catalog.iter().enumerate() {
            let dy = ic[0] - cc[0];
            let dx = ic[1] - cc[1];
            if (dy * dy + dx * dx).sqrt() < r {
                pairs.push((i, j));
            }
        }
    }
    // unique by image index (first occurrence, result sorted by i — already
    // row-major so first occurrence per ascending i is in order).
    let mut seen_i = std::collections::BTreeMap::new();
    for &(i, j) in &pairs {
        seen_i.entry(i).or_insert((i, j));
    }
    // unique by catalog index, result ordered by ascending j.
    let mut seen_j = std::collections::BTreeMap::new();
    for &(i, j) in seen_i.values() {
        seen_j.entry(j).or_insert((i, j));
    }
    seen_j.values().copied().collect()
}

/// Lexicographic k-combinations of 0..n (itertools.combinations order).
struct Combinations {
    n: usize,
    k: usize,
    idx: Vec<usize>,
    done: bool,
}

impl Combinations {
    fn new(n: usize, k: usize) -> Self {
        Combinations {
            n,
            k,
            idx: (0..k).collect(),
            done: k > n,
        }
    }
}

impl Iterator for Combinations {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        let out = self.idx.clone();
        // Advance.
        let mut i = self.k;
        loop {
            if i == 0 {
                self.done = true;
                break;
            }
            i -= 1;
            if self.idx[i] != i + self.n - self.k {
                self.idx[i] += 1;
                for j in i + 1..self.k {
                    self.idx[j] = self.idx[j - 1] + 1;
                }
                break;
            }
        }
        Some(out)
    }
}

pub fn solve_from_centroids(
    db: &SolverDatabase,
    star_centroids: &[[f64; 2]],
    height: usize,
    width: usize,
    params: &SolveParams,
) -> Option<Solution> {
    let (h, w) = (height as f64, width as f64);
    let fov_estimate = params.fov_estimate_deg.to_radians();
    let fov_initial = fov_estimate;
    let fov_max_error = params.fov_max_error_deg.map(|e| e.to_radians());
    let num_patterns = db.pattern_catalog.len() / 2;
    let match_threshold = params.match_threshold / num_patterns as f64;

    let p_size = db.props.pattern_size; // 4
    let p_bins = db.props.pattern_bins;
    let p_max_err = db.props.pattern_max_error;
    let num_stars = db.props.verification_stars_per_fov;

    let image_centroids: Vec<[f64; 2]> = star_centroids
        .iter()
        .take(num_stars)
        .copied()
        .collect();
    // Known distortion: undistort once up front (identity for k = 0).
    let image_centroids = undistort(&image_centroids, h, w, params.distortion);

    let n_check = image_centroids.len().min(params.pattern_checking_stars);
    let pattlen = p_size * (p_size - 1) / 2 - 1; // 5

    for pattern_indices in Combinations::new(n_check, p_size) {
        let pattern_centroids: Vec<[f64; 2]> =
            pattern_indices.iter().map(|&i| image_centroids[i]).collect();

        // Edge ratios at the initial FOV, +/- tolerance.
        let vectors = compute_vectors(&pattern_centroids, h, w, fov_initial);
        let mut angles = pairwise_angles(&vectors);
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let image_largest_edge = angles[angles.len() - 1];
        let ratios: Vec<f64> = angles[..pattlen]
            .iter()
            .map(|&a| a / image_largest_edge)
            .collect();
        let ratio_min: Vec<f64> = ratios.iter().map(|r| r - p_max_err).collect();
        let ratio_max: Vec<f64> = ratios.iter().map(|r| r + p_max_err).collect();

        // Hash-code ranges -> all combinations, row-sorted, unique
        // (np.unique sorts lexicographically; BTreeSet reproduces that).
        let lows: Vec<i64> = ratio_min
            .iter()
            .map(|&v| (v * p_bins as f64).max(0.0) as i64)
            .collect();
        let highs: Vec<i64> = ratio_max
            .iter()
            .map(|&v| ((v * p_bins as f64).min(p_bins as f64)) as i64)
            .collect();
        let mut codes: std::collections::BTreeSet<Vec<u64>> = std::collections::BTreeSet::new();
        let mut stack = vec![Vec::with_capacity(pattlen)];
        while let Some(prefix) = stack.pop() {
            let d = prefix.len();
            if d == pattlen {
                let mut sorted = prefix.clone();
                sorted.sort_unstable();
                codes.insert(sorted);
                continue;
            }
            for v in lows[d]..=highs[d] {
                let mut next = prefix.clone();
                next.push(v as u64);
                stack.push(next);
            }
        }

        for code in &codes {
            let hash_index = key_to_index(code, p_bins, db.pattern_catalog.len() as u64);
            let match_inds = probe_indices(hash_index, &db.pattern_catalog);
            if match_inds.is_empty() {
                continue;
            }

            for &row in &match_inds {
                let cat_rows = db.pattern_catalog[row];
                let cat_vectors: Vec<Vector3<f64>> = cat_rows
                    .iter()
                    .map(|&si| {
                        let s = &db.star_table[si as usize];
                        Vector3::new(s[2] as f64, s[3] as f64, s[4] as f64)
                    })
                    .collect();
                let mut cat_angles = pairwise_angles(&cat_vectors);
                cat_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cat_largest = cat_angles[cat_angles.len() - 1];
                let cat_ratios: Vec<f64> = cat_angles[..pattlen]
                    .iter()
                    .map(|&a| a / cat_largest)
                    .collect();
                if !cat_ratios
                    .iter()
                    .zip(ratio_min.iter().zip(ratio_max.iter()))
                    .all(|(r, (lo, hi))| r > lo && r < hi)
                {
                    continue;
                }

                // Coarse FOV by scaling the estimate.
                let fov = cat_largest / image_largest_edge * fov_initial;
                if let Some(max_err) = fov_max_error {
                    if (fov - fov_estimate).abs() > max_err {
                        continue;
                    }
                }

                // Image pattern vectors at the coarse FOV, radius-sorted.
                let mut image_pattern_vectors =
                    compute_vectors(&pattern_centroids, h, w, fov);
                sort_by_radius(&mut image_pattern_vectors);
                let mut catalog_pattern_vectors = cat_vectors.clone();
                if !db.props.presort_patterns {
                    sort_by_radius(&mut catalog_pattern_vectors);
                }

                let rotation =
                    find_rotation_matrix(&image_pattern_vectors, &catalog_pattern_vectors);

                // Nearby catalog stars inside the diagonal FOV.
                let center = Vector3::new(rotation[(0, 0)], rotation[(0, 1)], rotation[(0, 2)]);
                let fov_diag = fov * (w * w + h * h).sqrt() / w;
                let nearby = nearby_stars(db, &center, fov_diag / 2.0);
                let nearby_vectors: Vec<Vector3<f64>> = nearby
                    .iter()
                    .map(|&i| {
                        let s = &db.star_table[i];
                        Vector3::new(s[2] as f64, s[3] as f64, s[4] as f64)
                    })
                    .collect();
                let derot: Vec<Vector3<f64>> =
                    nearby_vectors.iter().map(|v| rotation * v).collect();
                let (mut cat_centroids, kept) = compute_centroids(&derot, h, w, fov);
                let mut kept_vectors: Vec<Vector3<f64>> =
                    kept.iter().map(|&i| nearby_vectors[i]).collect();
                cat_centroids.truncate(image_centroids.len());
                kept_vectors.truncate(image_centroids.len());

                let matched = find_centroid_matches(
                    &image_centroids,
                    &cat_centroids,
                    w * params.match_radius,
                );
                let n_extracted = image_centroids.len();
                let n_nearby = cat_centroids.len();
                let n_matches = matched.len();

                let prob_single = n_nearby as f64 * params.match_radius * params.match_radius;
                let prob_mismatch = binom_cdf(
                    n_extracted as f64 - (n_matches as f64 - 2.0),
                    n_extracted,
                    1.0 - prob_single,
                );

                if prob_mismatch >= match_threshold {
                    continue;
                }

                // Accepted: refine rotation on all matches.
                let matched_image_centroids: Vec<[f64; 2]> =
                    matched.iter().map(|&(i, _)| image_centroids[i]).collect();
                let matched_image_vectors =
                    compute_vectors(&matched_image_centroids, h, w, fov);
                let matched_cat_vectors: Vec<Vector3<f64>> =
                    matched.iter().map(|&(_, j)| kept_vectors[j]).collect();
                let rotation =
                    find_rotation_matrix(&matched_image_vectors, &matched_cat_vectors);

                let ra = rotation[(0, 1)]
                    .atan2(rotation[(0, 0)])
                    .to_degrees()
                    .rem_euclid(360.0);
                let dec = rotation[(0, 2)]
                    .atan2(
                        (rotation[(1, 2)] * rotation[(1, 2)]
                            + rotation[(2, 2)] * rotation[(2, 2)])
                            .sqrt(),
                    )
                    .to_degrees();
                let roll = rotation[(1, 2)]
                    .atan2(rotation[(2, 2)])
                    .to_degrees()
                    .rem_euclid(360.0);

                // Known-distortion branch: fit focal length + distortion by
                // least squares over the matches (tetra3's `else` branch).
                let derot_cat: Vec<Vector3<f64>> =
                    matched_cat_vectors.iter().map(|v| rotation * v).collect();
                let tangent: Vec<f64> = derot_cat
                    .iter()
                    .map(|v| (v[1] * v[1] + v[2] * v[2]).sqrt() / v[0])
                    .collect();
                let radius: Vec<f64> = matched_image_centroids
                    .iter()
                    .map(|c| {
                        let dy = c[0] - h / 2.0;
                        let dx = c[1] - w / 2.0;
                        (dy * dy + dx * dx).sqrt() / w * 2.0
                    })
                    .collect();
                // lstsq for [f, k]: A = [tangent, radius^3], b = radius.
                let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for (t, r) in tangent.iter().zip(radius.iter()) {
                    let r3 = r * r * r;
                    a11 += t * t;
                    a12 += t * r3;
                    a22 += r3 * r3;
                    b1 += t * r;
                    b2 += r3 * r;
                }
                let det = a11 * a22 - a12 * a12;
                let (f, k) = if det.abs() > 1e-20 {
                    ((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det)
                } else {
                    (b1 / a11, 0.0)
                };
                let f = f / (1.0 - k);
                let fov_final = 2.0 * (1.0 / f).atan();
                let matched_undist = undistort(&matched_image_centroids, h, w, k);

                let final_vectors = compute_vectors(&matched_undist, h, w, fov_final);
                let mut sq_sum = 0.0;
                for (v, c) in final_vectors.iter().zip(matched_cat_vectors.iter()) {
                    let d = (rotation.transpose() * v - c).norm();
                    let ang = 2.0 * (0.5 * d).asin();
                    sq_sum += ang * ang;
                }
                let rmse = (sq_sum / n_matches as f64).sqrt().to_degrees() * 3600.0;

                return Some(Solution {
                    ra_deg: ra,
                    dec_deg: dec,
                    roll_deg: roll,
                    fov_deg: fov_final.to_degrees(),
                    distortion: k,
                    rmse_arcsec: rmse,
                    matches: n_matches,
                    prob: prob_mismatch * num_patterns as f64,
                    matched_centroids: matched_image_centroids,
                });
            }
        }
    }
    None
}
