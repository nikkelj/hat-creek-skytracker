//! Shared 7-term least-squares fit machinery: full/partial (seeded) solve,
//! MAD-robust outlier rejection, and the stats block — the exact structure
//! of pointing_model.PointingModel.fit / EquatorialPointingModel.fit.
//!
//! A "sample" here is already reduced to (c1_cmd, c2_cmd, d1, d2): the
//! commanded coordinates and the raw observed-minus-commanded errors
//! (refraction already stripped by the alt-az caller). Design rows come
//! from the model-specific closure; the great-circle weighting cos(c2)
//! and the wrap-180 on the first residual are common to both models.

use crate::lstsq::lstsq;
use nalgebra::{DMatrix, DVector};

pub const N_TERMS: usize = 7;

pub fn wrap180(deg: f64) -> f64 {
    (deg + 180.0).rem_euclid(360.0) - 180.0
}

#[derive(Clone, Debug)]
pub struct FitStats {
    pub terms: [f64; N_TERMS],
    pub n_samples: usize,
    pub n_rejected: usize,
    pub design_cond: f64,
    pub rms_before_deg: f64,
    pub rms_after_deg: f64,
}

pub struct FitOptions {
    /// Indices of free terms for a partial fit; None = fit all seven.
    pub free_idx: Option<Vec<usize>>,
    /// Seed values for held-fixed terms (partial fit).
    pub seed: [f64; N_TERMS],
    pub robust: bool,
    pub robust_sigma: f64,
    pub robust_floor_deg: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            free_idx: None,
            seed: [0.0; N_TERMS],
            robust: false,
            robust_sigma: 4.0,
            robust_floor_deg: 30.0 / 3600.0,
        }
    }
}

/// A prepared sample: commanded coords and raw errors, plus the two design
/// rows evaluated at the commanded position.
#[derive(Clone)]
pub struct Sample {
    pub c1_cmd: f64,
    pub c2_cmd: f64,
    pub d1: f64,
    pub d2: f64,
    pub row1: [f64; N_TERMS],
    pub row2: [f64; N_TERMS],
}

fn solve(samples: &[Sample], opts: &FitOptions) -> ([f64; N_TERMS], f64) {
    let m = samples.len() * 2;
    match &opts.free_idx {
        None => {
            let mut a = DMatrix::zeros(m, N_TERMS);
            let mut b = DVector::zeros(m);
            for (i, s) in samples.iter().enumerate() {
                for j in 0..N_TERMS {
                    a[(2 * i, j)] = s.row1[j];
                    a[(2 * i + 1, j)] = s.row2[j];
                }
                b[2 * i] = s.d1;
                b[2 * i + 1] = s.d2;
            }
            let r = lstsq(&a, &b);
            let mut terms = [0.0; N_TERMS];
            terms.copy_from_slice(&r.x);
            (terms, r.cond)
        }
        Some(free) => {
            // Partial: solve the free columns against the residual the
            // fixed (seeded) terms don't explain.
            let is_free: Vec<bool> = (0..N_TERMS).map(|i| free.contains(&i)).collect();
            let p_fixed: Vec<f64> = (0..N_TERMS)
                .map(|i| if is_free[i] { 0.0 } else { opts.seed[i] })
                .collect();
            let mut a = DMatrix::zeros(m, free.len());
            let mut b = DVector::zeros(m);
            for (i, s) in samples.iter().enumerate() {
                let fixed1: f64 = (0..N_TERMS).map(|j| s.row1[j] * p_fixed[j]).sum();
                let fixed2: f64 = (0..N_TERMS).map(|j| s.row2[j] * p_fixed[j]).sum();
                for (jj, &j) in free.iter().enumerate() {
                    a[(2 * i, jj)] = s.row1[j];
                    a[(2 * i + 1, jj)] = s.row2[j];
                }
                b[2 * i] = s.d1 - fixed1;
                b[2 * i + 1] = s.d2 - fixed2;
            }
            let r = lstsq(&a, &b);
            let mut terms = [0.0; N_TERMS];
            for i in 0..N_TERMS {
                terms[i] = if is_free[i] { 0.0 } else { opts.seed[i] };
            }
            for (jj, &j) in free.iter().enumerate() {
                terms[j] = r.x[jj];
            }
            (terms, r.cond)
        }
    }
}

fn predicted(s: &Sample, terms: &[f64; N_TERMS]) -> (f64, f64) {
    let p1: f64 = (0..N_TERMS).map(|j| s.row1[j] * terms[j]).sum();
    let p2: f64 = (0..N_TERMS).map(|j| s.row2[j] * terms[j]).sum();
    (p1, p2)
}

fn sky_rms(pairs: &[(f64, f64, f64, f64)]) -> f64 {
    let mut sq = 0.0;
    for &(_c1, c2, d1, d2) in pairs {
        let c = c2.to_radians().cos();
        sq += (d1 * c) * (d1 * c) + d2 * d2;
    }
    (sq / pairs.len().max(1) as f64).sqrt()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

const MIN_FIT: usize = 4;

/// The full fit: solve, optional robust reject + refit, stats.
pub fn fit(samples: &[Sample], opts: &FitOptions) -> FitStats {
    let mut samples: Vec<Sample> = samples.to_vec();
    let (mut terms, mut cond) = solve(&samples, opts);
    let mut n_rejected = 0;

    if opts.robust && samples.len() > MIN_FIT {
        let mut mags: Vec<f64> = samples
            .iter()
            .map(|s| {
                let (p1, p2) = predicted(s, &terms);
                let r1 = wrap180(s.d1 - p1) * s.c2_cmd.to_radians().cos();
                let r2 = s.d2 - p2;
                (r1 * r1 + r2 * r2).sqrt()
            })
            .collect();
        let mut sorted = mags.clone();
        let med = median(&mut sorted);
        let mut devs: Vec<f64> = mags.iter().map(|m| (m - med).abs()).collect();
        let mut mad = median(&mut devs);
        if mad == 0.0 {
            mad = 1e-9;
        }
        let thresh = (med + opts.robust_sigma * 1.4826 * mad).max(opts.robust_floor_deg);
        let keep: Vec<Sample> = samples
            .iter()
            .zip(mags.drain(..))
            .filter(|(_, m)| *m <= thresh)
            .map(|(s, _)| s.clone())
            .collect();
        if keep.len() > MIN_FIT && keep.len() < samples.len() {
            n_rejected = samples.len() - keep.len();
            samples = keep;
            let (t, c) = solve(&samples, opts);
            terms = t;
            cond = c;
        }
    }

    let raw: Vec<(f64, f64, f64, f64)> = samples
        .iter()
        .map(|s| (s.c1_cmd, s.c2_cmd, s.d1, s.d2))
        .collect();
    let resid: Vec<(f64, f64, f64, f64)> = samples
        .iter()
        .map(|s| {
            let (p1, p2) = predicted(s, &terms);
            (s.c1_cmd, s.c2_cmd, wrap180(s.d1 - p1), s.d2 - p2)
        })
        .collect();

    FitStats {
        terms,
        n_samples: samples.len(),
        n_rejected,
        design_cond: cond,
        rms_before_deg: sky_rms(&raw),
        rms_after_deg: sky_rms(&resid),
    }
}
