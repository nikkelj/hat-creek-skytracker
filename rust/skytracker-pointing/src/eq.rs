//! 7-term equatorial TPOINT model (IH, ID, NP, CH, ME, MA, TF) — port of
//! eq_pointing_model.py, including the parallactic-angle flexure terms.

use crate::fit::{self, wrap180, FitOptions, FitStats, Sample};

pub const TERM_NAMES: [&str; 7] = ["IH", "ID", "NP", "CH", "ME", "MA", "TF"];

/// The 7-term basis rows for the HA and Dec error equations at latitude.
pub fn design_rows(h_deg: f64, d_deg: f64, lat_deg: f64) -> ([f64; 7], [f64; 7]) {
    let h = h_deg.to_radians();
    let d = d_deg.to_radians();
    let phi = lat_deg.to_radians();
    let tan_d = d.tan();
    let sec_d = 1.0 / d.cos();
    let sin_el = (phi.sin() * d.sin() + phi.cos() * d.cos() * h.cos()).clamp(-1.0, 1.0);
    let cos_el = sin_el.asin().cos();
    let q = h.sin().atan2(phi.tan() * d.cos() - d.sin() * h.cos());
    let flex_h = cos_el * q.sin() * sec_d;
    let flex_d = -cos_el * q.cos();
    (
        [1.0, 0.0, tan_d, sec_d, h.sin() * tan_d, -h.cos() * tan_d, flex_h],
        [0.0, 1.0, 0.0, 0.0, h.cos(), h.sin(), flex_d],
    )
}

pub fn error(terms: &[f64; 7], h_deg: f64, d_deg: f64, lat_deg: f64) -> (f64, f64) {
    let (r1, r2) = design_rows(h_deg, d_deg, lat_deg);
    let d_ha = (0..7).map(|i| r1[i] * terms[i]).sum();
    let d_dec = (0..7).map(|i| r2[i] * terms[i]).sum();
    (d_ha, d_dec)
}

/// Fit from (h_cmd, d_cmd, h_obs, d_obs) samples at the given latitude.
#[allow(clippy::too_many_arguments)]
pub fn fit_eq(
    samples: &[[f64; 4]],
    lat_deg: f64,
    seed: [f64; 7],
    free_idx: Option<Vec<usize>>,
    robust: bool,
    robust_sigma: f64,
    robust_floor_deg: f64,
) -> FitStats {
    let prepared: Vec<Sample> = samples
        .iter()
        .map(|&[h_cmd, d_cmd, h_obs, d_obs]| {
            let (row1, row2) = design_rows(h_cmd, d_cmd, lat_deg);
            Sample {
                c1_cmd: h_cmd,
                c2_cmd: d_cmd,
                d1: wrap180(h_obs - h_cmd),
                d2: d_obs - d_cmd,
                row1,
                row2,
            }
        })
        .collect();
    fit::fit(
        &prepared,
        &FitOptions {
            free_idx,
            seed,
            robust,
            robust_sigma,
            robust_floor_deg,
        },
    )
}

/// Backtest: great-circle sky RMS (deg) on held-out samples.
pub fn backtest_eq(terms: &[f64; 7], samples: &[[f64; 4]], lat_deg: f64) -> f64 {
    let mut sq = 0.0;
    for &[h_cmd, d_cmd, h_obs, d_obs] in samples {
        let (d_ha_m, d_dec_m) = error(terms, h_cmd, d_cmd, lat_deg);
        let pr_h = wrap180(h_cmd + d_ha_m);
        let pr_d = d_cmd + d_dec_m;
        let d_ha = wrap180(h_obs - pr_h);
        let d_dec = d_obs - pr_d;
        let c = d_cmd.to_radians().cos();
        sq += (d_ha * c) * (d_ha * c) + d_dec * d_dec;
    }
    (sq / samples.len().max(1) as f64).sqrt()
}
