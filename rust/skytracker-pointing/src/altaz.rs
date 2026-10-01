//! 7-term alt-az TPOINT model (IA, IE, AN, AW, NPAE, CA, TF) — port of
//! pointing_model.py. Coefficient order and formulas are identical.

use crate::fit::{self, wrap180, FitOptions, FitStats, Sample};

pub const TERM_NAMES: [&str; 7] = ["IA", "IE", "AN", "AW", "NPAE", "CA", "TF"];

/// Bennett refraction (apparent - true elevation), degrees.
pub fn bennett_refraction_deg(el_deg: f64, pressure_mbar: f64, temperature_c: f64) -> f64 {
    if el_deg < -1.0 {
        return 0.0;
    }
    let r_arcmin = 1.0 / (el_deg + 7.31 / (el_deg + 4.4)).to_radians().tan();
    r_arcmin * (pressure_mbar / 1010.0) * (283.0 / (273.0 + temperature_c)) / 60.0
}

/// The 7-term basis rows for the az and el error equations.
pub fn design_rows(az_deg: f64, el_deg: f64) -> ([f64; 7], [f64; 7]) {
    let a = az_deg.to_radians();
    let e = el_deg.to_radians();
    let tan_e = e.tan();
    let sec_e = 1.0 / e.cos();
    (
        [1.0, 0.0, -a.sin() * tan_e, -a.cos() * tan_e, tan_e, sec_e, 0.0],
        [0.0, 1.0, -a.cos(), a.sin(), 0.0, 0.0, -e.cos()],
    )
}

/// Pointing error (dAz, dEl) at a commanded position for given terms.
pub fn error(terms: &[f64; 7], az_deg: f64, el_deg: f64) -> (f64, f64) {
    let (r1, r2) = design_rows(az_deg, el_deg);
    let d_az = (0..7).map(|i| r1[i] * terms[i]).sum();
    let d_el = (0..7).map(|i| r2[i] * terms[i]).sum();
    (d_az, d_el)
}

/// Fit from (az_cmd, el_cmd, az_obs, el_obs) samples, mirroring
/// PointingModel.fit (incl. refraction stripping, partial fit, robust).
#[allow(clippy::too_many_arguments)]
pub fn fit_altaz(
    samples: &[[f64; 4]],
    remove_refraction: bool,
    seed: [f64; 7],
    free_idx: Option<Vec<usize>>,
    robust: bool,
    robust_sigma: f64,
    robust_floor_deg: f64,
) -> FitStats {
    let prepared: Vec<Sample> = samples
        .iter()
        .map(|&[az_cmd, el_cmd, az_obs, el_obs]| {
            let el_obs_geo = if remove_refraction {
                el_obs - bennett_refraction_deg(el_obs, 1010.0, 10.0)
            } else {
                el_obs
            };
            let (row1, row2) = design_rows(az_cmd, el_cmd);
            Sample {
                c1_cmd: az_cmd,
                c2_cmd: el_cmd,
                d1: wrap180(az_obs - az_cmd),
                d2: el_obs_geo - el_cmd,
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

/// Backtest: great-circle sky RMS (deg) of a model on held-out samples.
pub fn backtest_altaz(terms: &[f64; 7], samples: &[[f64; 4]], remove_refraction: bool) -> f64 {
    let mut sq = 0.0;
    for &[az_cmd, el_cmd, az_obs, el_obs] in samples {
        let el_obs_geo = if remove_refraction {
            el_obs - bennett_refraction_deg(el_obs, 1010.0, 10.0)
        } else {
            el_obs
        };
        let (d_az_m, d_el_m) = error(terms, az_cmd, el_cmd);
        let pr_az = (az_cmd + d_az_m).rem_euclid(360.0);
        let pr_el = el_cmd + d_el_m;
        let d_az = wrap180(az_obs - pr_az);
        let d_el = el_obs_geo - pr_el;
        let c = el_cmd.to_radians().cos();
        sq += (d_az * c) * (d_az * c) + d_el * d_el;
    }
    (sq / samples.len().max(1) as f64).sqrt()
}
