//! Satellite pass prediction: AOS / TCA / LOS against an elevation mask,
//! peak angular rate, range at culmination, TLE apogee/perigee, and the
//! pass-table brightness estimate — the native counterpart of the pygame
//! app's pass table (trajectory.extract_pass_data_from_trajectory /
//! build_satellite_pass_table / _estimate_pass_magnitude and
//! satellite_data.create_satellite_labels_and_metadata).
//!
//! Semantics carried over from the Python reference:
//!
//! * "Visible" means `el > min_el_deg` (strict), exactly as the Python
//!   `altitudes > elevation_mask_deg` test. AOS/LOS are the instants where
//!   the elevation crosses the mask (the Python table only ever saw 30 s
//!   samples; here the crossings are bisected to `fine_step_s` and then
//!   linearly interpolated, so `el(aos) == min_el` to well under 0.1 deg).
//! * TCA is the culmination (maximum elevation), which is where the Python
//!   table evaluates the magnitude (`max_el_time_tt`) and reads
//!   `azimuth_at_max`.
//! * Estimated magnitude: generic standard magnitude
//!   [`PASS_MAG_DEFAULT_STDMAG`] (apparent mag at 1000 km, 50% illuminated,
//!   McCants convention) with a Lambertian phase term,
//!   `stdmag - 15.75 + 2.5 log10(range_km^2 / frac_illum)`. A cylindrical
//!   Earth-shadow test (behind the terminator plane AND within
//!   [`R_EARTH_SHADOW_KM`] of the anti-sun axis) yields `None` = "ecl".
//! * Apogee/perigee come from the TLE mean motion + eccentricity with
//!   MU = 3.986004418e14 and R_EARTH = 6371 km (satellite_data.py).
//!
//! Frames. The satellite is propagated in TEME (SGP4's native frame) and the
//! observer's ECEF position is rotated into TEME with the same GMST1982
//! angle the alt/az path uses, so the satellite->observer vector is exact.
//! The sun vector supplied by the caller (`sun_eci_km`) is used as if it
//! were TEME. TEME, true-of-date and GCRS/ICRS axes differ by at most
//! ~0.4 deg (precession since J2000 + nutation); against a cylindrical
//! shadow test and a Lambertian phase term that error is negligible (the
//! Python reference itself evaluates the sun once per table build, i.e. up
//! to ~1 deg stale), so any of those frames is acceptable.
//! `Engine::predict_passes` supplies the sun rotated into the true equator
//! and equinox of date, which is within the equation of the equinoxes
//! (~20 arcsec) of TEME.

use crate::sgp4_pass::{altaz_from_teme, Observer, ObserverGeometry, Satellite};
use crate::time;
use rayon::prelude::*;

/// Generic standard magnitude (trajectory.PASS_MAG_DEFAULT_STDMAG).
pub const PASS_MAG_DEFAULT_STDMAG: f64 = 6.0;
/// Equatorial radius for the cylindrical shadow test (trajectory.R_EARTH_SHADOW_KM).
pub const R_EARTH_SHADOW_KM: f64 = 6378.137;
/// Earth gravitational parameter, m^3/s^2 (satellite_data.py MU).
pub const MU_EARTH_M3_S2: f64 = 3.986004418e14;
/// Mean Earth radius used for apogee/perigee altitude, km (satellite_data.py R_EARTH).
pub const R_EARTH_MEAN_KM: f64 = 6371.0;

/// One predicted pass of one satellite over the observer.
#[derive(Clone, Debug)]
pub struct Pass {
    pub satnum: String,
    pub name: String,
    /// Acquisition of signal: elevation rises through `min_el_deg` (TT JD).
    pub aos_jd_tt: f64,
    /// Time of closest approach = culmination (maximum elevation), TT JD.
    pub tca_jd_tt: f64,
    /// Loss of signal: elevation falls through `min_el_deg` (TT JD).
    pub los_jd_tt: f64,
    pub aos_az: f64,
    pub tca_az: f64,
    pub tca_el: f64,
    pub los_az: f64,
    pub duration_s: f64,
    /// Peak angular rate (deg/s) during the pass, from consecutive
    /// `fine_step_s` samples (great-circle separation / step).
    pub max_rate_dps: f64,
    pub range_tca_km: f64,
    /// Apogee / perigee altitude above R_EARTH_MEAN_KM, from the TLE.
    pub apogee_km: f64,
    pub perigee_km: f64,
    /// Estimated visual magnitude at TCA. `None` when eclipsed at TCA
    /// ("ecl" in the table) or when no sun vector was available ("--";
    /// see `eclipsed_at_tca` to tell the two apart).
    pub est_mag: Option<f64>,
    /// True when the cylindrical shadow test put the satellite in Earth's
    /// shadow at TCA. False when not eclipsed or when not computed.
    pub eclipsed_at_tca: bool,
    /// The satellite was already above the mask at `jd_tt_start`; `aos_jd_tt`
    /// is the window start, not a real crossing.
    pub aos_truncated: bool,
    /// The satellite was still above the mask at the end of the horizon;
    /// `los_jd_tt` is the window end, not a real crossing.
    pub los_truncated: bool,
}

#[derive(Clone, Debug)]
pub struct PassParams {
    /// Elevation mask, degrees. A pass is the interval with `el > min_el_deg`.
    pub min_el_deg: f64,
    /// Search horizon from `jd_tt_start`, hours.
    pub horizon_hours: f64,
    /// Coarse sampling step used to find passes, seconds. Passes shorter
    /// than this (or peaking above the mask for less than this) can be
    /// missed.
    pub coarse_step_s: f64,
    /// Refinement step for AOS/LOS bisection and TCA/rate sampling, seconds.
    pub fine_step_s: f64,
    /// Keep at most this many passes (earliest first) per satellite.
    pub max_passes_per_sat: usize,
}

impl Default for PassParams {
    fn default() -> Self {
        PassParams {
            min_el_deg: 10.0,
            horizon_hours: 6.0,
            coarse_step_s: 30.0,
            fine_step_s: 1.0,
            max_passes_per_sat: 3,
        }
    }
}

/// One sample of a satellite's track for the skyplot.
#[derive(Clone, Copy, Debug)]
pub struct ArcPoint {
    pub jd_tt: f64,
    pub az: f64,
    pub el: f64,
    pub range_km: f64,
}

/// TLE-derived apogee and perigee altitudes in km
/// (satellite_data.create_satellite_labels_and_metadata): Keplerian
/// semi-major axis from the Kozai mean motion, minus the 6371 km mean radius.
pub fn apogee_perigee_km(sat: &Satellite) -> (f64, f64) {
    let n_rad_s = sat.mean_motion_rev_per_day * time::TAU / time::DAY_S;
    let a_km = (MU_EARTH_M3_S2 / (n_rad_s * n_rad_s)).cbrt() / 1000.0;
    let e = sat.eccentricity;
    (
        a_km * (1.0 + e) - R_EARTH_MEAN_KM,
        a_km * (1.0 - e) - R_EARTH_MEAN_KM,
    )
}

/// Observer geocentric position in TEME (km) at a TT instant: the inverse
/// of the TEME->PEF rotation in `altaz_from_teme` applied to the ECEF site.
pub fn observer_teme_km(geom: &ObserverGeometry, jd_tt: f64) -> [f64; 3] {
    let jd_ut1 = time::tt_to_ut1(jd_tt);
    let (theta, _rate) = time::theta_gmst1982(jd_ut1);
    let (s, c) = theta.sin_cos();
    let p = geom.pos_km;
    // rot_z(+theta): PEF -> TEME.
    [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Cylindrical Earth-shadow test (trajectory._estimate_pass_magnitude):
/// true when the satellite is behind the terminator plane AND within one
/// equatorial radius of the anti-sun axis. `r_sat_km` and `r_sun_km` are
/// geocentric vectors in the same (or nearly the same) ECI frame.
pub fn is_eclipsed(r_sat_km: &[f64; 3], r_sun_km: &[f64; 3]) -> bool {
    let sun_norm = norm(r_sun_km);
    if sun_norm <= 0.0 {
        return false;
    }
    let u = [
        r_sun_km[0] / sun_norm,
        r_sun_km[1] / sun_norm,
        r_sun_km[2] / sun_norm,
    ];
    let along = dot(r_sat_km, &u);
    if along >= 0.0 {
        return false;
    }
    let perp = [
        r_sat_km[0] - along * u[0],
        r_sat_km[1] - along * u[1],
        r_sat_km[2] - along * u[2],
    ];
    norm(&perp) < R_EARTH_SHADOW_KM
}

/// Estimated apparent magnitude (trajectory._estimate_pass_magnitude) from
/// geocentric satellite, observer and sun vectors (km, one ECI frame).
/// Returns `None` when eclipsed.
pub fn estimate_magnitude(
    r_sat_km: &[f64; 3],
    r_obs_km: &[f64; 3],
    r_sun_km: &[f64; 3],
) -> Option<f64> {
    if is_eclipsed(r_sat_km, r_sun_km) {
        return None;
    }
    let v_obs = [
        r_obs_km[0] - r_sat_km[0],
        r_obs_km[1] - r_sat_km[1],
        r_obs_km[2] - r_sat_km[2],
    ];
    let v_sun = [
        r_sun_km[0] - r_sat_km[0],
        r_sun_km[1] - r_sat_km[1],
        r_sun_km[2] - r_sat_km[2],
    ];
    let range_km = norm(&v_obs);
    let cos_phase = dot(&v_obs, &v_sun) / (range_km * norm(&v_sun));
    let frac_illum = (0.5 * (1.0 + cos_phase)).max(1e-3);
    Some(PASS_MAG_DEFAULT_STDMAG - 15.75 + 2.5 * (range_km * range_km / frac_illum).log10())
}

/// Great-circle separation (deg) between two (az, el) directions in degrees.
fn angular_sep_deg(az1: f64, el1: f64, az2: f64, el2: f64) -> f64 {
    let (a1, e1, a2, e2) = (
        az1.to_radians(),
        el1.to_radians(),
        az2.to_radians(),
        el2.to_radians(),
    );
    // Haversine form: well conditioned for the small steps used here.
    let sde = ((e2 - e1) / 2.0).sin();
    let sda = ((a2 - a1) / 2.0).sin();
    let h = sde * sde + e1.cos() * e2.cos() * sda * sda;
    2.0 * h.sqrt().clamp(0.0, 1.0).asin().to_degrees()
}

/// (alt, az, range) of `sat` at `jd_tt`, or None when SGP4 fails
/// (decayed/divergent orbit). Failures are treated as "below the mask",
/// matching `visibility_mask`.
fn altaz(sat: &Satellite, jd_tt: f64, geom: &ObserverGeometry) -> Option<(f64, f64, f64)> {
    let r = sat.position_teme_km(jd_tt).ok()?;
    let (alt, az, range) = altaz_from_teme(r, jd_tt, geom);
    if alt.is_finite() && az.is_finite() && range.is_finite() {
        Some((alt, az, range))
    } else {
        None
    }
}

fn el_above(sat: &Satellite, jd_tt: f64, geom: &ObserverGeometry, min_el: f64) -> bool {
    matches!(altaz(sat, jd_tt, geom), Some((alt, _, _)) if alt > min_el)
}

/// Refine a mask crossing inside [lo, hi] (exactly one side above the
/// mask) by bisection to `fine_step_s`, then linear interpolation of the
/// elevation across the final bracket. Returns the crossing TT JD.
fn refine_crossing(
    sat: &Satellite,
    geom: &ObserverGeometry,
    min_el: f64,
    mut lo: f64,
    mut hi: f64,
    fine_step_s: f64,
) -> f64 {
    let lo_above = el_above(sat, lo, geom, min_el);
    let tol_days = fine_step_s.max(1e-3) / time::DAY_S;
    let mut iters = 0;
    while (hi - lo) > tol_days && iters < 64 {
        let mid = 0.5 * (lo + hi);
        if el_above(sat, mid, geom, min_el) == lo_above {
            lo = mid;
        } else {
            hi = mid;
        }
        iters += 1;
    }
    // Linear interpolation on elevation across the final bracket.
    match (altaz(sat, lo, geom), altaz(sat, hi, geom)) {
        (Some((e_lo, _, _)), Some((e_hi, _, _))) if (e_hi - e_lo).abs() > 1e-12 => {
            let f = ((min_el - e_lo) / (e_hi - e_lo)).clamp(0.0, 1.0);
            lo + f * (hi - lo)
        }
        _ => 0.5 * (lo + hi),
    }
}

/// Build one `Pass` from a refined [aos, los] interval: fine-sample the
/// interval for TCA (parabolic refinement about the max sample), peak
/// angular rate, and the endpoint azimuths; then evaluate the magnitude.
#[allow(clippy::too_many_arguments)]
fn build_pass(
    sat: &Satellite,
    geom: &ObserverGeometry,
    params: &PassParams,
    aos: f64,
    los: f64,
    aos_truncated: bool,
    los_truncated: bool,
    sun_eci_km: &(dyn Fn(f64) -> Option<[f64; 3]> + Sync),
) -> Option<Pass> {
    let step_days = params.fine_step_s.max(1e-3) / time::DAY_S;
    let span = los - aos;
    let n = ((span / step_days).floor() as usize).max(1) + 1;
    let mut samples: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = aos + (i as f64) * step_days;
        if let Some((alt, az, rng)) = altaz(sat, t, geom) {
            samples.push((t, alt, az, rng));
        }
    }
    // Always include the exact LOS instant as the last sample.
    if samples.last().map(|s| (los - s.0) > 1e-9).unwrap_or(true) {
        if let Some((alt, az, rng)) = altaz(sat, los, geom) {
            samples.push((los, alt, az, rng));
        }
    }
    if samples.is_empty() {
        return None;
    }

    // Culmination: max-elevation sample, refined with a parabola through
    // its neighbours, then re-evaluated exactly at the vertex time.
    let mut imax = 0;
    for (i, s) in samples.iter().enumerate() {
        if s.1 > samples[imax].1 {
            imax = i;
        }
    }
    let mut tca = samples[imax].0;
    if imax > 0 && imax + 1 < samples.len() {
        let (t0, y0) = (samples[imax - 1].0, samples[imax - 1].1);
        let (t1, y1) = (samples[imax].0, samples[imax].1);
        let (t2, y2) = (samples[imax + 1].0, samples[imax + 1].1);
        let denom = y0 - 2.0 * y1 + y2;
        if denom.abs() > 1e-12 && (t2 - t0) > 0.0 {
            // Vertex offset in units of the (nominally uniform) step.
            let h = 0.5 * (t2 - t0);
            let off = 0.5 * (y0 - y2) / denom;
            let cand = t1 + off.clamp(-1.0, 1.0) * h;
            tca = cand.clamp(t0, t2);
        }
    }
    let (tca_el, tca_az, range_tca) = altaz(sat, tca, geom).unwrap_or((
        samples[imax].1,
        samples[imax].2,
        samples[imax].3,
    ));
    // Keep the sampled maximum if the refined point is (numerically) lower.
    let (tca, tca_el, tca_az, range_tca) = if tca_el < samples[imax].1 {
        (samples[imax].0, samples[imax].1, samples[imax].2, samples[imax].3)
    } else {
        (tca, tca_el, tca_az, range_tca)
    };

    // Peak angular rate over consecutive fine samples.
    let mut max_rate = 0.0_f64;
    for w in samples.windows(2) {
        let dt_s = (w[1].0 - w[0].0) * time::DAY_S;
        if dt_s > 1e-6 {
            let sep = angular_sep_deg(w[0].2, w[0].1, w[1].2, w[1].1);
            max_rate = max_rate.max(sep / dt_s);
        }
    }

    let aos_az = samples.first().map(|s| s.2).unwrap_or(0.0);
    let los_az = samples.last().map(|s| s.2).unwrap_or(0.0);

    // Brightness at culmination.
    let (est_mag, eclipsed) = match (sun_eci_km(tca), sat.position_teme_km(tca)) {
        (Some(r_sun), Ok(r_sat)) => {
            let r_obs = observer_teme_km(geom, tca);
            let ecl = is_eclipsed(&r_sat, &r_sun);
            (estimate_magnitude(&r_sat, &r_obs, &r_sun), ecl)
        }
        _ => (None, false),
    };

    let (apogee_km, perigee_km) = apogee_perigee_km(sat);
    Some(Pass {
        satnum: sat.satnum.clone(),
        name: sat.name.clone(),
        aos_jd_tt: aos,
        tca_jd_tt: tca,
        los_jd_tt: los,
        aos_az,
        tca_az,
        tca_el,
        los_az,
        duration_s: (los - aos) * time::DAY_S,
        max_rate_dps: max_rate,
        range_tca_km: range_tca,
        apogee_km,
        perigee_km,
        est_mag,
        eclipsed_at_tca: eclipsed,
        aos_truncated,
        los_truncated,
    })
}

/// All passes of one satellite over the horizon (earliest first, at most
/// `max_passes_per_sat`).
pub fn predict_passes_one(
    sat: &Satellite,
    geom: &ObserverGeometry,
    jd_tt_start: f64,
    params: &PassParams,
    sun_eci_km: &(dyn Fn(f64) -> Option<[f64; 3]> + Sync),
) -> Vec<Pass> {
    let mut out = Vec::new();
    if params.max_passes_per_sat == 0 || params.horizon_hours <= 0.0 {
        return out;
    }
    let coarse_days = params.coarse_step_s.max(0.1) / time::DAY_S;
    let horizon_days = params.horizon_hours * 3600.0 / time::DAY_S;
    let jd_end = jd_tt_start + horizon_days;
    let n = (horizon_days / coarse_days).ceil() as usize + 1;

    let mut prev_t = jd_tt_start;
    let mut prev_above = el_above(sat, jd_tt_start, geom, params.min_el_deg);
    let mut aos: Option<(f64, bool)> = if prev_above {
        Some((jd_tt_start, true))
    } else {
        None
    };

    for i in 1..=n {
        let t = (jd_tt_start + (i as f64) * coarse_days).min(jd_end);
        let above = el_above(sat, t, geom, params.min_el_deg);
        if above && !prev_above {
            let t_aos = refine_crossing(sat, geom, params.min_el_deg, prev_t, t, params.fine_step_s);
            aos = Some((t_aos, false));
        } else if !above && prev_above {
            if let Some((t_aos, trunc)) = aos.take() {
                let t_los =
                    refine_crossing(sat, geom, params.min_el_deg, prev_t, t, params.fine_step_s);
                if let Some(p) =
                    build_pass(sat, geom, params, t_aos, t_los, trunc, false, sun_eci_km)
                {
                    out.push(p);
                    if out.len() >= params.max_passes_per_sat {
                        return out;
                    }
                }
            }
        }
        prev_t = t;
        prev_above = above;
        if t >= jd_end {
            break;
        }
    }
    // Still above the mask at the end of the horizon: truncated LOS.
    if let Some((t_aos, trunc)) = aos.take() {
        if let Some(p) = build_pass(sat, geom, params, t_aos, jd_end, trunc, true, sun_eci_km) {
            out.push(p);
        }
    }
    out
}

/// Passes for many satellites (rayon over satellites), sorted by AOS.
///
/// `sun_eci_km(jd_tt)` returns the geocentric sun position in km in an
/// Earth-centered inertial frame close to TEME (TEME, true-of-date or
/// GCRS/ICRS axes are all acceptable — see the module docs), or `None`
/// when no ephemeris is available (then `est_mag` is `None` and
/// `eclipsed_at_tca` is false for every pass).
pub fn predict_passes(
    sats: &[&Satellite],
    observer: &Observer,
    jd_tt_start: f64,
    params: &PassParams,
    sun_eci_km: &(dyn Fn(f64) -> Option<[f64; 3]> + Sync),
) -> Vec<Pass> {
    let geom = observer.geometry();
    let mut passes: Vec<Pass> = sats
        .par_iter()
        .flat_map_iter(|sat| predict_passes_one(sat, &geom, jd_tt_start, params, sun_eci_km))
        .collect();
    passes.sort_by(|a, b| {
        a.aos_jd_tt
            .partial_cmp(&b.aos_jd_tt)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    passes
}

/// Track of `sat` from `center - before_s` to `center + after_s` at
/// `step_s`, keeping only points with `el > -5` deg (for drawing the
/// selected satellite's arc on the skyplot). Propagation failures are
/// skipped.
pub fn trajectory_arc(
    sat: &Satellite,
    geom: &ObserverGeometry,
    jd_tt_center: f64,
    before_s: f64,
    after_s: f64,
    step_s: f64,
) -> Vec<ArcPoint> {
    let step = step_s.max(1e-3);
    let t0 = jd_tt_center - before_s.max(0.0) / time::DAY_S;
    let span_s = before_s.max(0.0) + after_s.max(0.0);
    let n = (span_s / step).floor() as usize + 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = t0 + (i as f64) * step / time::DAY_S;
        if let Some((el, az, range_km)) = altaz(sat, t, geom) {
            if el > -5.0 {
                out.push(ArcPoint {
                    jd_tt: t,
                    az,
                    el,
                    range_km,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eclipse_geometry() {
        let sun = [1.5e8, 0.0, 0.0];
        // Directly behind Earth on the anti-sun axis: eclipsed.
        assert!(is_eclipsed(&[-7000.0, 0.0, 0.0], &sun));
        // Behind the terminator plane but outside the cylinder: lit.
        assert!(!is_eclipsed(&[-7000.0, 7000.0, 0.0], &sun));
        // Sunward side: lit.
        assert!(!is_eclipsed(&[7000.0, 0.0, 0.0], &sun));
    }

    #[test]
    fn magnitude_brighter_when_closer_and_fuller() {
        // Pure formula check (frame-free): sun on +x, observer on the
        // terminator, satellite anti-sunward of the observer (full phase,
        // outside the shadow cylinder) at several ranges.
        let sun = [1.5e8, 0.0, 0.0];
        let obs = [0.0, 7000.0, 0.0];
        let near = estimate_magnitude(&[-500.0, 7000.0, 0.0], &obs, &sun).unwrap();
        let far = estimate_magnitude(&[-1500.0, 7000.0, 0.0], &obs, &sun).unwrap();
        assert!(near < far, "near {near} should be brighter than far {far}");
        // 1000 km, full phase: stdmag - 15.75 + 2.5 log10(1e6) = stdmag - 0.75
        let full = estimate_magnitude(&[-1000.0, 7000.0, 0.0], &obs, &sun).unwrap();
        assert!(
            (full - (PASS_MAG_DEFAULT_STDMAG - 0.75)).abs() < 1e-6,
            "full {full}"
        );
        // Same range, ~90 deg phase (half illuminated): stdmag exactly.
        let half = estimate_magnitude(&[0.0, 7000.0, 1000.0], &obs, &sun).unwrap();
        assert!((half - PASS_MAG_DEFAULT_STDMAG).abs() < 0.01, "half {half}");
        assert!(full < half);
        // Sunward of the observer (new phase): very dim but finite.
        let new = estimate_magnitude(&[1000.0, 7000.0, 0.0], &obs, &sun).unwrap();
        assert!(new > half + 5.0, "new {new}");
    }

    #[test]
    fn apogee_perigee_from_elements() {
        // Circular 15.5 rev/day orbit: a ~ 6790 km -> ~420 km altitude.
        let text = "ISS (ZARYA)\n1 25544U 98067A   26228.56710022  .00005115  00000+0  99348-4 0  9991\n2 25544  51.6334   1.2594 0007609  53.1141 307.0544 15.49461657581119\n";
        let cat = crate::tle::TleCatalog::from_text(text);
        let sat = cat.get("25544").unwrap();
        let (ap, pe) = apogee_perigee_km(sat);
        assert!(ap > pe);
        assert!((400.0..460.0).contains(&ap), "apogee {ap}");
        assert!((400.0..460.0).contains(&pe), "perigee {pe}");
    }
}
