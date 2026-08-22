//! Satellite passes: SGP4 propagation to topocentric alt/az/range, matching
//! skyfield's `(sat - observer).at(t).altaz()` pipeline.
//!
//! Why this is so short: skyfield's full chain (TEME -> GCRS via
//! `rot_z(theta82 - gast) . M`, observer ITRS -> GCRS via `M^T rot_z(gast)`,
//! then projection onto the observer's ENU axes) algebraically collapses —
//! the `M` and `gast` factors cancel in the observer-relative projection —
//! to: `ENU( rot_z(-theta_GMST1982(ut1)) . r_TEME  -  r_observer_ECEF )`.
//! Polar motion is zero under skyfield's builtin timescale, which is what
//! the app (and the golden vectors) use. SGP4 epochs and theta82 both take
//! UT1, exactly as skyfield feeds them.

use crate::time;

/// WGS84 observer site.
#[derive(Clone, Copy, Debug)]
pub struct Observer {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub elevation_m: f64,
}

pub struct ObserverGeometry {
    /// Geocentric ECEF position, km.
    pub pos_km: [f64; 3],
    /// ENU unit vectors in ECEF: east, north, up.
    pub east: [f64; 3],
    pub north: [f64; 3],
    pub up: [f64; 3],
}

impl Observer {
    /// WGS84 geodetic -> geocentric ECEF + local ENU axes
    /// (same constants as trajectory.py `_batched_visibility_mask`).
    pub fn geometry(&self) -> ObserverGeometry {
        let a_km = 6378.137;
        let f = 1.0 / 298.257223563;
        let e2 = f * (2.0 - f);
        let phi = self.lat_deg.to_radians();
        let lam = self.lon_deg.to_radians();
        let h_km = self.elevation_m / 1000.0;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let (sin_lam, cos_lam) = lam.sin_cos();
        let n = a_km / (1.0 - e2 * sin_phi * sin_phi).sqrt();
        ObserverGeometry {
            pos_km: [
                (n + h_km) * cos_phi * cos_lam,
                (n + h_km) * cos_phi * sin_lam,
                (n * (1.0 - e2) + h_km) * sin_phi,
            ],
            east: [-sin_lam, cos_lam, 0.0],
            north: [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
            up: [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi],
        }
    }
}

/// A parsed TLE ready for propagation.
pub struct Satellite {
    pub name: String,
    /// NORAD catalog field (line-1 cols 3-7), alpha-5 safe.
    pub satnum: String,
    pub epoch_jd_utc: f64,
    constants: sgp4::Constants,
}

#[derive(Debug)]
pub enum SatError {
    Tle(String),
    Propagation(String),
}

impl std::fmt::Display for SatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SatError::Tle(s) => write!(f, "TLE parse error: {s}"),
            SatError::Propagation(s) => write!(f, "SGP4 propagation error: {s}"),
        }
    }
}

impl std::error::Error for SatError {}

/// Parse the TLE epoch from line 1 (cols 19-32, `YYDDD.DDDDDDDD`) into a
/// UTC Julian date, using the same convention as python-sgp4's jdsatepoch.
fn tle_epoch_jd(line1: &str) -> Result<f64, SatError> {
    let field = line1
        .get(18..32)
        .ok_or_else(|| SatError::Tle("line 1 too short for epoch".into()))?;
    let yy: i64 = field[..2]
        .trim()
        .parse()
        .map_err(|_| SatError::Tle(format!("bad epoch year in {field:?}")))?;
    let doy: f64 = field[2..]
        .trim()
        .parse()
        .map_err(|_| SatError::Tle(format!("bad epoch day in {field:?}")))?;
    let year = if yy < 57 { 2000 + yy } else { 1900 + yy };
    Ok(time::julian_date(year, 1, 1, 0.0, 0.0, 0.0) + doy - 1.0)
}

impl Satellite {
    pub fn from_tle(name: &str, line1: &str, line2: &str) -> Result<Self, SatError> {
        let epoch_jd_utc = tle_epoch_jd(line1)?;
        let elements = sgp4::Elements::from_tle(
            Some(name.to_string()),
            line1.as_bytes(),
            line2.as_bytes(),
        )
        .map_err(|e| SatError::Tle(e.to_string()))?;
        // AFSPC-compatibility mode selects WGS72 gravity constants — the
        // convention python-sgp4 (and therefore skyfield and the golden
        // vectors) uses. The crate's plain `from_elements` is WGS84 and
        // diverges from python-sgp4 by ~0.5 km over two weeks.
        let constants = sgp4::Constants::from_elements_afspc_compatibility_mode(&elements)
            .map_err(|e| SatError::Tle(e.to_string()))?;
        Ok(Satellite {
            name: name.to_string(),
            satnum: line1.get(2..7).unwrap_or("").trim().to_string(),
            epoch_jd_utc,
            constants,
        })
    }

    /// TEME position (km) at a TT Julian date. Matches skyfield's timing
    /// convention: SGP4 runs on the UTC scale ("we assume the TLE epoch to
    /// be a UTC date", sgp4lib), while the TEME->PEF rotation uses UT1.
    pub fn position_teme_km(&self, jd_tt: f64) -> Result<[f64; 3], SatError> {
        let jd_utc = time::tt_to_utc(jd_tt);
        let minutes = (jd_utc - self.epoch_jd_utc) * 1440.0;
        let prediction = self
            .constants
            .propagate(sgp4::MinutesSinceEpoch(minutes))
            .map_err(|e| SatError::Propagation(e.to_string()))?;
        Ok(prediction.position)
    }
}

/// Topocentric (alt_deg, az_deg, dist_km) of a TEME position at TT.
pub fn altaz_from_teme(
    r_teme_km: [f64; 3],
    jd_tt: f64,
    geom: &ObserverGeometry,
) -> (f64, f64, f64) {
    let jd_ut1 = time::tt_to_ut1(jd_tt);
    let (theta, _rate) = time::theta_gmst1982(jd_ut1);
    let (s, c) = theta.sin_cos();
    // rot_z(-theta): TEME -> PEF (== ECEF with zero polar motion).
    let xe = c * r_teme_km[0] + s * r_teme_km[1];
    let ye = -s * r_teme_km[0] + c * r_teme_km[1];
    let ze = r_teme_km[2];

    let rho = [
        xe - geom.pos_km[0],
        ye - geom.pos_km[1],
        ze - geom.pos_km[2],
    ];
    let dist = (rho[0] * rho[0] + rho[1] * rho[1] + rho[2] * rho[2]).sqrt();
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let e = dot(&rho, &geom.east);
    let n = dot(&rho, &geom.north);
    let u = dot(&rho, &geom.up);
    let alt = (u / dist).clamp(-1.0, 1.0).asin().to_degrees();
    let az = e.atan2(n).to_degrees().rem_euclid(360.0);
    (alt, az, dist)
}

/// Convenience: satellite topocentric alt/az/range at one TT instant.
pub fn satellite_altaz(
    sat: &Satellite,
    jd_tt: f64,
    geom: &ObserverGeometry,
) -> Result<(f64, f64, f64), SatError> {
    let r = sat.position_teme_km(jd_tt)?;
    Ok(altaz_from_teme(r, jd_tt, geom))
}

/// Shortest-arc azimuth difference in degrees (port of
/// trajectory._unwrap_az_diff).
pub fn unwrap_az_diff(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

/// One satellite's trajectory over `times_tt` in the app's canonical
/// 8-column format: [time_tt, alt, az, dist_km, px, py, az_rate, el_rate].
/// Rates use the same central/forward/backward finite differences and
/// azimuth unwrapping as trajectory._compute_one_trajectory; px/py are the
/// polar skyplot pixel coordinates.
pub fn compute_trajectory_rows(
    sat: &Satellite,
    times_tt: &[f64],
    geom: &ObserverGeometry,
    cx: f64,
    cy: f64,
    radius: f64,
) -> Result<Vec<[f64; 8]>, SatError> {
    let n = times_tt.len();
    let mut alt = vec![0.0; n];
    let mut az = vec![0.0; n];
    let mut dist = vec![0.0; n];
    for i in 0..n {
        let (a, z, d) = satellite_altaz(sat, times_tt[i], geom)?;
        alt[i] = a;
        az[i] = z;
        dist[i] = d;
    }

    let dt_seconds = if n > 1 {
        (times_tt[1] - times_tt[0]) * time::DAY_S
    } else {
        1.0
    };
    let mut az_rate = vec![0.0; n];
    let mut el_rate = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        az_rate[i] = unwrap_az_diff(az[i + 1] - az[i - 1]) / (2.0 * dt_seconds);
        el_rate[i] = (alt[i + 1] - alt[i - 1]) / (2.0 * dt_seconds);
    }
    if n > 1 {
        az_rate[0] = unwrap_az_diff(az[1] - az[0]) / dt_seconds;
        az_rate[n - 1] = unwrap_az_diff(az[n - 1] - az[n - 2]) / dt_seconds;
        el_rate[0] = (alt[1] - alt[0]) / dt_seconds;
        el_rate[n - 1] = (alt[n - 1] - alt[n - 2]) / dt_seconds;
    }

    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let az_rad = (az[i].rem_euclid(360.0)).to_radians();
        let polar_radius = (90.0 - alt[i]) / 90.0 * radius;
        rows.push([
            times_tt[i],
            alt[i],
            az[i],
            dist[i],
            cx + polar_radius * az_rad.sin(),
            cy - polar_radius * az_rad.cos(),
            az_rate[i],
            el_rate[i],
        ]);
    }
    Ok(rows)
}

/// Coarse visibility gate over many satellites (port of
/// trajectory._batched_visibility_mask, parallelized with rayon): true if
/// the satellite rises above `min_alt_deg` at any sample time. Propagation
/// failures (decayed orbits) are treated as not visible.
pub fn visibility_mask(
    sats: &[Satellite],
    times_tt: &[f64],
    observer: &Observer,
    min_alt_deg: f64,
) -> Vec<bool> {
    use rayon::prelude::*;
    let geom = observer.geometry();
    sats.par_iter()
        .map(|sat| {
            times_tt.iter().any(|&tt| {
                matches!(satellite_altaz(sat, tt, &geom),
                         Ok((alt, _, _)) if alt > min_alt_deg)
            })
        })
        .collect()
}
