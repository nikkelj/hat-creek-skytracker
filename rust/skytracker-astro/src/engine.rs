//! App-level facade over the astro modules: the API surface that
//! skytracker-ffi exposes to Python today and skytracker-app consumes
//! natively in Phase 7. Owns the TLE catalog and the DE421 ephemeris.

use crate::ephemeris::{body_id, Ephemeris};
use crate::passes::{self, Pass, PassParams};
use crate::sgp4_pass::{self, Observer, Satellite};
use crate::spk::SpkError;
use crate::stars;
use crate::time;
use crate::tle::TleCatalog;
use rayon::prelude::*;
use std::path::Path;

pub struct Engine {
    pub eph: Option<Ephemeris>,
    pub tles: Option<TleCatalog>,
}

#[derive(Debug)]
pub enum EngineError {
    Spk(SpkError),
    Sat(sgp4_pass::SatError),
    NoEphemeris,
    NoTles,
    UnknownKey(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Spk(e) => write!(f, "{e}"),
            EngineError::Sat(e) => write!(f, "{e}"),
            EngineError::NoEphemeris => write!(f, "no ephemeris loaded (de421.bsp)"),
            EngineError::NoTles => write!(f, "no TLE catalog loaded"),
            EngineError::UnknownKey(k) => write!(f, "unknown body/target key: {k}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<SpkError> for EngineError {
    fn from(e: SpkError) -> Self {
        EngineError::Spk(e)
    }
}

impl From<sgp4_pass::SatError> for EngineError {
    fn from(e: sgp4_pass::SatError) -> Self {
        EngineError::Sat(e)
    }
}

impl Engine {
    pub fn new(de421_path: Option<&Path>) -> Result<Self, EngineError> {
        let eph = match de421_path {
            Some(p) => Some(Ephemeris::open(p)?),
            None => None,
        };
        Ok(Engine { eph, tles: None })
    }

    pub fn load_tle_file(&mut self, path: &Path) -> Result<usize, EngineError> {
        let cat = TleCatalog::load(path)?;
        let n = cat.len();
        self.tles = Some(cat);
        Ok(n)
    }

    fn sat(&self, satnum: &str) -> Result<&Satellite, EngineError> {
        self.tles
            .as_ref()
            .ok_or(EngineError::NoTles)?
            .get(satnum)
            .ok_or_else(|| EngineError::UnknownKey(satnum.to_string()))
    }

    /// One satellite's canonical 8-column rows (bulk or fine grid).
    pub fn satellite_rows(
        &self,
        satnum: &str,
        times_tt: &[f64],
        observer: &Observer,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> Result<Vec<[f64; 8]>, EngineError> {
        let geom = observer.geometry();
        Ok(sgp4_pass::compute_trajectory_rows(
            self.sat(satnum)?,
            times_tt,
            &geom,
            cx,
            cy,
            radius,
        )?)
    }

    /// Bulk precompute for many satellites in parallel. Satellites that
    /// fail to propagate (decayed) are silently omitted, matching the
    /// Python path's behavior of skipping errored satellites.
    pub fn precompute(
        &self,
        satnums: &[String],
        times_tt: &[f64],
        observer: &Observer,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> Result<Vec<(String, Vec<[f64; 8]>)>, EngineError> {
        let tles = self.tles.as_ref().ok_or(EngineError::NoTles)?;
        let geom = observer.geometry();
        Ok(satnums
            .par_iter()
            .filter_map(|sn| {
                let sat = tles.get(sn)?;
                let rows =
                    sgp4_pass::compute_trajectory_rows(sat, times_tt, &geom, cx, cy, radius)
                        .ok()?;
                Some((sn.clone(), rows))
            })
            .collect())
    }

    /// Coarse visibility gate: which of `satnums` rise above `min_alt_deg`
    /// at any of the sample times.
    pub fn visible_satnums(
        &self,
        satnums: &[String],
        times_tt: &[f64],
        observer: &Observer,
        min_alt_deg: f64,
    ) -> Result<Vec<String>, EngineError> {
        let tles = self.tles.as_ref().ok_or(EngineError::NoTles)?;
        let geom = observer.geometry();
        Ok(satnums
            .par_iter()
            .filter(|sn| {
                let Some(sat) = tles.get(sn) else { return false };
                times_tt.iter().any(|&tt| {
                    matches!(sgp4_pass::satellite_altaz(sat, tt, &geom),
                             Ok((alt, _, _)) if alt > min_alt_deg)
                })
            })
            .cloned()
            .collect())
    }

    /// Geocentric sun position (km) on the true equator and equinox of
    /// date — within the equation of the equinoxes (~20") of TEME, the
    /// frame the pass magnitude/eclipse test works in. Light-time and
    /// aberration corrected (the ~20" aberration is irrelevant there).
    pub fn sun_tod_km(&self, jd_tt: f64) -> Option<[f64; 3]> {
        let eph = self.eph.as_ref()?;
        let ctx = crate::apparent::FrameContext::new(jd_tt);
        let place = eph.apparent(10, &ctx, None).ok()?;
        let p_km = [
            place.position_au[0] * crate::apparent::AU_KM,
            place.position_au[1] * crate::apparent::AU_KM,
            place.position_au[2] * crate::apparent::AU_KM,
        ];
        Some(crate::frames::mat_vec(&ctx.m, &p_km))
    }

    /// Pass prediction for `satnums` (unknown keys skipped), sorted by AOS.
    /// The sun vector for the brightness/eclipse estimate comes from the
    /// loaded DE421 ephemeris; without one every `est_mag` is `None`
    /// (`eclipsed_at_tca` false = "not computed").
    pub fn predict_passes(
        &self,
        satnums: &[String],
        observer: &Observer,
        jd_tt_start: f64,
        params: &PassParams,
    ) -> Vec<Pass> {
        let Some(tles) = self.tles.as_ref() else {
            return Vec::new();
        };
        let sats: Vec<&Satellite> = satnums.iter().filter_map(|sn| tles.get(sn)).collect();
        let sun = |jd_tt: f64| self.sun_tod_km(jd_tt);
        passes::predict_passes(&sats, observer, jd_tt_start, params, &sun)
    }

    /// Apparent topocentric (alt, az, dist_km) of a solar-system body at
    /// one instant — celestial.solar_system_altaz's per-body quantity.
    pub fn body_altaz_dist(
        &self,
        name: &str,
        jd_tt: f64,
        observer: &Observer,
    ) -> Result<(f64, f64, f64), EngineError> {
        let eph = self.eph.as_ref().ok_or(EngineError::NoEphemeris)?;
        let id = body_id(name).ok_or_else(|| EngineError::UnknownKey(name.to_string()))?;
        let ctx = crate::apparent::FrameContext::new(jd_tt);
        let geom = observer.geometry();
        let place = eph.apparent(id, &ctx, Some(&geom))?;
        let (alt, az) = ctx.altaz_from_icrs(&place.position_au, &geom);
        let dist_km = crate::apparent::norm(&place.position_au) * crate::apparent::AU_KM;
        Ok((alt, az, dist_km))
    }

    /// 8-column tracking rows for a body or fixed ICRS RA/Dec target —
    /// celestial.build_trajectory's contract (px/py zero, same rate
    /// scheme, dist_km from the apparent range).
    pub fn target_rows(
        &self,
        target: &TrackTarget,
        times_tt: &[f64],
        observer: &Observer,
    ) -> Result<Vec<[f64; 8]>, EngineError> {
        let eph = self.eph.as_ref().ok_or(EngineError::NoEphemeris)?;
        let geom = observer.geometry();
        let n = times_tt.len();
        let mut alt = vec![0.0; n];
        let mut az = vec![0.0; n];
        let mut dist = vec![0.0; n];
        for (i, &jd_tt) in times_tt.iter().enumerate() {
            let (a, z, d) = match target {
                TrackTarget::Body(name) => self.body_altaz_dist(name, jd_tt, observer)?,
                TrackTarget::FixedRadec { ra_deg, dec_deg } => {
                    let star = stars::Star {
                        hip: 0,
                        magnitude: 0.0,
                        ra_deg: *ra_deg,
                        dec_deg: *dec_deg,
                        pm_ra_mas_yr: 0.0,
                        pm_dec_mas_yr: 0.0,
                        parallax_mas: 0.0,
                    };
                    let app = stars::star_apparent(&star, eph, jd_tt, &geom)?;
                    // Fixed targets have no meaningful range; Python stores
                    // skyfield's ~gigaparsec placeholder. Use 0.0: nothing
                    // reads the dist column for fixed targets.
                    (app.alt_deg, app.az_deg, 0.0)
                }
            };
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
            az_rate[i] = sgp4_pass::unwrap_az_diff(az[i + 1] - az[i - 1]) / (2.0 * dt_seconds);
            el_rate[i] = (alt[i + 1] - alt[i - 1]) / (2.0 * dt_seconds);
        }
        if n > 1 {
            az_rate[0] = sgp4_pass::unwrap_az_diff(az[1] - az[0]) / dt_seconds;
            az_rate[n - 1] = sgp4_pass::unwrap_az_diff(az[n - 1] - az[n - 2]) / dt_seconds;
            el_rate[0] = (alt[1] - alt[0]) / dt_seconds;
            el_rate[n - 1] = (alt[n - 1] - alt[n - 2]) / dt_seconds;
        }

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push([
                times_tt[i],
                alt[i],
                az[i],
                dist[i],
                0.0,
                0.0,
                az_rate[i],
                el_rate[i],
            ]);
        }
        Ok(rows)
    }
}

pub enum TrackTarget {
    Body(String),
    FixedRadec { ra_deg: f64, dec_deg: f64 },
}
