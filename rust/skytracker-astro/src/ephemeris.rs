//! Solar-system bodies from a JPL SPK file (de421.bsp): apparent
//! topocentric places, replacing skyfield's
//! `(earth + site).at(t).observe(body).apparent()` chain for celestial.py.

use crate::apparent::{self, add_aberration, FrameContext, AU_KM, C_AUDAY};
use crate::frames::Vec3;
use crate::sgp4_pass::ObserverGeometry;
use crate::spk::{Spk, SpkError};
use crate::time;
use std::path::Path;

pub struct Ephemeris {
    spk: Spk,
}

/// NAIF id for a body name as celestial.py uses them. Accepts both bare
/// names ("moon", "jupiter barycenter") and celestial.py's selection keys
/// ("planet:Jupiter").
pub fn body_id(name: &str) -> Option<i32> {
    let lower = name.to_ascii_lowercase();
    let name = lower.strip_prefix("planet:").unwrap_or(&lower);
    Some(match name {
        "sun" => 10,
        "moon" => 301,
        "mercury" => 199,
        "venus" => 299,
        "mars" => 499,
        "earth" => 399,
        "jupiter" | "jupiter barycenter" => 5,
        "saturn" | "saturn barycenter" => 6,
        "uranus" | "uranus barycenter" => 7,
        "neptune" | "neptune barycenter" => 8,
        "pluto" | "pluto barycenter" => 9,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::body_id;

    #[test]
    fn celestial_selection_keys_resolve() {
        for key in [
            "sun", "moon", "planet:Mercury", "planet:Venus", "planet:Mars",
            "planet:Jupiter", "planet:Saturn", "planet:Uranus",
            "planet:Neptune", "planet:Pluto",
        ] {
            assert!(body_id(key).is_some(), "unresolved key {key}");
        }
    }
}

pub struct ApparentPlace {
    /// Aberrated ICRS direction from the observer, AU.
    pub position_au: Vec3,
    /// One-way light time, days.
    pub light_time_days: f64,
}

impl Ephemeris {
    pub fn open(path: &Path) -> Result<Self, SpkError> {
        Ok(Ephemeris {
            spk: Spk::open(path)?,
        })
    }

    /// Barycentric state of the observing site (km, km/s, ICRS axes).
    /// Public because the star path (stars.rs) shares it.
    pub fn observer_ssb_state(
        &self,
        ctx: &FrameContext,
        geom: Option<&ObserverGeometry>,
    ) -> Result<(Vec3, Vec3), SpkError> {
        let (earth_p, earth_v) = self.spk.ssb_state(399, ctx.jd_tdb)?;
        match geom {
            None => Ok((earth_p, earth_v)),
            Some(g) => {
                let (op, ov) = ctx.observer_gcrs(g);
                Ok((
                    [earth_p[0] + op[0], earth_p[1] + op[1], earth_p[2] + op[2]],
                    [earth_v[0] + ov[0], earth_v[1] + ov[1], earth_v[2] + ov[2]],
                ))
            }
        }
    }

    /// Apparent place of a body (light-time + aberration; no deflection)
    /// as seen from the site (or the geocenter if `geom` is None).
    pub fn apparent(
        &self,
        target: i32,
        ctx: &FrameContext,
        geom: Option<&ObserverGeometry>,
    ) -> Result<ApparentPlace, SpkError> {
        let (obs_p_km, obs_v_kms) = self.observer_ssb_state(ctx, geom)?;

        // Light-time iteration on the barycentric target position.
        let mut lt_days = 0.0;
        let mut p_km = [0.0; 3];
        for _ in 0..4 {
            let (tp, _tv) = self.spk.ssb_state(target, ctx.jd_tdb - lt_days)?;
            for i in 0..3 {
                p_km[i] = tp[i] - obs_p_km[i];
            }
            lt_days = apparent::norm(&p_km) / AU_KM / C_AUDAY;
        }

        let mut p_au = [p_km[0] / AU_KM, p_km[1] / AU_KM, p_km[2] / AU_KM];
        let obs_v_auday = [
            obs_v_kms[0] * time::DAY_S / AU_KM,
            obs_v_kms[1] * time::DAY_S / AU_KM,
            obs_v_kms[2] * time::DAY_S / AU_KM,
        ];
        add_aberration(&mut p_au, &obs_v_auday, lt_days);
        Ok(ApparentPlace {
            position_au: p_au,
            light_time_days: lt_days,
        })
    }

    /// Apparent topocentric (alt_deg, az_deg, ra_icrs_hours, dec_icrs_deg)
    /// of a body — the exact quantities celestial.py reads off skyfield.
    pub fn apparent_altaz_radec(
        &self,
        target: i32,
        jd_tt: f64,
        geom: &ObserverGeometry,
    ) -> Result<(f64, f64, f64, f64), SpkError> {
        let ctx = FrameContext::new(jd_tt);
        let place = self.apparent(target, &ctx, Some(geom))?;
        let (alt, az) = ctx.altaz_from_icrs(&place.position_au, geom);
        let (ra, dec) = apparent::radec_of(&place.position_au);
        Ok((alt, az, ra, dec))
    }
}
