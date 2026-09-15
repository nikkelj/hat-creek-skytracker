//! Hipparcos star catalog: hip_main.dat parsing and apparent places —
//! ports skyfield's starlib model (position + proper motion + parallax,
//! aberration) for star_catalog.py.

use crate::apparent::{self, add_aberration, light_time_difference, FrameContext, AU_KM, C_AUDAY};
use crate::ephemeris::Ephemeris;
use crate::frames::{self, Vec3};
use crate::sgp4_pass::ObserverGeometry;
use crate::spk::SpkError;
use crate::time;
use std::io::BufRead;
use std::path::Path;

/// Hipparcos epoch J1991.25 as a TT/TDB Julian date, matching skyfield's
/// `1721045.0 + epoch_year * 365.25`.
pub const HIP_EPOCH_JD: f64 = 1721045.0 + 1991.25 * 365.25;

#[derive(Clone, Debug)]
pub struct Star {
    pub hip: i64,
    pub magnitude: f64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// mu_alpha* (includes cos-delta), mas/yr — catalog convention.
    pub pm_ra_mas_yr: f64,
    pub pm_dec_mas_yr: f64,
    pub parallax_mas: f64,
}

/// Parse hip_main.dat (CDS I/239, pipe-delimited). Rows without a valid
/// position or magnitude are skipped, matching star_catalog.py's cleanup.
pub fn parse_hip_main(path: &Path) -> std::io::Result<Vec<Star>> {
    let f = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::with_capacity(120_000);
    for line in reader.lines() {
        let line = line?;
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() < 14 {
            continue;
        }
        let parse = |i: usize| fields[i].trim().parse::<f64>().ok();
        let (Some(hip), Some(mag), Some(ra), Some(dec)) = (
            fields[1].trim().parse::<i64>().ok(),
            parse(5),
            parse(8),
            parse(9),
        ) else {
            continue;
        };
        out.push(Star {
            hip,
            magnitude: mag,
            ra_deg: ra,
            dec_deg: dec,
            parallax_mas: parse(11).unwrap_or(0.0),
            pm_ra_mas_yr: parse(12).unwrap_or(0.0),
            pm_dec_mas_yr: parse(13).unwrap_or(0.0),
        });
    }
    Ok(out)
}

/// Barycentric position (AU) and velocity (AU/day) of the star, port of
/// starlib._compute_vectors (radial velocity 0, as the app's catalog has).
fn star_vectors(star: &Star) -> (Vec3, Vec3) {
    let parallax_mas = if star.parallax_mas <= 0.0 || star.parallax_mas.is_nan() {
        1.0e-6
    } else {
        star.parallax_mas
    };
    let dist = 1.0 / (parallax_mas * 1.0e-3 * time::ASEC2RAD).sin();
    let r = star.ra_deg.to_radians();
    let d = star.dec_deg.to_radians();
    let (sra, cra) = r.sin_cos();
    let (sdc, cdc) = d.sin_cos();
    let position = [dist * cdc * cra, dist * cdc * sra, dist * sdc];
    // k = 1 with zero radial velocity.
    let pmr = star.pm_ra_mas_yr / (parallax_mas * 365.25);
    let pmd = star.pm_dec_mas_yr / (parallax_mas * 365.25);
    let velocity = [
        -pmr * sra - pmd * sdc * cra,
        pmr * cra - pmd * sdc * sra,
        pmd * cdc,
    ];
    (position, velocity)
}

pub struct StarApparent {
    pub ra_apparent_hours: f64,
    pub dec_apparent_deg: f64,
    pub alt_deg: f64,
    pub az_deg: f64,
}

/// Apparent place of a star for a ground observer: proper motion to date,
/// parallax (via the finite catalog distance), aberration, then of-date
/// RA/Dec (radec(epoch='date')) and topocentric alt/az — the quantities
/// star_catalog.py reads off skyfield.
pub fn star_apparent(
    star: &Star,
    eph: &Ephemeris,
    jd_tt: f64,
    geom: &ObserverGeometry,
) -> Result<StarApparent, SpkError> {
    let ctx = FrameContext::new(jd_tt);
    let (obs_p_km, obs_v_kms) = eph.observer_ssb_state(&ctx, Some(geom))?;
    let obs_p_au = [
        obs_p_km[0] / AU_KM,
        obs_p_km[1] / AU_KM,
        obs_p_km[2] / AU_KM,
    ];
    let obs_v_auday = [
        obs_v_kms[0] * time::DAY_S / AU_KM,
        obs_v_kms[1] * time::DAY_S / AU_KM,
        obs_v_kms[2] * time::DAY_S / AU_KM,
    ];

    let (pos0, vel) = star_vectors(star);
    // starlib._observe_from_bcrs: advance by proper motion to (tdb + dt).
    let dt = light_time_difference(&pos0, &obs_p_au);
    let dt_days = ctx.jd_tdb + dt - HIP_EPOCH_JD;
    let mut vector = [
        pos0[0] + vel[0] * dt_days - obs_p_au[0],
        pos0[1] + vel[1] * dt_days - obs_p_au[1],
        pos0[2] + vel[2] * dt_days - obs_p_au[2],
    ];
    let light_time = apparent::norm(&vector) / C_AUDAY;
    add_aberration(&mut vector, &obs_v_auday, light_time);

    let (alt, az) = ctx.altaz_from_icrs(&vector, geom);
    // radec(epoch='date'): rotate the apparent ICRS vector by M.
    let of_date = frames::mat_vec(&ctx.m, &vector);
    let (ra, dec) = apparent::radec_of(&of_date);
    Ok(StarApparent {
        ra_apparent_hours: ra,
        dec_apparent_deg: dec,
        alt_deg: alt,
        az_deg: az,
    })
}
