//! Shared apparent-place machinery: observer barycentric state, the NOVAS
//! aberration correction (port of skyfield.relativity.add_aberration), and
//! ICRS-vector -> topocentric alt/az / RA-Dec conversions.
//!
//! Light deflection is intentionally omitted: it exceeds 0.05" only within
//! a few degrees of the Sun's limb (max 1.75"), far inside the 60" gate,
//! and the tracker never points there.

use crate::frames::{self, Mat3, Vec3};
use crate::sgp4_pass::ObserverGeometry;
use crate::time;

pub const AU_KM: f64 = 149_597_870.700;
pub const C_M_S: f64 = 299_792_458.0;
pub const C_AUDAY: f64 = C_M_S * time::DAY_S / (AU_KM * 1000.0);
/// Earth rotation rate, rad/s (skyfield earthlib.ANGVEL).
pub const ANGVEL: f64 = 7.2921150e-5;

pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: &Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// Standard CCW rotation about z: rot_z(a) * v rotates v by +a.
pub fn rot_z(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Precomputed per-instant rotation context.
pub struct FrameContext {
    pub jd_tt: f64,
    pub jd_tdb: f64,
    /// ICRS -> true equator/equinox of date (skyfield's Time.M).
    pub m: Mat3,
    pub gast_rad: f64,
}

impl FrameContext {
    pub fn new(jd_tt: f64) -> Self {
        FrameContext {
            jd_tt,
            jd_tdb: jd_tt + time::tdb_minus_tt(jd_tt) / time::DAY_S,
            m: frames::icrs_to_true_of_date(jd_tt),
            gast_rad: time::gast_hours(jd_tt) / 24.0 * time::TAU,
        }
    }

    /// GCRS/ICRS-axes position (km) and velocity (km/s) of a ground
    /// observer relative to the geocenter: ITRS -> GCRS is M^T rot_z(gast),
    /// and the velocity is Earth rotation (omega x r) carried through.
    pub fn observer_gcrs(&self, geom: &ObserverGeometry) -> (Vec3, Vec3) {
        let r_itrs = geom.pos_km;
        let v_itrs = [
            -ANGVEL * r_itrs[1],
            ANGVEL * r_itrs[0],
            0.0,
        ];
        let rz = rot_z(self.gast_rad);
        let mt = frames::transpose(&self.m);
        let pos = frames::mat_vec(&mt, &frames::mat_vec(&rz, &r_itrs));
        let vel = frames::mat_vec(&mt, &frames::mat_vec(&rz, &v_itrs));
        (pos, vel)
    }

    /// ICRS direction vector -> topocentric (alt_deg, az_deg):
    /// d_ITRS = rot_z(-gast) . M . d_ICRS, then the observer's ENU basis.
    /// Topocentric (alt, az, range_km) of a geocentric POSITION vector (km,
    /// ICRS/J2000 axes): rotate to ITRS, subtract the observer, project onto
    /// ENU. The workhorse for ephemeris-file tracking (Starlink MEME states).
    pub fn altaz_range_of_position(&self, r_icrs_km: &Vec3, geom: &ObserverGeometry) -> (f64, f64, f64) {
        let rz = rot_z(-self.gast_rad);
        let itrs = frames::mat_vec(&rz, &frames::mat_vec(&self.m, r_icrs_km));
        let d = [itrs[0] - geom.pos_km[0], itrs[1] - geom.pos_km[1], itrs[2] - geom.pos_km[2]];
        let e = dot(&d, &geom.east);
        let n = dot(&d, &geom.north);
        let u = dot(&d, &geom.up);
        let r = norm(&d);
        let alt = (u / r).clamp(-1.0, 1.0).asin().to_degrees();
        let az = e.atan2(n).to_degrees().rem_euclid(360.0);
        (alt, az, r)
    }

    pub fn altaz_from_icrs(&self, p_icrs: &Vec3, geom: &ObserverGeometry) -> (f64, f64) {
        let rz = rot_z(-self.gast_rad);
        let d_itrs = frames::mat_vec(&rz, &frames::mat_vec(&self.m, p_icrs));
        let e = dot(&d_itrs, &geom.east);
        let n = dot(&d_itrs, &geom.north);
        let u = dot(&d_itrs, &geom.up);
        let r = norm(&d_itrs);
        let alt = (u / r).clamp(-1.0, 1.0).asin().to_degrees();
        let az = e.atan2(n).to_degrees().rem_euclid(360.0);
        (alt, az)
    }
}

/// (ra_hours, dec_deg) of a vector in whatever equatorial frame it is
/// expressed in.
pub fn radec_of(v: &Vec3) -> (f64, f64) {
    let r = norm(v);
    let ra = v[1].atan2(v[0]).rem_euclid(time::TAU) / time::TAU * 24.0;
    let dec = (v[2] / r).clamp(-1.0, 1.0).asin().to_degrees();
    (ra, dec)
}

/// NOVAS aberration of light (port of skyfield.relativity.add_aberration):
/// `position` AU (observer-relative), `velocity` AU/day (observer
/// barycentric), `light_time` days. Modifies position in place.
pub fn add_aberration(position: &mut Vec3, velocity: &Vec3, light_time: f64) {
    const AVOID_DIVIDE_BY_ZERO: f64 = 1e-80;
    let p1mag = light_time * C_AUDAY;
    let vemag = norm(velocity);
    let beta = vemag / C_AUDAY;
    let d = dot(position, velocity);
    let cosd = d / (p1mag * vemag + AVOID_DIVIDE_BY_ZERO);
    let gammai = (1.0 - beta * beta).sqrt();
    let p = beta * cosd;
    let q = (1.0 + p / (1.0 + gammai)) * light_time;
    let r = 1.0 + p;
    for i in 0..3 {
        position[i] = (position[i] * gammai + q * velocity[i]) / r;
    }
}

/// Star-observer light-time difference relative to the SSB, days
/// (port of skyfield.relativity.light_time_difference).
pub fn light_time_difference(position_au: &Vec3, observer_position_au: &Vec3) -> f64 {
    const AVOID_DIVIDE_BY_ZERO: f64 = 1e-80;
    let dis = norm(position_au);
    let mut u1 = *position_au;
    for v in u1.iter_mut() {
        *v /= dis + AVOID_DIVIDE_BY_ZERO;
    }
    dot(&u1, observer_position_au) / C_AUDAY
}
