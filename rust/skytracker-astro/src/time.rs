//! Timescales: UTC/TT/UT1/TDB conversions, ERA, GMST, GMST1982, GAST.
//!
//! Faithful port of the skyfield formulas the goldens were recorded with
//! (earthlib.sidereal_time / earth_rotation_angle, sgp4lib.theta_GMST1982,
//! timelib.tdb_minus_tt) plus a leap-second table and the delta-T curve
//! sampled from skyfield's model in `generated_tables.rs`.
//!
//! Convention notes (these are skyfield's, and therefore the app's):
//! - `delta_t` = TT - UT1 seconds; UT1 = TT - delta_t.
//! - SGP4 propagation and the TEME->PEF rotation both take UT1, matching
//!   skyfield's satellite pipeline (it feeds `t.ut1` into sgp4).

use crate::generated_tables as gt;

pub const T0: f64 = 2451545.0; // J2000.0 as a TT Julian date
pub const DAY_S: f64 = 86400.0;
pub const TAU: f64 = std::f64::consts::TAU;
pub const ASEC2RAD: f64 = 4.848136811095359935899141e-6;
pub const ASEC360: f64 = 1296000.0;

/// Leap seconds: (first UTC Julian date it applies, TAI-UTC seconds).
/// TT = UTC + (TAI-UTC) + 32.184.
static LEAP_SECONDS: &[(f64, f64)] = &[
    (julian_date_const(1972, 1, 1), 10.0),
    (julian_date_const(1972, 7, 1), 11.0),
    (julian_date_const(1973, 1, 1), 12.0),
    (julian_date_const(1974, 1, 1), 13.0),
    (julian_date_const(1975, 1, 1), 14.0),
    (julian_date_const(1976, 1, 1), 15.0),
    (julian_date_const(1977, 1, 1), 16.0),
    (julian_date_const(1978, 1, 1), 17.0),
    (julian_date_const(1979, 1, 1), 18.0),
    (julian_date_const(1980, 1, 1), 19.0),
    (julian_date_const(1981, 7, 1), 20.0),
    (julian_date_const(1982, 7, 1), 21.0),
    (julian_date_const(1983, 7, 1), 22.0),
    (julian_date_const(1985, 7, 1), 23.0),
    (julian_date_const(1988, 1, 1), 24.0),
    (julian_date_const(1990, 1, 1), 25.0),
    (julian_date_const(1991, 1, 1), 26.0),
    (julian_date_const(1992, 7, 1), 27.0),
    (julian_date_const(1993, 7, 1), 28.0),
    (julian_date_const(1994, 7, 1), 29.0),
    (julian_date_const(1996, 1, 1), 30.0),
    (julian_date_const(1997, 7, 1), 31.0),
    (julian_date_const(1999, 1, 1), 32.0),
    (julian_date_const(2006, 1, 1), 33.0),
    (julian_date_const(2009, 1, 1), 34.0),
    (julian_date_const(2012, 7, 1), 35.0),
    (julian_date_const(2015, 7, 1), 36.0),
    (julian_date_const(2017, 1, 1), 37.0),
];

/// Julian date at 00:00 UTC for a Gregorian calendar date (midnight, so the
/// value ends in .5). Fliegel & Van Flandern algorithm, valid for all
/// Gregorian dates of interest.
const fn julian_date_const(y: i64, m: i64, d: i64) -> f64 {
    let a = (14 - m) / 12;
    let yy = y + 4800 - a;
    let mm = m + 12 * a - 3;
    let jdn = d + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32045;
    jdn as f64 - 0.5
}

/// Julian date for a Gregorian date + time-of-day (runtime variant).
pub fn julian_date(y: i64, m: i64, d: i64, hour: f64, minute: f64, second: f64) -> f64 {
    julian_date_const(y, m, d) + (hour + minute / 60.0 + second / 3600.0) / 24.0
}

/// TAI-UTC seconds in force at the given UTC Julian date.
pub fn tai_minus_utc(jd_utc: f64) -> f64 {
    let mut out = LEAP_SECONDS[0].1;
    for &(jd, s) in LEAP_SECONDS {
        if jd_utc >= jd {
            out = s;
        } else {
            break;
        }
    }
    out
}

/// UTC Julian date -> TT Julian date.
pub fn utc_to_tt(jd_utc: f64) -> f64 {
    jd_utc + (tai_minus_utc(jd_utc) + 32.184) / DAY_S
}

/// TT Julian date -> UTC Julian date (leap lookup on the TT value is fine:
/// no leap second lands within 70 s of another).
pub fn tt_to_utc(jd_tt: f64) -> f64 {
    jd_tt - (tai_minus_utc(jd_tt) + 32.184) / DAY_S
}

/// delta_t = TT - UT1 in seconds, linearly interpolated from the curve
/// sampled off skyfield's model (monthly, 2015-2045); clamped outside.
pub fn delta_t(jd_tt: f64) -> f64 {
    let table = &gt::DELTA_T_TABLE;
    let pos = (jd_tt - gt::DELTA_T_TT0_JD) / gt::DELTA_T_STEP_DAYS;
    if pos <= 0.0 {
        return table[0];
    }
    let i = pos.floor() as usize;
    if i + 1 >= table.len() {
        return table[table.len() - 1];
    }
    let f = pos - i as f64;
    table[i] * (1.0 - f) + table[i + 1] * f
}

/// TT -> UT1 Julian date.
pub fn tt_to_ut1(jd_tt: f64) -> f64 {
    jd_tt - delta_t(jd_tt) / DAY_S
}

/// TDB - TT in seconds (USNO Circular 179 eq. 2.6; skyfield timelib port).
pub fn tdb_minus_tt(jd_tdb: f64) -> f64 {
    let t = (jd_tdb - T0) / 36525.0;
    0.001657 * (628.3076 * t + 6.2401).sin()
        + 0.000022 * (575.3385 * t + 4.2970).sin()
        + 0.000014 * (1256.6152 * t + 6.1969).sin()
        + 0.000005 * (606.9777 * t + 4.0212).sin()
        + 0.000005 * (52.9691 * t + 0.4444).sin()
        + 0.000002 * (21.3299 * t + 5.5431).sin()
        + 0.000010 * t * (628.3076 * t + 4.2490).sin()
}

/// Earth Rotation Angle as a fraction of a full rotation [0, 1).
/// (skyfield earthlib.earth_rotation_angle)
pub fn earth_rotation_angle(jd_ut1: f64) -> f64 {
    let th = 0.7790572732640 + 0.00273781191135448 * (jd_ut1 - T0);
    (th.rem_euclid(1.0) + jd_ut1.rem_euclid(1.0)).rem_euclid(1.0)
}

/// Greenwich Mean Sidereal Time in hours (skyfield earthlib.sidereal_time).
pub fn gmst_hours(jd_tt: f64) -> f64 {
    let jd_ut1 = tt_to_ut1(jd_tt);
    let theta = earth_rotation_angle(jd_ut1);
    let t = (jd_tt + tdb_minus_tt(jd_tt) / DAY_S - T0) / 36525.0;
    let st = 0.014506
        + ((((-0.0000000368 * t - 0.000029956) * t - 0.00000044) * t + 1.3915817) * t
            + 4612.156534)
            * t;
    (st / 54000.0 + theta * 24.0).rem_euclid(24.0)
}

/// GMST1982 angle theta (radians) and its rate (radians/day of UT1).
/// Defines the TEME -> PEF rotation used by the satellite path
/// (skyfield sgp4lib.theta_GMST1982, AIAA 2006-6753 Appendix C).
pub fn theta_gmst1982(jd_ut1: f64) -> (f64, f64) {
    let t = (jd_ut1 - T0) / 36525.0;
    let g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t;
    let dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t;
    let theta = (jd_ut1.rem_euclid(1.0) + (g / DAY_S).rem_euclid(1.0)).rem_euclid(1.0) * TAU;
    let theta_dot = (1.0 + dg / (DAY_S * 36525.0)) * TAU;
    (theta, theta_dot)
}

/// Greenwich Apparent Sidereal Time in hours (skyfield timelib.Time.gast):
/// GMST + equation of the equinoxes (IAU 2000A nutation in longitude times
/// the cosine of the mean obliquity, plus the complementary terms).
pub fn gast_hours(jd_tt: f64) -> f64 {
    let jd_tdb = jd_tt + tdb_minus_tt(jd_tt) / DAY_S;
    let t = (jd_tdb - T0) / 36525.0;
    let (d_psi, _d_eps) = crate::frames::iau2000a_radians(t);
    let eps_m = crate::frames::mean_obliquity_asec(jd_tdb) * ASEC2RAD;
    let c_terms = crate::frames::ee_complementary_terms_radians(jd_tt);
    let eq_eq = d_psi * eps_m.cos() + c_terms;
    (gmst_hours(jd_tt) + eq_eq / TAU * 24.0).rem_euclid(24.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_dates() {
        // J2000.0: 2000-01-01 12:00 TT
        assert_eq!(julian_date(2000, 1, 1, 12.0, 0.0, 0.0), T0);
        // Unix epoch midnight
        assert_eq!(julian_date(1970, 1, 1, 0.0, 0.0, 0.0), 2440587.5);
    }

    #[test]
    fn leap_seconds_current_era() {
        // Since 2017-01-01, TT-UTC = 69.184 s. (f64 granularity at
        // JD ~2.46e6 is ~4e-5 s, so the tolerance reflects representation,
        // not the model.)
        let jd = julian_date(2026, 8, 1, 0.0, 0.0, 0.0);
        assert!(((utc_to_tt(jd) - jd) * DAY_S - 69.184).abs() < 1e-4);
    }

    #[test]
    fn utc_tt_roundtrip() {
        let jd = julian_date(2026, 8, 15, 7.0, 58.0, 0.0);
        let tt = utc_to_tt(jd);
        assert!((tt_to_utc(tt) - jd).abs() * DAY_S < 1e-9);
    }
}
