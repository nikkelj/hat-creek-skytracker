//! Precession, nutation, frame bias, and the equatorial true-of-date
//! transformation — faithful ports of skyfield's nutationlib and
//! precessionlib, using the coefficient tables generated verbatim from
//! skyfield into `generated_tables.rs`.
//!
//! Matrix convention matches skyfield: `M = N · P · B` maps ICRS/GCRS
//! vectors into the true equator and equinox of date. Apply `transpose`
//! for the reverse direction.

use crate::generated_tables as gt;
use crate::time::{ASEC2RAD, ASEC360, DAY_S, T0, TAU};

pub type Mat3 = [[f64; 3]; 3];
pub type Vec3 = [f64; 3];

// 0.1 microarcsecond in radians: the unit of the IAU2000A series output.
const TENTH_USEC_2_RAD: f64 = ASEC2RAD * 1e-7;

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

pub fn mat_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn transpose(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Fundamental arguments (Simon et al. 1994) in radians, at TDB centuries t.
/// (skyfield nutationlib.fundamental_arguments, all 5 polynomial terms)
pub fn fundamental_arguments(t: f64) -> [f64; 5] {
    let mut a = [0.0; 5];
    for i in 0..5 {
        // Horner from t^4 down to t^0: FA_POLY[k][i] is the t^k coefficient.
        let mut v = gt::FA_POLY[4][i] * t;
        v = (v + gt::FA_POLY[3][i]) * t;
        v = (v + gt::FA_POLY[2][i]) * t;
        v = (v + gt::FA_POLY[1][i]) * t;
        v += gt::FA_POLY[0][i];
        a[i] = (v % ASEC360) * ASEC2RAD;
    }
    a
}

/// IAU 2000A nutation: (delta_psi, delta_eps) in radians at TDB centuries t.
/// Full series: 678 luni-solar + 687 planetary terms, exactly as skyfield
/// evaluates them (nutationlib.iau2000a).
pub fn iau2000a_radians(t: f64) -> (f64, f64) {
    let a = fundamental_arguments(t);

    let mut dpsi = 0.0;
    let mut deps = 0.0;
    for (row, (lon, obl)) in gt::NALS_T
        .iter()
        .zip(gt::LUNISOLAR_LONGITUDE.iter().zip(gt::LUNISOLAR_OBLIQUITY.iter()))
    {
        let mut arg = 0.0;
        for k in 0..5 {
            arg += row[k] as f64 * a[k];
        }
        let (s, c) = arg.sin_cos();
        dpsi += s * lon[0] + s * lon[1] * t + c * lon[2];
        deps += c * obl[0] + c * obl[1] * t + s * obl[2];
    }

    // Planetary components.
    let mut ap = [0.0; 14];
    for i in 0..14 {
        ap[i] = gt::ANOMALY_CONSTANT[i] + gt::ANOMALY_COEFFICIENT[i] * t;
    }
    ap[13] *= t;

    for (row, (lon, obl)) in gt::NAPL_T
        .iter()
        .zip(gt::PLANETARY_LONGITUDE.iter().zip(gt::PLANETARY_OBLIQUITY.iter()))
    {
        let mut arg = 0.0;
        for k in 0..14 {
            arg += row[k] as f64 * ap[k];
        }
        let (s, c) = arg.sin_cos();
        dpsi += s * lon[0] + c * lon[1];
        deps += s * obl[0] + c * obl[1];
    }

    (dpsi * TENTH_USEC_2_RAD, deps * TENTH_USEC_2_RAD)
}

/// Mean obliquity of the ecliptic in ARCSECONDS at a TDB Julian date
/// (skyfield nutationlib.mean_obliquity; Capitaine et al. 2003 eq. 37/39).
pub fn mean_obliquity_asec(jd_tdb: f64) -> f64 {
    let t = (jd_tdb - T0) / 36525.0;
    ((((-0.0000000434 * t - 0.000000576) * t + 0.00200340) * t - 0.0001831) * t - 46.836769)
        * t
        + 84381.406
}

/// Complementary terms of the equation of the equinoxes, radians, at TT
/// (skyfield nutationlib.equation_of_the_equinoxes_complimentary_terms).
pub fn ee_complementary_terms_radians(jd_tt: f64) -> f64 {
    let t = (jd_tt - T0) / 36525.0;

    let mut fa = [0.0_f64; 14];
    // Luni-solar arguments with the (k*t % 1)*tau fast-wrap convention.
    fa[0] = (485868.249036
        + (715923.2178 + (31.8792 + (0.051635 + (-0.00024470) * t) * t) * t) * t)
        * ASEC2RAD
        + (1325.0 * t).rem_euclid(1.0) * TAU;
    fa[1] = (1287104.793048
        + (1292581.0481 + (-0.5532 + (0.000136 + (-0.00001149) * t) * t) * t) * t)
        * ASEC2RAD
        + (99.0 * t).rem_euclid(1.0) * TAU;
    fa[2] = (335779.526232
        + (295262.8478 + (-12.7512 + (-0.001037 + (0.00000417) * t) * t) * t) * t)
        * ASEC2RAD
        + (1342.0 * t).rem_euclid(1.0) * TAU;
    fa[3] = (1072260.703692
        + (1105601.2090 + (-6.3706 + (0.006593 + (-0.00003169) * t) * t) * t) * t)
        * ASEC2RAD
        + (1236.0 * t).rem_euclid(1.0) * TAU;
    fa[4] = (450160.398036
        + (-482890.5431 + (7.4722 + (0.007702 + (-0.00005939) * t) * t) * t) * t)
        * ASEC2RAD
        + (-5.0 * t).rem_euclid(1.0) * TAU;
    fa[5] = 4.402608842 + 2608.7903141574 * t;
    fa[6] = 3.176146697 + 1021.3285546211 * t;
    fa[7] = 1.753470314 + 628.3075849991 * t;
    fa[8] = 6.203480913 + 334.0612426700 * t;
    fa[9] = 0.599546497 + 52.9690962641 * t;
    fa[10] = 0.874016757 + 21.3299104960 * t;
    fa[11] = 5.481293872 + 7.4781598567 * t;
    fa[12] = 5.311886287 + 3.8133035638 * t;
    fa[13] = (0.024381750 + 0.00000538691 * t) * t;
    for v in fa.iter_mut() {
        *v = v.rem_euclid(TAU);
    }

    // t^1 term.
    let mut a1 = 0.0;
    for k in 0..14 {
        a1 += gt::KE1[0][k] as f64 * fa[k];
    }
    let mut c_terms = (gt::SE1_0 * a1.sin() + gt::SE1_1 * a1.cos()) * t;

    // t^0 terms.
    for (row, (s0, s1)) in gt::KE0_T
        .iter()
        .zip(gt::SE0_T_0.iter().zip(gt::SE0_T_1.iter()))
    {
        let mut arg = 0.0;
        for k in 0..14 {
            arg += row[k] as f64 * fa[k];
        }
        c_terms += s0 * arg.sin() + s1 * arg.cos();
    }

    c_terms * ASEC2RAD
}

/// Precession matrix (J2000 mean equator -> mean of date) at a TDB Julian
/// date (skyfield precessionlib.compute_precession; Capitaine et al. 2003
/// 4-angle formulation).
pub fn precession_matrix(jd_tdb: f64) -> Mat3 {
    let eps0_asec = 84381.406;
    let t = (jd_tdb - T0) / 36525.0;

    let psia = ((((-0.0000000951 * t + 0.000132851) * t - 0.00114045) * t - 1.0790069) * t
        + 5038.481507)
        * t;
    let omegaa = ((((0.0000003337 * t - 0.000000467) * t - 0.00772503) * t + 0.0512623) * t
        - 0.025754)
        * t
        + eps0_asec;
    let chia = ((((-0.0000000560 * t + 0.000170663) * t - 0.00121197) * t - 2.3814292) * t
        + 10.556403)
        * t;

    let eps0 = eps0_asec * ASEC2RAD;
    let psia = psia * ASEC2RAD;
    let omegaa = omegaa * ASEC2RAD;
    let chia = chia * ASEC2RAD;

    let (sa, ca) = eps0.sin_cos();
    let (sb, cb) = (-psia).sin_cos();
    let (sc, cc) = (-omegaa).sin_cos();
    let (sd, cd) = chia.sin_cos();

    [
        [
            cd * cb - sb * sd * cc,
            cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
            cd * sb * sa + sd * cc * cb * sa + ca * sd * sc,
        ],
        [
            -sd * cb - sb * cd * cc,
            -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
            -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc,
        ],
        [
            sb * sc,
            -sc * cb * ca - sa * cc,
            -sc * cb * sa + cc * ca,
        ],
    ]
}

/// Nutation matrix (mean of date -> true of date)
/// (skyfield nutationlib.build_nutation_matrix).
pub fn nutation_matrix(mean_obliquity_rad: f64, true_obliquity_rad: f64, psi_rad: f64) -> Mat3 {
    let (sobm, cobm) = mean_obliquity_rad.sin_cos();
    let (sobt, cobt) = true_obliquity_rad.sin_cos();
    let (spsi, cpsi) = psi_rad.sin_cos();
    [
        [cpsi, -spsi * cobm, -spsi * sobm],
        [
            spsi * cobt,
            cpsi * cobm * cobt + sobm * sobt,
            cpsi * sobm * cobt - cobm * sobt,
        ],
        [
            spsi * sobt,
            cpsi * cobm * sobt - sobm * cobt,
            cpsi * sobm * sobt + cobm * cobt,
        ],
    ]
}

/// The ICRS frame-bias matrix B (skyfield framelib.ICRS_to_J2000).
pub fn frame_bias() -> Mat3 {
    let b = &gt::FRAME_BIAS_ICRS_TO_J2000;
    [
        [b[0][0], b[0][1], b[0][2]],
        [b[1][0], b[1][1], b[1][2]],
        [b[2][0], b[2][1], b[2][2]],
    ]
}

/// The full equatorial matrix `M = N · P · B` (ICRS -> true equator and
/// equinox of date) at a TT Julian date, matching skyfield's `Time.M`.
pub fn icrs_to_true_of_date(jd_tt: f64) -> Mat3 {
    let jd_tdb = jd_tt + crate::time::tdb_minus_tt(jd_tt) / DAY_S;
    let t = (jd_tdb - T0) / 36525.0;
    let (d_psi, d_eps) = iau2000a_radians(t);
    let eps_mean = mean_obliquity_asec(jd_tdb) * ASEC2RAD;
    let eps_true = eps_mean + d_eps;

    let n = nutation_matrix(eps_mean, eps_true, d_psi);
    let p = precession_matrix(jd_tdb);
    let b = frame_bias();
    mat_mul(&n, &mat_mul(&p, &b))
}
