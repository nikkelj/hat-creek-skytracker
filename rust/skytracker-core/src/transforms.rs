//! Coordinate transforms — port of `transformations.py`.
//!
//! Pure f64 math over 3-vectors and 3x3 matrices (no external linear-algebra
//! crate). The general `az_alt_to_az_el` reproduces the scipy
//! `Rotation.align_vectors` path with a Rodrigues rotation: aligning a single
//! vector pair is just the minimal geodesic rotation about their cross product.
//!
//! The sky->mount functions (`az_el_to_az_alt_*`, `local_elev_az_to_telescope`)
//! are the per-cycle hot path used by the control loop's PROGRAM error
//! computation. Validated against `test_rust_transforms_parity.py`.

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &Vec3) -> f64 {
    dot(a, a).sqrt()
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn matvec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub fn cartesian_from_az_el(az_deg: f64, el_deg: f64) -> Vec3 {
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
}

pub fn az_el_from_cartesian(v: &Vec3) -> (f64, f64) {
    let r = norm(v);
    if r == 0.0 {
        return (0.0, 0.0);
    }
    let az = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    let el = (v[2] / r).asin().to_degrees();
    (az, el)
}

/// Rodrigues rotation matrix about a (non-zero) axis by `angle_rad`.
pub fn rotation_matrix_around_axis(axis_in: &Vec3, angle_rad: f64) -> Mat3 {
    let n = norm(axis_in);
    let axis = [axis_in[0] / n, axis_in[1] / n, axis_in[2] / n];
    let k = [
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ];
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let eye = if i == j { 1.0 } else { 0.0 };
            r[i][j] = eye * cos_a + (1.0 - cos_a) * axis[i] * axis[j] + sin_a * k[i][j];
        }
    }
    r
}

fn identity() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// numpy `np.maximum` semantics: NaN-propagating, unlike Rust's `f64::max`
/// (which ignores NaN). Needed so the degenerate (out-of-domain arcsin) path
/// reproduces transformations.py exactly instead of recovering a finite value.
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Rotation mapping the zenith [0,0,1] onto the alignment direction.
pub fn alignment_rotation_matrix(alignment_azimuth: f64, alignment_elevation: f64) -> Mat3 {
    let north = [0.0, 0.0, 1.0];
    let alignment_vector = cartesian_from_az_el(alignment_azimuth, alignment_elevation);
    let axis = cross(&north, &alignment_vector);
    let axis_norm = norm(&axis);
    if axis_norm < 1e-10 {
        return identity();
    }
    let cos_theta = dot(&north, &alignment_vector).clamp(-1.0, 1.0);
    let angle = cos_theta.acos();
    rotation_matrix_around_axis(&axis, angle)
}

/// Mount (AZM, ALT) -> true (az, el). General alignment. Mirrors `AzAlt2AzEl`
/// (the scipy align_vectors path), composed as align * azm * alt applied to the
/// zenith.
pub fn az_alt_to_az_el(
    azm: f64,
    alt: f64,
    alignment_azimuth: f64,
    alignment_elevation: f64,
) -> (f64, f64) {
    let azm_rad = azm.to_radians();
    let alt_rad = alt.to_radians();
    let zenith = [0.0, 0.0, 1.0];

    let alignment = alignment_rotation_matrix(alignment_azimuth, alignment_elevation);
    let azm_rotation = rotation_matrix_around_axis(&[0.0, 0.0, 1.0], azm_rad);
    let alt_axis = [azm_rad.sin(), -azm_rad.cos(), 0.0];
    let alt_rotation = rotation_matrix_around_axis(&alt_axis, alt_rad);

    let final_rotation = matmul(&matmul(&alignment, &azm_rotation), &alt_rotation);
    let pointing = matvec(&final_rotation, &zenith);
    az_el_from_cartesian(&pointing)
}

pub fn apply_rotation_to_az_el(az: f64, el: f64, rotation_angle_deg: f64) -> (f64, f64) {
    if rotation_angle_deg.abs() < 1e-10 {
        return (az, el);
    }
    let v = cartesian_from_az_el(az, el);
    let m = rotation_matrix_around_axis(&v, rotation_angle_deg.to_radians());
    let rotated = matvec(&m, &v);
    az_el_from_cartesian(&rotated)
}

// --- AltAz mode (simplified, gravity-aligned) ---

pub fn az_alt_to_az_el_altaz(azm: f64, alt: f64, alignment_azimuth: f64) -> (f64, f64) {
    let az = (azm + alignment_azimuth).rem_euclid(360.0);
    let el = 90.0 - alt;
    (az, el)
}

pub fn az_el_to_az_alt_altaz(
    az: f64,
    el: f64,
    alignment_azimuth: f64,
    alignment_elevation: f64,
) -> (f64, f64) {
    // True inverse of az_alt_to_az_el_altaz (el = 90 - alt); see transformations.py.
    let azm = (az - alignment_azimuth).rem_euclid(360.0);
    let alt = 90.0 - el - alignment_elevation;
    (azm, alt)
}

// --- Passthrough mode (identity) ---

pub fn az_alt_to_az_el_passthrough(azm: f64, alt: f64) -> (f64, f64) {
    (azm, alt)
}

pub fn az_el_to_az_alt_passthrough(az: f64, el: f64) -> (f64, f64) {
    (az, el)
}

// --- Wedge / equatorial-frame transforms ---

/// Telescope alt/az (wedge frame) -> local horizontal elev/az.
pub fn telescope_to_local_elev_az(alt_tel_deg: f64, az_tel_deg: f64, lat_deg: f64) -> (f64, f64) {
    let alt_t = alt_tel_deg.to_radians();
    let az_t = az_tel_deg.to_radians();
    let lat = lat_deg.to_radians();

    let sin_el = alt_t.sin() * lat.cos() + alt_t.cos() * lat.sin() * az_t.cos();
    let el_local = sin_el.asin().to_degrees();

    let cos_el = np_maximum(el_local.to_radians().cos(), 1e-8);
    let sin_az = alt_t.cos() * az_t.sin() / cos_el;
    let cos_az = (alt_t.cos() * az_t.cos() * lat.cos() - alt_t.sin() * lat.sin()) / cos_el;
    let az_local = sin_az.atan2(cos_az).to_degrees().rem_euclid(360.0);
    (el_local, az_local)
}

/// Local horizontal elev/az -> telescope alt/az (wedge frame). Returns
/// (alt_tel_deg, az_tel_deg). This is the Eq-mode sky->mount transform.
pub fn local_elev_az_to_telescope(el_local_deg: f64, az_local_deg: f64, lat_deg: f64) -> (f64, f64) {
    let el = el_local_deg.to_radians();
    let az = az_local_deg.to_radians();
    let lat = lat_deg.to_radians();

    let (sin_lat, cos_lat) = (lat.sin(), lat.cos());
    let (sin_el, cos_el) = (el.sin(), el.cos());
    let (sin_az, cos_az) = (az.sin(), az.cos());

    let d = cos_lat * sin_el - sin_lat * cos_el * cos_az;
    let a = sin_lat * sin_el + cos_lat * cos_el * cos_az;
    let b = cos_el * sin_az;

    let alt_tel = d.asin().to_degrees();
    let cos_alt = np_maximum((1.0 - d * d).sqrt(), 1e-8);
    let sin_az_tel = b / cos_alt;
    let cos_az_tel = a / cos_alt;
    let az_tel = sin_az_tel.atan2(cos_az_tel).to_degrees().rem_euclid(360.0);
    (alt_tel, az_tel)
}

/// Equatorial (ra_hours, dec_deg) from local horizontal.
pub fn altaz_local_to_radec(
    el_deg: f64,
    az_deg: f64,
    lat_deg: f64,
    lst_hours: f64,
) -> (f64, f64) {
    let el = el_deg.to_radians();
    let az = az_deg.to_radians();
    let lat = lat_deg.to_radians();

    let sin_dec = lat.sin() * el.sin() + lat.cos() * el.cos() * az.cos();
    let dec = sin_dec.asin().to_degrees();

    let cos_dec = np_maximum(dec.to_radians().cos(), 1e-10);
    let sin_ha = -el.cos() * az.sin() / cos_dec;
    let cos_ha = (lat.cos() * el.sin() - lat.sin() * el.cos() * az.cos()) / cos_dec;
    let ha_deg = sin_ha.atan2(cos_ha).to_degrees();
    let ha_hours = ha_deg / 15.0;
    let ra_hours = (lst_hours - ha_hours).rem_euclid(24.0);
    (ra_hours, dec)
}

/// FOV parameters for a camera. Mirrors `compute_fov_for_camera`.
pub struct Fov {
    pub spot_size_arcsec_per_pixel: f64,
    pub fov_width_deg: f64,
    pub fov_height_deg: f64,
    pub roi_pixel_width: f64,
    pub roi_pixel_height: f64,
}

pub fn compute_fov_for_camera(
    pixel_size_um: f64,
    focal_length_mm: f64,
    roi_width_pct: f64,
    roi_height_pct: f64,
    camera_width_pixels: f64,
    camera_height_pixels: f64,
) -> Fov {
    let spot = 206.0 * pixel_size_um / focal_length_mm;
    let roi_w = camera_width_pixels * roi_width_pct;
    let roi_h = camera_height_pixels * roi_height_pct;
    Fov {
        spot_size_arcsec_per_pixel: spot,
        fov_width_deg: (spot * roi_w) / 3600.0,
        fov_height_deg: (spot * roi_h) / 3600.0,
        roi_pixel_width: roi_w,
        roi_pixel_height: roi_h,
    }
}

/// Side-mount index-home offset (deg). Field-calibrated 2026-07-26: at the
/// AVX index marks the scope points ALONG the horizontal polar axis (at the
/// horizon, azimuth `alignment_azimuth`) with the dec axis vertical, so
/// H = AZM + 90 and dec = 90 - ALT. Mirrors
/// transformations.ALTAZ_SIDE_H0_DEG; `flip` mirrors the H origin for a rig
/// laid down on its other side.
pub const ALTAZ_SIDE_H0_DEG: f64 = 90.0;

#[inline]
fn altaz_side_h0(flip: bool) -> f64 {
    if flip {
        -ALTAZ_SIDE_H0_DEG
    } else {
        ALTAZ_SIDE_H0_DEG
    }
}

/// Side-mounted AltAz (mount lying on its side for center-of-gravity):
/// mount (AZM, ALT) -> true sky (az, el). The AZM axis is HORIZONTAL,
/// pointing at azimuth `alignment_azimuth`; geometrically an equatorial
/// mount whose pole sits ON the horizon, with the AVX index marks as the
/// encoder home (H = AZM + h0, dec = 90 - ALT). Mirrors
/// transformations.AzAlt2AzEl_AltAzSide, which is eq_mount_to_azel with
/// pole_alt exactly 0: basis p = pole (horizontal), x = zenith
/// (up - (up.p)p with up.p = 0), y = p x up.
pub fn az_alt_to_az_el_altaz_side(
    azm: f64,
    alt: f64,
    alignment_azimuth: f64,
    flip: bool,
) -> (f64, f64) {
    let h = (azm + altaz_side_h0(flip)).to_radians();
    let d = (90.0 - alt).to_radians();
    let a0 = alignment_azimuth.to_radians();
    // v = cos(d)cos(h)*x + cos(d)sin(h)*y + sin(d)*p in (north, east, up)
    // with x = (0,0,1), y = (sin a0, -cos a0, 0), p = (cos a0, sin a0, 0).
    let north = d.cos() * h.sin() * a0.sin() + d.sin() * a0.cos();
    let east = -(d.cos() * h.sin() * a0.cos()) + d.sin() * a0.sin();
    let up = d.cos() * h.cos();
    let el = up.clamp(-1.0, 1.0).asin().to_degrees();
    let az = east.atan2(north).to_degrees().rem_euclid(360.0);
    (az, el)
}

/// Inverse of `az_alt_to_az_el_altaz_side`: true sky (az, el) -> mount
/// (AZM, ALT). Mirrors transformations.AzEl2AzAlt_AltAzSide: the eq-frame
/// (ha, dec) = (asin(v.p), atan2(v.y, v.x)) solution mapped back through the
/// index home, AZM = (ha - h0) mod 360, ALT = 90 - dec.
pub fn az_el_to_az_alt_altaz_side(
    az: f64,
    el: f64,
    alignment_azimuth: f64,
    flip: bool,
) -> (f64, f64) {
    let a = az.to_radians();
    let e = el.to_radians();
    let a0 = alignment_azimuth.to_radians();
    let north = e.cos() * a.cos();
    let east = e.cos() * a.sin();
    let up = e.sin();
    let dec = (north * a0.cos() + east * a0.sin()).clamp(-1.0, 1.0).asin().to_degrees();
    let ha = (north * a0.sin() - east * a0.cos()).atan2(up).to_degrees();
    ((ha - altaz_side_h0(flip)).rem_euclid(360.0), 90.0 - dec)
}

/// Mount mode selector for the control loop's sky->mount transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountMode {
    AltAz,
    AltAzSide,
    Passthrough,
    Eq,
}

/// Mount (azm, alt) -> sky (az, el) for the given mount mode: the inverse of
/// `sky_to_mount` (same alignment / lat / flip conventions).
pub fn mount_to_sky(
    mode: MountMode,
    azm: f64,
    alt: f64,
    alignment_azimuth: f64,
    alignment_elevation: f64,
    altaz_side_flip: bool,
) -> (f64, f64) {
    match mode {
        // Exact inverse of az_el_to_az_alt_altaz (which sky_to_mount uses):
        // alt = 90 - el - alignment_elevation. (transformations.py's forward
        // AzAlt2AzEl_AltAz drops the alignment elevation; its inverse keeps
        // it, so the pair is not a round trip when alignment_elevation != 0.)
        MountMode::AltAz => ((azm + alignment_azimuth).rem_euclid(360.0), 90.0 - alt - alignment_elevation),
        MountMode::AltAzSide => az_alt_to_az_el_altaz_side(azm, alt, alignment_azimuth, altaz_side_flip),
        MountMode::Passthrough => az_alt_to_az_el_passthrough(azm, alt),
        MountMode::Eq => {
            let (el, az) = telescope_to_local_elev_az(alt, azm, alignment_elevation);
            (az, el)
        }
    }
}

#[cfg(test)]
mod mount_sky_roundtrip {
    use super::*;

    #[test]
    fn mount_to_sky_inverts_sky_to_mount_in_every_mode() {
        for mode in [MountMode::AltAz, MountMode::AltAzSide, MountMode::Passthrough, MountMode::Eq] {
            for &(az, el) in &[(10.0, 20.0), (123.4, 45.6), (250.0, 70.0), (359.0, 15.0), (180.0, 5.0)] {
                let (azm, alt) = sky_to_mount(mode, az, el, 274.8, -0.6, false);
                let (az2, el2) = mount_to_sky(mode, azm, alt, 274.8, -0.6, false);
                let daz = ((az2 - az + 540.0).rem_euclid(360.0) - 180.0).abs();
                assert!(daz < 1e-6 && (el2 - el).abs() < 1e-6, "{mode:?}: ({az},{el}) -> ({azm},{alt}) -> ({az2},{el2})");
            }
        }
    }
}

/// Sky (az, el) -> mount (azm, alt) for the given mount mode. This is the
/// transform the PROGRAM control path uses each cycle. For Eq mode, `lat_deg`
/// is the configured alignment elevation (matching control.py's call site).
/// `altaz_side_flip` selects the side-mount tip side (ignored elsewhere).
pub fn sky_to_mount(
    mode: MountMode,
    az: f64,
    el: f64,
    alignment_azimuth: f64,
    alignment_elevation: f64,
    altaz_side_flip: bool,
) -> (f64, f64) {
    match mode {
        MountMode::AltAz => az_el_to_az_alt_altaz(az, el, alignment_azimuth, alignment_elevation),
        MountMode::AltAzSide => {
            az_el_to_az_alt_altaz_side(az, el, alignment_azimuth, altaz_side_flip)
        }
        MountMode::Passthrough => az_el_to_az_alt_passthrough(az, el),
        // Eq: lat is the alignment elevation; returns (alt, azm) -> reorder.
        MountMode::Eq => {
            let (alt, azm) = local_elev_az_to_telescope(el, az, alignment_elevation);
            (azm, alt)
        }
    }
}
