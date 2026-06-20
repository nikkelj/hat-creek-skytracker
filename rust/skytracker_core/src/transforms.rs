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

/// Mount mode selector for the control loop's sky->mount transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountMode {
    AltAz,
    Passthrough,
    Eq,
}

/// Sky (az, el) -> mount (azm, alt) for the given mount mode. This is the
/// transform the PROGRAM control path uses each cycle. For Eq mode, `lat_deg`
/// is the configured alignment elevation (matching control.py's call site).
pub fn sky_to_mount(
    mode: MountMode,
    az: f64,
    el: f64,
    alignment_azimuth: f64,
    alignment_elevation: f64,
) -> (f64, f64) {
    match mode {
        MountMode::AltAz => az_el_to_az_alt_altaz(az, el, alignment_azimuth, alignment_elevation),
        MountMode::Passthrough => az_el_to_az_alt_passthrough(az, el),
        // Eq: lat is the alignment elevation; returns (alt, azm) -> reorder.
        MountMode::Eq => {
            let (alt, azm) = local_elev_az_to_telescope(el, az, alignment_elevation);
            (azm, alt)
        }
    }
}
