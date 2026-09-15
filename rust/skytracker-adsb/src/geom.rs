//! WGS84 topocentric geometry (port of adsb_receiver.py's
//! geodetic_to_ecef / ecef_to_enu / enu_to_azel_range chain).

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257223563;
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> (f64, f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (sin_lat, cos_lat) = lat.sin_cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    (
        (n + alt_m) * cos_lat * lon.cos(),
        (n + alt_m) * cos_lat * lon.sin(),
        (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
    )
}

pub fn ecef_to_enu(dx: f64, dy: f64, dz: f64, ref_lat_deg: f64, ref_lon_deg: f64) -> (f64, f64, f64) {
    let lat = ref_lat_deg.to_radians();
    let lon = ref_lon_deg.to_radians();
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    (
        -so * dx + co * dy,
        -sl * co * dx - sl * so * dy + cl * dz,
        cl * co * dx + cl * so * dy + sl * dz,
    )
}

/// ENU -> (az deg [0,360), el deg, range km).
pub fn enu_to_azel_range(e: f64, n: f64, u: f64) -> (f64, f64, f64) {
    let horiz = e.hypot(n);
    let rng = (e * e + n * n + u * u).sqrt();
    let az = e.atan2(n).to_degrees().rem_euclid(360.0);
    let el = if horiz > 0.0 || u != 0.0 {
        u.atan2(horiz).to_degrees()
    } else {
        0.0
    };
    (az, el, rng / 1000.0)
}

/// Observer-relative topocentric az/el/range for a geodetic target.
pub fn geodetic_to_azel_range(
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
    obs_lat: f64,
    obs_lon: f64,
    obs_alt_m: f64,
) -> (f64, f64, f64) {
    let (tx, ty, tz) = geodetic_to_ecef(lat_deg, lon_deg, alt_m);
    let (ox, oy, oz) = geodetic_to_ecef(obs_lat, obs_lon, obs_alt_m);
    let (e, n, u) = ecef_to_enu(tx - ox, ty - oy, tz - oz, obs_lat, obs_lon);
    enu_to_azel_range(e, n, u)
}
