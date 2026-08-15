//! Golden-vector parity tests against skyfield outputs recorded by
//! tools/record_golden.py into tests/golden/ (repo root).
//!
//! Tolerances come from tests/golden/MANIFEST.json: GAST 5 ms, satellites
//! 20 arcsec, rates checked against the same finite-difference scheme.

use skytracker_astro::sgp4_pass::{self, Observer, Satellite};
use skytracker_astro::time;

use std::io::Read;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

struct Npz {
    archive: npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
}

impl Npz {
    fn open(name: &str) -> Self {
        let path = golden_dir().join(name);
        Npz {
            archive: npyz::npz::NpzArchive::open(&path)
                .unwrap_or_else(|e| panic!("open {path:?}: {e}")),
        }
    }

    fn f64s(&mut self, name: &str) -> Vec<f64> {
        let arr = self
            .archive
            .by_name(name)
            .unwrap_or_else(|e| panic!("read {name}: {e}"))
            .unwrap_or_else(|| panic!("missing array {name}"));
        arr.into_vec::<f64>()
            .unwrap_or_else(|e| panic!("decode {name}: {e}"))
    }

    fn f64s_2d(&mut self, name: &str) -> (Vec<usize>, Vec<f64>) {
        let arr = self
            .archive
            .by_name(name)
            .unwrap_or_else(|e| panic!("read {name}: {e}"))
            .unwrap_or_else(|| panic!("missing array {name}"));
        let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
        let data = arr
            .into_vec::<f64>()
            .unwrap_or_else(|e| panic!("decode {name}: {e}"));
        (shape, data)
    }
}

/// Angular difference of two hour angles, in hours, wrapped to [-12, 12).
fn hour_diff(a: f64, b: f64) -> f64 {
    (a - b + 12.0).rem_euclid(24.0) - 12.0
}

/// Great-circle separation between two (alt, az) pairs in degrees.
fn sky_sep_deg(alt1: f64, az1: f64, alt2: f64, az2: f64) -> f64 {
    let (a1, z1) = (alt1.to_radians(), az1.to_radians());
    let (a2, z2) = (alt2.to_radians(), az2.to_radians());
    let cosd =
        a1.sin() * a2.sin() + a1.cos() * a2.cos() * (z1 - z2).cos();
    cosd.clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
fn gast_gmst_delta_t_match_skyfield() {
    let mut npz = Npz::open("astro_time.npz");
    let tt = npz.f64s("time_tt_jd");
    let gast_g = npz.f64s("gast_hours");
    let gmst_g = npz.f64s("gmst_hours");
    let delta_t_g = npz.f64s("delta_t_s");

    // 5 ms of time, expressed in sidereal hours.
    let tol_hours = 0.005 / 3600.0;
    let mut worst_gast = 0.0_f64;
    let mut worst_gmst = 0.0_f64;
    let mut worst_dt = 0.0_f64;
    for i in 0..tt.len() {
        worst_dt = worst_dt.max((time::delta_t(tt[i]) - delta_t_g[i]).abs());
        worst_gmst = worst_gmst.max(hour_diff(time::gmst_hours(tt[i]), gmst_g[i]).abs());
        worst_gast = worst_gast.max(hour_diff(time::gast_hours(tt[i]), gast_g[i]).abs());
    }
    println!(
        "worst: delta_t {:.3e} s, gmst {:.3e} s, gast {:.3e} s",
        worst_dt,
        worst_gmst * 3600.0,
        worst_gast * 3600.0
    );
    assert!(worst_dt < 0.005, "delta_t worst {worst_dt} s");
    assert!(worst_gmst < tol_hours, "gmst worst {} s", worst_gmst * 3600.0);
    assert!(worst_gast < tol_hours, "gast worst {} s", worst_gast * 3600.0);
}

fn load_tles() -> Vec<(String, String, String)> {
    let mut text = String::new();
    std::fs::File::open(golden_dir().join("sat_tles.txt"))
        .expect("sat_tles.txt (run tools/record_golden.py)")
        .read_to_string(&mut text)
        .unwrap();
    let lines: Vec<&str> = text.lines().collect();
    lines
        .chunks(3)
        .filter(|c| c.len() == 3)
        .map(|c| (c[0].to_string(), c[1].to_string(), c[2].to_string()))
        .collect()
}

#[test]
fn satellite_altaz_matches_skyfield() {
    let mut npz = Npz::open("astro_sats.npz");
    let tt = npz.f64s("time_tt_jd");
    let site = npz.f64s("site_lat_lon_alt");
    let (shape_alt, alt_g) = npz.f64s_2d("alt_deg");
    let (_, az_g) = npz.f64s_2d("az_deg");
    let (_, dist_g) = npz.f64s_2d("dist_km");

    let tles = load_tles();
    assert_eq!(shape_alt[0], tles.len(), "TLE count vs golden rows");
    assert_eq!(shape_alt[1], tt.len());

    let observer = Observer {
        lat_deg: site[0],
        lon_deg: site[1],
        elevation_m: site[2],
    };
    let geom = observer.geometry();

    let tol_deg = 20.0 / 3600.0; // 20 arcsec
    let mut worst = 0.0_f64;
    let mut worst_dist_rel = 0.0_f64;
    for (s, (name, l1, l2)) in tles.iter().enumerate() {
        let sat = Satellite::from_tle(name, l1, l2).unwrap();
        for (i, &jd_tt) in tt.iter().enumerate() {
            let (alt, az, dist) = sgp4_pass::satellite_altaz(&sat, jd_tt, &geom).unwrap();
            let idx = s * tt.len() + i;
            let sep = sky_sep_deg(alt, az, alt_g[idx], az_g[idx]);
            worst = worst.max(sep);
            worst_dist_rel = worst_dist_rel.max((dist - dist_g[idx]).abs() / dist_g[idx]);
            assert!(
                sep < tol_deg,
                "{name} sample {i}: sep {:.2} arcsec (alt {alt:.4} vs {:.4}, az {az:.4} vs {:.4})",
                sep * 3600.0,
                alt_g[idx],
                az_g[idx],
            );
        }
    }
    println!(
        "worst sky sep {:.2} arcsec, worst relative range error {:.2e}",
        worst * 3600.0,
        worst_dist_rel
    );
    assert!(worst_dist_rel < 1e-4);
}

#[test]
fn satellite_rates_match_golden_scheme() {
    let mut npz = Npz::open("astro_sats.npz");
    let tt = npz.f64s("time_tt_jd");
    let site = npz.f64s("site_lat_lon_alt");
    let (shape, az_rate_g) = npz.f64s_2d("az_rate_dps");
    let (_, el_rate_g) = npz.f64s_2d("el_rate_dps");

    let tles = load_tles();
    let observer = Observer {
        lat_deg: site[0],
        lon_deg: site[1],
        elevation_m: site[2],
    };
    let geom = observer.geometry();

    // The golden rates are +/-0.5 s central differences of skyfield alt/az.
    // Reproduce the same scheme with the Rust engine; the tolerance follows
    // from 20 arcsec position agreement over a 1 s baseline (plus margin
    // for az near the pole; use sky-projected rate error).
    let dt_s = 0.5;
    let dt_days = dt_s / time::DAY_S;
    let mut worst = 0.0_f64;
    for (s, (name, l1, l2)) in tles.iter().enumerate() {
        let sat = Satellite::from_tle(name, l1, l2).unwrap();
        for (i, &jd_tt) in tt.iter().enumerate() {
            let (alt_m, az_m, _) =
                sgp4_pass::satellite_altaz(&sat, jd_tt - dt_days, &geom).unwrap();
            let (alt_p, az_p, _) =
                sgp4_pass::satellite_altaz(&sat, jd_tt + dt_days, &geom).unwrap();
            let (alt_c, _, _) = sgp4_pass::satellite_altaz(&sat, jd_tt, &geom).unwrap();
            let az_rate = sgp4_pass::unwrap_az_diff(az_p - az_m) / (2.0 * dt_s);
            let el_rate = (alt_p - alt_m) / (2.0 * dt_s);
            let idx = s * tt.len() + i;
            // Compare sky-projected rates (az error shrinks with cos(alt)).
            let cosa = alt_c.to_radians().cos();
            let d_az = (az_rate - az_rate_g[idx]).abs() * cosa;
            let d_el = (el_rate - el_rate_g[idx]).abs();
            worst = worst.max(d_az.max(d_el));
        }
    }
    println!("worst sky-projected rate error {:.3e} deg/s", worst);
    // 20 arcsec / 1 s baseline = 0.0056 deg/s worst-case bound; the real
    // agreement is far better because errors are common-mode across the
    // +/-0.5 s pair.
    assert!(worst < 0.002, "worst rate diff {worst} deg/s");
}
