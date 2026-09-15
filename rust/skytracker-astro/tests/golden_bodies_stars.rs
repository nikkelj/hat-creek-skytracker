//! Golden-vector parity for the SPK reader, solar-system bodies, and
//! Hipparcos star apparent places (gates: SPK ~exact, bodies/stars 60").

use skytracker_astro::ephemeris::{body_id, Ephemeris};
use skytracker_astro::sgp4_pass::Observer;
use skytracker_astro::spk::Spk;
use skytracker_astro::stars::{self, Star};

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden(name: &str) -> npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>> {
    let path = repo_root().join("tests/golden").join(name);
    npyz::npz::NpzArchive::open(&path).unwrap_or_else(|e| panic!("open {path:?}: {e}"))
}

fn f64s(
    npz: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
    name: &str,
) -> Vec<f64> {
    npz.by_name(name)
        .unwrap()
        .unwrap_or_else(|| panic!("missing {name}"))
        .into_vec::<f64>()
        .unwrap()
}

fn i32s(
    npz: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
    name: &str,
) -> Vec<i32> {
    npz.by_name(name)
        .unwrap()
        .unwrap_or_else(|| panic!("missing {name}"))
        .into_vec::<i32>()
        .unwrap()
}

fn i64s(
    npz: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
    name: &str,
) -> Vec<i64> {
    npz.by_name(name)
        .unwrap()
        .unwrap_or_else(|| panic!("missing {name}"))
        .into_vec::<i64>()
        .unwrap()
}

fn sky_sep_deg(alt1: f64, az1: f64, alt2: f64, az2: f64) -> f64 {
    let (a1, z1) = (alt1.to_radians(), az1.to_radians());
    let (a2, z2) = (alt2.to_radians(), az2.to_radians());
    let cosd = a1.sin() * a2.sin() + a1.cos() * a2.cos() * (z1 - z2).cos();
    cosd.clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
fn spk_reader_matches_jplephem() {
    let mut npz = golden("spk_states.npz");
    let centers = i32s(&mut npz, "center");
    let targets = i32s(&mut npz, "target");
    let jd_tdb = f64s(&mut npz, "jd_tdb");
    let pos_g = f64s(&mut npz, "position_km");
    let vel_g = f64s(&mut npz, "velocity_km_s");

    let spk = Spk::open(&repo_root().join("de421.bsp")).unwrap();
    let nt = jd_tdb.len();
    let mut worst_pos = 0.0_f64;
    let mut worst_vel = 0.0_f64;
    for (si, (&c, &t)) in centers.iter().zip(targets.iter()).enumerate() {
        for (ti, &jd) in jd_tdb.iter().enumerate() {
            let (p, v) = spk.state(c, t, jd).unwrap();
            for ax in 0..3 {
                let idx = (si * nt + ti) * 3 + ax;
                worst_pos = worst_pos.max((p[ax] - pos_g[idx]).abs());
                worst_vel = worst_vel.max((v[ax] - vel_g[idx]).abs());
            }
        }
    }
    println!("worst SPK position diff {worst_pos:.3e} km, velocity diff {worst_vel:.3e} km/s");
    // Same Chebyshev coefficients; different summation order costs ~1 cm at
    // planetary magnitudes (~1e8 km), i.e. pure f64 rounding. 10 cm gate.
    assert!(worst_pos < 1e-4, "position {worst_pos} km");
    assert!(worst_vel < 1e-9, "velocity {worst_vel} km/s");
}

#[test]
fn body_altaz_radec_match_skyfield() {
    let mut npz = golden("astro_bodies.npz");
    let tt = f64s(&mut npz, "time_tt_jd");
    let site = f64s(&mut npz, "site_lat_lon_alt");
    let alt_g = f64s(&mut npz, "alt_deg");
    let az_g = f64s(&mut npz, "az_deg");
    let ra_g = f64s(&mut npz, "ra_hours");
    let dec_g = f64s(&mut npz, "dec_deg");

    // Body order written by record_golden.py.
    let names = [
        "sun", "moon", "mercury", "venus", "mars", "jupiter barycenter", "saturn barycenter",
    ];
    let eph = Ephemeris::open(&repo_root().join("de421.bsp")).unwrap();
    let observer = Observer {
        lat_deg: site[0],
        lon_deg: site[1],
        elevation_m: site[2],
    };
    let geom = observer.geometry();

    let tol_deg = 60.0 / 3600.0;
    let mut worst = 0.0_f64;
    let mut worst_radec = 0.0_f64;
    for (bi, name) in names.iter().enumerate() {
        let id = body_id(name).unwrap();
        for (ti, &jd_tt) in tt.iter().enumerate() {
            let (alt, az, ra, dec) = eph.apparent_altaz_radec(id, jd_tt, &geom).unwrap();
            let idx = bi * tt.len() + ti;
            let sep = sky_sep_deg(alt, az, alt_g[idx], az_g[idx]);
            worst = worst.max(sep);
            assert!(
                sep < tol_deg,
                "{name} t{ti}: altaz sep {:.2}\" (alt {alt:.5} vs {:.5})",
                sep * 3600.0,
                alt_g[idx]
            );
            let sep_rd = sky_sep_deg(dec, ra * 15.0, dec_g[idx], ra_g[idx] * 15.0);
            worst_radec = worst_radec.max(sep_rd);
            assert!(
                sep_rd < tol_deg,
                "{name} t{ti}: radec sep {:.2}\"",
                sep_rd * 3600.0
            );
        }
    }
    println!(
        "worst body altaz sep {:.2}\", radec sep {:.2}\"",
        worst * 3600.0,
        worst_radec * 3600.0
    );
}

#[test]
fn star_apparent_matches_skyfield() {
    let mut npz = golden("astro_stars.npz");
    let hip_id = i64s(&mut npz, "hip_id");
    let cat_ra = f64s(&mut npz, "catalog_ra_deg");
    let cat_dec = f64s(&mut npz, "catalog_dec_deg");
    let pm_ra = f64s(&mut npz, "pm_ra_mas_yr");
    let pm_dec = f64s(&mut npz, "pm_dec_mas_yr");
    let plx = f64s(&mut npz, "parallax_mas");
    let tt = f64s(&mut npz, "time_tt_jd");
    let site = f64s(&mut npz, "site_lat_lon_alt");
    let ra_g = f64s(&mut npz, "ra_apparent_hours");
    let dec_g = f64s(&mut npz, "dec_apparent_deg");
    let alt_g = f64s(&mut npz, "alt_deg");
    let az_g = f64s(&mut npz, "az_deg");

    let eph = Ephemeris::open(&repo_root().join("de421.bsp")).unwrap();
    let observer = Observer {
        lat_deg: site[0],
        lon_deg: site[1],
        elevation_m: site[2],
    };
    let geom = observer.geometry();

    let tol_deg = 60.0 / 3600.0;
    let mut worst = 0.0_f64;
    for si in 0..hip_id.len() {
        let star = Star {
            hip: hip_id[si],
            magnitude: 0.0,
            ra_deg: cat_ra[si],
            dec_deg: cat_dec[si],
            pm_ra_mas_yr: pm_ra[si],
            pm_dec_mas_yr: pm_dec[si],
            parallax_mas: plx[si],
        };
        for (ti, &jd_tt) in tt.iter().enumerate() {
            let app = stars::star_apparent(&star, &eph, jd_tt, &geom).unwrap();
            let idx = si * tt.len() + ti;
            let sep_aa = sky_sep_deg(app.alt_deg, app.az_deg, alt_g[idx], az_g[idx]);
            let sep_rd = sky_sep_deg(
                app.dec_apparent_deg,
                app.ra_apparent_hours * 15.0,
                dec_g[idx],
                ra_g[idx] * 15.0,
            );
            worst = worst.max(sep_aa.max(sep_rd));
            assert!(
                sep_aa < tol_deg && sep_rd < tol_deg,
                "HIP {} t{ti}: altaz {:.2}\" radec {:.2}\"",
                hip_id[si],
                sep_aa * 3600.0,
                sep_rd * 3600.0
            );
        }
    }
    println!("worst star sep {:.2}\"", worst * 3600.0);
}

#[test]
fn hip_parser_matches_golden_catalog_fields() {
    let hip_path = repo_root().join("hip_main.dat");
    if !hip_path.exists() {
        eprintln!("hip_main.dat absent; skipping parser check");
        return;
    }
    let stars = stars::parse_hip_main(&hip_path).unwrap();
    let mut npz = golden("astro_stars.npz");
    let hip_id = i64s(&mut npz, "hip_id");
    let cat_ra = f64s(&mut npz, "catalog_ra_deg");
    let cat_dec = f64s(&mut npz, "catalog_dec_deg");
    let plx = f64s(&mut npz, "parallax_mas");

    for si in 0..hip_id.len() {
        let s = stars
            .iter()
            .find(|s| s.hip == hip_id[si])
            .unwrap_or_else(|| panic!("HIP {} not parsed", hip_id[si]));
        assert!((s.ra_deg - cat_ra[si]).abs() < 1e-8, "HIP {} ra", s.hip);
        assert!((s.dec_deg - cat_dec[si]).abs() < 1e-8, "HIP {} dec", s.hip);
        assert!((s.parallax_mas - plx[si]).abs() < 1e-8, "HIP {} plx", s.hip);
    }
    println!("hip parser: {} stars parsed, 50 golden rows matched", stars.len());
}
