//! Pure-Rust transform sanity tests. Run: cargo test
//! Full parity against transformations.py (incl. the scipy AzAlt2AzEl path) is
//! in test_rust_transforms_parity.py.

use skytracker_core::transforms::*;

fn ang_close(a: f64, b: f64, tol: f64) -> bool {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs() < tol
}

#[test]
fn altaz_formulas() {
    // AzEl2AzAlt_AltAz must be the true inverse of AzAlt2AzEl_AltAz on elevation
    // (forward el = 90 - ALT  =>  inverse ALT = 90 - el). Azimuth round-trips too.
    // With align_el = 0 the elevation round-trip is exact (the forward ignores
    // align_el, matching transformations.py).
    let align_az = 12.0;
    for &az in &[0.0, 45.0, 123.0, 270.0, 359.0] {
        for &el in &[-5.0, 0.0, 30.0, 85.0] {
            let (azm, alt) = az_el_to_az_alt_altaz(az, el, align_az, 0.0);
            assert!(ang_close(azm, az - align_az, 1e-9), "azm");
            assert!((alt - (90.0 - el)).abs() < 1e-9, "alt");
            // full round-trip through the inverse
            let (az2, el2) = az_alt_to_az_el_altaz(azm, alt, align_az);
            assert!(ang_close(az, az2, 1e-9), "az {az}->{az2}");
            assert!((el - el2).abs() < 1e-9, "el {el}->{el2}");
        }
    }
}

#[test]
fn passthrough_identity() {
    let (azm, alt) = az_el_to_az_alt_passthrough(33.0, 47.0);
    assert_eq!((azm, alt), (33.0, 47.0));
    assert_eq!(az_alt_to_az_el_passthrough(33.0, 47.0), (33.0, 47.0));
}

#[test]
fn wedge_roundtrip() {
    let lat = 40.8;
    for &el in &[5.0, 30.0, 75.0] {
        for &az in &[10.0, 100.0, 250.0] {
            let (alt_t, az_t) = local_elev_az_to_telescope(el, az, lat);
            let (el2, az2) = telescope_to_local_elev_az(alt_t, az_t, lat);
            assert!((el - el2).abs() < 1e-7, "el {el}->{el2}");
            assert!(ang_close(az, az2, 1e-6), "az {az}->{az2}");
        }
    }
}

#[test]
fn cartesian_known_values() {
    // az=0,el=0 -> +x ; az=90,el=0 -> +y ; el=90 -> +z
    let n = cartesian_from_az_el(0.0, 0.0);
    assert!((n[0] - 1.0).abs() < 1e-12 && n[1].abs() < 1e-12 && n[2].abs() < 1e-12);
    let e = cartesian_from_az_el(90.0, 0.0);
    assert!(e[0].abs() < 1e-12 && (e[1] - 1.0).abs() < 1e-12);
    let up = cartesian_from_az_el(0.0, 90.0);
    assert!((up[2] - 1.0).abs() < 1e-12);
}

#[test]
fn azalt_to_azel_zenith_alignment() {
    // Alignment at zenith (el=90): the alignment rotation is identity-ish; a
    // mount alt of 0 with azm sweeps the horizon. Mostly a smoke test that the
    // general path runs and returns sane ranges.
    let (az, el) = az_alt_to_az_el(30.0, 0.0, 0.0, 90.0);
    assert!((0.0..360.0).contains(&az));
    assert!((-90.0..=90.0).contains(&el));
}

#[test]
fn sky_to_mount_modes() {
    // AltAz and Passthrough match their dedicated functions. (The final bool
    // is the AltAz-Side flip flag; irrelevant to these modes.)
    assert_eq!(
        sky_to_mount(MountMode::AltAz, 100.0, 40.0, 10.0, 2.0, false),
        az_el_to_az_alt_altaz(100.0, 40.0, 10.0, 2.0)
    );
    assert_eq!(
        sky_to_mount(MountMode::Passthrough, 100.0, 40.0, 10.0, 2.0, false),
        (100.0, 40.0)
    );
    // Eq returns (azm, alt) = reordered local_elev_az_to_telescope.
    let (alt, azm) = local_elev_az_to_telescope(40.0, 100.0, 2.0);
    assert_eq!(
        sky_to_mount(MountMode::Eq, 100.0, 40.0, 10.0, 2.0, false),
        (azm, alt)
    );
}
