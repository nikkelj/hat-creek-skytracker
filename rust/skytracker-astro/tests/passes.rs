//! Pass prediction: plausibility (ISS, GEO), brightness/eclipse behaviour,
//! and parity against the skyfield/Python reference fixture produced by
//! tests/gen_passes_ref.py (which calls the app's own
//! trajectory._estimate_pass_magnitude for the magnitude column).

use skytracker_astro::engine::Engine;
use skytracker_astro::passes::{self, Pass, PassParams};
use skytracker_astro::sgp4_pass::{satellite_altaz, Observer, Satellite};
use skytracker_astro::time;
use skytracker_astro::tle::TleCatalog;

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/passes_ref.txt")
}

/// Hat Creek config.json site (lat/lon/alt keys).
fn observer() -> Observer {
    Observer {
        lat_deg: 34.8740289,
        lon_deg: -120.4461237,
        elevation_m: 100.0,
    }
}

// Embedded fallbacks so the tests do not depend on tle_cache.tle contents
// (the cache is refreshed by the app and synced across machines).
const ISS_TLE: (&str, &str, &str) = (
    "ISS (ZARYA)",
    "1 25544U 98067A   26228.56710022  .00005115  00000+0  99348-4 0  9991",
    "2 25544  51.6334   1.2594 0007609  53.1141 307.0544 15.49461657581119",
);
// DIRECTV 12, ~103 W: above the mask from California around the clock.
const GEO_VISIBLE_TLE: (&str, &str, &str) = (
    "DIRECTV 12",
    "1 36131U 09075A   26228.47744006 -.00000113  00000+0  00000+0 0  9991",
    "2 36131   0.0109 294.1459 0000160 321.5779 138.2885  1.00271546 47159",
);
// INTELSAT 10-02, ~1 W: never above the mask from California.
const GEO_HIDDEN_TLE: (&str, &str, &str) = (
    "INTELSAT 10-02",
    "1 28358U 04022A   26228.34978772 -.00000021  00000+0  00000+0 0  9993",
    "2 28358   0.0574 269.0222 0000198 308.5880 232.0452  1.00272422 81150",
);

/// ISS from the repo's tle_cache.tle when present, else the embedded TLE.
fn iss_from_cache() -> Satellite {
    if let Ok(text) = std::fs::read_to_string(repo_root().join("tle_cache.tle")) {
        for (name, l1, l2) in skytracker_astro::tle::parse_tle_text(&text) {
            if skytracker_astro::tle::satnum_str(&l1) == "25544" {
                if let Ok(s) = Satellite::from_tle(&name, &l1, &l2) {
                    return s;
                }
            }
        }
    }
    Satellite::from_tle(ISS_TLE.0, ISS_TLE.1, ISS_TLE.2).unwrap()
}

fn no_sun(_jd: f64) -> Option<[f64; 3]> {
    None
}

#[test]
fn iss_24h_plausible_passes() {
    let sat = iss_from_cache();
    let obs = observer();
    let geom = obs.geometry();
    let start_tt = time::utc_to_tt(sat.epoch_jd_utc);
    let params = PassParams {
        min_el_deg: 10.0,
        horizon_hours: 24.0,
        coarse_step_s: 30.0,
        fine_step_s: 1.0,
        max_passes_per_sat: 100,
    };
    let passes = passes::predict_passes(&[&sat], &obs, start_tt, &params, &no_sun);
    eprintln!("ISS passes over 24 h: {}", passes.len());
    for p in &passes {
        eprintln!(
            "  aos {:.5} tca {:.5} los {:.5} dur {:.0}s el {:.2} az {:.1}->{:.1}->{:.1} rate {:.3} rng {:.0}",
            p.aos_jd_tt, p.tca_jd_tt, p.los_jd_tt, p.duration_s, p.tca_el, p.aos_az, p.tca_az,
            p.los_az, p.max_rate_dps, p.range_tca_km
        );
    }
    assert!(
        (2..=8).contains(&passes.len()),
        "expected 2..=8 ISS passes, got {}",
        passes.len()
    );
    let mut prev_aos = f64::MIN;
    for p in &passes {
        assert!(p.aos_jd_tt >= prev_aos, "passes not sorted by AOS");
        prev_aos = p.aos_jd_tt;
        assert!(p.aos_jd_tt < p.tca_jd_tt && p.tca_jd_tt < p.los_jd_tt, "AOS<TCA<LOS");
        assert!(
            p.duration_s > 180.0 && p.duration_s < 900.0,
            "duration {} s out of 3..15 min",
            p.duration_s
        );
        assert!(p.tca_el > params.min_el_deg, "tca_el {} <= mask", p.tca_el);
        assert!(!p.aos_truncated && !p.los_truncated);
        let (el_aos, _, _) = satellite_altaz(&sat, p.aos_jd_tt, &geom).unwrap();
        let (el_los, _, _) = satellite_altaz(&sat, p.los_jd_tt, &geom).unwrap();
        assert!((el_aos - 10.0).abs() < 0.1, "el(aos) = {el_aos}");
        assert!((el_los - 10.0).abs() < 0.1, "el(los) = {el_los}");
        // ISS peaks under ~1.2 deg/s overhead; never slower than ~0.1 deg/s.
        assert!(p.max_rate_dps > 0.05 && p.max_rate_dps < 1.5, "rate {}", p.max_rate_dps);
        assert!(p.range_tca_km > 400.0 && p.range_tca_km < 2400.0, "range {}", p.range_tca_km);
        assert!((380.0..480.0).contains(&p.apogee_km) && p.perigee_km <= p.apogee_km);
        // Without a sun vector: not computed, not flagged eclipsed.
        assert!(p.est_mag.is_none() && !p.eclipsed_at_tca);
        assert_eq!(p.satnum, "25544");
    }
    // The TCA really is the maximum: sample either side at 2 s.
    for p in &passes {
        let (el_tca, _, _) = satellite_altaz(&sat, p.tca_jd_tt, &geom).unwrap();
        for dt in [-2.0, 2.0] {
            let (el, _, _) = satellite_altaz(&sat, p.tca_jd_tt + dt / time::DAY_S, &geom).unwrap();
            assert!(el <= el_tca + 1e-6, "el({dt:+}s) {el} > el(tca) {el_tca}");
        }
    }
}

#[test]
fn geo_satellites_no_pass_or_one_long_pass() {
    let obs = observer();
    let params = PassParams {
        horizon_hours: 24.0,
        max_passes_per_sat: 10,
        ..PassParams::default()
    };
    let vis = Satellite::from_tle(GEO_VISIBLE_TLE.0, GEO_VISIBLE_TLE.1, GEO_VISIBLE_TLE.2).unwrap();
    let hid = Satellite::from_tle(GEO_HIDDEN_TLE.0, GEO_HIDDEN_TLE.1, GEO_HIDDEN_TLE.2).unwrap();
    let start_tt = time::utc_to_tt(vis.epoch_jd_utc);

    let pv = passes::predict_passes(&[&vis], &obs, start_tt, &params, &no_sun);
    assert_eq!(pv.len(), 1, "visible GEO: expected one long pass, got {}", pv.len());
    let p = &pv[0];
    assert!(p.aos_truncated && p.los_truncated);
    assert!((p.duration_s - 24.0 * 3600.0).abs() < 1.0, "dur {}", p.duration_s);
    assert!(p.tca_el > 10.0 && p.tca_el < 60.0, "GEO el {}", p.tca_el);
    assert!(p.max_rate_dps < 0.01, "GEO rate {}", p.max_rate_dps);
    assert!(p.range_tca_km > 35000.0 && p.range_tca_km < 40000.0);
    assert!((35000.0..36500.0).contains(&p.apogee_km), "apogee {}", p.apogee_km);

    let ph = passes::predict_passes(&[&hid], &obs, start_tt, &params, &no_sun);
    assert!(ph.is_empty(), "hidden GEO produced {} passes", ph.len());

    // Both at once through the multi-sat entry point: still exactly one.
    let both = passes::predict_passes(&[&vis, &hid], &obs, start_tt, &params, &no_sun);
    assert_eq!(both.len(), 1);
}

#[test]
fn magnitude_and_eclipse_with_real_sun() {
    // Real DE421 sun: over 24 h of ISS passes some culminations are lit and
    // (at this site/season) at least one is in Earth's shadow; lit passes
    // get a finite magnitude in a sane range; closer passes at similar
    // phase come out brighter.
    let de421 = repo_root().join("de421.bsp");
    if !de421.exists() {
        eprintln!("de421.bsp missing; skipping");
        return;
    }
    let mut engine = Engine::new(Some(&de421)).unwrap();
    let text = format!("{}\n{}\n{}\n", ISS_TLE.0, ISS_TLE.1, ISS_TLE.2);
    engine.tles = Some(TleCatalog::from_text(&text));
    let sat = engine.tles.as_ref().unwrap().get("25544").unwrap();
    let start_tt = time::utc_to_tt(sat.epoch_jd_utc);
    let params = PassParams {
        horizon_hours: 24.0,
        max_passes_per_sat: 100,
        ..PassParams::default()
    };
    let passes = engine.predict_passes(&["25544".to_string()], &observer(), start_tt, &params);
    assert!(passes.len() >= 2);
    let lit: Vec<&Pass> = passes.iter().filter(|p| p.est_mag.is_some()).collect();
    let ecl: Vec<&Pass> = passes.iter().filter(|p| p.eclipsed_at_tca).collect();
    for p in &passes {
        eprintln!(
            "  tca {:.5} el {:.1} rng {:.0} mag {:?} ecl {}",
            p.tca_jd_tt, p.tca_el, p.range_tca_km, p.est_mag, p.eclipsed_at_tca
        );
        // est_mag None <=> eclipsed (a sun vector is always available here).
        assert_eq!(p.est_mag.is_none(), p.eclipsed_at_tca);
        if let Some(m) = p.est_mag {
            assert!((-2.0..12.0).contains(&m), "mag {m}");
        }
    }
    assert!(!lit.is_empty(), "no lit pass");
    assert!(!ecl.is_empty(), "no eclipsed pass (fixture epoch has one)");

    // Independent eclipse check at each TCA: satellite TEME vs sun TOD.
    for p in &passes {
        let r_sat = sat.position_teme_km(p.tca_jd_tt).unwrap();
        let r_sun = engine.sun_tod_km(p.tca_jd_tt).unwrap();
        assert_eq!(passes::is_eclipsed(&r_sat, &r_sun), p.eclipsed_at_tca);
    }

    // Brightness ordering: at 1000 km the formula is stdmag - 0.75 + 2.5
    // log10(1/frac); for lit ISS passes range dominates, so the closest lit
    // culmination must not be dimmer than the farthest by more than the
    // phase term allows (and typically is brighter).
    if lit.len() >= 2 {
        let closest = lit
            .iter()
            .min_by(|a, b| a.range_tca_km.partial_cmp(&b.range_tca_km).unwrap())
            .unwrap();
        let farthest = lit
            .iter()
            .max_by(|a, b| a.range_tca_km.partial_cmp(&b.range_tca_km).unwrap())
            .unwrap();
        let range_term = 5.0 * (farthest.range_tca_km / closest.range_tca_km).log10();
        let diff = farthest.est_mag.unwrap() - closest.est_mag.unwrap();
        // |diff - range_term| is the phase-term difference, bounded by
        // 2.5 log10(1/1e-3) = 7.5 mag.
        assert!((diff - range_term).abs() <= 7.5, "diff {diff} range_term {range_term}");
    }
}

#[test]
fn trajectory_arc_brackets_a_pass() {
    let sat = iss_from_cache();
    let obs = observer();
    let geom = obs.geometry();
    let start_tt = time::utc_to_tt(sat.epoch_jd_utc);
    let params = PassParams {
        horizon_hours: 24.0,
        max_passes_per_sat: 1,
        ..PassParams::default()
    };
    let p = &passes::predict_passes(&[&sat], &obs, start_tt, &params, &no_sun)[0];
    let arc = passes::trajectory_arc(&sat, &geom, p.tca_jd_tt, 900.0, 900.0, 5.0);
    assert!(!arc.is_empty());
    for a in &arc {
        assert!(a.el > -5.0);
        assert!((0.0..360.0).contains(&a.az));
        assert!(a.range_km > 300.0);
    }
    // Monotonic time, and the arc spans the pass.
    for w in arc.windows(2) {
        assert!(w[1].jd_tt > w[0].jd_tt);
    }
    assert!(arc.first().unwrap().jd_tt <= p.aos_jd_tt);
    assert!(arc.last().unwrap().jd_tt >= p.los_jd_tt);
    let max_el = arc.iter().map(|a| a.el).fold(f64::MIN, f64::max);
    assert!((max_el - p.tca_el).abs() < 0.5, "arc max el {max_el} vs tca {}", p.tca_el);
}

// ---------------------------------------------------------------------------
// Parity with the skyfield/Python reference (tests/fixtures/passes_ref.txt).
// ---------------------------------------------------------------------------

struct RefPass {
    satnum: String,
    aos: f64,
    tca: f64,
    los: f64,
    tca_el: f64,
    tca_az: f64,
    range_km: f64,
    mag: Option<f64>, // None = "ecl"
    apogee: f64,
    perigee: f64,
}

struct Fixture {
    observer: Observer,
    start_tt: f64,
    horizon_h: f64,
    min_el: f64,
    tles: Vec<(String, String, String, String)>, // satnum, name, l1, l2
    passes: Vec<RefPass>,
}

fn load_fixture() -> Fixture {
    let text = std::fs::read_to_string(fixture_path()).expect("passes_ref.txt");
    let mut fx = Fixture {
        observer: observer(),
        start_tt: 0.0,
        horizon_h: 24.0,
        min_el: 10.0,
        tles: Vec::new(),
        passes: Vec::new(),
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = line.split_once(' ').unwrap();
        match key {
            "observer" => {
                let v: Vec<f64> = rest.split_whitespace().map(|s| s.parse().unwrap()).collect();
                fx.observer = Observer { lat_deg: v[0], lon_deg: v[1], elevation_m: v[2] };
            }
            "start_tt" => fx.start_tt = rest.trim().parse().unwrap(),
            "horizon_h" => fx.horizon_h = rest.trim().parse().unwrap(),
            "min_el" => fx.min_el = rest.trim().parse().unwrap(),
            "tle" => {
                let parts: Vec<&str> = rest.split('|').collect();
                fx.tles.push((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    parts[3].to_string(),
                ));
            }
            "pass" => {
                let f: Vec<&str> = rest.split_whitespace().collect();
                fx.passes.push(RefPass {
                    satnum: f[0].to_string(),
                    aos: f[1].parse().unwrap(),
                    tca: f[2].parse().unwrap(),
                    los: f[3].parse().unwrap(),
                    tca_el: f[4].parse().unwrap(),
                    tca_az: f[5].parse().unwrap(),
                    range_km: f[6].parse().unwrap(),
                    mag: if f[7] == "ecl" { None } else { Some(f[7].parse().unwrap()) },
                    apogee: f[8].parse().unwrap(),
                    perigee: f[9].parse().unwrap(),
                });
            }
            _ => {}
        }
    }
    assert!(fx.start_tt > 2.4e6 && !fx.tles.is_empty() && !fx.passes.is_empty());
    fx
}

#[test]
fn parity_with_python_reference() {
    let fx = load_fixture();
    let de421 = repo_root().join("de421.bsp");
    let mut engine = Engine::new(if de421.exists() { Some(de421.as_path()) } else { None }).unwrap();
    let mut text = String::new();
    for (_, name, l1, l2) in &fx.tles {
        text.push_str(&format!("{name}\n{l1}\n{l2}\n"));
    }
    engine.tles = Some(TleCatalog::from_text(&text));
    let satnums: Vec<String> = fx.tles.iter().map(|t| t.0.clone()).collect();
    let params = PassParams {
        min_el_deg: fx.min_el,
        horizon_hours: fx.horizon_h,
        coarse_step_s: 30.0,
        fine_step_s: 1.0,
        max_passes_per_sat: 100,
    };
    let got = engine.predict_passes(&satnums, &fx.observer, fx.start_tt, &params);

    eprintln!("python passes: {}  rust passes: {}", fx.passes.len(), got.len());
    assert_eq!(got.len(), fx.passes.len(), "pass count differs");

    let (mut max_aos, mut max_los, mut max_tca_t, mut max_el, mut max_az, mut max_rng, mut max_mag) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut sats_compared = std::collections::HashSet::new();
    for r in &fx.passes {
        // Match by satnum + nearest AOS.
        let g = got
            .iter()
            .filter(|g| g.satnum == r.satnum)
            .min_by(|a, b| {
                (a.aos_jd_tt - r.aos).abs().partial_cmp(&(b.aos_jd_tt - r.aos).abs()).unwrap()
            })
            .unwrap_or_else(|| panic!("no Rust pass for {} near {}", r.satnum, r.aos));
        sats_compared.insert(r.satnum.clone());
        let d_aos = (g.aos_jd_tt - r.aos).abs() * time::DAY_S;
        let d_los = (g.los_jd_tt - r.los).abs() * time::DAY_S;
        let d_tca_t = (g.tca_jd_tt - r.tca).abs() * time::DAY_S;
        let d_el = (g.tca_el - r.tca_el).abs();
        let d_az = skytracker_astro::sgp4_pass::unwrap_az_diff(g.tca_az - r.tca_az).abs();
        let d_rng = (g.range_tca_km - r.range_km).abs();
        let d_mag = match (g.est_mag, r.mag) {
            (Some(a), Some(b)) => (a - b).abs(),
            (None, None) => 0.0,
            (a, b) => panic!("{} tca {}: eclipse mismatch rust {:?} python {:?}", r.satnum, r.tca, a, b),
        };
        eprintln!(
            "{:>5} aos {:.2}s los {:.2}s tca {:.2}s el {:.3} az {:.3} rng {:.3}km mag {:.4} (py {:?} rs {:?}) apo {:.3} per {:.3}",
            r.satnum, d_aos, d_los, d_tca_t, d_el, d_az, d_rng, d_mag, r.mag, g.est_mag,
            (g.apogee_km - r.apogee).abs(), (g.perigee_km - r.perigee).abs()
        );
        assert!(d_aos < 2.0, "{} AOS differs by {d_aos} s", r.satnum);
        assert!(d_los < 2.0, "{} LOS differs by {d_los} s", r.satnum);
        assert!(d_el < 0.2, "{} TCA el differs by {d_el}", r.satnum);
        assert!(d_mag < 0.2, "{} mag differs by {d_mag}", r.satnum);
        assert!((g.apogee_km - r.apogee).abs() < 0.5, "{} apogee", r.satnum);
        assert!((g.perigee_km - r.perigee).abs() < 0.5, "{} perigee", r.satnum);
        assert!(d_rng < 5.0, "{} range at TCA differs by {d_rng} km", r.satnum);
        assert!(!g.aos_truncated && !g.los_truncated);
        max_aos = max_aos.max(d_aos);
        max_los = max_los.max(d_los);
        max_tca_t = max_tca_t.max(d_tca_t);
        max_el = max_el.max(d_el);
        max_az = max_az.max(d_az);
        max_rng = max_rng.max(d_rng);
        max_mag = max_mag.max(d_mag);
    }
    assert!(sats_compared.len() >= 3, "need 3+ satellites with passes");
    eprintln!(
        "PARITY max: aos {max_aos:.3}s los {max_los:.3}s tca_t {max_tca_t:.3}s el {max_el:.4} az {max_az:.4} rng {max_rng:.3}km mag {max_mag:.4} over {} passes / {} sats",
        fx.passes.len(),
        sats_compared.len()
    );
}
