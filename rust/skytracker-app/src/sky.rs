//! Sky worker: owns the astro engine (TLE catalog, DE421, Hipparcos) and
//! publishes SkySnapshot — satellites + bodies at 2 Hz (the UI dead-reckons
//! between snapshots), stars every five seconds — plus PassesSnapshot (the
//! upcoming-pass table every minute and the selected satellite's track
//! every 2 s), entirely off the UI thread.

use crate::state::{ArcPoint, BodyMark, PassRow, PassesSnapshot, SatMark, Shared, SkySnapshot, StarMark};
use skytracker_astro::engine::Engine;
use skytracker_astro::sgp4_pass::{self, Observer};
use skytracker_astro::stars;
use skytracker_astro::time;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

pub fn now_jd_tt() -> f64 {
    time::utc_to_tt(2440587.5 + now_unix() / 86400.0)
}

pub fn jd_tt_to_unix(jd_tt: f64) -> f64 {
    (time::tt_to_utc(jd_tt) - 2440587.5) * 86400.0
}

/// Civil (y, m, d, h, min, s) from unix seconds (Howard Hinnant's algorithm).
pub fn civil_from_unix(unix: f64) -> (i64, i64, i64, i64, i64, i64) {
    let secs = unix.floor() as i64;
    let days = secs.div_euclid(86400);
    let sod = secs.rem_euclid(86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, sod / 3600, (sod % 3600) / 60, sod % 60)
}

pub fn utc_iso_from_unix(unix: f64) -> String {
    let (y, m, d, hh, mm, ss) = civil_from_unix(unix);
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02} UTC")
}

pub fn hms_from_unix(unix: f64) -> String {
    let (_, _, _, hh, mm, ss) = civil_from_unix(unix);
    format!("{hh:02}:{mm:02}:{ss:02}")
}

/// `YYYY_MM_DD_HH_MM_SS` like the Python capture directories.
pub fn utc_stamp_compact() -> String {
    let (y, m, d, hh, mm, ss) = civil_from_unix(now_unix());
    format!("{y:04}_{m:02}_{d:02}_{hh:02}_{mm:02}_{ss:02}")
}

fn utc_iso(jd_tt: f64) -> String {
    utc_iso_from_unix(jd_tt_to_unix(jd_tt))
}

pub fn spawn(shared: Arc<Shared>, repo_root: std::path::PathBuf) {
    std::thread::Builder::new()
        .name("sky-worker".into())
        .spawn(move || run(shared, repo_root))
        .expect("spawn sky worker");
}

fn run(shared: Arc<Shared>, root: std::path::PathBuf) {
    let cfg = shared.config.clone();
    let observer = Observer {
        lat_deg: cfg.lat_deg,
        lon_deg: cfg.lon_deg,
        elevation_m: cfg.alt_m,
    };
    let geom = observer.geometry();

    let de421 = root.join("de421.bsp");
    let mut engine = match Engine::new(if de421.exists() { Some(de421.as_path()) } else { None }) {
        Ok(e) => e,
        Err(e) => {
            publish_status(&shared, format!("astro engine failed: {e}"));
            return;
        }
    };
    let tle_path = root.join("tle_cache.tle");
    let mut load_error = String::new();
    let n_catalog = match engine.load_tle_file(&tle_path) {
        Ok(n) => n,
        Err(e) => {
            load_error = format!("TLE load failed ({}): {e} -- run from the repo root or set SKYTRACKER_ROOT", tle_path.display());
            0
        }
    };
    if !de421.exists() {
        load_error = format!("{load_error} | de421.bsp not found at {}", de421.display());
    }
    let all_satnums: Vec<String> = engine
        .tles
        .as_ref()
        .map(|c| c.sats.iter().map(|s| s.satnum.clone()).collect())
        .unwrap_or_default();

    // Hipparcos: brightest N within the limiting magnitude.
    let hip_path = root.join("hip_main.dat");
    let mut star_cat: Vec<stars::Star> = if hip_path.exists() {
        stars::parse_hip_main(&hip_path).unwrap_or_default()
    } else {
        Vec::new()
    };
    star_cat.retain(|s| s.magnitude <= cfg.star_limit_mag);
    star_cat.sort_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap());
    star_cat.truncate(cfg.max_stars);

    let bodies = [
        "sun", "moon", "planet:Mercury", "planet:Venus", "planet:Mars",
        "planet:Jupiter", "planet:Saturn", "planet:Uranus", "planet:Neptune",
    ];

    let mut visible: Vec<String> = Vec::new();
    let mut last_gate = Instant::now() - Duration::from_secs(3600);
    let mut last_stars = Instant::now() - Duration::from_secs(3600);
    let mut last_passes = Instant::now() - Duration::from_secs(3600);
    let mut star_marks: Vec<StarMark> = Vec::new();
    let mut pass_rows: Vec<PassRow> = Vec::new();
    let mut pass_ms = 0.0;
    let mut pass_computed = 0.0;

    loop {
        let t0 = Instant::now();
        let jd_tt = now_jd_tt();

        // Visibility gate over the whole catalog every 60 s (window +-15 min).
        if last_gate.elapsed() > Duration::from_secs(60) && !all_satnums.is_empty() {
            let times: Vec<f64> = (0..8)
                .map(|i| jd_tt + (i as f64 * 4.0 - 14.0) / 1440.0)
                .collect();
            if let Ok(v) = engine.visible_satnums(&all_satnums, &times, &observer, cfg.elevation_mask_deg.max(15.0)) {
                visible = v;
            }
            last_gate = Instant::now();
        }

        // Satellites: live alt/az + 1 s finite-difference rates.
        let mut sats = Vec::with_capacity(visible.len());
        if let Some(cat) = engine.tles.as_ref() {
            let dt = 0.5 / 86400.0;
            for sn in &visible {
                let Some(sat) = cat.get(sn) else { continue };
                let (Ok((alt, az, rng)), Ok((a_m, z_m, _)), Ok((a_p, z_p, _))) = (
                    sgp4_pass::satellite_altaz(sat, jd_tt, &geom),
                    sgp4_pass::satellite_altaz(sat, jd_tt - dt, &geom),
                    sgp4_pass::satellite_altaz(sat, jd_tt + dt, &geom),
                ) else {
                    continue;
                };
                if alt < -5.0 {
                    continue;
                }
                sats.push(SatMark {
                    satnum: sn.clone(),
                    name: sat.name.clone(),
                    az,
                    el: alt,
                    range_km: rng,
                    az_rate: sgp4_pass::unwrap_az_diff(z_p - z_m),
                    el_rate: a_p - a_m,
                });
            }
        }
        sats.sort_by(|a, b| b.el.partial_cmp(&a.el).unwrap());

        // Bodies.
        let mut body_marks = Vec::new();
        for name in bodies {
            if let Ok((alt, az, dist)) = engine.body_altaz_dist(name, jd_tt, &observer) {
                body_marks.push(BodyMark {
                    name: name.trim_start_matches("planet:").to_string(),
                    az,
                    el: alt,
                    dist_km: dist,
                });
            }
        }

        // Stars every 5 s (apparent places; ~0.1 s for 2000 stars).
        if last_stars.elapsed() > Duration::from_secs(5) && !star_cat.is_empty() {
            if let Some(eph) = engine.eph.as_ref() {
                let mut marks = Vec::with_capacity(star_cat.len());
                for s in &star_cat {
                    if let Ok(app) = stars::star_apparent(s, eph, jd_tt, &geom) {
                        if app.alt_deg > -2.0 {
                            marks.push(StarMark {
                                hip: s.hip,
                                mag: s.magnitude,
                                az: app.az_deg,
                                el: app.alt_deg,
                            });
                        }
                    }
                }
                star_marks = marks;
            }
            last_stars = Instant::now();
        }

        let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;
        shared.sky.store(Arc::new(SkySnapshot {
            jd_tt,
            utc_iso: utc_iso(jd_tt),
            n_visible: sats.len(),
            sats,
            stars: star_marks.clone(),
            bodies: body_marks,
            n_catalog,
            compute_ms,
            status: if load_error.is_empty() {
                format!("{n_catalog} TLEs · {} stars", star_marks.len())
            } else {
                load_error.clone()
            },
        }));

        // Pass table every 60 s over the gated set (+ everything that will
        // rise in the horizon: the gate window is short, so use the whole
        // catalog through the engine's coarse prediction).
        if last_passes.elapsed() > Duration::from_secs(60) && !all_satnums.is_empty() {
            let tp = Instant::now();
            pass_rows = compute_passes(&engine, &all_satnums, &observer, jd_tt, cfg.elevation_mask_deg);
            pass_ms = tp.elapsed().as_secs_f64() * 1000.0;
            pass_computed = now_unix();
            last_passes = Instant::now();
        }

        // Selected satellite's track, -10..+10 min at 5 s steps.
        let selected = shared.selected.load();
        let mut arc = Vec::new();
        if let (Some(sn), Some(cat)) = (selected.as_ref().as_ref(), engine.tles.as_ref()) {
            if let Some(sat) = cat.get(sn) {
                let mut t = -600.0;
                while t <= 600.0 {
                    if let Ok((alt, az, _)) = sgp4_pass::satellite_altaz(sat, jd_tt + t / 86400.0, &geom) {
                        if alt > -3.0 {
                            arc.push(ArcPoint { t_rel_s: t, az, el: alt });
                        }
                    }
                    t += 5.0;
                }
            }
        }
        shared.passes.store(Arc::new(PassesSnapshot {
            computed_unix: pass_computed,
            horizon_h: 6.0,
            rows: pass_rows.clone(),
            compute_ms: pass_ms,
            arc,
            arc_satnum: selected.as_ref().clone(),
        }));

        // 2 Hz: the UI dead-reckons marks between snapshots with their rates.
        let sleep = Duration::from_millis(500).saturating_sub(t0.elapsed());
        std::thread::sleep(sleep);
    }
}

/// Upcoming passes over the next 6 h (skytracker-astro `passes`).
fn compute_passes(engine: &Engine, satnums: &[String], observer: &Observer, jd_tt: f64, mask_deg: f64) -> Vec<PassRow> {
    crate::passes_bridge::compute(engine, satnums, observer, jd_tt, mask_deg)
}

fn publish_status(shared: &Shared, msg: String) {
    let mut snap = (**shared.sky.load()).clone();
    snap.status = msg;
    shared.sky.store(Arc::new(snap));
}
