//! Sky worker: owns the astro engine (TLE catalog, DE421, Hipparcos) and
//! publishes SkySnapshot — satellites + bodies at 2 Hz (the UI dead-reckons
//! between snapshots), stars every five seconds — plus PassesSnapshot (the
//! upcoming-pass table every minute and the selected satellite's track
//! every 2 s), entirely off the UI thread.

use crate::catalogs;
use crate::state::{ArcPoint, BodyMark, DsoMark, PassRow, PassesSnapshot, SatMark, Shared, SkySnapshot, StarMark, TargetTrack};
use skytracker_astro::apparent::FrameContext;
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
    let mut all_satnums: Vec<String> = engine
        .tles
        .as_ref()
        .map(|c| c.sats.iter().map(|s| s.satnum.clone()).collect())
        .unwrap_or_default();
    let mut n_catalog = n_catalog;
    // TLE freshness: download from Celestrak when the cache is older than
    // tle_cache_age_hours (background thread; the catalog is reloaded when
    // it lands) or when the UI asks.
    let age_h = skytracker_astro::tle::cache_age_s(&tle_path).map(|s| s / 3600.0);
    shared.tle_status.store(Arc::new(match age_h {
        Some(a) => format!("{n_catalog} TLEs · cache {a:.1} h old"),
        None => "no TLE cache".into(),
    }));
    if cfg.tle_max_age_h > 0.0 && age_h.map_or(true, |a| a > cfg.tle_max_age_h) {
        shared.tle_refresh.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let (dl_tx, dl_rx) = crossbeam_channel::unbounded::<Result<usize, String>>();
    let mut downloading = false;

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

    // Catalogue layers: IAU names, Messier + NGC (OpenNGC).
    let star_names = Arc::new(catalogs::iau_star_names(&root.join("catalogs/iau-csn.txt")));
    let (messier, ngc) = catalogs::load_openngc(&root.join("catalogs/openngc.csv"));
    eprintln!("skytracker: {} star names, {} Messier, {} NGC objects", star_names.len(), messier.len(), ngc.len());
    let mut dso_marks: Vec<DsoMark> = Vec::new();
    let launches_dir = root.join(&cfg.launches_dir);
    let mut launches = Arc::new(crate::launches::load_dir(&launches_dir, &observer));
    if !launches.is_empty() {
        eprintln!("skytracker: {} launch trajectories from {}", launches.len(), launches_dir.display());
    }
    let mut last_launch_scan = Instant::now();
    let dso_altaz = |d: &catalogs::Dso, jd_tt: f64| -> (f64, f64) {
        let ctx = FrameContext::new(jd_tt);
        let (sr, cr) = d.ra_deg.to_radians().sin_cos();
        let (sd, cd) = d.dec_deg.to_radians().sin_cos();
        let (alt, az) = ctx.altaz_from_icrs(&[cd * cr, cd * sr, sd], &geom);
        (az, alt)
    };

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

        // TLE download / reload.
        if shared.tle_refresh.swap(false, std::sync::atomic::Ordering::Relaxed) && !downloading {
            downloading = true;
            shared.tle_status.store(Arc::new("downloading TLEs from Celestrak…".into()));
            let (url, path, tx) = (cfg.tle_url.clone(), tle_path.clone(), dl_tx.clone());
            std::thread::Builder::new().name("tle-download".into()).spawn(move || {
                let _ = tx.send(skytracker_astro::tle::download_to_cache(&url, &path, 60));
            }).ok();
        }
        if let Ok(res) = dl_rx.try_recv() {
            downloading = false;
            match res {
                Ok(n) => {
                    match engine.load_tle_file(&tle_path) {
                        Ok(m) => {
                            n_catalog = m;
                            all_satnums = engine.tles.as_ref().map(|c| c.sats.iter().map(|s| s.satnum.clone()).collect()).unwrap_or_default();
                            last_gate = Instant::now() - Duration::from_secs(3600);
                            last_passes = Instant::now() - Duration::from_secs(3600);
                            shared.tle_version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            shared.tle_status.store(Arc::new(format!("{m} TLEs · refreshed just now ({n} downloaded)")));
                            eprintln!("skytracker: TLEs refreshed from Celestrak ({m} satellites)");
                        }
                        Err(e) => shared.tle_status.store(Arc::new(format!("TLE reload failed: {e}"))),
                    }
                }
                Err(e) => shared.tle_status.store(Arc::new(format!("TLE download failed: {e}"))),
            }
        }

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
                let (apo, per) = skytracker_astro::passes::apogee_perigee_km(sat);
                sats.push(SatMark {
                    satnum: sn.clone(),
                    name: sat.name.clone(),
                    az,
                    el: alt,
                    range_km: rng,
                    az_rate: sgp4_pass::unwrap_az_diff(z_p - z_m),
                    el_rate: a_p - a_m,
                    intl_desg: sat.intl_desg.clone(),
                    inclination_deg: sat.inclination_deg,
                    raan_deg: sat.raan_deg,
                    arg_perigee_deg: sat.arg_perigee_deg,
                    mean_anomaly_deg: sat.mean_anomaly_deg,
                    mean_motion_rev_day: sat.mean_motion_rev_per_day,
                    eccentricity: sat.eccentricity,
                    rev_number: sat.rev_number,
                    epoch_unix: (sat.epoch_jd_utc - 2440587.5) * 86400.0,
                    apogee_km: apo,
                    perigee_km: per,
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

        if last_launch_scan.elapsed() > Duration::from_secs(60) {
            launches = Arc::new(crate::launches::load_dir(&launches_dir, &observer));
            last_launch_scan = Instant::now();
        }

        // DSOs every 5 s (fixed ICRS directions; Messier always, NGC to the limit).
        if last_stars.elapsed() > Duration::from_secs(5) {
            let mut marks = Vec::new();
            if cfg.messier_enabled {
                for d in &messier {
                    let (az, el) = dso_altaz(d, jd_tt);
                    if el > -2.0 {
                        marks.push(DsoMark { key: d.key.clone(), name: d.name.clone(), az, el, mag: d.mag, messier: true });
                    }
                }
            }
            if cfg.ngc_enabled {
                for d in ngc.iter().filter(|d| d.mag <= cfg.ngc_limit_mag) {
                    let (az, el) = dso_altaz(d, jd_tt);
                    if el > 0.0 {
                        marks.push(DsoMark { key: d.key.clone(), name: d.name.clone(), az, el, mag: d.mag, messier: false });
                    }
                }
            }
            dso_marks = marks;
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

        // Non-satellite selection: live az/el + rates from +-0.5 s.
        let selected_key = shared.selected.load();
        let target = selected_key.as_ref().as_ref().and_then(|key| {
            let dt = 0.5 / 86400.0;
            let pos_at = |jd: f64| -> Option<(f64, f64, String)> {
                if let Some(b) = key.strip_prefix("body:") {
                    let name = if matches!(b, "sun" | "moon") { b.to_string() } else { format!("planet:{b}") };
                    let (alt, az, _) = engine.body_altaz_dist(&name, jd, &observer).ok()?;
                    Some((az, alt, b.to_string()))
                } else if let Some(h) = key.strip_prefix("star:HIP") {
                    let hip: i64 = h.parse().ok()?;
                    let s = star_cat.iter().find(|s| s.hip == hip)?;
                    let eph = engine.eph.as_ref()?;
                    let app = stars::star_apparent(s, eph, jd, &geom).ok()?;
                    let name = star_names.get(&hip).cloned().unwrap_or_else(|| format!("HIP {hip}"));
                    Some((app.az_deg, app.alt_deg, name))
                } else if key.starts_with("dso:") {
                    let d = messier.iter().chain(ngc.iter()).find(|d| &d.key == key)?;
                    let (az, el) = dso_altaz(d, jd);
                    Some((az, el, d.name.clone()))
                } else if key.starts_with("launch:") {
                    let l = launches.iter().find(|l| &l.key == key)?;
                    let (az, el, _) = crate::launches::interp(l, jd_tt_to_unix(jd))?;
                    Some((az, el, l.name.clone()))
                } else {
                    None
                }
            };
            let (az, el, name) = pos_at(jd_tt)?;
            let (a_m, e_m, _) = pos_at(jd_tt - dt)?;
            let (a_p, e_p, _) = pos_at(jd_tt + dt)?;
            Some(TargetTrack { key: key.clone(), name, jd_tt, az, el, az_rate: sgp4_pass::unwrap_az_diff(a_p - a_m), el_rate: e_p - e_m })
        });

        let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;
        shared.sky.store(Arc::new(SkySnapshot {
            jd_tt,
            utc_iso: utc_iso(jd_tt),
            n_visible: sats.len(),
            sats,
            stars: star_marks.clone(),
            bodies: body_marks,
            dsos: dso_marks.clone(),
            launches: launches.clone(),
            star_names: star_names.clone(),
            target,
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
