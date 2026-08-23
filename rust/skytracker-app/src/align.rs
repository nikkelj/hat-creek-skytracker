//! Alignment worker: plate-solves the live camera frame (skytracker-platesolve,
//! the tetra3 port) into a true sky direction, and runs the alignment
//! sequence (skytracker-pointing's runner) that collects (mount, true)
//! samples and fits the 7-term pointing model. All off the UI thread; the
//! UI sees SolveSnapshot / AlignSnapshot and sends AlignCmd.

use crate::state::{AlignCmd, AlignSample, AlignSnapshot, MountCmd, Shared, SolveSnapshot};
use skytracker_astro::apparent::FrameContext;
use skytracker_astro::sgp4_pass::{Observer, ObserverGeometry};
use skytracker_platesolve::centroid::{get_centroids, CentroidParams};
use skytracker_platesolve::db::SolverDatabase;
use skytracker_platesolve::solve::{solve_from_centroids, SolveParams};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn spawn(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<AlignCmd>, mount_tx: crossbeam_channel::Sender<MountCmd>) {
    std::thread::Builder::new()
        .name("align-worker".into())
        .spawn(move || run(shared, rx, mount_tx))
        .expect("spawn align worker");
}

fn find_db(shared: &Shared) -> Result<(String, std::path::PathBuf), String> {
    let cfg = &shared.config;
    let cam = &cfg.cam[shared.solve_slot()];
    let name = cam.tetra3_db.clone().unwrap_or_else(|| "db_cam1_tyc".into());
    for dir in cfg.tetra3_search_dirs() {
        let p = dir.join(format!("{name}.npz"));
        if p.exists() {
            return Ok((name, p));
        }
    }
    Err(format!("{name}.npz not found (set tetra3_db_dir or SKYTRACKER_TETRA3_DIR)"))
}

pub struct Solved {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    pub fov_deg: f64,
    pub rmse_arcsec: f64,
    pub matches: usize,
    pub n_centroids: usize,
    pub true_az: f64,
    pub true_el: f64,
    pub centroids: Vec<[f64; 2]>,
    pub matched: Vec<[f64; 2]>,
    pub solve_ms: f64,
}

/// Solve one mono frame; RA/Dec (catalogue equinox, treated as ICRS) ->
/// topocentric az/el at the frame's exposure midpoint.
pub fn solve_frame(db: &SolverDatabase, data: &[u8], w: usize, h: usize, fov_deg: f64, jd_tt: f64, geom: &ObserverGeometry) -> Result<Solved, String> {
    let t0 = Instant::now();
    let cents = get_centroids(data, h, w, &CentroidParams::default());
    if cents.len() < 4 {
        return Err(format!("only {} centroids", cents.len()));
    }
    let params = SolveParams {
        fov_estimate_deg: fov_deg,
        fov_max_error_deg: Some(fov_deg * 0.6),
        pattern_checking_stars: 12,
        match_radius: 0.02,
        match_threshold: 1e-2,
        distortion: 0.0,
    };
    let sol = solve_from_centroids(db, &cents, h, w, &params).ok_or_else(|| format!("no solution ({} centroids)", cents.len()))?;
    let (sr, cr) = sol.ra_deg.to_radians().sin_cos();
    let (sd, cd) = sol.dec_deg.to_radians().sin_cos();
    let ctx = FrameContext::new(jd_tt);
    let (alt, az) = ctx.altaz_from_icrs(&[cd * cr, cd * sr, sd], geom);
    Ok(Solved {
        ra_deg: sol.ra_deg,
        dec_deg: sol.dec_deg,
        roll_deg: sol.roll_deg,
        fov_deg: sol.fov_deg,
        rmse_arcsec: sol.rmse_arcsec,
        matches: sol.matches,
        n_centroids: cents.len(),
        true_az: az,
        true_el: alt,
        centroids: cents,
        matched: sol.matched_centroids,
        solve_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

fn run(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<AlignCmd>, mount_tx: crossbeam_channel::Sender<MountCmd>) {
    let cfg = shared.config.clone();
    let observer = Observer {
        lat_deg: cfg.lat_deg,
        lon_deg: cfg.lon_deg,
        elevation_m: cfg.alt_m,
    };
    let geom = observer.geometry();
    let mut db: Option<SolverDatabase> = None;
    let mut snap = SolveSnapshot::default();
    match find_db(&shared) {
        Ok((name, path)) => {
            snap.db_name = name.clone();
            let t = Instant::now();
            match SolverDatabase::load(&path) {
                Ok(d) => {
                    snap.db_loaded = true;
                    snap.message = format!(
                        "{name} loaded in {:.1} s · FOV {:.2}-{:.2}°",
                        t.elapsed().as_secs_f64(),
                        d.props.min_fov_deg,
                        d.props.max_fov_deg
                    );
                    db = Some(d);
                }
                Err(e) => snap.message = format!("{name}: load failed: {e:?}"),
            }
        }
        Err(e) => snap.message = e,
    }
    shared.solve.store(Arc::new(snap.clone()));

    let mut runner = crate::align_runner::Runner::new(&cfg);
    let mut align = AlignSnapshot::default();
    align.status = "idle".into();
    shared.align.store(Arc::new(align.clone()));

    loop {
        let mut want_solve = false;
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                AlignCmd::SolveNow => want_solve = true,
                AlignCmd::Start { n_points, supervised } => {
                    let m = shared.mount.load();
                    runner.start(n_points, supervised, m.az, m.el, crate::sky::now_unix());
                    align.log.clear();
                }
                AlignCmd::Accept => runner.user(true),
                AlignCmd::Reject => runner.user(false),
                AlignCmd::Abort => runner.abort(),
                AlignCmd::ApplyModel => {
                    if let Some(t) = runner.terms() {
                        shared.pointing.store(Arc::new((true, t)));
                        // Persist like accept_alignment: terms by name + enabled.
                        if let Ok(text) = std::fs::read_to_string(&cfg.path) {
                            if let Ok(mut raw) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some(o) = raw.as_object_mut() {
                                    let m: serde_json::Map<String, serde_json::Value> = skytracker_pointing::altaz::TERM_NAMES.iter().zip(t.iter()).map(|(n, v)| (n.to_string(), serde_json::json!(v))).collect();
                                    o.insert("pointing_model_terms".into(), serde_json::Value::Object(m));
                                    o.insert("pointing_model_enabled".into(), serde_json::json!(true));
                                    if let Ok(s) = serde_json::to_string_pretty(&raw) {
                                        let _ = std::fs::write(&cfg.path, s);
                                    }
                                }
                            }
                        }
                        align.log.push(format!("pointing model APPLIED + saved: {}", skytracker_pointing::altaz::TERM_NAMES.iter().zip(t.iter()).map(|(n, v)| format!("{n} {:+.2}′", v * 60.0)).collect::<Vec<_>>().join("  ")));
                    } else {
                        align.log.push("no fit to apply yet".into());
                    }
                }
            }
        }

        // Runner step: it may ask for a slew, a solve, or a pause.
        let m = shared.mount.load();
        let now = crate::sky::now_unix();
        match runner.step(now, m.az, m.el) {
            crate::align_runner::Step::Idle => {}
            crate::align_runner::Step::Slew { az, el } => {
                let _ = mount_tx.send(MountCmd::Goto { az, el });
            }
            crate::align_runner::Step::Solve => want_solve = true,
            crate::align_runner::Step::Wait => {}
        }

        if want_solve {
            let cam = shared.cam(shared.solve_slot());
            match (db.as_ref(), cam.as_ref()) {
                (Some(d), Some(f)) => {
                    snap.busy = true;
                    shared.solve.store(Arc::new(snap.clone()));
                    let jd_tt = skytracker_astro::time::utc_to_tt(2440587.5 + f.utc_midpoint_s / 86400.0);
                    let m = shared.mount.load();
                    match solve_frame(d, &f.data, f.width, f.height, f.fov_deg, jd_tt, &geom) {
                        Ok(s) => {
                            snap.last_ok = true;
                            snap.message = format!("solved: {} matches, rmse {:.1}″, {:.0} ms", s.matches, s.rmse_arcsec, s.solve_ms);
                            snap.ra_deg = s.ra_deg;
                            snap.dec_deg = s.dec_deg;
                            snap.roll_deg = s.roll_deg;
                            snap.fov_deg = s.fov_deg;
                            snap.rmse_arcsec = s.rmse_arcsec;
                            snap.matches = s.matches;
                            snap.n_centroids = s.n_centroids;
                            snap.solve_ms = s.solve_ms;
                            snap.true_az = s.true_az;
                            snap.true_el = s.true_el;
                            snap.mount_az = m.az;
                            snap.mount_el = m.el;
                            snap.centroids = s.centroids;
                            snap.matched = s.matched;
                            snap.frame_seq = f.seq;
                            runner.on_solve(now, Some((s.true_az, s.true_el, s.rmse_arcsec)), m.az, m.el);
                        }
                        Err(e) => {
                            snap.last_ok = false;
                            snap.message = format!("solve failed: {e}");
                            snap.frame_seq = f.seq;
                            snap.centroids.clear();
                            snap.matched.clear();
                            runner.on_solve(now, None, m.az, m.el);
                        }
                    }
                    snap.busy = false;
                    shared.solve.store(Arc::new(snap.clone()));
                }
                (None, _) => {
                    snap.message = "no solver database".into();
                    shared.solve.store(Arc::new(snap.clone()));
                    runner.on_solve(now, None, m.az, m.el);
                }
                _ => {}
            }
        }

        // Publish runner state.
        let st = runner.snapshot();
        align.running = st.running;
        align.status = st.status;
        align.action = st.action;
        align.point = st.point;
        align.n_points = st.n_points;
        align.targets = st.targets;
        align.samples = st
            .samples
            .iter()
            .map(|s| AlignSample {
                mount_az: s.0,
                mount_el: s.1,
                true_az: s.2,
                true_el: s.3,
                residual_arcsec: s.4,
            })
            .collect();
        align.terms = st.terms;
        align.rms_arcsec = st.rms_arcsec;
        for l in st.new_log {
            align.log.push(l);
            if align.log.len() > 40 {
                align.log.remove(0);
            }
        }
        shared.align.store(Arc::new(align.clone()));

        std::thread::sleep(Duration::from_millis(50));
    }
}
