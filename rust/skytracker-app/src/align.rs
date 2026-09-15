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
    let base = cam.tetra3_db.clone().unwrap_or_else(|| "db_cam1_tyc".into());
    // Sim frames are a centred crop (~1.28° for the guide cam), well below
    // the hardware full-frame FOV the main DB indexes — prefer a matching
    // "<name>_sim" DB when the cameras are simulated, fall back otherwise.
    let mut names = Vec::new();
    if cfg.camera_source.eq_ignore_ascii_case("sim") {
        names.push(format!("{base}_sim"));
    }
    names.push(base.clone());
    for name in &names {
        for dir in cfg.tetra3_search_dirs() {
            let p = dir.join(format!("{name}.npz"));
            if p.exists() {
                return Ok((name.clone(), p));
            }
        }
    }
    Err(format!("{base}.npz not found (set tetra3_db_dir or SKYTRACKER_TETRA3_DIR)"))
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
    // Continuous background solving (plate_solve_enabled) + polar align state.
    let mut continuous = cfg.raw["plate_solve_enabled"].as_bool().unwrap_or(false);
    let mut last_auto_solve = Instant::now();
    let mut align_az_live = cfg.alignment_az;
    struct PolarRun {
        targets: Vec<f64>,
        alt_axis: f64,
        idx: usize,
        samples: Vec<[f64; 2]>,
        awaiting: bool,
        started: Instant,
    }
    let mut polar: Option<PolarRun> = None;
    let mut polar_result: Option<(f64, f64, f64, f64, usize)> = None;
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
                AlignCmd::Abort => {
                    runner.abort();
                    if polar.is_some() {
                        polar = None;
                        align.log.push("polar align aborted".into());
                    }
                }
                AlignCmd::RetryFailed => {
                    runner.retry_failed(crate::sky::now_unix());
                }
                AlignCmd::Skip => runner.skip(),
                AlignCmd::QuickRefit => {
                    let (_, seed) = **shared.pointing.load();
                    let m = shared.mount.load();
                    runner.start_quick(seed, m.az, m.el, crate::sky::now_unix());
                }
                AlignCmd::Continuous(on) => {
                    continuous = on;
                    crate::mount::persist_config_key(&cfg.path, "plate_solve_enabled", serde_json::json!(on));
                    align.log.push(format!("continuous plate solve {}", if on { "ON" } else { "OFF" }));
                }
                AlignCmd::ApplyAlign => {
                    // Fold the latest solve's azimuth error into alignment_azimuth
                    // (apply_instantaneous_alignment): one-star align.
                    if snap.last_ok {
                        let daz = (snap.true_az - snap.mount_az + 540.0).rem_euclid(360.0) - 180.0;
                        align_az_live += daz;
                        let _ = mount_tx.send(MountCmd::SetAlignmentOffsets { az: align_az_live, el: cfg.alignment_el });
                        align.log.push(format!("instantaneous align: alignment_azimuth {:+.4}° -> {:.4}°", daz, align_az_live));
                    } else {
                        align.log.push("apply align: no good solve yet".into());
                    }
                }
                AlignCmd::PolarStart { n_points, sweep_deg } => {
                    let m = shared.mount.load();
                    let n = n_points.clamp(3, 24);
                    let half = sweep_deg.abs().max(10.0) / 2.0;
                    let targets: Vec<f64> = (0..n).map(|i| m.azm - half + sweep_deg.abs() * i as f64 / (n - 1) as f64).collect();
                    polar = Some(PolarRun { targets, alt_axis: m.alt, idx: 0, samples: Vec::new(), awaiting: false, started: Instant::now() });
                    polar_result = None;
                    align.log.push(format!("polar align: sweeping RA axis {half:.0}° each side, {n} solves"));
                }
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
                        if (**shared.mount_mode.load()).eq_ignore_ascii_case("eq") {
                            align.log.push("note: this fit is ALT-AZ; the Eq residual model (eq_pointing_model_*) is separate and unchanged".into());
                        }
                    } else {
                        align.log.push("no fit to apply yet".into());
                    }
                }
            }
        }

        // Runner step: it may ask for a slew, a solve, or a pause.
        let m = shared.mount.load();
        let now = crate::sky::now_unix();
        // Polar-align sweep: goto next axis target, solve when arrived.
        if let Some(p) = polar.as_mut() {
            if p.started.elapsed() > Duration::from_secs(600) {
                align.log.push("polar align TIMED OUT".into());
                polar = None;
            } else if p.idx >= p.targets.len() {
                let (pole_az, pole_el) = if cfg.lat_deg >= 0.0 { (0.0, cfg.lat_deg) } else { (180.0, -cfg.lat_deg) };
                if let Some((ax_az, ax_el)) = skytracker_pointing::polar::fit_polar_axis(&p.samples, pole_az, pole_el) {
                    let daz = (ax_az - pole_az + 540.0).rem_euclid(360.0) - 180.0;
                    let del = ax_el - pole_el;
                    polar_result = Some((ax_az, ax_el, daz, del, p.samples.len()));
                    align.log.push(format!(
                        "polar axis: az {ax_az:.3}° el {ax_el:.3}° — turn base {daz:+.3}° {}, tilt {del:+.3}° {}",
                        if daz > 0.0 { "west" } else { "east" },
                        if del > 0.0 { "down" } else { "up" }
                    ));
                } else {
                    align.log.push(format!("polar align: fit failed ({} samples)", p.samples.len()));
                }
                polar = None;
            } else if !p.awaiting {
                let tgt = p.targets[p.idx];
                let d = ((m.azm - tgt + 540.0).rem_euclid(360.0) - 180.0).abs();
                if d < 0.2 {
                    p.awaiting = true;
                    want_solve = true;
                } else {
                    let _ = mount_tx.send(MountCmd::GotoAxes { azm: tgt, alt: p.alt_axis });
                }
            }
        }
        if continuous && !want_solve && polar.is_none() && !align.running && last_auto_solve.elapsed() > Duration::from_secs(2) {
            last_auto_solve = Instant::now();
            want_solve = true;
        }
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
                            if let Some(p) = polar.as_mut() {
                                if p.awaiting {
                                    p.samples.push([s.true_az, s.true_el]);
                                    p.idx += 1;
                                    p.awaiting = false;
                                }
                            }
                        }
                        Err(e) => {
                            snap.last_ok = false;
                            snap.message = format!("solve failed: {e}");
                            snap.frame_seq = f.seq;
                            snap.centroids.clear();
                            snap.matched.clear();
                            runner.on_solve(now, None, m.az, m.el);
                            if let Some(p) = polar.as_mut() {
                                if p.awaiting {
                                    align.log.push(format!("polar point {} failed to solve — skipped", p.idx + 1));
                                    p.idx += 1;
                                    p.awaiting = false;
                                }
                            }
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
        align.manual = st.manual;
        align.n_failed = st.failed.len();
        align.failed = st.failed;
        align.continuous = continuous;
        align.polar = polar_result;
        align.polar_running = polar.is_some();
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
