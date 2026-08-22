//! Camera worker. Source is either the synthetic star-field simulator (the
//! hardware simulator's essence in Rust — Tycho/Hipparcos stars and the
//! live satellites projected through camera 1's pinhole at the mount's
//! *true* pointing, i.e. reported pose + injected misalignment + periodic
//! error, Gaussian PSFs + read noise) or a real ZWO ASI camera through the
//! SDK binding (config `camera_source`). Either way frames go through the
//! real skytracker-camera pump/ring, the newest frame is published for the
//! display, and in HANDOFF/HOTSPOT it is also pushed straight into the
//! core loop's frame slot — the in-process camera->hotspot feed.

use crate::deepsky::DeepCatalog;
use crate::state::{CamCmd, CamSnapshot, Shared};
use skytracker_camera::capture::CaptureRecorder;
use skytracker_camera::pump::{Pump, PushShared, PushSource};
use skytracker_camera::ring::Frame;
use skytracker_core::controller::{Frame as LoopFrame, Mode};
use skytracker_core::hotspot::angles_to_pixel_offset;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const CAM_W: usize = 1248;
pub const CAM_H: usize = 936;
pub const CAM_FPS: f64 = 100.0;

pub fn spawn(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<CamCmd>, repo_root: std::path::PathBuf) {
    std::thread::Builder::new()
        .name("camera".into())
        .spawn(move || run(shared, rx, repo_root))
        .expect("spawn camera worker");
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }
}

struct Projector {
    w: f64,
    h: f64,
    pixel_um: f64,
    focal_mm: f64,
    rotation_deg: f64,
    x_sign: f64,
    y_sign: f64,
}

impl Projector {
    /// Sky direction -> pixel, through the exact inverse of the hotspot
    /// pixel->angle mapping, so the closed loop and the simulator agree by
    /// construction (the hardware signs still need rig calibration).
    fn project(&self, bore_az: f64, bore_el: f64, az: f64, el: f64) -> Option<(f64, f64)> {
        let daz = (az - bore_az + 540.0).rem_euclid(360.0) - 180.0;
        let del = el - bore_el;
        if daz.abs() > 20.0 || del.abs() > 20.0 {
            return None;
        }
        let (dx, dy) = angles_to_pixel_offset(
            daz, del, self.pixel_um, self.focal_mm, self.rotation_deg, bore_el, self.x_sign, self.y_sign, true,
        );
        let x = self.w / 2.0 + dx;
        let y = self.h / 2.0 + dy;
        if x < -6.0 || y < -6.0 || x > self.w + 6.0 || y > self.h + 6.0 {
            return None;
        }
        Some((x, y))
    }
}

fn splat(buf: &mut [u8], w: usize, h: usize, x: f64, y: f64, amp: f32, sigma: f32) {
    let r = (sigma * 3.0).ceil() as i64;
    let (xi, yi) = (x.round() as i64, y.round() as i64);
    for yy in (yi - r).max(0)..=(yi + r).min(h as i64 - 1) {
        for xx in (xi - r).max(0)..=(xi + r).min(w as i64 - 1) {
            let dx = xx as f64 - x;
            let dy = yy as f64 - y;
            let g = amp * (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp();
            let i = yy as usize * w + xx as usize;
            buf[i] = (buf[i] as f32 + g).clamp(0.0, 255.0) as u8;
        }
    }
}

struct SimRenderer {
    proj: Projector,
    rng: Lcg,
    deep: Arc<Mutex<Option<DeepCatalog>>>,
    deep_cache: Vec<(f64, f64, f64)>, // az, el, mag
    deep_cache_at: Option<(f64, f64, Instant)>,
    geom: skytracker_astro::sgp4_pass::ObserverGeometry,
    t0: Instant,
}

impl SimRenderer {
    fn render(&mut self, shared: &Shared, buf: &mut [u8]) -> usize {
        let mount = shared.mount.load();
        let sky = shared.sky.load();
        let sim = shared.sim.load();
        let (w, h) = (CAM_W, CAM_H);
        // Background + read noise.
        let bg = sim.background_level as f32;
        let rn = sim.read_noise as f32;
        for v in buf.iter_mut() {
            let n = self.rng.next() * 2.0 - 1.0;
            *v = (bg + n * rn * 1.7).clamp(0.0, 255.0) as u8;
        }
        // True boresight = reported pose + misalignment + periodic error.
        let t = self.t0.elapsed().as_secs_f64();
        let pe = if sim.pe_period_s > 0.0 {
            sim.pe_amplitude_deg * (t / sim.pe_period_s * std::f64::consts::TAU).sin()
        } else {
            0.0
        };
        let bore_az = mount.az + sim.misalign_az_deg + pe;
        let bore_el = mount.el + sim.misalign_el_deg;
        let fov_w = self.proj.w * ((self.proj.pixel_um * 1e-3) / self.proj.focal_mm).to_degrees();
        let fov_h = fov_w * self.proj.h / self.proj.w;
        let half_diag = (fov_w * fov_w + fov_h * fov_h).sqrt() / 2.0 + 0.05;

        // Stars: deep catalogue cone (refreshed at 10 Hz or on motion), else Hipparcos.
        let mut deep_n = 0;
        let use_deep = sim.use_deep_catalog;
        if use_deep {
            let stale = match self.deep_cache_at {
                None => true,
                Some((a, e, at)) => {
                    at.elapsed() > Duration::from_millis(100) || (a - bore_az).abs() > 1e-4 || (e - bore_el).abs() > 1e-4
                }
            };
            if stale {
                if let Some(cat) = self.deep.lock().unwrap().as_ref() {
                    let jd_tt = crate::sky::now_jd_tt();
                    self.deep_cache = cat
                        .stars_near(jd_tt, &self.geom, bore_az, bore_el, half_diag, sim.star_limit_mag, 1500)
                        .into_iter()
                        .map(|s| (s.az, s.el, s.mag))
                        .collect();
                    self.deep_cache_at = Some((bore_az, bore_el, Instant::now()));
                }
            }
            deep_n = self.deep_cache.len();
            for &(az, el, mag) in &self.deep_cache {
                if let Some((x, y)) = self.proj.project(bore_az, bore_el, az, el) {
                    // mag 9 saturates at the star cap; mag 10 ~ 40% of it -- bright enough that
                    // every rendered (DB) star is a centroid for the solver.
                    // ...but never brighter than the target, which is what the
                    // hot-spot detector keys on (real passes: the sat is the brightest thing).
                    let amp = (sim.target_brightness * 1.27 * 10f64.powf(-0.4 * (mag - 9.0))).clamp(6.0, sim.target_brightness * 0.55) as f32;
                    splat(buf, w, h, x, y, amp, 1.25);
                }
            }
        }
        if !use_deep || deep_n == 0 {
            for s in &sky.stars {
                if let Some((x, y)) = self.proj.project(bore_az, bore_el, s.az, s.el) {
                    let amp = (250.0 * 10f64.powf(-0.4 * (s.mag - 1.0))).clamp(25.0, 250.0) as f32;
                    splat(buf, w, h, x, y, amp, 1.3);
                }
            }
        }
        // Satellites (the tracking targets).
        let age_s = ((crate::sky::now_jd_tt() - sky.jd_tt) * 86400.0).clamp(0.0, 5.0);
        for s in &sky.sats {
            let az = s.az + s.az_rate * age_s;
            let el = s.el + s.el_rate * age_s;
            if el > 0.0 {
                if let Some((x, y)) = self.proj.project(bore_az, bore_el, az, el) {
                    splat(buf, w, h, x, y, sim.target_brightness as f32, 1.8);
                }
            }
        }
        deep_n
    }
}

enum Source {
    Sim { push: Arc<PushShared>, renderer: SimRenderer },
    Asi,
}

fn open_asi(shared: &Shared, root: &std::path::Path) -> Result<(Pump, usize, usize), String> {
    use skytracker_camera::asi::{AsiSdk, AsiSource, ASI_EXPOSURE, ASI_GAIN, ASI_IMG_RAW8};
    let cfg = &shared.config;
    let dll = if std::path::Path::new(&cfg.asi_dll).is_absolute() {
        cfg.asi_dll.clone()
    } else {
        root.join(&cfg.asi_dll).to_string_lossy().to_string()
    };
    let sdk = AsiSdk::load(&dll).map_err(|e| e.0)?;
    let n = sdk.num_cameras().map_err(|e| e.0)?;
    if n < 1 {
        return Err("no ASI cameras found".into());
    }
    let info = sdk.camera_info(0).map_err(|e| e.0)?;
    let id = info.camera_id;
    let (w, h) = (info.max_width.max(16) as usize, info.max_height.max(16) as usize);
    let cam = &cfg.cam[0];
    sdk.open(id).map_err(|e| e.0)?;
    sdk.set_roi(id, w as i32, h as i32, 1, ASI_IMG_RAW8).map_err(|e| e.0)?;
    sdk.set_control(id, ASI_EXPOSURE, cam.exposure_ms.max(1) * 1000, false).map_err(|e| e.0)?;
    sdk.set_control(id, ASI_GAIN, cam.gain, false).map_err(|e| e.0)?;
    sdk.start_video(id).map_err(|e| e.0)?;
    let source = AsiSource {
        sdk,
        camera_id: id,
        width: w,
        height: h,
        channels: 1,
        wait_ms: 1000,
        stopped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    Ok((Pump::spawn(source, 600), w, h))
}

fn run(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<CamCmd>, root: std::path::PathBuf) {
    let cfg = shared.config.clone();
    let cam_cfg = cfg.cam[0].clone();
    let observer = skytracker_astro::sgp4_pass::Observer {
        lat_deg: cfg.lat_deg,
        lon_deg: cfg.lon_deg,
        elevation_m: cfg.alt_m,
    };
    let geom = observer.geometry();

    // Source selection.
    let mut source_name = "sim".to_string();
    let (pump, mut source, w, h) = if cfg.camera_source.eq_ignore_ascii_case("asi") {
        match open_asi(&shared, &root) {
            Ok((p, w, h)) => {
                source_name = "ASI".into();
                (p, Source::Asi, w, h)
            }
            Err(e) => {
                eprintln!("skytracker: ASI camera unavailable ({e}); using the simulator");
                source_name = format!("sim (ASI: {e})");
                let (s, push) = PushSource::new();
                (Pump::spawn(s, 600), Source::Sim { push, renderer: SimRenderer::placeholder(&cam_cfg, geom.clone_geom()) }, CAM_W, CAM_H)
            }
        }
    } else {
        let (s, push) = PushSource::new();
        (Pump::spawn(s, 600), Source::Sim { push, renderer: SimRenderer::placeholder(&cam_cfg, geom.clone_geom()) }, CAM_W, CAM_H)
    };

    // Deep catalogue loads in the background (371 MB text; a few seconds).
    if let Source::Sim { renderer, .. } = &mut source {
        let slot = renderer.deep.clone();
        let tyc = root.join("tyc_main.dat");
        let limit = cfg.sim.star_limit_mag.max(6.0) + 0.5;
        std::thread::Builder::new()
            .name("tycho-load".into())
            .spawn(move || {
                if let Ok(cat) = DeepCatalog::load(&tyc, limit) {
                    eprintln!("skytracker: Tycho deep catalogue loaded ({} stars <= mag {limit:.1})", cat.len());
                    *slot.lock().unwrap() = Some(cat);
                }
            })
            .ok();
    }

    let fov_deg = cam_cfg.fov_deg(w as f64).max(0.05);
    let recorder = Arc::new(CaptureRecorder::new());
    let period = Duration::from_secs_f64(1.0 / CAM_FPS);
    let mut fps_t0 = Instant::now();
    let mut fps_n = 0u64;
    let mut fps = 0.0;
    let mut last_seq = u64::MAX;
    let mut armed_frames = 0usize;
    let mut last_dump: Option<String> = None;
    let mut deep_n = 0usize;
    let core = shared.core.clone();

    loop {
        let t0 = Instant::now();

        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                CamCmd::Arm => {
                    recorder.arm();
                    armed_frames = 0;
                }
                CamCmd::Disarm => {
                    let _ = recorder.disarm_and_dump(&std::env::temp_dir().join("skytracker_discard"));
                    armed_frames = 0;
                }
                CamCmd::Dump { name } => {
                    let stamp = crate::sky::utc_stamp_compact();
                    let dir = root.join(&cfg.captures_dir).join(format!("{name}_{stamp}"));
                    match recorder.disarm_and_dump(&dir) {
                        Ok((n, times)) => {
                            // Python / replay layout: Camera1_000000__YYYY_MM_DD_HH_MM_SS.ffffffZ.bmp
                            for (i, t) in times.iter().enumerate() {
                                let (y, mo, d, hh, mm, ss) = crate::sky::civil_from_unix(*t);
                                let frac = ((t - t.floor()) * 1e6).round() as i64;
                                let new = dir.join(format!("Camera1_{i:06}__{y:04}_{mo:02}_{d:02}_{hh:02}_{mm:02}_{ss:02}.{frac:06}Z.bmp"));
                                let _ = std::fs::rename(dir.join(format!("frame_{i:05}.bmp")), new);
                            }
                            let mut csv = String::from("frame,utc_midpoint_unix\n");
                            for (i, t) in times.iter().enumerate() {
                                csv.push_str(&format!("{i},{t:.6}\n"));
                            }
                            let _ = std::fs::write(dir.join("timestamps.csv"), csv);
                            let meta = serde_json::json!({
                                "target": name,
                                "camera": "camera1",
                                "source": source_name,
                                "width": w, "height": h,
                                "fov_deg": fov_deg,
                                "frames": n,
                                "start_unix": times.first().copied().unwrap_or(0.0),
                                "end_unix": times.last().copied().unwrap_or(0.0),
                            });
                            let _ = std::fs::write(dir.join("run.json"), serde_json::to_string_pretty(&meta).unwrap());
                            last_dump = Some(format!("{n} frames -> {}", dir.display()));
                        }
                        Err(e) => last_dump = Some(format!("dump failed: {e}")),
                    }
                    armed_frames = 0;
                }
            }
        }

        if let Source::Sim { push, renderer } = &mut source {
            let mut buf = vec![0u8; w * h];
            deep_n = renderer.render(&shared, &mut buf);
            let render_s = t0.elapsed().as_secs_f64();
            push.push(buf, w, h, 1, render_s.max(1.0 / CAM_FPS));
        }

        // Publish the newest pumped frame (Arc, no copy).
        if let Some(f) = pump.ring.latest() {
            if f.seq != last_seq {
                last_seq = f.seq;
                fps_n += 1;
                if fps_t0.elapsed() >= Duration::from_secs(1) {
                    fps = fps_n as f64 / fps_t0.elapsed().as_secs_f64();
                    fps_n = 0;
                    fps_t0 = Instant::now();
                }
                if recorder.is_armed() {
                    recorder.offer(&f);
                    armed_frames += 1;
                }
                feed_loop(&core, &f);
                shared.cam.store(Arc::new(Some(CamSnapshot {
                    data: f.data.clone(),
                    width: f.width,
                    height: f.height,
                    seq: f.seq,
                    fps,
                    utc_midpoint_s: f.utc_midpoint_s,
                    fov_deg,
                    source: source_name.clone(),
                    armed: recorder.is_armed(),
                    armed_frames,
                    last_dump: last_dump.clone(),
                    deep_stars: deep_n,
                })));
            }
        }
        let sleep = period.saturating_sub(t0.elapsed());
        if !sleep.is_zero() {
            std::thread::sleep(sleep);
        }
    }
}

/// HANDOFF / HOTSPOT: hand the frame to the core loop as f32 with its
/// exposure-midpoint time on the loop clock.
fn feed_loop(core: &skytracker_core::core_loop::Shared, f: &Frame) {
    let mode = core.inputs.lock().unwrap().mode;
    if !matches!(mode, Mode::Handoff | Mode::Hotspot) {
        return;
    }
    let n = f.width * f.height;
    let data: Vec<f32> = f.data.iter().take(n).map(|&v| v as f32).collect();
    let age = (skytracker_camera::ring::now_unix() - f.utc_midpoint_s).max(0.0);
    let time = core.epoch.elapsed().as_secs_f64() - age;
    *core.frame.lock().unwrap() = Some(Arc::new(LoopFrame {
        data: Arc::new(data),
        h: f.height,
        w: f.width,
        seq: f.seq,
        time,
    }));
}

trait CloneGeom {
    fn clone_geom(&self) -> Self;
}
impl CloneGeom for skytracker_astro::sgp4_pass::ObserverGeometry {
    fn clone_geom(&self) -> Self {
        skytracker_astro::sgp4_pass::ObserverGeometry {
            pos_km: self.pos_km,
            east: self.east,
            north: self.north,
            up: self.up,
        }
    }
}

impl SimRenderer {
    fn placeholder(cam: &crate::state::CameraConfig, geom: skytracker_astro::sgp4_pass::ObserverGeometry) -> Self {
        SimRenderer {
            proj: Projector {
                w: CAM_W as f64,
                h: CAM_H as f64,
                pixel_um: cam.pixel_um,
                focal_mm: cam.focal_mm,
                rotation_deg: cam.alignment_rotation_deg,
                x_sign: 1.0,
                y_sign: -1.0,
            },
            rng: Lcg(0x1234_5678_9abc_def0),
            deep: Arc::new(Mutex::new(None)),
            deep_cache: Vec::new(),
            deep_cache_at: None,
            geom,
            t0: Instant::now(),
        }
    }
}
