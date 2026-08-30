//! Camera workers — one per logical slot (guide / main / bubble …). Each
//! slot's source is either the synthetic simulator (catalogue stars, live
//! satellites, sun/moon projected through the camera's model — pinhole on
//! the mount-borne scopes at the mount's *true* pointing, equidistant
//! fisheye fixed at the zenith for the bubble cam; Gaussian PSFs + read
//! noise) or a real ZWO ASI camera opened by the slot's hardware index
//! (config `camera_source`). Frames go through the real skytracker-camera
//! pump/ring, the newest frame is published per slot, live settings (gain /
//! exposure / ROI) are applied on change, and the HOTSPOT camera's frames
//! are pushed straight into the core loop's frame slot in HANDOFF/HOTSPOT.
//! Captures arm every connected camera into one run directory (CameraN_
//! files, the Python layout). "Swap cameras" exchanges the hardware index
//! behind two slots and reconnects them.

use crate::deepsky::DeepCatalog;
use crate::state::{CamCmd, CamSettings, CamSnapshot, CameraConfig, Projection, Shared};
use skytracker_camera::capture::CaptureRecorder;
use skytracker_camera::pump::{Pump, PushShared, PushSource};
use skytracker_camera::ring::Frame;
use skytracker_core::controller::{Frame as LoopFrame, Mode};
use skytracker_core::hotspot::angles_to_pixel_offset;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const CAM_FPS: f64 = 100.0;
/// Sim frames are a centred crop of the (binned) sensor so the plate scale
/// stays that of the real camera while the per-frame cost stays bounded.
pub const SIM_MAX_W: usize = 1248;
pub const SIM_MAX_H: usize = 936;

/// Frame size the simulator produces for a camera (centred crop of the binned sensor).
pub fn sim_frame_size(c: &CameraConfig) -> (usize, usize) {
    let (w, h) = c.frame_size();
    (w.min(SIM_MAX_W), h.min(SIM_MAX_H))
}

/// Legacy width used by callers that need "the" frame width of the solve camera.
pub const CAM_W: usize = SIM_MAX_W;

pub fn spawn(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<CamCmd>, repo_root: std::path::PathBuf) {
    // One channel per slot; a router fans commands out (Arm/Dump/Disarm to all).
    let n = shared.config.cam.len();
    let mut txs = Vec::new();
    let deep: Arc<Mutex<Option<DeepCatalog>>> = Arc::new(Mutex::new(None));
    // Deep catalogue loads once in the background for every sim camera.
    {
        let slot = deep.clone();
        let tyc = repo_root.join("tyc_main.dat");
        let limit = shared.config.sim.star_limit_mag.max(6.0) + 0.5;
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
    for slot in 0..n {
        let (tx, rx_slot) = crossbeam_channel::unbounded::<CamCmd>();
        txs.push(tx);
        let shared = shared.clone();
        let root = repo_root.clone();
        let deep = deep.clone();
        std::thread::Builder::new()
            .name(format!("camera{}", slot + 1))
            .spawn(move || run_slot(shared, slot, rx_slot, root, deep))
            .expect("spawn camera worker");
    }
    std::thread::Builder::new()
        .name("camera-router".into())
        .spawn(move || {
            while let Ok(cmd) = rx.recv() {
                match &cmd {
                    CamCmd::Apply { slot } => {
                        if let Some(t) = txs.get(*slot) {
                            let _ = t.send(cmd.clone());
                        }
                    }
                    CamCmd::Swap { a, b } => {
                        {
                            let mut hw = shared.cam_hw.lock().unwrap();
                            if *a < hw.len() && *b < hw.len() {
                                hw.swap(*a, *b);
                            }
                        }
                        for s in [*a, *b] {
                            if let Some(t) = txs.get(s) {
                                let _ = t.send(CamCmd::Apply { slot: s });
                            }
                        }
                    }
                    _ => {
                        for t in &txs {
                            let _ = t.send(cmd.clone());
                        }
                    }
                }
            }
        })
        .ok();
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
    projection: Projection,
}

impl Projector {
    /// Sky direction -> pixel. Pinhole: the exact inverse of the hotspot
    /// pixel->angle mapping (the closed loop and the simulator agree by
    /// construction). Fisheye: equidistant r = f*theta about the zenith,
    /// north up, east right, then the alignment rotation.
    fn project(&self, bore_az: f64, bore_el: f64, az: f64, el: f64) -> Option<(f64, f64)> {
        match self.projection {
            Projection::Pinhole => {
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
            Projection::FisheyeEquidistant => {
                if el < -2.0 {
                    return None;
                }
                let theta = (90.0 - el).to_radians();
                let r_px = self.focal_mm * theta / (self.pixel_um * 1e-3);
                let a = (az + self.rotation_deg).to_radians();
                let x = self.w / 2.0 + r_px * a.sin();
                let y = self.h / 2.0 - r_px * a.cos();
                if x < -6.0 || y < -6.0 || x > self.w + 6.0 || y > self.h + 6.0 {
                    return None;
                }
                Some((x, y))
            }
        }
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
    w: usize,
    h: usize,
    rng: Lcg,
    deep: Arc<Mutex<Option<DeepCatalog>>>,
    deep_cache: Vec<(f64, f64, f64)>, // az, el, mag
    deep_cache_at: Option<(f64, f64, Instant)>,
    geom: skytracker_astro::sgp4_pass::ObserverGeometry,
    t0: Instant,
    fixed_zenith: bool,
    /// Exposure/gain scale relative to the slot's config defaults.
    base_gain: f64,
    base_exposure: f64,
    slot: usize,
}

impl SimRenderer {
    fn render(&mut self, shared: &Shared, settings: &CamSettings, buf: &mut [u8]) -> usize {
        let mount = shared.mount.load();
        let sky = shared.sky.load();
        let sim = shared.sim.load();
        let (w, h) = (self.w, self.h);
        // Exposure / gain scale the signal (gain ~ 0.1 dB per unit on ASI; keep it gentle).
        let gain_scale = 10f64.powf((settings.gain as f64 - self.base_gain) / 200.0);
        let exp_scale = (settings.exposure_us.max(1) as f64 / self.base_exposure.max(1.0)).clamp(0.02, 50.0);
        let signal = (gain_scale * exp_scale) as f32;
        // Background + read noise.
        let bg = sim.background_level as f32 * exp_scale as f32;
        let rn = sim.read_noise as f32;
        for v in buf.iter_mut() {
            let n = self.rng.next() * 2.0 - 1.0;
            *v = (bg + n * rn * 1.7).clamp(0.0, 255.0) as u8;
        }
        // True boresight = reported pose + misalignment + periodic error (or the zenith).
        let t = self.t0.elapsed().as_secs_f64();
        let pe = if sim.pe_period_s > 0.0 { sim.pe_amplitude_deg * (t / sim.pe_period_s * std::f64::consts::TAU).sin() } else { 0.0 };
        let (mut bore_az, mut bore_el) = if self.fixed_zenith { (0.0, 90.0) } else { (mount.az + sim.misalign_az_deg + pe, mount.el + sim.misalign_el_deg) };
        self.proj.rotation_deg = settings.rotation_deg;
        // Camera-2 co-boresight offset injection (hw_sim cam2_offset_*):
        // pixel shift expressed as a sky offset at this plate scale + a
        // rotation delta, exercised by the combined-view calibration.
        if self.slot == 1 && !self.fixed_zenith && (sim.cam2_dx_px != 0.0 || sim.cam2_dy_px != 0.0 || sim.cam2_rot_deg != 0.0) {
            let deg_per_px = ((self.proj.pixel_um * 1e-3) / self.proj.focal_mm).to_degrees();
            let cos_el = bore_el.to_radians().cos().max(0.087);
            bore_az += sim.cam2_dx_px * deg_per_px / cos_el;
            bore_el += sim.cam2_dy_px * deg_per_px;
            self.proj.rotation_deg += sim.cam2_rot_deg;
        }

        let mut deep_n = 0;
        if self.fixed_zenith {
            // Whole sky: Hipparcos stars, bodies, every satellite above the horizon.
            for s in &sky.stars {
                if let Some((x, y)) = self.proj.project(bore_az, bore_el, s.az, s.el) {
                    let amp = (255.0 * 10f64.powf(-0.4 * (s.mag - 1.5))).clamp(4.0, 255.0) as f32 * signal;
                    splat(buf, w, h, x, y, amp, 1.2);
                }
            }
            for b in &sky.bodies {
                if let Some((x, y)) = self.proj.project(bore_az, bore_el, b.az, b.el) {
                    let (amp, sigma) = match b.name.as_str() {
                        "sun" => (255.0, 6.0),
                        "moon" => (255.0, 4.0),
                        _ => (200.0, 1.5),
                    };
                    splat(buf, w, h, x, y, amp, sigma);
                }
            }
        } else {
            let fov_w = self.proj.w * ((self.proj.pixel_um * 1e-3) / self.proj.focal_mm).to_degrees();
            let fov_h = fov_w * self.proj.h / self.proj.w;
            let half_diag = (fov_w * fov_w + fov_h * fov_h).sqrt() / 2.0 + 0.05;
            let use_deep = sim.use_deep_catalog;
            if use_deep {
                let stale = match self.deep_cache_at {
                    None => true,
                    Some((a, e, at)) => at.elapsed() > Duration::from_millis(100) || (a - bore_az).abs() > 1e-4 || (e - bore_el).abs() > 1e-4,
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
                        // mag 9 saturates at the star cap; never brighter than the target.
                        let amp = (sim.target_brightness * 1.27 * 10f64.powf(-0.4 * (mag - 9.0))).clamp(6.0, sim.target_brightness * 0.55) as f32 * signal;
                        splat(buf, w, h, x, y, amp, 1.25);
                    }
                }
            }
            if !use_deep || deep_n == 0 {
                for s in &sky.stars {
                    if let Some((x, y)) = self.proj.project(bore_az, bore_el, s.az, s.el) {
                        let amp = (250.0 * 10f64.powf(-0.4 * (s.mag - 1.0))).clamp(25.0, 250.0) as f32 * signal;
                        splat(buf, w, h, x, y, amp, 1.3);
                    }
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
                    let amp = if self.fixed_zenith { (sim.target_brightness * (800.0 / s.range_km.max(300.0)).min(1.0)) as f32 } else { sim.target_brightness as f32 };
                    splat(buf, w, h, x, y, amp * signal, if self.fixed_zenith { 1.2 } else { 1.8 });
                }
            }
        }
        deep_n
    }
}

enum Source {
    Sim { push: Arc<PushShared>, renderer: SimRenderer },
    Asi { sdk: Arc<skytracker_camera::asi::AsiSdk>, id: i32, stop: Arc<std::sync::atomic::AtomicBool> },
    None,
}

fn open_asi(cfg: &crate::state::Config, cam: &CameraConfig, hw_index: usize, settings: &CamSettings, root: &std::path::Path) -> Result<(Pump, usize, usize, Arc<skytracker_camera::asi::AsiSdk>, i32, Arc<std::sync::atomic::AtomicBool>), String> {
    use skytracker_camera::asi::{AsiSdk, AsiSource, ASI_EXPOSURE, ASI_GAIN, ASI_IMG_RAW8};
    // Serialize enumerate→open→configure across slot workers: the SDK's
    // enumeration is not thread-safe and concurrent slots were handed the
    // same camera (guide frames showing up in the main view).
    let _open_guard = skytracker_camera::asi::OPEN_LOCK.lock().unwrap();
    let dll = if std::path::Path::new(&cfg.asi_dll).is_absolute() { cfg.asi_dll.clone() } else { root.join(&cfg.asi_dll).to_string_lossy().to_string() };
    let sdk = AsiSdk::load(&dll).map_err(|e| e.0)?;
    let n = sdk.num_cameras().map_err(|e| e.0)?;
    if hw_index as i32 >= n {
        return Err(format!("hardware index {hw_index} but only {n} ASI camera(s) found"));
    }
    let info = sdk.camera_info(hw_index as i32).map_err(|e| e.0)?;
    let id = info.camera_id;
    let bin = cam.bin.max(1) as i32;
    let (fw, fh) = ((info.max_width.max(16) as usize) / bin as usize, (info.max_height.max(16) as usize) / bin as usize);
    // Hardware ROI (camera_manager.set_camera_roi): fraction of the binned
    // sensor, width a multiple of 8, height of 2, centred on (cx, cy) and
    // clamped inside the sensor.
    let frac = settings.roi_frac.clamp(0.03, 1.0);
    let (w, h) = ((((fw as f64 * frac) as usize) / 8 * 8).max(8), (((fh as f64 * frac) as usize) / 2 * 2).max(2));
    let sx = ((settings.roi_cx * fw as f64) as i64 - w as i64 / 2).clamp(0, fw as i64 - w as i64) as i32;
    let sy = ((settings.roi_cy * fh as f64) as i64 - h as i64 / 2).clamp(0, fh as i64 - h as i64) as i32;
    sdk.open(id).map_err(|e| e.0)?;
    sdk.set_roi(id, w as i32, h as i32, bin, ASI_IMG_RAW8).map_err(|e| e.0)?;
    if frac < 0.999 {
        sdk.set_start_pos(id, sx, sy).map_err(|e| e.0)?;
    }
    sdk.set_control(id, ASI_EXPOSURE, settings.exposure_us.max(1), false).map_err(|e| e.0)?;
    sdk.set_control(id, ASI_GAIN, settings.gain, false).map_err(|e| e.0)?;
    sdk.start_video(id).map_err(|e| e.0)?;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sdk = Arc::new(sdk);
    // AsiSource owns an SDK handle too; load a second binding for it.
    let sdk2 = AsiSdk::load(&dll).map_err(|e| e.0)?;
    let source = AsiSource { sdk: sdk2, camera_id: id, width: w, height: h, channels: 1, wait_ms: 1000, stopped: stop.clone() };
    Ok((Pump::spawn(source, 600), w, h, sdk, id, stop))
}

fn run_slot(shared: Arc<Shared>, slot: usize, rx: crossbeam_channel::Receiver<CamCmd>, root: std::path::PathBuf, deep: Arc<Mutex<Option<DeepCatalog>>>) {
    let cfg = shared.config.clone();
    let cam_cfg = cfg.cam[slot].clone();
    let observer = skytracker_astro::sgp4_pass::Observer { lat_deg: cfg.lat_deg, lon_deg: cfg.lon_deg, elevation_m: cfg.alt_m };
    let geom = observer.geometry();
    let recorder = Arc::new(CaptureRecorder::new());
    // Sweep stale spool dirs left by a crashed/killed previous run.
    if let Ok(rd) = std::fs::read_dir(root.join(&cfg.captures_dir)) {
        let prefix = format!(".spool_cam{}_", slot + 1);
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }
    let period = Duration::from_secs_f64(1.0 / CAM_FPS);
    let core = shared.core.clone();
    let is_hotspot_cam = slot == shared.hotspot_slot();

    let mut source = Source::None;
    let mut pump: Option<Pump> = None;
    let (mut w, mut h) = sim_frame_size(&cam_cfg);
    let mut source_name = String::from("off");
    let mut hw_index = shared.cam_hw.lock().unwrap()[slot];
    let mut settings = (**shared.cam_settings[slot].load()).clone();
    let mut reconnect = true;
    let mut fps_t0 = Instant::now();
    let mut fps_n = 0u64;
    let mut fps = 0.0;
    let mut last_seq = u64::MAX;
    let mut armed_frames = 0usize;
    let mut last_dump: Option<String> = None;
    let mut deep_n = 0usize;
    let mut dump_result: Option<Arc<Mutex<Option<String>>>> = None;
    let mut last_applied = settings.clone();
    let mut connected = false;

    loop {
        let t0 = Instant::now();

        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                CamCmd::Arm => {
                    if connected {
                        let spool = root.join(&cfg.captures_dir).join(format!(".spool_cam{}_{}", slot + 1, crate::sky::utc_stamp_compact()));
                        match recorder.arm_spool(&spool, cfg.capture_buffer_frames) {
                            Ok(()) => armed_frames = 0,
                            Err(e) => last_dump = Some(format!("arm failed: {e}")),
                        }
                    }
                }
                CamCmd::Disarm => {
                    // Discard: finish the spool off-thread and delete it.
                    if recorder.is_armed() {
                        recorder.disarm();
                        let rec = recorder.clone();
                        std::thread::Builder::new().name("capture-discard".into()).spawn(move || {
                            if let Ok((dir, _, _)) = rec.finish() {
                                let _ = std::fs::remove_dir_all(dir);
                            }
                        }).ok();
                    }
                    armed_frames = 0;
                }
                CamCmd::Dump { name } => {
                    if recorder.is_armed() {
                        // One run directory per capture: name_<stamp to the second>.
                        // The BMP writes run on their own thread so the capture
                        // pump (and the HOTSPOT frame feed) never stalls.
                        // Auto-name from the selection (capture_manager: NAME_NORAD),
                        // so replay can recover the target.
                        let name = if name.trim().is_empty() || name == "manual" {
                            let sel = (**shared.selected.load()).clone();
                            match sel {
                                Some(k) if !k.contains(':') => {
                                    let sky = shared.sky.load();
                                    sky.sats
                                        .iter()
                                        .find(|s| s.satnum == k)
                                        .map(|s| format!("{}_{}", sanitize_name(&s.name), s.satnum))
                                        .unwrap_or_else(|| format!("sat_{k}"))
                                }
                                Some(k) => sanitize_name(&k),
                                None => "manual".into(),
                            }
                        } else {
                            name
                        };
                        let stamp = crate::sky::utc_stamp_compact();
                        let dir = root.join(&cfg.captures_dir).join(format!("{name}_{stamp}"));
                        // Snapshot the selected target's arc now (worker thread has
                        // no sky access): rows become trajectory.csv in the dump.
                        let arc_t0 = crate::sky::now_unix();
                        let arc: Vec<crate::state::ArcPoint> = shared.passes.load().arc.clone();
                        let target_label = (**shared.selected.load()).clone().unwrap_or_else(|| "manual".into());
                        let config_path = cfg.path.clone();
                        let rec = recorder.clone();
                        let (w2, h2, src2, fov2, cname, crole, cn) = (w, h, source_name.clone(), cam_cfg.fov_deg(w as f64), cam_cfg.name.clone(), cam_cfg.role.clone(), name.clone());
                        let slot2 = slot;
                        let done = Arc::new(Mutex::new(None::<String>));
                        dump_result = Some(done.clone());
                        let png = cfg.image_format == "png";
                        recorder.disarm();
                        std::thread::Builder::new().name(format!("capture-dump{}", slot + 1)).spawn(move || {
                            // Most frames are already on disk in the spool dir;
                            // finish() only flushes the in-RAM tail, then the
                            // files move (same-volume rename) into the run dir.
                            let res = match rec.finish() {
                                Ok((spool_dir, times, dropped)) => {
                                    let n = times.len();
                                    let _ = std::fs::create_dir_all(&dir);
                                    for (i, t) in times.iter().enumerate() {
                                        let (y, mo, d, hh, mm, ss) = crate::sky::civil_from_unix(*t);
                                        let frac = ((t - t.floor()) * 1e6).round() as i64;
                                        let src = spool_dir.join(format!("frame_{i:05}.bmp"));
                                        if png {
                                            // image_format=png: re-encode the BMP (Python capture PNG option).
                                            let new = dir.join(format!("Camera{}_{i:06}__{y:04}_{mo:02}_{d:02}_{hh:02}_{mm:02}_{ss:02}.{frac:06}Z.png", slot2 + 1));
                                            match image::open(&src) {
                                                Ok(img) => {
                                                    let _ = img.save(&new);
                                                    let _ = std::fs::remove_file(&src);
                                                }
                                                Err(_) => {
                                                    let _ = std::fs::rename(&src, new.with_extension("bmp"));
                                                }
                                            }
                                        } else {
                                            let new = dir.join(format!("Camera{}_{i:06}__{y:04}_{mo:02}_{d:02}_{hh:02}_{mm:02}_{ss:02}.{frac:06}Z.bmp", slot2 + 1));
                                            let _ = std::fs::rename(&src, new);
                                        }
                                    }
                                    let _ = std::fs::remove_dir_all(&spool_dir);
                                    let meta = serde_json::json!({
                                        "target": cn, "camera": format!("camera{}", slot2 + 1), "name": cname, "role": crole,
                                        "source": src2, "width": w2, "height": h2, "fov_deg": fov2, "frames": n,
                                        "dropped": dropped,
                                        "start_unix": times.first().copied().unwrap_or(0.0), "end_unix": times.last().copied().unwrap_or(0.0),
                                    });
                                    let _ = std::fs::write(dir.join(format!("run_camera{}.json", slot2 + 1)), serde_json::to_string_pretty(&meta).unwrap());
                                    // trajectory.csv + config snapshot: written once per
                                    // run dir (first camera thread to finish wins).
                                    let csv_path = dir.join("trajectory.csv");
                                    if !csv_path.exists() && arc.len() >= 2 {
                                        let _ = std::fs::write(&csv_path, trajectory_csv(&arc, arc_t0, &target_label, &times, slot2));
                                    }
                                    let cfg_copy = dir.join(format!("config_{}.json", crate::sky::utc_stamp_compact()));
                                    if dir.read_dir().map(|mut d| !d.any(|e| e.as_ref().map(|e| e.file_name().to_string_lossy().starts_with("config_")).unwrap_or(false))).unwrap_or(false) {
                                        let _ = std::fs::copy(&config_path, cfg_copy);
                                    }
                                    if dropped > 0 {
                                        format!("{n} frames -> {} ({dropped} dropped: disk slower than capture)", dir.display())
                                    } else {
                                        format!("{n} frames -> {}", dir.display())
                                    }
                                }
                                Err(e) => format!("dump failed: {e}"),
                            };
                            *done.lock().unwrap() = Some(res);
                        }).ok();
                        last_dump = Some("saving…".into());
                    }
                    armed_frames = 0;
                }
                CamCmd::Apply { .. } => {
                    settings = (**shared.cam_settings[slot].load()).clone();
                    let new_hw = shared.cam_hw.lock().unwrap()[slot];
                    let roi_changed = (settings.roi_frac, settings.roi_cx, settings.roi_cy) != (last_applied.roi_frac, last_applied.roi_cx, last_applied.roi_cy);
                    if new_hw != hw_index || settings.connected != connected || (roi_changed && matches!(source, Source::Asi { .. })) {
                        hw_index = new_hw;
                        reconnect = true;
                    }
                }
                CamCmd::Swap { .. } => {}
            }
        }

        let finished = dump_result.as_ref().and_then(|d| d.lock().unwrap().clone());
        if let Some(r) = finished {
            last_dump = Some(r);
            dump_result = None;
        }

        // (Re)connect the source when asked.
        if reconnect {
            reconnect = false;
            // Tear down. The pump thread owns the close: signal stop, nudge a
            // blocking ASIGetVideoData with stop_video, then JOIN the pump so
            // the camera is fully closed before anything reopens it. Closing
            // here while the pump is mid-ASIGetVideoData hung and then
            // crashed the SDK.
            match &source {
                Source::Asi { sdk, id, stop } => {
                    stop.store(true, std::sync::atomic::Ordering::SeqCst);
                    let _ = sdk.stop_video(*id);
                }
                Source::Sim { push, .. } => push.close(),
                Source::None => {}
            }
            if let Some(p) = pump.take() {
                p.join();
            }
            source = Source::None;
            connected = false;
            if settings.connected {
                let want_asi = cfg.camera_source.eq_ignore_ascii_case("asi");
                if want_asi {
                    match open_asi(&cfg, &cam_cfg, hw_index, &settings, &root) {
                        Ok((p, pw, ph, sdk, id, stop)) => {
                            w = pw;
                            h = ph;
                            pump = Some(p);
                            source = Source::Asi { sdk, id, stop };
                            source_name = format!("ASI #{hw_index}");
                            connected = true;
                        }
                        Err(e) => {
                            eprintln!("skytracker: camera{} ASI #{hw_index} unavailable ({e}); using the simulator", slot + 1);
                            source_name = format!("sim (ASI #{hw_index}: {e})");
                        }
                    }
                }
                if !connected {
                    let (sw, sh) = sim_frame_size(&cam_cfg);
                    w = sw;
                    h = sh;
                    let (s, push) = PushSource::new();
                    pump = Some(Pump::spawn(s, 600));
                    let renderer = SimRenderer {
                        proj: Projector {
                            w: w as f64,
                            h: h as f64,
                            pixel_um: cam_cfg.pixel_eff_um(),
                            focal_mm: cam_cfg.focal_mm,
                            rotation_deg: settings.rotation_deg,
                            x_sign: 1.0,
                            y_sign: -1.0,
                            projection: cam_cfg.projection,
                        },
                        w,
                        h,
                        rng: Lcg(0x1234_5678_9abc_def0 ^ (slot as u64 * 0x9e37_79b9)),
                        deep: deep.clone(),
                        deep_cache: Vec::new(),
                        deep_cache_at: None,
                        geom: clone_geom(&geom),
                        t0: Instant::now(),
                        fixed_zenith: cam_cfg.fixed_zenith(),
                        slot,
                        base_gain: cam_cfg.gain as f64,
                        base_exposure: cam_cfg.exposure_us as f64,
                    };
                    source = Source::Sim { push, renderer };
                    if !want_asi {
                        source_name = "sim".into();
                    }
                    connected = true;
                }
            } else {
                source_name = "disconnected".into();
            }
            last_applied = settings.clone();
            if !connected {
                shared.cams[slot].store(Arc::new(Some(CamSnapshot {
                    slot,
                    name: cam_cfg.name.clone(),
                    role: cam_cfg.role.clone(),
                    data: Arc::new(Vec::new()),
                    width: 0,
                    height: 0,
                    seq: 0,
                    fps: 0.0,
                    utc_midpoint_s: 0.0,
                    fov_deg: cam_cfg.fov_deg(w as f64),
                    deg_per_px: 0.0,
                    fisheye: cam_cfg.fixed_zenith(),
                    source: source_name.clone(),
                    connected: false,
                    armed: false,
                    armed_frames: 0,
                    armed_dropped: 0,
                    last_dump: last_dump.clone(),
                    deep_stars: 0,
                    hw_index,
                })));
            }
        }

        // Live ASI controls.
        if let Source::Asi { sdk, id, .. } = &source {
            if settings.gain != last_applied.gain {
                let _ = sdk.set_control(*id, skytracker_camera::asi::ASI_GAIN, settings.gain, false);
            }
            if settings.exposure_us != last_applied.exposure_us {
                let _ = sdk.set_control(*id, skytracker_camera::asi::ASI_EXPOSURE, settings.exposure_us.max(1), false);
            }
            last_applied = settings.clone();
        }

        if !connected {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        if let Source::Sim { push, renderer } = &mut source {
            let mut buf = vec![0u8; w * h];
            deep_n = renderer.render(&shared, &settings, &mut buf);
            let render_s = t0.elapsed().as_secs_f64();
            if settings.roi_frac < 0.999 {
                // Emulate the hardware ROI: crop the rendered frame.
                let frac = settings.roi_frac.clamp(0.03, 1.0);
                let (rw, rh) = ((((w as f64 * frac) as usize) / 8 * 8).max(8), (((h as f64 * frac) as usize) / 2 * 2).max(2));
                let sx = ((settings.roi_cx * w as f64) as i64 - rw as i64 / 2).clamp(0, w as i64 - rw as i64) as usize;
                let sy = ((settings.roi_cy * h as f64) as i64 - rh as i64 / 2).clamp(0, h as i64 - rh as i64) as usize;
                let mut crop = vec![0u8; rw * rh];
                for y in 0..rh {
                    crop[y * rw..(y + 1) * rw].copy_from_slice(&buf[(sy + y) * w + sx..(sy + y) * w + sx + rw]);
                }
                push.push(crop, rw, rh, 1, render_s.max(1.0 / CAM_FPS));
            } else {
                push.push(buf, w, h, 1, render_s.max(1.0 / CAM_FPS));
            }
        }

        if let Some(p) = pump.as_ref() {
            if let Some(f) = p.ring.latest() {
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
                    if is_hotspot_cam {
                        feed_loop(&core, &f);
                    }
                    let fov = cam_cfg.fov_deg(f.width as f64);
                    shared.cams[slot].store(Arc::new(Some(CamSnapshot {
                        slot,
                        name: cam_cfg.name.clone(),
                        role: cam_cfg.role.clone(),
                        data: f.data.clone(),
                        width: f.width,
                        height: f.height,
                        seq: f.seq,
                        fps,
                        utc_midpoint_s: f.utc_midpoint_s,
                        fov_deg: fov,
                        deg_per_px: if cam_cfg.projection == Projection::Pinhole {
                            ((cam_cfg.pixel_eff_um() * 1e-3) / cam_cfg.focal_mm).to_degrees()
                        } else {
                            (cam_cfg.pixel_eff_um() * 1e-3 / cam_cfg.focal_mm).to_degrees()
                        },
                        fisheye: cam_cfg.fixed_zenith(),
                        source: source_name.clone(),
                        connected: true,
                        armed: recorder.is_armed(),
                        armed_frames,
                        armed_dropped: recorder.dropped(),
                        last_dump: last_dump.clone(),
                        deep_stars: deep_n,
                        hw_index,
                    })));
                }
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
    *core.frame.lock().unwrap() = Some(Arc::new(LoopFrame { data: Arc::new(data), h: f.height, w: f.width, seq: f.seq, time }));
}

fn clone_geom(g: &skytracker_astro::sgp4_pass::ObserverGeometry) -> skytracker_astro::sgp4_pass::ObserverGeometry {
    skytracker_astro::sgp4_pass::ObserverGeometry { pos_km: g.pos_km, east: g.east, north: g.north, up: g.up }
}

/// Folder-name-safe target label (capture_manager style).
fn sanitize_name(s: &str) -> String {
    let out: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    out.trim_matches('_').to_string()
}

fn iso_utc(t: f64) -> String {
    let (y, mo, d, hh, mm, ss) = crate::sky::civil_from_unix(t);
    let frac = ((t - t.floor()) * 1e6).round().clamp(0.0, 999_999.0) as i64;
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{frac:06}Z")
}

/// trajectory.csv in the capture_manager format the replay screen reads:
/// the full pass arc first (camera_index -1), then a row interpolated at each
/// frame's exposure-midpoint time (camera_index = this camera's slot).
fn trajectory_csv(arc: &[crate::state::ArcPoint], arc_t0: f64, target: &str, frame_times: &[f64], slot: usize) -> String {
    let mut out = String::from("timestamp,satellite_name,altitude_deg,azimuth_deg,distance_km,pixel_x,pixel_y,sequence_in_capture,camera_index\n");
    for p in arc {
        out.push_str(&format!("{},{},{:.5},{:.5},0,0,0,0,-1\n", iso_utc(arc_t0 + p.t_rel_s), target, p.el, p.az));
    }
    let interp = |t: f64| -> Option<(f64, f64)> {
        let rel = t - arc_t0;
        let i = arc.partition_point(|p| p.t_rel_s <= rel);
        if i == 0 || i >= arc.len() {
            return None;
        }
        let (a, b) = (&arc[i - 1], &arc[i]);
        let k = if b.t_rel_s > a.t_rel_s { (rel - a.t_rel_s) / (b.t_rel_s - a.t_rel_s) } else { 0.0 };
        let daz = (b.az - a.az + 540.0).rem_euclid(360.0) - 180.0;
        Some(((a.az + daz * k).rem_euclid(360.0), a.el + (b.el - a.el) * k))
    };
    for (i, t) in frame_times.iter().enumerate() {
        if let Some((az, el)) = interp(*t) {
            out.push_str(&format!("{},{},{:.5},{:.5},0,0,0,{},{}\n", iso_utc(*t), target, el, az, i + 1, slot));
        }
    }
    out
}
