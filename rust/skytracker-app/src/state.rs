#![allow(dead_code)]
//! State architecture for the native app (the plan's Phase 7 design):
//! single-owner worker threads publish immutable snapshots through
//! `ArcSwap`, the UI renders snapshots and sends commands over channels —
//! it never mutates shared state. This replaces the GIL-reliant global
//! rebinds of the Python app by construction.

use arc_swap::ArcSwap;
use serde_json::{json, Value};
use skytracker_core::core_loop::Shared as LoopShared;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Per-camera configuration (config.json `camera_configs.cameraN`).
#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub pixel_um: f64,
    pub focal_mm: f64,
    pub alignment_rotation_deg: f64,
    pub gain: i64,
    pub exposure_ms: i64,
    pub tetra3_db: Option<String>,
}

impl CameraConfig {
    pub fn fov_deg(&self, width_px: f64) -> f64 {
        let sensor_w_mm = width_px * self.pixel_um / 1000.0;
        2.0 * (sensor_w_mm / (2.0 * self.focal_mm)).atan().to_degrees()
    }
}

/// Hardware-simulator settings (config.json `sim_config`), live-editable
/// from the Sim screen; the camera + mount workers read the ArcSwap copy.
#[derive(Clone, Debug)]
pub struct SimSettings {
    pub misalign_az_deg: f64,
    pub misalign_el_deg: f64,
    pub encoder_noise_deg: f64,
    pub rate_noise_dps: f64,
    pub pe_amplitude_deg: f64,
    pub pe_period_s: f64,
    pub background_level: f64,
    pub read_noise: f64,
    pub target_brightness: f64,
    pub star_limit_mag: f64,
    pub use_deep_catalog: bool,
    pub seed: u64,
}

impl Default for SimSettings {
    fn default() -> Self {
        SimSettings {
            misalign_az_deg: 0.0,
            misalign_el_deg: 0.0,
            encoder_noise_deg: 0.0007,
            rate_noise_dps: 0.003,
            pe_amplitude_deg: 0.003,
            pe_period_s: 600.0,
            background_level: 6.0,
            read_noise: 2.0,
            target_brightness: 200.0,
            star_limit_mag: 10.0,
            use_deep_catalog: true,
            seed: 1234,
        }
    }
}

/// The config.json surface the native app reads. `raw` keeps every other
/// key so saving round-trips unknown fields untouched.
#[derive(Clone, Debug)]
pub struct Config {
    pub path: PathBuf,
    pub raw: Value,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    pub elevation_mask_deg: f64,
    pub mount_mode: String,
    pub alignment_az: f64,
    pub alignment_el: f64,
    pub altaz_side_flip: bool,
    pub azm_gains: (f64, f64, f64),
    pub alt_gains: (f64, f64, f64),
    pub azm_limit: (f64, f64),
    pub alt_limit: (f64, f64),
    pub offsets: (f64, f64),
    pub ff_azm: bool,
    pub ff_alt: bool,
    pub lead_time_s: f64,
    pub continuous_rate: bool,
    pub guide_rate_max_dps: f64,
    pub output_filter_tau: f64,
    pub loop_hz: f64,
    pub handoff_min_frames: u32,
    pub hotspot_snr: f64,
    pub hotspot_gate_radius: f64,
    pub hotspot_coast_s: f64,
    pub hotspot_max_rate_dps: f64,
    pub hotspot_x_sign: f64,
    pub hotspot_y_sign: f64,
    pub hotspot_star_filter: bool,
    pub hotspot_rate_gate_dps: f64,
    pub hotspot_camera_index: usize,
    pub plate_solve_camera_index: usize,
    pub cam: [CameraConfig; 2],
    pub star_limit_mag: f64,
    pub max_stars: usize,
    pub ui_vsync: bool,
    pub mount_transport: String, // "sim" | "serial"
    pub serial_port: String,
    pub serial_baud: u32,
    pub camera_source: String, // "sim" | "asi"
    pub asi_dll: String,
    pub captures_dir: String,
    pub tetra3_db_dir: String,
    pub alignment_points: usize,
    pub alignment_settle_s: f64,
    pub sim: SimSettings,
}

fn num(v: &Value, d: f64) -> f64 {
    match v {
        Value::String(s) => s.parse().unwrap_or(d),
        Value::Number(n) => n.as_f64().unwrap_or(d),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => d,
    }
}

fn boolean(v: &Value, d: bool) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(s) => matches!(s.to_ascii_lowercase().as_str(), "true" | "1" | "yes"),
        _ => d,
    }
}

impl Config {
    pub fn load(path: &Path) -> Config {
        let raw: Value = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        let v = &raw;
        let f = |k: &str, d: f64| num(&v[k], d);
        let b = |k: &str, d: bool| boolean(&v[k], d);
        let s = |k: &str, d: &str| v[k].as_str().unwrap_or(d).to_string();
        let cam = |name: &str, pix: f64, foc: f64| {
            let c = &v["camera_configs"][name];
            CameraConfig {
                pixel_um: num(&c["pixel_size"], pix),
                focal_mm: num(&c["focal_length"], foc),
                alignment_rotation_deg: num(&c["alignment_rotation"], 180.0),
                gain: num(&c["gain"], 200.0) as i64,
                exposure_ms: num(&c["exposure"], 50.0) as i64,
                tetra3_db: c["tetra3_db"].as_str().map(|s| s.to_string()),
            }
        };
        let sc = &v["sim_config"];
        let sd = SimSettings::default();
        let sim = SimSettings {
            misalign_az_deg: num(&sc["mount_misalignment_az_deg"], sd.misalign_az_deg),
            misalign_el_deg: num(&sc["mount_misalignment_el_deg"], sd.misalign_el_deg),
            encoder_noise_deg: num(&sc["mount_encoder_noise_deg"], sd.encoder_noise_deg),
            rate_noise_dps: num(&sc["mount_rate_noise_dps"], sd.rate_noise_dps),
            pe_amplitude_deg: num(&sc["mount_pe_amplitude_deg"], sd.pe_amplitude_deg),
            pe_period_s: num(&sc["mount_pe_period_sec"], sd.pe_period_s),
            background_level: num(&sc["background_level"], sd.background_level),
            read_noise: num(&sc["read_noise"], sd.read_noise),
            target_brightness: num(&sc["target_brightness"], sd.target_brightness),
            star_limit_mag: num(&sc["sim_star_limit_mag"], sd.star_limit_mag),
            use_deep_catalog: boolean(&sc["sim_use_deep_catalog"], sd.use_deep_catalog),
            seed: num(&sc["seed"], 1234.0) as u64,
        };
        Config {
            path: path.to_path_buf(),
            lat_deg: f("lat", 34.8740289),
            lon_deg: f("lon", -120.4461237),
            alt_m: f("alt", 100.0),
            elevation_mask_deg: f("elevation_mask", 10.0),
            mount_mode: std::env::var("SKYTRACKER_MOUNT_MODE").unwrap_or_else(|_| s("mount_mode", "AltAz")),
            alignment_az: f("alignment_azimuth", 0.0),
            alignment_el: f("alignment_elevation", 0.0),
            altaz_side_flip: b("altaz_side_flip", false),
            azm_gains: (f("pid_azm_p_gain", 0.0023), f("pid_azm_i_gain", 0.00025), f("pid_azm_d_gain", 0.00027)),
            alt_gains: (f("pid_alt_p_gain", 0.0027), f("pid_alt_i_gain", 0.00028), f("pid_alt_d_gain", 0.00027)),
            azm_limit: (f("azm_limit_min", -360.0), f("azm_limit_max", 360.0)),
            alt_limit: (f("alt_limit_min", -5.0), f("alt_limit_max", 95.0)),
            offsets: (f("azm_offset", 0.0), f("alt_offset", 0.0)),
            ff_azm: b("feed_forward_azm_enabled", true),
            ff_alt: b("feed_forward_alt_enabled", true),
            lead_time_s: f("pid_lead_time_sec", 0.0),
            continuous_rate: b("continuous_rate_tracking", false),
            guide_rate_max_dps: f("guide_rate_max_dps", 4.5),
            output_filter_tau: f("pid_output_filter_tau_sec", 0.0),
            loop_hz: f("rust_core_loop_hz", 15.0),
            handoff_min_frames: f("handoff_min_frames", 5.0) as u32,
            hotspot_snr: f("hotspot_snr_threshold", 5.0),
            hotspot_gate_radius: f("hotspot_gate_radius", 120.0),
            hotspot_coast_s: f("hotspot_coast_time_sec", 1.0),
            hotspot_max_rate_dps: f("hotspot_max_rate_dps", 2.0),
            hotspot_x_sign: f("hotspot_x_sign", 1.0),
            hotspot_y_sign: f("hotspot_y_sign", -1.0),
            hotspot_star_filter: b("hotspot_star_filter_enabled", true),
            hotspot_rate_gate_dps: f("hotspot_rate_gate_dps", 0.15),
            hotspot_camera_index: f("hotspot_camera_index", 0.0) as usize,
            plate_solve_camera_index: f("plate_solve_camera_index", 0.0) as usize,
            cam: [cam("camera1", 2.9, 162.0), cam("camera2", 2.4, 2000.0)],
            star_limit_mag: f("star_limiting_magnitude", 6.5),
            max_stars: f("max_rendered_star_count", 2000.0) as usize,
            ui_vsync: b("ui_vsync", true),
            mount_transport: s("mount_transport", "sim"),
            serial_port: s("mount_serial_port", "COM3"),
            serial_baud: f("mount_serial_baud", 9600.0) as u32,
            camera_source: s("camera_source", "sim"),
            asi_dll: s("asi_dll", "ASICamera2.dll"),
            captures_dir: s("captures_dir", "data"),
            tetra3_db_dir: s("tetra3_db_dir", ""),
            alignment_points: f("alignment_points", 12.0) as usize,
            alignment_settle_s: f("alignment_settle_sec", 2.0),
            sim,
            raw,
        }
    }

    /// Write the editable surface back into `raw` and save, keeping every
    /// key the native app does not (yet) understand.
    pub fn save(&self) -> std::io::Result<()> {
        let mut raw = if self.raw.is_object() { self.raw.clone() } else { json!({}) };
        let o = raw.as_object_mut().unwrap();
        let mut set = |k: &str, v: Value| {
            o.insert(k.to_string(), v);
        };
        set("lat", json!(self.lat_deg));
        set("lon", json!(self.lon_deg));
        set("alt", json!(self.alt_m));
        set("elevation_mask", json!(self.elevation_mask_deg));
        set("mount_mode", json!(self.mount_mode));
        set("alignment_azimuth", json!(self.alignment_az));
        set("alignment_elevation", json!(self.alignment_el));
        set("altaz_side_flip", json!(self.altaz_side_flip));
        set("pid_azm_p_gain", json!(self.azm_gains.0));
        set("pid_azm_i_gain", json!(self.azm_gains.1));
        set("pid_azm_d_gain", json!(self.azm_gains.2));
        set("pid_alt_p_gain", json!(self.alt_gains.0));
        set("pid_alt_i_gain", json!(self.alt_gains.1));
        set("pid_alt_d_gain", json!(self.alt_gains.2));
        set("azm_offset", json!(self.offsets.0));
        set("alt_offset", json!(self.offsets.1));
        set("azm_limit_min", json!(self.azm_limit.0));
        set("azm_limit_max", json!(self.azm_limit.1));
        set("alt_limit_min", json!(self.alt_limit.0));
        set("alt_limit_max", json!(self.alt_limit.1));
        set("feed_forward_azm_enabled", json!(self.ff_azm));
        set("feed_forward_alt_enabled", json!(self.ff_alt));
        set("pid_lead_time_sec", json!(self.lead_time_s));
        set("continuous_rate_tracking", json!(self.continuous_rate));
        set("guide_rate_max_dps", json!(self.guide_rate_max_dps));
        set("pid_output_filter_tau_sec", json!(self.output_filter_tau));
        set("rust_core_loop_hz", json!(self.loop_hz));
        set("handoff_min_frames", json!(self.handoff_min_frames));
        set("hotspot_snr_threshold", json!(self.hotspot_snr));
        set("hotspot_gate_radius", json!(self.hotspot_gate_radius));
        set("hotspot_coast_time_sec", json!(self.hotspot_coast_s));
        set("hotspot_max_rate_dps", json!(self.hotspot_max_rate_dps));
        set("hotspot_x_sign", json!(self.hotspot_x_sign));
        set("hotspot_y_sign", json!(self.hotspot_y_sign));
        set("hotspot_star_filter_enabled", json!(self.hotspot_star_filter));
        set("hotspot_rate_gate_dps", json!(self.hotspot_rate_gate_dps));
        set("hotspot_camera_index", json!(self.hotspot_camera_index));
        set("plate_solve_camera_index", json!(self.plate_solve_camera_index));
        set("star_limiting_magnitude", json!(self.star_limit_mag));
        set("max_rendered_star_count", json!(self.max_stars));
        set("ui_vsync", json!(self.ui_vsync));
        set("mount_transport", json!(self.mount_transport));
        set("mount_serial_port", json!(self.serial_port));
        set("mount_serial_baud", json!(self.serial_baud));
        set("camera_source", json!(self.camera_source));
        set("asi_dll", json!(self.asi_dll));
        set("captures_dir", json!(self.captures_dir));
        set("tetra3_db_dir", json!(self.tetra3_db_dir));
        set("alignment_points", json!(self.alignment_points));
        set("alignment_settle_sec", json!(self.alignment_settle_s));
        for (i, name) in ["camera1", "camera2"].iter().enumerate() {
            let c = &self.cam[i];
            let entry = o
                .entry("camera_configs")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .unwrap()
                .entry(*name)
                .or_insert_with(|| json!({}));
            let e = entry.as_object_mut().unwrap();
            e.insert("pixel_size".into(), json!(c.pixel_um));
            e.insert("focal_length".into(), json!(c.focal_mm));
            e.insert("alignment_rotation".into(), json!(c.alignment_rotation_deg));
            e.insert("gain".into(), json!(c.gain));
            e.insert("exposure".into(), json!(c.exposure_ms));
            e.insert("tetra3_db".into(), json!(c.tetra3_db));
        }
        let sc = o.entry("sim_config").or_insert_with(|| json!({})).as_object_mut().unwrap();
        let s = &self.sim;
        sc.insert("mount_misalignment_az_deg".into(), json!(s.misalign_az_deg));
        sc.insert("mount_misalignment_el_deg".into(), json!(s.misalign_el_deg));
        sc.insert("mount_encoder_noise_deg".into(), json!(s.encoder_noise_deg));
        sc.insert("mount_rate_noise_dps".into(), json!(s.rate_noise_dps));
        sc.insert("mount_pe_amplitude_deg".into(), json!(s.pe_amplitude_deg));
        sc.insert("mount_pe_period_sec".into(), json!(s.pe_period_s));
        sc.insert("background_level".into(), json!(s.background_level));
        sc.insert("read_noise".into(), json!(s.read_noise));
        sc.insert("target_brightness".into(), json!(s.target_brightness));
        sc.insert("sim_star_limit_mag".into(), json!(s.star_limit_mag));
        sc.insert("sim_use_deep_catalog".into(), json!(s.use_deep_catalog));
        sc.insert("seed".into(), json!(s.seed));
        let text = serde_json::to_string_pretty(&raw)?;
        std::fs::write(&self.path, text)
    }

    pub fn cam1_fov_deg(&self, width_px: f64) -> f64 {
        self.cam[0].fov_deg(width_px)
    }

    pub fn repo_root(&self) -> PathBuf {
        self.path.parent().map(|p| p.to_path_buf()).unwrap_or_default()
    }

    /// Directories searched for tetra3 `.npz` databases.
    pub fn tetra3_search_dirs(&self) -> Vec<PathBuf> {
        let mut v = Vec::new();
        if !self.tetra3_db_dir.is_empty() {
            v.push(PathBuf::from(&self.tetra3_db_dir));
        }
        if let Ok(d) = std::env::var("SKYTRACKER_TETRA3_DIR") {
            v.push(PathBuf::from(d));
        }
        let root = self.repo_root();
        v.push(root.join("catalogs"));
        v.push(root.join("data"));
        if let Ok(home) = std::env::var("USERPROFILE") {
            v.push(PathBuf::from(home).join("anaconda3/envs/track/Lib/site-packages/tetra3/data"));
        }
        v
    }
}

#[derive(Clone, Debug)]
pub struct SatMark {
    pub satnum: String,
    pub name: String,
    pub az: f64,
    pub el: f64,
    pub range_km: f64,
    pub az_rate: f64,
    pub el_rate: f64,
}

#[derive(Clone, Debug)]
pub struct StarMark {
    pub hip: i64,
    pub mag: f64,
    pub az: f64,
    pub el: f64,
}

#[derive(Clone, Debug)]
pub struct BodyMark {
    pub name: String,
    pub az: f64,
    pub el: f64,
    pub dist_km: f64,
}

/// Published by the sky worker (~2 Hz sats/bodies, ~0.2 Hz stars).
#[derive(Clone, Debug, Default)]
pub struct SkySnapshot {
    pub jd_tt: f64,
    pub utc_iso: String,
    pub sats: Vec<SatMark>,
    pub stars: Vec<StarMark>,
    pub bodies: Vec<BodyMark>,
    pub n_catalog: usize,
    pub n_visible: usize,
    pub compute_ms: f64,
    pub status: String,
}

/// One point of the selected satellite's track on the skyplot.
#[derive(Clone, Copy, Debug)]
pub struct ArcPoint {
    pub t_rel_s: f64,
    pub az: f64,
    pub el: f64,
}

/// Upcoming-pass row (published by the sky worker every minute).
#[derive(Clone, Debug)]
pub struct PassRow {
    pub satnum: String,
    pub name: String,
    pub aos_unix: f64,
    pub tca_unix: f64,
    pub los_unix: f64,
    pub aos_az: f64,
    pub tca_az: f64,
    pub tca_el: f64,
    pub los_az: f64,
    pub duration_s: f64,
    pub max_rate_dps: f64,
    pub range_tca_km: f64,
    pub apogee_km: f64,
    pub est_mag: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct PassesSnapshot {
    pub computed_unix: f64,
    pub horizon_h: f64,
    pub rows: Vec<PassRow>,
    pub compute_ms: f64,
    /// Track of the selected satellite, -10..+10 min around now.
    pub arc: Vec<ArcPoint>,
    pub arc_satnum: Option<String>,
}

/// Published by the mount worker every control cycle.
#[derive(Clone, Debug, Default)]
pub struct MountSnapshot {
    /// Sky az/el (mount frame transformed through the configured mount mode).
    pub az: f64,
    pub el: f64,
    /// Mount-frame axis angles (encoder minus offsets).
    pub azm: f64,
    pub alt: f64,
    pub mode: String,
    pub rate_cmd: (i32, i32),
    pub az_error: f64,
    pub el_error: f64,
    pub actual_hz: f64,
    pub gear_ceiling: i32,
    pub joystick: Option<String>,
    pub stick: (f64, f64),
    pub status: Vec<String>,
    pub transport: String,
    pub connected: bool,
    pub target: Option<String>,
    pub setpoint: Option<(f64, f64)>,
    pub hotspot_acquired: bool,
    pub hotspot_status: String,
    pub hotspot_snr: f64,
    pub hotspot_centroid: Option<(f64, f64)>,
    pub handoff_count: u32,
    pub gains: [[f64; 3]; 2],
    pub autotune: Option<String>,
    pub loop_dead: bool,
}

/// Published by the camera worker for every pumped frame.
#[derive(Clone)]
pub struct CamSnapshot {
    pub data: Arc<Vec<u8>>,
    pub width: usize,
    pub height: usize,
    pub seq: u64,
    pub fps: f64,
    pub utc_midpoint_s: f64,
    pub fov_deg: f64,
    pub source: String,
    pub armed: bool,
    pub armed_frames: usize,
    pub last_dump: Option<String>,
    pub deep_stars: usize,
}

/// Plate-solve result on the live frame (alignment worker).
#[derive(Clone, Debug, Default)]
pub struct SolveSnapshot {
    pub busy: bool,
    pub db_name: String,
    pub db_loaded: bool,
    pub last_ok: bool,
    pub message: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    pub fov_deg: f64,
    pub rmse_arcsec: f64,
    pub matches: usize,
    pub n_centroids: usize,
    pub solve_ms: f64,
    pub true_az: f64,
    pub true_el: f64,
    pub mount_az: f64,
    pub mount_el: f64,
    pub centroids: Vec<[f64; 2]>,
    pub matched: Vec<[f64; 2]>,
    pub frame_seq: u64,
}

#[derive(Clone, Debug)]
pub struct AlignSample {
    pub mount_az: f64,
    pub mount_el: f64,
    pub true_az: f64,
    pub true_el: f64,
    pub residual_arcsec: f64,
}

#[derive(Clone, Debug, Default)]
pub struct AlignSnapshot {
    pub running: bool,
    pub status: String,
    pub action: String,
    pub point: usize,
    pub n_points: usize,
    pub targets: Vec<(f64, f64)>,
    pub samples: Vec<AlignSample>,
    pub terms: Option<[f64; 7]>,
    pub rms_arcsec: Option<f64>,
    pub log: Vec<String>,
}

/// UI -> mount worker.
#[derive(Clone, Debug)]
pub enum MountCmd {
    SetMode(String),
    SelectTarget(Option<String>),
    Stop,
    Goto { az: f64, el: f64 },
    SetGains { azm: (f64, f64, f64), alt: (f64, f64, f64) },
    SetHotspotSigns { x: f64, y: f64 },
    AutotuneStart,
    AutotuneStop { revert: bool },
    /// Nudge the commanded position in RATE-less modes (alignment paddles).
    Nudge { daz: f64, del: f64 },
}

/// UI -> camera worker.
#[derive(Clone, Debug)]
pub enum CamCmd {
    Arm,
    Dump { name: String },
    Disarm,
}

/// UI -> alignment/plate-solve worker.
#[derive(Clone, Debug)]
pub enum AlignCmd {
    SolveNow,
    Start { n_points: usize, supervised: bool },
    Accept,
    Reject,
    Abort,
    ApplyModel,
}

pub struct Shared {
    pub sky: ArcSwap<SkySnapshot>,
    pub passes: ArcSwap<PassesSnapshot>,
    pub mount: ArcSwap<MountSnapshot>,
    pub cam: ArcSwap<Option<CamSnapshot>>,
    pub solve: ArcSwap<SolveSnapshot>,
    pub align: ArcSwap<AlignSnapshot>,
    pub sim: ArcSwap<SimSettings>,
    pub config: Config,
    /// The core control loop's shared block (inputs / outputs / frame slot).
    pub core: Arc<LoopShared>,
    /// Selected satellite (mirrored from the mount worker for the sky worker).
    pub selected: ArcSwap<Option<String>>,
}

impl Shared {
    pub fn new(config: Config, core: Arc<LoopShared>) -> Arc<Self> {
        let sim = config.sim.clone();
        Arc::new(Shared {
            sky: ArcSwap::from_pointee(SkySnapshot::default()),
            passes: ArcSwap::from_pointee(PassesSnapshot::default()),
            mount: ArcSwap::from_pointee(MountSnapshot::default()),
            cam: ArcSwap::from_pointee(None),
            solve: ArcSwap::from_pointee(SolveSnapshot::default()),
            align: ArcSwap::from_pointee(AlignSnapshot::default()),
            sim: ArcSwap::from_pointee(sim),
            config,
            core,
            selected: ArcSwap::from_pointee(None),
        })
    }
}
