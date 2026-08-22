#![allow(dead_code)]
//! State architecture for the native app (the plan's Phase 7 design):
//! single-owner worker threads publish immutable snapshots through
//! `ArcSwap`, the UI renders snapshots and sends commands over channels —
//! it never mutates shared state. This replaces the GIL-reliant global
//! rebinds of the Python app by construction.

use arc_swap::ArcSwap;
use std::sync::Arc;

/// The subset of config.json the native app reads today.
#[derive(Clone, Debug)]
pub struct Config {
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
    pub cam1_pixel_um: f64,
    pub cam1_focal_mm: f64,
    pub star_limit_mag: f64,
    pub max_stars: usize,
    pub ui_vsync: bool,
}

impl Config {
    pub fn load(path: &std::path::Path) -> Config {
        let v: serde_json::Value = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let f = |k: &str, d: f64| -> f64 {
            match &v[k] {
                serde_json::Value::String(s) => s.parse().unwrap_or(d),
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(d),
                _ => d,
            }
        };
        let cam = &v["camera_configs"]["camera1"];
        let camf = |k: &str, d: f64| cam[k].as_f64().unwrap_or(d);
        Config {
            lat_deg: f("lat", 34.8740289),
            lon_deg: f("lon", -120.4461237),
            alt_m: f("alt", 100.0),
            elevation_mask_deg: f("elevation_mask", 10.0),
            mount_mode: v["mount_mode"].as_str().unwrap_or("AltAz").to_string(),
            alignment_az: f("alignment_azimuth", 0.0),
            alignment_el: f("alignment_elevation", 0.0),
            altaz_side_flip: v["altaz_side_flip"].as_bool().unwrap_or(false),
            azm_gains: (f("pid_azm_p_gain", 0.0023), f("pid_azm_i_gain", 0.00025), f("pid_azm_d_gain", 0.00027)),
            alt_gains: (f("pid_alt_p_gain", 0.0027), f("pid_alt_i_gain", 0.00028), f("pid_alt_d_gain", 0.00027)),
            cam1_pixel_um: camf("pixel_size", 2.9),
            cam1_focal_mm: camf("focal_length", 162.0),
            star_limit_mag: f("star_limiting_magnitude", 6.5),
            max_stars: f("max_rendered_star_count", 2000.0) as usize,
            ui_vsync: v["ui_vsync"].as_bool().unwrap_or(true),
        }
    }

    /// Horizontal FOV (deg) of camera 1 for a sensor `width_px` wide.
    pub fn cam1_fov_deg(&self, width_px: f64) -> f64 {
        let sensor_w_mm = width_px * self.cam1_pixel_um / 1000.0;
        2.0 * (sensor_w_mm / (2.0 * self.cam1_focal_mm)).atan().to_degrees()
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

/// Published by the sky worker (~1 Hz sats/bodies, ~0.2 Hz stars).
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

/// Published by the mount worker every control cycle.
#[derive(Clone, Debug, Default)]
pub struct MountSnapshot {
    pub az: f64,
    pub el: f64,
    pub mode: String,
    pub rate_cmd: (i32, i32),
    pub az_error: f64,
    pub el_error: f64,
    pub actual_hz: f64,
    pub gear_ceiling: i32,
    pub joystick: Option<String>,
    pub stick: (f64, f64),
    pub status: Vec<String>,
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
}

/// UI -> mount worker.
#[derive(Clone, Debug)]
pub enum MountCmd {
    SetMode(String),
    SelectTarget(Option<String>),
    Stop,
    Goto { az: f64, el: f64 },
}

pub struct Shared {
    pub sky: ArcSwap<SkySnapshot>,
    pub mount: ArcSwap<MountSnapshot>,
    pub cam: ArcSwap<Option<CamSnapshot>>,
    pub config: Config,
}

impl Shared {
    pub fn new(config: Config) -> Arc<Self> {
        Arc::new(Shared {
            sky: ArcSwap::from_pointee(SkySnapshot::default()),
            mount: ArcSwap::from_pointee(MountSnapshot::default()),
            cam: ArcSwap::from_pointee(None),
            config,
        })
    }
}
