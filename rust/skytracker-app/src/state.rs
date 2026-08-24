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

/// Optical projection of a camera: the mount-borne scopes are pinholes;
/// the hemispheric "bubble" camera is an equidistant fisheye fixed at the
/// zenith (r = f * theta).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Pinhole,
    FisheyeEquidistant,
}

/// Per-camera configuration (config.json `camera_configs.cameraN`). Slots
/// are logical (1 = guide scope, 2 = main scope, 3 = bubble by default);
/// `asi_index` is the hardware enumeration index behind the slot and is
/// what "swap cameras" exchanges.
#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub name: String,
    pub role: String, // "guide" | "main" | "bubble"
    pub asi_index: usize,
    pub enabled: bool,
    pub pixel_um: f64,
    pub focal_mm: f64,
    pub sensor_w: usize,
    pub sensor_h: usize,
    pub bin: usize,
    pub projection: Projection,
    pub alignment_rotation_deg: f64,
    pub gain: i64,
    /// Integration time in MICROSECONDS (config `exposure_us`, falling back
    /// to the Python-era `exposure` in ms). ASI cameras go down to ~32 µs.
    pub exposure_us: i64,
    pub gamma: f64,
    pub gamma_enabled: bool,
    pub tetra3_db: Option<String>,
}

impl CameraConfig {
    /// Horizontal FOV (deg) for a frame `width_px` wide at the binned plate scale.
    pub fn fov_deg(&self, width_px: f64) -> f64 {
        match self.projection {
            Projection::Pinhole => {
                let sensor_w_mm = width_px * self.pixel_um * self.bin as f64 / 1000.0;
                2.0 * (sensor_w_mm / (2.0 * self.focal_mm)).atan().to_degrees()
            }
            Projection::FisheyeEquidistant => (width_px * self.pixel_um * self.bin as f64 / 1000.0 / self.focal_mm).to_degrees().min(200.0),
        }
    }
    /// Effective pixel size at the current binning (um).
    pub fn pixel_eff_um(&self) -> f64 {
        self.pixel_um * self.bin as f64
    }
    /// Frame size the worker produces (binned sensor).
    pub fn frame_size(&self) -> (usize, usize) {
        ((self.sensor_w / self.bin.max(1)).max(16), (self.sensor_h / self.bin.max(1)).max(16))
    }
    pub fn fixed_zenith(&self) -> bool {
        self.projection == Projection::FisheyeEquidistant
    }
}

/// Live, UI-editable camera settings (gain / exposure / gamma / rotation /
/// ROI) — one ArcSwap per slot; the worker applies changes to the ASI
/// controls and the UI applies gamma + rotation on display.
#[derive(Clone, Debug, PartialEq)]
pub struct CamSettings {
    pub gain: i64,
    pub exposure_us: i64,
    pub gamma: f64,
    pub gamma_enabled: bool,
    pub rotation_deg: f64,
    /// ROI: fraction of the sensor (1.0 = full) and centre (0..1).
    pub roi_frac: f64,
    pub roi_cx: f64,
    pub roi_cy: f64,
    pub connected: bool,
}

impl CamSettings {
    pub fn from_config(c: &CameraConfig) -> Self {
        CamSettings {
            gain: c.gain,
            exposure_us: c.exposure_us,
            gamma: c.gamma,
            gamma_enabled: c.gamma_enabled,
            rotation_deg: c.alignment_rotation_deg,
            roi_frac: 1.0,
            roi_cx: 0.5,
            roi_cy: 0.5,
            connected: c.enabled,
        }
    }
}

/// A filter wheel: ASI EFW (electronic, controlled through EFW_filter.dll)
/// or manual (position tracked by the operator). Slot names are the filter
/// assignments; persisted in config.json `filter_wheels`.
#[derive(Clone, Debug)]
pub struct FilterWheelConfig {
    pub name: String,
    pub kind: String, // "efw" | "manual"
    pub camera: usize, // logical camera slot it sits in front of
    pub efw_index: usize,
    pub slots: Vec<String>,
    pub position: usize,
}

/// Hardware-simulator settings (config.json `sim_config`), live-editable
/// from the Sim screen; the camera + mount workers read the ArcSwap copy.
#[derive(Clone, Debug)]
pub struct SimSettings {
    pub misalign_az_deg: f64,
    pub misalign_el_deg: f64,
    pub encoder_noise_deg: f64,
    pub rate_noise_dps: f64,
    pub backlash_deg: f64,
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
            backlash_deg: 0.006,
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
    pub cam: Vec<CameraConfig>,
    pub filter_wheels: Vec<FilterWheelConfig>,
    pub efw_dll: String,
    pub star_limit_mag: f64,
    pub max_stars: usize,
    pub ui_vsync: bool,
    pub mount_transport: String, // "sim" | "serial"
    pub serial_port: String,
    pub serial_baud: u32,
    pub camera_source: String, // "sim" | "asi"
    pub asi_dll: String,
    pub captures_dir: String,
    /// Armed-capture ring depth in frames (Python `buffer_size`).
    pub capture_buffer_frames: usize,
    pub tetra3_db_dir: String,
    pub alignment_points: usize,
    pub alignment_settle_s: f64,
    pub alignment_grid_search_cells: usize,
    pub alignment_target_rms_arcmin: f64,
    pub alignment_pause_on_fail: bool,
    pub meo_enabled: bool,
    pub geo_enabled: bool,
    pub satellite_labels_enabled: bool,
    pub ngc_limit_mag: f64,
    pub messier_enabled: bool,
    pub ngc_enabled: bool,
    pub adsb_source_mode: String,
    pub adsb_host: String,
    pub adsb_port: u16,
    pub adsb_fit_points: usize,
    pub adsb_predict_horizon_s: f64,
    pub adsb_predict_step_s: f64,
    pub adsb_stale_timeout_s: f64,
    pub adsb_device_index: usize,
    pub adsb_gain: String,
    pub adsb_dll: String,
    /// Per-mode PID gain profiles (config `pid_mode_profiles`): mode -> (azm (p,i,d), alt (p,i,d)).
    pub pid_profiles: std::collections::HashMap<String, ((f64, f64, f64), (f64, f64, f64))>,
    pub launches_dir: String,
    /// Refresh tle_cache.tle from Celestrak when older than this (hours; 0 = never).
    pub tle_max_age_h: f64,
    pub tle_url: String,
    /// 7-term alt-az pointing model (IA IE AN AW NPAE CA TF, degrees) and whether it pre-corrects setpoints.
    pub pointing_model_enabled: bool,
    pub pointing_model_terms: [f64; 7],
    pub eq_pointing_model_enabled: bool,
    /// IH ID NP CH ME MA TF (skytracker_pointing::eq::TERM_NAMES order).
    pub eq_pointing_model_terms: [f64; 7],
    /// Gearbox: base ceiling step, wind-up delay (joy_rate_base_ceiling / joy_rate_windup_delay_s).
    pub joy_rate_base_ceiling: i32,
    pub joy_rate_windup_delay_s: f64,
    /// Which stick drives RATE mode ("right" like the Python app, or "left").
    pub joystick_rate_stick: String,
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
        // Cameras: defaults describe the 2026-08 rig -- slot 1 guide scope
        // (the tetra3 camera), slot 2 ASI432MM on the main scope, slot 3
        // ASI462MM hemispheric bubble cam (fisheye, fixed at the zenith).
        let cam_defaults: [(&str, &str, f64, f64, usize, usize, usize, &str, usize); 3] = [
            ("Guide 50 mm", "guide", 2.9, 162.0, 3096, 2080, 1, "pinhole", 0),
            ("Main ASI432MM", "main", 9.0, 2000.0, 1608, 1104, 1, "pinhole", 1),
            ("Bubble ASI462MM", "bubble", 2.9, 1.55, 1936, 1096, 2, "fisheye", 2),
        ];
        let cam = |i: usize| {
            let (dname, drole, pix, foc, sw, sh, bin, proj, asi) = cam_defaults[i];
            let name = format!("camera{}", i + 1);
            let c = &v["camera_configs"][&name];
            let proj_s = c["projection"].as_str().unwrap_or(proj);
            CameraConfig {
                name: c["name"].as_str().unwrap_or(dname).to_string(),
                role: c["role"].as_str().unwrap_or(drole).to_string(),
                asi_index: num(&c["asi_index"], asi as f64) as usize,
                enabled: boolean(&c["enabled"], true),
                pixel_um: num(&c["pixel_size"], pix),
                focal_mm: num(&c["focal_length"], foc),
                sensor_w: num(&c["sensor_width"], sw as f64) as usize,
                sensor_h: num(&c["sensor_height"], sh as f64) as usize,
                bin: (num(&c["bin"], bin as f64) as usize).clamp(1, 8),
                projection: if proj_s.starts_with("fish") { Projection::FisheyeEquidistant } else { Projection::Pinhole },
                alignment_rotation_deg: num(&c["alignment_rotation"], if i == 2 { 0.0 } else { 180.0 }),
                gain: num(&c["gain"], 200.0) as i64,
                exposure_us: {
                    // Python semantics: `exposure` is MICROSECONDS (it goes
                    // straight to ASI_EXPOSURE). `exposure_us` is the same
                    // value under the native app's explicit name.
                    let us = num(&c["exposure_us"], f64::NAN);
                    if us.is_finite() { us as i64 } else { num(&c["exposure"], 10_000.0) as i64 }
                },
                gamma: num(&c["gamma"], 0.1),
                gamma_enabled: boolean(&c["gamma_enabled"], false),
                tetra3_db: c["tetra3_db"].as_str().map(|s| s.to_string()).or(if i == 0 { Some("db_cam1_tyc".into()) } else { None }),
            }
        };
        let n_cams = (1..=8).rev().find(|i| !v["camera_configs"][format!("camera{i}")].is_null()).unwrap_or(0).max(3);
        let cams: Vec<CameraConfig> = (0..n_cams.min(3).max(3)).map(cam).collect();
        let mut filter_wheels: Vec<FilterWheelConfig> = v["filter_wheels"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|w| FilterWheelConfig {
                        name: w["name"].as_str().unwrap_or("wheel").to_string(),
                        kind: w["kind"].as_str().unwrap_or("manual").to_string(),
                        camera: num(&w["camera"], 1.0) as usize,
                        efw_index: num(&w["efw_index"], 0.0) as usize,
                        slots: w["slots"].as_array().map(|s| s.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default(),
                        position: num(&w["position"], 0.0) as usize,
                    })
                    .collect()
            })
            .unwrap_or_default();
        if filter_wheels.is_empty() {
            filter_wheels = vec![
                FilterWheelConfig {
                    name: "Main EFW".into(),
                    kind: "efw".into(),
                    camera: 1,
                    efw_index: 0,
                    slots: ["L", "R", "G", "B", "Ha", "OIII", "SII"].iter().map(|s| s.to_string()).collect(),
                    position: 0,
                },
                FilterWheelConfig {
                    name: "Guide manual wheel".into(),
                    kind: "manual".into(),
                    camera: 0,
                    efw_index: 0,
                    slots: ["Clear", "R", "G", "B", "ND"].iter().map(|s| s.to_string()).collect(),
                    position: 0,
                },
            ];
        }
        let sc = &v["sim_config"];
        let sd = SimSettings::default();
        let sim = SimSettings {
            misalign_az_deg: num(&sc["mount_misalignment_az_deg"], sd.misalign_az_deg),
            misalign_el_deg: num(&sc["mount_misalignment_el_deg"], sd.misalign_el_deg),
            encoder_noise_deg: num(&sc["mount_encoder_noise_deg"], sd.encoder_noise_deg),
            rate_noise_dps: num(&sc["mount_rate_noise_dps"], sd.rate_noise_dps),
            backlash_deg: num(&sc["mount_backlash_deg"], sd.backlash_deg),
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
            elevation_mask_deg: f("elevation_mask", 0.0),
            mount_mode: std::env::var("SKYTRACKER_MOUNT_MODE").unwrap_or_else(|_| s("mount_mode", "AltAz")),
            alignment_az: f("alignment_azimuth", 0.0),
            alignment_el: f("alignment_elevation", 0.0),
            altaz_side_flip: b("altaz_side_flip", false),
            azm_gains: (f("pid_azm_p_gain", 0.0023), f("pid_azm_i_gain", 0.00025), f("pid_azm_d_gain", 0.00027)),
            alt_gains: (f("pid_alt_p_gain", 0.0027), f("pid_alt_i_gain", 0.00028), f("pid_alt_d_gain", 0.00027)),
            azm_limit: (f("azm_limit_min", -180.0), f("azm_limit_max", 180.0)),
            alt_limit: (f("alt_limit_min", -100.0), f("alt_limit_max", 100.0)),
            offsets: (f("azm_offset", 0.0), f("alt_offset", 0.0)),
            ff_azm: b("feed_forward_azm_enabled", false),
            ff_alt: b("feed_forward_alt_enabled", false),
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
            cam: cams,
            filter_wheels,
            efw_dll: s("efw_dll", "EFW_filter.dll"),
            star_limit_mag: f("star_limiting_magnitude", 6.5),
            max_stars: f("max_rendered_star_count", 2000.0) as usize,
            ui_vsync: b("ui_vsync", true),
            mount_transport: s("mount_transport", "sim"),
            serial_port: s("mount_serial_port", "COM3"),
            serial_baud: f("mount_serial_baud", 9600.0) as u32,
            camera_source: s("camera_source", "sim"),
            asi_dll: s("asi_dll", "ASICamera2.dll"),
            captures_dir: s("captures_dir", "data"),
            capture_buffer_frames: f("buffer_size", 1000.0).max(1.0) as usize,
            tetra3_db_dir: s("tetra3_db_dir", ""),
            alignment_points: f("alignment_points", 12.0) as usize,
            alignment_settle_s: f("alignment_settle_sec", 2.0),
            alignment_grid_search_cells: f("alignment_grid_search_cells", 100.0) as usize,
            alignment_target_rms_arcmin: f("alignment_target_rms_arcmin", 0.0),
            alignment_pause_on_fail: b("alignment_pause_on_fail", false),
            meo_enabled: b("meo_enabled", true),
            geo_enabled: b("geo_enabled", true),
            satellite_labels_enabled: b("satellite_labels_enabled", true),
            ngc_limit_mag: f("ngc_limiting_magnitude", 10.0),
            messier_enabled: b("messier_enabled", true),
            ngc_enabled: b("ngc_enabled", false),
            adsb_source_mode: s("adsb_source_mode", "off"),
            adsb_host: s("adsb_dump1090_host", "127.0.0.1"),
            adsb_port: f("adsb_dump1090_port", 30003.0) as u16,
            adsb_fit_points: f("adsb_fit_points", 5.0) as usize,
            adsb_predict_horizon_s: f("adsb_predict_horizon_sec", 60.0),
            adsb_predict_step_s: f("adsb_predict_step_sec", 2.0),
            adsb_stale_timeout_s: f("adsb_stale_timeout_sec", 30.0),
            adsb_device_index: f("adsb_device_index", 0.0) as usize,
            adsb_gain: match &v["adsb_gain"] { Value::String(g) => g.clone(), Value::Number(n) => n.to_string(), _ => "auto".into() },
            adsb_dll: s("adsb_rtlsdr_dll", "rtlsdr.dll"),
            pid_profiles: v["pid_mode_profiles"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .filter_map(|(mode, p)| {
                            let g = &p["gains"];
                            if g.is_null() {
                                return None;
                            }
                            Some((
                                mode.to_ascii_uppercase(),
                                (
                                    (num(&g["pid_azm_p_gain"], 0.0023), num(&g["pid_azm_i_gain"], 0.00025), num(&g["pid_azm_d_gain"], 0.00027)),
                                    (num(&g["pid_alt_p_gain"], 0.0027), num(&g["pid_alt_i_gain"], 0.00028), num(&g["pid_alt_d_gain"], 0.00027)),
                                ),
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            launches_dir: s("launches_dir", "launches"),
            tle_max_age_h: f("tle_cache_age_hours", 12.0),
            tle_url: s("tle_url", skytracker_astro::tle::CELESTRAK_ACTIVE_URL),
            pointing_model_enabled: b("pointing_model_enabled", false),
            pointing_model_terms: {
                let t = &v["pointing_model_terms"];
                let mut arr = [0.0; 7];
                for (i, name) in skytracker_pointing::altaz::TERM_NAMES.iter().enumerate() {
                    arr[i] = num(&t[*name], 0.0);
                }
                arr
            },
            eq_pointing_model_enabled: b("eq_pointing_model_enabled", false),
            eq_pointing_model_terms: {
                let t = &v["eq_pointing_model_terms"];
                let mut arr = [0.0; 7];
                for (i, name) in skytracker_pointing::eq::TERM_NAMES.iter().enumerate() {
                    arr[i] = num(&t[*name], 0.0);
                }
                arr
            },
            joy_rate_base_ceiling: f("joy_rate_base_ceiling", 5.0) as i32,
            joy_rate_windup_delay_s: f("joy_rate_windup_delay_s", 0.8),
            joystick_rate_stick: s("joystick_rate_stick", "right"),
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
        set("lat", json!(format!("{}", self.lat_deg)));
        set("lon", json!(format!("{}", self.lon_deg)));
        set("alt", json!(format!("{}", self.alt_m)));
        set("elevation_mask", json!(format!("{}", self.elevation_mask_deg)));
        set("mount_mode", json!(self.mount_mode));
        set("alignment_azimuth", json!(format!("{}", self.alignment_az)));
        set("alignment_elevation", json!(format!("{}", self.alignment_el)));
        set("altaz_side_flip", json!(self.altaz_side_flip));
        set("pid_azm_p_gain", json!(self.azm_gains.0));
        set("pid_azm_i_gain", json!(self.azm_gains.1));
        set("pid_azm_d_gain", json!(self.azm_gains.2));
        set("pid_alt_p_gain", json!(self.alt_gains.0));
        set("pid_alt_i_gain", json!(self.alt_gains.1));
        set("pid_alt_d_gain", json!(self.alt_gains.2));
        set("azm_offset", json!(format!("{}", self.offsets.0)));
        set("alt_offset", json!(format!("{}", self.offsets.1)));
        set("azm_limit_min", json!(format!("{}", self.azm_limit.0)));
        set("azm_limit_max", json!(format!("{}", self.azm_limit.1)));
        set("alt_limit_min", json!(format!("{}", self.alt_limit.0)));
        set("alt_limit_max", json!(format!("{}", self.alt_limit.1)));
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
        set("pointing_model_enabled", json!(self.pointing_model_enabled));
        set(
            "pointing_model_terms",
            json!(skytracker_pointing::altaz::TERM_NAMES.iter().zip(self.pointing_model_terms.iter()).map(|(n, v)| (n.to_string(), json!(v))).collect::<serde_json::Map<String, Value>>()),
        );
        set("eq_pointing_model_enabled", json!(self.eq_pointing_model_enabled));
        set(
            "eq_pointing_model_terms",
            json!(skytracker_pointing::eq::TERM_NAMES.iter().zip(self.eq_pointing_model_terms.iter()).map(|(n, v)| (n.to_string(), json!(v))).collect::<serde_json::Map<String, Value>>()),
        );
        set("joy_rate_base_ceiling", json!(self.joy_rate_base_ceiling));
        set("joy_rate_windup_delay_s", json!(self.joy_rate_windup_delay_s));
        set("joystick_rate_stick", json!(self.joystick_rate_stick));
        set("mount_transport", json!(self.mount_transport));
        set("mount_serial_port", json!(self.serial_port));
        set("mount_serial_baud", json!(self.serial_baud));
        set("camera_source", json!(self.camera_source));
        set("asi_dll", json!(self.asi_dll));
        set("captures_dir", json!(self.captures_dir));
        set("tetra3_db_dir", json!(self.tetra3_db_dir));
        set("alignment_points", json!(self.alignment_points));
        set("alignment_settle_sec", json!(self.alignment_settle_s));
        set("alignment_grid_search_cells", json!(self.alignment_grid_search_cells));
        set("alignment_target_rms_arcmin", json!(self.alignment_target_rms_arcmin));
        set("alignment_pause_on_fail", json!(self.alignment_pause_on_fail));
        set("meo_enabled", json!(self.meo_enabled));
        set("geo_enabled", json!(self.geo_enabled));
        set("satellite_labels_enabled", json!(self.satellite_labels_enabled));
        for (i, c) in self.cam.iter().enumerate() {
            let name = format!("camera{}", i + 1);
            let entry = o
                .entry("camera_configs")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .unwrap()
                .entry(name)
                .or_insert_with(|| json!({}));
            let e = entry.as_object_mut().unwrap();
            e.insert("name".into(), json!(c.name));
            e.insert("role".into(), json!(c.role));
            e.insert("asi_index".into(), json!(c.asi_index));
            e.insert("enabled".into(), json!(c.enabled));
            e.insert("pixel_size".into(), json!(c.pixel_um));
            e.insert("focal_length".into(), json!(c.focal_mm));
            e.insert("sensor_width".into(), json!(c.sensor_w));
            e.insert("sensor_height".into(), json!(c.sensor_h));
            e.insert("bin".into(), json!(c.bin));
            e.insert("projection".into(), json!(if c.projection == Projection::Pinhole { "pinhole" } else { "fisheye" }));
            e.insert("alignment_rotation".into(), json!(c.alignment_rotation_deg));
            e.insert("gain".into(), json!(c.gain));
            e.insert("exposure".into(), json!(c.exposure_us));
            e.insert("exposure_us".into(), json!(c.exposure_us));
            e.insert("gamma".into(), json!(c.gamma));
            e.insert("gamma_enabled".into(), json!(c.gamma_enabled));
            e.insert("tetra3_db".into(), json!(c.tetra3_db));
        }
        o.insert(
            "filter_wheels".into(),
            json!(self
                .filter_wheels
                .iter()
                .map(|w| json!({"name": w.name, "kind": w.kind, "camera": w.camera, "efw_index": w.efw_index, "slots": w.slots, "position": w.position}))
                .collect::<Vec<_>>()),
        );
        o.insert("efw_dll".into(), json!(self.efw_dll));
        let sc = o.entry("sim_config").or_insert_with(|| json!({})).as_object_mut().unwrap();
        let s = &self.sim;
        sc.insert("mount_misalignment_az_deg".into(), json!(s.misalign_az_deg));
        sc.insert("mount_misalignment_el_deg".into(), json!(s.misalign_el_deg));
        sc.insert("mount_encoder_noise_deg".into(), json!(s.encoder_noise_deg));
        sc.insert("mount_rate_noise_dps".into(), json!(s.rate_noise_dps));
        sc.insert("mount_backlash_deg".into(), json!(s.backlash_deg));
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
    /// Classical elements for the info pane (from the TLE).
    pub intl_desg: String,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub arg_perigee_deg: f64,
    pub mean_anomaly_deg: f64,
    pub mean_motion_rev_day: f64,
    pub eccentricity: f64,
    pub rev_number: u32,
    pub epoch_unix: f64,
    pub apogee_km: f64,
    pub perigee_km: f64,
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

#[derive(Clone, Debug)]
pub struct DsoMark {
    pub key: String,
    pub name: String,
    pub az: f64,
    pub el: f64,
    pub mag: f64,
    pub messier: bool,
}

/// A launch trajectory (launches/*.txt: T0 + ECEF states, or *.csv with
/// time/azimuthDegs/elevationDegs/rangeKm), converted to topocentric rows.
#[derive(Clone, Debug)]
pub struct LaunchTrack {
    pub key: String,
    pub name: String,
    /// (unix s, az, el, range_km)
    pub rows: Vec<(f64, f64, f64, f64)>,
}

/// Live track of a non-satellite selection (body / star / DSO), so the
/// mount worker can follow it without its own ephemeris.
#[derive(Clone, Debug)]
pub struct TargetTrack {
    pub key: String,
    pub name: String,
    pub jd_tt: f64,
    pub az: f64,
    pub el: f64,
    pub az_rate: f64,
    pub el_rate: f64,
}

/// Published by the sky worker (~2 Hz sats/bodies, ~0.2 Hz stars).
#[derive(Clone, Debug, Default)]
pub struct SkySnapshot {
    pub jd_tt: f64,
    pub utc_iso: String,
    pub sats: Vec<SatMark>,
    pub stars: Vec<StarMark>,
    pub bodies: Vec<BodyMark>,
    pub dsos: Vec<DsoMark>,
    /// Rocket launch trajectories from the launches directory.
    pub launches: Arc<Vec<LaunchTrack>>,
    /// HIP -> IAU proper name (shared, built once).
    pub star_names: Arc<std::collections::HashMap<i64, String>>,
    pub target: Option<TargetTrack>,
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

/// One aircraft from the ADS-B worker.
#[derive(Clone, Debug)]
pub struct AircraftMark {
    pub icao: String,
    pub label: String,
    pub az: f64,
    pub el: f64,
    pub range_km: f64,
    pub alt_m: f64,
    pub speed_kt: Option<f64>,
    pub track_deg: Option<f64>,
    pub t_fix_unix: f64,
    pub age_s: f64,
    /// Linear-fit state at `fit_t_unix` (deg, deg/s) for dead reckoning.
    pub fit_az: f64,
    pub fit_el: f64,
    pub fit_t_unix: f64,
    pub az_rate: f64,
    pub el_rate: f64,
    /// Predicted (unix, az, el) ahead of the last fix.
    pub predicted: Vec<(f64, f64, f64)>,
    /// Observed (unix, az, el) history.
    pub history: Vec<(f64, f64, f64)>,
}

#[derive(Clone, Debug, Default)]
pub struct AdsbSnapshot {
    pub status: String,
    pub mode: String,
    pub n_aircraft: usize,
    pub n_msgs: u64,
    pub aircraft: Vec<AircraftMark>,
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
    /// Operator bias (cross-el, el) in degrees and the D-pad step mode.
    pub bias: (f64, f64),
    pub bias_fine: bool,
    pub parking: bool,
    pub mount_mode: String,
    pub pointing_model_on: bool,
    /// Last pointing-model correction applied to the setpoint (deg).
    pub pointing_corr: (f64, f64),
    pub joystick_tare: (f64, f64),
    /// Live gamepad state (virtual-controller panel). Bit order mirrors the
    /// Python BUTTON_LABELS: 0=Cross 1=Circle 2=Square 3=Triangle 4=Share
    /// 5=PS 6=Options 7=L3 8=R3 9=L1 10=R1 11..14=DPad U/D/L/R.
    /// Universal STOP latch (Circle / STOP button): rates forced to zero in
    /// any mode until released by a mode selection or a second press.
    pub stopped: bool,
    /// Focus motor: commanded discrete rate and encoder counts (24-bit).
    pub focus_rate: i32,
    pub focus_pos: i64,
    /// Operator-bias frame: "azel" or "alongcross" (Share cycles 4 states).
    pub bias_frame: String,
    /// In-track / cross-track bias (deg) when the alongcross frame is active.
    pub bias_it: f64,
    pub bias_ct: f64,
    /// Live feed-forward enables (per axis) and lead time, for the panel.
    pub ff_azm: bool,
    pub ff_alt: bool,
    pub lead_time_s: f64,
    pub star_filter: bool,
    pub joy_buttons: u32,
    pub joy_left: (f64, f64),
    pub joy_right: (f64, f64),
    /// L2 / R2 analog values (0..1).
    pub joy_triggers: (f64, f64),
    pub gear_base: i32,
    pub gear_max: i32,
}

/// Published by the camera worker for every pumped frame.
#[derive(Clone)]
pub struct CamSnapshot {
    pub slot: usize,
    pub name: String,
    pub role: String,
    pub data: Arc<Vec<u8>>,
    pub width: usize,
    pub height: usize,
    pub seq: u64,
    pub fps: f64,
    pub utc_midpoint_s: f64,
    pub fov_deg: f64,
    /// Effective plate scale (deg/px) at the published frame size.
    pub deg_per_px: f64,
    pub fisheye: bool,
    pub source: String,
    pub connected: bool,
    pub armed: bool,
    pub armed_frames: usize,
    pub last_dump: Option<String>,
    pub deep_stars: usize,
    pub hw_index: usize,
}

/// Filter wheel state published by the filter-wheel worker.
#[derive(Clone, Debug, Default)]
pub struct WheelState {
    pub name: String,
    pub kind: String,
    pub camera: usize,
    pub slots: Vec<String>,
    pub position: usize,
    pub target: Option<usize>,
    pub moving: bool,
    pub connected: bool,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct FilterWheelSnapshot {
    pub wheels: Vec<WheelState>,
    pub sdk: String,
}

/// UI -> filter-wheel worker.
#[derive(Clone, Debug)]
pub enum FwCmd {
    Goto { wheel: usize, slot: usize },
    SetSlotName { wheel: usize, slot: usize, name: String },
    /// Manual wheel: the operator reports the position.
    MarkPosition { wheel: usize, slot: usize },
    Save,
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
    ToggleFeedForward,
    /// Operator bias on the setpoint: on-sky cross-elevation / elevation (deg).
    Bias { daz: f64, del: f64 },
    BiasReset,
    BiasFine(bool),
    /// Cycle the bias mode: coarse/azel → fine/azel → coarse/alongcross → fine/alongcross.
    CycleBiasMode,
    SetLeadTime(f64),
    SetStarFilter(bool),
    SetFeedForward { az: bool, el: bool },
    /// Park: drive the axes to the configured offsets (mount frame 0/0).
    Park,
    /// Options button: cycle AltAz -> AltAz-Side -> Eq -> Passthrough (persisted).
    CycleMountMode,
    SetMountMode(String),
    /// Square button: tare the joystick axes (current stick = zero).
    TareJoystick,
    /// Enable / disable the pointing-model pre-correction (persisted).
    PointingModel(bool),
    /// Runtime mount connection: (re)open the loop on "sim" or "serial:<port>".
    Connect { transport: String, port: String, baud: u32 },
    Disconnect,
    ListPorts,
}

/// UI -> camera worker.
#[derive(Clone, Debug)]
pub enum CamCmd {
    /// Arm every connected camera (one run directory, CameraN_ files).
    Arm,
    Dump { name: String },
    Disarm,
    /// Live settings changed (gain / exposure / gamma / rotation / ROI / connect).
    Apply { slot: usize },
    /// Exchange the hardware indices behind two logical slots and reconnect.
    Swap { a: usize, b: usize },
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
    pub adsb: ArcSwap<AdsbSnapshot>,
    pub sky: ArcSwap<SkySnapshot>,
    pub passes: ArcSwap<PassesSnapshot>,
    pub mount: ArcSwap<MountSnapshot>,
    /// Per logical camera slot.
    pub cams: Vec<ArcSwap<Option<CamSnapshot>>>,
    pub cam_settings: Vec<ArcSwap<CamSettings>>,
    /// Hardware (ASI enumeration) index behind each logical slot; swapped by CamCmd::Swap.
    pub cam_hw: std::sync::Mutex<Vec<usize>>,
    pub filter_wheels: ArcSwap<FilterWheelSnapshot>,
    pub solve: ArcSwap<SolveSnapshot>,
    pub align: ArcSwap<AlignSnapshot>,
    pub sim: ArcSwap<SimSettings>,
    pub config: Config,
    /// The core control loop's shared block (inputs / outputs / frame slot).
    pub core: Arc<LoopShared>,
    /// Selected satellite (mirrored from the mount worker for the sky worker).
    pub selected: ArcSwap<Option<String>>,
    /// TLE catalog generation: bumped by the sky worker after a (re)load so
    /// the mount worker reloads its own catalog; `tle_refresh` asks for a download.
    pub tle_version: std::sync::atomic::AtomicU64,
    pub tle_refresh: std::sync::atomic::AtomicBool,
    pub tle_status: ArcSwap<String>,
    /// Live pointing model (enabled, terms) applied to PROGRAM/HANDOFF setpoints.
    pub pointing: ArcSwap<(bool, [f64; 7])>,
    /// Eq-mode 7-term residual model (enabled, IH..TF), applied in the mount
    /// (HA/Dec) frame when mount_mode is Eq.
    pub eq_pointing: ArcSwap<(bool, [f64; 7])>,
    /// ADS-B linear-fit depth (fixes), live-tunable from the UI.
    pub adsb_fit_points: std::sync::atomic::AtomicUsize,
    /// Launch override: LAUNCH button re-bases the selected trajectory's T0
    /// to "now" and pre-empts every other target until aborted.
    pub launch_armed: ArcSwap<Option<(String, f64)>>,
    /// Live mount mode name (Options button cycles it).
    pub mount_mode: ArcSwap<String>,
    /// Serial ports enumerated on request (MountCmd::ListPorts).
    pub serial_ports: ArcSwap<Vec<String>>,
    /// ADS-B source switch request: Some("off" | "rtlsdr" | "dump1090" | "sim").
    pub adsb_request: ArcSwap<Option<String>>,
}

impl Shared {
    /// Snapshot of one camera slot (None when the slot has no frames yet).
    pub fn cam(&self, slot: usize) -> Option<CamSnapshot> {
        self.cams.get(slot).and_then(|c| c.load().as_ref().clone())
    }
    /// The camera slot HOTSPOT/HANDOFF detect on.
    pub fn hotspot_slot(&self) -> usize {
        self.config.hotspot_camera_index.min(self.config.cam.len().saturating_sub(1))
    }
    /// The camera slot the plate solver uses.
    pub fn solve_slot(&self) -> usize {
        self.config.plate_solve_camera_index.min(self.config.cam.len().saturating_sub(1))
    }

    pub fn new(config: Config, core: Arc<LoopShared>) -> Arc<Self> {
        let sim = config.sim.clone();
        let pointing0 = (config.pointing_model_enabled, config.pointing_model_terms);
        let mount_mode0 = config.mount_mode.clone();
        let eq0 = (config.eq_pointing_model_enabled, config.eq_pointing_model_terms);
        let fit0 = config.adsb_fit_points.max(2);
        Arc::new(Shared {
            adsb: ArcSwap::from_pointee(AdsbSnapshot::default()),
            sky: ArcSwap::from_pointee(SkySnapshot::default()),
            passes: ArcSwap::from_pointee(PassesSnapshot::default()),
            mount: ArcSwap::from_pointee(MountSnapshot::default()),
            cams: config.cam.iter().map(|_| ArcSwap::from_pointee(None)).collect(),
            cam_settings: config.cam.iter().map(|c| ArcSwap::from_pointee(CamSettings::from_config(c))).collect(),
            cam_hw: std::sync::Mutex::new(config.cam.iter().map(|c| c.asi_index).collect()),
            filter_wheels: ArcSwap::from_pointee(FilterWheelSnapshot::default()),
            solve: ArcSwap::from_pointee(SolveSnapshot::default()),
            align: ArcSwap::from_pointee(AlignSnapshot::default()),
            sim: ArcSwap::from_pointee(sim),
            config,
            core,
            selected: ArcSwap::from_pointee(None),
            tle_version: std::sync::atomic::AtomicU64::new(0),
            tle_refresh: std::sync::atomic::AtomicBool::new(false),
            tle_status: ArcSwap::from_pointee(String::new()),
            pointing: ArcSwap::from_pointee(pointing0),
            eq_pointing: ArcSwap::from_pointee(eq0),
            adsb_fit_points: std::sync::atomic::AtomicUsize::new(fit0),
            launch_armed: ArcSwap::from_pointee(None),
            mount_mode: ArcSwap::from_pointee(mount_mode0),
            serial_ports: ArcSwap::from_pointee(Vec::new()),
            adsb_request: ArcSwap::from_pointee(None),
        })
    }
}
