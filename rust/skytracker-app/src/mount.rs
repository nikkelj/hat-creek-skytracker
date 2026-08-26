//! Mount worker: the Rust core loop over either the byte-level simulated
//! mount or a real NexStar serial port (config `mount_transport`), fed by
//! the gamepad through the adaptive rate gearbox (RATE), by live satellite
//! setpoints (PROGRAM / HANDOFF), or by the camera frame slot (HOTSPOT —
//! the camera worker publishes frames straight into the loop). UI talks to
//! it only through MountCmd; it publishes MountSnapshot every cycle.

use crate::state::{Config, MountCmd, MountSnapshot, Shared};
use crossbeam_channel::Receiver;
use skytracker_astro::sgp4_pass::{self, Observer};
use skytracker_astro::tle::TleCatalog;
use skytracker_core::autotune::PidAutoTuner;
use skytracker_core::controller::{HotspotParams, Inputs, Mode, Setpoint};
use skytracker_core::core_loop::{Command, CoreLoop, Shared as LoopShared};
use skytracker_core::rate::AdaptiveRateMapper;
use skytracker_core::sim::{LoopbackTransport, Mount, SimResponder};
use skytracker_core::transforms::{self, MountMode};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn spawn(shared: Arc<Shared>, rx: Receiver<MountCmd>, repo_root: std::path::PathBuf, tx_cam: crossbeam_channel::Sender<crate::state::CamCmd>) {
    std::thread::Builder::new()
        .name("mount-worker".into())
        .spawn(move || run(shared, rx, repo_root, tx_cam))
        .expect("spawn mount worker");
}

const MODE_CYCLE: [Mode; 5] = [Mode::Standby, Mode::Rate, Mode::Program, Mode::Handoff, Mode::Hotspot];

pub fn parse_mount_mode(s: &str) -> MountMode {
    match s.to_ascii_lowercase().replace('-', "_").as_str() {
        "altaz_side" | "altazside" => MountMode::AltAzSide,
        "passthrough" => MountMode::Passthrough,
        "eq" => MountMode::Eq,
        _ => MountMode::AltAz,
    }
}

pub fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Standby => "STANDBY",
        Mode::Rate => "RATE",
        Mode::Program => "PROGRAM",
        Mode::Handoff => "HANDOFF",
        Mode::Hotspot => "HOTSPOT",
        Mode::Mti => "MTI",
    }
}

/// Persist the departing mode's live gains into its pid_mode_profiles entry
/// (service_gain_profiles): read-modify-write of config.json keeping the
/// tuned_on / tuned_at / tuned_rms provenance stamps intact. `stamp` adds or
/// refreshes those stamps (after an autotune completes).
fn save_gain_profile(path: &std::path::Path, key: &str, azm: (f64, f64, f64), alt: (f64, f64, f64), stamp: Option<(&str, f64)>) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let Ok(mut raw) = serde_json::from_str::<serde_json::Value>(&text) else { return };
    let Some(o) = raw.as_object_mut() else { return };
    let profiles = o.entry("pid_mode_profiles").or_insert_with(|| serde_json::json!({}));
    let Some(po) = profiles.as_object_mut() else { return };
    let entry = po.entry(key).or_insert_with(|| serde_json::json!({}));
    if let Some(eo) = entry.as_object_mut() {
        eo.insert("gains".into(), serde_json::json!({
            "pid_azm_p_gain": azm.0, "pid_azm_i_gain": azm.1, "pid_azm_d_gain": azm.2,
            "pid_alt_p_gain": alt.0, "pid_alt_i_gain": alt.1, "pid_alt_d_gain": alt.2,
        }));
        if let Some((target, rms)) = stamp {
            eo.insert("tuned_on".into(), serde_json::json!(target));
            let (y, mo, d, _, _, _) = crate::sky::civil_from_unix(crate::sky::now_unix());
            eo.insert("tuned_at".into(), serde_json::json!(format!("{y:04}-{mo:02}-{d:02}")));
            if rms.is_finite() {
                eo.insert("tuned_rms".into(), serde_json::json!(rms));
            }
        }
    }
    if let Ok(out) = serde_json::to_string_pretty(&raw) {
        let _ = std::fs::write(path, out);
    }
}

/// Persist the operator bias (Python bias_azm_deg / bias_alt_deg /
/// bias_control_mode) so it survives restarts.
fn persist_bias(path: &std::path::Path, bias: (f64, f64), fine: bool) {
    persist_config_key(path, "bias_azm_deg", serde_json::json!(bias.0));
    persist_config_key(path, "bias_alt_deg", serde_json::json!(bias.1));
    persist_config_key(path, "bias_control_mode", serde_json::json!(if fine { "fine" } else { "coarse" }));
}

fn raw_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

/// Core-loop inputs from config: gains, limits, transforms, hotspot params
/// (camera `hotspot_camera_index` supplies the plate scale + rotation).
pub fn make_inputs(cfg: &Config) -> Inputs {
    let mut inputs = Inputs::default();
    inputs.connected = true;
    inputs.mode = Mode::Standby;
    inputs.azm_gains = cfg.azm_gains;
    inputs.alt_gains = cfg.alt_gains;
    inputs.azm_limit = cfg.azm_limit;
    inputs.alt_limit = cfg.alt_limit;
    inputs.offsets = cfg.offsets;
    inputs.mount_mode = parse_mount_mode(&cfg.mount_mode);
    inputs.alignment_az = cfg.alignment_az;
    inputs.alignment_el = cfg.alignment_el;
    inputs.altaz_side_flip = cfg.altaz_side_flip;
    inputs.ff_azm_enabled = cfg.ff_azm;
    inputs.ff_alt_enabled = cfg.ff_alt;
    inputs.lead_time_sec = cfg.lead_time_s;
    inputs.continuous_rate = cfg.continuous_rate;
    inputs.guide_rate_max_dps = cfg.guide_rate_max_dps;
    inputs.output_filter_tau = cfg.output_filter_tau;
    inputs.handoff_min_frames = cfg.handoff_min_frames.max(1);
    let cam = &cfg.cam[cfg.hotspot_camera_index.min(cfg.cam.len().saturating_sub(1))];
    inputs.hotspot = HotspotParams {
        snr_threshold: cfg.hotspot_snr,
        gate_radius: cfg.hotspot_gate_radius,
        coast_time_s: cfg.hotspot_coast_s,
        max_rate_dps: cfg.hotspot_max_rate_dps,
        x_sign: cfg.hotspot_x_sign,
        y_sign: cfg.hotspot_y_sign,
        pixel_size_um: cam.pixel_eff_um(),
        focal_length_mm: cam.focal_mm,
        rotation_deg: cam.alignment_rotation_deg,
        star_filter: cfg.hotspot_star_filter,
        rate_gate_dps: cfg.hotspot_rate_gate_dps,
    };
    inputs
}

/// Mount (azm, alt) -> sky (az, el) for the configured mode (inverse of
/// transforms::sky_to_mount, the PROGRAM path).
pub fn mount_to_sky(cfg: &Config, azm: f64, alt: f64) -> (f64, f64) {
    transforms::mount_to_sky(parse_mount_mode(&cfg.mount_mode), azm, alt, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip)
}

const MOUNT_MODE_CYCLE: [&str; 4] = ["AltAz", "AltAz-Side", "Eq", "Passthrough"];

/// Persist one top-level config key (read-modify-write, other keys untouched).
pub(crate) fn persist_config_key(path: &std::path::Path, key: &str, value: serde_json::Value) {
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Ok(mut raw) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(o) = raw.as_object_mut() {
                o.insert(key.to_string(), value);
                if let Ok(s) = serde_json::to_string_pretty(&raw) {
                    let _ = std::fs::write(path, s);
                }
            }
        }
    }
}

fn spawn_loop(cfg: &Config, loop_shared: &Arc<LoopShared>) -> (CoreLoop, String, Vec<String>, Option<Arc<std::sync::Mutex<skytracker_core::sim::SimNoise>>>) {
    let mut notes = Vec::new();
    let hz = if cfg.loop_hz > 0.0 { cfg.loop_hz } else { 15.0 };
    if cfg.mount_transport.eq_ignore_ascii_case("serial") {
        #[cfg(feature = "serial")]
        {
            match skytracker_core::serial::SerialTransport::open(&cfg.serial_port, cfg.serial_baud, 1500) {
                Ok(port) => {
                    notes.push(format!("serial {} @ {}", cfg.serial_port, cfg.serial_baud));
                    let mount = Mount::new(port);
                    return (CoreLoop::spawn(mount, loop_shared.clone(), hz), format!("serial {}", cfg.serial_port), notes, None);
                }
                Err(e) => notes.push(format!("serial {} failed: {e} -- using sim", cfg.serial_port)),
            }
        }
        #[cfg(not(feature = "serial"))]
        notes.push("built without the serial feature -- using sim".into());
    }
    // Wall-clock byte-level sim mount, parked at SKY az 180 / el 45 (raw
    // encoder angles through the configured mode + offsets).
    let (azm, alt) = transforms::sky_to_mount(parse_mount_mode(&cfg.mount_mode), 180.0, 45.0, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip);
    let responder = SimResponder::new_wall(azm + cfg.offsets.0, alt + cfg.offsets.1);
    let noise = responder.noise.clone();
    let mount = Mount::new(LoopbackTransport::new(responder));
    (CoreLoop::spawn(mount, loop_shared.clone(), hz), "sim".into(), notes, Some(noise))
}

fn run(shared: Arc<Shared>, rx: Receiver<MountCmd>, root: std::path::PathBuf, tx_cam: crossbeam_channel::Sender<crate::state::CamCmd>) {
    let cfg = shared.config.clone();
    let loop_shared = shared.core.clone();
    let (mut core, mut transport, notes, mut sim_noise) = spawn_loop(&cfg, &loop_shared);

    // Gamepad.
    let mut gilrs = gilrs::Gilrs::new().ok();
    let mut mapper = AdaptiveRateMapper::new(cfg.joy_rate_base_ceiling.clamp(1, 9), 9, cfg.joy_rate_windup_delay_s.max(0.0));
    let mut tare = (0.0f64, 0.0f64);
    let mut last_stick_raw = (0.0f64, 0.0f64);
    let mut cur_mode_name = cfg.mount_mode.clone();
    let mut cfg = cfg.clone();
    let epoch = loop_shared.epoch;

    // Satellite catalog for PROGRAM / HANDOFF setpoints (shares the TLE file).
    let mut catalog = TleCatalog::load(&root.join("tle_cache.tle")).ok();
    let mut tle_version = shared.tle_version.load(std::sync::atomic::Ordering::Relaxed);
    let observer = Observer {
        lat_deg: cfg.lat_deg,
        lon_deg: cfg.lon_deg,
        elevation_m: cfg.alt_m,
    };
    let geom = observer.geometry();

    let mut mode = Mode::Standby;
    let mut target: Option<String> = None;
    let mut status: Vec<String> = notes;
    let push_status = |status: &mut Vec<String>, s: String| {
        // 300-message scrollback (Python keeps 500) — the UI shows the tail
        // inline and the full list in a scrollable rollup.
        status.push(s);
        if status.len() > 300 {
            status.remove(0);
        }
    };
    let mut tuner = PidAutoTuner::new(
        [cfg.azm_gains.0, cfg.azm_gains.1, cfg.azm_gains.2],
        [cfg.alt_gains.0, cfg.alt_gains.1, cfg.alt_gains.2],
    );
    let mut tuner_active = false;
    let mut last_setpoint: Option<(f64, f64)> = None;
    // Large initial errors are closed with a GOTO (as the Python app does on
    // target select) and the PID takes over inside 1 degree.
    let mut slewing = false;
    let mut last_goto = Instant::now() - Duration::from_secs(10);
    let mut prev_target: Option<String> = None;
    // Operator bias (on-sky cross-el / el, deg) and its D-pad step;
    // persisted across runs (bias_azm_deg / bias_alt_deg / bias_control_mode).
    let mut bias = (
        raw_f64(&cfg.raw["bias_azm_deg"]).unwrap_or(0.0).clamp(-3.0, 3.0),
        raw_f64(&cfg.raw["bias_alt_deg"]).unwrap_or(0.0).clamp(-3.0, 3.0),
    );
    let mut bias_fine = cfg.raw["bias_control_mode"].as_str().map_or(false, |s| s == "fine");
    // Bias frame: "azel" nudges cross-el/el; "alongcross" projects onto the
    // target's sky-velocity direction (in-track / cross-track).
    let mut bias_frame_ac = false;
    let mut bias_it = 0.0f64;
    let mut bias_ct = 0.0f64;
    let mut focus_last_rate = 0i32;
    let mut park_state: Option<(f64, f64)> = None;
    let mut parking = false;
    let mut stopped = false;
    let mut rate_limit_warned = false;
    let mut pm_corr_last = (0.0f64, 0.0f64);
    // Per-mode PID gain profiles (service_gain_profiles): the departing
    // mode's live gains are SAVED into its profile (and persisted), the
    // arriving mode's profile is loaded; first entry seeds from live gains.
    let mut profiles: std::collections::HashMap<String, ((f64, f64, f64), (f64, f64, f64))> =
        cfg.pid_profiles.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let mut active_profile: Option<&'static str> = None;
    let profile_key = |mode: Mode| -> Option<&'static str> {
        match mode {
            Mode::Program | Mode::Handoff => Some("PROGRAM"),
            Mode::Hotspot => Some("HOTSPOT"),
            _ => None,
        }
    };

    loop {
        let t0 = Instant::now();
        let now = epoch.elapsed().as_secs_f64();

        // Commands from the UI.
        while let Ok(cmd) = rx.try_recv() {
            let cmd_copy = cmd.clone();
            match cmd {
                MountCmd::SetMode(m) => {
                    stopped = false;
                    rate_limit_warned = false;
                    mode = match m.as_str() {
                        "RATE" => Mode::Rate,
                        "PROGRAM" => Mode::Program,
                        "HANDOFF" => Mode::Handoff,
                        "HOTSPOT" => Mode::Hotspot,
                        "MTI" => Mode::Mti,
                        _ => Mode::Standby,
                    };
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.mode = mode;
                    i.stopped = false;
                    if !matches!(mode, Mode::Program | Mode::Handoff | Mode::Hotspot) {
                        i.setpoint = None;
                    }
                    drop(i);
                    push_status(&mut status, format!("mode -> {}", mode_name(mode)));
                    {
                    let key = profile_key(mode);
                    if key != active_profile {
                        let live = {
                            let i = loop_shared.inputs.lock().unwrap();
                            (i.azm_gains, i.alt_gains)
                        };
                        if let Some(old_key) = active_profile {
                            profiles.insert(old_key.to_string(), live);
                            save_gain_profile(&cfg.path, old_key, live.0, live.1, None);
                        }
                        if let Some(k) = key {
                            if let Some(&(a, e)) = profiles.get(k) {
                                let mut i = loop_shared.inputs.lock().unwrap();
                                if i.azm_gains != a || i.alt_gains != e {
                                    i.azm_gains = a;
                                    i.alt_gains = e;
                                    drop(i);
                                    push_status(&mut status, format!("gains <- {k} profile"));
                                }
                            } else {
                                profiles.insert(k.to_string(), live);
                                save_gain_profile(&cfg.path, k, live.0, live.1, None);
                                push_status(&mut status, format!("PID gains: seeded new {k} profile"));
                            }
                        }
                        active_profile = key;
                    }
                }
                    parking = false;
                }
                MountCmd::SelectTarget(t) => {
                    target = t;
                    shared.selected.store(Arc::new(target.clone()));
                    push_status(
                        &mut status,
                        match &target {
                            Some(s) => format!("target {s}"),
                            None => "target cleared".into(),
                        },
                    );
                    if target.as_deref().map_or(false, |t| t.eq_ignore_ascii_case("body:sun")) {
                        push_status(&mut status, "⚠ SUN selected — NEVER observe without a proper solar filter!".into());
                    }
                }
                MountCmd::Stop => {
                    mode = Mode::Standby;
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.mode = Mode::Standby;
                    i.rate_cmd = (0, 0);
                    i.setpoint = None;
                    drop(i);
                    loop_shared.commands.lock().unwrap().push_back(Command::Stop);
                    stopped = true;
                    push_status(&mut status, "STOP latched — select a mode or press STOP again to release".into());
                }
                MountCmd::Goto { az, el } => {
                    // Sky az/el -> mount axes -> raw encoder degrees (offsets).
                    let (azm, alt) = transforms::sky_to_mount(parse_mount_mode(&cfg.mount_mode), az, el, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip);
                    loop_shared
                        .commands
                        .lock()
                        .unwrap()
                        .push_back(Command::GotoMount { azm_deg: azm + cfg.offsets.0, alt_deg: alt + cfg.offsets.1 });
                    push_status(&mut status, format!("goto sky {az:.2} / {el:.2}"));
                }
                MountCmd::Nudge { daz, del } => {
                    let out = loop_shared.outputs.lock().unwrap().clone();
                    loop_shared.commands.lock().unwrap().push_back(Command::GotoMount {
                        azm_deg: out.azm + daz,
                        alt_deg: out.alt + del,
                    });
                }
                MountCmd::SetGains { azm, alt } => {
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.azm_gains = azm;
                    i.alt_gains = alt;
                    drop(i);
                    if !tuner_active {
                        tuner = PidAutoTuner::new([azm.0, azm.1, azm.2], [alt.0, alt.1, alt.2]);
                    }
                    push_status(&mut status, "gains updated".into());
                }
                MountCmd::SetHotspotSigns { x, y } => {
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.hotspot.x_sign = x;
                    i.hotspot.y_sign = y;
                    drop(i);
                    push_status(&mut status, format!("hotspot signs x{x:+.0} y{y:+.0}"));
                }
                MountCmd::Bias { daz, del } => {
                    if bias_frame_ac {
                        bias_it = (bias_it + daz).clamp(-3.0, 3.0);
                        bias_ct = (bias_ct + del).clamp(-3.0, 3.0);
                    } else {
                        bias.0 = (bias.0 + daz).clamp(-3.0, 3.0);
                        bias.1 = (bias.1 + del).clamp(-3.0, 3.0);
                        persist_bias(&cfg.path, bias, bias_fine);
                    }
                }
                MountCmd::BiasReset => {
                    bias = (0.0, 0.0);
                    bias_it = 0.0;
                    bias_ct = 0.0;
                    persist_bias(&cfg.path, bias, bias_fine);
                }
                MountCmd::CycleBiasMode => {
                    // coarse/azel -> fine/azel -> coarse/alongcross -> fine/alongcross
                    let idx = (bias_fine as usize) | ((bias_frame_ac as usize) << 1);
                    let next = (idx + 1) % 4;
                    bias_fine = next & 1 == 1;
                    bias_frame_ac = next & 2 == 2;
                    persist_bias(&cfg.path, bias, bias_fine);
                    push_status(&mut status, format!("bias mode: {} / {}", if bias_fine { "fine" } else { "coarse" }, if bias_frame_ac { "along/cross-track" } else { "az/el" }));
                }
                MountCmd::SetLeadTime(t) => {
                    loop_shared.inputs.lock().unwrap().lead_time_sec = t.clamp(0.0, 0.5);
                    persist_config_key(&cfg.path, "pid_lead_time_sec", serde_json::json!(t.clamp(0.0, 0.5)));
                }
                MountCmd::SetStarFilter(on) => {
                    loop_shared.inputs.lock().unwrap().hotspot.star_filter = on;
                    persist_config_key(&cfg.path, "hotspot_star_filter_enabled", serde_json::json!(on));
                    push_status(&mut status, format!("hotspot star filter {}", if on { "ON" } else { "OFF" }));
                }
                MountCmd::SyncHome => {
                    // Tare az/alt on the physical indices: offsets := raw
                    // encoders here, so working position (raw − offset)
                    // reads 0/0 at the marks and PARK returns to them.
                    let (ra, re) = {
                        let o = loop_shared.outputs.lock().unwrap();
                        (o.azm_raw, o.alt_raw)
                    };
                    cfg.offsets = (ra, re);
                    loop_shared.inputs.lock().unwrap().offsets = (ra, re);
                    persist_config_key(&cfg.path, "azm_offset", serde_json::json!(format!("{ra}")));
                    persist_config_key(&cfg.path, "alt_offset", serde_json::json!(format!("{re}")));
                    push_status(&mut status, format!("HOME SYNCED: offsets = raw {ra:.3}° / {re:.3}° (indices are now 0/0)"));
                }
                MountCmd::SetAlignmentOffsets { az, el } => {
                    cfg.alignment_az = az;
                    cfg.alignment_el = el;
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.alignment_az = az;
                    i.alignment_el = el;
                    drop(i);
                    persist_config_key(&cfg.path, "alignment_azimuth", serde_json::json!(format!("{az}")));
                    persist_config_key(&cfg.path, "alignment_elevation", serde_json::json!(format!("{el}")));
                    push_status(&mut status, format!("alignment offsets -> az {az:.4}° / el {el:.4}°"));
                }
                MountCmd::GotoAxes { azm, alt } => {
                    loop_shared.commands.lock().unwrap().push_back(Command::GotoMount { azm_deg: azm + cfg.offsets.0, alt_deg: alt + cfg.offsets.1 });
                }
                MountCmd::SetFeedForward { az, el } => {
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.ff_azm_enabled = az;
                    i.ff_alt_enabled = el;
                    drop(i);
                    persist_config_key(&cfg.path, "feed_forward_azm_enabled", serde_json::json!(az));
                    persist_config_key(&cfg.path, "feed_forward_alt_enabled", serde_json::json!(el));
                    push_status(&mut status, format!("feed-forward az {} / el {}", if az { "ON" } else { "OFF" }, if el { "ON" } else { "OFF" }));
                }
                MountCmd::BiasFine(f) => {
                    bias_fine = f;
                    persist_bias(&cfg.path, bias, bias_fine);
                }
                MountCmd::Park => {
                    mode = Mode::Standby;
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.mode = Mode::Standby;
                    i.rate_cmd = (0, 0);
                    i.setpoint = None;
                    drop(i);
                    loop_shared.commands.lock().unwrap().push_back(Command::GotoMount { azm_deg: cfg.offsets.0, alt_deg: cfg.offsets.1 });
                    parking = true;
                    push_status(&mut status, "PARK -> configured offsets".into());
                }
                MountCmd::CycleMountMode | MountCmd::SetMountMode(_) => {
                    let next = match &cmd_copy {
                        MountCmd::SetMountMode(m) => m.clone(),
                        _ => {
                            let cur = MOUNT_MODE_CYCLE.iter().position(|m| parse_mount_mode(m) == parse_mount_mode(&cur_mode_name)).unwrap_or(0);
                            MOUNT_MODE_CYCLE[(cur + 1) % MOUNT_MODE_CYCLE.len()].to_string()
                        }
                    };
                    cur_mode_name = next.clone();
                    cfg.mount_mode = next.clone();
                    loop_shared.inputs.lock().unwrap().mount_mode = parse_mount_mode(&next);
                    shared.mount_mode.store(Arc::new(next.clone()));
                    persist_config_key(&cfg.path, "mount_mode", serde_json::json!(next));
                    push_status(&mut status, format!("mount mode -> {next}"));
                }
                MountCmd::TareJoystick => {
                    tare = last_stick_raw;
                    push_status(&mut status, format!("joystick tared at {:+.2} / {:+.2}", tare.0, tare.1));
                }
                MountCmd::PointingModel(on) => {
                    let (_, t) = **shared.pointing.load();
                    shared.pointing.store(Arc::new((on, t)));
                    persist_config_key(&cfg.path, "pointing_model_enabled", serde_json::json!(on));
                    push_status(&mut status, format!("pointing model {}", if on { "ON" } else { "OFF" }));
                }
                MountCmd::ListPorts => {
                    let ports: Vec<String> = serialport::available_ports()
                        .map(|v| v.into_iter().map(|p| p.port_name).collect())
                        .unwrap_or_default();
                    shared.serial_ports.store(Arc::new(ports));
                }
                MountCmd::Connect { transport: tr, port, baud } => {
                    // Stop the running loop, then spawn a new one on the requested transport.
                    core.stop();
                    cfg.mount_transport = tr.clone();
                    cfg.serial_port = port.clone();
                    cfg.serial_baud = baud;
                    let (c2, t2, notes2, n2) = spawn_loop(&cfg, &loop_shared);
                    core = c2;
                    transport = t2;
                    sim_noise = n2;
                    for n in notes2 {
                        push_status(&mut status, n);
                    }
                    push_status(&mut status, format!("connected: {transport}"));
                    persist_config_key(&cfg.path, "mount_transport", serde_json::json!(tr));
                    persist_config_key(&cfg.path, "mount_serial_port", serde_json::json!(port));
                    persist_config_key(&cfg.path, "mount_serial_baud", serde_json::json!(baud));
                    mode = Mode::Standby;
                    loop_shared.inputs.lock().unwrap().mode = Mode::Standby;
                }
                MountCmd::Disconnect => {
                    core.stop();
                    // A stopped loop publishes loop_dead; keep a sim loop alive so the UI stays sane.
                    cfg.mount_transport = "sim".into();
                    let (c2, t2, _, n2) = spawn_loop(&cfg, &loop_shared);
                    core = c2;
                    sim_noise = n2;
                    transport = format!("{t2} (mount disconnected)");
                    mode = Mode::Standby;
                    loop_shared.inputs.lock().unwrap().mode = Mode::Standby;
                    push_status(&mut status, "mount disconnected -- sim loop".into());
                }
                MountCmd::ToggleFeedForward => {
                    let mut i = loop_shared.inputs.lock().unwrap();
                    let on = !(i.ff_azm_enabled && i.ff_alt_enabled);
                    i.ff_azm_enabled = on;
                    i.ff_alt_enabled = on;
                    drop(i);
                    push_status(&mut status, format!("feed-forward {}", if on { "ON" } else { "OFF" }));
                }
                MountCmd::AutotuneStart => {
                    let i = loop_shared.inputs.lock().unwrap();
                    let (a, b) = (i.azm_gains, i.alt_gains);
                    drop(i);
                    tuner = PidAutoTuner::new([a.0, a.1, a.2], [b.0, b.1, b.2]);
                    tuner.start(now);
                    tuner_active = true;
                    push_status(&mut status, "autotune started".into());
                }
                MountCmd::AutotuneStop { revert } => {
                    tuner.stop(revert);
                    tuner_active = false;
                    let g = tuner.applied_gains();
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.azm_gains = (g[0][0], g[0][1], g[0][2]);
                    i.alt_gains = (g[1][0], g[1][1], g[1][2]);
                    drop(i);
                    push_status(&mut status, format!("autotune stopped ({})", if revert { "reverted" } else { "kept" }));
                }
            }
        }

        // Live sim imperfections from the Sim screen.
        if let Some(n) = &sim_noise {
            let s = shared.sim.load();
            let want = skytracker_core::sim::SimNoise { encoder_deg: s.encoder_noise_deg, rate_dps: s.rate_noise_dps, backlash_deg: s.backlash_deg, serial_latency_s: s.serial_latency_s, serial_garbage_prob: s.serial_garbage_prob };
            let mut g = n.lock().unwrap();
            if *g != want {
                *g = want;
            }
        }

        // Reload the satellite catalog after a TLE refresh.
        let tv = shared.tle_version.load(std::sync::atomic::Ordering::Relaxed);
        if tv != tle_version {
            tle_version = tv;
            catalog = TleCatalog::load(&root.join("tle_cache.tle")).ok();
            push_status(&mut status, "TLE catalog reloaded".into());
        }

        // Gamepad -> gearbox -> rate command.
        let mut stick = (0.0, 0.0);
        let mut stick_raw = (0.0, 0.0);
        let mut joystick_name = None;
        let mut joy_buttons = 0u32;
        let mut joy_left = (0.0, 0.0);
        let mut joy_right = (0.0, 0.0);
        let mut joy_trig = (0.0, 0.0);
        let mut want_tare = false;
        let mut want_cycle_mode = false;
        let mut armed_toggle = false;
        if let Some(g) = gilrs.as_mut() {
            while let Some(ev) = g.next_event() {
                // DualShock layout (mirrors joystick_controller.process_joystick_events):
                // Cross = capture arm/save, Circle = STOP, R3 = cycle mode
                // forward, touchpad/L3 = cycle backward, L1 = feed-forward toggle.
                if let gilrs::EventType::ButtonPressed(b, _) = ev.event {
                    use gilrs::Button;
                    match b {
                        Button::East => {
                            stopped = !stopped;
                            if stopped {
                                let mut i = loop_shared.inputs.lock().unwrap();
                                i.rate_cmd = (0, 0);
                                i.setpoint = None;
                                drop(i);
                                loop_shared.commands.lock().unwrap().push_back(Command::Stop);
                                push_status(&mut status, "STOP latched (gamepad)".into());
                            } else {
                                push_status(&mut status, "STOP released (gamepad)".into());
                            }
                        }
                        Button::RightThumb | Button::LeftThumb => {
                            let cur = MODE_CYCLE.iter().position(|m| *m == mode).unwrap_or(0);
                            let next = if b == Button::RightThumb { (cur + 1) % MODE_CYCLE.len() } else { (cur + MODE_CYCLE.len() - 1) % MODE_CYCLE.len() };
                            mode = MODE_CYCLE[next];
                            stopped = false;
                            rate_limit_warned = false;
                            let mut i = loop_shared.inputs.lock().unwrap();
                            i.mode = mode;
                            i.stopped = false;
                            if !matches!(mode, Mode::Program | Mode::Handoff | Mode::Hotspot) {
                                i.setpoint = None;
                            }
                            drop(i);
                            push_status(&mut status, format!("mode -> {} (gamepad)", mode_name(mode)));
                        }
                        Button::LeftTrigger => {
                            let mut i = loop_shared.inputs.lock().unwrap();
                            let on = !(i.ff_azm_enabled && i.ff_alt_enabled);
                            i.ff_azm_enabled = on;
                            i.ff_alt_enabled = on;
                            drop(i);
                            push_status(&mut status, format!("feed-forward {} (gamepad)", if on { "ON" } else { "OFF" }));
                        }
                        Button::South => armed_toggle = true,
                        Button::North => {
                            mode = Mode::Standby;
                            let mut i = loop_shared.inputs.lock().unwrap();
                            i.mode = Mode::Standby;
                            i.rate_cmd = (0, 0);
                            i.setpoint = None;
                            drop(i);
                            loop_shared.commands.lock().unwrap().push_back(Command::GotoMount { azm_deg: cfg.offsets.0, alt_deg: cfg.offsets.1 });
                            parking = true;
                            push_status(&mut status, "PARK (gamepad)".into());
                        }
                        Button::West => want_tare = true,
                        Button::Start => want_cycle_mode = true,
                        Button::Select => {
                            let idx = (bias_fine as usize) | ((bias_frame_ac as usize) << 1);
                            let next = (idx + 1) % 4;
                            bias_fine = next & 1 == 1;
                            bias_frame_ac = next & 2 == 2;
                            persist_bias(&cfg.path, bias, bias_fine);
                            push_status(&mut status, format!("bias mode: {} / {}", if bias_fine { "fine 0.01°" } else { "coarse 0.1°" }, if bias_frame_ac { "in/cross-track" } else { "az/el" }));
                        }
                        Button::DPadUp | Button::DPadDown | Button::DPadLeft | Button::DPadRight => {
                            let step = if bias_fine { 0.01 } else { 0.1 };
                            let (dx, dy) = match b {
                                Button::DPadLeft => (-step, 0.0),
                                Button::DPadRight => (step, 0.0),
                                Button::DPadUp => (0.0, step),
                                _ => (0.0, -step),
                            };
                            if bias_frame_ac {
                                bias_it = (bias_it + dx).clamp(-3.0, 3.0);
                                bias_ct = (bias_ct + dy).clamp(-3.0, 3.0);
                                push_status(&mut status, format!("bias in-track {:+.2}° / cross {:+.2}°", bias_it, bias_ct));
                            } else {
                                bias.0 = (bias.0 + dx).clamp(-3.0, 3.0);
                                bias.1 = (bias.1 + dy).clamp(-3.0, 3.0);
                                persist_bias(&cfg.path, bias, bias_fine);
                                push_status(&mut status, format!("bias {:+.2}° / {:+.2}°", bias.0, bias.1));
                            }
                        }
                        _ => {}
                    }
                }
            }
            if let Some((_id, pad)) = g.gamepads().next() {
                joystick_name = Some(pad.name().to_string());
                use gilrs::{Axis as Ax, Button as B};
                joy_left = (pad.value(Ax::LeftStickX) as f64, pad.value(Ax::LeftStickY) as f64);
                joy_right = (pad.value(Ax::RightStickX) as f64, pad.value(Ax::RightStickY) as f64);
                let trig = |b: B| pad.button_data(b).map(|d| d.value() as f64).unwrap_or(0.0);
                joy_trig = (trig(B::LeftTrigger2), trig(B::RightTrigger2));
                for (bit, b) in [
                    (0u32, B::South), (1, B::East), (2, B::West), (3, B::North),
                    (4, B::Select), (5, B::Mode), (6, B::Start),
                    (7, B::LeftThumb), (8, B::RightThumb),
                    (9, B::LeftTrigger), (10, B::RightTrigger),
                    (11, B::DPadUp), (12, B::DPadDown), (13, B::DPadLeft), (14, B::DPadRight),
                ] {
                    if pad.is_pressed(b) {
                        joy_buttons |= 1 << bit;
                    }
                }
                let (ax, ay) = if cfg.joystick_rate_stick.eq_ignore_ascii_case("left") { (gilrs::Axis::LeftStickX, gilrs::Axis::LeftStickY) } else { (gilrs::Axis::RightStickX, gilrs::Axis::RightStickY) };
                stick_raw = (pad.value(ax) as f64, -(pad.value(ay) as f64));
                stick = ((stick_raw.0 - tare.0).clamp(-1.0, 1.0), (stick_raw.1 - tare.1).clamp(-1.0, 1.0));
            }
        }
        last_stick_raw = stick_raw;
        if want_tare {
            tare = stick_raw;
            push_status(&mut status, format!("joystick tared at {:+.2} / {:+.2} (gamepad)", tare.0, tare.1));
        }
        if want_cycle_mode {
            let cur = MOUNT_MODE_CYCLE.iter().position(|m| parse_mount_mode(m) == parse_mount_mode(&cur_mode_name)).unwrap_or(0);
            let next = MOUNT_MODE_CYCLE[(cur + 1) % MOUNT_MODE_CYCLE.len()].to_string();
            cur_mode_name = next.clone();
            cfg.mount_mode = next.clone();
            loop_shared.inputs.lock().unwrap().mount_mode = parse_mount_mode(&next);
            shared.mount_mode.store(Arc::new(next.clone()));
            persist_config_key(&cfg.path, "mount_mode", serde_json::json!(next));
            push_status(&mut status, format!("mount mode -> {next} (gamepad)"));
        }
        if armed_toggle {
            let armed = shared.cam(shared.hotspot_slot()).map(|c| c.armed).unwrap_or(false);
            let _ = tx_cam.send(if armed {
                crate::state::CamCmd::Dump { name: target.clone().unwrap_or_else(|| "manual".into()) }
            } else {
                crate::state::CamCmd::Arm
            });
            push_status(&mut status, if armed { "capture saved (gamepad)".into() } else { "capture ARMED (gamepad)".into() });
        }
        let (az_rate, el_rate) = mapper.update(stick.0, stick.1, now);
        if mode == Mode::Rate {
            // Hardware-limit gating (joystick_controller._rate_control_track):
            // an axis at its limit refuses to slew further out, but can always
            // come back in-range.
            let prev = shared.mount.load();
            let mut cmd = (az_rate, el_rate);
            let mut hit = false;
            if (cmd.0 > 0 && prev.azm >= cfg.azm_limit.1) || (cmd.0 < 0 && prev.azm <= cfg.azm_limit.0) {
                cmd.0 = 0;
                hit = true;
            }
            if (cmd.1 > 0 && prev.alt >= cfg.alt_limit.1) || (cmd.1 < 0 && prev.alt <= cfg.alt_limit.0) {
                cmd.1 = 0;
                hit = true;
            }
            if hit && !rate_limit_warned {
                push_status(&mut status, format!("axis limit — slew blocked (azm {:.1} alt {:.1})", prev.azm, prev.alt));
                rate_limit_warned = true;
            } else if !hit {
                rate_limit_warned = false;
            }
            loop_shared.inputs.lock().unwrap().rate_cmd = cmd;
        }
        // Universal STOP latch: rates forced to zero in any mode until released.
        {
            let mut i = loop_shared.inputs.lock().unwrap();
            i.stopped = stopped;
            if stopped {
                i.rate_cmd = (0, 0);
            }
        }
        // Focus motor from the triggers (R2 forward / L2 backward), rate
        // proportional to deflection; send only on change (_handle_focus_control).
        let focus_rate = if stopped {
            0
        } else {
            const DB: f64 = 0.05;
            let (l2, r2) = joy_trig;
            if r2 > DB && r2 >= l2 {
                (r2 * 9.0).round() as i32
            } else if l2 > DB {
                -((l2 * 9.0).round() as i32)
            } else {
                0
            }
        }
        .clamp(-9, 9);
        if focus_rate != focus_last_rate {
            loop_shared.commands.lock().unwrap().push_back(Command::FocusRate(focus_rate));
            focus_last_rate = focus_rate;
        }

        // PROGRAM / HANDOFF / HOTSPOT: live setpoint from the selected target
        // (HOTSPOT rides its trajectory feed-forward + star-filter rate reference).
        if matches!(mode, Mode::Program | Mode::Handoff | Mode::Hotspot) {
            let sp = target.as_ref().and_then(|sn| {
                // Non-satellite selections (body / star / DSO / aircraft) ride
                // the sky or ADS-B worker's live track, dead-reckoned.
                if let Some(tt) = shared.sky.load().target.as_ref().filter(|t| &t.key == sn) {
                    let age = (crate::sky::now_jd_tt() - tt.jd_tt) * 86400.0;
                    let el = tt.el + tt.el_rate * age;
                    return Some(Setpoint {
                        az_deg: (tt.az + tt.az_rate * age).rem_euclid(360.0),
                        el_deg: if el <= 0.0 { cfg.elevation_mask_deg } else { el },
                        ff_az_dps: if el <= 0.0 { 0.0 } else { tt.az_rate },
                        ff_el_dps: if el <= 0.0 { 0.0 } else { tt.el_rate },
                    });
                }
                if let Some(icao) = sn.strip_prefix("adsb:") {
                    let adsb = shared.adsb.load();
                    let a = adsb.aircraft.iter().find(|a| a.icao == icao)?;
                    let age = crate::sky::now_unix() - a.fit_t_unix;
                    let el = a.fit_el + a.el_rate * age;
                    return Some(Setpoint {
                        az_deg: (a.fit_az + a.az_rate * age).rem_euclid(360.0),
                        el_deg: if el <= 0.0 { cfg.elevation_mask_deg } else { el },
                        ff_az_dps: if el <= 0.0 { 0.0 } else { a.az_rate },
                        ff_el_dps: if el <= 0.0 { 0.0 } else { a.el_rate },
                    });
                }
                // Ephemeris override (Starlink public ephemerides): maneuvering
                // birds where the TLE is hours-stale. Falls through to the TLE
                // outside the ephemeris span.
                if let Some(eph) = shared.ephemerides.load().get(sn) {
                    let t = crate::sky::now_unix();
                    let jd_tt = crate::sky::now_jd_tt();
                    if let (Some(p0), Some(pm), Some(pp)) = (
                        skytracker_astro::starlink::position_at(eph, t),
                        skytracker_astro::starlink::position_at(eph, t - 0.5),
                        skytracker_astro::starlink::position_at(eph, t + 0.5),
                    ) {
                        let ctx = skytracker_astro::apparent::FrameContext::new(jd_tt);
                        let (alt, az, _) = ctx.altaz_range_of_position(&p0, &geom);
                        let (a_m, z_m, _) = ctx.altaz_range_of_position(&pm, &geom);
                        let (a_p, z_p, _) = ctx.altaz_range_of_position(&pp, &geom);
                        let el = if alt <= 0.0 { cfg.elevation_mask_deg } else { alt };
                        let (ff_az, ff_el) = if alt <= 0.0 { (0.0, 0.0) } else { (sgp4_pass::unwrap_az_diff(z_p - z_m), a_p - a_m) };
                        return Some(Setpoint { az_deg: az, el_deg: el, ff_az_dps: ff_az, ff_el_dps: ff_el });
                    }
                }
                let cat = catalog.as_ref()?;
                let sat = cat.get(sn)?;
                let jd_tt = crate::sky::now_jd_tt();
                let dt = 0.5 / 86400.0;
                let (alt, az, _) = sgp4_pass::satellite_altaz(sat, jd_tt, &geom).ok()?;
                let (a_m, z_m, _) = sgp4_pass::satellite_altaz(sat, jd_tt - dt, &geom).ok()?;
                let (a_p, z_p, _) = sgp4_pass::satellite_altaz(sat, jd_tt + dt, &geom).ok()?;
                let el = if alt <= 0.0 { cfg.elevation_mask_deg } else { alt };
                let (ff_az, ff_el) = if alt <= 0.0 {
                    (0.0, 0.0)
                } else {
                    (sgp4_pass::unwrap_az_diff(z_p - z_m), a_p - a_m)
                };
                Some(Setpoint {
                    az_deg: az,
                    el_deg: el,
                    ff_az_dps: ff_az,
                    ff_el_dps: ff_el,
                })
            });
            let mut pm_corr = (0.0, 0.0);
            let sp = sp.map(|mut s| {
                if bias != (0.0, 0.0) || bias_it != 0.0 || bias_ct != 0.0 {
                    // _apply_bias_to_target: az/el bias is a cross-el/el nudge;
                    // in/cross-track projects onto the sky-velocity direction.
                    let cos_el = s.el_deg.to_radians().cos().max(0.087);
                    let mut crossel = bias.0;
                    let mut elb = bias.1;
                    if bias_it != 0.0 || bias_ct != 0.0 {
                        let vx = s.ff_az_dps * cos_el;
                        let vy = s.ff_el_dps;
                        let norm = vx.hypot(vy);
                        if norm > 1e-6 {
                            let (ux, uy) = (vx / norm, vy / norm);
                            crossel += bias_it * ux - bias_ct * uy;
                            elb += bias_it * uy + bias_ct * ux;
                        }
                    }
                    s.az_deg = (s.az_deg + crossel / cos_el).rem_euclid(360.0);
                    s.el_deg += elb;
                }
                // Pointing-model pre-correction (control.apply_pointing_model):
                // command = desired - error(desired), first-order inverse.
                let (on, terms) = **shared.pointing.load();
                if on && parse_mount_mode(&cur_mode_name) != MountMode::Eq {
                    let (da, de) = skytracker_pointing::altaz::error(&terms, s.az_deg, s.el_deg);
                    pm_corr = (da, de);
                    s.az_deg = (s.az_deg - da).rem_euclid(360.0);
                    s.el_deg -= de;
                }
                // Eq residual model (apply_eq_pointing_model): correct in the
                // mount (HA/Dec) frame through the existing geometric transform.
                let (eq_on, eq_terms) = **shared.eq_pointing.load();
                if eq_on && parse_mount_mode(&cur_mode_name) == MountMode::Eq {
                    use skytracker_core::transforms::{mount_to_sky as m2s, sky_to_mount};
                    let (h, d) = sky_to_mount(MountMode::Eq, s.az_deg, s.el_deg, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip);
                    let (dh, dd) = skytracker_pointing::eq::error(&eq_terms, h, d, cfg.lat_deg);
                    let (az2, el2) = m2s(MountMode::Eq, h - dh, d - dd, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip);
                    pm_corr = (s.az_deg - az2, s.el_deg - el2);
                    s.az_deg = az2.rem_euclid(360.0);
                    s.el_deg = el2;
                }
                s
            });
            pm_corr_last = pm_corr;
            last_setpoint = sp.as_ref().map(|s| (s.az_deg, s.el_deg));
            if prev_target != target {
                prev_target = target.clone();
                slewing = true;
            }
            if let Some(s) = sp.as_ref() {
                let (taz, tal) = transforms::sky_to_mount(parse_mount_mode(&cfg.mount_mode), s.az_deg, s.el_deg, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip);
                let out = loop_shared.outputs.lock().unwrap().clone();
                let e_az = ((taz - out.azm + 540.0).rem_euclid(360.0) - 180.0).abs();
                let e_al = (tal - out.alt).abs();
                if (e_az > 2.0 || e_al > 2.0) && mode != Mode::Hotspot {
                    slewing = true;
                }
                if slewing && e_az < 0.7 && e_al < 0.7 {
                    slewing = false;
                    push_status(&mut status, "slew done -- tracking".into());
                }
                if slewing && last_goto.elapsed() > Duration::from_millis(1500) {
                    loop_shared.commands.lock().unwrap().push_back(Command::GotoMount {
                        azm_deg: taz + cfg.offsets.0,
                        alt_deg: tal + cfg.offsets.1,
                    });
                    last_goto = Instant::now();
                }
            }
            // While slewing the loop must not fight the goto: hold the setpoint back.
            loop_shared.inputs.lock().unwrap().setpoint = if slewing && mode != Mode::Hotspot { None } else { sp };
        } else {
            last_setpoint = None;
            slewing = false;
            prev_target = target.clone();
        }

        // Loop outputs; honour mode requests (HANDOFF -> HOTSPOT, safety -> STANDBY).
        let out = loop_shared.outputs.lock().unwrap().clone();
        if let Some(req) = out.requested_mode {
            if req != mode {
                mode = req;
                let mut i = loop_shared.inputs.lock().unwrap();
                i.mode = mode;
                if !matches!(mode, Mode::Program | Mode::Handoff | Mode::Hotspot) {
                    i.setpoint = None;
                }
                drop(i);
                push_status(&mut status, format!("loop -> {}", mode_name(mode)));
                {
                    let key = profile_key(mode);
                    if key != active_profile {
                        let live = {
                            let i = loop_shared.inputs.lock().unwrap();
                            (i.azm_gains, i.alt_gains)
                        };
                        if let Some(old_key) = active_profile {
                            profiles.insert(old_key.to_string(), live);
                            save_gain_profile(&cfg.path, old_key, live.0, live.1, None);
                        }
                        if let Some(k) = key {
                            if let Some(&(a, e)) = profiles.get(k) {
                                let mut i = loop_shared.inputs.lock().unwrap();
                                if i.azm_gains != a || i.alt_gains != e {
                                    i.azm_gains = a;
                                    i.alt_gains = e;
                                    drop(i);
                                    push_status(&mut status, format!("gains <- {k} profile"));
                                }
                            } else {
                                profiles.insert(k.to_string(), live);
                                save_gain_profile(&cfg.path, k, live.0, live.1, None);
                                push_status(&mut status, format!("PID gains: seeded new {k} profile"));
                            }
                        }
                        active_profile = key;
                    }
                }
            }
        }
        // LAUNCH override (joystick_controller launch_active): while armed, the
        // launch is the target, PROGRAM is forced, other selections are dropped.
        if let Some((lkey, _t0)) = (**shared.launch_armed.load()).clone() {
            if target.as_deref() != Some(lkey.as_str()) {
                target = Some(lkey.clone());
                shared.selected.store(Arc::new(target.clone()));
                push_status(&mut status, "LAUNCH: overriding target -> launch trajectory".into());
            }
            if mode != Mode::Program {
                mode = Mode::Program;
                stopped = false;
                let mut i = loop_shared.inputs.lock().unwrap();
                i.mode = mode;
                i.stopped = false;
                drop(i);
                push_status(&mut status, "LAUNCH: mode -> PROGRAM".into());
            }
        }

        // Park servicing (_service_park): wrap-aware convergence in the raw
        // encoder frame, gentle goto re-issue, 1° tolerance, 90 s timeout.
        if parking {
            if park_state.is_none() {
                park_state = Some((now, now));
            }
            if let Some((start, last_cmd)) = park_state.as_mut() {
                let err_a = (cfg.offsets.0 - out.azm_raw + 180.0).rem_euclid(360.0) - 180.0;
                let err_e = (cfg.offsets.1 - out.alt_raw + 180.0).rem_euclid(360.0) - 180.0;
                if err_a.abs() <= 1.0 && err_e.abs() <= 1.0 {
                    parking = false;
                    park_state = None;
                    push_status(&mut status, "Park complete".into());
                } else if now - *start > 90.0 {
                    parking = false;
                    park_state = None;
                    push_status(&mut status, "Park TIMED OUT after 90s — check the mount".into());
                } else if now - *last_cmd >= 2.0 {
                    loop_shared.commands.lock().unwrap().push_back(Command::GotoMount { azm_deg: cfg.offsets.0, alt_deg: cfg.offsets.1 });
                    *last_cmd = now;
                }
            }
        } else {
            park_state = None;
        }
        for m in &out.status_msgs {
            if status.last() != Some(m) {
                push_status(&mut status, m.clone());
            }
        }

        // Autotune rides on the tracking error.
        let mut autotune_text = None;
        if tuner_active {
            let tracking = matches!(mode, Mode::Program | Mode::Handoff | Mode::Hotspot);
            tuner.update(now, tracking, out.azm_error, out.alt_error);
            let g = tuner.applied_gains();
            let mut i = loop_shared.inputs.lock().unwrap();
            i.azm_gains = (g[0][0], g[0][1], g[0][2]);
            i.alt_gains = (g[1][0], g[1][1], g[1][2]);
            drop(i);
            for m in tuner.take_messages() {
                push_status(&mut status, m);
            }
            autotune_text = Some(tuner.status_text(now));
            if !tuner.active {
                tuner_active = false;
            }
        }

        let gains = {
            let i = loop_shared.inputs.lock().unwrap();
            [[i.azm_gains.0, i.azm_gains.1, i.azm_gains.2], [i.alt_gains.0, i.alt_gains.1, i.alt_gains.2]]
        };

        let live_ff = {
            let i = loop_shared.inputs.lock().unwrap();
            (i.ff_azm_enabled, i.ff_alt_enabled, i.lead_time_sec, i.hotspot.star_filter)
        };
        let (sky_az, sky_el) = mount_to_sky(&cfg, out.azm, out.alt);
        shared.mount.store(Arc::new(MountSnapshot {
            az: sky_az,
            el: sky_el,
            azm: out.azm,
            alt: out.alt,
            mode: mode_name(mode).to_string(),
            rate_cmd: if mode == Mode::Rate { (az_rate, el_rate) } else { (out.azm_rate_cmd, out.alt_rate_cmd) },
            az_error: out.azm_error,
            el_error: out.alt_error,
            actual_hz: out.actual_hz,
            gear_ceiling: mapper.ceiling,
            joystick: joystick_name,
            stick,
            status: status.clone(),
            transport: transport.clone(),
            connected: !out.loop_dead && out.cycle_count > 0,
            target: target.clone(),
            setpoint: last_setpoint,
            hotspot_acquired: out.hotspot_acquired,
            hotspot_status: out.hotspot_status.clone(),
            hotspot_snr: out.hotspot_snr,
            hotspot_centroid: out.hotspot_centroid,
            handoff_count: out.handoff_detection_count,
            gains,
            autotune: autotune_text,
            loop_dead: out.loop_dead,
            bias,
            bias_fine,
            parking,
            mount_mode: cur_mode_name.clone(),
            pointing_model_on: shared.pointing.load().0,
            pointing_corr: pm_corr_last,
            joystick_tare: tare,
            stopped,
            focus_rate: focus_last_rate,
            focus_pos: (out.focus_frac * 16_777_216.0).round() as i64,
            bias_frame: if bias_frame_ac { "alongcross".into() } else { "azel".into() },
            bias_it,
            bias_ct,
            ff_azm: live_ff.0,
            ff_alt: live_ff.1,
            lead_time_s: live_ff.2,
            star_filter: live_ff.3,
            joy_buttons,
            joy_left,
            joy_right,
            joy_triggers: joy_trig,
            gear_base: cfg.joy_rate_base_ceiling.clamp(1, 9),
            gear_max: 9,
        }));

        let sleep = Duration::from_millis(33).saturating_sub(t0.elapsed());
        std::thread::sleep(sleep);
    }
}
