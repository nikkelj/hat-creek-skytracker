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

fn spawn_loop(cfg: &Config, loop_shared: &Arc<LoopShared>) -> (CoreLoop, String, Vec<String>) {
    let mut notes = Vec::new();
    let hz = if cfg.loop_hz > 0.0 { cfg.loop_hz } else { 15.0 };
    if cfg.mount_transport.eq_ignore_ascii_case("serial") {
        #[cfg(feature = "serial")]
        {
            match skytracker_core::serial::SerialTransport::open(&cfg.serial_port, cfg.serial_baud, 1500) {
                Ok(port) => {
                    notes.push(format!("serial {} @ {}", cfg.serial_port, cfg.serial_baud));
                    let mount = Mount::new(port);
                    return (CoreLoop::spawn(mount, loop_shared.clone(), hz), format!("serial {}", cfg.serial_port), notes);
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
    let mount = Mount::new(LoopbackTransport::new(SimResponder::new_wall(azm + cfg.offsets.0, alt + cfg.offsets.1)));
    (CoreLoop::spawn(mount, loop_shared.clone(), hz), "sim".into(), notes)
}

fn run(shared: Arc<Shared>, rx: Receiver<MountCmd>, root: std::path::PathBuf, tx_cam: crossbeam_channel::Sender<crate::state::CamCmd>) {
    let cfg = shared.config.clone();
    let loop_shared = shared.core.clone();
    let (_core, transport, notes) = spawn_loop(&cfg, &loop_shared);

    // Gamepad.
    let mut gilrs = gilrs::Gilrs::new().ok();
    let mut mapper = AdaptiveRateMapper::new(5, 9, 0.8);
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
        status.push(s);
        if status.len() > 8 {
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
    // Operator bias (on-sky cross-el / el, deg) and its D-pad step.
    let mut bias = (0.0f64, 0.0f64);
    let mut bias_fine = false;
    let mut parking = false;
    let apply_profile = |mode: Mode, loop_shared: &Arc<LoopShared>, status: &mut Vec<String>| {
        let key = match mode {
            Mode::Program | Mode::Handoff => "PROGRAM",
            Mode::Hotspot => "HOTSPOT",
            _ => return,
        };
        if let Some((a, e)) = cfg.pid_profiles.get(key) {
            let mut i = loop_shared.inputs.lock().unwrap();
            if i.azm_gains != *a || i.alt_gains != *e {
                i.azm_gains = *a;
                i.alt_gains = *e;
                drop(i);
                status.push(format!("gains <- {key} profile"));
                if status.len() > 8 {
                    status.remove(0);
                }
            }
        }
    };

    loop {
        let t0 = Instant::now();
        let now = epoch.elapsed().as_secs_f64();

        // Commands from the UI.
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                MountCmd::SetMode(m) => {
                    mode = match m.as_str() {
                        "RATE" => Mode::Rate,
                        "PROGRAM" => Mode::Program,
                        "HANDOFF" => Mode::Handoff,
                        "HOTSPOT" => Mode::Hotspot,
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
                    apply_profile(mode, &loop_shared, &mut status);
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
                }
                MountCmd::Stop => {
                    mode = Mode::Standby;
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.mode = Mode::Standby;
                    i.rate_cmd = (0, 0);
                    i.setpoint = None;
                    drop(i);
                    loop_shared.commands.lock().unwrap().push_back(Command::Stop);
                    push_status(&mut status, "STOP".into());
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
                    bias.0 = (bias.0 + daz).clamp(-3.0, 3.0);
                    bias.1 = (bias.1 + del).clamp(-3.0, 3.0);
                }
                MountCmd::BiasReset => bias = (0.0, 0.0),
                MountCmd::BiasFine(f) => bias_fine = f,
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

        // Reload the satellite catalog after a TLE refresh.
        let tv = shared.tle_version.load(std::sync::atomic::Ordering::Relaxed);
        if tv != tle_version {
            tle_version = tv;
            catalog = TleCatalog::load(&root.join("tle_cache.tle")).ok();
            push_status(&mut status, "TLE catalog reloaded".into());
        }

        // Gamepad -> gearbox -> rate command.
        let mut stick = (0.0, 0.0);
        let mut joystick_name = None;
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
                            mode = Mode::Standby;
                            let mut i = loop_shared.inputs.lock().unwrap();
                            i.mode = Mode::Standby;
                            i.rate_cmd = (0, 0);
                            i.setpoint = None;
                            drop(i);
                            loop_shared.commands.lock().unwrap().push_back(Command::Stop);
                            push_status(&mut status, "STOP (gamepad)".into());
                        }
                        Button::RightThumb | Button::LeftThumb => {
                            let cur = MODE_CYCLE.iter().position(|m| *m == mode).unwrap_or(0);
                            let next = if b == Button::RightThumb { (cur + 1) % MODE_CYCLE.len() } else { (cur + MODE_CYCLE.len() - 1) % MODE_CYCLE.len() };
                            mode = MODE_CYCLE[next];
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
                        Button::Select => {
                            bias_fine = !bias_fine;
                            push_status(&mut status, format!("bias step {}", if bias_fine { "fine 0.01°" } else { "coarse 0.1°" }));
                        }
                        Button::DPadUp | Button::DPadDown | Button::DPadLeft | Button::DPadRight => {
                            let step = if bias_fine { 0.01 } else { 0.1 };
                            let (dx, dy) = match b {
                                Button::DPadLeft => (-step, 0.0),
                                Button::DPadRight => (step, 0.0),
                                Button::DPadUp => (0.0, step),
                                _ => (0.0, -step),
                            };
                            bias.0 = (bias.0 + dx).clamp(-3.0, 3.0);
                            bias.1 = (bias.1 + dy).clamp(-3.0, 3.0);
                            push_status(&mut status, format!("bias {:+.2}° / {:+.2}°", bias.0, bias.1));
                        }
                        _ => {}
                    }
                }
            }
            if let Some((_id, pad)) = g.gamepads().next() {
                joystick_name = Some(pad.name().to_string());
                stick = (
                    pad.value(gilrs::Axis::LeftStickX) as f64,
                    -(pad.value(gilrs::Axis::LeftStickY) as f64),
                );
            }
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
            loop_shared.inputs.lock().unwrap().rate_cmd = (az_rate, el_rate);
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
                        ff_az_dps: tt.az_rate,
                        ff_el_dps: tt.el_rate,
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
                        ff_az_dps: a.az_rate,
                        ff_el_dps: a.el_rate,
                    });
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
            let sp = sp.map(|mut s| {
                if bias != (0.0, 0.0) {
                    let cos_el = s.el_deg.to_radians().cos().max(0.087);
                    s.az_deg = (s.az_deg + bias.0 / cos_el).rem_euclid(360.0);
                    s.el_deg += bias.1;
                }
                s
            });
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
                apply_profile(mode, &loop_shared, &mut status);
            }
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
        }));

        let sleep = Duration::from_millis(33).saturating_sub(t0.elapsed());
        std::thread::sleep(sleep);
    }
}
