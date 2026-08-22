//! Mount worker: the Rust core loop over the byte-level simulated mount,
//! fed by the gamepad through the adaptive rate gearbox (RATE mode) or by
//! live satellite setpoints from the sky snapshot (PROGRAM mode). UI talks
//! to it only through MountCmd; it publishes MountSnapshot every cycle.
//! Swapping in the serial transport at rig time changes only the Mount.

use crate::state::{MountCmd, MountSnapshot, Shared};
use crossbeam_channel::Receiver;
use skytracker_astro::sgp4_pass::{self, Observer};
use skytracker_astro::tle::TleCatalog;
use skytracker_core::controller::{Inputs, Mode, Setpoint};
use skytracker_core::core_loop::{Command, CoreLoop, Shared as LoopShared};
use skytracker_core::rate::AdaptiveRateMapper;
use skytracker_core::sim::{LoopbackTransport, Mount, SimResponder};
use skytracker_core::transforms::MountMode;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn spawn(shared: Arc<Shared>, rx: Receiver<MountCmd>, repo_root: std::path::PathBuf) {
    std::thread::Builder::new()
        .name("mount-worker".into())
        .spawn(move || run(shared, rx, repo_root))
        .expect("spawn mount worker");
}

fn parse_mount_mode(s: &str) -> MountMode {
    match s.to_ascii_lowercase().replace('-', "_").as_str() {
        "altaz_side" | "altazside" => MountMode::AltAzSide,
        "passthrough" => MountMode::Passthrough,
        "eq" => MountMode::Eq,
        _ => MountMode::AltAz,
    }
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Standby => "STANDBY",
        Mode::Rate => "RATE",
        Mode::Program => "PROGRAM",
        Mode::Handoff => "HANDOFF",
        Mode::Hotspot => "HOTSPOT",
        Mode::Mti => "MTI",
    }
}

fn run(shared: Arc<Shared>, rx: Receiver<MountCmd>, root: std::path::PathBuf) {
    let cfg = shared.config.clone();
    let mut inputs = Inputs::default();
    inputs.connected = true;
    inputs.mode = Mode::Standby;
    inputs.azm_gains = cfg.azm_gains;
    inputs.alt_gains = cfg.alt_gains;
    inputs.mount_mode = parse_mount_mode(&cfg.mount_mode);
    inputs.alignment_az = cfg.alignment_az;
    inputs.alignment_el = cfg.alignment_el;
    inputs.altaz_side_flip = cfg.altaz_side_flip;
    inputs.ff_azm_enabled = true;
    inputs.ff_alt_enabled = true;
    let loop_shared: Arc<LoopShared> = LoopShared::new(inputs);

    // Wall-clock byte-level sim mount, parked at az 180 / el 45.
    let mount = Mount::new(LoopbackTransport::new(SimResponder::new_wall(180.0, 45.0)));
    let _core = CoreLoop::spawn(mount, loop_shared.clone(), 15.0);

    // Gamepad.
    let mut gilrs = gilrs::Gilrs::new().ok();
    let mut mapper = AdaptiveRateMapper::new(5, 9, 0.8);
    let epoch = Instant::now();

    // Satellite catalog for PROGRAM setpoints (cheap, shares the TLE file).
    let catalog = TleCatalog::load(&root.join("tle_cache.tle")).ok();
    let observer = Observer {
        lat_deg: cfg.lat_deg,
        lon_deg: cfg.lon_deg,
        elevation_m: cfg.alt_m,
    };
    let geom = observer.geometry();

    let mut mode = Mode::Standby;
    let mut target: Option<String> = None;
    let mut status: Vec<String> = Vec::new();

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
                    if mode != Mode::Program {
                        i.setpoint = None;
                    }
                    status.push(format!("mode -> {}", mode_name(mode)));
                    if status.len() > 6 {
                        status.remove(0);
                    }
                }
                MountCmd::SelectTarget(t) => {
                    target = t;
                    status.push(match &target {
                        Some(s) => format!("target {s}"),
                        None => "target cleared".into(),
                    });
                }
                MountCmd::Stop => {
                    mode = Mode::Standby;
                    let mut i = loop_shared.inputs.lock().unwrap();
                    i.mode = Mode::Standby;
                    i.rate_cmd = (0, 0);
                    i.setpoint = None;
                    loop_shared.commands.lock().unwrap().push_back(Command::Stop);
                    status.push("STOP".into());
                }
                MountCmd::Goto { az, el } => {
                    loop_shared
                        .commands
                        .lock()
                        .unwrap()
                        .push_back(Command::GotoMount { azm_deg: az, alt_deg: el });
                    status.push(format!("goto {az:.1}/{el:.1}"));
                }
            }
        }

        // Gamepad -> gearbox -> rate command.
        let mut stick = (0.0, 0.0);
        let mut joystick_name = None;
        if let Some(g) = gilrs.as_mut() {
            while let Some(_ev) = g.next_event() {}
            if let Some((_id, pad)) = g.gamepads().next() {
                joystick_name = Some(pad.name().to_string());
                stick = (
                    pad.value(gilrs::Axis::LeftStickX) as f64,
                    -(pad.value(gilrs::Axis::LeftStickY) as f64),
                );
            }
        }
        let (az_rate, el_rate) = mapper.update(stick.0, stick.1, now);
        if mode == Mode::Rate {
            loop_shared.inputs.lock().unwrap().rate_cmd = (az_rate, el_rate);
        }

        // PROGRAM: live setpoint from the selected satellite.
        if mode == Mode::Program {
            let sp = target.as_ref().and_then(|sn| {
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
            loop_shared.inputs.lock().unwrap().setpoint = sp;
        }

        // Publish.
        let out = loop_shared.outputs.lock().unwrap().clone();
        shared.mount.store(Arc::new(MountSnapshot {
            az: out.azm,
            el: out.alt,
            mode: mode_name(mode).to_string(),
            rate_cmd: if mode == Mode::Rate { (az_rate, el_rate) } else { (out.azm_rate_cmd, out.alt_rate_cmd) },
            az_error: out.azm_error,
            el_error: out.alt_error,
            actual_hz: out.actual_hz,
            gear_ceiling: mapper.ceiling,
            joystick: joystick_name,
            stick,
            status: status.clone(),
        }));

        let sleep = Duration::from_millis(33).saturating_sub(t0.elapsed());
        std::thread::sleep(sleep);
    }
}
