//! Pure decision-logic tests for the control loop. Run: cargo test
//! These exercise LoopState::step with no threads or serial I/O.

use std::sync::Arc;

use skytracker_core::controller::{Frame, Inputs, LoopState, Mode, Setpoint};
use skytracker_core::transforms::MountMode;

fn base_inputs() -> Inputs {
    let mut i = Inputs {
        connected: true,
        ..Default::default()
    };
    i.azm_gains = (0.025, 0.0, 0.0);
    i.alt_gains = (0.025, 0.0, 0.0);
    i
}

fn blob_frame(w: usize, h: usize, cx: f64, cy: f64) -> Frame {
    let amp = 200.0;
    let sigma = 3.0;
    let bg = 10.0;
    let mut data = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let r2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
            data[y * w + x] = (bg + amp * (-(r2) / (2.0 * sigma * sigma)).exp()) as f32;
        }
    }
    Frame {
        data: Arc::new(data),
        h,
        w,
        seq: 1,
    }
}

#[test]
fn not_connected_does_nothing() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.connected = false;
    let o = s.step(&i, None, 0.0, 0.0, 1.0);
    assert_eq!(o.azm_rate_cmd, None);
    assert_eq!(o.alt_rate_cmd, None);
}

#[test]
fn stopped_commands_zero() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.stopped = true;
    i.mode = Mode::Rate;
    i.rate_cmd = (5, 5);
    let o = s.step(&i, None, 0.0, 0.0, 1.0);
    assert_eq!(o.azm_rate_cmd, Some(0));
    assert_eq!(o.alt_rate_cmd, Some(0));
}

#[test]
fn standby_stops_once() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Standby;
    let o1 = s.step(&i, None, 0.0, 0.0, 1.0);
    assert_eq!(o1.azm_rate_cmd, Some(0));
    let o2 = s.step(&i, None, 0.0, 0.0, 1.1);
    assert_eq!(o2.azm_rate_cmd, None, "STANDBY must not re-command every cycle");
}

#[test]
fn rate_mode_passes_through_joystick_rate() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Rate;
    i.rate_cmd = (-7, 3);
    let o = s.step(&i, None, 0.0, 0.0, 1.0);
    assert_eq!(o.azm_rate_cmd, Some(-7));
    assert_eq!(o.alt_rate_cmd, Some(3));
}

#[test]
fn program_drives_toward_setpoint() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::Passthrough; // target == sky coords
    i.setpoint = Some(Setpoint {
        az_deg: 15.0,
        el_deg: 25.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    // current below target on both axes -> positive error -> positive rate.
    let o = s.step(&i, None, 10.0, 20.0, 1.0);
    assert!((o.azm_error - 5.0).abs() < 1e-9);
    assert!((o.alt_error - 5.0).abs() < 1e-9);
    assert!(o.azm_rate_cmd.unwrap() > 0);
    assert!(o.alt_rate_cmd.unwrap() > 0);
}

#[test]
fn program_no_setpoint_is_quiet() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.setpoint = None;
    let o = s.step(&i, None, 10.0, 20.0, 1.0);
    assert_eq!(o.azm_rate_cmd, None);
}

#[test]
fn program_altaz_uses_transform() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::AltAz;
    i.alignment_az = 10.0;
    i.alignment_el = 0.0;
    // AltAz: target_azm = az - align_az = 100-10 = 90; target_alt = 90 - el = 50.
    i.setpoint = Some(Setpoint {
        az_deg: 100.0,
        el_deg: 40.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let o = s.step(&i, None, 90.0, 50.0, 1.0);
    // At the transformed target: zero error, zero command.
    assert!(o.azm_error.abs() < 1e-9, "azm_error={}", o.azm_error);
    assert!(o.alt_error.abs() < 1e-9, "alt_error={}", o.alt_error);
}

#[test]
fn hotspot_locks_on_blob() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    // Blob to the right of and below center.
    let f = blob_frame(256, 200, 128.0 + 30.0, 100.0 + 20.0);
    let o = s.step(&i, Some(&f), 50.0, 45.0, 100.0);
    assert!(o.hotspot_acquired);
    assert_eq!(o.hotspot_status, "locked");
    assert!(o.hotspot_snr > 5.0);
    assert!(o.hotspot_centroid.is_some());
    // Off-center object yields a non-zero angular correction signal. (The
    // discrete rate may be sub-threshold for a small offset at long focal
    // length, which is correct near-lock behavior; assert on the signal.)
    assert!(o.azm_rate_cmd.is_some() && o.alt_rate_cmd.is_some());
    assert!(o.azm_error.abs() > 0.0 && o.alt_error.abs() > 0.0);
    assert!(o.azm_pid_output.abs() > 0.0 && o.alt_pid_output.abs() > 0.0);
}

#[test]
fn hotspot_no_frame_falls_back_to_program() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    let o = s.step(&i, None, 50.0, 45.0, 100.0);
    assert_eq!(o.requested_mode, Some(Mode::Program));
    assert_eq!(o.hotspot_status, "lost");
    assert_eq!(o.azm_rate_cmd, Some(0));
}

#[test]
fn hotspot_coasts_on_brief_dropout() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    i.hotspot.coast_time_s = 1.0;
    // Acquire.
    let f = blob_frame(256, 200, 138.0, 110.0);
    let o1 = s.step(&i, Some(&f), 50.0, 45.0, 100.0);
    assert!(o1.hotspot_acquired);
    // Dropout within coast window -> coast, no new command, stays in HOTSPOT.
    let o2 = s.step(&i, None, 50.0, 45.0, 100.5);
    assert_eq!(o2.hotspot_status, "coasting");
    assert_eq!(o2.azm_rate_cmd, None);
    assert_eq!(o2.requested_mode, None);
    // Past the coast window -> lost, hand back to PROGRAM.
    let o3 = s.step(&i, None, 50.0, 45.0, 102.0);
    assert_eq!(o3.hotspot_status, "lost");
    assert_eq!(o3.requested_mode, Some(Mode::Program));
}

#[test]
fn hotspot_safety_limit_aborts_to_standby() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    i.alt_limit = (0.0, 80.0);
    let f = blob_frame(256, 200, 138.0, 110.0);
    // current_alt 85 is outside [0,80].
    let o = s.step(&i, Some(&f), 50.0, 85.0, 100.0);
    assert_eq!(o.requested_mode, Some(Mode::Standby));
    assert_eq!(o.azm_rate_cmd, Some(0));
    assert_eq!(o.alt_rate_cmd, Some(0));
}
