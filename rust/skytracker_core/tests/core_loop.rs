//! Closed-loop integration tests for the control loop driving the in-memory
//! byte-level mount sim (Mount<LoopbackTransport> + SimResponder). Run:
//!   cargo test --test core_loop

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use skytracker_core::controller::{Inputs, LoopState, Mode, Setpoint};
use skytracker_core::core_loop::{run_cycle, CoreLoop, Shared};
use skytracker_core::sim::{LoopbackTransport, Mount, SimResponder};
use skytracker_core::transforms::MountMode;

fn program_inputs(az: f64, el: f64) -> Inputs {
    let mut i = Inputs {
        connected: true,
        ..Default::default()
    };
    i.mode = Mode::Program;
    i.mount_mode = MountMode::Passthrough;
    i.azm_gains = (0.025, 0.0, 0.0);
    i.alt_gains = (0.025, 0.0, 0.0);
    i.setpoint = Some(Setpoint {
        az_deg: az,
        el_deg: el,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    i
}

#[test]
fn program_closed_loop_converges() {
    let shared = Shared::new(program_inputs(50.0, 30.0));
    let mut mount = Mount::new(LoopbackTransport::new(SimResponder::new_manual(0.0, 0.0)));
    let mut state = LoopState::new();

    let mut now = 0.0;
    for _ in 0..400 {
        now += 0.1;
        mount.io.responder.advance_time(0.1); // integrate the previous rate
        run_cycle(&mut mount, &mut state, &shared, now);
    }

    let o = shared.outputs.lock().unwrap();
    assert!((o.azm - 50.0).abs() < 2.0, "azm converged to {}", o.azm);
    assert!((o.alt - 30.0).abs() < 2.0, "alt converged to {}", o.alt);
    assert!(o.fresh);
}

#[test]
fn program_tracks_a_moving_setpoint() {
    // Ramp the azimuth setpoint and confirm the loop follows within a small lag.
    let shared = Shared::new(program_inputs(0.0, 20.0));
    let mut mount = Mount::new(LoopbackTransport::new(SimResponder::new_manual(0.0, 0.0)));
    let mut state = LoopState::new();

    let mut now = 0.0;
    let mut target_az = 0.0;
    for _ in 0..600 {
        now += 0.1;
        target_az += 0.05; // 0.5 deg/s ramp
        shared.inputs.lock().unwrap().setpoint = Some(Setpoint {
            az_deg: target_az,
            el_deg: 20.0,
            ff_az_dps: 0.0,
            ff_el_dps: 0.0,
        });
        mount.io.responder.advance_time(0.1);
        run_cycle(&mut mount, &mut state, &shared, now);
    }

    let o = shared.outputs.lock().unwrap();
    // After a 30 deg ramp the loop should be tracking within a few degrees.
    assert!((o.azm - target_az).abs() < 3.0, "azm {} vs target {}", o.azm, target_az);
}

#[test]
fn focus_axis_moves_and_reads_back_independently() {
    // A focus move (MC_MOVE on Targets::FOCUS) integrates a separate focus axis
    // and must NOT disturb az/el — the regression this guards is FOCUS being
    // folded into the azimuth branch.
    use skytracker_core::protocol::targets::{ALT, AZM, FOCUS};
    let mut mount = Mount::new(LoopbackTransport::new(SimResponder::new_manual(0.0, 0.0)));

    mount.hc_slew_fixed(FOCUS, 9).unwrap(); // max forward focus rate (10 deg/s)
    mount.io.responder.advance_time(1.0);

    let focus = mount.hc_get_position(FOCUS).unwrap() * 360.0;
    let az = mount.hc_get_position(AZM).unwrap() * 360.0;
    let alt = mount.hc_get_position(ALT).unwrap() * 360.0;
    // Tolerance is the 24-bit encoder LSB (360/2^24 ≈ 2.1e-5 deg) from the
    // pack/unpack round-trip, not a physics error.
    assert!((focus - 10.0).abs() < 1e-3, "focus advanced to {}", focus);
    assert!(az.abs() < 1e-6, "azimuth must be undisturbed, got {}", az);
    assert!(alt.abs() < 1e-6, "altitude must be undisturbed, got {}", alt);

    // Stopping the focus rate freezes the position.
    mount.hc_slew_fixed(FOCUS, 0).unwrap();
    mount.io.responder.advance_time(1.0);
    let focus2 = mount.hc_get_position(FOCUS).unwrap() * 360.0;
    assert!((focus2 - 10.0).abs() < 1e-3, "focus held at {}", focus2);
}

#[test]
fn poll_fault_skips_cycle_safely() {
    // A short-read transport makes every poll fail; the loop must skip cycles
    // and never panic or publish a fresh position.
    use skytracker_core::sim::{Mount as M, RecordingTransport};
    let shared = Shared::new(program_inputs(50.0, 30.0));
    let mut mount = M::new(RecordingTransport::new(b"")); // empty -> short read
    let mut state = LoopState::new();
    for i in 0..10 {
        run_cycle(&mut mount, &mut state, &shared, i as f64 * 0.1);
    }
    assert!(!shared.outputs.lock().unwrap().fresh);
}

#[test]
fn threaded_loop_spawns_runs_and_stops() {
    let mut i = Inputs {
        connected: true,
        ..Default::default()
    };
    i.mode = Mode::Rate;
    i.rate_cmd = (9, 0); // max +azm rate
    let shared = Shared::new(i);
    let mount = Mount::new(LoopbackTransport::new(SimResponder::new_wall(0.0, 0.0)));

    let mut cl = CoreLoop::spawn(mount, Arc::clone(&shared), 50.0);

    // Wait for the loop to publish a fresh snapshot.
    let mut fresh = false;
    for _ in 0..100 {
        thread::sleep(Duration::from_millis(20));
        if shared.outputs.lock().unwrap().fresh {
            fresh = true;
            break;
        }
    }
    assert!(fresh, "loop never published a snapshot");

    thread::sleep(Duration::from_millis(200));
    let az = shared.outputs.lock().unwrap().azm;
    cl.stop();
    assert!(az > 0.0, "azm should have advanced under a +rate command, got {az}");
}
