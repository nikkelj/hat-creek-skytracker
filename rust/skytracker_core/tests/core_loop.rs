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

#[test]
fn consecutive_poll_faults_stop_motion() {
    // A transport whose reads always time out (short read) but whose writes
    // are observable: after MAX_CONSECUTIVE_FAULTS the loop must command a
    // stop on both axes and surface a status message -- BEFORE shutdown.
    use skytracker_core::sim::Transport;
    use std::sync::Mutex;

    struct FailingRecorder {
        written: Arc<Mutex<Vec<u8>>>,
    }
    impl Transport for FailingRecorder {
        fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
            self.written.lock().unwrap().extend_from_slice(data);
            Ok(())
        }
        fn read(&mut self, _n: usize) -> Vec<u8> {
            Vec::new() // every poll short-reads -> comm fault
        }
    }

    let written = Arc::new(Mutex::new(Vec::new()));
    let mount = skytracker_core::sim::Mount::new(FailingRecorder {
        written: Arc::clone(&written),
    });
    let shared = Shared::new(program_inputs(50.0, 30.0));
    let mut cl = CoreLoop::spawn(mount, Arc::clone(&shared), 50.0);

    // Give the loop time for well over MAX_CONSECUTIVE_FAULTS cycles, then
    // inspect the wire traffic BEFORE stop() (which also sends a stop).
    thread::sleep(Duration::from_millis(400));
    let wire: Vec<u8> = written.lock().unwrap().clone();
    let msgs = shared.outputs.lock().unwrap().status_msgs.clone();
    let fresh = shared.outputs.lock().unwrap().fresh;
    cl.stop();

    // MC_MOVE_POS rate 0 (stop): 50 02 <target> 24 00 00 00 00
    let has_stop = |t: u8| {
        wire.windows(8)
            .any(|c| c == [0x50, 0x02, t, 0x24, 0x00, 0x00, 0x00, 0x00])
    };
    assert!(has_stop(0x10), "no AZM stop commanded under sustained faults");
    assert!(has_stop(0x11), "no ALT stop commanded under sustained faults");
    assert!(
        msgs.iter().any(|m| m.contains("consecutive comm faults")),
        "no fault status message: {msgs:?}"
    );
    assert!(!fresh, "snapshot must be marked stale under comm faults");
}

#[test]
fn west_to_east_converges_over_the_zenith_not_the_long_way() {
    // Mount west (270, 45), target east (az=90, el=45), Passthrough: the
    // closed loop must converge to the FLIPPED solution (270, 135) with the
    // azimuth axis essentially parked -- not drag the azimuth 180 deg around.
    let shared = Shared::new(program_inputs(90.0, 45.0));
    let mut mount = Mount::new(LoopbackTransport::new(SimResponder::new_manual(270.0, 45.0)));
    let mut state = LoopState::new();

    let mut now = 0.0;
    let mut az_excursion: f64 = 0.0;
    for _ in 0..500 {
        now += 0.1;
        mount.io.responder.advance_time(0.1);
        run_cycle(&mut mount, &mut state, &shared, now);
        let az = mount.io.responder.az_true_deg;
        let d = ((az - 270.0 + 180.0).rem_euclid(360.0) - 180.0).abs();
        az_excursion = az_excursion.max(d);
    }

    let o = shared.outputs.lock().unwrap();
    assert!((o.azm - 270.0).abs() < 2.0, "azm should stay ~270, got {}", o.azm);
    assert!((o.alt - 135.0).abs() < 2.0, "alt should arrive at 135, got {}", o.alt);
    assert!(
        az_excursion < 10.0,
        "azimuth slewed {az_excursion} deg -- it took the long way around"
    );
}
