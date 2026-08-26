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
        time: 0.0,
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
fn hotspot_no_frame_holds_then_falls_back() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    i.hotspot.coast_time_s = 1.0; // acquisition grace = max(coast, 1.0) = 1.0s
    // First frameless cycle on entry: hold in the acquisition grace rather than
    // bailing immediately (the async frame push can lag the mode change). Leave
    // the last slew running (no command).
    let o1 = s.step(&i, None, 50.0, 45.0, 100.0);
    assert_eq!(o1.requested_mode, None);
    assert_eq!(o1.hotspot_status, "acquiring");
    assert_eq!(o1.azm_rate_cmd, None);
    // Past the grace window with still no detection -> fall back to PROGRAM.
    let o2 = s.step(&i, None, 50.0, 45.0, 101.5);
    assert_eq!(o2.requested_mode, Some(Mode::Program));
    assert_eq!(o2.hotspot_status, "lost");
    assert_eq!(o2.azm_rate_cmd, Some(0));
}

#[test]
fn handoff_program_tracks_and_hands_off_after_n_detections() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Handoff;
    i.mount_mode = MountMode::Passthrough;
    i.handoff_min_frames = 3;
    i.setpoint = Some(Setpoint {
        az_deg: 15.0,
        el_deg: 25.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    // Each cycle: program-tracks (commands a rate toward the setpoint) AND counts
    // a detection on each FRESH frame (stale frames neither count nor reset).
    // Hand-off to HOTSPOT fires on the 3rd consecutive fresh detection.
    let mut f = blob_frame(256, 200, 138.0, 110.0);
    f.seq = 1;
    let o1 = s.step(&i, Some(&f), 10.0, 20.0, 1.0);
    assert!(o1.azm_rate_cmd.unwrap() > 0, "handoff must keep program-tracking");
    assert_eq!(o1.hotspot_status, "detecting");
    assert_eq!(o1.requested_mode, None);
    // A STALE repeat of the same frame is not new information: no count change.
    let stale = s.step(&i, Some(&f), 10.0, 20.0, 1.05);
    assert_eq!(stale.requested_mode, None);
    assert_eq!(stale.handoff_detection_count, 1);
    f.seq = 2;
    let o2 = s.step(&i, Some(&f), 10.0, 20.0, 1.1);
    assert_eq!(o2.requested_mode, None);
    f.seq = 3;
    let o3 = s.step(&i, Some(&f), 10.0, 20.0, 1.2);
    assert_eq!(o3.requested_mode, Some(Mode::Hotspot));
}

#[test]
fn handoff_resets_count_on_missed_detection() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Handoff;
    i.mount_mode = MountMode::Passthrough;
    i.handoff_min_frames = 2;
    i.setpoint = Some(Setpoint {
        az_deg: 15.0,
        el_deg: 25.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let mut f = blob_frame(256, 200, 138.0, 110.0);
    f.seq = 1;
    let _ = s.step(&i, Some(&f), 10.0, 20.0, 1.0); // count 1
    let miss = s.step(&i, None, 10.0, 20.0, 1.1); // no frame -> reset
    assert_eq!(miss.hotspot_status, "program");
    assert_eq!(miss.requested_mode, None);
    // Needs two fresh consecutive detections again before handing off.
    f.seq = 2;
    let a = s.step(&i, Some(&f), 10.0, 20.0, 1.2);
    assert_eq!(a.requested_mode, None);
    f.seq = 3;
    let b = s.step(&i, Some(&f), 10.0, 20.0, 1.3);
    assert_eq!(b.requested_mode, Some(Mode::Hotspot));
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
    assert!(o1.azm_pid_output != 0.0, "off-center blob must command a correction");
    // Dropout within coast window -> coast, stays in HOTSPOT, but the held
    // correction decays (halves) rather than integrating unconfirmed motion.
    let o2 = s.step(&i, None, 50.0, 45.0, 100.5);
    assert_eq!(o2.hotspot_status, "coasting");
    assert_eq!(o2.requested_mode, None);
    assert!(
        (o2.azm_pid_output - 0.5 * o1.azm_pid_output).abs() < 1e-12,
        "held rate must halve per missed frame"
    );
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

#[test]
fn program_setpoint_clear_commands_stop_once() {
    // A tracked target that sets (or is deselected) must produce an explicit
    // stop -- "no new command" leaves the mount at its last rate (runaway).
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::Passthrough;
    i.setpoint = Some(Setpoint {
        az_deg: 50.0,
        el_deg: 30.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let o = s.step(&i, None, 40.0, 20.0, 100.0);
    assert!(o.azm_rate_cmd.is_some(), "tracking should command a rate");

    i.setpoint = None;
    let o = s.step(&i, None, 40.0, 20.0, 100.1);
    assert_eq!(o.azm_rate_cmd, Some(0), "cleared setpoint must stop AZM");
    assert_eq!(o.alt_rate_cmd, Some(0), "cleared setpoint must stop ALT");
    assert!(o.status_msg.is_some());

    // The stop is one-shot; subsequent no-target cycles stay quiet.
    let o = s.step(&i, None, 40.0, 20.0, 100.2);
    assert_eq!(o.azm_rate_cmd, None);
    assert_eq!(o.alt_rate_cmd, None);
}

#[test]
fn program_without_ever_having_setpoint_stays_quiet() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.setpoint = None;
    let o = s.step(&i, None, 40.0, 20.0, 100.0);
    assert_eq!(o.azm_rate_cmd, None);
    assert_eq!(o.alt_rate_cmd, None);
}

#[test]
fn program_target_beyond_mount_limits_aborts_to_standby() {
    // The azm/alt limits gate MOUNT-frame positions; a target that transforms
    // outside them must stop motion and hand back to STANDBY (mirrors
    // program_track's gate, which also converts sky->mount before comparing).
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::Passthrough; // mount alt == sky el
    i.alt_limit = (0.0, 80.0);
    i.setpoint = Some(Setpoint {
        az_deg: 50.0,
        el_deg: 85.0, // -> mount alt 85 > 80
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let o = s.step(&i, None, 40.0, 20.0, 100.0);
    assert_eq!(o.requested_mode, Some(Mode::Standby));
    assert_eq!(o.azm_rate_cmd, Some(0));
    assert_eq!(o.alt_rate_cmd, Some(0));

    // An in-limits target tracks normally.
    let mut s = LoopState::new();
    i.setpoint = Some(Setpoint {
        az_deg: 50.0,
        el_deg: 30.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let o = s.step(&i, None, 40.0, 20.0, 100.0);
    assert_eq!(o.requested_mode, None);
    assert!(o.azm_rate_cmd.is_some());
}

#[test]
fn program_west_to_east_goes_over_the_zenith() {
    // Mount west (270, 45), target east (az=90, el=45), Passthrough. The
    // canonical solution is a 180-deg azimuth slew; the flipped one
    // (az+180, 180-el) = (270, 135) is pure alt motion over the top. The
    // loop must choose the flip: azm error ~0, alt error ~+90.
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::Passthrough;
    i.setpoint = Some(Setpoint {
        az_deg: 90.0,
        el_deg: 45.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let o = s.step(&i, None, 270.0, 45.0, 100.0);
    assert!(o.azm_error.abs() < 1e-6, "azm must not slew the long way: {}", o.azm_error);
    assert!((o.alt_error - 90.0).abs() < 1e-6, "alt error {}", o.alt_error);
    assert!(o.alt_rate_cmd.unwrap_or(0) > 0, "ALT must drive up over the zenith");
}

#[test]
fn program_flip_vetoed_by_alt_limits() {
    // Same scenario but ALT limits exclude the flipped solution (alt 135):
    // fall back to the legal canonical 180-deg azimuth path, not STANDBY.
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::Passthrough;
    i.alt_limit = (0.0, 90.0);
    i.setpoint = Some(Setpoint {
        az_deg: 90.0,
        el_deg: 45.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let o = s.step(&i, None, 270.0, 45.0, 100.0);
    assert_eq!(o.requested_mode, None, "canonical is legal - no abort");
    assert!((o.azm_error.abs() - 180.0).abs() < 1e-6, "azm error {}", o.azm_error);
    assert!(o.alt_error.abs() < 1e-6, "alt error {}", o.alt_error);
}

#[test]
fn program_flip_altaz_error_convention() {
    // AltAz convention (ALT = 90 - el): the flipped solution for the
    // west-east scenario is (AZM=270, ALT=-45) -> alt error -90.
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Program;
    i.mount_mode = MountMode::AltAz;
    i.setpoint = Some(Setpoint {
        az_deg: 90.0,
        el_deg: 45.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    // current mount: AZM=270, ALT=45 (el=45, pointing west)
    let o = s.step(&i, None, 270.0, 45.0, 100.0);
    assert!(o.azm_error.abs() < 1e-6, "azm error {}", o.azm_error);
    assert!((o.alt_error + 90.0).abs() < 1e-6, "alt error {}", o.alt_error);
    assert!(o.alt_rate_cmd.unwrap_or(0) < 0);
}

#[test]
fn handoff_star_filter_rejects_static_blob() {
    // Boresight static + blob static in frame => the detection's implied sky
    // rate is ~0 (a star). The trajectory says the target moves at 0.5 deg/s,
    // so every post-baseline detection must be rejected and the hand-off must
    // never fire.
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Handoff;
    i.mount_mode = MountMode::Passthrough;
    i.handoff_min_frames = 2;
    i.hotspot.star_filter = true;
    i.hotspot.rate_gate_dps = 0.15;
    i.setpoint = Some(Setpoint {
        az_deg: 15.0,
        el_deg: 25.0,
        ff_az_dps: 0.5,
        ff_el_dps: 0.0,
    });
    let mut f = blob_frame(256, 200, 138.0, 110.0);
    f.seq = 1;
    f.time = 1.0; // star-filter rates are frame-time based
    // First fresh detection: rate baseline warming up -- neither counted nor
    // rejected (a star must not build count during the warm-up window).
    let o1 = s.step(&i, Some(&f), 10.0, 20.0, 1.0);
    assert_eq!(o1.hotspot_status, "detecting");
    assert_eq!(o1.handoff_detection_count, 0);
    // Subsequent frames >= the 0.35s baseline are verified and rejected.
    for seq in 2..=6u64 {
        f.seq = seq;
        f.time = 1.0 + 0.4 * (seq - 1) as f64;
        let o = s.step(&i, Some(&f), 10.0, 20.0, 1.0 + 0.4 * (seq - 1) as f64);
        assert_eq!(o.hotspot_status, "star-reject");
        assert_eq!(o.requested_mode, None, "a star must never trigger the hand-off");
        assert_eq!(o.handoff_detection_count, 0);
    }
}

#[test]
fn handoff_star_filter_accepts_matching_rate() {
    // Same geometry, but the trajectory rate is ~0 -- equivalent to a target
    // being tracked perfectly (zero relative pixel motion, boresight moving
    // with the target). The filter must accept and the hand-off fire.
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Handoff;
    i.mount_mode = MountMode::Passthrough;
    i.handoff_min_frames = 2;
    i.hotspot.star_filter = true;
    i.hotspot.rate_gate_dps = 0.15;
    i.setpoint = Some(Setpoint {
        az_deg: 15.0,
        el_deg: 25.0,
        ff_az_dps: 0.0,
        ff_el_dps: 0.0,
    });
    let mut f = blob_frame(256, 200, 138.0, 110.0);
    f.seq = 1;
    f.time = 1.0; // star-filter rates are frame-time based
    let o1 = s.step(&i, Some(&f), 10.0, 20.0, 1.0); // warm-up: not counted
    assert_eq!(o1.hotspot_status, "detecting");
    assert_eq!(o1.handoff_detection_count, 0);
    f.seq = 2;
    f.time = 1.4;
    let o2 = s.step(&i, Some(&f), 10.0, 20.0, 1.4); // verified: count 1
    assert_eq!(o2.requested_mode, None);
    assert_eq!(o2.handoff_detection_count, 1);
    f.seq = 3;
    f.time = 1.8;
    let o3 = s.step(&i, Some(&f), 10.0, 20.0, 1.8); // verified: count 2 -> engage
    assert_eq!(o3.requested_mode, Some(Mode::Hotspot));
}

#[test]
fn hotspot_rides_trajectory_feed_forward() {
    // A dead-centered target has zero optical error; the commanded rate must
    // still carry the trajectory feed-forward (3 deg/s), and a subsequent
    // miss must keep the feed-forward running while the correction decays.
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    i.mount_mode = MountMode::Passthrough;
    i.ff_azm_enabled = true;
    i.ff_alt_enabled = true;
    i.setpoint = Some(Setpoint {
        az_deg: 50.0,
        el_deg: 45.0,
        ff_az_dps: 3.0,
        ff_el_dps: 0.0,
    });
    let f = blob_frame(256, 200, 128.0, 100.0); // exact frame center
    let o1 = s.step(&i, Some(&f), 50.0, 45.0, 100.0);
    assert_eq!(o1.hotspot_status, "locked");
    assert!(
        (o1.azm_pid_output * 360.0 - 3.0).abs() < 0.3,
        "centered target should command ~3 deg/s of feed-forward, got {}",
        o1.azm_pid_output * 360.0
    );
    // Miss (no frame): coast at the trajectory rate, not a frozen total.
    let o2 = s.step(&i, None, 50.0, 45.0, 100.1);
    assert_eq!(o2.hotspot_status, "coasting");
    assert!(
        (o2.azm_pid_output * 360.0 - 3.0).abs() < 0.3,
        "coast must keep the feed-forward running, got {}",
        o2.azm_pid_output * 360.0
    );
}

/// Two overlapping-brightness blobs: with no gate history the tracker must
/// take the one near the reticle, not the globally brightest corner blob.
#[test]
fn hotspot_bare_acquires_nearest_to_center() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    let (w, h) = (256usize, 200usize);
    // Bright corner star + dimmer (still solid SNR) star near center.
    let mut data = vec![10.0f32; w * h];
    let mut splat = |cx: f64, cy: f64, amp: f64| {
        for y in 0..h {
            for x in 0..w {
                let r2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                data[y * w + x] += (amp * (-(r2) / (2.0 * 9.0)).exp()) as f32;
            }
        }
    };
    splat(20.0, 20.0, 220.0);
    splat(120.0, 108.0, 120.0);
    let f = Frame { data: Arc::new(data), h, w, seq: 1, time: 0.0 };
    let o = s.step(&i, Some(&f), 50.0, 45.0, 100.0);
    assert!(o.hotspot_acquired, "{}", o.hotspot_status);
    let (cx, cy) = o.hotspot_centroid.unwrap();
    assert!(
        (cx - 120.0).abs() < 4.0 && (cy - 108.0).abs() < 4.0,
        "locked ({cx:.1},{cy:.1}) — should be the near-center star, not the corner one"
    );
}

/// Tracking bare with the star filter ON, a sidereal-slow star must stay
/// locked indefinitely — the old bare branch rejected it once the rate
/// baseline warmed up, giving a lock/reject/decay limit cycle.
#[test]
fn hotspot_bare_star_stays_locked_with_filter_on() {
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Hotspot;
    i.hotspot.star_filter = true;
    let (w, h) = (256usize, 200usize);
    let mut locked = 0;
    for k in 0..80u64 {
        let t = k as f64 * 0.1;
        // Sidereal-ish drift: ~0.2 px/s at this plate scale.
        let mut f = blob_frame(w, h, 128.0 + 12.0 + 0.02 * k as f64, 100.0 - 8.0);
        f.seq = k + 1;
        f.time = t;
        let o = s.step(&i, Some(&f), 50.0, 45.0, 100.0 + t);
        assert_ne!(o.hotspot_status, "star-reject", "cycle {k}: bare tracking must accept a star");
        if o.hotspot_acquired {
            locked += 1;
        }
    }
    assert!(locked >= 78, "star lock held only {locked}/80 cycles");
}
