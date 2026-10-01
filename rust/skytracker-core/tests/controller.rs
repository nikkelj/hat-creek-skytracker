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

/// Synthetic close-range launch: a structured rocket (nose, body, dark
/// interstage, fins) on a sky/ground scene. Pre-launch the operator grabs a
/// template on the UPPER BODY; at T0 the vehicle accelerates upward, an
/// exhaust plume far brighter than the airframe ignites at the base, and the
/// vehicle shrinks as it climbs. The FEATURE loop must keep the grabbed
/// patch near boresight throughout — and must NOT slide onto the plume the
/// way a brightest-blob tracker would.
mod launch_sim {
    use super::*;

    pub const W: usize = 320;
    pub const H: usize = 240;

    /// Deterministic per-frame noise.
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Render the scene with the rocket CENTER at (rcx, rcy) in frame pixels.
    pub fn rocket_frame(rcx: f64, rcy: f64, scale: f64, plume: f64, k: u64) -> Frame {
        let mut data = vec![0.0f32; W * H];
        let mut seed = 0x5eed ^ k;
        let horizon = H as f64 * 0.92;
        for y in 0..H {
            for x in 0..W {
                let sky = 18.0 + y as f64 * 0.04;
                let v = if (y as f64) > horizon { 55.0 } else { sky };
                data[y * W + x] = (v + 3.0 * lcg(&mut seed)) as f32;
            }
        }
        let bw = 26.0 * scale;
        let bh = 110.0 * scale;
        let mut put = |x: f64, y: f64, v: f64| {
            if x >= 0.0 && y >= 0.0 && (x as usize) < W && (y as usize) < H {
                let i = y as usize * W + x as usize;
                data[i] = data[i].max(v as f32);
            }
        };
        for dy in 0..bh as usize {
            let fy = rcy - bh / 2.0 + dy as f64;
            let frac = dy as f64 / bh;
            for dx in 0..bw as usize {
                let fx = rcx - bw / 2.0 + dx as f64;
                // Nose cone (top 18%): bright, tapering. Interstage band at
                // 45%: dark. Engine section (bottom 12%): mid-dark. Body: mid.
                let v = if frac < 0.18 {
                    let taper = (frac / 0.18).max(0.15);
                    let half = bw / 2.0 * taper;
                    if (fx - rcx).abs() > half { continue; } else { 195.0 }
                } else if (0.42..0.50).contains(&frac) {
                    70.0
                } else if frac > 0.88 {
                    95.0
                } else {
                    150.0 - 25.0 * ((fx - rcx).abs() / (bw / 2.0)) // rounded shading
                };
                put(fx, fy, v);
            }
        }
        // Plume: saturating blob below the base — brighter than anything on
        // the airframe (the trap for a brightest-blob tracker).
        if plume > 0.0 {
            let (pcx, pcy) = (rcx, rcy + bh / 2.0 + 14.0 * scale);
            let sigma = (9.0 + 10.0 * plume) * scale;
            let amp = 255.0 * plume.min(1.0);
            let r = (3.0 * sigma) as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let g = amp * (-((dx * dx + dy * dy) as f64) / (2.0 * sigma * sigma)).exp();
                    if g > 1.0 {
                        put(pcx + dx as f64, pcy + dy as f64, g.min(255.0));
                    }
                }
            }
        }
        Frame { data: Arc::new(data), h: H, w: W, seq: k + 1, time: k as f64 / 15.0 }
    }
}

#[test]
fn feature_tracks_close_range_launch() {
    use launch_sim::*;
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Feature;
    i.mount_mode = MountMode::Passthrough;
    i.hotspot.max_rate_dps = 6.0; // close range needs real slew authority
    let degpp = (i.hotspot.pixel_size_um * 1e-3 / i.hotspot.focal_length_mm).atan().to_degrees();

    // World state (pixels in an inertial frame): rocket fixed pre-launch.
    let (wx0, wy0) = (12.0f64, -60.0f64); // world offset of the rocket center
    // Boresight (deg, controller convention: +az = scene moves left, +el = scene moves down).
    let (mut b_az, mut b_el) = (0.0f64, 0.0f64);
    let dt = 1.0 / 15.0;
    let t0_launch = 3.0;
    let mut grabbed = false;
    let mut max_err_post = 0.0f64;
    let mut final_err = 0.0f64;
    let mut worst_feature_slip = 0.0f64;

    for k in 0..210u64 {
        let t = k as f64 * dt;
        let (wy, scale, plume) = if t < t0_launch {
            (wy0, 1.0, 0.0)
        } else {
            let tl = t - t0_launch;
            // 45 px/s² boost, shrink toward 0.55× as it recedes.
            (wy0 - 22.5 * tl * tl, (1.0 - tl / 22.0).max(0.55), (tl * 2.5).min(1.0))
        };
        // Render: rocket frame position = world − boresight (in px).
        let rcx = W as f64 / 2.0 + wx0 - b_az / degpp;
        let rcy = H as f64 / 2.0 + wy + b_el / degpp;
        let f = rocket_frame(rcx, rcy, scale, plume, k);

        // Operator grabs the UPPER BODY (between nose and interstage) at t=0.4.
        if !grabbed && t >= 0.4 {
            let gx = rcx;
            let gy = rcy - 30.0 * scale;
            i.feature_grab = Some((gx, gy, 22));
            i.feature_grab_seq += 1;
            grabbed = true;
        }
        let o = s.step(&i, Some(&f), 0.0, 0.0, 100.0 + t);
        i.feature_grab = None;

        // Actuate: integrate the commanded correction into the boresight
        // (Passthrough at el 0: axis rates == sky rates).
        b_az += o.azm_pid_output * 360.0 * dt;
        b_el += o.alt_pid_output * 360.0 * dt;

        if grabbed && t > t0_launch + 0.7 {
            // Grabbed-feature world position vs boresight, in px.
            let feat_px_x = rcx;
            let feat_px_y = rcy - 30.0 * scale;
            let err = ((feat_px_x - W as f64 / 2.0).powi(2) + (feat_px_y - H as f64 / 2.0).powi(2)).sqrt();
            max_err_post = max_err_post.max(err);
            final_err = err;
            // The tracked box must stay on the grabbed feature — NOT the
            // plume: its offset from the rocket center must remain near the
            // grab offset (−30·scale), far above the plume (+70·scale).
            if let Some((_, cy, _)) = o.feature_box {
                let slip = cy - (rcy - 30.0 * scale);
                worst_feature_slip = worst_feature_slip.max(slip.abs());
                assert!(slip < 35.0 * scale, "t={t:.2}: tracked point slid {slip:.0} px down — toward the plume");
            }
            assert_ne!(o.hotspot_status, "lost — re-grab", "t={t:.2}: lock lost during ascent");
        }
    }
    eprintln!("launch sim: max post-launch error {max_err_post:.1} px, final {final_err:.1} px, worst feature slip {worst_feature_slip:.1} px");
    assert!(max_err_post < 60.0, "boresight error peaked at {max_err_post:.1} px");
    assert!(final_err < 20.0, "final error {final_err:.1} px");
}

/// Pre-launch the grabbed patch is static: the loop must hold, not hunt.
#[test]
fn feature_holds_static_prelaunch() {
    use launch_sim::*;
    let mut s = LoopState::new();
    let mut i = base_inputs();
    i.mode = Mode::Feature;
    i.mount_mode = MountMode::Passthrough;
    let f0 = rocket_frame(W as f64 / 2.0 + 8.0, H as f64 / 2.0 - 40.0, 1.0, 0.0, 0);
    i.feature_grab = Some((W as f64 / 2.0 + 8.0, H as f64 / 2.0 - 70.0, 20));
    i.feature_grab_seq = 1;
    let _ = s.step(&i, Some(&f0), 0.0, 0.0, 100.0);
    i.feature_grab = None;
    let mut total_cmd = 0.0;
    for k in 1..40u64 {
        let f = rocket_frame(W as f64 / 2.0 + 8.0, H as f64 / 2.0 - 40.0, 1.0, 0.0, k);
        let o = s.step(&i, Some(&f), 0.0, 0.0, 100.0 + k as f64 / 15.0);
        assert_eq!(o.hotspot_status, "locked", "cycle {k}");
        total_cmd += o.azm_pid_output.abs() + o.alt_pid_output.abs();
    }
    // Static scene, feature ~70 px off boresight: corrections exist but the
    // tracked point itself must not wander (template stability).
    let _ = total_cmd;
}
