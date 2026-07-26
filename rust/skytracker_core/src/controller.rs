//! Control-loop decision logic — Rust port of `tracking_control()` plus the
//! per-cycle parts of `program_track` and `hotspot_track`.
//!
//! The decision logic is a PURE function of (inputs, frame, current position,
//! now): `LoopState::step` returns the rate commands to issue, any requested
//! mode transition, and diagnostics. It performs no serial I/O and touches no
//! Python objects, so it is exhaustively unit-testable (`tests/controller.rs`)
//! and runs off the GIL when wrapped by the threaded loop (`core_loop.rs`).
//!
//! Inputs the loop cannot own (joystick axes, camera frames, skyfield setpoint)
//! are pushed in by Python as plain Rust data; see `Inputs` / `Frame`.

use crate::hotspot::{self, Detection};
use crate::pid::PidController;
use crate::transforms::{self, MountMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Standby,
    Rate,
    Program,
    Handoff,
    Hotspot,
    Mti,
}

/// Sky-frame target pushed by Python (from skyfield). Feed-forward rates are
/// deg/sec; the PID converts them to rev/sec internally.
#[derive(Debug, Clone, Copy)]
pub struct Setpoint {
    pub az_deg: f64,
    pub el_deg: f64,
    pub ff_az_dps: f64,
    pub ff_el_dps: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct HotspotParams {
    pub snr_threshold: f64,
    pub gate_radius: f64,
    pub coast_time_s: f64,
    /// Cap on the centering-correction rate (deg/s). A centering loop that
    /// lunges at slew rates yanks the target out of its own tracking gate.
    pub max_rate_dps: f64,
    pub x_sign: f64,
    pub y_sign: f64,
    pub pixel_size_um: f64,
    pub focal_length_mm: f64,
    pub rotation_deg: f64,
    /// Star-rejection rate filter: accept a detection only when its implied
    /// sky angular rate matches the trajectory (or, tracking bare, is NOT
    /// star-like near-zero). Toggle off to deliberately track a star.
    pub star_filter: bool,
    /// Tolerance (deg/s) for the rate comparison above; doubles as the
    /// "near-sidereal" threshold in bare mode.
    pub rate_gate_dps: f64,
}

/// Everything Python pushes into the loop each time it changes. Cloned out under
/// the lock once per cycle.
#[derive(Clone)]
pub struct Inputs {
    pub connected: bool,
    pub stopped: bool,
    pub mode: Mode,

    pub azm_gains: (f64, f64, f64),
    pub alt_gains: (f64, f64, f64),
    pub azm_limit: (f64, f64),
    pub alt_limit: (f64, f64),
    /// Position offsets applied by the poll (raw - offset = working position).
    pub offsets: (f64, f64),

    // RATE_CONTROL: Python maps joystick -> discrete rate and pushes it.
    pub rate_cmd: (i32, i32),

    // PROGRAM
    pub mount_mode: MountMode,
    pub alignment_az: f64,
    pub alignment_el: f64,
    /// Side-mount tip side for MountMode::AltAzSide (see transforms).
    pub altaz_side_flip: bool,
    pub setpoint: Option<Setpoint>,
    pub ff_azm_enabled: bool,
    pub ff_alt_enabled: bool,
    /// Lead time (s): extrapolate the sky setpoint forward by this much using the
    /// trajectory (feed-forward) rates to compensate read/command transport
    /// latency. Mirrors joystick_controller.program_track's pid_lead_time_sec.
    /// 0 = disabled. Kept congruent with the Python control loop.
    pub lead_time_sec: f64,

    // Continuous variable-rate (guide-rate) tracking instead of discrete MC_MOVE.
    // Applies to the tracking modes (PROGRAM/HANDOFF/HOTSPOT); above
    // guide_rate_max_dps the loop falls back to the discrete step (e.g. the
    // near-zenith keyhole).
    pub continuous_rate: bool,
    pub guide_rate_max_dps: f64,

    /// Feedback low-pass time constant for both PIDs (seconds; 0 disables).
    pub output_filter_tau: f64,

    // HANDOFF: hand to HOTSPOT after this many consecutive solid detections.
    pub handoff_min_frames: u32,

    // HOTSPOT
    pub hotspot: HotspotParams,
}

impl Default for Inputs {
    fn default() -> Self {
        Inputs {
            connected: false,
            stopped: false,
            mode: Mode::Standby,
            azm_gains: (1.0, 0.0, 0.0),
            alt_gains: (1.0, 0.0, 0.0),
            azm_limit: (f64::NEG_INFINITY, f64::INFINITY),
            alt_limit: (f64::NEG_INFINITY, f64::INFINITY),
            offsets: (0.0, 0.0),
            rate_cmd: (0, 0),
            mount_mode: MountMode::AltAz,
            alignment_az: 0.0,
            alignment_el: 0.0,
            altaz_side_flip: false,
            setpoint: None,
            ff_azm_enabled: false,
            ff_alt_enabled: false,
            lead_time_sec: 0.0,
            continuous_rate: false,
            // 24-bit guide-rate full scale is 4.551 dps (arcsec/s Q10); the
            // encoder clamps there, so default the MC_MOVE fallback gate below it.
            guide_rate_max_dps: 4.5,
            output_filter_tau: 0.0,
            handoff_min_frames: 5,
            hotspot: HotspotParams {
                snr_threshold: 5.0,
                gate_radius: 120.0,
                coast_time_s: 1.0,
                max_rate_dps: 2.0,
                x_sign: 1.0,
                y_sign: -1.0,
                pixel_size_um: 4.0,
                focal_length_mm: 1000.0,
                rotation_deg: 0.0,
                star_filter: false,
                rate_gate_dps: 0.15,
            },
        }
    }
}

/// A camera frame as a Rust-owned intensity map (the camera shim converts the
/// numpy frame under the GIL and publishes this; the loop reads it GIL-free).
pub struct Frame {
    pub data: std::sync::Arc<Vec<f32>>,
    pub h: usize,
    pub w: usize,
    pub seq: u64,
}

/// What the loop wants done this cycle. `None` rate = leave the axis as-is
/// (HOTSPOT coast leaves the last continuous slew running); `Some(r)` = command.
#[derive(Debug, Clone, Default)]
pub struct StepOutput {
    pub azm_rate_cmd: Option<i32>,
    pub alt_rate_cmd: Option<i32>,
    pub requested_mode: Option<Mode>,

    pub azm_error: f64,
    pub alt_error: f64,
    pub azm_pid_output: f64,
    pub alt_pid_output: f64,

    pub hotspot_acquired: bool,
    pub hotspot_status: &'static str,
    pub hotspot_snr: f64,
    pub hotspot_centroid: Option<(f64, f64)>,
    pub handoff_detection_count: u32,

    pub status_msg: Option<String>,
}

fn wrap180(deg: f64) -> f64 {
    (deg + 180.0).rem_euclid(360.0) - 180.0
}

/// Discrete MC_MOVE step for a capped HOTSPOT rate (rev/s): the largest step
/// whose physical rate fits under |total|, minimum 1 so a small capped
/// correction still creeps toward center. The PID's own discretizer zeroes
/// anything below 0.01 rev/s (3.6 deg/s) -- fine for raw PID outputs, but a
/// hotspot correction capped to ~2 deg/s must still actuate in discrete
/// mode. Mirrors joystick_controller.hotspot_discrete_step.
fn hotspot_discrete_step(total_rev_s: f64) -> i32 {
    if total_rev_s.abs() <= 1e-6 {
        return 0;
    }
    let sign = if total_rev_s > 0.0 { 1 } else { -1 };
    let mut mag = 1i32;
    for idx in (1..=9u8).rev() {
        if crate::protocol::rate(idx) <= total_rev_s.abs() {
            mag = idx as i32;
            break;
        }
    }
    sign * mag
}

/// Run hot-spot detection on a frame (gated or full-frame). Returns None when
/// there is no frame or nothing qualifies. Pure; issues no commands.
fn detect_in_frame(
    frame: Option<&Frame>,
    hp: &HotspotParams,
    gate: Option<(f64, f64, f64)>,
) -> Option<Detection> {
    let f = frame?;
    let view = ndarray::ArrayView2::from_shape((f.h, f.w), f.data.as_slice()).ok()?;
    hotspot::detect_hotspot(&view, gate, hp.snr_threshold, 12, 0.5, 3)
}

/// Per-axis PID + the HOTSPOT lock/coast state machine + mode-entry guards.
pub struct LoopState {
    pub azm_pid: PidController,
    pub alt_pid: PidController,
    pid_last_update: f64,

    // STANDBY one-shot stop guard.
    standby_stopped: bool,
    prev_mode: Option<Mode>,

    // HOTSPOT lock state (ported from JoystickModeState).
    hotspot_gate_center: Option<(f64, f64)>,
    hotspot_acquired: bool,
    hotspot_miss_count: u32,
    hotspot_last_detection_time: f64,
    hotspot_entry_time: f64,
    hotspot_last_frame_seq: Option<u64>,
    // Fresh-frame arrival time + measured camera cadence (sec) — corrections
    // are capped so one frame's command can't outrun the next measurement.
    hotspot_last_fresh_time: f64,
    hotspot_frame_interval: f64,
    // Last commanded slew (az_dps, el_dps) + when the gate was anchored, so
    // the search gate can be shifted by the loop's own commanded motion.
    hotspot_cmd_dps: (f64, f64),
    hotspot_gate_time: f64,
    // Optical-correction part of hotspot_cmd_dps: the miss-path decay bleeds
    // only this, while the trajectory feed-forward keeps running.
    hotspot_corr_dps: (f64, f64),
    // Star-rejection rate filter: recent fresh-frame detection candidates as
    // (time, cx, cy, boresight_az_sky, boresight_el_sky). Rates are measured
    // against a candidate >= RATE_FILTER_BASELINE_S old (see the constants).
    track_candidates: Vec<(f64, f64, f64, f64, f64)>,

    // HANDOFF: consecutive-detection counter toward the auto hand-off.
    handoff_detection_count: u32,

    // PROGRAM: whether a setpoint was active on the previous step, so a
    // cleared setpoint (target set / deselected) commands an explicit stop
    // once instead of silently leaving the mount at its last commanded rate.
    program_had_setpoint: bool,
}

impl LoopState {
    pub fn new() -> Self {
        LoopState {
            azm_pid: PidController::new(1.0, 0.0, 0.0, "AZM".to_string(), false),
            alt_pid: PidController::new(1.0, 0.0, 0.0, "ALT".to_string(), false),
            pid_last_update: 0.0,
            standby_stopped: false,
            prev_mode: None,
            hotspot_gate_center: None,
            hotspot_acquired: false,
            hotspot_miss_count: 0,
            hotspot_last_detection_time: 0.0,
            hotspot_entry_time: 0.0,
            hotspot_last_frame_seq: None,
            hotspot_last_fresh_time: 0.0,
            hotspot_frame_interval: 0.2,
            hotspot_cmd_dps: (0.0, 0.0),
            hotspot_gate_time: 0.0,
            hotspot_corr_dps: (0.0, 0.0),
            track_candidates: Vec::new(),
            handoff_detection_count: 0,
            program_had_setpoint: false,
        }
    }

    fn dt(&mut self, now: f64) -> f64 {
        let dt = if self.pid_last_update == 0.0 {
            0.1
        } else {
            now - self.pid_last_update
        };
        self.pid_last_update = now;
        dt
    }

    /// Pure per-cycle decision. `current_azm`/`current_alt` are offset-applied
    /// degrees (the threaded loop polls them once and passes them here).
    pub fn step(
        &mut self,
        inputs: &Inputs,
        frame: Option<&Frame>,
        current_azm: f64,
        current_alt: f64,
        now: f64,
    ) -> StepOutput {
        let mut out = StepOutput::default();

        // Not connected: nothing to do (the threaded loop also guards this).
        if !inputs.connected {
            return out;
        }

        // Universal stop in any mode.
        if inputs.stopped {
            out.azm_rate_cmd = Some(0);
            out.alt_rate_cmd = Some(0);
            return out;
        }

        // Reset the STANDBY one-shot whenever in an active mode.
        if inputs.mode != Mode::Standby {
            self.standby_stopped = false;
        }

        // Per-mode entry resets (mirrors JoystickModeState mode-transition logic).
        let entering = self.prev_mode != Some(inputs.mode);
        if entering {
            match inputs.mode {
                Mode::Hotspot => {
                    self.hotspot_gate_center = None;
                    self.hotspot_acquired = false;
                    self.hotspot_miss_count = 0;
                    self.hotspot_last_detection_time = 0.0;
                    self.hotspot_entry_time = now;
                    self.hotspot_last_frame_seq = None;
                    self.hotspot_last_fresh_time = 0.0;
                    self.hotspot_frame_interval = 0.2;
                    self.hotspot_cmd_dps = (0.0, 0.0);
                    self.hotspot_corr_dps = (0.0, 0.0);
                    self.hotspot_gate_time = 0.0;
                    self.track_candidates.clear();
                }
                Mode::Handoff => {
                    self.handoff_detection_count = 0;
                    self.hotspot_last_frame_seq = None;
                    self.track_candidates.clear();
                }
                _ => {}
            }
        }
        self.prev_mode = Some(inputs.mode);

        match inputs.mode {
            Mode::Standby => {
                // Single stop on entry, then stay quiet.
                if !self.standby_stopped {
                    out.azm_rate_cmd = Some(0);
                    out.alt_rate_cmd = Some(0);
                    self.standby_stopped = true;
                }
            }
            Mode::Rate => {
                out.azm_rate_cmd = Some(inputs.rate_cmd.0);
                out.alt_rate_cmd = Some(inputs.rate_cmd.1);
            }
            Mode::Program => self.step_program(inputs, current_azm, current_alt, now, &mut out),
            Mode::Handoff => {
                self.step_handoff(inputs, frame, current_azm, current_alt, now, &mut out)
            }
            Mode::Hotspot => {
                self.step_hotspot(inputs, frame, current_azm, current_alt, now, &mut out)
            }
            // MTI: stub, as in Python.
            Mode::Mti => {}
        }

        out
    }

    fn step_program(
        &mut self,
        inputs: &Inputs,
        current_azm: f64,
        current_alt: f64,
        now: f64,
        out: &mut StepOutput,
    ) {
        let sp = match inputs.setpoint {
            Some(s) => {
                self.program_had_setpoint = true;
                s
            }
            None => {
                // No target. If we *were* tracking one (it set below the
                // horizon, or was deselected), command an explicit stop once —
                // issuing no new command would leave the mount at its last
                // rate, which is a slow-motion runaway, not a hold.
                if self.program_had_setpoint {
                    self.program_had_setpoint = false;
                    out.azm_rate_cmd = Some(0);
                    out.alt_rate_cmd = Some(0);
                    out.status_msg =
                        Some("PROGRAM: setpoint cleared - motion stopped".to_string());
                }
                return;
            }
        };

        // Lead/extrapolate the sky setpoint by lead_time using the trajectory
        // (feed-forward) rates, to compensate read/command transport latency.
        // Mirrors program_track's `target += rate * lead_s` (applied to the sky
        // az/el before sky_to_mount). Operator bias is applied upstream in the
        // Python adapter; lead and bias are both additive to the setpoint, so the
        // net target matches program_track regardless of which side applies which.
        let lead = inputs.lead_time_sec;
        let led_az = sp.az_deg + sp.ff_az_dps * lead;
        let led_el = sp.el_deg + sp.ff_el_dps * lead;

        // Sky -> mount (the per-cycle transform we ported in step 5), choosing
        // the SHORTEST-slew axis solution. An alt-az style mount reaches every
        // sky pointing in two configurations: the canonical one and the
        // over-the-zenith one from the mirrored sky representation
        // (az+180, 180-el). Per-axis wrap alone cannot discover the second
        // solution -- pointed west with the target in the east it drives the
        // azimuth axis ~180 deg the long way instead of ~90 deg of ALT motion
        // straight over the top. Mirrors control.choose_mount_target
        // (same minimax metric, same hysteresis, limits-aware; Eq mode has no
        // usable second solution here).
        let canon = transforms::sky_to_mount(
            inputs.mount_mode,
            led_az,
            led_el,
            inputs.alignment_az,
            inputs.alignment_el,
            inputs.altaz_side_flip,
        );
        let (azm_min, azm_max) = inputs.azm_limit;
        let (alt_min, alt_max) = inputs.alt_limit;
        let in_limits = |t: (f64, f64)| {
            t.0 >= azm_min && t.0 <= azm_max && t.1 >= alt_min && t.1 <= alt_max
        };
        let metric = |t: (f64, f64)| {
            wrap180(t.0 - current_azm)
                .abs()
                .max(wrap180(t.1 - current_alt).abs())
        };
        const FLIP_HYSTERESIS_DEG: f64 = 0.5;
        let mut target_azm = canon.0;
        let mut target_alt = canon.1;
        let mut flipped = false;
        // Only true alt-az geometries have the mirrored-sky second solution;
        // Eq and AltAzSide map (az+180, 180-el) to the same principal axes.
        if matches!(inputs.mount_mode, MountMode::AltAz | MountMode::Passthrough) {
            let flip = transforms::sky_to_mount(
                inputs.mount_mode,
                (led_az + 180.0).rem_euclid(360.0),
                180.0 - led_el,
                inputs.alignment_az,
                inputs.alignment_el,
                inputs.altaz_side_flip,
            );
            let canon_ok = in_limits(canon);
            let flip_ok = in_limits(flip);
            let use_flip = match (canon_ok, flip_ok) {
                (_, false) => false,
                (false, true) => true,
                (true, true) => metric(flip) < metric(canon) - FLIP_HYSTERESIS_DEG,
            };
            if use_flip {
                target_azm = flip.0;
                target_alt = flip.1;
                flipped = true;
            }
        }

        // Safety: abort to STANDBY if the MOUNT-frame target exceeds the
        // configured axis limits (they gate encoder positions, so the check
        // must happen after sky_to_mount). Mirrors program_track.
        if target_azm > azm_max
            || target_azm < azm_min
            || target_alt > alt_max
            || target_alt < alt_min
        {
            out.azm_rate_cmd = Some(0);
            out.alt_rate_cmd = Some(0);
            out.requested_mode = Some(Mode::Standby);
            out.status_msg = Some(format!(
                "PROGRAM: target (mount AZM:{target_azm:.1} ALT:{target_alt:.1}) exceeds safety limits - switched to STANDBY"
            ));
            return;
        }

        let azm_error = wrap180(target_azm - current_azm);
        // ALT is wrapped too: the ALT encoder is modular ([0, 360)) and in AltAz
        // el = 90 - ALT, so the boresight sits at zenith when ALT ~= 0, on the
        // 0/360 seam. On a near-overhead pass current_alt can read ~359 while
        // target_alt is a few degrees; without wrap the PID would drive a near-
        // full 360 ALT traversal instead of the short hop. Mirrors
        // control.compute_mount_position_error.
        let alt_error = wrap180(target_alt - current_alt);

        self.azm_pid
            .update_gains(inputs.azm_gains.0, inputs.azm_gains.1, inputs.azm_gains.2);
        self.alt_pid
            .update_gains(inputs.alt_gains.0, inputs.alt_gains.1, inputs.alt_gains.2);
        self.azm_pid.set_output_filter_tau(inputs.output_filter_tau);
        self.alt_pid.set_output_filter_tau(inputs.output_filter_tau);
        self.azm_pid.set_feed_forward_enabled(inputs.ff_azm_enabled);
        self.alt_pid.set_feed_forward_enabled(inputs.ff_alt_enabled);
        // Sky rates -> mount feed-forward. AZM tracks azimuth directly; the
        // ALT direction follows the mount convention (in AltAz the axis runs
        // opposite sky elevation, ALT = 90 - el) AND the chosen configuration
        // (the over-the-zenith solution runs opposite the canonical one).
        // Mirrors joystick_controller.program_track.
        let mut alt_sign = if inputs.mount_mode == MountMode::AltAz {
            -1.0
        } else {
            1.0
        };
        if flipped {
            alt_sign = -alt_sign;
        }
        let alt_ff = alt_sign * sp.ff_el_dps;
        self.azm_pid.set_feed_forward_rate(sp.ff_az_dps);
        self.alt_pid.set_feed_forward_rate(alt_ff);

        let dt = self.dt(now);
        let (az_out, az_rate) =
            self.azm_pid
                .compute_pid_output(azm_error, dt, Some(current_azm));
        let (al_out, al_rate) =
            self.alt_pid
                .compute_pid_output(alt_error, dt, Some(current_alt));

        out.azm_error = azm_error;
        out.alt_error = alt_error;
        out.azm_pid_output = az_out;
        out.alt_pid_output = al_out;
        out.azm_rate_cmd = Some(az_rate);
        out.alt_rate_cmd = Some(al_rate);
    }

    /// Star-rejection rate gate (mirrors JoystickModeState._detection_rate_filter).
    ///
    /// A detection's implied SKY angular rate is the boresight motion plus
    /// the pixel drift, measured against the newest recorded candidate at
    /// least RATE_FILTER_BASELINE_S old -- pixel positions are captured at
    /// frame time but the boresight at loop-cycle time, so a short
    /// (consecutive-frame) baseline corrupts the estimate by
    /// ~rate * timing_skew / baseline, falsely rejecting fast targets. The
    /// program target moves at the trajectory rate; a star moves at
    /// ~sidereal (near zero inertially). With a setpoint available, reject
    /// mismatches beyond max(rate_gate_dps, REL_FRACTION * |trajectory
    /// rate|); tracking bare, reject near-zero (star-like) rates.
    ///
    /// Returns Some(true) verified / Some(false) rejected / None while the
    /// baseline is still warming up. ALWAYS records the candidate, so a
    /// persistent star keeps failing.
    fn detection_rate_verdict(
        &mut self,
        det: &Detection,
        inputs: &Inputs,
        current_azm: f64,
        current_alt: f64,
        el_sky: f64,
        now: f64,
    ) -> Option<bool> {
        const RATE_FILTER_BASELINE_S: f64 = 0.35;
        const RATE_FILTER_MAX_AGE_S: f64 = 3.0;
        const RATE_FILTER_REL_FRACTION: f64 = 0.35;

        let hp = &inputs.hotspot;
        let (bs_az, bs_el) = if inputs.mount_mode == MountMode::AltAz {
            (current_azm, 90.0 - current_alt)
        } else {
            (current_azm, current_alt)
        };
        self.track_candidates
            .retain(|c| now - c.0 <= RATE_FILTER_MAX_AGE_S);
        let base = self
            .track_candidates
            .iter()
            .rev()
            .find(|c| now - c.0 >= RATE_FILTER_BASELINE_S)
            .copied();
        self.track_candidates
            .push((now, det.cx, det.cy, bs_az, bs_el));
        if !hp.star_filter {
            return Some(true);
        }
        let (t0, cx0, cy0, az0, el0) = base?; // None: baseline warming up
        let dt = now - t0;
        let (da, de) = hotspot::pixel_offset_to_angles(
            det.cx - cx0,
            det.cy - cy0,
            hp.pixel_size_um,
            hp.focal_length_mm,
            hp.rotation_deg,
            el_sky,
            hp.x_sign,
            hp.y_sign,
            true,
        );
        let impl_az = (wrap180(bs_az - az0) + da) / dt;
        let impl_el = (bs_el - el0 + de) / dt;
        let gate = if hp.rate_gate_dps > 0.0 {
            hp.rate_gate_dps
        } else {
            0.15
        };
        let cos_el = el_sky.to_radians().cos().abs().max(1e-3);
        Some(match inputs.setpoint {
            Some(sp) => {
                let ref_mag =
                    ((sp.ff_az_dps * cos_el).powi(2) + sp.ff_el_dps.powi(2)).sqrt();
                let thresh = gate.max(RATE_FILTER_REL_FRACTION * ref_mag);
                let diff = (((impl_az - sp.ff_az_dps) * cos_el).powi(2)
                    + (impl_el - sp.ff_el_dps).powi(2))
                .sqrt();
                diff <= thresh
            }
            None => {
                let mag = ((impl_az * cos_el).powi(2) + impl_el.powi(2)).sqrt();
                mag >= gate
            }
        })
    }

    /// HANDOFF: keep PROGRAM track closing the loop while running the hot-spot
    /// detector in parallel; hand the loop to HOTSPOT after `handoff_min_frames`
    /// consecutive solid detections. Mirrors JoystickModeState.handoff_track.
    #[allow(clippy::too_many_arguments)]
    fn step_handoff(
        &mut self,
        inputs: &Inputs,
        frame: Option<&Frame>,
        current_azm: f64,
        current_alt: f64,
        now: f64,
        out: &mut StepOutput,
    ) {
        // 1) Keep PROGRAM-tracking (drives the mount; includes lead + feed-forward).
        self.step_program(inputs, current_azm, current_alt, now, out);

        // 2) Run the hot-spot detector in parallel (full frame, no commanding),
        //    on FRESH frames only: a stale frame carries no new information, so
        //    it must neither count toward nor reset the consecutive-detection
        //    requirement (which also gives the star filter distinct frames to
        //    difference for its rate estimate).
        let frame_is_stale = matches!(
            (frame, self.hotspot_last_frame_seq),
            (Some(f), Some(last)) if f.seq == last
        );
        if let Some(f) = frame {
            if !frame_is_stale {
                self.hotspot_last_frame_seq = Some(f.seq);
            }
        }
        if frame_is_stale {
            out.handoff_detection_count = self.handoff_detection_count;
            return;
        }

        let need = inputs.handoff_min_frames.max(1);
        match detect_in_frame(frame, &inputs.hotspot, None) {
            Some(d) => {
                out.hotspot_snr = d.snr;
                out.hotspot_centroid = Some((d.cx, d.cy));
                let el_sky = if inputs.mount_mode == MountMode::AltAz {
                    90.0 - current_alt
                } else {
                    current_alt
                };
                match self.detection_rate_verdict(&d, inputs, current_azm, current_alt, el_sky, now)
                {
                    Some(true) => {
                        self.handoff_detection_count += 1;
                        out.hotspot_status = "detecting";
                        if self.handoff_detection_count >= need {
                            self.handoff_detection_count = 0;
                            out.requested_mode = Some(Mode::Hotspot);
                            out.status_msg = Some(
                                "HANDOFF: solid detection - engaging HOTSPOT tracker".to_string(),
                            );
                        }
                    }
                    Some(false) => {
                        // A star (or wrong object) is not the target: reset the
                        // consecutive count so it can never trigger the hand-off.
                        self.handoff_detection_count = 0;
                        out.hotspot_status = "star-reject";
                    }
                    None => {
                        // Filter warming up its rate baseline: neither count
                        // nor reset, so a star can't ride the warm-up window
                        // into a hand-off.
                        out.hotspot_status = "detecting";
                    }
                }
            }
            None => {
                // Require *consecutive* detections; any miss resets the counter.
                self.handoff_detection_count = 0;
                out.hotspot_status = "program";
            }
        }
        out.handoff_detection_count = self.handoff_detection_count;
    }

    #[allow(clippy::too_many_arguments)]
    fn step_hotspot(
        &mut self,
        inputs: &Inputs,
        frame: Option<&Frame>,
        current_azm: f64,
        current_alt: f64,
        now: f64,
        out: &mut StepOutput,
    ) {
        let hp = inputs.hotspot;

        // Safety: abort to STANDBY if outside configured limits.
        let (azm_min, azm_max) = inputs.azm_limit;
        let (alt_min, alt_max) = inputs.alt_limit;
        if !(azm_min <= current_azm && current_azm <= azm_max
            && alt_min <= current_alt && current_alt <= alt_max)
        {
            out.azm_rate_cmd = Some(0);
            out.alt_rate_cmd = Some(0);
            out.requested_mode = Some(Mode::Standby);
            out.hotspot_status = "limit";
            out.status_msg =
                Some("HOTSPOT: mount at safety limit - switched to STANDBY".to_string());
            self.hotspot_acquired = false;
            self.hotspot_gate_center = None;
            self.hotspot_corr_dps = (0.0, 0.0);
            self.hotspot_cmd_dps = (0.0, 0.0);
            return;
        }

        // Stale-frame gate (mirrors hotspot_track): a frame we already
        // processed carries no new measurement, so don't re-detect it -- the
        // PID would re-integrate the same centroid with an advancing dt. The
        // time-based coast/loss logic below still runs, so a camera that
        // stops producing frames coasts out and falls back.
        let frame_is_stale = matches!(
            (frame, self.hotspot_last_frame_seq),
            (Some(f), Some(last)) if f.seq == last
        );
        if let Some(f) = frame {
            if !frame_is_stale {
                self.hotspot_last_frame_seq = Some(f.seq);
            }
        }
        let frame = if frame_is_stale { None } else { frame };

        // Measured fresh-frame interval (hit OR miss): gate prediction and the
        // correction-rate cap are sized to it, so the loop never outruns its
        // own measurements. Mirrors hotspot_track.
        if frame.is_some() {
            if self.hotspot_last_fresh_time > 0.0 {
                self.hotspot_frame_interval =
                    (now - self.hotspot_last_fresh_time).clamp(0.05, 2.0);
            }
            self.hotspot_last_fresh_time = now;
        }

        // The optical geometry needs the SKY elevation (azimuth compresses by
        // cos(el) on the sky); in AltAz the mount ALT axis is 90 - el, and
        // passing it raw overstates the azimuth error by cos(el)/cos(ALT) --
        // a divergent overshoot at moderate elevations.
        let el_sky = if inputs.mount_mode == MountMode::AltAz {
            90.0 - current_alt
        } else {
            current_alt
        };

        // Detect on this cycle's frame, gated to the last lock once acquired.
        // The gate is PREDICTED forward by the boresight motion we ourselves
        // commanded since the frame it was anchored on (with a narrow FOV a
        // legitimate correction sweeps a large pixel distance between frames),
        // and GROWS on consecutive misses so residual prediction error or
        // target motion can't strand it while the target is still in frame.
        let gate = self.hotspot_gate_center.map(|(cx, cy)| {
            // Predict with the CORRECTION rates, not the total command: the
            // trajectory feed-forward moves the boresight WITH the target
            // (no apparent pixel drift); only the correction closes on it.
            // Predicting with the total slid the gate off a well-tracked
            // fast target at the full feed-forward rate.
            let (az_dps, alt_dps) = self.hotspot_corr_dps;
            let gate_dt = (now - self.hotspot_gate_time).max(0.0);
            let (mut gx, mut gy) = (cx, cy);
            if gate_dt > 0.0 && (az_dps != 0.0 || alt_dps != 0.0) {
                let el_sky_dps = if inputs.mount_mode == MountMode::AltAz {
                    -alt_dps
                } else {
                    alt_dps
                };
                let (pdx, pdy) = hotspot::angles_to_pixel_offset(
                    -az_dps * gate_dt,
                    -el_sky_dps * gate_dt,
                    hp.pixel_size_um,
                    hp.focal_length_mm,
                    hp.rotation_deg,
                    el_sky,
                    hp.x_sign,
                    hp.y_sign,
                    true,
                );
                gx += pdx;
                gy += pdy;
            }
            let growth = 1.5f64.powi(self.hotspot_miss_count.min(8) as i32).min(4.0);
            (gx, gy, hp.gate_radius * growth)
        });
        let detection = detect_in_frame(frame, &hp, gate);

        // Star-rejection rate gate: a detection whose implied sky rate doesn't
        // match the trajectory (or is star-like when tracking bare) is treated
        // as a miss -- the gate grows and the correction decays, so a bright
        // star drifting through the gate can't capture the loop.
        // None (baseline warming up) is accepted: HOTSPOT must keep
        // commanding, and HANDOFF already verified the object before
        // promoting -- only an explicit mismatch demotes to a miss.
        let detection = match detection {
            Some(d) => {
                if self.detection_rate_verdict(&d, inputs, current_azm, current_alt, el_sky, now)
                    == Some(false)
                {
                    out.hotspot_status = "star-reject";
                    None
                } else {
                    Some(d)
                }
            }
            None => None,
        };

        // Trajectory feed-forward (mount frame): the optical correction rides
        // on the program target's sky rates so a moving target no longer needs
        // the capped correction to supply ALL of the tracking rate. Honors the
        // same per-axis FF toggles as PROGRAM; zero when tracking bare.
        let (ff_az, ff_el) = match inputs.setpoint {
            Some(sp) => (
                if inputs.ff_azm_enabled { sp.ff_az_dps } else { 0.0 },
                if inputs.ff_alt_enabled {
                    if inputs.mount_mode == MountMode::AltAz {
                        -sp.ff_el_dps
                    } else {
                        sp.ff_el_dps
                    }
                } else {
                    0.0
                },
            ),
            None => (0.0, 0.0),
        };

        if let Some(det) = detection {
            let (h, w) = frame.map(|f| (f.h, f.w)).unwrap_or((0, 0));
            let dx = det.cx - (w as f64 / 2.0);
            let dy = det.cy - (h as f64 / 2.0);
            let (az_error, el_error) = hotspot::pixel_offset_to_angles(
                dx,
                dy,
                hp.pixel_size_um,
                hp.focal_length_mm,
                hp.rotation_deg,
                el_sky,
                hp.x_sign,
                hp.y_sign,
                true,
            );

            // pixel_offset_to_angles gives the optical/sky-frame correction. In
            // AltAz the ALT axis runs opposite sky elevation (el = 90 - ALT), so
            // negate the elevation term to command the mount the right way --
            // matching the mount-frame error PROGRAM track feeds the same PID.
            // Mirrors joystick_controller.hotspot_track.
            let el_error = if inputs.mount_mode == MountMode::AltAz {
                -el_error
            } else {
                el_error
            };

            self.azm_pid
                .update_gains(inputs.azm_gains.0, inputs.azm_gains.1, inputs.azm_gains.2);
            self.alt_pid
                .update_gains(inputs.alt_gains.0, inputs.alt_gains.1, inputs.alt_gains.2);
            self.azm_pid.set_output_filter_tau(inputs.output_filter_tau);
            self.alt_pid.set_output_filter_tau(inputs.output_filter_tau);
            // Feed-forward is added MANUALLY below (the correction cap must
            // not clamp it), so zero the PIDs' own FF term -- these
            // controllers are shared with step_program, whose last
            // set_feed_forward_rate would otherwise leak in as a stale bias.
            self.azm_pid.set_feed_forward_rate(0.0);
            self.alt_pid.set_feed_forward_rate(0.0);

            let dt = self.dt(now);
            let (az_out, _az_rate) =
                self.azm_pid
                    .compute_pid_output(az_error, dt, Some(current_azm));
            let (al_out, _al_rate) =
                self.alt_pid
                    .compute_pid_output(el_error, dt, Some(current_alt));

            // Cap each axis's CORRECTION so it covers at most ~90% of the
            // remaining error before the NEXT measurement (measured frame
            // interval): a correction that outruns its measurements
            // overshoots, exits its own gate, and coasts into a stair-step.
            // The trajectory feed-forward rides UNDER the cap: it is not a
            // correction, it's the target's own motion.
            let interval = self.hotspot_frame_interval.max(0.05);
            let cap_az = (hp.max_rate_dps / 360.0).min(0.9 * az_error.abs() / interval / 360.0);
            let cap_al = (hp.max_rate_dps / 360.0).min(0.9 * el_error.abs() / interval / 360.0);
            let az_corr = az_out.clamp(-cap_az, cap_az);
            let al_corr = al_out.clamp(-cap_al, cap_al);
            let az_total = ff_az / 360.0 + az_corr;
            let al_total = ff_el / 360.0 + al_corr;
            let az_rate = hotspot_discrete_step(az_total);
            let al_rate = hotspot_discrete_step(al_total);
            // Remember what we commanded (mount frame, deg/s): the gate
            // prediction uses the TOTAL boresight motion; the miss-path decay
            // bleeds only the correction while feed-forward keeps running.
            self.hotspot_corr_dps = (az_corr * 360.0, al_corr * 360.0);
            self.hotspot_cmd_dps = (az_total * 360.0, al_total * 360.0);

            self.hotspot_gate_center = Some((det.cx, det.cy));
            self.hotspot_gate_time = now;
            self.hotspot_acquired = true;
            self.hotspot_miss_count = 0;
            self.hotspot_last_detection_time = now;

            out.azm_error = az_error;
            out.alt_error = el_error;
            out.azm_pid_output = az_total;
            out.alt_pid_output = al_total;
            out.hotspot_acquired = true;
            out.hotspot_status = "locked";
            out.hotspot_snr = det.snr;
            out.hotspot_centroid = Some((det.cx, det.cy));
            out.azm_rate_cmd = Some(az_rate);
            out.alt_rate_cmd = Some(al_rate);
            return;
        }

        // No detection this cycle. A stale frame is "no new information",
        // not a miss -- only count misses against frames actually examined.
        if !frame_is_stale {
            self.hotspot_miss_count += 1;
            // Bleed off the held CORRECTION: a correction that hasn't been
            // re-confirmed by a detection must not keep integrating (a single
            // overshoot would otherwise coast into an FOV-sized stair-step).
            // Halve per missed frame. The trajectory feed-forward is NOT
            // decayed -- coasting follows the target's motion.
            let (mut az_c, mut el_c) = self.hotspot_corr_dps;
            az_c *= 0.5;
            el_c *= 0.5;
            if az_c.abs() < 1e-3 {
                az_c = 0.0;
            }
            if el_c.abs() < 1e-3 {
                el_c = 0.0;
            }
            self.hotspot_corr_dps = (az_c, el_c);
            if az_c != 0.0 || el_c != 0.0 || ff_az != 0.0 || ff_el != 0.0 {
                let az_total = (ff_az + az_c) / 360.0;
                let el_total = (ff_el + el_c) / 360.0;
                self.hotspot_cmd_dps = (az_total * 360.0, el_total * 360.0);
                out.azm_pid_output = az_total;
                out.alt_pid_output = el_total;
                out.azm_rate_cmd = Some(hotspot_discrete_step(az_total));
                out.alt_rate_cmd = Some(hotspot_discrete_step(el_total));
            }
        }

        if self.hotspot_acquired {
            // Coast: leave the last continuous slew running (no new command)
            // through a brief dropout before declaring the lock lost.
            if now - self.hotspot_last_detection_time < hp.coast_time_s {
                out.hotspot_status = "coasting";
                out.hotspot_acquired = true;
                return;
            }
        } else if now - self.hotspot_entry_time < hp.coast_time_s.max(1.0) {
            // Acquisition grace: just entered HOTSPOT (manual or handed off). Give
            // frames time to arrive / the target time to be found before bailing,
            // and leave the last slew running so a moving target stays roughly
            // framed. Without this the loop fell back to PROGRAM on the very first
            // frameless cycle (the async frame push lags the mode change).
            out.hotspot_status = "acquiring";
            return;
        }

        // Lock lost (or never acquired within the grace window): stop and hand back.
        out.azm_rate_cmd = Some(0);
        out.alt_rate_cmd = Some(0);
        out.requested_mode = Some(Mode::Program);
        out.hotspot_status = "lost";
        out.status_msg = Some("HOTSPOT: lost lock - falling back to PROGRAM track".to_string());
        self.hotspot_acquired = false;
        self.hotspot_gate_center = None;
        self.hotspot_corr_dps = (0.0, 0.0);
        self.hotspot_cmd_dps = (0.0, 0.0);
    }
}

impl Default for LoopState {
    fn default() -> Self {
        Self::new()
    }
}
