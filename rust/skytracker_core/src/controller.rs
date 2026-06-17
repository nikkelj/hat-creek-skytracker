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
    pub x_sign: f64,
    pub y_sign: f64,
    pub pixel_size_um: f64,
    pub focal_length_mm: f64,
    pub rotation_deg: f64,
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
    pub setpoint: Option<Setpoint>,
    pub ff_azm_enabled: bool,
    pub ff_alt_enabled: bool,
    /// Lead time (s): extrapolate the sky setpoint forward by this much using the
    /// trajectory (feed-forward) rates to compensate read/command transport
    /// latency. Mirrors joystick_controller.program_track's pid_lead_time_sec.
    /// 0 = disabled. Kept congruent with the Python control loop.
    pub lead_time_sec: f64,

    // Continuous variable-rate (guide-rate) tracking instead of discrete MC_MOVE.
    // Applies only to PROGRAM/HOTSPOT; above guide_rate_max_dps the loop falls
    // back to the discrete step (e.g. the near-zenith keyhole).
    pub continuous_rate: bool,
    pub guide_rate_max_dps: f64,

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
            setpoint: None,
            ff_azm_enabled: false,
            ff_alt_enabled: false,
            lead_time_sec: 0.0,
            continuous_rate: false,
            guide_rate_max_dps: 5.0,
            handoff_min_frames: 5,
            hotspot: HotspotParams {
                snr_threshold: 5.0,
                gate_radius: 120.0,
                coast_time_s: 1.0,
                x_sign: 1.0,
                y_sign: -1.0,
                pixel_size_um: 4.0,
                focal_length_mm: 1000.0,
                rotation_deg: 0.0,
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

    pub status_msg: Option<String>,
}

fn wrap180(deg: f64) -> f64 {
    (deg + 180.0).rem_euclid(360.0) - 180.0
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

    // HANDOFF: consecutive-detection counter toward the auto hand-off.
    handoff_detection_count: u32,
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
            handoff_detection_count: 0,
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
                }
                Mode::Handoff => {
                    self.handoff_detection_count = 0;
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
            Some(s) => s,
            None => return, // no target yet
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

        // Sky -> mount (the per-cycle transform we ported in step 5).
        let (target_azm, target_alt) = transforms::sky_to_mount(
            inputs.mount_mode,
            led_az,
            led_el,
            inputs.alignment_az,
            inputs.alignment_el,
        );

        let azm_error = wrap180(target_azm - current_azm);
        let alt_error = target_alt - current_alt;

        self.azm_pid
            .update_gains(inputs.azm_gains.0, inputs.azm_gains.1, inputs.azm_gains.2);
        self.alt_pid
            .update_gains(inputs.alt_gains.0, inputs.alt_gains.1, inputs.alt_gains.2);
        self.azm_pid.set_feed_forward_enabled(inputs.ff_azm_enabled);
        self.alt_pid.set_feed_forward_enabled(inputs.ff_alt_enabled);
        // Sky rates -> mount feed-forward. AZM tracks azimuth directly; in AltAz
        // the ALT axis runs opposite sky elevation (ALT = 90 - el), so the ALT
        // feed-forward is -ff_el_dps. Mirrors joystick_controller.program_track.
        let alt_ff = if inputs.mount_mode == MountMode::AltAz {
            -sp.ff_el_dps
        } else {
            sp.ff_el_dps
        };
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

        // 2) Run the hot-spot detector in parallel (full frame, no commanding).
        let need = inputs.handoff_min_frames.max(1);
        match detect_in_frame(frame, &inputs.hotspot, None) {
            Some(d) => {
                self.handoff_detection_count += 1;
                out.hotspot_snr = d.snr;
                out.hotspot_centroid = Some((d.cx, d.cy));
                out.hotspot_status = "detecting";
                if self.handoff_detection_count >= need {
                    self.handoff_detection_count = 0;
                    out.requested_mode = Some(Mode::Hotspot);
                    out.status_msg =
                        Some("HANDOFF: solid detection - engaging HOTSPOT tracker".to_string());
                }
            }
            None => {
                // Require *consecutive* detections; any miss resets the counter.
                self.handoff_detection_count = 0;
                out.hotspot_status = "program";
            }
        }
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
            return;
        }

        // Detect on this cycle's frame (gated to the last lock once acquired).
        let gate = self
            .hotspot_gate_center
            .map(|(cx, cy)| (cx, cy, hp.gate_radius));
        let detection = detect_in_frame(frame, &hp, gate);

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
                current_alt,
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

            let dt = self.dt(now);
            let (az_out, az_rate) =
                self.azm_pid
                    .compute_pid_output(az_error, dt, Some(current_azm));
            let (al_out, al_rate) =
                self.alt_pid
                    .compute_pid_output(el_error, dt, Some(current_alt));

            self.hotspot_gate_center = Some((det.cx, det.cy));
            self.hotspot_acquired = true;
            self.hotspot_miss_count = 0;
            self.hotspot_last_detection_time = now;

            out.azm_error = az_error;
            out.alt_error = el_error;
            out.azm_pid_output = az_out;
            out.alt_pid_output = al_out;
            out.hotspot_acquired = true;
            out.hotspot_status = "locked";
            out.hotspot_snr = det.snr;
            out.hotspot_centroid = Some((det.cx, det.cy));
            out.azm_rate_cmd = Some(az_rate);
            out.alt_rate_cmd = Some(al_rate);
            return;
        }

        // No detection this cycle.
        self.hotspot_miss_count += 1;

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
    }
}

impl Default for LoopState {
    fn default() -> Self {
        Self::new()
    }
}
