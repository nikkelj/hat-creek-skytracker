//! PID controller — port of `control.py::PIDController`.
//!
//! Faithful to the Python behavior, including the three subtleties the Python
//! unit tests pin down:
//!   * derivative-on-measurement (no derivative kick on a setpoint/feed-forward
//!     step), with the AZM wrap-aware delta,
//!   * conditional-integration anti-windup (the integrator is rolled back on any
//!     cycle where the command is saturated *and* the error pushes it further
//!     into saturation), and
//!   * feed-forward unit conversion (deg/sec trajectory rate -> rev/sec command).
//!
//! Output units are revolutions/second; the discrete rate is the signed 0..=9
//! MC_MOVE step. Validated against `test_pid_control.py` (mirrored in
//! `tests/pid.rs`) and cross-checked step-for-step in `test_rust_pid_parity.py`.

use crate::protocol;

/// Saturation threshold the controller uses: RATES[9] (rev/sec).
fn max_rate() -> f64 {
    protocol::rate(9)
}

/// numpy-style sign: -1.0 / 0.0 / 1.0.
fn signum(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

pub struct PidController {
    pub p_gain: f64,
    pub i_gain: f64,
    pub d_gain: f64,
    pub axis_name: String,
    pub feed_forward_enabled: bool,

    pub integral_error: f64,
    pub previous_error: f64,
    pub previous_measurement: Option<f64>,

    pub current_feed_forward_rate: f64,

    // Low-pass (EMA) on the FEEDBACK term only (mirrors control.py): tau in
    // seconds, 0 disables. Feed-forward is added after the filter so
    // trajectory-rate changes pass through unfiltered.
    pub output_filter_tau: f64,
    filtered_feedback: f64,

    // Diagnostics (read by the UI in Python).
    pub current_position_error: f64,
    pub current_pid_output: f64,
}

impl PidController {
    pub fn new(
        p_gain: f64,
        i_gain: f64,
        d_gain: f64,
        axis_name: String,
        feed_forward_enabled: bool,
    ) -> Self {
        Self {
            p_gain,
            i_gain,
            d_gain,
            axis_name,
            feed_forward_enabled,
            integral_error: 0.0,
            previous_error: 0.0,
            previous_measurement: None,
            current_feed_forward_rate: 0.0,
            output_filter_tau: 0.0,
            filtered_feedback: 0.0,
            current_position_error: 0.0,
            current_pid_output: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.previous_error = 0.0;
        self.previous_measurement = None;
        self.filtered_feedback = 0.0;
    }

    /// Feedback low-pass time constant (seconds; 0 disables).
    pub fn set_output_filter_tau(&mut self, tau_seconds: f64) {
        self.output_filter_tau = tau_seconds.max(0.0);
    }

    pub fn update_gains(&mut self, p_gain: f64, i_gain: f64, d_gain: f64) {
        self.p_gain = p_gain;
        self.i_gain = i_gain;
        self.d_gain = d_gain;
    }

    /// Trajectory rates arrive in deg/sec; the command is rev/sec. Convert here
    /// (1 rev = 360 deg) so feed-forward shares units with the feedback output.
    pub fn set_feed_forward_rate(&mut self, feed_forward_rate_deg_per_sec: f64) {
        self.current_feed_forward_rate = if self.feed_forward_enabled {
            feed_forward_rate_deg_per_sec / 360.0
        } else {
            0.0
        };
    }

    pub fn set_feed_forward_enabled(&mut self, enabled: bool) {
        self.feed_forward_enabled = enabled;
        if !enabled {
            self.current_feed_forward_rate = 0.0;
        }
    }

    /// Returns (total_output_rev_per_sec, signed_discrete_rate).
    pub fn compute_pid_output(
        &mut self,
        error_degrees: f64,
        dt_seconds: f64,
        measurement_degrees: Option<f64>,
    ) -> (f64, i32) {
        if dt_seconds <= 0.0 {
            return (0.0, 0);
        }

        let error_degrees = clip(error_degrees, -180.0, 180.0);

        let p_term = self.p_gain * error_degrees;

        // Derivative on measurement when available (no derivative kick),
        // otherwise on error.
        let d_term = match measurement_degrees {
            Some(meas) => {
                let d = match self.previous_measurement {
                    None => 0.0,
                    Some(prev) => {
                        let mut delta = meas - prev;
                        // AZM wraps at 360; keep the delta on the short arc.
                        if self.axis_name == "AZM" {
                            delta = (delta + 180.0).rem_euclid(360.0) - 180.0;
                        }
                        -self.d_gain * delta / dt_seconds
                    }
                };
                self.previous_measurement = Some(meas);
                d
            }
            None => self.d_gain * (error_degrees - self.previous_error) / dt_seconds,
        };

        self.previous_error = error_degrees;

        // Tentatively accumulate the integrator.
        self.integral_error += error_degrees * dt_seconds;
        self.integral_error = clip(self.integral_error, -100.0, 100.0);
        let mut i_term = self.i_gain * self.integral_error;

        let mut pid_feedback_output = p_term + i_term + d_term;
        let mut total_output = pid_feedback_output + self.current_feed_forward_rate;

        // Conditional-integration anti-windup: roll back this cycle's
        // accumulation if saturated and the error pushes further into it.
        if total_output.abs() > max_rate() && signum(error_degrees) == signum(total_output) {
            self.integral_error -= error_degrees * dt_seconds;
            self.integral_error = clip(self.integral_error, -100.0, 100.0);
            i_term = self.i_gain * self.integral_error;
            pid_feedback_output = p_term + i_term + d_term;
            total_output = pid_feedback_output + self.current_feed_forward_rate;
        }

        // Feedback low-pass (EMA): smooth measurement jitter out of the rate
        // command; feed-forward is added after so trajectory-rate changes
        // pass through unfiltered. Mirrors control.py.
        if self.output_filter_tau > 0.0 {
            let alpha = dt_seconds / (self.output_filter_tau + dt_seconds);
            self.filtered_feedback += alpha * (pid_feedback_output - self.filtered_feedback);
            pid_feedback_output = self.filtered_feedback;
            total_output = pid_feedback_output + self.current_feed_forward_rate;
        } else {
            self.filtered_feedback = pid_feedback_output;
        }

        self.current_position_error = error_degrees;
        self.current_pid_output = total_output;

        let discrete_rate = Self::error_to_discrete_rate(total_output);
        (total_output, discrete_rate)
    }

    /// Map output (rev/sec) to a signed discrete rate (-9..=9), preserving sign
    /// and biasing slightly toward higher rates to reduce undershoot. Public so
    /// HOTSPOT can recompute the discrete step for its feed-forward + correction
    /// total (the PID's own discrete output covers the correction only).
    pub fn error_to_discrete_rate(pid_output_rev_per_sec: f64) -> i32 {
        let sign: i32 = if pid_output_rev_per_sec >= 0.0 { 1 } else { -1 };
        let requested = pid_output_rev_per_sec.abs();
        if requested < 0.01 {
            return 0;
        }
        let mut best_rate_magnitude: i32 = 0;
        let mut min_distance = f64::INFINITY;
        for idx in 0..10u8 {
            let rate_velocity = protocol::rate(idx);
            let mut distance = (rate_velocity - requested).abs();
            if rate_velocity >= requested {
                distance *= 0.9; // prefer higher rates
            }
            if distance < min_distance {
                min_distance = distance;
                best_rate_magnitude = idx as i32;
            }
        }
        sign * best_rate_magnitude
    }
}
