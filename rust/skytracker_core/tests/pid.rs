//! Pure-Rust PID tests mirroring test_pid_control.py. Run: cargo test
//! Cross-language step-for-step parity lives in test_rust_pid_parity.py.

use skytracker_core::pid::PidController;
use skytracker_core::protocol;

fn pid(p: f64, i: f64, d: f64, axis: &str, ff: bool) -> PidController {
    PidController::new(p, i, d, axis.to_string(), ff)
}

#[test]
fn no_derivative_kick_on_setpoint_step() {
    let mut c = pid(0.0, 0.0, 1.0, "ALT", false);
    c.compute_pid_output(0.0, 0.1, Some(100.0));
    let (out, _) = c.compute_pid_output(50.0, 0.1, Some(100.0));
    assert!(out.abs() < 1e-9, "out={out}");
}

#[test]
fn error_mode_does_kick() {
    let mut c = pid(0.0, 0.0, 1.0, "ALT", false);
    c.compute_pid_output(0.0, 0.1, None);
    let (out, _) = c.compute_pid_output(50.0, 0.1, None);
    assert!((out - 50.0 / 0.1).abs() < 1e-6, "out={out}");
}

#[test]
fn derivative_tracks_measurement_motion() {
    let mut c = pid(0.0, 0.0, 2.0, "ALT", false);
    c.compute_pid_output(0.0, 0.1, Some(10.0));
    let (out, _) = c.compute_pid_output(0.0, 0.1, Some(11.0));
    assert!((out - (-20.0)).abs() < 1e-6, "out={out}");
}

#[test]
fn azm_wrap_no_spike() {
    let mut c = pid(0.0, 0.0, 1.0, "AZM", false);
    c.compute_pid_output(0.0, 0.1, Some(359.0));
    let (out, _) = c.compute_pid_output(0.0, 0.1, Some(1.0));
    assert!((out - (-20.0)).abs() < 1e-6, "out={out}");
}

#[test]
fn integral_frozen_while_saturated() {
    let mut c = pid(0.025, 0.01, 0.0, "ALT", false);
    let max_rate = protocol::rate(9);
    for _ in 0..100 {
        let (out, _) = c.compute_pid_output(10.0, 0.1, None);
        assert!(out.abs() > max_rate);
    }
    assert!(c.integral_error.abs() < 1e-9, "int={}", c.integral_error);
}

#[test]
fn integral_accumulates_when_unsaturated() {
    let mut c = pid(0.025, 0.01, 0.0, "ALT", false);
    for _ in 0..10 {
        c.compute_pid_output(0.01, 0.1, None);
    }
    assert!(c.integral_error > 0.0);
}

#[test]
fn feed_forward_unit_conversion() {
    let mut c = pid(1.0, 0.0, 0.0, "", true);
    c.set_feed_forward_rate(360.0); // 360 deg/s == 1 rev/s
    assert!((c.current_feed_forward_rate - 1.0).abs() < 1e-9);

    let mut off = pid(1.0, 0.0, 0.0, "", false);
    off.set_feed_forward_rate(360.0);
    assert_eq!(off.current_feed_forward_rate, 0.0);
}

#[test]
fn feed_forward_flows_to_output() {
    let mut c = pid(0.0, 0.0, 0.0, "ALT", true);
    c.set_feed_forward_rate(3.6); // -> 0.01 rev/s
    let (out, _) = c.compute_pid_output(0.0, 0.1, Some(5.0));
    assert!((out - 0.01).abs() < 1e-9, "out={out}");
}

#[test]
fn sign_conventions() {
    let mut c = pid(0.025, 0.0, 0.0, "ALT", false);
    let (_, r) = c.compute_pid_output(5.0, 0.1, None);
    assert!(r > 0);
    let mut c2 = pid(0.025, 0.0, 0.0, "ALT", false);
    let (_, r2) = c2.compute_pid_output(-5.0, 0.1, None);
    assert!(r2 < 0);
}
