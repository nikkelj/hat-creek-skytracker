#!/usr/bin/env python
"""
Unit tests for the PID controller improvements in control.py:
  * derivative-on-measurement (no derivative kick on setpoint/feed-forward steps),
  * AZM wrap-aware derivative (no spike across the 0/360 boundary),
  * conditional-integration anti-windup (integrator frozen while saturated),
  * feed-forward unit conversion (deg/sec trajectory rate -> rev/sec command).

Pure math; needs only numpy. Run: python test_pid_control.py
"""

import unittest

from control import PIDController
from lib.auxstar import RATES

MAX_RATE = RATES[9]  # saturation threshold the controller uses (rev/sec)


class DerivativeOnMeasurementTests(unittest.TestCase):

    def test_no_derivative_kick_on_setpoint_step(self):
        # d-only controller; steady measurement but a sudden error (setpoint)
        # step must NOT produce a derivative spike.
        pid = PIDController(p_gain=0.0, i_gain=0.0, d_gain=1.0, axis_name="ALT")
        pid.compute_pid_output(error_degrees=0.0, dt_seconds=0.1, measurement_degrees=100.0)
        out, _ = pid.compute_pid_output(error_degrees=50.0, dt_seconds=0.1,
                                        measurement_degrees=100.0)
        self.assertAlmostEqual(out, 0.0, places=9)

    def test_error_mode_does_kick(self):
        # Same step without a measurement (legacy derivative-on-error) DOES kick.
        pid = PIDController(p_gain=0.0, i_gain=0.0, d_gain=1.0, axis_name="ALT")
        pid.compute_pid_output(error_degrees=0.0, dt_seconds=0.1)
        out, _ = pid.compute_pid_output(error_degrees=50.0, dt_seconds=0.1)
        self.assertAlmostEqual(out, 1.0 * 50.0 / 0.1, places=6)

    def test_derivative_tracks_measurement_motion(self):
        pid = PIDController(p_gain=0.0, i_gain=0.0, d_gain=2.0, axis_name="ALT")
        pid.compute_pid_output(0.0, 0.1, measurement_degrees=10.0)
        # measurement moved +1 deg in 0.1 s -> d = -d_gain * (1/0.1) = -20
        out, _ = pid.compute_pid_output(0.0, 0.1, measurement_degrees=11.0)
        self.assertAlmostEqual(out, -20.0, places=6)

    def test_azm_wrap_no_spike(self):
        pid = PIDController(p_gain=0.0, i_gain=0.0, d_gain=1.0, axis_name="AZM")
        pid.compute_pid_output(0.0, 0.1, measurement_degrees=359.0)
        # 359 -> 1 is a real +2 deg move, not -358. d = -1 * (2/0.1) = -20.
        out, _ = pid.compute_pid_output(0.0, 0.1, measurement_degrees=1.0)
        self.assertAlmostEqual(out, -20.0, places=6)


class AntiWindupTests(unittest.TestCase):

    def test_integral_frozen_while_saturated(self):
        pid = PIDController(p_gain=0.025, i_gain=0.01, d_gain=0.0, axis_name="ALT")
        # 10 deg error -> p_term alone (0.25) exceeds the rate ceiling, so the
        # command is saturated every cycle and the integrator must not wind up.
        for _ in range(100):
            out, _ = pid.compute_pid_output(10.0, 0.1)
            self.assertTrue(abs(out) > MAX_RATE)  # genuinely saturated
        self.assertAlmostEqual(pid.integral_error, 0.0, places=9)

    def test_integral_accumulates_when_unsaturated(self):
        pid = PIDController(p_gain=0.025, i_gain=0.01, d_gain=0.0, axis_name="ALT")
        # Tiny error keeps the command well below saturation -> integrator works.
        for _ in range(10):
            pid.compute_pid_output(0.01, 0.1)
        self.assertGreater(pid.integral_error, 0.0)


class FeedForwardUnitTests(unittest.TestCase):

    def test_deg_per_sec_converted_to_rev_per_sec(self):
        pid = PIDController(feed_forward_enabled=True)
        pid.set_feed_forward_rate(360.0)  # 360 deg/s == 1 rev/s
        self.assertAlmostEqual(pid.current_feed_forward_rate, 1.0, places=9)

    def test_feed_forward_disabled_is_zero(self):
        pid = PIDController(feed_forward_enabled=False)
        pid.set_feed_forward_rate(360.0)
        self.assertEqual(pid.current_feed_forward_rate, 0.0)

    def test_feed_forward_flows_to_output(self):
        pid = PIDController(p_gain=0.0, i_gain=0.0, d_gain=0.0,
                            feed_forward_enabled=True, axis_name="ALT")
        pid.set_feed_forward_rate(3.6)  # -> 0.01 rev/s
        out, _ = pid.compute_pid_output(0.0, 0.1, measurement_degrees=5.0)
        self.assertAlmostEqual(out, 0.01, places=9)


class SignConventionTests(unittest.TestCase):

    def test_positive_error_positive_rate(self):
        pid = PIDController(p_gain=0.025, i_gain=0.0, d_gain=0.0, axis_name="ALT")
        _, rate = pid.compute_pid_output(5.0, 0.1)
        self.assertGreater(rate, 0)

    def test_negative_error_negative_rate(self):
        pid = PIDController(p_gain=0.025, i_gain=0.0, d_gain=0.0, axis_name="ALT")
        _, rate = pid.compute_pid_output(-5.0, 0.1)
        self.assertLess(rate, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
