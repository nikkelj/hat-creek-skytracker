#!/usr/bin/env python
"""
Unit tests pinning the RATE_CONTROL adaptive gearbox (AdaptiveRateMapper):
full stick deflection is capped at the base ceiling on first touch, the
ceiling winds up only while the stick is deliberately held pinned, and it
winds back down (faster) on back-off, release, direction reversal, or a
service gap. Deterministic: all timing is driven through the mapper's `now`
parameter, no wall clock.

Field symptom this exists to fix: a small extra stick deflection jumped the
commanded rate from ~0.5 deg/s to 5-10 deg/s (the MC_MOVE table is
~exponential), flinging the moon out of the FOV.

Headless. Run: python test_adaptive_rate.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest

import pygame
pygame.init()

from joystick_controller import (
    AdaptiveRateMapper,
    axis_to_rate,
    JOY_BASE_CEILING,
    JOY_MAX_CEILING,
    JOY_ENGAGE_DELAY_S,
    JOY_STEP_UP_S,
    JOY_STEP_DOWN_S,
    JOY_IDLE_RESET_S,
    JOY_REVERSAL_DROP,
    JOY_STALE_RESET_S,
)


def drive(mapper, az, alt, t_start, t_end, dt=0.05):
    """Service the mapper at control-loop cadence from t_start to t_end with
    fixed axis values, always including t_end itself exactly; returns the
    last (az_rate, alt_rate)."""
    n = max(0, int((t_end - t_start) / dt))
    rates = None
    for k in range(n + 1):
        rates = mapper.update(az, alt, now=t_start + k * dt)
    if t_start + n * dt < t_end - 1e-9:
        rates = mapper.update(az, alt, now=t_end)
    return rates


def wind_to_max(mapper, t_start=0.0):
    """Pin the az axis until the ceiling saturates; returns the end time."""
    t_end = t_start + JOY_ENGAGE_DELAY_S + JOY_STEP_UP_S * (
        JOY_MAX_CEILING - JOY_BASE_CEILING) + 0.2
    drive(mapper, 1.0, 0.0, t_start, t_end)
    assert mapper.ceiling == JOY_MAX_CEILING
    return t_end


class TestLegacyMapping(unittest.TestCase):
    """axis_to_rate with the default ceiling must stay the historical mapping
    (the Rust/Python parity contract)."""

    def test_default_ceiling_unchanged(self):
        self.assertEqual(axis_to_rate(0.0), 0)
        self.assertEqual(axis_to_rate(0.005), 0)      # deadband
        self.assertEqual(axis_to_rate(0.05), 1)
        self.assertEqual(axis_to_rate(0.5), 5)
        self.assertEqual(axis_to_rate(0.95), 9)
        self.assertEqual(axis_to_rate(-0.5), -5)
        self.assertEqual(axis_to_rate(-1.0), -9)

    def test_ceiling_caps_and_rescales(self):
        # Full deflection lands exactly on the ceiling...
        self.assertEqual(axis_to_rate(1.0, ceiling=5), 5)
        # ...and partial deflection spreads over 1..ceiling, giving finer
        # control than the same deflection under the full 1..9 map.
        self.assertEqual(axis_to_rate(0.5, ceiling=5), 3)
        self.assertEqual(axis_to_rate(-0.5, ceiling=5), -3)
        self.assertEqual(axis_to_rate(0.3, ceiling=5), 2)
        self.assertEqual(axis_to_rate(0.3, ceiling=9), 3)


class TestAdaptiveCeiling(unittest.TestCase):
    def setUp(self):
        self.m = AdaptiveRateMapper()

    def test_first_touch_is_capped_at_base(self):
        az, alt = self.m.update(1.0, -1.0, now=0.0)
        self.assertEqual((az, alt), (JOY_BASE_CEILING, -JOY_BASE_CEILING))
        self.assertEqual(self.m.ceiling, JOY_BASE_CEILING)

    def test_windup_requires_sustained_pin(self):
        # Held pinned but short of the engage delay: still in base gear.
        drive(self.m, 1.0, 0.0, 0.0, JOY_ENGAGE_DELAY_S - 0.1)
        self.assertEqual(self.m.ceiling, JOY_BASE_CEILING)

    def test_windup_step_timing(self):
        expected = JOY_BASE_CEILING
        t = 0.0
        drive(self.m, 1.0, 0.0, 0.0, JOY_ENGAGE_DELAY_S - 0.05)
        # First step lands at the engage delay, then one per interval.
        for step in range(JOY_MAX_CEILING - JOY_BASE_CEILING):
            t = JOY_ENGAGE_DELAY_S + step * JOY_STEP_UP_S
            drive(self.m, 1.0, 0.0, t - 0.04, t)
            expected += 1
            self.assertEqual(self.m.ceiling, expected, f"at t={t}")
        # Saturates at max and stays there while held.
        drive(self.m, 1.0, 0.0, t, t + 2.0)
        self.assertEqual(self.m.ceiling, JOY_MAX_CEILING)

    def test_hysteresis_band_holds_gear(self):
        t = wind_to_max(self.m)
        # Between the release (0.70) and engage (0.95) thresholds: hold.
        drive(self.m, 0.8, 0.0, t, t + 3.0)
        self.assertEqual(self.m.ceiling, JOY_MAX_CEILING)

    def test_backoff_winds_down_to_base(self):
        t = wind_to_max(self.m)
        # Below the release threshold the ceiling sheds one step per
        # JOY_STEP_DOWN_S -- faster than it wound up -- and floors at base.
        steps = JOY_MAX_CEILING - JOY_BASE_CEILING
        drive(self.m, 0.5, 0.0, t, t + JOY_STEP_DOWN_S * steps + 0.1)
        self.assertEqual(self.m.ceiling, JOY_BASE_CEILING)
        drive(self.m, 0.5, 0.0, t + 2.0, t + 4.0)
        self.assertEqual(self.m.ceiling, JOY_BASE_CEILING)  # never below base

    def test_release_snaps_back_to_base(self):
        t = wind_to_max(self.m)
        drive(self.m, 0.0, 0.0, t, t + JOY_IDLE_RESET_S + 0.05)
        self.assertEqual(self.m.ceiling, JOY_BASE_CEILING)

    def test_hard_reversal_sheds_gears(self):
        t = wind_to_max(self.m)
        az, _ = self.m.update(-1.0, 0.0, now=t + 0.05)
        self.assertEqual(self.m.ceiling, JOY_MAX_CEILING - JOY_REVERSAL_DROP)
        self.assertEqual(az, -(JOY_MAX_CEILING - JOY_REVERSAL_DROP))

    def test_service_gap_resets(self):
        # RATE_CONTROL not serviced (mode switch / STOP / disconnect): the
        # earned gear is stale context and must not survive.
        t = wind_to_max(self.m)
        self.m.update(1.0, 0.0, now=t + JOY_STALE_RESET_S + 1.0)
        self.assertEqual(self.m.ceiling, JOY_BASE_CEILING)

    def test_ceiling_is_shared_across_axes(self):
        # Pin az to earn top gear; a light el deflection then scales against
        # the shared ceiling (one "gear" for the whole stick).
        t = wind_to_max(self.m)
        _, alt = self.m.update(1.0, 0.3, now=t + 0.05)
        self.assertEqual(alt, axis_to_rate(0.3, JOY_MAX_CEILING))
        fresh_alt = AdaptiveRateMapper().update(0.0, 0.3, now=0.0)[1]
        self.assertEqual(fresh_alt, axis_to_rate(0.3, JOY_BASE_CEILING))
        self.assertLess(fresh_alt, alt)

    def test_constructor_overrides(self):
        m = AdaptiveRateMapper(base_ceiling=3, engage_delay_s=2.0)
        self.assertEqual(m.update(1.0, 0.0, now=0.0)[0], 3)
        drive(m, 1.0, 0.0, 0.0, 1.9)
        self.assertEqual(m.ceiling, 3)      # longer delay honored
        drive(m, 1.0, 0.0, 1.9, 2.05)
        self.assertEqual(m.ceiling, 4)


class _RecordingController:
    """Fake mount that records every hc_slew_fixed call."""

    def __init__(self):
        self.sent = []   # list of (target, rate)

    def hc_slew_fixed(self, target, rate):
        self.sent.append((target, rate))
        return True


class _StickJoystick:
    def __init__(self):
        self.axes = [0.0] * 6

    def get_numaxes(self):
        return 6

    def get_axis(self, i):
        return self.axes[i]

    def get_instance_id(self):
        return 0


class TestRateControlSend(unittest.TestCase):
    """The Python RATE_CONTROL path through the real tracking_control dispatch:
    gearbox-capped rates go out on change only, and -- the fix that matters in
    the field -- releasing the stick sends the one rate-0 command that actually
    stops the slew (the old rate!=0 guard never sent it)."""

    def setUp(self):
        from joystick_controller import JoystickModeState, TrackingMode, Targets
        self.Targets = Targets
        self.st = JoystickModeState()
        self.ctrl = _RecordingController()
        self.joy = _StickJoystick()
        self.st.telescope_connected = True
        self.st.telescope_controller = self.ctrl
        self.st.joysticks = {0: self.joy}
        self.st.connected_joystick = 0
        self.st.tracking_mode = TrackingMode.RATE_CONTROL

    def az_sends(self):
        return [r for t, r in self.ctrl.sent if t == self.Targets.AZM]

    def test_send_on_change_and_stop_on_release(self):
        # Held pinned: the base-gear rate goes out once, not every cycle.
        self.joy.axes[2] = 1.0
        for _ in range(3):
            self.st.tracking_control()
        self.assertEqual(self.az_sends(), [JOY_BASE_CEILING])

        # Released: exactly one stop (rate 0) goes out.
        self.joy.axes[2] = 0.0
        for _ in range(3):
            self.st.tracking_control()
        self.assertEqual(self.az_sends(), [JOY_BASE_CEILING, 0])

    def test_unstop_resends_current_rate(self):
        self.joy.axes[2] = 1.0
        self.st.tracking_control()
        # Universal STOP writes zeros directly, invalidating the cache...
        self.st.stopped = True
        self.st.tracking_control()
        self.assertEqual(self.az_sends()[-1], 0)
        # ...so releasing STOP with the stick still held must re-send.
        self.st.stopped = False
        self.st.tracking_control()
        self.assertEqual(self.az_sends()[-1], JOY_BASE_CEILING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
