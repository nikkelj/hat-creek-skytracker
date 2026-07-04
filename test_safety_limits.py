#!/usr/bin/env python
"""
Safety-envelope regression tests.

Two behaviors introduced after the 2026-07 project review:

  * PROGRAM/launch target gating happens in the MOUNT frame. The
    azm/alt_limit_* config values gate encoder positions (that's how RATE and
    HOTSPOT use them), but PROGRAM used to gate the raw *sky* az/el against
    them. In AltAz mode mount ALT = 90 - el, so the old check rejected safe
    targets and passed unsafe ones.
  * MountControlThread._safe_stop_motion escalates instead of swallowing
    failure: it retries, and if the stop never succeeds it reports through the
    status callback (the mount may still be slewing on a dead link).

Run: python test_safety_limits.py
"""

import unittest

from control import compute_mount_position_error, sky_target_to_mount
from mount_control import MountControlThread


class DummyConfig:
    mount_mode = 'AltAz'
    alignment_azimuth_str = '0.0'
    alignment_elevation_str = '0.0'
    pointing_model_enabled = False
    eq_pointing_model_enabled = False
    lat_str = '35.0'


class SkyTargetToMountTests(unittest.TestCase):

    def test_altaz_alt_axis_runs_opposite_sky_elevation(self):
        cfg = DummyConfig()
        _, mount_alt_high_el = sky_target_to_mount(cfg, 120.0, 85.0)
        _, mount_alt_low_el = sky_target_to_mount(cfg, 120.0, 5.0)
        self.assertAlmostEqual(mount_alt_high_el, 5.0, places=6)
        self.assertAlmostEqual(mount_alt_low_el, 85.0, places=6)

    def test_mount_frame_gating_differs_from_sky_frame(self):
        # The scenario the fix exists for: mount ALT limits [0, 80].
        # A sky target at el=85 is mount ALT=5 -> SAFE (the old sky-frame check
        # wrongly rejected it); a sky target at el=5 is mount ALT=85 -> UNSAFE
        # (the old check wrongly allowed it).
        cfg = DummyConfig()
        alt_min, alt_max = 0.0, 80.0

        _, mount_alt = sky_target_to_mount(cfg, 120.0, 85.0)
        self.assertTrue(alt_min <= mount_alt <= alt_max,
                        "high-elevation sky target must be inside mount limits")

        _, mount_alt = sky_target_to_mount(cfg, 120.0, 5.0)
        self.assertFalse(alt_min <= mount_alt <= alt_max,
                         "low-elevation sky target must exceed mount ALT limit")

    def test_error_computation_consistent_with_transform(self):
        # compute_mount_position_error must be exactly "transform, then
        # shortest-arc difference" -- the refactor that exposed
        # sky_target_to_mount must not have changed the error path.
        cfg = DummyConfig()
        target_azm, target_alt = sky_target_to_mount(cfg, 120.0, 40.0)
        az_err, el_err = compute_mount_position_error(cfg, 100.0, 45.0, 120.0, 40.0)
        self.assertAlmostEqual(az_err, (target_azm - 100.0 + 180.0) % 360.0 - 180.0, places=9)
        self.assertAlmostEqual(el_err, (target_alt - 45.0 + 180.0) % 360.0 - 180.0, places=9)


class _FailingController:
    def __init__(self, fail_times=999):
        self.fail_times = fail_times
        self.calls = 0
        self.stopped_axes = []

    def hc_slew_fixed(self, target, rate):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise IOError("short read")
        self.stopped_axes.append((target, rate))
        return True


class _State:
    def __init__(self, controller):
        self.telescope_controller = controller
        self.telescope_connected = True
        self.status_messages = []
        self.update_status_callback = self.status_messages.append


class SafeStopEscalationTests(unittest.TestCase):

    def _thread(self, controller):
        state = _State(controller)
        t = MountControlThread(state, DummyConfig())
        return t, state

    def test_stop_success_returns_true_quietly(self):
        ctrl = _FailingController(fail_times=0)
        t, state = self._thread(ctrl)
        self.assertTrue(t._safe_stop_motion())
        self.assertEqual(state.status_messages, [])

    def test_stop_retries_through_transient_fault(self):
        # First attempt fails, second succeeds -> True, no operator alert.
        ctrl = _FailingController(fail_times=1)
        t, state = self._thread(ctrl)
        self.assertTrue(t._safe_stop_motion())
        self.assertEqual(state.status_messages, [])

    def test_persistent_stop_failure_alerts_operator(self):
        ctrl = _FailingController()  # never succeeds
        t, state = self._thread(ctrl)
        self.assertFalse(t._safe_stop_motion())
        self.assertEqual(len(state.status_messages), 1)
        self.assertIn("FAILED to stop motion", state.status_messages[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
