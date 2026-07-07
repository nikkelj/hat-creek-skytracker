#!/usr/bin/env python
"""
SimCap exposure/readout timing model + frame-timestamp midpoint tests.

The real ASI camera holds ASI_EXP_WORKING for the exposure duration plus a
readout latency; the sim used to return SUCCESS instantly, so control-loop
latency was ~0 and frames could never be slower than the loop -- hiding the
whole stale-frame / windup regime. These tests pin the timing model and the
exposure-midpoint timestamp back-dating.

Headless. Run: python test_sim_exposure.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest
from datetime import datetime, timezone

import numpy as np
import zwoasi as asi

from camera_buffer import exposure_midpoint_utc
from simulator import SimCap


class _FakeSim:
    """SimCap only needs the simulator's _sim() config accessor."""

    def __init__(self, sim):
        self._s = sim

    def _sim(self):
        return self._s


def _simcap(sim_overrides=None, exposure_us=50_000):
    sim = {'enabled': True, 'cam_width': 64, 'cam_height': 48, 'seed': 3}
    sim.update(sim_overrides or {})
    cap = SimCap(0, _FakeSim(sim))
    cap.set_control_value(asi.ASI_EXPOSURE, exposure_us)
    return cap, sim


class ExposureTimingTests(unittest.TestCase):

    def test_idle_before_first_exposure(self):
        cap, _ = _simcap()
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_IDLE)

    def test_working_until_exposure_plus_readout_elapses(self):
        cap, _ = _simcap(sim_overrides={'sim_readout_latency_s': 0.02},
                         exposure_us=50_000)
        cap.start_exposure()
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_WORKING)
        time.sleep(0.03)  # inside exposure window (50 ms + 20 ms)
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_WORKING)
        time.sleep(0.06)  # past 70 ms total
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_SUCCESS)

    def test_exposure_setting_is_honored(self):
        # A shorter exposure completes sooner.
        cap, _ = _simcap(sim_overrides={'sim_readout_latency_s': 0.0},
                         exposure_us=5_000)
        cap.start_exposure()
        time.sleep(0.02)
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_SUCCESS)

    def test_timing_model_can_be_disabled(self):
        cap, _ = _simcap(sim_overrides={'sim_exposure_timing': False})
        # No start_exposure needed; legacy instant-success behavior.
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_SUCCESS)

    def test_failure_injection(self):
        cap, _ = _simcap(sim_overrides={'sim_exposure_failure_prob': 1.0})
        cap.start_exposure()
        self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_FAILED)

    def test_no_failures_by_default(self):
        cap, _ = _simcap(exposure_us=1_000)
        for _ in range(20):
            cap.start_exposure()
            time.sleep(0.007)
            self.assertEqual(cap.get_exposure_status(), asi.ASI_EXP_SUCCESS)


class MidpointTimestampTests(unittest.TestCase):

    def test_backdates_by_half_capture_time(self):
        now = datetime(2026, 7, 3, 12, 0, 10, tzinfo=timezone.utc)
        mid = exposure_midpoint_utc(1.0, now=now)
        self.assertEqual((now - mid).total_seconds(), 0.5)

    def test_zero_capture_time_is_now(self):
        now = datetime(2026, 7, 3, 12, 0, 10, tzinfo=timezone.utc)
        self.assertEqual(exposure_midpoint_utc(0.0, now=now), now)

    def test_negative_capture_time_clamped(self):
        now = datetime(2026, 7, 3, 12, 0, 10, tzinfo=timezone.utc)
        self.assertEqual(exposure_midpoint_utc(-5.0, now=now), now)


if __name__ == '__main__':
    unittest.main(verbosity=2)
