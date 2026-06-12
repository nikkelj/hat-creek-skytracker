#!/usr/bin/env python
"""
Tests for the hardware simulator (simulator.py): mount dynamics, geometry that is
the exact inverse of the tracker's, render->detect, and a sim-in-the-loop test
that closes SimMount -> rendered frame -> hotspot_track and asserts the object
converges to frame center.

Headless. Run: python test_simulator.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
import time
import unittest

import numpy as np
import pygame
pygame.init()

from config import ConfigState
from lib.auxstar import RATES, Targets
from hotspot import detect_hotspot, pixel_offset_to_angles
import simulator
from simulator import SimMount, HardwareSimulator, angles_to_pixel


class MountDynamicsTests(unittest.TestCase):

    def test_zero_rate_holds(self):
        m = SimMount(ConfigState(), az0_deg=100.0, el0_deg=30.0)
        time.sleep(0.05)
        self.assertAlmostEqual(m.hc_get_position(Targets.AZM) * 360.0, 100.0, places=3)

    def test_slew_advances_at_rate(self):
        m = SimMount(ConfigState(), az0_deg=10.0)
        m.hc_slew_fixed(Targets.AZM, 9)   # RATES[9] rev/s
        time.sleep(0.2)
        pos = m.hc_get_position(Targets.AZM) * 360.0
        expected = 10.0 + RATES[9] * 360.0 * 0.2
        self.assertAlmostEqual(pos, expected, delta=0.5)  # delta covers timing slop

    def test_misalignment_in_report(self):
        cfg = ConfigState()
        cfg.sim_config["mount_misalignment_az_deg"] = 3.0
        m = SimMount(cfg, az0_deg=50.0)
        self.assertAlmostEqual(m.hc_get_position(Targets.AZM) * 360.0, 53.0, places=3)

    def test_encoder_noise_bounded(self):
        cfg = ConfigState()
        cfg.sim_config["mount_encoder_noise_deg"] = 0.5
        m = SimMount(cfg, az0_deg=50.0)
        errs = [abs(m.hc_get_position(Targets.AZM) * 360.0 - 50.0) for _ in range(200)]
        self.assertLessEqual(max(errs), 0.5 + 1e-6)

    def test_goto_jumps(self):
        m = SimMount(ConfigState())
        m.hc_goto_fast(Targets.ALT, 42, 0, 0)
        self.assertAlmostEqual(m.hc_get_position(Targets.ALT) * 360.0, 42.0, places=3)


class GeometryTests(unittest.TestCase):

    def test_angles_to_pixel_is_inverse(self):
        px_um, fl_mm, rot, el = 4.0, 1000.0, 17.0, 35.0
        for dx, dy in [(123, -45), (-200, 80), (5, 300)]:
            az, elr = pixel_offset_to_angles(dx, dy, px_um, fl_mm, rotation_deg=rot,
                                             el_deg=el, x_sign=1.0, y_sign=-1.0)
            rdx, rdy = angles_to_pixel(az, elr, px_um, fl_mm, rotation_deg=rot,
                                       el_deg=el, x_sign=1.0, y_sign=-1.0)
            self.assertAlmostEqual(rdx, dx, places=4)
            self.assertAlmostEqual(rdy, dy, places=4)

    def test_boresight_maps_to_center(self):
        dx, dy = angles_to_pixel(0.0, 0.0, 4.0, 1000.0)
        self.assertAlmostEqual(dx, 0.0, places=6)
        self.assertAlmostEqual(dy, 0.0, places=6)


class RenderDetectTests(unittest.TestCase):

    def _sim(self, target):
        cfg = ConfigState()
        cfg.sim_config["enabled"] = True
        cfg.sim_config["mount_rate_noise_dps"] = 0.0
        cfg.sim_config["mount_encoder_noise_deg"] = 0.0
        sim = HardwareSimulator(cfg, None, None)
        sim.mount.az_true_deg, sim.mount.el_true_deg = 100.0, 30.0
        sim.current_target_azel = lambda: target
        return sim

    def test_satellite_dot_detected_in_wide_cam(self):
        # Target slightly off boresight so it lands off-center but in frame.
        sim = self._sim((100.5, 30.3, 'satellite', True))
        raw = sim.render_frame(0)
        d = detect_hotspot(raw, snr_threshold=5.0)
        self.assertIsNotNone(d)
        # off-center
        h, w = raw.shape
        self.assertGreater(abs(d.cx - w / 2) + abs(d.cy - h / 2), 3.0)

    def test_v2mini_detected_in_narrow_cam(self):
        sim = self._sim((100.02, 30.02, 'satellite', True))
        raw = sim.render_frame(1)
        d = detect_hotspot(raw, snr_threshold=5.0)
        self.assertIsNotNone(d)

    def test_streak_grows_with_rate(self):
        sim = self._sim((999.0, 999.0, None, False))  # no target, stars only
        static = sim.render_frame(0).astype(np.int32).sum()
        sim.mount._az_rate_dps = 5.0  # moving -> stars streak -> more lit pixels
        streaked = sim.render_frame(0).astype(np.int32).sum()
        self.assertGreater(streaked, static)


class SimInTheLoopTests(unittest.TestCase):

    def test_hotspot_converges_on_simulated_object(self):
        import joystick_controller as jc
        from joystick_controller import JoystickModeState, TrackingMode

        cfg = ConfigState()
        cfg.sim_config["enabled"] = True
        # Stable proportional control for the test.
        cfg.pid_azm_p_gain = cfg.pid_alt_p_gain = 0.3
        cfg.pid_azm_i_gain = cfg.pid_alt_i_gain = 0.0
        cfg.pid_azm_d_gain = cfg.pid_alt_d_gain = 0.0
        cfg.azm_offset_str = "0"; cfg.alt_offset_str = "0"

        target_az, target_el = 100.0, 30.0
        sim = HardwareSimulator(cfg, None, None)
        sim.current_target_azel = lambda: (target_az, target_el, 'satellite', True)
        # Start pointing off-target but within the wide-cam FOV.
        sim.mount.az_true_deg = target_az + 2.0
        sim.mount.el_true_deg = target_el + 1.0

        state = JoystickModeState(None, cfg, lambda m: None)
        state.hardware_sim = sim
        state.telescope_connected = True
        state.telescope_controller = sim.mount
        state.tracking_mode = TrackingMode.HOTSPOT
        state._enter_hotspot_mode()

        # Fake camera that renders the wide cam fresh on each read.
        class _Thread:
            def get_latest_raw(self_inner):
                return sim.render_frame(0)

        class _Cam:
            thread = _Thread()

        jc.camera_manager.get_camera = lambda idx: _Cam()

        initial_err = math.hypot(2.0, 1.0)
        last_centroid = None
        for _ in range(60):
            state.current_azm = sim.mount.hc_get_position(Targets.AZM) * 360.0
            state.current_alt = sim.mount.hc_get_position(Targets.ALT) * 360.0
            state.hotspot_track()
            last_centroid = state.hotspot_centroid
            time.sleep(0.03)

        final_err = math.hypot(
            simulator.wrap180(sim.mount.az_true_deg - target_az),
            sim.mount.el_true_deg - target_el)

        self.assertTrue(state.hotspot_acquired, "should hold lock")
        self.assertLess(final_err, initial_err * 0.4,
                        f"pointing error should shrink: {initial_err:.2f} -> {final_err:.2f}")
        self.assertIsNotNone(last_centroid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
