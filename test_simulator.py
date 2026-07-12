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
from simulator import SimMount, HardwareSimulator, angles_to_pixel, injected_boresight


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
        # AltAz (the default): boresight sky el = 90 - mount ALT, so to point the
        # camera at sky el 30 the mount sits at ALT 60.
        sim.mount.az_true_deg, sim.mount.el_true_deg = 100.0, 60.0
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

    def test_stars_render_past_zenith(self):
        # Regression: ALT past zero tips the boresight just past the zenith, where
        # the AltAz transform el = 90 - ALT leaves [-90, 90] (ALT=355 -> el=-265).
        # Un-normalized, every star failed the visibility gate and a near-zenith
        # frame rendered empty even though the FOV was pointed at the sky.
        sim = self._sim((999.0, 999.0, None, False))  # stars only, no target
        sim.config_state.sim_config["star_density"] = 1500
        sim._stars = None  # force regen at the higher density
        bg = float(sim.config_state.sim_config.get("background_level", 6.0))
        thr = bg + 15.0

        def star_pixels(az, alt):
            sim.mount.az_true_deg, sim.mount.el_true_deg = az, alt
            return int((sim.render_frame(0) > thr).sum())

        baseline = star_pixels(100.0, 60.0)     # el 30, well above horizon (sanity)
        past_zenith = star_pixels(0.0, 355.0)    # ALT 355 == -5 -> el 85, near zenith

        self.assertGreater(baseline, 0, "sanity: stars should render above horizon")
        self.assertGreater(past_zenith, 0,
                           "stars must render with ALT past zero (boresight near zenith)")


class InjectedErrorTests(unittest.TestCase):
    """The sim can inject a full pointing model + refraction so an alignment run recovers
    all seven terms end-to-end (not just the encoder-bias IA/IE)."""

    def test_no_injection_is_identity(self):
        cfg = ConfigState()
        az, el = injected_boresight(cfg, 123.0, 45.0)
        self.assertAlmostEqual(az, 123.0, places=9)
        self.assertAlmostEqual(el, 45.0, places=9)

    def test_pointing_model_matches_predict_observed(self):
        from pointing_model import PointingModel
        terms = {"IA": 0.4, "IE": -0.2, "AN": 0.05, "AW": -0.03, "NPAE": 0.04, "CA": 0.02, "TF": 0.03}
        cfg = ConfigState()
        cfg.sim_config["mount_pointing_model"] = terms
        exp_az, exp_el = PointingModel(terms).predict_observed(80.0, 50.0)
        az, el = injected_boresight(cfg, 80.0, 50.0)
        self.assertAlmostEqual(az, exp_az, places=9)
        self.assertAlmostEqual(el, exp_el, places=9)

    def test_refraction_lifts_elevation(self):
        from pointing_model import bennett_refraction_deg
        cfg = ConfigState()
        cfg.sim_config["sim_refraction"] = True
        az, el = injected_boresight(cfg, 200.0, 25.0)
        self.assertAlmostEqual(az, 200.0, places=9)
        self.assertAlmostEqual(el, 25.0 + bennett_refraction_deg(25.0), places=9)
        self.assertGreater(el, 25.0)  # apparent is higher than geometric

    def test_render_shifts_field_by_injected_model(self):
        """A large injected IA shifts the rendered boresight, so a target placed at the
        nominal boresight no longer lands at frame center."""
        cfg = ConfigState()
        cfg.sim_config["enabled"] = True
        cfg.sim_config["mount_encoder_noise_deg"] = 0.0
        cfg.sim_config["read_noise"] = 0.0
        sim = HardwareSimulator(cfg, None, None)
        sim.mount.az_true_deg, sim.mount.el_true_deg = 100.0, 60.0  # nominal sky boresight az100 el30
        # Target sits exactly at the NOMINAL boresight (az 100, el 30).
        sim.current_target_azel = lambda: (100.0, 30.0, 'satellite', True)

        cfg.sim_config["mount_pointing_model"] = {}
        d0 = detect_hotspot(sim.render_frame(0), snr_threshold=5.0)
        self.assertIsNotNone(d0)
        h, w = cfg.sim_config["cam_height"], cfg.sim_config["cam_width"]
        self.assertLess(abs(d0.cx - w / 2) + abs(d0.cy - h / 2), 3.0)  # centered, no model

        cfg.sim_config["mount_pointing_model"] = {"IA": 0.3}  # ~0.3 deg azimuth shift
        d1 = detect_hotspot(sim.render_frame(0), snr_threshold=5.0)
        self.assertIsNotNone(d1)
        self.assertGreater(abs(d1.cx - w / 2) + abs(d1.cy - h / 2), 5.0)  # shifted off-center


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
        # This target is STATIC in the sky -- exactly what the star filter
        # rejects in bare mode (a static object IS star-like by definition).
        # The test exercises the centering loop, so opt out of the filter
        # (the operator's "stars OK" toggle).
        cfg.hotspot_star_filter_enabled = False

        target_az, target_el = 100.0, 30.0
        sim = HardwareSimulator(cfg, None, None)
        sim.current_target_azel = lambda: (target_az, target_el, 'satellite', True)
        # Start pointing off-target but within the wide-cam FOV. AltAz: mount ALT
        # for sky el 30 is 60; offset by 1 in ALT (= 1 in sky el) plus 2 in az.
        sim.mount.az_true_deg = target_az + 2.0
        sim.mount.el_true_deg = (90.0 - target_el) + 1.0

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

        # Measure the residual in sky coordinates (AltAz: sky el = 90 - mount ALT).
        final_err = math.hypot(
            simulator.wrap180(sim.mount.az_true_deg - target_az),
            (90.0 - sim.mount.el_true_deg) - target_el)

        self.assertTrue(state.hotspot_acquired, "should hold lock")
        self.assertLess(final_err, initial_err * 0.4,
                        f"pointing error should shrink: {initial_err:.2f} -> {final_err:.2f}")
        self.assertIsNotNone(last_centroid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
