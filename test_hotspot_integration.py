#!/usr/bin/env python
"""
Integration test for the HOTSPOT closed-loop wiring in JoystickModeState.

Drives hotspot_track() with a fake camera (synthetic blob) and a fake mount
controller (records slew commands) - no hardware, no real serial. Verifies:
  * a detected hot spot acquires lock and issues slew commands,
  * an off-center object commands motion (non-zero rate on at least one axis),
  * loss of lock past the coast window stops the mount and falls back to PROGRAM.

Headless. Run: python test_hotspot_integration.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest

import numpy as np
import pygame
pygame.init()

import joystick_controller as jc
from joystick_controller import JoystickModeState, TrackingMode
from lib.auxstar import Targets


def blob_frame(w=256, h=256, cx=160, cy=128, amp=220, sigma=3.0, bg=8.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = bg + amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    return img.astype(np.float32)


def noise_frame(w=256, h=256, seed=0):
    rng = np.random.default_rng(seed)
    return (8 + rng.normal(0, 1.0, (h, w))).astype(np.float32)


class _FakeThread:
    def __init__(self, raw):
        self.raw = raw
        self.latest_raw_seq = 0
        # Explicit frame timestamp (perf_counter-clock seconds). Tests set
        # this per frame so rate-filter dt math is deterministic instead of
        # depending on real sleep() jitter.
        self.latest_raw_time = None

    def get_latest_raw(self):
        return self.raw

    def get_latest_raw_with_meta(self):
        return self.raw, self.latest_raw_seq, self.latest_raw_time


class _FakeCamera:
    def __init__(self, raw):
        self.thread = _FakeThread(raw)


class _FakeController:
    def __init__(self):
        self.cmds = []

    def hc_slew_fixed(self, target, rate):
        self.cmds.append((target, rate))
        return True


class _FakeGuideController(_FakeController):
    """Adds the continuous guide-rate primitive (the field configuration)."""

    def __init__(self):
        super().__init__()
        self.rate_cmds = []

    def hc_set_rate_dps(self, target, dps):
        self.rate_cmds.append((target, dps))
        return True


def _restore_camera_manager():
    """Drop the per-instance get_camera override so the shared singleton's
    real class method is back for whatever test module runs next."""
    jc.camera_manager.__dict__.pop("get_camera", None)


class HotspotIntegrationTests(unittest.TestCase):

    def tearDown(self):
        _restore_camera_manager()

    def _make_state(self, raw):
        from config import ConfigState
        cfg = ConfigState()
        # These tests exercise lock/coast/fallback mechanics with a STATIC
        # blob and a static mount -- exactly what the star filter is built to
        # reject. Disable it here; StarFilterTests covers it explicitly.
        cfg.hotspot_star_filter_enabled = False
        self.cfg = cfg
        self.messages = []
        state = JoystickModeState(None, cfg, self.messages.append)
        state.telescope_connected = True
        state.telescope_controller = _FakeController()
        state.current_azm = 10.0
        state.current_alt = 30.0
        # Point the loop at our fake camera regardless of configured index.
        jc.camera_manager.get_camera = lambda idx: _FakeCamera(raw)
        return state

    def test_acquires_and_commands(self):
        state = self._make_state(blob_frame())
        state.tracking_mode = TrackingMode.HOTSPOT
        state._enter_hotspot_mode()
        state.hotspot_track()

        self.assertTrue(state.hotspot_acquired)
        self.assertEqual(state.hotspot_status, "locked")
        # Two commands issued (AZM, ALT), and the off-center object drives motion.
        targets = [t for (t, r) in state.telescope_controller.cmds]
        self.assertIn(Targets.AZM, targets)
        self.assertIn(Targets.ALT, targets)
        rates = [r for (t, r) in state.telescope_controller.cmds]
        self.assertTrue(any(r != 0 for r in rates), "expected non-zero slew rate")
        # Gate centered on the detected blob (~x=160).
        self.assertIsNotNone(state.hotspot_gate_center)
        self.assertLess(abs(state.hotspot_gate_center[0] - 160), 3.0)

    def test_loss_falls_back_to_program(self):
        state = self._make_state(blob_frame())
        state.tracking_mode = TrackingMode.HOTSPOT
        state._enter_hotspot_mode()
        state.hotspot_track()  # acquire lock
        self.assertTrue(state.hotspot_acquired)

        # Now feed noise (no detection) and pretend the coast window elapsed.
        jc.camera_manager.get_camera = lambda idx: _FakeCamera(noise_frame())
        # The hotspot loop clock is time.perf_counter() (frame/coast timing);
        # backdate on the same clock.
        state.hotspot_last_detection_time = time.perf_counter() - 5.0
        state.telescope_controller.cmds.clear()
        state.hotspot_track()

        self.assertEqual(state.tracking_mode, TrackingMode.PROGRAM)
        self.assertFalse(state.hotspot_acquired)
        self.assertIn((Targets.AZM, 0), state.telescope_controller.cmds)
        self.assertIn((Targets.ALT, 0), state.telescope_controller.cmds)

    def test_brief_dropout_coasts(self):
        state = self._make_state(blob_frame())
        state.tracking_mode = TrackingMode.HOTSPOT
        state._enter_hotspot_mode()
        state.hotspot_track()  # acquire

        # One missed frame, still within coast window -> stay locked, no fallback.
        jc.camera_manager.get_camera = lambda idx: _FakeCamera(noise_frame())
        state.hotspot_track()
        self.assertEqual(state.tracking_mode, TrackingMode.HOTSPOT)
        self.assertEqual(state.hotspot_status, "coasting")

    def test_stale_frame_is_not_reprocessed(self):
        # With real exposures the same frame stays "latest" across several
        # control cycles; re-detecting it would re-integrate the same centroid
        # (PID windup). A camera thread that exposes latest_raw_seq enables the
        # stale gate: same seq -> no new commands, no miss, lock retained.
        state = self._make_state(blob_frame())
        cam = _FakeCamera(blob_frame())
        cam.thread.latest_raw_seq = 1  # persistent camera with a frame counter
        jc.camera_manager.get_camera = lambda idx: cam

        state.tracking_mode = TrackingMode.HOTSPOT
        state._enter_hotspot_mode()
        state.hotspot_track()  # fresh frame: acquire + command
        self.assertTrue(state.hotspot_acquired)
        n_cmds = len(state.telescope_controller.cmds)
        self.assertGreater(n_cmds, 0)

        # Same seq across cycles: stale -> no detection, no commands, no miss.
        state.hotspot_track()
        state.hotspot_track()
        self.assertEqual(len(state.telescope_controller.cmds), n_cmds)
        self.assertEqual(state.hotspot_miss_count, 0)
        self.assertTrue(state.hotspot_acquired)

        # A new frame (seq bump) is processed and commands again. The blob
        # moves so the correction differs: the rate-command dedup layer
        # (deliberately) suppresses re-sends of an IDENTICAL rate, so a
        # same-position blob would be processed without a visible send.
        cam.thread.raw = blob_frame(cx=180, cy=110)
        cam.thread.latest_raw_seq += 1
        state.hotspot_track()
        self.assertGreater(len(state.telescope_controller.cmds), n_cmds)
        self.assertEqual(state.hotspot_status, "locked")


class StarFilterTests(unittest.TestCase):
    """Star-rejection rate gate: a detection whose implied sky rate doesn't
    match the program trajectory (or is star-like near-zero when tracking
    bare) must be rejected -- and the toggle must disable that. The filter
    verifies against a baseline candidate >= RATE_FILTER_BASELINE_S old
    (timing-skew robustness); tests shrink the baseline so two quick frames
    are enough."""

    def setUp(self):
        self._orig_baseline = jc.RATE_FILTER_BASELINE_S
        jc.RATE_FILTER_BASELINE_S = 0.02

    def tearDown(self):
        jc.RATE_FILTER_BASELINE_S = self._orig_baseline
        _restore_camera_manager()

    def _make_state(self, cam, sky_rates):
        from config import ConfigState
        cfg = ConfigState()
        self.cfg = cfg
        state = JoystickModeState(None, cfg, lambda m: None)
        state.telescope_connected = True
        state.telescope_controller = _FakeController()
        state.current_azm = 10.0
        state.current_alt = 30.0
        state._program_target_sky_rates = lambda: sky_rates
        jc.camera_manager.get_camera = lambda idx: cam
        return state

    def _fresh_detect_twice(self, state, cam):
        """Two _handoff_detect calls on FRESH frames far enough apart for a
        rate estimate. The static blob + static mount = implied rate ~0.
        Frame times are stamped explicitly (0.05 s apart) so the filter's dt
        is exact -- no real sleep, no timer-jitter flakiness."""
        t0 = time.perf_counter()
        cam.thread.latest_raw_seq = 1
        cam.thread.latest_raw_time = t0
        first = state._handoff_detect(self.cfg)
        cam.thread.latest_raw_seq = 2
        cam.thread.latest_raw_time = t0 + 0.05
        return first, state._handoff_detect(self.cfg)

    def test_static_blob_rejected_when_target_moves(self):
        cam = _FakeCamera(blob_frame())
        state = self._make_state(cam, (0.5, 0.0))  # trajectory says 0.5 deg/s
        first, second = self._fresh_detect_twice(state, cam)
        self.assertIsNone(first, "first detection is warm-up: no verdict")
        self.assertFalse(second, "static (star-like) blob must be rejected")
        self.assertIn("rejected", state.handoff_reject_reason)

    def test_static_blob_rejected_bare_mode(self):
        cam = _FakeCamera(blob_frame())
        state = self._make_state(cam, None)  # no program target
        _first, second = self._fresh_detect_twice(state, cam)
        self.assertFalse(second, "near-zero rate must read as a star")
        self.assertIn("star-like", state.handoff_reject_reason)

    def test_matching_rate_accepted(self):
        # Static blob + static mount = implied rate 0; a trajectory rate of 0
        # matches it, which is the same geometry as a perfectly-tracked target.
        cam = _FakeCamera(blob_frame())
        state = self._make_state(cam, (0.0, 0.0))
        _first, second = self._fresh_detect_twice(state, cam)
        self.assertTrue(second, "rate matching the trajectory must be accepted")

    def test_relative_gate_tolerates_fast_target_noise(self):
        # At 2 deg/s the acceptance threshold grows to 0.35 * |rate|, so the
        # timing-skew noise floor (which scales the same way) doesn't reject
        # the real target. Static blob + trajectory (0,0)... instead emulate:
        # implied error of 0.4 deg/s against a 2 deg/s trajectory must pass
        # (0.4 < 0.7), while the same absolute error against a slow 0.2 deg/s
        # trajectory must fail (0.4 > max(0.15, 0.07)).
        for rate, expect in ((2.0, True), (0.2, False)):
            cam = _FakeCamera(blob_frame())
            state = self._make_state(cam, (rate, 0.0))
            # Fake the boresight moving at (rate - 0.4): implied rate ends up
            # 0.4 deg/s short of the trajectory. dt is exact because the fake
            # frames carry explicit timestamps (the filter measures frame
            # time, not wall time).
            dt = 0.05
            t0 = time.perf_counter()
            cam.thread.latest_raw_seq = 1
            cam.thread.latest_raw_time = t0
            self.assertIsNone(state._handoff_detect(self.cfg))
            state.current_azm += (rate - 0.4) * dt
            cam.thread.latest_raw_seq = 2
            cam.thread.latest_raw_time = t0 + dt
            got = state._handoff_detect(self.cfg)
            self.assertEqual(bool(got), expect,
                             f"rate={rate}: expected accept={expect}")

    def test_toggle_off_accepts_stars(self):
        cam = _FakeCamera(blob_frame())
        state = self._make_state(cam, None)
        self.cfg.hotspot_star_filter_enabled = False
        first, second = self._fresh_detect_twice(state, cam)
        self.assertTrue(first and second, "filter disabled: stars are trackable")

    def test_stale_frame_returns_none(self):
        cam = _FakeCamera(blob_frame())
        state = self._make_state(cam, (0.0, 0.0))
        self.cfg.hotspot_star_filter_enabled = False  # isolate the stale gate
        cam.thread.latest_raw_seq = 1
        self.assertTrue(state._handoff_detect(self.cfg))
        # Same seq again: no new information -- neither count nor reset.
        self.assertIsNone(state._handoff_detect(self.cfg))


class HotspotFeedForwardTests(unittest.TestCase):
    """HOTSPOT rides its optical correction on the trajectory rates."""

    def tearDown(self):
        _restore_camera_manager()

    def test_centered_target_still_commands_trajectory_rate(self):
        from config import ConfigState
        cfg = ConfigState()
        cfg.hotspot_star_filter_enabled = False
        cfg.continuous_rate_tracking = True   # the field configuration
        state = JoystickModeState(None, cfg, lambda m: None)
        state.telescope_connected = True
        state.telescope_controller = _FakeGuideController()
        state.current_azm = 10.0
        state.current_alt = 30.0
        state.feed_forward_azm_enabled = True
        state.feed_forward_alt_enabled = True
        state._program_target_sky_rates = lambda: (1.2, 0.0)
        # Blob dead-center: zero optical error, so any commanded azimuth rate
        # comes from the trajectory feed-forward alone.
        cam = _FakeCamera(blob_frame(cx=128, cy=128))
        jc.camera_manager.get_camera = lambda idx: cam

        state.tracking_mode = TrackingMode.HOTSPOT
        state._enter_hotspot_mode()
        state.hotspot_track()

        self.assertTrue(state.hotspot_acquired)
        az_rates = [d for (t, d) in state.telescope_controller.rate_cmds
                    if t == Targets.AZM]
        self.assertTrue(az_rates, "expected a continuous guide-rate command")
        self.assertAlmostEqual(az_rates[-1], 1.2, delta=0.3,
                               msg="commanded azimuth rate should carry the "
                                   "1.2 deg/s trajectory feed-forward")
        self.assertGreater(state._hotspot_cmd_dps[0], 0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
