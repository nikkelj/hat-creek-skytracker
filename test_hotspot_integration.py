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

    def get_latest_raw(self):
        return self.raw


class _FakeCamera:
    def __init__(self, raw):
        self.thread = _FakeThread(raw)


class _FakeController:
    def __init__(self):
        self.cmds = []

    def hc_slew_fixed(self, target, rate):
        self.cmds.append((target, rate))
        return True


class HotspotIntegrationTests(unittest.TestCase):

    def _make_state(self, raw):
        from config import ConfigState
        cfg = ConfigState()
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
        state.hotspot_last_detection_time = time.time() - 5.0
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

        # A new frame (seq bump) is processed and commands again.
        cam.thread.latest_raw_seq += 1
        state.hotspot_track()
        self.assertGreater(len(state.telescope_controller.cmds), n_cmds)
        self.assertEqual(state.hotspot_status, "locked")


if __name__ == '__main__':
    unittest.main(verbosity=2)
