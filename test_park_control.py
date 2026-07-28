#!/usr/bin/env python
"""
Tests for the control-thread park sequence (_service_park).

The park button used to run a blocking goto/poll busy-loop on the UI thread
with no timeout and a non-wrap-aware convergence test (an offset near 0 with
the encoder reading 359.9 looped forever, frozen UI, serial contention with
the control thread). Now the UI only sets park_requested and the control
thread drives the sequence. These tests pin the new behavior: wrap-aware
convergence, command throttling, timeout, and completion reporting.

Headless. Run: python test_park_control.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest

import pygame
pygame.init()

from joystick_controller import JoystickModeState
from lib.auxstar import Targets


class _RecordingController:
    def __init__(self):
        self.gotos = []

    def hc_goto_fast(self, target, dd, mm, ss):
        self.gotos.append((target, dd))
        return True

    def hc_slew_fixed(self, target, rate):
        return True


class _Cfg:
    azm_offset_str = "10.0"
    alt_offset_str = "350.0"
    mount_mode = "AltAz"
    bias_control_mode = "coarse"


class ParkServiceTests(unittest.TestCase):

    def _state(self, cfg=None):
        cfg = cfg or _Cfg()
        self.messages = []
        state = JoystickModeState(None, cfg, self.messages.append)
        state.telescope_connected = True
        state.telescope_controller = _RecordingController()
        state.park_requested = True
        return state

    def test_issues_gotos_toward_offsets(self):
        state = self._state()
        state.current_azm_raw = 200.0
        state.current_alt_raw = 200.0
        state._service_park()
        self.assertTrue(state.park_requested, "still slewing - park stays active")
        self.assertIn((Targets.AZM, 10.0), state.telescope_controller.gotos)
        self.assertIn((Targets.ALT, 350.0), state.telescope_controller.gotos)

    def test_commands_are_throttled(self):
        state = self._state()
        state.current_azm_raw = 200.0
        state.current_alt_raw = 200.0
        state._service_park()
        n = len(state.telescope_controller.gotos)
        state._service_park()  # immediate next cycle: within reissue window
        self.assertEqual(len(state.telescope_controller.gotos), n)

    def test_wrap_aware_completion_across_seam(self):
        # The failure mode of the old UI busy-loop: target 0.2, encoder 359.9.
        # Shortest-arc error is 0.3 deg -> parked; the unwrapped difference is
        # 359.7 -> the old loop never terminated.
        class SeamCfg(_Cfg):
            azm_offset_str = "0.2"
            alt_offset_str = "0.0"
        state = self._state(SeamCfg())
        state.current_azm_raw = 359.9
        state.current_alt_raw = 0.4
        state._service_park()
        self.assertFalse(state.park_requested)
        self.assertEqual(state.telescope_controller.gotos, [])
        self.assertTrue(any("Park complete" in m for m in self.messages))

    def test_times_out_instead_of_looping_forever(self):
        state = self._state()
        state.current_azm_raw = 200.0
        state.current_alt_raw = 200.0
        state._service_park()  # arms _park_state
        # _service_park runs on time.perf_counter() (NTP-step immune), so the
        # backdate must use the same clock.
        state._park_state['start'] = time.perf_counter() - (state.PARK_TIMEOUT_SEC + 1)
        state._service_park()
        self.assertFalse(state.park_requested)
        self.assertTrue(any("TIMED OUT" in m for m in self.messages))


if __name__ == '__main__':
    unittest.main(verbosity=2)
