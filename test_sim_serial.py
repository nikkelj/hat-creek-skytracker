#!/usr/bin/env python
"""
Byte-level sim serial transport tests (simulator.SimSerialDevice).

The method-level SimMount bypasses the app's command encoders, response
parsers, timeouts, and SerialCommError recovery -- the exact seam where a
sim-invisible crash class lives. These tests drive the REAL
NexstarHandController (and the real MountControlThread fault policy) through
the AUX wire protocol against SimMount physics, with injected faults.

Headless, no hardware. Run: python test_sim_serial.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest

import numpy as np

from lib.auxstar import NexstarHandController, SerialCommError, Targets
from simulator import SimMount, SimSerialDevice, make_sim_serial_controller


class _Cfg:
    sim_config = {}


def _rig(az0=0.0, el0=0.0, seed=1):
    mount = SimMount(_Cfg(), az0_deg=az0, el0_deg=el0,
                     rng=np.random.default_rng(seed))
    dev = SimSerialDevice(mount, rng=np.random.default_rng(seed + 1))
    return mount, dev, NexstarHandController(dev)


class WireRoundTripTests(unittest.TestCase):

    def test_position_read_through_real_parser(self):
        _, _, ctrl = _rig(az0=123.5, el0=42.25)
        az = ctrl.hc_get_position(Targets.AZM) * 360.0
        el = ctrl.hc_get_position(Targets.ALT) * 360.0
        # 24-bit encoder LSB is ~2.1e-5 deg.
        self.assertAlmostEqual(az, 123.5, places=3)
        self.assertAlmostEqual(el, 42.25, places=3)

    def test_goto_through_wire_moves_mount(self):
        mount, _, ctrl = _rig()
        self.assertTrue(ctrl.hc_goto_fast(Targets.AZM, 200.0, 0, 0))
        self.assertTrue(ctrl.hc_goto_fast(Targets.ALT, 35.0, 0, 0))
        self.assertAlmostEqual(mount.az_true_deg, 200.0, places=3)
        self.assertAlmostEqual(mount.el_true_deg, 35.0, places=3)

    def test_slew_fixed_through_wire_sets_rate(self):
        mount, _, ctrl = _rig()
        self.assertTrue(ctrl.hc_slew_fixed(Targets.AZM, 9))
        self.assertAlmostEqual(mount.az_rate_dps, 10.0, places=6)  # RATES[9]
        self.assertTrue(ctrl.hc_slew_fixed(Targets.AZM, -5))
        self.assertAlmostEqual(mount.az_rate_dps, -0.5, places=6)  # -RATES[5]
        self.assertTrue(ctrl.hc_slew_fixed(Targets.AZM, 0))
        self.assertEqual(mount.az_rate_dps, 0.0)

    def test_variable_guide_rate_through_wire(self):
        # End-to-end over the encoding whose '{:06x}' crash shipped unseen:
        # real encoder -> wire bytes -> sim decode -> mount rate.
        mount, _, ctrl = _rig()
        self.assertTrue(ctrl.hc_set_rate_dps(Targets.AZM, 0.25))
        self.assertAlmostEqual(mount.az_rate_dps, 0.25, places=4)
        self.assertTrue(ctrl.hc_set_rate_dps(Targets.ALT, -1.5))
        self.assertAlmostEqual(mount.el_rate_dps, -1.5, places=4)

    def test_focus_axis_through_wire(self):
        mount, _, ctrl = _rig()
        ctrl.hc_slew_fixed(Targets.FOCUS, 9)
        time.sleep(0.05)
        pos = ctrl.hc_get_position(Targets.FOCUS) * 360.0
        ctrl.hc_slew_fixed(Targets.FOCUS, 0)
        self.assertGreater(pos, 0.0)

    def test_get_version(self):
        _, _, ctrl = _rig()
        self.assertEqual(ctrl.hc_get_version(Targets.AZM), "040f")


class FaultInjectionTests(unittest.TestCase):

    def test_short_read_raises_and_flushes(self):
        _, dev, ctrl = _rig()
        dev.fail_next(1)
        with self.assertRaises(SerialCommError):
            ctrl.hc_get_position(Targets.AZM)
        self.assertEqual(dev.reset_count, 1, "input buffer must be flushed")
        # Next transaction is clean again.
        az = ctrl.hc_get_position(Targets.AZM)
        self.assertIsInstance(az, float)

    def test_probabilistic_short_reads_recovered(self):
        _, dev, ctrl = _rig(seed=42)
        dev.short_read_prob = 0.3
        errors = successes = 0
        for _ in range(200):
            try:
                ctrl.hc_get_position(Targets.AZM)
                successes += 1
            except SerialCommError:
                errors += 1
        self.assertGreater(errors, 10, "fault injection should fire")
        self.assertGreater(successes, 50, "most transactions should survive")

    def test_garbage_ack_reported_as_failure_not_crash(self):
        # A corrupted ack must make the command return False (response != '#'),
        # never crash the parser.
        _, dev, ctrl = _rig(seed=7)
        dev.garbage_prob = 1.0
        results = [ctrl.hc_slew_fixed(Targets.AZM, 3) for _ in range(20)]
        self.assertIn(False, results)

    def test_latency_is_survivable(self):
        _, dev, ctrl = _rig()
        dev.latency_s = 0.02
        t0 = time.perf_counter()
        ctrl.hc_get_position(Targets.AZM)
        self.assertGreaterEqual(time.perf_counter() - t0, 0.02)


class ControlThreadFaultPolicyTests(unittest.TestCase):
    """The real MountControlThread against the byte-level sim: sustained comm
    faults must trip the consecutive-fault watchdog and stop motion."""

    class _State:
        def __init__(self, ctrl):
            self.telescope_connected = True
            self.telescope_controller = ctrl
            self.tracking_vis_state = None
            self.status_messages = []
            self.update_status_callback = self.status_messages.append
            self.current_azm = self.current_alt = 0.0
            self.current_azm_raw = self.current_alt_raw = 0.0
            self.position_fresh = False
            self.azm_display_str = self.alt_display_str = ""

        def tracking_control(self):
            pass

        def _poll_focus_position(self):
            pass

    def test_sustained_faults_trip_safe_stop(self):
        from mount_control import MountControlThread

        mount, dev, ctrl = _rig()
        mount.hc_slew_fixed(Targets.AZM, 9)  # motion in progress
        state = self._State(ctrl)

        class _Cfg2:
            azm_offset_str = "0.0"
            alt_offset_str = "0.0"

        thread = MountControlThread(state, _Cfg2(), target_hz=50)
        dev.fail_next(10_000)  # link goes dead
        thread.start()
        time.sleep(0.4)        # >> 3 consecutive fault cycles at 50 Hz
        thread.stop()
        thread.join(timeout=2.0)

        # The watchdog's stop attempts also fail on the dead link, so the
        # operator alert must have fired (the P0 escalation behavior).
        self.assertTrue(
            any("FAILED to stop motion" in m for m in state.status_messages),
            f"expected stop-failure alert, got: {state.status_messages}")

    def test_faults_then_recovery_stops_cleanly(self):
        from mount_control import MountControlThread

        mount, dev, ctrl = _rig()
        mount.hc_slew_fixed(Targets.AZM, 9)
        state = self._State(ctrl)

        class _Cfg2:
            azm_offset_str = "0.0"
            alt_offset_str = "0.0"

        thread = MountControlThread(state, _Cfg2(), target_hz=50)
        dev.fail_next(8)       # transient outage, then the link heals
        thread.start()
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=2.0)

        # After the watchdog fired with a healed link, the mount must be
        # stopped (rate zeroed by the safe stop).
        self.assertEqual(mount.az_rate_dps, 0.0)
        self.assertEqual(mount.el_rate_dps, 0.0)


class FactoryTests(unittest.TestCase):

    def test_factory_returns_real_controller(self):
        mount = SimMount(_Cfg())
        ctrl = make_sim_serial_controller(mount)
        self.assertIsInstance(ctrl, NexstarHandController)
        ctrl.hc_goto_fast(Targets.AZM, 90.0, 0, 0)
        self.assertAlmostEqual(mount.az_true_deg, 90.0, places=3)
        ctrl.close()
        self.assertEqual(mount.az_rate_dps, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
