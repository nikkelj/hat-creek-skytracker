#!/usr/bin/env python
"""
Validate the THREADED Rust CoreLoop driving a Python mount object through the
PyMountTransport bridge (skytracker_core.CoreLoop.wrap_mount). This is the path
used for in-UI hardware-simulator validation: the Rust loop runs on a background
thread and drives the existing Python SimMount across the GIL.

Uses a small Python fake mount with the same Targets-enum interface as
simulator.SimMount, so no hardware and no full app are needed.

Build first, then run:
    cd rust/skytracker_core && maturin develop --release
    python test_rust_core_loop_bridge.py
"""

import threading
import time
import unittest

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

from lib.auxstar import RATES, Targets


class FakeMount:
    """Duck-typed NexstarHandController subset, like simulator.SimMount."""

    def __init__(self):
        self.az = 0.0
        self.el = 0.0
        self.az_rate = 0.0
        self.el_rate = 0.0
        self.t = time.time()
        self.lock = threading.Lock()

    def _advance(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        self.az = (self.az + self.az_rate * dt) % 360.0
        self.el += self.el_rate * dt

    def hc_get_position(self, target):
        with self.lock:
            self._advance()
            val = self.el if target == Targets.ALT else self.az
            return (val % 360.0) / 360.0

    def hc_slew_fixed(self, target, rate):
        with self.lock:
            self._advance()
            sign = 1.0 if rate >= 0 else -1.0
            dps = sign * RATES.get(abs(int(rate)), 0.0) * 360.0
            if target == Targets.ALT:
                self.el_rate = dps
            else:
                self.az_rate = dps
            return True

    def hc_goto_fast(self, target, dd, mm, ss):
        with self.lock:
            self._advance()
            deg = abs(dd) + mm / 60.0 + ss / 3600.0
            deg = deg if dd >= 0 else -deg
            if target == Targets.ALT:
                self.el = deg
                self.el_rate = 0.0
            else:
                self.az = deg % 360.0
                self.az_rate = 0.0
            return True


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class CoreLoopBridgeTests(unittest.TestCase):

    def test_program_converges_via_bridge(self):
        fm = FakeMount()
        loop = rc.CoreLoop.wrap_mount(fm, target_hz=200.0)
        try:
            loop.set_mode("program")
            loop.set_mount_mode("passthrough")
            loop.set_gains(0.025, 0.0, 0.0, 0.025, 0.0, 0.0)
            loop.set_setpoint(40.0, 25.0)

            converged = False
            deadline = time.time() + 10.0
            while time.time() < deadline:
                time.sleep(0.1)
                s = loop.snapshot()
                if abs(s["azm"] - 40.0) < 2.5 and abs(s["alt"] - 25.0) < 2.5:
                    converged = True
                    break
            snap = loop.snapshot()
        finally:
            loop.stop()
        self.assertTrue(converged, f"did not converge: {snap}")

    def test_goto_command_via_bridge(self):
        fm = FakeMount()
        loop = rc.CoreLoop.wrap_mount(fm, target_hz=100.0)
        try:
            loop.set_mode("standby")
            loop.submit_goto(100.0, 30.0)
            time.sleep(0.4)
            s = loop.snapshot()
        finally:
            loop.stop()
        self.assertAlmostEqual(s["azm"], 100.0, delta=1.0)
        self.assertAlmostEqual(s["alt"], 30.0, delta=1.0)

    def test_loop_publishes_rate_stats(self):
        fm = FakeMount()
        loop = rc.CoreLoop.wrap_mount(fm, target_hz=100.0)
        try:
            loop.set_mode("standby")
            time.sleep(1.3)  # let a 1s rate-stats window close
            s = loop.snapshot()
        finally:
            loop.stop()
        self.assertGreater(s["actual_hz"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
