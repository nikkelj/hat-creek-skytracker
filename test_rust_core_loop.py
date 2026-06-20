#!/usr/bin/env python
"""
Drive the Rust control loop from Python via skytracker_core.SimCoreLoop.

Validates the message-passing surface (push inputs/frame/commands, pull
snapshot) end to end against the in-memory byte-level mount sim — the same loop
that runs on a background thread on hardware, exercised deterministically here.

Build first, then run:
    cd rust/skytracker_core && maturin develop --release
    python test_rust_core_loop.py
"""

import unittest

import numpy as np

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False


def gaussian_frame(w, h, cx, cy, amp=200.0, sigma=3.0, bg=10.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = bg + amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    return img.astype(np.float32)


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class SimCoreLoopTests(unittest.TestCase):

    def test_program_converges(self):
        loop = rc.SimCoreLoop(0.0, 0.0)
        loop.set_mode("program")
        loop.set_mount_mode("passthrough")
        loop.set_gains(0.025, 0.0, 0.0, 0.025, 0.0, 0.0)
        loop.set_setpoint(50.0, 30.0)
        for _ in range(400):
            loop.step(0.1)
        snap = loop.snapshot()
        self.assertTrue(snap["fresh"])
        self.assertAlmostEqual(snap["azm"], 50.0, delta=2.0)
        self.assertAlmostEqual(snap["alt"], 30.0, delta=2.0)

    def test_rate_mode_moves(self):
        loop = rc.SimCoreLoop(0.0, 0.0)
        loop.set_mode("rate")
        loop.set_rate_cmd(9, 0)
        for _ in range(20):
            loop.step(0.1)
        self.assertGreater(loop.az_true_deg, 0.0)

    def test_goto_command(self):
        loop = rc.SimCoreLoop(0.0, 0.0)
        loop.set_mode("standby")
        loop.submit_goto(123.0, 45.0)
        loop.step(0.1)
        self.assertAlmostEqual(loop.az_true_deg, 123.0, delta=0.5)
        self.assertAlmostEqual(loop.el_true_deg, 45.0, delta=0.5)

    def test_hotspot_locks_on_frame(self):
        loop = rc.SimCoreLoop(50.0, 45.0)
        loop.set_mode("hotspot")
        loop.set_gains(0.025, 0.0, 0.0, 0.025, 0.0, 0.0)
        frame = gaussian_frame(256, 200, 128 + 25, 100 + 15)
        loop.push_frame(frame)
        loop.step(0.1)
        snap = loop.snapshot()
        self.assertTrue(snap["hotspot_acquired"])
        self.assertEqual(snap["hotspot_status"], "locked")
        self.assertGreater(snap["hotspot_snr"], 5.0)
        self.assertIsNotNone(snap["hotspot_centroid"])

    def test_hotspot_no_frame_holds_then_falls_back(self):
        loop = rc.SimCoreLoop(50.0, 45.0)
        loop.set_mode("hotspot")
        # First frameless cycle on entry: the acquisition grace holds (it must NOT
        # bail immediately -- the async frame push can lag the mode change).
        loop.step(0.1)
        snap = loop.snapshot()
        self.assertIsNone(snap["requested_mode"])
        self.assertEqual(snap["hotspot_status"], "acquiring")
        # Past the grace window with still no detection -> fall back to PROGRAM.
        for _ in range(15):
            loop.step(0.1)
        snap = loop.snapshot()
        self.assertEqual(snap["requested_mode"], "program")
        self.assertEqual(snap["hotspot_status"], "lost")

    def test_hotspot_safety_limit(self):
        loop = rc.SimCoreLoop(50.0, 85.0)
        loop.set_mode("hotspot")
        loop.set_limits(0.0, 360.0, 0.0, 80.0)  # alt 85 is out of range
        loop.push_frame(gaussian_frame(256, 200, 138, 110))
        loop.step(0.1)
        snap = loop.snapshot()
        self.assertEqual(snap["requested_mode"], "standby")
        msgs = loop.drain_status()
        self.assertTrue(any("safety limit" in m for m in msgs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
