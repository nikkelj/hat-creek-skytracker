#!/usr/bin/env python
"""
Cross-language parity for the Rust PID port (skytracker_core.PidController).

Drives the real Python control.PIDController and the Rust port through the SAME
randomized input sequences across a sweep of gain/axis/feed-forward configs and
asserts the total output (rev/sec) and the discrete rate (0..9) agree at every
step. This is the strong check: identical dynamics over hundreds of cycles, not
just a handful of named scenarios.

Build first, then run:
    cd rust/skytracker-ffi && maturin develop --release
    python test_rust_pid_parity.py
"""

import random
import unittest

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

from control import PIDController


CONFIGS = [
    # (p, i, d, axis, feed_forward)
    (1.0, 0.0, 0.0, "ALT", False),
    (0.025, 0.01, 0.0, "ALT", False),
    (0.025, 0.0, 0.5, "ALT", False),
    (0.025, 0.01, 0.2, "AZM", False),
    (0.05, 0.005, 0.1, "AZM", True),
    (0.0, 0.0, 1.0, "AZM", False),
]


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class RustPidParity(unittest.TestCase):

    def test_step_for_step_parity(self):
        for (p, i, d, axis, ff) in CONFIGS:
            py = PIDController(p, i, d, axis, ff)
            rs = rc.PidController(p, i, d, axis, ff)
            if ff:
                py.set_feed_forward_rate(7.2)   # 0.02 rev/s
                rs.set_feed_forward_rate(7.2)

            rng = random.Random(12345)
            with self.subTest(config=(p, i, d, axis, ff)):
                for step in range(400):
                    err = rng.uniform(-30.0, 30.0)
                    # Alternate between error-mode and measurement-mode cycles,
                    # sweeping measurement across the AZM wrap boundary.
                    if step % 2 == 0:
                        meas = rng.uniform(0.0, 360.0)
                    else:
                        meas = None
                    po, pr = py.compute_pid_output(err, 0.1, measurement_degrees=meas)
                    ro, rr = rs.compute_pid_output(err, 0.1, meas)
                    self.assertAlmostEqual(po, ro, places=9,
                                           msg=f"output @step{step} cfg={(p,i,d,axis,ff)}")
                    self.assertEqual(pr, rr,
                                     msg=f"discrete rate @step{step} cfg={(p,i,d,axis,ff)}")

    def test_feed_forward_conversion(self):
        py = PIDController(feed_forward_enabled=True)
        rs = rc.PidController(feed_forward_enabled=True)
        py.set_feed_forward_rate(360.0)
        rs.set_feed_forward_rate(360.0)
        self.assertAlmostEqual(py.current_feed_forward_rate,
                               rs.current_feed_forward_rate, places=12)

    def test_integral_state_matches(self):
        py = PIDController(0.025, 0.01, 0.0, "ALT")
        rs = rc.PidController(0.025, 0.01, 0.0, "ALT")
        for _ in range(50):
            py.compute_pid_output(0.5, 0.1)
            rs.compute_pid_output(0.5, 0.1)
        self.assertAlmostEqual(py.integral_error, rs.integral_error, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
