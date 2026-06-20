#!/usr/bin/env python
"""
Performance regression guard: assert the Rust core stays at least as fast as the
Python baseline on the hot paths. Thresholds are deliberately conservative (well
below observed speedups) so the test is robust to machine load, but still fails
if a change makes the Rust core slower than the Python it replaces.

For the actual numbers, run bench_rust_vs_python.py instead.

    cd rust/skytracker_core && maturin develop --release
    python test_rust_perf.py
"""

import time
import unittest

import numpy as np

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

import transformations as tf
from control import PIDController
from hotspot import detect_hotspot as py_detect


def per_call(fn, n):
    for _ in range(max(1, n // 20)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def gaussian_frame(w, h, cx, cy):
    yy, xx = np.mgrid[0:h, 0:w]
    img = 10.0 + 200.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 9.0)))
    img = img + np.random.default_rng(1).normal(0.0, 1.0, img.shape)
    return np.clip(img, 0, None).astype(np.float32)


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class PerfGuard(unittest.TestCase):
    """Each test asserts rust_time * threshold <= python_time."""

    def _assert_faster(self, py_s, rs_s, min_speedup, label):
        speedup = py_s / rs_s if rs_s > 0 else float("inf")
        self.assertGreaterEqual(
            speedup,
            min_speedup,
            f"{label}: Rust only {speedup:.2f}x python "
            f"(py={py_s*1e9:.0f}ns rs={rs_s*1e9:.0f}ns); expected >= {min_speedup}x",
        )

    def test_detect_hotspot_not_slower(self):
        frame = gaussian_frame(640, 480, 330.0, 250.0)
        py = per_call(lambda: py_detect(frame, snr_threshold=5.0), 100)
        rs = per_call(lambda: rc.detect_hotspot(frame, snr_threshold=5.0), 100)
        # Observed ~1.5x; require at least parity with margin.
        self._assert_faster(py, rs, 1.1, "detect_hotspot")

    def test_transforms_much_faster(self):
        py = per_call(lambda: tf.local_elev_az_to_telescope(40.0, 123.0, lat_deg=45.0), 3000)
        rs = per_call(lambda: rc.local_elev_az_to_telescope(40.0, 123.0, lat_deg=45.0), 3000)
        # Observed ~58x; require a large margin.
        self._assert_faster(py, rs, 10.0, "local_elev_az_to_telescope")

    def test_scipy_transform_much_faster(self):
        py = per_call(lambda: tf.AzAlt2AzEl(123.0, 40.0, 10.0, 2.0), 1000)
        rs = per_call(lambda: rc.AzAlt2AzEl(123.0, 40.0, 10.0, 2.0), 1000)
        # Observed ~560x; require a large margin.
        self._assert_faster(py, rs, 20.0, "AzAlt2AzEl")

    def test_pid_step_faster(self):
        pp = PIDController(0.025, 0.01, 0.005, "AZM")
        rp = rc.PidController(0.025, 0.01, 0.005, "AZM")
        py = per_call(lambda: pp.compute_pid_output(5.0, 0.1, 100.0), 5000)
        rs = per_call(lambda: rp.compute_pid_output(5.0, 0.1, 100.0), 5000)
        # Observed ~110x (Python baseline is numpy-on-scalars heavy).
        self._assert_faster(py, rs, 5.0, "PID step")


if __name__ == "__main__":
    unittest.main(verbosity=2)
