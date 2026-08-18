#!/usr/bin/env python
"""Phase 6a A/B parity: the Rust AdaptiveRateMapper + axis_to_rate vs the
Python originals, driven by identical deterministic stick timelines
(random walks with pins, releases, reversals, and stale gaps).
"""

import unittest

import numpy as np

try:
    import skytracker_core
    _HAVE = hasattr(skytracker_core, "RustAdaptiveRateMapper")
except ImportError:
    _HAVE = False

from joystick_controller import AdaptiveRateMapper, axis_to_rate


@unittest.skipUnless(_HAVE, "skytracker_core wheel lacks RustAdaptiveRateMapper")
class RateParity(unittest.TestCase):
    def test_axis_to_rate_exhaustive(self):
        for ceiling in (1, 3, 5, 9):
            for v in np.linspace(-1.2, 1.2, 4001):
                self.assertEqual(
                    axis_to_rate(float(v), ceiling),
                    skytracker_core.rust_axis_to_rate(float(v), ceiling),
                    f"axis {v} ceiling {ceiling}")
        print("\n[parity] axis_to_rate: 16004 samples identical")

    def test_gearbox_timelines(self):
        rng = np.random.default_rng(66)
        for trial in range(20):
            py = AdaptiveRateMapper()
            rs = skytracker_core.RustAdaptiveRateMapper()
            t = 0.0
            # Segments: pin / hold / release / reverse / gap.
            for _ in range(60):
                kind = rng.integers(0, 5)
                if kind == 0:
                    ax, ay = (1.0 if rng.random() < 0.5 else -1.0), 0.0
                elif kind == 1:
                    ax, ay = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
                elif kind == 2:
                    ax = ay = 0.0
                elif kind == 3:
                    ax, ay = 0.0, (1.0 if rng.random() < 0.5 else -1.0)
                else:
                    t += float(rng.uniform(1.1, 2.0))  # stale gap
                    ax, ay = rng.uniform(-1, 1), rng.uniform(-1, 1)
                dur = float(rng.uniform(0.1, 1.5))
                steps = max(1, int(dur / 0.066))
                for k in range(steps):
                    tk = t + k * 0.066
                    p = py.update(ax, ay, now=tk)
                    r = rs.update(ax, ay, tk)
                    self.assertEqual(tuple(p), tuple(r),
                                     f"trial {trial} t={tk:.3f} axes=({ax:.2f},{ay:.2f}) "
                                     f"py ceiling {py.ceiling} rs {rs.ceiling}")
                    self.assertEqual(py.ceiling, rs.ceiling)
                t += dur
        print("[parity] gearbox: 20 random timelines, every step identical")


if __name__ == "__main__":
    unittest.main(verbosity=2)
