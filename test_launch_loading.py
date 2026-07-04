#!/usr/bin/env python3
"""
Launch-trajectory loading tests against the committed launches/ fixtures.
Converted from a print-only script (which returned success even on
questionable data) to real assertions.
"""

import os
import unittest

import numpy as np

from trajectory import read_launch_trajectories

LAUNCHES_DIR = os.path.join(os.path.dirname(__file__), "launches")


class LaunchLoadingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trajectories = read_launch_trajectories(LAUNCHES_DIR)
        cls.names = [n for n in cls.trajectories if not n.endswith("_arcs")]

    def test_fixture_launches_load(self):
        self.assertGreater(len(self.names), 0,
                           f"no launch trajectories parsed from {LAUNCHES_DIR}")

    def test_each_launch_is_well_formed(self):
        for name in self.names:
            with self.subTest(launch=name):
                trajectory_data, times = self.trajectories[name]
                self.assertGreater(len(trajectory_data), 1)
                self.assertEqual(len(trajectory_data), len(times))

                times = np.asarray(times, dtype=float)
                self.assertTrue(np.all(np.diff(times) >= 0),
                                "times must be non-decreasing")

                for point in (trajectory_data[0], trajectory_data[-1]):
                    # (time, alt, az, range_km, px, py, az_rate, el_rate, ...)
                    self.assertGreaterEqual(len(point), 6)
                    _, alt, az, rng = point[0], point[1], point[2], point[3]
                    self.assertTrue(-90.0 <= alt <= 90.0, f"alt {alt}")
                    self.assertTrue(-360.0 <= az <= 720.0, f"az {az}")
                    self.assertGreaterEqual(rng, 0.0, f"range {rng}")

    def test_arc_segments_present(self):
        # Every launch gets an <name>_arcs companion (possibly empty list).
        for name in self.names:
            with self.subTest(launch=name):
                self.assertIn(name + "_arcs", self.trajectories)


if __name__ == "__main__":
    unittest.main(verbosity=2)
