#!/usr/bin/env python3
"""
PID position-error sign/consistency tests for compute_mount_position_error.
Converted from a print-and-eyeball script to real asserts.
"""

import unittest

from control import compute_mount_position_error, sky_target_to_mount
from transformations import AzEl2AzAlt_AltAz


class MockConfigState:
    def __init__(self, alignment_az=45.0, alignment_el=30.0):
        self.alignment_azimuth_str = str(alignment_az)
        self.alignment_elevation_str = str(alignment_el)
        self.mount_mode = 'AltAz'
        self.pointing_model_enabled = False


class PidErrorTests(unittest.TestCase):

    def setUp(self):
        self.config = MockConfigState()

    def test_zero_error_at_target(self):
        # Telescope already at the mount coordinates of the target -> ~0 error.
        target_azm, target_alt = sky_target_to_mount(self.config, 45.0, 30.0)
        azm_err, alt_err = compute_mount_position_error(
            self.config, target_azm, target_alt, 45.0, 30.0)
        self.assertAlmostEqual(azm_err, 0.0, places=9)
        self.assertAlmostEqual(alt_err, 0.0, places=9)

    def test_azm_error_sign(self):
        # Telescope 10 deg past the target in mount AZM -> negative correction.
        target_azm, target_alt = sky_target_to_mount(self.config, 45.0, 30.0)
        azm_err, _ = compute_mount_position_error(
            self.config, target_azm + 10.0, target_alt, 45.0, 30.0)
        self.assertAlmostEqual(azm_err, -10.0, places=9)

    def test_alt_error_sign(self):
        # Telescope 5 deg short of the target in mount ALT -> positive correction.
        target_azm, target_alt = sky_target_to_mount(self.config, 45.0, 30.0)
        _, alt_err = compute_mount_position_error(
            self.config, target_azm, target_alt - 5.0, 45.0, 30.0)
        self.assertAlmostEqual(alt_err, 5.0, places=9)

    def test_errors_are_shortest_arc(self):
        # Across the wrap: current 350, target mount 10 -> +20, never +340.
        cfg = MockConfigState(alignment_az=0.0, alignment_el=0.0)
        target_azm, target_alt = sky_target_to_mount(cfg, 10.0, 30.0)
        azm_err, _ = compute_mount_position_error(
            cfg, target_azm + 340.0, target_alt, 10.0, 30.0)
        self.assertAlmostEqual(azm_err, 20.0, places=6)

    def test_transform_matches_altaz_branch(self):
        # sky_target_to_mount in AltAz mode must be exactly AzEl2AzAlt_AltAz
        # (with the pointing model disabled).
        got = sky_target_to_mount(self.config, 120.0, 40.0)
        expected = AzEl2AzAlt_AltAz(120.0, 40.0, 45.0, 30.0)
        self.assertAlmostEqual(got[0], expected[0], places=9)
        self.assertAlmostEqual(got[1], expected[1], places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
