#!/usr/bin/env python3
"""Regression tests for shortest-arc error on the modular AZM and ALT axes.

Both the AZM and ALT encoders are modular: hc_get_position() returns a fraction
of a revolution, so current_azm / current_alt live in [0, 360) (offset-shifted),
while compute_mount_position_error()'s targets come from the %360 transform.
The error on each axis MUST be reduced to the shortest arc (-180, 180], or the
PID drives the mount the long way around:

  * AZM: an azm_offset can push current_azm outside [0, 360); a one-shot +/-360
    wrap failed past ~1.5 turns and produced a ~360-deg "lap" on acquire.
  * ALT: in AltAz, el = 90 - ALT, so the boresight is at zenith when ALT ~= 0 --
    right on the 0/360 seam. On a near-overhead pass current_alt reads ~359 while
    target_alt is a few degrees, and an unwrapped error of ~-357 commanded a
    near-full 360 elevation traversal the mount cannot physically make.

These guard against both bugs recurring (they bit AZM then ALT). The reference
is the pure modular reduction; the production code must match it for any input.
"""

import unittest

from control import compute_mount_position_error


def shortest_arc(deg):
    """True shortest-arc reduction to (-180, 180] -- the reference behavior."""
    return (deg + 180.0) % 360.0 - 180.0


class MockConfigState:
    """Minimal config for compute_mount_position_error in AltAz/Passthrough.

    alignment 0/0 keeps the transform a clean identity-plus-complement so the
    expected errors are easy to reason about:
        target_azm = target_sky_az % 360
        target_alt = 90 - target_sky_el
    """

    def __init__(self, mount_mode="AltAz", alignment_az=0.0, alignment_el=0.0):
        self.mount_mode = mount_mode
        self.alignment_azimuth_str = str(alignment_az)
        self.alignment_elevation_str = str(alignment_el)
        self.pointing_model_enabled = False


class TestAzmShortestPath(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfigState()

    def test_straddle_zero_seam(self):
        # current 358, target 2 -> short path is +4, not -356.
        az_err, _ = compute_mount_position_error(self.cfg, 358.0, 45.0, 2.0, 45.0)
        self.assertAlmostEqual(az_err, 4.0, places=6)

    def test_straddle_zero_seam_other_way(self):
        az_err, _ = compute_mount_position_error(self.cfg, 2.0, 45.0, 358.0, 45.0)
        self.assertAlmostEqual(az_err, -4.0, places=6)

    def test_offset_shifted_negative_current(self):
        # azm_offset can drive current_azm negative (mount_control applies it
        # before this function sees it). current -190 (== 170) to target 175 is
        # +5, NOT a long lap. This is the original AZM "360 lap" case.
        az_err, _ = compute_mount_position_error(self.cfg, -190.0, 45.0, 175.0, 45.0)
        self.assertAlmostEqual(az_err, 5.0, places=6)

    def test_offset_more_than_full_turn_apart(self):
        # Raw diff > 540 deg: the old one-shot if/elif wrap returned +190 (then
        # clamped to +180 -> long way). Shortest arc is -170.
        # current -200 (== 160), target 350 -> +190 raw -> -170 short.
        az_err, _ = compute_mount_position_error(self.cfg, -200.0, 45.0, 350.0, 45.0)
        self.assertAlmostEqual(az_err, -170.0, places=6)


class TestAltShortestPath(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfigState()

    def test_near_zenith_seam(self):
        # sky el 88 -> target ALT = 2. Mount a hair below zenith reads ALT ~359.
        # Short, physical move is +3 across the seam; the bug commanded ~-357.
        _, alt_err = compute_mount_position_error(self.cfg, 90.0, 359.0, 120.0, 88.0)
        self.assertAlmostEqual(alt_err, 3.0, places=6)

    def test_near_zenith_seam_other_way(self):
        # sky el 85 -> target ALT = 5. current ALT 358 -> +7 short hop.
        _, alt_err = compute_mount_position_error(self.cfg, 90.0, 358.0, 120.0, 85.0)
        self.assertAlmostEqual(alt_err, 7.0, places=6)

    def test_no_wrap_when_far_from_seam(self):
        # Ordinary mid-sky geometry: error is the plain difference, unwrapped.
        # sky el 40 -> target ALT 50, current ALT 30 -> +20.
        _, alt_err = compute_mount_position_error(self.cfg, 90.0, 30.0, 120.0, 40.0)
        self.assertAlmostEqual(alt_err, 20.0, places=6)


class TestZeroErrorAtTarget(unittest.TestCase):
    def test_zero_error_both_axes(self):
        cfg = MockConfigState()
        # At the target: target_azm = 120, target_alt = 90 - 25 = 65.
        az_err, alt_err = compute_mount_position_error(cfg, 120.0, 65.0, 120.0, 25.0)
        self.assertAlmostEqual(az_err, 0.0, places=6)
        self.assertAlmostEqual(alt_err, 0.0, places=6)


class TestBothAxesAlwaysShortestArc(unittest.TestCase):
    """Sweep: for any current position, the error must equal the true shortest
    arc and never exceed 180 deg in magnitude -- on BOTH axes."""

    def test_sweep_matches_modular_reference(self):
        for mode in ("AltAz", "Passthrough"):
            cfg = MockConfigState(mount_mode=mode)
            for current in range(-360, 721, 7):        # well outside [0, 360)
                for target_sky_az in range(0, 360, 13):
                    for target_sky_el in range(0, 90, 11):
                        az_err, alt_err = compute_mount_position_error(
                            cfg, float(current), float(current),
                            float(target_sky_az), float(target_sky_el),
                        )
                        if mode == "AltAz":
                            target_azm = target_sky_az % 360.0
                            target_alt = 90.0 - target_sky_el
                        else:  # Passthrough: sky == mount
                            target_azm = float(target_sky_az)
                            target_alt = float(target_sky_el)
                        self.assertAlmostEqual(
                            az_err, shortest_arc(target_azm - current), places=6,
                            msg=f"AZM {mode} current={current} target_az={target_sky_az}")
                        self.assertAlmostEqual(
                            alt_err, shortest_arc(target_alt - current), places=6,
                            msg=f"ALT {mode} current={current} target_el={target_sky_el}")
                        self.assertLessEqual(abs(az_err), 180.0 + 1e-9)
                        self.assertLessEqual(abs(alt_err), 180.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
