#!/usr/bin/env python
"""
AltAz-Side mount mode: the AVX lying on its side for center-of-gravity, so the
mount's AZM axis is HORIZONTAL (pointing at azimuth alignment_azimuth) and
sweeps a vertical great circle, while the ALT axis initially sweeps the
horizon. Geometrically an equatorial mount whose pole sits ON the horizon.

HOME CONVENTION (field-calibrated 2026-07-26): encoder (0, 0) = the AVX
index marks, where the scope points ALONG the horizontal polar axis -- at the
horizon, at azimuth alignment_azimuth -- and the dec axis stands vertical.
So H = AZM + 90 and dec = 90 - ALT (flip mirrors the H origin for a rig laid
on its other side).

Covers the transform pair's round trip, the physical semantics the rig was
built around, the control-path dispatch (sky_target_to_mount), and that the
axis-flip search stays disabled (the mirrored-sky trick has no second solution
in this geometry).

Headless. Run: python test_altaz_side.py
"""

import unittest

from transformations import AzAlt2AzEl_AltAzSide, AzEl2AzAlt_AltAzSide


class _Cfg:
    """Minimal config stub for control-path dispatch tests."""
    mount_mode = "AltAz-Side"
    alignment_azimuth_str = "30.0"
    alignment_elevation_str = "0.0"
    lat_str = "40.0"  # must be IGNORED in this mode (no latitude fallback)
    altaz_side_flip = False
    pointing_model_enabled = False
    eq_pointing_model_enabled = False


class TransformRoundTrip(unittest.TestCase):

    def test_round_trip_sky_to_mount_to_sky(self):
        for a0 in (0.0, 30.0, 145.7, 300.0):
            for az in range(0, 360, 30):
                for el in range(-75, 90, 15):
                    azm, alt = AzEl2AzAlt_AltAzSide(az, el, a0)
                    # ALT -> 0 or 180 means the boresight lies along the
                    # horizontal pole itself (dec +/-90), where mount azimuth
                    # is undefined (gimbal pole); skip the singular ring.
                    if min(alt % 360.0, 360.0 - (alt % 360.0)) < 0.5 or abs(alt - 180.0) < 0.5:
                        continue
                    raz, rel = AzAlt2AzEl_AltAzSide(azm, alt, a0)
                    self.assertAlmostEqual((raz - az + 180.0) % 360.0 - 180.0, 0.0,
                                           places=8, msg=f"az {az} el {el} a0 {a0}")
                    self.assertAlmostEqual(rel, el, places=8)


class RigSemantics(unittest.TestCase):
    """The behaviors the side-mounted rig is defined by (pole az = 0: the
    mount's horizontal AZM axis points due north). Home = AVX index marks."""

    def test_index_home_points_along_the_pole(self):
        # At the index marks the scope points along the horizontal polar
        # axis: at the horizon, at the pole azimuth. (First field session:
        # scope at index read horizon az 259 with Alignment Azimuth 259.)
        az, el = AzAlt2AzEl_AltAzSide(0.0, 0.0, 0.0)
        self.assertAlmostEqual(el, 0.0, places=9)
        self.assertAlmostEqual(az % 360.0, 0.0, places=9)

    def test_alt_axis_sweeps_the_horizon_from_home(self):
        # "What is normally Alt sweeps the horizon": from the index marks,
        # +ALT stays at elevation 0 and walks azimuth toward pole_az - 90.
        for a in (5.0, 30.0, 60.0, 120.0):
            az, el = AzAlt2AzEl_AltAzSide(0.0, a, 0.0)
            self.assertAlmostEqual(el, 0.0, places=9)
            self.assertAlmostEqual(az, (-a) % 360.0, places=9)

    def test_azm_axis_sweeps_a_vertical_circle(self):
        # "What is normally Az sweeps like Alt": with the boresight off the
        # pole (ALT=90 -> dec 0), AZM motion sweeps the vertical great circle
        # through the zenith; AZM=270 is the zenith itself.
        az, el = AzAlt2AzEl_AltAzSide(270.0, 90.0, 0.0)
        self.assertAlmostEqual(el, 90.0, places=9)
        for h in (10.0, 45.0, 80.0):
            az, el = AzAlt2AzEl_AltAzSide(270.0 + h, 90.0, 0.0)
            self.assertAlmostEqual(el, 90.0 - h, places=9)
            self.assertAlmostEqual(az, 270.0, places=9)  # pole_az - 90 side

    def test_flip_mirrors_the_alt_sweep(self):
        # A rig laid down on its other side: +ALT walks the horizon the
        # other way (toward pole_az + 90).
        for a in (5.0, 30.0, 60.0):
            az, el = AzAlt2AzEl_AltAzSide(0.0, a, 0.0, flip=True)
            self.assertAlmostEqual(el, 0.0, places=9)
            self.assertAlmostEqual(az, a % 360.0, places=9)

    def test_pole_azimuth_rotates_the_frame(self):
        # Rotating the rig (alignment_azimuth) rotates the whole sky mapping.
        az0, el0 = AzAlt2AzEl_AltAzSide(60.0, 25.0, 0.0)
        az1, el1 = AzAlt2AzEl_AltAzSide(60.0, 25.0, 40.0)
        self.assertAlmostEqual((az1 - az0) % 360.0, 40.0, places=9)
        self.assertAlmostEqual(el1, el0, places=9)


class ControlPathDispatch(unittest.TestCase):

    def test_sky_target_to_mount_uses_side_transform(self):
        from control import sky_target_to_mount
        cfg = _Cfg()
        got = sky_target_to_mount(cfg, 123.0, 40.0)
        want = AzEl2AzAlt_AltAzSide(123.0, 40.0, 30.0)
        self.assertAlmostEqual(got[0], want[0], places=9)
        self.assertAlmostEqual(got[1], want[1], places=9)

    def test_latitude_is_not_a_fallback(self):
        # Unlike Eq mode, alignment_elevation 0 must mean a pole exactly on
        # the horizon -- NOT "use site latitude".
        from control import sky_target_to_mount
        cfg = _Cfg()
        cfg.lat_str = "77.0"  # would shift the answer if it leaked in
        got = sky_target_to_mount(cfg, 200.0, 10.0)
        want = AzEl2AzAlt_AltAzSide(200.0, 10.0, 30.0)
        self.assertAlmostEqual(got[0], want[0], places=9)
        self.assertAlmostEqual(got[1], want[1], places=9)

    def test_flip_config_reaches_the_transform(self):
        from control import sky_target_to_mount
        cfg = _Cfg()
        cfg.altaz_side_flip = True
        got = sky_target_to_mount(cfg, 123.0, 40.0)
        want = AzEl2AzAlt_AltAzSide(123.0, 40.0, 30.0, flip=True)
        self.assertAlmostEqual(got[0], want[0], places=9)
        self.assertAlmostEqual(got[1], want[1], places=9)

    def test_no_flip_solution_offered(self):
        # The mirrored-sky (az+180, 180-el) trick maps to the SAME principal
        # axes in this geometry, so choose_mount_target must not flip.
        from control import choose_mount_target
        cfg = _Cfg()
        azm, alt, flipped = choose_mount_target(cfg, 0.0, 0.0, 123.0, 40.0)
        self.assertFalse(flipped)
        want = AzEl2AzAlt_AltAzSide(123.0, 40.0, 30.0)
        self.assertAlmostEqual(azm, want[0], places=9)
        self.assertAlmostEqual(alt, want[1], places=9)


class SimulatorPole(unittest.TestCase):

    def test_sim_boresight_matches_side_transform(self):
        # The simulator's true pole in AltAz-Side must sit on the horizon
        # (base_alt = 0), not at the site latitude.
        from config import ConfigState
        from simulator import HardwareSimulator
        cfg = ConfigState()
        cfg.mount_mode = "AltAz-Side"
        cfg.alignment_azimuth_str = "30.0"
        cfg.alignment_elevation_str = "0.0"
        cfg.lat_str = "40.0"
        sim = HardwareSimulator(cfg, None, None)
        sim.mount.az_true_deg = 60.0   # HA about the horizontal pole
        sim.mount.el_true_deg = 25.0   # dec from its equator
        frame = sim.render_frame(0)
        self.assertIsNotNone(frame)
        want_az, want_el = AzAlt2AzEl_AltAzSide(60.0, 25.0, 30.0)
        got_az, got_el = sim.last_boresight_azel
        self.assertAlmostEqual((got_az - want_az + 180.0) % 360.0 - 180.0,
                               0.0, places=6)
        self.assertAlmostEqual(got_el, want_el, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
