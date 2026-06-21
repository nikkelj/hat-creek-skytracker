"""
Equatorial mount transform (transformations.eq_mount_to_azel / azel_to_eq_mount): a pure
rotation about the polar axis. Validates round-trip, agreement with the textbook
hour-angle/declination -> alt/az formula when the pole is correctly set, the HA sign
convention (increasing HA moves a target westward), and that the control-loop Eq branch
no longer crashes (it was a stub that raised NameError).
"""

import math
import unittest

from transformations import eq_mount_to_azel, azel_to_eq_mount
from config import ConfigState
from control import compute_mount_position_error


def _standard_hadec_to_altaz(ha_deg, dec_deg, lat_deg):
    """Textbook HA/Dec -> Alt/Az (az from north through east), pole in the meridian."""
    H = math.radians(ha_deg); d = math.radians(dec_deg); phi = math.radians(lat_deg)
    sin_alt = math.sin(phi) * math.sin(d) + math.cos(phi) * math.cos(d) * math.cos(H)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    # azimuth measured from North, increasing toward East
    y = -math.cos(d) * math.sin(H)
    x = math.sin(d) * math.cos(phi) - math.cos(d) * math.cos(H) * math.sin(phi)
    az = math.degrees(math.atan2(y, x)) % 360.0
    return az, math.degrees(alt)


class TestEqTransform(unittest.TestCase):
    def test_round_trip(self):
        for az_p, alt_p in [(0.0, 40.0), (0.0, 52.0), (2.0, 38.0)]:
            for az, el in [(120, 30), (200, 60), (300, 15), (45, 75), (180, 50)]:
                ha, dec = azel_to_eq_mount(az, el, az_p, alt_p)
                raz, rel = eq_mount_to_azel(ha, dec, az_p, alt_p)
                self.assertAlmostEqual(raz % 360.0, az % 360.0, places=4)
                self.assertAlmostEqual(rel, el, places=4)

    def test_matches_textbook_when_pole_in_meridian(self):
        lat = 40.0
        for ha in (-60, -15, 0, 30, 90):
            for dec in (-20, 0, 35, 70):
                az, el = eq_mount_to_azel(ha, dec, 0.0, lat)
                eaz, eel = _standard_hadec_to_altaz(ha, dec, lat)
                self.assertAlmostEqual(el, eel, places=3, msg=f"el ha={ha} dec={dec}")
                # az can wrap; compare on the circle
                self.assertAlmostEqual(((az - eaz + 180) % 360) - 180, 0.0, places=3,
                                       msg=f"az ha={ha} dec={dec}")

    def test_pole_maps_to_zenith_distance_latitude(self):
        # The celestial pole (dec=90) sits due north at altitude=latitude.
        az, el = eq_mount_to_azel(123.0, 90.0, 0.0, 47.0)
        self.assertAlmostEqual(el, 47.0, places=4)
        self.assertAlmostEqual(((az - 0 + 180) % 360) - 180, 0.0, places=3)

    def test_increasing_ha_moves_west(self):
        # On the equator at the meridian (az=180 south), +HA should head toward west (az->270).
        az0, _ = eq_mount_to_azel(0.0, 0.0, 0.0, 40.0)
        az1, _ = eq_mount_to_azel(20.0, 0.0, 0.0, 40.0)
        self.assertAlmostEqual(az0, 180.0, places=3)
        self.assertGreater(az1, 180.0)  # moved toward the western horizon

    def test_control_eq_branch_does_not_crash(self):
        cfg = ConfigState()
        cfg.mount_mode = 'Eq'
        cfg.lat_str = '40.0'
        cfg.alignment_azimuth_str = '0.0'
        cfg.alignment_elevation_str = '0.0'  # -> falls back to latitude
        # Regression: this path used to raise NameError (the Eq branch was a stub).
        az_err, alt_err = compute_mount_position_error(cfg, 10.0, 45.0, 200.0, 35.0)
        self.assertTrue(math.isfinite(az_err) and math.isfinite(alt_err))


if __name__ == '__main__':
    unittest.main(verbosity=2)
