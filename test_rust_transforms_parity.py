#!/usr/bin/env python
"""
Cross-language parity for the Rust transforms port (skytracker_core) against
transformations.py, including the scipy-backed AzAlt2AzEl path.

Build first, then run:
    cd rust/skytracker_core && maturin develop --release
    python test_rust_transforms_parity.py
"""

import math
import unittest

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

import transformations as tf

try:
    import scipy  # noqa: F401
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

AZ = [0.0, 17.0, 45.0, 123.4, 200.0, 270.0, 359.5]
EL = [-20.0, -1.0, 0.0, 15.0, 45.0, 80.0]
ALIGN = [(0.0, 0.0), (10.0, 2.0), (350.0, -3.0)]
LATS = [0.0, 30.0, 45.0, 60.0]


def ang_close(a, b, tol=1e-6):
    # Both-NaN counts as agreement: the two impls degenerate identically when an
    # arcsin argument grazes +/-1 for non-physical input combinations.
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(((a - b + 180.0) % 360.0) - 180.0) < tol


def num_close(a, b, places=7):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) < 10.0 ** (-places)


def hour_close(a, b, tol=1e-6):
    return abs(((a - b + 12.0) % 24.0) - 12.0) < tol


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class RustTransformsParity(unittest.TestCase):

    def test_cartesian_from_az_el(self):
        for az in AZ:
            for el in EL:
                py = tf.cartesian_from_az_el(az, el)
                rs = rc.cartesian_from_az_el(az, el)
                for k in range(3):
                    self.assertAlmostEqual(py[k], rs[k], places=12)

    def test_az_el_from_cartesian(self):
        for az in AZ:
            for el in EL:
                v = tf.cartesian_from_az_el(az, el)
                pa, pe = tf.az_el_from_cartesian(v)
                ra, re = rc.az_el_from_cartesian(float(v[0]), float(v[1]), float(v[2]))
                self.assertTrue(ang_close(pa, ra), f"az {pa} vs {ra}")
                self.assertAlmostEqual(pe, re, places=9)

    def test_altaz_pair(self):
        for az in AZ:
            for el in EL:
                for (aa, ae) in ALIGN:
                    p = tf.AzEl2AzAlt_AltAz(az, el, aa, ae)
                    r = rc.AzEl2AzAlt_AltAz(az, el, aa, ae)
                    self.assertTrue(ang_close(p[0], r[0]), f"azm {p[0]} vs {r[0]}")
                    self.assertAlmostEqual(p[1], r[1], places=9)

                    p2 = tf.AzAlt2AzEl_AltAz(az, el, aa)
                    r2 = rc.AzAlt2AzEl_AltAz(az, el, aa)
                    self.assertTrue(ang_close(p2[0], r2[0]))
                    self.assertAlmostEqual(p2[1], r2[1], places=9)

    def test_passthrough(self):
        self.assertEqual(tf.AzEl2AzAlt_Passthrough(40, 50), rc.AzEl2AzAlt_Passthrough(40, 50))
        self.assertEqual(tf.AzAlt2AzEl_Passthrough(40, 50), rc.AzAlt2AzEl_Passthrough(40, 50))

    def test_altaz_side_pair(self):
        # Side-mounted AltAz (pole on the horizon at alignment_azimuth,
        # index-mark home: H = AZM + h0, dec = 90 - ALT). The Python side goes
        # through the generic numpy eq-basis; the Rust side is the closed form
        # -- they must agree everywhere except the exact frame poles, where
        # the swept angle is undefined (atan2 of signed zeros) and only the
        # well-defined axis is compared.
        for az in AZ:
            for el in EL:
                for a0 in (0.0, 10.0, 77.3, 200.0, 350.0):
                    for flip in (False, True):
                        p = tf.AzEl2AzAlt_AltAzSide(az, el, a0, flip=flip)
                        r = rc.AzEl2AzAlt_AltAzSide(az, el, a0, flip)
                        self.assertTrue(num_close(p[1], r[1], places=9),
                                        f"alt {p[1]} vs {r[1]}")
                        # ALT -> 0 or 180 puts the boresight along the pole,
                        # where AZM is undefined.
                        if 1e-9 < p[1] < 180.0 - 1e-9:
                            self.assertTrue(ang_close(p[0], r[0]),
                                            f"azm {p[0]} vs {r[0]}")

                        p2 = tf.AzAlt2AzEl_AltAzSide(az, el, a0, flip=flip)
                        r2 = rc.AzAlt2AzEl_AltAzSide(az, el, a0, flip)
                        self.assertTrue(num_close(p2[1], r2[1], places=9),
                                        f"el {p2[1]} vs {r2[1]}")
                        if abs(p2[1]) < 90.0 - 1e-9:  # sky az undefined at zenith/nadir
                            self.assertTrue(ang_close(p2[0], r2[0]),
                                            f"az {p2[0]} vs {r2[0]}")

    def test_wedge_pair(self):
        for az in AZ:
            for el in [e for e in EL if e > -90]:
                for lat in LATS:
                    p = tf.local_elev_az_to_telescope(el, az, lat_deg=lat)
                    r = rc.local_elev_az_to_telescope(el, az, lat_deg=lat)
                    self.assertTrue(num_close(p[0], r[0]), f"alt {p} {r}")
                    self.assertTrue(ang_close(p[1], r[1], 1e-5), f"az {p[1]} vs {r[1]}")

                    p2 = tf.telescope_to_local_elev_az(el, az, lat_deg=lat)
                    r2 = rc.telescope_to_local_elev_az(el, az, lat_deg=lat)
                    self.assertTrue(num_close(p2[0], r2[0]), f"el {p2} {r2}")
                    self.assertTrue(ang_close(p2[1], r2[1], 1e-5))

    def test_radec(self):
        for az in AZ:
            for el in [5.0, 30.0, 70.0]:
                for lat in LATS:
                    p = tf.altaz_local_to_radec(el, az, lat_deg=lat, lst_hours=17.89)
                    r = rc.altaz_local_to_radec(el, az, lat_deg=lat, lst_hours=17.89)
                    self.assertTrue(hour_close(p[0], r[0]), f"ra {p[0]} vs {r[0]}")
                    self.assertAlmostEqual(p[1], r[1], places=7)

    def test_apply_rotation(self):
        for az in AZ:
            for el in [0.0, 30.0, 60.0]:
                for rot in [0.0, 15.0, -40.0, 90.0]:
                    p = tf.apply_rotation_to_az_el(az, el, rot)
                    r = rc.apply_rotation_to_az_el(az, el, rot)
                    self.assertTrue(ang_close(p[0], r[0], 1e-6), f"az {p[0]} vs {r[0]}")
                    self.assertAlmostEqual(p[1], r[1], places=7)

    @unittest.skipUnless(_HAVE_SCIPY, "scipy not installed")
    def test_azalt_to_azel_general(self):
        # The scipy align_vectors path vs the Rust Rodrigues port.
        for azm in AZ:
            for alt in [0.0, 20.0, 60.0]:
                for (aa, ae) in ALIGN:
                    p = tf.AzAlt2AzEl(azm, alt, aa, ae)
                    r = rc.AzAlt2AzEl(azm, alt, aa, ae)
                    self.assertTrue(ang_close(p[0], r[0], 1e-6), f"az {p[0]} vs {r[0]}")
                    self.assertAlmostEqual(p[1], r[1], places=6, msg=f"el {p[1]} vs {r[1]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
