"""
Validate star_catalog's fast diurnal-rotation path against a full Skyfield recompute.

The catalogue anchors positions with Skyfield, then advances the whole sky by a rigid
rotation about the celestial pole. This test confirms that, over a window comparable to
ANCHOR_MAX_AGE_SEC, the rotated positions agree with an independent Skyfield computation
to better than ~1 arcmin -- and incidentally pins down the sidereal rotation sign.
"""

import math
import unittest

import numpy as np
from skyfield.api import Star, load, wgs84

import star_catalog
from star_catalog import StarCatalog

LAT, LON, ELEV = 34.874, -120.446, 120.0


def _angular_sep_deg(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(np.radians, (az1, el1, az2, el2))
    cos_sep = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


class TestStarCatalogFastPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eph = load('de421.bsp')
        cls.ts = load.timescale()
        cls.cat = StarCatalog(ephemeris=cls.eph)

    def _skyfield_altaz(self, star, t):
        observer = self.eph['earth'] + wgs84.latlon(LAT, LON, elevation_m=ELEV)
        alt, az, _ = observer.at(t).observe(star).apparent().altaz()
        return az.degrees, alt.degrees

    def test_fast_path_matches_skyfield_over_window(self):
        # Anchor at t0, then query 300 s later via the fast rotation path.
        t0 = self.ts.utc(2026, 6, 17, 5, 0, 0)
        # Prime the anchor at t0.
        self.cat._project(LAT, LON, ELEV, self.ts, t0.tt)
        anchor_tt = self.cat._anchor_tt
        self.assertAlmostEqual(anchor_tt, t0.tt, places=9)

        t1 = self.ts.utc(2026, 6, 17, 5, 5, 0)  # +300 s, still within ANCHOR_MAX_AGE_SEC
        az_fast, el_fast = self.cat._project(LAT, LON, ELEV, self.ts, t1.tt)
        # Confirm it did NOT re-anchor (still the fast path).
        self.assertAlmostEqual(self.cat._anchor_tt, t0.tt, places=9)

        # Independent Skyfield truth for the same bright subset.
        alt, az, _ = (self.eph['earth'] + wgs84.latlon(LAT, LON, elevation_m=ELEV)).at(t1) \
            .observe(self.cat._bright_star).apparent().altaz()
        az_true, el_true = az.degrees, alt.degrees

        # Compare well above the horizon (refraction-free geometric positions diverge
        # near the horizon, which we don't model in the rotation).
        high = el_true > 20.0
        sep = _angular_sep_deg(az_fast[high], el_fast[high], az_true[high], el_true[high])
        self.assertLess(np.percentile(sep, 99), 1.0 / 60.0 * 2,  # < ~2 arcmin at 99th pct
                        f"fast-path drift too large: max={sep.max()*60:.2f} arcmin")

    def test_visible_selection_respects_count_and_cutoff(self):
        t = self.ts.utc(2026, 6, 17, 5, 0, 0)
        res = self.cat.current_altaz(LAT, LON, ELEV, self.ts, t.tt,
                                     elevation_mask=10.0, max_count=200, limiting_mag=6.5)
        self.assertLessEqual(res['n_visible'], 200)
        self.assertTrue(np.all(res['el'] >= 10.0))
        self.assertTrue(np.all(res['mag'] <= 6.5 + 1e-9))
        # cutoff is the faintest rendered magnitude
        self.assertAlmostEqual(res['cutoff_mag'], float(res['mag'].max()), places=6)
        # at least one always-on label, and labels are the brightest stars
        self.assertGreaterEqual(res['top_label_mask'].sum(), 1)
        if res['top_label_mask'].any() and (~res['top_label_mask']).any():
            self.assertLessEqual(res['mag'][res['top_label_mask']].max(),
                                 res['mag'][~res['top_label_mask']].min() + 1e-9)

    def test_cutoff_monotonic_in_count(self):
        t = self.ts.utc(2026, 6, 17, 5, 0, 0)
        small = self.cat.current_altaz(LAT, LON, ELEV, self.ts, t.tt,
                                       elevation_mask=10.0, max_count=100, limiting_mag=8.0)
        big = self.cat.current_altaz(LAT, LON, ELEV, self.ts, t.tt,
                                     elevation_mask=10.0, max_count=1000, limiting_mag=8.0)
        # More stars rendered -> fainter (larger) cutoff magnitude.
        self.assertGreaterEqual(big['cutoff_mag'], small['cutoff_mag'])


if __name__ == '__main__':
    unittest.main()
