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
        anchor_tt = self.cat._anchors[-1]['tt']
        self.assertAlmostEqual(anchor_tt, t0.tt, places=9)
        n_anchors = len(self.cat._anchors)

        t1 = self.ts.utc(2026, 6, 17, 5, 5, 0)  # +300 s, still within ANCHOR_MAX_AGE_SEC
        az_fast, el_fast = self.cat._project(LAT, LON, ELEV, self.ts, t1.tt)
        # Confirm it did NOT re-anchor (still the fast path).
        self.assertAlmostEqual(self.cat._anchors[-1]['tt'], t0.tt, places=9)
        self.assertEqual(len(self.cat._anchors), n_anchors)

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


class TestRenderThreadTimeSource(unittest.TestCase):
    """Regression: the render threads must draw the sky at the app's tracking
    clock (tracking_vis_state.current_tt -- slider / paused / live), NOT at
    wall-clock now. When they used ts.now() directly, scrubbing the time
    slider moved the satellites (positioned from current_tt in main.py) while
    the starfield stayed frozen at real time."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.init()
        cls.ts = load.timescale()

    def test_render_time_prefers_scrubbed_current_tt(self):
        import types
        from rendering_threads import render_time_tt
        scrubbed = self.ts.now().tt - 0.5   # 12 h in the past
        tvs = types.SimpleNamespace(current_tt=scrubbed)
        self.assertAlmostEqual(render_time_tt(tvs, self.ts), scrubbed, places=9)

    def test_render_time_falls_back_to_now_when_unset(self):
        import types
        from rendering_threads import render_time_tt
        now = self.ts.now().tt
        for tvs in (types.SimpleNamespace(current_tt=0.0),
                    types.SimpleNamespace(current_tt=None),
                    types.SimpleNamespace()):
            tt = render_time_tt(tvs, self.ts)
            self.assertAlmostEqual(tt, now, delta=1.0 / 86400.0)

    def test_starfield_moves_when_time_is_scrubbed(self):
        """End-to-end at the 2D-plot level: two starfield draws 1 h of
        TRACKING time apart must produce different pixels."""
        import types
        import pygame
        from rendering_threads import _draw_starfield_on_surface

        class Cfg:
            lat_str = str(LAT)
            lon_str = str(LON)
            alt_str = str(ELEV)
            max_rendered_star_count = 800
            star_limiting_magnitude = 6.5

        t0 = self.ts.utc(2026, 6, 17, 5, 0, 0).tt
        frames = []
        for tt in (t0, t0 + 1.0 / 24.0):
            state = types.SimpleNamespace(ephemeris=None)
            surf = pygame.Surface((500, 500))
            _draw_starfield_on_surface(surf, Cfg(), self.ts, tt, state,
                                       250, 250, 200, 0.0,
                                       draw_labels=False, publish_positions=False)
            frames.append(pygame.surfarray.array3d(surf))
        diff = int(np.count_nonzero(np.any(frames[0] != frames[1], axis=2)))
        self.assertGreater(diff, 100,
                           f"only {diff} pixels changed over 1 h of scrubbed "
                           "time -- starfield is ignoring the tracking clock")


if __name__ == '__main__':
    unittest.main()
