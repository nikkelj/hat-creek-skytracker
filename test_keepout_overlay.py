#!/usr/bin/env python
"""Unit tests for the skyplot mount-keepout overlay (rendering_threads.py).

The azm/alt safety limits are mount-frame quantities; the overlay paints the
sky directions with NO in-limits mount-axis solution (matching what PROGRAM
track's safety gate would refuse). These tests pin the geometry per mount
mode, the flip-solution rescue in AltAz, cache behavior, and build cost.

Run: python test_keepout_overlay.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
import time
import unittest

import pygame
pygame.init()

from config import ConfigState
import rendering_threads as rt

RADIUS = 60


def make_cfg(**over):
    cfg = ConfigState()
    cfg.mount_mode = "AltAz"
    cfg.alignment_azimuth_str = "0.0"
    cfg.alignment_elevation_str = "0.0"
    cfg.azm_limit_min_str = "0"
    cfg.azm_limit_max_str = "360"
    cfg.alt_limit_min_str = "0"
    cfg.alt_limit_max_str = "180"
    cfg.pointing_model_enabled = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def alpha_at(surface, az_deg, el_deg, radius=RADIUS):
    """Overlay alpha at a sky (az, el): plot projection r=(90-el)/90."""
    r = (90.0 - el_deg) / 90.0 * radius
    x = radius + r * math.sin(math.radians(az_deg))
    y = radius - r * math.cos(math.radians(az_deg))
    return surface.get_at((int(x), int(y))).a


class AltAzGeometryTests(unittest.TestCase):

    def test_alt_limit_paints_low_elevation_annulus(self):
        # Mount ALT = 90 - el; ALT limited to [0, 80] forbids el < 10 (the
        # flip solution's ALT is negative there, also outside). Zenith stays
        # reachable.
        cfg = make_cfg(alt_limit_min_str="0", alt_limit_max_str="80")
        ov = rt._build_keepout_surface(cfg, RADIUS)
        self.assertEqual(alpha_at(ov, 0, 90), 0, "zenith must be reachable")
        self.assertEqual(alpha_at(ov, 45, 45), 0)
        for az in (0, 90, 180, 270):
            self.assertGreater(alpha_at(ov, az, 3), 0,
                               f"el=3 at az={az} should be kept out")

    def test_flip_solution_rescues_azm_limited_sky(self):
        # AZM confined to [0, 180] (east half). A WESTERN sky target's
        # canonical solution is out of limits, but the over-the-zenith flip
        # (azm+180) is inside -- the loop would take it, so the overlay must
        # NOT paint the west sky at altitudes the flip can reach.
        cfg = make_cfg(azm_limit_min_str="0", azm_limit_max_str="180",
                       alt_limit_min_str="-180", alt_limit_max_str="180")
        ov = rt._build_keepout_surface(cfg, RADIUS)
        self.assertEqual(alpha_at(ov, 270, 45), 0,
                         "flip solution reaches the west sky; not a keepout")

    def test_azm_limit_without_flip_room_paints_west(self):
        # Same AZM restriction but ALT confined to [0, 90]: the flip solution
        # (negative ALT) is now also out of limits, so the west sky IS kept
        # out while the east sky is not.
        cfg = make_cfg(azm_limit_min_str="0", azm_limit_max_str="180",
                       alt_limit_min_str="0", alt_limit_max_str="90")
        ov = rt._build_keepout_surface(cfg, RADIUS)
        self.assertGreater(alpha_at(ov, 270, 40), 0, "west sky must be kept out")
        self.assertEqual(alpha_at(ov, 90, 40), 0, "east sky must stay clear")


class AltAzSideGeometryTests(unittest.TestCase):

    def test_keepout_follows_the_horizontal_pole(self):
        # Side rig, pole on the horizon at az=180. Mount ALT = 90 - dec
        # (dec measured from that horizontal pole); ALT limited to [0, 90]
        # keeps only the sky hemisphere within 90 deg of the pole direction:
        # az=180 low sky reachable, az=0 low sky (dec ~ -90 -> ALT ~ 180)
        # kept out. The boundary great circle passes near the zenith.
        cfg = make_cfg(mount_mode="AltAz-Side", alignment_azimuth_str="180.0",
                       alt_limit_min_str="0", alt_limit_max_str="90",
                       azm_limit_min_str="0", azm_limit_max_str="360")
        ov = rt._build_keepout_surface(cfg, RADIUS)
        self.assertEqual(alpha_at(ov, 180, 20), 0, "sky toward the pole is reachable")
        self.assertGreater(alpha_at(ov, 0, 20), 0, "sky opposite the pole is kept out")


class CacheAndCostTests(unittest.TestCase):

    def test_cache_reuses_and_invalidates(self):
        rt._keepout_cache.clear()
        cfg = make_cfg(alt_limit_max_str="80")
        surf = pygame.Surface((2 * RADIUS + 20, 2 * RADIUS + 20), pygame.SRCALPHA)
        rt.draw_keepout_overlay_on_surface(surf, cfg, RADIUS + 10, RADIUS + 10, RADIUS)
        self.assertEqual(len(rt._keepout_cache), 1)
        first = next(iter(rt._keepout_cache.values()))
        rt.draw_keepout_overlay_on_surface(surf, cfg, RADIUS + 10, RADIUS + 10, RADIUS)
        self.assertIs(next(iter(rt._keepout_cache.values())), first, "cache must reuse")
        cfg.alt_limit_max_str = "70"  # shape changed -> new entry
        rt.draw_keepout_overlay_on_surface(surf, cfg, RADIUS + 10, RADIUS + 10, RADIUS)
        self.assertEqual(len(rt._keepout_cache), 2)

    def test_build_cost_is_a_one_time_fraction_of_a_second(self):
        cfg = make_cfg(alt_limit_max_str="80")
        t0 = time.perf_counter()
        rt._build_keepout_surface(cfg, RADIUS)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 2.0, f"keepout build took {dt:.2f}s -- too slow "
                                 "even for a one-time render-thread stall")

    def test_bad_limits_yield_no_overlay(self):
        cfg = make_cfg(alt_limit_max_str="not a number")
        self.assertIsNone(rt._build_keepout_surface(cfg, RADIUS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
