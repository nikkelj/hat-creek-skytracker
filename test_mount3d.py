#!/usr/bin/env python
"""Unit tests for the Mount 3D screen (mount3d.py).

The load-bearing test is boresight PARITY: the articulated 3D model's
boresight must equal the app's own forward transform (AzAlt2AzEl_* /
eq_mount_to_azel -- the same functions the tracker uses) for every mount
mode, across the full angle range including past-zenith poses. Every sign
and offset in the kinematic chain is pinned here, not by review.

Run: python test_mount3d.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
import unittest

import numpy as np
import pygame
pygame.init()

from mount3d import (unit_from_azel, azel_from_unit, rot_about,
                     mount_forward, mount_pose)


class Cfg:
    def __init__(self, mode='AltAz', align_az=0.0, align_el=0.0,
                 lat=34.87, flip=False):
        self.mount_mode = mode
        self.alignment_azimuth_str = str(align_az)
        self.alignment_elevation_str = str(align_el)
        self.lat_str = str(lat)
        self.altaz_side_flip = flip


MODES = [
    Cfg('AltAz'),
    Cfg('AltAz', align_az=137.4),
    Cfg('Passthrough'),
    Cfg('AltAz-Side', align_az=0.0),
    Cfg('AltAz-Side', align_az=259.0),
    Cfg('AltAz-Side', align_az=259.0, flip=True),
    Cfg('Eq', align_az=0.0, align_el=0.0, lat=34.87),   # pole alt <- latitude
    Cfg('Eq', align_az=182.0, align_el=40.0),
]

AZM_GRID = [0.0, 30.0, 90.0, 137.0, 180.0, 270.0, 330.0]
ALT_GRID = [-20.0, 0.0, 30.0, 60.0, 90.0, 120.0]   # includes past-zenith poses


class FrameGuardTests(unittest.TestCase):
    """Catches (N,E,U)/(E,N,U) frame mixups at the door."""

    def test_enu_basis(self):
        np.testing.assert_allclose(unit_from_azel(0, 0), [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(unit_from_azel(90, 0), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(unit_from_azel(123, 90), [0, 0, 1], atol=1e-9)

    def test_azel_round_trip(self):
        for az in (0.0, 45.0, 190.0, 359.0):
            for el in (-45.0, 0.0, 30.0, 89.0):
                a2, e2 = azel_from_unit(unit_from_azel(az, el))
                self.assertAlmostEqual(a2, az, places=8)
                self.assertAlmostEqual(e2, el, places=8)

    def test_rot_about_right_hand_rule(self):
        # +90 about Up takes East -> North (right-hand rule).
        v = rot_about(np.array([0.0, 0.0, 1.0]), 90.0) @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-12)


class BoresightParityTests(unittest.TestCase):
    """mount_pose boresight == unit vector of the app's forward transform,
    for every mode x angle combination. Compared as VECTORS (dot product), so
    past-zenith poses (el > 90 folds to az+180/180-el) compare correctly."""

    def test_parity_all_modes(self):
        for cfg in MODES:
            for azm in AZM_GRID:
                for alt in ALT_GRID:
                    with self.subTest(mode=cfg.mount_mode,
                                      align=cfg.alignment_azimuth_str,
                                      flip=cfg.altaz_side_flip,
                                      azm=azm, alt=alt):
                        az, el = mount_forward(cfg, azm, alt)
                        expect = unit_from_azel(az, el)
                        got = mount_pose(cfg, azm, alt)['boresight']
                        dot = float(np.dot(expect, got))
                        self.assertGreater(dot, 1.0 - 1e-9,
                                           f"boresight mismatch: dot={dot}")


class AxisGeometryTests(unittest.TestCase):

    def test_altaz_axis1_is_up(self):
        pose = mount_pose(Cfg('AltAz', align_az=77.0), 12.0, 34.0)
        np.testing.assert_allclose(pose['p'], [0, 0, 1], atol=1e-12)

    def test_altaz_side_axis1_is_horizontal_at_align_az(self):
        # The headline feature: the AZM axis lies ON the horizon at the
        # alignment azimuth.
        pose = mount_pose(Cfg('AltAz-Side', align_az=259.0), 45.0, 20.0)
        self.assertAlmostEqual(float(pose['p'][2]), 0.0, places=12)
        np.testing.assert_allclose(pose['p'], unit_from_azel(259.0, 0.0),
                                   atol=1e-12)

    def test_altaz_side_index_home_boresight_on_axis(self):
        # At the index marks (AZM=ALT=0) the scope points ALONG the polar
        # axis (LEARNINGS 2026-07-26), for both tip sides.
        for flip in (False, True):
            pose = mount_pose(Cfg('AltAz-Side', align_az=100.0, flip=flip), 0.0, 0.0)
            dot = float(np.dot(pose['boresight'], pose['p']))
            self.assertGreater(dot, 1.0 - 1e-9)

    def test_eq_axis1_is_alignment_vector(self):
        pose = mount_pose(Cfg('Eq', align_az=5.0, align_el=42.0), 30.0, 10.0)
        np.testing.assert_allclose(pose['p'], unit_from_azel(5.0, 42.0),
                                   atol=1e-12)

    def test_eq_pole_alt_falls_back_to_latitude(self):
        pose = mount_pose(Cfg('Eq', align_az=0.0, align_el=0.0, lat=34.87), 0.0, 0.0)
        np.testing.assert_allclose(pose['p'], unit_from_azel(0.0, 34.87),
                                   atol=1e-12)

    def test_axis2_perpendicular_to_boresight_and_axis1(self):
        for cfg in MODES:
            pose = mount_pose(cfg, 40.0, 25.0)
            self.assertAlmostEqual(
                float(np.dot(pose['axis2_world'], pose['p'])), 0.0, places=9)

    def test_stage_rotations_are_rotations(self):
        for cfg in MODES:
            pose = mount_pose(cfg, 123.0, 45.0)
            for key in ('R_ax1', 'R_ax2'):
                R = pose[key]
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)

    def test_stage_chain_moves_home_boresight_to_current(self):
        # The geometry contract: R_ax2 applied to the home boresight yields
        # the current boresight (parts sculpted at home pose track the tube).
        for cfg in MODES:
            for azm, alt in ((0.0, 0.0), (50.0, 30.0), (200.0, -15.0)):
                pose = mount_pose(cfg, azm, alt)
                moved = pose['R_ax2'] @ pose['home']['boresight']
                dot = float(np.dot(moved, pose['boresight']))
                self.assertGreater(dot, 1.0 - 1e-9)


class ProjectionTests(unittest.TestCase):

    def _cam(self, view_mode='orbit', **cfg_kw):
        from mount3d import Mount3DState, _camera
        m3d = Mount3DState()
        m3d.view_mode = view_mode
        cfg = Cfg(**cfg_kw)
        cfg.mount3d_observer_bearing_deg = 200.0
        cfg.mount3d_observer_distance_m = 3.0
        cfg.mount3d_eye_height_m = 1.5
        return m3d, cfg, _camera(m3d, cfg, 800, 600)

    def test_view_axis_point_projects_to_center(self):
        from mount3d import project_points, HEAD_HEIGHT
        m3d, cfg, (R, cam, focal) = self._cam()
        head = np.array([[0.0, 0.0, HEAD_HEIGHT]])
        sx, sy, z = project_points(head, R, cam, 400, 300, focal)
        self.assertAlmostEqual(float(sx[0]), 400.0, places=6)
        self.assertAlmostEqual(float(sy[0]), 300.0, places=6)
        self.assertGreater(float(z[0]), 0.0)

    def test_behind_camera_is_culled_by_sign(self):
        from mount3d import project_points, HEAD_HEIGHT, unit_from_azel, BASE_DIST
        m3d, cfg, (R, cam, focal) = self._cam()
        behind = cam + unit_from_azel(m3d.view_az, m3d.view_el) * 2.0
        _sx, _sy, z = project_points(behind[None, :], R, cam, 400, 300, focal)
        self.assertLess(float(z[0]), 0.0)

    def test_operator_camera_sits_at_configured_seat(self):
        from mount3d import unit_from_azel
        m3d, cfg, (R, cam, focal) = self._cam(view_mode='operator')
        seat = unit_from_azel(200.0, 0.0) * 3.0
        np.testing.assert_allclose(cam, [seat[0], seat[1], 1.5], atol=1e-9)
        # Initial look direction aims at the mount head.
        from mount3d import HEAD_HEIGHT
        to_head = np.array([0, 0, HEAD_HEIGHT]) - cam
        to_head /= np.linalg.norm(to_head)
        fwd = R[2]
        self.assertGreater(float(np.dot(fwd, to_head)), 1.0 - 1e-9)

    def test_zoom_moves_orbit_camera_closer(self):
        from mount3d import Mount3DState, _camera, HEAD_HEIGHT
        m3d = Mount3DState()
        cfg = Cfg()
        _R1, cam1, _f1 = _camera(m3d, cfg, 800, 600)
        m3d.zoom = 2.0
        _R2, cam2, _f2 = _camera(m3d, cfg, 800, 600)
        head = np.array([0, 0, HEAD_HEIGHT])
        self.assertLess(np.linalg.norm(cam2 - head), np.linalg.norm(cam1 - head))


class FovConeTests(unittest.TestCase):

    def test_corner_angles_match_pinhole_geometry(self):
        from mount3d import fov_corner_dirs
        b = unit_from_azel(40.0, 25.0)
        a2 = unit_from_azel(130.0, 0.0)
        w, h = 6.0, 4.0
        expect = math.degrees(math.atan(math.hypot(
            math.tan(math.radians(w / 2)), math.tan(math.radians(h / 2)))))
        for c in fov_corner_dirs(b, a2, w, h, 0.0):
            ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(c, b))))))
            self.assertAlmostEqual(ang, expect, places=9)
            # ...and roughly hypot(w,h)/2 in the small-angle sense.
            self.assertAlmostEqual(ang, math.hypot(w, h) / 2, delta=0.02)

    def test_rotation_rolls_the_corner_pattern(self):
        from mount3d import fov_corner_dirs
        b = unit_from_azel(0.0, 30.0)
        a2 = unit_from_azel(90.0, 0.0)
        c0 = fov_corner_dirs(b, a2, 8.0, 2.0, 0.0)
        c90 = fov_corner_dirs(b, a2, 8.0, 2.0, 90.0)
        # A 90-deg roll maps each corner onto the NEXT corner's position
        # pattern (wide axis becomes tall axis): corner 0 rotated should align
        # with where a (2.0, 8.0) cone's corner 0 sits.
        c_swap = fov_corner_dirs(b, a2, 2.0, 8.0, 0.0)
        # rotating (w,h)=(8,2) by 90 about the boresight = the (2,8) cone's
        # corner set, cyclically shifted by one.
        for i in range(4):
            dot = float(np.dot(c90[i], c_swap[(i + 1) % 4]))
            self.assertGreater(dot, 1.0 - 1e-9)


def _prep_render_cfg(cfg, stars=False):
    """Minimal extra config surface the renderer reads."""
    cfg.camera_configs = {
        'camera1': {'pixel_size': 2.9, 'focal_length': 162.0,
                    'alignment_rotation': 0.0},
        'camera2': {'pixel_size': 2.4, 'focal_length': 2000.0,
                    'alignment_rotation': 3.0},
    }
    cfg.azm_limit_min_str = "25"
    cfg.azm_limit_max_str = "335"
    cfg.alt_limit_min_str = "5"
    cfg.alt_limit_max_str = "78"
    cfg.elevation_mask_str = "10"
    cfg.starfield_enabled = stars   # catalog access only when asked
    cfg.lon_str = "-120.4"
    cfg.alt_str = "110"
    cfg.pointing_model_enabled = False
    cfg.mount3d_observer_bearing_deg = 200.0
    cfg.mount3d_observer_distance_m = 2.5
    cfg.mount3d_eye_height_m = 1.2
    return cfg


class SmokeRenderTests(unittest.TestCase):

    def test_render_every_mode(self):
        from mount3d import Mount3DState, render_mount3d_on_surface
        for cfg in (Cfg('AltAz'), Cfg('AltAz-Side', align_az=259.0),
                    Cfg('Eq', align_az=0, align_el=34.9), Cfg('Passthrough')):
            _prep_render_cfg(cfg)
            for view in ('orbit', 'operator'):
                m3d = Mount3DState()
                m3d.view_mode = view
                m3d.follow_live = False
                m3d.manual_azm, m3d.manual_alt = 40.0, 30.0
                surf = pygame.Surface((800, 600))
                render_mount3d_on_surface(surf, cfg, None, None, m3d, None)
                arr = pygame.surfarray.pixels3d(surf)
                nonbg = int(np.count_nonzero(arr.sum(axis=2) > 60))
                del arr
                self.assertGreater(nonbg, 500,
                                   f"{cfg.mount_mode}/{view}: scene looks empty")

    def test_below_horizon_orbit_looks_up_through_the_mount(self):
        """The orbit camera may dive to view_el=-89 (under the ground plane,
        looking straight up). The scene must still render non-empty."""
        from mount3d import Mount3DState, render_mount3d_on_surface
        cfg = _prep_render_cfg(Cfg('AltAz'))
        m3d = Mount3DState()
        m3d.view_mode = 'orbit'
        m3d.follow_live = False
        m3d.manual_azm, m3d.manual_alt = 40.0, 30.0
        m3d.view_el = -85.0
        surf = pygame.Surface((800, 600))
        render_mount3d_on_surface(surf, cfg, None, None, m3d, None)
        arr = pygame.surfarray.pixels3d(surf)
        nonbg = int(np.count_nonzero(arr.sum(axis=2) > 60))
        del arr
        self.assertGreater(nonbg, 500, "below-horizon view renders empty")

    def test_view_el_clamp_allows_minus_89(self):
        from mount3d import Mount3DState, handle_mount3d_mouse_motion
        m3d = Mount3DState()
        m3d.view_mode = 'orbit'
        m3d.dragging_view = True
        m3d.dragging_slider = None
        # Drag convention: positive rel[1] (mouse down) RAISES view_el
        # (grab-the-scene feel), negative lowers it.
        handle_mount3d_mouse_motion((400, 300), (0, 1000), (1, 0, 0),
                                    None, m3d)
        self.assertLessEqual(m3d.view_el, 89.0, "upper clamp")
        handle_mount3d_mouse_motion((400, 300), (0, -1000), (1, 0, 0),
                                    None, m3d)
        self.assertGreaterEqual(m3d.view_el, -89.0, "lower clamp")
        self.assertLess(m3d.view_el, -80.0,
                        "drag should reach the below-horizon range")


class StarTimeProgressionTests(unittest.TestCase):
    """The 3D sky must rotate as tracking time advances (the catalog is NOT
    frozen at one epoch): two renders 2 h apart must differ in many pixels,
    with everything except the stars held identical."""

    def test_sky_rotates_between_renders(self):
        import types
        from mount3d import Mount3DState, render_mount3d_on_surface
        try:
            from skyfield.api import load
            ts = load.timescale()
            from star_catalog import get_catalog
            probe = get_catalog(None).current_altaz(
                34.87, -120.4, 110.0, ts, ts.now().tt,
                elevation_mask=0.0, max_count=50, limiting_mag=4.0)
            if probe is None or len(probe.get('az', [])) == 0:
                raise RuntimeError("catalog empty")
        except Exception as e:
            self.skipTest(f"star catalog unavailable here: {e}")

        cfg = _prep_render_cfg(Cfg('AltAz'), stars=True)
        t0 = ts.now().tt
        frames = []
        for tt in (t0, t0 + 2.0 / 24.0):
            tvs = types.SimpleNamespace(current_tt=tt, ephemeris=None)
            m3d = Mount3DState()
            m3d.view_mode = 'orbit'
            m3d.follow_live = False
            m3d.manual_azm, m3d.manual_alt = 40.0, 30.0
            surf = pygame.Surface((800, 600))
            render_mount3d_on_surface(surf, cfg, tvs, None, m3d, ts)
            frames.append(pygame.surfarray.array3d(surf))
        diff = int(np.count_nonzero(np.any(frames[0] != frames[1], axis=2)))
        self.assertGreater(diff, 200,
                           f"only {diff} pixels changed over 2 h -- star sky "
                           "appears frozen in time")


if __name__ == '__main__':
    unittest.main(verbosity=2)
