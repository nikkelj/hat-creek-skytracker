#!/usr/bin/env python
"""
Tests for the celestial-targets subsystem (celestial.py + star_catalog names):
catalogue loading (solar-system bodies, Messier incl. the M45 patch, NGC),
the fast fixed-RA/Dec transform against full Skyfield, tracking-trajectory
format/interp parity, the sliding-window maintenance, selection semantics
(mutual exclusivity, toggle-off, solar warning), the PROGRAM resolver branch,
and config round-tripping of the overlay toggles.

Headless. Run: python test_celestial.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
import unittest
from types import SimpleNamespace

import numpy as np
import pygame
pygame.init()

from skyfield.api import Star, load, wgs84

import celestial
from celestial import get_celestial, select_celestial, ensure_selected_trajectory
from trajectory import interpolate_position_data_and_rates, live_tt

LAT, LON, ELEV = 34.874, -120.446, 120.0
TS = load.timescale()


def fake_cfg(**kw):
    d = dict(lat_str=str(LAT), lon_str=str(LON), alt_str=str(ELEV),
             messier_enabled=True, ngc_enabled=False, ngc_limiting_magnitude=10.0)
    d.update(kw)
    return SimpleNamespace(**d)


def fake_tvs():
    return SimpleNamespace(selected_satellite=None, selected_aircraft=None,
                           selected_celestial=None, celestial_trajectories={},
                           satellite_trajectories={}, aircraft_trajectories={},
                           ephemeris=None)


class TestCatalogue(unittest.TestCase):
    def setUp(self):
        self.cat = get_celestial()

    def test_solar_system_bodies_present(self):
        keys = set()
        for key, name, az, el, dist, color, rad in self.cat.solar_system_altaz(
                LAT, LON, ELEV, TS, TS.now().tt):
            keys.add(key)
            self.assertTrue(-90.0 <= el <= 90.0)
            self.assertTrue(0.0 <= az < 360.0)
        for want in ("sun", "moon", "planet:Jupiter", "planet:Saturn",
                     "planet:Mars", "planet:Venus"):
            self.assertIn(want, keys)

    def test_messier_catalogue_complete_enough(self):
        m = self.cat.messier
        self.assertGreaterEqual(len(m["key"]), 105)
        self.assertIn("M045", m["key"], "Pleiades patch must be present")
        self.assertIn("M031", m["key"])
        self.assertIn("Andromeda", self.cat.display_name("M031"))

    def test_ngc_catalogue_loaded(self):
        self.assertGreater(len(self.cat.ngc["key"]), 5000)
        self.assertEqual(self.cat.display_name("moon"), "Moon")

    def test_fixed_altaz_matches_skyfield(self):
        # The fast hour-angle transform (J2000, no precession) must agree with
        # full Skyfield apparent altaz to well under a degree -- visualisation
        # tolerance; tracking uses the full path via build_trajectory.
        t_tt = TS.now().tt
        m = self.cat.messier
        idx = [m["key"].index("M031"), m["key"].index("M045")]
        ra = m["ra_deg"][idx]
        dec = m["dec_deg"][idx]
        az_fast, el_fast = self.cat.fixed_altaz(ra, dec, LAT, LON, TS, t_tt)
        observer = load("de421.bsp")["earth"] + wgs84.latlon(LAT, LON, elevation_m=ELEV)
        star = Star(ra_hours=ra / 15.0, dec_degrees=dec)
        alt, az, _ = observer.at(TS.tt_jd(t_tt)).observe(star).apparent().altaz()
        for i in range(len(idx)):
            self.assertLess(abs(el_fast[i] - alt.degrees[i]), 0.6)
            d_az = abs((az_fast[i] - az.degrees[i] + 180.0) % 360.0 - 180.0)
            self.assertLess(d_az * math.cos(math.radians(el_fast[i])), 0.6)

    def test_plot_objects_kinds_and_toggles(self):
        t_tt = TS.now().tt
        objs = self.cat.compute_plot_objects(fake_cfg(), TS, t_tt)
        kinds = {o["kind"] for o in objs}
        self.assertIn("sun", kinds)
        self.assertIn("moon", kinds)
        self.assertIn("planet", kinds)
        self.assertIn("star", kinds, "named bright stars must be selectable")
        self.assertIn("messier", kinds)
        self.assertNotIn("ngc", kinds, "NGC defaults off")
        objs2 = self.cat.compute_plot_objects(
            fake_cfg(ngc_enabled=True, ngc_limiting_magnitude=8.0), TS, t_tt)
        self.assertIn("ngc", {o["kind"] for o in objs2})
        # NGC magnitude limit respected
        for o in objs2:
            if o["kind"] == "ngc":
                self.assertLessEqual(o["mag"], 8.0)


class TestTrajectories(unittest.TestCase):
    def setUp(self):
        self.cat = get_celestial()

    def test_format_and_interp_parity(self):
        now = TS.now().tt
        for key in ("moon", "planet:Jupiter", "M031"):
            traj = self.cat.build_trajectory(key, LAT, LON, ELEV, TS, now)
            self.assertIsNotNone(traj)
            rows, times = traj
            self.assertEqual(len(rows[0]), 8)
            self.assertTrue(np.all(np.diff(times) > 0))
            r = interpolate_position_data_and_rates(traj, now)
            self.assertIsNotNone(r[2], f"{key} interp failed")
            # Rates on a sample boundary should match the stored row rates.
            mid = len(rows) // 2
            r2 = interpolate_position_data_and_rates(traj, times[mid])
            self.assertAlmostEqual(r2[2], rows[mid][1], places=6)  # el
            self.assertAlmostEqual(r2[4], rows[mid][2], places=6)  # az

    def test_fixed_object_rates_are_sidereal_scale(self):
        now = TS.now().tt
        rows, times = self.cat.build_trajectory("star:HIP32349", LAT, LON,
                                                ELEV, TS, now)
        SIDEREAL_DPS = 360.0 / 86164.0  # ~0.00418
        for row in rows[1:-1]:
            self.assertLess(abs(row[7]), SIDEREAL_DPS * 1.05,
                            "el rate cannot exceed sidereal")
            # az rate exceeds sidereal near meridian transit (measured up to
            # ~3.0x for Sirius from this latitude, time-of-day dependent).
            # The loose 5x bound still catches unit errors (deg/hr, wraps).
            self.assertLess(abs(row[6]), SIDEREAL_DPS * 5.0)

    def test_unknown_key_returns_none(self):
        self.assertIsNone(self.cat.build_trajectory("NGC99999x", LAT, LON,
                                                    ELEV, TS, TS.now().tt))

    def test_sliding_window_maintenance(self):
        tvs = fake_tvs()
        cfg = fake_cfg()
        tvs.selected_celestial = "moon"
        ensure_selected_trajectory(tvs, cfg)
        rows, times = tvs.celestial_trajectories["moon"]
        now = live_tt()
        self.assertLess(times[0], now)
        self.assertGreater(times[-1], now + 5.0 * 60.0 / 86400.0)
        # Fresh window: no rebuild (object identity preserved).
        before = tvs.celestial_trajectories["moon"]
        ensure_selected_trajectory(tvs, cfg)
        self.assertIs(tvs.celestial_trajectories["moon"], before)
        # Stale window (ends before now): rebuilt.
        old = self.cat.build_trajectory("moon", LAT, LON, ELEV, TS,
                                        now - 0.5)  # centered 12 h ago
        tvs.celestial_trajectories = {"moon": old}
        ensure_selected_trajectory(tvs, cfg)
        self.assertIsNot(tvs.celestial_trajectories["moon"], old)
        self.assertGreater(tvs.celestial_trajectories["moon"][1][-1], now)


class TestSelection(unittest.TestCase):
    def test_select_toggle_and_exclusivity(self):
        tvs = fake_tvs()
        tvs.selected_satellite = object()
        tvs.selected_aircraft = "ABC123"
        select_celestial(tvs, fake_cfg(), "M045")
        self.assertEqual(tvs.selected_celestial, "M045")
        self.assertIsNone(tvs.selected_satellite)
        self.assertIsNone(tvs.selected_aircraft)
        self.assertIn("M045", tvs.celestial_trajectories)
        select_celestial(tvs, fake_cfg(), "M045")  # toggle off
        self.assertIsNone(tvs.selected_celestial)

    def test_sun_selection_warns(self):
        tvs = fake_tvs()
        msgs = []
        select_celestial(tvs, fake_cfg(), "sun", status_cb=msgs.append)
        self.assertTrue(any("solar filter" in m for m in msgs))

    def test_no_config_defers_trajectory(self):
        # Full-screen click path has no config: selection sticks, trajectory
        # is left for the control loop's ensure call.
        tvs = fake_tvs()
        select_celestial(tvs, None, "moon")
        self.assertEqual(tvs.selected_celestial, "moon")
        self.assertEqual(tvs.celestial_trajectories, {})
        ensure_selected_trajectory(tvs, fake_cfg())
        self.assertIn("moon", tvs.celestial_trajectories)


class TestProgramResolver(unittest.TestCase):
    def test_resolver_prefers_satellite_then_celestial(self):
        from joystick_controller import active_program_trajectory
        tvs = fake_tvs()
        select_celestial(tvs, fake_cfg(), "planet:Saturn")
        traj, kind, key = active_program_trajectory(tvs)
        self.assertEqual((kind, key), ("celestial", "planet:Saturn"))
        # A satellite selection outranks the celestial one.
        sat = object()
        tvs.selected_satellite = sat
        tvs.satellite_trajectories = {sat: ([[0] * 8], np.array([0.0]))}
        traj, kind, key = active_program_trajectory(tvs)
        self.assertEqual(kind, "satellite")


class TestStarNames(unittest.TestCase):
    def test_top_named_set_and_names(self):
        from star_catalog import get_catalog, iau_star_names
        names = iau_star_names()
        self.assertGreater(len(names), 300)
        self.assertEqual(names[32349], "Sirius")
        self.assertEqual(names[71683], "Rigil Kentaurus")  # trailing-'*' row
        sc = get_catalog()
        top = sc.top_named_hips(100)
        self.assertEqual(len(top), 100)
        for hip in (32349, 91262, 69673, 24436):  # Sirius Vega Arcturus Rigel
            self.assertIn(hip, top)
        self.assertEqual(sc.name_for(32349), "Sirius")


class TestProgramTrackEndToEnd(unittest.TestCase):
    """PROGRAM mode slews to and tracks a real celestial target in the hardware
    sim, through the REAL tracking_control dispatch and the REAL trajectory
    interpolation (unlike test_mode_machine's rig, nothing is monkeypatched).
    Target = whichever always-up named star is currently highest, so the test
    is deterministic day or night. Wall-clock (~8 s), like the mode-machine
    suite."""

    def _pick_high_star(self):
        cat = get_celestial()
        objs = cat.compute_plot_objects(fake_cfg(messier_enabled=False), TS,
                                        TS.now().tt)
        stars = [o for o in objs if o["kind"] == "star"]
        best = max(stars, key=lambda o: o["el"])
        self.assertGreater(best["el"], 20.0,
                           "some top-100 star must be well above the horizon")
        return best["key"]

    def test_program_acquires_and_tracks_star(self):
        import json
        import time as _time
        import simulator
        from simulator import HardwareSimulator
        from config import ConfigState
        from lib.auxstar import Targets
        from joystick_controller import JoystickModeState, TrackingMode

        cfg = ConfigState()
        cfg.load_from_dict(json.load(open("config.example.json")))
        cfg.sim_config["enabled"] = True
        cfg.azm_offset_str = "0"
        cfg.alt_offset_str = "0"
        cfg.alignment_azimuth_str = "0.0"
        cfg.alignment_elevation_str = "0.0"
        cfg.mount_mode = "AltAz"
        cfg.lat_str, cfg.lon_str, cfg.alt_str = str(LAT), str(LON), str(ELEV)

        key = self._pick_high_star()
        tvs = fake_tvs()
        select_celestial(tvs, cfg, key)
        traj = tvs.celestial_trajectories[key]

        def target_azel():
            r = interpolate_position_data_and_rates(traj, live_tt())
            return r[4], r[2]

        sim = HardwareSimulator(cfg, None, None)
        taz, tel = target_azel()
        # Start 2 deg off in azimuth, 1.5 deg in elevation.
        sim.mount.az_true_deg = (taz + 2.0) % 360.0
        sim.mount.el_true_deg = 90.0 - (tel + 1.5)

        class _FakeJoy:
            def get_numaxes(self):
                return 6

            def get_axis(self, i):
                return 0.0

            def get_instance_id(self):
                return 0

        js = JoystickModeState(None, cfg, lambda m: None)
        js.hardware_sim = sim
        js.telescope_connected = True
        js.telescope_controller = sim.mount
        js.tracking_vis_state = tvs
        # tracking_control is a no-op without a connected joystick
        js.joysticks = {0: _FakeJoy()}
        js.connected_joystick = 0
        js.joystick_tare = {}
        js.feed_forward_azm_enabled = True
        js.feed_forward_alt_enabled = True
        js.tracking_mode = TrackingMode.PROGRAM

        errors = []
        end = _time.time() + 8.0
        while _time.time() < end:
            c0 = _time.perf_counter()
            js.current_azm = sim.mount.hc_get_position(Targets.AZM) * 360.0
            js.current_alt = sim.mount.hc_get_position(Targets.ALT) * 360.0
            js.current_azm_raw = js.current_azm
            js.current_alt_raw = js.current_alt
            js.tracking_control()
            taz, tel = target_azel()
            maz, mel = simulator.normalize_azel(
                sim.mount.az_true_deg, 90.0 - sim.mount.el_true_deg)
            errors.append(math.hypot(
                simulator.wrap180(maz - taz) * math.cos(math.radians(tel)),
                mel - tel))
            sleep = 1.0 / 15.0 - (_time.perf_counter() - c0)
            if sleep > 0:
                _time.sleep(sleep)

        self.assertEqual(js.tracking_mode, TrackingMode.PROGRAM,
                         "PROGRAM must hold a celestial target, not fall to STANDBY")
        self.assertGreater(errors[0], 1.5, "test must start off-target")
        settled = errors[len(errors) * 3 // 4:]
        self.assertLess(max(settled), 0.3,
                        f"celestial track did not converge: settled errors "
                        f"{[f'{e:.2f}' for e in settled[-8:]]}")


class TestObjectToggleButtons(unittest.TestCase):
    """The shared object-type toggle spec + the tracking-vis click handler.
    (The Mount 3D HUD drives the same config attrs through its own buttons.)"""

    def test_spec_covers_types_but_not_always_on_objects(self):
        from tracking_visuals import OBJECT_TOGGLES
        attrs = [a for a, _n, _d in OBJECT_TOGGLES]
        for want in ("satellites_enabled", "aircraft_enabled", "starfield_enabled",
                     "messier_enabled", "ngc_enabled"):
            self.assertIn(want, attrs)
        # Kid-friendly invariant: no toggle may hide the sun/moon/planets or
        # the named bright stars -- they have no config attr at all.
        self.assertFalse(any("planet" in a or "moon" in a or "sun" in a
                             for a in attrs))

    def test_click_flips_config_flag(self):
        import pygame
        from tracking_visuals import handle_object_toggle_click
        from config import ConfigState
        cfg = ConfigState()
        state = SimpleNamespace(object_toggle_rects={
            "ngc_enabled": pygame.Rect(10, 10, 110, 22)})
        self.assertFalse(cfg.ngc_enabled)
        self.assertTrue(handle_object_toggle_click(state, cfg, (15, 15)))
        self.assertTrue(cfg.ngc_enabled)
        self.assertTrue(handle_object_toggle_click(state, cfg, (15, 15)))
        self.assertFalse(cfg.ngc_enabled)
        self.assertFalse(handle_object_toggle_click(state, cfg, (500, 500)))

    def test_aircraft_flag_round_trips(self):
        from config import ConfigState
        cfg = ConfigState()
        self.assertTrue(cfg.aircraft_enabled)
        cfg.aircraft_enabled = False
        cfg2 = ConfigState()
        cfg2.load_from_dict(cfg.get_config_dict())
        self.assertFalse(cfg2.aircraft_enabled)


class TestObjectInfoLines(unittest.TestCase):
    """The any-object details box: selected_object_info_lines."""

    def test_lines_for_body_and_dso(self):
        from celestial import selected_object_info_lines
        tvs = fake_tvs()
        self.assertIsNone(selected_object_info_lines(tvs))
        tvs.selected_celestial = "moon"
        tvs.celestial_plot_objects = {
            "moon": {"key": "moon", "kind": "moon", "name": "Moon", "az": 120.0,
                     "el": 35.0, "mag": 0.0, "dist_km": 384400.0}}
        lines = dict(selected_object_info_lines(tvs))
        self.assertEqual(lines["Target"], "Moon")
        self.assertIn("384,400", lines["Distance"])
        tvs.selected_celestial = "M031"
        tvs.celestial_plot_objects = {
            "M031": {"key": "M031", "kind": "messier", "name": "M31 Andromeda Galaxy",
                     "az": 60.0, "el": 20.0, "mag": 3.4,
                     "ra_deg": 10.68, "dec_deg": 41.27}}
        lines = dict(selected_object_info_lines(tvs))
        self.assertEqual(lines["Type"], "Messier object")
        self.assertEqual(lines["V mag"], "3.4")
        self.assertIn("00h 42.7m", lines["RA"])

    def test_hidden_object_reports_status(self):
        from celestial import selected_object_info_lines
        tvs = fake_tvs()
        tvs.selected_celestial = "NGC0001"
        tvs.celestial_plot_objects = {}
        lines = dict(selected_object_info_lines(tvs))
        self.assertIn("below horizon", lines["Status"])


class TestMount3DSelection(unittest.TestCase):
    """Sky-object click selection in the 3D view (nearest marker in 12 px)."""

    def _fixture(self):
        from mount3d import Mount3DState
        display = SimpleNamespace(sub_x=100, sub_y=50)
        m3d = Mount3DState()
        m3d.ui_rects = {}
        sat = object()
        m3d.sky_click_targets = [
            (200, 200, 'satellite', sat),
            (206, 200, 'celestial', 'moon'),   # closer to the click below
            (400, 400, 'aircraft', 'ABC123'),
        ]
        tvs = fake_tvs()
        return display, m3d, sat, tvs

    def test_nearest_target_wins_and_selects(self):
        from mount3d import handle_mount3d_mouse_down
        display, m3d, sat, tvs = self._fixture()
        cfg = fake_cfg()
        # Screen pos = local (207, 200) -> nearest is the moon (celestial).
        self.assertTrue(handle_mount3d_mouse_down((307, 250), 1, display, m3d, cfg, tvs))
        self.assertEqual(tvs.selected_celestial, "moon")
        # Local (199, 200) -> the satellite; celestial cleared.
        self.assertTrue(handle_mount3d_mouse_down((299, 250), 1, display, m3d, cfg, tvs))
        self.assertIs(tvs.selected_satellite, sat)
        self.assertIsNone(tvs.selected_celestial)
        # Clicking the selected satellite again toggles it off.
        handle_mount3d_mouse_down((299, 250), 1, display, m3d, cfg, tvs)
        self.assertIsNone(tvs.selected_satellite)

    def test_empty_sky_click_starts_drag(self):
        from mount3d import handle_mount3d_mouse_down
        display, m3d, _sat, tvs = self._fixture()
        self.assertTrue(handle_mount3d_mouse_down((900, 900), 1, display, m3d,
                                                  fake_cfg(), tvs))
        self.assertTrue(m3d.dragging_view)
        self.assertIsNone(tvs.selected_satellite)


class TestConfigToggles(unittest.TestCase):
    def test_round_trip(self):
        from config import ConfigState
        cfg = ConfigState()
        self.assertTrue(cfg.messier_enabled)
        self.assertFalse(cfg.ngc_enabled)
        cfg.messier_enabled = False
        cfg.ngc_enabled = True
        cfg.ngc_limiting_magnitude = 8.5
        d = cfg.get_config_dict()
        cfg2 = ConfigState()
        cfg2.load_from_dict(d)
        self.assertFalse(cfg2.messier_enabled)
        self.assertTrue(cfg2.ngc_enabled)
        self.assertAlmostEqual(cfg2.ngc_limiting_magnitude, 8.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
