#!/usr/bin/env python
"""Live A/B parity: the Rust astro engine vs the skyfield paths it strangles.

Unlike the pure-Rust golden tests (cargo test -p skytracker-astro, frozen
vectors), this suite runs BOTH live implementations side by side through
the real app entry points -- trajectory._compute_one_trajectory, the
visibility gate, and celestial.solar_system_altaz / build_trajectory --
on the current tle_cache.tle, asserting the same tolerances the golden
gates use (sats 20 arcsec, bodies/fixed targets 60 arcsec).

Build first:
    cd rust/skytracker-ffi && maturin develop --release
Run: python -m pytest test_rust_astro_parity.py
"""

import math
import os
import types
import unittest

import numpy as np

try:
    import skytracker_core  # noqa: F401
    _HAVE_ASTRO = getattr(skytracker_core, "ASTRO_ENGINE_AVAILABLE", False)
except ImportError:
    _HAVE_ASTRO = False

import rust_astro_adapter

TLE_FILE = ("tle_cache.tle" if os.path.exists("tle_cache.tle")
            else os.path.join("tests", "golden", "sat_tles.txt"))
_HAVE_TLES = os.path.exists(TLE_FILE)
_HAVE_DE421 = os.path.exists("de421.bsp")

LAT, LON, ELEV = 34.8740289, -120.4461237, 100.0


def _sky_sep_deg(alt1, az1, alt2, az2):
    a1, z1 = math.radians(alt1), math.radians(az1)
    a2, z2 = math.radians(alt2), math.radians(az2)
    cosd = (math.sin(a1) * math.sin(a2)
            + math.cos(a1) * math.cos(a2) * math.cos(z1 - z2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosd))))


def _enable(on):
    rust_astro_adapter.configure(types.SimpleNamespace(use_rust_astro=on))


@unittest.skipUnless(_HAVE_ASTRO, "skytracker_core wheel lacks AstroEngine")
@unittest.skipUnless(_HAVE_TLES, "tle_cache.tle absent")
class SatelliteParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from skyfield.api import load, wgs84

        cls.ts = load.timescale()
        cls.observer = wgs84.latlon(LAT, LON, elevation_m=ELEV)
        cls.sats = load.tle_file(TLE_FILE)[:250]
        t0 = cls.ts.now()
        cls.times = cls.ts.linspace(t0, cls.ts.tt_jd(t0.tt + 15.0 / 1440.0), 31)

    def tearDown(self):
        _enable(False)

    def test_compute_one_trajectory_rows_match(self):
        import trajectory as tj

        _enable(False)
        worst = 0.0
        checked = 0
        for sat in self.sats[:40]:
            py_rows = np.array(
                tj._compute_one_trajectory(sat, self.observer, self.times,
                                           500, 400, 300)[0])
            rust_rows = rust_astro_adapter.satellite_rows(
                sat, self.times, self.observer, 500, 400, 300)
            if rust_rows is None:
                continue
            checked += 1
            self.assertEqual(rust_rows.shape, py_rows.shape)
            for i in range(py_rows.shape[0]):
                sep = _sky_sep_deg(rust_rows[i, 1], rust_rows[i, 2],
                                   py_rows[i, 1], py_rows[i, 2])
                worst = max(worst, sep)
                self.assertLess(
                    sep, 20.0 / 3600.0,
                    f"{sat.name} row {i}: {sep * 3600:.2f} arcsec")
            # px/py sub-pixel; range relative; rates near-identical.
            np.testing.assert_allclose(rust_rows[:, 4:6], py_rows[:, 4:6], atol=0.05)
            np.testing.assert_allclose(rust_rows[:, 3], py_rows[:, 3], rtol=1e-4)
            np.testing.assert_allclose(rust_rows[:, 6:8], py_rows[:, 6:8], atol=2e-3)
        self.assertGreater(checked, 20, "too few satellites checked")
        print(f"\n[parity] sat rows: {checked} sats, worst {worst * 3600:.3f} arcsec")

    def test_visibility_gate_matches(self):
        vis = rust_astro_adapter.visible_satnums(
            self.sats, self.times, self.observer, 15.0)
        self.assertIsNotNone(vis)
        # Python coarse gate (GAST z-rotation) vs Rust (skyfield-parity
        # path): both approximate the same elevations well inside the 15 deg
        # margin, so disagreements may exist only for satellites whose peak
        # elevation grazes the threshold.
        import trajectory as tj

        _enable(False)
        mask = tj._batched_visibility_mask(
            self.sats, self.observer, self.times, 15.0)
        disagreements = 0
        for sat, py_vis in zip(self.sats, mask):
            satnum = sat.model.satnum_str
            if (satnum in vis) != bool(py_vis):
                disagreements += 1
                # Verify the satellite is a genuine boundary case.
                alts = (sat - self.observer).at(self.times).altaz()[0].degrees
                self.assertLess(abs(float(np.max(alts)) - 15.0), 0.2,
                                f"{sat.name}: non-boundary disagreement")
        print(f"[parity] visibility: {len(self.sats)} sats, "
              f"{disagreements} boundary-only disagreements")

    def test_flag_routes_compute_one_trajectory(self):
        import trajectory as tj

        sat = self.sats[0]
        _enable(False)
        py = np.array(tj._compute_one_trajectory(
            sat, self.observer, self.times, 500, 400, 300)[0])
        _enable(True)
        rust = np.array(tj._compute_one_trajectory(
            sat, self.observer, self.times, 500, 400, 300)[0])
        sep = max(
            _sky_sep_deg(rust[i, 1], rust[i, 2], py[i, 1], py[i, 2])
            for i in range(py.shape[0]))
        self.assertLess(sep, 20.0 / 3600.0)


@unittest.skipUnless(_HAVE_ASTRO, "skytracker_core wheel lacks AstroEngine")
@unittest.skipUnless(_HAVE_DE421, "de421.bsp absent")
class CelestialParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from skyfield.api import load

        from celestial import get_celestial

        cls.ts = load.timescale()
        cls.cel = get_celestial()
        cls.t_tt = float(cls.ts.now().tt)

    def tearDown(self):
        _enable(False)

    def test_solar_system_altaz_matches(self):
        # Guard against a silent skyfield fallback masquerading as parity:
        # every body key must resolve on the Rust side before the A/B runs.
        for key in self.cel._bodies:
            self.assertIsNotNone(
                rust_astro_adapter.body_altaz_dist(key, self.t_tt, LAT, LON, ELEV),
                f"Rust engine cannot resolve body key {key!r} - the A/B "
                "would silently compare skyfield to itself")

        _enable(False)
        py = {k: (az, el, d) for k, _n, az, el, d, _c, _r in
              self.cel.solar_system_altaz(LAT, LON, ELEV, self.ts, self.t_tt)}
        _enable(True)
        rust = {k: (az, el, d) for k, _n, az, el, d, _c, _r in
                self.cel.solar_system_altaz(LAT, LON, ELEV, self.ts, self.t_tt)}
        self.assertEqual(set(py), set(rust))
        worst = 0.0
        for k in py:
            sep = _sky_sep_deg(rust[k][1], rust[k][0], py[k][1], py[k][0])
            worst = max(worst, sep)
            self.assertLess(sep, 60.0 / 3600.0, f"{k}: {sep * 3600:.1f} arcsec")
            self.assertLess(abs(rust[k][2] - py[k][2]) / max(py[k][2], 1.0), 1e-3, k)
        print(f"\n[parity] bodies: worst {worst * 3600:.2f} arcsec")

    def test_build_trajectory_body_and_fixed(self):
        keys = ["moon", "planet:Jupiter"]
        star_arrays = self.cel._named_star_arrays()
        if star_arrays["key"]:
            keys.append(star_arrays["key"][0])  # first named-star anchor
        for key in keys:
            _enable(False)
            py = self.cel.build_trajectory(key, LAT, LON, ELEV, self.ts, self.t_tt)
            self.assertIsNotNone(py, f"{key}: python path must resolve")
            _enable(True)
            rust = self.cel.build_trajectory(key, LAT, LON, ELEV, self.ts, self.t_tt)
            self.assertIsNotNone(rust, key)
            py_rows, _ = py
            rust_rows, _ = rust
            self.assertEqual(len(py_rows), len(rust_rows), key)
            worst = max(
                _sky_sep_deg(r[1], r[2], p[1], p[2])
                for r, p in zip(rust_rows, py_rows))
            self.assertLess(worst, 60.0 / 3600.0,
                            f"{key}: worst {worst * 3600:.1f} arcsec")
            worst_rate = max(
                max(abs(r[6] - p[6]), abs(r[7] - p[7]))
                for r, p in zip(rust_rows, py_rows))
            self.assertLess(worst_rate, 1e-4, f"{key} rates")
            print(f"[parity] build_trajectory {key}: worst "
                  f"{worst * 3600:.2f} arcsec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
