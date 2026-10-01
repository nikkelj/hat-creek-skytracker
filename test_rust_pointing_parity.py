#!/usr/bin/env python
"""Live A/B parity: Rust pointing-model fits vs the numpy implementations.

Synthesizes ground-truth models, generates noisy plate-solve samples (with
planted outliers for the robust mode), and fits with the flag off and on
through the REAL classmethod entry points. Gate: coefficients within 1e-6
degrees, identical rejection counts, matching stats.

Build first: cd rust/skytracker-ffi && maturin develop --release
Run: python -m pytest test_rust_pointing_parity.py
"""

import math
import os
import unittest

import numpy as np

try:
    import skytracker_core
    _HAVE = getattr(skytracker_core, "POINTING_AVAILABLE", False)
except ImportError:
    _HAVE = False

from eq_pointing_model import EquatorialPointingModel
from pointing_model import PointingModel, fibonacci_sky_grid

LAT = 34.8740289
TRUTH = {"IA": -0.021, "IE": 0.065, "AN": 0.0015, "AW": -0.0035,
         "NPAE": -0.023, "CA": 0.030, "TF": 0.10}
TRUTH_EQ = {"IH": 0.012, "ID": -0.04, "NP": 0.008, "CH": -0.015,
            "ME": 0.03, "MA": -0.02, "TF": 0.05}


def _flag(on):
    os.environ["SKYTRACKER_RUST_POINTING"] = "1" if on else "0"


def _make_altaz_samples(rng, n=24, noise=0.002, outliers=0):
    model = PointingModel(TRUTH)
    samples = []
    for az, el in fibonacci_sky_grid(n, 20.0, 75.0):
        obs_az, obs_el = model.predict_observed(az, el)
        obs_az += rng.normal(0, noise)
        obs_el += rng.normal(0, noise)
        samples.append((az, el, obs_az, obs_el))
    for i in range(outliers):
        az, el, oa, oe = samples[i * 3]
        samples[i * 3] = (az, el, oa + 0.5, oe - 0.4)
    return samples


def _make_eq_samples(rng, n=24, noise=0.002):
    model = EquatorialPointingModel(TRUTH_EQ, lat_deg=LAT)
    samples = []
    for i in range(n):
        h = -60.0 + 120.0 * i / (n - 1)
        d = -20.0 + 70.0 * ((i * 7) % n) / (n - 1)
        oh, od = model.predict_observed(h, d)
        samples.append((h, d, oh + rng.normal(0, noise), od + rng.normal(0, noise)))
    return samples


@unittest.skipUnless(_HAVE, "skytracker_core wheel lacks pointing fits")
class PointingParity(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("SKYTRACKER_RUST_POINTING", None)

    def _compare(self, py, rs, label, tol=1e-6):
        py_model, py_stats = py
        rs_model, rs_stats = rs
        worst = 0.0
        for k in py_model.terms:
            d = abs(py_model.terms[k] - rs_model.terms[k])
            worst = max(worst, d)
            self.assertLess(d, tol, f"{label} term {k}: {d:.2e}")
        self.assertEqual(py_stats["n_samples"], rs_stats["n_samples"], label)
        self.assertEqual(py_stats["n_rejected"], rs_stats["n_rejected"], label)
        self.assertLess(
            abs(py_stats["rms_after_deg"] - rs_stats["rms_after_deg"]), 1e-9, label)
        self.assertLess(
            abs(py_stats["design_cond"] - rs_stats["design_cond"])
            / max(py_stats["design_cond"], 1.0), 1e-6, label)
        return worst

    def test_altaz_full_fit(self):
        rng = np.random.default_rng(1)
        samples = _make_altaz_samples(rng)
        _flag(False)
        py = PointingModel.fit(samples, remove_refraction=True)
        _flag(True)
        rs = PointingModel.fit(samples, remove_refraction=True)
        worst = self._compare(py, rs, "altaz full")
        print(f"\n[parity] altaz full fit: worst term diff {worst:.2e} deg")

    def test_altaz_partial_seeded_fit(self):
        rng = np.random.default_rng(2)
        samples = _make_altaz_samples(rng, n=8)
        seed = dict(TRUTH)
        _flag(False)
        py = PointingModel.fit(samples, seed_terms=seed, free_terms=["IA", "IE"])
        _flag(True)
        rs = PointingModel.fit(samples, seed_terms=seed, free_terms=["IA", "IE"])
        self._compare(py, rs, "altaz partial")
        # Non-free terms must stay at the seed on both sides.
        for k in ("AN", "AW", "NPAE", "CA", "TF"):
            self.assertEqual(py[0].terms[k], seed[k])
            self.assertEqual(rs[0].terms[k], seed[k])

    def test_altaz_robust_rejects_outliers(self):
        rng = np.random.default_rng(3)
        samples = _make_altaz_samples(rng, n=24, outliers=3)
        _flag(False)
        py = PointingModel.fit(samples, robust=True)
        _flag(True)
        rs = PointingModel.fit(samples, robust=True)
        self.assertGreater(py[1]["n_rejected"], 0, "outliers should be rejected")
        self._compare(py, rs, "altaz robust")
        print(f"[parity] altaz robust: both rejected {rs[1]['n_rejected']}")

    def test_eq_full_and_robust(self):
        rng = np.random.default_rng(4)
        samples = _make_eq_samples(rng)
        for robust in (False, True):
            _flag(False)
            py = EquatorialPointingModel.fit(samples, LAT, robust=robust)
            _flag(True)
            rs = EquatorialPointingModel.fit(samples, LAT, robust=robust)
            worst = self._compare(py, rs, f"eq robust={robust}")
        print(f"[parity] eq fits: worst term diff {worst:.2e} deg")

    def test_polar_axis_fit(self):
        import polar_align

        rng = np.random.default_rng(5)
        # Points on a cone about a tilted axis (RA sweep).
        axis_az, axis_el = 2.5, LAT + 0.8
        from polar_align import cartesian_from_az_el

        ax = np.asarray(cartesian_from_az_el(axis_az, axis_el), dtype=float)
        ref = np.array([0.0, 0.0, 1.0])
        u = np.cross(ax, ref)
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        samples = []
        for ang in np.linspace(0, 300, 9):
            r = math.radians(ang)
            cone = math.radians(30.0)
            p = math.cos(cone) * ax + math.sin(cone) * (math.cos(r) * u + math.sin(r) * v)
            p += rng.normal(0, 1e-4, 3)
            p /= np.linalg.norm(p)
            az = math.degrees(math.atan2(p[0], p[1])) % 360.0
            el = math.degrees(math.asin(np.clip(p[2], -1, 1)))
            samples.append((az, el))

        _flag(False)
        py = polar_align.fit_polar_axis(samples, toward_az_deg=0.0, toward_alt_deg=45.0)
        _flag(True)
        rs = polar_align.fit_polar_axis(samples, toward_az_deg=0.0, toward_alt_deg=45.0)
        d_az = abs((py[0] - rs[0] + 180.0) % 360.0 - 180.0)
        d_el = abs(py[1] - rs[1])
        self.assertLess(d_az * math.cos(math.radians(py[1])), 1e-6)
        self.assertLess(d_el, 1e-6)
        print(f"[parity] polar axis: daz {d_az:.2e} del {d_el:.2e} deg "
              f"(axis {rs[0]:.3f}, {rs[1]:.3f})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
