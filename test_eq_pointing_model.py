"""
Equatorial 7-term pointing model (eq_pointing_model.py): the least-squares fit recovers
injected TPOINT coefficients in hour-angle/dec space, the correct/observe inverse holds,
robust rejection drops a blunder, and the control-loop Eq correction is wired (apply ==
model.correct, and the mount-position error vanishes at the corrected command).
"""

import math
import unittest

import numpy as np

from eq_pointing_model import EquatorialPointingModel, TERM_NAMES
from config import ConfigState
from control import apply_eq_pointing_model, compute_mount_position_error

LAT = 42.0
TRUTH = {"IH": 0.5, "ID": -0.3, "NP": 0.04, "CH": 0.06, "ME": 0.03, "MA": -0.05, "TF": 0.08}


def _grid_hd():
    return [(h, d) for h in range(-80, 81, 20) for d in range(-10, 71, 20)]


def _samples(model, grid, noise_arcsec=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for h, d in grid:
        h_obs, d_obs = model.predict_observed(h, d)
        if noise_arcsec:
            n = noise_arcsec / 3600.0
            h_obs += rng.normal(0, n) / max(0.2, math.cos(math.radians(d)))
            d_obs += rng.normal(0, n)
        out.append((h, d, h_obs, d_obs))
    return out


class TestEqModelFit(unittest.TestCase):
    def test_recovers_injected_terms_noiseless(self):
        truth = EquatorialPointingModel(TRUTH, lat_deg=LAT)
        model, stats = EquatorialPointingModel.fit(_samples(truth, _grid_hd()), LAT)
        for k in TERM_NAMES:
            self.assertAlmostEqual(model.terms[k], TRUTH[k], delta=1e-3, msg=k)
        self.assertLess(stats["rms_after_arcmin"], 0.05)
        self.assertIn("design_cond", stats)

    def test_recovers_with_noise(self):
        truth = EquatorialPointingModel(TRUTH, lat_deg=LAT)
        model, stats = EquatorialPointingModel.fit(
            _samples(truth, _grid_hd(), noise_arcsec=10.0, seed=2), LAT)
        for k in TERM_NAMES:
            self.assertAlmostEqual(model.terms[k], TRUTH[k], delta=0.05, msg=k)

    def test_correct_then_observe_is_identity(self):
        truth = EquatorialPointingModel(TRUTH, lat_deg=LAT)
        for h, d in [(0, 20), (45, -10), (-60, 55)]:
            h_cmd, d_cmd = truth.correct(h, d)
            h_obs, d_obs = truth.predict_observed(h_cmd, d_cmd)
            self.assertLess(abs(((h_obs - h + 180) % 360 - 180)) * math.cos(math.radians(d)), 0.01)
            self.assertLess(abs(d_obs - d), 0.01)

    def test_robust_rejects_outlier(self):
        truth = EquatorialPointingModel(TRUTH, lat_deg=LAT)
        samples = _samples(truth, _grid_hd(), noise_arcsec=8.0, seed=4)
        bad = list(samples[5]); bad[2] += 2.0; bad[3] += 2.0
        samples[5] = tuple(bad)
        naive, st_n = EquatorialPointingModel.fit(samples, LAT, robust=False)
        robust, st_r = EquatorialPointingModel.fit(samples, LAT, robust=True)
        # The blunder is rejected (a high-leverage 2 deg outlier can also swamp a few clean
        # neighbours into rejection -- harmless with 45 points; the point is a far better fit).
        self.assertGreaterEqual(st_r["n_rejected"], 1)
        self.assertLess(st_r["rms_after_arcmin"], st_n["rms_after_arcmin"])
        for k in TERM_NAMES:
            self.assertAlmostEqual(robust.terms[k], TRUTH[k], delta=0.05, msg=k)

    def test_flexure_depends_on_latitude(self):
        # The TF basis is latitude-dependent (gravity), unlike the other terms.
        m_lo = EquatorialPointingModel({"TF": 0.1}, lat_deg=20.0)
        m_hi = EquatorialPointingModel({"TF": 0.1}, lat_deg=60.0)
        self.assertNotAlmostEqual(m_lo.error(40, 10)[1], m_hi.error(40, 10)[1], places=4)


class TestEqControlIntegration(unittest.TestCase):
    def _cfg(self):
        cfg = ConfigState()
        cfg.mount_mode = 'Eq'
        cfg.lat_str = str(LAT)
        cfg.alignment_azimuth_str = '0.0'
        cfg.alignment_elevation_str = '0.0'  # pole_alt falls back to latitude
        cfg.eq_pointing_model_enabled = True
        cfg.eq_pointing_model_terms = dict(TRUTH)
        return cfg

    def test_apply_matches_model_correct(self):
        cfg = self._cfg()
        model = EquatorialPointingModel(TRUTH, lat_deg=LAT)
        self.assertEqual(apply_eq_pointing_model(cfg, 30.0, 25.0), model.correct(30.0, 25.0))

    def test_disabled_is_noop(self):
        cfg = self._cfg(); cfg.eq_pointing_model_enabled = False
        self.assertEqual(apply_eq_pointing_model(cfg, 30.0, 25.0), (30.0, 25.0))

    def test_error_zero_at_corrected_command(self):
        # If the mount sits exactly where the model says to command for a target, the
        # computed mount-position error is ~0 (the correction is self-consistent).
        from transformations import azel_to_eq_mount
        cfg = self._cfg()
        target_az, target_el = 210.0, 40.0
        h_des, d_des = azel_to_eq_mount(target_az, target_el, 0.0, LAT)
        model = EquatorialPointingModel(TRUTH, lat_deg=LAT)
        h_cmd, d_cmd = model.correct(h_des, d_des)
        az_err, alt_err = compute_mount_position_error(cfg, h_cmd, d_cmd, target_az, target_el)
        self.assertLess(abs(az_err), 0.02)
        self.assertLess(abs(alt_err), 0.02)


if __name__ == '__main__':
    unittest.main(verbosity=2)
