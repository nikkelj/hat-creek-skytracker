"""
Validate the 7-term alt-az pointing model: a least-squares fit recovers injected TPOINT
coefficients from plate-solve-style samples, and the residual/backtest RMS collapses.
"""

import math
import unittest

import numpy as np

from pointing_model import PointingModel, fibonacci_sky_grid, TERM_NAMES


class TestPointingModelFit(unittest.TestCase):
    def _make_samples(self, truth, grid, noise_arcsec=0.0, seed=0):
        rng = np.random.default_rng(seed)
        samples = []
        for az_cmd, el_cmd in grid:
            az_obs, el_obs = truth.predict_observed(az_cmd, el_cmd)
            if noise_arcsec:
                n = noise_arcsec / 3600.0
                az_obs += rng.normal(0, n) / max(0.2, math.cos(math.radians(el_cmd)))
                el_obs += rng.normal(0, n)
            samples.append((az_cmd, el_cmd, az_obs, el_obs))
        return samples

    def test_recovers_injected_terms_noiseless(self):
        truth = PointingModel({"IA": 1.5, "IE": -0.8, "AN": 0.05, "AW": -0.03,
                               "NPAE": 0.04, "CA": 0.06, "TF": 0.02})
        grid = fibonacci_sky_grid(40, el_min=15, el_max=80)
        samples = self._make_samples(truth, grid)
        model, stats = PointingModel.fit(samples)
        for k in TERM_NAMES:
            self.assertAlmostEqual(model.terms[k], truth.terms[k], delta=1e-3,
                                   msg=f"{k}: {model.terms[k]} vs {truth.terms[k]}")
        self.assertLess(stats["rms_after_arcmin"], 0.05)
        self.assertGreater(stats["rms_before_arcmin"], stats["rms_after_arcmin"])

    def test_recovers_terms_with_noise_and_backtest(self):
        truth = PointingModel({"IA": 2.0, "IE": 1.0, "AN": 0.08, "AW": 0.05,
                               "NPAE": -0.05, "CA": 0.03, "TF": 0.04})
        grid = fibonacci_sky_grid(50, el_min=15, el_max=80)
        # Split into fit / held-out backtest sets.
        fit_grid = grid[::2]
        test_grid = grid[1::2]
        fit_samples = self._make_samples(truth, fit_grid, noise_arcsec=10.0, seed=1)
        test_samples = self._make_samples(truth, test_grid, noise_arcsec=10.0, seed=2)

        model, stats = PointingModel.fit(fit_samples)
        # Individual coefficients are partly correlated (NPAE vs CA), so under 10" noise
        # they recover to a few hundredths of a degree; predictive RMS is the real test.
        for k in TERM_NAMES:
            self.assertAlmostEqual(model.terms[k], truth.terms[k], delta=0.05)
        # backtest RMS on held-out points should be near the injected noise floor (~10")
        backtest_rms_arcmin = model.backtest(test_samples) * 60.0
        self.assertLess(backtest_rms_arcmin, 1.0)

    def test_correct_then_observe_is_identity(self):
        truth = PointingModel({"IA": 1.0, "IE": -0.5, "AN": 0.1, "AW": 0.0,
                               "NPAE": 0.05, "CA": 0.0, "TF": 0.03})
        # Commanding correct(desired) should land the boresight on desired.
        for az_d, el_d in [(30, 45), (200, 60), (310, 25)]:
            az_cmd, el_cmd = truth.correct(az_d, el_d)
            az_obs, el_obs = truth.predict_observed(az_cmd, el_cmd)
            # first-order correction: within a few arcsec for these small terms
            self.assertLess(abs(((az_obs - az_d + 180) % 360 - 180)) * math.cos(math.radians(el_d)), 0.01)
            self.assertLess(abs(el_obs - el_d), 0.01)

    def test_grid_respects_elevation_band(self):
        grid = fibonacci_sky_grid(30, el_min=20, el_max=78)
        self.assertGreater(len(grid), 10)
        for az, el in grid:
            self.assertGreaterEqual(el, 20.0)
            self.assertLessEqual(el, 78.0)
            self.assertGreaterEqual(az, 0.0)
            self.assertLess(az, 360.0)


if __name__ == '__main__':
    unittest.main()
