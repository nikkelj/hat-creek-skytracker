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

    def test_partial_fit_refits_index_only(self):
        """A seeded partial fit recovers drifted IA/IE while holding the (correctly seeded)
        mechanical terms fixed -- the nightly-refresh path."""
        mech = {"AN": 0.07, "AW": -0.04, "NPAE": 0.05, "CA": 0.03, "TF": 0.04}
        truth = PointingModel({"IA": 2.3, "IE": -0.9, **mech})
        grid = fibonacci_sky_grid(8, el_min=20, el_max=78)
        samples = self._make_samples(truth, grid, noise_arcsec=8.0, seed=3)
        seed = dict(mech)  # last night's mechanical terms, zero index errors
        seed.update({"IA": 0.0, "IE": 0.0})
        model, stats = PointingModel.fit(samples, seed_terms=seed, free_terms=["IA", "IE"])
        # Mechanical terms must be held exactly at the seed (not re-fit).
        for k in mech:
            self.assertEqual(model.terms[k], mech[k])
        # Index errors recovered from only 8 points.
        self.assertAlmostEqual(model.terms["IA"], truth.terms["IA"], delta=0.03)
        self.assertAlmostEqual(model.terms["IE"], truth.terms["IE"], delta=0.03)
        self.assertLess(stats["rms_after_arcmin"], 1.0)

    def test_stats_report_condition_number(self):
        """A well-spread grid is well-conditioned; a clustered one is not."""
        truth = PointingModel({"IA": 1.0, "IE": -0.5, "AN": 0.05, "AW": 0.03,
                               "NPAE": 0.04, "CA": 0.02, "TF": 0.03})
        wide = self._make_samples(truth, fibonacci_sky_grid(40, el_min=15, el_max=80))
        _, st_wide = PointingModel.fit(wide)
        self.assertIn("design_cond", st_wide)
        # A grid clustered in a tiny az/el patch can't separate the terms -> huge cond.
        clustered = self._make_samples(truth, [(100 + 0.5 * i, 45 + 0.3 * i) for i in range(12)])
        _, st_cl = PointingModel.fit(clustered)
        self.assertGreater(st_cl["design_cond"], st_wide["design_cond"])

    def test_robust_fit_rejects_outlier(self):
        """A single grossly-wrong solve is rejected so it can't drag the model."""
        truth = PointingModel({"IA": 1.5, "IE": -0.6, "AN": 0.06, "AW": -0.04,
                               "NPAE": 0.05, "CA": 0.03, "TF": 0.04})
        grid = fibonacci_sky_grid(30, el_min=15, el_max=80)
        samples = self._make_samples(truth, grid, noise_arcsec=8.0, seed=5)
        # Corrupt one sample with a 2-degree plate-solve blunder.
        bad = list(samples[7]); bad[2] = (bad[2] + 2.0) % 360.0; bad[3] += 2.0
        samples[7] = tuple(bad)

        naive, st_naive = PointingModel.fit(samples, robust=False)
        robust, st_robust = PointingModel.fit(samples, robust=True)
        self.assertEqual(st_robust["n_rejected"], 1)
        self.assertLess(st_robust["rms_after_arcmin"], st_naive["rms_after_arcmin"])
        # Robust terms stay close to truth; the naive fit is pulled off by the blunder.
        for k in TERM_NAMES:
            self.assertAlmostEqual(robust.terms[k], truth.terms[k], delta=0.05)

    def test_robust_keeps_clean_samples(self):
        """With clean data, robust mode rejects nothing (the absolute floor protects inliers)."""
        truth = PointingModel({"IA": 1.0, "IE": 0.5, "AN": 0.05, "AW": 0.02,
                               "NPAE": 0.03, "CA": 0.02, "TF": 0.03})
        grid = fibonacci_sky_grid(24, el_min=20, el_max=78)
        samples = self._make_samples(truth, grid, noise_arcsec=8.0, seed=6)
        _, st = PointingModel.fit(samples, robust=True)
        self.assertEqual(st["n_rejected"], 0)

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
