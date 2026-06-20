"""
Phase 4 orchestration: the AlignmentRunner samples a sky grid, fits the pointing model,
backtests on held-out points, and accept_alignment persists it. Uses a synthetic
sample_fn that emulates a misaligned mount via a known PointingModel (so the run is
deterministic and hardware-free); the live slew+solve path is covered by test_plate_solve.
"""

import unittest

import numpy as np

from config import ConfigState
from pointing_model import PointingModel
from alignment import AlignmentState, AlignmentRunner, accept_alignment, slew_to_azel, DONE
from simulator import SimMount
from transformations import AzEl2AzAlt_AltAz


TRUTH = {"IA": 1.5, "IE": -0.6, "AN": 0.06, "AW": -0.04, "NPAE": 0.05, "CA": 0.03, "TF": 0.04}


class TestAlignmentRunner(unittest.TestCase):
    def _synthetic_sample_fn(self, noise_arcsec=8.0, seed=0):
        truth = PointingModel(TRUTH)
        rng = np.random.default_rng(seed)

        def sample_fn(az_nom, el_nom):
            az_obs, el_obs = truth.predict_observed(az_nom, el_nom)
            if noise_arcsec:
                n = noise_arcsec / 3600.0
                az_obs += rng.normal(0, n)
                el_obs += rng.normal(0, n)
            return az_obs % 360.0, el_obs
        return sample_fn

    def test_full_run_recovers_model_and_backtests(self):
        cfg = ConfigState()
        cfg.mount_mode = 'AltAz'
        cfg.elevation_mask_str = '10.0'
        state = AlignmentState()
        runner = AlignmentRunner(state, cfg, self._synthetic_sample_fn(),
                                 n_points=24, holdout_frac=0.25)
        runner.run()  # synchronous (no real hardware) for the test

        self.assertEqual(state.phase, DONE, state.error)
        self.assertIsNotNone(state.model)
        # fit collapses the error
        self.assertLess(state.stats['rms_after_arcmin'], 0.5)
        self.assertGreater(state.stats['rms_before_arcmin'], state.stats['rms_after_arcmin'])
        # recovered terms close to truth
        for k, v in TRUTH.items():
            self.assertAlmostEqual(state.model.terms[k], v, delta=0.05)
        # backtest on held-out points is near the noise floor
        self.assertIsNotNone(state.backtest_rms_deg)
        self.assertLess(state.backtest_rms_deg * 60.0, 1.0)

    def test_accept_writes_and_enables(self):
        cfg = ConfigState()
        cfg.mount_mode = 'AltAz'
        state = AlignmentState()
        runner = AlignmentRunner(state, cfg, self._synthetic_sample_fn(noise_arcsec=0.0),
                                 n_points=20)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        ok = accept_alignment(state, cfg, save=False)
        self.assertTrue(ok)
        self.assertTrue(cfg.pointing_model_enabled)
        for k, v in TRUTH.items():
            self.assertAlmostEqual(cfg.pointing_model_terms[k], v, delta=0.01)

    def test_holdout_disjoint_from_fit(self):
        cfg = ConfigState()
        state = AlignmentState()
        runner = AlignmentRunner(state, cfg, self._synthetic_sample_fn(), n_points=20)
        runner.run()
        fit_set = set(state.fit_points)
        hold_set = set(state.holdout_points)
        self.assertTrue(fit_set.isdisjoint(hold_set))
        self.assertGreater(len(hold_set), 0)


class TestSlewSettle(unittest.TestCase):
    """The closed-loop slew must actually converge in the sim (regression for the
    'alignment gets stuck' report) -- coarse goto then rate-settle onto the target."""

    def test_slew_converges_with_misalignment(self):
        cfg = ConfigState()
        cfg.mount_mode = 'AltAz'
        cfg.alignment_azimuth_str = '0.0'
        cfg.alignment_elevation_str = '0.0'
        cfg.sim_config['mount_misalignment_az_deg'] = 2.0
        cfg.sim_config['mount_misalignment_el_deg'] = 1.0
        mount = SimMount(cfg)

        ok = slew_to_azel(mount, cfg, 100.0, 45.0, timeout=15.0, tol_deg=0.03)
        self.assertTrue(ok, "slew failed to settle")

        # Reported encoder should sit on the commanded mount coords (the loop drives the
        # encoder, which includes the misalignment, to target).
        azm_t, alt_t = AzEl2AzAlt_AltAz(100.0, 45.0, 0.0, 0.0)
        from lib.auxstar import Targets
        rep_azm = mount.hc_get_position(Targets.AZM) * 360.0
        rep_alt = mount.hc_get_position(Targets.ALT) * 360.0
        self.assertLess(abs(((rep_azm - azm_t + 180) % 360) - 180), 0.1)
        self.assertLess(abs(rep_alt - alt_t), 0.1)


if __name__ == '__main__':
    unittest.main()
