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
import alignment as al
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

    def test_early_stop_halts_before_full_grid(self):
        """With a target RMS set, a clean fit should stop sampling well before the whole
        grid is consumed, yet still recover the model."""
        cfg = ConfigState()
        cfg.mount_mode = 'AltAz'
        cfg.elevation_mask_str = '10.0'
        state = AlignmentState()
        runner = AlignmentRunner(state, cfg, self._synthetic_sample_fn(noise_arcsec=0.0),
                                 n_points=40, holdout_frac=0.25, target_rms_arcmin=0.5)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        # Stopped early: far fewer fit samples than the available fit points.
        self.assertLess(len(state.samples), len(state.fit_points))
        self.assertLessEqual(len(state.samples), 20)
        for k, v in TRUTH.items():
            self.assertAlmostEqual(state.model.terms[k], v, delta=0.05)

    def test_holdout_disjoint_from_fit(self):
        cfg = ConfigState()
        state = AlignmentState()
        runner = AlignmentRunner(state, cfg, self._synthetic_sample_fn(), n_points=20)
        runner.run()
        fit_set = set(state.fit_points)
        hold_set = set(state.holdout_points)
        self.assertTrue(fit_set.isdisjoint(hold_set))
        self.assertGreater(len(hold_set), 0)


class _SimSolveSampler:
    """Live-style sampler (slew + capture) backed by SimMount + the sim's injected pointing
    model, bypassing the camera/plate-solve so the test is deterministic and tetra3-free but
    still drives the REAL closed-loop slew (slew_to_azel) and the AlignmentRunner live path.
    Closes the 'sim only recovers IA/IE' gap: with a full injected model it recovers all 7."""

    def __init__(self, config_state, mount, fov_deg=1.0):
        self.config_state = config_state
        self.mount = mount
        self.fov_deg = fov_deg

    def slew(self, az, el):
        return slew_to_azel(self.mount, self.config_state, az, el,
                            timeout=8.0, tol_deg=0.1, settle_cycles=2)

    def capture(self):
        from lib.auxstar import Targets
        from transformations import AzAlt2AzEl_AltAz
        from simulator import injected_boresight
        align_az = float(self.config_state.alignment_azimuth_str or 0.0)
        cur_azm = self.mount.hc_get_position(Targets.AZM) * 360.0
        cur_alt = self.mount.hc_get_position(Targets.ALT) * 360.0
        az_cmd, el_cmd = AzAlt2AzEl_AltAz(cur_azm, cur_alt, align_az)
        az_obs, el_obs = injected_boresight(self.config_state, az_cmd, el_cmd)
        return (az_cmd % 360.0, el_cmd, az_obs % 360.0, el_obs,
                {'kind': 'fit', 'n_matches': 12, 'rmse': 0.05, 'fov': self.fov_deg})

    def set_manual(self, on):
        pass


class TestSimMountEndToEnd(unittest.TestCase):
    """Full pointing model recovered through the real SimMount + slew loop + orchestration."""

    def _cfg(self):
        cfg = ConfigState()
        cfg.mount_mode = 'AltAz'
        cfg.alignment_azimuth_str = '0.0'
        cfg.alignment_elevation_str = '0.0'
        cfg.elevation_mask_str = '10.0'
        cfg.sim_config['mount_misalignment_az_deg'] = 0.0
        cfg.sim_config['mount_misalignment_el_deg'] = 0.0
        cfg.sim_config['mount_encoder_noise_deg'] = 0.0
        return cfg

    def test_recovers_full_injected_model(self):
        cfg = self._cfg()
        cfg.sim_config['mount_pointing_model'] = dict(TRUTH)
        mount = SimMount(cfg)
        state = AlignmentState()
        runner = AlignmentRunner(state, cfg, _SimSolveSampler(cfg, mount),
                                 n_points=18, holdout_frac=0.25)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        self.assertLess(state.stats['rms_after_arcmin'], 0.5)
        for k, v in TRUTH.items():
            self.assertAlmostEqual(state.model.terms[k], v, delta=0.05)
        self.assertIsNotNone(state.backtest_rms_deg)
        self.assertLess(state.backtest_rms_deg * 60.0, 1.0)

    def test_refraction_removed_beats_not_removed(self):
        cfg = self._cfg()
        cfg.sim_config['mount_pointing_model'] = {}
        cfg.sim_config['sim_refraction'] = True
        mount = SimMount(cfg)
        state = AlignmentState()
        # el_min lifted so refraction stays modest (and the geometric model ~ identity).
        runner = AlignmentRunner(state, cfg, _SimSolveSampler(cfg, mount),
                                 n_points=18, holdout_frac=0.25, el_min=30.0,
                                 remove_refraction=True)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        # Stripping refraction recovers a near-identity geometric model.
        self.assertLess(state.stats['rms_after_arcmin'], 0.5)
        # Same samples fit WITHOUT removing refraction leave a larger residual (refraction's
        # cot(el) shape isn't in the geometric basis).
        _, st_keep = PointingModel.fit(state.samples, remove_refraction=False)
        self.assertGreater(st_keep['rms_after_arcmin'], state.stats['rms_after_arcmin'])


class _SimEqSampler:
    """Eq live-style sampler: slews map sky (az,el) -> mount (HA,Dec) about the assumed pole,
    and capture returns (h_cmd, d_cmd, h_obs, d_obs) with the boresight distorted by an
    injected equatorial pointing model. Deterministic, tetra3-free."""

    fov_deg = 1.0

    def __init__(self, config_state, truth_model):
        self.config_state = config_state
        self.truth = truth_model
        self._h = 0.0
        self._d = 0.0

    def slew(self, az, el):
        from transformations import azel_to_eq_mount
        pole_az, pole_alt = al._assumed_pole(self.config_state)
        self._h, self._d = azel_to_eq_mount(az, el, pole_az, pole_alt)
        return True

    def capture(self):
        h_obs, d_obs = self.truth.predict_observed(self._h, self._d)
        return (self._h, self._d, h_obs, d_obs, {'kind': 'fit', 'n_matches': 12})

    def set_manual(self, on):
        pass


class TestEqAlignmentEndToEnd(unittest.TestCase):
    """The equatorial pointing model is recovered through the alignment orchestration."""

    def test_recovers_eq_model(self):
        from eq_pointing_model import EquatorialPointingModel
        lat = 42.0
        truth = {"IH": 0.5, "ID": -0.3, "NP": 0.04, "CH": 0.06, "ME": 0.03, "MA": -0.05, "TF": 0.08}
        cfg = ConfigState()
        cfg.mount_mode = 'Eq'
        cfg.lat_str = str(lat)
        cfg.alignment_azimuth_str = '0.0'
        cfg.alignment_elevation_str = '0.0'
        cfg.elevation_mask_str = '10.0'
        state = AlignmentState()
        sampler = _SimEqSampler(cfg, EquatorialPointingModel(truth, lat_deg=lat))
        runner = AlignmentRunner(state, cfg, sampler, n_points=30, holdout_frac=0.25)
        self.assertTrue(runner.eq_mode)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        self.assertIsInstance(state.model, EquatorialPointingModel)
        for k, v in truth.items():
            self.assertAlmostEqual(state.model.terms[k], v, delta=0.05, msg=k)
        # accept_alignment writes the EQUATORIAL config (not the alt-az terms).
        accept_alignment(state, cfg, save=False)
        self.assertTrue(cfg.eq_pointing_model_enabled)
        self.assertFalse(cfg.pointing_model_enabled)
        for k, v in truth.items():
            self.assertAlmostEqual(cfg.eq_pointing_model_terms[k], v, delta=0.05)


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
