"""
Plate-solve polar alignment (polar_align.py): the axis-fit math, the error/correction
decomposition, and an end-to-end run against a deterministic sampler that rotates the RA
axis about an injected (mis-set) polar axis -- the routine must recover that axis and report
the right azimuth/altitude correction.
"""

import math
import unittest

import numpy as np

from config import ConfigState
from transformations import eq_mount_to_azel
from polar_align import (PolarAlignState, PolarAlignRunner, fit_polar_axis,
                         polar_error, DONE, ERROR)


class _SimPolarSampler:
    """Boresight = eq_mount_to_azel(AZM, ALT) about the injected TRUE pole; optional noise."""

    def __init__(self, true_pole_az, true_pole_alt, noise_deg=0.0, seed=0, fail_all=False):
        self.az = true_pole_az
        self.alt = true_pole_alt
        self.noise = noise_deg
        self.fail_all = fail_all
        self._azm = 0.0
        self._alt = 0.0
        self._rng = np.random.default_rng(seed)

    def slew(self, azm, alt):
        self._azm, self._alt = azm, alt
        return True

    def capture(self):
        if self.fail_all:
            return None
        az, el = eq_mount_to_azel(self._azm, self._alt, self.az, self.alt)
        if self.noise:
            az += self._rng.normal(0, self.noise) / max(0.2, math.cos(math.radians(el)))
            el += self._rng.normal(0, self.noise)
        return (az % 360.0, el, {'n_matches': 10})


LAT = 40.0


class TestPolarMath(unittest.TestCase):
    def test_fit_axis_recovers_known_pole(self):
        # Boresights 20 deg from a pole at (3, 42), swept in HA.
        pole = (3.0, 42.0)
        samples = [eq_mount_to_azel(ha, 20.0, *pole) for ha in (-40, -20, 0, 20, 40)]
        az, el = fit_polar_axis(samples, toward_az_deg=0.0, toward_alt_deg=LAT)
        self.assertAlmostEqual(az, 3.0, places=2)
        self.assertAlmostEqual(el, 42.0, places=2)

    def test_error_directions(self):
        # Pole too far east and too high.
        r = polar_error(measured_az=2.0, measured_alt=42.0, target_az=0.0, target_alt=40.0)
        self.assertAlmostEqual(r['az_error'], 2.0, places=6)
        self.assertAlmostEqual(r['alt_error'], 2.0, places=6)
        self.assertEqual(r['az_dir'], 'west')   # correction: move west
        self.assertEqual(r['alt_dir'], 'down')  # correction: lower
        self.assertGreater(r['total_deg'], 0.0)

    def test_perfect_alignment_is_zero(self):
        r = polar_error(0.0, LAT, 0.0, LAT)
        self.assertLess(r['total_deg'], 1e-4)  # acos near 1.0 leaves ~sub-arcsec float noise


class TestPolarRun(unittest.TestCase):
    def _run(self, true_az, true_alt, **kw):
        cfg = ConfigState()
        cfg.mount_mode = 'Eq'
        cfg.lat_str = str(LAT)
        state = PolarAlignState()
        sampler = _SimPolarSampler(true_az, true_alt, **kw)
        runner = PolarAlignRunner(state, cfg, sampler, target_az=0.0, target_alt=LAT,
                                  n_points=6, dec_deg=20.0, ha_span_deg=90.0)
        runner.run()
        return state

    def test_recovers_injected_misalignment(self):
        state = self._run(2.5, 41.5)
        self.assertEqual(state.phase, DONE, state.error)
        az_m, alt_m = state.measured_pole
        self.assertAlmostEqual(az_m, 2.5, places=2)
        self.assertAlmostEqual(alt_m, 41.5, places=2)
        self.assertAlmostEqual(state.result['az_error'], 2.5, delta=0.02)
        self.assertAlmostEqual(state.result['alt_error'], 1.5, delta=0.02)

    def test_recovers_under_noise(self):
        state = self._run(-1.5, 38.5, noise_deg=10.0 / 3600.0, seed=3)
        self.assertEqual(state.phase, DONE, state.error)
        self.assertAlmostEqual(state.result['az_error'], -1.5, delta=0.1)
        self.assertAlmostEqual(state.result['alt_error'], -1.5, delta=0.1)

    def test_total_solve_failure_errors(self):
        cfg = ConfigState(); cfg.lat_str = str(LAT)
        state = PolarAlignState()
        runner = PolarAlignRunner(state, cfg, _SimPolarSampler(0, LAT, fail_all=True),
                                  target_az=0.0, target_alt=LAT, n_points=5)
        runner.run()
        self.assertEqual(state.phase, ERROR)
        self.assertIsNone(state.measured_pole)


if __name__ == '__main__':
    unittest.main(verbosity=2)
