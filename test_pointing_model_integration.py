"""
Verify the pointing-model correction is wired consistently into the control path.

Both the Python control loop (control.compute_mount_position_error) and the Rust-loop
adapter call the SAME control.apply_pointing_model, so the correction can't drift between
the two cores. The transforms themselves stay geometric (Rust parity preserved).
"""

import math
import unittest

from config import ConfigState
from control import apply_pointing_model, compute_mount_position_error
from pointing_model import PointingModel
from transformations import AzEl2AzAlt_AltAz

TERMS = {"IA": 1.2, "IE": -0.7, "AN": 0.05, "AW": 0.03, "NPAE": 0.04, "CA": 0.02, "TF": 0.03}


class TestPointingModelIntegration(unittest.TestCase):
    def _cfg(self, enabled):
        cfg = ConfigState()
        cfg.mount_mode = 'AltAz'
        cfg.alignment_azimuth_str = '0.0'
        cfg.alignment_elevation_str = '0.0'
        cfg.pointing_model_enabled = enabled
        cfg.pointing_model_terms = dict(TERMS)
        return cfg

    def test_apply_matches_model_correct(self):
        cfg = self._cfg(True)
        model = PointingModel(TERMS)
        for az, el in [(10, 30), (130, 55), (300, 20)]:
            self.assertEqual(apply_pointing_model(cfg, az, el), model.correct(az, el))

    def test_disabled_is_noop(self):
        cfg = self._cfg(False)
        for az, el in [(10, 30), (200, 70)]:
            self.assertEqual(apply_pointing_model(cfg, az, el), (az, el))

    def test_error_zero_when_at_corrected_target(self):
        """If the mount sits exactly where the model says to command, error is ~0."""
        cfg = self._cfg(True)
        model = PointingModel(TERMS)
        target_az, target_el = 120.0, 50.0
        cmd_az, cmd_el = model.correct(target_az, target_el)
        cur_azm, cur_alt = AzEl2AzAlt_AltAz(cmd_az, cmd_el, 0.0, 0.0)
        az_err, el_err = compute_mount_position_error(cfg, cur_azm, cur_alt, target_az, target_el)
        self.assertLess(abs(az_err), 1e-6)
        self.assertLess(abs(el_err), 1e-6)

    def test_correction_actually_changes_command(self):
        """With the model on, the commanded mount position differs from the naive one."""
        cfg_on = self._cfg(True)
        cfg_off = self._cfg(False)
        target_az, target_el = 80.0, 40.0
        # naive mount coords (model off) vs corrected (model on): compare the error each
        # reports at the SAME current position -> they must differ by the correction.
        cur_azm, cur_alt = AzEl2AzAlt_AltAz(target_az, target_el, 0.0, 0.0)
        e_off = compute_mount_position_error(cfg_off, cur_azm, cur_alt, target_az, target_el)
        e_on = compute_mount_position_error(cfg_on, cur_azm, cur_alt, target_az, target_el)
        self.assertGreater(abs(e_off[0] - e_on[0]) + abs(e_off[1] - e_on[1]), 1e-3)


if __name__ == '__main__':
    unittest.main()
