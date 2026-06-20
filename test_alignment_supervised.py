"""
Supervised-alignment engine: the GuiSampler-style object path (slew + capture returning
the full encoder-derived 5-tuple), the automatic grid search around a failed point, and
the resume/retry-failed pass that appends samples and refits. Uses a fake sampler that
emulates a misaligned mount via a known PointingModel so the run is deterministic and
hardware-free (the live camera/solve path is covered by test_plate_solve).
"""

import unittest

from config import ConfigState
from pointing_model import PointingModel
from alignment import AlignmentState, AlignmentRunner, DONE, MIN_SAMPLES, _spiral_offsets


TRUTH = {"IA": 1.5, "IE": -0.6, "AN": 0.06, "AW": -0.04, "NPAE": 0.05, "CA": 0.03, "TF": 0.04}


class FakeSampler:
    """Object-style sampler (has .capture, so the runner treats it as the live path).

    capture() returns the full (az_cmd, el_cmd, az_obs, el_obs, meta) tuple where az_cmd
    is the *current slewed position* -- i.e. the encoder-derived command, exactly as the
    real GuiSampler reads it. ``fail_until`` makes the first N capture() calls return None
    (to exercise grid search / failed-point handling); after that everything solves.
    """

    def __init__(self, fail_until=0, fov_deg=2.0):
        self.truth = PointingModel(TRUTH)
        self.fov_deg = fov_deg
        self.fail_until = fail_until
        self.calls = 0
        self.manual_calls = []
        self._pos = (0.0, 0.0)

    def slew(self, az, el):
        self._pos = (az % 360.0, el)
        return True

    def capture(self):
        self.calls += 1
        if self.calls <= self.fail_until:
            return None
        az, el = self._pos
        az_obs, el_obs = self.truth.predict_observed(az, el)
        return (az, el, az_obs % 360.0, el_obs, {'n_matches': 12, 'rmse': 0.4, 'fov': self.fov_deg})

    def set_manual(self, on):
        self.manual_calls.append(on)


def _cfg():
    cfg = ConfigState()
    cfg.mount_mode = 'AltAz'
    cfg.elevation_mask_str = '10.0'
    return cfg


class TestObjectSamplerPath(unittest.TestCase):
    def test_full_run_recovers_model(self):
        state = AlignmentState()
        state.pause_on_fail = False
        runner = AlignmentRunner(state, _cfg(), FakeSampler(), n_points=24)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        self.assertIsNotNone(state.model)
        for k, v in TRUTH.items():
            self.assertAlmostEqual(state.model.terms[k], v, delta=0.05)
        # The 5-tuple capture path records per-sample meta aligned with samples.
        self.assertEqual(len(state.sample_meta), len(state.samples))
        self.assertEqual(state.sample_meta[0]['n_matches'], 12)
        # Mount handed back from any manual control at the end.
        self.assertEqual(state.manual_active, False)

    def test_grid_search_recovers_first_point(self):
        # The nominal capture of point 1 fails once; the grid search around it then solves,
        # so no point is lost and az_cmd reflects the (offset) encoder position.
        state = AlignmentState()
        state.pause_on_fail = False
        sampler = FakeSampler(fail_until=1)
        runner = AlignmentRunner(state, _cfg(), sampler, n_points=24)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        self.assertEqual(len(state.failed_points), 0, "grid search should have recovered the point")
        self.assertEqual(len(state.samples), len(state.fit_points))


class TestRetryFailed(unittest.TestCase):
    def test_failed_point_then_retry_appends_and_refits(self):
        # With a 12-cell grid search, fail the nominal + all 12 grid offsets of point 1
        # (13 capture calls) so it lands in failed_points; everything after solves.
        # pause_on_fail off so we don't block.
        state = AlignmentState()
        state.pause_on_fail = False
        sampler = FakeSampler(fail_until=13)
        runner = AlignmentRunner(state, _cfg(), sampler, n_points=24, grid_cells=12)
        runner.run()
        self.assertEqual(state.phase, DONE, state.error)
        self.assertEqual(len(state.failed_points), 1)
        self.assertGreaterEqual(len(state.samples), MIN_SAMPLES)
        n_before = len(state.samples)
        failed = list(state.failed_points)

        # Retry pass: re-run only the failed point, appending (no reset) and refitting.
        state.failed_points = []
        retry = AlignmentRunner(state, _cfg(), sampler,
                                pending_points=failed, append=True, grid_cells=12)
        retry.run()
        self.assertEqual(state.phase, DONE, state.error)
        self.assertEqual(len(state.failed_points), 0)
        self.assertEqual(len(state.samples), n_before + 1)
        self.assertIsNotNone(state.model)


class TestGridSearchSizing(unittest.TestCase):
    def test_spiral_offsets_count_unique_and_nearest_first(self):
        pts = _spiral_offsets(100)
        self.assertEqual(len(pts), 100)
        self.assertEqual(len(set(pts)), 100, "offsets must be unique")
        self.assertNotIn((0, 0), pts, "origin (the failed point itself) is excluded")
        # Nearest-first: the first ring (Chebyshev radius 1) is the 8 closest cells.
        ring1 = {(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)} - {(0, 0)}
        self.assertEqual(set(pts[:8]), ring1)

    def test_grid_cells_config_is_honored(self):
        # The runner reads config.alignment_grid_search_cells; a small value bounds the
        # number of capture attempts in the grid search.
        cfg = _cfg()
        cfg.alignment_grid_search_cells = 7
        sampler = FakeSampler(fail_until=10 ** 9)  # never solves
        runner = AlignmentRunner(AlignmentState(), cfg, sampler)
        self.assertEqual(runner.grid_cells, 7)
        out = runner._grid_search(120.0, 40.0)
        self.assertIsNone(out)
        self.assertEqual(sampler.calls, 7, "grid search should try exactly grid_cells cells")


if __name__ == '__main__':
    unittest.main()
