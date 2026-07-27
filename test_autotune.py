#!/usr/bin/env python
"""Unit tests for the online PID auto-tuner (autotune.py).

The tuner is driven on a simulated clock against a synthetic plant: a
deterministic cost surface mapping the current config gains to a tracking-RMS
value, fed back to update() as the per-cycle position error. Deterministic
errors make every assertion exact -- these tests exercise the twiddle
mechanics (descent, clamping, pause/resume, divergence revert, stop
semantics), not control-loop physics; the full-wiring behavior is covered by
the live tracking-quality suite.

Run: python test_autotune.py
"""

import math
import unittest

import autotune
from autotune import PIDAutoTuner, GAIN_MIN, GAIN_MAX, PARAM_ORDER


class FakeConfig:
    def __init__(self, **overrides):
        self.pid_azm_p_gain = 0.10
        self.pid_azm_i_gain = 0.0001
        self.pid_azm_d_gain = 0.0005
        self.pid_alt_p_gain = 0.05
        self.pid_alt_i_gain = 0.0002
        self.pid_alt_d_gain = 0.0010
        for k, v in overrides.items():
            setattr(self, k, v)


LOG = math.log10

# Synthetic plant optima (log-space quadratic bowls, one per axis).
AZM_OPT = {'p': 0.02, 'd': 0.002, 'i': 0.0003}
ALT_OPT = {'p': 0.03, 'd': 0.001, 'i': 0.0005}
FLOOR_DEG = 0.01


def bowl_rms(cfg, axis, opt, floor=FLOOR_DEG):
    """Tracking RMS (deg) for the axis's current gains: a quadratic bowl in
    log-gain space with its minimum `floor` at the optimum."""
    p = getattr(cfg, f'pid_{axis}_p_gain')
    d = getattr(cfg, f'pid_{axis}_d_gain')
    i = getattr(cfg, f'pid_{axis}_i_gain')
    return floor * (1.0
                    + (LOG(p) - LOG(opt['p'])) ** 2
                    + 0.3 * (LOG(d) - LOG(opt['d'])) ** 2
                    + 0.1 * (LOG(i) - LOG(opt['i'])) ** 2)


def drive(tuner, error_fn, sim_seconds=900.0, hz=10.0, tracking_fn=None,
          on_step=None):
    """Run the tuner on a simulated clock until it finishes (or the budget
    runs out). error_fn(axis) -> current error magnitude (deg)."""
    t0 = 1000.0
    tuner.start(now=t0)
    t, dt = t0, 1.0 / hz
    while tuner.active and t < t0 + sim_seconds:
        t += dt
        tracking = True if tracking_fn is None else tracking_fn(t - t0)
        tuner.update(t, tracking, error_fn('azm'), error_fn('alt'))
        if on_step is not None:
            on_step(t - t0)
    return t - t0


class ConvergenceTests(unittest.TestCase):

    def test_descends_both_axes_and_stays_clamped(self):
        cfg = FakeConfig()
        err = lambda axis: bowl_rms(cfg, axis, AZM_OPT if axis == 'azm' else ALT_OPT)
        initial = {'azm': err('azm'), 'alt': err('alt')}
        tuner = PIDAutoTuner(cfg)

        def check_clamps(_t):
            for axis in ('azm', 'alt'):
                for k in PARAM_ORDER:
                    g = getattr(cfg, f'pid_{axis}_{k}_gain')
                    self.assertGreaterEqual(g, GAIN_MIN)
                    self.assertLessEqual(g, GAIN_MAX)

        drive(tuner, err, on_step=check_clamps)

        self.assertEqual(tuner.phase, 'done', "tuner should converge on a smooth bowl")
        for axis, opt in (('azm', AZM_OPT), ('alt', ALT_OPT)):
            final = err(axis)
            # Score the reducible part: the bowl bottoms out at FLOOR_DEG, so
            # compare the excess above the floor, not the raw cost.
            self.assertLess(final - FLOOR_DEG, 0.35 * (initial[axis] - FLOOR_DEG),
                            f"{axis}: cost should drop toward the floor "
                            f"({initial[axis]:.4f} -> {final:.4f})")
            p = getattr(cfg, f'pid_{axis}_p_gain')
            self.assertLess(abs(LOG(p) - LOG(opt['p'])), 0.6,
                            f"{axis}: P should land near the optimum (got {p})")

    def test_status_text_is_string_all_phases(self):
        cfg = FakeConfig()
        err = lambda axis: bowl_rms(cfg, axis, AZM_OPT)
        tuner = PIDAutoTuner(cfg)
        self.assertIsInstance(tuner.status_text(), str)
        drive(tuner, err, sim_seconds=20.0)
        self.assertIsInstance(tuner.status_text(), str)
        tuner.stop()
        self.assertIsInstance(tuner.status_text(), str)


class PauseResumeTests(unittest.TestCase):

    def test_pause_holds_best_gains_and_resume_completes(self):
        cfg = FakeConfig()
        err = lambda axis: bowl_rms(cfg, axis, AZM_OPT if axis == 'azm' else ALT_OPT)
        tuner = PIDAutoTuner(cfg)

        # Tracking drops out for 30 s mid-tune (e.g. lost optical lock).
        tracking_fn = lambda t: not (60.0 <= t < 90.0)
        paused_gain_snapshots = []

        def snoop(t):
            if 61.0 < t < 89.0 and tuner.phase == 'paused':
                paused_gain_snapshots.append(
                    {f'{a}_{k}': getattr(cfg, f'pid_{a}_{k}_gain')
                     for a in ('azm', 'alt') for k in PARAM_ORDER})

        drive(tuner, err, tracking_fn=tracking_fn, on_step=snoop)

        self.assertEqual(tuner.phase, 'done', "tuner should finish despite the gap")
        # While paused, the live gains must equal the tuner's best-known set
        # (no half-probed candidate left driving the mount).
        self.assertTrue(paused_gain_snapshots, "pause window was never observed")
        for snap in paused_gain_snapshots:
            for a, ax in tuner.axes.items():
                for k in PARAM_ORDER:
                    self.assertAlmostEqual(
                        snap[f'{a}_{k}'], round(min(GAIN_MAX, max(GAIN_MIN, ax.best[k])), 6),
                        places=9)


class DivergenceTests(unittest.TestCase):

    def test_diverging_candidate_is_reverted(self):
        # Plant blows up (10 deg error) the moment AZM P exceeds 0.05: the
        # up-probe from P=0.04 must be reverted by the in-window guard and the
        # tuner must still finish with P at or below the cliff.
        cfg = FakeConfig(pid_azm_p_gain=0.04)
        blown = []

        def err(axis):
            if axis == 'azm' and cfg.pid_azm_p_gain > 0.05:
                blown.append(cfg.pid_azm_p_gain)
                return 10.0
            return bowl_rms(cfg, axis, AZM_OPT if axis == 'azm' else ALT_OPT)

        tuner = PIDAutoTuner(cfg)
        drive(tuner, err)

        self.assertTrue(blown, "the up-probe never crossed the cliff -- test is vacuous")
        self.assertLessEqual(cfg.pid_azm_p_gain, 0.05,
                             "tuner finished with a diverging P gain")
        self.assertFalse(tuner.active)


class StopSemanticsTests(unittest.TestCase):

    def _partial_run(self):
        cfg = FakeConfig()
        err = lambda axis: bowl_rms(cfg, axis, AZM_OPT if axis == 'azm' else ALT_OPT)
        tuner = PIDAutoTuner(cfg)
        drive(tuner, err, sim_seconds=120.0)  # a couple of sweeps, not converged
        return cfg, tuner

    def test_stop_keeps_best(self):
        cfg, tuner = self._partial_run()
        tuner.stop()
        self.assertFalse(tuner.active)
        for a, ax in tuner.axes.items():
            for k in PARAM_ORDER:
                self.assertAlmostEqual(
                    getattr(cfg, f'pid_{a}_{k}_gain'),
                    round(min(GAIN_MAX, max(GAIN_MIN, ax.best[k])), 6), places=9)

    def test_stop_revert_restores_initial(self):
        cfg, tuner = self._partial_run()
        tuner.stop(revert=True)
        for a, ax in tuner.axes.items():
            for k in PARAM_ORDER:
                self.assertAlmostEqual(
                    getattr(cfg, f'pid_{a}_{k}_gain'),
                    round(min(GAIN_MAX, max(GAIN_MIN, ax.initial[k])), 6), places=9)

    def test_messages_are_drained(self):
        cfg, tuner = self._partial_run()
        tuner.stop()
        msgs = tuner.take_messages()
        self.assertTrue(any("stopped" in m for m in msgs))
        self.assertEqual(tuner.take_messages(), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
