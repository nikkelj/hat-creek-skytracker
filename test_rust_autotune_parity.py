#!/usr/bin/env python
"""Phase 6b A/B parity: Rust PIDAutoTuner vs the Python tuner on identical
deterministic plants and clocks.

Both tuners drive a synthetic plant whose per-cycle axis error depends on
the gains THEY applied (a smooth cost bowl in log-gain space + seeded
noise + an occasional divergence spike and tracking dropout), sampled at
15 Hz. Since each tuner only sees the consequences of its own decisions,
any divergence in decision logic cascades and is caught by the per-cycle
comparison of phase, stage, applied gains, and messages.
"""

import math
import os
import types
import unittest

import numpy as np

try:
    import skytracker_core
    _HAVE = hasattr(skytracker_core, "RustPIDAutoTuner")
except ImportError:
    _HAVE = False

from autotune import PIDAutoTuner, RustPIDAutoTuner

FIELDS = ('pid_azm_p_gain', 'pid_azm_d_gain', 'pid_azm_i_gain',
          'pid_alt_p_gain', 'pid_alt_d_gain', 'pid_alt_i_gain')


def _cfg():
    return types.SimpleNamespace(
        pid_azm_p_gain=0.0023, pid_azm_d_gain=0.00027, pid_azm_i_gain=0.00025,
        pid_alt_p_gain=0.0027, pid_alt_d_gain=0.00027, pid_alt_i_gain=0.00028)


def _plant_error(cfg, axis, rng_val, t):
    """Axis error (deg): cost bowl centred on ideal gains + noise."""
    p = getattr(cfg, f'pid_{axis}_p_gain')
    d = getattr(cfg, f'pid_{axis}_d_gain')
    i = getattr(cfg, f'pid_{axis}_i_gain')
    ideal = (0.006, 0.0004, 0.0001) if axis == 'azm' else (0.004, 0.0006, 0.0002)
    cost = 0.0
    for g, gi in zip((p, d, i), ideal):
        cost += (math.log10(max(g, 1e-9)) - math.log10(gi)) ** 2
    rms = 0.01 * (1.0 + cost)
    return rms * (math.sin(2.1 * t + 0.3) + 0.5 * rng_val)


@unittest.skipUnless(_HAVE, "skytracker_core wheel lacks RustPIDAutoTuner")
class AutotuneParity(unittest.TestCase):
    def _run(self, seed, cycles, dropout_at=None, spike_at=None):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, size=(cycles, 2))
        cfg_py, cfg_rs = _cfg(), _cfg()
        py = PIDAutoTuner(cfg_py, mode="PROGRAM")
        rs = RustPIDAutoTuner(cfg_rs, mode="PROGRAM")
        py.start(now=0.0)
        rs.start(now=0.0)
        n_compared = 0
        for k in range(cycles):
            t = k / 15.0
            tracking = not (dropout_at and dropout_at[0] <= t < dropout_at[1])
            errs = {}
            for tuner_cfg, key in ((cfg_py, 'py'), (cfg_rs, 'rs')):
                ea = _plant_error(tuner_cfg, 'azm', noise[k, 0], t)
                el = _plant_error(tuner_cfg, 'alt', noise[k, 1], t)
                if spike_at and abs(t - spike_at) < 0.05:
                    ea, el = 9.0, -9.0
                errs[key] = (ea, el)
            py.update(t, tracking, *errs['py'])
            rs.update(t, tracking, *errs['rs'])
            # Per-cycle parity.
            self.assertEqual(py.phase, rs.phase, f"cycle {k} phase")
            self.assertEqual(py.active, rs.active, f"cycle {k} active")
            for f in FIELDS:
                self.assertAlmostEqual(getattr(cfg_py, f), getattr(cfg_rs, f), places=9,
                                       msg=f"cycle {k} {f}")
            pm, rm = py.take_messages(), rs.take_messages()
            self.assertEqual(pm, rm, f"cycle {k} messages")
            n_compared += 1
            if not py.active and not rs.active:
                break
        self.assertEqual(py.summary(), rs.summary())
        return n_compared, py, rs

    def test_convergence_run(self):
        n, py, rs = self._run(seed=1, cycles=15 * 60 * 12)
        print(f"\n[parity] autotune converge: {n} cycles identical; final phase {rs.phase}, "
              f"sweeps {rs.sweep}, {rs.summary()}")

    def test_dropout_and_divergence(self):
        n, py, rs = self._run(seed=2, cycles=15 * 60 * 4, dropout_at=(20.0, 26.0), spike_at=40.0)
        print(f"[parity] autotune w/ pause+spike: {n} cycles identical, phase {rs.phase}")

    def test_stop_revert_parity(self):
        cfg_py, cfg_rs = _cfg(), _cfg()
        py = PIDAutoTuner(cfg_py, mode="PROGRAM")
        rs = RustPIDAutoTuner(cfg_rs, mode="PROGRAM")
        py.start(now=0.0)
        rs.start(now=0.0)
        for k in range(15 * 30):
            t = k / 15.0
            py.update(t, True, 0.01 * math.sin(t), 0.01 * math.cos(t))
            rs.update(t, True, 0.01 * math.sin(t), 0.01 * math.cos(t))
        py.stop(revert=True)
        rs.stop(revert=True)
        for f in FIELDS:
            self.assertAlmostEqual(getattr(cfg_py, f), getattr(cfg_rs, f), places=9)
            self.assertAlmostEqual(getattr(cfg_rs, f), getattr(_cfg(), f), places=9,
                                   msg="revert must restore initial gains")
        self.assertEqual(py.phase, rs.phase)
        print("[parity] autotune stop(revert): identical, initial gains restored")


if __name__ == "__main__":
    unittest.main(verbosity=2)
