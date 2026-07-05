#!/usr/bin/env python
"""
SimMount physics-realism tests: finite gotos, EL hard stops, sqrt(dt)-scaled
drift noise, and travel-driven (worm-angle) periodic error.

These behaviors close specific sim-vs-hardware gaps flagged in the 2026-07
project review: instant gotos meant every settle/timeout path was validated
against a mount that cannot mis-settle; time-based PE didn't speed up during
slews; and rate-noise amplitude depended on the caller's polling cadence.

Headless. Run: python test_sim_mount_physics.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest

import numpy as np

from lib.auxstar import Targets
from simulator import SimMount


class _Cfg:
    def __init__(self, **sim):
        self.sim_config = sim


def _step(m, dt):
    """Advance the sim mount by dt simulated seconds (poll-driven physics)."""
    m._last_t -= dt
    m.hc_get_position(Targets.AZM)


class FiniteGotoTests(unittest.TestCase):

    def test_default_goto_is_instant(self):
        m = SimMount(_Cfg())
        m.hc_goto_fast(Targets.AZM, 200.0, 0, 0)
        self.assertAlmostEqual(m.az_true_deg, 200.0, places=6)

    def test_finite_goto_takes_time_and_arrives(self):
        m = SimMount(_Cfg(sim_goto_rate_dps=4.0), az0_deg=0.0)
        m.hc_goto_fast(Targets.AZM, 20.0, 0, 0)
        _step(m, 1.0)
        self.assertAlmostEqual(m.az_true_deg, 4.0, delta=0.2,
                               msg="1 s at 4 deg/s should cover ~4 deg")
        for _ in range(10):
            _step(m, 1.0)
        self.assertAlmostEqual(m.az_true_deg, 20.0, places=3)

    def test_finite_goto_takes_shortest_arc(self):
        m = SimMount(_Cfg(sim_goto_rate_dps=4.0), az0_deg=350.0)
        m.hc_goto_fast(Targets.AZM, 10.0, 0, 0)  # 20 deg across the seam
        _step(m, 1.0)
        # Should be moving UP through 354, not down the 340-deg way.
        self.assertAlmostEqual(m.az_true_deg, 354.0, delta=0.2)
        for _ in range(10):
            _step(m, 1.0)
        self.assertAlmostEqual(m.az_true_deg, 10.0, places=3)

    def test_rate_command_cancels_goto(self):
        m = SimMount(_Cfg(sim_goto_rate_dps=4.0), az0_deg=0.0)
        m.hc_goto_fast(Targets.AZM, 40.0, 0, 0)
        _step(m, 1.0)
        m.hc_slew_fixed(Targets.AZM, 0)  # stop overrides the goto
        pos = m.az_true_deg
        for _ in range(5):
            _step(m, 1.0)
        self.assertAlmostEqual(m.az_true_deg, pos, places=6)


class ElHardStopTests(unittest.TestCase):

    def test_el_stops_clamp_and_stall(self):
        m = SimMount(_Cfg(sim_el_stop_min_deg=0.0, sim_el_stop_max_deg=90.0),
                     el0_deg=85.0)
        m.hc_slew_fixed(Targets.ALT, 9)  # drive up at 10 deg/s
        _step(m, 2.0)
        self.assertAlmostEqual(m.el_true_deg, 90.0, places=6,
                               msg="axis must clamp at the hard stop")
        self.assertEqual(m.el_rate_dps, 0.0, "mount stalls at a hard stop")
        # Further polling does not push through the stop.
        _step(m, 2.0)
        self.assertAlmostEqual(m.el_true_deg, 90.0, places=6)

    def test_no_stops_by_default(self):
        m = SimMount(_Cfg(), el0_deg=85.0)
        m.hc_slew_fixed(Targets.ALT, 9)
        _step(m, 2.0)
        self.assertGreater(m.el_true_deg, 100.0)


class NoiseScalingTests(unittest.TestCase):

    def _walk_std(self, dt, n_steps, total_time, seed):
        """Std of final position after random-walking total_time in dt steps."""
        finals = []
        for i in range(40):
            m = SimMount(_Cfg(mount_rate_noise_dps=0.05),
                         rng=np.random.default_rng(seed + i))
            for _ in range(n_steps):
                _step(m, dt)
            finals.append((m.az_true_deg + 180) % 360 - 180)
        return float(np.std(finals))

    def test_walk_amplitude_independent_of_polling_cadence(self):
        # Same total span (2 s), polled at 10 Hz vs 2 Hz: the random-walk
        # spread must match (the old rate*dt kick made fine polling look
        # 5x quieter than coarse polling).
        std_fine = self._walk_std(dt=0.1, n_steps=20, total_time=2.0, seed=100)
        std_coarse = self._walk_std(dt=0.5, n_steps=4, total_time=2.0, seed=900)
        ratio = std_coarse / std_fine
        self.assertGreater(ratio, 0.5, f"ratio {ratio}")
        self.assertLess(ratio, 2.0, f"ratio {ratio}")


class WormAnglePeTests(unittest.TestCase):

    def test_pe_frozen_when_parked_in_travel_mode(self):
        m = SimMount(_Cfg(mount_pe_amplitude_deg=0.01, mount_pe_period_deg=2.0),
                     az0_deg=100.0)
        _step(m, 0.01)  # seed PE state
        p0 = m.az_true_deg
        for _ in range(20):
            _step(m, 1.0)  # parked: no travel -> worm not turning -> no PE drift
        self.assertAlmostEqual(m.az_true_deg, p0, places=9)

    def test_pe_advances_with_travel(self):
        # Slewing 1 deg through a 2-deg worm period = half a PE cycle: the
        # pointing must deviate from the ideal (slew + nothing) trajectory.
        m = SimMount(_Cfg(mount_pe_amplitude_deg=0.05, mount_pe_period_deg=2.0),
                     az0_deg=0.0)
        _step(m, 0.01)
        m.hc_slew_fixed(Targets.AZM, 6)  # 1 deg/s
        deviations = []
        travelled = 0.0
        for _ in range(10):
            _step(m, 0.1)
            travelled += 0.1  # 1 deg/s * 0.1 s
            deviations.append(abs(m.az_true_deg - travelled))
        m.hc_slew_fixed(Targets.AZM, 0)
        self.assertGreater(max(deviations), 0.01,
                           "PE must ride on the pointing during a slew")

    def test_time_based_pe_still_works(self):
        m = SimMount(_Cfg(mount_pe_amplitude_deg=0.05, mount_pe_period_sec=2.0),
                     az0_deg=100.0)
        _step(m, 0.01)
        p0 = m.az_true_deg
        moved = False
        for _ in range(8):
            m._t0 -= 0.25          # advance wall-clock phase
            _step(m, 0.01)
            if abs(m.az_true_deg - p0) > 1e-4:
                moved = True
        self.assertTrue(moved, "legacy time-based PE must still drift while parked")


if __name__ == '__main__':
    unittest.main(verbosity=2)
