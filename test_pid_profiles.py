#!/usr/bin/env python
"""Unit tests for per-mode PID gain profiles (service_gain_profiles).

PROGRAM (encoder loop) and HOTSPOT (optical loop) are different plants, so
each keeps its own auto-looked-up gain profile in config.pid_mode_profiles;
the six live pid_*_gain fields are "the active set" and the service swaps
them on mode transitions. These tests cover the swap/seed/restore mechanics,
the HANDOFF-shares-PROGRAM mapping, the tune-provenance stamp (target label),
the tuner-stop-on-plant-change interaction, and config persistence.

Run: python test_pid_profiles.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import types
import unittest

import pygame
pygame.init()

from config import ConfigState
from joystick_controller import JoystickModeState, TrackingMode
from autotune import PIDAutoTuner

GAINS = JoystickModeState._GAIN_PROFILE_FIELDS


def make_state():
    cfg = ConfigState()
    cfg.pid_azm_p_gain = 0.014
    cfg.pid_azm_i_gain = 0.00025
    cfg.pid_azm_d_gain = 0.0009
    cfg.pid_alt_p_gain = 0.015
    cfg.pid_alt_i_gain = 0.00028
    cfg.pid_alt_d_gain = 0.0009
    msgs = []
    st = JoystickModeState(None, cfg, msgs.append)
    return st, cfg, msgs


def live(cfg):
    return {f: getattr(cfg, f) for f in GAINS}


def set_live(cfg, value_map):
    for f, v in value_map.items():
        setattr(cfg, f, v)


class ProfileSwapTests(unittest.TestCase):

    def test_seed_swap_and_automatic_restore(self):
        st, cfg, _ = make_state()

        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        program_tune = dict(live(cfg))
        self.assertIn("PROGRAM", cfg.pid_mode_profiles)

        # "Tune" PROGRAM, then hand off to HOTSPOT: PROGRAM's gains must be
        # saved, and HOTSPOT seeds from them (first entry).
        cfg.pid_azm_p_gain = 0.111
        program_tune = dict(live(cfg))
        st.tracking_mode = TrackingMode.HOTSPOT
        st.service_gain_profiles()
        self.assertEqual(cfg.pid_mode_profiles["PROGRAM"]["gains"], program_tune)
        self.assertEqual(live(cfg), program_tune)  # seeded, not zeroed

        # "Tune" HOTSPOT differently, drop back to PROGRAM: PROGRAM's tune
        # comes back automatically.
        cfg.pid_azm_p_gain = 0.033
        cfg.pid_alt_d_gain = 0.002
        hotspot_tune = dict(live(cfg))
        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        self.assertEqual(live(cfg), program_tune)
        self.assertEqual(cfg.pid_mode_profiles["HOTSPOT"]["gains"], hotspot_tune)

        # And HOTSPOT's tune comes back when HOTSPOT re-engages.
        st.tracking_mode = TrackingMode.HOTSPOT
        st.service_gain_profiles()
        self.assertEqual(live(cfg), hotspot_tune)

    def test_handoff_shares_program_profile(self):
        st, cfg, _ = make_state()
        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        before = live(cfg)

        st.tracking_mode = TrackingMode.HANDOFF
        st.service_gain_profiles()
        self.assertEqual(live(cfg), before, "HANDOFF must not swap gains")
        self.assertEqual(st._active_gain_profile, "PROGRAM")

        st.tracking_mode = TrackingMode.HOTSPOT
        st.service_gain_profiles()
        self.assertEqual(st._active_gain_profile, "HOTSPOT")

    def test_non_pid_modes_do_not_swap(self):
        st, cfg, _ = make_state()
        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        cfg.pid_azm_p_gain = 0.5
        for mode in (TrackingMode.STANDBY, TrackingMode.RATE_CONTROL):
            st.tracking_mode = mode
            st.service_gain_profiles()
            self.assertEqual(cfg.pid_azm_p_gain, 0.5)
            self.assertEqual(st._active_gain_profile, "PROGRAM")


class TunerInteractionTests(unittest.TestCase):

    def _measured_tuner(self, st, cfg):
        """Arm a tuner in PROGRAM and drive it through its baseline window on
        a simulated clock so it holds a real measurement."""
        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        st.autotuner = tuner = PIDAutoTuner(cfg, mode=TrackingMode.PROGRAM)
        tuner.target_label = "satellite TESTSAT-42"
        t = 1000.0
        tuner.start(now=t)
        for _ in range(80):  # > settle (1.5 s) + eval (4 s) at 10 Hz
            t += 0.1
            tuner.update(t, True, 0.01, 0.02)
        self.assertIsNotNone(tuner.axes['azm'].best_cost,
                             "baseline window never completed")
        return tuner

    def test_plant_change_stops_tune_and_stamps_target(self):
        st, cfg, msgs = make_state()
        tuner = self._measured_tuner(st, cfg)

        st.tracking_mode = TrackingMode.HOTSPOT
        st.service_gain_profiles()

        self.assertFalse(tuner.active, "plant change must stop the tune")
        prof = cfg.pid_mode_profiles["PROGRAM"]
        self.assertEqual(prof["tuned_on"], "satellite TESTSAT-42")
        self.assertIn("tuned_at", prof)
        self.assertIn("tuned_rms", prof)
        # Saved gains are the tuner's best (stop() applied them to the live
        # fields before the profile save ran).
        for f in GAINS:
            ax = tuner.axes['azm' if 'azm' in f else 'alt']
            param = f.split('_')[2]  # pid_<axis>_<p|i|d>_gain
            self.assertAlmostEqual(prof["gains"][f],
                                   round(min(2.0, max(2e-5, ax.best[param])), 6),
                                   places=9)

    def test_done_tune_is_stamped_via_service_autotune(self):
        st, cfg, _ = make_state()
        tuner = self._measured_tuner(st, cfg)
        # Force convergence and let service_autotune notice the 'done' phase.
        for ax in tuner.axes.values():
            for k in ax.steps:
                ax.steps[k] = 0.0
        t = 2000.0
        while tuner.active and t < 2600.0:
            t += 0.1
            tuner.update(t, True, 0.01, 0.02)
        self.assertEqual(tuner.phase, 'done')
        st.telescope_connected = True
        st.stopped = False
        st.service_autotune()
        self.assertEqual(cfg.pid_mode_profiles["PROGRAM"]["tuned_on"],
                         "satellite TESTSAT-42")


class PersistenceTests(unittest.TestCase):

    def test_profiles_survive_config_round_trip(self):
        st, cfg, _ = make_state()
        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        cfg.pid_azm_p_gain = 0.077
        st.tracking_mode = TrackingMode.HOTSPOT
        st.service_gain_profiles()
        cfg.pid_mode_profiles["PROGRAM"]["tuned_on"] = "aircraft A1B2C3"

        blob = json.dumps(cfg.get_config_dict())
        cfg2 = ConfigState()
        cfg2.load_from_dict(json.loads(blob))
        self.assertEqual(cfg2.pid_mode_profiles["PROGRAM"]["gains"]["pid_azm_p_gain"], 0.077)
        self.assertEqual(cfg2.pid_mode_profiles["PROGRAM"]["tuned_on"], "aircraft A1B2C3")
        self.assertIn("HOTSPOT", cfg2.pid_mode_profiles)

    def test_load_message_names_tune_target(self):
        st, cfg, msgs = make_state()
        cfg.pid_mode_profiles = {
            "HOTSPOT": {"gains": {f: 0.02 for f in GAINS},
                        "tuned_on": "launch Starship"},
        }
        st.tracking_mode = TrackingMode.PROGRAM
        st.service_gain_profiles()
        st.tracking_mode = TrackingMode.HOTSPOT
        st.service_gain_profiles()
        self.assertEqual(cfg.pid_azm_p_gain, 0.02)
        self.assertTrue(any("tuned on launch Starship" in m for m in msgs),
                        f"no provenance in status messages: {msgs}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
