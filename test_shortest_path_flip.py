#!/usr/bin/env python
"""
Shortest-spherical-path regression tests (the west-to-east "long way around"
bug).

An alt-az style mount reaches every sky pointing in TWO axis configurations:
canonical, and over-the-zenith (the mirrored sky representation
(az+180, 180-el)). The per-axis shortest-arc wrap -- the earlier '360 lap'
fix, still covered by test_wrap_shortest_path.py -- only ever considers the
canonical solution, so with the mount pointed west and the target in the
east the loop drove the azimuth axis ~180 deg the long way around instead of
~90 deg of ALT motion straight over the zenith. choose_mount_target now
picks between both solutions each cycle; these tests pin that choice, its
limit awareness, its hysteresis, and the sign consequences all the way down
to the commanded rates.

Headless. Run: python test_shortest_path_flip.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest

import pygame
pygame.init()

from control import (FLIP_HYSTERESIS_DEG, choose_mount_target,
                     compute_mount_position_error, mount_target_for)


class _Cfg:
    mount_mode = 'AltAz'
    alignment_azimuth_str = '0.0'
    alignment_elevation_str = '0.0'
    pointing_model_enabled = False
    eq_pointing_model_enabled = False
    lat_str = '35.0'


class _PassCfg(_Cfg):
    mount_mode = 'Passthrough'


class ChooseMountTargetTests(unittest.TestCase):

    def test_west_to_east_goes_over_the_zenith(self):
        # Mount west (AZM=270, ALT=45 i.e. el=45), target east (az=90, el=45).
        # Canonical demands a 180-deg azimuth slew; the flipped solution is
        # (AZM=270, ALT=-45): pure ALT motion through the zenith.
        azm, alt, flipped = choose_mount_target(_Cfg(), 270.0, 45.0, 90.0, 45.0)
        self.assertTrue(flipped)
        self.assertAlmostEqual(azm, 270.0, places=6)
        self.assertAlmostEqual(alt, -45.0, places=6)

    def test_errors_take_the_zenith_path(self):
        az_err, el_err = compute_mount_position_error(_Cfg(), 270.0, 45.0, 90.0, 45.0)
        self.assertAlmostEqual(az_err, 0.0, places=6)
        self.assertAlmostEqual(el_err, -90.0, places=6,
                               msg="ALT should drive -90 deg through the zenith, "
                                   "not spin azimuth 180 deg")

    def test_nearby_target_stays_canonical(self):
        azm, alt, flipped = choose_mount_target(_Cfg(), 270.0, 45.0, 280.0, 40.0)
        self.assertFalse(flipped)
        self.assertAlmostEqual(azm, 280.0, places=6)
        self.assertAlmostEqual(alt, 50.0, places=6)

    def test_passthrough_flip_convention(self):
        # Passthrough: mount alt == sky el, so the flipped solution is
        # (az+180, 180-el).
        azm, alt, flipped = choose_mount_target(_PassCfg(), 270.0, 45.0, 90.0, 45.0)
        self.assertTrue(flipped)
        self.assertAlmostEqual(azm, 270.0, places=6)
        self.assertAlmostEqual(alt, 135.0, places=6)

    def test_limits_veto_the_flip(self):
        # ALT limits [0, 90] exclude the flipped solution (ALT=-45): fall back
        # to the (legal) canonical 180-deg azimuth path rather than aborting.
        azm, alt, flipped = choose_mount_target(
            _Cfg(), 270.0, 45.0, 90.0, 45.0, limits=(0.0, 360.0, 0.0, 90.0))
        self.assertFalse(flipped)
        self.assertAlmostEqual(azm, 90.0, places=6)
        self.assertAlmostEqual(alt, 45.0, places=6)

    def test_flip_chosen_when_only_it_is_legal(self):
        # Canonical (ALT=45) outside limits, flipped (ALT=-45) inside.
        azm, alt, flipped = choose_mount_target(
            _Cfg(), 270.0, 45.0, 90.0, 45.0, limits=(0.0, 360.0, -90.0, 0.0))
        self.assertTrue(flipped)
        self.assertAlmostEqual(alt, -45.0, places=6)

    def test_both_illegal_returns_canonical_for_the_gate(self):
        azm, alt, flipped = choose_mount_target(
            _Cfg(), 270.0, 45.0, 90.0, 45.0, limits=(0.0, 360.0, 50.0, 60.0))
        self.assertFalse(flipped)
        self.assertAlmostEqual(alt, 45.0, places=6)  # gate will abort on this

    def test_hysteresis_prefers_canonical_on_near_ties(self):
        # Canonical metric 90.3 vs flip metric 90.0: within the hysteresis
        # band, so no flip (a flapping choice mid-slew is worse than 0.3 deg).
        self.assertLess(0.3, FLIP_HYSTERESIS_DEG)
        azm, alt, flipped = choose_mount_target(_Cfg(), 0.0, 45.0, 90.3, 45.0)
        self.assertFalse(flipped)

    def test_eq_mode_never_flips(self):
        class EqCfg(_Cfg):
            mount_mode = 'Eq'
            alignment_elevation_str = '35.0'
        azm, alt, flipped = choose_mount_target(EqCfg(), 270.0, 45.0, 90.0, 45.0)
        self.assertFalse(flipped)

    def test_mount_target_for_matches_choice(self):
        # The gate chooses once; the error path re-derives the SAME solution
        # from the flag (post lead/bias). The two must agree.
        for cfg, cur in ((_Cfg(), (270.0, 45.0)), (_PassCfg(), (270.0, 45.0))):
            azm, alt, flipped = choose_mount_target(cfg, cur[0], cur[1], 90.0, 45.0)
            azm2, alt2 = mount_target_for(cfg, 90.0, 45.0, flipped)
            self.assertAlmostEqual(azm, azm2, places=9)
            self.assertAlmostEqual(alt, alt2, places=9)


class ProgramTrackDirectionTests(unittest.TestCase):
    """End-to-end: program_track pointed west with the target east must
    command ALT motion (over the top), not a 180-deg azimuth slew."""

    def test_commands_alt_not_azimuth(self):
        import joystick_controller as jc
        from joystick_controller import JoystickModeState, TrackingMode

        class Cfg(_Cfg):
            pid_azm_p_gain, pid_azm_i_gain, pid_azm_d_gain = 0.025, 0.0, 0.0
            pid_alt_p_gain, pid_alt_i_gain, pid_alt_d_gain = 0.025, 0.0, 0.0
            pid_lead_time_sec = 0.0
            continuous_rate_tracking = False
            azm_limit_min_str = "-100000"
            azm_limit_max_str = "100000"
            alt_limit_min_str = "-100000"
            alt_limit_max_str = "100000"

        class _RecController:
            def __init__(self):
                self.rates = {}

            def hc_slew_fixed(self, target, rate):
                self.rates[target] = rate
                return True

        class _Vis:
            selected_launch = None
            launch_launched = False
            selected_aircraft = None
            aircraft_trajectories = {}
            current_tt = 0.0
            selected_satellite = "EASTSAT"
            satellite_trajectories = {"EASTSAT": ([(0,)] * 8, [0.0])}

        state = JoystickModeState(None, Cfg(), lambda m: None)
        state.telescope_connected = True
        ctrl = _RecController()
        state.telescope_controller = ctrl
        state.tracking_vis_state = _Vis()
        state.tracking_mode = TrackingMode.PROGRAM
        state.current_azm = state.current_azm_raw = 270.0
        state.current_alt = state.current_alt_raw = 45.0

        orig = jc.interpolate_position_data_and_rates
        jc.interpolate_position_data_and_rates = (
            lambda traj, tt, *a, **k: (1.0, 1.0, 45.0, 500.0, 90.0, 0.0, 0.0))
        try:
            state.program_track()
        finally:
            jc.interpolate_position_data_and_rates = orig

        self.assertAlmostEqual(state.azm_position_error, 0.0, places=6)
        self.assertAlmostEqual(state.alt_position_error, -90.0, places=6)
        self.assertEqual(ctrl.rates.get(jc.Targets.AZM, 0), 0,
                         "azimuth must not slew the long way around")
        self.assertLess(ctrl.rates.get(jc.Targets.ALT, 0), 0,
                        "ALT must drive negative, over the zenith")


if __name__ == "__main__":
    unittest.main(verbosity=2)
