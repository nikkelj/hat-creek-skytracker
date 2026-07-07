#!/usr/bin/env python
"""
Closed-loop A/B parity harness: the REAL Python PROGRAM-track path vs the
Rust core loop on the same scenario (the promotion gate promised in
rust/STEP4_DESIGN.md).

Module-level parity (PID step-for-step, transforms to 1e-9, byte-identical
wire output) is covered by test_rust_*_parity.py. What was never tested is
the LOOP level: does the composed Rust decision cycle behave like the real
`JoystickModeState.program_track` when both close the loop against an ideal
mount? These tests drive:

  * Python: the real program_track (unmodified), 10 Hz wall-clock cadence
    against an ideal SimMount (its PID takes dt from the wall clock, so the
    Python side cannot be stepped on a fake clock without bypassing the code
    under test);
  * Rust: SimCoreLoop stepped deterministically at the same dt.

and compares closed-loop BEHAVIOR: convergence, approach direction,
steady-state error, and moving-setpoint lag. Exact per-cycle rate-command
traces are not compared -- Python's wall-clock dt makes them jittery by
construction; behavioral envelopes are the honest comparison.

Skipped (via conftest) when the skytracker_core wheel isn't built.
Run: python test_loop_ab_parity.py   (takes ~15 s: real-time Python cycles)
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest

import pygame
pygame.init()

import skytracker_core as rc

import joystick_controller as jc
from joystick_controller import JoystickModeState, TrackingMode
from simulator import SimMount

GAINS = (0.025, 0.0, 0.0)
DT = 0.1          # 10 Hz
CYCLES = 70
START = (10.0, 20.0)   # mount az/el at t0
TARGET = (50.0, 45.0)  # sky setpoint (Passthrough: mount == sky)


class _Cfg:
    mount_mode = "Passthrough"
    pid_azm_p_gain, pid_azm_i_gain, pid_azm_d_gain = GAINS
    pid_alt_p_gain, pid_alt_i_gain, pid_alt_d_gain = GAINS
    pid_lead_time_sec = 0.0
    pointing_model_enabled = False
    continuous_rate_tracking = False
    azm_limit_min_str = "-100000"
    azm_limit_max_str = "100000"
    alt_limit_min_str = "-100000"
    alt_limit_max_str = "100000"
    alignment_azimuth_str = "0.0"
    alignment_elevation_str = "0.0"
    sim_config = {}


class _Vis:
    """Minimal tracking-vis: one selected satellite with a placeholder
    trajectory; the interpolator is monkeypatched to emit the scenario."""
    selected_launch = None
    launch_launched = False
    selected_aircraft = None
    aircraft_trajectories = {}
    current_tt = 0.0

    def __init__(self):
        self.selected_satellite = "ABSAT"
        self.satellite_trajectories = {"ABSAT": ([(0,)] * 8, [0.0])}


def _run_python_loop(setpoint_fn, cycles=CYCLES, dt=DT):
    """Drive the REAL program_track against an ideal SimMount in real time.
    Returns the trace of offset-free mount (az, el) after each cycle."""
    cfg = _Cfg()
    state = JoystickModeState(None, cfg, lambda m: None)
    state.telescope_connected = True
    mount = SimMount(cfg, az0_deg=START[0], el0_deg=START[1])
    state.telescope_controller = mount
    state.tracking_vis_state = _Vis()
    state.tracking_mode = TrackingMode.PROGRAM

    orig = jc.interpolate_position_data_and_rates
    trace = []
    try:
        for i in range(cycles):
            az, el, az_rate, el_rate = setpoint_fn(i * dt)
            jc.interpolate_position_data_and_rates = (
                lambda traj, tt, *a, **k: (1.0, 1.0, el, 500.0, az, az_rate, el_rate))
            # Mimic MountControlThread._poll_position: one poll per cycle.
            state.current_azm = mount.hc_get_position(jc.Targets.AZM) * 360.0
            state.current_alt = mount.hc_get_position(jc.Targets.ALT) * 360.0
            state.current_azm_raw = state.current_azm
            state.current_alt_raw = state.current_alt
            state.program_track()
            time.sleep(dt)  # the Python PID takes dt from the wall clock
            trace.append((mount.az_true_deg, mount.el_true_deg))
    finally:
        jc.interpolate_position_data_and_rates = orig
    return trace


def _run_rust_loop(setpoint_fn, cycles=CYCLES, dt=DT):
    """Drive the Rust loop deterministically on the same scenario."""
    loop = rc.SimCoreLoop(az0_deg=START[0], el0_deg=START[1])
    loop.set_mode("program")
    loop.set_mount_mode("passthrough")
    loop.set_gains(*GAINS, *GAINS)
    trace = []
    for i in range(cycles):
        az, el, az_rate, el_rate = setpoint_fn(i * dt)
        loop.set_setpoint(az, el, az_rate, el_rate)
        loop.step(dt)
        trace.append((loop.az_true_deg, loop.el_true_deg))
    return trace


def _first_within(trace, target, tol):
    for i, (az, el) in enumerate(trace):
        if abs(az - target[0]) < tol and abs(el - target[1]) < tol:
            return i
    return None


class AbLoopParityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixed = lambda t: (TARGET[0], TARGET[1], 0.0, 0.0)
        cls.py_fixed = _run_python_loop(fixed)
        cls.rs_fixed = _run_rust_loop(fixed)

    def test_both_converge_to_the_target(self):
        for name, trace in (("python", self.py_fixed), ("rust", self.rs_fixed)):
            az, el = trace[-1]
            self.assertLess(abs(az - TARGET[0]), 1.0, f"{name} az={az}")
            self.assertLess(abs(el - TARGET[1]), 1.0, f"{name} el={el}")

    def test_acquisition_time_comparable(self):
        py_t = _first_within(self.py_fixed, TARGET, tol=2.0)
        rs_t = _first_within(self.rs_fixed, TARGET, tol=2.0)
        self.assertIsNotNone(py_t, "python never acquired")
        self.assertIsNotNone(rs_t, "rust never acquired")
        ratio = (py_t + 1) / (rs_t + 1)
        self.assertTrue(0.4 < ratio < 2.5,
                        f"acquisition times diverge: py={py_t} rs={rs_t} cycles")

    def test_approach_is_monotonic_no_overshoot_blowup(self):
        # Neither loop may fly past the target by more than a few degrees --
        # the wrap/limit/sign bugs this harness exists to catch all manifest
        # as gross overshoot or motion in the wrong direction.
        for name, trace in (("python", self.py_fixed), ("rust", self.rs_fixed)):
            azs = [az for az, _ in trace]
            self.assertLess(max(azs), TARGET[0] + 4.0, f"{name} overshoot: {max(azs)}")
            self.assertGreaterEqual(min(azs), START[0] - 1.0,
                                    f"{name} moved away from target first")

    def test_moving_setpoint_lag_comparable(self):
        ramp = lambda t: (TARGET[0] + 0.5 * t, TARGET[1], 0.5, 0.0)
        py = _run_python_loop(ramp, cycles=60)
        rs = _run_rust_loop(ramp, cycles=60)
        t_end = 59 * DT
        target_az = TARGET[0] + 0.5 * t_end
        py_lag = abs(py[-1][0] - target_az)
        rs_lag = abs(rs[-1][0] - target_az)
        self.assertLess(py_lag, 4.0, f"python lag {py_lag}")
        self.assertLess(rs_lag, 4.0, f"rust lag {rs_lag}")
        self.assertLess(abs(py_lag - rs_lag), 3.0,
                        f"lags diverge: py={py_lag:.2f} rs={rs_lag:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
