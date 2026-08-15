#!/usr/bin/env python
"""Regression tests for the 2026-07 project-wide timing audit fixes.

Each test pins one fix:
  * PID dt clamp: a long idle gap cannot wind the integrator in one cycle,
    and get_current_rates honors an explicitly passed dt (it used to shadow
    and discard the parameter).
  * Azimuth unwrap: rates and interpolation across the 0/360 seam (a due-north
    crossing used to inject a ~360 deg/dt feed-forward spike and swing the
    interpolated azimuth to the antipode).
  * live_tt(): the control loops' live time base is a real TT Julian date.
  * Out-of-window scrub hides satellites instead of freezing them at the
    window edge.
  * Camera raw-frame metadata: the seq/payload/seq tear-free read protocol,
    and the stop_capture tail-drop only removing explicit seq-0 raced frames.
  * Pass-table time math carries raw TT (no local HH:MM string round-trip).

Headless. Run: python -m pytest test_timing_fixes.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import unittest

import numpy as np

from control import PIDController, DT_MAX_SECONDS, DT_MIN_SECONDS


class PidDtClampTests(unittest.TestCase):
    def test_compute_pid_output_clamps_large_dt(self):
        # 60 s idle gap with 1 deg of error: the integrator step must be
        # bounded by DT_MAX_SECONDS, not pinned at the clip in one cycle.
        pid = PIDController(p_gain=0.0, i_gain=1.0, d_gain=0.0, axis_name="ALT")
        pid.compute_pid_output(1.0, 60.0)
        self.assertLessEqual(abs(pid.integral_error), DT_MAX_SECONDS + 1e-9)

    def test_get_current_rates_honors_explicit_dt(self):
        # The old implementation shadowed dt_seconds and recomputed it from
        # wall time, silently discarding every caller's value. i_gain is kept
        # small so the output stays under the rate ceiling (the conditional
        # anti-windup would otherwise roll the integrator back).
        pid = PIDController(p_gain=0.0, i_gain=0.001, d_gain=0.0, axis_name="ALT")
        pid.get_current_rates(1.0, dt_seconds=0.25)
        self.assertAlmostEqual(pid.integral_error, 0.25, places=9)

    def test_get_current_rates_self_times_on_monotonic_clock(self):
        # The controller self-times on time.perf_counter() (monotonic and
        # high-resolution -- Windows time.monotonic() has a 15.6 ms quantum
        # on CPython<=3.12).
        pid = PIDController(p_gain=0.0, i_gain=0.001, d_gain=0.0, axis_name="ALT")
        pid.get_current_rates(1.0)  # first call: default dt 0.1
        self.assertAlmostEqual(pid.integral_error, 0.1, places=9)
        # Simulate a 60 s standby on the SAME clock the controller uses.
        pid.last_update_time = time.perf_counter() - 60.0
        pid.get_current_rates(1.0)
        self.assertLessEqual(abs(pid.integral_error), 0.1 + DT_MAX_SECONDS + 1e-9)

    def test_dt_floor(self):
        # A duplicate call in the same cycle must not divide the derivative
        # by a near-zero dt: measured dt is floored at DT_MIN_SECONDS.
        pid = PIDController(p_gain=0.0, i_gain=0.0, d_gain=1.0, axis_name="ALT")
        pid.get_current_rates(0.0, measurement_degrees=0.0)
        out, _ = pid.get_current_rates(0.0, dt_seconds=1e-9,
                                       measurement_degrees=0.001)
        # d = -d_gain * delta / dt; with the floor, bounded by 0.001/DT_MIN.
        self.assertLessEqual(abs(out), 0.001 / DT_MIN_SECONDS + 1e-6)


class AzimuthUnwrapTests(unittest.TestCase):
    def _rows(self, az0, az1):
        # 8-column rows: [time, alt, az, dist, px, py, az_rate, el_rate]
        return ([[0.0, 45.0, az0, 500.0, 100.0, 100.0, 0.0, 0.0],
                 [1.0 / 86400.0, 45.0, az1, 500.0, 110.0, 110.0, 0.0, 0.0]],
                np.array([0.0, 1.0 / 86400.0]))

    def test_interpolation_across_north(self):
        from trajectory import interpolate_position_data_and_rates
        traj = self._rows(359.0, 1.0)
        _px, _py, _alt, _dist, az, _ar, _er = interpolate_position_data_and_rates(
            traj, 0.5 / 86400.0)
        # Midpoint of 359 -> 1 through north is 0, NOT 180.
        self.assertLess(min(az, 360.0 - az), 1.5,
                        f"interpolated az {az} swung to the antipode")

    def test_unwrap_helper(self):
        from trajectory import _unwrap_az_diff
        self.assertAlmostEqual(_unwrap_az_diff(1.0 - 359.0), 2.0)
        self.assertAlmostEqual(_unwrap_az_diff(359.0 - 1.0), -2.0)
        self.assertAlmostEqual(_unwrap_az_diff(10.0), 10.0)


class LiveTtTests(unittest.TestCase):
    def test_live_tt_is_a_real_julian_date(self):
        from trajectory import live_tt
        tt = live_tt()
        self.assertGreater(tt, 2400000.0)
        self.assertLess(tt, 2600000.0)


class OutOfWindowScrubTests(unittest.TestCase):
    def test_positions_hidden_outside_window(self):
        from trajectory import update_satellite_positions
        from skyfield.api import load

        ts = load.timescale()
        t0 = ts.tt_jd(2460000.0)
        t1 = ts.tt_jd(2460000.0 + 30.0 / 1440.0)  # 30 min window

        class _State:
            pass

        state = _State()
        state.t0, state.t1 = t0, t1
        state.satellite_positions = {"sentinel": (0, 0, 0, 0)}
        state.satellite_trajectories = {}
        state.filter_text = ""
        state.filter_above_alt_text = ""
        state.filter_below_alt_text = ""
        state.satellite_mean_altitudes = {}
        state.position_stack = None

        # Scrub 2 h past the window: everything hides (no frozen edge ghosts).
        update_satellite_positions(state, t1.tt + 2.0 / 24.0)
        self.assertEqual(state.satellite_positions, {})


class RawFrameMetaTests(unittest.TestCase):
    def _thread(self):
        from camera_buffer import CameraThread
        t = CameraThread.__new__(CameraThread)  # no hardware init
        t.latest_raw = None
        t.latest_raw_seq = 0
        t.latest_raw_time = None
        return t

    def test_meta_triple_consistent(self):
        t = self._thread()
        frame = np.zeros((4, 4), dtype=np.uint8)
        # Producer order: payload first, seq last.
        t.latest_raw = frame
        t.latest_raw_time = 123.5
        t.latest_raw_seq = 7
        raw, seq, stamp = t.get_latest_raw_with_meta()
        self.assertIs(raw, frame)
        self.assertEqual(seq, 7)
        self.assertEqual(stamp, 123.5)

    def test_meta_none_before_first_frame(self):
        raw, seq, stamp = self._thread().get_latest_raw_with_meta()
        self.assertIsNone(raw)
        self.assertEqual(seq, 0)
        self.assertIsNone(stamp)


class StopCaptureTailDropTests(unittest.TestCase):
    def _thread(self):
        from camera_buffer import CameraThread, CircularBuffer
        t = CameraThread.__new__(CameraThread)
        t.camera_index = 0
        t.circular_buffer = CircularBuffer(10)
        t.capture_active = False
        t.capture_start_idx = -1
        t.capture_start_time = None
        return t

    def test_raced_seq0_tail_frame_dropped(self):
        t = self._thread()
        t.start_capture()
        for i in range(3):
            t.circular_buffer.append({'i': i, 'sequence_in_capture': i + 1})
            t.capture_frame_count += 1
        # A frame the capture thread appended after the stop flag flipped:
        # explicitly stamped sequence_in_capture 0.
        t.circular_buffer.append({'i': 99, 'sequence_in_capture': 0})
        info, frames = t.stop_capture()
        self.assertEqual([f['i'] for f in frames], [0, 1, 2])
        self.assertEqual(info['end_idx'], 2)

    def test_frames_without_seq_key_are_kept(self):
        # Synthetic producers (tests, tools) may omit the key entirely; only
        # an EXPLICIT seq 0 marks a raced non-capture frame.
        t = self._thread()
        t.start_capture()
        for i in range(3):
            t.circular_buffer.append({'i': i})
            t.capture_frame_count += 1
        _info, frames = t.stop_capture()
        self.assertEqual([f['i'] for f in frames], [0, 1, 2])


class PassTableTimeTests(unittest.TestCase):
    def test_pass_data_carries_raw_tt(self):
        from trajectory import extract_pass_data_from_trajectory
        from skyfield.api import load

        ts = load.timescale()
        base = 2460000.0
        rows = []
        times = []
        for k in range(5):
            tt = base + k * 30.0 / 86400.0
            alt = 20.0 + k  # rising pass, all above mask
            dist = 1000.0 - 50.0 * k  # closest at the last sample
            rows.append([tt, alt, 100.0, dist, 0.0, 0.0, 0.0, 0.0])
            times.append(tt)

        class _Sat:
            name = "TESTSAT"

            class model:
                satnum_str = "99999"

        data = extract_pass_data_from_trajectory(
            (rows, np.array(times)), _Sat(), {}, elevation_mask_deg=10.0, ts=ts)
        self.assertIsNotNone(data)
        # Raw TT of the min-distance sample, exact -- no HH:MM round trip.
        self.assertAlmostEqual(data['closest_time_tt'], times[-1], places=9)
        self.assertNotEqual(data['closest_approach_time'], '--:--')


if __name__ == "__main__":
    unittest.main(verbosity=2)
