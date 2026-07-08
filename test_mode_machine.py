#!/usr/bin/env python
"""
Live-fidelity mode-machine integration tests: PROGRAM -> HANDOFF -> HOTSPOT
against the hardware simulator, driven through the REAL tracking_control
dispatch at control-loop cadence with camera frames arriving at a realistic
frame rate (so the stale-frame gate is exercised, unlike the older
convergence test whose fake camera rendered fresh on every read).

Field symptoms this suite exists to reproduce/pin:
  * HANDOFF centering briefly then walking off in FOV-sized stair-steps,
  * HOTSPOT failing to acquire and reverting to PROGRAM immediately.

Wall-clock based (the Python PID takes dt from time.time()), so each
scenario runs for a few real seconds.

Headless. Run: python test_mode_machine.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
import time
import unittest

import pygame
pygame.init()

from config import ConfigState
from lib.auxstar import Targets
import simulator
from simulator import HardwareSimulator

import joystick_controller as jc
from joystick_controller import JoystickModeState, TrackingMode


class _FakeJoystick:
    def get_numaxes(self):
        return 6

    def get_axis(self, i):
        return 0.0

    def get_instance_id(self):
        return 0


class _PacedThread:
    """Camera thread stand-in that renders at a fixed frame rate and exposes
    latest_raw_seq -- so the control loop sees stale frames between captures,
    exactly like the real CameraThread."""

    def __init__(self, sim, cam_index=0, fps=20.0):
        self.sim = sim
        self.cam_index = cam_index
        self.period = 1.0 / fps
        self._last = 0.0
        self.latest_raw = None
        self.latest_raw_seq = 0

    def get_latest_raw(self):
        now = time.time()
        if self.latest_raw is None or (now - self._last) >= self.period:
            self._last = now
            self.latest_raw = self.sim.render_frame(self.cam_index)
            self.latest_raw_seq += 1
        return self.latest_raw


class _Cam:
    def __init__(self, thread):
        self.thread = thread


class _Vis:
    selected_launch = None
    launch_launched = False
    selected_aircraft = None
    aircraft_trajectories = {}
    current_tt = 0.0

    def __init__(self):
        self.selected_satellite = "SIMSAT"
        self.satellite_trajectories = {"SIMSAT": ([(0,)] * 8, [0.0])}
        self.telescope_azimuth = 0.0
        self.telescope_altitude = 0.0


class _Rig:
    """One live-fidelity tracking rig: sim mount + paced sim camera + real
    JoystickModeState driven through tracking_control at 15 Hz."""

    def __init__(self, az_rate_dps=0.25, el_rate_dps=0.0, start_offset=None,
                 fps=20.0, tuned=True):
        cfg = ConfigState()
        if tuned:
            # Live fidelity: the field-tuned configuration (continuous
            # guide-rate tracking, tuned PID gains, lead time) from the
            # committed example config -- with the sim-scenario site geometry
            # (alignment/offsets zero) overlaid.
            import json
            cfg.load_from_dict(json.load(open("config.example.json")))
        else:
            # Legacy/fallback configuration: discrete MC_MOVE rate steps and
            # a plain P controller -- coarser (sawtooth) by design.
            cfg.continuous_rate_tracking = False
            cfg.pid_azm_p_gain = cfg.pid_alt_p_gain = 0.3
            cfg.pid_azm_i_gain = cfg.pid_alt_i_gain = 0.0
            cfg.pid_azm_d_gain = cfg.pid_alt_d_gain = 0.0
        cfg.sim_config["enabled"] = True
        cfg.azm_offset_str = "0"
        cfg.alt_offset_str = "0"
        cfg.alignment_azimuth_str = "0.0"
        cfg.alignment_elevation_str = "0.0"
        cfg.mount_mode = "AltAz"
        self.cfg = cfg

        self.t0 = time.time()
        self.az0, self.el0 = 100.0, 30.0
        self.az_rate, self.el_rate = az_rate_dps, el_rate_dps

        self.sim = HardwareSimulator(cfg, None, None)
        self.sim.current_target_azel = lambda: (*self.target_azel(), 'satellite', True)
        # Start off-target but INSIDE the tracking camera's FOV (the optical
        # modes can only acquire what is in frame; the configured camera
        # geometry may be far narrower than "wide" suggests). Default offset:
        # half the half-FOV on each axis.
        pix = float(cfg.get_camera_pixel_size("camera1"))
        foc = float(cfg.get_camera_focal_length("camera1"))
        ifd = 206.0 * pix / foc / 3600.0    # deg per pixel
        w = int(cfg.sim_config.get("cam_width", 960))
        h = int(cfg.sim_config.get("cam_height", 720))
        self.fov_x = w * ifd / 2.0          # half-FOV, deg
        self.fov_y = h * ifd / 2.0
        if start_offset is None:
            start_offset = (self.fov_x * 0.5, self.fov_y * 0.5)
        self.sim.mount.az_true_deg = self.az0 + start_offset[0]
        self.sim.mount.el_true_deg = (90.0 - self.el0) + start_offset[1]

        self.state = JoystickModeState(None, cfg, self._status)
        self.state.hardware_sim = self.sim
        self.state.telescope_connected = True
        self.state.telescope_controller = self.sim.mount
        self.state.feed_forward_azm_enabled = tuned
        self.state.feed_forward_alt_enabled = tuned
        self.state.tracking_vis_state = _Vis()
        joy = _FakeJoystick()
        self.state.joysticks = {0: joy}
        self.state.connected_joystick = 0
        self.state.joystick_tare = {}

        self.thread = _PacedThread(self.sim, 0, fps=fps)
        self._orig_get_camera = jc.camera_manager.get_camera
        jc.camera_manager.get_camera = lambda idx: _Cam(self.thread)

        self._orig_interp = jc.interpolate_position_data_and_rates
        jc.interpolate_position_data_and_rates = self._interp

        self.messages = []
        self.pointing_log = []   # (t, sky_err_deg)

    def _status(self, msg):
        self.messages.append(msg)

    def target_azel(self, t=None):
        dt = (t if t is not None else time.time()) - self.t0
        return (self.az0 + self.az_rate * dt) % 360.0, self.el0 + self.el_rate * dt

    def _interp(self, traj, tt, *a, **k):
        az, el = self.target_azel()
        return (1.0, 1.0, el, 500.0, az, self.az_rate, self.el_rate)

    def sky_error_deg(self):
        """Angular error between the sim mount's true pointing and the target."""
        taz, tel = self.target_azel()
        maz, mel = simulator.normalize_azel(
            self.sim.mount.az_true_deg, 90.0 - self.sim.mount.el_true_deg)
        return math.hypot(simulator.wrap180(maz - taz) *
                          math.cos(math.radians(tel)), mel - tel)

    def run(self, seconds, hz=15.0):
        """Drive the REAL control dispatch (mode entry logic included) like
        MountControlThread does, logging the sky error each cycle."""
        period = 1.0 / hz
        end = time.time() + seconds
        while time.time() < end:
            c0 = time.perf_counter()
            self.state.current_azm = self.sim.mount.hc_get_position(Targets.AZM) * 360.0
            self.state.current_alt = self.sim.mount.hc_get_position(Targets.ALT) * 360.0
            self.state.current_azm_raw = self.state.current_azm
            self.state.current_alt_raw = self.state.current_alt
            self.state.tracking_control()
            self.pointing_log.append((time.time() - self.t0, self.sky_error_deg()))
            sleep = period - (time.perf_counter() - c0)
            if sleep > 0:
                time.sleep(sleep)

    def close(self):
        jc.camera_manager.get_camera = self._orig_get_camera
        jc.interpolate_position_data_and_rates = self._orig_interp

    def max_error_after(self, t_settle):
        errs = [e for (t, e) in self.pointing_log
                if t >= t_settle + (self.pointing_log[0][0] if self.pointing_log else 0)]
        return max(errs) if errs else float("inf")


class ProgramTrackingTests(unittest.TestCase):

    def test_program_tracks_tightly_with_tuned_config(self):
        # Continuous guide-rate + feed-forward + tuned gains (the field
        # config): the loop should hold a moving target within ~0.2 deg.
        rig = _Rig()
        try:
            rig.state.tracking_mode = TrackingMode.PROGRAM
            rig.run(6.0)
        finally:
            rig.close()
        self.assertEqual(rig.state.tracking_mode, TrackingMode.PROGRAM)
        errs = [e for (t, e) in rig.pointing_log]
        settled = errs[len(errs) // 2:]
        self.assertLess(max(settled), 0.2,
                        f"PROGRAM (tuned) walked off: max settled error {max(settled):.2f} deg "
                        f"(errors tail: {[f'{e:.2f}' for e in settled[-10:]]})")

    def test_program_bounded_with_discrete_fallback_config(self):
        # Discrete MC_MOVE steps produce the known sawtooth; it must stay
        # BOUNDED (a limit cycle), never walk off.
        rig = _Rig(tuned=False)
        try:
            rig.state.tracking_mode = TrackingMode.PROGRAM
            rig.run(6.0)
        finally:
            rig.close()
        self.assertEqual(rig.state.tracking_mode, TrackingMode.PROGRAM)
        errs = [e for (t, e) in rig.pointing_log]
        settled = errs[len(errs) // 2:]
        self.assertLess(max(settled), 1.2,
                        f"PROGRAM (discrete) diverged: max settled error {max(settled):.2f} deg")


class HandoffTests(unittest.TestCase):

    def test_handoff_engages_hotspot_and_keeps_tracking(self):
        rig = _Rig()
        try:
            rig.state.tracking_mode = TrackingMode.HANDOFF
            rig.run(8.0)
        finally:
            rig.close()
        self.assertEqual(
            rig.state.tracking_mode, TrackingMode.HOTSPOT,
            f"HANDOFF should have engaged HOTSPOT and HELD it; "
            f"mode={rig.state.tracking_mode} status={rig.state.hotspot_status!r} "
            f"messages={rig.messages[-6:]}")
        errs = [e for (t, e) in rig.pointing_log]
        settled = errs[len(errs) // 2:]
        self.assertLess(max(settled), 0.8,
                        f"walked off after handoff: max settled error {max(settled):.2f} deg")


class HotspotTests(unittest.TestCase):

    def test_hotspot_acquires_and_holds_on_moving_target(self):
        rig = _Rig()
        try:
            rig.state.tracking_mode = TrackingMode.HOTSPOT
            rig.run(6.0)
        finally:
            rig.close()
        self.assertEqual(
            rig.state.tracking_mode, TrackingMode.HOTSPOT,
            f"HOTSPOT reverted (status={rig.state.hotspot_status!r}, "
            f"messages={rig.messages[-6:]})")
        self.assertTrue(rig.state.hotspot_acquired)
        errs = [e for (t, e) in rig.pointing_log]
        settled = errs[len(errs) // 2:]
        self.assertLess(max(settled), 0.8,
                        f"HOTSPOT walked off: max settled error {max(settled):.2f} deg")

    def test_hotspot_survives_slow_frames(self):
        # 4 fps camera vs 15 Hz control loop: mostly-stale frames must ride
        # the stale gate + coast, not trip the loss fallback.
        rig = _Rig(fps=4.0)
        try:
            rig.state.tracking_mode = TrackingMode.HOTSPOT
            rig.run(5.0)
        finally:
            rig.close()
        self.assertEqual(rig.state.tracking_mode, TrackingMode.HOTSPOT,
                         f"reverted on slow frames (status={rig.state.hotspot_status!r})")

    def test_hotspot_with_target_out_of_fov_falls_back_to_program(self):
        # Engaged with nothing (or only stars) in frame, HOTSPOT must decline
        # gracefully -- fall back to PROGRAM, which re-centers -- rather than
        # chase a star or sit dead. This is the designed division of labor:
        # PROGRAM owns acquisition, HOTSPOT owns centering.
        rig = _Rig(start_offset=(3.0, 1.5))  # far outside the ~0.5 deg FOV
        try:
            rig.state.tracking_mode = TrackingMode.HOTSPOT
            rig.run(6.0)
        finally:
            rig.close()
        self.assertEqual(rig.state.tracking_mode, TrackingMode.PROGRAM,
                         "should have handed back to PROGRAM for re-acquisition")
        errs = [e for (t, e) in rig.pointing_log]
        self.assertLess(errs[-1], 0.3,
                        f"PROGRAM should be re-centering after fallback, err={errs[-1]:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
