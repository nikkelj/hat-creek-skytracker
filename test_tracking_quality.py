#!/usr/bin/env python
"""Live-fidelity tracking-QUALITY metrics for the Rust-loop optical chain.

Unlike the mode-machine tests (which assert the state machine works at all),
this suite runs the FULL live wiring -- camera_manager singleton + SimCap +
real CameraThread + RustCoreLoopAdapter on its own pump thread + real
JoystickModeState -- with the field configuration (example config: noise,
backlash, periodic error, pointing model, ~10 fps camera) and MEASURES the
behaviors reported from live runs:

  * HANDOFF latency (arm -> HOTSPOT engaged) and how often the star filter
    falsely rejects the real target in a clean scene,
  * HOTSPOT hold time over a long window (does it drop the lock?),
  * tracking jitter (RMS / P95 sky error) in PROGRAM and HOTSPOT.

Each scenario prints its metrics and asserts coarse quality floors, so a
behavioral regression shows up as numbers, not just a red/green bit.

Wall-clock based (real threads); the suite takes ~1.5 minutes.
Run: python test_tracking_quality.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import math
import time
import types
import unittest

import numpy as np
import pygame
pygame.init()

from config import ConfigState
from simulator import HardwareSimulator
import simulator

import joystick_controller as jc
import rust_loop_adapter as rla
from joystick_controller import JoystickModeState, TrackingMode

try:
    import skytracker_core  # noqa: F401
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False


class Rig:
    """Full-wiring tracking rig with an analytic target."""

    def __init__(self, az_rate_dps=-0.17, el_rate_dps=0.5, fps=10.0):
        cfg = ConfigState()
        cfg.load_from_dict(json.load(open("config.example.json")))
        cfg.sim_config["enabled"] = True
        # Live fidelity: the real star field ON -- stars stream through the
        # FOV at the tracking rate and are exactly what the star filter (and
        # the brightest-object detector) must contend with in the field.
        cfg.sim_config["sim_use_real_stars"] = True
        # ~fps camera cadence via the sim's exposure/readout timing model
        # (live cameras run ~10 fps; the control loop runs 15 Hz, so frames
        # arrive with the same capture->process timing skew as hardware).
        cfg.sim_config["sim_readout_latency_s"] = max(0.01, 1.0 / fps - 0.01)
        cfg.azm_offset_str = "0"; cfg.alt_offset_str = "0"
        cfg.alignment_azimuth_str = "0.0"; cfg.alignment_elevation_str = "0.0"
        cfg.mount_mode = "AltAz"
        # Sim-measured transport latency is ~65 ms; the hardware-tuned 0.15 s
        # lead OVER-leads the sim and reads as a constant ~0.03-0.05 deg
        # offset ahead of the target (bias, not jitter). Measured sweep:
        # lead 0.15 -> 158 arcsec bias, 0.05 -> 44, 0.0 -> 120.
        cfg.pid_lead_time_sec = 0.06
        self.cfg = cfg

        self.sim = HardwareSimulator(cfg, None, None)
        self.t0 = time.time()
        self.az0, self.el0 = 100.0, 30.0
        self.az_rate, self.el_rate = az_rate_dps, el_rate_dps
        self.sim.current_target_azel = lambda: (*self.target_azel(), 'satellite', True)
        self.sim.mount.az_true_deg = self.az0
        self.sim.mount.el_true_deg = 90.0 - self.el0

        from camera_manager import camera_manager as cm
        self.cm = cm
        cm.simulator = self.sim
        assert cm.connect_camera(0), "sim camera must connect"

        self.messages = []
        st = JoystickModeState(None, cfg, self.messages.append)
        st.telescope_connected = True
        st.telescope_controller = self.sim.mount
        st.feed_forward_azm_enabled = True
        st.feed_forward_alt_enabled = True
        st.tracking_vis_state = types.SimpleNamespace(
            selected_launch=None, launch_launched=False, launch_trajectories={},
            selected_satellite="SIMSAT", satellite_trajectories={"SIMSAT": object()},
            selected_aircraft=None, aircraft_trajectories={},
            current_tt=0.0, telescope_azimuth=0.0, telescope_altitude=0.0)
        st.joysticks = {}
        st.connected_joystick = None
        st.joystick_tare = {}
        self.state = st

        self._orig_apt = jc.active_program_trajectory
        self._orig_interp = rla.interpolate_position_data_and_rates
        jc.active_program_trajectory = lambda v: (object(), 'satellite', 'SIMSAT')
        rla.interpolate_position_data_and_rates = (
            lambda traj, tt, *a, **k: (1.0, 1.0, self.target_azel()[1], 500.0,
                                       self.target_azel()[0],
                                       self.az_rate, self.el_rate))

        self.adapter = rla.RustCoreLoopAdapter(st, cfg, target_hz=15)
        self.adapter.start()
        time.sleep(1.0)  # bridge + settle into PROGRAM-quality pointing

    def target_azel(self, t=None):
        dt = (t if t is not None else time.time()) - self.t0
        return (self.az0 + self.az_rate * dt) % 360.0, self.el0 + self.el_rate * dt

    def sky_err(self):
        """Angular distance from the CAMERA BORESIGHT to the true target --
        i.e. what the operator sees in the tracking camera. The boresight is
        the mount's true pointing run through the sim's injected pointing
        model (the same transform render_frame uses), so a pointing-model
        pre-correction that lands the camera on target reads as zero error."""
        taz, tel = self.target_azel()
        maz, mel = simulator.normalize_azel(
            self.sim.mount.az_true_deg, 90.0 - self.sim.mount.el_true_deg)
        maz, mel = simulator.injected_boresight(self.cfg, maz, mel)
        return math.hypot(
            simulator.wrap180(maz - taz) * math.cos(math.radians(tel)), mel - tel)

    def close(self):
        self.adapter.stop()
        self.adapter.join(timeout=3.0)
        self.cm.disconnect_camera(0)
        jc.active_program_trajectory = self._orig_apt
        rla.interpolate_position_data_and_rates = self._orig_interp

    # -- metric sampling -----------------------------------------------------
    def sample(self, seconds, hz=30.0):
        """Sample (t, mode, handoff_status, hotspot_status, sky_err) tuples."""
        out = []
        end = time.time() + seconds
        while time.time() < end:
            out.append((time.time() - self.t0, self.state.tracking_mode,
                        getattr(self.state, 'handoff_status', ''),
                        self.state.hotspot_status, self.sky_err()))
            time.sleep(1.0 / hz)
        return out


def _transitions(samples, col, value_substr):
    """Count entries INTO a status containing value_substr (dedup consecutive)."""
    n, prev = 0, False
    for s in samples:
        cur = value_substr in (s[col] or "")
        if cur and not prev:
            n += 1
        prev = cur
    return n


def _rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("inf")


def _bias_jitter(vals):
    """(mean, std) in degrees -- separates a systematic offset from wobble."""
    if not vals:
        return float("inf"), float("inf")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def _p95(vals):
    if not vals:
        return float("inf")
    return sorted(vals)[int(0.95 * (len(vals) - 1))]


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built")
class TrackingQualityTests(unittest.TestCase):

    def test_1_program_jitter(self):
        """PROGRAM-track jitter floor: RMS/P95 sky error over 12 s settled."""
        rig = Rig()
        try:
            rig.state.tracking_mode = TrackingMode.PROGRAM
            time.sleep(3.0)  # settle
            samples = rig.sample(12.0)
        finally:
            rig.close()
        errs = [s[4] for s in samples]
        bias, jit = _bias_jitter(errs)
        print(f"\n[metrics] PROGRAM: bias={bias*3600:.1f} jitter(std)={jit*3600:.1f} "
              f"rms={_rms(errs)*3600:.1f} p95={_p95(errs)*3600:.1f} "
              f"peak={max(errs)*3600:.1f} arcsec")
        self.assertLess(_rms(errs), 0.05, "PROGRAM should hold a moving target to <0.05 deg RMS")

    def test_2_handoff_latency_and_false_rejects(self):
        """Arm HANDOFF on a clean scene (target only): it must engage HOTSPOT
        promptly, and the star filter must not reject the real target."""
        rig = Rig()
        try:
            rig.state.tracking_mode = TrackingMode.PROGRAM
            time.sleep(2.5)  # PROGRAM settles the target near boresight
            t_arm = time.time()
            rig.state.tracking_mode = TrackingMode.HANDOFF
            latency = None
            samples = []
            end = time.time() + 20.0
            while time.time() < end:
                samples.append((time.time() - rig.t0, rig.state.tracking_mode,
                                getattr(rig.state, 'handoff_status', ''),
                                rig.state.hotspot_status, rig.sky_err()))
                if rig.state.tracking_mode == TrackingMode.HOTSPOT:
                    latency = time.time() - t_arm
                    break
                time.sleep(1.0 / 30.0)
            false_rejects = _transitions(samples, 2, "star")
        finally:
            rig.close()
        print(f"\n[metrics] HANDOFF latency: "
              f"{'NEVER' if latency is None else f'{latency:.2f}s'}; "
              f"target false-reject episodes: {false_rejects}")
        self.assertIsNotNone(latency, "HANDOFF never engaged HOTSPOT in 20s")
        self.assertLess(latency, 3.0,
                        f"HANDOFF took {latency:.2f}s (star filter fighting the target?)")
        self.assertLessEqual(false_rejects, 1,
                             f"{false_rejects} false star-rejections of the real target")

    def test_3_hotspot_hold_and_jitter(self):
        """After hand-off, HOTSPOT must HOLD the lock over a long window and
        keep the target centered."""
        rig = Rig()
        try:
            rig.state.tracking_mode = TrackingMode.PROGRAM
            time.sleep(2.5)
            rig.state.tracking_mode = TrackingMode.HANDOFF
            # Wait (up to 20 s) for the hand-off.
            end = time.time() + 20.0
            while time.time() < end and rig.state.tracking_mode != TrackingMode.HOTSPOT:
                time.sleep(0.05)
            self.assertEqual(rig.state.tracking_mode, TrackingMode.HOTSPOT,
                             "hand-off never engaged; cannot measure hold")
            t_engage = time.time()
            hold_window = 30.0
            samples = []
            first_loss = None
            end = time.time() + hold_window
            while time.time() < end:
                samples.append((time.time() - rig.t0, rig.state.tracking_mode,
                                getattr(rig.state, 'handoff_status', ''),
                                rig.state.hotspot_status, rig.sky_err()))
                if (first_loss is None
                        and rig.state.tracking_mode != TrackingMode.HOTSPOT):
                    first_loss = time.time() - t_engage
                time.sleep(1.0 / 30.0)
            errs = [s[4] for s in samples if s[1] == TrackingMode.HOTSPOT]
            losses = _transitions(samples, 3, "lost")
            rejects = _transitions(samples, 3, "star-reject")
        finally:
            rig.close()
        hold = hold_window if first_loss is None else first_loss
        bias, jit = _bias_jitter(errs)
        print(f"\n[metrics] HOTSPOT hold: {hold:.1f}s of {hold_window:.0f}s "
              f"(loss events: {losses}, star-reject episodes: {rejects}); "
              f"bias={bias*3600:.1f} jitter(std)={jit*3600:.1f} "
              f"rms={_rms(errs)*3600:.1f} p95={_p95(errs)*3600:.1f} arcsec")
        self.assertIsNone(first_loss,
                          f"HOTSPOT dropped the lock after {hold:.1f}s")
        self.assertLess(_rms(errs), 0.08, "HOTSPOT jitter above 0.08 deg RMS")

    def test_4_fast_target_handoff_and_hold(self):
        """Launch-profile rates (2.0 az / 1.0 el deg/s). Pins two regressions:
        (a) the rate filter's short-baseline timing-skew noise scaled with
        target rate and falsely star-rejected the real target (HANDOFF never
        engaged); (b) the HOTSPOT gate prediction used the TOTAL commanded
        rate, so with feed-forward it slid off a well-tracked fast target and
        the lock dropped ~1 s after engagement."""
        rig = Rig(az_rate_dps=2.0, el_rate_dps=1.0)
        try:
            rig.state.tracking_mode = TrackingMode.PROGRAM
            time.sleep(3.0)
            t_arm = time.time()
            rig.state.tracking_mode = TrackingMode.HANDOFF
            latency = None
            samples = []
            end = time.time() + 25.0
            while time.time() < end:
                samples.append((time.time() - rig.t0, rig.state.tracking_mode,
                                getattr(rig.state, 'handoff_status', ''),
                                rig.state.hotspot_status, rig.sky_err()))
                if latency is None and rig.state.tracking_mode == TrackingMode.HOTSPOT:
                    latency = time.time() - t_arm
                time.sleep(1.0 / 30.0)
            false_rejects = _transitions(samples, 2, "star")
            losses = _transitions(samples, 3, "lost")
            locked_errs = [s[4] for s in samples if s[3] == "locked"]
        finally:
            rig.close()
        print(f"\n[metrics] FAST target: handoff latency="
              f"{'NEVER' if latency is None else f'{latency:.2f}s'}, "
              f"false-rejects={false_rejects}, losses={losses}, "
              f"locked rms={_rms(locked_errs)*3600:.1f} arcsec")
        self.assertIsNotNone(latency, "HANDOFF never engaged on the fast target")
        self.assertLess(latency, 3.0)
        self.assertLessEqual(false_rejects, 1)
        self.assertEqual(losses, 0, "HOTSPOT dropped the fast target")


if __name__ == '__main__':
    unittest.main(verbosity=2)
