#!/usr/bin/env python
"""
Headless test for RustCoreLoopAdapter: validates the glue (bridge build on
connect, static-config + mode push, RATE joystick mapping, and snapshot
read-back into app state) without the pygame UI, a real mount, or skyfield.

Uses a fake mount (SimMount-like Targets interface), fake joystick, and a
SimpleNamespace config/state.

    cd rust/skytracker-ffi && maturin develop --release
    python test_rust_loop_adapter.py
"""

import threading
import time
import types
import unittest

try:
    import skytracker_core  # noqa: F401
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

from lib.auxstar import RATES, Targets


class FakeMount:
    def __init__(self):
        self.az = 0.0
        self.el = 0.0
        self.focus = 0.0
        self.azr = 0.0
        self.elr = 0.0
        self.focusr = 0.0
        self.t = time.time()
        self.lock = threading.Lock()

    def _adv(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        self.az = (self.az + self.azr * dt) % 360.0
        self.el += self.elr * dt
        self.focus = (self.focus + self.focusr * dt) % 360.0

    def hc_get_position(self, target):
        with self.lock:
            self._adv()
            if target == Targets.ALT:
                v = self.el
            elif target == Targets.FOCUS:
                v = self.focus
            else:
                v = self.az
            return (v % 360.0) / 360.0

    def hc_slew_fixed(self, target, rate):
        with self.lock:
            self._adv()
            dps = (1.0 if rate >= 0 else -1.0) * RATES.get(abs(int(rate)), 0.0) * 360.0
            if target == Targets.ALT:
                self.elr = dps
            elif target == Targets.FOCUS:
                self.focusr = dps
            else:
                self.azr = dps
            return True

    def hc_goto_fast(self, target, dd, mm, ss):
        with self.lock:
            self._adv()
            deg = abs(dd) + mm / 60.0 + ss / 3600.0
            deg = deg if dd >= 0 else -deg
            if target == Targets.ALT:
                self.el = deg
                self.elr = 0.0
            else:
                self.az = deg % 360.0
                self.azr = 0.0
            return True


class FakeJoystick:
    def __init__(self, axes):
        self.axes = axes

    def get_numaxes(self):
        return len(self.axes)

    def get_axis(self, i):
        return self.axes[i]


def fake_config():
    return types.SimpleNamespace(
        pid_azm_p_gain=0.025, pid_azm_i_gain=0.0, pid_azm_d_gain=0.0,
        pid_alt_p_gain=0.025, pid_alt_i_gain=0.0, pid_alt_d_gain=0.0,
        azm_limit_min_str="0", azm_limit_max_str="360",
        alt_limit_min_str="-10", alt_limit_max_str="90",
        azm_offset_str="0", alt_offset_str="0",
        mount_mode="AltAz",
        alignment_azimuth_str="0", alignment_elevation_str="0",
    )


def fake_state(controller, mode, joystick=None):
    from joystick_controller import JoystickModeState, TrackingMode, AdaptiveRateMapper
    st = types.SimpleNamespace(
        rate_mapper=AdaptiveRateMapper(),  # adaptive gearbox (_push_rate needs it)
        telescope_connected=True,
        telescope_controller=controller,
        update_status_callback=None,
        stopped=False,
        tracking_mode=mode,
        connected_joystick=(0 if joystick else None),
        joysticks=({0: joystick} if joystick else {}),
        joystick_tare={},
        tracking_vis_state=types.SimpleNamespace(telescope_azimuth=0.0, telescope_altitude=0.0),
        feed_forward_azm_enabled=False,
        feed_forward_alt_enabled=False,
        # written by read-back:
        current_azm=0.0, current_alt=0.0, current_azm_raw=0.0, current_alt_raw=0.0,
        position_fresh=False, azm_position_error=0.0, alt_position_error=0.0,
        azm_pid_output=0.0, alt_pid_output=0.0,
        hotspot_snr=0.0, hotspot_status="", hotspot_acquired=False, hotspot_centroid=None,
        azm_display_str="--", alt_display_str="--",
        # Focus motor state: the adapter drives focus directly off the controller
        # via the real JoystickModeState methods (bound below). Triggers are
        # pre-"seen" so a pressed value commands immediately in the test.
        focus_axis_forward=5, focus_axis_backward=4,
        _focus_trigger_seen={4: True, 5: True}, _focus_last_rate=0,
        focus_rate=0, current_focus=0,
    )
    # Exercise the production focus code (not stubs) through the adapter.
    st._handle_focus_control = types.MethodType(JoystickModeState._handle_focus_control, st)
    st._poll_focus_position = types.MethodType(JoystickModeState._poll_focus_position, st)
    return st


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class AdapterTests(unittest.TestCase):

    def test_rate_mode_moves_and_reads_back(self):
        from joystick_controller import TrackingMode
        from rust_loop_adapter import RustCoreLoopAdapter

        fm = FakeMount()
        joy = FakeJoystick([0.0, 0.0, 1.0, 0.0])  # axis 2 = full +AZM
        st = fake_state(fm, TrackingMode.RATE_CONTROL, joystick=joy)
        adapter = RustCoreLoopAdapter(st, fake_config(), target_hz=50)
        adapter.start()
        try:
            time.sleep(1.2)
        finally:
            adapter.stop()
            adapter.join(timeout=2.0)
        # The mount moved under the +AZM rate and the snapshot was read back.
        # With the adaptive gearbox a pinned stick starts at the base ceiling
        # (step 5 = 0.5 dps) and winds up after 0.8 s, so 1.2 s pinned gives
        # ~0.8 deg -- comfortably past the 0.5 deg threshold, but far less
        # than the old instant step 9 (10 dps) would have.
        self.assertGreater(fm.az, 0.5, "fake mount AZM should have advanced")
        self.assertTrue(st.position_fresh)
        self.assertGreater(st.current_azm, 0.5, "read-back should reflect motion")

    def test_focus_driven_on_rust_path(self):
        # The Rust loop doesn't own the focus axis, so the adapter must command it
        # directly off the controller and read its position back. Hold R2 (axis 5
        # = +1) and confirm the focus advances and the read-back reflects it.
        from joystick_controller import TrackingMode
        from rust_loop_adapter import RustCoreLoopAdapter

        fm = FakeMount()
        joy = FakeJoystick([0.0, 0.0, 0.0, 0.0, -1.0, 1.0])  # axis 5 (R2) = full +
        st = fake_state(fm, TrackingMode.STANDBY, joystick=joy)
        adapter = RustCoreLoopAdapter(st, fake_config(), target_hz=50)
        adapter.start()
        try:
            time.sleep(1.2)
        finally:
            adapter.stop()
            adapter.join(timeout=2.0)
        self.assertGreater(fm.focus, 0.5, "focus axis should have advanced under +R2")
        self.assertEqual(st.focus_rate, 9, "R2 fully pressed maps to focus rate +9")
        self.assertGreater(st.current_focus, 0, "focus position should read back")

    def test_disconnected_is_safe(self):
        from joystick_controller import TrackingMode
        from rust_loop_adapter import RustCoreLoopAdapter

        st = fake_state(None, TrackingMode.STANDBY)
        st.telescope_connected = False
        adapter = RustCoreLoopAdapter(st, fake_config(), target_hz=50)
        adapter.start()
        try:
            time.sleep(0.3)
        finally:
            adapter.stop()
            adapter.join(timeout=2.0)
        # No loop built, no crash, no fresh data.
        self.assertFalse(st.position_fresh)
        self.assertIsNone(adapter.loop)


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class HandoffFramePushTests(unittest.TestCase):
    """_push_hotspot must resolve the real camera_manager SINGLETON and push
    the latest camera frame into the Rust loop. Regression: it imported the
    camera_manager MODULE (which has no get_camera attribute), so every
    HANDOFF/HOTSPOT pump raised AttributeError -- swallowed by the pump's
    catch-all -- and the loop never received a single frame: HANDOFF sat at
    'no detection' forever even with the target dead-centered."""

    def test_push_hotspot_reaches_singleton_and_pushes_frame(self):
        import numpy as np
        from camera_manager import camera_manager as cm
        from joystick_controller import TrackingMode
        from rust_loop_adapter import RustCoreLoopAdapter

        cfg = fake_config()
        cfg.hotspot_camera_index = 0
        cfg.hotspot_snr_threshold = 5.0
        cfg.hotspot_gate_radius = 120
        cfg.hotspot_coast_time_sec = 1.0
        cfg.hotspot_x_sign = 1.0
        cfg.hotspot_y_sign = -1.0
        cfg.hotspot_max_rate_dps = 2.0
        cfg.get_camera_pixel_size = lambda name: 2.9
        cfg.get_camera_focal_length = lambda name: 162.0
        cfg.get_camera_alignment_rotation = lambda name: 0.0

        st = fake_state(FakeMount(), TrackingMode.HANDOFF)
        adapter = RustCoreLoopAdapter(st, cfg, target_hz=50)

        calls = []

        class FakeLoop:
            def set_hotspot_params(self, *a):
                calls.append(("params",))

            def push_frame(self, arr, age_s=None):
                calls.append(("frame", arr.shape, str(arr.dtype)))

        adapter.loop = FakeLoop()

        frame = np.zeros((720, 960), dtype=np.uint8)
        frame[360, 480] = 255
        cam = cm.get_camera(0)
        self.assertIsNotNone(cam, "real singleton must expose camera 0")
        orig_thread = cam.thread
        cam.thread = types.SimpleNamespace(
            get_latest_raw=lambda: frame, latest_raw_seq=1)
        try:
            adapter._push_hotspot(st, cfg)  # must not raise
            kinds = [c[0] for c in calls]
            self.assertIn("params", kinds, "hotspot params must be pushed")
            self.assertIn("frame", kinds, "camera frame must reach the loop")
            _, shape, dtype = next(c for c in calls if c[0] == "frame")
            self.assertEqual(shape, (720, 960))
            self.assertEqual(dtype, "float32")

            # Stale-seq gate: the same frame seq is not re-pushed...
            calls.clear()
            adapter._push_hotspot(st, cfg)
            self.assertNotIn("frame", [c[0] for c in calls])
            # ...but a new capture (seq bump) is.
            cam.thread.latest_raw_seq = 2
            adapter._push_hotspot(st, cfg)
            self.assertIn("frame", [c[0] for c in calls])
        finally:
            cam.thread = orig_thread


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class WatchdogTests(unittest.TestCase):
    """Liveness-watchdog behavior, including the old-wheel regression: a
    skytracker_core build without cycle_count in snapshots must DISABLE the
    stall watchdog, not trip it 2 s after every connect (which presented as
    'the sim mount tracks briefly after reconnect, then stops')."""

    def _adapter(self, mode=None):
        from joystick_controller import TrackingMode
        from rust_loop_adapter import RustCoreLoopAdapter
        ctrl = FakeMount()
        st = fake_state(ctrl, mode or TrackingMode.STANDBY)
        msgs = []
        st.update_status_callback = msgs.append
        adapter = RustCoreLoopAdapter(st, fake_config(), target_hz=50)
        return adapter, st, ctrl, msgs

    def test_missing_cycle_count_does_not_trip_watchdog(self):
        adapter, st, ctrl, msgs = self._adapter()
        self.assertTrue(adapter._ensure_loop())
        loop = adapter.loop
        # Simulate an old-wheel snapshot: no cycle_count / loop_dead keys, and
        # let far more than WATCHDOG_STALL_SEC of apparent silence accumulate.
        adapter._last_cycle_advance -= 10 * adapter.WATCHDOG_STALL_SEC
        old_snap = {"fresh": True, "azm": 0.0, "alt": 0.0}
        for _ in range(5):
            self.assertFalse(adapter._watchdog(old_snap),
                             "missing liveness API must not read as a stall")
        self.assertIs(adapter.loop, loop, "loop must not be torn down")
        self.assertTrue(any("wheel" in m for m in msgs),
                        f"operator should be told to rebuild: {msgs}")
        adapter.loop.stop()
        adapter.loop = None

    def test_real_stall_trips_then_retries_after_cooldown(self):
        adapter, st, ctrl, msgs = self._adapter()
        self.assertTrue(adapter._ensure_loop())
        # A snapshot whose cycle_count stops advancing IS a stall.
        adapter._last_cycle_count = 42
        adapter._last_cycle_advance = time.monotonic() - (adapter.WATCHDOG_STALL_SEC + 1)
        tripped = adapter._watchdog({"cycle_count": 42, "loop_dead": False})
        self.assertTrue(tripped)
        self.assertIsNone(adapter.loop)
        self.assertTrue(any("retrying" in m for m in msgs), msgs)
        # Within the cooldown: no rebuild.
        self.assertFalse(adapter._ensure_loop())
        # After the cooldown: rebuilds on its own (no reconnect needed).
        adapter._loop_retry_at = time.monotonic() - 0.01
        self.assertTrue(adapter._ensure_loop())
        adapter.loop.stop()
        adapter.loop = None

    def test_reconnect_clears_retry_cooldown_immediately(self):
        adapter, st, ctrl, msgs = self._adapter()
        adapter._loop_retry_at = time.monotonic() + 1000.0
        st.telescope_connected = False
        self.assertFalse(adapter._ensure_loop())   # disconnect resets cooldown
        st.telescope_connected = True
        self.assertTrue(adapter._ensure_loop(), "reconnect must retry immediately")
        adapter.loop.stop()
        adapter.loop = None

    def test_live_loop_with_new_wheel_never_trips(self):
        # End-to-end: real bridged loop running for > WATCHDOG_STALL_SEC with
        # real snapshots must never trip the watchdog (cycle_count advances).
        adapter, st, ctrl, msgs = self._adapter()
        self.assertTrue(adapter._ensure_loop())
        deadline = time.time() + 2.5 * adapter.WATCHDOG_STALL_SEC
        while time.time() < deadline:
            snap = adapter.loop.snapshot()
            self.assertIn("cycle_count", snap,
                          "wheel in this env must expose the liveness API")
            self.assertFalse(adapter._watchdog(snap),
                             "healthy loop must not trip the watchdog")
            time.sleep(0.1)
        adapter.loop.stop()
        adapter.loop = None


if __name__ == "__main__":
    unittest.main(verbosity=2)
