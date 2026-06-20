"""
Drop-in replacement for MountControlThread that runs the control loop in Rust
(skytracker_core.CoreLoop), behind the `use_rust_core_loop` flag.

Design (see rust/STEP4_DESIGN.md): the Rust loop owns the cadence + PID +
transforms + hotspot detection on its own thread (off the GIL); Python pushes
the inputs it owns (joystick rate, skyfield setpoint, camera frame) and pulls a
snapshot for display.

Integration choice: the loop BRIDGES to the already-connected Python controller
(`CoreLoop.wrap_mount`), which works identically for the real
NexstarHandController and the sim SimMount. This needs no change to the connect
flow, and manual UI commands can still call the controller directly (its
internal lock serializes them with the loop's calls). The pure-Rust serial path
(`CoreLoop.open_serial`) is available as a later, fully-off-GIL optimization.

This class mirrors MountControlThread's interface (start/stop/join) so main.py
swaps it in with a one-line branch.
"""

import threading
import time

import numpy as np

import skytracker_core as rc
from joystick_controller import TrackingMode, axis_to_rate
from trajectory import interpolate_position_data_and_rates

_MODE_TO_STR = {
    TrackingMode.STANDBY: "standby",
    TrackingMode.RATE_CONTROL: "rate",
    TrackingMode.PROGRAM: "program",
    TrackingMode.HANDOFF: "handoff",
    TrackingMode.HOTSPOT: "hotspot",
    TrackingMode.MTI: "mti",
}
_STR_TO_MODE = {v: k for k, v in _MODE_TO_STR.items()}


def _mount_mode_str(cfg):
    mm = str(getattr(cfg, "mount_mode", "AltAz")).lower()
    if mm == "passthrough":
        return "passthrough"
    if mm == "eq":
        return "eq"
    return "altaz"


class RustCoreLoopAdapter:
    """Runs skytracker_core.CoreLoop and bridges it to the app state."""

    def __init__(self, joystick_mode_state, config_state, target_hz=15):
        self.state = joystick_mode_state
        self.config_state = config_state
        self.target_hz = float(target_hz)
        self.loop = None
        self._stop = threading.Event()
        self._thread = None

    # ---- MountControlThread-compatible lifecycle ----
    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="RustCoreLoopAdapter", daemon=True
        )
        self._thread.start()
        print(f"RustCoreLoopAdapter starting ({self.target_hz:.0f} Hz target)...")

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout=timeout)
        if self.loop is not None:
            self.loop.stop()
            self.loop = None

    # ---- loop ----
    def _run(self):
        period = 1.0 / self.target_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._pump()
            except Exception as e:  # never let the pump die silently
                print(f"RustCoreLoopAdapter pump error: {e}")
            self._stop.wait(max(0.0, period - (time.perf_counter() - t0)))
        # Shutdown: stop the loop thread (it sends a final mount stop).
        if self.loop is not None:
            self.loop.stop()
            self.loop = None
        print("RustCoreLoopAdapter stopped.")

    def _ensure_loop(self):
        """Build the Rust loop on connect; tear it down on disconnect."""
        st = self.state
        connected = bool(st.telescope_connected) and st.telescope_controller is not None
        if self.loop is None and connected:
            self.loop = rc.CoreLoop.wrap_mount(st.telescope_controller, self.target_hz)
            print("RustCoreLoopAdapter: bridged Rust loop to telescope controller")
        elif self.loop is not None and not connected:
            self.loop.stop()
            self.loop = None
        return self.loop is not None

    def _pump(self):
        if not self._ensure_loop():
            return
        st = self.state
        cfg = self.config_state
        loop = self.loop

        # 1) Read snapshot back into app state for display + apply loop-originated
        #    mode transitions (hotspot loss/limit) and status messages.
        snap = loop.snapshot()
        if snap.get("fresh"):
            self._read_back(st, snap)
        req = snap.get("requested_mode")
        if req:
            st.tracking_mode = _STR_TO_MODE.get(req, st.tracking_mode)
        if st.update_status_callback:
            for msg in loop.drain_status():
                st.update_status_callback(msg)

        # 2) Push connection/stop/mode + static config.
        loop.set_connected(bool(st.telescope_connected))
        loop.set_stopped(bool(st.stopped))
        loop.set_mode(_MODE_TO_STR.get(st.tracking_mode, "standby"))
        self._push_static(cfg)

        # 3) Per-mode dynamic inputs.
        mode = st.tracking_mode
        if mode == TrackingMode.RATE_CONTROL:
            self._push_rate(st)
        elif mode == TrackingMode.PROGRAM:
            self._push_program_setpoint(st)
        elif mode == TrackingMode.HANDOFF:
            # HANDOFF needs BOTH the program setpoint (it keeps program-tracking)
            # and camera frames + hotspot params (it detects in parallel).
            self._push_program_setpoint(st)
            self._push_hotspot(st, cfg)
        elif mode == TrackingMode.HOTSPOT:
            self._push_hotspot(st, cfg)

    def _read_back(self, st, snap):
        st.current_azm = snap["azm"]
        st.current_alt = snap["alt"]
        st.current_azm_raw = snap["azm_raw"]
        st.current_alt_raw = snap["alt_raw"]
        st.position_fresh = True
        st.azm_position_error = snap["azm_error"]
        st.alt_position_error = snap["alt_error"]
        st.azm_pid_output = snap["azm_pid_output"]
        st.alt_pid_output = snap["alt_pid_output"]
        st.hotspot_snr = snap.get("hotspot_snr", 0.0)
        st.hotspot_status = snap.get("hotspot_status", "")
        st.hotspot_acquired = snap.get("hotspot_acquired", False)
        st.hotspot_centroid = snap.get("hotspot_centroid")
        try:
            from lib.auxstar import f2dms
            ad, am, asec = f2dms(st.current_azm / 360.0)
            ld, lm, lsec = f2dms(st.current_alt / 360.0)
            st.azm_display_str = f"{ad:3d}°{am:02d}'{asec:04.1f}\""
            st.alt_display_str = f"{ld:3d}°{lm:02d}'{lsec:04.1f}\""
        except Exception:
            pass
        if st.tracking_vis_state is not None:
            st.tracking_vis_state.telescope_azimuth = st.current_azm
            st.tracking_vis_state.telescope_altitude = st.current_alt

    def _push_static(self, cfg):
        loop = self.loop
        loop.set_gains(
            cfg.pid_azm_p_gain, cfg.pid_azm_i_gain, cfg.pid_azm_d_gain,
            cfg.pid_alt_p_gain, cfg.pid_alt_i_gain, cfg.pid_alt_d_gain,
        )
        # Lead time mirrors the Python program_track path so both cores extrapolate
        # the setpoint forward by the same amount (transport-latency compensation).
        loop.set_lead_time(float(getattr(cfg, "pid_lead_time_sec", 0.0) or 0.0))
        try:
            loop.set_limits(
                float(cfg.azm_limit_min_str), float(cfg.azm_limit_max_str),
                float(cfg.alt_limit_min_str), float(cfg.alt_limit_max_str),
            )
        except (ValueError, AttributeError):
            pass
        loop.set_offsets(
            float(getattr(cfg, "azm_offset_str", 0.0) or 0.0),
            float(getattr(cfg, "alt_offset_str", 0.0) or 0.0),
        )
        loop.set_mount_mode(_mount_mode_str(cfg))
        loop.set_continuous_rate(
            bool(getattr(cfg, "continuous_rate_tracking", False)),
            float(getattr(cfg, "guide_rate_max_dps", 5.0)),
        )
        loop.set_handoff_min_frames(int(getattr(cfg, "handoff_min_frames", 5) or 5))
        try:
            loop.set_alignment(
                float(cfg.alignment_azimuth_str), float(cfg.alignment_elevation_str)
            )
        except (ValueError, AttributeError):
            pass

    def _push_rate(self, st):
        joy = None
        if st.connected_joystick is not None:
            joy = st.joysticks.get(st.connected_joystick)
        if joy is None:
            self.loop.set_rate_cmd(0, 0)
            return
        tare = st.joystick_tare.get(st.connected_joystick)

        def axis(i):
            v = joy.get_axis(i) if i < joy.get_numaxes() else 0.0
            if tare and i < len(tare):
                v -= tare[i]
            return v

        self.loop.set_rate_cmd(axis_to_rate(axis(2)), axis_to_rate(axis(3)))

    def _push_program_setpoint(self, st):
        """Satellite tracking setpoint. NOTE: launch tracking and below-horizon
        mask-exit (handled in JoystickModeState.program_track) are deferred; in
        those cases we clear the setpoint so the loop holds rather than misbehave.
        """
        vis = st.tracking_vis_state
        trajectories = getattr(vis, "satellite_trajectories", {}) if vis else {}
        sat = getattr(vis, "selected_satellite", None) if vis else None
        if vis is None or sat is None or sat not in trajectories:
            self.loop.clear_setpoint()
            return
        px, py, target_alt, dist, target_az, az_rate, el_rate = (
            interpolate_position_data_and_rates(trajectories[sat], vis.current_tt)
        )
        if px is None or target_az is None or target_alt is None or target_alt <= 0:
            self.loop.clear_setpoint()
            return
        self.loop.set_ff_enabled(
            bool(st.feed_forward_azm_enabled), bool(st.feed_forward_alt_enabled)
        )
        # Apply operator bias to the setpoint (the Rust loop owns the setpoint
        # here, so it must be applied here or the bias controls have no effect).
        # Reuse JoystickModeState._apply_bias_to_target so both control loops use
        # the EXACT same Az/El + along/cross-track projection (single source of
        # truth -- the two capabilities are co-developed and must not drift). The
        # along/cross-track terms need the target's sky-velocity direction, which
        # is the (az_rate, el_rate) we already have.
        az_rate_f = float(az_rate) if az_rate is not None else 0.0
        el_rate_f = float(el_rate) if el_rate is not None else 0.0
        # Cache target sky-velocity for the camera bias-direction axes (the Python
        # program_track path caches the same fields).
        st.target_az_rate = az_rate_f
        st.target_el_rate = el_rate_f
        st.target_el_deg = float(target_alt)
        biased_az, biased_el = st._apply_bias_to_target(
            float(target_az), float(target_alt), az_rate_f, el_rate_f
        )
        # Pointing-model pre-correction: the Rust loop runs the plain geometric
        # transform, so the 7-term correction must be applied here in Python (mirrors
        # control.compute_mount_position_error so both control paths agree).
        from control import apply_pointing_model
        biased_az, biased_el = apply_pointing_model(self.config_state, biased_az, biased_el)
        self.loop.set_setpoint(biased_az, biased_el, az_rate_f, el_rate_f)

    def _push_hotspot(self, st, cfg):
        import camera_manager
        import hotspot as hs

        cam_index = int(getattr(cfg, "hotspot_camera_index", 0))
        cam_name = f"camera{cam_index + 1}"
        self.loop.set_hotspot_params(
            float(getattr(cfg, "hotspot_snr_threshold", 5.0)),
            float(getattr(cfg, "hotspot_gate_radius", 120)),
            float(getattr(cfg, "hotspot_coast_time_sec", 1.0)),
            float(getattr(cfg, "hotspot_x_sign", 1.0)),
            float(getattr(cfg, "hotspot_y_sign", -1.0)),
            float(cfg.get_camera_pixel_size(cam_name)),
            float(cfg.get_camera_focal_length(cam_name)),
            float(cfg.get_camera_alignment_rotation(cam_name)),
        )
        camera = camera_manager.get_camera(cam_index)
        if camera is not None and getattr(camera, "thread", None) is not None:
            raw = camera.thread.get_latest_raw()
            if raw is not None:
                self.loop.push_frame(hs.to_intensity(raw).astype(np.float32))
