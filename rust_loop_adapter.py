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
    if mm in ("altaz-side", "altaz_side", "altazside"):
        return "altaz_side"
    return "altaz"


class RustCoreLoopAdapter:
    """Runs skytracker_core.CoreLoop and bridges it to the app state."""

    # Loop-liveness watchdog: if the Rust loop's cycle_count stops advancing
    # for this long (or the loop reports itself dead), stop the mount from
    # Python, tear the loop down, and retry after WATCHDOG_RETRY_SEC (or
    # immediately on a telescope reconnect).
    WATCHDOG_STALL_SEC = 2.0
    WATCHDOG_RETRY_SEC = 5.0

    def __init__(self, joystick_mode_state, config_state, target_hz=15):
        self.state = joystick_mode_state
        self.config_state = config_state
        self.target_hz = float(target_hz)
        self.loop = None
        self._stop = threading.Event()
        self._thread = None
        self._last_cycle_count = -1
        self._last_cycle_advance = time.monotonic()
        self._loop_retry_at = 0.0        # monotonic time before which we don't rebuild
        self._warned_no_liveness = False
        self._last_pushed_frame_seq = None
        self._park_state = None
        self._reported_pump_errors = set()   # one UI report per distinct error

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
                # Surface each distinct failure to the UI once — a repeating
                # pump error otherwise scrolls by in the terminal while the
                # operator only sees the symptom (e.g. HANDOFF never
                # detecting because the camera push kept failing).
                msg = f"{type(e).__name__}: {e}"
                if msg not in self._reported_pump_errors:
                    self._reported_pump_errors.add(msg)
                    cb = self.state.update_status_callback
                    if cb:
                        cb(f"Rust loop pump error: {msg}")
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
        if not connected:
            # Reconnecting is the operator's explicit "retry now" for a
            # watchdog-tripped loop (see _watchdog).
            self._loop_retry_at = 0.0
        if (self.loop is None and connected
                and time.monotonic() >= self._loop_retry_at):
            self.loop = rc.CoreLoop.wrap_mount(st.telescope_controller, self.target_hz)
            self._last_cycle_count = -1
            self._last_cycle_advance = time.monotonic()
            print("RustCoreLoopAdapter: bridged Rust loop to telescope controller")
        elif self.loop is not None and not connected:
            self.loop.stop()
            self.loop = None
        return self.loop is not None

    def _watchdog(self, snap):
        """Detect a dead/stalled loop thread and fail safe.

        The loop increments snapshot['cycle_count'] every cycle and sets
        'loop_dead' when its thread exits (including a contained panic). A
        snapshot whose count stops advancing is stale no matter what 'fresh'
        says, so: stop the mount from Python (the controller lock serializes
        this with any remaining loop traffic), tear the loop down, and retry
        after WATCHDOG_RETRY_SEC (a reconnect retries immediately). Snapshots
        without a cycle_count (an old wheel) disable stall detection instead
        of tripping it.

        Returns True when the loop was torn down (pump must not keep using it).
        """
        now = time.monotonic()
        if "cycle_count" not in snap:
            # An installed skytracker_core wheel that predates the liveness
            # API never reports cycle_count. That is NOT a stall: without
            # this guard the "never advancing" count tripped the watchdog
            # exactly WATCHDOG_STALL_SEC after every connect (the loop
            # tracked for ~2 s, was torn down, and stayed dead until a
            # reconnect). Disable stall detection and tell the operator to
            # rebuild once.
            if not self._warned_no_liveness:
                self._warned_no_liveness = True
                msg = ("RustCoreLoopAdapter: installed skytracker_core wheel "
                       "predates the liveness API - loop-stall watchdog "
                       "disabled. Rebuild it: maturin build --release -m "
                       "rust/skytracker_core/Cargo.toml && pip install "
                       "--force-reinstall rust/skytracker_core/target/wheels/"
                       "skytracker_core-*.whl")
                print(msg)
                if self.state.update_status_callback:
                    self.state.update_status_callback(
                        "Rust loop: OLD skytracker_core wheel - rebuild it "
                        "(see console); stall watchdog disabled")
            return False

        count = snap["cycle_count"]
        if count != self._last_cycle_count:
            self._last_cycle_count = count
            self._last_cycle_advance = now
        dead = bool(snap.get("loop_dead"))
        stalled = (now - self._last_cycle_advance) > self.WATCHDOG_STALL_SEC
        if not (dead or stalled):
            return False

        reason = "loop thread died" if dead else (
            f"loop stalled >{self.WATCHDOG_STALL_SEC:.0f}s")
        msg = (f"RustCoreLoopAdapter: {reason} - stopping mount; retrying in "
               f"{self.WATCHDOG_RETRY_SEC:.0f}s (reconnect telescope to retry now)")
        print(msg)
        st = self.state
        if st.update_status_callback:
            st.update_status_callback(msg)
        try:
            controller = st.telescope_controller
            if controller is not None:
                from lib.auxstar import Targets
                controller.hc_slew_fixed(Targets.AZM, 0)
                controller.hc_slew_fixed(Targets.ALT, 0)
        except Exception as e:
            print(f"RustCoreLoopAdapter: python-side safe stop failed: {e}")
        try:
            self.loop.stop()
        except Exception:
            pass
        self.loop = None
        st.position_fresh = False
        self._loop_retry_at = now + self.WATCHDOG_RETRY_SEC
        return True

    def _pump(self):
        if not self._ensure_loop():
            return
        st = self.state
        cfg = self.config_state
        loop = self.loop

        # 1) Read snapshot back into app state for display + apply loop-originated
        #    mode transitions (hotspot loss/limit) and status messages.
        snap = loop.snapshot()
        if self._watchdog(snap):
            return
        if snap.get("fresh"):
            self._read_back(st, snap)
        req = snap.get("requested_mode")
        if req:
            st.tracking_mode = _STR_TO_MODE.get(req, st.tracking_mode)
        if st.update_status_callback:
            for msg in loop.drain_status():
                st.update_status_callback(msg)

        # Launch override: a launched rocket is the sole target. Force PROGRAM and
        # drive the launch setpoint regardless of the operator's current mode or
        # any satellite selection -- the Rust loop otherwise only knows satellite
        # setpoints. Mirrors JoystickModeState.tracking_control()'s launch override
        # (which the Rust path bypasses entirely).
        vis = st.tracking_vis_state
        launch_active = bool(
            vis and getattr(vis, "selected_launch", None)
            and getattr(vis, "launch_launched", False)
        )
        if launch_active:
            if getattr(vis, "selected_satellite", None) is not None:
                vis.selected_satellite = None
            st.tracking_mode = TrackingMode.PROGRAM

        # 1b) Focus motor. The Rust loop doesn't own the focus axis, so drive it
        #     directly off the Python controller (its internal lock serializes
        #     these calls with the loop's own traffic). Mirrors the Python loop,
        #     where focus is commanded in tracking_control() and read back in the
        #     poll cycle -- neither of which runs on the Rust path.
        joy = st.joysticks.get(st.connected_joystick) if st.connected_joystick is not None else None
        if joy is not None:
            st._handle_focus_control(joy)
        st._poll_focus_position()

        # 2) Push connection/stop/mode + static config.
        loop.set_connected(bool(st.telescope_connected))
        loop.set_stopped(bool(st.stopped))

        # Park request (mirrors tracking_control's control-thread park). STOP
        # cancels it; while parking, hold the loop in standby so the tracking
        # modes don't fight the goto, and drive/converge from here.
        if st.stopped and getattr(st, "park_requested", False):
            st.park_requested = False
            self._park_state = None
        if getattr(st, "park_requested", False):
            loop.set_mode("standby")
            self._push_static(cfg)
            self._service_park(st, cfg)
            return
        self._park_state = None

        loop.set_mode(_MODE_TO_STR.get(st.tracking_mode, "standby"))
        self._push_static(cfg)

        # 3) Per-mode dynamic inputs.
        mode = st.tracking_mode
        if mode == TrackingMode.RATE_CONTROL:
            self._push_rate(st)
        elif mode == TrackingMode.PROGRAM:
            if launch_active:
                self._push_launch_setpoint(st, cfg)
            else:
                self._push_program_setpoint(st)
        elif mode == TrackingMode.HANDOFF:
            # HANDOFF needs BOTH the program setpoint (it keeps program-tracking)
            # and camera frames + hotspot params (it detects in parallel).
            self._push_program_setpoint(st)
            self._push_hotspot(st, cfg)
        elif mode == TrackingMode.HOTSPOT:
            # HOTSPOT also gets the program setpoint when a target is selected:
            # its trajectory rates feed the optical loop's feed-forward (the
            # correction rides on the target's own motion) and give the star
            # filter the expected detection rate. Tracking bare (no target),
            # this clears the setpoint and the loop runs correction-only.
            self._push_program_setpoint(st)
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
        # HANDOFF progress for the PID-diagnostics panel. The Python loop
        # writes handoff_status from handoff_track(), which the Rust path
        # bypasses entirely -- without this the panel shows a static "armed"
        # and the operator can't tell whether detection is even firing.
        if st.tracking_mode == TrackingMode.HANDOFF:
            count = snap.get("handoff_detection_count")
            if count is None:
                st.handoff_status = "armed (rebuild skytracker_core for progress)"
            elif st.hotspot_status == "star-reject":
                st.handoff_status = "star rejected (rate gate)"
            elif st.hotspot_status == "detecting" or count > 0:
                need = max(1, int(getattr(self.config_state,
                                          "handoff_min_frames", 5) or 5))
                st.handoff_status = f"detecting {count}/{need}"
            else:
                st.handoff_status = "program track (no detection)"
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
        # Side-mount tip side (older wheels predate the setter).
        if hasattr(loop, "set_altaz_side_flip"):
            loop.set_altaz_side_flip(bool(getattr(cfg, "altaz_side_flip", False)))
        loop.set_continuous_rate(
            bool(getattr(cfg, "continuous_rate_tracking", False)),
            float(getattr(cfg, "guide_rate_max_dps", 5.0)),
        )
        loop.set_handoff_min_frames(int(getattr(cfg, "handoff_min_frames", 5) or 5))
        loop.set_output_filter(
            float(getattr(cfg, "pid_output_filter_tau_sec", 0.0) or 0.0))
        try:
            loop.set_alignment(
                float(cfg.alignment_azimuth_str), float(cfg.alignment_elevation_str)
            )
        except (ValueError, AttributeError):
            pass

    def _service_park(self, st, cfg):
        """Drive both axes to the configured offsets via the loop's goto queue,
        with the same wrap-aware convergence + timeout as
        JoystickModeState._service_park (which the Rust path bypasses)."""
        now = time.monotonic()
        try:
            target_azm = float(getattr(cfg, "azm_offset_str", 0.0) or 0.0)
            target_alt = float(getattr(cfg, "alt_offset_str", 0.0) or 0.0)
        except (TypeError, ValueError):
            target_azm = target_alt = 0.0

        if self._park_state is None:
            self._park_state = {"start": now, "last_cmd": 0.0}
            if st.update_status_callback:
                st.update_status_callback(
                    f"Parking to AZM {target_azm:.1f}° / ALT {target_alt:.1f}°...")
        ps = self._park_state

        timeout = float(getattr(st, "PARK_TIMEOUT_SEC", 90.0))
        tol = float(getattr(st, "PARK_TOLERANCE_DEG", 1.0))
        reissue = float(getattr(st, "PARK_REISSUE_SEC", 1.0))

        if now - ps["start"] > timeout:
            st.park_requested = False
            self._park_state = None
            if st.update_status_callback:
                st.update_status_callback(
                    f"Park TIMED OUT after {timeout:.0f}s - check the mount")
            return

        err_azm = (target_azm - st.current_azm_raw + 180.0) % 360.0 - 180.0
        err_alt = (target_alt - st.current_alt_raw + 180.0) % 360.0 - 180.0
        if abs(err_azm) <= tol and abs(err_alt) <= tol:
            st.park_requested = False
            self._park_state = None
            if st.update_status_callback:
                st.update_status_callback("Park complete")
            return

        if now - ps["last_cmd"] >= reissue:
            self.loop.submit_goto(target_azm, target_alt)
            ps["last_cmd"] = now

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
        """Satellite/aircraft tracking setpoint. (Launch tracking is handled
        separately by _push_launch_setpoint, which takes priority in _pump when a
        launch is active.) A below-horizon target clears the setpoint so the loop
        holds. Uses the shared active_program_trajectory resolver so the Rust and
        Python loops agree on the target (selected satellite first, then aircraft)."""
        vis = st.tracking_vis_state
        from joystick_controller import active_program_trajectory
        target_traj, _kind, _key = active_program_trajectory(vis)
        if vis is None or target_traj is None:
            self.loop.clear_setpoint()
            return
        px, py, target_alt, dist, target_az, az_rate, el_rate = (
            interpolate_position_data_and_rates(target_traj, vis.current_tt)
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

    def _push_launch_setpoint(self, st, cfg):
        """Push the active launch trajectory as the program setpoint, so the Rust
        loop slews to and tracks the rocket. Mirrors
        JoystickModeState._program_track_launch:

          * Below the elevation mask the rocket isn't trackable yet, so aim at the
            horizon point nearest the rocket -- its current azimuth at the mask
            elevation -- and follow that azimuth so the mount is pre-positioned and
            ready when it rises.
          * Above the mask, follow the rocket directly with feed-forward + bias.
        """
        vis = st.tracking_vis_state
        name = getattr(vis, "selected_launch", None) if vis else None
        traj = vis.launch_trajectories.get(name) if (vis and name and getattr(vis, "launch_trajectories", None)) else None
        if not traj:
            self.loop.clear_setpoint()
            return

        # launched=True applies the relative-time indexing (file T-0 + elapsed).
        px, py, alt, dist, az, az_rate, el_rate = interpolate_position_data_and_rates(
            traj, vis.current_tt,
            getattr(vis, "launch_start_time", 0) or 0,
            bool(getattr(vis, "launch_launched", False)),
        )
        if px is None or az is None or alt is None:
            self.loop.clear_setpoint()
            return

        try:
            mask = float(getattr(cfg, "elevation_mask_str", None) or getattr(cfg, "elevation_mask", 10.0) or 10.0)
        except (TypeError, ValueError):
            mask = 10.0

        az_rate_f = float(az_rate) if az_rate is not None else 0.0
        el_rate_f = float(el_rate) if el_rate is not None else 0.0

        below = float(alt) <= mask
        # Below the mask: hold elevation at the mask (horizon point), still follow
        # the rocket's azimuth; above it: track the rocket's actual elevation.
        set_el = mask if below else float(alt)
        set_el_rate = 0.0 if below else el_rate_f

        # Cache target sky-velocity / elevation for the camera bias-direction axes.
        st.target_az_rate = az_rate_f
        st.target_el_rate = set_el_rate
        st.target_el_deg = set_el

        self.loop.set_ff_enabled(
            bool(st.feed_forward_azm_enabled), bool(st.feed_forward_alt_enabled)
        )
        biased_az, biased_el = st._apply_bias_to_target(
            float(az), set_el, az_rate_f, set_el_rate
        )
        # Pointing-model pre-correction (Rust runs the plain geometric transform).
        from control import apply_pointing_model
        biased_az, biased_el = apply_pointing_model(self.config_state, biased_az, biased_el)
        self.loop.set_setpoint(biased_az, biased_el, az_rate_f, set_el_rate)

    def _push_hotspot(self, st, cfg):
        # The instance, NOT the module: get_camera is a method on the singleton
        # (module-level `import camera_manager` made every HANDOFF/HOTSPOT pump
        # die with AttributeError, silently starving the loop of frames).
        from camera_manager import camera_manager
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
            float(getattr(cfg, "hotspot_max_rate_dps", 2.0) or 2.0),
            bool(getattr(cfg, "hotspot_star_filter_enabled", True)),
            float(getattr(cfg, "hotspot_rate_gate_dps", 0.15) or 0.15),
        )
        camera = camera_manager.get_camera(cam_index)
        if camera is not None and getattr(camera, "thread", None) is not None:
            raw = camera.thread.get_latest_raw()
            # Only push frames the camera hasn't already delivered: push_frame
            # assigns a new seq per call, so re-pushing the same raw frame would
            # defeat the loop's stale-frame gate (and costs a full-frame FFI
            # copy every pump cycle for nothing).
            seq = getattr(camera.thread, "latest_raw_seq", None)
            if raw is not None and (seq is None or seq != self._last_pushed_frame_seq):
                self.loop.push_frame(hs.to_intensity(raw).astype(np.float32))
                self._last_pushed_frame_seq = seq
