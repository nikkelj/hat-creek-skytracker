"""
Dedicated real-time mount control thread.

Historically the telescope read -> PID -> command cycle ran inline inside the
pygame render/event loop. That coupled mount commanding to rendering jitter and,
worse, to blocking serial I/O: a single slow/timed-out read could freeze the
whole loop for seconds while an axis kept slewing ("keep moving until stop"
semantics), causing the mount to wander off target.

This thread owns ALL serial traffic to the mount and runs the control cycle at a
fixed cadence, independent of the UI. The UI thread only reads shared state
(position, errors, display strings) populated here. Safety is enforced by:
  * a short serial timeout (set in lib.auxstar) so cycles cannot stall, and
  * a watchdog that stops motion on any cycle fault or on shutdown.
"""

import threading
import time

from lib.auxstar import Targets, f2dms


class MountControlThread(threading.Thread):
    """Fixed-rate control loop that owns the serial link to the mount."""

    def __init__(self, joystick_mode_state, config_state, target_hz=15,
                 max_consecutive_faults=3):
        super().__init__(daemon=True, name="MountControlThread")
        self.state = joystick_mode_state
        self.config_state = config_state
        self.period = 1.0 / float(target_hz)
        self.max_consecutive_faults = max_consecutive_faults

        self._stop_event = threading.Event()
        self._consecutive_faults = 0

        # Instrumentation (read by the UI for diagnostics if desired)
        self.actual_hz = 0.0
        self.last_cycle_ms = 0.0
        self.last_poll_ms = 0.0
        self._cycle_count = 0
        self._rate_window_start = time.time()

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------ loop
    def run(self):
        print(f"MountControlThread starting ({1.0 / self.period:.0f} Hz target)...")
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()
            try:
                self._run_cycle()
                self._consecutive_faults = 0
            except Exception as e:
                # Any fault in a control cycle: never leave the mount running
                # open-loop. After a few consecutive faults, stop motion.
                self._consecutive_faults += 1
                if self._consecutive_faults >= self.max_consecutive_faults:
                    print(f"MountControlThread: {self._consecutive_faults} "
                          f"consecutive faults ({e}) - stopping motion")
                    self._safe_stop_motion()

            elapsed = time.perf_counter() - cycle_start
            self.last_cycle_ms = elapsed * 1000.0
            self._update_rate_stats()

            sleep_time = self.period - elapsed
            if sleep_time > 0:
                # Interruptible sleep so stop() takes effect promptly.
                self._stop_event.wait(sleep_time)

        # On shutdown, guarantee the mount is not left slewing.
        self._safe_stop_motion()
        print("MountControlThread stopped.")

    def _run_cycle(self):
        state = self.state
        if not state.telescope_connected:
            return
        controller = state.telescope_controller
        if controller is None:
            return

        # 1) Single position poll per cycle, shared by control and display.
        self._poll_position(controller)

        # 2) Run the active tracking mode (PID + rate commands). Handlers read
        #    the cached position set above rather than hitting the wire again.
        state.tracking_control()

        # 3) Post-cycle services: per-mode PID gain profile swap first (so a
        #    mode change lands the right gains before the tuner sees them),
        #    then feed the auto-tuner this cycle's position errors.
        #    getattr: test rigs drive this thread with minimal fake states.
        for name in ("service_gain_profiles", "service_autotune"):
            service = getattr(state, name, None)
            if service is not None:
                service()

    # -------------------------------------------------------------- polling
    def _poll_position(self, controller):
        poll_start = time.perf_counter()
        state = self.state
        cfg = self.config_state

        azm_raw = controller.hc_get_position(Targets.AZM) * 360.0
        alt_raw = controller.hc_get_position(Targets.ALT) * 360.0

        # Focus encoder read-back for the joystick-mode display. The focus motor
        # is commanded from the triggers inside tracking_control() below; this
        # only reads its position back. (Commanding lives there so this poll stays
        # a pure read.)
        state._poll_focus_position()

        offset_azm = float(getattr(cfg, 'azm_offset_str', 0.0) or 0.0)
        offset_alt = float(getattr(cfg, 'alt_offset_str', 0.0) or 0.0)

        # Cache raw and offset-applied positions so tracking handlers reuse this
        # single read instead of re-querying the serial port.
        state.current_azm_raw = azm_raw
        state.current_alt_raw = alt_raw
        state.current_azm = azm_raw - offset_azm
        state.current_alt = alt_raw - offset_alt
        state.position_fresh = True

        # Display strings (degrees -> DMS).
        try:
            azm_d, azm_m, azm_s = f2dms(state.current_azm / 360.0)
            alt_d, alt_m, alt_s = f2dms(state.current_alt / 360.0)
            state.azm_display_str = f"{azm_d:3d}°{azm_m:02d}'{azm_s:04.1f}\""
            state.alt_display_str = f"{alt_d:3d}°{alt_m:02d}'{alt_s:04.1f}\""
        except Exception:
            pass

        # Sync to visualization state for the polar plot / overlays.
        if state.tracking_vis_state is not None:
            state.tracking_vis_state.telescope_azimuth = state.current_azm
            state.tracking_vis_state.telescope_altitude = state.current_alt

        self.last_poll_ms = (time.perf_counter() - poll_start) * 1000.0

    # --------------------------------------------------------------- safety
    def _safe_stop_motion(self, retries=3):
        """Stop both axes, retrying briefly. Returns True when the stop was
        acknowledged. Also invalidates the rate-command dedup cache: these
        stops bypass _send_rate_command, so without this the tracking path
        could skip its next (differing) send for up to the keepalive window.
        A stop that keeps failing is the one fault the operator
        MUST hear about -- the mount may still be slewing on a dead link -- so
        it is surfaced through the status callback instead of swallowed.
        """
        state = self.state
        controller = state.telescope_controller
        getattr(state, '_rate_cmd_cache', {}).clear()
        if controller is None or not state.telescope_connected:
            return True
        last_err = None
        for _ in range(max(1, retries)):
            try:
                controller.hc_slew_fixed(Targets.AZM, 0)
                controller.hc_slew_fixed(Targets.ALT, 0)
                return True
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        msg = (f"MountControlThread: FAILED to stop motion after {retries} attempts "
               f"({last_err}) - mount may still be slewing! Check serial link/power.")
        print(msg)
        try:
            if getattr(state, 'update_status_callback', None):
                state.update_status_callback(msg)
        except Exception:
            pass
        return False

    def _update_rate_stats(self):
        self._cycle_count += 1
        now = time.time()
        elapsed = now - self._rate_window_start
        if elapsed >= 1.0:
            self.actual_hz = self._cycle_count / elapsed
            self._cycle_count = 0
            self._rate_window_start = now
            print(f"MountControl: {self.actual_hz:.1f} Hz | "
                  f"cycle {self.last_cycle_ms:.1f}ms | poll {self.last_poll_ms:.1f}ms")
