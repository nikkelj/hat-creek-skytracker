import pygame
import serial
import serial.tools.list_ports
import math
import time
import threading
from collections import deque
from datetime import datetime, timezone
from skyfield.api import load
from enum import Enum

# ==============================================================================
# TRACKING MODES ENUM
# ==============================================================================

class TrackingMode(Enum):
    STANDBY = 1      # Poll telescope position only, bypass rate control
    RATE_CONTROL = 2 # Current rate control functionality
    PROGRAM = 3      # Program track mode (stub)
    HANDOFF = 4      # Handoff mode (stub)
    HOTSPOT = 5      # Hotspot mode (stub)
    MTI = 6          # MTI mode (stub)

# Import existing components
from lib.auxstar import NexstarHandController, RATES, Targets
from tracking_visuals import PolarPlotMode
from camera_manager import camera_manager, update_camera_frames_from_buffers
from camera_manager import render_sensor_calibration
from camera_manager import apply_gamma_correction, roi_sizes, roi_label_texts
from utils import draw_button

# Import PID controller and helper functions
from control import create_pid_controllers, compute_mount_position_error
from trajectory import interpolate_position_data_and_rates
from hotspot import detect_hotspot, pixel_offset_to_angles

# PS4 Controller Button Labels (zero-indexed)
BUTTON_LABELS = ["X", "O", "[]", "/\\", "Sh", "PS5", "Op", "LS", "RS", "L1", "R1", "D/\\", "D\\/", "D<", "D>", "Pad"]

# Functionality currently assigned to each joystick button, keyed by the pygame
# button index (i.e. what process_joystick_events() acts on for `event.button`).
# Buttons with no mapping are shown with a dash so the layout still documents
# every physical button. The D-pad (11-14) and Op (6) get frame-dependent labels
# at render time via button_function_map().
BUTTON_FUNCTIONS = {
    0:  "Capture",
    1:  "Stop",
    2:  "Tare axes",
    3:  "Park",
    4:  "Bias mode",
    6:  "Mount mode",
    8:  "Track mode +",
    9:  "Feed-fwd",
    15: "Track mode -",
}


def button_function_map(joystick_state=None):
    """Button index -> function label, including the bias-frame-dependent labels
    for the D-pad (bias adjust) and Op (bias mode). The horizontal D-pad axis is
    Az or In-track and the vertical axis is El or Cross-track, depending on the
    active bias frame; the labels flip with it so the controller always documents
    what the buttons currently do."""
    fmap = dict(BUTTON_FUNCTIONS)
    frame = getattr(joystick_state, 'bias_frame', 'azel') if joystick_state else 'azel'
    res = getattr(joystick_state, 'bias_resolution', 'coarse') if joystick_state else 'coarse'
    h, v = ("InTk", "XTk") if frame == 'alongcross' else ("Az", "El")
    fmap[11] = f"+{v} bias"   # D-pad up
    fmap[12] = f"-{v} bias"   # D-pad down
    fmap[13] = f"-{h} bias"   # D-pad left
    fmap[14] = f"+{h} bias"   # D-pad right
    fmap[4] = f"Bias {res[:4]}/{'IX' if frame == 'alongcross' else 'AzEl'}"
    return fmap


# Lead-time slider range (seconds) for the joystick-loop PID pane.
JL_LEAD_MIN = 0.0
JL_LEAD_MAX = 0.5


def _draw_disabled_scrim(display, rect):
    """Draw a translucent grey scrim over a panel to show it is visible but
    inactive (not interactable in the current mode/state)."""
    scrim = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    scrim.fill((25, 25, 25, 165))
    display.menu_screen.blit(scrim, rect.topleft)


def joystick_panel_layout(display):
    """Geometry for the control panes that hug the RIGHT edge of the joystick
    mode's upper-left quadrant, stacked bottom-up: PID Gain (bottom), Bias above
    it, and PID Diagnostics above that. All right-aligned to the quadrant
    divider, which keeps the center of the quadrant clear (reserved for the
    forthcoming navball)."""
    qx, qy = display.sub_x, display.sub_y
    qw, qh = display.sub_width // 2, display.sub_height // 2

    pid_w, pid_h = 250, 215
    pid_x = qx + qw - pid_w - 12
    pid_y = qy + qh - pid_h - 12

    bias_w, bias_h = 250, 150
    bias_x = qx + qw - bias_w - 12
    bias_y = pid_y - bias_h - 12

    diag_w, diag_h = 250, 112
    diag_x = qx + qw - diag_w - 12
    diag_y = bias_y - diag_h - 12

    # Plate-solve pane above the diagnostics pane, clamped so it never rides up over
    # the connect/port/status rows at the top of the quadrant.
    plate_w, plate_h = 250, 92
    plate_x = qx + qw - plate_w - 12
    plate_y = max(qy + 104, diag_y - plate_h - 12)

    return {
        'pid': pygame.Rect(pid_x, pid_y, pid_w, pid_h),
        'bias': pygame.Rect(bias_x, bias_y, bias_w, bias_h),
        'diag': pygame.Rect(diag_x, diag_y, diag_w, diag_h),
        'plate': pygame.Rect(plate_x, plate_y, plate_w, plate_h),
    }


def joystick_center_layout(display):
    """Geometry for the previously-unused center column of the joystick mode's
    upper-left quadrant: the navball up top and the two PID-tuning strip charts
    (tracking rate, position error) stacked below it. The column sits between the
    left button/axes status block and the right-hand PID/bias/diag panes.
    `valid` is False when the quadrant is too small to host them."""
    qx, qy = display.sub_x, display.sub_y
    qw, qh = display.sub_width // 2, display.sub_height // 2

    left = qx + 280              # clear of the button/axes status block
    right = qx + qw - 250 - 14   # clear of the right-hand panes (250 wide + margin)
    top = qy + 104               # below the connect/port/baud/status rows
    bottom = qy + qh - 12
    cw = right - left
    ch = bottom - top
    valid = cw >= 120 and ch >= 220

    nb_side = max(60, min(cw, int(ch * 0.42), 150))
    navball = pygame.Rect(left + (cw - nb_side) // 2, top + 6, nb_side, nb_side)

    charts_top = navball.bottom + 24
    chart_h = max(40, (bottom - charts_top - 8) // 2)
    chart_rate = pygame.Rect(left, charts_top, cw, chart_h)
    chart_err = pygame.Rect(left, charts_top + chart_h + 8, cw, chart_h)

    return {
        'valid': valid,
        'navball': navball,
        'chart_rate': chart_rate,
        'chart_err': chart_err,
        'center': pygame.Rect(left, top, cw, ch),
    }

# ==============================================================================
# JOYSTICK MODE STATE CLASS
# ==============================================================================

def axis_to_rate(axis_value):
    """Map a (tared) joystick axis value to a discrete rate (-9..=9).

    0 = stop; |rate| ramps from 1 at the deadband edge to 9 by ~0.9 deflection.
    Extracted from the RATE_CONTROL handler so the Rust-core-loop adapter pushes
    exactly the same mapping (single source of truth).
    """
    axis_value = max(-1.0, min(1.0, axis_value))
    if abs(axis_value) < 0.01:  # deadband
        return 0
    normalized = min((abs(axis_value) - 0.01) / (0.9 - 0.01), 1.0)
    rate = max(1, min(9, int(math.ceil(normalized * 9))))
    return rate * (1 if axis_value > 0 else -1)


class JoystickModeState:
    """
    Encapsulates all state for joystick mode, following state-direct object mutation pattern.
    Similar to TrackingVisState, this manages all joystick mode specific state.
    """

    def __init__(self, tracking_vis_state=None, config_state=None, update_status_callback=None):
        # Initialize Pygame joystick subsystem
        pygame.joystick.init()
        print(f"Pygame joystick initialized: {pygame.joystick.get_count()} joysticks detected")

        # Store global references for PROGRAM tracking mode
        self.tracking_vis_state = tracking_vis_state
        self.config_state = config_state
        self.update_status_callback = update_status_callback

        # Initialize mount mode from config
        self.mount_mode = self.config_state.mount_mode if self.config_state else "AltAz"

        # Joystick state
        self.joysticks = {}  # Dict of active joysticks
        self.connected_joystick = None  # Currently active joystick
        self.joystick_tare = {}  # Tare values for deadzone calibration
        self.stopped = False  # Stop button state

        # Tracking mode state
        self.tracking_mode = TrackingMode.STANDBY  # Default to standby mode (user preference)
        # Tracks the launch_launched edge so pressing "launch" can auto-engage
        # PROGRAM tracking of the rocket (overriding any selected satellite target).
        self._prev_launch_launched = False

        # Telescope connection state
        self.telescope_connected = False
        self.telescope_controller = None
        self.selected_port = None
        self.available_ports = []

        # Hardware simulator (set by main.py); when sim is enabled the connect
        # calls below hand back the sim mount instead of a real serial controller.
        self.hardware_sim = None

        # Plate-solving (tetra3). Runs in a background worker at ~1 Hz, solving the
        # latest frame from the configured camera and deriving the instantaneous
        # alignment. last_solve holds the most recent result dict for the UI/overlay.
        self.plate_solver = None
        self.plate_solve_thread = None
        self.plate_solve_running = False
        self.last_solve = None        # dict: result, az, el, align_az, t, mount_azm/alt
        self.plate_solve_status = ""
        self.ps_button_rects = []

        # UI state
        self.connect_button_hover = False
        self.disconnect_button_hover = False
        self.port_dropdown_open = False
        self.port_options_rects = []

        # Polar graph integration
        self.ts = None  # Loaded timescope for polar plot
        self.current_tt = None  # Current time for polar plot

        # Capture integration
        self.capture_active = False
        self.capture_progress = 0.0
        self.capture_status = ""
        self.capture_button_rect = None

        # Telescope position tracking for display.
        # current_azm/current_alt are offset-applied degrees; the *_raw values
        # are pre-offset. These are populated once per cycle by the
        # MountControlThread and reused by the tracking handlers below so that
        # each control cycle reads the mount position exactly once.
        self.current_azm = 0.0
        self.current_alt = 0.0
        self.current_azm_raw = 0.0
        self.current_alt_raw = 0.0
        self.position_fresh = False
        self.azm_display_str = "--"
        self.alt_display_str = "--"

        # One-shot guard so STANDBY sends a single stop on entry rather than
        # re-commanding zero every cycle (and never leaves a residual slew).
        self._standby_motion_stopped = False

        # Tracks the last dispatched mode so we can run per-mode entry logic.
        self._prev_dispatch_mode = None

        # HOTSPOT (closed-loop optical) tracker state
        self.hotspot_gate_center = None      # (cx, cy) in image px once locked
        self.hotspot_acquired = False
        self.hotspot_miss_count = 0
        self.hotspot_last_detection_time = 0.0
        self.hotspot_entry_time = 0.0        # when HOTSPOT was engaged (acq. grace)
        self.hotspot_snr = 0.0               # diagnostics
        self.hotspot_centroid = None
        self.hotspot_status = ""

        # PID controllers for PROGRAM track mode
        self.azm_pid = None
        self.alt_pid = None
        self.pid_last_update = 0.0

        # Feed-forward and bias control state. Honor the saved config flags so
        # feed-forward (which supplies the target's trajectory rate and removes
        # the velocity-lag that the integrator would otherwise wind up slowly)
        # can be on by default; the FF AZ/EL buttons still toggle it live.
        self.feed_forward_azm_enabled = bool(getattr(config_state, 'feed_forward_azm_enabled', False))
        self.feed_forward_alt_enabled = bool(getattr(config_state, 'feed_forward_alt_enabled', False))
        self.bias_azm_deg = 0.0          # on-sky cross-elevation (Az) bias
        self.bias_alt_deg = 0.0          # elevation (El) bias
        self.bias_intrack_deg = 0.0      # along-track (target velocity dir) bias
        self.bias_crosstrack_deg = 0.0   # cross-track (perpendicular) bias
        # Bias adjust mode cycled by the Op button: resolution (coarse/fine) x
        # frame (azel/alongcross). bias_control_mode is kept as a compat alias of
        # the resolution for older callers.
        self.bias_resolution = "coarse"
        self.bias_frame = "azel"
        self.bias_control_mode = "coarse"

        # Focus motor: R2 trigger drives it forward, L2 backward, at a rate
        # proportional to trigger deflection (sent via Targets.FOCUS / MC_MOVE).
        # Triggers are latched as "seen" once they report their released value so
        # an untouched-axis reading of 0 cannot command spurious focus motion.
        self.focus_axis_forward = 5      # R2 trigger axis
        self.focus_axis_backward = 4     # L2 trigger axis
        self._focus_last_rate = 0
        self._focus_trigger_seen = {4: False, 5: False}
        self.focus_rate = 0              # last commanded rate (for display)
        # Focus encoder read-back for display: raw 24-bit position counts
        # (hc_get_position returns a fraction of a revolution; *2**24 is the
        # underlying MC_GET_POSITION count). Refreshed each control cycle.
        self.current_focus = 0

        # HANDOFF mode: run PROGRAM track while the hotspot detector evaluates the
        # frame in parallel; auto-engage HOTSPOT after N consecutive detections.
        self.handoff_detection_count = 0
        self.handoff_status = ""

        # Rolling diagnostics history for the PID tuning strip charts (sampled
        # each control cycle while actively tracking).
        self.diag_history_len = 600
        self.az_rate_history = deque(maxlen=self.diag_history_len)
        self.el_rate_history = deque(maxlen=self.diag_history_len)
        self.az_err_history = deque(maxlen=self.diag_history_len)
        self.el_err_history = deque(maxlen=self.diag_history_len)
        # Persistent per-chart vertical-axis scale for prompt auto-ranging.
        self._chart_scale = {}

        # PID diagnostic state
        self.azm_position_error = 0.0
        self.alt_position_error = 0.0
        self.azm_rate_error = 0.0
        self.alt_rate_error = 0.0
        self.azm_target_rate = 0.0
        self.alt_target_rate = 0.0
        self.azm_pid_output = 0.0
        self.alt_pid_output = 0.0

        # Current target sky-velocity (deg/s) and elevation, cached each cycle so
        # the camera overlays can draw the along/cross-track bias direction axes.
        # Populated by program_track (Python loop) and the Rust adapter setpoint
        # push, so the overlays work under either control loop.
        self.target_az_rate = 0.0
        self.target_el_rate = 0.0
        self.target_el_deg = 0.0

    def reset_tare(self):
        """Reset tare values for all connected joysticks"""
        self.joystick_tare = {}
        for joy in self.joysticks.values():
            self.joystick_tare[joy.get_instance_id()] = [0] * joy.get_numaxes()

    def tare_current_joystick(self):
        """Tare the currently connected joystick"""
        if self.connected_joystick is not None:
            joy = self.joysticks[self.connected_joystick]
            tare_values = []
            for i in range(joy.get_numaxes()):
                axis_value = joy.get_axis(i)
                tare_values.append(axis_value)
                print(f"Tared Axis {i} value: {axis_value:>6.3f}")
            self.joystick_tare[self.connected_joystick] = tare_values

    def get_available_serial_ports(self):
        """Get list of available serial ports"""
        self.available_ports = []
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.available_ports.append({
                'device': port.device,
                'description': port.description,
                'name': port.name or port.device
            })

        # If no port selected, pick first one available
        if not self.selected_port and self.available_ports:
            self.selected_port = self.available_ports[0]['device']

    def connect_telescope(self):
        """Connect to telescope via serial port (or the sim mount in sim mode)."""
        # Simulation: hand back the sim mount, no serial port required.
        if self.hardware_sim is not None and self.hardware_sim.sim_enabled():
            self.telescope_controller = self.hardware_sim.mount
            self.telescope_connected = True
            print("Connected to SIMULATED telescope")
            return True

        if not self.selected_port:
            return False

        try:
            self.telescope_controller = NexstarHandController(self.selected_port)
            self.telescope_connected = True
            print(f"Connected to telescope on {self.selected_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to telescope: {e}")
            self.telescope_controller = None
            self.telescope_connected = False
            return False

    def disconnect_telescope(self):
        """Disconnect from telescope"""
        if self.telescope_controller:
            try:
                self.telescope_controller.close()
            except:
                pass
        self.telescope_controller = None
        self.telescope_connected = False
        print("Disconnected from telescope")

    def tracking_control(self):
        """Handle tracking control based on connected joystick and tracking mode"""
        if not self.telescope_connected or self.connected_joystick is None:
            return

        if self.connected_joystick not in self.joysticks:
            return

        joy = self.joysticks[self.connected_joystick]

        # Focus motor on the triggers, independent of tracking mode (zeroed when
        # stopped). Runs before the stop/mode dispatch so focusing always works.
        self._handle_focus_control(joy)

        # Universal stop check - stop movement in any mode when stopped
        if self.stopped:
            try:
                self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
            except Exception as e:
                print(f"Error sending stop commands: {e}")
            return

        # A launched rocket overrides any current target AND any tracking mode:
        # drive the launch trajectory directly, every cycle it stays active.
        # Dispatching here (instead of relying on mode==PROGRAM + the launch
        # priority inside program_track) makes "launch overrides the selected
        # satellite" unconditional -- it cannot be defeated by a stray satellite
        # selection or by whatever mode the operator last left it in. Emergency
        # STOP still wins (it returned above); toggling the launch off hands
        # control back to the normal mode dispatch.
        launch_active = bool(
            self.tracking_vis_state
            and getattr(self.tracking_vis_state, 'selected_launch', None)
            and getattr(self.tracking_vis_state, 'launch_launched', False)
        )
        if launch_active:
            # Drop any selected satellite so it neither competes as a target nor
            # lingers as the highlighted "selected" object in the sky plots.
            if getattr(self.tracking_vis_state, 'selected_satellite', None) is not None:
                self.tracking_vis_state.selected_satellite = None
            if self.tracking_mode != TrackingMode.PROGRAM:
                self.tracking_mode = TrackingMode.PROGRAM  # reflect override in the UI
                self._prev_dispatch_mode = TrackingMode.PROGRAM
                self._standby_motion_stopped = False
            if not self._prev_launch_launched:
                print("LAUNCH: rocket launched -> overriding target, tracking launch trajectory")
            self._prev_launch_launched = True
            self._program_track_launch()
            return
        self._prev_launch_launched = False

        # Reset the STANDBY one-shot stop guard whenever we are in any active
        # mode, so re-entering STANDBY will again issue a single stop.
        if self.tracking_mode != TrackingMode.STANDBY:
            self._standby_motion_stopped = False

        # Run per-mode entry logic on a mode transition.
        if self.tracking_mode != self._prev_dispatch_mode:
            if self.tracking_mode == TrackingMode.HOTSPOT:
                self._enter_hotspot_mode()
            elif self.tracking_mode == TrackingMode.HANDOFF:
                self.handoff_detection_count = 0
                self.handoff_status = "armed"
                print("HANDOFF: armed - program track + parallel hotspot detection")
            self._prev_dispatch_mode = self.tracking_mode

        # Dispatch to appropriate tracking method based on mode
        if self.tracking_mode == TrackingMode.STANDBY:
            # STANDBY: no rate control. Position polling happens in the control
            # thread. Send a single stop on entry so any residual slew from a
            # prior mode (e.g. a safety-limit trip) is halted, then stay quiet.
            if not self._standby_motion_stopped:
                try:
                    self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                    self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
                except Exception as e:
                    print(f"Error sending STANDBY stop commands: {e}")
                self._standby_motion_stopped = True
        elif self.tracking_mode == TrackingMode.RATE_CONTROL:
            # RATE_CONTROL: Current joystick control logic
            self._handle_rate_control(joy)
        elif self.tracking_mode == TrackingMode.PROGRAM:
            # PROGRAM: Stub implementation
            self.program_track()
        elif self.tracking_mode == TrackingMode.HANDOFF:
            # HANDOFF: Stub implementation
            self.handoff_track()
        elif self.tracking_mode == TrackingMode.HOTSPOT:
            # HOTSPOT: Stub implementation
            self.hotspot_track()
        elif self.tracking_mode == TrackingMode.MTI:
            # MTI: Stub implementation
            self.mti_track()

    def chart_axis_scale(self, key, values, floor):
        """Auto-range a strip chart's vertical axis (±return value). Tracks the
        peak over a RECENT window, not the whole history buffer, so the axis
        shrinks promptly when the signal collapses instead of staying zoomed out
        for the full buffer duration. Hysteresis: rescale only when the recent
        peak exceeds the current axis or drops to within 10% of it (so a settled
        signal stays put without jitter)."""
        recent_n = max(20, self.diag_history_len // 12)
        recent = values[-recent_n:] if values else []
        peak = max((abs(v) for v in recent), default=0.0)
        cur = self._chart_scale.get(key, 0.0)
        if peak > cur or peak < 0.1 * cur:
            cur = max(peak * 1.2, floor)   # 20% headroom; never below the floor
        self._chart_scale[key] = cur
        return cur

    def sample_tracking_history(self):
        """Append the current az/el rate and position-error diagnostics to the
        strip-chart history. Called once per UI frame from the chart renderer so
        it works for BOTH the Python control thread and the Rust core loop (both
        populate azm_pid_output / azm_position_error etc.); sampling inside the
        Python tracking_control would leave the charts blank under the Rust loop."""
        if self.tracking_mode not in (TrackingMode.PROGRAM, TrackingMode.HANDOFF,
                                      TrackingMode.HOTSPOT):
            return
        self.az_rate_history.append(getattr(self, 'azm_pid_output', 0.0) * 360.0)
        self.el_rate_history.append(getattr(self, 'alt_pid_output', 0.0) * 360.0)
        self.az_err_history.append(getattr(self, 'azm_position_error', 0.0))
        self.el_err_history.append(getattr(self, 'alt_position_error', 0.0))

    def _handle_focus_control(self, joy):
        """Drive the focus motor from the triggers: R2 forward, L2 backward, at a
        rate proportional to deflection. Triggers report -1 (released) .. +1
        (pressed); an untouched trigger may read 0, so each is ignored until it
        has been seen at its released value at least once. Commands are only sent
        when the integer rate changes, to avoid flooding the serial link."""
        if self.telescope_controller is None:
            return
        num_axes = joy.get_numaxes()

        def deflection(idx):
            # 0 (released) .. 1 (fully pressed); None until the trigger is "seen".
            if idx >= num_axes:
                return None
            raw = joy.get_axis(idx)
            if raw < -0.5:
                self._focus_trigger_seen[idx] = True
            if not self._focus_trigger_seen.get(idx, False):
                return None
            return max(0.0, min(1.0, (raw + 1.0) / 2.0))

        fwd = deflection(self.focus_axis_forward)
        bwd = deflection(self.focus_axis_backward)
        fwd = fwd if fwd is not None else 0.0
        bwd = bwd if bwd is not None else 0.0

        DEADBAND = 0.05
        if self.stopped:
            rate = 0
        elif fwd > DEADBAND and fwd >= bwd:
            rate = int(round(fwd * 9))       # forward = positive
        elif bwd > DEADBAND:
            rate = -int(round(bwd * 9))      # backward = negative
        else:
            rate = 0
        rate = max(-9, min(9, rate))

        if rate != self._focus_last_rate:
            try:
                self.telescope_controller.hc_slew_fixed(Targets.FOCUS, rate)
                self._focus_last_rate = rate
                self.focus_rate = rate
            except Exception as e:
                print(f"Focus control error: {e}")

    def _poll_focus_position(self):
        """Refresh the focus encoder read-back for the UI. Cheap single read of
        the FOCUS axis position, stored as raw 24-bit counts. Driven directly off
        the controller so it works under either control loop; a read fault just
        leaves the last value in place (never stalls the caller)."""
        if self.telescope_controller is None or not self.telescope_connected:
            return
        try:
            frac = self.telescope_controller.hc_get_position(Targets.FOCUS)
            self.current_focus = int(round(frac * (2 ** 24)))
        except Exception:
            pass

    def _handle_rate_control(self, joy):
        """Handle the original rate control logic with hardware safety limits"""
        # Use the position cached by the control thread this cycle (already
        # offset-applied) instead of re-reading the serial port.
        current_azm = self.current_azm
        current_alt = self.current_alt

        # Get configured limits
        try:
            azm_limit_min = float(self.config_state.azm_limit_min_str)
            azm_limit_max = float(self.config_state.azm_limit_max_str)
            alt_limit_min = float(self.config_state.alt_limit_min_str)
            alt_limit_max = float(self.config_state.alt_limit_max_str)
        except Exception as e:
            # If limits are unparseable, disable limit gating (original behavior)
            print(f"Warning: Could not read telescope limits for safety checks: {e}")
            azm_limit_min = azm_limit_max = alt_limit_min = alt_limit_max = float('inf')

        # Process axes 2 and 3 (PlayStation right stick)
        for i in [2, 3]:  # AZM and ALT axes
            if i >= joy.get_numaxes():
                continue

            axis_value = joy.get_axis(i)

            # Apply tare if available
            if self.connected_joystick in self.joystick_tare:
                tare_value = self.joystick_tare[self.connected_joystick][i]
                axis_value -= tare_value

            # Map to telescope rates (-9 to 9)
            # Clamp values to avoid extreme movements
            rate = axis_to_rate(axis_value)

            # Hardware safety checks - prevent movement that would approach limits
            safe_to_move = True
            if rate > 0 and i == 2:  # Positive AZM rate
                if current_azm >= azm_limit_max:
                    safe_to_move = False
                    self.tracking_mode = TrackingMode.STANDBY
                    if self.update_status_callback:
                        self.update_status_callback(f"AZM safety limit exceeded ({current_azm:.1f} >= {azm_limit_max}) - switched to STANDBY")
            elif rate < 0 and i == 2:  # Negative AZM rate
                if current_azm <= azm_limit_min:
                    safe_to_move = False
                    self.tracking_mode = TrackingMode.STANDBY
                    if self.update_status_callback:
                        self.update_status_callback(f"AZM safety limit exceeded ({current_azm:.1f} <= {azm_limit_min}) - switched to STANDBY")
            elif rate > 0 and i == 3:  # Positive ALT rate
                if current_alt >= alt_limit_max:
                    safe_to_move = False
                    self.tracking_mode = TrackingMode.STANDBY
                    if self.update_status_callback:
                        self.update_status_callback(f"ALT safety limit exceeded ({current_alt:.1f} >= {alt_limit_max}) - switched to STANDBY")
            elif rate < 0 and i == 3:  # Negative ALT rate
                if current_alt <= alt_limit_min:
                    safe_to_move = False
                    self.tracking_mode = TrackingMode.STANDBY
                    if self.update_status_callback:
                        self.update_status_callback(f"ALT safety limit exceeded ({current_alt:.1f} <= {alt_limit_min}) - switched to STANDBY")

            # Send command only if safe
            if safe_to_move:
                try:
                    if i == 2:  # AZM
                        if rate != 0:
                            success = self.telescope_controller.hc_slew_fixed(Targets.AZM, rate)
                            if not success:
                                print(f"Warning: AZM slew command failed (rate={rate})")
                    elif i == 3:  # ALT
                        if rate != 0:
                            success = self.telescope_controller.hc_slew_fixed(Targets.ALT, rate)
                            if not success:
                                print(f"Warning: ALT slew command failed (rate={rate})")
                except Exception as e:
                    print(f"Error sending slew command: {e}")
                    # Try to reconnect or handle the error
                    if "serial" in str(e).lower() or "timeout" in str(e).lower():
                        print("Serial communication error detected. Attempting to reconnect...")
                        try:
                            self.disconnect_telescope()
                            time.sleep(0.1)
                            self.connect_telescope()
                        except Exception as reconnect_e:
                            print(f"Failed to reconnect: {reconnect_e}")

    def cycle_bias_mode(self):
        """Op button: cycle the bias adjust mode through the four combinations of
        resolution (coarse/fine) and frame (Az/El vs along/cross-track). The
        D-pad labels and the bias panel update to match."""
        order = [("coarse", "azel"), ("fine", "azel"),
                 ("coarse", "alongcross"), ("fine", "alongcross")]
        try:
            idx = order.index((self.bias_resolution, self.bias_frame))
        except ValueError:
            idx = -1
        self.bias_resolution, self.bias_frame = order[(idx + 1) % len(order)]
        self.bias_control_mode = self.bias_resolution  # compat alias
        print(f"Bias mode: {self.bias_resolution} / {self.bias_frame}")

    def adjust_bias(self, horizontal, vertical):
        """Apply one D-pad bias step. horizontal/vertical are in {-1, 0, +1}. The
        active frame decides whether the horizontal axis is Az or In-track and the
        vertical axis is El or Cross-track. Values clamp to +/-3 degrees."""
        step = 0.01 if self.bias_resolution == "fine" else 0.1

        def clamp(v):
            return max(-3.0, min(3.0, v))

        if self.bias_frame == "alongcross":
            if horizontal:
                self.bias_intrack_deg = clamp(self.bias_intrack_deg + horizontal * step)
            if vertical:
                self.bias_crosstrack_deg = clamp(self.bias_crosstrack_deg + vertical * step)
        else:
            if horizontal:
                self.bias_azm_deg = clamp(self.bias_azm_deg + horizontal * step)
            if vertical:
                self.bias_alt_deg = clamp(self.bias_alt_deg + vertical * step)

    def _apply_bias_to_target(self, target_az_deg, target_el_deg, az_rate, el_rate):
        """Return (az, el) with operator bias applied. The Az/El bias is an on-sky
        cross-elevation (Az) and elevation (El) nudge; the along/cross-track bias
        is projected onto the target's sky-velocity direction. Cross-elevation is
        converted to an azimuth offset by dividing by cos(el) (azimuth compresses
        toward the zenith). cos(el) is clamped to >= cos(85 deg)."""
        cos_el = max(math.cos(math.radians(target_el_deg)), 0.087)
        crossel = self.bias_azm_deg
        elbias = self.bias_alt_deg
        if self.bias_intrack_deg or self.bias_crosstrack_deg:
            # On-sky tangent-plane velocity: cross-el component is az_rate*cos(el).
            vx = az_rate * cos_el
            vy = el_rate
            norm = math.hypot(vx, vy)
            if norm > 1e-6:
                ux, uy = vx / norm, vy / norm
                crossel += self.bias_intrack_deg * ux - self.bias_crosstrack_deg * uy
                elbias += self.bias_intrack_deg * uy + self.bias_crosstrack_deg * ux
        return target_az_deg + crossel / cos_el, target_el_deg + elbias

    def cycle_tracking_mode_forward(self):
        """Cycle to the next tracking mode in forward sequence"""
        modes = list(TrackingMode)
        current_index = modes.index(self.tracking_mode)
        next_index = (current_index + 1) % len(modes)
        self.tracking_mode = modes[next_index]
        print(f"Tracking mode: {self.tracking_mode.name}")

    def cycle_tracking_mode_backward(self):
        """Cycle to the next tracking mode in reverse sequence"""
        modes = list(TrackingMode)
        current_index = modes.index(self.tracking_mode)
        prev_index = (current_index - 1) % len(modes)
        self.tracking_mode = modes[prev_index]
        print(f"Tracking mode: {self.tracking_mode.name}")

    def program_track(self):
        """Program track mode - implement launch or satellite tracking with PID control"""
        # Import required modules locally to avoid circular imports
        import time

        # Priority: Check for actively launched launch first
        if (self.tracking_vis_state and
            hasattr(self.tracking_vis_state, 'selected_launch') and
            self.tracking_vis_state.selected_launch and
            hasattr(self.tracking_vis_state, 'launch_launched') and
            self.tracking_vis_state.launch_launched):

            # Launch is actively tracking - follow launch trajectory
            return self._program_track_launch()

        # Check if satellite is selected
        if self.tracking_vis_state is None or self.tracking_vis_state.selected_satellite is None:
            # No satellite selected - switch to STANDBY and send message
            self.tracking_mode = TrackingMode.STANDBY
            if self.update_status_callback:
                self.update_status_callback("Select a satellite for PROGRAM tracking")
            else:
                print("PROGRAM TRACK: No satellite selected - switched to STANDBY mode")
            return

        # Ensure PID controllers are initialized
        if self.azm_pid is None or self.alt_pid is None:
            if self.config_state is None:
                print("PROGRAM TRACK: Config state not available")
                return
            self.azm_pid, self.alt_pid = create_pid_controllers(self.config_state)
            print("PROGRAM TRACK: PID controllers initialized")

        # Update PID controller gains from config state in case they were changed in the UI
        if self.azm_pid and self.alt_pid:
            self.azm_pid.update_gains(
                self.config_state.pid_azm_p_gain,
                self.config_state.pid_azm_i_gain,
                self.config_state.pid_azm_d_gain
            )
            self.alt_pid.update_gains(
                self.config_state.pid_alt_p_gain,
                self.config_state.pid_alt_i_gain,
                self.config_state.pid_alt_d_gain
            )
            # Keep each PID's feed-forward-enabled flag in sync with ours.
            # create_pid_controllers() builds them with FF off, so without this the
            # set_feed_forward_rate() calls below would be silently ignored until
            # the FF buttons were toggled.
            self.azm_pid.set_feed_forward_enabled(self.feed_forward_azm_enabled)
            self.alt_pid.set_feed_forward_enabled(self.feed_forward_alt_enabled)

        try:
            # Get configured safety limits
            azm_limit_min = float(self.config_state.azm_limit_min_str)
            azm_limit_max = float(self.config_state.azm_limit_max_str)
            alt_limit_min = float(self.config_state.alt_limit_min_str)
            alt_limit_max = float(self.config_state.alt_limit_max_str)

            # Use the position cached by the control thread this cycle (already
            # offset-applied) instead of re-reading the serial port.
            current_azm = self.current_azm
            current_alt = self.current_alt

            # Get target position from satellite trajectory
            selected_sat = self.tracking_vis_state.selected_satellite
            if self.tracking_vis_state and selected_sat in self.tracking_vis_state.satellite_trajectories:
                # Interpolate current satellite position and rates
                px, py, target_alt, dist, target_az_deg, az_rate, el_rate = interpolate_position_data_and_rates(
                    self.tracking_vis_state.satellite_trajectories[selected_sat],
                    self.tracking_vis_state.current_tt
                )

                if px is not None and target_az_deg is not None:
                    target_el_deg = target_alt

                    # Check if satellite is visible (above horizon)
                    if target_el_deg <= 0:
                        # Satellite is below horizon - drive to mask exit point and wait
                        mask_exit_az, mask_exit_el = self._compute_mask_exit_point(current_azm, current_alt, target_az_deg, self.config_state)

                        print(f"PROGRAM TRACK: Satellite below horizon. Driving to mask exit point: AZ={mask_exit_az:.1f}°, EL={mask_exit_el:.1f}°")

                        # Compute position errors for driving to mask exit point
                        az_error, el_error = compute_mount_position_error(
                            self.config_state, current_azm, current_alt, mask_exit_az, mask_exit_el
                        )

                        # Use maximum rate to drive to mask exit point quickly
                        if abs(az_error) > 1.0:  # Only move if error is significant
                            az_rate_cmd = 9 if az_error > 0 else -9
                        else:
                            az_rate_cmd = 0

                        if abs(el_error) > 1.0:
                            el_rate_cmd = 9 if el_error > 0 else -9
                        else:
                            el_rate_cmd = 0

                        # Apply rate commands to drive to mask exit point
                        if self.telescope_connected and not self.stopped:
                            try:
                                self.telescope_controller.hc_slew_fixed(Targets.AZM, az_rate_cmd)
                                self.telescope_controller.hc_slew_fixed(Targets.ALT, el_rate_cmd)
                            except Exception as e:
                                print(f"Error sending mask exit commands: {e}")

                        # Store diagnostic values
                        self.azm_position_error = az_error
                        self.alt_position_error = el_error
                        self.azm_target_rate = 0.0
                        self.alt_target_rate = 0.0

                        return

                    # Satellite is above horizon - use PID control for tracking
                    # Check hardware safety limits against target position first
                    # If target exceeds limits, abort tracking immediately
                    if (target_az_deg > azm_limit_max or target_az_deg < azm_limit_min or
                        target_el_deg > alt_limit_max or target_el_deg < alt_limit_min):
                        self.tracking_mode = TrackingMode.STANDBY
                        target_info = f"AZ:{target_az_deg:.1f}°/{azm_limit_min:.0f}-{azm_limit_max:.0f}° EL:{target_el_deg:.1f}°/{alt_limit_min:.0f}-{alt_limit_max:.0f}°"
                        if self.update_status_callback:
                            self.update_status_callback(f"Satellite target ({target_info}) exceeds safety limits - switched to STANDBY")
                        return

                    # Lead the target by a configurable time to compensate for
                    # read/command transport latency (the target keeps moving
                    # while we poll position and issue commands). Uses the
                    # interpolated trajectory rates (deg/sec). Defaults to 0.
                    lead_s = float(getattr(self.config_state, 'pid_lead_time_sec', 0.0) or 0.0)
                    if lead_s > 0.0:
                        target_az_deg += az_rate * lead_s
                        target_el_deg += el_rate * lead_s

                    # Cache target sky-velocity for the camera bias-direction axes.
                    self.target_az_rate = az_rate
                    self.target_el_rate = el_rate
                    self.target_el_deg = target_el_deg

                    # Apply operator bias (Az/El or along/cross-track, projected
                    # onto the target's sky-velocity direction).
                    target_az_deg, target_el_deg = self._apply_bias_to_target(
                        target_az_deg, target_el_deg, az_rate, el_rate)

                    # Compute position errors using config state instance
                    az_error, el_error = compute_mount_position_error(
                        self.config_state, current_azm, current_alt, target_az_deg, target_el_deg
                    )

                    # Set feed-forward rates from trajectory. The trajectory gives
                    # sky rates; the PID drives the mount. AZM tracks azimuth
                    # directly, but in AltAz the ALT axis runs opposite sky
                    # elevation (ALT = 90 - el), so the ALT feed-forward is -el_rate.
                    # Without this the elevation feed-forward pushes the wrong way.
                    if self.feed_forward_azm_enabled:
                        self.azm_pid.set_feed_forward_rate(az_rate)
                    if self.feed_forward_alt_enabled:
                        alt_ff_rate = -el_rate if getattr(self.config_state, 'mount_mode', 'Eq') == 'AltAz' else el_rate
                        self.alt_pid.set_feed_forward_rate(alt_ff_rate)

                    # Update PID controllers
                    current_time = time.time()
                    az_pid_output, az_rate_cmd = self.azm_pid.get_current_rates(
                        az_error, current_time - self.pid_last_update, measurement_degrees=current_azm)
                    el_pid_output, el_rate_cmd = self.alt_pid.get_current_rates(
                        el_error, current_time - self.pid_last_update, measurement_degrees=current_alt)
                    self.pid_last_update = current_time

                    # Store diagnostic values for display
                    self.azm_position_error = az_error
                    self.alt_position_error = el_error
                    self.azm_rate_error = self.azm_pid.current_rate_error
                    self.alt_rate_error = self.alt_pid.current_rate_error
                    self.azm_target_rate = az_pid_output
                    self.alt_target_rate = el_pid_output
                    self.azm_pid_output = az_pid_output
                    self.alt_pid_output = el_pid_output

                    # Map rate to signed discrete command
                    az_command = az_rate_cmd if az_rate_cmd != 0 else 0
                    el_command = el_rate_cmd if el_rate_cmd != 0 else 0

                    # Apply rate commands (continuous variable-rate when enabled,
                    # else the discrete MC_MOVE step).
                    if self.telescope_connected and not self.stopped:
                        try:
                            self._issue_axis_rate(Targets.AZM, az_pid_output, az_command)
                            self._issue_axis_rate(Targets.ALT, el_pid_output, el_command)
                        except Exception as e:
                            print(f"Error sending satellite tracking commands: {e}")

                    # Debug output (throttled)
                    if int(current_time) % 5 == 0:  # Every 5 seconds
                        print(f"PROGRAM TRACK: {selected_sat} | "
                              f"AZ:{current_azm:.2f}->{target_az_deg:.2f}({az_error:+.2f}) | "
                              f"EL:{current_alt:.2f}->{target_el_deg:.2f}({el_error:+.2f}) | "
                              f"CMD AZ:{az_command} EL:{el_command}")

        except Exception as e:
            print(f"PROGRAM TRACK: Error in tracking loop: {e}")

    def _program_track_launch(self):
        """Program track mode - implement launch trajectory tracking with PID control"""
        # Import required modules locally to avoid circular imports
        import time

        # Ensure PID controllers are initialized
        if self.azm_pid is None or self.alt_pid is None:
            if self.config_state is None:
                print("LAUNCH TRACK: Config state not available")
                return
            self.azm_pid, self.alt_pid = create_pid_controllers(self.config_state)
            print("LAUNCH TRACK: PID controllers initialized")

        # Update PID controller gains from config state in case they were changed in the UI
        if self.azm_pid and self.alt_pid:
            self.azm_pid.update_gains(
                self.config_state.pid_azm_p_gain,
                self.config_state.pid_azm_i_gain,
                self.config_state.pid_azm_d_gain
            )
            self.alt_pid.update_gains(
                self.config_state.pid_alt_p_gain,
                self.config_state.pid_alt_i_gain,
                self.config_state.pid_alt_d_gain
            )
            # Keep each PID's feed-forward-enabled flag in sync with ours.
            # create_pid_controllers() builds them with FF off, so without this the
            # set_feed_forward_rate() calls below would be silently ignored until
            # the FF buttons were toggled.
            self.azm_pid.set_feed_forward_enabled(self.feed_forward_azm_enabled)
            self.alt_pid.set_feed_forward_enabled(self.feed_forward_alt_enabled)

        try:
            # Get configured safety limits
            azm_limit_min = float(self.config_state.azm_limit_min_str)
            azm_limit_max = float(self.config_state.azm_limit_max_str)
            alt_limit_min = float(self.config_state.alt_limit_min_str)
            alt_limit_max = float(self.config_state.alt_limit_max_str)

            # Use the position cached by the control thread this cycle (already
            # offset-applied) instead of re-reading the serial port.
            current_azm = self.current_azm
            current_alt = self.current_alt

            # Get launch trajectory data
            launch_name = self.tracking_vis_state.selected_launch
            if not launch_name or launch_name not in self.tracking_vis_state.launch_trajectories:
                print(f"LAUNCH TRACK: Launch '{launch_name}' not found in trajectories")
                self.tracking_mode = TrackingMode.STANDBY
                return

            px, py, alt, dist, az_deg, az_rate_dps, el_rate_dps = interpolate_position_data_and_rates(
                self.tracking_vis_state.launch_trajectories[launch_name],
                self.tracking_vis_state.current_tt,
                self.tracking_vis_state.launch_start_time,
                self.tracking_vis_state.launch_launched
            )

            if px is None or az_deg is None:
                print(f"LAUNCH TRACK: Could not interpolate launch position for '{launch_name}'")
                self.tracking_mode = TrackingMode.STANDBY
                if self.update_status_callback:
                    self.update_status_callback(f"Launch '{launch_name}' tracking failed - switched to STANDBY")
                return

            target_el_deg = alt

            # Check if launch is visible (above horizon)
            if target_el_deg <= 0:
                # Launch below horizon - drive to horizon mask point
                mask_exit_az, mask_exit_el = self._compute_mask_exit_point(current_azm, current_alt, az_deg, self.config_state)

                print(f"LAUNCH TRACK: Launch below horizon. Driving to mask exit point: AZ={mask_exit_az:.1f}°, EL={mask_exit_el:.1f}°")

                # Compute position errors for driving to mask exit point
                az_error, el_error = compute_mount_position_error(
                    self.config_state, current_azm, current_alt, mask_exit_az, mask_exit_el
                )

                # Use maximum rate to drive to mask exit point quickly
                if abs(az_error) > 1.0:  # Only move if error is significant
                    az_rate_cmd = 9 if az_error > 0 else -9
                else:
                    az_rate_cmd = 0

                if abs(el_error) > 1.0:
                    el_rate_cmd = 9 if el_error > 0 else -9
                else:
                    el_rate_cmd = 0

                # Apply rate commands to drive to mask exit point
                if self.telescope_connected and not self.stopped:
                    try:
                        self.telescope_controller.hc_slew_fixed(Targets.AZM, az_rate_cmd)
                        self.telescope_controller.hc_slew_fixed(Targets.ALT, el_rate_cmd)
                    except Exception as e:
                        print(f"Error sending launch mask exit commands: {e}")

                # Store diagnostic values
                self.azm_position_error = az_error
                self.alt_position_error = el_error
                self.azm_target_rate = 0.0
                self.alt_target_rate = 0.0

                return

            # Launch is above horizon - use PID control for tracking
            # Check hardware safety limits against target position first
            if (az_deg > azm_limit_max or az_deg < azm_limit_min or
                target_el_deg > alt_limit_max or target_el_deg < alt_limit_min):
                self.tracking_mode = TrackingMode.STANDBY
                target_info = f"AZ:{az_deg:.1f}°/{azm_limit_min:.0f}-{azm_limit_max:.0f}° EL:{target_el_deg:.1f}°/{alt_limit_min:.0f}-{alt_limit_max:.0f}°"
                if self.update_status_callback:
                    self.update_status_callback(f"Launch target ({target_info}) exceeds safety limits - switched to STANDBY")
                return

            # Lead the target by a configurable time to compensate for
            # read/command transport latency (the target keeps moving while we
            # poll position and issue commands). Uses the interpolated
            # trajectory rates (deg/sec). Defaults to 0.
            lead_s = float(getattr(self.config_state, 'pid_lead_time_sec', 0.0) or 0.0)
            if lead_s > 0.0:
                az_deg += az_rate_dps * lead_s
                target_el_deg += el_rate_dps * lead_s

            # Apply operator bias (Az/El or along/cross-track, projected onto the
            # target's sky-velocity direction).
            az_deg, target_el_deg = self._apply_bias_to_target(
                az_deg, target_el_deg, az_rate_dps, el_rate_dps)

            # Compute position errors using config state instance
            az_error, el_error = compute_mount_position_error(
                self.config_state, current_azm, current_alt, az_deg, target_el_deg
            )

            # Set feed-forward rates from launch trajectory
            if self.feed_forward_azm_enabled:
                self.azm_pid.set_feed_forward_rate(az_rate_dps)
            if self.feed_forward_alt_enabled:
                self.alt_pid.set_feed_forward_rate(el_rate_dps)

            # Update PID controllers
            current_time = time.time()
            az_pid_output, az_rate_cmd = self.azm_pid.get_current_rates(
                az_error, current_time - self.pid_last_update, measurement_degrees=current_azm)
            el_pid_output, el_rate_cmd = self.alt_pid.get_current_rates(
                el_error, current_time - self.pid_last_update, measurement_degrees=current_alt)
            self.pid_last_update = current_time

            # Store diagnostic values for display
            self.azm_position_error = az_error
            self.alt_position_error = el_error
            self.azm_rate_error = self.azm_pid.current_rate_error
            self.alt_rate_error = self.alt_pid.current_rate_error
            self.azm_target_rate = az_pid_output
            self.alt_target_rate = el_pid_output
            self.azm_pid_output = az_pid_output
            self.alt_pid_output = el_pid_output

            # Map rate to signed discrete command
            az_command = az_rate_cmd if az_rate_cmd != 0 else 0
            el_command = el_rate_cmd if el_rate_cmd != 0 else 0

            # Apply rate commands (continuous variable-rate when enabled).
            if self.telescope_connected and not self.stopped:
                try:
                    self._issue_axis_rate(Targets.AZM, az_pid_output, az_command)
                    self._issue_axis_rate(Targets.ALT, el_pid_output, el_command)
                except Exception as e:
                    print(f"Error sending launch tracking commands: {e}")

            # Debug output (throttled)
            if int(current_time) % 5 == 0:  # Every 5 seconds
                print(f"LAUNCH TRACK: {launch_name} | AZ:{current_azm:.2f}->{az_deg:.2f}({az_error:+.2f}) | EL:{current_alt:.2f}->{target_el_deg:.2f}({el_error:+.2f}) | CMD AZ:{az_command} EL:{el_command}")

        except Exception as e:
            print(f"LAUNCH TRACK: Error in tracking loop: {e}")

    def _compute_mask_exit_point(self, current_azm, current_alt, target_az, config_state):
        """
        Compute the optimal mask exit point for tracking a satellite.

        The mask exit point is typically the point on the horizon where the satellite
        will first appear, allowing the telescope to position itself optimally
        for when the satellite rises above the horizon.

        Args:
            current_azm: Current azimuth of mount (degrees)
            current_alt: Current elevation of mount (degrees)
            target_az: Target azimuth (degrees)
            config_state: Configuration state with elevation mask

        Returns:
            tuple: (az_exit, el_exit) - azimuth and elevation of mask exit point
        """
        try:
            # Get elevation mask from config
            elevation_mask = float(config_state.elevation_mask_str) if hasattr(config_state, 'elevation_mask_str') else 10.0

            # The mask exit point is typically at elevation = elevation_mask, azimuth = target_az
            # This represents where the satellite will first appear above the horizon
            az_exit = target_az
            el_exit = elevation_mask

            return az_exit, el_exit

        except Exception as e:
            print(f"PROGRAM TRACK: Error computing mask exit point: {e}")
            # Fallback: return a safe default position
            return 0.0, 10.0

    def reset_program_tracking(self):
        """Reset program tracking state"""
        if self.azm_pid:
            self.azm_pid.reset()
        if self.alt_pid:
            self.alt_pid.reset()
        self.pid_last_update = 0.0

    def handoff_track(self):
        """HANDOFF: keep PROGRAM track closing the loop while running the hotspot
        detector on the camera frame in parallel (without commanding from it).
        After N consecutive solid detections (config: handoff_min_frames),
        auto-engage HOTSPOT to take over the loop. HOTSPOT itself coasts/falls
        back to PROGRAM on loss, so a bad hand-off self-corrects."""
        cfg = self.config_state

        # 1) Continue program tracking (this drives the mount this cycle).
        self.program_track()

        # program_track() may bail to STANDBY (no target / outside limits); if it
        # left HANDOFF, abort the hand-off logic for this cycle.
        if self.tracking_mode != TrackingMode.HANDOFF:
            return

        # 2) Run the hotspot detector in parallel (detection only, no commanding).
        needed = max(1, int(getattr(cfg, 'handoff_min_frames', 5) or 5))
        if self._handoff_detect(cfg):
            self.handoff_detection_count += 1
            self.handoff_status = f"detecting {self.handoff_detection_count}/{needed}"
            if self.handoff_detection_count >= needed:
                self.handoff_detection_count = 0
                self.handoff_status = "engaged HOTSPOT"
                self.tracking_mode = TrackingMode.HOTSPOT
                if self.update_status_callback:
                    self.update_status_callback(
                        "HANDOFF: solid detection - engaging HOTSPOT tracker")
        else:
            # Require *consecutive* detections; any miss resets the counter.
            self.handoff_detection_count = 0
            self.handoff_status = "program track (no detection)"

    def _handoff_detect(self, cfg):
        """Run hotspot detection on the tracking camera frame and return True for
        a solid detection. Updates the hotspot diagnostics (centroid, SNR) but
        never commands the mount."""
        try:
            cam_index = int(getattr(cfg, 'hotspot_camera_index', 0))
            camera = camera_manager.get_camera(cam_index)
            if camera is None or getattr(camera, 'thread', None) is None:
                self.hotspot_snr = 0.0
                return False
            raw = camera.thread.get_latest_raw()
            if raw is None:
                return False
            detection = detect_hotspot(
                raw,
                gate_center=None,
                gate_radius=None,
                snr_threshold=float(getattr(cfg, 'hotspot_snr_threshold', 5.0)),
            )
            if detection is not None:
                self.hotspot_centroid = (detection.cx, detection.cy)
                self.hotspot_snr = detection.snr
                return True
            self.hotspot_snr = 0.0
            return False
        except Exception as e:
            print(f"HANDOFF: detection error: {e}")
            return False

    def _enter_hotspot_mode(self):
        """Reset state when HOTSPOT is engaged (handed off from another mode)."""
        self.hotspot_gate_center = None      # force a full-frame acquisition
        self.hotspot_acquired = False
        self.hotspot_miss_count = 0
        self.hotspot_last_detection_time = 0.0
        self.hotspot_centroid = None
        self.hotspot_snr = 0.0
        if self.azm_pid:
            self.azm_pid.reset()
        if self.alt_pid:
            self.alt_pid.reset()
        self.pid_last_update = time.time()
        self.hotspot_entry_time = time.time()
        self.hotspot_status = "acquiring"
        print("HOTSPOT: engaged - acquiring...")

    def hotspot_track(self):
        """Closed-loop optical tracker: lock onto the brightest ('hot') object in
        the camera frame and drive the mount to keep it centered.

        Intended as a hand-off target once PROGRAM track has the object in frame.
        On loss of lock it coasts briefly, then falls back to PROGRAM track.
        Best for bright objects (rockets, aircraft); dim satellites amid streaking
        stars need a different mode.
        """
        cfg = self.config_state

        # Ensure PID controllers exist (shared with PROGRAM track).
        if self.azm_pid is None or self.alt_pid is None:
            if cfg is None:
                print("HOTSPOT: config state not available")
                return
            self.azm_pid, self.alt_pid = create_pid_controllers(cfg)
        self.azm_pid.update_gains(cfg.pid_azm_p_gain, cfg.pid_azm_i_gain, cfg.pid_azm_d_gain)
        self.alt_pid.update_gains(cfg.pid_alt_p_gain, cfg.pid_alt_i_gain, cfg.pid_alt_d_gain)

        # Grab the latest raw frame from the configured tracking camera.
        cam_index = int(getattr(cfg, 'hotspot_camera_index', 0))
        camera = camera_manager.get_camera(cam_index)
        raw = None
        if camera is not None and getattr(camera, 'thread', None) is not None:
            raw = camera.thread.get_latest_raw()

        # Mount position cached by the control thread this cycle.
        current_azm = self.current_azm
        current_alt = self.current_alt

        # Safety: abort to STANDBY if the mount is outside configured limits.
        try:
            azm_min = float(cfg.azm_limit_min_str); azm_max = float(cfg.azm_limit_max_str)
            alt_min = float(cfg.alt_limit_min_str); alt_max = float(cfg.alt_limit_max_str)
            if not (azm_min <= current_azm <= azm_max and alt_min <= current_alt <= alt_max):
                self._hotspot_stop_motion()
                self.tracking_mode = TrackingMode.STANDBY
                if self.update_status_callback:
                    self.update_status_callback("HOTSPOT: mount at safety limit - switched to STANDBY")
                return
        except (ValueError, AttributeError):
            pass

        detection = None
        if raw is not None:
            gate_center = self.hotspot_gate_center
            gate_radius = int(getattr(cfg, 'hotspot_gate_radius', 120)) if gate_center else None
            try:
                detection = detect_hotspot(
                    raw,
                    gate_center=gate_center,
                    gate_radius=gate_radius,
                    snr_threshold=float(getattr(cfg, 'hotspot_snr_threshold', 5.0)),
                )
            except Exception as e:
                print(f"HOTSPOT: detection error: {e}")
                detection = None

        now = time.time()

        if detection is not None:
            # Pixel error of the object from boresight (image center).
            h, w = raw.shape[:2]
            dx = detection.cx - (w / 2.0)
            dy = detection.cy - (h / 2.0)

            cam_name = f"camera{cam_index + 1}"
            az_error, el_error = pixel_offset_to_angles(
                dx, dy,
                pixel_size_um=float(cfg.get_camera_pixel_size(cam_name)),
                focal_length_mm=float(cfg.get_camera_focal_length(cam_name)),
                rotation_deg=float(cfg.get_camera_alignment_rotation(cam_name)),
                el_deg=current_alt,
                x_sign=float(getattr(cfg, 'hotspot_x_sign', 1.0)),
                y_sign=float(getattr(cfg, 'hotspot_y_sign', -1.0)),
            )

            # pixel_offset_to_angles returns the correction in the optical/sky frame
            # (elevation increasing upward). The PID drives the mount in mount
            # coordinates; in AltAz mode the ALT axis runs opposite sky elevation
            # (el = 90 - ALT), so negate the elevation term -- this makes HOTSPOT
            # feed the same mount-frame error sign that PROGRAM track gets from
            # compute_mount_position_error.
            if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz':
                el_error = -el_error

            # Drive the mount to null the optical error using the shared PID.
            current_time = now
            az_pid_output, az_rate_cmd = self.azm_pid.get_current_rates(
                az_error, current_time - self.pid_last_update, measurement_degrees=current_azm)
            el_pid_output, el_rate_cmd = self.alt_pid.get_current_rates(
                el_error, current_time - self.pid_last_update, measurement_degrees=current_alt)
            self.pid_last_update = current_time

            # Update lock state and diagnostics.
            self.hotspot_gate_center = (detection.cx, detection.cy)
            self.hotspot_centroid = (detection.cx, detection.cy)
            self.hotspot_snr = detection.snr
            self.hotspot_acquired = True
            self.hotspot_miss_count = 0
            self.hotspot_last_detection_time = now
            self.hotspot_status = "locked"
            self.azm_position_error = az_error
            self.alt_position_error = el_error
            self.azm_target_rate = az_pid_output
            self.alt_target_rate = el_pid_output

            if self.telescope_connected and not self.stopped:
                try:
                    self._issue_axis_rate(Targets.AZM, az_pid_output, az_rate_cmd)
                    self._issue_axis_rate(Targets.ALT, el_pid_output, el_rate_cmd)
                except Exception as e:
                    print(f"HOTSPOT: error sending slew commands: {e}")
            return

        # No detection this cycle.
        self.hotspot_miss_count += 1
        coast_time = float(getattr(cfg, 'hotspot_coast_time_sec', 1.0))

        if self.hotspot_acquired:
            if (now - self.hotspot_last_detection_time) < coast_time:
                # Coast: leave the last continuous slew running so a brief dropout
                # (cloud, frame glitch) doesn't jerk the mount.
                self.hotspot_status = "coasting"
                return
        elif (now - self.hotspot_entry_time) < max(coast_time, 1.0):
            # Acquisition grace: just engaged (manual or handed off). Give frames
            # time to arrive / the target time to be found before falling back,
            # leaving the last slew running so a moving target stays framed.
            self.hotspot_status = "acquiring"
            return

        # Lock lost (or never acquired within the grace window): stop and hand
        # back to PROGRAM track per configured behavior.
        self._hotspot_stop_motion()
        self.hotspot_acquired = False
        self.hotspot_gate_center = None
        self.hotspot_status = "lost"
        self.tracking_mode = TrackingMode.PROGRAM
        if self.update_status_callback:
            self.update_status_callback("HOTSPOT: lost lock - falling back to PROGRAM track")

    def _issue_axis_rate(self, target, pid_output_rev_s, discrete_cmd):
        """Send one axis's rate to the mount.

        In continuous-rate mode use the fine 24-bit variable-rate (guide-rate)
        primitive so the commanded rate matches the target's motion exactly and
        avoids the MC_MOVE quantization sawtooth. Fall back to the discrete
        MC_MOVE step when the requested rate exceeds guide_rate_max_dps (the
        regime where the guide-rate command may not be honored, e.g. the
        near-zenith keyhole) or when continuous mode is off.
        """
        cfg = self.config_state
        if getattr(cfg, 'continuous_rate_tracking', False) and hasattr(self.telescope_controller, 'hc_set_rate_dps'):
            dps = pid_output_rev_s * 360.0
            if abs(dps) <= float(getattr(cfg, 'guide_rate_max_dps', 5.0)):
                self.telescope_controller.hc_set_rate_dps(target, dps)
                return
        self.telescope_controller.hc_slew_fixed(target, discrete_cmd)

    def _hotspot_stop_motion(self):
        """Best-effort stop of both axes."""
        if self.telescope_connected and self.telescope_controller is not None:
            try:
                self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
            except Exception as e:
                print(f"HOTSPOT: error stopping motion: {e}")

    def mti_track(self):
        """MTI track mode - stub implementation"""
        # TODO: Implement MTI track mode
        print("MTI track mode - stub")

    def process_joystick_events(self, event, current_mode=None, current_tracking_surface=None, tracking_vis_state=None, config_state=None):
        """Process pygame joystick events (non-mode-specific)"""
        if event.type == pygame.JOYDEVICEADDED:
            joy = pygame.joystick.Joystick(event.device_index)
            self.joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connected: {joy.get_name()}")

            # Auto-connect to first joystick
            if self.connected_joystick is None:
                self.connected_joystick = joy.get_instance_id()
                # Initialize tare values
                self.reset_tare()

        elif event.type == pygame.JOYDEVICEREMOVED:
            if event.instance_id in self.joysticks:
                del self.joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")

                # If this was the connected joystick, disconnect it
                if self.connected_joystick == event.instance_id:
                    self.connected_joystick = None

                    # Connect to remaining joystick if any
                    if self.joysticks:
                        self.connected_joystick = next(iter(self.joysticks.keys()))

        elif event.type == pygame.JOYBUTTONDOWN:
            # Handle button events based on mode
            print(f"process_joystick_events: Button {event.button} pressed in mode {current_mode}")

            # Capture button (X button) - works in ANY mode
            if event.button == 0:  # X button
                print("process_joystick_events: X button pressed - calling _handle_capture_toggle")
                self._handle_capture_toggle(current_tracking_surface, tracking_vis_state, config_state)
                return  # Don't process any other buttons after handling capture

            # Handle stop button (Circle button) - this is universal
            if event.button == 1:  # Circle button
                self.stopped = not self.stopped
                print(f"Stop toggled: {self.stopped}")

                # Stop movement immediately when stopped
                if self.stopped and self.telescope_connected:
                    self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                    self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)

            elif event.button == 2:  # Square button for tare
                print(f"Taring joystick axes")
                self.tare_current_joystick()

            elif event.button == 3:  # Triangle/Park button for park command
                if self.telescope_connected:
                    print("Parking telescope to 0, 0...")
                    # Loop until parked
                    angle = 1
                    while abs(angle - float(config_state.azm_offset_str)) > 1:
                        success = self.telescope_controller.hc_goto_fast(Targets.AZM, float(config_state.azm_offset_str), 0, 0)
                        print("AZM Park command sent")
                        time.sleep(0.1)
                        angle = self.telescope_controller.hc_get_position(Targets.AZM) * 360
                    angle = 1
                    while abs(angle - float(config_state.alt_offset_str)) > 1: # keep trying until <1 degree
                        success = self.telescope_controller.hc_goto_fast(Targets.ALT, float(config_state.alt_offset_str), 0, 0)
                        print("ALT Park command sent")
                        time.sleep(0.1)
                        angle = self.telescope_controller.hc_get_position(Targets.ALT) * 360
                else:
                    print("Cannot park: telescope not connected")

            elif event.button == 4:  # Share button - cycle bias adjust mode
                self.cycle_bias_mode()
                return

            elif event.button == 6:  # Op (Options) button - mount mode toggle
                # Cycle through AltAz, Eq, and Passthrough modes
                if self.mount_mode == "AltAz":
                    self.mount_mode = "Eq"
                elif self.mount_mode == "Eq":
                    self.mount_mode = "Passthrough"
                else:  # Passthrough or unknown
                    self.mount_mode = "AltAz"

                if self.config_state:
                    self.config_state.mount_mode = self.mount_mode
                print(f"Mount mode toggled to: {self.mount_mode}")
                return  # Don't process any other buttons after handling mode toggle

            elif event.button == 8:  # RS button for forward tracking mode cycle
                self.cycle_tracking_mode_forward()
                return  # Don't process any other buttons after handling mode cycle

            elif event.button == 9:  # L1 button - toggle feed-forward (both axes)
                self.feed_forward_azm_enabled = not self.feed_forward_azm_enabled
                self.feed_forward_alt_enabled = not self.feed_forward_alt_enabled
                if self.azm_pid:
                    self.azm_pid.set_feed_forward_enabled(self.feed_forward_azm_enabled)
                if self.alt_pid:
                    self.alt_pid.set_feed_forward_enabled(self.feed_forward_alt_enabled)
                print(f"Feed-forward AZ: {self.feed_forward_azm_enabled}, EL: {self.feed_forward_alt_enabled}")
                return

            # D-pad as discrete buttons (some controllers report it this way
            # instead of as a hat). Horizontal = Az/In-track, vertical = El/Cross.
            elif event.button == 11:  # D-pad up
                self.adjust_bias(0, +1)
                return
            elif event.button == 12:  # D-pad down
                self.adjust_bias(0, -1)
                return
            elif event.button == 13:  # D-pad left
                self.adjust_bias(-1, 0)
                return
            elif event.button == 14:  # D-pad right
                self.adjust_bias(+1, 0)
                return

            elif event.button == 15:  # Pad button for backward tracking mode cycle
                self.cycle_tracking_mode_backward()
                return  # Don't process any other buttons after handling mode cycle

        elif event.type == pygame.JOYHATMOTION:
            # D-pad reported as a hat (the other common mapping). Route to the same
            # bias adjust as the D-pad buttons above. hat_y is +1 up in pygame.
            hat_x, hat_y = event.value
            if hat_x:
                self.adjust_bias(1 if hat_x > 0 else -1, 0)
            if hat_y:
                self.adjust_bias(0, 1 if hat_y > 0 else -1)

        elif event.type == pygame.JOYAXISMOTION:
            # Update joystick axis state for display (this stays in process for all modes)
            if self.connected_joystick is not None and event.joy == self.connected_joystick:
                # Could store axis state here if needed for display updates
                pass

    def _handle_capture_toggle(self, tracking_surface, tracking_vis_state, config_state):
        """Handle capture toggle for joystick"""
        from camera_manager import camera_manager
        from capture_manager import capture_manager

        if self.capture_active:
            # Stop capture and begin dump process for all cameras
            capture_manager.stop_capture(None, tracking_vis_state, tracking_vis_state.selected_satellite, config_state, tracking_surface)
            print("Capture stopped on all cameras, dump process started")
            self.capture_active = False
        else:
            # Start capture on all connected cameras
            # Check if any camera is available and connected
            any_camera_available = False
            for idx in range(len(camera_manager.cameras)):
                camera = camera_manager.get_camera(idx)
                if camera and camera.connected:
                    any_camera_available = True
                    break

            if any_camera_available:
                capture_manager.start_capture()  # Start capture on all cameras
                self.capture_active = True
                print("Capture started on all connected cameras")
            else:
                print("No cameras available for capture")

    # Helper methods to get states from global scope - removed to avoid circular import

    def update_polar_plot_time(self):
        """Update current time for polar plot"""
        if not self.ts:
            self.ts = load.timescale()
        self.current_tt = self.ts.now().tt

def render_position_display(display, joystick_state):
    """Render the current mode / AZM / ALT position box.

    Anchored to the top-right corner of the joystick mode's upper-left
    quadrant. The box is always drawn (so its location is visible) but is
    greyed out when the telescope is not connected and there is no live data.
    """
    connected = joystick_state.telescope_connected

    # Box sized to envelope its text (the old 100x65 box clipped the values).
    width, height = 140, 86
    x_start = display.sub_x + display.sub_width // 2 - width - 12
    y_start = display.sub_y + 12

    # Background color based on stop state
    if joystick_state.stopped:
        bg_color = (120, 60, 60)  # Reddish background when stopped
        border_color = (200, 100, 100)
    else:
        bg_color = (60, 60, 60)   # Normal grey when active
        border_color = (100, 100, 100)

    # Background rectangle for position display
    box_rect = pygame.Rect(x_start, y_start, width, height)
    pygame.draw.rect(display.menu_screen, bg_color, box_rect)
    pygame.draw.rect(display.menu_screen, border_color, box_rect, 1)

    # Mode/status indicator at the top
    if joystick_state.stopped:
        status_text = "STOPPED"
        status_color = (255, 100, 100)  # Red when stopped
    else:
        status_text = joystick_state.tracking_mode.name  # Show current tracking mode
        status_color = (100, 255, 100)  # Green when active
    status_render = display.small_font.render(status_text, True, status_color)
    display.menu_screen.blit(status_render, (x_start + 6, y_start + 4))

    # Add mount mode indicator
    mount_mode_text = display.tiny_font.render(f"Mount: {joystick_state.mount_mode}", True, (200, 200, 100))
    display.menu_screen.blit(mount_mode_text, (x_start + 6, y_start + 22))

    # AZM display
    azm_label_text = display.tiny_font.render("AZM", True, (255, 255, 255))
    display.menu_screen.blit(azm_label_text, (x_start + 6, y_start + 38))
    azm_value = joystick_state.azm_display_str[:14] if connected else "--"
    azm_value_text = display.tiny_font.render(azm_value, True, (0, 255, 0))
    display.menu_screen.blit(azm_value_text, (x_start + 34, y_start + 38))

    # ALT display
    alt_label_text = display.tiny_font.render("ALT", True, (255, 255, 255))
    display.menu_screen.blit(alt_label_text, (x_start + 6, y_start + 54))
    alt_value = joystick_state.alt_display_str[:14] if connected else "--"
    alt_value_text = display.tiny_font.render(alt_value, True, (0, 255, 0))
    display.menu_screen.blit(alt_value_text, (x_start + 34, y_start + 54))

    # Connection hint at the bottom
    conn_text = "Connected" if connected else "Disconnected"
    conn_color = (120, 220, 120) if connected else (200, 120, 120)
    conn_render = display.tiny_font.render(conn_text, True, conn_color)
    display.menu_screen.blit(conn_render, (x_start + 6, y_start + 70))

    # Grey the box out when there is no live telescope data
    if not connected:
        _draw_disabled_scrim(display, box_rect)

# ==============================================================================
# JOYSTICK MODE RENDERING FUNCTIONS
# ==============================================================================

def render_connection_controls(display, joystick_state):
    """Render connection controls in upper left"""
    # Update available ports
    joystick_state.get_available_serial_ports()

    # Connect/Disconnect buttons
    button_y = display.sub_y + 10
    button_width = 80
    button_height = 30

    # Connect button
    connect_rect = pygame.Rect(display.sub_x + 10, button_y, button_width, button_height)
    joystick_state.connect_button_hover = connect_rect.collidepoint(pygame.mouse.get_pos())

    if joystick_state.telescope_connected:
        button_color = (100, 100, 100)  # Grey when connected
    else:
        button_color = (100, 150, 100) if joystick_state.connect_button_hover else (100, 120, 100)

    pygame.draw.rect(display.menu_screen, button_color, connect_rect)
    connect_text = display.small_font.render("Connect", True, (255, 255, 255))
    display.menu_screen.blit(connect_text, (connect_rect.x + 5, connect_rect.y + 5))

    # Disconnect button
    disconnect_rect = pygame.Rect(display.sub_x + 100, button_y, button_width, button_height)
    joystick_state.disconnect_button_hover = disconnect_rect.collidepoint(pygame.mouse.get_pos())

    if not joystick_state.telescope_connected:
        button_color = (100, 100, 100)  # Grey when disconnected
    else:
        button_color = (150, 100, 100) if joystick_state.disconnect_button_hover else (120, 100, 100)

    pygame.draw.rect(display.menu_screen, button_color, disconnect_rect)
    disconnect_text = display.small_font.render("Disconnect", True, (255, 255, 255))
    display.menu_screen.blit(disconnect_text, (disconnect_rect.x + 5, disconnect_rect.y + 5))

    # Port selector
    port_y = button_y + 40
    port_label = display.small_font.render("Port:", True, (255, 255, 255))
    display.menu_screen.blit(port_label, (display.sub_x + 10, port_y))

    # Port dropdown box
    dropdown_width = 120
    dropdown_height = 25
    dropdown_rect = pygame.Rect(display.sub_x + 50, port_y, dropdown_width, dropdown_height)
    pygame.draw.rect(display.menu_screen, (70, 70, 70), dropdown_rect)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), dropdown_rect, 1)

    if joystick_state.selected_port:
        port_text = display.small_font.render(joystick_state.selected_port, True, (255, 255, 255))
    else:
        port_text = display.small_font.render("Select Port", True, (255, 255, 255))
    display.menu_screen.blit(port_text, (dropdown_rect.x + 5, dropdown_rect.y + 3))

    # Baud rate display (fixed)
    baud_y = port_y + 30
    baud_text = display.small_font.render("Baud: 9600 (fixed)", True, (255, 255, 255))
    display.menu_screen.blit(baud_text, (display.sub_x + 10, baud_y))

    # Connection status
    status_y = baud_y + 25
    if joystick_state.telescope_connected:
        status_text = display.small_font.render("Status: Connected", True, (0, 255, 0))
    else:
        status_text = display.small_font.render("Status: Disconnected", True, (255, 0, 0))
    display.menu_screen.blit(status_text, (display.sub_x + 10, status_y))

def render_joystick_status(display, joystick_state):
    """Render the joystick status (button map + axes) below the connection
    controls.

    The full button map and axis displays are always drawn so their location
    is visible, with each button labelled by the functionality currently
    assigned to it. When no joystick is connected the whole block is greyed
    out (visible but inactive).
    """
    connected = (joystick_state.connected_joystick is not None and
                 joystick_state.connected_joystick in joystick_state.joysticks)
    joy = joystick_state.joysticks[joystick_state.connected_joystick] if connected else None
    num_buttons = joy.get_numbuttons() if connected else 0
    num_axes = joy.get_numaxes() if connected else 0

    base_x = display.sub_x + 10
    y_start = display.sub_y + 140

    # Joystick name (kept outside the grey scrim so its status stays readable)
    if connected:
        name_text = display.small_font.render(f"Joystick: {joy.get_name()}", True, (255, 255, 255))
    else:
        name_text = display.small_font.render("Joystick: None", True, (255, 0, 0))
    display.menu_screen.blit(name_text, (base_x, y_start))

    # ---- Buttons section ---------------------------------------------------
    region_top = y_start + 25
    buttons_label = display.small_font.render("Buttons (function):", True, (255, 255, 255))
    display.menu_screen.blit(buttons_label, (base_x, region_top))

    buttons_top = region_top + 20
    col_w = 130
    func_map = button_function_map(joystick_state)
    swatch_w, swatch_h = 26, 18
    row_h = 22
    num_slots = len(BUTTON_LABELS)
    rows_per_col = (num_slots + 1) // 2  # 8 rows over two columns for 16 buttons

    for i in range(num_slots):
        col = i // rows_per_col
        row = i % rows_per_col
        bx = base_x + col * col_w
        by = buttons_top + row * row_h

        active = connected and i < num_buttons and joy.get_button(i)
        swatch_color = (0, 255, 0) if active else (90, 90, 90)
        swatch_rect = pygame.Rect(bx, by, swatch_w, swatch_h)
        pygame.draw.rect(display.menu_screen, swatch_color, swatch_rect)
        pygame.draw.rect(display.menu_screen, (150, 150, 150), swatch_rect, 1)

        label = BUTTON_LABELS[i] if i < len(BUTTON_LABELS) else str(i)
        label_color = (0, 0, 0) if active else (230, 230, 230)
        label_text = display.tiny_font.render(label, True, label_color)
        display.menu_screen.blit(label_text, label_text.get_rect(center=swatch_rect.center))

        func = func_map.get(i, "-")
        func_color = (220, 220, 220) if i in func_map else (120, 120, 120)
        func_text = display.tiny_font.render(func, True, func_color)
        display.menu_screen.blit(func_text, (bx + swatch_w + 4, by + 4))

    current_y = buttons_top + rows_per_col * row_h + 12

    # ---- Axes section ------------------------------------------------------
    axes_label = display.small_font.render("Axes:", True, (255, 255, 255))
    display.menu_screen.blit(axes_label, (base_x, current_y))
    current_y += 20

    # First two axis pairs as 2D crosshair boxes (left/right sticks)
    box_size = 60
    crosshair_range = 20
    for pair in range(2):
        axis_x, axis_y = pair * 2, pair * 2 + 1
        box_x, box_y = base_x, current_y
        center_x = box_x + box_size // 2
        center_y = box_y + box_size // 2

        pygame.draw.rect(display.menu_screen, (80, 80, 80), (box_x, box_y, box_size, box_size))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (box_x, box_y, box_size, box_size), 1)

        x_val = joy.get_axis(axis_x) if connected and axis_x < num_axes else 0.0
        y_val = joy.get_axis(axis_y) if connected and axis_y < num_axes else 0.0
        crosshair_x = center_x + int(x_val * crosshair_range)
        crosshair_y = center_y + int(y_val * crosshair_range)

        pygame.draw.line(display.menu_screen, (255, 255, 255),
                         (crosshair_x, center_y - crosshair_range),
                         (crosshair_x, center_y + crosshair_range), 1)
        pygame.draw.line(display.menu_screen, (255, 255, 255),
                         (center_x - crosshair_range, crosshair_y),
                         (center_x + crosshair_range, crosshair_y), 1)

        pair_label = "Left Stick" if pair == 0 else "Right Stick"
        label_text = display.tiny_font.render(pair_label, True, (255, 255, 255))
        display.menu_screen.blit(label_text, (base_x + box_size + 10, current_y + 20))

        current_y += box_size + 10

    # Triggers (L2/R2) as linear sliders
    for idx, ax_label in ((4, "L2"), (5, "R2")):
        slider_width, slider_height = 100, 12
        slider_x, slider_y = base_x, current_y
        pygame.draw.rect(display.menu_screen, (80, 80, 80), (slider_x, slider_y, slider_width, slider_height))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (slider_x, slider_y, slider_width, slider_height), 1)

        axis_val = joy.get_axis(idx) if connected and idx < num_axes else -1.0
        slider_pos = int((axis_val + 1) / 2 * slider_width)
        pygame.draw.rect(display.menu_screen, (255, 255, 0),
                         (slider_x + slider_pos - 2, slider_y - 2, 4, slider_height + 4))

        val_text = display.tiny_font.render(f"{ax_label}: {axis_val:+.2f}", True, (255, 255, 255))
        display.menu_screen.blit(val_text, (slider_x + slider_width + 10, slider_y))
        current_y += slider_height + 8

    # Focus motor: the L2/R2 triggers above drive it (L2 = retract/-, R2 = extend
    # /+). Show the commanded rate as a centre-zero bar plus the live encoder
    # read-back so the operator can see focus moving.
    tele_connected = joystick_state.telescope_connected
    focus_rate = int(getattr(joystick_state, 'focus_rate', 0))
    focus_pos = int(getattr(joystick_state, 'current_focus', 0))

    focus_label = display.small_font.render("Focus (L2-/R2+):", True, (255, 255, 255))
    display.menu_screen.blit(focus_label, (base_x, current_y))
    current_y += 20

    bar_w, bar_h = 100, 12
    bar_x, bar_y = base_x, current_y
    pygame.draw.rect(display.menu_screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(display.menu_screen, (150, 150, 150), (bar_x, bar_y, bar_w, bar_h), 1)
    center_x = bar_x + bar_w // 2
    pygame.draw.line(display.menu_screen, (150, 150, 150),
                     (center_x, bar_y), (center_x, bar_y + bar_h), 1)
    pos_color, neg_color, idle_color = (0, 220, 0), (240, 140, 0), (200, 200, 200)
    if focus_rate != 0:
        fill = int(max(-1.0, min(1.0, focus_rate / 9.0)) * (bar_w // 2))
        fill_color = pos_color if focus_rate > 0 else neg_color
        x0 = center_x if fill >= 0 else center_x + fill
        pygame.draw.rect(display.menu_screen, fill_color, (x0, bar_y + 1, abs(fill), bar_h - 1))

    rate_color = pos_color if focus_rate > 0 else neg_color if focus_rate < 0 else idle_color
    rate_text = display.tiny_font.render(f"rate {focus_rate:+d}", True, rate_color)
    display.menu_screen.blit(rate_text, (bar_x + bar_w + 10, bar_y - 2))
    pos_str = f"pos {focus_pos}" if tele_connected else "pos --"
    pos_text = display.tiny_font.render(pos_str, True, (0, 255, 0) if tele_connected else (160, 160, 160))
    display.menu_screen.blit(pos_text, (bar_x + bar_w + 10, bar_y + 9))
    current_y += bar_h + 10

    # Hat information
    num_hats = joy.get_numhats() if connected else 0
    hats_label = display.small_font.render(f"Hats: {num_hats}", True, (255, 255, 255))
    display.menu_screen.blit(hats_label, (base_x, current_y))
    current_y += 20

    # Grey the whole button/axes block out when no joystick is connected
    if not connected:
        region = pygame.Rect(display.sub_x + 5, region_top - 2,
                             2 * col_w + 20, current_y - region_top + 2)
        _draw_disabled_scrim(display, region)

def render_capture_controls(display, joystick_state):
    """Render capture controls and progress indicator below polar graph"""
    try:
        # Update capture progress from capture manager
        from capture_manager import capture_manager
        progress, status = capture_manager.get_dump_progress()

        # Update joystick state
        joystick_state.capture_progress = progress
        joystick_state.capture_status = status if status else ""

        # Determine if cameras are connected and get individual camera status
        camera_connected = False
        camera_buffer_info = {}
        joystick_state.capture_active = False

        for idx in range(len(camera_manager.cameras)):
            camera = camera_manager.get_camera(idx)
            if camera and camera.connected and camera.thread:
                camera_connected = True
                buffer_info = camera.thread.get_capture_buffer_info()

                # Store individual camera info
                camera_buffer_info[idx] = buffer_info

                # Check if any camera is actively capturing
                if buffer_info.get('capture_active', False):
                    joystick_state.capture_active = True

        # Capture toggle button
        button_y = display.sub_y + display.sub_height // 2 - 80  # Between polar graph and camera feeds
        button_width = 90
        button_height = 35
        button_x = display.sub_x + display.sub_width - button_width - 20  # Right side of screen

        joystick_state.capture_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        # Button color based on capture state
        mouse_pos = pygame.mouse.get_pos()
        hover = joystick_state.capture_button_rect.collidepoint(mouse_pos)

        if joystick_state.capture_active:
            button_color = (0, 150, 0) if hover else (0, 100, 0)  # Green when active
            button_text = "Stop Capture"
        else:
            button_color = (150, 150, 150) if hover else (120, 120, 120)  # Grey when inactive
            button_text = "Start Capture"

        pygame.draw.rect(display.menu_screen, button_color, joystick_state.capture_button_rect)
        pygame.draw.rect(display.menu_screen, (200, 200, 200), joystick_state.capture_button_rect, 1)

        # Button text
        button_surface = display.small_font.render(button_text, True, (255, 255, 255))
        text_rect = button_surface.get_rect(center=joystick_state.capture_button_rect.center)
        display.menu_screen.blit(button_surface, text_rect)

        # Buffer fill progress bar (below button)
        progress_y = button_y + button_height + 10
        progress_width = button_width
        progress_height = 15

        # Progress bar background
        pygame.draw.rect(display.menu_screen, (50, 50, 50), (button_x, progress_y, progress_width, progress_height))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (button_x, progress_y, progress_width, progress_height), 1)

        # Display individual camera buffer status
        if camera_connected and camera_buffer_info:
            # Display format: C1: [frames] [percent]% | C2: [frames] [percent]%
            camera_status_lines = []
            camera_fill_ratios = []
            total_capacity = 0

            for cam_idx in sorted(camera_buffer_info.keys()):
                buffer_info = camera_buffer_info[cam_idx]
                cam_num = cam_idx + 1
                max_buffer_size = buffer_info.get('max_buffer_size', 1000)
                total_capacity += max_buffer_size

                if joystick_state.capture_active:
                    # During capture: show frame count and capture progress percentage
                    captured_frames = buffer_info.get('capture_frame_count', 0)
                    capture_progress_ratio = buffer_info.get('capture_progress_ratio', 0.0)
                    camera_fill_ratios.append(capture_progress_ratio)
                    camera_status_lines.append(f"C{cam_num}: {captured_frames} {int(capture_progress_ratio * 100)}%")
                else:
                    # When idle: show buffer capacity with zero percent
                    camera_status_lines.append(f"C{cam_num}: {max_buffer_size} 0%")

            # Draw progress bar only when capturing or dumping
            if camera_fill_ratios:
                # Use average fill ratio for progress bar color and length
                avg_fill_ratio = sum(camera_fill_ratios) / len(camera_fill_ratios)

                if avg_fill_ratio < 0.7:
                    # Green for low fill
                    fill_color = (0, int(255 * avg_fill_ratio / 0.7), 0)
                elif avg_fill_ratio < 0.9:
                    # Yellow to orange for medium fill
                    fill_level = (avg_fill_ratio - 0.7) / 0.2
                    fill_color = (int(255 * fill_level), int(200 * fill_level), 0)
                else:
                    # Red for high fill
                    fill_color = (255, int(100 * (avg_fill_ratio - 0.9) / 0.1), 0)

                # Draw progress fill based on average of both cameras
                fill_width = int(progress_width * avg_fill_ratio)
                if fill_width > 0:
                    pygame.draw.rect(display.menu_screen, fill_color,
                                    (button_x, progress_y, fill_width, progress_height))

            # Show camera status lines (C1: frames %, C2: frames %)
            if camera_status_lines:
                status_text = " | ".join(camera_status_lines)
                fill_text = display.tiny_font.render(status_text, True, (255, 255, 255))

                text_width = fill_text.get_width()
                text_y = progress_y + progress_height + 5
                text_x = button_x + (progress_width - text_width) // 2
                display.menu_screen.blit(fill_text, (text_x, text_y))

        # Status and progress information (to the left of progress bar)
        info_x = button_x - 200
        info_y = progress_y - 5

        # Capture status
        if joystick_state.capture_active:
            status_text = display.small_font.render("REC", True, (255, 0, 0))
            display.menu_screen.blit(status_text, (info_x, info_y))

            # Recording time (would need to track actual time)
            # For now, just show "Recording..." when active
            recording_text = display.tiny_font.render("Recording...", True, (255, 0, 0))
            display.menu_screen.blit(recording_text, (info_x, info_y + 15))
        else:
            status_text = display.small_font.render("Ready", True, (0, 200, 0))
            display.menu_screen.blit(status_text, (info_x, info_y))

        # Dump progress (when dumping)
        if progress > 0:
            if progress < 1.0:
                dump_progress = int(progress * 100)
                dump_text = display.tiny_font.render(f"Dumping: {dump_progress}%", True, (255, 165, 0))
            else:
                dump_text = display.tiny_font.render("Dump Complete!", True, (0, 255, 0))
            display.menu_screen.blit(dump_text, (info_x, info_y + 30))

        # Camera status
        if not camera_connected:
            no_cam_text = display.tiny_font.render("Camera not connected", True, (255, 0, 0))
            display.menu_screen.blit(no_cam_text, (info_x + 20, info_y + 15))

    except Exception as e:
        print(f"Error in render_capture_controls: {e}")
        # Fallback: render simple error message
        error_text = display.tiny_font.render("Capture Error", True, (255, 0, 0))
        display.menu_screen.blit(error_text, (display.sub_x + 10, display.sub_y + display.sub_height // 2 - 60))

# ==============================================================================
# JOYSTICK-LOOP CAMERA CONTROLS (half-height feeds)
# ==============================================================================
# The Joystick Loop view shows the two camera feeds at half height in the bottom
# camera area. The helpers below give those feeds the same camera/view controls
# as the Sensor Calibration view (gain, exposure, gamma, alignment rotation, ROI,
# combined view + opacity, reset/save). A single layout helper is shared by the
# renderer and the event handler so hit-testing never drifts from what is drawn.

# Slider/control geometry constants for the joystick-loop camera area
JL_SLIDER_W = 100          # gain / exposure / gamma slider track width
JL_ROT_SLIDER_W = 120      # alignment-rotation slider track width
JL_GAMMA_MIN = 0.01
JL_GAMMA_MAX = 2.0
JL_ROTATION_RANGE = 90.0   # -90deg .. +90deg


def _joystick_camera_layout(display):
    """Compute every draw/hit rect for the joystick-loop camera controls.

    Returns a dict shared by render_joystick_camera_controls() and
    handle_joystick_camera_control_events() so the two never diverge.
    """
    camera_area_x = display.sub_x + 10
    camera_area_y = display.sub_y + display.sub_height // 2 + 10
    camera_area_width = display.sub_width - 20
    camera_area_height = display.sub_height // 2 - 20

    cam_w = camera_area_width // 2 - 5
    cam_h = camera_area_height

    layout = {
        "camera_area": pygame.Rect(camera_area_x, camera_area_y, camera_area_width, camera_area_height),
        "cameras": [],
    }

    for idx in range(2):
        if idx == 0:
            fx = camera_area_x
        else:
            fx = camera_area_x + camera_area_width // 2 + 5
        fy = camera_area_y

        strip_y = fy + 4
        gain_x = fx + 84
        exp_x = gain_x + JL_SLIDER_W + 30
        gamma_x = exp_x + JL_SLIDER_W + 30

        # ROI size buttons: two columns of three at the top-right of the feed
        roi_rects = []
        roi_col1_x = fx + cam_w - 70
        roi_col2_x = fx + cam_w - 36
        roi_top = fy + 26
        for i in range(6):
            col_x = roi_col1_x if i < 3 else roi_col2_x
            row = i % 3
            roi_rects.append(pygame.Rect(col_x, roi_top + row * 15, 30, 12))

        rot_x = fx + (cam_w - JL_ROT_SLIDER_W) // 2
        rot_y = fy + cam_h - 26

        layout["cameras"].append({
            "fx": fx, "fy": fy, "cam_w": cam_w, "cam_h": cam_h,
            "image_rect": pygame.Rect(fx, fy, cam_w, cam_h),
            # Connect and disconnect share a slot; only the relevant one is shown.
            "connect_rect": pygame.Rect(fx + 4, strip_y, 70, 18),
            "disconnect_rect": pygame.Rect(fx + 4, strip_y, 70, 18),
            "gain_track": pygame.Rect(gain_x, strip_y, JL_SLIDER_W, 18),
            "exposure_track": pygame.Rect(exp_x, strip_y, JL_SLIDER_W, 18),
            "gamma_track": pygame.Rect(gamma_x, strip_y, JL_SLIDER_W, 18),
            "gamma_toggle": pygame.Rect(gamma_x + JL_SLIDER_W + 6, strip_y, 34, 18),
            "roi_rects": roi_rects,
            "rotation_track": pygame.Rect(rot_x, rot_y, JL_ROT_SLIDER_W, 14),
        })

    # Global view controls grouped at the bottom-center, clear of the per-camera
    # ROI buttons that sit at each feed's top-right (the left feed's ROI buttons
    # are right next to the seam, so the seam-top is not free).
    seam_x = camera_area_x + camera_area_width // 2
    bottom = camera_area_y + camera_area_height
    layout["combined_btn"] = pygame.Rect(seam_x - 115, bottom - 48, 110, 18)
    layout["opacity_track"] = pygame.Rect(seam_x + 15, bottom - 44, 100, 14)
    layout["reset_btn"] = pygame.Rect(seam_x - 105, bottom - 24, 100, 20)
    layout["save_btn"] = pygame.Rect(seam_x + 5, bottom - 24, 100, 20)
    return layout


def _process_feed_surface(camera, frame, width, height):
    """Apply gamma + scale + alignment rotation, matching the Sensor Calibration view."""
    processed = frame
    if camera.gamma_enabled:
        processed = apply_gamma_correction(frame, camera.gamma)
    surface = pygame.transform.scale(processed, (width, height))
    if camera.alignment_rotation != 0.0:
        rotated = pygame.transform.rotate(surface, camera.alignment_rotation)
        surface = pygame.Surface((width, height))
        surface.fill((0, 0, 0))
        surface.blit(rotated, (width // 2 - rotated.get_width() // 2,
                               height // 2 - rotated.get_height() // 2))
    return surface


def _draw_feed_roi(screen, camera, image_rect):
    """Draw the green ROI box + crosshair over a feed, matching the Sensor Calibration view."""
    if camera.roi_size is None or camera.roi_size < 0:
        return
    roi_w_pct, roi_h_pct = roi_sizes[camera.roi_size]
    roi_w = int(roi_w_pct * image_rect.width)
    roi_h = int(roi_h_pct * image_rect.height)
    roi_x = int(camera.roi_x * (image_rect.width - roi_w))
    roi_y = int(camera.roi_y * (image_rect.height - roi_h))
    rect = pygame.Rect(image_rect.x + roi_x, image_rect.y + roi_y, roi_w, roi_h)
    pygame.draw.rect(screen, (0, 255, 0), rect, 2)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(screen, (0, 255, 0), (cx - 10, cy), (cx + 10, cy), 1)
    pygame.draw.line(screen, (0, 255, 0), (cx, cy - 10), (cx, cy + 10), 1)


def _draw_feed_timestamps(display, camera, fx, fy, cam_w, cam_h, align):
    """Draw UTC / Local / FPS stacked in a feed's outer-bottom corner.

    The left feed uses its bottom-left corner and the right feed its bottom-right
    corner, so the timestamps never collide with the centered rotation slider or
    the seam-centered combined/opacity/reset/save controls.
    """
    info_font = pygame.font.Font(None, 16)
    lines = [
        f"UTC: {camera.utc_ts}",
        f"Local: {camera.local_ts}",
        f"FPS: {camera.fps:.1f}",
    ]
    line_h = 15
    bottom = fy + cam_h - 4
    for i, line in enumerate(lines):
        surf = info_font.render(line, True, (255, 255, 255))
        y = bottom - (len(lines) - i) * line_h  # stack upward; last line at the bottom
        x = fx + 8 if align == 'left' else fx + cam_w - 8 - surf.get_width()
        display.menu_screen.blit(surf, (x, y))


def _draw_jl_slider(screen, font, track, ratio, label, track_color, handle_color):
    """Draw a compact horizontal slider (track + handle + label above)."""
    line_y = track.centery
    pygame.draw.rect(screen, track_color, (track.x, line_y - 2, track.width, 4))
    handle_x = track.x + int(max(0.0, min(1.0, ratio)) * track.width)
    pygame.draw.rect(screen, handle_color, (handle_x - 4, line_y - 6, 8, 12))
    if label:
        screen.blit(font.render(label, True, (255, 255, 255)), (track.x, track.y - 12))


def _camera_fov_deg(config_state, camera, cam_key, default_focal):
    """Approximate (fov_w_deg, fov_h_deg) from config optics + camera resolution."""
    cc = config_state.camera_configs.get(cam_key, {}) if config_state else {}
    pixel_size_um = float(cc.get('pixel_size', 2.9))
    focal_length_mm = float(cc.get('focal_length', default_focal))
    if focal_length_mm <= 0:
        return 0.0, 0.0
    w = getattr(camera, 'width_res', 0) or 1920
    h = getattr(camera, 'height_res', 0) or 1080
    fov_w = math.degrees(2 * math.atan((w * pixel_size_um * 1e-3) / (2 * focal_length_mm)))
    fov_h = math.degrees(2 * math.atan((h * pixel_size_um * 1e-3) / (2 * focal_length_mm)))
    return fov_w, fov_h


def _cam_rotation(config_state, camera, cam_key):
    """Live camera alignment rotation (deg), falling back to the config value."""
    rot = getattr(camera, 'alignment_rotation', None)
    if rot is not None:
        return float(rot)
    cc = config_state.camera_configs.get(cam_key, {}) if config_state else {}
    return float(cc.get('alignment_rotation', 0.0))


def _draw_cam2_fov_in_cam1(display, joystick_state, image_rect):
    """Overlay camera2's FOV as a red box centered in camera1's frame, sized by the
    FOV ratio and rotated by cam2's alignment rotation relative to cam1. Lets us
    see how close the narrow (cam2) field is to centered in the finder (cam1)."""
    cfg = joystick_state.config_state
    if cfg is None:
        return
    c1 = camera_manager.get_camera(0)
    c2 = camera_manager.get_camera(1)
    fov1 = _camera_fov_deg(cfg, c1, 'camera1', 162.0)
    fov2 = _camera_fov_deg(cfg, c2, 'camera2', 2000.0)
    if fov1[0] <= 0 or fov1[1] <= 0 or fov2[0] <= 0:
        return
    hw = max(3.0, (fov2[0] / fov1[0]) * image_rect.width / 2.0)
    hh = max(3.0, (fov2[1] / fov1[1]) * image_rect.height / 2.0)
    rel = math.radians(_cam_rotation(cfg, c2, 'camera2') - _cam_rotation(cfg, c1, 'camera1'))
    ct, st = math.cos(rel), math.sin(rel)
    cx, cy = image_rect.center
    pts = [(int(cx + x * ct - y * st), int(cy + x * st + y * ct))
           for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    pygame.draw.polygon(display.menu_screen, (255, 0, 0), pts, 2)
    lbl = display.tiny_font.render("Cam2 FOV", True, (255, 90, 90))
    display.menu_screen.blit(lbl, (cx + hw + 3, cy - hh - 2))


def _draw_feed_axes(display, joystick_state, image_rect, camera, cam_key):
    """Draw HUD axes from the frame center: the +along-track (IT) and +cross-track
    (CT) bias-direction axes, plus a target-position axis (TGT) pointing to where
    the tracked target currently sits in the image (from the live tracking error,
    so its tip lands on the spot). A central GAP keeps the boresight/target itself
    un-obscured -- without it the converging axes hid a centered target. Only
    shown while tracking."""
    cfg = joystick_state.config_state
    if cfg is None or joystick_state.tracking_mode not in (
            TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT):
        return
    surf = display.menu_screen
    cx, cy = image_rect.center
    span = min(image_rect.width, image_rect.height)
    gap = 18  # clear radius around boresight so the target spot stays visible
    x_sign = float(getattr(cfg, 'hotspot_x_sign', 1.0)) or 1.0
    y_sign = float(getattr(cfg, 'hotspot_y_sign', -1.0)) or -1.0
    altaz = getattr(cfg, 'mount_mode', 'AltAz') == 'AltAz'
    el = getattr(joystick_state, 'target_el_deg', 0.0)
    cos_el = max(math.cos(math.radians(el)), 0.087)

    def draw_axis(dx, dy, length, color, label):
        n = math.hypot(dx, dy)
        if n < 1e-9 or length <= gap + 4:
            return
        ux, uy = dx / n, dy / n
        sx, sy = cx + ux * gap, cy + uy * gap          # start past the central gap
        ex, ey = cx + ux * length, cy + uy * length
        pygame.draw.line(surf, color, (sx, sy), (ex, ey), 2)
        ang = math.atan2(ey - cy, ex - cx)
        for da in (2.618, -2.618):  # +/-150 deg arrowhead
            pygame.draw.line(surf, color, (ex, ey),
                             (ex + 8 * math.cos(ang + da), ey + 8 * math.sin(ang + da)), 2)
        surf.blit(display.tiny_font.render(label, True, color), (ex + 3, ey - 6))

    # Bias direction axes (along/cross-track), from the target's sky velocity.
    # Feed is rotation-corrected by alignment_rotation, so only signs apply here.
    vx = getattr(joystick_state, 'target_az_rate', 0.0) * cos_el  # on-sky cross-el
    vy = getattr(joystick_state, 'target_el_rate', 0.0)           # on-sky el
    vn = math.hypot(vx, vy)
    if vn >= 1e-4:
        ux, uy = vx / vn, vy / vn
        bias_len = 0.26 * span
        draw_axis(ux / x_sign, uy / y_sign, bias_len, (255, 230, 80), "+IT")
        draw_axis(-uy / x_sign, ux / y_sign, bias_len, (80, 230, 255), "+CT")

    # Target-position axis: map the live tracking position error to the target's
    # image offset (true pixels, scaled to the displayed feed) so the arrow tip
    # lands on the spot. Shrinks to nothing as you center up.
    az_err = getattr(joystick_state, 'azm_position_error', 0.0)
    alt_err = getattr(joystick_state, 'alt_position_error', 0.0)
    sky_el_err = -alt_err if altaz else alt_err          # ALT axis runs opposite el
    cross_el = az_err * cos_el
    pix = float(cfg.get_camera_pixel_size(cam_key))
    foc = float(cfg.get_camera_focal_length(cam_key))
    ifov = math.degrees(pix * 1e-3 / foc) if foc > 0 else 0.0
    if ifov > 0:
        raw_w = getattr(camera, 'width_res', 0) or 1920
        scale = image_rect.width / float(raw_w)
        tdx = (cross_el / ifov) / x_sign * scale
        tdy = (sky_el_err / ifov) / y_sign * scale
        tlen = math.hypot(tdx, tdy)
        if tlen > gap + 4:
            draw_axis(tdx, tdy, min(tlen, 0.46 * span), (255, 90, 255), "TGT")


def render_camera_feeds(display, joystick_state=None):
    """Render camera feeds with cross-hairs using existing camera manager"""
    try:
        # Update camera frames
        update_camera_frames_from_buffers()

        # Safely get camera status
        camera1_connected = False
        camera2_connected = False
        camera1_frame = None
        camera2_frame = None

        # Safely access cameras with bounds checking and get time/fps info
        if len(camera_manager.cameras) > 0:
            camera1 = camera_manager.cameras[0]
            camera1_connected = camera1.connected
            camera1_frame = camera1.frame

        if len(camera_manager.cameras) > 1:
            camera2 = camera_manager.cameras[1]
            camera2_connected = camera2.connected
            camera2_frame = camera2.frame

        display_width = display.sub_width - 20
        display_height = display.sub_height // 2 - 20  # Bottom half minus some margin

        # Create surface for camera area (bottom half)
        camera_area_x = display.sub_x + 10
        camera_area_y = display.sub_y + display.sub_height // 2 + 10
        camera_area_width = display.sub_width - 20
        camera_area_height = display.sub_height // 2 - 20

        display.menu_screen.fill((0, 0, 0), (camera_area_x, camera_area_y,
                                            camera_area_width, camera_area_height))

        # Check if any cameras are expected to be present
        num_available = camera_manager.get_num_cameras()
        if num_available == 0:
            # No cameras detected at all
            no_cams_text = display.small_font.render("No cameras detected", True, (255, 0, 0))
            text_rect = no_cams_text.get_rect(center=(display.sub_x + display.sub_width // 2,
                                                     display.sub_y + display.sub_height * 3 // 4))
            display.menu_screen.blit(no_cams_text, text_rect)

            if joystick_state:
                # Send status update (we'll use print for now since no callback is passed)
                print("No cameras detected - joystick mode camera features will not function")
            return

        # Combined view: overlay both feeds (opacity-blended) across the full camera
        # area, matching the Sensor Calibration view's combined mode.
        combined_active = (camera_manager.combined_view_toggle and
                           camera1_connected and camera2_connected and
                           camera1_frame is not None and camera2_frame is not None)
        if combined_active:
            try:
                comb_w = camera_area_width
                comb_h = camera_area_height
                surf1 = _process_feed_surface(camera1, camera1_frame, comb_w, comb_h).convert_alpha()
                surf2 = _process_feed_surface(camera2, camera2_frame, comb_w, comb_h).convert_alpha()
                opacity = camera_manager.camera_opacities[0] if camera_manager.camera_opacities else 0.5
                surf1.set_alpha(int(opacity * 255))
                surf2.set_alpha(int((1 - opacity) * 255))
                display.menu_screen.blit(surf1, (camera_area_x, camera_area_y))
                display.menu_screen.blit(surf2, (camera_area_x, camera_area_y))

                center_x = camera_area_x + comb_w // 2
                center_y = camera_area_y + comb_h // 2
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - 20, center_y), (center_x + 20, center_y), 1)
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - 20), (center_x, center_y + 20), 1)
                combined_label = display.small_font.render(f"Combined View - Opacity: {opacity:.1f}", True, (0, 200, 200))
                display.menu_screen.blit(combined_label, (camera_area_x + 10, camera_area_y + 10))
            except Exception as e:
                print(f"Error rendering combined joystick feed: {e}")
                combined_active = False

        # Camera 1 (left half of camera area)
        cam1_width = camera_area_width // 2 - 5
        cam1_height = camera_area_height

        if not combined_active and camera1_connected and camera1_frame is not None:
            try:
                cam1_scaled = _process_feed_surface(camera1, camera1_frame, cam1_width, cam1_height)
                display.menu_screen.blit(cam1_scaled, (camera_area_x, camera_area_y))
                _draw_feed_roi(display.menu_screen, camera1,
                               pygame.Rect(camera_area_x, camera_area_y, cam1_width, cam1_height))

                # Draw cross-hair at center
                center_x = camera_area_x + cam1_width // 2
                center_y = camera_area_y + cam1_height // 2
                crosshair_length = 20
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y),
                                (center_x + crosshair_length, center_y), 1)  # Horizontal
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length),
                                (center_x, center_y + crosshair_length), 1)  # Vertical

                # Camera 1 label
                cam1_text = display.small_font.render("Camera 1", True, (255, 255, 255))
                display.menu_screen.blit(cam1_text, (camera_area_x + 10, camera_area_y + 10))

                # Time/FPS stacked in the left feed's bottom-left corner (clear of the
                # centered rotation slider and the seam-centered view controls)
                _draw_feed_timestamps(display, camera1, camera_area_x, camera_area_y, cam1_width, cam1_height, 'left')

                # Overlays: camera2's FOV box (centering aid) + bias-direction and
                # target-position HUD axes (with a central gap so they don't hide
                # a centered target).
                if joystick_state is not None:
                    cam1_rect = pygame.Rect(camera_area_x, camera_area_y, cam1_width, cam1_height)
                    _draw_cam2_fov_in_cam1(display, joystick_state, cam1_rect)
                    _draw_feed_axes(display, joystick_state, cam1_rect, camera1, 'camera1')
                    draw_solve_centroids_on_feed(display, joystick_state, cam1_rect, 0)
            except Exception as e:
                error_text = display.small_font.render("Camera 1 Error", True, (255, 0, 0))
                display.menu_screen.blit(error_text, (camera_area_x + 10, camera_area_y + 10))
                if joystick_state:
                    print(f"Error rendering Camera 1: {e}")
        elif len(camera_manager.cameras) > 0 and not camera1_connected:
            # Camera available but not connected
            not_connected_text = display.small_font.render("Not Connected", True, (255, 0, 0))
            text_rect = not_connected_text.get_rect(center=(camera_area_x + cam1_width // 2,
                                                           camera_area_y + cam1_height // 2))
            display.menu_screen.blit(not_connected_text, text_rect)

        # Camera 2 (right half of camera area)
        cam2_x = camera_area_x + camera_area_width // 2 + 5
        cam2_width = camera_area_width // 2 - 5
        cam2_height = camera_area_height

        if not combined_active and camera2_connected and camera2_frame is not None:
            try:
                cam2_scaled = _process_feed_surface(camera2, camera2_frame, cam2_width, cam2_height)
                display.menu_screen.blit(cam2_scaled, (cam2_x, camera_area_y))
                _draw_feed_roi(display.menu_screen, camera2,
                               pygame.Rect(cam2_x, camera_area_y, cam2_width, cam2_height))

                # Draw cross-hair at center
                center_x = cam2_x + cam2_width // 2
                center_y = camera_area_y + cam2_height // 2
                crosshair_length = 20
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y),
                                (center_x + crosshair_length, center_y), 1)  # Horizontal
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length),
                                (center_x, center_y + crosshair_length), 1)  # Vertical

                # Camera 2 label
                cam2_text = display.small_font.render("Camera 2", True, (255, 255, 255))
                display.menu_screen.blit(cam2_text, (cam2_x + 10, camera_area_y + 10))

                # Time/FPS stacked in the right feed's bottom-right corner (clear of the
                # centered rotation slider and the seam-centered view controls)
                _draw_feed_timestamps(display, camera2, cam2_x, camera_area_y, cam2_width, cam2_height, 'right')

                # Bias-direction + target-position HUD axes (camera2 only -- the
                # FOV box is a finder/cam1 aid).
                if joystick_state is not None:
                    _draw_feed_axes(display, joystick_state,
                                    pygame.Rect(cam2_x, camera_area_y, cam2_width, cam2_height),
                                    camera2, 'camera2')
            except Exception as e:
                error_text = display.small_font.render("Camera 2 Error", True, (255, 0, 0))
                display.menu_screen.blit(error_text, (cam2_x + 10, camera_area_y + 10))
                if joystick_state:
                    print(f"Error rendering Camera 2: {e}")
        elif len(camera_manager.cameras) > 1 and not camera2_connected:
            # Camera available but not connected
            not_connected_text = display.small_font.render("Not Connected", True, (255, 0, 0))
            text_rect = not_connected_text.get_rect(center=(cam2_x + cam2_width // 2,
                                                           camera_area_y + cam2_height // 2))
            display.menu_screen.blit(not_connected_text, text_rect)
        elif len(camera_manager.cameras) == 1 and num_available >= 2:
            # Second camera slot available but camera not present
            missing_text = display.small_font.render("Camera 2 Missing", True, (255, 165, 0))
            text_rect = missing_text.get_rect(center=(cam2_x + cam2_width // 2,
                                                     camera_area_y + cam2_height // 2))
            display.menu_screen.blit(missing_text, text_rect)

        # Camera and view controls overlaid on the half-height feeds
        render_joystick_camera_controls(display, joystick_state)

    except Exception as e:
        # Catch any unexpected errors and display gracefully
        display.menu_screen.fill((0, 0, 0), (display.sub_x + 10, display.sub_y + display.sub_height // 2 + 10,
                                            display.sub_width - 20, display.sub_height // 2 - 20))

        error_text = display.small_font.render(f"Camera Error: {str(e)[:40]}", True, (255, 0, 0))
        text_rect = error_text.get_rect(center=(display.sub_x + display.sub_width // 2,
                                               display.sub_y + display.sub_height * 3 // 4))
        display.menu_screen.blit(error_text, text_rect)
        print(f"Camera rendering error in joystick mode: {e}")


def render_joystick_camera_controls(display, joystick_state=None):
    """Render the camera/view controls overlaid on the half-height joystick-loop feeds.

    Mirrors the Sensor Calibration view's control set: per-camera connect/disconnect,
    gain, exposure, gamma (slider + toggle) and alignment-rotation sliders, ROI size
    buttons, plus the global combined-view toggle, opacity slider and reset/save config.
    """
    try:
        screen = display.menu_screen
        mouse_pos = pygame.mouse.get_pos()
        font = pygame.font.Font(None, 14)
        layout = _joystick_camera_layout(display)

        camera1 = camera_manager.get_camera(0)
        camera2 = camera_manager.get_camera(1)
        cameras = [camera1, camera2]

        for idx, camera in enumerate(cameras):
            if camera is None:
                continue
            cam = layout["cameras"][idx]

            # Connect / Disconnect button (one shown depending on state)
            if camera.connected:
                rect = cam["disconnect_rect"]
                hovered = rect.collidepoint(mouse_pos)
                color = (255, 100, 100) if hovered else (200, 70, 70)
                pygame.draw.rect(screen, color, rect)
                screen.blit(font.render("Disconnect", True, (255, 255, 255)),
                            font.render("Disconnect", True, (255, 255, 255)).get_rect(center=rect.center))
            else:
                rect = cam["connect_rect"]
                hovered = rect.collidepoint(mouse_pos)
                color = (100, 100, 255) if hovered else (70, 70, 200)
                pygame.draw.rect(screen, color, rect)
                screen.blit(font.render("Connect", True, (255, 255, 255)),
                            font.render("Connect", True, (255, 255, 255)).get_rect(center=rect.center))
                # No further per-camera controls when disconnected
                continue

            # Gain slider
            max_gain = camera.prop.get('MaxGain', 500) if camera.prop else 500
            gain_ratio = min(1.0, camera.gain / max_gain) if max_gain else 0.0
            _draw_jl_slider(screen, font, cam["gain_track"], gain_ratio,
                            f"Gain {camera.gain}", (100, 100, 100), (200, 0, 0))

            # Exposure slider (logarithmic)
            max_exp = camera.prop.get('MaxExposure', 500000) if camera.prop else 500000
            min_exp = 1
            if camera.exposure > 0 and max_exp > min_exp:
                exp_ratio = math.log10(camera.exposure / min_exp) / math.log10(max_exp / min_exp)
                exp_ratio = min(1.0, max(0.0, exp_ratio))
            else:
                exp_ratio = 0.0
            exp_us = camera.exposure
            if exp_us < 1000:
                exp_val = f"{exp_us:g}us"
            elif exp_us < 1000000:
                exp_val = f"{exp_us / 1000.0:.1f}ms"
            else:
                exp_val = f"{exp_us / 1000000.0:.1f}s"
            _draw_jl_slider(screen, font, cam["exposure_track"], exp_ratio,
                            f"Exp {exp_val}", (100, 100, 100), (0, 200, 0))

            # Gamma slider + toggle
            gamma_ratio = (camera.gamma - JL_GAMMA_MIN) / (JL_GAMMA_MAX - JL_GAMMA_MIN)
            _draw_jl_slider(screen, font, cam["gamma_track"], gamma_ratio,
                            f"Gamma {camera.gamma:.2f}", (150, 150, 150), (200, 200, 255))
            toggle = cam["gamma_toggle"]
            toggle_color = (0, 150, 0) if camera.gamma_enabled else (150, 0, 0)
            if toggle.collidepoint(mouse_pos):
                toggle_color = tuple(min(255, c + 50) for c in toggle_color)
            pygame.draw.rect(screen, toggle_color, toggle)
            t_surf = font.render("ON" if camera.gamma_enabled else "OFF", True, (255, 255, 255))
            screen.blit(t_surf, t_surf.get_rect(center=toggle.center))

            # Alignment rotation slider (bottom-center)
            rot = cam["rotation_track"]
            rot_ratio = (camera.alignment_rotation + JL_ROTATION_RANGE) / (2 * JL_ROTATION_RANGE)
            _draw_jl_slider(screen, font, rot, rot_ratio,
                            f"Rot {camera.alignment_rotation:+.1f}", (50, 50, 150), (150, 150, 255))
            # Center marker at 0 degrees
            center_marker_x = rot.x + rot.width // 2
            pygame.draw.line(screen, (255, 255, 255),
                             (center_marker_x, rot.centery - 5), (center_marker_x, rot.centery + 5), 1)

            # ROI size buttons
            for i, roi_rect in enumerate(cam["roi_rects"]):
                is_selected = (camera.roi_size == i)
                is_hovered = roi_rect.collidepoint(mouse_pos)
                if is_selected:
                    pygame.draw.rect(screen, (0, 100, 0), roi_rect)
                    pygame.draw.rect(screen, (0, 255, 0), roi_rect, 1)
                    text_color = (255, 255, 255)
                elif is_hovered:
                    pygame.draw.rect(screen, (100, 100, 100), roi_rect)
                    pygame.draw.rect(screen, (200, 200, 200), roi_rect, 1)
                    text_color = (255, 255, 0)
                else:
                    pygame.draw.rect(screen, (70, 70, 70), roi_rect)
                    pygame.draw.rect(screen, (150, 150, 150), roi_rect, 1)
                    text_color = (255, 255, 255)
                r_surf = font.render(roi_label_texts[i], True, text_color)
                screen.blit(r_surf, r_surf.get_rect(center=roi_rect.center))

        # Global view controls -----------------------------------------------
        # Combined view toggle
        combined_btn = layout["combined_btn"]
        is_toggled = camera_manager.combined_view_toggle
        is_hovered = combined_btn.collidepoint(mouse_pos)
        if is_toggled:
            btn_color = (50, 150, 50) if is_hovered else (0, 100, 0)
        else:
            btn_color = (100, 100, 100) if is_hovered else (70, 70, 70)
        pygame.draw.rect(screen, btn_color, combined_btn)
        pygame.draw.rect(screen, (150, 150, 150), combined_btn, 1)
        c_surf = font.render("Combined View", True, (255, 255, 255))
        screen.blit(c_surf, c_surf.get_rect(center=combined_btn.center))

        # Opacity slider (only when both cameras connected)
        if camera1 and camera2 and camera1.connected and camera2.connected:
            opacity = camera_manager.camera_opacities[0] if camera_manager.camera_opacities else 0.5
            _draw_jl_slider(screen, font, layout["opacity_track"], opacity,
                            f"Opacity {opacity:.1f}", (100, 100, 100), (255, 255, 0))

        # Reset / Save config buttons
        reset_btn = layout["reset_btn"]
        pygame.draw.rect(screen, (100, 70, 70) if reset_btn.collidepoint(mouse_pos) else (70, 50, 50), reset_btn)
        pygame.draw.rect(screen, (150, 150, 150), reset_btn, 1)
        rs_surf = font.render("Reset", True, (255, 255, 255))
        screen.blit(rs_surf, rs_surf.get_rect(center=reset_btn.center))

        save_btn = layout["save_btn"]
        pygame.draw.rect(screen, (70, 100, 70) if save_btn.collidepoint(mouse_pos) else (50, 70, 50), save_btn)
        pygame.draw.rect(screen, (150, 150, 150), save_btn, 1)
        sv_surf = font.render("Save", True, (255, 255, 255))
        screen.blit(sv_surf, sv_surf.get_rect(center=save_btn.center))

    except Exception as e:
        print(f"Error rendering joystick camera controls: {e}")


def _apply_jl_camera_drag(current_pos, display):
    """Apply slider drags for the joystick-loop camera controls (gain/exposure/gamma/rotation/opacity)."""
    layout = _joystick_camera_layout(display)
    for idx in range(2):
        camera = camera_manager.get_camera(idx)
        if camera is None or not camera.connected:
            continue
        cam = layout["cameras"][idx]

        # Gain
        track = cam["gain_track"]
        if track.collidepoint(current_pos):
            max_gain = camera.prop.get('MaxGain', 500) if camera.prop else 500
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            camera_manager.set_camera_gain(idx, int((rel / track.width) * max_gain))

        # Exposure (logarithmic)
        track = cam["exposure_track"]
        if track.collidepoint(current_pos):
            max_exp = camera.prop.get('MaxExposure', 500000) if camera.prop else 500000
            min_exp = 1
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            slider_pos = rel / track.width
            if max_exp > min_exp:
                new_exp = int(min_exp * (10 ** (slider_pos * math.log10(max_exp / min_exp))))
                new_exp = max(min_exp, min(new_exp, max_exp))
            else:
                new_exp = max_exp
            camera_manager.set_camera_exposure(idx, new_exp)

        # Gamma
        track = cam["gamma_track"]
        if track.collidepoint(current_pos):
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            new_gamma = JL_GAMMA_MIN + (rel / track.width) * (JL_GAMMA_MAX - JL_GAMMA_MIN)
            camera.gamma = round(max(JL_GAMMA_MIN, min(JL_GAMMA_MAX, new_gamma)), 2)

        # Alignment rotation
        track = cam["rotation_track"]
        if track.collidepoint(current_pos):
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            new_rot = ((rel / track.width) - 0.5) * (2 * JL_ROTATION_RANGE)
            camera.alignment_rotation = max(-JL_ROTATION_RANGE, min(JL_ROTATION_RANGE, new_rot))

    # Opacity (global, only when both cameras connected)
    c1 = camera_manager.get_camera(0)
    c2 = camera_manager.get_camera(1)
    if c1 and c2 and c1.connected and c2.connected:
        track = layout["opacity_track"]
        if track.collidepoint(current_pos):
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            camera_manager.camera_opacities[0] = max(0.0, min(1.0, rel / track.width))


def _jl_pos_over_control(layout, idx, pos):
    """True if pos hits any control overlaying the feed (so an image click is not an ROI-origin set)."""
    cam = layout["cameras"][idx]
    rects = [cam["connect_rect"], cam["disconnect_rect"], cam["gain_track"],
             cam["exposure_track"], cam["gamma_track"], cam["gamma_toggle"],
             cam["rotation_track"]] + cam["roi_rects"]
    rects += [layout["combined_btn"], layout["opacity_track"], layout["reset_btn"], layout["save_btn"]]
    return any(r.collidepoint(pos) for r in rects)


def _jl_reset_camera_config(config_state, update_status_callback):
    """Reset both cameras' settings to config-file defaults (mirrors Sensor Calibration reset)."""
    if not config_state:
        update_status_callback("Error: Config state not available")
        return
    c0 = camera_manager.get_camera(0)
    c1 = camera_manager.get_camera(1)
    c0.alignment_rotation = float(config_state.get_camera_alignment_rotation("camera1") or 0.0)
    c0.gain = int(float(config_state.get_camera_gain("camera1") or 1))
    c0.exposure = int(float(config_state.get_camera_exposure("camera1") or 10000))
    c0.gamma = float(config_state.camera_configs["camera1"].get("gamma", 0.1))
    c0.gamma_enabled = bool(config_state.camera_configs["camera1"].get("gamma_enabled", False))
    c1.alignment_rotation = float(config_state.get_camera_alignment_rotation("camera2") or 0.0)
    c1.gain = int(float(config_state.get_camera_gain("camera2") or 1))
    c1.exposure = int(float(config_state.get_camera_exposure("camera2") or 10000))
    c1.gamma = float(config_state.camera_configs["camera2"].get("gamma", 0.1))
    c1.gamma_enabled = bool(config_state.camera_configs["camera2"].get("gamma_enabled", False))
    if c0.connected and c0.cap:
        camera_manager.set_camera_gain(0, c0.gain, update_status_callback)
        camera_manager.set_camera_exposure(0, c0.exposure, update_status_callback)
    if c1.connected and c1.cap:
        camera_manager.set_camera_gain(1, c1.gain, update_status_callback)
        camera_manager.set_camera_exposure(1, c1.exposure, update_status_callback)
    update_status_callback("Camera settings reset to config file defaults")


def _jl_save_camera_config(config_state, update_status_callback):
    """Save both cameras' current settings to config.json (mirrors Sensor Calibration save)."""
    if not config_state:
        update_status_callback("Error: Config state not available")
        return
    c0 = camera_manager.get_camera(0)
    c1 = camera_manager.get_camera(1)
    config_state.camera_configs["camera1"]["alignment_rotation"] = c0.alignment_rotation
    config_state.camera_configs["camera1"]["gain"] = c0.gain
    config_state.camera_configs["camera1"]["exposure"] = c0.exposure
    config_state.camera_configs["camera1"]["gamma"] = c0.gamma
    config_state.camera_configs["camera1"]["gamma_enabled"] = c0.gamma_enabled
    config_state.camera_configs["camera2"]["alignment_rotation"] = c1.alignment_rotation
    config_state.camera_configs["camera2"]["gain"] = c1.gain
    config_state.camera_configs["camera2"]["exposure"] = c1.exposure
    config_state.camera_configs["camera2"]["gamma"] = c1.gamma
    config_state.camera_configs["camera2"]["gamma_enabled"] = c1.gamma_enabled
    config_state.save_to_file()
    update_status_callback("Camera settings saved to config.json")


def handle_joystick_camera_control_events(event, display, config_state=None, update_status_callback=None):
    """Handle mouse events for the joystick-loop half-height camera controls.

    Returns True if the event was consumed by a camera/view control."""
    if update_status_callback is None:
        update_status_callback = print

    if event.type == pygame.MOUSEMOTION:
        if event.buttons[0]:
            _apply_jl_camera_drag(event.pos, display)
        return False

    if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
        return False

    pos = event.pos
    layout = _joystick_camera_layout(display)

    for idx in range(2):
        camera = camera_manager.get_camera(idx)
        if camera is None:
            continue
        cam = layout["cameras"][idx]

        # Connect / Disconnect
        if not camera.connected:
            if cam["connect_rect"].collidepoint(pos):
                camera_manager.connect_camera(idx, update_status_callback)
                return True
            continue  # no other per-camera controls while disconnected
        if cam["disconnect_rect"].collidepoint(pos):
            camera_manager.disconnect_camera(idx, update_status_callback)
            return True

        # Gamma toggle
        if cam["gamma_toggle"].collidepoint(pos):
            camera.gamma_enabled = not camera.gamma_enabled
            return True

        # Slider clicks set the value immediately (then drag continues to adjust)
        if (cam["gain_track"].collidepoint(pos) or cam["exposure_track"].collidepoint(pos) or
                cam["gamma_track"].collidepoint(pos) or cam["rotation_track"].collidepoint(pos)):
            _apply_jl_camera_drag(pos, display)
            return True

        # ROI size buttons
        for i, roi_rect in enumerate(cam["roi_rects"]):
            if roi_rect.collidepoint(pos):
                if camera.roi_size == i:
                    camera.roi_size = -1
                    camera.roi_x = 0.5
                    camera.roi_y = 0.5
                    camera_manager.set_camera_roi(idx, update_status_callback, None, None, -1)
                else:
                    was_first = (camera.roi_size == -1)
                    camera.roi_size = i
                    if was_first:
                        camera.roi_x = 0.5
                        camera.roi_y = 0.5
                    camera_manager.set_camera_roi(idx, update_status_callback, camera.roi_x, camera.roi_y, i)
                return True

    # Global view controls
    if layout["combined_btn"].collidepoint(pos):
        camera_manager.combined_view_toggle = not camera_manager.combined_view_toggle
        return True

    c1 = camera_manager.get_camera(0)
    c2 = camera_manager.get_camera(1)
    if c1 and c2 and c1.connected and c2.connected and layout["opacity_track"].collidepoint(pos):
        track = layout["opacity_track"]
        rel = min(max(pos[0] - track.x, 0), track.width)
        camera_manager.camera_opacities[0] = max(0.0, min(1.0, rel / track.width))
        return True

    if layout["reset_btn"].collidepoint(pos):
        _jl_reset_camera_config(config_state, update_status_callback)
        return True
    if layout["save_btn"].collidepoint(pos):
        _jl_save_camera_config(config_state, update_status_callback)
        return True

    # ROI origin selection by clicking on a connected feed (skip if over any control)
    for idx in range(2):
        camera = camera_manager.get_camera(idx)
        if camera is None or not camera.connected or camera.frame is None:
            continue
        cam = layout["cameras"][idx]
        image_rect = cam["image_rect"]
        if not image_rect.collidepoint(pos) or _jl_pos_over_control(layout, idx, pos):
            continue
        camera.roi_x = max(0.0, min(1.0, (pos[0] - image_rect.x) / image_rect.width))
        camera.roi_y = max(0.0, min(1.0, (pos[1] - image_rect.y) / image_rect.height))
        if camera.roi_size >= 0:
            camera_manager.set_camera_roi(idx, update_status_callback, camera.roi_x, camera.roi_y, camera.roi_size)
        return True

    return False

# ==============================================================================
# JOYSTICK MODE EVENT HANDLING
# ==============================================================================

def handle_joystick_mode_mouse_events(event, joystick_state, display, tracking_vis_state, config_state, current_tracking_surface):
    """Handle mouse events specific to joystick mode"""
    mouse_pos = pygame.mouse.get_pos()
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = event.pos

        # Connect button
        connect_rect = pygame.Rect(display.sub_x + 10, display.sub_y + 10, 80, 30)
        if connect_rect.collidepoint(pos) and not joystick_state.telescope_connected:
            success = joystick_state.connect_telescope()
            if success:
                print("Telescope connected")

        # Disconnect button
        disconnect_rect = pygame.Rect(display.sub_x + 100, display.sub_y + 10, 80, 30)
        if disconnect_rect.collidepoint(pos) and joystick_state.telescope_connected:
            joystick_state.disconnect_telescope()
            print("Telescope disconnected")

        # Port dropdown (simplified - click to cycle through ports)
        dropdown_rect = pygame.Rect(display.sub_x + 50, display.sub_y + 50, 120, 25)
        if dropdown_rect.collidepoint(pos):
            joystick_state.get_available_serial_ports()
            if joystick_state.available_ports:
                # Force refresh and cycle to next port
                current_index = 0
                if joystick_state.selected_port:
                    for i, port in enumerate(joystick_state.available_ports):
                        if port['device'] == joystick_state.selected_port:
                            current_index = i
                            break

                next_index = (current_index + 1) % len(joystick_state.available_ports)
                joystick_state.selected_port = joystick_state.available_ports[next_index]['device']
                print(f"Selected port: {joystick_state.selected_port}")

        # Handle satellite selection/hover in polar plot area
        quadrant_x = display.sub_x + display.sub_width // 2
        quadrant_y = display.sub_y
        quadrant_width = display.sub_width // 2
        quadrant_height = display.sub_height // 2

        quadrant_rect = pygame.Rect(quadrant_x, quadrant_y, quadrant_width, quadrant_height)
        if quadrant_rect.collidepoint(pos):
            # Mouse is over polar plot quadrant - check for satellite hover/selection
            hovered_sat = None

            # Debug: print mouse position on click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                print(f"Click at ({pos[0]}, {pos[1]}) in polar plot quadrant")

            # Pre-calculate centers and scale factor to match draw_satellites exactly
            full_screen_center_x = display.sub_x + display.sub_width // 2
            full_screen_center_y = display.sub_y + display.sub_height // 2
            quadrant_center_x = display.sub_x + display.sub_width // 2
            quadrant_center_y = display.sub_y + display.sub_height // 2
            scale_factor = 0.45

            for sat, (px, py, alt, _) in tracking_vis_state.satellite_positions.items():
                # Exact coordinate transformation matching draw_satellites
                rel_x = px - full_screen_center_x
                rel_y = py - full_screen_center_y
                trans_x = quadrant_center_x + rel_x * scale_factor
                trans_y = quadrant_center_y + rel_y * scale_factor

                if alt > 0:  # Only consider satellites above horizon
                    # Debug satellite positions on click
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        print(f"Satellite {sat.name}: orig({px},{py}) -> trans({trans_x},{trans_y})")

                    # Check if mouse is over satellite (larger hit area for easier clicking)
                    dist_to_sat = math.sqrt((pos[0] - trans_x)**2 + (pos[1] - trans_y)**2)
                    if dist_to_sat <= 15:  # 15 pixel radius for easier clicking/hovering
                        hovered_sat = sat
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            print(f"  -> Hit! Distance: {dist_to_sat}")
                        break

            # Update hover state on motion
            if event.type == pygame.MOUSEMOTION:
                tracking_vis_state.hovered_satellite = hovered_sat

            # Handle satellite selection on click. While a launch is active it is
            # the sole target, so plot clicks must not select/track a satellite
            # (this also covers a launch-button press that lands over the plot).
            launch_active = bool(getattr(tracking_vis_state, 'selected_launch', None)
                                 and getattr(tracking_vis_state, 'launch_launched', False))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not launch_active:
                if hovered_sat is not None:
                    if tracking_vis_state.selected_satellite == hovered_sat:
                        tracking_vis_state.selected_satellite = None  # Deselect if clicking same
                        print("Deselected satellite")
                    else:
                        tracking_vis_state.selected_satellite = hovered_sat  # Select new satellite
                        print(f"Selected satellite: {hovered_sat.name}")
                else:
                    print("  -> Clicked empty area")
                    # Click in empty area - deselect current selection
                    tracking_vis_state.selected_satellite = None
                    print("Deselected satellite (empty area clicked)")
        else:
            # Mouse not over polar plot area - clear hover state
            if event.type == pygame.MOUSEMOTION:
                tracking_vis_state.hovered_satellite = None

        # Capture button (mouse click)
        if (hasattr(joystick_state, 'capture_button_rect') and
            joystick_state.capture_button_rect and
            joystick_state.capture_button_rect.collidepoint(pos)):
            _handle_capture_toggle(joystick_state, tracking_vis_state, config_state, current_tracking_surface)

        # Handle bias control button clicks
        if handle_bias_control_mouse_events(joystick_state, mouse_pos):
            return True

        # Handle feed-forward toggle button clicks
        if handle_ff_toggle_mouse_events(joystick_state, mouse_pos):
            return True

        # Handle plate-solve pane clicks (toggle / apply alignment)
        if handle_plate_solve_mouse_events(joystick_state, mouse_pos, config_state):
            return True

        # Handle joystick launch button clicks
        if (hasattr(display, 'joystick_launch_button') and
            display.joystick_launch_button and
            display.joystick_launch_button.collidepoint(pos)):
            # Handle launch button click - same logic as main tracking mode
            if hasattr(tracking_vis_state, 'selected_launch') and tracking_vis_state.selected_launch:
                if tracking_vis_state.launch_launched:
                    # Turn off launch visualization and reset to start
                    tracking_vis_state.launch_launched = False
                    tracking_vis_state.launch_start_time = None
                    print("Launch visualization stopped")
                else:
                    # Start launch visualization from current time
                    from skyfield.api import load
                    tracking_vis_state.launch_launched = True
                    tracking_vis_state.launch_start_time = load.timescale().now().tt
                    print(f"Launch visualization started for {tracking_vis_state.selected_launch}")
                return True

        # Handle PID slider clicks
        if handle_pid_sliders_mouse_events(joystick_state, display, mouse_pos):
            return True

        # Handle Lead-time slider clicks
        if handle_lead_slider_mouse_events(joystick_state, display, mouse_pos):
            return True

def _handle_capture_toggle(joystick_state, tracking_vis_state, config_state, tracking_surface=None):
    """Handle capture toggle from UI or joystick"""
    from camera_manager import camera_manager
    from capture_manager import capture_manager

    if joystick_state.capture_active:
        # Stop capture and begin dump process for all cameras
        capture_manager.stop_capture(None, tracking_vis_state, tracking_vis_state.selected_satellite, config_state, tracking_surface)
        print("Capture stopped on all cameras, dump process started")
        joystick_state.capture_active = False
    else:
        # Start capture on all connected cameras
        # Check if any camera is available and connected
        any_camera_available = False
        for idx in range(len(camera_manager.cameras)):
            camera = camera_manager.get_camera(idx)
            if camera and camera.connected:
                any_camera_available = True
                break

        if any_camera_available:
            capture_manager.start_capture()  # Start capture on all cameras
            joystick_state.capture_active = True
            print("Capture started on all connected cameras")
        else:
            print("No cameras available for capture")

def _plate_solve_worker(joystick_state):
    """Background loop: solve the latest camera frame and derive the instantaneous
    alignment. Pairs each solve with the mount-reported encoder position and the frame
    time so the az/el conversion and alignment are self-consistent."""
    import time as _t
    from skyfield.api import load as _sf_load
    from camera_manager import camera_manager
    import plate_solver as ps_mod

    cfg = joystick_state.config_state
    tvs = joystick_state.tracking_vis_state
    ts = joystick_state.ts or _sf_load.timescale()
    cam_index = int(getattr(cfg, 'plate_solve_camera_index', 0))
    cam_name = f"camera{cam_index + 1}"
    eph = getattr(tvs, 'ephemeris', None)

    try:
        solver = ps_mod.PlateSolver(cfg, cam_name)
        joystick_state.plate_solve_status = f"loading DB {solver.db_name}..."
        solver.ensure_loaded()
        joystick_state.plate_solver = solver
    except Exception as e:
        joystick_state.plate_solve_status = f"solver init failed: {e}"
        joystick_state.plate_solve_running = False
        return

    while joystick_state.plate_solve_running:
        try:
            camera = camera_manager.get_camera(cam_index)
            raw = (camera.thread.get_latest_raw()
                   if camera is not None and getattr(camera, 'thread', None) is not None else None)
            if raw is None:
                joystick_state.plate_solve_status = "no camera frame"
                _t.sleep(0.5)
                continue

            # Snapshot the encoder position and time paired with this frame.
            mount_azm = joystick_state.current_azm_raw
            mount_alt = joystick_state.current_alt_raw
            t = ts.now()

            result = solver.solve(raw)
            if result is None or not result.solved:
                joystick_state.plate_solve_status = "no solution"
                joystick_state.last_solve = None
                if tvs is not None:
                    tvs.last_solve = None
                _t.sleep(1.0)
                continue

            az = el = align_az = None
            if eph is not None:
                lat = float(cfg.lat_str or 0.0)
                lon = float(cfg.lon_str or 0.0)
                elev = float(cfg.alt_str or 0.0)
                az, el = ps_mod.solved_azel(result.ra_deg, result.dec_deg, lat, lon, elev, eph, ts, t)
                align_az = ps_mod.instantaneous_alignment_azimuth(az, mount_azm)

            solve_info = {
                'result': result, 'az': az, 'el': el, 'align_az': align_az,
                'mount_azm': mount_azm, 'mount_alt': mount_alt, 't': t,
                'src_shape': tuple(raw.shape[:2]), 'cam_index': cam_index,
            }
            joystick_state.last_solve = solve_info
            if tvs is not None:
                tvs.last_solve = solve_info  # consumed by the skyplot overlay
            joystick_state.plate_solve_status = f"OK {result.n_matches}m FOV {result.fov_deg:.2f}"
        except Exception as e:
            joystick_state.plate_solve_status = f"err: {e}"
        _t.sleep(1.0)


def toggle_plate_solve(joystick_state):
    """Start/stop the background plate-solve worker."""
    import plate_solver as ps_mod
    if joystick_state.plate_solve_running:
        joystick_state.plate_solve_running = False
        joystick_state.plate_solve_status = "stopped"
        if joystick_state.config_state is not None:
            joystick_state.config_state.plate_solve_enabled = False
        return
    if not ps_mod.tetra3_available():
        joystick_state.plate_solve_status = "tetra3 not installed"
        return
    joystick_state.plate_solve_running = True
    if joystick_state.config_state is not None:
        joystick_state.config_state.plate_solve_enabled = True
    th = threading.Thread(target=_plate_solve_worker, args=(joystick_state,), daemon=True)
    joystick_state.plate_solve_thread = th
    th.start()


def apply_instantaneous_alignment(joystick_state, config_state):
    """Write the most recent solved alignment azimuth into config (AltAz mode)."""
    ls = joystick_state.last_solve
    if ls is None or ls.get('align_az') is None or config_state is None:
        joystick_state.plate_solve_status = "no solution to apply"
        return
    align_az = ls['align_az']
    config_state.alignment_azimuth_str = f"{align_az:.4f}"
    try:
        config_state.save_to_file()
    except Exception as e:
        print(f"Apply-alignment save failed: {e}")
    tvs = joystick_state.tracking_vis_state
    if tvs is not None:
        tvs.alignment_azimuth = align_az
    joystick_state.plate_solve_status = f"applied align az {align_az:+.3f}"


def draw_solve_centroids_on_feed(display, joystick_state, cam_rect, cam_index):
    """Overlay the plate solver's matched star centroids on a camera feed.

    matched_centroids are in solved-frame pixel coords (row, col); scale them into the
    on-screen feed rect. Drawn only on the camera the solver is reading.
    """
    ls = getattr(joystick_state, 'last_solve', None)
    if ls is None or ls.get('cam_index') != cam_index:
        return
    result = ls.get('result')
    src = ls.get('src_shape')
    if result is None or src is None:
        return
    cents = getattr(result, 'matched_centroids', None)
    if cents is None or len(cents) == 0:
        return
    src_h, src_w = src[0], src[1]
    if src_h <= 0 or src_w <= 0:
        return
    sx_scale = cam_rect.width / float(src_w)
    sy_scale = cam_rect.height / float(src_h)
    for c in cents:
        row, col = float(c[0]), float(c[1])
        px = int(cam_rect.x + col * sx_scale)
        py = int(cam_rect.y + row * sy_scale)
        pygame.draw.circle(display.menu_screen, (0, 230, 230), (px, py), 5, 1)


def render_plate_solve_panel(display, joystick_state):
    """Plate-solve pane: on/off toggle, Apply-Alignment button, and solve status."""
    pane = joystick_panel_layout(display)['plate']
    pygame.draw.rect(display.menu_screen, (35, 40, 50), pane)
    pygame.draw.rect(display.menu_screen, (120, 140, 180), pane, 2)
    display.menu_screen.blit(display.small_font.render("Plate Solve", True, (200, 210, 235)),
                             (pane.x + 8, pane.y + 4))

    mouse_pos = pygame.mouse.get_pos()
    btn_w, btn_h = 78, 22
    toggle_rect = pygame.Rect(pane.x + 8, pane.y + 26, btn_w, btn_h)
    on = joystick_state.plate_solve_running
    tcol = (0, 150, 0) if on else (100, 100, 100)
    if toggle_rect.collidepoint(mouse_pos):
        tcol = tuple(min(255, c + 40) for c in tcol)
    pygame.draw.rect(display.menu_screen, tcol, toggle_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), toggle_rect, 1)
    tlabel = display.tiny_font.render("Solve ON" if on else "Solve OFF", True, (255, 255, 255))
    display.menu_screen.blit(tlabel, tlabel.get_rect(center=toggle_rect.center))

    have = joystick_state.last_solve is not None and joystick_state.last_solve.get('align_az') is not None
    apply_rect = pygame.Rect(pane.x + 8 + btn_w + 8, pane.y + 26, 110, btn_h)
    acol = (60, 90, 140) if have else (70, 70, 70)
    if have and apply_rect.collidepoint(mouse_pos):
        acol = tuple(min(255, c + 40) for c in acol)
    pygame.draw.rect(display.menu_screen, acol, apply_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), apply_rect, 1)
    alabel = display.tiny_font.render("Apply Align", True, (255, 255, 255))
    display.menu_screen.blit(alabel, alabel.get_rect(center=apply_rect.center))

    joystick_state.ps_button_rects = [('ps_toggle', toggle_rect), ('ps_apply', apply_rect)]

    ls = joystick_state.last_solve
    y = pane.y + 50
    if ls is not None and ls.get('az') is not None:
        r = ls['result']
        rmse = f"{r.rmse:.1f}\"" if r.rmse == r.rmse else "--"  # NaN-safe
        display.menu_screen.blit(
            display.tiny_font.render(f"SOLVED  {r.n_matches} stars  RMSE {rmse}", True, (110, 220, 140)),
            (pane.x + 8, y)); y += 12
        lines = [f"sky az {ls['az']:.2f}  el {ls['el']:.2f}",
                 f"align az -> {ls['align_az']:+.3f}°"]
        col = (200, 200, 200)
    else:
        st = joystick_state.plate_solve_status or "idle"
        col = (220, 205, 120) if joystick_state.plate_solve_running else (150, 150, 150)
        display.menu_screen.blit(display.tiny_font.render(
            ("searching... " + st) if joystick_state.plate_solve_running else st, True, col),
            (pane.x + 8, y)); y += 12
        lines = []
    for ln in lines:
        display.menu_screen.blit(display.tiny_font.render(ln, True, col), (pane.x + 8, y))
        y += 12


def handle_plate_solve_mouse_events(joystick_state, mouse_pos, config_state):
    """Route clicks on the plate-solve pane's toggle / apply buttons."""
    if not getattr(joystick_state, 'ps_button_rects', None):
        return False
    for name, rect in joystick_state.ps_button_rects:
        if rect.collidepoint(mouse_pos):
            if name == 'ps_toggle':
                toggle_plate_solve(joystick_state)
            elif name == 'ps_apply':
                apply_instantaneous_alignment(joystick_state, config_state)
            return True
    return False


def render_pid_diagnostics(display, joystick_state):
    """
    Live PID tracking diagnostics for both axes: position error and commanded
    rate, plus the active rate-command mode. Always drawn so its location is
    visible (greyed out when not actively tracking), matching the PID Gain pane.
    Reads only fields both the Python and Rust loops populate (position error and
    pid output), so it works regardless of which control loop is running.
    """
    pane = joystick_panel_layout(display)['diag']
    x_start, y_start, width, height = pane.x, pane.y, pane.width, pane.height

    # "Needed" = actively tracking (PROGRAM / HANDOFF / HOTSPOT); otherwise greyed.
    active = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    # Palette: dim when idle, lit when active.
    bg = (70, 70, 90) if active else (45, 45, 55)
    border = (140, 140, 170) if active else (80, 80, 95)
    label_c = (255, 200, 0) if active else (110, 100, 70)
    val_c = (255, 255, 255) if active else (110, 110, 120)
    rate_c = (160, 255, 160) if active else (90, 110, 90)

    pygame.draw.rect(display.menu_screen, bg, (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, border, (x_start, y_start, width, height), 1)
    display.menu_screen.blit(
        display.small_font.render("PID Diagnostics", True, val_c), (x_start + 10, y_start + 5))

    az_err = getattr(joystick_state, 'azm_position_error', 0.0)
    el_err = getattr(joystick_state, 'alt_position_error', 0.0)
    # pid_output is in rev/sec (the command scale); show it as deg/sec.
    az_rate = getattr(joystick_state, 'azm_pid_output', 0.0) * 360.0
    el_rate = getattr(joystick_state, 'alt_pid_output', 0.0) * 360.0

    cx = [x_start + 10, x_start + 130]   # AZ column, EL column
    display.menu_screen.blit(display.tiny_font.render("AZIMUTH", True, label_c), (cx[0], y_start + 26))
    display.menu_screen.blit(display.tiny_font.render("ELEVATION", True, label_c), (cx[1], y_start + 26))
    for col, err, rate in ((0, az_err, az_rate), (1, el_err, el_rate)):
        display.menu_screen.blit(
            display.tiny_font.render(f"err {err:+.2f}°", True, val_c), (cx[col], y_start + 42))
        display.menu_screen.blit(
            display.tiny_font.render(f"rate {rate:+.2f}°/s", True, rate_c), (cx[col], y_start + 58))

    # Active rate-command primitive (the control-theory lesson at a glance).
    cfg = getattr(joystick_state, 'config_state', None)
    continuous = bool(getattr(cfg, 'continuous_rate_tracking', False)) if cfg else False
    if joystick_state.tracking_mode == TrackingMode.HANDOFF:
        # In HANDOFF this line reports the parallel-detection progress instead.
        hs = getattr(joystick_state, 'handoff_status', '') or 'armed'
        mode_label = f"HANDOFF: {hs}"
        mode_c = (255, 200, 120)
    else:
        mode_label = "rate cmd: CONTINUOUS (guide-rate)" if continuous else "rate cmd: DISCRETE (MC_MOVE)"
        mode_c = (120, 200, 255) if (active and continuous) else (val_c if active else (90, 90, 100))
    display.menu_screen.blit(display.tiny_font.render(mode_label, True, mode_c), (x_start + 10, y_start + 80))
    # Bias line: show along/cross if either is set, else Az/El.
    it = getattr(joystick_state, 'bias_intrack_deg', 0.0)
    xt = getattr(joystick_state, 'bias_crosstrack_deg', 0.0)
    if it or xt:
        bias_text = f"bias InTk {it:+.1f}° XTk {xt:+.1f}°"
    else:
        bias_text = f"bias AZ {getattr(joystick_state,'bias_azm_deg',0.0):+.1f}° EL {getattr(joystick_state,'bias_alt_deg',0.0):+.1f}°"
    bias_c = val_c if active else (90, 90, 100)
    display.menu_screen.blit(display.tiny_font.render(bias_text, True, bias_c), (x_start + 10, y_start + 94))

def _draw_strip_chart(display, rect, title, series, active, scale=None):
    """Draw a single zero-centered strip chart. `series` is a list of
    (label, values, color); all series share one symmetric vertical scale.
    `scale` sets the ±vertical fullscale (auto-ranged by the caller); if None it
    falls back to the peak of the supplied data. Greyed when not tracking."""
    surf = display.menu_screen
    bg = (25, 25, 35) if active else (30, 30, 36)
    pygame.draw.rect(surf, bg, rect)
    pygame.draw.rect(surf, (90, 90, 110) if active else (60, 60, 70), rect, 1)

    title_c = (210, 210, 220) if active else (120, 120, 130)
    surf.blit(display.tiny_font.render(title, True, title_c), (rect.x + 4, rect.y + 2))

    # Legend at the top-right.
    lx = rect.right - 6
    for label, _vals, color in reversed(series):
        t = display.tiny_font.render(label, True, color if active else (110, 110, 120))
        lx -= t.get_width() + 8
        surf.blit(t, (lx, rect.y + 2))

    plot = pygame.Rect(rect.x + 4, rect.y + 15, rect.width - 8, rect.height - 28)
    zero_y = plot.centery
    pygame.draw.line(surf, (70, 70, 80), (plot.x, zero_y), (plot.right, zero_y), 1)

    if scale is not None:
        amax = max(scale, 1e-6)
    else:
        allvals = [v for _l, vals, _c in series for v in vals]
        amax = max(max((abs(v) for v in allvals), default=1.0), 1e-6)

    for _label, vals, color in series:
        n = len(vals)
        if n < 2:
            continue
        pts = []
        for i, v in enumerate(vals):
            x = plot.x + int(i / (n - 1) * plot.width)
            y = zero_y - int(v / amax * (plot.height / 2 - 1))
            y = max(plot.y, min(plot.bottom, y))
            pts.append((x, y))
        pygame.draw.lines(surf, color if active else (90, 90, 100), False, pts, 1)

    scale_c = (130, 130, 140) if active else (90, 90, 100)
    surf.blit(display.tiny_font.render(f"±{amax:.3g}", True, scale_c),
              (plot.x, rect.bottom - 12))


def render_tracking_strip_charts(display, joystick_state):
    """Render the az/el tracking-rate and position-error strip charts in the
    center column of the upper-left quadrant, for live PID tuning. Always drawn
    (greyed when not tracking) so their location is stable."""
    layout = joystick_center_layout(display)
    if not layout['valid']:
        return

    # Sample fresh data each frame (loop-agnostic: Python thread or Rust loop).
    joystick_state.sample_tracking_history()

    active = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    az_c, el_c = (120, 255, 120), (120, 200, 255)
    az_rate, el_rate = list(joystick_state.az_rate_history), list(joystick_state.el_rate_history)
    az_err, el_err = list(joystick_state.az_err_history), list(joystick_state.el_err_history)
    # Auto-range each chart from a recent window so the axis shrinks promptly when
    # the signal collapses (floors keep it from over-zooming on noise).
    rate_scale = joystick_state.chart_axis_scale('rate', az_rate + el_rate, floor=0.05)
    err_scale = joystick_state.chart_axis_scale('err', az_err + el_err, floor=0.02)
    _draw_strip_chart(display, layout['chart_rate'], "Tracking Rate °/s", [
        ("AZ", az_rate, az_c),
        ("EL", el_rate, el_c),
    ], active, scale=rate_scale)
    _draw_strip_chart(display, layout['chart_err'], "Pos Error °", [
        ("AZ", az_err, az_c),
        ("EL", el_err, el_c),
    ], active, scale=err_scale)


def render_navball(display, joystick_state):
    """Render a KSP-style navball in the center of the upper-left quadrant,
    showing the mount's current attitude (azimuth = heading tape, elevation =
    pitch). A fixed boresight reticle marks where the scope points; the ball
    moves under it. Greyed/blanked when the telescope is not connected."""
    layout = joystick_center_layout(display)
    if not layout['valid']:
        return
    rect = layout['navball']
    cx, cy = rect.center
    R = rect.width // 2 - 2
    if R < 30:
        return

    connected = joystick_state.telescope_connected
    # Show the same SKY az/el the skyplot draws, not raw mount coordinates. In
    # AltAz the mount ALT axis runs opposite sky elevation (sky el = 90 - ALT)
    # and azimuth carries the alignment offset, so using current_azm/current_alt
    # directly made the navball read inverted/rotated vs the camera pointing.
    cfg = getattr(joystick_state, 'config_state', None)
    if connected and cfg is not None:
        try:
            align_az = float(getattr(cfg, 'alignment_azimuth_str', 0.0) or 0.0)
            align_el = float(getattr(cfg, 'alignment_elevation_str', 0.0) or 0.0)
        except (TypeError, ValueError):
            align_az = align_el = 0.0
        if getattr(cfg, 'mount_mode', 'AltAz') == 'AltAz':
            from transformations import AzAlt2AzEl_AltAz
            az, el = AzAlt2AzEl_AltAz(joystick_state.current_azm,
                                     joystick_state.current_alt, align_az)
        else:
            from transformations import AzAlt2AzEl
            az, el = AzAlt2AzEl(joystick_state.current_azm,
                                joystick_state.current_alt, align_az, align_el)
        az = az % 360.0
        el = max(-90.0, min(90.0, el))
    else:
        az = el = 0.0

    # Title above the ball.
    display.menu_screen.blit(display.small_font.render("Navball", True, (220, 220, 230)),
                             (rect.x, rect.y - 18))

    ball = pygame.Surface((2 * R, 2 * R), pygame.SRCALPHA)
    bcx, bcy = R, R

    # Sky / ground split at the horizon; pitch-up drives the horizon downward.
    horizon_y = int(bcy + (el / 90.0) * R)
    ball.fill((60, 110, 200))                                   # sky
    pygame.draw.rect(ball, (150, 95, 55), (0, horizon_y, 2 * R, 2 * R))  # ground
    pygame.draw.line(ball, (255, 255, 255), (0, horizon_y), (2 * R, horizon_y), 2)

    # Pitch ladder (ticks every 30 deg; current pitch sits at center).
    for a in range(-60, 91, 30):
        if a == 0:
            continue
        y = bcy + ((el - a) / 90.0) * R
        if 2 <= y <= 2 * R - 2:
            half = R * 0.22
            pygame.draw.line(ball, (225, 225, 225), (bcx - half, y), (bcx + half, y), 1)
            t = display.tiny_font.render(f"{a}", True, (225, 225, 225))
            ball.blit(t, (bcx + half + 2, y - 5))

    # Heading tape across the upper third; current azimuth at center, 90 deg ~ R.
    tape_y = int(R * 0.30)
    for h, lbl in ((0, "N"), (45, "NE"), (90, "E"), (135, "SE"),
                   (180, "S"), (225, "SW"), (270, "W"), (315, "NW")):
        d = ((h - az + 180) % 360) - 180
        x = bcx + (d / 90.0) * R
        if 2 <= x <= 2 * R - 2:
            cardinal = len(lbl) == 1
            col = (255, 235, 130) if cardinal else (190, 190, 200)
            t = display.tiny_font.render(lbl, True, col)
            ball.blit(t, (x - t.get_width() // 2, tape_y))
            pygame.draw.line(ball, col, (x, tape_y + 12), (x, tape_y + 17), 1)

    # Clip the square down to a disc via an alpha mask.
    mask = pygame.Surface((2 * R, 2 * R), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (R, R), R)
    ball.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    display.menu_screen.blit(ball, (cx - R, cy - R))

    # Bezel + fixed boresight reticle.
    pygame.draw.circle(display.menu_screen, (180, 180, 190), (cx, cy), R, 2)
    pygame.draw.circle(display.menu_screen, (255, 255, 0), (cx, cy), 3, 1)
    pygame.draw.line(display.menu_screen, (255, 255, 0), (cx - 13, cy), (cx - 4, cy), 2)
    pygame.draw.line(display.menu_screen, (255, 255, 0), (cx + 4, cy), (cx + 13, cy), 2)
    pygame.draw.line(display.menu_screen, (255, 255, 0), (cx, cy - 13), (cx, cy - 4), 2)

    # Readout below the ball.
    readout = f"Az {az:5.1f}°  El {el:+5.1f}°" if connected else "Az --   El --"
    rc = (200, 255, 200) if connected else (150, 150, 150)
    t = display.tiny_font.render(readout, True, rc)
    display.menu_screen.blit(t, (cx - t.get_width() // 2, cy + R + 4))

    if not connected:
        scrim = pygame.Surface((2 * R, 2 * R), pygame.SRCALPHA)
        pygame.draw.circle(scrim, (25, 25, 30, 170), (R, R), R)
        display.menu_screen.blit(scrim, (cx - R, cy - R))


def render_bias_control_grid(display, joystick_state):
    """
    Render the manual bias control grid (D-pad mirror) with the active frame /
    resolution and current values. Anchored above the PID pane in the bottom-
    right of the upper-left quadrant. Always drawn so its location is visible;
    greyed out and non-interactable unless in a tracking mode that uses bias.
    """
    enabled = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    frame = getattr(joystick_state, 'bias_frame', 'azel')
    res = getattr(joystick_state, 'bias_resolution', 'coarse')
    alongcross = (frame == 'alongcross')
    h_lbl, v_lbl = ("InTk", "XTk") if alongcross else ("Az", "El")

    # Position above the PID pane, hugging the bottom-right of the quadrant
    pane = joystick_panel_layout(display)['bias']
    x_start, y_start = pane.x, pane.y
    width, height = pane.width, pane.height

    # Background rectangle
    pygame.draw.rect(display.menu_screen, (60, 60, 80),
                     (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, (120, 120, 150),
                     (x_start, y_start, width, height), 1)

    # Title (note the Op button cycles the mode)
    title_text = display.small_font.render("Bias Control", True, (255, 255, 255))
    display.menu_screen.blit(title_text, (x_start + 10, y_start + 5))
    hint = display.tiny_font.render("Op: mode", True, (150, 150, 180))
    display.menu_screen.blit(hint, (x_start + width - hint.get_width() - 8, y_start + 8))

    # Current bias mode indicator: resolution + frame
    res_color = (100, 255, 100) if res == "coarse" else (255, 180, 100)
    frame_color = (120, 200, 255) if alongcross else (200, 200, 120)
    res_text = display.tiny_font.render(f"{res.upper()}", True, res_color)
    display.menu_screen.blit(res_text, (x_start + 10, y_start + 25))
    frame_text = display.tiny_font.render(
        "ALONG/CROSS-TRACK" if alongcross else "AZ/EL", True, frame_color)
    display.menu_screen.blit(frame_text, (x_start + 55, y_start + 25))

    # Current values: show the active frame's pair brightly, the other dimmed.
    azel_c = (140, 140, 150) if alongcross else (255, 255, 255)
    ac_c = (255, 255, 255) if alongcross else (140, 140, 150)
    azel_str = f"Az {joystick_state.bias_azm_deg:+.2f}° El {joystick_state.bias_alt_deg:+.2f}°"
    ac_str = f"InTk {joystick_state.bias_intrack_deg:+.2f}° XTk {joystick_state.bias_crosstrack_deg:+.2f}°"
    display.menu_screen.blit(display.tiny_font.render(azel_str, True, azel_c), (x_start + 10, y_start + 40))
    display.menu_screen.blit(display.tiny_font.render(ac_str, True, ac_c), (x_start + 10, y_start + 52))

    # Button grid for manual adjustment (mirrors the D-pad). Each entry carries
    # its (horizontal, vertical) direction so the handler calls adjust_bias().
    button_size = 25
    button_spacing = 8
    grid_start_x = x_start + 20
    grid_start_y = y_start + 72

    h_color, v_color = (255, 150, 150), (150, 255, 150)
    button_positions = [
        (f"-{h_lbl}", -1, 0, grid_start_x, grid_start_y, h_color),
        (f"+{v_lbl}", 0, 1, grid_start_x + button_size + button_spacing, grid_start_y, v_color),
        (f"+{h_lbl}", 1, 0, grid_start_x + 2 * (button_size + button_spacing), grid_start_y, h_color),
        (f"-{v_lbl}", 0, -1, grid_start_x + button_size + button_spacing,
         grid_start_y + button_size + button_spacing, v_color),
    ]

    button_rects = []
    mouse_pos = pygame.mouse.get_pos()
    for label, hdir, vdir, bx, by, color in button_positions:
        button_rect = pygame.Rect(bx, by, button_size, button_size)
        button_rects.append((hdir, vdir, button_rect))

        hover = button_rect.collidepoint(mouse_pos)
        bg_color = tuple(min(255, c + 40) for c in color) if hover else color
        pygame.draw.rect(display.menu_screen, bg_color, button_rect)
        pygame.draw.rect(display.menu_screen, (200, 200, 200), button_rect, 1)

        label_text = display.tiny_font.render(label, True, (0, 0, 0))
        text_rect = label_text.get_rect(center=button_rect.center)
        display.menu_screen.blit(label_text, text_rect)

    # Store button rects in joystick state for mouse handling
    joystick_state.bias_button_rects = button_rects

    # Grey out when not in a bias-using tracking mode (visible but not interactable)
    if not enabled:
        _draw_disabled_scrim(display, pane)

def render_feed_forward_toggle_buttons(display, joystick_state):
    """
    Render feed-forward toggle buttons (FF AZ / FF EL / FF OFF) as a row inside
    the bottom of the PID Gain Control pane. Always drawn so the controls are
    visible; greyed out and non-interactable unless in PROGRAM mode.
    """
    enabled = joystick_state.tracking_mode == TrackingMode.PROGRAM

    # Lay the buttons out along the bottom strip of the PID pane
    pane = joystick_panel_layout(display)['pid']
    button_width, button_height = 72, 22
    button_spacing = 6
    x_start = pane.x + 12
    row_y = pane.bottom - button_height - 8

    mouse_pos = pygame.mouse.get_pos()

    # Section label
    ff_label = display.tiny_font.render("Feed-Forward:", True, (220, 220, 220))
    display.menu_screen.blit(ff_label, (x_start, row_y - 13))

    # AZ Feed-forward button
    az_ff_rect = pygame.Rect(x_start, row_y, button_width, button_height)
    az_hover = enabled and az_ff_rect.collidepoint(mouse_pos)

    az_color = (0, 150, 0) if joystick_state.feed_forward_azm_enabled else (100, 100, 100)
    if az_hover:
        az_color = tuple(min(255, c + 40) for c in az_color)

    pygame.draw.rect(display.menu_screen, az_color, az_ff_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), az_ff_rect, 1)

    az_text = display.tiny_font.render("FF AZ", True, (255, 255, 255))
    text_rect = az_text.get_rect(center=az_ff_rect.center)
    display.menu_screen.blit(az_text, text_rect)

    # EL Feed-forward button
    el_ff_rect = pygame.Rect(x_start + button_width + button_spacing, row_y, button_width, button_height)
    el_hover = enabled and el_ff_rect.collidepoint(mouse_pos)

    el_color = (0, 150, 0) if joystick_state.feed_forward_alt_enabled else (100, 100, 100)
    if el_hover:
        el_color = tuple(min(255, c + 40) for c in el_color)

    pygame.draw.rect(display.menu_screen, el_color, el_ff_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), el_ff_rect, 1)

    el_text = display.tiny_font.render("FF EL", True, (255, 255, 255))
    text_rect = el_text.get_rect(center=el_ff_rect.center)
    display.menu_screen.blit(el_text, text_rect)

    # Disable All button
    disable_rect = pygame.Rect(x_start + 2 * (button_width + button_spacing), row_y, button_width, button_height)
    disable_hover = enabled and disable_rect.collidepoint(mouse_pos)

    disable_color = (150, 100, 100) if disable_hover else (120, 100, 100)
    pygame.draw.rect(display.menu_screen, disable_color, disable_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), disable_rect, 1)

    disable_all_text = display.tiny_font.render("FF OFF", True, (255, 255, 255))
    text_rect = disable_all_text.get_rect(center=disable_rect.center)
    display.menu_screen.blit(disable_all_text, text_rect)

    # Store button rects for mouse click handling
    joystick_state.ff_button_rects = [
        (" ff_az", az_ff_rect),
        (" ff_el", el_ff_rect),
        (" ff_off", disable_rect)
    ]

    # Grey out the feed-forward strip when not in PROGRAM mode
    if not enabled:
        strip = pygame.Rect(pane.x + 1, row_y - 15, pane.width - 2, button_height + 19)
        _draw_disabled_scrim(display, strip)

def handle_bias_control_mouse_events(joystick_state, mouse_pos):
    """
    Handle mouse clicks on the bias control grid buttons (on-screen mirror of the
    D-pad). Routes through adjust_bias() so it honors the active frame/resolution.
    """
    if not hasattr(joystick_state, 'bias_button_rects'):
        return False

    # Interactable only in tracking modes that use bias (greyed out otherwise).
    if joystick_state.tracking_mode not in (
            TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT):
        return False

    for hdir, vdir, rect in joystick_state.bias_button_rects:
        if rect.collidepoint(mouse_pos):
            joystick_state.adjust_bias(hdir, vdir)
            print(f"Bias [{joystick_state.bias_frame}/{joystick_state.bias_resolution}]: "
                  f"Az {joystick_state.bias_azm_deg:+.2f}° El {joystick_state.bias_alt_deg:+.2f}° "
                  f"InTk {joystick_state.bias_intrack_deg:+.2f}° XTk {joystick_state.bias_crosstrack_deg:+.2f}°")
            return True
    return False

def render_pid_gain_sliders(display, joystick_state):
    """
    Render PID gain sliders for adjusting P, I, D gains in joystick mode.
    Anchored to the bottom-right of the upper-left quadrant (below the Bias
    pane). The feed-forward toggle buttons are drawn inside the bottom of this
    pane by render_feed_forward_toggle_buttons(). Always drawn so its location
    is visible; greyed out and non-interactable unless in PROGRAM mode.
    """
    # Only render when there's an active config_state
    if not hasattr(joystick_state, 'config_state') or joystick_state.config_state is None:
        return

    config_state = joystick_state.config_state
    enabled = joystick_state.tracking_mode == TrackingMode.PROGRAM

    # Pane hugging the bottom-right of the quadrant
    pane = joystick_panel_layout(display)['pid']
    x_start, y_start = pane.x, pane.y
    width, height = pane.width, pane.height

    # Background rectangle
    pygame.draw.rect(display.menu_screen, (70, 70, 90),
                     (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, (140, 140, 170),
                     (x_start, y_start, width, height), 1)

    # Title
    title_text = display.small_font.render("PID Gain Control", True, (255, 255, 255))
    display.menu_screen.blit(title_text, (x_start + 10, y_start + 5))

    # Define PID gain ranges (similar to camera settings)
    PID_GAIN_RANGE = (0.0, 2.0)  # From 0 to 2.0
    SLIDER_WIDTH = 80  # Width for each PID slider

    # Mouse position for hover detection
    mouse_pos = pygame.mouse.get_pos()

    # PID gain range for logarithmic sliders spanning 5 orders of magnitude (0.00002 to 2.0)
    PID_MAX_VALUE = 2.0
    PID_MIN_VALUE = 2.0 / 100000.0  # 5 orders of magnitude: 2.0e-5
    LOG_SCALE_FACTOR = 5.0  # 5 orders of magnitude
    LOG_SCALE_OFFSET = PID_MIN_VALUE  # Start just above zero

    # Recompute slider input and track rectangles every frame so they track the
    # pane's (quadrant-relative) position rather than being frozen at first render.
    display.joystick_pid_rects = {
        'pid_azm_p_gain': pygame.Rect(x_start + 40 - 10, y_start + 30, 60, 20),
        'pid_azm_i_gain': pygame.Rect(x_start + 40 - 10, y_start + 65, 60, 20),
        'pid_azm_d_gain': pygame.Rect(x_start + 40 - 10, y_start + 100, 60, 20),
        'pid_alt_p_gain': pygame.Rect(x_start + 160 - 10, y_start + 30, 60, 20),
        'pid_alt_i_gain': pygame.Rect(x_start + 160 - 10, y_start + 65, 60, 20),
        'pid_alt_d_gain': pygame.Rect(x_start + 160 - 10, y_start + 100, 60, 20),
    }
    display.joystick_pid_slider_rects = {
        'pid_azm_p_gain': pygame.Rect(x_start + 40 - 10, y_start + 55, SLIDER_WIDTH, 5),
        'pid_azm_i_gain': pygame.Rect(x_start + 40 - 10, y_start + 90, SLIDER_WIDTH, 5),
        'pid_azm_d_gain': pygame.Rect(x_start + 40 - 10, y_start + 125, SLIDER_WIDTH, 5),
        'pid_alt_p_gain': pygame.Rect(x_start + 160 - 10, y_start + 55, SLIDER_WIDTH, 5),
        'pid_alt_i_gain': pygame.Rect(x_start + 160 - 10, y_start + 90, SLIDER_WIDTH, 5),
        'pid_alt_d_gain': pygame.Rect(x_start + 160 - 10, y_start + 125, SLIDER_WIDTH, 5),
    }

    # AZM PID labels and inputs (left column)
    azm_title = display.tiny_font.render("AZM:", True, (255, 200, 100))
    display.menu_screen.blit(azm_title, (x_start, y_start + 20))

    # P gain
    p_label = display.tiny_font.render("P:", True, (255, 255, 255))
    display.menu_screen.blit(p_label, (x_start, y_start + 30))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_azm_p_gain'])
    p_value = getattr(config_state, 'pid_azm_p_gain', 0.0)
    p_text = display.tiny_font.render(f"{p_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(p_text, (display.joystick_pid_rects['pid_azm_p_gain'].x + 3, display.joystick_pid_rects['pid_azm_p_gain'].y + 2))

    # P slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_azm_p_gain'])
    # Position calculation: log10(value / min_val) / log10(max_val / min_val)
    log_position = math.log10(max(p_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_azm_p_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_azm_p_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_azm_p_gain'].y - 3, 6, 11))

    # I gain
    i_label = display.tiny_font.render("I:", True, (255, 255, 255))
    display.menu_screen.blit(i_label, (x_start, y_start + 65))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_azm_i_gain'])
    i_value = getattr(config_state, 'pid_azm_i_gain', 0.0)
    i_text = display.tiny_font.render(f"{i_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(i_text, (display.joystick_pid_rects['pid_azm_i_gain'].x + 3, display.joystick_pid_rects['pid_azm_i_gain'].y + 2))

    # I slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_azm_i_gain'])
    log_position = math.log10(max(i_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_azm_i_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_azm_i_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_azm_i_gain'].y - 3, 6, 11))

    # D gain
    d_label = display.tiny_font.render("D:", True, (255, 255, 255))
    display.menu_screen.blit(d_label, (x_start, y_start + 105))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_azm_d_gain'])
    d_value = getattr(config_state, 'pid_azm_d_gain', 0.0)
    d_text = display.tiny_font.render(f"{d_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(d_text, (display.joystick_pid_rects['pid_azm_d_gain'].x + 3, display.joystick_pid_rects['pid_azm_d_gain'].y + 2))

    # D slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_azm_d_gain'])
    log_position = math.log10(max(d_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_azm_d_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_azm_d_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_azm_d_gain'].y - 3, 6, 11))

    # ALT PID labels and inputs (right column)
    alt_title = display.tiny_font.render("ALT:", True, (255, 200, 100))
    display.menu_screen.blit(alt_title, (x_start + 120, y_start + 20))

    # P gain
    alt_p_label = display.tiny_font.render("P:", True, (255, 255, 255))
    display.menu_screen.blit(alt_p_label, (x_start + 120, y_start + 30))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_alt_p_gain'])
    alt_p_value = getattr(config_state, 'pid_alt_p_gain', 0.0)
    alt_p_text = display.tiny_font.render(f"{alt_p_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_p_text, (display.joystick_pid_rects['pid_alt_p_gain'].x + 3, display.joystick_pid_rects['pid_alt_p_gain'].y + 2))

    # P slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_alt_p_gain'])
    log_position = math.log10(max(alt_p_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_alt_p_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_alt_p_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_alt_p_gain'].y - 3, 6, 11))

    # I gain
    alt_i_label = display.tiny_font.render("I:", True, (255, 255, 255))
    display.menu_screen.blit(alt_i_label, (x_start + 120, y_start + 65))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_alt_i_gain'])
    alt_i_value = getattr(config_state, 'pid_alt_i_gain', 0.0)
    alt_i_text = display.tiny_font.render(f"{alt_i_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_i_text, (display.joystick_pid_rects['pid_alt_i_gain'].x + 3, display.joystick_pid_rects['pid_alt_i_gain'].y + 2))

    # I slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_alt_i_gain'])
    log_position = math.log10(max(alt_i_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_alt_i_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_alt_i_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_alt_i_gain'].y - 3, 6, 11))

    # D gain
    alt_d_label = display.tiny_font.render("D:", True, (255, 255, 255))
    display.menu_screen.blit(alt_d_label, (x_start + 120, y_start + 105))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_alt_d_gain'])
    alt_d_value = getattr(config_state, 'pid_alt_d_gain', 0.0)
    alt_d_text = display.tiny_font.render(f"{alt_d_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_d_text, (display.joystick_pid_rects['pid_alt_d_gain'].x + 3, display.joystick_pid_rects['pid_alt_d_gain'].y + 2))

    # D slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_alt_d_gain'])
    log_position = math.log10(max(alt_d_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_alt_d_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_alt_d_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_alt_d_gain'].y - 3, 6, 11))

    # Focus highlights
    for field, rect in display.joystick_pid_rects.items():
        if hasattr(config_state, 'focused_field') and config_state.focused_field == field:
            pygame.draw.rect(display.menu_screen, (0, 0, 255), rect, 2)
            if hasattr(config_state, 'cursor_pos'):
                field_display_str = ""
                if field == 'pid_azm_p_gain':
                    field_display_str = f"{p_value:.5f}"
                elif field == 'pid_azm_i_gain':
                    field_display_str = f"{i_value:.5f}"
                elif field == 'pid_azm_d_gain':
                    field_display_str = f"{d_value:.5f}"
                elif field == 'pid_alt_p_gain':
                    field_display_str = f"{alt_p_value:.5f}"
                elif field == 'pid_alt_i_gain':
                    field_display_str = f"{alt_i_value:.5f}"
                elif field == 'pid_alt_d_gain':
                    field_display_str = f"{alt_d_value:.5f}"

                if field_display_str and field in config_state.cursor_pos:
                    text_width, _ = display.tiny_font.size(field_display_str[:config_state.cursor_pos[field]])
                    pygame.draw.line(display.menu_screen, (0, 0, 255),
                                   (rect.x + 5 + text_width, rect.y + 5),
                                   (rect.x + 5 + text_width, rect.y + 20), 2)

    # Lead-time slider (transport-latency compensation). Live-tunable like the
    # gains; takes effect immediately in both the Python and Rust loops, which
    # re-read pid_lead_time_sec each cycle. Range JL_LEAD_MIN..JL_LEAD_MAX.
    lead_val = float(getattr(config_state, 'pid_lead_time_sec', 0.0) or 0.0)
    lead_y = y_start + 143
    display.menu_screen.blit(display.tiny_font.render("Lead s:", True, (255, 200, 100)),
                             (x_start, lead_y))
    display.menu_screen.blit(display.tiny_font.render(f"{lead_val:.2f}", True, (255, 255, 255)),
                             (x_start + 50, lead_y))
    lead_track = pygame.Rect(x_start + 85, lead_y + 6, width - 97, 4)
    display.joystick_lead_slider_rect = lead_track
    pygame.draw.rect(display.menu_screen, (150, 150, 150), lead_track)
    lead_ratio = 0.0
    if JL_LEAD_MAX > JL_LEAD_MIN:
        lead_ratio = min(1.0, max(0.0, (lead_val - JL_LEAD_MIN) / (JL_LEAD_MAX - JL_LEAD_MIN)))
    lead_handle_x = lead_track.x + int(lead_ratio * lead_track.width)
    lead_hover = pygame.Rect(lead_handle_x - 3, lead_track.y - 4, 6, 12).collidepoint(mouse_pos)
    pygame.draw.rect(display.menu_screen, (255, 0, 0) if lead_hover else (200, 0, 0),
                     (lead_handle_x - 3, lead_track.y - 4, 6, 12))

    # Grey out the slider area when not in PROGRAM mode (the feed-forward strip
    # below is greyed separately by render_feed_forward_toggle_buttons()).
    if not enabled:
        slider_region = pygame.Rect(x_start + 1, y_start + 18, width - 2, 151)
        _draw_disabled_scrim(display, slider_region)

def _lead_from_track_x(track, x):
    """Map an x pixel on the lead slider track to a lead time (seconds)."""
    rel = min(max(x - track.x, 0), track.width)
    frac = rel / track.width if track.width else 0.0
    return round(JL_LEAD_MIN + frac * (JL_LEAD_MAX - JL_LEAD_MIN), 3)


def handle_lead_slider_mouse_events(joystick_state, display, mouse_pos):
    """Click on the joystick PID pane's Lead slider sets pid_lead_time_sec live.
    Dragging is handled in main.py's MOUSEMOTION path (same as the PID gain
    sliders). Returns True if the click hit the lead track."""
    if not hasattr(display, 'joystick_lead_slider_rect'):
        return False
    track = display.joystick_lead_slider_rect
    cfg = getattr(joystick_state, 'config_state', None)
    if cfg is None or not track.collidepoint(mouse_pos):
        return False
    cfg.pid_lead_time_sec = _lead_from_track_x(track, mouse_pos[0])
    print(f"Lead time: {cfg.pid_lead_time_sec:.3f}s")
    return True


def handle_pid_sliders_mouse_events(joystick_state, display, mouse_pos):
    """
    Handle mouse clicks on PID gain slider input fields.
    """
    if not hasattr(display, 'joystick_pid_rects') or not hasattr(joystick_state, 'config_state'):
        return False

    # PID gain sliders are only interactable in PROGRAM mode (greyed otherwise)
    if joystick_state.tracking_mode != TrackingMode.PROGRAM:
        return False

    config_state = joystick_state.config_state

    for field, rect in display.joystick_pid_rects.items():
        if rect.collidepoint(mouse_pos):
            config_state.focused_field = field
            if hasattr(config_state, 'cursor_pos'):
                config_state.cursor_pos[field] = 0
            if hasattr(config_state, 'selection_start'):
                config_state.selection_start[field] = None
            return True

    return False

def handle_ff_toggle_mouse_events(joystick_state, mouse_pos):
    """
    Handle mouse clicks on feed-forward toggle buttons.
    Called from main event loop when buttons are clicked.
    """
    if not hasattr(joystick_state, 'ff_button_rects'):
        return False

    # Feed-forward toggles are only interactable in PROGRAM mode (greyed otherwise)
    if joystick_state.tracking_mode != TrackingMode.PROGRAM:
        return False

    for label, rect in joystick_state.ff_button_rects:
        if rect.collidepoint(mouse_pos):
            if " ff_az" in label:
                # Toggle AZ feed-forward
                joystick_state.feed_forward_azm_enabled = not joystick_state.feed_forward_azm_enabled
                if joystick_state.azm_pid:
                    joystick_state.azm_pid.set_feed_forward_enabled(joystick_state.feed_forward_azm_enabled)
                print(f"FF AZ toggled: {joystick_state.feed_forward_azm_enabled}")
            elif " ff_el" in label:
                # Toggle EL feed-forward
                joystick_state.feed_forward_alt_enabled = not joystick_state.feed_forward_alt_enabled
                if joystick_state.alt_pid:
                    joystick_state.alt_pid.set_feed_forward_enabled(joystick_state.feed_forward_alt_enabled)
                print(f"FF EL toggled: {joystick_state.feed_forward_alt_enabled}")
            elif " ff_off" in label:
                # Disable all feed-forward
                joystick_state.feed_forward_azm_enabled = False
                joystick_state.feed_forward_alt_enabled = False
                if joystick_state.azm_pid:
                    joystick_state.azm_pid.set_feed_forward_enabled(False)
                if joystick_state.alt_pid:
                    joystick_state.alt_pid.set_feed_forward_enabled(False)
                print("All feed-forward disabled")
            return True
    return False
