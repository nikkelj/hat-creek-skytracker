import pygame
import serial
import serial.tools.list_ports
import math
import numpy as np
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
from lib.auxstar import NexstarHandController, RATES, Targets, GUIDE_COUNTS_PER_DPS
from tracking_visuals import PolarPlotMode
from camera_manager import camera_manager, update_camera_frames_from_buffers
from camera_manager import render_sensor_calibration
from camera_manager import apply_gamma_correction, roi_sizes, roi_label_texts
from utils import draw_button

# Import PID controller and helper functions
from control import (create_pid_controllers, compute_mount_position_error,
                     sky_target_to_mount, choose_mount_target, mount_target_for)
from trajectory import interpolate_position_data_and_rates
from hotspot import detect_hotspot, pixel_offset_to_angles

# Star-rejection rate filter tuning (shared by HANDOFF and HOTSPOT).
# The implied-rate estimate differences a detection against a candidate at
# least BASELINE_S older: pixel positions are captured at frame time but the
# boresight is sampled at loop-cycle time, so the per-sample timing skew
# (tens of ms) corrupts short-baseline rates by ~rate * skew / baseline --
# with a consecutive-frame baseline that falsely rejected any fast target.
# The acceptance threshold also scales with the expected rate
# (REL_FRACTION * |trajectory rate|) for the same reason, with the
# configured hotspot_rate_gate_dps as the absolute floor.
RATE_FILTER_BASELINE_S = 0.35
RATE_FILTER_MAX_AGE_S = 3.0
RATE_FILTER_REL_FRACTION = 0.35


def hotspot_discrete_step(total_rev_s):
    """Discrete MC_MOVE step for a capped HOTSPOT rate (rev/s): the largest
    step whose physical rate fits under |total|, minimum 1 so a small capped
    correction still creeps toward center. The PID's own discretizer zeroes
    anything below 0.01 rev/s (3.6 deg/s) -- fine for raw PID outputs, but a
    hotspot correction capped to ~2 deg/s must still actuate in discrete
    mode. (The continuous guide-rate path commands the exact rate instead.)"""
    if abs(total_rev_s) <= 1e-6:
        return 0
    sign = 1 if total_rev_s > 0 else -1
    mag = 1
    for idx in range(9, 0, -1):
        if RATES.get(idx, 0.0) <= abs(total_rev_s):
            mag = idx
            break
    return sign * mag

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

# ADS-B linear-fit slider: number of recent fixes fit for trajectory prediction.
JL_ADSB_FIT_MIN = 2
JL_ADSB_FIT_MAX = 20


def _draw_disabled_scrim(display, rect):
    """Draw a translucent grey scrim over a panel to show it is visible but
    inactive (not interactable in the current mode/state)."""
    scrim = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    scrim.fill((25, 25, 25, 165))
    display.menu_screen.blit(scrim, rect.topleft)


def joystick_panel_layout(display):
    """Geometry for the control panes that hug the RIGHT edge of the joystick
    mode's upper-left quadrant. On tall screens they stack bottom-up in one
    column: PID Gain (bottom), Bias, PID Diagnostics, Plate Solve. On short
    screens (the stack doesn't fit -- e.g. 1080p) they reflow into two
    bottom-aligned columns: pid+bias rightmost, diag+plate to their left, and
    the quadrant divider slides right (display.joystick_layout_params) so the
    navball column keeps its width. All right-aligned to the divider."""
    params = display.joystick_layout_params()
    qx, qy = display.sub_x, display.sub_y
    qh = display.sub_height // 2
    gap = params['gap']
    right = params['divider_x'] - 12  # right edge of the pane block
    bottom = qy + qh - 12
    top_min = qy + display.JOYSTICK_PANE_TOP

    pid_w, pid_h = 250, 215
    bias_w, bias_h = 250, 150
    diag_w, diag_h = 250, 112
    plate_w, plate_h = 250, 92

    pid_x = right - pid_w
    pid_y = bottom - pid_h
    bias_x = pid_x
    bias_y = pid_y - gap - bias_h

    if params['two_col']:
        # Second column left of pid/bias, bottom-aligned: plate at the bottom,
        # diagnostics above it.
        diag_x = pid_x - gap - diag_w
        plate_x = diag_x
        plate_y = bottom - plate_h
        diag_y = plate_y - gap - diag_h
        pane_left = diag_x
    else:
        diag_x = pid_x
        diag_y = bias_y - gap - diag_h
        plate_x = pid_x
        # Clamped so it never rides up over the connect/port/status rows at
        # the top of the quadrant.
        plate_y = max(top_min, diag_y - gap - plate_h)
        pane_left = pid_x

    # ADS-B pane: top-center band, above the navball and between the left
    # status block and the pane block / position-display box (which already
    # own the top-left and top-right corners).
    center_left = qx + display.JOYSTICK_STATUS_W
    center_right = pane_left - 14
    center_w = max(150, center_right - center_left)
    adsb_w = min(260, center_w)
    adsb_x = center_left + (center_w - adsb_w) // 2
    adsb_y = qy + 4
    adsb_h = 82

    return {
        'pid': pygame.Rect(pid_x, pid_y, pid_w, pid_h),
        'bias': pygame.Rect(bias_x, bias_y, bias_w, bias_h),
        'diag': pygame.Rect(diag_x, diag_y, diag_w, diag_h),
        'plate': pygame.Rect(plate_x, plate_y, plate_w, plate_h),
        'adsb': pygame.Rect(adsb_x, adsb_y, adsb_w, adsb_h),
        'pane_left': pane_left,
    }


def joystick_center_layout(display):
    """Geometry for the center column of the joystick mode's upper-left
    quadrant: the navball up top and the two PID-tuning strip charts
    (tracking rate, position error) stacked below it. The column sits between
    the left button/axes status block and the right-hand pane block (one or
    two columns wide -- see joystick_panel_layout). `valid` is False when the
    quadrant is too small to host them."""
    qx, qy = display.sub_x, display.sub_y
    qh = display.sub_height // 2

    left = qx + display.JOYSTICK_STATUS_W  # clear of the button/axes status block
    right = joystick_panel_layout(display)['pane_left'] - 14  # clear of the pane block
    top = qy + display.JOYSTICK_PANE_TOP   # below the connect/port/baud/status rows
    bottom = qy + qh - 12
    cw = right - left
    ch = bottom - top
    valid = cw >= 120 and ch >= 220

    nb_side = max(60, min(cw, int(ch * 0.52), 210))
    navball = pygame.Rect(left + (cw - nb_side) // 2, top + 6, nb_side, nb_side)

    charts_top = navball.bottom + 28
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
        # Last rate command actually sent per axis: {target: (wire_key, time)}.
        # See _send_rate_command -- unchanged commands are not re-sent (the
        # firmware holds the rate; the 9600-baud wire is the loop bottleneck).
        self._rate_cmd_cache = {}
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

        # ADS-B receiver (RTL-SDR aircraft tracking). The receiver owns the SDR
        # source thread + the aircraft tracker; it's created lazily on first
        # connect so the app starts without the SDR libs installed.
        self.adsb = None                       # AdsbReceiver instance (lazy)
        self.adsb_connected = False
        self.adsb_status = ""
        self.adsb_connect_button_hover = False
        self.adsb_disconnect_button_hover = False
        self.adsb_button_rects = {}            # {'connect': rect, 'disconnect': rect}

        # Skyplot "Targets" overlay (filters + sortable passes/launches table) in
        # the upper-right quadrant. Always-on toggle strip; panel opens on demand.
        self.targets_panel_open = False
        self.jl_target_btn_rects = {}          # {'targets'|'sats'|'labels': rect}
        self.jl_filter_rects = {}              # {'filter'|'filter_above_alt'|'filter_below_alt': rect}
        self.jl_clear_filters_rect = None
        self.jl_pass_table_rect = None         # table bounding box (for wheel hit-test)

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
        self.hotspot_last_frame_seq = None   # last camera frame processed (stale gate)
        self._hotspot_last_fresh_time = 0.0  # for the measured frame interval
        self._hotspot_frame_interval = 0.2   # conservative until measured
        self._hotspot_cmd_dps = (0.0, 0.0)   # last commanded TOTAL rates (mount frame)
        self._hotspot_corr_dps = (0.0, 0.0)  # optical-correction part of the command
        self._hotspot_gate_time = 0.0        # frame time the gate was anchored on
        # Star-rejection rate filter: recent fresh-frame detection candidates
        # as (time, cx, cy, boresight_az_sky, boresight_el_sky). Rates are
        # measured against a candidate >= RATE_FILTER_BASELINE_S old -- a
        # short consecutive-frame baseline made the estimate hostage to the
        # capture-to-processing timing skew (error ~ rate * skew/dt), which
        # falsely rejected fast targets.
        self._track_candidates = deque()
        self.handoff_last_frame_seq = None   # HANDOFF's own stale-frame gate
        self.handoff_reject_reason = ""      # last star-filter rejection (UI)

        # PID controllers for PROGRAM track mode
        self.azm_pid = None
        self.alt_pid = None
        self.pid_last_update = 0.0

        # Park request, set by the UI thread and serviced by the control
        # thread (tracking_control) so no blocking serial happens on the UI.
        self.park_requested = False
        self._park_state = None

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

        # PID auto-tuner (autotune.py). Lives outside both control loops: it
        # watches the per-cycle position errors published on this state and
        # writes candidate gains into config_state, which both loops re-read
        # every cycle. Serviced via service_autotune() from whichever loop is
        # active; started/stopped from the PID pane's AUTOTUNE button.
        self.autotuner = None
        # Which per-mode gain profile currently occupies the live pid_* fields
        # (see service_gain_profiles). None until the first PID mode is entered.
        self._active_gain_profile = None

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
            sim_cfg = getattr(self.config_state, 'sim_config', {}) or {}
            if sim_cfg.get('sim_serial_transport', False):
                # Byte-level sim: a REAL NexstarHandController drives the sim
                # mount through the AUX wire protocol, so the app's encoders,
                # parsers, timeouts and SerialCommError recovery are exercised
                # in sim (they are bypassed by the method-level SimMount).
                from simulator import make_sim_serial_controller
                self.telescope_controller = make_sim_serial_controller(
                    self.hardware_sim.mount, self.config_state)
                self.telescope_connected = True
                print("Connected to SIMULATED telescope (byte-level serial transport)")
                return True
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

    def connect_adsb(self):
        """Connect the RTL-SDR ADS-B receiver and start sampling aircraft. The
        receiver is created on first use (deps imported lazily there)."""
        if self.adsb is None:
            from adsb_receiver import AdsbReceiver
            self.adsb = AdsbReceiver(
                self.config_state, self.tracking_vis_state,
                ts=getattr(self.tracking_vis_state, 'ts', None) or self.ts,
                update_status=self.update_status_callback)
            if self.tracking_vis_state is not None:
                self.tracking_vis_state.adsb_tracker = self.adsb.tracker
        ok = self.adsb.connect()
        self.adsb_connected = self.adsb.connected
        self.adsb_status = self.adsb.status
        return ok

    def disconnect_adsb(self):
        """Stop the ADS-B receiver (the tracked aircraft fade out via pruning)."""
        if self.adsb is not None:
            self.adsb.disconnect()
            self.adsb_status = self.adsb.status
        self.adsb_connected = False
        print("Disconnected from ADS-B receiver")

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
            self.park_requested = False   # STOP cancels an in-flight park
            self._park_state = None
            try:
                self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
            except Exception as e:
                print(f"Error sending stop commands: {e}")
            return

        # Park request (Triangle button). Serviced HERE on the control thread:
        # the UI thread only sets the flag, so a slow/parked/dead serial link
        # can no longer freeze the pygame loop, and all wire traffic keeps a
        # single owner. Park preempts tracking until it converges or times out.
        if self.park_requested:
            self._service_park()
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
                self.handoff_last_frame_seq = None
                self._track_candidates.clear()
                self.handoff_reject_reason = ""
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

    def toggle_autotune(self):
        """Start/stop the PID auto-tuner (PID pane's AUTOTUNE button).
        Armable while tracking in PROGRAM or HOTSPOT; stopping keeps the
        best gains found so far (they are already live in config_state --
        save the config to persist them)."""
        from autotune import PIDAutoTuner
        tuner = self.autotuner
        if tuner is not None and tuner.active:
            tuner.stop()
            self._stamp_autotune_label(tuner)
            self._drain_autotune_messages(tuner)
            return
        if self.tracking_mode not in (TrackingMode.PROGRAM, TrackingMode.HOTSPOT):
            if self.update_status_callback:
                self.update_status_callback(
                    "Auto-tune: start PROGRAM or HOTSPOT tracking first")
            return
        self.autotuner = PIDAutoTuner(self.config_state, mode=self.tracking_mode)
        # Remember what we're tuning ON (captured at arm time; the profile of
        # this mode gets stamped with it when the tune finishes).
        self.autotuner.target_label = self._current_target_label()
        self.autotuner.start()
        self._drain_autotune_messages(self.autotuner)

    def service_autotune(self):
        """Feed the auto-tuner one control-cycle sample. Called from BOTH
        control paths (the mount-control cycle and the Rust adapter pump)
        right after the cycle's position errors are published on this state.
        The tuner only scores samples taken while the mode it was armed in is
        genuinely closed-loop on the target -- PROGRAM and HOTSPOT are
        different plants (encoder loop vs. optical loop), so a tune must not
        mix their measurements."""
        tuner = self.autotuner
        if tuner is None:
            return
        if not tuner.active:
            # A tune that finished (converged) but hasn't had its provenance
            # stamped yet gets it here; _stamp_autotune_label is a no-op once
            # stamped, so this costs nothing on later cycles.
            if tuner.phase == 'done':
                self._stamp_autotune_label(tuner)
            return
        tracking = (self.telescope_connected and not self.stopped
                    and self.tracking_mode == tuner.armed_mode)
        if tracking and tuner.armed_mode == TrackingMode.HOTSPOT:
            tracking = self.hotspot_status == "locked"
        tuner.update(time.time(), tracking,
                     self.azm_position_error, self.alt_position_error)
        if not tuner.active and tuner.phase == 'done':
            self._stamp_autotune_label(tuner)
        self._drain_autotune_messages(tuner)

    def _drain_autotune_messages(self, tuner):
        for msg in tuner.take_messages():
            if self.update_status_callback:
                self.update_status_callback(msg)

    # ---------------------------------------------------- per-mode gain profiles
    # The six live gain fields form "the active set" that both control loops,
    # the UI sliders, and the auto-tuner all read/write.
    _GAIN_PROFILE_FIELDS = ('pid_azm_p_gain', 'pid_azm_i_gain', 'pid_azm_d_gain',
                            'pid_alt_p_gain', 'pid_alt_i_gain', 'pid_alt_d_gain')

    def _gain_profile_key(self, mode):
        """Gain-profile key for a tracking mode. HANDOFF runs program-track,
        so it shares PROGRAM's plant (and gains); modes with no PID loop
        (STANDBY, RATE, MTI) have no profile."""
        if mode in (TrackingMode.PROGRAM, TrackingMode.HANDOFF):
            return "PROGRAM"
        if mode == TrackingMode.HOTSPOT:
            return "HOTSPOT"
        return None

    def _current_target_label(self):
        """Short 'what are we tracking' label for the gain-profile stamp,
        e.g. 'satellite ISS (ZARYA)', 'aircraft A1B2C3', 'launch Starship'."""
        vis = self.tracking_vis_state
        if (vis is not None and getattr(vis, 'selected_launch', None)
                and getattr(vis, 'launch_launched', False)):
            return f"launch {vis.selected_launch}"
        _traj, kind, key = active_program_trajectory(vis)
        return f"{kind} {key}" if key is not None else None

    def service_gain_profiles(self):
        """Automatic per-mode PID gain lookup. PROGRAM (encoder loop) and
        HOTSPOT (optical loop) are different plants and want different gains,
        so each keeps its own profile in config.pid_mode_profiles: on a mode
        transition the departing mode's live gains are saved into its profile
        and the arriving mode's profile is loaded into the live fields (first
        entry seeds the profile from the live gains). Runs every control
        cycle, BEFORE service_autotune, from both control paths."""
        cfg = self.config_state
        if cfg is None:
            return
        key = self._gain_profile_key(self.tracking_mode)
        if key is None or key == self._active_gain_profile:
            return

        # A plant change ends any running tune: its measurements belong to
        # the departing mode, and stop() writes its best gains into the live
        # fields so the profile save below captures them rather than a
        # half-tested probe candidate.
        tuner = self.autotuner
        if (tuner is not None and tuner.active
                and self._gain_profile_key(tuner.armed_mode) != key):
            tuner.stop()
            self._stamp_autotune_label(tuner)
            self._drain_autotune_messages(tuner)

        profiles = getattr(cfg, 'pid_mode_profiles', None)
        if not isinstance(profiles, dict):
            profiles = cfg.pid_mode_profiles = {}
        live = {f: float(getattr(cfg, f)) for f in self._GAIN_PROFILE_FIELDS}

        if self._active_gain_profile is not None:
            entry = profiles.get(self._active_gain_profile)
            entry = dict(entry) if isinstance(entry, dict) else {}
            entry['gains'] = live
            profiles[self._active_gain_profile] = entry

        stored = profiles.get(key)
        gains = stored.get('gains') if isinstance(stored, dict) else None
        if isinstance(gains, dict):
            for f in self._GAIN_PROFILE_FIELDS:
                try:
                    setattr(cfg, f, float(gains.get(f, getattr(cfg, f))))
                except (TypeError, ValueError):
                    pass
            tuned_on = stored.get('tuned_on')
            msg = f"PID gains: {key} profile loaded" + (
                f" (tuned on {tuned_on})" if tuned_on else "")
            if self.update_status_callback:
                self.update_status_callback(msg)
        else:
            profiles[key] = {'gains': dict(live)}
            print(f"PID gains: seeded new {key} profile from current gains")
        self._active_gain_profile = key

    def _stamp_autotune_label(self, tuner):
        """Stamp the tuned mode's profile with what the tune ran against
        (target, date, achieved RMS) and the resulting gains. Once per tune,
        and only if it completed at least one measurement."""
        if getattr(tuner, '_label_stamped', False):
            return
        if not any(ax.best_cost is not None for ax in tuner.axes.values()):
            return
        cfg = self.config_state
        key = self._gain_profile_key(tuner.armed_mode)
        if cfg is None or key is None:
            return
        profiles = getattr(cfg, 'pid_mode_profiles', None)
        if not isinstance(profiles, dict):
            profiles = cfg.pid_mode_profiles = {}
        entry = profiles.get(key)
        entry = dict(entry) if isinstance(entry, dict) else {}
        # The tuner has already applied its best gains to the live fields
        # (stop()/convergence do so), so the live snapshot IS the tuned set.
        entry['gains'] = {f: float(getattr(cfg, f))
                          for f in self._GAIN_PROFILE_FIELDS}
        entry['tuned_on'] = tuner.target_label or "unknown target"
        entry['tuned_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        entry['tuned_rms'] = tuner.summary()
        profiles[key] = entry
        tuner._label_stamped = True

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

        # Resolve the program target: a selected satellite first, then a selected
        # aircraft (ADS-B). Both supply the same 8-column trajectory format, so the
        # rest of the loop (bias, feed-forward, limits, PID) is target-agnostic.
        target_traj, target_kind, target_key = active_program_trajectory(self.tracking_vis_state)
        if target_traj is None:
            # Nothing selected - switch to STANDBY and send message
            self.tracking_mode = TrackingMode.STANDBY
            if self.update_status_callback:
                self.update_status_callback("Select a satellite or aircraft for PROGRAM tracking")
            else:
                print("PROGRAM TRACK: No target selected - switched to STANDBY mode")
            return

        # Reset the PIDs when the tracked target changes: integrator state and
        # derivative history accumulated against the OLD target's error is
        # wrong for the new one (it produces a burst of stale correction on
        # switch). reset_program_tracking existed for exactly this but was
        # never called.
        if (target_kind, target_key) != getattr(self, '_last_program_target', (None, None)):
            self._last_program_target = (target_kind, target_key)
            self.reset_program_tracking()

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
            _tau = getattr(self.config_state, 'pid_output_filter_tau_sec', 0.0)
            self.azm_pid.set_output_filter_tau(_tau)
            self.alt_pid.set_output_filter_tau(_tau)

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

            # Get target position from the resolved trajectory (satellite or aircraft)
            if target_traj is not None:
                # Interpolate current target position and rates
                px, py, target_alt, dist, target_az_deg, az_rate, el_rate = interpolate_position_data_and_rates(
                    target_traj,
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
                    # Check hardware safety limits against target position first.
                    # The azm/alt_limit_* values are MOUNT-axis limits (they gate
                    # encoder positions in RATE/HOTSPOT modes), so convert the sky
                    # target through the command transform before comparing --
                    # in AltAz mode mount ALT = 90 - el, so gating raw sky el
                    # against a mount limit is checking the wrong quantity.
                    # choose_mount_target additionally picks the SHORTEST-slew
                    # axis solution (canonical, or over-the-zenith when the
                    # target is on the far side of the sky) among solutions
                    # inside the limits -- pointed west with the target in the
                    # east, the loop must go up through the zenith, not drive
                    # the azimuth axis 180 deg the long way around.
                    target_azm_mount, target_alt_mount, target_flipped = choose_mount_target(
                        self.config_state, current_azm, current_alt,
                        target_az_deg, target_el_deg,
                        limits=(azm_limit_min, azm_limit_max, alt_limit_min, alt_limit_max))
                    if (target_azm_mount > azm_limit_max or target_azm_mount < azm_limit_min or
                        target_alt_mount > alt_limit_max or target_alt_mount < alt_limit_min):
                        self.tracking_mode = TrackingMode.STANDBY
                        target_info = (
                            f"mount AZM:{target_azm_mount:.1f}°/{azm_limit_min:.0f}-{azm_limit_max:.0f}° "
                            f"ALT:{target_alt_mount:.1f}°/{alt_limit_min:.0f}-{alt_limit_max:.0f}° "
                            f"(sky AZ:{target_az_deg:.1f}° EL:{target_el_deg:.1f}°)")
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

                    # Position errors in mount coordinates, in the SAME axis
                    # configuration the gate chose above (recomputing the choice
                    # after lead/bias could disagree at the decision boundary).
                    target_azm_mount, target_alt_mount = mount_target_for(
                        self.config_state, target_az_deg, target_el_deg, target_flipped)
                    az_error = (target_azm_mount - current_azm + 180.0) % 360.0 - 180.0
                    el_error = (target_alt_mount - current_alt + 180.0) % 360.0 - 180.0

                    # Set feed-forward rates from trajectory. The trajectory gives
                    # sky rates; the PID drives the mount. AZM tracks azimuth
                    # directly, but the ALT axis direction depends on the mount
                    # convention AND the chosen configuration: in AltAz the ALT
                    # axis runs opposite sky elevation (ALT = 90 - el), and the
                    # over-the-zenith (flipped) solution runs opposite the
                    # canonical one. Without the right sign the elevation
                    # feed-forward pushes the wrong way.
                    if self.feed_forward_azm_enabled:
                        self.azm_pid.set_feed_forward_rate(az_rate)
                    if self.feed_forward_alt_enabled:
                        alt_sign = -1.0 if getattr(self.config_state, 'mount_mode', 'Eq') == 'AltAz' else 1.0
                        if target_flipped:
                            alt_sign = -alt_sign
                        self.alt_pid.set_feed_forward_rate(alt_sign * el_rate)

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
                        print(f"PROGRAM TRACK: {target_kind}:{target_key} | "
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
            _tau = getattr(self.config_state, 'pid_output_filter_tau_sec', 0.0)
            self.azm_pid.set_output_filter_tau(_tau)
            self.alt_pid.set_output_filter_tau(_tau)

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
            # Check hardware safety limits against target position first, in the
            # MOUNT frame; choose_mount_target also picks the shortest-slew axis
            # solution (see program_track: canonical vs over-the-zenith).
            target_azm_mount, target_alt_mount, target_flipped = choose_mount_target(
                self.config_state, current_azm, current_alt, az_deg, target_el_deg,
                limits=(azm_limit_min, azm_limit_max, alt_limit_min, alt_limit_max))
            if (target_azm_mount > azm_limit_max or target_azm_mount < azm_limit_min or
                target_alt_mount > alt_limit_max or target_alt_mount < alt_limit_min):
                self.tracking_mode = TrackingMode.STANDBY
                target_info = (
                    f"mount AZM:{target_azm_mount:.1f}°/{azm_limit_min:.0f}-{azm_limit_max:.0f}° "
                    f"ALT:{target_alt_mount:.1f}°/{alt_limit_min:.0f}-{alt_limit_max:.0f}° "
                    f"(sky AZ:{az_deg:.1f}° EL:{target_el_deg:.1f}°)")
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

            # Position errors in mount coordinates, in the SAME configuration
            # the gate chose (see program_track).
            target_azm_mount, target_alt_mount = mount_target_for(
                self.config_state, az_deg, target_el_deg, target_flipped)
            az_error = (target_azm_mount - current_azm + 180.0) % 360.0 - 180.0
            el_error = (target_alt_mount - current_alt + 180.0) % 360.0 - 180.0

            # Set feed-forward rates from launch trajectory. ALT sign follows
            # the mount convention and the chosen configuration -- the same
            # rule as program_track (this path used to omit the AltAz
            # negation, a long-standing divergence between the two).
            if self.feed_forward_azm_enabled:
                self.azm_pid.set_feed_forward_rate(az_rate_dps)
            if self.feed_forward_alt_enabled:
                alt_sign = -1.0 if getattr(self.config_state, 'mount_mode', 'Eq') == 'AltAz' else 1.0
                if target_flipped:
                    alt_sign = -alt_sign
                self.alt_pid.set_feed_forward_rate(alt_sign * el_rate_dps)

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
        result = self._handoff_detect(cfg)
        if result is None:
            return  # stale frame: no new information, leave the counter alone
        if result:
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
            self.handoff_status = (self.handoff_reject_reason
                                   or "program track (no detection)")

    def _handoff_detect(self, cfg):
        """Run hotspot detection on a FRESH tracking-camera frame. Returns True
        for an accepted detection, False for a miss (or a detection the star
        filter rejected), and None when no new frame has arrived since the
        last cycle (a stale frame carries no information, so it must neither
        count nor reset the consecutive-detection counter). Updates the
        hotspot diagnostics (centroid, SNR) but never commands the mount."""
        self.handoff_reject_reason = ""
        try:
            cam_index = int(getattr(cfg, 'hotspot_camera_index', 0))
            camera = camera_manager.get_camera(cam_index)
            if camera is None or getattr(camera, 'thread', None) is None:
                self.hotspot_snr = 0.0
                return False
            raw = camera.thread.get_latest_raw()
            if raw is None:
                return False
            frame_seq = getattr(camera.thread, 'latest_raw_seq', None)
            if frame_seq is not None:
                if frame_seq == self.handoff_last_frame_seq:
                    return None
                self.handoff_last_frame_seq = frame_seq
            detection = detect_hotspot(
                raw,
                gate_center=None,
                gate_radius=None,
                snr_threshold=float(getattr(cfg, 'hotspot_snr_threshold', 5.0)),
            )
            if detection is None:
                self.hotspot_snr = 0.0
                return False
            self.hotspot_centroid = (detection.cx, detection.cy)
            self.hotspot_snr = detection.snr
            el_sky = (90.0 - self.current_alt
                      if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz'
                      else self.current_alt)
            verdict, reason = self._detection_rate_filter(
                detection.cx, detection.cy, time.time(),
                f"camera{cam_index + 1}", el_sky)
            if verdict is None:
                # Filter warming up its rate baseline: the detection is
                # neither confirmed nor refuted -- treat like a stale frame
                # (no count, no reset) so a star can't ride the warm-up
                # window into a hand-off.
                return None
            if verdict is False:
                self.handoff_reject_reason = f"rejected: {reason}"
                return False
            return True
        except Exception as e:
            print(f"HANDOFF: detection error: {e}")
            return False

    def _program_target_sky_rates(self):
        """Sky-frame trajectory rates (az_dps, el_dps) of the active program
        target, or None when nothing is selected / interpolable. HOTSPOT uses
        them as trajectory feed-forward; the star filter uses them as the
        expected angular rate of a REAL detection of the selected target."""
        try:
            vis = self.tracking_vis_state
            if vis is None:
                return None
            target_traj, _kind, _key = active_program_trajectory(vis)
            if target_traj is None:
                return None
            px, _py, alt, _dist, az, az_rate, el_rate = (
                interpolate_position_data_and_rates(target_traj, vis.current_tt))
            if px is None or az is None or az_rate is None or el_rate is None:
                return None
            return float(az_rate), float(el_rate)
        except Exception:
            return None

    def _detection_rate_filter(self, det_cx, det_cy, now, cam_name, el_sky):
        """Star-rejection rate gate. Returns (verdict, reason) where verdict
        is True (verified: the rate matches), False (rejected: a star or
        wrong object), or None (unverifiable: the filter is still warming up
        its measurement baseline).

        The detection's implied SKY angular rate is the boresight motion plus
        the pixel drift, measured against the newest recorded candidate at
        least RATE_FILTER_BASELINE_S old (short baselines are corrupted by
        capture-to-processing timing skew; see the constants above). A
        detection of the program target moves at the trajectory rate; a star
        moves at ~sidereal (near zero inertially). With a trajectory
        available, reject mismatches beyond max(hotspot_rate_gate_dps,
        REL_FRACTION * |trajectory rate|); tracking bare, reject near-zero
        (star-like) rates instead. Candidates are recorded regardless of the
        verdict, so a persistent star keeps failing the gate."""
        cfg = self.config_state
        if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz':
            bs_az, bs_el = self.current_azm, 90.0 - self.current_alt
        else:
            bs_az, bs_el = self.current_azm, self.current_alt
        hist = self._track_candidates
        while hist and now - hist[0][0] > RATE_FILTER_MAX_AGE_S:
            hist.popleft()
        base = None
        for cand in reversed(hist):
            if now - cand[0] >= RATE_FILTER_BASELINE_S:
                base = cand
                break
        hist.append((now, det_cx, det_cy, bs_az, bs_el))
        if not getattr(cfg, 'hotspot_star_filter_enabled', True):
            return True, ""
        if base is None:
            return None, ""  # baseline still warming up
        t0, cx0, cy0, az0, el0 = base
        dt = now - t0
        da, de = pixel_offset_to_angles(
            det_cx - cx0, det_cy - cy0,
            pixel_size_um=float(cfg.get_camera_pixel_size(cam_name)),
            focal_length_mm=float(cfg.get_camera_focal_length(cam_name)),
            rotation_deg=float(cfg.get_camera_alignment_rotation(cam_name)),
            el_deg=el_sky,
            x_sign=float(getattr(cfg, 'hotspot_x_sign', 1.0)),
            y_sign=float(getattr(cfg, 'hotspot_y_sign', -1.0)),
        )
        impl_az = ((bs_az - az0 + 180.0) % 360.0 - 180.0 + da) / dt
        impl_el = (bs_el - el0 + de) / dt
        gate = float(getattr(cfg, 'hotspot_rate_gate_dps', 0.15) or 0.15)
        cos_el = max(abs(math.cos(math.radians(el_sky))), 1e-3)
        ref = self._program_target_sky_rates()
        if ref is not None:
            ref_mag = math.hypot(ref[0] * cos_el, ref[1])
            thresh = max(gate, RATE_FILTER_REL_FRACTION * ref_mag)
            diff = math.hypot((impl_az - ref[0]) * cos_el, impl_el - ref[1])
            if diff > thresh:
                return False, f"rate off trajectory by {diff:.2f} deg/s"
        else:
            mag = math.hypot(impl_az * cos_el, impl_el)
            if mag < gate:
                return False, f"star-like rate {mag:.2f} deg/s"
        return True, ""

    def _enter_hotspot_mode(self):
        """Reset state when HOTSPOT is engaged (handed off from another mode)."""
        self.hotspot_gate_center = None      # force a full-frame acquisition
        self.hotspot_acquired = False
        self.hotspot_miss_count = 0
        self.hotspot_last_detection_time = 0.0
        self.hotspot_centroid = None
        self.hotspot_snr = 0.0
        self.hotspot_last_frame_seq = None
        self._hotspot_last_fresh_time = 0.0
        self._hotspot_frame_interval = 0.2
        self._hotspot_cmd_dps = (0.0, 0.0)
        self._hotspot_corr_dps = (0.0, 0.0)
        self._hotspot_gate_time = 0.0
        self._track_candidates.clear()
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
        tau = getattr(cfg, 'pid_output_filter_tau_sec', 0.0)
        self.azm_pid.set_output_filter_tau(tau)
        self.alt_pid.set_output_filter_tau(tau)
        # Trajectory feed-forward is added MANUALLY below (the correction cap
        # must not clamp it), so zero the PIDs' own feed-forward term -- these
        # controllers are shared with PROGRAM track, whose last
        # set_feed_forward_rate would otherwise leak in as a stale rate bias.
        self.azm_pid.set_feed_forward_rate(0.0)
        self.alt_pid.set_feed_forward_rate(0.0)

        # Grab the latest raw frame from the configured tracking camera.
        cam_index = int(getattr(cfg, 'hotspot_camera_index', 0))
        camera = camera_manager.get_camera(cam_index)
        raw = None
        frame_seq = None
        if camera is not None and getattr(camera, 'thread', None) is not None:
            raw = camera.thread.get_latest_raw()
            frame_seq = getattr(camera.thread, 'latest_raw_seq', None)

        # Stale-frame gate: with real exposures longer than the control period
        # the same frame stays "latest" across several cycles. Re-detecting it
        # would feed the PID the same centroid with an advancing dt (integral
        # windup / overshoot), so treat a stale frame as "no new measurement":
        # skip detection this cycle and let the time-based coast/loss logic
        # below decide (a camera that stops producing frames entirely still
        # coasts out and falls back rather than tracking a frozen image).
        frame_is_stale = (frame_seq is not None
                          and frame_seq == self.hotspot_last_frame_seq)
        if frame_seq is not None and not frame_is_stale:
            self.hotspot_last_frame_seq = frame_seq
        if frame_is_stale:
            raw = None

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

        now = time.time()

        # Measured fresh-frame interval (hit OR miss): gate prediction and the
        # correction-rate cap are sized to it, so the loop never outruns its
        # own measurements.
        elapsed_since_fresh = 0.0
        if raw is not None:
            if self._hotspot_last_fresh_time > 0.0:
                elapsed_since_fresh = max(0.0, now - self._hotspot_last_fresh_time)
                self._hotspot_frame_interval = max(0.05, min(2.0, elapsed_since_fresh))
            self._hotspot_last_fresh_time = now

        detection = None
        if raw is not None:
            gate_center = self.hotspot_gate_center
            gate_radius = None
            if gate_center:
                # Predict where the target moved in the frame due to OUR OWN
                # commanded slew since the last fresh frame (the boresight
                # moved, so the target streams the other way in pixels). With
                # a narrow FOV a legitimate correction sweeps a large pixel
                # distance between frames; a static gate loses the target the
                # loop is successfully converging on.
                cam_name = f"camera{cam_index + 1}"
                # Predict with the CORRECTION rates, not the total command:
                # the trajectory feed-forward moves the boresight WITH the
                # target (no apparent pixel drift); only the correction
                # closes on it. Predicting with the total slid the gate off
                # a well-tracked fast target at the full feed-forward rate.
                az_dps, alt_dps = getattr(self, '_hotspot_corr_dps', (0.0, 0.0))
                # Predict from the frame the gate was anchored on (the last
                # detection) -- missed frames in between extend the span, and
                # the commanded rate has been held constant since then.
                gate_dt = max(0.0, now - getattr(self, '_hotspot_gate_time', now))
                if gate_dt > 0.0 and (az_dps or alt_dps):
                    el_sky_dps = (-alt_dps
                                  if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz'
                                  else alt_dps)
                    from simulator import angles_to_pixel
                    el_sky_geom = (90.0 - current_alt
                                   if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz'
                                   else current_alt)
                    pdx, pdy = angles_to_pixel(
                        -az_dps * gate_dt,
                        -el_sky_dps * gate_dt,
                        float(cfg.get_camera_pixel_size(cam_name)),
                        float(cfg.get_camera_focal_length(cam_name)),
                        float(cfg.get_camera_alignment_rotation(cam_name)),
                        el_sky_geom,
                        float(getattr(cfg, 'hotspot_x_sign', 1.0)),
                        float(getattr(cfg, 'hotspot_y_sign', -1.0)))
                    gate_center = (gate_center[0] + pdx, gate_center[1] + pdy)
                # Grow the gate on consecutive misses (classic track-gate
                # growth) so residual prediction error or target motion can't
                # strand the gate while the target is still in frame.
                base = int(getattr(cfg, 'hotspot_gate_radius', 120))
                growth = min(4.0, 1.5 ** min(self.hotspot_miss_count, 8))
                gate_radius = int(base * growth)
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

        # Star-rejection rate gate: a detection whose implied sky rate doesn't
        # match the trajectory (or is star-like when tracking bare) is treated
        # as a miss -- the gate grows and the correction decays, exactly as if
        # nothing was found, so a bright star drifting through the gate can't
        # capture the loop.
        if detection is not None:
            el_sky_gate = (90.0 - current_alt
                           if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz'
                           else current_alt)
            verdict, reason = self._detection_rate_filter(
                detection.cx, detection.cy, now,
                f"camera{cam_index + 1}", el_sky_gate)
            # None = filter warming up: HOTSPOT must keep commanding, so an
            # unverifiable detection is accepted (HANDOFF already verified
            # the object before promoting; only an explicit mismatch demotes
            # it to a miss here).
            if verdict is False:
                self.hotspot_status = f"rejected: {reason}"
                detection = None

        # Trajectory feed-forward (mount frame): the optical correction rides
        # on the program target's sky rates so a moving target no longer needs
        # the capped correction to supply ALL of the tracking rate. Honors the
        # same per-axis FF toggles as PROGRAM track; zero when tracking bare.
        ff_az_mount = ff_el_mount = 0.0
        ff = self._program_target_sky_rates()
        if ff is not None:
            if self.feed_forward_azm_enabled:
                ff_az_mount = ff[0]
            if self.feed_forward_alt_enabled:
                ff_el_mount = (-ff[1]
                               if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz'
                               else ff[1])

        if detection is not None:
            # Pixel error of the object from boresight (image center).
            h, w = raw.shape[:2]
            dx = detection.cx - (w / 2.0)
            dy = detection.cy - (h / 2.0)

            cam_name = f"camera{cam_index + 1}"
            # The pixel->angle geometry needs the SKY elevation (azimuth
            # compresses by cos(el) on the sky). In AltAz the mount ALT axis
            # is 90-el: passing it raw overstates the azimuth error by
            # cos(el)/cos(ALT) -- >1.5x at el=30 -- making every correction
            # overshoot and the loop oscillate divergently.
            el_sky = (90.0 - current_alt
                      if getattr(cfg, 'mount_mode', 'Eq') == 'AltAz'
                      else current_alt)
            az_error, el_error = pixel_offset_to_angles(
                dx, dy,
                pixel_size_um=float(cfg.get_camera_pixel_size(cam_name)),
                focal_length_mm=float(cfg.get_camera_focal_length(cam_name)),
                rotation_deg=float(cfg.get_camera_alignment_rotation(cam_name)),
                el_deg=el_sky,
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
            self._hotspot_gate_time = now
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
                    # Cap each axis's CORRECTION so it covers at most ~90% of
                    # the remaining error before the NEXT measurement arrives
                    # (measured fresh-frame interval). Without this, a single
                    # strong correction at a slow frame rate overshoots, exits
                    # its own tracking gate, and then COASTS at full rate
                    # through the loss window -- the FOV-sized stair-step.
                    # The trajectory feed-forward rides UNDER the cap: it is
                    # not a correction, it's the target's own motion.
                    cap = float(getattr(cfg, 'hotspot_max_rate_dps', 2.0) or 2.0)
                    interval = getattr(self, '_hotspot_frame_interval', 0.2)
                    az_cap = min(cap, 0.9 * abs(az_error) / interval)
                    el_cap = min(cap, 0.9 * abs(el_error) / interval)
                    az_corr = max(-az_cap, min(az_cap, az_pid_output * 360.0))
                    el_corr = max(-el_cap, min(el_cap, el_pid_output * 360.0))
                    az_total = (ff_az_mount + az_corr) / 360.0
                    el_total = (ff_el_mount + el_corr) / 360.0

                    az_eff = self._issue_axis_rate(
                        Targets.AZM, az_total, hotspot_discrete_step(az_total))
                    el_eff = self._issue_axis_rate(
                        Targets.ALT, el_total, hotspot_discrete_step(el_total))
                    # Remember what we commanded (mount frame): gate prediction
                    # uses the TOTAL boresight motion; the miss-path decay
                    # bleeds only the correction while feed-forward keeps
                    # running (coasting follows the trajectory, not a frozen
                    # rate).
                    self._hotspot_corr_dps = (az_corr, el_corr)
                    self._hotspot_cmd_dps = (az_eff or 0.0, el_eff or 0.0)
                except Exception as e:
                    print(f"HOTSPOT: error sending slew commands: {e}")
            return

        # No detection this cycle. A stale frame is "no new information", not a
        # miss -- only count misses against frames we actually examined.
        if not frame_is_stale:
            self.hotspot_miss_count += 1
            # Bleed off the held CORRECTION: a correction that hasn't been
            # re-confirmed by a detection must not keep integrating (that
            # runaway is what turned a single overshoot into an FOV-sized
            # stair-step). Halve per missed frame. The trajectory feed-forward
            # is NOT decayed -- coasting follows the target's motion.
            az_c, el_c = getattr(self, '_hotspot_corr_dps', (0.0, 0.0))
            az_c *= 0.5
            el_c *= 0.5
            if abs(az_c) < 1e-3:
                az_c = 0.0
            if abs(el_c) < 1e-3:
                el_c = 0.0
            self._hotspot_corr_dps = (az_c, el_c)
            if (self.telescope_connected and not self.stopped
                    and (az_c or el_c or ff_az_mount or ff_el_mount)):
                try:
                    az_total = (ff_az_mount + az_c) / 360.0
                    el_total = (ff_el_mount + el_c) / 360.0
                    az_eff = self._issue_axis_rate(
                        Targets.AZM, az_total, hotspot_discrete_step(az_total))
                    el_eff = self._issue_axis_rate(
                        Targets.ALT, el_total, hotspot_discrete_step(el_total))
                    self._hotspot_cmd_dps = (az_eff or 0.0, el_eff or 0.0)
                except Exception as e:
                    print(f"HOTSPOT: error decaying rates: {e}")
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

    def _issue_axis_rate(self, target, pid_output_rev_s, discrete_cmd, max_dps=None):
        """Send one axis's rate to the mount.

        In continuous-rate mode use the fine 24-bit variable-rate (guide-rate)
        primitive so the commanded rate matches the target's motion exactly and
        avoids the MC_MOVE quantization sawtooth. Fall back to the discrete
        MC_MOVE step when the requested rate exceeds guide_rate_max_dps (the
        regime where the guide-rate command may not be honored, e.g. the
        near-zenith keyhole) or when continuous mode is off.

        ``max_dps`` caps the commanded rate in BOTH paths. HOTSPOT passes its
        hotspot_max_rate_dps: a centering loop that lunges at slew rates yanks
        the target out of its own tracking gate before the next frame arrives
        (the acquire -> yank -> lose -> fallback stair-step failure).
        """
        cfg = self.config_state
        dps = pid_output_rev_s * 360.0
        if max_dps is not None:
            dps = max(-float(max_dps), min(float(max_dps), dps))
        if getattr(cfg, 'continuous_rate_tracking', False) and hasattr(self.telescope_controller, 'hc_set_rate_dps'):
            if abs(dps) <= float(getattr(cfg, 'guide_rate_max_dps', 5.0)):
                # Dedup: the firmware HOLDS the last commanded rate, and each
                # AUX transaction costs ~30 ms of 9600-baud wire time (measured
                # 2026-07-26: full read+command cycle = 7.7 Hz vs 15 Hz target;
                # reads alone support ~16 Hz). Resend only when the wire-
                # quantized value changes, with a keepalive so an out-of-band
                # stop can't leave the cache lying for more than a second.
                wire = ('g', int(round(dps * GUIDE_COUNTS_PER_DPS)))
                if not self._rate_cmd_repeats(target, wire):
                    self.telescope_controller.hc_set_rate_dps(target, dps)
                    self._rate_cmd_cache[target] = (wire, time.monotonic())
                return dps
        if max_dps is not None and discrete_cmd != 0:
            sign = 1 if discrete_cmd > 0 else -1
            mag = abs(int(discrete_cmd))
            # Largest discrete step whose rate fits under the cap (keep at
            # least 1 so a capped axis still creeps toward center).
            while mag > 1 and RATES.get(mag, 0.0) * 360.0 > float(max_dps):
                mag -= 1
            discrete_cmd = sign * mag
        # Same dedup for the discrete path: MC_MOVE also persists until changed.
        wire = ('m', int(discrete_cmd))
        if not self._rate_cmd_repeats(target, wire):
            self.telescope_controller.hc_slew_fixed(target, discrete_cmd)
            self._rate_cmd_cache[target] = (wire, time.monotonic())
        sign = 1.0 if discrete_cmd >= 0 else -1.0
        return sign * RATES.get(abs(int(discrete_cmd)), 0.0) * 360.0

    # Resend an unchanged rate at least this often: bounds how long a stale
    # cache entry can mask an out-of-band stop/change sent past the cache.
    RATE_CMD_KEEPALIVE_SEC = 1.0

    def _rate_cmd_repeats(self, target, wire):
        """True when `wire` matches the last command sent to `target` recently
        enough that the firmware is already holding it (skip the transaction)."""
        last = self._rate_cmd_cache.get(target)
        return (last is not None and last[0] == wire
                and time.monotonic() - last[1] < self.RATE_CMD_KEEPALIVE_SEC)

    PARK_TIMEOUT_SEC = 90.0
    PARK_TOLERANCE_DEG = 1.0
    PARK_REISSUE_SEC = 1.0

    def _service_park(self):
        """One control-thread cycle of the park sequence.

        Drives both axes to the configured offsets (raw encoder frame, like
        the goto command itself) with wrap-aware convergence and a timeout --
        the old UI-thread busy-loop compared unwrapped angles, so an offset
        near 0 with the encoder reading 359.9 looped forever while the UI was
        frozen. Uses the position already polled this cycle instead of extra
        serial reads. Serial faults propagate to MountControlThread's
        consecutive-fault watchdog like any other cycle fault.
        """
        cfg = self.config_state
        try:
            target_azm = float(getattr(cfg, 'azm_offset_str', 0.0) or 0.0)
            target_alt = float(getattr(cfg, 'alt_offset_str', 0.0) or 0.0)
        except (TypeError, ValueError):
            target_azm = target_alt = 0.0

        now = time.time()
        if self._park_state is None:
            self._park_state = {'start': now, 'last_cmd': 0.0}
            if self.update_status_callback:
                self.update_status_callback(
                    f"Parking to AZM {target_azm:.1f}° / ALT {target_alt:.1f}°...")
        ps = self._park_state

        if now - ps['start'] > self.PARK_TIMEOUT_SEC:
            self.park_requested = False
            self._park_state = None
            if self.update_status_callback:
                self.update_status_callback(
                    f"Park TIMED OUT after {self.PARK_TIMEOUT_SEC:.0f}s - check the mount")
            return

        # Wrap-aware error in the raw encoder frame (goto targets raw degrees).
        err_azm = (target_azm - self.current_azm_raw + 180.0) % 360.0 - 180.0
        err_alt = (target_alt - self.current_alt_raw + 180.0) % 360.0 - 180.0
        if abs(err_azm) <= self.PARK_TOLERANCE_DEG and abs(err_alt) <= self.PARK_TOLERANCE_DEG:
            self.park_requested = False
            self._park_state = None
            if self.update_status_callback:
                self.update_status_callback("Park complete")
            print("Park complete")
            return

        # Goto is a persistent command; re-issue at a gentle cadence rather
        # than every cycle so the wire isn't saturated while the mount slews.
        if now - ps['last_cmd'] >= self.PARK_REISSUE_SEC:
            self.telescope_controller.hc_goto_fast(Targets.AZM, target_azm, 0, 0)
            self.telescope_controller.hc_goto_fast(Targets.ALT, target_alt, 0, 0)
            ps['last_cmd'] = now

    def _hotspot_stop_motion(self):
        """Best-effort stop of both axes."""
        self._hotspot_cmd_dps = (0.0, 0.0)
        self._hotspot_corr_dps = (0.0, 0.0)
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

                # Stop movement immediately when stopped. Guarded: a serial
                # fault on the UI thread must not crash the app -- the control
                # thread's stop check re-issues the stop every cycle anyway.
                if self.stopped and self.telescope_connected:
                    try:
                        self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                        self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
                    except Exception as e:
                        print(f"Stop command failed on UI thread ({e}) - "
                              "control thread will retry")

            elif event.button == 2:  # Square button for tare
                print(f"Taring joystick axes")
                self.tare_current_joystick()

            elif event.button == 3:  # Triangle/Park button for park command
                if self.telescope_connected:
                    # Request only -- the control thread runs the park sequence
                    # (_service_park) so the UI never blocks on serial I/O.
                    self.park_requested = True
                    print("Park requested - control thread will drive to the "
                          "configured offsets")
                else:
                    print("Cannot park: telescope not connected")

            elif event.button == 4:  # Share button - cycle bias adjust mode
                self.cycle_bias_mode()
                return

            elif event.button == 6:  # Op (Options) button - mount mode toggle
                # Cycle through AltAz, AltAz-Side, Eq, and Passthrough modes
                if self.mount_mode == "AltAz":
                    self.mount_mode = "AltAz-Side"
                elif self.mount_mode == "AltAz-Side":
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


# ---------------------------------------------------------------------------
# Rendering / panel half of the old monolith, split into joystick_panels.py.
# Re-import EVERY name it defines (including underscore-private ones) so this
# module remains a complete facade: all existing `from joystick_controller
# import X` and `jc.X` references keep working unchanged.
# ---------------------------------------------------------------------------
from joystick_panels import (  # noqa: E402,F401
    render_position_display,
    render_connection_controls,
    render_adsb_connection_controls,
    _adsb_fit_from_track_x,
    handle_adsb_fit_slider_mouse_events,
    render_joystick_target_panel,
    handle_joystick_target_panel_click,
    _filter_field_text,
    render_joystick_status,
    render_capture_controls,
    JL_SLIDER_W,
    JL_ROT_SLIDER_W,
    JL_GAMMA_MIN,
    JL_GAMMA_MAX,
    JL_ROTATION_RANGE,
    _joystick_camera_layout,
    _process_feed_surface,
    _draw_feed_roi,
    _draw_feed_timestamps,
    _draw_jl_slider,
    _camera_fov_deg,
    _cam_rotation,
    _draw_cam2_fov_in_cam1,
    _draw_feed_axes,
    render_camera_feeds,
    render_joystick_camera_controls,
    _apply_jl_camera_drag,
    _jl_pos_over_control,
    _jl_reset_camera_config,
    _jl_save_camera_config,
    handle_joystick_camera_control_events,
    handle_joystick_mode_mouse_events,
    _handle_capture_toggle,
    _plate_solve_worker,
    toggle_plate_solve,
    apply_instantaneous_alignment,
    draw_solve_centroids_on_feed,
    render_plate_solve_panel,
    handle_plate_solve_mouse_events,
    render_pid_diagnostics,
    _draw_strip_chart,
    render_tracking_strip_charts,
    _NAVBALL_GRID_CACHE,
    _NAVBALL_BASE_CACHE,
    _navball_grid,
    active_program_trajectory,
    _navball_active_target,
    render_navball,
    render_bias_control_grid,
    render_feed_forward_toggle_buttons,
    handle_bias_control_mouse_events,
    render_pid_gain_sliders,
    _lead_from_track_x,
    handle_lead_slider_mouse_events,
    handle_pid_sliders_mouse_events,
    handle_ff_toggle_mouse_events,
)
