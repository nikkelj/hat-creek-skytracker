import pygame
import serial
import serial.tools.list_ports
import math
import time
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
# every physical button.
BUTTON_FUNCTIONS = {
    0:  "Capture",
    1:  "Stop",
    2:  "Tare axes",
    3:  "Park",
    4:  "Mount mode",
    8:  "Track mode +",
    10: "Bias fine/crs",
    11: "Feed-fwd",
    15: "Track mode -",
}


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

    return {
        'pid': pygame.Rect(pid_x, pid_y, pid_w, pid_h),
        'bias': pygame.Rect(bias_x, bias_y, bias_w, bias_h),
        'diag': pygame.Rect(diag_x, diag_y, diag_w, diag_h),
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

        # Telescope connection state
        self.telescope_connected = False
        self.telescope_controller = None
        self.selected_port = None
        self.available_ports = []

        # Hardware simulator (set by main.py); when sim is enabled the connect
        # calls below hand back the sim mount instead of a real serial controller.
        self.hardware_sim = None

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
        self.bias_azm_deg = 0.0
        self.bias_alt_deg = 0.0
        self.bias_control_mode = "coarse"

        # PID diagnostic state
        self.azm_position_error = 0.0
        self.alt_position_error = 0.0
        self.azm_rate_error = 0.0
        self.alt_rate_error = 0.0
        self.azm_target_rate = 0.0
        self.alt_target_rate = 0.0
        self.azm_pid_output = 0.0
        self.alt_pid_output = 0.0

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

        # Universal stop check - stop movement in any mode when stopped
        if self.stopped:
            try:
                self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
            except Exception as e:
                print(f"Error sending stop commands: {e}")
            return

        # Reset the STANDBY one-shot stop guard whenever we are in any active
        # mode, so re-entering STANDBY will again issue a single stop.
        if self.tracking_mode != TrackingMode.STANDBY:
            self._standby_motion_stopped = False

        # Run per-mode entry logic on a mode transition.
        if self.tracking_mode != self._prev_dispatch_mode:
            if self.tracking_mode == TrackingMode.HOTSPOT:
                self._enter_hotspot_mode()
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

                    # Apply bias corrections. The AZ bias is an on-sky
                    # (cross-elevation) nudge so it shifts the image consistently;
                    # convert to an azimuth offset by dividing by cos(el), since
                    # azimuth compresses toward the zenith (a raw azimuth bias is
                    # nearly invisible at high elevation). Clamp cos(el) so the
                    # scaling stays finite right at the zenith.
                    cos_el = max(math.cos(math.radians(target_el_deg)), 0.087)  # >= cos(85°)
                    target_az_deg += self.bias_azm_deg / cos_el
                    target_el_deg += self.bias_alt_deg

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
                              f"AZ:{current_azm:.2f}→{target_az_deg:.2f}({az_error:+.2f}) | "
                              f"EL:{current_alt:.2f}→{target_el_deg:.2f}({el_error:+.2f}) | "
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

            # Apply bias corrections (AZ bias is on-sky cross-elevation; divide by
            # cos(el) so it shifts the image consistently near the zenith).
            cos_el = max(math.cos(math.radians(target_el_deg)), 0.087)  # >= cos(85°)
            az_deg += self.bias_azm_deg / cos_el
            target_el_deg += self.bias_alt_deg

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
                print(f"LAUNCH TRACK: {launch_name} | AZ:{current_azm:.2f}→{az_deg:.2f}({az_error:+.2f}) | EL:{current_alt:.2f}→{target_el_deg:.2f}({el_error:+.2f}) | CMD AZ:{az_command} EL:{el_command}")

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
        """Handoff track mode - stub implementation"""
        # TODO: Implement handoff track mode
        print("Handoff track mode - stub")

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
        elapsed = (now - self.hotspot_last_detection_time) if self.hotspot_acquired else float('inf')

        if self.hotspot_acquired and elapsed < coast_time:
            # Coast: leave the last continuous slew running so a brief dropout
            # (cloud, frame glitch) doesn't jerk the mount.
            self.hotspot_status = "coasting"
            return

        # Lock lost (or never acquired within the coast window): stop and hand
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

            elif event.button == 4:  # Share button for mount mode toggle
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

            elif event.button == 15:  # Pad button for backward tracking mode cycle
                self.cycle_tracking_mode_backward()
                return  # Don't process any other buttons after handling mode cycle

        elif event.type == pygame.JOYHATMOTION:
            # Handle D-pad input for bias control
            hat_x, hat_y = event.value

            # Bias adjustment increment (coarse vs fine depending on L1 toggle)
            step = 0.01 if self.bias_control_mode == "fine" else 0.1

            # D-pad left/right for azimuth bias
            if hat_x > 0:  # D-pad right
                self.bias_azm_deg = max(-3.0, min(3.0, self.bias_azm_deg + step))
            elif hat_x < 0:  # D-pad left
                self.bias_azm_deg = max(-3.0, min(3.0, self.bias_azm_deg - step))

            # D-pad up/down for elevation bias
            if hat_y > 0:  # D-pad up
                self.bias_alt_deg = max(-3.0, min(3.0, self.bias_alt_deg + step))
            elif hat_y < 0:  # D-pad down
                self.bias_alt_deg = max(-3.0, min(3.0, self.bias_alt_deg - step))

        elif event.type == pygame.JOYBUTTONDOWN:
            # Handle L1/R1 buttons if not already handled above (for bias mode toggle)
            if event.button == 10:  # L1 button
                self.bias_control_mode = "fine" if self.bias_control_mode == "coarse" else "coarse"
                print(f"Bias control mode: {self.bias_control_mode}")
                return
            elif event.button == 11:  # R1 button - toggle feed-forward

                # Toggle feed-forward for both axes
                self.feed_forward_azm_enabled = not self.feed_forward_azm_enabled
                self.feed_forward_alt_enabled = not self.feed_forward_alt_enabled

                # Update PID controllers if they exist
                if self.azm_pid:
                    self.azm_pid.set_feed_forward_enabled(self.feed_forward_azm_enabled)
                if self.alt_pid:
                    self.alt_pid.set_feed_forward_enabled(self.feed_forward_alt_enabled)

                print(f"Feed-forward AZ: {self.feed_forward_azm_enabled}, EL: {self.feed_forward_alt_enabled}")
                return

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

        func = BUTTON_FUNCTIONS.get(i, "-")
        func_color = (220, 220, 220) if i in BUTTON_FUNCTIONS else (120, 120, 120)
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

            # Handle satellite selection on click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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

    # "Needed" = actively tracking (PROGRAM or HOTSPOT); otherwise greyed.
    active = joystick_state.tracking_mode in (TrackingMode.PROGRAM, TrackingMode.HOTSPOT)

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
    mode_label = "rate cmd: CONTINUOUS (guide-rate)" if continuous else "rate cmd: DISCRETE (MC_MOVE)"
    mode_c = (120, 200, 255) if (active and continuous) else (val_c if active else (90, 90, 100))
    display.menu_screen.blit(display.tiny_font.render(mode_label, True, mode_c), (x_start + 10, y_start + 80))
    bias_text = f"bias AZ {getattr(joystick_state,'bias_azm_deg',0.0):+.1f}° EL {getattr(joystick_state,'bias_alt_deg',0.0):+.1f}°"
    display.menu_screen.blit(display.tiny_font.render(bias_text, True, val_c if active else (90, 90, 100)),
                             (x_start + 10, y_start + 94))

def render_bias_control_grid(display, joystick_state):
    """
    Render visual grid for manual bias control with buttons and current values.
    Anchored above the PID pane in the bottom-right of the upper-left quadrant.
    Always drawn so its location is visible; greyed out and non-interactable
    unless in PROGRAM mode.
    """
    enabled = joystick_state.tracking_mode == TrackingMode.PROGRAM

    # Position above the PID pane, hugging the bottom-right of the quadrant
    pane = joystick_panel_layout(display)['bias']
    x_start, y_start = pane.x, pane.y
    width, height = pane.width, pane.height

    # Background rectangle
    pygame.draw.rect(display.menu_screen, (60, 60, 80),
                     (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, (120, 120, 150),
                     (x_start, y_start, width, height), 1)

    # Title
    title_text = display.small_font.render("Bias Control", True, (255, 255, 255))
    display.menu_screen.blit(title_text, (x_start + 10, y_start + 5))

    # Current bias mode indicator
    mode_color = (100, 255, 100) if joystick_state.bias_control_mode == "coarse" else (255, 100, 100)
    mode_text = display.tiny_font.render(f"Mode: {joystick_state.bias_control_mode.upper()}", True, mode_color)
    display.menu_screen.blit(mode_text, (x_start + 10, y_start + 25))

    # Current values display
    current_values_text = display.tiny_font.render(f"AZ: {joystick_state.bias_azm_deg:.1f}° EL: {joystick_state.bias_alt_deg:.1f}°", True, (255, 255, 255))
    display.menu_screen.blit(current_values_text, (x_start + 10, y_start + 40))

    # Button grid for manual adjustment
    button_size = 25
    button_spacing = 8
    grid_start_x = x_start + 20
    grid_start_y = y_start + 60

    # Button layout (arrow keys grid)
    button_positions = [
        ("← Az", grid_start_x, grid_start_y, (255, 150, 150)),                    # Left Arrow - Decrease AZM
        ("↑ El", grid_start_x + button_size + button_spacing, grid_start_y, (150, 255, 150)),  # Up Arrow - Increase ALT
        ("→ Az", grid_start_x + 2*(button_size + button_spacing), grid_start_y, (255, 150, 150)), # Right Arrow - Increase AZM
        ("↓ El", grid_start_x + button_size + button_spacing, grid_start_y + button_size + button_spacing, (150, 255, 150))  # Down Arrow - Decrease ALT
    ]

    # Store button rects for mouse click handling
    button_rects = []

    # Draw buttons
    for label, bx, by, color in button_positions:
        button_rect = pygame.Rect(bx, by, button_size, button_size)
        button_rects.append((label, button_rect))

        # Check mouse hover
        mouse_pos = pygame.mouse.get_pos()
        hover = button_rect.collidepoint(mouse_pos)

        # Draw button background
        bg_color = tuple(min(255, c + 40) for c in color) if hover else color
        pygame.draw.rect(display.menu_screen, bg_color, button_rect)
        pygame.draw.rect(display.menu_screen, (200, 200, 200), button_rect, 1)

        # Draw label
        label_text = display.tiny_font.render(label, True, (0, 0, 0))
        text_rect = label_text.get_rect(center=button_rect.center)
        display.menu_screen.blit(label_text, text_rect)

    # Store button rects in joystick state for mouse handling
    joystick_state.bias_button_rects = button_rects

    # Grey out when not in PROGRAM mode (visible but not interactable)
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
    Handle mouse clicks on bias control buttons.
    Called from main event loop when buttons are clicked.
    """
    if not hasattr(joystick_state, 'bias_button_rects'):
        return False

    # Bias control is only interactable in PROGRAM mode (greyed out otherwise)
    if joystick_state.tracking_mode != TrackingMode.PROGRAM:
        return False

    step = 0.01 if joystick_state.bias_control_mode == "fine" else 0.1

    for label, rect in joystick_state.bias_button_rects:
        if rect.collidepoint(mouse_pos):
            if "← Az" in label:
                joystick_state.bias_azm_deg = max(-3.0, min(3.0, joystick_state.bias_azm_deg - step))
                print(f"Bias control: AZ decreased by {step}° to {joystick_state.bias_azm_deg:.2f}°")
            elif "→ Az" in label:
                joystick_state.bias_azm_deg = max(-3.0, min(3.0, joystick_state.bias_azm_deg + step))
                print(f"Bias control: AZ increased by {step}° to {joystick_state.bias_azm_deg:.2f}°")
            elif "↑ El" in label:
                joystick_state.bias_alt_deg = max(-3.0, min(3.0, joystick_state.bias_alt_deg + step))
                print(f"Bias control: EL increased by {step}° to {joystick_state.bias_alt_deg:.2f}°")
            elif "↓ El" in label:
                joystick_state.bias_alt_deg = max(-3.0, min(3.0, joystick_state.bias_alt_deg - step))
                print(f"Bias control: EL decreased by {step}° to {joystick_state.bias_alt_deg:.2f}°")
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

    # Grey out the slider area when not in PROGRAM mode (the feed-forward strip
    # below is greyed separately by render_feed_forward_toggle_buttons()).
    if not enabled:
        slider_region = pygame.Rect(x_start + 1, y_start + 18, width - 2, 151)
        _draw_disabled_scrim(display, slider_region)

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
