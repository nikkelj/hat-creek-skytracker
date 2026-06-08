"""
PID Control System for Satellite Tracking

This module contains the PID controller implementation and related functions
for precise satellite tracking control using azimuth and elevation axes.
"""

import numpy as np
import math
import time
from lib.auxstar import RATES


class PIDController:
    """
    PID Controller for satellite tracking using azimuth and elevation axes.
    Maps position error to discrete rate commands (0-9) for telescope control.
    Supports feed-forward control using trajectory rates.
    """

    def __init__(self, p_gain=1.0, i_gain=0.0, d_gain=0.0, axis_name="", feed_forward_enabled=False):
        """
        Initialize PID controller.

        Args:
            p_gain: Proportional gain
            i_gain: Integral gain
            d_gain: Derivative gain
            axis_name: Name of axis (for debug logging)
            feed_forward_enabled: Whether to use feed-forward from trajectory rates
        """
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.axis_name = axis_name
        self.feed_forward_enabled = feed_forward_enabled

        # PID state
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.previous_measurement = None  # for derivative-on-measurement
        self.last_update_time = None

        # Feed-forward state
        self.current_feed_forward_rate = 0.0

        # Error tracking for display
        self.current_position_error = 0.0
        self.current_rate_error = 0.0
        self.current_target_rate = 0.0
        self.current_pid_output = 0.0

        # Rate limiting
        self.max_rate = 9  # Maximum discrete rate
        self.min_rate = 1  # Minimum positive rate

    def reset(self):
        """Reset PID controller state."""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.previous_measurement = None
        self.last_update_time = None

    def update_gains(self, p_gain, i_gain, d_gain):
        """Update PID gains."""
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain

    def set_feed_forward_rate(self, feed_forward_rate_deg_per_sec):
        """Set the current feed-forward rate from trajectory tracking.

        The PID output and the discrete RATES table are expressed in
        revolutions/second, but trajectory rates arrive in degrees/second.
        Convert here (1 rev = 360 deg) so the feed-forward term shares the same
        units as the feedback output. Without this conversion the feed-forward
        contribution is ~360x too large and immediately saturates the rate
        command, which is why feed-forward tracking was previously unusable.
        """
        if self.feed_forward_enabled:
            self.current_feed_forward_rate = feed_forward_rate_deg_per_sec / 360.0
        else:
            self.current_feed_forward_rate = 0.0

    def set_feed_forward_enabled(self, enabled):
        """Enable or disable feed-forward control."""
        self.feed_forward_enabled = enabled
        if not enabled:
            self.current_feed_forward_rate = 0.0

    def compute_pid_output(self, error_degrees, dt_seconds, measurement_degrees=None):
        """
        Compute PID output for given error with optional feed-forward.

        Args:
            error_degrees: Position error in degrees
            dt_seconds: Time since last update (for integral/derivative)
            measurement_degrees: Current measured position in degrees. When
                provided, the derivative term is taken on the measurement
                instead of the error. This avoids the "derivative kick" spike
                that occurs when the setpoint (or feed-forward) changes, which
                is common when tracking a moving trajectory.

        Returns:
            tuple: (pid_output, discrete_rate) - discrete_rate is signed (-9 to +9)
        """
        if dt_seconds <= 0:
            return 0.0, 0

        # Clamp error to reasonable range
        error_degrees = np.clip(error_degrees, -180, 180)

        # Proportional term
        p_term = self.p_gain * error_degrees

        # Derivative term: on measurement when available (no derivative kick),
        # otherwise on error (legacy behavior).
        if measurement_degrees is not None:
            if self.previous_measurement is None:
                d_term = 0.0
            else:
                delta = measurement_degrees - self.previous_measurement
                # AZM wraps at 360 deg; keep the delta on the short arc so a
                # 0<->360 crossing doesn't produce a spurious derivative spike.
                if self.axis_name == "AZM":
                    delta = (delta + 180.0) % 360.0 - 180.0
                d_term = -self.d_gain * delta / dt_seconds
            self.previous_measurement = measurement_degrees
        else:
            d_term = self.d_gain * (error_degrees - self.previous_error) / dt_seconds

        self.previous_error = error_degrees

        # Integral term with conditional-integration anti-windup. We tentatively
        # accumulate, compute the output, and if the command is saturated *and*
        # the error would push it further into saturation, we roll back this
        # cycle's accumulation so the integrator cannot wind up against the rate
        # ceiling.
        self.integral_error += error_degrees * dt_seconds
        self.integral_error = np.clip(self.integral_error, -100, 100)
        i_term = self.i_gain * self.integral_error

        pid_feedback_output = p_term + i_term + d_term
        total_output = pid_feedback_output + self.current_feed_forward_rate

        max_rate = RATES[self.max_rate]  # saturation threshold (rev/sec)
        if abs(total_output) > max_rate and np.sign(error_degrees) == np.sign(total_output):
            self.integral_error -= error_degrees * dt_seconds
            self.integral_error = np.clip(self.integral_error, -100, 100)
            i_term = self.i_gain * self.integral_error
            pid_feedback_output = p_term + i_term + d_term
            total_output = pid_feedback_output + self.current_feed_forward_rate

        # Store feedback components for display
        self.current_feedback_output = pid_feedback_output

        # Store diagnostic values for display
        self.current_position_error = error_degrees
        self.current_pid_output = total_output

        # Convert to signed discrete rate using combined output
        discrete_rate = self._error_to_discrete_rate(total_output, dt_seconds)

        # Store target rate and rate error for display (use sign and magnitude correctly)
        rate_magnitude = abs(discrete_rate)
        expected_rate = RATES[rate_magnitude] if rate_magnitude in RATES else 0.0
        self.current_rate_error = total_output - (expected_rate * (1 if discrete_rate >= 0 else -1))

        return total_output, discrete_rate

    def _error_to_discrete_rate(self, pid_output_deg_per_sec, dt_seconds):
        """
        Map PID output to signed discrete rate setting (-9 to +9).

        This function properly preserves the sign information from PID output,
        ensuring correct directionality for telescope motion.

        Algorithm:
        1. Extract sign from PID output (direction)
        2. Compute optimal rate magnitude for requested velocity
        3. Return signed rate (-9 to +9)

        Returns:
            Signed discrete rate: positive for clockwise/increasing,
                                negative for counter-clockwise/decreasing
        """
        # Store the sign of the requested motion
        sign = 1 if pid_output_deg_per_sec >= 0 else -1

        # Get magnitude for rate computation
        requested_velocity = abs(pid_output_deg_per_sec)

        # Zero velocity gets zero rate
        if requested_velocity < 0.01:
            return 0

        # Find the discrete rate magnitude that best matches the requested velocity
        best_rate_magnitude = 0
        min_distance = float('inf')

        for rate_idx in range(10):  # 0 to 9
            rate_velocity = RATES[rate_idx]
            distance = abs(rate_velocity - requested_velocity)

            # Bias toward higher rates for stability (reduce undershoot)
            if rate_velocity >= requested_velocity:
                distance *= 0.9  # Prefer higher rates

            if distance < min_distance:
                min_distance = distance
                best_rate_magnitude = rate_idx

        # Return signed rate (-9 to +9) - CRITICAL: preserves directionality
        return sign * best_rate_magnitude

    def _max_fixable_error_degrees(self, dt_seconds):
        """
        Calculate maximum position error that can be fixed in one control cycle.

        Args:
            dt_seconds: Control cycle time

        Returns:
            Maximum fixable error in degrees
        """
        # Maximum theoretical correction per cycle at max rate
        max_correction_per_cycle = RATES[self.max_rate] * dt_seconds
        return max_correction_per_cycle

    def get_current_rates(self, error_degrees, dt_seconds=0.1, measurement_degrees=None):
        """
        Get current PID output and discrete rate for this control cycle.

        Args:
            error_degrees: Current position error in degrees
            dt_seconds: Time since last update
            measurement_degrees: Current measured position (enables
                derivative-on-measurement; see compute_pid_output)

        Returns:
            tuple: (pid_output_deg_per_sec, discrete_rate)
        """
        import time
        current_time = time.time()

        if self.last_update_time is None:
            dt_seconds = 0.1  # Default dt for first call
        else:
            dt_seconds = current_time - self.last_update_time

        self.last_update_time = current_time

        pid_output, discrete_rate = self.compute_pid_output(
            error_degrees, dt_seconds, measurement_degrees=measurement_degrees
        )

        return pid_output, discrete_rate

    def compute_axis_error(self, current_position_rad, target_position_rad):
        """
        Compute position error for an axis, handling wraparound for azimuth.

        Args:
            current_position_rad: Current position in radians
            target_position_rad: Target position in radians

        Returns:
            Error in radians (-pi to pi)
        """
        import math

        # Handle azimuth wraparound
        if self.axis_name == "AZM":
            error_rad = math.atan2(math.sin(target_position_rad - current_position_rad),
                                  math.cos(target_position_rad - current_position_rad))
        else:
            # For elevation, no wraparound needed
            error_rad = target_position_rad - current_position_rad

        return math.degrees(error_rad)


def compute_mount_position_error(config_state, current_azm_deg, current_alt_deg, target_az_deg, target_el_deg):
    """
    Compute position error between current mount position and target position.

    This function computes the error by converting the target sky position to mount coordinates
    and then computing the difference in mount coordinate space. This ensures correct
    sign conventions for control systems.

    Args:
        config_state: Configuration state with offsets and alignment
        current_azm_deg: Current AZM position (degrees, already includes offsets)
        current_alt_deg: Current ALT position (degrees, already includes offsets)
        target_az_deg: Target azimuth (degrees)
        target_el_deg: Target elevation (degrees)

    Returns:
        tuple: (az_error_deg, el_error_deg) - errors in degrees
    """
    # Check mount mode and use appropriate transformation
    mount_mode = getattr(config_state, 'mount_mode', 'Eq')

    if mount_mode == 'AltAz':
        # Import here to avoid circular import
        from transformations import AzEl2AzAlt_AltAz

        # Convert target sky position to mount coordinates (simplified AltAz mode)
        target_azm_deg, target_alt_deg = AzEl2AzAlt_AltAz(
            target_az_deg,
            target_el_deg,
            float(config_state.alignment_azimuth_str),
            float(config_state.alignment_elevation_str)
        )
    elif mount_mode == 'Passthrough':
        # Import here to avoid circular import
        from transformations import AzEl2AzAlt_Passthrough

        # Convert target sky position to mount coordinates (passthrough mode - sky to mount)
        target_azm_deg, target_alt_deg = AzEl2AzAlt_Passthrough(
            target_az_deg,
            target_el_deg
        )
    else:
        # Use full equatorial transformation for Eq mode
        # Import here to avoid circular import
        from transformations import local_elev_az_to_telescope #AzEl2AzAlt
        local_elev_az_to_telescope(target_el_deg, target_az_deg, lat_deg=float(config_state.alignment_elevation_str))

        # Convert target sky position to mount coordinates 
        # target_azm_deg, target_alt_deg = AzEl2AzAlt(
        #    target_az_deg,
        #    target_el_deg,
        #    float(config_state.alignment_azimuth_str),
        #    float(config_state.alignment_elevation_str)
        #)

    # Compute error in mount coordinates
    azm_error_deg = target_azm_deg - current_azm_deg
    alt_error_deg = target_alt_deg - current_alt_deg

    # Handle azimuth wraparound (mount coordinates)
    if azm_error_deg > 180:
        azm_error_deg -= 360
    elif azm_error_deg < -180:
        azm_error_deg += 360

    return azm_error_deg, alt_error_deg


def create_pid_controllers(config_state):
    """
    Create PID controllers for AZM and ALT axes using config values.

    Args:
        config_state: Configuration state with PID gains

    Returns:
        tuple: (azm_controller, alt_controller)
    """
    # Default gains if not set
    azm_p = getattr(config_state, 'pid_azm_p_gain', 1.0)
    azm_i = getattr(config_state, 'pid_azm_i_gain', 0.0)
    azm_d = getattr(config_state, 'pid_azm_d_gain', 0.0)

    alt_p = getattr(config_state, 'pid_alt_p_gain', 1.0)
    alt_i = getattr(config_state, 'pid_alt_i_gain', 0.0)
    alt_d = getattr(config_state, 'pid_alt_d_gain', 0.0)

    azm_controller = PIDController(azm_p, azm_i, azm_d, axis_name="AZM")
    alt_controller = PIDController(alt_p, alt_i, alt_d, axis_name="ALT")

    return azm_controller, alt_controller
