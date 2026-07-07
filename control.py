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


def apply_pointing_model(config_state, target_az_deg, target_el_deg):
    """Correct a desired sky (az, el) through the configured 7-term pointing model.

    Returns the commanded sky position to feed the (unmodified) mount transform. A no-op
    when the model is disabled. Shared by the Python control path and the Rust-loop
    adapter so both apply the same correction (the Rust transform stays geometric).
    """
    if not getattr(config_state, 'pointing_model_enabled', False):
        return target_az_deg, target_el_deg
    try:
        from pointing_model import PointingModel
        model = PointingModel(getattr(config_state, 'pointing_model_terms', None))
        return model.correct(target_az_deg, target_el_deg)
    except Exception as e:
        print(f"Pointing-model correction skipped: {e}")
        return target_az_deg, target_el_deg


def apply_eq_pointing_model(config_state, target_ha_deg, target_dec_deg):
    """Correct a desired (hour-angle, dec) through the configured equatorial pointing model.

    Returns the commanded (HA, Dec) to feed the mount transform. A no-op when disabled.
    The Eq counterpart of apply_pointing_model; applied in HA/Dec space in the Eq branch.
    """
    if not getattr(config_state, 'eq_pointing_model_enabled', False):
        return target_ha_deg, target_dec_deg
    try:
        from eq_pointing_model import EquatorialPointingModel
        lat = float(getattr(config_state, 'lat_str', 0.0) or 0.0)
        model = EquatorialPointingModel(getattr(config_state, 'eq_pointing_model_terms', None), lat_deg=lat)
        return model.correct(target_ha_deg, target_dec_deg)
    except Exception as e:
        print(f"Eq pointing-model correction skipped: {e}")
        return target_ha_deg, target_dec_deg


def sky_target_to_mount(config_state, target_az_deg, target_el_deg):
    """Convert a desired sky (az, el) to mount-axis coordinates for the
    configured mount mode, including the pointing-model pre-correction.

    This is the exact command transform compute_mount_position_error uses,
    exposed separately so callers that reason about the *mount axes* -- most
    importantly the safety-limit gates, whose azm/alt_limit_* config values
    are mount-frame quantities (they gate encoder positions in RATE and
    HOTSPOT modes) -- compare the target in the same frame. Gating the raw sky
    az/el against mount limits is frame-inconsistent: in AltAz mode the ALT
    axis runs opposite sky elevation (ALT = 90 - el), and in Eq mode the axes
    are HA/Dec.

    Returns:
        tuple: (target_azm_deg, target_alt_deg) mount-axis coordinates
    """
    mount_mode = getattr(config_state, 'mount_mode', 'AltAz')

    if mount_mode == 'AltAz':
        # Import here to avoid circular import
        from transformations import AzEl2AzAlt_AltAz

        # Pointing-model pre-step: correct the desired sky position so the (imperfect)
        # mount lands the boresight on target. Kept OUT of the transform itself, which is
        # mirrored in Rust and parity-tested; this stays Python-only.
        target_az_deg, target_el_deg = apply_pointing_model(config_state, target_az_deg, target_el_deg)

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
        # Equatorial mode: convert the target sky position to mount (hour-angle, dec) axes
        # for a polar axis pointing at (pole_az, pole_alt). pole_az defaults to due north
        # (alignment_azimuth) and pole_alt to the site latitude unless an explicit polar-axis
        # elevation has been entered (alignment_elevation, the "Eq" field in the config UI).
        from transformations import azel_to_eq_mount
        pole_az = float(getattr(config_state, 'alignment_azimuth_str', 0.0) or 0.0)
        pole_alt = float(getattr(config_state, 'alignment_elevation_str', 0.0) or 0.0)
        if pole_alt == 0.0:
            pole_alt = float(getattr(config_state, 'lat_str', 0.0) or 0.0)
        target_ha_deg, target_dec_deg = azel_to_eq_mount(
            target_az_deg, target_el_deg, pole_az, pole_alt)
        # Pointing-model pre-step (HA/Dec space): correct the desired axes so the imperfect
        # mount lands the boresight on target. Kept Python-only, like the AltAz path.
        target_azm_deg, target_alt_deg = apply_eq_pointing_model(
            config_state, target_ha_deg, target_dec_deg)

    return target_azm_deg, target_alt_deg


# Prefer the flipped mount solution only when it is decisively shorter, so a
# near-tie (target ~90 deg away in both configurations) can't flap between
# solutions from cycle to cycle mid-slew.
FLIP_HYSTERESIS_DEG = 0.5


def mount_target_for(config_state, target_az_deg, target_el_deg, flipped):
    """Mount-axis coordinates of a sky target in a CHOSEN configuration.

    ``flipped=False`` is the canonical solution (sky_target_to_mount).
    ``flipped=True`` pushes the mirrored sky representation
    ``(az+180, 180-el)`` -- the same physical pointing -- through the same
    transform, yielding the 'over the zenith' axis solution (for the AltAz
    convention that is (AZM+180, -ALT)).
    """
    if not flipped:
        return sky_target_to_mount(config_state, target_az_deg, target_el_deg)
    return sky_target_to_mount(config_state,
                               (target_az_deg + 180.0) % 360.0,
                               180.0 - target_el_deg)


def choose_mount_target(config_state, current_azm_deg, current_alt_deg,
                        target_az_deg, target_el_deg, limits=None):
    """Pick the mount-axis solution with the shortest slew to a sky target.

    An alt-az style mount reaches every sky pointing in TWO axis
    configurations: the canonical one, and the over-the-zenith one from the
    mirrored sky representation (az+180, 180-el). The per-axis shortest-arc
    wrap (the earlier '360 lap' fix) cannot discover the second solution:
    pointed west with the target in the east, the canonical solution demands
    a ~180-deg azimuth slew the long way around, while the flipped solution
    is a ~90-deg ALT move straight over the zenith. Choose per cycle by
    minimax wrapped axis error, with FLIP_HYSTERESIS_DEG of preference for
    the canonical solution so near-ties don't oscillate.

    Only AltAz and Passthrough modes have a usable second solution here; in
    Eq mode the mirrored representation maps to the same axis coordinates
    (a true meridian flip changes pier side and pointing-model terms, so it
    is deliberately not attempted by the tracking loop).

    ``limits``: optional (azm_min, azm_max, alt_min, alt_max) mount-frame
    safety limits; a solution outside them is never chosen. If no solution
    is inside, the canonical one is returned so the caller's safety gate
    aborts exactly as it would have before.

    Note: with the pointing model enabled, its correction in the flipped
    configuration is an extrapolation of a fit made in the canonical one --
    fine for choosing the path, but expect to re-center after a flip.

    Returns (target_azm_deg, target_alt_deg, flipped).
    """
    mount_mode = getattr(config_state, 'mount_mode', 'AltAz')
    canon = mount_target_for(config_state, target_az_deg, target_el_deg, False)
    candidates = [(canon, False)]
    if mount_mode in ('AltAz', 'Passthrough'):
        candidates.append(
            (mount_target_for(config_state, target_az_deg, target_el_deg, True), True))

    def _wrap(d):
        return (d + 180.0) % 360.0 - 180.0

    def metric(t):
        # Axes slew simultaneously, so slew time ~ the larger axis error.
        return max(abs(_wrap(t[0] - current_azm_deg)),
                   abs(_wrap(t[1] - current_alt_deg)))

    def in_limits(t):
        if limits is None:
            return True
        azm_min, azm_max, alt_min, alt_max = limits
        return (azm_min <= t[0] <= azm_max) and (alt_min <= t[1] <= alt_max)

    legal = [(t, f) for t, f in candidates if in_limits(t)]
    if not legal:
        return canon[0], canon[1], False
    (best_t, best_f) = legal[0]
    best_m = metric(best_t)
    for t, f in legal[1:]:
        m = metric(t)
        if m < best_m - FLIP_HYSTERESIS_DEG:
            best_t, best_f, best_m = t, f, m
    return best_t[0], best_t[1], best_f


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
    # Shortest-slew solution: canonical, or over-the-zenith when that is
    # decisively shorter (see choose_mount_target).
    target_azm_deg, target_alt_deg, _flipped = choose_mount_target(
        config_state, current_azm_deg, current_alt_deg,
        target_az_deg, target_el_deg)

    # Compute error in mount coordinates
    azm_error_deg = target_azm_deg - current_azm_deg
    alt_error_deg = target_alt_deg - current_alt_deg

    # Handle azimuth wraparound (mount coordinates). Use modular reduction rather
    # than a single-step +/-360 so the result is the true shortest-arc error for
    # ANY input range. current_azm carries azm_offset (mount_control.py) and can
    # fall outside [0, 360), while target_azm is %360 from the transform; when the
    # two land in different 360-deg windows a one-shot if/elif fails to wrap past
    # ~1.5 turns and returns e.g. +340 instead of -20. That value is then clamped
    # to +180 (compute_pid_output), driving the mount the long way ("360 lap" on
    # acquire). Matches the Rust wrap180() and the AZM derivative wrap above.
    azm_error_deg = (azm_error_deg + 180.0) % 360.0 - 180.0

    # Handle ALT wraparound too. The ALT encoder is just as modular as AZM
    # (hc_get_position returns a fraction of a revolution -> current_alt in
    # [0, 360); mount_control.py), and in AltAz the convention is el = 90 - ALT,
    # so the boresight points at zenith when ALT ~= 0 -- right on the 0/360 seam.
    # On a near-overhead pass current_alt can read ~359 while target_alt is a few
    # degrees, and the raw difference becomes ~-357. Without this reduction the
    # PID drives the ALT axis the long way -- a near-full 360 traversal the mount
    # cannot physically make -- instead of the few-degree hop across the seam.
    alt_error_deg = (alt_error_deg + 180.0) % 360.0 - 180.0

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
