import numpy as np
import math
from datetime import datetime, timedelta, timezone
from skyfield.api import wgs84, load

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Simulation and Trajectory Parameters
TRAJECTORY_DURATION_MINUTES_DEFAULT = 15  # Default duration for trajectory computation
TIME_SAMPLES_COUNT = 301  # Number of time samples for trajectory (every ~6 seconds)
MIN_VISIBLE_ALTITUDE_DEGREES = 15.0  # Minimum altitude considered visible (degrees)
DEFAULT_ELEVATION_MASK_DEGREES = 10.0  # Default elevation mask
FRUSTUM_UNCERTAINTY_MARGIN_DEGREES = 15.0  # Extra margin for visibility checking
VISIBILITY_SEED_SAMPLES_MIN = 5  # Minimum number of time samples for visibility checking
VISIBILITY_SEED_INTERVAL_MINUTES = 2  # Minutes between visibility check samples

# Future Pass Window
FUTURE_PASS_WINDOW_HOURS = 2.0  # Hours of future passes to display in table
FUTURE_PASS_WINDOW_SECONDS = FUTURE_PASS_WINDOW_HOURS * 3600  # Convert to seconds

# Table and Display Limits
PASS_TABLE_MAX_ROWS_DEFAULT = 20  # Maximum rows to display in pass table
SATELLITE_NAME_MAX_LENGTH_TABLE = 20  # Max satellite name length in table display

# Progress Reporting
VISIBILITY_FILTER_BATCH_SIZE = 500  # Batch size for visibility filter progress reporting
TRAJECTORY_COMPUTE_BATCH_SIZE = 100  # Batch size for trajectory computation progress reporting

# Trajectory Rendering
TRAJECTORY_FUTURE_COLOR = (255, 0, 0)  # Red color for future trajectory segments
TRAJECTORY_PAST_COLOR = (128, 128, 128)  # Gray color for past trajectory segments
TRAJECTORY_START_TIME_BUFFER = 0.0  # Additional buffer for trajectory start time

# Cache Configuration
CACHE_PRECISION_DIGITS = 6  # Decimal precision for cache keys
TIME_RANGE_BUFFER_MINUTES = 0.0  # Buffer added to time ranges

# Horizon and Elevation Thresholds
ELEVATION_THRESHOLD_DEGREES = 0.0  # Minimum elevation above horizon
MINIMUM_VISIBLE_ALTITUDE_DEGREES = 0.0  # Minimum altitude to consider for visibility

# ==============================================================================
# FUNCTIONS
# ==============================================================================

# Global cache for trajectories to avoid recomputation
TRAJECTORY_CACHE = {}
SUN_EPHEMERIS_CACHE = None

def clear_trajectory_cache():
    """Clear the trajectory cache to free memory"""
    global TRAJECTORY_CACHE, SUN_EPHEMERIS_CACHE
    TRAJECTORY_CACHE.clear()
    SUN_EPHEMERIS_CACHE = None

def _is_satellite_potentially_visible(satellite, observer, time_samples, elevation_mask_deg=10.0):
    """
    Check if a satellite might be visible during the time period by sampling at coarse intervals.
    This is much faster than computing full trajectory and reduces computation for satellites
    that will remain below the horizon.
    """
    try:
        difference = satellite - observer
        topocentrics = difference.at(time_samples)

        # Get altitudes (vectorized)
        alts = topocentrics.altaz()[0]
        alt_deg = np.array(alts.degrees)

        # If any altitude is above the elevation mask + margin, consider it potentially visible
        min_visible_alt = max(elevation_mask_deg, FRUSTUM_UNCERTAINTY_MARGIN_DEGREES)
        return np.any(alt_deg > min_visible_alt)

    except Exception:
        # If there's an error calculating position, assume it's not visible to be safe
        return False

def _filter_satellites_by_visibility(satellites, observer, ts, center_time, duration_minutes, elevation_mask_deg=10.0, update_status_callback=None):
    """
    Filter satellites to only compute trajectories for those that could potentially be visible.
    Uses coarse time sampling (every 2 minutes) to quickly check visibility.
    """
    # Calculate time range
    if center_time is not None:
        t0 = ts.utc(center_time - timedelta(minutes=duration_minutes/2))
        t1 = ts.utc(center_time + timedelta(minutes=duration_minutes/2))
    else:
        current_utc = datetime.now(timezone.utc)
        t0 = ts.utc(current_utc - timedelta(minutes=duration_minutes/2))
        t1 = ts.utc(current_utc + timedelta(minutes=duration_minutes/2))

    # Create coarse time samples for visibility checking
    num_samples = max(VISIBILITY_SEED_SAMPLES_MIN,
                     int(duration_minutes / VISIBILITY_SEED_INTERVAL_MINUTES))
    visibility_times = ts.linspace(t0, t1, num_samples)

    visible_satellites = []
    total_checked = 0

    for sat in satellites:
        total_checked += 1
        if _is_satellite_potentially_visible(sat, observer, visibility_times, elevation_mask_deg):
            visible_satellites.append(sat)

        # Progress update for filtering
        if update_status_callback and total_checked % VISIBILITY_FILTER_BATCH_SIZE == 0 and total_checked > 0:
            update_status_callback(f"Visibility filtered {len(visible_satellites)}/{total_checked} satellites...")

    if visible_satellites:
        visibility_ratio = len(visible_satellites) / len(satellites) * 100
    else:
        visibility_ratio = 0

    if update_status_callback:
        update_status_callback(f"Visibility filtering: {len(visible_satellites)}/{len(satellites)} satellites potentially visible ({visibility_ratio:.1f}%)")

    return visible_satellites

def precompute_trajectories(state, observer, ts, display, update_status_callback=None, center_time=None, duration_minutes=15):
    """Optimized trajectory computation with caching, reduced resolution, vectorization, and visibility filtering.
    Modifies state object directly with trajectories and arc segments."""

    # OPTIMIZATION 8: Filter satellites by potential visibility first (potentially 2-10x speedup)
    visible_satellites = _filter_satellites_by_visibility(
        state.satellites, observer, ts, center_time, duration_minutes, elevation_mask_deg=10.0, update_status_callback=update_status_callback
    )

    # Setup constants
    cx = display.sub_x + display.sub_width // 2
    cy = display.sub_y + display.sub_height // 2
    radius = min(display.sub_width, display.sub_height) // 2 - 50

    # Get global sun ephemeris if not already loaded
    global SUN_EPHEMERIS_CACHE
    if SUN_EPHEMERIS_CACHE is None:
        SUN_EPHEMERIS_CACHE = load('de421.bsp')['sun']

    # Calculate time range
    if center_time is not None:
        t0 = ts.utc(center_time - timedelta(minutes=duration_minutes/2))
        t1 = ts.utc(center_time + timedelta(minutes=duration_minutes/2))
    else:
        current_utc = datetime.now(timezone.utc)
        t0 = ts.utc(current_utc - timedelta(minutes=duration_minutes/2))
        t1 = ts.utc(current_utc + timedelta(minutes=duration_minutes/2))

    # OPTIMIZATION 1: Reduce time resolution from 1-second to 6-second intervals (5x speedup)
    times = ts.linspace(t0, t1, 301)  # Every 6 seconds instead of 1 second

    # Filter satellites that are visible AND in our label dictionary and not cached
    satellites_to_compute = []
    satellites_in_label_dict = 0

    for sat in visible_satellites:
        if sat in state.satellite_labels:
            satellites_in_label_dict += 1
            # Create cache key based on satellite ID and time range
            sat_model = getattr(sat, 'model', None)
            sat_id = getattr(sat_model, 'satnum_str', str(sat)) if sat_model else str(sat)
            cache_key = f"{sat_id}_{t0.tt:.6f}_{t1.tt:.6f}"

            if cache_key not in TRAJECTORY_CACHE:
                satellites_to_compute.append((sat, cache_key))

    total_sats = len(satellites_to_compute)
    state.satellite_trajectories = {}
    state.satellite_arc_segments = {}

    if update_status_callback:
        update_status_callback(f"Computing optimized trajectories for {total_sats}/{satellites_in_label_dict} satellites...")

    processed_count = 0
    for sat, cache_key in satellites_to_compute:
        processed_count += 1

        # OPTIMIZATION 2: Batch processing of satellite positions
        difference = sat - observer
        topocentrics = difference.at(times)

        # Get position arrays (vectorized operation)
        alts, azs, distances = topocentrics.altaz()

        # OPTIMIZATION 3: Convert to numpy arrays for vectorized operations
        alt_deg = np.array(alts.degrees)
        az_deg = np.array(azs.degrees)
        dist_km = np.array(distances.km)
        time_vals = np.array(times.tt)

        # OPTIMIZATION 4: Vectorized pixel coordinate calculation (eliminate Python loops)
        az_rad = np.radians(az_deg % 360)
        polar_radius = (90 - alt_deg) / 90 * radius

        # Compute all pixel coordinates at once
        px_coords = cx + polar_radius * np.sin(az_rad)
        py_coords = cy - polar_radius * np.cos(az_rad)

        # OPTIMIZATION 5: Pre-compute time-to-index mapping for faster interpolation
        times_array = time_vals.copy()

        # Add rate calculations (degrees/second)
        dt = times_array[1] - times_array[0]  # Time step in days
        dt_seconds = dt * 86400.0  # Convert to seconds

        # Calculate rates (finite differences)
        az_rates_dps = np.zeros(len(times_array))
        el_rates_dps = np.zeros(len(times_array))

        for i in range(1, len(times_array) - 1):
            az_rates_dps[i] = (az_deg[i + 1] - az_deg[i - 1]) / (2 * dt_seconds)
            el_rates_dps[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / (2 * dt_seconds)

        # Handle endpoints with forward/backward differences
        az_rates_dps[0] = (az_deg[1] - az_deg[0]) / dt_seconds
        az_rates_dps[-1] = (az_deg[-1] - az_deg[-2]) / dt_seconds
        el_rates_dps[0] = (alt_deg[1] - alt_deg[0]) / dt_seconds
        el_rates_dps[-1] = (alt_deg[-1] - alt_deg[-2]) / dt_seconds

        # OPTIMIZATION 6: Create trajectory data structure with rates
        trajectory_data = np.column_stack([time_vals, alt_deg, az_deg, dist_km, px_coords, py_coords, az_rates_dps, el_rates_dps])
        trajectory = trajectory_data.tolist()

        # OPTIMIZATION 7: Create arc segments (color will be determined during rendering)
        segments = _create_arc_segments_simple(trajectory_data, times.tt[0])

        # Cache the computed trajectory
        TRAJECTORY_CACHE[cache_key] = (trajectory, times_array, segments)

        state.satellite_trajectories[sat] = (trajectory, times_array)
        state.satellite_arc_segments[sat] = segments

        # Progress update
        if update_status_callback and processed_count % TRAJECTORY_COMPUTE_BATCH_SIZE == 0 and processed_count > 0:
            update_status_callback(f"Computed traj {processed_count}/{total_sats} sats...")

    # Load cached trajectories for satellites that don't need recomputation
    cache_hits = 0
    for sat in state.satellites:
        if sat in state.satellite_labels:
            sat_model = getattr(sat, 'model', None)
            sat_id = getattr(sat_model, 'satnum_str', str(sat)) if sat_model else str(sat)
            cache_key = f"{sat_id}_{t0.tt:.6f}_{t1.tt:.6f}"

            if cache_key in TRAJECTORY_CACHE:
                cached_data = TRAJECTORY_CACHE[cache_key]
                state.satellite_trajectories[sat] = (cached_data[0], cached_data[1])
                state.satellite_arc_segments[sat] = cached_data[2]
                cache_hits += 1

    if cache_hits > 0:
        print(f"DEBUG: Trajectory optimization: {cache_hits} satellites loaded from cache")

def _create_arc_segments_simple(trajectory_data, start_time):
    """Create arc segments with simple time-based color coding"""
    if len(trajectory_data) < 2:
        return []

    # Unpack trajectory data
    times = trajectory_data[:, 0]
    alts = trajectory_data[:, 1]
    azs = trajectory_data[:, 2]
    px = trajectory_data[:-1, 4]  # Start positions
    py = trajectory_data[:-1, 5]
    px_next = trajectory_data[1:, 4]  # End positions
    py_next = trajectory_data[1:, 5]

    # Create masks for segments that are above horizon
    above_horizon = (alts[:-1] > 0) | (alts[1:] > 0)

    if not above_horizon.any():
        return []

    # Determine colors based on time and sunlit status
    segments = []
    for i in range(len(trajectory_data) - 1):
        if above_horizon[i]:
            is_future = times[i] > start_time
            # For sunlit trajectory display, all satellites get time-based coloring
            # Individual satellite sunlit status will be recomputed for selected satellites
            color = (255, 0, 0) if is_future else (128, 128, 128)
            segments.append((px[i], py[i], px_next[i], py_next[i], color))

    return segments

def interpolate_position_and_rates(trajectory_data, current_tt):
    """
    Interpolate satellite position and motion rates at a given time.

    Args:
        trajectory_data: Tuple of (trajectory, times_array)
        current_tt: Current time in terrestrial time

    Returns:
        tuple: (px, py, alt, dist, az_rate_dps, el_rate_dps)
    """
    if not trajectory_data or not trajectory_data[0]:  # Check if trajectory is empty
        return None, None, None, None, 0.0, 0.0
    trajectory, times_array = trajectory_data
    # Find the insertion point for current_tt
    idx = np.searchsorted(times_array, current_tt) - 1
    if idx < 0:
        # Before the first point, use the first point
        return trajectory[0][4], trajectory[0][5], trajectory[0][1], trajectory[0][3], trajectory[0][6], trajectory[0][7]
    elif idx >= len(times_array) - 1:
        # After the last point, use the last point
        return trajectory[-1][4], trajectory[-1][5], trajectory[-1][1], trajectory[-1][3], trajectory[-1][6], trajectory[-1][7]
    else:
        # Linear interpolation between idx and idx+1
        t0 = times_array[idx]
        t1 = times_array[idx + 1]
        fraction = (current_tt - t0) / (t1 - t0)
        px0, py0, alt0, dist0 = trajectory[idx][4], trajectory[idx][5], trajectory[idx][1], trajectory[idx][3]
        px1, py1, alt1, dist1 = trajectory[idx + 1][4], trajectory[idx + 1][5], trajectory[idx + 1][1], trajectory[idx + 1][3]
        az_rate0, el_rate0 = trajectory[idx][6], trajectory[idx][7]
        az_rate1, el_rate1 = trajectory[idx + 1][6], trajectory[idx + 1][7]

        px = px0 + fraction * (px1 - px0)
        py = py0 + fraction * (py1 - py0)
        alt = alt0 + fraction * (alt1 - alt0)
        dist = dist0 + fraction * (dist1 - dist0)
        az_rate = az_rate0 + fraction * (az_rate1 - az_rate0)
        el_rate = el_rate0 + fraction * (el_rate1 - el_rate0)

        return px, py, alt, dist, az_rate, el_rate

def interpolate_position(trajectory_data, current_tt):
    """
    Legacy function that returns only position data.
    """
    px, py, alt, dist, *_ = interpolate_position_data_and_rates(trajectory_data, current_tt)
    return px, py, alt, dist

def interpolate_position_data_and_rates(trajectory_data, current_tt, launch_tt=0, launched=False):
    """
    Parse trajectory data (either old 7-column or new 8-column format) and return position + rates.
    Now returns azimuth (az_deg) as the 7th element for precise tracking.
    """
    if not trajectory_data or not trajectory_data[0]:
        return None, None, None, None, None, 0.0, 0.0

    trajectory, times_array = trajectory_data
    
    # Relative indexing adjustment for repeated launches
    if launched:
        t0_0 = times_array[0] # base t0 (when file was loaded or selected)
        dt0 = launch_tt - t0_0
        current_tt = current_tt - dt0

    # Handle both old format (time, alt, az, dist, px, py) and new format (with rates)
    if len(trajectory[0]) == 6:  # Old format
        idx = np.searchsorted(times_array, current_tt) - 1
        if idx < 0:
            return trajectory[0][4], trajectory[0][5], trajectory[0][1], trajectory[0][3], trajectory[0][2], 0.0, 0.0
        elif idx >= len(times_array) - 1:
            return trajectory[-1][4], trajectory[-1][5], trajectory[-1][1], trajectory[-1][3], trajectory[-1][2], 0.0, 0.0

        t0 = times_array[idx]
        t1 = times_array[idx + 1]
        fraction = (current_tt - t0) / (t1 - t0)
        px0, py0, alt0, dist0, az0 = trajectory[idx][4], trajectory[idx][5], trajectory[idx][1], trajectory[idx][3], trajectory[idx][2]
        px1, py1, alt1, dist1, az1 = trajectory[idx + 1][4], trajectory[idx + 1][5], trajectory[idx + 1][1], trajectory[idx + 1][3], trajectory[idx + 1][2]
        px = px0 + fraction * (px1 - px0)
        py = py0 + fraction * (py1 - py0)
        alt = alt0 + fraction * (alt1 - alt0)
        dist = dist0 + fraction * (dist1 - dist0)
        az = az0 + fraction * (az1 - az0)
        return px, py, alt, dist, az, 0.0, 0.0

    elif len(trajectory[0]) >= 8:  # New format with rates
        idx = np.searchsorted(times_array, current_tt) - 1
        if idx < 0:
            return trajectory[0][4], trajectory[0][5], trajectory[0][1], trajectory[0][3], trajectory[0][2], trajectory[0][6], trajectory[0][7]
        elif idx >= len(times_array) - 1:
            return trajectory[-1][4], trajectory[-1][5], trajectory[-1][1], trajectory[-1][3], trajectory[-1][2], trajectory[-1][6], trajectory[-1][7]

        t0 = times_array[idx]
        t1 = times_array[idx + 1]
        fraction = (current_tt - t0) / (t1 - t0)
        px0, py0, alt0, dist0, az0 = trajectory[idx][4], trajectory[idx][5], trajectory[idx][1], trajectory[idx][3], trajectory[idx][2]
        px1, py1, alt1, dist1, az1 = trajectory[idx + 1][4], trajectory[idx + 1][5], trajectory[idx + 1][1], trajectory[idx + 1][3], trajectory[idx + 1][2]
        az_rate0, el_rate0 = trajectory[idx][6], trajectory[idx][7]
        az_rate1, el_rate1 = trajectory[idx + 1][6], trajectory[idx + 1][7]

        px = px0 + fraction * (px1 - px0)
        py = py0 + fraction * (py1 - py0)
        alt = alt0 + fraction * (alt1 - alt0)
        dist = dist0 + fraction * (dist1 - dist0)
        az = az0 + fraction * (az1 - az0)
        az_rate = az_rate0 + fraction * (az_rate1 - az_rate0)
        el_rate = el_rate0 + fraction * (el_rate1 - el_rate0)

        return px, py, alt, dist, az, az_rate, el_rate

    else:
        # Unknown format, fall back to old behavior
        return None, None, None, None, None, 0.0, 0.0

def update_satellite_positions(state, current_tt, elevation_mask_deg=10.0):
    """
    Update satellite positions for the current timestamp with filtering.
    Modifies state object directly with updated satellite_positions dictionary.
    """
    # Build into a local dict and publish it in a single atomic rebind at the
    # end. The rendering thread reads state.satellite_positions live every
    # frame; if we cleared and incrementally repopulated the live attribute
    # here, a render landing mid-loop would copy a partially-filled dict and
    # draw only some satellites -> the visualization "blink"/flicker. Assigning
    # the finished dict once means readers always see a complete frame's worth.
    new_positions = {}

    # Always handle all satellites with filtering (don't exclude other satellites when one is selected)
    for sat in state.satellites:
        if sat in state.satellite_trajectories:
            px, py, alt, dist = interpolate_position(state.satellite_trajectories[sat], current_tt)
            if px is not None and alt > elevation_mask_deg:
                # Apply filters
                include_sat = True
                if state.filter_text:
                    include_sat = state.filter_text.lower() in sat.name.lower() or state.filter_text in sat.model.satnum_str
                if state.filter_above_alt_text:
                    try:
                        alt_filter = float(state.filter_above_alt_text)
                        include_sat = include_sat and state.satellite_mean_altitudes[sat] >= alt_filter
                    except ValueError:
                        include_sat = False
                if state.filter_below_alt_text:
                    try:
                        alt_filter = float(state.filter_below_alt_text)
                        include_sat = include_sat and state.satellite_mean_altitudes[sat] <= alt_filter
                    except ValueError:
                        include_sat = False

                if include_sat:
                    new_positions[sat] = (px, py, alt, dist)

    # Atomic publish: readers see either the complete old dict or this new one.
    state.satellite_positions = new_positions

def extract_pass_data_from_trajectory(trajectory_data, satellite, satellite_labels, elevation_mask_deg=10.0, ts=None):
    """
    Extract pass information from a single satellite's trajectory.
    Returns pass data or None if no significant pass exists.
    """
    if not trajectory_data or not trajectory_data[0]:
        return None

    trajectory, times_array = trajectory_data
    trajectory_array = np.array(trajectory)

    # Extract relevant columns: time, altitude, azimuth
    altitudes = trajectory_array[:, 1]  # altitude in degrees
    azimuths = trajectory_array[:, 2]   # azimuth in degrees

    # Filter for altitudes above elevation mask
    visible_points = altitudes > elevation_mask_deg

    if not np.any(visible_points):
        return None

    # Find max altitude and its index
    max_alt_idx = np.argmax(altitudes)
    max_elevation = altitudes[max_alt_idx]
    azimuth_at_max = azimuths[max_alt_idx]

    # Extract satellite name and NORAD ID
    name = satellite.name.strip() if satellite and hasattr(satellite, 'name') else "Unknown"
    norad_id = satellite.model.satnum_str if satellite and hasattr(satellite, 'model') else "Unknown"

    # Find closest approach (minimum distance) time
    closest_time = None
    if np.any(visible_points):
        visible_distances = trajectory_array[visible_points, 3]  # distances for visible points
        visible_times = trajectory_array[visible_points, 0]       # times for visible points

        if len(visible_distances) > 0:
            min_dist_idx = np.argmin(visible_distances)
            closest_time = visible_times[min_dist_idx]

    # Format closest approach time as local time string
    closest_approach_time = "--:--"
    if closest_time is not None:
        try:
            # Convert TT (Terrestrial Time) to UTC datetime
            closest_datetime = ts.tt(jd=closest_time).utc_datetime()
            # Convert to local time (PDT/PDT)
            closest_datetime = closest_datetime.replace(tzinfo=timezone.utc)
            closest_datetime = closest_datetime.astimezone()
            closest_approach_time = closest_datetime.strftime("%H:%M")
        except Exception:
            closest_approach_time = "--:--"

    return {
        'satellite': satellite,
        'name': name,
        'norad_id': norad_id,
        'max_elevation': max_elevation,
        'azimuth_at_max': azimuth_at_max,
        'closest_approach_time': closest_approach_time,
        'is_visible': np.any(visible_points)
    }

def build_satellite_pass_table(state, elevation_mask_deg=10.0, max_rows=20, ts=None):
    """
    Build pass table data from all satellite trajectories.
    Only includes satellites that are currently in view OR will be in view soon (future closest approach).
    Modifies state object directly with sorted satellite pass table.
    """
    pass_entries = []

    # Get current time in local timezone for comparison
    current_datetime = datetime.now(timezone.utc)
    current_local = current_datetime.astimezone()

    for sat in state.satellites:
        if sat in state.satellite_trajectories and sat in state.satellite_labels:
            pass_data = extract_pass_data_from_trajectory(state.satellite_trajectories[sat], sat, state.satellite_labels, elevation_mask_deg, ts)
            if pass_data and pass_data['is_visible']:
                # Filter for satellites that are either:
                # 1. Currently in view (checked against satellite_positions)
                # 2. Will be in view soon (future closest approach time)
                include_in_table = False

                # Check if satellite is currently in view
                if state.satellite_positions and sat in state.satellite_positions:
                    include_in_table = True
                else:
                    # Check if the closest approach time is in the future
                    closest_time_str = pass_data.get('closest_approach_time', '--:--')
                    if closest_time_str != '--:--':
                        try:
                            # Convert time string back to local time for comparison
                            h, m = map(int, closest_time_str.split(':'))                                                            
                            closest_datetime = current_local.replace(hour=h, minute=m, second=0, microsecond=0)

                            # Handle case where closest time is before current time (next day)
                            if closest_datetime < current_local:
                                closest_datetime = closest_datetime.replace(
                                    day=current_local.day + 1,
                                    month=current_local.month if current_local.day < 31 else current_local.month + 1,
                                    year=current_local.year
                                )

                            # If closest approach is within 2 hours from now, include it
                            time_diff = closest_datetime - current_local
                            if time_diff.total_seconds() >= 0 and time_diff.total_seconds() <= 7200:  # 7200 seconds = 2 hours
                                include_in_table = True
                        except (ValueError, IndexError):
                            # If time parsing fails, check if satellite has any visibility points
                            # This is a fallback for cases where the time format is unexpected
                            pass

                if include_in_table:
                    pass_entries.append(pass_data)

    # Store the full candidate set; name/altitude filtering and multi-column
    # sorting are applied live every frame in filter_and_sort_pass_table so the
    # table responds immediately to filter edits and column-header clicks.
    state.satellite_pass_table_full = pass_entries

    # Compute the initial filtered + sorted view (the render loop refreshes this
    # each frame, but populate it now so callers that draw immediately are correct).
    from tracking_visuals import filter_and_sort_pass_table
    state.satellite_pass_table = filter_and_sort_pass_table(state)

def sort_pass_table(pass_table, sort_keys=None, reverse_flags=None):
    """
    Sort pass table by specified keys.
    sort_keys: array of booleans indicating active columns (True = active sort column)
    reverse_flags: array of booleans indicating sort direction for each column
    Legacy support: single sort_key string if arrays not provided
    """

    # Handle legacy single key sorting
    if sort_keys is None or not hasattr(sort_keys, '__iter__'):
        # Legacy mode - sort_key should be passed as second positional argument
        sort_key = sort_keys if sort_keys else 'max_elevation'
        reverse = reverse_flags if isinstance(reverse_flags, bool) else False

        if sort_key == 'name':
            return sorted(pass_table, key=lambda x: x['name'], reverse=reverse)
        elif sort_key == 'norad_id':
            return sorted(pass_table, key=lambda x: x['norad_id'], reverse=reverse)
        elif sort_key == 'azimuth_at_max':
            return sorted(pass_table, key=lambda x: x['azimuth_at_max'], reverse=reverse)
        elif sort_key == 'max_elevation':
            return sorted(pass_table, key=lambda x: x['max_elevation'], reverse=reverse)
        elif sort_key == 'closest_approach_time':
            # Custom sorter for time strings (handle "--:--" cases)
            def time_sort_key(x):
                time_str = x['closest_approach_time']
                if time_str == '--:--':
                    return '23:59'  # Put unknown times at the end
                return time_str
            return sorted(pass_table, key=time_sort_key, reverse=reverse)
        else:
            return pass_table

    # Handle array-based sorting (column header clicks)
    # Find the active sort column
    active_column = None
    for i, is_active in enumerate(sort_keys):
        if is_active:
            active_column = i
            break

    if active_column is None:
        # No active column, sort by max elevation by default
        active_column = 3  # max_elevation column

    # Map column index to sort key
    column_mapping = {
        0: 'name',
        1: 'norad_id',
        2: 'azimuth_at_max',
        3: 'max_elevation',
        4: 'closest_approach_time'
    }

    sort_key = column_mapping.get(active_column, 'max_elevation')
    reverse = reverse_flags[active_column] if reverse_flags and len(reverse_flags) > active_column else False

    if sort_key == 'name':
        return sorted(pass_table, key=lambda x: x['name'], reverse=reverse)
    elif sort_key == 'norad_id':
        return sorted(pass_table, key=lambda x: x['norad_id'], reverse=reverse)
    elif sort_key == 'azimuth_at_max':
        return sorted(pass_table, key=lambda x: x['azimuth_at_max'], reverse=reverse)
    elif sort_key == 'max_elevation':
        return sorted(pass_table, key=lambda x: x['max_elevation'], reverse=reverse)
    elif sort_key == 'closest_approach_time':
        # Custom sorter for time strings (handle "--:--" cases)
        def time_sort_key(x):
            time_str = x['closest_approach_time']
            if time_str == '--:--':
                return '23:59'  # Put unknown times at the end
            return time_str
        return sorted(pass_table, key=time_sort_key, reverse=reverse)
    else:
        return pass_table

def read_tracking_trajectory(filepath, display=None, update_status_callback=None):
    """
    Read a tracking trajectory file in twilight format with ECEF coordinates.
    Returns a dictionary with trajectory data and arc segments, or None on failure.
    """
    try:
        # Load observer coordinates from config
        import json
        import os
        from datetime import datetime
        from skyfield.api import load, Topos
        from skyfield.framelib import itrs
        from skyfield.toposlib import ITRSPosition
        from skyfield.units import Distance

        config_data = {}
        # Try to find config.json in the project root (same directory as trajectory.py)
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to hardcoded values if config can't be loaded
            config_data = {"lat": "34.584532", "lon": "-120.632245", "alt": "120.0"}

        obs_lat = float(config_data.get("lat", "34.584532"))
        obs_lon = float(config_data.get("lon", "-120.632245"))
        obs_alt = float(config_data.get("alt", "120.0"))

        # Read the file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            if update_status_callback:
                update_status_callback(f"File {filepath} too short")
            return None

        # Parse Unix timestamp from header
        unix_timestamp = None
        data_start_idx = None

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#Unix T+0 Time (s)'):
                # Next line should contain the timestamp
                if i + 1 < len(lines):
                    try:
                        unix_timestamp = float(lines[i + 1].strip())
                    except ValueError:
                        if update_status_callback:
                            update_status_callback(f"Invalid Unix timestamp in {filepath}")
                        return None
            elif line.startswith('# Delta T+ Time (s), ECEF Target Position (m), ECEF Target Velocity (m/s)'):
                # Data starts after this header
                data_start_idx = i + 1
                break

        if unix_timestamp is None or data_start_idx is None:
            if update_status_callback:
                update_status_callback(f"Missing headers in {filepath}")
            return None

        # Parse data rows
        times_rel = []  # Relative time in seconds
        ecef_positions = []  # [x, y, z] in meters
        ecef_velocities = []  # [vx, vy, vz] in m/s

        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 7:
                continue  # Skip malformed lines

            try:
                time_rel = float(parts[0])
                ecef_x = float(parts[1])
                ecef_y = float(parts[2])
                ecef_z = float(parts[3])
                vel_x = float(parts[4])
                vel_y = float(parts[5])
                vel_z = float(parts[6])

                times_rel.append(time_rel)
                ecef_positions.append([ecef_x, ecef_y, ecef_z])
                ecef_velocities.append([vel_x, vel_y, vel_z])
            except ValueError:
                continue  # Skip lines with invalid numbers

        if not times_rel:
            if update_status_callback:
                update_status_callback(f"No valid data rows in {filepath}")
            return None

        # Convert to numpy arrays
        times_rel = np.array(times_rel)
        ecef_positions = np.array(ecef_positions)
        ecef_velocities = np.array(ecef_velocities)

        # Create Skyfield observer and timescale
        ts = load.timescale()
        # Sun lighting computation using Skyfield
        global SUN_EPHEMERIS_CACHE
        if SUN_EPHEMERIS_CACHE is None:
            SUN_EPHEMERIS_CACHE = load('de421.bsp')['sun']
        eph = load('de421.bsp')

        # observer = wgs84.latlon(obs_lat, obs_lon, elevation_m=obs_alt)
        observer = Topos(latitude_degrees=obs_lat, longitude_degrees=obs_lon, elevation_m=obs_alt)

        # Convert Unix timestamp to Skyfield time
        launch_time = ts.utc(datetime.fromtimestamp(unix_timestamp, tz=timezone.utc))

        # Create absolute times for each data point
        times_tt = []
        times = []
        for time_rel in times_rel:
            abs_time = launch_time + timedelta(seconds=time_rel)
            times_tt.append(abs_time.tt)
            times.append(abs_time)
        times_array = np.array(times)
        tt_array = np.array(times_tt)
        

        # Convert ECEF positions to topocentric coordinates
        altitudes = []
        azimuths = []
        ranges_km = []
        sunlit_status = []

        for pos, tt in zip(ecef_positions, times_array):
            # Create ECEF position vector (input is in meters)
            d = Distance(m=pos)
            ecef_pos = ITRSPosition(d)
            topocentric = (ecef_pos - observer).at(tt)
            alt_deg, az_deg, range_m = topocentric.altaz()
            is_sunlit = ecef_pos.at(tt).is_sunlit(eph)
            sunlit_status.append(is_sunlit)
            ranges_km.append(range_m.km)

            # Elevation
            altitudes.append(alt_deg.degrees)

            # Azimuth (measured from north, clockwise)
            azimuths.append(az_deg.degrees)

        # Convert to numpy arrays
        alt_deg = np.array(altitudes)
        az_deg = np.array(azimuths)
        dist_km = np.array(ranges_km)
        sun_stat = np.array(sunlit_status)

        # Calculate pixel coordinates
        if display:
            cx = display.sub_x + display.sub_width // 2
            cy = display.sub_y + display.sub_height // 2
            radius = min(display.sub_width, display.sub_height) // 2 - 50
        else:
            cx = 400
            cy = 300
            radius = 300

        az_rad = np.radians(az_deg % 360)
        polar_radius = (90 - alt_deg) / 90 * radius
        px_coords = cx + polar_radius * np.sin(az_rad)
        py_coords = cy - polar_radius * np.cos(az_rad)

        # Calculate angular rates from position data
        az_rates_dps = np.zeros(len(times_array))
        el_rates_dps = np.zeros(len(times_array))

        # Time step (assume uniform spacing)
        dt_seconds = 1.0  # Default 1 second intervals
        if len(times_rel) > 1:
            dt_seconds = times_rel[1] - times_rel[0]

        # Calculate rates using finite differences
        for i in range(1, len(times_array) - 1):
            az_rates_dps[i] = (az_deg[i + 1] - az_deg[i - 1]) / (2 * dt_seconds)
            el_rates_dps[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / (2 * dt_seconds)

        # Handle endpoints
        if len(times_array) > 1:
            az_rates_dps[0] = (az_deg[1] - az_deg[0]) / dt_seconds
            az_rates_dps[-1] = (az_deg[-1] - az_deg[-2]) / dt_seconds
            el_rates_dps[0] = (alt_deg[1] - alt_deg[0]) / dt_seconds
            el_rates_dps[-1] = (alt_deg[-1] - alt_deg[-2]) / dt_seconds

        # Create trajectory data array (same format as satellite trajectories)
        trajectory_data = np.column_stack([
            tt_array,         # time_vals (TT)
            alt_deg,          # alt_deg
            az_deg,           # az_deg
            dist_km,          # dist_km
            px_coords,        # px_coords
            py_coords,        # py_coords
            az_rates_dps,     # az_rates_dps
            el_rates_dps,     # el_rates_dps
            sun_stat          # sunlit status
            
        ]).tolist()

        # Create arc segments
        segments = _create_tracking_arc_segments(trajectory_data, launch_time.tt)

        return {
            'trajectory': (trajectory_data, tt_array),
            'arcs': segments
        }

    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Error reading tracking trajectory {filepath}: {e}")
        return None


def _create_tracking_arc_segments(trajectory_data, start_time):
    """Create arc segments for tracking trajectories"""
    if len(trajectory_data) < 2:
        return []

    times = np.array([row[0] for row in trajectory_data])
    alts = np.array([row[1] for row in trajectory_data])
    azs = np.array([row[2] for row in trajectory_data])
    px = np.array([row[4] for row in trajectory_data[:-1]])
    py = np.array([row[5] for row in trajectory_data[:-1]])
    px_next = np.array([row[4] for row in trajectory_data[1:]])
    py_next = np.array([row[5] for row in trajectory_data[1:]])
    az_next = np.array([row[2] for row in trajectory_data[1:]])

    above_horizon = (alts[:-1] > 0) | (alts[1:] > 0)

    if not above_horizon.any():
        return []

    segments = []
    for i in range(len(trajectory_data) - 1):
        if above_horizon[i]:
            # Check for large azimuth jumps to prevent "clipping" across the plot
            az_diff = abs(azs[i] - az_next[i])
            az_diff = min(az_diff, 360 - az_diff)  # Handle wraparound
            if az_diff > 90:  # Skip segments with large azimuth changes
                continue

            is_future = times[i] > start_time
            color = (255, 255, 0) if is_future else (128, 128, 128)  # Yellow for tracking trajectories
            segments.append((px[i], py[i], px_next[i], py_next[i], color))

    return segments


def read_launch_trajectories(launches_dir="./launches", display=None, update_status_callback=None):
    """
    Read all launch trajectory CSV files from the specified directory.
    Files are sorted by creation time (newest first) and parsed into trajectory format.
    Returns a dictionary of launch_name -> (trajectory_data, times_array)
    """
    import os
    import csv
    from datetime import datetime, timezone

    launch_trajectories = {}

    if not os.path.exists(launches_dir):
        if update_status_callback:
            update_status_callback(f"Launch directory '{launches_dir}' not found")
        return launch_trajectories

    # Get all CSV and TXT files and sort by creation time (newest first)
    trajectory_files = [f for f in os.listdir(launches_dir) if f.endswith(('.csv', '.txt'))]
    if not trajectory_files:
        if update_status_callback:
            update_status_callback("No launch trajectory files found")
        return launch_trajectories

    # Sort by creation time (newest first)
    trajectory_files_with_times = []
    for trajectory_file in trajectory_files:
        filepath = os.path.join(launches_dir, trajectory_file)
        try:
            # Use modification time as approximation of creation time
            mod_time = os.path.getmtime(filepath)
            trajectory_files_with_times.append((trajectory_file, mod_time))
        except OSError:
            # If we can't get mod time, put at end
            trajectory_files_with_times.append((trajectory_file, 0))

    # Sort by time (newest first)
    trajectory_files_with_times.sort(key=lambda x: x[1], reverse=True)

    if update_status_callback:
        update_status_callback(f"Reading {len(trajectory_files)} launch trajectory files...")

    success_count = 0
    for trajectory_file, _ in trajectory_files_with_times:
        filepath = os.path.join(launches_dir, trajectory_file)

        try:
            if trajectory_file.endswith('.txt'):
                # Handle new TXT format with ECEF coordinates
                trajectory_result = read_tracking_trajectory(filepath, display, update_status_callback)
                if trajectory_result:
                    launch_name = trajectory_file[:-4]  # Remove .txt extension
                    launch_trajectories[launch_name] = trajectory_result['trajectory']
                    launch_trajectories[launch_name + '_arcs'] = trajectory_result['arcs']
                    success_count += 1
                    if update_status_callback:
                        update_status_callback(f"Loaded tracking trajectory: {launch_name}")
                else:
                    if update_status_callback:
                        update_status_callback(f"Failed to load {trajectory_file}")
            else:
                # Handle existing CSV format
                # Read CSV data
                times_seconds = []
                elevations_deg = []
                azimuths_deg = []
                ranges_km = []

                with open(filepath, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            # Convert time from seconds to absolute Skyfield time starting from current time
                            time_sec = float(row['time'])
                            elevation = float(row['elevationDegs'])
                            azimuth = float(row['azimuthDegs'])
                            range_km = float(row['rangeKm'])

                            times_seconds.append(time_sec)
                            elevations_deg.append(elevation)
                            azimuths_deg.append(azimuth)
                            ranges_km.append(range_km)

                        except (KeyError, ValueError) as e:
                            # Skip malformed rows
                            continue

                if not times_seconds:
                    if update_status_callback:
                        update_status_callback(f"No valid data in {trajectory_file}, skipping")
                    continue

                # Convert to trajectory format matching satellite trajectories:
                # [time_vals, alt_deg, az_deg, dist_km, px_coords, py_coords, az_rates_dps, el_rates_dps]

                # Convert times to Skyfield terrestrial time starting from current time
                current_tt = load.timescale().now().tt
                times_tt = []
                for time_sec in times_seconds:
                    # Convert relative time to absolute TT starting from now
                    abs_tt = current_tt + time_sec / 86400.0  # Convert seconds to days
                    times_tt.append(abs_tt)

                times_array = np.array(times_tt)
                alt_deg = np.array(elevations_deg)
                az_deg = np.array(azimuths_deg)
                dist_km = np.array(ranges_km)

                # Calculate pixel coordinates (polar plot coordinates)
                # Use display info if available, otherwise use default values
                if display:
                    cx = display.sub_x + display.sub_width // 2
                    cy = display.sub_y + display.sub_height // 2
                    radius = min(display.sub_width, display.sub_height) // 2 - 50
                else:
                    # Default values for when display is not available
                    cx = 400  # Approximate center
                    cy = 300  # Approximate center
                    radius = 300  # Approximate radius

                # Convert to pixel coordinates
                az_rad = np.radians(az_deg % 360)
                polar_radius = (90 - alt_deg) / 90 * radius
                px_coords = cx + polar_radius * np.sin(az_rad)
                py_coords = cy - polar_radius * np.cos(az_rad)

                # Calculate rates (azimuth and elevation rates in degrees/second)
                dt_seconds = 1.0  # Assume 1-second intervals (may need adjustment based on actual data spacing)

                az_rates_dps = np.zeros(len(times_array))
                el_rates_dps = np.zeros(len(times_array))

                for i in range(1, len(times_array) - 1):
                    # Use central differences for better accuracy
                    actual_dt = (times_tt[i+1] - times_tt[i-1]) * 86400  # Time step in seconds
                    if actual_dt > 0:
                        az_rates_dps[i] = (az_deg[i + 1] - az_deg[i - 1]) / actual_dt
                        el_rates_dps[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / actual_dt

                # Handle endpoints with forward/backward differences
                if len(times_array) > 1:
                    actual_dt_end = (times_tt[1] - times_tt[0]) * 86400
                    actual_dt_start = (times_tt[-1] - times_tt[-2]) * 86400

                    if actual_dt_end > 0:
                        az_rates_dps[0] = (az_deg[1] - az_deg[0]) / actual_dt_end
                        el_rates_dps[0] = (alt_deg[1] - alt_deg[0]) / actual_dt_end

                    if actual_dt_start > 0:
                        az_rates_dps[-1] = (az_deg[-1] - az_deg[-2]) / actual_dt_start
                        el_rates_dps[-1] = (alt_deg[-1] - alt_deg[-2]) / actual_dt_start

                # Create trajectory data array
                trajectory_data = np.column_stack([
                    times_array,      # time_vals (TT)
                    alt_deg,          # alt_deg
                    az_deg,           # az_deg
                    dist_km,          # dist_km
                    px_coords,        # px_coords
                    py_coords,        # py_coords
                    az_rates_dps,     # az_rates_dps
                    el_rates_dps      # el_rates_dps
                ]).tolist()

                # Create arc segments (similar to satellite trajectory rendering)
                segments = _create_launch_arc_segments(trajectory_data, current_tt)

                # Store in launch_trajectories dict
                launch_name = trajectory_file[:-4]  # Remove .csv extension
                launch_trajectories[launch_name] = (trajectory_data, times_array)

                # Also store arc segments separately
                launch_trajectories[launch_name + '_arcs'] = segments

                success_count += 1
                if update_status_callback:
                    update_status_callback(f"Loaded launch trajectory: {launch_name}")

        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Error reading {trajectory_file}: {e}")
            continue

    if update_status_callback:
        update_status_callback(f"Successfully loaded {success_count}/{len(trajectory_files)} launch trajectories")

    return launch_trajectories

def _create_launch_arc_segments(trajectory_data, start_time):
    """Create arc segments for launch trajectories (similar to satellite arcs)"""
    if len(trajectory_data) < 2:
        return []

    # Unpack trajectory data
    times = np.array([row[0] for row in trajectory_data])
    alts = np.array([row[1] for row in trajectory_data])
    px = np.array([row[4] for row in trajectory_data[:-1]])  # Start positions
    py = np.array([row[5] for row in trajectory_data[:-1]])
    px_next = np.array([row[4] for row in trajectory_data[1:]])  # End positions
    py_next = np.array([row[5] for row in trajectory_data[1:]])

    # Create masks for segments that are above horizon
    above_horizon = (alts[:-1] > 0) | (alts[1:] > 0)

    if not above_horizon.any():
        return []

    # Create segments with cyan/white color scheme for launches
    segments = []
    for i in range(len(trajectory_data) - 1):
        if above_horizon[i]:
            is_future = times[i] > start_time
            color = (0, 255, 255) if is_future else (128, 128, 128)  # Cyan for future, gray for past
            segments.append((px[i], py[i], px_next[i], py_next[i], color))

    return segments

def update_launch_positions(state, current_tt):
    """
    Update launch positions based on current time and launch state.
    Modifies state object with launch_positions dictionary.
    """
    state.launch_positions = {}  # launch_name -> (px, py, alt, dist)

    if not state.launch_trajectories:
        return

    for launch_name, trajectory_data in state.launch_trajectories.items():
        if launch_name.endswith('_arcs'):
            continue  # Skip arc segment data

        trajectory, times_array = trajectory_data

        if not trajectory:
            continue

        # For launches, behavior depends on whether it's launched or not
        if state.selected_launch == launch_name:
            if state.launch_launched and state.launch_start_time is not None:
                # Launched: advance with real time starting from trajectory beginning
                # Both launch_start_time and current_tt are in TT format (days)
                launch_tt = times_array[0] + (current_tt - state.launch_start_time)
            else:
                # Not launched: pinned at T-0
                launch_tt = times_array[0]  # Use first point (T-0)
        else:
            # Not selected: use current time for display purposes
            launch_tt = current_tt

        # Interpolate position at the desired time
        px, py, alt, dist = interpolate_launch_position(trajectory_data, launch_tt)

        if px is not None and alt > 0:  # Only show if above horizon
            state.launch_positions[launch_name] = (px, py, alt, dist)

def interpolate_launch_position(trajectory_data, current_tt):
    """
    Interpolate launch position at a given time (launch-relative or absolute).
    Returns (px, py, alt, dist)
    """
    if not trajectory_data or not trajectory_data[0]:
        return None, None, None, None

    trajectory, times_array = trajectory_data

    # Find the insertion point for current_tt
    idx = np.searchsorted(times_array, current_tt) - 1
    if idx < 0:
        # Before the first point, use the first point (T-0)
        return trajectory[0][4], trajectory[0][5], trajectory[0][1], trajectory[0][3]
    elif idx >= len(times_array) - 1:
        # After the last point, use the last point
        return trajectory[-1][4], trajectory[-1][5], trajectory[-1][1], trajectory[-1][3]
    else:
        # Linear interpolation between idx and idx+1
        t0 = times_array[idx]
        t1 = times_array[idx + 1]
        fraction = (current_tt - t0) / (t1 - t0)

        px0, py0, alt0, dist0 = trajectory[idx][4], trajectory[idx][5], trajectory[idx][1], trajectory[idx][3]
        px1, py1, alt1, dist1 = trajectory[idx + 1][4], trajectory[idx + 1][5], trajectory[idx + 1][1], trajectory[idx + 1][3]

        px = px0 + fraction * (px1 - px0)
        py = py0 + fraction * (py1 - py0)
        alt = alt0 + fraction * (alt1 - alt0)
        dist = dist0 + fraction * (dist1 - dist0)

        return px, py, alt, dist
