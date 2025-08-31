from skyfield.api import wgs84, load
from datetime import datetime, timedelta, timezone
import numpy as np
import math

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

        # If any altitude is above the elevation mask + some margin, consider it potentially visible
        min_visible_alt = max(elevation_mask_deg, 15.0)  # At least 15 degrees to account for sample sparsity
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

    # Create coarse time samples for visibility checking (every 2 minutes for speed)
    num_samples = max(5, int(duration_minutes / 2))  # At least 5 samples
    visibility_times = ts.linspace(t0, t1, num_samples)

    visible_satellites = []
    total_checked = 0

    for sat in satellites:
        total_checked += 1
        if _is_satellite_potentially_visible(sat, observer, visibility_times, elevation_mask_deg):
            visible_satellites.append(sat)

        # Progress update for filtering
        if update_status_callback and total_checked % 500 == 0 and total_checked > 0:
            update_status_callback(f"Visibility filtered {len(visible_satellites)}/{total_checked} satellites...")

    if visible_satellites:
        visibility_ratio = len(visible_satellites) / len(satellites) * 100
    else:
        visibility_ratio = 0

    if update_status_callback:
        update_status_callback(f"Visibility filtering: {len(visible_satellites)}/{len(satellites)} satellites potentially visible ({visibility_ratio:.1f}%)")

    return visible_satellites

def precompute_trajectories(satellites, observer, ts, sub_x, sub_y, sub_width, sub_height, satellite_labels, update_status_callback=None, center_time=None, duration_minutes=15):
    """Optimized trajectory computation with caching, reduced resolution, vectorization, and visibility filtering"""

    # OPTIMIZATION 8: Filter satellites by potential visibility first (potentially 2-10x speedup)
    visible_satellites = _filter_satellites_by_visibility(
        satellites, observer, ts, center_time, duration_minutes, elevation_mask_deg=10.0, update_status_callback=update_status_callback
    )

    # Setup constants
    cx = sub_x + sub_width // 2
    cy = sub_y + sub_height // 2
    radius = min(sub_width, sub_height) // 2 - 50

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
        if sat in satellite_labels:
            satellites_in_label_dict += 1
            # Create cache key based on satellite ID and time range
            sat_model = getattr(sat, 'model', None)
            sat_id = getattr(sat_model, 'satnum_str', str(sat)) if sat_model else str(sat)
            cache_key = f"{sat_id}_{t0.tt:.6f}_{t1.tt:.6f}"

            if cache_key not in TRAJECTORY_CACHE:
                satellites_to_compute.append((sat, cache_key))

    total_sats = len(satellites_to_compute)
    trajectories = {}
    arc_segments = {}

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

        # OPTIMIZATION 6: Create trajectory data structure
        trajectory_data = np.column_stack([time_vals, alt_deg, az_deg, dist_km, px_coords, py_coords])
        trajectory = trajectory_data.tolist()

        # OPTIMIZATION 7: Create arc segments (color will be determined during rendering)
        segments = _create_arc_segments_simple(trajectory_data, times.tt[0])

        # Cache the computed trajectory
        TRAJECTORY_CACHE[cache_key] = (trajectory, times_array, segments)

        trajectories[sat] = (trajectory, times_array)
        arc_segments[sat] = segments

        # Progress update
        if update_status_callback and processed_count % 100 == 0 and processed_count > 0:
            update_status_callback(f"Computed traj {processed_count}/{total_sats} sats...")

    # Load cached trajectories for satellites that don't need recomputation
    cache_hits = 0
    for sat in satellites:
        if sat in satellite_labels:
            sat_model = getattr(sat, 'model', None)
            sat_id = getattr(sat_model, 'satnum_str', str(sat)) if sat_model else str(sat)
            cache_key = f"{sat_id}_{t0.tt:.6f}_{t1.tt:.6f}"

            if cache_key in TRAJECTORY_CACHE:
                cached_data = TRAJECTORY_CACHE[cache_key]
                trajectories[sat] = (cached_data[0], cached_data[1])
                arc_segments[sat] = cached_data[2]
                cache_hits += 1

    if cache_hits > 0:
        print(f"DEBUG: Trajectory optimization: {cache_hits} satellites loaded from cache")

    return trajectories, arc_segments

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

def interpolate_position(trajectory_data, current_tt):
    if not trajectory_data[0]:  # Check if trajectory is empty
        return None, None, None, None
    trajectory, times_array = trajectory_data
    # Find the insertion point for current_tt
    idx = np.searchsorted(times_array, current_tt) - 1
    if idx < 0:
        # Before the first point, use the first point
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

def build_satellite_pass_table(satellite_trajectories, satellites, satellite_labels, elevation_mask_deg=10.0, max_rows=20, ts=None, current_satellite_positions=None):
    """
    Build pass table data from all satellite trajectories.
    Only includes satellites that are currently in view OR will be in view soon (future closest approach).
    Returns sorted list of pass data (default: max elevation descending).
    """
    pass_entries = []

    # Get current time in local timezone for comparison
    current_datetime = datetime.now(timezone.utc)
    current_local = current_datetime.astimezone()

    for sat in satellites:
        if sat in satellite_trajectories and sat in satellite_labels:
            pass_data = extract_pass_data_from_trajectory(satellite_trajectories[sat], sat, satellite_labels, elevation_mask_deg, ts)
            if pass_data and pass_data['is_visible']:
                # Filter for satellites that are either:
                # 1. Currently in view (checked against satellite_positions)
                # 2. Will be in view soon (future closest approach time)
                include_in_table = False

                # Check if satellite is currently in view
                if current_satellite_positions and sat in current_satellite_positions:
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

    # Sort by max elevation descending by default
    pass_entries.sort(key=lambda x: x['max_elevation'], reverse=True)

    # Limit to max_rows
    return pass_entries[:max_rows]

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
