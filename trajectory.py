import numpy as np
import math
import os
from datetime import datetime, timedelta, timezone
from skyfield.api import wgs84, load

# Rust astro engine bridge (Phase 1 strangler seam). Every call returns
# None on failure, so the skyfield paths below remain the safety net.
import rust_astro_adapter

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Simulation and Trajectory Parameters
TRAJECTORY_DURATION_MINUTES_DEFAULT = 15  # Default duration for trajectory computation
TIME_SAMPLES_COUNT = 301  # Legacy: sample count at the historical 30 min window (kept for reference)
# Bulk trajectories (all visible sats) are for DISPLAY + the pass table, so they
# are sampled coarsely -- this keeps a long (e.g. 120 min) window cheap to compute
# and cache. The actively tracked satellite is recomputed separately at the fine
# interval below (see compute_fine_selected_trajectory), because live PROGRAM
# tracking interpolates that one trajectory and needs tight spacing on fast passes.
TRAJECTORY_SAMPLE_INTERVAL_SECONDS = 30.0       # bulk/display spacing (sample count scales with duration)
TRAJECTORY_FINE_SAMPLE_INTERVAL_SECONDS = 6.0   # selected/tracked satellite spacing
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

# Pass-table brightness estimate. TLEs carry no per-satellite intrinsic
# magnitude, so every satellite gets the same standard magnitude (apparent mag
# at 1000 km range, 50% illuminated -- the McCants convention); the sortable
# column is therefore dominated by range and phase geometry. 6.0 is a
# reasonable middle for the active catalog (Starlink ~5.5-6.5; ISS is much
# brighter, cubesats dimmer).
PASS_MAG_DEFAULT_STDMAG = 6.0
R_EARTH_SHADOW_KM = 6378.137  # equatorial radius for the cylindrical shadow test

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

# Global cache for trajectories to avoid recomputation. Scoped to one
# (observer, plot geometry) tuple -- see the scope guard in
# precompute_trajectories; a site/geometry change clears it wholesale.
TRAJECTORY_CACHE = {}
_TRAJECTORY_CACHE_SCOPE = None
SUN_EPHEMERIS_CACHE = None

# ---- Disk-backed trajectory cache --------------------------------------------
# Persists the most recent window's computed trajectories so a quick relaunch can
# serve the precise pass from disk instead of recomputing it (~2.5s for ~1700
# sats). Hits require an exact (TLE fingerprint, t0, t1) match, which is why the
# auto/launch window center is snapped to a coarse grid -- see quantized_now().
TRAJECTORY_DISK_CACHE_FILE = "trajectory_disk_cache.npz"
# Window-center quantization. Two launches within the same bucket produce
# identical t0/t1 -> identical cache keys -> a full disk hit. The displayed window
# center can lag true "now" by up to this many seconds, which is imperceptible
# against the default +/-15 min window. Tune up for more hits, down for less lag.
AUTO_CENTER_QUANTIZE_SECONDS = 120


def quantized_now(bucket_seconds=AUTO_CENTER_QUANTIZE_SECONDS):
    """Current UTC time floored to a bucket grid, so the launch/auto trajectory
    window is reproducible across quick relaunches (enables disk-cache hits)."""
    now = datetime.now(timezone.utc)
    snapped = (int(now.timestamp()) // int(bucket_seconds)) * int(bucket_seconds)
    return datetime.fromtimestamp(snapped, tz=timezone.utc)


def clear_trajectory_cache():
    """Clear the trajectory cache to free memory"""
    global TRAJECTORY_CACHE, SUN_EPHEMERIS_CACHE
    TRAJECTORY_CACHE.clear()
    SUN_EPHEMERIS_CACHE = None


_LIVE_TS = None


def live_tt():
    """Current TT Julian date from the real (wall) clock.

    The control loops must aim the mount at where the target is NOW.
    `tracking_vis_state.current_tt` is the UI's *tracking clock* -- it follows
    the time slider while scrubbing and freezes while paused -- so it is the
    right time base for visualization but the wrong one for hardware: a mount
    cannot time-travel. Every control-path setpoint interpolation uses this
    instead."""
    global _LIVE_TS
    if _LIVE_TS is None:
        _LIVE_TS = load.timescale()
    return _LIVE_TS.now().tt


def _unwrap_az_diff(d):
    """Shortest-arc azimuth difference in degrees (wraps 0/360 seam)."""
    return (d + 180.0) % 360.0 - 180.0


def _file_fingerprint(path):
    """Cheap content fingerprint (size + mtime) so an edited source file
    invalidates any cache derived from it automatically."""
    try:
        st = os.stat(path)
        return f"{int(st.st_size)}_{int(st.st_mtime)}"
    except OSError:
        return ""


def save_trajectory_disk_cache(state, t0_tt, t1_tt, tle_path="tle_cache.tle",
                               cache_file=TRAJECTORY_DISK_CACHE_FILE):
    """Persist the current window's trajectories to disk as packed float arrays.

    Stores a single window (overwrites any previous one). All satellites in a
    window share one time grid, so it's stored once. Arc segments are NOT stored
    -- they're cheaply recomputed from the trajectory on load. Best-effort: any
    failure is swallowed (the cache is an optimization, never required)."""
    try:
        entries = []  # (sid, traj_f32, times_f64)
        for s in state.satellites:
            if s not in state.satellite_trajectories:
                continue
            traj, ta = state.satellite_trajectories[s]
            if not traj:
                continue
            sid = getattr(getattr(s, 'model', None), 'satnum_str', None)
            if sid is None:
                continue
            entries.append((sid, np.asarray(traj, dtype=np.float32), np.asarray(ta, dtype=np.float64)))
        if not entries:
            return 0

        # Bulk trajectories share one sample count, but the selected satellite may
        # carry a fine-resolution entry of a different length. Keep only the modal
        # length so np.stack and the single shared time grid stay valid; the fine
        # selected-sat trajectory is recomputed live next run, so it needn't persist.
        from collections import Counter
        modal_len = Counter(e[1].shape[0] for e in entries).most_common(1)[0][0]
        entries = [e for e in entries if e[1].shape[0] == modal_len]
        if not entries:
            return 0

        ids = [e[0] for e in entries]
        times_array = entries[0][2]
        data = np.stack([e[1] for e in entries], axis=0)  # (n_sats, n_samples, n_cols) float32
        # np.savez appends ".npz" if the name lacks it, so keep the suffix on the
        # temp file too; then atomically swap it into place.
        tmp = cache_file + ".tmp.npz"
        np.savez(tmp,
                 fingerprint=np.array(_file_fingerprint(tle_path)),
                 t0_tt=np.array(float(t0_tt)), t1_tt=np.array(float(t1_tt)),
                 times=times_array, ids=np.array(ids), data=data)
        os.replace(tmp, cache_file)
        return len(ids)
    except Exception:
        return 0


def load_trajectory_disk_cache(t0_tt, t1_tt, tle_path="tle_cache.tle",
                               cache_file=TRAJECTORY_DISK_CACHE_FILE):
    """If a disk cache matching this exact (TLE fingerprint, t0, t1) window
    exists, reconstruct its entries into the in-memory TRAJECTORY_CACHE so the
    next precompute_trajectories() serves them as hits. Returns entries loaded
    (0 on miss). Cheap to call on a miss (reads only the small header arrays)."""
    if not os.path.exists(cache_file):
        return 0
    try:
        with np.load(cache_file, allow_pickle=False) as z:
            if str(z['fingerprint']) != _file_fingerprint(tle_path):
                return 0
            if abs(float(z['t0_tt']) - float(t0_tt)) > 1e-6 or \
               abs(float(z['t1_tt']) - float(t1_tt)) > 1e-6:
                return 0
            times = z['times']
            ids = z['ids']
            data = z['data']
    except Exception:
        return 0

    start_tt = float(t0_tt)
    times = np.asarray(times, dtype=np.float64)
    loaded = 0
    for i in range(len(ids)):
        arr = np.asarray(data[i], dtype=np.float64)  # (n_samples, n_cols)
        # Column 0 is the absolute TT Julian date (~2.46e6); float32 can't hold it
        # to better than ~0.1 day, which would corrupt the future/past arc coloring.
        # It's identical to the float64 `times` grid we stored separately, so
        # restore it exactly rather than trusting the packed float32 copy.
        if arr.shape[1] > 0:
            arr[:, 0] = times
        traj = arr.tolist()
        segments = _create_arc_segments_simple(arr, start_tt)
        cache_key = f"{ids[i]}_{start_tt:.6f}_{float(t1_tt):.6f}"
        TRAJECTORY_CACHE[cache_key] = (traj, times, segments)
        loaded += 1
    return loaded

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

def _batched_visibility_mask(satellites, observer, visibility_times, min_visible_alt_deg):
    """
    Vectorized coarse visibility test for ALL satellites at once.

    Propagates every satellite at every sample time in a single compiled
    SGP4 call (sgp4.api.SatrecArray) instead of one Python call per satellite,
    then converts the TEME positions to topocentric elevation with pure NumPy.
    Returns a boolean array (len == len(satellites)); True == the satellite rises
    above `min_visible_alt_deg` at one or more sample times.

    This is an approximation of Skyfield's full altaz() (it uses a single GAST
    z-rotation for TEME->ECEF and skips precession/nutation/polar-motion), but
    the error is < ~0.05 deg -- far inside the 15 deg visibility margin this gate
    runs with -- so it does not change which satellites pass. Raises on any
    unexpected condition so the caller can fall back to the per-satellite path.
    """
    from sgp4.api import SatrecArray  # part of skyfield's own sgp4 dependency

    # Observer geocentric position + local "up" unit vector (WGS84 ellipsoid), km.
    a_km = 6378.137
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    phi = math.radians(observer.latitude.degrees)
    lam = math.radians(observer.longitude.degrees)
    h_km = observer.elevation.m / 1000.0
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    N = a_km / math.sqrt(1.0 - e2 * sin_phi * sin_phi)
    ox = (N + h_km) * cos_phi * cos_lam
    oy = (N + h_km) * cos_phi * sin_lam
    oz = (N * (1.0 - e2) + h_km) * sin_phi
    # Local up (ENU) unit vector in ECEF.
    ux, uy, uz = cos_phi * cos_lam, cos_phi * sin_lam, sin_phi

    # Batched SGP4: r has shape (n_sats, n_times, 3) in TEME km.
    satrec_array = SatrecArray([sat.model for sat in satellites])
    jd = np.ascontiguousarray(visibility_times.whole)
    fr = np.ascontiguousarray(visibility_times.ut1_fraction)
    err, r, _v = satrec_array.sgp4(jd, fr)

    x, y, z = r[:, :, 0], r[:, :, 1], r[:, :, 2]

    # TEME -> ECEF: rotation about z by GAST (broadcast across the time axis).
    theta = np.asarray(visibility_times.gast) * (np.pi / 12.0)  # hours -> radians
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xe = cos_t * x + sin_t * y
    ye = -sin_t * x + cos_t * y
    ze = z

    # Topocentric range vector and elevation (only the "up" projection is needed).
    rho_x, rho_y, rho_z = xe - ox, ye - oy, ze - oz
    up = rho_x * ux + rho_y * uy + rho_z * uz
    rng = np.sqrt(rho_x * rho_x + rho_y * rho_y + rho_z * rho_z)
    with np.errstate(invalid='ignore', divide='ignore'):
        elev_deg = np.degrees(np.arcsin(np.clip(up / rng, -1.0, 1.0)))

    # Propagation failures (decayed orbits etc.) can't be visible; drop them so
    # they never reach the precise pass (where they'd error out anyway).
    elev_deg = np.where(err != 0, -np.inf, elev_deg)

    return np.any(elev_deg > min_visible_alt_deg, axis=1)

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

    min_visible_alt = max(elevation_mask_deg, FRUSTUM_UNCERTAINTY_MARGIN_DEGREES)

    # Rust gate: parallel propagation of the whole catalog off the GIL. The
    # engine matches skyfield's full pipeline (0.03 arcsec), tighter than the
    # Python coarse gate below, and the 15 deg margin dwarfs both.
    if satellites and rust_astro_adapter.enabled():
        vis = rust_astro_adapter.visible_satnums(
            satellites, visibility_times, observer, min_visible_alt)
        if vis is not None:
            visible_satellites = [
                sat for sat in satellites
                if getattr(getattr(sat, 'model', None), 'satnum_str', None) in vis]
            if update_status_callback:
                update_status_callback(
                    f"Visibility filtering (rust): {len(visible_satellites)}/{len(satellites)} satellites potentially visible")
            return visible_satellites

    # Fast path: propagate all satellites in one batched SGP4 call. Falls back to
    # the per-satellite loop if anything about the vectorized path goes wrong.
    visible_satellites = None
    if satellites:
        try:
            mask = _batched_visibility_mask(satellites, observer, visibility_times, min_visible_alt)
            visible_satellites = [sat for sat, vis in zip(satellites, mask) if vis]
        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Batched visibility filter unavailable ({e}); using per-sat path")
            visible_satellites = None

    if visible_satellites is None:
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

def _rows_to_trajectory(rows):
    """(n, 8) ndarray -> the (trajectory_list, times_array, segments) triple
    _compute_one_trajectory returns; shared by the Rust fast paths."""
    times_array = rows[:, 0].copy()
    trajectory = rows.tolist()
    segments = _create_arc_segments_simple(rows, times_array[0])
    return trajectory, times_array, segments


def _compute_one_trajectory(sat, observer, times, cx, cy, radius):
    """Compute a single satellite's trajectory + arc segments over `times`.

    Shared by the bulk precompute (coarse spacing, all sats) and the fine
    selected-satellite recompute (tight spacing, one sat). Returns
    (trajectory_list, times_array, segments) in the standard 8-column format:
    [time_tt, alt_deg, az_deg, dist_km, px, py, az_rate_dps, el_rate_dps]."""
    # Rust fast path: same rows (0.03 arcsec parity), same segment builder.
    if rust_astro_adapter.enabled():
        rows = rust_astro_adapter.satellite_rows(sat, times, observer, cx, cy, radius)
        if rows is not None:
            return _rows_to_trajectory(rows)

    topocentrics = (sat - observer).at(times)
    alts, azs, distances = topocentrics.altaz()

    alt_deg = np.array(alts.degrees)
    az_deg = np.array(azs.degrees)
    dist_km = np.array(distances.km)
    time_vals = np.array(times.tt)

    # Vectorized polar-plot pixel coordinates.
    az_rad = np.radians(az_deg % 360)
    polar_radius = (90 - alt_deg) / 90 * radius
    px_coords = cx + polar_radius * np.sin(az_rad)
    py_coords = cy - polar_radius * np.cos(az_rad)

    times_array = time_vals.copy()

    # Angular rates (deg/s) via finite differences over the sample spacing.
    dt_seconds = (times_array[1] - times_array[0]) * 86400.0 if len(times_array) > 1 else 1.0
    az_rates_dps = np.zeros(len(times_array))
    el_rates_dps = np.zeros(len(times_array))
    # Azimuth differences take the short arc: a pass crossing due north jumps
    # 359->1 deg, and the raw difference would feed a spurious ~360deg/dt spike
    # straight into the feed-forward rate.
    for i in range(1, len(times_array) - 1):
        az_rates_dps[i] = _unwrap_az_diff(az_deg[i + 1] - az_deg[i - 1]) / (2 * dt_seconds)
        el_rates_dps[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / (2 * dt_seconds)
    if len(times_array) > 1:
        az_rates_dps[0] = _unwrap_az_diff(az_deg[1] - az_deg[0]) / dt_seconds
        az_rates_dps[-1] = _unwrap_az_diff(az_deg[-1] - az_deg[-2]) / dt_seconds
        el_rates_dps[0] = (alt_deg[1] - alt_deg[0]) / dt_seconds
        el_rates_dps[-1] = (alt_deg[-1] - alt_deg[-2]) / dt_seconds

    trajectory_data = np.column_stack([time_vals, alt_deg, az_deg, dist_km, px_coords, py_coords, az_rates_dps, el_rates_dps])
    trajectory = trajectory_data.tolist()
    segments = _create_arc_segments_simple(trajectory_data, time_vals[0])
    return trajectory, times_array, segments


def compute_fine_selected_trajectory(state, sat, observer, ts, display, t0, t1,
                                     fine_interval_seconds=TRAJECTORY_FINE_SAMPLE_INTERVAL_SECONDS):
    """Recompute ONE satellite (the tracked/selected one) at fine time resolution
    and publish it into state.satellite_trajectories / _arc_segments, replacing
    the coarse bulk entry. Bulk trajectories are sampled coarsely for display;
    PROGRAM tracking interpolates this entry, so the selected sat needs tight
    spacing to keep interpolation + feed-forward error low on fast passes.

    Single satellite, so it's only a few milliseconds -- safe to call inline from
    the main loop on selection/window change. Returns True on success."""
    if sat is None or t0 is None or t1 is None:
        return False
    try:
        cx = display.sub_x + display.sub_width // 2
        cy = display.sub_y + display.sub_height // 2
        radius = min(display.sub_width, display.sub_height) // 2 - 50
        span_days = float(t1.tt - t0.tt)
        n = max(2, round(span_days * 86400.0 / fine_interval_seconds) + 1)
        times = ts.linspace(t0, t1, n)
        trajectory, times_array, segments = _compute_one_trajectory(sat, observer, times, cx, cy, radius)

        # Publish via copy-rebind so a concurrent reader (render thread / position
        # loop) always sees a complete dict, consistent with precompute's atomic
        # publish. Cheap: copies ~dict-of-refs, only on selection/window change.
        new_traj = dict(state.satellite_trajectories)
        new_arcs = dict(state.satellite_arc_segments)
        new_traj[sat] = (trajectory, times_array)
        new_arcs[sat] = segments
        state.satellite_trajectories = new_traj
        state.satellite_arc_segments = new_arcs
        # The sunlit cache is a list index-parallel to this sat's times_array;
        # the fine grid just changed its length (and times), so the cached
        # entry would silently mis-index (tail painted shadowed). Drop it.
        try:
            new_sunlit = dict(state.sunlit_status_cache)
            new_sunlit.pop(getattr(sat, 'name', None), None)
            state.sunlit_status_cache = new_sunlit
        except AttributeError:
            pass
        return True
    except Exception:
        return False


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

    # Cache scope guard: TRAJECTORY_CACHE entries bake in the observer AND the
    # polar-plot pixel geometry (columns 4/5), but the per-sat keys only carry
    # the time window. If the site or the plot geometry changed since the cache
    # was filled, every entry is stale -- and the dict otherwise grows without
    # bound as the auto-refresh advances the window. Scope the whole cache to
    # one (observer, geometry) tuple and clear it on any change.
    global _TRAJECTORY_CACHE_SCOPE
    try:
        scope = (round(float(observer.latitude.degrees), 6),
                 round(float(observer.longitude.degrees), 6),
                 round(float(observer.elevation.m), 1),
                 int(cx), int(cy), int(radius))
    except AttributeError:
        scope = (int(cx), int(cy), int(radius))
    if scope != _TRAJECTORY_CACHE_SCOPE:
        TRAJECTORY_CACHE.clear()
        _TRAJECTORY_CACHE_SCOPE = scope
    elif len(TRAJECTORY_CACHE) > 6000:
        # ~3 windows of ~1700 sats: entries from old windows are never
        # revisited (keys carry t0/t1), so growth is pure leak -- reset.
        TRAJECTORY_CACHE.clear()

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

    # Bulk/display sampling: coarse, fixed cadence; sample count scales with the
    # window so a long (120 min) span stays cheap. The tracked satellite is
    # refined elsewhere (compute_fine_selected_trajectory) for tracking accuracy.
    num_time_samples = max(2, round(duration_minutes * 60.0 / TRAJECTORY_SAMPLE_INTERVAL_SECONDS) + 1)
    times = ts.linspace(t0, t1, num_time_samples)

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
    # Build into local dicts and publish them in a single atomic rebind at the
    # end (see the tail of this function). precompute may now run on a background
    # worker thread while the render thread reads state.satellite_trajectories /
    # _arc_segments live every frame; incrementally filling the live attributes
    # here would let a render land mid-fill and see a partial frame. Assigning the
    # finished dicts once means readers always see the complete old or new set.
    trajectories = {}
    arc_segments = {}

    if update_status_callback:
        update_status_callback(f"Computing optimized trajectories for {total_sats}/{satellites_in_label_dict} satellites...")

    # Rust bulk fast path: one FFI call computes every satellite's rows in
    # parallel (rayon) with the GIL released; falls back per-sat below for
    # any satellite the engine skipped (or entirely, when the flag is off).
    rust_bulk = None
    if satellites_to_compute and rust_astro_adapter.enabled():
        rust_bulk = rust_astro_adapter.bulk_rows(
            [sat for sat, _ in satellites_to_compute], times, observer, cx, cy, radius)

    processed_count = 0
    for sat, cache_key in satellites_to_compute:
        processed_count += 1

        rows = None
        if rust_bulk is not None:
            model = getattr(sat, 'model', None)
            rows = rust_bulk.get(getattr(model, 'satnum_str', None)) if model else None
        if rows is not None and rows.shape[0] == len(times.tt):
            trajectory, times_array, segments = _rows_to_trajectory(rows)
        else:
            trajectory, times_array, segments = _compute_one_trajectory(sat, observer, times, cx, cy, radius)

        # Cache the computed trajectory
        TRAJECTORY_CACHE[cache_key] = (trajectory, times_array, segments)

        trajectories[sat] = (trajectory, times_array)
        arc_segments[sat] = segments

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
                trajectories[sat] = (cached_data[0], cached_data[1])
                arc_segments[sat] = cached_data[2]
                cache_hits += 1

    if cache_hits > 0:
        print(f"DEBUG: Trajectory optimization: {cache_hits} satellites loaded from cache")

    # Atomic publish: readers (render thread / position updates) see either the
    # complete previous set or this complete new one, never a partial fill.
    state.satellite_trajectories = trajectories
    state.satellite_arc_segments = arc_segments
    # New time grid: every cached per-sat sunlit list is index-parallel to the
    # OLD grid (same length, different times after a window refresh), so it
    # would color the new arcs with the old window's shadow pattern.
    if hasattr(state, 'sunlit_status_cache'):
        state.sunlit_status_cache = {}
    # Also publish a packed array view of the columns the per-frame position
    # update needs (alt, dist, px, py), so that update can interpolate every
    # satellite in one vectorized shot instead of a Python loop. Single-tuple
    # rebind so a concurrent reader sees a consistent (stack, sats, times) set.
    state.position_stack = _build_position_stack(trajectories)


def _build_position_stack(trajectories):
    """Pack the bulk trajectories into one contiguous array for vectorized
    per-frame position interpolation. Returns (stack, sats, times) where
    stack has shape (n_sats, n_samples, 4) holding [alt, dist, px, py], sats is
    the matching satellite list, and times is the shared TT grid. Satellites that
    don't match the modal grid length (e.g. the fine-resolution selected sat,
    present here at its coarse length when the stack is built) are simply the
    ones kept by the modal filter. Returns None if there's nothing to pack."""
    sats = [s for s, (traj, _ta) in trajectories.items() if traj]
    if not sats:
        return None
    from collections import Counter
    modal_len = Counter(len(trajectories[s][0]) for s in sats).most_common(1)[0][0]
    keep = [s for s in sats if len(trajectories[s][0]) == modal_len]
    if not keep:
        return None
    times = np.asarray(trajectories[keep[0]][1], dtype=np.float64)
    stack = np.empty((len(keep), modal_len, 4), dtype=np.float64)
    for i, s in enumerate(keep):
        arr = np.asarray(trajectories[s][0], dtype=np.float64)
        stack[i] = arr[:, (1, 3, 4, 5)]  # alt, dist, px, py
    return stack, keep, times

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

    # Segments carry (x0, y0, x1, y1, future_color, seg_tt): the future/past
    # choice is made at DRAW time against the tracking clock (current_tt), not
    # baked here -- baking froze every segment's color at window start, so the
    # arcs never followed the time slider. `start_time` only seeds the legacy
    # baked color for any consumer that ignores seg_tt.
    segments = []
    for i in range(len(trajectory_data) - 1):
        if above_horizon[i]:
            # For sunlit trajectory display, all satellites get time-based coloring
            # Individual satellite sunlit status will be recomputed for selected satellites
            color = (255, 0, 0) if times[i] > start_time else (128, 128, 128)
            segments.append((px[i], py[i], px_next[i], py_next[i], color, times[i]))

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
        # Interpolate azimuth on the short arc: raw interpolation across the
        # 0/360 seam swings the target to the antipode for one sample interval.
        az = (az0 + fraction * _unwrap_az_diff(az1 - az0)) % 360.0
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
        # Short-arc azimuth interpolation (see the 6-column branch above).
        az = (az0 + fraction * _unwrap_az_diff(az1 - az0)) % 360.0
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
    # Scrubbed outside the computed window: there is no data at this time, so
    # show nothing. The old clamp-to-endpoint behavior froze every satellite
    # at the window edge -- a static, fictitious constellation at both ends
    # of the slider.
    t0 = getattr(state, 't0', None)
    t1 = getattr(state, 't1', None)
    if t0 is not None and t1 is not None:
        try:
            if current_tt < t0.tt - 1e-9 or current_tt > t1.tt + 1e-9:
                state.satellite_positions = {}
                return
        except AttributeError:
            pass

    # Build into a local dict and publish it in a single atomic rebind at the
    # end. The rendering thread reads state.satellite_positions live every
    # frame; if we cleared and incrementally repopulated the live attribute
    # here, a render landing mid-loop would copy a partially-filled dict and
    # draw only some satellites -> the visualization "blink"/flicker. Assigning
    # the finished dict once means readers always see a complete frame's worth.
    new_positions = {}

    trajectories = state.satellite_trajectories
    filter_text = state.filter_text
    filter_above = state.filter_above_alt_text
    filter_below = state.filter_below_alt_text
    mean_alts = state.satellite_mean_altitudes
    filters_active = bool(filter_text or filter_above or filter_below)

    # Fast path: one vectorized interpolation over the packed stack. This runs on
    # the main loop every frame; the stack lets us interpolate all ~5k sats and
    # reject below-mask ones in NumPy, then build the dict only for the ~hundreds
    # actually visible -- ~0.6 ms vs ~25 ms for the per-sat Python loop. Skipped
    # when name/altitude filters are active (handled by the inline path below).
    position_stack = getattr(state, 'position_stack', None)
    if position_stack is not None and not filters_active:
        stack, stack_sats, stack_times = position_stack
        n = len(stack_times)
        if n >= 2 and len(stack_sats):
            idx = int(np.searchsorted(stack_times, current_tt) - 1)
            idx = 0 if idx < 0 else (n - 2 if idx > n - 2 else idx)
            ta0 = stack_times[idx]
            ta1 = stack_times[idx + 1]
            frac = (current_tt - ta0) / (ta1 - ta0)
            # Clamp so before-start/after-end hold the endpoint (matches the
            # per-sat clamp semantics of interpolate_position).
            frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
            a = stack[:, idx, :]
            b = stack[:, idx + 1, :]
            alt = a[:, 0] + frac * (b[:, 0] - a[:, 0])
            visible = np.nonzero(alt > elevation_mask_deg)[0]
            if len(visible):
                px = a[:, 2] + frac * (b[:, 2] - a[:, 2])
                py = a[:, 3] + frac * (b[:, 3] - a[:, 3])
                dist = a[:, 1] + frac * (b[:, 1] - a[:, 1])
                for i in visible:
                    new_positions[stack_sats[i]] = (float(px[i]), float(py[i]), float(alt[i]), float(dist[i]))
        # Stack covers all bulk sats (the selected sat is present at its coarse
        # length); publish and return without the per-sat loop.
        state.satellite_positions = new_positions
        return

    # Inline fallback: filters active, or no stack yet. Every bulk trajectory
    # shares ONE time grid, so the search index + fraction are identical across
    # them -- compute once per distinct grid (keyed by id) rather than a
    # np.searchsorted per satellite.
    grid_cache = {}  # id(times_array) -> (kind, idx, frac)

    def _interp_params(times_array):
        key = id(times_array)
        params = grid_cache.get(key)
        if params is None:
            n = len(times_array)
            idx = int(np.searchsorted(times_array, current_tt) - 1)
            if idx < 0:
                params = ('start', 0, 0.0)
            elif idx >= n - 1:
                params = ('end', n - 1, 0.0)
            else:
                ta0 = times_array[idx]
                ta1 = times_array[idx + 1]
                frac = (current_tt - ta0) / (ta1 - ta0)
                params = ('mid', idx, frac)
            grid_cache[key] = params
        return params

    for sat, (trajectory, times_array) in trajectories.items():
        if not trajectory:
            continue
        kind, idx, frac = _interp_params(times_array)
        if kind == 'mid':
            r0 = trajectory[idx]
            r1 = trajectory[idx + 1]
            alt = r0[1] + frac * (r1[1] - r0[1])
            if alt <= elevation_mask_deg:
                continue  # below mask: skip the rest of the interpolation
            px = r0[4] + frac * (r1[4] - r0[4])
            py = r0[5] + frac * (r1[5] - r0[5])
            dist = r0[3] + frac * (r1[3] - r0[3])
        else:
            row = trajectory[idx]
            alt = row[1]
            if alt <= elevation_mask_deg:
                continue
            px, py, dist = row[4], row[5], row[3]

        # Optional name / altitude filters (usually all empty -> skipped).
        if filter_text:
            if not (filter_text.lower() in sat.name.lower() or filter_text in sat.model.satnum_str):
                continue
        if filter_above:
            try:
                if mean_alts[sat] < float(filter_above):
                    continue
            except ValueError:
                continue
        if filter_below:
            try:
                if mean_alts[sat] > float(filter_below):
                    continue
            except ValueError:
                continue

        new_positions[sat] = (px, py, alt, dist)

    # Atomic publish: readers see either the complete old dict or this new one.
    state.satellite_positions = new_positions

def _estimate_pass_magnitude(sat, t, observer, r_sun_km):
    """Estimated apparent visual magnitude of `sat` at time `t` for `observer`.

    Returns None when the satellite sits in Earth's shadow (eclipsed -> not
    visible regardless of geometry) or the computation fails. Uses the
    standard-magnitude formula (PASS_MAG_DEFAULT_STDMAG at 1000 km / 50%
    illuminated) with a Lambertian phase term; the eclipse test is a simple
    cylindrical Earth-shadow model. `r_sun_km` is the geocentric sun position
    vector, computed once per table build (sun direction moves ~0.06 deg over
    a 90 min window -- negligible against a shared default intrinsic mag).
    """
    try:
        r_sat = sat.at(t).position.km
        r_obs = observer.at(t).position.km
        u_sun = r_sun_km / np.linalg.norm(r_sun_km)
        # Cylindrical shadow: behind the terminator plane AND within one Earth
        # radius of the anti-sun axis -> eclipsed.
        along = float(np.dot(r_sat, u_sun))
        if along < 0.0 and np.linalg.norm(r_sat - along * u_sun) < R_EARTH_SHADOW_KM:
            return None
        v_obs = r_obs - r_sat            # satellite -> observer
        v_sun = r_sun_km - r_sat         # satellite -> sun
        range_km = float(np.linalg.norm(v_obs))
        cos_phase = float(np.dot(v_obs, v_sun)) / (range_km * float(np.linalg.norm(v_sun)))
        frac_illum = max(0.5 * (1.0 + cos_phase), 1e-3)
        return PASS_MAG_DEFAULT_STDMAG - 15.75 + 2.5 * math.log10(range_km * range_km / frac_illum)
    except Exception:
        return None

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

    # Format closest approach time as local time string (display only -- all
    # comparisons use the raw TT below).
    closest_approach_time = "--:--"
    if closest_time is not None:
        try:
            closest_datetime = ts.tt_jd(closest_time).utc_datetime()
            closest_datetime = closest_datetime.astimezone()  # system local tz
            closest_approach_time = closest_datetime.strftime("%H:%M")
        except Exception:
            closest_approach_time = "--:--"

    return {
        'satellite': satellite,
        'name': name,
        'norad_id': norad_id,
        'max_elevation': max_elevation,
        'azimuth_at_max': azimuth_at_max,
        # Raw TT time of the max-elevation sample -- the brightness estimate
        # is evaluated at this instant (culmination) in build_satellite_pass_table.
        'max_el_time_tt': float(trajectory_array[max_alt_idx, 0]),
        'closest_approach_time': closest_approach_time,
        # Raw TT Julian date of the closest approach: time math must use
        # this, never re-parse the local "HH:MM" display string (that
        # round-trip lost the date and produced day=32 ValueErrors when
        # splicing onto month-end "today").
        'closest_time_tt': closest_time,
        'is_visible': np.any(visible_points)
    }

def build_satellite_pass_table(state, elevation_mask_deg=10.0, max_rows=20, ts=None, observer=None):
    """
    Build pass table data from all satellite trajectories.
    Only includes satellites that are currently in view OR will be in view soon (future closest approach).
    Modifies state object directly with sorted satellite pass table.

    When `observer` is given (and the state carries the DE421 ephemeris), each
    entry also gets an estimated visual magnitude at culmination; apogee
    altitude comes from the precomputed state.satellite_apogee metadata.
    """
    pass_entries = []

    # Geocentric sun vector, computed once for the whole table (see
    # _estimate_pass_magnitude for why once is enough).
    eph = getattr(state, 'ephemeris', None)
    r_sun_km = None
    if observer is not None and eph is not None and ts is not None:
        try:
            r_sun_km = eph['earth'].at(ts.now()).observe(eph['sun']).position.km
        except Exception:
            r_sun_km = None
    apogee_map = getattr(state, 'satellite_apogee', None) or {}

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
                    # Include when the closest approach lies 0..2 h in the
                    # future. Compared directly in TT -- re-parsing the local
                    # "HH:MM" display string lost the date and required a
                    # hand-rolled (and month-end-broken) day rollover.
                    closest_tt = pass_data.get('closest_time_tt')
                    if closest_tt is not None and ts is not None:
                        try:
                            diff_seconds = (float(closest_tt) - ts.now().tt) * 86400.0
                            if 0.0 <= diff_seconds <= 7200.0:  # within 2 hours
                                include_in_table = True
                        except Exception:
                            pass

                if include_in_table:
                    pass_data['apogee_km'] = apogee_map.get(sat)
                    # Magnitude key only set when the geometry inputs exist:
                    # a present-but-None value means "eclipsed at culmination"
                    # (drawn as 'ecl'), a missing key means "not computed" ('--').
                    if r_sun_km is not None:
                        t_max = pass_data.get('max_el_time_tt')
                        if t_max is not None:
                            pass_data['magnitude'] = _estimate_pass_magnitude(
                                sat, ts.tt_jd(t_max), observer, r_sun_km)
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
        4: 'closest_approach_time',
        5: 'apogee_km',
        6: 'magnitude'
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
    elif sort_key == 'apogee_km':
        return sorted(pass_table, key=lambda x: x.get('apogee_km') if x.get('apogee_km') is not None else -1.0, reverse=reverse)
    elif sort_key == 'magnitude':
        # None (eclipsed/unknown) sorts as very dim
        return sorted(pass_table, key=lambda x: x.get('magnitude') if x.get('magnitude') is not None else 99.0, reverse=reverse)
    else:
        return pass_table

# Binary cache for parsed launch/tracking trajectories. The ECEF (.txt) parse does
# a per-point Skyfield altaz() + is_sunlit() loop (~1.3s for the current files),
# re-run on every launch and every 15 min refresh even though the files never
# change. We cache the geometry-INDEPENDENT result arrays (absolute time, alt, az,
# range, rates, sunlit) keyed by the source file's size+mtime; pixel coordinates
# and arc segments depend on the display geometry, so they're recomputed cheaply
# on load. Caches live in a sibling ".traj_cache" dir, invisible to the .txt/.csv glob.
LAUNCH_CACHE_DIRNAME = ".traj_cache"


def _launch_cache_path(filepath):
    d = os.path.join(os.path.dirname(filepath), LAUNCH_CACHE_DIRNAME)
    return os.path.join(d, os.path.basename(filepath) + ".npz")


def _load_launch_trajectory_cache(filepath):
    """Return the cached geometry-independent arrays for `filepath` as a dict, or
    None on miss/stale/error."""
    cache_path = _launch_cache_path(filepath)
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as z:
            if str(z['fingerprint']) != _file_fingerprint(filepath):
                return None
            return {k: z[k] for k in ('tt', 'alt', 'az', 'dist', 'az_rate', 'el_rate', 'sunlit', 'launch_tt')}
    except Exception:
        return None


def _save_launch_trajectory_cache(filepath, tt, alt, az, dist, az_rate, el_rate, sunlit, launch_tt):
    """Persist the geometry-independent arrays for `filepath` (best-effort)."""
    try:
        cache_path = _launch_cache_path(filepath)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp.npz"
        np.savez(tmp,
                 fingerprint=np.array(_file_fingerprint(filepath)),
                 tt=np.asarray(tt, dtype=np.float64),
                 alt=np.asarray(alt, dtype=np.float64),
                 az=np.asarray(az, dtype=np.float64),
                 dist=np.asarray(dist, dtype=np.float64),
                 az_rate=np.asarray(az_rate, dtype=np.float64),
                 el_rate=np.asarray(el_rate, dtype=np.float64),
                 sunlit=np.asarray(sunlit),
                 launch_tt=np.array(float(launch_tt)))
        os.replace(tmp, cache_path)
    except Exception:
        pass


def _finalize_tracking_trajectory(tt_array, alt_deg, az_deg, dist_km, az_rates_dps,
                                  el_rates_dps, sun_stat, launch_tt, display):
    """Build the display-ready tracking-trajectory dict (pixel coords + arc
    segments) from the geometry-independent arrays. Cheap and pure-NumPy; run on
    both the fresh-parse and cache-hit paths so the cache is display-independent."""
    if display:
        cx = display.sub_x + display.sub_width // 2
        cy = display.sub_y + display.sub_height // 2
        radius = min(display.sub_width, display.sub_height) // 2 - 50
    else:
        cx, cy, radius = 400, 300, 300

    az_rad = np.radians(az_deg % 360)
    polar_radius = (90 - alt_deg) / 90 * radius
    px_coords = cx + polar_radius * np.sin(az_rad)
    py_coords = cy - polar_radius * np.cos(az_rad)

    trajectory_data = np.column_stack([
        tt_array, alt_deg, az_deg, dist_km, px_coords, py_coords,
        az_rates_dps, el_rates_dps, sun_stat,
    ]).tolist()

    segments = _create_tracking_arc_segments(trajectory_data, launch_tt)
    return {'trajectory': (trajectory_data, np.asarray(tt_array)), 'arcs': segments}


def read_tracking_trajectory(filepath, display=None, update_status_callback=None):
    """
    Read a tracking trajectory file in twilight format with ECEF coordinates.
    Returns a dictionary with trajectory data and arc segments, or None on failure.
    """
    # Fast path: reuse the cached parse if the source file is unchanged. This skips
    # the expensive per-point Skyfield altaz()/is_sunlit() loop below entirely.
    cached = _load_launch_trajectory_cache(filepath)
    if cached is not None:
        try:
            return _finalize_tracking_trajectory(
                cached['tt'], cached['alt'], cached['az'], cached['dist'],
                cached['az_rate'], cached['el_rate'], cached['sunlit'],
                float(cached['launch_tt']), display)
        except Exception:
            pass  # fall through to a full re-parse on any cache problem

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

        # Calculate angular rates from position data (geometry-independent)
        az_rates_dps = np.zeros(len(times_array))
        el_rates_dps = np.zeros(len(times_array))

        # Time step (assume uniform spacing)
        dt_seconds = 1.0  # Default 1 second intervals
        if len(times_rel) > 1:
            dt_seconds = times_rel[1] - times_rel[0]

        # Calculate rates using finite differences (azimuth on the short arc --
        # see the north-crossing note in _compute_one_trajectory)
        for i in range(1, len(times_array) - 1):
            az_rates_dps[i] = _unwrap_az_diff(az_deg[i + 1] - az_deg[i - 1]) / (2 * dt_seconds)
            el_rates_dps[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / (2 * dt_seconds)

        # Handle endpoints
        if len(times_array) > 1:
            az_rates_dps[0] = _unwrap_az_diff(az_deg[1] - az_deg[0]) / dt_seconds
            az_rates_dps[-1] = _unwrap_az_diff(az_deg[-1] - az_deg[-2]) / dt_seconds
            el_rates_dps[0] = (alt_deg[1] - alt_deg[0]) / dt_seconds
            el_rates_dps[-1] = (alt_deg[-1] - alt_deg[-2]) / dt_seconds

        # Persist the geometry-independent parse result so future loads skip the
        # per-point Skyfield work, then build the display-ready trajectory.
        _save_launch_trajectory_cache(filepath, tt_array, alt_deg, az_deg, dist_km,
                                      az_rates_dps, el_rates_dps, sun_stat, launch_time.tt)
        return _finalize_tracking_trajectory(tt_array, alt_deg, az_deg, dist_km,
                                             az_rates_dps, el_rates_dps, sun_stat,
                                             launch_time.tt, display)

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
            # seg_tt appended so renderers can recolor against the tracking
            # clock (see _create_arc_segments_simple).
            segments.append((px[i], py[i], px_next[i], py_next[i], color, times[i]))

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
                    # Use central differences for better accuracy (azimuth on
                    # the short arc -- see the north-crossing note in
                    # _compute_one_trajectory)
                    actual_dt = (times_tt[i+1] - times_tt[i-1]) * 86400  # Time step in seconds
                    if actual_dt > 0:
                        az_rates_dps[i] = _unwrap_az_diff(az_deg[i + 1] - az_deg[i - 1]) / actual_dt
                        el_rates_dps[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / actual_dt

                # Handle endpoints with forward/backward differences
                if len(times_array) > 1:
                    actual_dt_end = (times_tt[1] - times_tt[0]) * 86400
                    actual_dt_start = (times_tt[-1] - times_tt[-2]) * 86400

                    if actual_dt_end > 0:
                        az_rates_dps[0] = _unwrap_az_diff(az_deg[1] - az_deg[0]) / actual_dt_end
                        el_rates_dps[0] = (alt_deg[1] - alt_deg[0]) / actual_dt_end

                    if actual_dt_start > 0:
                        az_rates_dps[-1] = _unwrap_az_diff(az_deg[-1] - az_deg[-2]) / actual_dt_start
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
            # seg_tt appended so renderers can recolor against the tracking
            # clock (see _create_arc_segments_simple).
            segments.append((px[i], py[i], px_next[i], py_next[i], color, times[i]))

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
