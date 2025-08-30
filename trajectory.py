from skyfield.api import wgs84, load
from datetime import datetime, timedelta, timezone
import numpy as np
import math

def precompute_trajectories(satellites, observer, ts, sub_x, sub_y, sub_width, sub_height, satellite_labels, update_status_callback=None, center_time=None, duration_minutes=15):
    if center_time is not None:
        # Use the provided center time and duration
        t0 = ts.utc(center_time - timedelta(minutes=duration_minutes/2))
        t1 = ts.utc(center_time + timedelta(minutes=duration_minutes/2))
    else:
        # Default behavior: current time with 15 minutes on each side
        current_utc = datetime.now(timezone.utc)
        t0 = ts.utc(current_utc - timedelta(minutes=duration_minutes/2))
        t1 = ts.utc(current_utc + timedelta(minutes=duration_minutes/2))
    times = ts.linspace(t0, t1, 1801)  # 1-second intervals over 30 minutes (1800 intervals + 1)
    trajectories = {}
    arc_segments = {}
    cx = sub_x + sub_width // 2
    cy = sub_y + sub_height // 2
    radius = min(sub_width, sub_height) // 2 - 50
    total_sats = len(satellites)
    for i, sat in enumerate(satellites):
        if sat in satellite_labels:
            if update_status_callback and (i + 1) % 1000 == 0:
                update_status_callback(f"Computed traj for {i+1}/{total_sats} sats...")
            difference = sat - observer
            topocentrics = difference.at(times)
            alts, azs, distances = topocentrics.altaz()
            # Precompute pixel coordinates and trajectory
            trajectory = [(t, alt, az, dist,
                          cx + ((90 - alt) / 90 * radius) * math.sin(math.radians(az % 360)),
                          cy - ((90 - alt) / 90 * radius) * math.cos(math.radians(az % 360)))
                          for t, alt, az, dist in zip(times.tt, alts.degrees, azs.degrees, distances.km)]
            trajectories[sat] = trajectory
            # Precompute time array
            times_array = np.array([t for t, _, _, _, _, _ in trajectory])
            trajectories[sat] = (trajectory, times_array)
            # Precompute arc segments with colors
            segments = []
            for i in range(len(trajectory) - 1):
                t0, alt0, az0, dist0, x0, y0 = trajectory[i]
                t1, alt1, az1, dist1, x1, y1 = trajectory[i + 1]
                if alt0 > 0 or alt1 > 0:
                    color = (128, 128, 128)  # Grey for past
                    if t0 > times.tt[0]:  # Future if after start time
                        color = (255, 0, 0)  # Red for future
                        # Simplified sunlit check (precompute based on time order)
                        if i > 0 and trajectory[i-1][0] <= times.tt[0] <= t0:
                            sat_pos = sat.at(ts.tt_jd(t0))
                            sun_pos = load('de421.bsp')['sun'].at(ts.tt_jd(t0))
                            sat_vec = sat_pos.position.km
                            sun_vec = sun_pos.position.km
                            dot_product = np.dot(sat_vec, sun_vec)
                            mag_sat = np.linalg.norm(sat_vec)
                            mag_sun = np.linalg.norm(sun_vec)
                            cos_angle = dot_product / (mag_sat * mag_sun)
                            angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                            if angle_deg < 90:
                                color = (255, 255, 0)  # Yellow for sunlit
                    segments.append((x0, y0, x1, y1, color))
            arc_segments[sat] = segments
    return trajectories, arc_segments

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
