"""
Threaded rendering system for skytracker visualization components.
Provides off-main-thread rendering for polar plots, satellite drawing, and other display elements.
"""

import threading
import time
import pygame

# Local import to avoid scoping issues with time module in thread methods
import time as safe_time
import math
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from tracking_visuals import draw_legend, draw_filters, draw_time_display, draw_satellite_count, draw_scroll_bar, draw_scroll_time_display, PolarPlotMode
from skyfield.api import wgs84, load
from utils import get_altitude_color, draw_button
from transformations import AzAlt2AzEl, apply_rotation_to_az_el, compute_fov_for_camera
import numpy as np
import json
import os

# Font for surface rendering (will be created when needed)
SATELLITE_LABEL_FONT = None

# Cache for rendered satellite label surfaces to avoid expensive re-rendering
SATELLITE_LABEL_CACHE = {}

# Font + label-surface cache for star names (created off the main thread).
STAR_LABEL_FONT = None
STAR_LABEL_CACHE = {}

# Polar plot constants (must match tracking_visuals.py)
POLAR_RADIUS_OFFSET = 50

# ==============================================================================
# SURFACE-BASED DRAWING FUNCTIONS FOR THREADED RENDERING
# ==============================================================================

def draw_hexagon_on_surface(surface, x, y, color, size=5):
    """Draw hexagon shape on a specified surface."""
    points = [(x + size * math.cos(math.radians(60 * i)), y + size * math.sin(math.radians(60 * i))) for i in range(6)]
    pygame.draw.polygon(surface, color, points)

def draw_triangle_on_surface(surface, x, y, color, size=5):
    """Draw triangle shape on a specified surface."""
    points = [(x, y - size), (x - size * math.sin(math.radians(60)), y + size * 0.5), (x + size * math.sin(math.radians(60)), y + size * 0.5)]
    pygame.draw.polygon(surface, color, points)


def draw_dashed_line_on_surface(surface, color, p0, p1, dash=5, gap=4, width=1):
    """Draw a single dashed line segment from p0 to p1."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    step = dash + gap
    n = int(length // step) + 1
    for i in range(n):
        s = i * step
        e = min(s + dash, length)
        sx, sy = x0 + ux * s, y0 + uy * s
        ex, ey = x0 + ux * e, y0 + uy * e
        pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)


def draw_dashed_polygon_on_surface(surface, color, points, dash=5, gap=4, width=1):
    """Draw a closed dashed polygon through points."""
    n = len(points)
    for i in range(n):
        draw_dashed_line_on_surface(surface, color, points[i], points[(i + 1) % n],
                                    dash=dash, gap=gap, width=width)

def _draw_starfield_on_surface(surface, config_state, ts, current_tt, state, cx, cy, radius,
                               elevation_mask, draw_labels=True, publish_positions=True):
    """Draw catalogue stars on the polar plot and publish their screen positions.

    Stars are sized/brightened by visual magnitude. Only the brightest ~5% in view
    (top_label_mask) and the hovered star are labelled (full-screen only). Publishes
    state.star_screen_positions (for main-thread hover hit-testing) and
    state.starfield_cutoff_mag (the faintest rendered magnitude, shown in the UI).
    publish_positions=False (the small joystick quadrant) leaves the hover data alone.
    Wrapped by the caller in try/except so a catalogue hiccup never kills rendering.
    """
    global STAR_LABEL_FONT, STAR_LABEL_CACHE

    from star_catalog import get_catalog
    catalog = get_catalog(ephemeris=getattr(state, 'ephemeris', None))

    lat = float(config_state.lat_str or 0.0)
    lon = float(config_state.lon_str or 0.0)
    elev = float(config_state.alt_str or 0.0)
    max_count = int(getattr(config_state, 'max_rendered_star_count', 2000))
    limiting_mag = float(getattr(config_state, 'star_limiting_magnitude', 6.5))

    res = catalog.current_altaz(lat, lon, elev, ts, current_tt,
                                elevation_mask=elevation_mask,
                                max_count=max_count, limiting_mag=limiting_mag)
    if publish_positions:
        state.starfield_cutoff_mag = res['cutoff_mag']

    n = res['n_visible']
    if n == 0:
        if publish_positions:
            state.star_screen_positions = []
        return

    az = res['az']
    el = res['el']
    mag = res['mag']
    hip = res['hip']
    top_mask = res['top_label_mask']

    # Vectorised polar projection: r=(90-el)/90*radius, az 0=N=up, east=right.
    az_rad = np.radians(az)
    r = (90.0 - el) / 90.0 * radius
    sx = cx + r * np.sin(az_rad)
    sy = cy - r * np.cos(az_rad)

    # Magnitude -> dot radius and brightness. Brightest stars are bigger/whiter.
    bright = float(mag.min())
    span = max(0.5, limiting_mag - bright)
    norm = np.clip((mag - bright) / span, 0.0, 1.0)  # 0 = brightest, 1 = faintest
    dot_r = np.clip(2.6 - 2.0 * norm, 0.6, 2.6)
    intensity = np.clip(255.0 - 150.0 * norm, 90.0, 255.0).astype(int)

    if STAR_LABEL_FONT is None:
        pygame.font.init()
        STAR_LABEL_FONT = pygame.font.Font(None, 14)

    hovered = getattr(state, 'hovered_star', None)
    positions = []
    sx_i = sx.astype(int)
    sy_i = sy.astype(int)
    for i in range(n):
        col = (intensity[i], intensity[i], min(255, intensity[i] + 10))
        ri = int(round(dot_r[i]))
        if ri <= 0:
            surface.set_at((sx_i[i], sy_i[i]), col)
        else:
            pygame.draw.circle(surface, col, (sx_i[i], sy_i[i]), ri)

        h = int(hip[i])
        positions.append((sx_i[i], sy_i[i], h))

        if draw_labels and (top_mask[i] or h == hovered):
            name = catalog.name_for(h)
            label = STAR_LABEL_CACHE.get(name)
            if label is None:
                try:
                    label = STAR_LABEL_FONT.render(name, True, (170, 190, 220))
                    STAR_LABEL_CACHE[name] = label
                except pygame.error:
                    label = None
            if label is not None:
                surface.blit(label, (sx_i[i] + 4, sy_i[i] - 6))

    if publish_positions:
        state.star_screen_positions = positions


def draw_polar_plot_on_surface(surface, config_state, ts, current_tt, state, display_bounds, mode=PolarPlotMode.FULL_SCREEN, full_screen_bounds=None):
    """
    Draw polar plot on a specified surface with given bounds.
    surface: pygame.Surface to draw on
    display_bounds: dict with sub_x, sub_y, sub_width, sub_height
    """
    elevation_mask = float(config_state.elevation_mask_str or 0)

    # For surface rendering, cx and cy are relative to the surface (0,0 is top-left of surface)
    # Since we render to a surface that will be blitted to the correct screen position,
    # we use center coordinates relative to the surface
    surface_width, surface_height = surface.get_size()
    if mode == PolarPlotMode.FULL_SCREEN:
        cx = surface_width // 2
        cy = surface_height // 2
    elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
        # Upper right quadrant mode - center is adjusted for the quadrant
        cx = surface_width // 2
        cy = surface_height // 2

    radius = min(surface_width, surface_height) // 2 - 50

    # Draw elevation mask circle first (thick red)
    mask_radius = (90 - elevation_mask) / 90 * radius
    pygame.draw.circle(surface, (255, 0, 0), (cx, cy), mask_radius, 2)

    # Draw elevation circles (subtle gray)
    for el in [30, 60]:
        r = (90 - el) / 90 * radius
        pygame.draw.circle(surface, (50, 50, 50), (cx, cy), int(r), 1)

    # Draw azimuth lines (subtle gray)
    for az_deg in range(0, 360, 30):
        az_rad = math.radians(az_deg)
        x1 = cx + radius * math.sin(az_rad)
        y1 = cy - radius * math.cos(az_rad)
        pygame.draw.line(surface, (50, 50, 50), (cx, cy), (x1, y1), 1)

    # Draw horizon circle LAST so it appears on top (bright white, thicker)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius, 3)

    # Draw center dot explicitly to ensure it's visible over grid lines
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3, 0)

    # ---- Cardinal directions + numeric axis labels --------------------------
    # Restored after a prior iteration dropped them. Drawn off the main thread,
    # so we create local fonts rather than relying on display fonts.
    pygame.font.init()
    _dir_font = pygame.font.Font(None, 20)
    _num_font = pygame.font.Font(None, 15)

    def _blit_centered(text, color, px, py, font):
        s = font.render(text, True, color)
        surface.blit(s, (int(px - s.get_width() / 2), int(py - s.get_height() / 2)))

    # Numeric azimuth ticks every 30 deg just outside the horizon ring (the four
    # cardinals get letters instead). Azimuth: 0 = N = up, 90 = E = right.
    for az_deg in range(0, 360, 30):
        if az_deg in (0, 90, 180, 270):
            continue
        az_rad = math.radians(az_deg)
        lx = cx + (radius + 13) * math.sin(az_rad)
        ly = cy - (radius + 13) * math.cos(az_rad)
        _blit_centered(f"{az_deg}", (140, 140, 150), lx, ly, _num_font)

    # Cardinal direction letters, further out; N highlighted.
    for az_deg, lbl, col in ((0, "N", (255, 130, 130)), (90, "E", (225, 225, 235)),
                             (180, "S", (225, 225, 235)), (270, "W", (225, 225, 235))):
        az_rad = math.radians(az_deg)
        lx = cx + (radius + 18) * math.sin(az_rad)
        ly = cy - (radius + 18) * math.cos(az_rad)
        _blit_centered(lbl, col, lx, ly, _dir_font)

    # Elevation ring labels (30 and 60 deg), placed just right of the N-S spoke.
    for el in (30, 60):
        r = (90 - el) / 90 * radius
        _blit_centered(f"{el}°", (120, 170, 120), cx + 14, cy - r, _num_font)

    # Catalogue starfield, drawn under satellites/FOV. Rendered in both the full-screen
    # plot and the smaller joystick quadrant; labels + hover hit-testing only full-screen.
    if getattr(config_state, 'starfield_enabled', True):
        full_screen = (mode == PolarPlotMode.FULL_SCREEN)
        try:
            _draw_starfield_on_surface(surface, config_state, ts, current_tt, state,
                                       cx, cy, radius, elevation_mask,
                                       draw_labels=full_screen, publish_positions=full_screen)
        except Exception as e:
            print(f"Starfield render error: {e}")

        # Limiting-magnitude readout just inside the horizon ring (full-screen only).
        if full_screen:
            cutoff = getattr(state, 'starfield_cutoff_mag', None)
            if cutoff is not None and not math.isnan(cutoff):
                n_stars = len(getattr(state, 'star_screen_positions', []) or [])
                txt = f"lim mag {cutoff:.1f}  ({n_stars} stars)"
                _blit_centered(txt, (150, 170, 200), cx, cy + radius - 8, _num_font)
    elif mode == PolarPlotMode.FULL_SCREEN:
        state.star_screen_positions = []

    # Plate-solve "solved pointing" marker (cyan cross at the solved boresight az/el).
    if mode == PolarPlotMode.FULL_SCREEN:
        ls = getattr(state, 'last_solve', None)
        if ls is not None and ls.get('az') is not None and ls.get('el') is not None and ls['el'] >= 0:
            saz_rad = math.radians(ls['az'])
            sr = (90.0 - ls['el']) / 90.0 * radius
            px = int(cx + sr * math.sin(saz_rad))
            py = int(cy - sr * math.cos(saz_rad))
            cyan = (0, 230, 230)
            pygame.draw.line(surface, cyan, (px - 7, py), (px + 7, py), 1)
            pygame.draw.line(surface, cyan, (px, py - 7), (px, py + 7), 1)
            pygame.draw.circle(surface, cyan, (px, py), 9, 1)
            r = ls.get('result')
            tag = f"solved {r.n_matches}m" if r is not None else "solved"
            _blit_centered(tag, cyan, px, py - 16, _num_font)

    # Draw precomputed arc segments for selected satellite
    if state.selected_satellite and state.tle_loaded and state.selected_satellite in state.satellite_arc_segments:
    # Draw arcs with sunlit detection (preferred method)
        if state.selected_satellite in state.satellite_trajectories and state.satellite_trajectories[state.selected_satellite]:
            trajectory_data, times_array = state.satellite_trajectories[state.selected_satellite]

            # Use cached sunlit status for performance optimization
            sat_name = state.selected_satellite.name
            if sat_name not in state.sunlit_status_cache:
                # Compute sunlit status and cache it
                try:
                    # Create Skyfield Time objects for current trajectory times
                    state.sunlit_status_cache[sat_name] = []
                    for tt_time in times_array:
                        state.sunlit_status_cache[sat_name].append(state.selected_satellite.at(ts.tt(jd=tt_time)).is_sunlit(state.ephemeris))

                except Exception as e:
                    # Fallback if sunlit calculation fails
                    state.sunlit_status_cache[sat_name] = [False] * len(times_array)
                    print('!!!!!! SUNLIT CACHING FAILED !!!!!!!')

            # Draw segments with sunlit-aware colors (same logic as original)
            for i in range(len(trajectory_data) - 1):
                x0, y0, x1, y1 = trajectory_data[i][4], trajectory_data[i][5], trajectory_data[i + 1][4], trajectory_data[i + 1][5]
                alt = trajectory_data[i][1]

                if alt > 0:  # Above horizon
                    # Determine if this time point is in the future or past
                    is_future = times_array[i] > current_tt if current_tt else False
                    is_sunlit = state.sunlit_status_cache[sat_name][i] if i < len(state.sunlit_status_cache[sat_name]) else False

                    # Apply coloring logic from original code:
                    # Future trajectories: Yellow if sunlit, Red if shadowed
                    # Past trajectories: Gray
                    if is_future:
                        color = (255, 255, 0) if is_sunlit else (255, 0, 0)  # Yellow for sunlit future, red for shadowed
                    else:
                        color = (128, 128, 128)  # Grey for past

                    # Apply coordinate transformation same as arc segments
                    if mode == PolarPlotMode.FULL_SCREEN:
                        # screen to surface: subtract display origin
                        draw_x0 = x0 - display_bounds['sub_x']
                        draw_y0 = y0 - display_bounds['sub_y']
                        draw_x1 = x1 - display_bounds['sub_x']
                        draw_y1 = y1 - display_bounds['sub_y']
                    elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
                        # Trajectory points are in full screen coordinates
                        # Convert to quadrant surface coordinates (scaled and centered)
                        # Use full_screen_bounds if provided for correct center calculation
                        if full_screen_bounds:
                            full_center_x = full_screen_bounds['sub_x'] + full_screen_bounds['sub_width'] // 2
                            full_center_y = full_screen_bounds['sub_y'] + full_screen_bounds['sub_height'] // 2
                        else:
                            full_center_x = display_bounds['sub_x'] + display_bounds['sub_width'] // 2
                            full_center_y = display_bounds['sub_y'] + display_bounds['sub_height'] // 2

                        quadrant_surface_width = surface.get_width()
                        quadrant_surface_height = surface.get_height()

                        # Transform relative to full screen center
                        rel_x0 = x0 - full_center_x
                        rel_y0 = y0 - full_center_y
                        rel_x1 = x1 - full_center_x
                        rel_y1 = y1 - full_center_y

                        # Apply quadrant scaling factor (0.45 as in original display.py)
                        scale_factor = 0.45
                        rel_x0 *= scale_factor
                        rel_y0 *= scale_factor
                        rel_x1 *= scale_factor
                        rel_y1 *= scale_factor

                        # Center in quadrant surface
                        draw_x0 = quadrant_surface_width // 2 + rel_x0
                        draw_y0 = quadrant_surface_height // 2 + rel_y0
                        draw_x1 = quadrant_surface_width // 2 + rel_x1
                        draw_y1 = quadrant_surface_height // 2 + rel_y1

                    pygame.draw.line(surface, color, (draw_x0, draw_y0), (draw_x1, draw_y1), 1)

        else:
            # Fallback: draw basic arc segments without sunlit detection when no trajectory data available
            for x0, y0, x1, y1, color in state.satellite_arc_segments[state.selected_satellite]:
                # Apply coordinate transformation to surface coordinates
                if mode == PolarPlotMode.FULL_SCREEN:
                    # screen to surface: subtract display origin
                    draw_x0 = x0 - display_bounds['sub_x']
                    draw_y0 = y0 - display_bounds['sub_y']
                    draw_x1 = x1 - display_bounds['sub_x']
                    draw_y1 = y1 - display_bounds['sub_y']
                elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
                    # Arc segments are in full screen coordinates
                    # Convert to quadrant surface coordinates (scaled and centered)
                    # Use full_screen_bounds if provided for correct center calculation
                    if full_screen_bounds:
                        full_center_x = full_screen_bounds['sub_x'] + full_screen_bounds['sub_width'] // 2
                        full_center_y = full_screen_bounds['sub_y'] + full_screen_bounds['sub_height'] // 2
                    else:
                        full_center_x = display_bounds['sub_x'] + display_bounds['sub_width'] // 2
                        full_center_y = display_bounds['sub_y'] + display_bounds['sub_height'] // 2

                    quadrant_surface_width = surface.get_width()
                    quadrant_surface_height = surface.get_height()

                    # Transform relative to full screen center
                    rel_x0 = x0 - full_center_x
                    rel_y0 = y0 - full_center_y
                    rel_x1 = x1 - full_center_x
                    rel_y1 = y1 - full_center_y

                    # Apply quadrant scaling factor (0.45 as in original display.py)
                    scale_factor = 0.45
                    rel_x0 *= scale_factor
                    rel_y0 *= scale_factor
                    rel_x1 *= scale_factor
                    rel_y1 *= scale_factor

                    # Center in quadrant surface
                    draw_x0 = quadrant_surface_width // 2 + rel_x0
                    draw_y0 = quadrant_surface_height // 2 + rel_y0
                    draw_x1 = quadrant_surface_width // 2 + rel_x1
                    draw_y1 = quadrant_surface_height // 2 + rel_y1

                pygame.draw.line(surface, color, (draw_x0, draw_y0), (draw_x1, draw_y1), 1)

def draw_camera_fov_details_on_surface(surface, state, display_bounds, y_offset, mode=PolarPlotMode.FULL_SCREEN, config_state=None):
    """
    Draw camera FOV details panels on a specified surface with given bounds.
    Each camera gets its own info pane positioned below the satellite details pane.
    """
    global SATELLITE_LABEL_FONT

    if not hasattr(state, 'camera_fov_data') or not state.camera_fov_data:
        return

    surface_width, surface_height = surface.get_size()
    current_y = y_offset + 10  # Start below satellite details with some spacing

    for i, fov_data in enumerate(state.camera_fov_data):
        # Use the same panel positioning as tracking visuals mode for consistency
        if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
            # Adjust for quadrant surface bounds
            panel_x = surface_width - 190
            panel_y = current_y
            panel_width = 170
            panel_height = 120
        else:
            # Full screen mode - same positioning as tracking visuals
            panel_x = surface_width - 190
            panel_y = current_y
            panel_width = 170
            panel_height = 120

        # Create font if needed (always use same font size as tracking visuals)
        if SATELLITE_LABEL_FONT is None:
            SATELLITE_LABEL_FONT = pygame.font.Font(None, 12)  # Same as tracking visuals

        # Use consistent font size for both modes (same as tracking visuals)
        dynamic_font = SATELLITE_LABEL_FONT

        # Draw panel background
        pygame.draw.rect(surface, (50, 50, 50), (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(surface, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2)

        # Panel title
        camera_num = fov_data.get('camera_id', i) + 1
        title = f"Camera {camera_num} FOV"
        title_surface = dynamic_font.render(title, True, (255, 255, 255))
        surface.blit(title_surface, (panel_x + 5, panel_y + 5))

        # FOV details
        details = [
            f"Az: {fov_data.get('az', 0.0):.2f}°",
            f"El: {fov_data.get('el', 0.0):.2f}°",
            f"FOV Width: {fov_data.get('fov_width_deg', 1.0):.3f}°",
            f"FOV Height: {fov_data.get('fov_height_deg', 1.0):.3f}°",
            f"Rotation: {fov_data.get('rotation', 0.0):.1f}°",
            f"Spot Size: {fov_data.get('spot_size_arcsec_per_pixel', 0.5):.2f}\"/pix",
        ]

        # Draw detail lines
        y_offset_line = panel_y + 25
        for line_no, line in enumerate(details):
            # Color-coded display
            if line_no == 0:  # Azimuth
                color = (200, 200, 255)  # Light blue
            elif line_no == 1:  # Elevation
                color = (200, 255, 200)  # Light green
            elif line_no == 5:  # Spot size
                color = (255, 255, 200)  # Light yellow
            else:  # FOV parameters
                color = (255, 255, 255)  # White

            line_surface = dynamic_font.render(line, True, color)
            surface.blit(line_surface, (panel_x + 5, y_offset_line))
            y_offset_line += 15

        # Update vertical position for next camera panel
        current_y += panel_height + 0  # Panel height + spacing


def draw_details_on_surface(surface, state, display_bounds, mode=PolarPlotMode.FULL_SCREEN, config_state=None):
    """
    Draw satellite details panel on a specified surface with given bounds.
    Surface-based version of draw_details from tracking_visuals.py
    """
    global SATELLITE_LABEL_FONT

    if not (state.hovered_satellite or state.selected_satellite):
        return

    # Get the satellite to display details for
    sat = state.selected_satellite if state.selected_satellite else state.hovered_satellite

    # Get satellite data the same way as the main draw_details function
    epoch_dt = sat.epoch.utc_datetime().strftime("%Y-%m-%d %H:%M:%S")
    details = [
        f"NORAD ID: {sat.model.satnum_str}",
        f"Name: {sat.name.strip()}",
        f"Int. Designator: {sat.model.intldesg}",
        f"Epoch: {epoch_dt}",
        f"Inclination (deg): {math.degrees(sat.model.inclo):.2f}",
        f"RAAN (deg): {math.degrees(sat.model.nodeo):.2f}",
        f"Arg. of Perigee (deg): {math.degrees(sat.model.argpo):.2f}",
        f"Mean Anomaly (deg): {math.degrees(sat.model.mo):.2f}",
        f"Mean Motion (rev/day): {sat.model.no_kozai:.4f}",
        f"Rev Number: {sat.model.revnum}",
    ]

    # Add altitude information
    dist = None
    if state.satellite_positions and sat in state.satellite_positions:
        dist = state.satellite_positions[sat][3]  # Slant range from position data

    if ((state.selected_satellite or state.hovered_satellite) and
        hasattr(state, 'satellite_perigee') and hasattr(state, 'satellite_apogee') and
        state.satellite_perigee and state.satellite_apogee):
        sat_perigee = state.satellite_perigee[sat] if sat in state.satellite_perigee else None
        sat_apogee = state.satellite_apogee[sat] if sat in state.satellite_apogee else None
        if sat_perigee is not None and sat_apogee is not None:
            details.extend([
                f"Apogee Altitude (km): {sat_apogee:.1f}",
                f"Perigee Altitude (km): {sat_perigee:.1f}"
            ])

    if dist is not None:
        details.append(f"Slant Range (km): {dist:.1f}")

    # Position the details panel based on mode and surface bounds
    surface_width, surface_height = surface.get_size()

    # Use the same panel size and font size as tracking visuals mode for consistency
    if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
        # In quadrant mode, position relative to the quadrant surface bounds
        panel_x = surface_width - 190  # Right side of quadrant surface (same as full-screen)
        panel_y = 20  # Top margin
        panel_width = 170  # Same width as tracking visuals mode
        panel_height = min(250, surface_height - 30)  # Same height as tracking visuals mode
    else:
        # Full screen mode - original positioning relative to surface
        panel_x = surface_width - 190
        panel_y = 20
        panel_width = 170
        panel_height = min(250, surface_height - 30)

    # Create font if needed (always use same font size as tracking visuals)
    if SATELLITE_LABEL_FONT is None:
        SATELLITE_LABEL_FONT = pygame.font.Font(None, 12)  # Same as tracking visuals

    # Use consistent font size for both modes (same as tracking visuals)
    dynamic_font = pygame.font.Font(None, 12)  # Same as tracking visuals

    # Draw panel background
    pygame.draw.rect(surface, (50, 50, 50), (panel_x, panel_y, panel_width, panel_height))  # Dark grey background
    pygame.draw.rect(surface, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2)  # Black border

    # Set consistent line height and padding (same as tracking visuals)
    line_height = 15  # Same as tracking visuals
    padding = 5  # Same as tracking visuals

    # Draw detail lines
    y_offset = panel_y + padding
    for i, line in enumerate(details):
        text_surface = dynamic_font.render(line, True, (255, 255, 255))
        surface.blit(text_surface, (panel_x + padding, y_offset))
        y_offset += line_height

        # Prevent drawing below panel
        if y_offset > panel_y + panel_height - padding:
            break

FOV_MAGNIFICATION = 10  # dotted boxes are drawn at this magnification for clarity


def draw_fov_on_surface(surface, state, cx, cy, display_bounds, mode=PolarPlotMode.FULL_SCREEN, joystick_mode_state=None):
    """Draw camera FOV boxes on the polar plot surface.

    Each camera's true FOV box is drawn solid, plus a FOV_MAGNIFICATION-times
    (10x) magnified copy in dotted lines so the small finder/imager fields are
    actually visible and their position/orientation can be read at a glance. A
    legend notes the 10x exaggeration."""
    if not hasattr(state, 'camera_fov_data') or not state.camera_fov_data:
        return

    surface_width, surface_height = surface.get_size()
    drew_any = False

    # Center coordinates for polar plot (relative to surface)
    center_x = surface_width // 2
    center_y = surface_height // 2
    radius = min(surface_width, surface_height) // 2 - POLAR_RADIUS_OFFSET

    for fov_data in state.camera_fov_data:
        camera_id = fov_data.get('camera_id', 0)
        az = fov_data.get('az', 0.0)
        el = fov_data.get('el', 0.0)
        width_deg = fov_data.get('fov_width_deg', 1.0)
        height_deg = fov_data.get('fov_height_deg', 1.0)
        rotation = fov_data.get('rotation', 0.0)
        color = fov_data.get('color', (255, 0, 0))

        # Normalize elevation to [-180, 180] so mount-angle wrap doesn't produce
        # a garbage value (e.g. ALT 351° -> el = 90 - 351 = -261°, really -9° past
        # the horizon i.e. 99° elevation pointing past zenith).
        el = (el + 180) % 360 - 180

        # Reflect a past-zenith pointing back onto the visible hemisphere: el just
        # over 90° means pointing slightly past the zenith toward the opposite
        # azimuth, which on the polar plot is el = 180 - el at az + 180.
        if el > 90:
            el = 180 - el
            az += 180
        elif el < -90:
            el = -180 - el
            az += 180

        # Skip if below horizon or invalid
        if el < 0:
            continue

        # Convert az/el to polar plot coordinates
        # Azimuth: 0° = north = top, 90° = east = right
        # Elevation: 90° = zenith = center, 0° = horizon = edge
        az_rad = np.radians(az)
        r = (90 - el) / 90 * radius
        x = center_x + r * np.sin(az_rad)
        y = center_y - r * np.cos(az_rad)

        # FOV box dimensions in pixels
        # Approximate: 1 degree ≈ radius * np.pi/180 pixels at horizon
        # But FOV boxes are small, so scale appropriately
        fov_scale = radius * np.pi / 360  # Base scale for FOV rendering
        half_width = width_deg * fov_scale / 2
        half_height = height_deg * fov_scale / 2

        # Determine rotation angle (camera rotation)
        # Convert position from north=0 to screen coordinates
        position_angle = (az - 90) % 360  # Adjust for screen coordinate system
        total_rotation = position_angle + rotation

        # Draw FOV rectangle as rotated polygon
        # Define rectangle corners in local coordinates (not rotated)
        corners_local = np.array([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])

        # Apply rotation matrix
        rot_angle = np.radians(total_rotation)
        rot_matrix = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)]
        ])

        corners_rotated = corners_local @ rot_matrix.T

        # Convert to screen coordinates (center at x,y)
        corners_screen = corners_rotated + np.array([x, y])

        # Draw the FOV box (use pygame polygon)
        corners_list = [(int(c[0]), int(c[1])) for c in corners_screen]

        # Close the polygon for pygame.draw.polygon
        if len(corners_list) > 2:
            pygame.draw.polygon(surface, color, corners_list, 2)
            drew_any = True

            # Draw rotation indicator line from center
            indicator_length = half_width * 0.3
            indicator_angle = np.radians(total_rotation)
            indicator_dx = indicator_length * np.cos(indicator_angle)
            indicator_dy = indicator_length * np.sin(indicator_angle)
            indicator_end_x = x + indicator_dx
            indicator_end_y = y + indicator_dy
            pygame.draw.line(surface, color, (int(x), int(y)),
                           (int(indicator_end_x), int(indicator_end_y)), 1)

            # 10x-magnified dotted copy: same center/rotation, corners scaled out
            # so the (tiny) true FOV is legible. Lighter shade to read as a hint.
            mag_corners = corners_local * FOV_MAGNIFICATION @ rot_matrix.T + np.array([x, y])
            mag_list = [(int(c[0]), int(c[1])) for c in mag_corners]
            mag_color = tuple(min(255, int(ch * 0.6 + 90)) for ch in color)
            draw_dashed_polygon_on_surface(surface, mag_color, mag_list, dash=5, gap=4, width=1)

        # Draw corner markers for better visibility
        for corner in corners_screen:
            cx, cy = corner
            pygame.draw.circle(surface, color, (int(cx), int(cy)), 2, 1)

    # Legend: note the dotted boxes are magnified (drawn once, bottom-left).
    if drew_any:
        pygame.font.init()
        legend_font = pygame.font.Font(None, 15)
        legend = legend_font.render(f"- - -  {FOV_MAGNIFICATION}x FOV", True, (200, 200, 210))
        surface.blit(legend, (10, surface_height - 18))

def draw_satellites_on_surface(surface, state, cx, cy, display_bounds, mode=PolarPlotMode.FULL_SCREEN, config_state=None):
    """Draw satellites on a specified surface with given bounds."""

    surface_center_x = surface.get_width() // 2
    surface_center_y = surface.get_height() // 2

    # If a satellite is selected, only draw the selected satellite for performance
    # Otherwise draw all satellites in view
    satellites_to_draw = {}
    if state.selected_satellite and state.selected_satellite in state.satellite_positions:
        # Only draw selected satellite when zoomed/focused
        satellites_to_draw = {state.selected_satellite: state.satellite_positions[state.selected_satellite]}
    else:
        # Draw all satellites when no selection (overview mode)
        satellites_to_draw = state.satellite_positions.copy()

    # Quick performance check: ensure we have valid data before attempting to render
    if not satellites_to_draw:
        return

    # While a rocket is actively launched, suppress satellite labels: the launch
    # is the focus and the dense label cloud is just visual noise (the dots stay).
    hide_labels = bool(getattr(state, 'launch_launched', False) and getattr(state, 'selected_launch', None))

    for sat, (px, py, alt, _) in satellites_to_draw.items():
        # Satellite positions are absolute screen coordinates
        # Convert to surface coordinates based on mode

        if mode == PolarPlotMode.FULL_SCREEN:
            # Direct transformation to surface coordinates (main area)
            draw_px = px - display_bounds['sub_x']
            draw_py = py - display_bounds['sub_y']
        elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
            # Transform through the same process as original quadrant code:
            # 1. Transform to full screen viewport coordinates
            # 2. Apply scaling relative to full screen center
            # 3. Place in surface coordinates

            # Step 1: Get coordinates relative to full screen center
            full_center_x = display_bounds['sub_x'] + display_bounds['sub_width'] // 2
            full_center_y = display_bounds['sub_y'] + display_bounds['sub_height'] // 2

            rel_x = px - full_center_x
            rel_y = py - full_center_y

            # Step 2: Apply scaling same as original code (0.45)
            scale_factor = 0.45
            rel_x *= scale_factor
            rel_y *= scale_factor

            # Step 3: Place relative to surface center (quadrant center)
            draw_px = surface_center_x + rel_x
            draw_py = surface_center_y + rel_y

        # Draw satellite shape based on altitude
        mean_altitude = state.satellite_mean_altitudes.get(sat, 0.0)
        eccentricity = sat.model.ecco

        if 2000 < mean_altitude <= 35786:  # MEO
            color = (255, 165, 0)  # Orange
            draw_hexagon_on_surface(surface, draw_px, draw_py, color)
        elif abs(mean_altitude - 35786) <= 1000:  # GEO or nearby
            color = (128, 0, 128)  # Purple
            draw_triangle_on_surface(surface, draw_px, draw_py, color)
        else:  # LEO (0-2000 km)
            color = get_altitude_color(mean_altitude) or (0, 255, 0)
            if eccentricity > 0.01:
                width = 6
                height = 3
                if cx and cy:
                    angle = math.degrees(math.atan2(draw_py - cy, draw_px - cx))
                    oval_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                    rotated_oval = pygame.transform.rotate(oval_surface, angle)
                    rotated_rect = rotated_oval.get_rect(center=(draw_px, draw_py))
                    surface.blit(rotated_oval, rotated_rect.topleft)
            else:
                pygame.draw.circle(surface, color, (draw_px, draw_py), 3)

        if sat == state.hovered_satellite or sat == state.selected_satellite:
            pygame.draw.circle(surface, (255, 255, 0), (draw_px, draw_py), 5, 1)

        # Draw satellite labels efficiently with caching (hidden during a launch)
        if not hide_labels and hasattr(state, 'satellite_labels') and sat in state.satellite_labels:
            global SATELLITE_LABEL_FONT, SATELLITE_LABEL_CACHE

            # Initialize font if not already done
            if SATELLITE_LABEL_FONT is None:
                SATELLITE_LABEL_FONT = pygame.font.Font(None, 12)

            # Clip satellite name to avoid overcrowding
            sat_name = sat.name.strip()[:10]
            if len(sat.name.strip()) > 10:
                sat_name += "..."

            # Check if there's a label surface for this satellite name
            if sat_name in SATELLITE_LABEL_CACHE:
                label_surface = SATELLITE_LABEL_CACHE[sat_name]
            else:
                # Create and cache the label surface
                try:
                    label_surface = SATELLITE_LABEL_FONT.render(sat_name, True, (255, 255, 255))
                    SATELLITE_LABEL_CACHE[sat_name] = label_surface
                except pygame.error:
                    # If font still not working, skip labels
                    label_surface = None

            # Only render labels that are within surface bounds (performance optimization)
            if label_surface:
                # Only render labels for selected/selected satellites to reduce rendering overhead
                surface.blit(label_surface, (int(draw_px + 5), int(draw_py)))

def draw_launch_trajectory_on_surface(surface, state, current_tt, display_bounds, mode=PolarPlotMode.FULL_SCREEN):
    """Draw selected launch trajectory on surface."""
    if not hasattr(state, 'selected_launch') or not state.selected_launch:
        return

    launch_name = state.selected_launch
    if launch_name not in state.launch_trajectories:
        return

    # Get arc segments for the launch trajectory
    arc_key = launch_name + '_arcs'
    if arc_key not in state.launch_trajectories:
        return

    arc_segments = state.launch_trajectories[arc_key]

    # Draw arc segments
    for start_x, start_y, end_x, end_y, color in arc_segments:
        # Apply coordinate transformation based on mode
        if mode == PolarPlotMode.FULL_SCREEN:
            # screen to surface: subtract display origin
            draw_x0 = start_x - display_bounds['sub_x']
            draw_y0 = start_y - display_bounds['sub_y']
            draw_x1 = end_x - display_bounds['sub_x']
            draw_y1 = end_y - display_bounds['sub_y']
        elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
            # Transform through the same process as original quadrant code
            full_center_x = display_bounds['sub_x'] + display_bounds['sub_width'] // 2
            full_center_y = display_bounds['sub_y'] + display_bounds['sub_height'] // 2

            # Transform relative to full screen center
            rel_x0 = start_x - full_center_x
            rel_y0 = start_y - full_center_y
            rel_x1 = end_x - full_center_x
            rel_y1 = end_y - full_center_y

            # Apply quadrant scaling factor (0.45 as in original display.py)
            scale_factor = 0.45
            rel_x0 *= scale_factor
            rel_y0 *= scale_factor
            rel_x1 *= scale_factor
            rel_y1 *= scale_factor

            # Center in quadrant surface
            surface_center_x = surface.get_width() // 2
            surface_center_y = surface.get_height() // 2
            draw_x0 = surface_center_x + rel_x0
            draw_y0 = surface_center_y + rel_y0
            draw_x1 = surface_center_x + rel_x1
            draw_y1 = surface_center_y + rel_y1

        # Draw the arc segment
        pygame.draw.line(surface, color, (draw_x0, draw_y0), (draw_x1, draw_y1), 2)  # Thicker line for launches

def draw_launch_position_on_surface(surface, state, cx, cy, display_bounds, mode=PolarPlotMode.FULL_SCREEN):
    """Draw selected launch position marker on surface."""
    if not hasattr(state, 'selected_launch') or not state.selected_launch:
        return

    launch_name = state.selected_launch
    if not hasattr(state, 'launch_positions') or launch_name not in state.launch_positions:
        return

    px, py, alt, dist = state.launch_positions[launch_name]

    # Convert to surface coordinates
    if mode == PolarPlotMode.FULL_SCREEN:
        draw_px = px - display_bounds['sub_x']
        draw_py = py - display_bounds['sub_y']
    elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
        full_center_x = display_bounds['sub_x'] + display_bounds['sub_width'] // 2
        full_center_y = display_bounds['sub_y'] + display_bounds['sub_height'] // 2

        rel_x = px - full_center_x
        rel_y = py - full_center_y

        scale_factor = 0.45
        rel_x *= scale_factor
        rel_y *= scale_factor

        surface_center_x = surface.get_width() // 2
        surface_center_y = surface.get_height() // 2
        draw_px = surface_center_x + rel_x
        draw_py = surface_center_y + rel_y

    # Draw launch position with distinctive cyan color (cross marker)
    marker_size = 6
    # Draw cross
    pygame.draw.line(surface, (0, 255, 255), (draw_px - marker_size, draw_py), (draw_px + marker_size, draw_py), 3)
    pygame.draw.line(surface, (0, 255, 255), (draw_px, draw_py - marker_size), (draw_px, draw_py + marker_size), 3)
    # Draw circle outline
    pygame.draw.circle(surface, (0, 255, 255), (int(draw_px), int(draw_py)), marker_size + 2, 2)

class RenderMode(Enum):
    FULL_SCREEN = "full_screen"
    UPPER_RIGHT_QUADRANT = "upper_right_quadrant"

class VisualizationRenderingThread(threading.Thread):
    """Base class for visualization rendering threads following CameraThread pattern."""

    def __init__(self, display, config_state, tracking_vis_state, mode=RenderMode.FULL_SCREEN, target_fps=10):
        super().__init__()
        self.display = display
        self.config_state = config_state
        self.tracking_vis_state = tracking_vis_state
        self.mode = mode
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0

        # Pre-allocate rendering surface
        if mode == RenderMode.FULL_SCREEN:
            self.surface = pygame.Surface((display.sub_width, display.sub_height))
        elif mode == RenderMode.UPPER_RIGHT_QUADRANT:
            self.surface = pygame.Surface((display.sub_width // 2, display.sub_height // 2))

        self.latest_surface = None
        self.surface_lock = threading.Condition()  # guards the latest_surface handoff
        self.running = True
        self.last_render_time = 0
        self.render_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()
        self.ts = None  # Shared timescale reference

        print(f"VisualizationRenderingThread initialized for mode: {mode.value}")

    def get_latest_surface(self):
        """Get the most recent rendered surface - thread-safe."""
        if hasattr(self, 'surface_lock'):
            with self.surface_lock:
                return self.latest_surface
        return self.latest_surface

    def _publish_surface(self):
        """Publish a complete snapshot of the just-rendered frame.

        The render loop clears and redraws self.surface in place every frame, so
        handing consumers a *reference* to it lets the main thread blit a frame
        that is mid-clear or mid-draw -> flicker. We copy the surface only after
        all drawing for the frame is finished, and swap the published reference
        under the lock, so consumers always receive a whole, stable frame.
        """
        snapshot = self.surface.copy()
        with self.surface_lock:
            self.latest_surface = snapshot

    def calculate_fps(self):
        """Calculate current rendering FPS."""
        if time.time() - self.fps_timer >= 1.0:
            self.fps = self.render_count / (time.time() - self.fps_timer)
            self.fps_timer = time.time()
            self.render_count = 0
            return True
        return False

    def set_timescale(self, ts):
        """Set shared timescale reference from main thread."""
        self.ts = ts

    def stop(self):
        """Stop the rendering thread."""
        print(f"Stopping rendering thread for mode: {self.mode.value}")
        self.running = False
        self.join(timeout=2.0)

class TrackingVisualizationThread(VisualizationRenderingThread):
    """Thread for full-screen tracking visualization rendering."""

    def __init__(self, display, config_state, tracking_vis_state, target_fps=10):
        super().__init__(display, config_state, tracking_vis_state, RenderMode.FULL_SCREEN, target_fps)
        self.ts = None  # To be set by main thread
        self.surface_lock = threading.Condition()  # Lock for reliable surface access

    def run(self):
        """Main rendering loop for tracking visualization."""
        print("TrackingVisualizationThread starting...")

        while self.running:
            current_time = time.time()

            # Throttle to target FPS
            if self.frame_interval > 0 and current_time - self.last_render_time < self.frame_interval:
                safe_time.sleep(0.001)  # Short sleep to prevent busy waiting
                continue

            # Check if we have required data
            if not self.tracking_vis_state or not hasattr(self.tracking_vis_state, 'satellite_positions'):
                safe_time.sleep(0.01)
                continue

            try:
                # Create time object for current rendering
                if not self.ts:
                    from skyfield.api import load
                    self.ts = load.timescale()
                current_tt = self.ts.now().tt

                # Capture consistent snapshot of shared state to avoid race conditions
                if hasattr(self.tracking_vis_state, 'satellite_positions') and self.tracking_vis_state.satellite_positions:
                    # Capture state after satellite position updates are complete
                    # This may happen after the main thread's position update
                    satellite_positions_snapshot = self.tracking_vis_state.satellite_positions.copy()
                    selected_satellite_snapshot = self.tracking_vis_state.selected_satellite
                    hovered_satellite_snapshot = self.tracking_vis_state.hovered_satellite

                    # If we have a selected satellite but it's not in our snapshot,
                    # try once more to get updated positions (may have been updated after our loop iteration)
                    if selected_satellite_snapshot and selected_satellite_snapshot not in satellite_positions_snapshot:
                        # Brief retry - position update might have happened after we checked
                        safe_time.sleep(0.001)  # Micro sleep for thread sync
                        satellite_positions_snapshot = self.tracking_vis_state.satellite_positions.copy()

                    # Only black out the surface if no satellites at all, not just if selected satellite is missing
                    if not satellite_positions_snapshot:
                        # Empty render if no satellites to display
                        self.surface.fill((0, 0, 0))

                    # Clear surface periodically to prevent trail accumulation
                    # but not every frame to avoid frequency mismatch with main loop
                    if not hasattr(self, '_frame_counter'):
                        self._frame_counter = 0

                    self._frame_counter += 1

                    # Clear every n-frames
                    # If the thread gets really slow, increase the modulo to ward off darkness from delay
                    if self._frame_counter % 1 == 0:
                        self.surface.fill((0, 0, 0))

                    # Set up display bounds for surface rendering
                    # Use the actual screen coordinates where the surface will be blitted
                    display_bounds = {
                        'sub_x': self.display.sub_x,
                        'sub_y': self.display.sub_y,
                        'sub_width': self.surface.get_width(),
                        'sub_height': self.surface.get_height()
                    }

                    # Center coordinates for satellite rendering calculations
                    cx = self.surface.get_width() // 2
                    cy = self.surface.get_height() // 2

                    # Draw polar plot and satellites using surface-based functions

                    # Compute camera FOV data
                    self.compute_camera_fov_data()

                    # Update launch positions for current timestamp (critical for launch visualization)
                    if hasattr(self.tracking_vis_state, 'launch_trajectories') and self.tracking_vis_state.launch_trajectories:
                        from trajectory import update_launch_positions
                        update_launch_positions(self.tracking_vis_state, current_tt)

                    # Draw polar plot background first
                    draw_polar_plot_on_surface(self.surface, self.config_state, self.ts, current_tt, self.tracking_vis_state, display_bounds, PolarPlotMode.FULL_SCREEN)
                    # Now draw FOV boxes
                    draw_fov_on_surface(self.surface, self.tracking_vis_state, cx, cy, display_bounds, PolarPlotMode.FULL_SCREEN, None)
                    # Now draw satellites on top
                    draw_satellites_on_surface(self.surface, self.tracking_vis_state, cx, cy, display_bounds, PolarPlotMode.FULL_SCREEN, self.config_state)
                    # Draw launch trajectory if one is selected
                    draw_launch_trajectory_on_surface(self.surface, self.tracking_vis_state, current_tt, display_bounds, PolarPlotMode.FULL_SCREEN)
                    # Draw launch position marker
                    draw_launch_position_on_surface(self.surface, self.tracking_vis_state, cx, cy, display_bounds, PolarPlotMode.FULL_SCREEN)
                else:
                    # No satellite data available, just clear and skip rendering
                    self.surface.fill((0, 0, 0))

                # For other overlay elements, we could add simplified versions
                # For now, just core polar plot and satellites to avoid complexity

                # Publish a complete snapshot so the main loop never blits a
                # half-cleared / half-drawn frame (root cause of viz flicker).
                self._publish_surface()
                self.last_render_time = current_time
                self.render_count += 1

            except Exception as e:
                print(f"Error in TrackingVisualizationThread: {e}")
                import traceback
                traceback.print_exc()
                safe_time.sleep(0.1)  # Brief pause on error

        print("TrackingVisualizationThread stopped.")

    def compute_camera_fov_data(self):
        """Compute FOV parameters for both cameras and store in tracking_vis_state."""
        try:
            # Use telescope position stored in tracking_vis_state
            current_azm = self.tracking_vis_state.telescope_azimuth
            current_alt = self.tracking_vis_state.telescope_altitude

            # Get alignment parameters from config_state (already available)
            try:
                alignment_azimuth = float(self.config_state.alignment_azimuth_str or 0.0)
                alignment_elevation = float(self.config_state.alignment_elevation_str or 0.0)
            except (AttributeError, ValueError):
                alignment_azimuth = 0.0
                alignment_elevation = 0.0

            # Initialize FOV data list
            fov_data_list = []

            # Process each camera
            for camera_idx in [0, 1]:  # Camera1 and Camera2
                try:
                    camera_key = f"camera{camera_idx + 1}"

                    # Get camera parameters from config
                    camera_config = self.config_state.camera_configs.get(camera_key, {})
                    pixel_size_um = float(camera_config.get('pixel_size', 2.9))
                    focal_length_mm = float(camera_config.get('focal_length', 162.0))
                    alignment_rotation = float(camera_config.get('alignment_rotation', 0.0))

                    # Get current ROI settings (from camera_manager)
                    from camera_manager import camera_manager
                    camera_obj = camera_manager.get_camera(camera_idx)
                    if not camera_obj:
                        continue

                    # Get camera resolution (use defaults if not connected)
                    width_pixels = camera_obj.width_res if camera_obj.connected else 1920
                    height_pixels = camera_obj.height_res if camera_obj.connected else 1280

                    # Get ROI settings
                    roi_width_pct = 0.5 ** camera_obj.roi_size if camera_obj.roi_size > 0 else 1.0
                    roi_height_pct = 0.5 ** camera_obj.roi_size if camera_obj.roi_size > 0 else 1.0

                    # Compute FOV for this camera
                    fov_params = compute_fov_for_camera(
                        pixel_size_um=pixel_size_um,
                        focal_length_mm=focal_length_mm,
                        roi_width_pct=roi_width_pct,
                        roi_height_pct=roi_height_pct,
                        camera_width_pixels=width_pixels,
                        camera_height_pixels=height_pixels
                    )

                    # Transform telescope position to sky coordinates based on mount mode
                    mount_mode = getattr(self.config_state, 'mount_mode', 'Eq')
                    if mount_mode == 'AltAz':
                        from transformations import AzAlt2AzEl_AltAz
                        true_az, true_el = AzAlt2AzEl_AltAz(
                            current_azm, current_alt,
                            alignment_azimuth
                        )
                    else:
                        # Use full equatorial transformation for Eq mode
                        true_az, true_el = AzAlt2AzEl(
                            current_azm, current_alt,
                            alignment_azimuth, alignment_elevation
                        )

                    # Apply camera alignment rotation
                    az, el = apply_rotation_to_az_el(true_az, true_el, alignment_rotation)

                    # Offset FOV center based on ROI position
                    roi_x = getattr(camera_obj, 'roi_x', 0.5)
                    roi_y = getattr(camera_obj, 'roi_y', 0.5)

                    # Offset calculations (simplified)
                    az_offset = (roi_x - 0.5) * fov_params['fov_width_deg'] / 4  # Scale down for better visibility
                    el_offset = (roi_y - 0.5) * fov_params['fov_height_deg'] / 4

                    fov_center_az = az + az_offset
                    fov_center_el = el + el_offset

                    # Create FOV data dictionary
                    fov_data = {
                        'camera_id': camera_idx,
                        'az': fov_center_az,
                        'el': fov_center_el,
                        'fov_width_deg': fov_params['fov_width_deg'],
                        'fov_height_deg': fov_params['fov_height_deg'],
                        'rotation': alignment_rotation,
                        'spot_size_arcsec_per_pixel': fov_params['spot_size_arcsec_per_pixel'],
                        'color': (255, 0, 0) if camera_idx == 0 else (255, 165, 0),  # Red for cam1, orange for cam2
                    }

                    fov_data_list.append(fov_data)

                except Exception as cam_e:
                    print(f"Error computing FOV for camera {camera_idx}: {cam_e}")
                    continue

            # Update the tracking_vis_state with computed FOV data
            self.tracking_vis_state.camera_fov_data = fov_data_list

        except Exception as e:
            print(f"Error computing camera FOV data: {e}")
            self.tracking_vis_state.camera_fov_data = []

class JoystickVisualizationThread(VisualizationRenderingThread):
    """Thread for upper right quadrant joystick mode visualization rendering."""

    def __init__(self, display, config_state, tracking_vis_state, target_fps=10):
        super().__init__(display, config_state, tracking_vis_state, RenderMode.UPPER_RIGHT_QUADRANT, target_fps)
        self.ts = None  # To be set by main thread
        self.surface_lock = threading.Condition()  # Lock for reliable surface access

    def run(self):
        """Main rendering loop for joystick visualization."""
        print("JoystickVisualizationThread starting...")

        while self.running:
            current_time = time.time()

            # Throttle to target FPS
            if self.frame_interval > 0 and current_time - self.last_render_time < self.frame_interval:
                safe_time.sleep(0.001)  # Short sleep to prevent busy waiting
                continue

            # Check if we have required data
            if not self.tracking_vis_state or not hasattr(self.tracking_vis_state, 'satellite_positions'):
                safe_time.sleep(0.01)
                continue

            try:
                # Create time object for current rendering
                if not self.ts:
                    from skyfield.api import load
                    self.ts = load.timescale()
                current_tt = self.ts.now().tt

                # Capture consistent snapshot of shared state to avoid race conditions
                if hasattr(self.tracking_vis_state, 'satellite_positions') and self.tracking_vis_state.satellite_positions:
                    # Capture state after satellite position updates are complete
                    # This may happen after the main thread's position update
                    satellite_positions_snapshot = self.tracking_vis_state.satellite_positions.copy()
                    selected_satellite_snapshot = self.tracking_vis_state.selected_satellite
                    hovered_satellite_snapshot = self.tracking_vis_state.hovered_satellite

                    # If we have a selected satellite but it's not in our snapshot,
                    # try once more to get updated positions (may have been updated after our loop iteration)
                    if selected_satellite_snapshot and selected_satellite_snapshot not in satellite_positions_snapshot:
                        # Brief retry - position update might have happened after we checked
                        safe_time.sleep(0.001)  # Micro sleep for thread sync
                        satellite_positions_snapshot = self.tracking_vis_state.satellite_positions.copy()

                    # Only black out the surface if no satellites at all, not just if selected satellite is missing
                    if not satellite_positions_snapshot:
                        # Empty render if no satellites to display
                        self.surface.fill((0, 0, 0))

                    # Clear surface periodically to prevent trail accumulation
                    # but not every frame to avoid frequency mismatch with main loop
                    if not hasattr(self, '_frame_counter'):
                        self._frame_counter = 0

                    self._frame_counter += 1

                    # Clear every n-frames
                    # If the thread gets really slow, increase the modulo to ward off darkness from delay
                    if self._frame_counter % 1 == 0:
                        self.surface.fill((0, 0, 0))

                    # Set up display bounds for quadrant mode
                    # The surface will be blitted at (display.sub_x + display.sub_width // 2, display.sub_y)
                    # So we need to set the bounds to represent the final screen position
                    display_bounds = {
                        'sub_x': self.display.sub_x + self.display.sub_width // 2,  # Upper right quadrant start
                        'sub_y': self.display.sub_y,
                        'sub_width': self.display.sub_width // 2,
                        'sub_height': self.display.sub_height // 2
                    }

                    # Center coordinates for quadrant surface (since surface is quadrant-sized, use surface center)
                    cx = self.surface.get_width() // 2
                    cy = self.surface.get_height() // 2

                    # Draw polar plot and satellites using surface-based functions for quadrant mode
                    # For satellites, we need to use full screen bounds for coordinate transformation
                    full_screen_bounds = {
                        'sub_x': self.display.sub_x,
                        'sub_y': self.display.sub_y,
                        'sub_width': self.display.sub_width,
                        'sub_height': self.display.sub_height
                    }

                    # Compute camera FOV data
                    self.compute_camera_fov_data()

                    # Update launch positions for current timestamp (critical for launch visualization)
                    if hasattr(self.tracking_vis_state, 'launch_trajectories') and self.tracking_vis_state.launch_trajectories:
                        from trajectory import update_launch_positions
                        update_launch_positions(self.tracking_vis_state, current_tt)

                    # Draw polar plot background first
                    draw_polar_plot_on_surface(self.surface, self.config_state, self.ts, current_tt, self.tracking_vis_state, display_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, full_screen_bounds)
                    # Now draw FOV boxes
                    draw_fov_on_surface(self.surface, self.tracking_vis_state, cx, cy, display_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, None)
                    # Now draw satellites on top
                    draw_satellites_on_surface(self.surface, self.tracking_vis_state, cx, cy, full_screen_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, self.config_state)
                    # Draw launch trajectory if one is selected
                    draw_launch_trajectory_on_surface(self.surface, self.tracking_vis_state, current_tt, full_screen_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT)
                    # Draw launch position marker
                    draw_launch_position_on_surface(self.surface, self.tracking_vis_state, cx, cy, full_screen_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT)

                    # Draw satellite details panel after satellites
                    draw_details_on_surface(self.surface, self.tracking_vis_state, display_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, self.config_state)
                    # Draw camera FOV details below satellite details (accounting for reduced height in quadrant mode)
                    draw_camera_fov_details_on_surface(self.surface, self.tracking_vis_state, display_bounds, 210, PolarPlotMode.UPPER_RIGHT_QUADRANT, self.config_state)
                else:
                    # No satellite data available, just clear and skip rendering
                    self.surface.fill((0, 0, 0))

                # Publish a complete snapshot so the main loop never blits a
                # half-cleared / half-drawn frame (root cause of viz flicker).
                self._publish_surface()
                self.last_render_time = current_time
                self.render_count += 1

            except Exception as e:
                print(f"Error in JoystickVisualizationThread: {e}")
                import traceback
                traceback.print_exc()
                safe_time.sleep(0.1)  # Brief pause on error

        print("JoystickVisualizationThread stopped.")

    def compute_camera_fov_data(self):
        """Compute FOV parameters for both cameras and store in tracking_vis_state."""
        try:
            # Use telescope position stored in tracking_vis_state
            current_azm = self.tracking_vis_state.telescope_azimuth
            current_alt = self.tracking_vis_state.telescope_altitude

            # Get alignment parameters from config_state (already available)
            try:
                alignment_azimuth = float(self.config_state.alignment_azimuth_str or 0.0)
                alignment_elevation = float(self.config_state.alignment_elevation_str or 0.0)
            except (AttributeError, ValueError):
                alignment_azimuth = 0.0
                alignment_elevation = 0.0

            # Initialize FOV data list
            fov_data_list = []

            # Process each camera
            for camera_idx in [0, 1]:  # Camera1 and Camera2
                try:
                    camera_key = f"camera{camera_idx + 1}"

                    # Get camera parameters from config
                    camera_config = self.config_state.camera_configs.get(camera_key, {})
                    pixel_size_um = float(camera_config.get('pixel_size', 2.9))
                    focal_length_mm = float(camera_config.get('focal_length', 162.0))
                    alignment_rotation = float(camera_config.get('alignment_rotation', 0.0))

                    # Get current ROI settings (from camera_manager)
                    from camera_manager import camera_manager
                    camera_obj = camera_manager.get_camera(camera_idx)
                    if not camera_obj:
                        continue

                    # Get camera resolution (use defaults if not connected)
                    width_pixels = camera_obj.width_res if camera_obj.connected else 1920
                    height_pixels = camera_obj.height_res if camera_obj.connected else 1280

                    # Get ROI settings
                    roi_width_pct = 0.5 ** camera_obj.roi_size if camera_obj.roi_size > 0 else 1.0
                    roi_height_pct = 0.5 ** camera_obj.roi_size if camera_obj.roi_size > 0 else 1.0

                    # Compute FOV for this camera
                    fov_params = compute_fov_for_camera(
                        pixel_size_um=pixel_size_um,
                        focal_length_mm=focal_length_mm,
                        roi_width_pct=roi_width_pct,
                        roi_height_pct=roi_height_pct,
                        camera_width_pixels=width_pixels,
                        camera_height_pixels=height_pixels
                    )

                    # Transform telescope position to sky coordinates based on mount mode
                    mount_mode = getattr(self.config_state, 'mount_mode', 'Eq')
                    if mount_mode == 'AltAz':
                        from transformations import AzAlt2AzEl_AltAz
                        true_az, true_el = AzAlt2AzEl_AltAz(
                            current_azm, current_alt,
                            alignment_azimuth
                        )
                    else:
                        # Use full equatorial transformation for Eq mode
                        true_az, true_el = AzAlt2AzEl(
                            current_azm, current_alt,
                            alignment_azimuth, alignment_elevation
                        )

                    # Apply camera alignment rotation
                    az, el = apply_rotation_to_az_el(true_az, true_el, alignment_rotation)

                    # Offset FOV center based on ROI position
                    roi_x = getattr(camera_obj, 'roi_x', 0.5)
                    roi_y = getattr(camera_obj, 'roi_y', 0.5)

                    # Offset calculations (simplified)
                    az_offset = (roi_x - 0.5) * fov_params['fov_width_deg'] / 4  # Scale down for better visibility
                    el_offset = (roi_y - 0.5) * fov_params['fov_height_deg'] / 4

                    fov_center_az = az + az_offset
                    fov_center_el = el + el_offset

                    # Create FOV data dictionary
                    fov_data = {
                        'camera_id': camera_idx,
                        'az': fov_center_az,
                        'el': fov_center_el,
                        'fov_width_deg': fov_params['fov_width_deg'],
                        'fov_height_deg': fov_params['fov_height_deg'],
                        'rotation': alignment_rotation,
                        'spot_size_arcsec_per_pixel': fov_params['spot_size_arcsec_per_pixel'],
                        'color': (255, 0, 0) if camera_idx == 0 else (255, 165, 0),  # Red for cam1, orange for cam2
                    }

                    fov_data_list.append(fov_data)

                except Exception as cam_e:
                    print(f"Error computing FOV for camera {camera_idx}: {cam_e}")
                    continue

            # Update the tracking_vis_state with computed FOV data
            self.tracking_vis_state.camera_fov_data = fov_data_list

        except Exception as e:
            print(f"Error computing camera FOV data: {e}")
            self.tracking_vis_state.camera_fov_data = []
