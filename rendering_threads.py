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

# Font for surface rendering (will be created when needed)
SATELLITE_LABEL_FONT = None

# Cache for rendered satellite label surfaces to avoid expensive re-rendering
SATELLITE_LABEL_CACHE = {}

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

        # Draw satellite labels efficiently with caching
        if hasattr(state, 'satellite_labels') and sat in state.satellite_labels:
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
        self.running = True
        self.last_render_time = 0
        self.render_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()
        self.ts = None  # Shared timescale reference

        print(f"VisualizationRenderingThread initialized for mode: {mode.value}")

    def get_latest_surface(self):
        """Get the most recent rendered surface - thread-safe."""
        with self.surface_lock:
            return self.latest_surface

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

                    # Clear every 6 frames (roughly 60fps/10fps = 6 frames sync)
                    if self._frame_counter % 6 == 0:
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

                    # Draw polar plot background first
                    draw_polar_plot_on_surface(self.surface, self.config_state, self.ts, current_tt, self.tracking_vis_state, display_bounds, PolarPlotMode.FULL_SCREEN)
                    # Now draw satellites on top
                    draw_satellites_on_surface(self.surface, self.tracking_vis_state, cx, cy, display_bounds, PolarPlotMode.FULL_SCREEN, self.config_state)
                else:
                    # No satellite data available, just clear and skip rendering
                    self.surface.fill((0, 0, 0))

                # For other overlay elements, we could add simplified versions
                # For now, just core polar plot and satellites to avoid complexity

                # Update latest surface reference (no locks needed)
                self.latest_surface = self.surface
                self.last_render_time = current_time
                self.render_count += 1

            except Exception as e:
                print(f"Error in TrackingVisualizationThread: {e}")
                import traceback
                traceback.print_exc()
                safe_time.sleep(0.1)  # Brief pause on error

        print("TrackingVisualizationThread stopped.")

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

                    # Clear every 6 frames (roughly 60fps/10fps = 6 frames sync)
                    if self._frame_counter % 6 == 0:
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

                    draw_polar_plot_on_surface(self.surface, self.config_state, self.ts, current_tt, self.tracking_vis_state, display_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, full_screen_bounds)
                    draw_satellites_on_surface(self.surface, self.tracking_vis_state, cx, cy, full_screen_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, self.config_state)

                    # Draw satellite details panel after satellites
                    draw_details_on_surface(self.surface, self.tracking_vis_state, display_bounds, PolarPlotMode.UPPER_RIGHT_QUADRANT, self.config_state)
                else:
                    # No satellite data available, just clear and skip rendering
                    self.surface.fill((0, 0, 0))

                # Update latest surface reference (no locks needed)
                self.latest_surface = self.surface
                self.last_render_time = current_time
                self.render_count += 1

            except Exception as e:
                print(f"Error in JoystickVisualizationThread: {e}")
                import traceback
                traceback.print_exc()
                safe_time.sleep(0.1)  # Brief pause on error

        print("JoystickVisualizationThread stopped.")
