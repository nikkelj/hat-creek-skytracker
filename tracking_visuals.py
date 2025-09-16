import math
import pygame
import numpy as np
from datetime import datetime, timedelta
from utils import get_altitude_color, draw_button
from skyfield.api import wgs84, load
from enum import Enum

# Polar plot mode enumeration
class PolarPlotMode(Enum):
    FULL_SCREEN = "full_screen"  # Full area polar plot (tracking visualization mode)
    UPPER_RIGHT_QUADRANT = "upper_right_quadrant"  # Upper right quadrant only (joystick mode)

# ==============================================================================
# TRACKING VISUALIZATION STATE CLASS
# ==============================================================================

class TrackingVisState:
    """Encapsulates all state for tracking visualization mode."""

    def __init__(self):
        from datetime import datetime, timezone, timedelta

        # Time-related state
        self.center_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # Initial to current UTC ISO
        self.duration_str = "30"
        self.t0 = None
        self.t1 = None
        self.current_tt = None
        self.paused = False
        self.paused_tt = None

        # Input field state
        self.focused_field = None
        self.cursor_pos = {
            "filter": 0,
            "filter_above_alt": 0,
            "filter_below_alt": 0,
            "center_time": 0,
            "duration": 0
        }
        self.selection_start = {
            "filter": None,
            "filter_above_alt": None,
            "filter_below_alt": None,
            "center_time": None,
            "duration": None
        }

        # Filter state
        self.filter_text = ""
        self.filter_above_alt_text = ""
        self.filter_below_alt_text = ""

        # Satellite state
        self.selected_satellite = None
        self.hovered_satellite = None
        self.satellite_positions = {}

        # Pass table state
        self.satellite_pass_table = []
        self.table_sort_keys = [False, False, False, True, False]  # Default: sort by max elevation
        self.table_sort_reverse = [True, True, True, True, True]  # Default: descending for all
        self.pass_table_clickable_areas = []

        # UI state
        self.dragging_slider = False
        self.scroll_start = None

        # UI labels for scroll bar
        self.scroll_bar_start_label = "-15.0 min"
        self.scroll_bar_end_label = "+15.0 min"

        # Flags
        self.recompute_triggered = False

        # Satellite data (moved from global variables)
        self.satellite_labels = {}
        self.satellite_mean_altitudes = {}
        self.satellite_perigee = {}
        self.satellite_apogee = {}
        self.satellite_trajectories = {}
        self.satellite_arc_segments = {}
        self.tle_loaded = False
        self.satellites = []
        self.cache_file = "tle_cache.tle"
        self.cache_age_limit = 24 * 3600  # 24 hours in seconds

        # Performance cache for sunlit trajectories (satellite -> sunlit_array)
        self.sunlit_status_cache = {}

    def reset_input_fields(self):
        """Reset input field positions when switching modes."""
        self.cursor_pos = {
            "filter": 0,
            "filter_above_alt": 0,
            "filter_below_alt": 0,
            "center_time": 0,
            "duration": 0
        }
        self.selection_start = {
            "filter": None,
            "filter_above_alt": None,
            "filter_below_alt": None,
            "center_time": None,
            "duration": None
        }
        self.focused_field = None

    def get_updated_params(self):
        """Get dictionary of parameters for event handlers (for backward compatibility)."""
        return {
            'center_time_str': self.center_time_str,
            'duration_str': self.duration_str,
            'focused_field': self.focused_field,
            'cursor_pos': self.cursor_pos,
            'selection_start': self.selection_start,
            'filter_text': self.filter_text,
            'filter_above_alt_text': self.filter_above_alt_text,
            'filter_below_alt_text': self.filter_below_alt_text,
            'satellite_positions': self.satellite_positions,
            'selected_satellite': self.selected_satellite,
            'satellite_pass_table': self.satellite_pass_table,
            'table_sort_keys': self.table_sort_keys,
            'table_sort_reverse': self.table_sort_reverse,
            'paused': self.paused,
            'paused_tt': self.paused_tt,
            'dragging_slider': self.dragging_slider,
            't0': self.t0,
            't1': self.t1,
            'pass_table_clickable_areas': self.pass_table_clickable_areas,
            'hovered_satellite': self.hovered_satellite,
            'scroll_start': self.scroll_start
        }


# ==============================================================================
# CONSTANTS
# ==============================================================================

# UI Dimensions and Layout
UI_MARGIN = 5                    # Standard margin for UI elements
UI_PADDING = 5                   # Standard padding for text inside elements
UI_SMALL_PADDING = 2             # Smaller padding for tight layouts
UI_TEXT_LABEL_OFFSET = 15        # Y offset for text labels above inputs
UI_DOUBLE_CLICK_THRESHOLD = 10   # Threshold for satellite selection/clicking
UI_LINE_THICKNESS = 1            # Default line thickness for drawing
UI_FOCUSED_BORDER_WIDTH = 2      # Width of focus border
UI_RECTANGLE_BORDER_WIDTH = 2    # Width of group rectangles
UI_RECTANGLE_BORDER_RADIUS = 5   # Border radius for rounded rectangles

# Shape and Satellite Drawing
SATELLITE_SHAPE_SIZE_DEFAULT = 5     # Default size for hexagon/triangle shapes
SATELLITE_CIRCLE_RADIUS = 3         # Radius of default satellite circles
SATELLITE_HIGHLIGHT_RADIUS = 5      # Radius of highlight circle for hovered/selected
SATELLITE_ELLIPSE_WIDTH = 6         # Width of elliptical marker for eccentric orbits
SATELLITE_ELLIPSE_HEIGHT = 3        # Height of elliptical marker for eccentric orbits
ECCENTRICITY_THRESHOLD = 0.01       # Threshold to determine if orbit is eccentric

# Altitude Ranges (km)
LEO_ALTITUDE_MAX = 2000            # Maximum LEO altitude
GEO_ALTITUDE_NOMINAL = 35786       # Nominal GEO altitude
GEO_ALTITUDE_TOLERANCE = 1000      # Tolerance for GEO altitude classification

# Orbit Classification Colors (RGB)
COLOR_MEO_ORBIT = (255, 165, 0)    # Orange for Medium Earth Orbit
COLOR_GEO_ORBIT = (128, 0, 128)    # Purple for Geostationary Earth Orbit
COLOR_LEO_DEFAULT = (0, 255, 0)    # Green default for Low Earth Orbit
COLOR_SATELLITE_HIGHLIGHT = (255, 255, 0)  # Yellow highlight for selected/hovered

# Drawing Colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GRAY_MEDIUM = (100, 100, 100)
COLOR_GRAY_DARK = (50, 50, 50)
COLOR_GRAY_DARK_BACKGROUND = (30, 30, 30)
COLOR_FOCUS_BLUE = (0, 0, 255)
COLOR_SELECTION = (0, 120, 215)

# Gradient Parameters
GRADIENT_BRIGHTNESS_FACTOR = 5     # Brightness adjustment factor for gradients
GRADIENT_BOTTOM_COLOR = (160, 160, 160)  # Base color for config background gradient

# Polar Plot Parameters
POLAR_RADIUS_OFFSET = 50           # Offset from display edge for polar plot radius
POLAR_SCREEN_OFFSET_X = 50         # X offset for polar plot from screen edge
POLAR_SCREEN_OFFSET_Y = 5          # Y offset for polar plot from screen edge
ELEVATION_CIRCLE_DEGREES = [30, 60]  # Elevation circles to draw
AZIMUTH_LINE_INTERVAL_DEGREES = 30  # Degrees between azimuth lines

# Elevation Mask and Horizon Parameters
HORIZON_ELEVATION_DEGREES = 90      # Elevation angle for horizon

# Arc Segment Drawing
ARC_SEGMENT_WIDTH = 1              # Line width for trajectory arcs

# Font Sizes
FONT_SIZE_ELEVATION_LABEL = 14     # Font size for elevation labels
FONT_SIZE_DIRECTION_LABEL = 14     # Font size for compass direction labels

# Legend Parameters
LEGEND_WIDTH = 150                # Width of orbit legend
LEGEND_HEIGHT = 140               # Height of orbit legend
LEGEND_TITLE_OFFSET_Y = 20         # Y offset for legend title line
LEGEND_CONTENT_OFFSET_Y = 40       # Y offset for legend content
LEGEND_ITEM_SPACING_Y = 20         # Y spacing between legend items
LEGEND_ALTITUDE_BAR_WIDTH = 150    # Width of altitude color bar
LEGEND_ALTITUDE_BAR_HEIGHT = 25    # Height of altitude color bar
LEGEND_ALTITUDE_HEATMAP_SCALE = 1000  # Max altitude for color bar (km)
LEGEND_ALTITUDE_LABELS = [0, 1000]    # Altitude labels (km)

# Table/Display Parameters
DETAILS_PANEL_WIDTH = 230          # Width of satellite details panel
DETAILS_PANEL_HEIGHT = 250         # Height of satellite details panel
DETAILS_PANEL_OFFSET_X = 250       # X offset for details panel from right edge
DETAILS_PANEL_OFFSET_Y = 20        # Y offset for details panel from top
DETAILS_LINE_HEIGHT = 15           # Line height for details text

# Satellite Count Display
SATELLITE_COUNT_OFFSET_Y = 240     # Y offset for satellite count from top

# Time Display
TIME_DISPLAY_OFFSET_Y_FROM_BOTTOM = 30  # Y offset for time from bottom
TIME_DISPLAY_OFFSET_X = 10          # X offset for time display

# Scroll Bar and Slider
SCROLL_BAR_HEIGHT = 10             # Height of scroll bar background
SLIDER_HEIGHT = 10                 # Height of scroll bar slider
SCROLL_BAR_OFFSET_Y = 35           # Y offset for scroll bar from visualization bottom
SLIDER_OFFSET_Y = 35               # Y offset for slider from visualization bottom
SCROLL_TIME_DISPLAY_OFFSET_Y = 15  # Y offset for scroll time display

# Pass Table Parameters
PASS_TABLE_OFFSET_Y = 370          # Y offset for pass table from visualization bottom
PASS_TABLE_WIDTH = 374             # Total width of pass table
PASS_TABLE_HEIGHT = 316            # Total height of pass table
PASS_TABLE_MAX_ROWS = 15           # Maximum number of rows to display
PASS_TABLE_ROW_HEIGHT = 18         # Height of each table row
PASS_TABLE_HEADER_HEIGHT = 25      # Height of table header
PASS_TABLE_MARGIN = 5              # Internal margin for table content
PASS_TABLE_BACKGROUND_COLOR = (40, 40, 40)   # Background color
PASS_TABLE_BORDER_COLOR = (200, 200, 200)   # Border color

# Pass Table Column Widths
PASS_TABLE_COLUMN_NAME = 120       # Name column width
PASS_TABLE_COLUMN_NORAD = 55       # NORAD ID column width
PASS_TABLE_COLUMN_AZIMUTH = 42     # Azimuth column width
PASS_TABLE_COLUMN_ELEVATION = 57   # Max elevation column width
PASS_TABLE_COLUMN_TIME = 80        # Time column width

# Pass Table Row Colors
PASS_TABLE_ROW_ODD_COLOR = (45, 45, 45)      # Odd row background
PASS_TABLE_ROW_EVEN_COLOR = (35, 35, 35)     # Even row background
PASS_TABLE_ROW_SELECTED_COLOR = (80, 100, 120)  # Selected row highlight
PASS_TABLE_HEADER_DEFAULT_COLOR = (50, 50, 50) # Default header background
PASS_TABLE_HEADER_SORTED_COLOR = (60, 60, 60)   # Sorted header background

# Text Shortening and Formatting
SATELLITE_NAME_MAX_LENGTH = 20     # Max length for displayed satellite names
TIME_STRING_MILLISECONDS_PRECISION = 3   # Precision for time string milliseconds
DEFAULT_ELEVATION_MASK_DEG = 10.0        # Default elevation mask (degrees)
HOUR_OFFSET_PDT = 7                     # PDT offset from UTC (hours)

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def draw_hexagon(surface, x, y, color, size=5):
    points = [(x + size * math.cos(math.radians(60 * i)), y + size * math.sin(math.radians(60 * i))) for i in range(6)]
    pygame.draw.polygon(surface, color, points)

def draw_triangle(surface, x, y, color, size=5):
    points = [(x, y - size), (x - size * math.sin(math.radians(60)), y + size * 0.5), (x + size * math.sin(math.radians(60)), y + size * 0.5)]
    pygame.draw.polygon(surface, color, points)

def draw_polar_plot(display, config_state, ts, current_tt, state=None, mode=PolarPlotMode.FULL_SCREEN):
    """
    State-direct mutation function for drawing polar plot.
    Takes display, config_state, and state objects directly. Supports different size modes.
    """
    # Adjust display bounds based on mode
    original_sub_x = display.sub_x
    original_sub_y = display.sub_y
    original_sub_width = display.sub_width
    original_sub_height = display.sub_height

    if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
        # Resize to upper right quadrant only
        display.sub_x = original_sub_x + original_sub_width // 2  # Right half
        display.sub_y = original_sub_y  # Top
        display.sub_width = original_sub_width // 2  # Half width
        display.sub_height = original_sub_height // 2  # Half height

    # Tracking visualization mode - requires state object
    if state is None:
        return  # Skip drawing if no state provided

    elevation_mask = float(config_state.elevation_mask_str or 0)
    # Draw polar plot for tracking vis
    cx = display.sub_x + display.sub_width // 2
    cy = display.sub_y + display.sub_height // 2
    radius = min(display.sub_width, display.sub_height) // 2 - 50
    # Draw horizon circle
    pygame.draw.circle(display.menu_screen, (255, 255, 255), (cx, cy), radius, 1)
    # Draw elevation mask circle
    mask_radius = (90 - elevation_mask) / 90 * radius
    pygame.draw.circle(display.menu_screen, (255, 0, 0), (cx, cy), mask_radius, 2)
    # Draw elevation circles
    for el in [30, 60]:
        r = (90 - el) / 90 * radius
        pygame.draw.circle(display.menu_screen, (100, 100, 100), (cx, cy), int(r), 1)
        el_label = pygame.font.Font(None, 14).render(f"{el}°", True, (255, 255, 255))
        display.menu_screen.blit(el_label, (cx + r + 5, cy - 5))
    # Draw azimuth lines and labels
    for az_deg in range(0, 360, 30):
        az_rad = math.radians(az_deg)
        x1 = cx + radius * math.sin(az_rad)
        y1 = cy - radius * math.cos(az_rad)
        pygame.draw.line(display.menu_screen, (100, 100, 100), (cx, cy), (x1, y1), 1)
        if az_deg % 90 == 0:
            direction = {0: "N", 90: "E", 180: "S", 270: "W"}[az_deg]
            direction_label = pygame.font.Font(None, 14).render(direction, True, (255, 255, 255))
            if az_deg == 0:  # North
                display.menu_screen.blit(direction_label, (cx - direction_label.get_width() // 2, cy - radius - 10))
            elif az_deg == 90:  # East
                display.menu_screen.blit(direction_label, (cx + radius + 10, cy - direction_label.get_height() // 2))
            elif az_deg == 180:  # South
                display.menu_screen.blit(direction_label, (cx - direction_label.get_width() // 2, cy + radius + 5))
            elif az_deg == 270:  # West
                display.menu_screen.blit(direction_label, (cx - radius - 10 - direction_label.get_width(), cy - direction_label.get_height() // 2))
    # Draw precomputed arc segments for selected satellite
    if state.selected_satellite and state.tle_loaded and state.selected_satellite in state.satellite_arc_segments:
        # Redraw selected satellite's arc segments with sunlit detection
        if state.selected_satellite in state.satellite_arc_segments and state.satellite_trajectories and state.selected_satellite in state.satellite_trajectories:
            trajectory_data, times_array = state.satellite_trajectories[state.selected_satellite]

            # Set up observer and sun for sunlit calculation
            try:
                observer = wgs84.latlon(float(config_state.lat_str or 0), float(config_state.lon_str or 0), elevation_m=float(config_state.alt_str or 0))

                # Load sun ephemeris
                sun_ephemeris = load('de421.bsp')['sun']

                # Create Skyfield Time objects for current trajectory times
                skyfield_times = []
                for tt_time in times_array:
                    skyfield_times.append(ts.tt(jd=tt_time))

                # Calculate sunlit status for each trajectory point
                sunlit_status = []
                for st in skyfield_times:
                    try:
                        # Get satellite position and sun position
                        sat_pos = state.selected_satellite.at(st)
                        sun_pos = sun_ephemeris.at(st)

                        # Calculate vectors
                        sat_to_sun = (sun_pos.position.km - sat_pos.position.km)
                        sat_to_center = -sat_pos.position.km  # Vector from satellite to Earth center

                        # Calculate angle between sun vector and Earth center vector
                        dot_product = np.dot(sat_to_sun, sat_to_center)
                        mag_sat_sun = np.linalg.norm(sat_to_sun)
                        mag_sat_center = np.linalg.norm(sat_to_center)

                        if mag_sat_sun > 0 and mag_sat_center > 0:
                            cos_angle = dot_product / (mag_sat_sun * mag_sat_center)
                            angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                            # If the satellite is closer to the sun-side relative to Earth, it's sunlit
                            # Invert the logic: if angle > 90°, then the sun is on the same side as the horizon
                            is_sunlit = angle > 90.0
                        else:
                            is_sunlit = False

                    except Exception:
                        is_sunlit = False

                    sunlit_status.append(is_sunlit)

                # Redraw segments with sunlit-aware colors
                segments = []
                current_start_time = times_array[0] if len(times_array) > 0 else 0

                for i in range(len(trajectory_data) - 1):
                    x0, y0, x1, y1 = trajectory_data[i][4], trajectory_data[i][5], trajectory_data[i + 1][4], trajectory_data[i + 1][5]
                    alt = trajectory_data[i][1]

                    if alt > 0:  # Above horizon
                        is_future = times_array[i] > current_tt if current_tt else False
                        is_sunlit = sunlit_status[i] if i < len(sunlit_status) else False

                        if is_future:
                            color = (255, 255, 0) if is_sunlit else (255, 0, 0)  # Yellow for sunlit future, red for shadowed
                        else:
                            color = (128, 128, 128)  # Grey for past

                        # Transform coordinates when in upper right quadrant mode
                        if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
                            # Calculate the quadrant center (where the polar plot circle is drawn)
                            quadrant_center_x = display.sub_x + display.sub_width // 2
                            quadrant_center_y = display.sub_y + display.sub_height // 2

                            # Calculate original full screen center (where arcs were calculated)
                            full_screen_center_x = original_sub_x + original_sub_width // 2
                            full_screen_center_y = original_sub_y + original_sub_height // 2

                            # First, translate coordinates relative to full screen center
                            rel_x0 = x0 - full_screen_center_x
                            rel_y0 = y0 - full_screen_center_y
                            rel_x1 = x1 - full_screen_center_x
                            rel_y1 = y1 - full_screen_center_y

                            # Scale down by factor of 2 (since quadrant is half the size of full screen)
                            scale_factor = 0.45
                            rel_x0 *= scale_factor
                            rel_y0 *= scale_factor
                            rel_x1 *= scale_factor
                            rel_y1 *= scale_factor

                            # Translate to quadrant center
                            x0 = rel_x0 + quadrant_center_x
                            y0 = rel_y0 + quadrant_center_y
                            x1 = rel_x1 + quadrant_center_x
                            y1 = rel_y1 + quadrant_center_y

                        pygame.draw.line(display.menu_screen, color, (x0, y0), (x1, y1), 1)

            except Exception as e:
                # Fallback to default arc segments if sunlit calculation fails
                for x0, y0, x1, y1, color in state.satellite_arc_segments[state.selected_satellite]:
                    # Transform coordinates when in upper right quadrant mode
                    if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
                        # Calculate the quadrant center (where the polar plot circle is drawn)
                        quadrant_center_x = display.sub_x + display.sub_width // 2
                        quadrant_center_y = display.sub_y + display.sub_height // 2

                        # Calculate original full screen center (where arcs were calculated)
                        full_screen_center_x = original_sub_x + original_sub_width // 2
                        full_screen_center_y = original_sub_y + original_sub_height // 2

                        # First, translate coordinates relative to full screen center
                        rel_x0 = x0 - full_screen_center_x
                        rel_y0 = y0 - full_screen_center_y
                        rel_x1 = x1 - full_screen_center_x
                        rel_y1 = y1 - full_screen_center_y

                        # Scale down by factor of 2 (since quadrant is half the size of full screen)
                        scale_factor = 0.45
                        rel_x0 *= scale_factor
                        rel_y0 *= scale_factor
                        rel_x1 *= scale_factor
                        rel_y1 *= scale_factor

                        # Translate to quadrant center
                        x0 = rel_x0 + quadrant_center_x
                        y0 = rel_y0 + quadrant_center_y
                        x1 = rel_x1 + quadrant_center_x
                        y1 = rel_y1 + quadrant_center_y
                    pygame.draw.line(display.menu_screen, color, (x0, y0), (x1, y1), 1)
        else:
                    # Default drawing if no trajectory data available
            for x0, y0, x1, y1, color in state.satellite_arc_segments[state.selected_satellite]:
                # Transform coordinates when in upper right quadrant mode
                if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
                    # Calculate the quadrant center (where the polar plot circle is drawn)
                    quadrant_center_x = display.sub_x + display.sub_width // 2
                    quadrant_center_y = display.sub_y + display.sub_height // 2

                    # Calculate original full screen center (where arcs were calculated)
                    full_screen_center_x = original_sub_x + original_sub_width // 2
                    full_screen_center_y = original_sub_y + original_sub_height // 2

                    # First, translate coordinates relative to full screen center
                    rel_x0 = x0 - full_screen_center_x
                    rel_y0 = y0 - full_screen_center_y
                    rel_x1 = x1 - full_screen_center_x
                    rel_y1 = y1 - full_screen_center_y

                    # Scale down by factor of 2 (since quadrant is half the size of full screen)
                    scale_factor = 0.45
                    rel_x0 *= scale_factor
                    rel_y0 *= scale_factor
                    rel_x1 *= scale_factor
                    rel_y1 *= scale_factor

                    # Translate to quadrant center
                    x0 = rel_x0 + quadrant_center_x
                    y0 = rel_y0 + quadrant_center_y
                    x1 = rel_x1 + quadrant_center_x
                    y1 = rel_y1 + quadrant_center_y
                pygame.draw.line(display.menu_screen, color, (x0, y0), (x1, y1), 1)

    # Restore original display bounds after drawing
    display.sub_x = original_sub_x
    display.sub_y = original_sub_y
    display.sub_width = original_sub_width
    display.sub_height = original_sub_height

def draw_satellites(display, state, cx, cy, mode=PolarPlotMode.FULL_SCREEN):
    """
    State-direct mutation function for drawing satellites.
    Takes display and state objects directly instead of individual parameters.
    """
    # Adjust display bounds if in upper right quadrant mode
    original_sub_x = display.sub_x
    original_sub_y = display.sub_y
    original_sub_width = display.sub_width
    original_sub_height = display.sub_height

    if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
        # Resize to upper right quadrant only
        display.sub_x = original_sub_x + original_sub_width // 2  # Right half
        display.sub_y = original_sub_y  # Top
        display.sub_width = original_sub_width // 2  # Half width
        display.sub_height = original_sub_height // 2  # Half height

    for sat, (px, py, alt, _) in state.satellite_positions.items():
        # Transform coordinates when in upper right quadrant mode
        draw_px, draw_py = px, py
        if mode == PolarPlotMode.FULL_SCREEN:
            # In full screen mode, use original coordinates
            draw_px = px
            draw_py = py
        elif mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
            # In quadrant mode, transform coordinates the same way as arcs

            # Full screen center where satellite positions were calculated
            full_cx = original_sub_x + original_sub_width // 2
            full_cy = original_sub_y + original_sub_height // 2

            # Quadrant center where we want to draw (this is correct quadrant center)
            quad_cx = display.sub_x + display.sub_width // 2
            quad_cy = display.sub_y + display.sub_height // 2

            # Transform: scale by 0.45 and place relative to quadrant center
            # Same transformation as arcs, but ensure we're using the right centers
            rel_x = px - full_cx
            rel_y = py - full_cy

            # Scale down by factor matching arcs
            scale_factor = 0.45
            rel_x *= scale_factor
            rel_y *= scale_factor

            # Place relative to quadrant center
            draw_px = quad_cx + rel_x
            draw_py = quad_cy + rel_y
        else:
            # Fallback for any other modes
            draw_px = px
            draw_py = py

        mean_altitude = state.satellite_mean_altitudes.get(sat, 0.0)
        eccentricity = sat.model.ecco
        if 2000 < mean_altitude <= 35786:  # MEO
            color = (255, 165, 0)  # Orange
            draw_hexagon(display.menu_screen, draw_px, draw_py, color)
        elif abs(mean_altitude - 35786) <= 1000:  # GEO or nearby
            color = (128, 0, 128)  # Purple
            draw_triangle(display.menu_screen, draw_px, draw_py, color)
        else:  # LEO (0-2000 km)
            color = get_altitude_color(mean_altitude) or (0, 255, 0)  # Fallback to green if out of range
            if eccentricity > 0.01:
                width = 6
                height = 3
                angle = math.degrees(math.atan2(draw_py - cy, draw_px - cx))
                oval_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                rotated_oval = pygame.transform.rotate(oval_surface, angle)
                rotated_rect = rotated_oval.get_rect(center=(draw_px, draw_py))
                display.menu_screen.blit(rotated_oval, rotated_rect.topleft)
            else:
                pygame.draw.circle(display.menu_screen, color, (draw_px, draw_py), 3)
        if sat == state.hovered_satellite or sat == state.selected_satellite:
            pygame.draw.circle(display.menu_screen, (255, 255, 0), (draw_px, draw_py), 5, 1)  # Highlight on hover or select
        display.menu_screen.blit(state.satellite_labels[sat], (draw_px + 5, draw_py))

    # Restore original display bounds after drawing
    display.sub_x = original_sub_x
    display.sub_y = original_sub_y
    display.sub_width = original_sub_width
    display.sub_height = original_sub_height

def draw_filters(display, state):
    """
    State-direct mutation function for drawing filter inputs.
    Takes display and state objects directly instead of individual parameters.
    """
    # Name filter
    filter_label = display.small_font.render("Name Filter:", True, (255, 255, 255))
    filter_rect = display.filter_rect
    display.menu_screen.blit(filter_label, (filter_rect.x, filter_rect.y - filter_label.get_height() - 5))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), filter_rect)
    filter_text_surface = display.small_font.render(state.filter_text, True, (0, 0, 0))
    display.menu_screen.blit(filter_text_surface, (filter_rect.x + 5, filter_rect.y + 5))
    if state.focused_field == "filter":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), filter_rect, 2)
        text_width, _ = display.small_font.size(state.filter_text[:state.cursor_pos["filter"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (filter_rect.x + 5 + text_width, filter_rect.y + 5),
                        (filter_rect.x + 5 + text_width, filter_rect.y + 25), 2)
        if state.selection_start["filter"] is not None:
            start_width, _ = display.small_font.size(state.filter_text[:min(state.cursor_pos["filter"], state.selection_start["filter"])])
            end_width, _ = display.small_font.size(state.filter_text[:max(state.cursor_pos["filter"], state.selection_start["filter"])])
            pygame.draw.rect(display.menu_screen, (0, 120, 215),
                            (filter_rect.x + 5 + start_width, filter_rect.y + 5,
                             end_width - start_width, 20), 2)

    # Filter Above Altitude
    filter_above_alt_label = display.small_font.render("Filter Above Alt (km):", True, (255, 255, 255))
    filter_above_alt_rect = display.filter_above_alt_rect
    display.menu_screen.blit(filter_above_alt_label, (filter_above_alt_rect.x, filter_above_alt_rect.y - filter_above_alt_label.get_height() - 5))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), filter_above_alt_rect)
    filter_above_alt_text_surface = display.small_font.render(state.filter_above_alt_text, True, (0, 0, 0))
    display.menu_screen.blit(filter_above_alt_text_surface, (filter_above_alt_rect.x + 5, filter_above_alt_rect.y + 5))
    if state.focused_field == "filter_above_alt":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), filter_above_alt_rect, 2)
        text_width, _ = display.small_font.size(state.filter_above_alt_text[:state.cursor_pos["filter_above_alt"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (filter_above_alt_rect.x + 5 + text_width, filter_above_alt_rect.y + 5),
                        (filter_above_alt_rect.x + 5 + text_width, filter_above_alt_rect.y + 25), 2)
        if state.selection_start["filter_above_alt"] is not None:
            start_width, _ = display.small_font.size(state.filter_above_alt_text[:min(state.cursor_pos["filter_above_alt"], state.selection_start["filter_above_alt"])])
            end_width, _ = display.small_font.size(state.filter_above_alt_text[:max(state.cursor_pos["filter_above_alt"], state.selection_start["filter_above_alt"])])
            pygame.draw.rect(display.menu_screen, (0, 120, 215),
                            (filter_above_alt_rect.x + 5 + start_width, filter_above_alt_rect.y + 5,
                             end_width - start_width, 20), 2)

    # Filter Below Altitude
    filter_below_alt_label = display.small_font.render("Filter Below Alt (km):", True, (255, 255, 255))
    filter_below_alt_rect = display.filter_below_alt_rect
    display.menu_screen.blit(filter_below_alt_label, (filter_below_alt_rect.x, filter_below_alt_rect.y - filter_below_alt_label.get_height() - 5))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), filter_below_alt_rect)
    filter_below_alt_text_surface = display.small_font.render(state.filter_below_alt_text, True, (0, 0, 0))
    display.menu_screen.blit(filter_below_alt_text_surface, (filter_below_alt_rect.x + 5, filter_below_alt_rect.y + 5))
    if state.focused_field == "filter_below_alt":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), filter_below_alt_rect, 2)
        text_width, _ = display.small_font.size(state.filter_below_alt_text[:state.cursor_pos["filter_below_alt"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (filter_below_alt_rect.x + 5 + text_width, filter_below_alt_rect.y + 5),
                        (filter_below_alt_rect.x + 5 + text_width, filter_below_alt_rect.y + 25), 2)
        if state.selection_start["filter_below_alt"] is not None:
            start_width, _ = display.small_font.size(state.filter_below_alt_text[:min(state.cursor_pos["filter_below_alt"], state.selection_start["filter_below_alt"])])
            end_width, _ = display.small_font.size(state.filter_below_alt_text[:max(state.cursor_pos["filter_below_alt"], state.selection_start["filter_below_alt"])])
            pygame.draw.rect(display.menu_screen, (0, 120, 215),
                            (filter_below_alt_rect.x + 5 + start_width, filter_below_alt_rect.y + 5,
                             end_width - start_width, 20), 2)

def draw_legend(display):
    """
    State-direct mutation function for drawing orbit legend.
    Takes display object directly instead of individual parameters.
    """
    pygame.draw.rect(display.menu_screen, (50, 50, 50), (display.legend_x, display.legend_y, 150, 140))  # Larger legend
    pygame.draw.line(display.menu_screen, (255, 255, 255), (display.legend_x, display.legend_y + 20), (display.legend_x + 150, display.legend_y + 20), 1)  # Title line
    legend_title = display.small_font.render("Orbit Legend", True, (255, 255, 255))
    display.menu_screen.blit(legend_title, (display.legend_x + 5, display.legend_y + 5))
    # LEO heatmap
    for i in range(150):
        alt = i / 150 * 1000  # Scale from 0 to 1000 km
        color = get_altitude_color(alt) or (0, 255, 0)  # Fallback to green if out of range
        pygame.draw.line(display.menu_screen, color, (display.legend_x + i, display.legend_y + 40), (display.legend_x + i, display.legend_y + 60))
    low_alt = display.small_font.render("0 km", True, (255, 255, 255))
    high_alt = display.small_font.render("1000 km", True, (255, 255, 255))
    display.menu_screen.blit(low_alt, (display.legend_x, display.legend_y + 70))
    display.menu_screen.blit(high_alt, (display.legend_x + 140, display.legend_y + 70))
    # MEO hexagon
    draw_hexagon(display.menu_screen, display.legend_x + 20, display.legend_y + 90, (255, 165, 0))  # Orange
    meo_label = display.small_font.render("MEO (Orange)", True, (255, 255, 255))
    display.menu_screen.blit(meo_label, (display.legend_x + 40, display.legend_y + 85))
    # GEO triangle with line break
    draw_triangle(display.menu_screen, display.legend_x + 20, display.legend_y + 110, (128, 0, 128))  # Purple
    geo_label = display.small_font.render("GEO (Purple)", True, (255, 255, 255))
    display.menu_screen.blit(geo_label, (display.legend_x + 40, display.legend_y + 105))

def draw_details(display, state, mode=PolarPlotMode.FULL_SCREEN):
    """
    State-direct mutation function for drawing satellite details panel.
    Takes display and state objects directly instead of individual parameters.
    """
    if (state.hovered_satellite or state.selected_satellite):
        sat = state.selected_satellite if state.selected_satellite else state.hovered_satellite
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
        dist = state.satellite_positions[sat][3] if state.satellite_positions and sat in state.satellite_positions else None
        if ((state.selected_satellite or state.hovered_satellite) and state.satellite_perigee and state.satellite_apogee):
            sat_perigee = state.satellite_perigee[sat] if sat in state.satellite_perigee else None
            sat_apogee = state.satellite_apogee[sat] if sat in state.satellite_apogee else None
            if sat_perigee is not None and sat_apogee is not None:
                details.extend([
                    f"Apogee Altitude (km): {sat_apogee:.1f}",
                    f"Perigee Altitude (km): {sat_perigee:.1f}"
                ])
        if ((state.selected_satellite or state.hovered_satellite) and dist is not None):
            details.append(f"Slant Range (km): {dist:.1f}")

        # Position the details panel based on mode
        if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
            # In quadrant mode, position relative to the quadrant bounds (scaled and offset)
            details_rect = pygame.Rect(display.sub_x + display.sub_width - 90, display.sub_y + 20, 85, 125)
        else:
            # Full screen mode - original positioning
            details_rect = pygame.Rect(display.sub_x + display.sub_width - 190, display.sub_y + 20, 170, 250)

        pygame.draw.rect(display.menu_screen, (50, 50, 50), details_rect)  # Dark grey background
        pygame.draw.rect(display.menu_screen, (0, 0, 0), details_rect, 2)  # Black border

        # Adjust font size and spacing for smaller quadrant panel
        if mode == PolarPlotMode.UPPER_RIGHT_QUADRANT:
            font_to_use = display.tiny_font
            line_height = 12
            padding = 3
        else:
            font_to_use = display.small_font
            line_height = 15
            padding = 5

        for i, line in enumerate(details):
            text_surface = font_to_use.render(line, True, (255, 255, 255))
            display.menu_screen.blit(text_surface, (details_rect.x + padding, details_rect.y + padding + i * line_height))

def draw_time_display(display):
    """
    State-direct mutation function for drawing time display.
    Takes display object directly instead of individual parameters.
    """
    current_utc = datetime.utcnow()
    current_local = current_utc - timedelta(hours=7)  # PDT is UTC-7
    utc_time_str = current_utc.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
    local_time_str = current_local.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
    time_text = f"UTC: {utc_time_str}  Local: {local_time_str}"
    time_surface = display.small_font.render(time_text, True, (255, 255, 255))
    display.menu_screen.blit(time_surface, (display.sub_x + 10, display.sub_y + display.sub_height - 30))

def draw_satellite_count(display, state):
    """
    State-direct mutation function for drawing satellite count display.
    Takes display and state objects directly instead of individual parameters.
    """
    sat_count = len(state.satellite_positions)
    count_text = f"Satellites in view: {sat_count}"
    count_surface = display.small_font.render(count_text, True, (255, 255, 255))
    display.menu_screen.blit(count_surface, (display.sub_x + 10, display.sub_y + 240))  # Moved below filter boxes

def draw_scroll_bar(display, state):
    """
    State-direct mutation function for drawing scroll bar.
    Takes display and state objects directly instead of individual parameters.
    """
    # Draw scroll bar track
    pygame.draw.rect(display.menu_screen, (100, 100, 100), display.scroll_bar_rect)
    # Draw slider
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.slider_rect)
    # Draw time labels from state
    start_label = display.small_font.render(state.scroll_bar_start_label, True, (255, 255, 255))
    end_label = display.small_font.render(state.scroll_bar_end_label, True, (255, 255, 255))
    display.menu_screen.blit(start_label, (display.scroll_bar_rect.x, display.scroll_bar_rect.y + display.scroll_bar_rect.height + 5))
    display.menu_screen.blit(end_label, (display.scroll_bar_rect.x + display.scroll_bar_rect.width - end_label.get_width(), display.scroll_bar_rect.y + display.scroll_bar_rect.height + 5))

def draw_scroll_time_display(display, current_tt, ts):
    """
    State-direct mutation function for drawing scroll time display.
    Takes display object, current_tt, and ts directly instead of individual parameters.
    """
    radius = min(display.sub_width, display.sub_height) // 2 - 50
    current_utc = ts.tt(jd=current_tt).utc_datetime()
    current_local = current_utc - timedelta(hours=7)  # PDT is UTC-7
    utc_time_str = current_utc.strftime("%H:%M:%S.%f")[:-3]
    local_time_str = current_local.strftime("%H:%M:%S.%f")[:-3]
    time_text = f"Scroll UTC: {utc_time_str}  Local: {local_time_str}"
    time_surface = display.small_font.render(time_text, True, (255, 255, 255))
    text_width, _ = display.small_font.size(time_text)
    display.menu_screen.blit(time_surface, (display.sub_x + display.sub_width // 2 - text_width // 2, display.sub_y + display.sub_height - 15))

def draw_satellite_pass_table(display, state):
    """
    Draw the satellite pass table in the lower left corner.
    State-direct mutation function that modifies state.pass_table_clickable_areas directly.
    """
    if not state.satellite_pass_table:
        state.pass_table_clickable_areas = []
        return

    # Table positioning and dimensions
    table_x = display.sub_x + 10
    table_y = display.sub_y + display.sub_height - 370  # Position above the status messages and controls, moved up 50 pixels
    table_width = 374  # Sum of column widths (354px) + margin (20px)
    table_height = 316  # Increased by ~2 rows to accommodate taller border
    row_height = 18
    header_height = 25

    # Background
    pygame.draw.rect(display.menu_screen, (40, 40, 40), (table_x, table_y, table_width, table_height))
    pygame.draw.rect(display.menu_screen, (200, 200, 200), (table_x, table_y, table_width, table_height), 2)

    # Column widths (Name, NORAD ID, Az, Max El, Closest Time)
    col_widths = [120, 55, 42, 57, 80]
    col_x_positions = [table_x + 5, table_x + col_widths[0] + 5, table_x + col_widths[0] + col_widths[1] + 5,
                      table_x + col_widths[0] + col_widths[1] + col_widths[2] + 5,
                      table_x + col_widths[0] + col_widths[1] + col_widths[2] + col_widths[3] + 5]

    # Table title
    title_text = "Satellite Passes"
    title_surface = display.small_font.render(title_text, True, (255, 255, 255))
    display.menu_screen.blit(title_surface, (table_x + 5, table_y + 5))

    # Column headers
    column_headers = ["Name", "NORAD", "Az", "Max El", "Closest"]
    state.pass_table_clickable_areas = []

    header_y = table_y + 25
    for i, header in enumerate(column_headers):
        # Draw header background
        header_bg_color = (60, 60, 60) if state.table_sort_keys and i < len(state.table_sort_keys) and state.table_sort_keys[i] else (50, 50, 50)
        pygame.draw.rect(display.menu_screen, header_bg_color, (col_x_positions[i] - 3, header_y, col_widths[i], header_height - 5))

        # Draw header text
        header_surface = display.small_font.render(header, True, (255, 255, 255))
        display.menu_screen.blit(header_surface, (col_x_positions[i], header_y + 3))

        # Add clickable border and area
        header_rect = pygame.Rect(col_x_positions[i] - 3, header_y, col_widths[i], header_height - 5)
        pygame.draw.rect(display.menu_screen, (150, 150, 150), header_rect, 1)
        state.pass_table_clickable_areas.append(('header', i, header_rect))

    # Draw table rows
    row_y = header_y + header_height - 5
    for row_idx, pass_entry in enumerate(state.satellite_pass_table[:15]):  # Limit to 15 rows maximum
        # Alternate row colors
        if row_idx % 2 == 0:
            row_bg_color = (45, 45, 45)
        else:
            row_bg_color = (35, 35, 35)

        # Highlight selected satellite's row
        if state.selected_satellite and pass_entry['satellite'] == state.selected_satellite:
            row_bg_color = (80, 100, 120)  # Blue highlight for selected

        # Draw row background
        pygame.draw.rect(display.menu_screen, row_bg_color, (table_x + 3, row_y, table_width - 6, row_height))

        # Draw cell contents
        cell_values = [
            pass_entry['name'][:20],  # Truncate long names
            pass_entry['norad_id'],
            f"{pass_entry['azimuth_at_max']:.0f}°",
            f"{pass_entry['max_elevation']:.1f}°",
            pass_entry.get('closest_approach_time', '--:--')  # Local time of minimum slant range
        ]

        for col_idx, value in enumerate(cell_values):
            # Draw cell text
            value_surface = display.small_font.render(value, True, (255, 255, 255))
            display.menu_screen.blit(value_surface, (col_x_positions[col_idx], row_y + 2))

        # Add row clickable area for selection
        row_rect = pygame.Rect(table_x + 3, row_y, table_width - 6, row_height)
        pygame.draw.rect(display.menu_screen, (100, 100, 100), row_rect, 1)
        state.pass_table_clickable_areas.append(('row', row_idx, row_rect))

        row_y += row_height

    # Add info text about sorting in title row, right-justified
    sort_index = state.table_sort_keys.index(True) if state.table_sort_keys and True in state.table_sort_keys else 3
    sort_order = 'Desc' if state.table_sort_reverse and sort_index < len(state.table_sort_reverse) and state.table_sort_reverse[sort_index] else 'Asc'
    info_text = f"Sorted by: {['Name', 'NORAD', 'Az', 'Max El', 'Closest'][sort_index]} ({sort_order})"
    info_surface = display.small_font.render(info_text, True, (180, 180, 180))
    info_x = table_x + table_width - info_surface.get_width() - 5  # Right-justified
    display.menu_screen.blit(info_surface, (info_x, table_y + 5))  # Same Y as title

def filter_and_sort_pass_table(state):
    """
    Filter and sort the satellite pass table based on name and altitude filters.
    State-direct mutation function that takes a tracking_vis_state object directly.
    Returns the filtered and sorted pass table.
    """
    if not state.satellite_pass_table:
        return []

    # Apply name and altitude filters
    filtered_table = [entry for entry in state.satellite_pass_table if (
        not state.filter_text or
        state.filter_text.lower() in entry['satellite'].name.lower() or
        state.filter_text in entry['satellite'].model.satnum_str
    ) and (
        not state.filter_above_alt_text or
        float(state.satellite_mean_altitudes.get(entry['satellite'], 0.0)) >= float(state.filter_above_alt_text)
    ) and (
        not state.filter_below_alt_text or
        float(state.satellite_mean_altitudes.get(entry['satellite'], 0.0)) <= float(state.filter_below_alt_text)
    )]

    # Apply sorting based on sort keys
    if state.table_sort_keys and any(state.table_sort_keys):
        # Find the column to sort by
        sort_column = None
        for i, is_sorted in enumerate(state.table_sort_keys):
            if is_sorted:
                sort_column = i
                break

        if sort_column is not None:
            reverse_sort = state.table_sort_reverse[sort_column] if state.table_sort_reverse and sort_column < len(state.table_sort_reverse) else True

            # Define sort key functions for each column
            sort_keys = {
                0: lambda x: x['satellite'].name.strip().lower(),  # Name (alphabetical)
                1: lambda x: x['satellite'].model.satnum_str,      # NORAD ID (numerical string)
                2: lambda x: x['azimuth_at_max'],                   # Azimuth
                3: lambda x: x['max_elevation'],                    # Max Elevation
                4: lambda x: x.get('closest_approach_time', '99:99')  # Closest Time (handle missing)
            }

            if sort_column in sort_keys:
                filtered_table.sort(key=sort_keys[sort_column], reverse=reverse_sort)

    return filtered_table
