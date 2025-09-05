import pygame
import os
import sys
import time
import requests
import math
import threading
from PIL import Image
from tkinter import Tk
from tkinter import filedialog
from skyfield.api import wgs84, load, utc
from datetime import datetime, timedelta, timezone
import numpy as np
from camera_buffer import CircularBuffer, CameraThread

# Camera imports (for ASI cameras)
try:
    import zwoasi as asi
    import numpy as np
    asi.init('lib/ASICamera2.dll')
    print("ZWO ASI SDK available for camera integration")
    ASI_AVAILABLE = True
except ImportError:
    print("Warning: ZWO ASI SDK not available - camera functionality will be disabled")
    ASI_AVAILABLE = False
    asi = None
except Exception as e:
    print(f"Error initializing ASI SDK: {e}")
    ASI_AVAILABLE = False
    asi = None
from utils import draw_button, create_negative_image
from trajectory import precompute_trajectories, interpolate_position, clear_trajectory_cache, build_satellite_pass_table, sort_pass_table
from config import load_config, save_config, handle_input
from visuals import draw_polar_plot, draw_satellites, draw_legend, draw_details, draw_filters, draw_time_display, draw_satellite_count, draw_scroll_bar, draw_scroll_time_display, draw_satellite_pass_table
from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
from camera_manager import *
from camera_manager import camera1_connected, camera2_connected
from events import *

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# UI Dimensions
MENU_WIDTH = 200
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 40
INPUT_WIDTH = 200
INPUT_HEIGHT = 30
CENTER_TIME_WIDTH = 130
DURATION_WIDTH = 130
FILTER_WIDTH = 100

# UI Spacing and Layout
BUTTON_GAP = 20
INPUT_GAP = 70
FILTER_GAP = 50
UI_MARGIN = 10
UI_HEIGHT_OFFSET = 5

# Background and Icon Images
BG_IMAGE_SIZE = (160, 160)
ICON_SIZE = (32, 32)
BACKGROUND_FILENAME = 'lucky.jpg'

# Font Sizes
LARGE_FONT_SIZE = 36
NORMAL_FONT_SIZE = 24
SMALL_FONT_SIZE = 14

# Default Values
DEFAULT_DURATION_MINUTES = "30"
DEFAULT_ELEVATION_MASK_DEG = 10.0

# Update Intervals (seconds)
POSITION_UPDATE_INTERVAL = 0.1  # 10 Hz
TRAJECTORY_UPDATE_INTERVAL = 900  # 15 minutes
FPS_TARGET = 60

# Cache Settings
CACHE_FILENAME = 'tle_cache.tle'
CACHE_AGE_LIMIT_HOURS = 24
CACHE_AGE_LIMIT_SECONDS = CACHE_AGE_LIMIT_HOURS * 3600

# Colors (RGB tuples)
COLOR_BACKGROUND_DARK = (30, 30, 30)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_INPUT_BACKGROUND = (255, 255, 255)
COLOR_INPUT_TEXT = (0, 0, 0)
COLOR_FOCUS_BLUE = (0, 0, 255)
COLOR_SELECTION = (0, 120, 215)

# TLE API Configuration
TLE_API_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'

# Astrodynamics Constants
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14  # m^3/s^2
EARTH_RADIUS_KM = 6371 # Mean radius
SECONDS_PER_HOUR = 3600
WINDOW_POSITION = "0,0"

# Camera Constants and Settings
CAMERA_UPDATE_INTERVAL = 0.05  # Update camera images at 15 FPS target (optimized timing)
CAMERA_TARGET_FPS = 15  # Increased from 10 for better frame rate while maintaining stability
CAMERA_WIDTH = 1920  # ASI462MM resolution (adjust based on actual camera)
CAMERA_HEIGHT = 1280
CAMERA2_WIDTH = 1936  # ASI178MC resolution
CAMERA2_HEIGHT = 1216

# ==============================================================================
# END CONSTANTS
# ==============================================================================

# Initialize Pygame and set up the display
pygame.init()
display_info = pygame.display.Info()
total_width = display_info.current_w
total_height = display_info.current_h
menu_screen = pygame.display.set_mode((total_width, total_height))
pygame.display.set_caption("Main Menu")

# Initialize camera system
initialize_camera_buttons(menu_screen, total_width, total_height)

# Enumerate cameras if available
camera_infos = []
camera1_name = "Camera 1"
camera2_name = "Camera 2"

# Update camera names with actual camera names if available
if ASI_AVAILABLE:
    from camera_manager import get_camera_names
    try:
        camera_names = get_camera_names()
        if len(camera_names) > 0:
            camera1_name = camera_names[0] if len(camera_names) > 0 else "Camera 1"
        if len(camera_names) > 1:
            camera2_name = camera_names[1] if len(camera_names) > 1 else "Camera 2"
        print(f"Camera names updated: Camera 1='{camera1_name}', Camera 2='{camera2_name}'")
    except Exception as e:
        print(f"Failed to get camera names: {e}")

# Load background image for menu and icon
try:
    bg_image = pygame.image.load(BACKGROUND_FILENAME)
    bg_image_menu = pygame.transform.scale(bg_image, BG_IMAGE_SIZE)
    bg_image_icon = pygame.transform.scale(bg_image, ICON_SIZE)
    pygame.display.set_icon(bg_image_icon)
    negative_image = create_negative_image(bg_image_menu)
    rotation_angle = 0
except pygame.error:
    bg_image_menu = None
    bg_image_icon = None
    negative_image = None
    rotation_angle = 0
    print(f"Warning: '{BACKGROUND_FILENAME}' not found. Using fallback color and no icon.")

font = pygame.font.Font(None, NORMAL_FONT_SIZE)
large_font = pygame.font.Font(None, LARGE_FONT_SIZE)
small_font = pygame.font.Font(None, SMALL_FONT_SIZE)
tiny_font = pygame.font.Font(None, 12)  # Smaller font for compact camera buttons
status_font = pygame.font.Font(None, SMALL_FONT_SIZE)

sub_x = MENU_WIDTH
sub_y = 0
sub_width = total_width - MENU_WIDTH
sub_height = total_height
radius = min(sub_width, sub_height) // 2 - 50  # For scroll bar width

# Combined view UI controls (bottom center of screen) - defined after pygame initialization
COMBINED_VIEW_BUTTON_RECT = pygame.Rect(total_width // 2 + 50, total_height - 60, 100, 20)  # Toggle button beneath the two images at center (shrunk height)
CAMERA_OPACITY_SLIDER_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 300, 20)  # Opacity slider
CAMERA_OPACITY_SLIDER_HANDLE_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 20, 20)  # Slider handle

input_rects = {
    'lat': pygame.Rect(sub_x + UI_MARGIN, sub_y + 60, INPUT_WIDTH, INPUT_HEIGHT),
    'lon': pygame.Rect(sub_x + UI_MARGIN, sub_y + 150, INPUT_WIDTH, INPUT_HEIGHT),
    'alt': pygame.Rect(sub_x + UI_MARGIN, sub_y + 240, INPUT_WIDTH, INPUT_HEIGHT),
    'elevation_mask': pygame.Rect(sub_x + UI_MARGIN, sub_y + 330, INPUT_WIDTH, INPUT_HEIGHT),
}
save_button = pygame.Rect(sub_x + 20, sub_y + sub_height - 50, 100, 30)
load_button = pygame.Rect(sub_x + 130, sub_y + sub_height - 50, 100, 30)
clear_filters_button = pygame.Rect(sub_x + 10, sub_y + 30, 100, 30)  # Aligned with Name Filter
recompute_button = pygame.Rect(sub_x + 230, sub_y + 30, 100, 30)  # Recompute Trajectories
reset_button = pygame.Rect(sub_x + 140, sub_y + 30, 80, 30)  # Reset
center_time_rect = pygame.Rect(sub_x + 140, sub_y + 90, 130, 30)  # Center Time ISO
duration_rect = pygame.Rect(sub_x + 140, sub_y + 140, 130, 30)  # Duration Minutes
filter_rect = pygame.Rect(sub_x + 10, sub_y + 70, 100, 30)  # Name Filter
filter_above_alt_rect = pygame.Rect(sub_x + 10, sub_y + 130, 100, 30)  # Filter Above Alt
filter_below_alt_rect = pygame.Rect(sub_x + 10, sub_y + 185, 100, 30)  # Filter Below Alt
scroll_bar_rect = pygame.Rect(sub_x + sub_width // 2 - radius, sub_y + sub_height - 35, 2 * radius, 10)  # Scroll bar below polar plot
slider_rect = pygame.Rect(sub_x + sub_width // 2 - radius, sub_y + sub_height - 35, 20, 10)  # Slider
pause_button = pygame.Rect(sub_x + sub_width // 2 + radius + 10, sub_y + sub_height - 45, 60, 30)  # Pause button
play_button = pygame.Rect(sub_x + sub_width // 2 + radius + 80, sub_y + sub_height - 45, 60, 30)  # Play button
legend_x = sub_x + sub_width - 170
legend_y = sub_y + sub_height - 160

buttons = [
    {"rect": pygame.Rect(10, 10, 180, 40), "text": "Tracking Vis", "mode": "tracking_vis"},
    {"rect": pygame.Rect(10, 60, 180, 40), "text": "Sensor Calib", "mode": "sensor_calib"},
    {"rect": pygame.Rect(10, 110, 180, 40), "text": "Joystick Loop", "mode": "joystick_loop"},
    {"rect": pygame.Rect(10, 160, 180, 40), "text": "Post Process", "mode": "post_process"},
    {"rect": pygame.Rect(10, 210, 180, 40), "text": "Config Options", "mode": "config_options"},
    {"rect": pygame.Rect(10, 260, 180, 40), "text": "Author Info", "mode": "author_info"},
    {"rect": pygame.Rect(10, 310, 180, 40), "text": "Exit", "mode": "exit"},
]

image_y = 550 + 80 + 10  # Position underneath the buttons
status_y_start = total_height - 14 * 4  # Adjusted for new font size, space for 4 lines

current_mode = None
clock = pygame.time.Clock()

# For author info
author_bg = None
try:
    author_bg = pygame.image.load('lucky.jpg')
except pygame.error:
    pass

# Load configuration
config = load_config()
lat_str, lon_str, alt_str, elevation_mask_str = config["lat"], config["lon"], config["alt"], config["elevation_mask"]
focused_field = None
cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0, "filter": 0, "filter_above_alt": 0, "filter_below_alt": 0, "center_time": 0, "duration": 0}
selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None, "filter": None, "filter_above_alt": None, "filter_below_alt": None, "center_time": None, "duration": None}
center_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # Initial to current UTC ISO
duration_str = "30"  # 30 minutes
filter_text = ""
filter_above_alt_text = ""
filter_below_alt_text = ""
paused = False
paused_tt = None
scroll_start = None
dragging_slider = False
t0 = None
t1 = None
recompute_triggered = False

# Scroll bar labels (global variables to persist between updates)
scroll_bar_start_label = "-15.0 min"
scroll_bar_end_label = "+15.0 min"

# Button states for embossment
button_states = {btn["mode"]: {"hover": False, "clicked": False} for btn in buttons}
button_states["save"] = {"hover": False, "clicked": False}
button_states["load"] = {"hover": False, "clicked": False}
button_states["clear_filters"] = {"hover": False, "clicked": False}
button_states["recompute"] = {"hover": False, "clicked": False}
button_states["reset"] = {"hover": False, "clicked": False}
button_states["pause"] = {"hover": False, "clicked": False}
button_states["play"] = {"hover": False, "clicked": False}

# Initial render of main menu
menu_screen.fill((30, 30, 30), (0, 0, MENU_WIDTH, total_height))  # Dark theme menu background
if bg_image_menu:
    menu_screen.blit(bg_image_menu, ((MENU_WIDTH - 160) // 2, image_y))  # Center horizontally, adjusted for 160px width
for btn in buttons:
    draw_button(menu_screen, btn["rect"], btn["text"], button_states[btn["mode"]])
status_messages = ["Starting TLE process..."]
for i, msg in enumerate(status_messages[-4:]):
    status_render = status_font.render(msg, True, (255, 255, 255))  # White text for dark theme
    menu_screen.blit(status_render, (10, status_y_start + i * 14))
pygame.display.flip()
print(f"Debug: Status - {'Starting TLE process...'}")

# Cache file management
cache_file = "tle_cache.tle"
cache_age_limit = 24 * 3600  # 24 hours in seconds

# Load satellite data using refactored module
tle_loaded = False
satellites = []
ts = load.timescale()

# Define status update callback for satellite loading
def status_callback(message):
    status_messages.append(message)
    status_render = status_font.render(status_messages[-1], True, (255, 255, 255))
    menu_screen.fill((30, 30, 30), (0, 0, MENU_WIDTH, total_height))  # Dark theme menu background
    for btn in buttons:
        draw_button(menu_screen, btn["rect"], btn["text"], button_states[btn["mode"]])
    if bg_image_menu:
        menu_screen.blit(bg_image_menu, ((MENU_WIDTH - 160) // 2, image_y))
    for i, msg in enumerate(status_messages[-4:]):
        status_render = status_font.render(msg, True, (255, 255, 255))
        menu_screen.blit(status_render, (10, status_y_start + i * 14))
    pygame.display.flip()
    pygame.time.wait(50)  # Brief pause to ensure display update
    print(f"Debug: Status - {message}")

try:
    satellites, tle_loaded = load_satellite_data(
        cache_file=cache_file,
        cache_age_limit_seconds=cache_age_limit,
        update_status_callback=status_callback
    )
except Exception as e:
    status_messages.append(f"TLE loading error: {str(e)}")
    print(f"Debug: Error loading TLEs: {e}")

# Pre-compute satellite labels and mean altitudes using refactored module
satellite_labels = {}
satellite_mean_altitudes = {}
satellite_perigee = {}
satellite_apogee = {}

if tle_loaded and satellites:
    satellite_labels, satellite_mean_altitudes, satellite_perigee, satellite_apogee = create_satellite_labels_and_metadata(satellites, ts)
    # Force immediate trajectory computation after TLEs are loaded
    last_trajectory_update = 0

# Force immediate trajectory computation with no delays
recompute_triggered = True
# Force immediate timer trigger
last_trajectory_update = -1
# Add clear status message that trajectory computation is starting immediately
status_messages.append("TLEs loaded - starting trajectory computation...")
print("IMMEDIATE: TLEs loaded - forcing trajectory computation to start NOW")

last_update_time = 0
update_interval = 0.1  # Target 10 Hz
last_trajectory_update = -1  # Force immediate trigger on first loop
trajectory_interval = 900  # 15 minutes
satellite_trajectories = {}
satellite_arc_segments = {}
hovered_satellite = None
selected_satellite = None
satellite_positions = {}

# Satellite pass table variables
satellite_pass_table = []
table_sort_keys = [False, False, False, True, False]  # Default: sort by max elevation (index 3), 5 columns total
table_sort_reverse = [True, True, True, True, True]  # Default: descending for all, 5 columns total
pass_table_clickable_areas = []

# Callback function for status updates during trajectory precomputation
def update_status_callback(message):
    status_messages.append(message)
    status_render = status_font.render(status_messages[-1], True, (255, 255, 255))
    menu_screen.fill((30, 30, 30), (0, 0, MENU_WIDTH, total_height))  # Dark theme menu background
    for btn in buttons:
        draw_button(menu_screen, btn["rect"], btn["text"], button_states[btn["mode"]])
    if bg_image_menu:
        menu_screen.blit(bg_image_menu, ((MENU_WIDTH - 160) // 2, image_y))
    for i, msg in enumerate(status_messages[-4:]):
        status_render = status_font.render(msg, True, (255, 255, 255))
        menu_screen.blit(status_render, (10, status_y_start + i * 14))
    pygame.display.flip()
    pygame.time.wait(50)  # Brief pause to ensure display update
    print(f"Debug: Status - {message}")



# ==============================================================================
# MAIN PROGRAM LOOP
# ==============================================================================

running = True
while running:
    current_time = time.time()
    mouse_pos = pygame.mouse.get_pos()
    # Check if mouse is over the background image
    image_rect = pygame.Rect((MENU_WIDTH - 160) // 2, image_y, 160, 160) if bg_image_menu else None
    # Define rectangles inside the loop to ensure they are always current
    filter_rect = pygame.Rect(sub_x + 10, sub_y + 90, 100, 30)  # Name Filter
    filter_above_alt_rect = pygame.Rect(sub_x + 10, sub_y + 140, 100, 30)  # Filter Above Alt
    filter_below_alt_rect = pygame.Rect(sub_x + 10, sub_y + 195, 100, 30)  # Filter Below Alt
    center_time_rect = pygame.Rect(sub_x + 140, sub_y + 90, 130, 30)  # Center Time ISO
    duration_rect = pygame.Rect(sub_x + 140, sub_y + 140, 130, 30)  # Duration Minutes

    if recompute_triggered and tle_loaded:
        try:
            center_time_obj = datetime.fromisoformat(center_time_str.replace('Z', '+00:00'))
            center_time = center_time_obj.replace(tzinfo=timezone.utc)
            duration_minutes = float(duration_str)
            lat = float(lat_str or 0)
            lon = float(lon_str or 0)
            alt_m = float(alt_str or 0)
            observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
            t0 = ts.utc((center_time - timedelta(minutes=duration_minutes / 2)))
            t1 = ts.utc((center_time + timedelta(minutes=duration_minutes / 2)))

            # Update scroll bar time range labels
            scroll_bar_start_label = "-" + str(duration_minutes/2) + " min"
            scroll_bar_end_label = "+" + str(duration_minutes/2) + " min"

            update_status_callback("Recomputing trajectories...")
            satellite_trajectories, satellite_arc_segments = precompute_trajectories(satellites, observer, ts, sub_x, sub_y, sub_width, sub_height, satellite_labels, update_status_callback, center_time, duration_minutes)

            # Build satellite pass table (filtered for current visibility + upcoming passes)
            satellite_pass_table = build_satellite_pass_table(satellite_trajectories, satellites, satellite_labels, elevation_mask_deg=float(elevation_mask_str or 10.0), ts=ts)
            # Modular table filtering and sorting - replaced ~30 lines of duplicate filtering logic
            satellite_pass_table = [entry for entry in satellite_pass_table if (
                not filter_text or
                filter_text.lower() in entry['satellite'].name.lower() or
                filter_text in entry['satellite'].model.satnum_str
            ) and (
                not filter_above_alt_text or
                float(satellite_mean_altitudes.get(entry['satellite'], 0.0)) >= float(filter_above_alt_text)
            ) and (
                not filter_below_alt_text or
                float(satellite_mean_altitudes.get(entry['satellite'], 0.0)) <= float(filter_below_alt_text)
            )]
            satellite_pass_table = sort_pass_table(satellite_pass_table, table_sort_keys, table_sort_reverse)

            last_trajectory_update = current_time
            update_status_callback("Trajectories recomputed")
            selected_satellite = None
            paused = False
            paused_tt = None
            fraction = max(0.0, min(1.0, (ts.now().tt - t0.tt) / (t1.tt - t0.tt)))
            slider_rect.x = scroll_bar_rect.x + int(fraction * (scroll_bar_rect.width - slider_rect.width))
        except Exception as e:
            update_status_callback(f"Error recomputing: {str(e)}")
        recompute_triggered = False

    # Precompute trajectories and arc segments every 15 minutes
    if current_time - last_trajectory_update >= trajectory_interval:
        update_status_callback("Starting trajectory precomputation...")
        lat = float(lat_str or 0)
        lon = float(lon_str or 0)
        alt_m = float(alt_str or 0)
        observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
        # Use current time as center and user-specified duration
        duration_minutes_auto = float(duration_str) if duration_str else 30.0
        current_utc = datetime.now(timezone.utc)
        t0 = ts.utc(current_utc - timedelta(minutes=duration_minutes_auto/2))
        t1 = ts.utc(current_utc + timedelta(minutes=duration_minutes_auto/2))

        # Update scroll bar time range labels based on the duration being used
        scroll_bar_start_label = "-" + str(duration_minutes_auto/2) + " min"
        scroll_bar_end_label = "+" + str(duration_minutes_auto/2) + " min"

        satellite_trajectories, satellite_arc_segments = precompute_trajectories(satellites, observer, ts, sub_x, sub_y, sub_width, sub_height, satellite_labels, update_status_callback)

        # Build satellite pass table (filtered for current visibility + upcoming passes)
        satellite_pass_table = build_satellite_pass_table(satellite_trajectories, satellites, satellite_labels, elevation_mask_deg=float(elevation_mask_str or 10.0), ts=ts, current_satellite_positions=satellite_positions)
        # Final consolidated filtering - replaced last duplicate ~30 lines of filtering logic
        satellite_pass_table = [entry for entry in satellite_pass_table if (
            not filter_text or
            filter_text.lower() in entry['satellite'].name.lower() or
            filter_text in entry['satellite'].model.satnum_str
        ) and (
            not filter_above_alt_text or
            float(satellite_mean_altitudes.get(entry['satellite'], 0.0)) >= float(filter_above_alt_text)
        ) and (
            not filter_below_alt_text or
            float(satellite_mean_altitudes.get(entry['satellite'], 0.0)) <= float(filter_below_alt_text)
        )]
        satellite_pass_table = sort_pass_table(satellite_pass_table, table_sort_keys, table_sort_reverse)

        last_trajectory_update = current_time
        update_status_callback("Trajectories updated")
        # Initialize slider to current time
        fraction = (ts.now().tt - t0.tt) / (t1.tt - t0.tt)
        slider_rect.x = scroll_bar_rect.x + int(fraction * (scroll_bar_rect.width - slider_rect.width))

    # Compute satellite positions at 10 Hz
    if current_time - last_update_time >= update_interval:
        satellite_positions = {}
        if dragging_slider:
            fraction = (slider_rect.x - scroll_bar_rect.x) / (scroll_bar_rect.width - slider_rect.width)
            current_tt = t0.tt + fraction * (t1.tt - t0.tt)
        elif paused:
            current_tt = paused_tt
        else:
            current_tt = ts.now().tt
            # Update slider position to reflect real-time
            if t0 is not None and t1 is not None:
                fraction = (current_tt - t0.tt) / (t1.tt - t0.tt)
                slider_rect.x = scroll_bar_rect.x + int(fraction * (scroll_bar_rect.width - slider_rect.width))
        if selected_satellite:
            if selected_satellite in satellite_trajectories:
                px, py, alt, dist = interpolate_position(satellite_trajectories[selected_satellite], current_tt)
                if px is not None and alt > float(elevation_mask_str or 0):
                    satellite_positions[selected_satellite] = (px, py, alt, dist)
        else:
            for sat in satellites:
                if sat in satellite_trajectories:
                    px, py, alt, dist = interpolate_position(satellite_trajectories[sat], current_tt)
                    if px is not None and alt > float(elevation_mask_str or 0):
                        # Apply filters
                        include_sat = True
                        if filter_text:
                            include_sat = filter_text.lower() in sat.name.lower() or filter_text in sat.model.satnum_str
                        if filter_above_alt_text:
                            try:
                                alt_filter = float(filter_above_alt_text)
                                include_sat = include_sat and satellite_mean_altitudes[sat] >= alt_filter
                            except ValueError:
                                include_sat = False
                        if filter_below_alt_text:
                            try:
                                alt_filter = float(filter_below_alt_text)
                                include_sat = include_sat and satellite_mean_altitudes[sat] <= alt_filter
                            except ValueError:
                                include_sat = False
                        if include_sat:
                            satellite_positions[sat] = (px, py, alt, dist)
        last_update_time = current_time

    # Handle events using modular event system
    # Handle events using modular event system
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif current_mode == "config_options" and focused_field:
                lat_str, lon_str, alt_str, elevation_mask_str = handle_input(event, focused_field, lat_str, lon_str, alt_str, elevation_mask_str, cursor_pos, selection_start)
            elif current_mode == "tracking_vis":
                if filter_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "filter"
                        cursor_pos['filter'] = len(filter_text)
                        selection_start['filter'] = None
                elif filter_above_alt_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "filter_above_alt"
                        cursor_pos['filter_above_alt'] = len(filter_above_alt_text)
                        selection_start['filter_above_alt'] = None
                elif filter_below_alt_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "filter_below_alt"
                        cursor_pos['filter_below_alt'] = len(filter_below_alt_text)
                        selection_start['filter_below_alt'] = None
                elif center_time_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "center_time"
                        cursor_pos['center_time'] = len(center_time_str)
                        selection_start['center_time'] = None
                elif duration_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "duration"
                        cursor_pos['duration'] = len(duration_str)
                        selection_start['duration'] = None
                if focused_field == "filter":
                    field_str = filter_text
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter"] = cursor_pos["filter"] if selection_start["filter"] is None else selection_start["filter"]
                            cursor_pos["filter"] = max(0, cursor_pos["filter"] - 1)
                        else:
                            cursor_pos["filter"] = max(0, cursor_pos["filter"] - 1)
                            selection_start["filter"] = None
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter"] = cursor_pos["filter"] if selection_start["filter"] is None else selection_start["filter"]
                            cursor_pos["filter"] = min(len(field_str), cursor_pos["filter"] + 1)
                        else:
                            cursor_pos["filter"] = min(len(field_str), cursor_pos["filter"] + 1)
                            selection_start["filter"] = None
                    elif event.key == pygame.K_HOME:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter"] = cursor_pos["filter"] if selection_start["filter"] is None else selection_start["filter"]
                        cursor_pos["filter"] = 0
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["filter"] = None
                    elif event.key == pygame.K_END:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter"] = cursor_pos["filter"] if selection_start["filter"] is None else selection_start["filter"]
                        cursor_pos["filter"] = len(field_str)
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["filter"] = None
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        start = min(cursor_pos["filter"], selection_start["filter"]) if selection_start["filter"] is not None else cursor_pos["filter"]
                        end = max(cursor_pos["filter"], selection_start["filter"]) if selection_start["filter"] is not None else cursor_pos["filter"] + 1 if event.key == pygame.K_DELETE else cursor_pos["filter"]
                        if start < end:
                            field_str = field_str[:start] + field_str[end:]
                            cursor_pos["filter"] = start
                            selection_start["filter"] = None
                        elif event.key == pygame.K_BACKSPACE and cursor_pos["filter"] > 0:
                            field_str = field_str[:cursor_pos["filter"] - 1] + field_str[cursor_pos["filter"]:]
                            cursor_pos["filter"] -= 1
                            selection_start["filter"] = None
                    elif event.key == pygame.K_RETURN:
                        focused_field = None
                        selection_start["filter"] = None
                    else:
                        char = event.unicode
                        if char.isalnum() or char in [' ', '-', '_']:
                            start = min(cursor_pos["filter"], selection_start["filter"]) if selection_start["filter"] is not None else cursor_pos["filter"]
                            end = max(cursor_pos["filter"], selection_start["filter"]) if selection_start["filter"] is not None else cursor_pos["filter"]
                            field_str = field_str[:start] + char + field_str[end:]
                            cursor_pos["filter"] += 1
                            selection_start["filter"] = None
                    filter_text = field_str
                elif focused_field == "center_time":
                    field_str = center_time_str
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["center_time"] = cursor_pos["center_time"] if selection_start["center_time"] is None else selection_start["center_time"]
                            cursor_pos["center_time"] = max(0, cursor_pos["center_time"] - 1)
                        else:
                            cursor_pos["center_time"] = max(0, cursor_pos["center_time"] - 1)
                            selection_start["center_time"] = None
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["center_time"] = cursor_pos["center_time"] if selection_start["center_time"] is None else selection_start["center_time"]
                            cursor_pos["center_time"] = min(len(field_str), cursor_pos["center_time"] + 1)
                        else:
                            cursor_pos["center_time"] = min(len(field_str), cursor_pos["center_time"] + 1)
                            selection_start["center_time"] = None
                    elif event.key == pygame.K_HOME:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["center_time"] = cursor_pos["center_time"] if selection_start["center_time"] is None else selection_start["center_time"]
                        cursor_pos["center_time"] = 0
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["center_time"] = None
                    elif event.key == pygame.K_END:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["center_time"] = cursor_pos["center_time"] if selection_start["center_time"] is None else selection_start["center_time"]
                        cursor_pos["center_time"] = len(field_str)
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["center_time"] = None
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        start = min(cursor_pos["center_time"], selection_start["center_time"]) if selection_start["center_time"] is not None else cursor_pos["center_time"]
                        end = max(cursor_pos["center_time"], selection_start["center_time"]) if selection_start["center_time"] is not None else cursor_pos["center_time"] + 1 if event.key == pygame.K_DELETE else cursor_pos["center_time"]
                        if start < end:
                            field_str = field_str[:start] + field_str[end:]
                            cursor_pos["center_time"] = start
                            selection_start["center_time"] = None
                        elif event.key == pygame.K_BACKSPACE and cursor_pos["center_time"] > 0:
                            field_str = field_str[:cursor_pos["center_time"] - 1] + field_str[cursor_pos["center_time"]:]
                            cursor_pos["center_time"] -= 1
                            selection_start["center_time"] = None
                    elif event.key == pygame.K_RETURN:
                        focused_field = None
                        selection_start["center_time"] = None
                    else:
                        char = event.unicode
                        if char.isalnum() or char in ['-', ':', 'T', 'Z', '.']:
                            start = min(cursor_pos["center_time"], selection_start["center_time"]) if selection_start["center_time"] is not None else cursor_pos["center_time"]
                            end = max(cursor_pos["center_time"], selection_start["center_time"]) if selection_start["center_time"] is not None else cursor_pos["center_time"]
                            field_str = field_str[:start] + char + field_str[end:]
                            cursor_pos["center_time"] += 1
                            selection_start["center_time"] = None
                    center_time_str = field_str
                elif focused_field == "duration":
                    field_str = duration_str
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["duration"] = cursor_pos["duration"] if selection_start["duration"] is None else selection_start["duration"]
                            cursor_pos["duration"] = max(0, cursor_pos["duration"] - 1)
                        else:
                            cursor_pos["duration"] = max(0, cursor_pos["duration"] - 1)
                            selection_start["duration"] = None
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["duration"] = cursor_pos["duration"] if selection_start["duration"] is None else selection_start["duration"]
                            cursor_pos["duration"] = min(len(field_str), cursor_pos["duration"] + 1)
                        else:
                            cursor_pos["duration"] = min(len(field_str), cursor_pos["duration"] + 1)
                            selection_start["duration"] = None
                    elif event.key == pygame.K_HOME:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["duration"] = cursor_pos["duration"] if selection_start["duration"] is None else selection_start["duration"]
                        cursor_pos["duration"] = 0
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["duration"] = None
                    elif event.key == pygame.K_END:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["duration"] = cursor_pos["duration"] if selection_start["duration"] is None else selection_start["duration"]
                        cursor_pos["duration"] = len(field_str)
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["duration"] = None
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        start = min(cursor_pos["duration"], selection_start["duration"]) if selection_start["duration"] is not None else cursor_pos["duration"]
                        end = max(cursor_pos["duration"], selection_start["duration"]) if selection_start["duration"] is not None else cursor_pos["duration"] + 1 if event.key == pygame.K_DELETE else cursor_pos["duration"]
                        if start < end:
                            field_str = field_str[:start] + field_str[end:]
                            cursor_pos["duration"] = start
                            selection_start["duration"] = None
                        elif event.key == pygame.K_BACKSPACE and cursor_pos["duration"] > 0:
                            field_str = field_str[:cursor_pos["duration"] - 1] + field_str[cursor_pos["duration"]:]
                            cursor_pos["duration"] -= 1
                            selection_start["duration"] = None
                    elif event.key == pygame.K_RETURN:
                        focused_field = None
                        selection_start["duration"] = None
                    else:
                        char = event.unicode
                        if char.isdigit():
                            start = min(cursor_pos["duration"], selection_start["duration"]) if selection_start["duration"] is not None else cursor_pos["duration"]
                            end = max(cursor_pos["duration"], selection_start["duration"]) if selection_start["duration"] is not None else cursor_pos["duration"]
                            field_str = field_str[:start] + char + field_str[end:]
                            cursor_pos["duration"] += 1
                            selection_start["duration"] = None
                    duration_str = field_str
                elif focused_field == "filter_above_alt":
                    field_str = filter_above_alt_text
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_above_alt"] = cursor_pos["filter_above_alt"] if selection_start["filter_above_alt"] is None else selection_start["filter_above_alt"]
                            cursor_pos["filter_above_alt"] = max(0, cursor_pos["filter_above_alt"] - 1)
                        else:
                            cursor_pos["filter_above_alt"] = max(0, cursor_pos["filter_above_alt"] - 1)
                            selection_start["filter_above_alt"] = None
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_above_alt"] = cursor_pos["filter_above_alt"] if selection_start["filter_above_alt"] is None else selection_start["filter_above_alt"]
                            cursor_pos["filter_above_alt"] = min(len(field_str), cursor_pos["filter_above_alt"] + 1)
                        else:
                            cursor_pos["filter_above_alt"] = min(len(field_str), cursor_pos["filter_above_alt"] + 1)
                            selection_start["filter_above_alt"] = None
                    elif event.key == pygame.K_HOME:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_above_alt"] = cursor_pos["filter_above_alt"] if selection_start["filter_above_alt"] is None else selection_start["filter_above_alt"]
                        cursor_pos["filter_above_alt"] = 0
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["filter_above_alt"] = None
                    elif event.key == pygame.K_END:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_above_alt"] = cursor_pos["filter_above_alt"] if selection_start["filter_above_alt"] is None else selection_start["filter_above_alt"]
                        cursor_pos["filter_above_alt"] = len(field_str)
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["filter_above_alt"] = None
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        start = min(cursor_pos["filter_above_alt"], selection_start["filter_above_alt"]) if selection_start["filter_above_alt"] is not None else cursor_pos["filter_above_alt"]
                        end = max(cursor_pos["filter_above_alt"], selection_start["filter_above_alt"]) if selection_start["filter_above_alt"] is not None else cursor_pos["filter_above_alt"] + 1 if event.key == pygame.K_DELETE else cursor_pos["filter_above_alt"]
                        if start < end:
                            field_str = field_str[:start] + field_str[end:]
                            cursor_pos["filter_above_alt"] = start
                            selection_start["filter_above_alt"] = None
                        elif event.key == pygame.K_BACKSPACE and cursor_pos["filter_above_alt"] > 0:
                            field_str = field_str[:cursor_pos["filter_above_alt"] - 1] + field_str[cursor_pos["filter_above_alt"]:]
                            cursor_pos["filter_above_alt"] -= 1
                            selection_start["filter_above_alt"] = None
                    elif event.key == pygame.K_RETURN:
                        focused_field = None
                        selection_start["filter_above_alt"] = None
                    else:
                        char = event.unicode
                        if char.isdigit() or char in ['.', '-']:
                            start = min(cursor_pos["filter_above_alt"], selection_start["filter_above_alt"]) if selection_start["filter_above_alt"] is not None else cursor_pos["filter_above_alt"]
                            end = max(cursor_pos["filter_above_alt"], selection_start["filter_above_alt"]) if selection_start["filter_above_alt"] is not None else cursor_pos["filter_above_alt"]
                            field_str = field_str[:start] + char + field_str[end:]
                            cursor_pos["filter_above_alt"] += 1
                            selection_start["filter_above_alt"] = None
                    filter_above_alt_text = field_str
                elif focused_field == "filter_below_alt":
                    field_str = filter_below_alt_text
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_below_alt"] = cursor_pos["filter_below_alt"] if selection_start["filter_below_alt"] is None else selection_start["filter_below_alt"]
                            cursor_pos["filter_below_alt"] = max(0, cursor_pos["filter_below_alt"] - 1)
                        else:
                            cursor_pos["filter_below_alt"] = max(0, cursor_pos["filter_below_alt"] - 1)
                            selection_start["filter_below_alt"] = None
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_below_alt"] = cursor_pos["filter_below_alt"] if selection_start["filter_below_alt"] is None else selection_start["filter_below_alt"]
                            cursor_pos["filter_below_alt"] = min(len(field_str), cursor_pos["filter_below_alt"] + 1)
                        else:
                            cursor_pos["filter_below_alt"] = min(len(field_str), cursor_pos["filter_below_alt"] + 1)
                            selection_start["filter_below_alt"] = None
                    elif event.key == pygame.K_HOME:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_below_alt"] = cursor_pos["filter_below_alt"] if selection_start["filter_below_alt"] is None else selection_start["filter_below_alt"]
                        cursor_pos["filter_below_alt"] = 0
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["filter_below_alt"] = None
                    elif event.key == pygame.K_END:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start["filter_below_alt"] = cursor_pos["filter_below_alt"] if selection_start["filter_below_alt"] is None else selection_start["filter_below_alt"]
                        cursor_pos["filter_below_alt"] = len(field_str)
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start["filter_below_alt"] = None
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        start = min(cursor_pos["filter_below_alt"], selection_start["filter_below_alt"]) if selection_start["filter_below_alt"] is not None else cursor_pos["filter_below_alt"]
                        end = max(cursor_pos["filter_below_alt"], selection_start["filter_below_alt"]) if selection_start["filter_below_alt"] is not None else cursor_pos["filter_below_alt"] + 1 if event.key == pygame.K_DELETE else cursor_pos["filter_below_alt"]
                        if start < end:
                            field_str = field_str[:start] + field_str[end:]
                            cursor_pos["filter_below_alt"] = start
                            selection_start["filter_below_alt"] = None
                        elif event.key == pygame.K_BACKSPACE and cursor_pos["filter_below_alt"] > 0:
                            field_str = field_str[:cursor_pos["filter_below_alt"] - 1] + field_str[cursor_pos["filter_below_alt"]:]
                            cursor_pos["filter_below_alt"] -= 1
                            selection_start["filter_below_alt"] = None
                    elif event.key == pygame.K_RETURN:
                        focused_field = None
                        selection_start["filter_below_alt"] = None
                    else:
                        char = event.unicode
                        if char.isdigit() or char in ['.', '-']:
                            start = min(cursor_pos["filter_below_alt"], selection_start["filter_below_alt"]) if selection_start["filter_below_alt"] is not None else cursor_pos["filter_below_alt"]
                            end = max(cursor_pos["filter_below_alt"], selection_start["filter_below_alt"]) if selection_start["filter_below_alt"] is not None else cursor_pos["filter_below_alt"]
                            field_str = field_str[:start] + char + field_str[end:]
                            cursor_pos["filter_below_alt"] += 1
                            selection_start["filter_below_alt"] = None
                    filter_below_alt_text = field_str
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            # Always check main menu buttons first (allows switching back to menu or between modes)
            menu_result = None
            for btn in buttons:
                if btn["rect"].collidepoint(pos):
                    menu_result = btn["mode"]
                    break

            if menu_result:
                from events import handle_main_menu_events
                result = handle_main_menu_events(event, buttons, current_mode)
                if result == "exit":
                    running = False
                    break
                elif result and result != current_mode:
                    current_mode = result
                    # Reset fields when switching modes
                    focused_field = None
                    cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0, "filter": 0, "filter_above_alt": 0, "filter_below_alt": 0, "center_time": 0, "duration": 0}
                    selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None, "filter": None, "filter_above_alt": None, "filter_below_alt": None, "center_time": None, "duration": None}
                    break  # Exit inner loop to process new mode
            elif current_mode is None:
                # Handle main menu button clicks when already in main menu mode
                from events import handle_main_menu_events
                result = handle_main_menu_events(event, buttons, current_mode)
                if result == "exit":
                    running = False
                    break
                elif result and result != current_mode:
                    current_mode = result
                    # Reset fields when switching modes
                    focused_field = None
                    cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0, "filter": 0, "filter_above_alt": 0, "filter_below_alt": 0, "center_time": 0, "duration": 0}
                    selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None, "filter": None, "filter_above_alt": None, "filter_below_alt": None, "center_time": None, "duration": None}
                    break  # Exit inner loop to process new mode

            # Handle mode-specific events
            elif current_mode == "sensor_calib":
                # Handle sensor calibration events using modular handler
                from camera_manager import handle_sensor_calib_events
                sensor_result = handle_sensor_calib_events(event, pos, sub_x, sub_y, sub_width, sub_height,
                                                         camera1_connected, camera2_connected)
                if sensor_result:
                    # Update camera connection states based on result
                    if "action" in sensor_result:
                        if sensor_result["action"] == "connect_camera1":
                            camera1_connected = True
                        elif sensor_result["action"] == "disconnect_camera1":
                            camera1_connected = False
                        elif sensor_result["action"] == "connect_camera2":
                            camera2_connected = True
                        elif sensor_result["action"] == "disconnect_camera2":
                            camera2_connected = False

                    # Add status message if provided
                    if "status" in sensor_result:
                        status_messages.append(sensor_result["status"])
                        status_messages[:] = status_messages[-4:]  # Keep last 4 messages
            elif current_mode == "config_options":
                from events import handle_config_events
                modified_vars = handle_config_events(event, pos, input_rects, save_button, load_button,
                                                     lat_str, lon_str, alt_str, elevation_mask_str,
                                                     focused_field, cursor_pos, selection_start,
                                                     button_states)
                if modified_vars:
                    for key, value in modified_vars.items():
                        locals()[key] = value  # Update local variables
                    status_messages.append("Configuration updated")

            # Handle tracking_vis button clicks
            elif current_mode == "tracking_vis":
                if clear_filters_button.collidepoint(pos):
                    button_states["clear_filters"]["clicked"] = True
                    filter_text = ""
                    filter_above_alt_text = ""
                    filter_below_alt_text = ""
                    cursor_pos["filter"] = 0
                    cursor_pos["filter_above_alt"] = 0
                    cursor_pos["filter_below_alt"] = 0
                    selection_start["filter"] = None
                    selection_start["filter_above_alt"] = None
                    selection_start["filter_below_alt"] = None
                    selected_satellite = None
                    # Re-build satellite pass table to remove filters
                    satellite_pass_table = build_satellite_pass_table(satellite_trajectories, satellites, satellite_labels, elevation_mask_deg=float(elevation_mask_str or 10.0), ts=ts, current_satellite_positions=satellite_positions)
                    satellite_pass_table = sort_pass_table(satellite_pass_table, table_sort_keys, table_sort_reverse)
                elif pause_button.collidepoint(pos):
                    button_states["pause"]["clicked"] = True
                    paused = True
                    paused_tt = ts.now().tt
                elif play_button.collidepoint(pos):
                    button_states["play"]["clicked"] = True
                    paused = False
                    paused_tt = None
                elif scroll_bar_rect.collidepoint(pos):
                    dragging_slider = True
                    scroll_start = pos[0]
                    slider_rect.x = max(scroll_bar_rect.x, min(pos[0] - slider_rect.width // 2, scroll_bar_rect.x + scroll_bar_rect.width - slider_rect.width))
                elif filter_rect.collidepoint(pos):
                    focused_field = 'filter'
                    cursor_pos['filter'] = len(filter_text)
                    selection_start['filter'] = None
                elif filter_above_alt_rect.collidepoint(pos):
                    focused_field = 'filter_above_alt'
                    cursor_pos['filter_above_alt'] = len(filter_above_alt_text)
                    selection_start['filter_above_alt'] = None
                elif filter_below_alt_rect.collidepoint(pos):
                    focused_field = 'filter_below_alt'
                    cursor_pos['filter_below_alt'] = len(filter_below_alt_text)
                    selection_start['filter_below_alt'] = None
                elif center_time_rect.collidepoint(pos):
                    focused_field = 'center_time'
                    cursor_pos['center_time'] = len(center_time_str)
                    selection_start['center_time'] = None
                elif duration_rect.collidepoint(pos):
                    focused_field = 'duration'
                    cursor_pos['duration'] = len(duration_str)
                    selection_start['duration'] = None
                elif recompute_button.collidepoint(pos):
                    recompute_triggered = True
                    button_states["recompute"]["clicked"] = True
                elif reset_button.collidepoint(pos):
                    button_states["reset"]["clicked"] = True
                    # Reset to current UTC time and default 30 minute duration
                    center_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    duration_str = "30"
                    cursor_pos["center_time"] = len(center_time_str)
                    cursor_pos["duration"] = len(duration_str)
                    selection_start["center_time"] = None
                    selection_start["duration"] = None
                    status_messages.append("Reset to current time (30 min duration)")
                    recompute_triggered = True  # Also trigger recomputation
                else:
                    # Check pass table clicks
                    if hasattr(pass_table_clickable_areas, '__iter__'):
                        for area_type, index, rect in pass_table_clickable_areas:
                            if rect.collidepoint(pos):
                                if area_type == 'row' and satellite_pass_table and index < len(satellite_pass_table):
                                    # Select satellite from pass table
                                    selected_satellite = satellite_pass_table[index]['satellite']
                                    status_messages.append(f"Selected: {satellite_pass_table[index]['name']}")
                                elif area_type == 'header':
                                    # Sort by column
                                    if table_sort_keys and index < len(table_sort_keys):
                                        if table_sort_keys[index]:
                                            table_sort_reverse[index] = not table_sort_reverse[index]
                                        else:
                                            table_sort_keys = [False] * len(table_sort_keys)
                                            table_sort_keys[index] = True
                                            table_sort_reverse = [False] * len(table_sort_reverse)
                                            table_sort_reverse[index] = True
                                        # Re-sort the table
                                        satellite_pass_table = sort_pass_table(satellite_pass_table, table_sort_keys, table_sort_reverse)
                                break

                    # Satellite selection - if no table click
                    deselected = False
                    if selected_satellite:
                        px, py, _, _ = satellite_positions.get(selected_satellite, (0, 0, 0, 0))
                        if math.hypot(px - pos[0], py - pos[1]) < 10:
                            selected_satellite = None
                            deselected = True
                    if not deselected:
                        for sat, (px, py, _, _) in satellite_positions.items():
                            if math.hypot(px - pos[0], py - pos[1]) < 10:
                                selected_satellite = sat
                                break
        elif event.type == pygame.MOUSEMOTION:
            if current_mode is None:
                # Handle main menu hover states
                for btn in buttons:
                    button_states[btn["mode"]]["hover"] = btn["rect"].collidepoint(event.pos)
            elif current_mode == "config_options":
                button_states["save"]["hover"] = save_button.collidepoint(event.pos)
                button_states["load"]["hover"] = load_button.collidepoint(event.pos)
            elif current_mode == "sensor_calib":
                # Camera slider hover states handled by modular camera code
                from camera_manager import handle_sensor_calib_events
                sensor_result = handle_sensor_calib_events(event, pos, sub_x, sub_y, sub_width, sub_height,
                                                         camera1_connected, camera2_connected)
            elif current_mode == "tracking_vis":
                button_states["clear_filters"]["hover"] = clear_filters_button.collidepoint(event.pos)
                button_states["recompute"]["hover"] = recompute_button.collidepoint(event.pos)
                button_states["reset"]["hover"] = reset_button.collidepoint(event.pos)
                button_states["pause"]["hover"] = pause_button.collidepoint(event.pos)
                button_states["play"]["hover"] = play_button.collidepoint(event.pos)
                if dragging_slider:
                    slider_rect.x = max(scroll_bar_rect.x, min(event.pos[0] - slider_rect.width // 2, scroll_bar_rect.x + scroll_bar_rect.width - slider_rect.width))
                hovered_satellite = None
                for sat, (px, py, _, _) in satellite_positions.items():
                    if math.hypot(px - event.pos[0], py - event.pos[1]) < 10:
                        hovered_satellite = sat
                        break

        elif event.type == pygame.MOUSEBUTTONUP:
            # Reset all button clicked states - essential for UI feedback
            for btn in buttons:
                button_states[btn["mode"]]["clicked"] = False
            if current_mode is None:
                pass
            elif current_mode == "config_options":
                button_states["save"]["clicked"] = False
                button_states["load"]["clicked"] = False
            elif current_mode == "sensor_calib":
                # Handle sensor calibration events using modular handler
                from camera_manager import handle_sensor_calib_events
                sensor_result = handle_sensor_calib_events(event, pos, sub_x, sub_y, sub_width, sub_height,
                                                         camera1_connected, camera2_connected)
            elif current_mode == "tracking_vis":
                button_states["clear_filters"]["clicked"] = False
                button_states["reset"]["clicked"] = False
                button_states["pause"]["clicked"] = False
                button_states["play"]["clicked"] = False
                # Reset dragging state
                if dragging_slider and paused:
                    fraction = (slider_rect.x - scroll_bar_rect.x) / (scroll_bar_rect.width - slider_rect.width)
                    paused_tt = t0.tt + fraction * (t1.tt - t0.tt)
                dragging_slider = False
                scroll_start = None



    # Render continuously
    menu_screen.fill((30, 30, 30), (0, 0, MENU_WIDTH, total_height))  # Dark theme menu background
    if bg_image_menu:
        # Blit rotated negative image if mouse is over the image area, normal otherwise
        if image_rect and image_rect.collidepoint(mouse_pos):
            # Rotate the negative image slowly
            rotated_image = pygame.transform.rotate(negative_image, rotation_angle)
            # Center the rotated image
            rotated_rect = rotated_image.get_rect(center=image_rect.center)
            menu_screen.blit(rotated_image, rotated_rect.topleft)
            rotation_angle = (rotation_angle + 1) % 360  # Increment angle, reset at 360
        else:
            menu_screen.blit(bg_image_menu, image_rect.topleft)

    for btn in buttons:
        draw_button(menu_screen, btn["rect"], btn["text"], button_states[btn["mode"]])
    # Render status messages each frame
    status_messages = status_messages[-4:]  # Keep last 4 messages
    for i, msg in enumerate(status_messages):
        status_render = status_font.render(msg, True, (255, 255, 255))  # White text for dark theme
        menu_screen.blit(status_render, (10, status_y_start + i * 14))

    cx = sub_x + sub_width // 2
    cy = sub_y + sub_height // 2
    if current_mode == "config_options":
        draw_polar_plot(menu_screen, sub_x, sub_y, sub_width, sub_height, input_rects, lat_str, lon_str, alt_str, elevation_mask_str, font, focused_field, cursor_pos, selection_start, save_button, load_button, button_states)
    elif current_mode == "tracking_vis" and tle_loaded:
        menu_screen.fill((0, 0, 0), (sub_x, sub_y, sub_width, sub_height))  # Clear the subplot area with black
        draw_polar_plot(menu_screen, sub_x, sub_y, sub_width, sub_height, input_rects, lat_str, lon_str, alt_str, elevation_mask_str, font, focused_field, cursor_pos, selection_start, save_button, load_button, button_states, ts, current_tt, satellite_trajectories, satellite_positions, satellite_labels, satellite_mean_altitudes, selected_satellite, satellite_arc_segments, filter_text, filter_above_alt_text, legend_x, legend_y, clear_filters_button, tle_loaded=tle_loaded, obs_lat=float(lat_str or 0), obs_lon=float(lon_str or 0), obs_alt=float(alt_str or 0))
        draw_satellites(menu_screen, satellite_positions, satellite_labels, satellite_mean_altitudes, hovered_satellite, selected_satellite, cx, cy)
        draw_filters(menu_screen, filter_rect, filter_above_alt_rect, filter_below_alt_rect, filter_text, filter_above_alt_text, filter_below_alt_text, focused_field, cursor_pos, selection_start, small_font)
        draw_legend(menu_screen, legend_x, legend_y, small_font)
        draw_details(menu_screen, hovered_satellite, selected_satellite, satellite_mean_altitudes, sub_x, sub_y, sub_width, sub_height, small_font, satellite_perigee, satellite_apogee, satellite_positions)
        draw_time_display(menu_screen, sub_x, sub_y, sub_height, small_font)
        draw_satellite_count(menu_screen, sub_x, sub_y + 240, satellite_positions, small_font)
        # Modular pass table filtering and sorting - replaced ~30 lines of inline filtering logic
        from visuals import filter_and_sort_pass_table
        sorted_filtered_pass_table = filter_and_sort_pass_table(satellite_pass_table, filter_text, filter_above_alt_text,
                                                                filter_below_alt_text, satellite_mean_altitudes,
                                                                table_sort_keys, table_sort_reverse)

        # Draw satellite pass table
        pass_table_clickable_areas = draw_satellite_pass_table(menu_screen, sub_x, sub_y, sub_height, sorted_filtered_pass_table, selected_satellite, small_font, table_sort_keys, table_sort_reverse)
        draw_button(menu_screen, clear_filters_button, "Clear Filters", button_states["clear_filters"])
        draw_button(menu_screen, pause_button, "Pause", button_states["pause"])
        draw_button(menu_screen, play_button, "Play", button_states["play"])
        draw_scroll_bar(menu_screen, scroll_bar_rect, slider_rect, small_font)
        draw_scroll_time_display(menu_screen, sub_x, sub_y, sub_width, sub_height, current_tt, ts, small_font)
        # Draw Recompute Button
        draw_button(menu_screen, recompute_button, "Update Traj", button_states["recompute"])
        # Draw Reset Button
        draw_button(menu_screen, reset_button, "Reset", button_states["reset"])
        # Draw Center Time Input
        pygame.draw.rect(menu_screen, (255, 255, 255), center_time_rect)
        if focused_field == 'center_time':
            pygame.draw.rect(menu_screen, (0, 0, 255), center_time_rect, 2)
        text_surface = small_font.render(center_time_str, True, (0, 0, 0))
        menu_screen.blit(text_surface, (center_time_rect.x + 5, center_time_rect.y + 5))
        if focused_field == 'center_time':
            text_width, _ = small_font.size(center_time_str[:cursor_pos['center_time']])
            pygame.draw.line(menu_screen, (0, 0, 255), (center_time_rect.x + 5 + text_width, center_time_rect.y + 5), (center_time_rect.x + 5 + text_width, center_time_rect.y + 25), 2)
        # Label for Center Time
        label = small_font.render("Center Time:", True, (255, 255, 255))
        menu_screen.blit(label, (center_time_rect.x, center_time_rect.y - 15))
        # Draw Duration Input
        pygame.draw.rect(menu_screen, (255, 255, 255), duration_rect)
        if focused_field == 'duration':
            pygame.draw.rect(menu_screen, (0, 0, 255), duration_rect, 2)
        text_surface = small_font.render(duration_str, True, (0, 0, 0))
        menu_screen.blit(text_surface, (duration_rect.x + 5, duration_rect.y + 5))
        if focused_field == 'duration':
            text_width, _ = small_font.size(duration_str[:cursor_pos['duration']])
            pygame.draw.line(menu_screen, (0, 0, 255), (duration_rect.x + 5 + text_width, duration_rect.y + 5), (duration_rect.x + 5 + text_width, duration_rect.y + 25), 2)
        # Label for Duration
        label = small_font.render("Duration (min.):", True, (255, 255, 255))
        menu_screen.blit(label, (duration_rect.x, duration_rect.y - 15))
    elif current_mode == "sensor_calib":
        # Modular camera display rendering - replaced ~200 lines of inline camera rendering code
        from camera_manager import render_sensor_calibration
        render_sensor_calibration(menu_screen, sub_x, sub_y, sub_width, sub_height,
                               camera1_connected, camera2_connected,
                               camera1_name, camera2_name)

        # Modular camera slider controls - positioned within camera display area
        from camera_manager import render_camera_sliders
        render_camera_sliders(menu_screen, tiny_font, sub_x, sub_y, sub_width, sub_height)

        # Modular ROI controls for both cameras - replaced ~100+ lines of inline ROI control code
        from camera_manager import render_camera_roi_controls
        render_camera_roi_controls(menu_screen, tiny_font)

        # Modular camera interface completion - replaced ~40 lines of inline combined view and status code
        from camera_manager import render_camera_interface_completion
        render_camera_interface_completion(menu_screen, sub_x, sub_y, sub_width, sub_height,
                                        small_font, tiny_font)
    elif current_mode == "joystick_loop":
        sub_rect = (sub_x, sub_y, sub_width, sub_height)
        menu_screen.fill((100, 100, 100), sub_rect)
        # Add manual joystick loop code here later
    elif current_mode == "post_process":
        sub_rect = (sub_x, sub_y, sub_width, sub_height)
        menu_screen.fill((150, 150, 150), sub_rect)
        # Add post-processing tool code here later
    elif current_mode == "author_info":
        sub_rect = (sub_x, sub_y, sub_width, sub_height)
        if author_bg:
            scaled_bg = pygame.transform.scale(author_bg, (sub_width, sub_height))
            menu_screen.blit(scaled_bg, (sub_x, sub_y))
        else:
            menu_screen.fill((0, 0, 0), sub_rect)
        text1 = large_font.render("Starlink-1060", True, (255, 255, 255))
        menu_screen.blit(text1, (sub_x + 10, sub_y + 10))
        contact_text = "Jonathan Nikkel - @NikkelJonathan"
        text2 = large_font.render(contact_text, True, (255, 255, 255))
        menu_screen.blit(text2, (sub_x + 10, sub_y + 50))

    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS for better responsiveness

pygame.quit()
