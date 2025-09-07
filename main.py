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
from tracking_visuals import draw_polar_plot, draw_satellites, draw_legend, draw_details, draw_filters, draw_time_display, draw_satellite_count, draw_scroll_bar, draw_scroll_time_display, draw_satellite_pass_table
from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
from camera_manager import camera_manager, render_sensor_calibration, handle_sensor_calib_events, render_camera_sliders, render_camera_roi_controls, render_combined_view_controls, update_camera_frames_from_buffers
# Camera button initialization is now handled internally by camera_manager
from events import *

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Update Intervals (seconds)
POSITION_UPDATE_INTERVAL = 0.1  # 10 Hz
TRAJECTORY_UPDATE_INTERVAL = 900  # 15 minutes

# Cache Settings
CACHE_FILENAME = 'tle_cache.tle'
CACHE_AGE_LIMIT_HOURS = 24
CACHE_AGE_LIMIT_SECONDS = CACHE_AGE_LIMIT_HOURS * 3600

# TLE API Configuration
TLE_API_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'

# Astrodynamics Constants
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14  # m^3/s^2
EARTH_RADIUS_KM = 6371 # Mean radius
SECONDS_PER_HOUR = 3600
WINDOW_POSITION = "0,0"



# ==============================================================================
# END CONSTANTS
# ==============================================================================

# Initialize Pygame and set up the display
from display import DisplaySetup
display = DisplaySetup()

# Enumerate cameras if available
camera_infos = []
camera1_name = "Camera 1"
camera2_name = "Camera 2"

# Update camera names with actual camera names if available
if ASI_AVAILABLE:
    try:
        camera_names = camera_manager.get_camera_names()
        if len(camera_names) > 0:
            camera1_name = camera_names[0] if len(camera_names) > 0 else "Camera 1"
        if len(camera_names) > 1:
            camera2_name = camera_names[1] if len(camera_names) > 1 else "Camera 2"
        print(f"Camera names updated: Camera 1='{camera1_name}', Camera 2='{camera2_name}'")
    except Exception as e:
        print(f"Failed to get camera names: {e}")

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

current_mode = None
clock = pygame.time.Clock()

# For author info
author_bg = None
try:
    author_bg = pygame.image.load('lucky.jpg')
except pygame.error:
    pass

# Initial render of main menu
status_messages = ["Starting TLE process..."]
display.render_initial_menu(status_messages)
print(f"Debug: Status - {'Starting TLE process...'}")

# Cache file management
cache_file = "tle_cache.tle"
cache_age_limit = 24 * 3600  # 24 hours in seconds

# Load satellite data using refactored module
tle_loaded = False
satellites = []
ts = load.timescale()

# Define status update callback for satellite loading
def update_status_callback(message):
    status_messages.append(message)
    display.menu_screen.fill(display.COLOR_BACKGROUND_DARK, (0, 0, display.MENU_WIDTH, display.total_height))  # Dark theme menu background
    for btn in display.buttons:
        draw_button(display.menu_screen, btn["rect"], btn["text"], display.button_states[btn["mode"]])
    if display.bg_image_menu:
        display.menu_screen.blit(display.bg_image_menu, display.image_rect.topleft if display.image_rect else ((display.MENU_WIDTH - 160) // 2, display.image_y))
    for i, msg in enumerate(status_messages[-4:]):
        status_render = display.status_font.render(msg, True, display.COLOR_TEXT_WHITE)
        display.menu_screen.blit(status_render, (10, display.status_y_start + i * 14))
    pygame.display.flip()
    pygame.time.wait(50)  # Brief pause to ensure display update
    print(f"Debug: Status - {message}")

try:
    satellites, tle_loaded = load_satellite_data(
        cache_file=cache_file,
        cache_age_limit_seconds=cache_age_limit,
        update_status_callback=update_status_callback
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

# ==============================================================================
# MAIN PROGRAM LOOP
# ==============================================================================

running = True
while running:
    current_time = time.time()
    mouse_pos = pygame.mouse.get_pos()


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
            satellite_trajectories, satellite_arc_segments = precompute_trajectories(satellites, observer, ts, display.sub_x, display.sub_y, display.sub_width, display.sub_height, satellite_labels, update_status_callback, center_time, duration_minutes)

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
            display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))
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

        satellite_trajectories, satellite_arc_segments = precompute_trajectories(satellites, observer, ts, display.sub_x, display.sub_y, display.sub_width, display.sub_height, satellite_labels, update_status_callback)

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
        display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))

    # Compute satellite positions at 10 Hz
    if current_time - last_update_time >= update_interval:
        satellite_positions = {}
        if dragging_slider:
            fraction = (display.slider_rect.x - display.scroll_bar_rect.x) / (display.scroll_bar_rect.width - display.slider_rect.width)
            current_tt = t0.tt + fraction * (t1.tt - t0.tt)
        elif paused:
            current_tt = paused_tt
        else:
            current_tt = ts.now().tt
            # Update slider position to reflect real-time
            if t0 is not None and t1 is not None:
                fraction = (current_tt - t0.tt) / (t1.tt - t0.tt)
                display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))
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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif current_mode == "config_options" and focused_field:
                lat_str, lon_str, alt_str, elevation_mask_str = handle_input(event, focused_field, lat_str, lon_str, alt_str, elevation_mask_str, cursor_pos, selection_start)
            elif current_mode == "tracking_vis":
                if display.filter_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "filter"
                        cursor_pos['filter'] = len(filter_text)
                        selection_start['filter'] = None
                elif display.filter_above_alt_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "filter_above_alt"
                        cursor_pos['filter_above_alt'] = len(filter_above_alt_text)
                        selection_start['filter_above_alt'] = None
                elif display.filter_below_alt_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "filter_below_alt"
                        cursor_pos['filter_below_alt'] = len(filter_below_alt_text)
                        selection_start['filter_below_alt'] = None
                elif display.center_time_rect.collidepoint(mouse_pos):
                    if event.key != pygame.K_TAB:
                        focused_field = "center_time"
                        cursor_pos['center_time'] = len(center_time_str)
                        selection_start['center_time'] = None
                elif display.duration_rect.collidepoint(mouse_pos):
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
            for btn in display.buttons:
                if btn["rect"].collidepoint(pos):
                    menu_result = btn["mode"]
                    break

            if menu_result:
                from events import handle_main_menu_events
                result = handle_main_menu_events(event, display.buttons, current_mode)
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
                result = handle_main_menu_events(event, display.buttons, current_mode)
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
                handle_sensor_calib_events(event, pos, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                                         camera_manager.get_camera(0).connected, camera_manager.get_camera(1).connected,
                                         update_status_callback)
            elif current_mode == "config_options":
                from events import handle_config_events
                modified_vars = handle_config_events(event, pos, display.input_rects, display.save_button, display.load_button,
                                                     lat_str, lon_str, alt_str, elevation_mask_str,
                                                     focused_field, cursor_pos, selection_start,
                                                     button_states)
                if modified_vars:
                    for key, value in modified_vars.items():
                        locals()[key] = value  # Update local variables
                    status_messages.append("Configuration updated")

            # Handle tracking_vis button clicks
            elif current_mode == "tracking_vis":
                if display.clear_filters_button.collidepoint(pos):
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
                elif display.pause_button.collidepoint(pos):
                    button_states["pause"]["clicked"] = True
                    paused = True
                    paused_tt = ts.now().tt
                elif display.play_button.collidepoint(pos):
                    button_states["play"]["clicked"] = True
                    paused = False
                    paused_tt = None
                elif display.scroll_bar_rect.collidepoint(pos):
                    dragging_slider = True
                    scroll_start = pos[0]
                    display.slider_rect.x = max(display.scroll_bar_rect.x, min(pos[0] - display.slider_rect.width // 2, display.scroll_bar_rect.x + display.scroll_bar_rect.width - display.slider_rect.width))
                elif display.filter_rect.collidepoint(pos):
                    focused_field = 'filter'
                    cursor_pos['filter'] = len(filter_text)
                    selection_start['filter'] = None
                elif display.filter_above_alt_rect.collidepoint(pos):
                    focused_field = 'filter_above_alt'
                    cursor_pos['filter_above_alt'] = len(filter_above_alt_text)
                    selection_start['filter_above_alt'] = None
                elif display.filter_below_alt_rect.collidepoint(pos):
                    focused_field = 'filter_below_alt'
                    cursor_pos['filter_below_alt'] = len(filter_below_alt_text)
                    selection_start['filter_below_alt'] = None
                elif display.center_time_rect.collidepoint(pos):
                    focused_field = 'center_time'
                    cursor_pos['center_time'] = len(center_time_str)
                    selection_start['center_time'] = None
                elif display.duration_rect.collidepoint(pos):
                    focused_field = 'duration'
                    cursor_pos['duration'] = len(duration_str)
                    selection_start['duration'] = None
                elif display.recompute_button.collidepoint(pos):
                    recompute_triggered = True
                    button_states["recompute"]["clicked"] = True
                elif display.reset_button.collidepoint(pos):
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
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
        elif event.type == pygame.MOUSEMOTION:
            if current_mode is None:
                # Handle main menu hover states
                for btn in display.buttons:
                    display.button_states[btn["mode"]]["hover"] = btn["rect"].collidepoint(event.pos)
            elif current_mode == "config_options":
                button_states["save"]["hover"] = display.save_button.collidepoint(event.pos)
                button_states["load"]["hover"] = display.load_button.collidepoint(event.pos)
            elif current_mode == "sensor_calib":
                # Camera slider hover states handled by modular camera code
                from camera_manager import handle_sensor_calib_events
                handle_sensor_calib_events(event, pos, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                                         camera_manager.get_camera(0).connected, camera_manager.get_camera(1).connected,
                                         update_status_callback)
            elif current_mode == "tracking_vis":
                button_states["clear_filters"]["hover"] = display.clear_filters_button.collidepoint(event.pos)
                button_states["recompute"]["hover"] = display.recompute_button.collidepoint(event.pos)
                button_states["reset"]["hover"] = display.reset_button.collidepoint(event.pos)
                button_states["pause"]["hover"] = display.pause_button.collidepoint(event.pos)
                button_states["play"]["hover"] = display.play_button.collidepoint(event.pos)
                if dragging_slider:
                    display.slider_rect.x = max(display.scroll_bar_rect.x, min(event.pos[0] - display.slider_rect.width // 2, display.scroll_bar_rect.x + display.scroll_bar_rect.width - display.slider_rect.width))
                hovered_satellite = None
                for sat, (px, py, _, _) in satellite_positions.items():
                    if math.hypot(px - event.pos[0], py - event.pos[1]) < 10:
                        hovered_satellite = sat
                        break

        elif event.type == pygame.MOUSEBUTTONUP:
            # Reset all button clicked states - essential for UI feedback
            for btn in display.buttons:
                display.button_states[btn["mode"]]["clicked"] = False
            if current_mode is None:
                pass
            elif current_mode == "config_options":
                display.button_states["save"]["clicked"] = False
                display.button_states["load"]["clicked"] = False
            elif current_mode == "sensor_calib":
                # Handle sensor calibration events using modular handler
                from camera_manager import handle_sensor_calib_events
                handle_sensor_calib_events(event, pos, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                                         camera_manager.get_camera(0).connected, camera_manager.get_camera(1).connected,
                                         update_status_callback)
            elif current_mode == "tracking_vis":
                button_states["clear_filters"]["clicked"] = False
                button_states["reset"]["clicked"] = False
                button_states["pause"]["clicked"] = False
                button_states["play"]["clicked"] = False
                # Reset dragging state
                if dragging_slider and paused:
                    fraction = (display.slider_rect.x - display.scroll_bar_rect.x) / (display.scroll_bar_rect.width - display.slider_rect.width)
                    paused_tt = t0.tt + fraction * (t1.tt - t0.tt)
                dragging_slider = False
                scroll_start = None



    # Render continuously
    display.menu_screen.fill(display.COLOR_BACKGROUND_DARK, (0, 0, display.MENU_WIDTH, display.total_height))  # Dark theme menu background
    if display.bg_image_menu:
        # Update background rotation
        display.update_background_rotation()
        # Get the appropriate image
        image_tool_use = display.get_rotated_image(display.image_rect)
        if display.image_rect:
            display.menu_screen.blit(image_tool_use, display.image_rect.topleft)

    for btn in display.buttons:
        draw_button(display.menu_screen, btn["rect"], btn["text"], display.button_states[btn["mode"]])
    # Render status messages each frame
    status_messages = status_messages[-4:]  # Keep last 4 messages
    for i, msg in enumerate(status_messages):
        status_render = display.status_font.render(msg, True, display.COLOR_TEXT_WHITE)  # White text for dark theme
        display.menu_screen.blit(status_render, (10, display.status_y_start + i * 14))

    cx = display.sub_x + display.sub_width // 2
    cy = display.sub_y + display.sub_height // 2
    if current_mode == "config_options":
        draw_polar_plot(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height, display.input_rects, lat_str, lon_str, alt_str, elevation_mask_str, display.font, focused_field, cursor_pos, selection_start, display.save_button, display.load_button, display.button_states)
    elif current_mode == "tracking_vis" and tle_loaded:
        display.menu_screen.fill((0, 0, 0), (display.sub_x, display.sub_y, display.sub_width, display.sub_height))  # Clear the subplot area with black
        draw_polar_plot(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height, display.input_rects, lat_str, lon_str, alt_str, elevation_mask_str, display.font, focused_field, cursor_pos, selection_start, display.save_button, display.load_button, display.button_states, ts, current_tt, satellite_trajectories, satellite_positions, satellite_labels, satellite_mean_altitudes, selected_satellite, satellite_arc_segments, filter_text, filter_above_alt_text, display.legend_x, display.legend_y, display.clear_filters_button, tle_loaded=tle_loaded, obs_lat=float(lat_str or 0), obs_lon=float(lon_str or 0), obs_alt=float(alt_str or 0))
        draw_satellites(display.menu_screen, satellite_positions, satellite_labels, satellite_mean_altitudes, hovered_satellite, selected_satellite, cx, cy)
        draw_filters(display.menu_screen, display.filter_rect, display.filter_above_alt_rect, display.filter_below_alt_rect, filter_text, filter_above_alt_text, filter_below_alt_text, focused_field, cursor_pos, selection_start, display.small_font)
        draw_legend(display.menu_screen, display.legend_x, display.legend_y, display.small_font)
        draw_details(display.menu_screen, hovered_satellite, selected_satellite, satellite_mean_altitudes, display.sub_x, display.sub_y, display.sub_width, display.sub_height, display.small_font, satellite_perigee, satellite_apogee, satellite_positions)
        draw_time_display(display.menu_screen, display.sub_x, display.sub_y, display.sub_height, display.small_font)
        draw_satellite_count(display.menu_screen, display.sub_x, display.sub_y + 240, satellite_positions, display.small_font)
        # Modular pass table filtering and sorting - replaced ~30 lines of inline filtering logic
        from tracking_visuals import filter_and_sort_pass_table
        sorted_filtered_pass_table = filter_and_sort_pass_table(satellite_pass_table, filter_text, filter_above_alt_text,
                                                                filter_below_alt_text, satellite_mean_altitudes,
                                                                table_sort_keys, table_sort_reverse)

        # Draw satellite pass table
        pass_table_clickable_areas = draw_satellite_pass_table(display.menu_screen, display.sub_x, display.sub_y, display.sub_height, sorted_filtered_pass_table, selected_satellite, display.small_font, table_sort_keys, table_sort_reverse)
        draw_button(display.menu_screen, display.clear_filters_button, "Clear Filters", display.button_states["clear_filters"])
        draw_button(display.menu_screen, display.pause_button, "Pause", display.button_states["pause"])
        draw_button(display.menu_screen, display.play_button, "Play", display.button_states["play"])
        draw_scroll_bar(display.menu_screen, display.scroll_bar_rect, display.slider_rect, display.small_font)
        draw_scroll_time_display(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height, current_tt, ts, display.small_font)
        # Draw Recompute Button
        draw_button(display.menu_screen, display.recompute_button, "Update Traj", display.button_states["recompute"])
        # Draw Reset Button
        draw_button(display.menu_screen, display.reset_button, "Reset", display.button_states["reset"])
        # Draw Center Time Input
        pygame.draw.rect(display.menu_screen, (255, 255, 255), display.center_time_rect)
        if focused_field == 'center_time':
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.center_time_rect, 2)
        text_surface = display.small_font.render(center_time_str, True, (0, 0, 0))
        display.menu_screen.blit(text_surface, (display.center_time_rect.x + 5, display.center_time_rect.y + 5))
        if focused_field == 'center_time':
            text_width, _ = display.small_font.size(center_time_str[:cursor_pos['center_time']])
            pygame.draw.line(display.menu_screen, (0, 0, 255), (display.center_time_rect.x + 5 + text_width, display.center_time_rect.y + 5), (display.center_time_rect.x + 5 + text_width, display.center_time_rect.y + 25), 2)
        # Label for Center Time
        label = display.small_font.render("Center Time:", True, (255, 255, 255))
        display.menu_screen.blit(label, (display.center_time_rect.x, display.center_time_rect.y - 15))
        # Draw Duration Input
        pygame.draw.rect(display.menu_screen, (255, 255, 255), display.duration_rect)
        if focused_field == 'duration':
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.duration_rect, 2)
        text_surface = display.small_font.render(duration_str, True, (0, 0, 0))
        display.menu_screen.blit(text_surface, (display.duration_rect.x + 5, display.duration_rect.y + 5))
        if focused_field == 'duration':
            text_width, _ = display.small_font.size(duration_str[:cursor_pos['duration']])
            pygame.draw.line(display.menu_screen, (0, 0, 255), (display.duration_rect.x + 5 + text_width, display.duration_rect.y + 5), (display.duration_rect.x + 5 + text_width, display.duration_rect.y + 25), 2)
        # Label for Duration
        label = display.small_font.render("Duration (min.):", True, (255, 255, 255))
        display.menu_screen.blit(label, (display.duration_rect.x, display.duration_rect.y - 15))
    elif current_mode == "sensor_calib":
        # Modular camera display rendering - replaced ~200 lines of inline camera rendering code
        from camera_manager import render_sensor_calibration
        render_sensor_calibration(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                               camera_manager.get_camera(0).connected, camera_manager.get_camera(1).connected,
                               camera1_name, camera2_name)

        # Modular camera slider controls - positioned within camera display area
        from camera_manager import render_camera_sliders
        render_camera_sliders(display.menu_screen, display.tiny_font, display.sub_x, display.sub_y, display.sub_width, display.sub_height)

        # Modular ROI controls for both cameras - replaced ~100+ lines of inline ROI control code
        from camera_manager import render_camera_roi_controls
        render_camera_roi_controls(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height)

        # Modular camera interface completion - replaced ~40 lines of inline combined view and status code
        from camera_manager import render_combined_view_controls
        # Initialize camera control rects with proper dimensions
        camera_manager._initialize_control_rects(display.menu_screen, display.total_width, display.total_height)
        render_combined_view_controls(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                                        display.small_font, display.tiny_font)
    elif current_mode == "joystick_loop":
        sub_rect = (display.sub_x, display.sub_y, display.sub_width, display.sub_height)
        display.menu_screen.fill((100, 100, 100), sub_rect)
        # Add manual joystick loop code here later
    elif current_mode == "post_process":
        sub_rect = (display.sub_x, display.sub_y, display.sub_width, display.sub_height)
        display.menu_screen.fill((150, 150, 150), sub_rect)
        # Add post-processing tool code here later
    elif current_mode == "author_info":
        sub_rect = (display.sub_x, display.sub_y, display.sub_width, display.sub_height)
        if author_bg:
            scaled_bg = pygame.transform.scale(author_bg, (display.sub_width, display.sub_height))
            display.menu_screen.blit(scaled_bg, (display.sub_x, display.sub_y))
        else:
            display.menu_screen.fill((0, 0, 0), sub_rect)
        text1 = display.large_font.render("Starlink-1060", True, (255, 255, 255))
        display.menu_screen.blit(text1, (display.sub_x + 10, display.sub_y + 10))
        contact_text = "Jonathan Nikkel - @NikkelJonathan"
        text2 = display.large_font.render(contact_text, True, (255, 255, 255))
        display.menu_screen.blit(text2, (display.sub_x + 10, display.sub_y + 50))

    pygame.display.flip()
    clock.tick(display.FPS_TARGET)  # Limit to FPS_TARGET FPS for better responsiveness

pygame.quit()
