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
EARTH_RADIUS_KM = 6371
SECONDS_PER_HOUR = 3600
WINDOW_POSITION = "0,0"

# Camera Constants and Settings
CAMERA_UPDATE_INTERVAL = 0.1  # Update camera images at 10 FPS target (reduced for stability)
CAMERA_TARGET_FPS = 10  # Reduced from 20 to prevent timeouts with exposure settings
CAMERA_WIDTH = 1920  # ASI462MM resolution (adjust based on actual camera)
CAMERA_HEIGHT = 1280
CAMERA2_WIDTH = 1936  # ASI178MC resolution
CAMERA2_HEIGHT = 1216

# ==============================================================================
# CAMERA VARIABLES (sensor calibration) - defined after pygame initialization
# ==============================================================================

# Camera connection buttons (shrunk to 50% size and arranged horizontally)
CAMERA1_CONNECT_BUTTON = pygame.Rect(MENU_WIDTH + UI_MARGIN + 200, UI_HEIGHT_OFFSET + 0, 60, 20)
CAMERA1_DISCONNECT_BUTTON = pygame.Rect(MENU_WIDTH + UI_MARGIN + 265, UI_HEIGHT_OFFSET + 0, 60, 20)
CAMERA2_CONNECT_BUTTON = pygame.Rect(MENU_WIDTH + UI_MARGIN + 1070, UI_HEIGHT_OFFSET + 0, 60, 20)
CAMERA2_DISCONNECT_BUTTON = pygame.Rect(MENU_WIDTH + UI_MARGIN + 1135, UI_HEIGHT_OFFSET + 0, 60, 20)

# Gain control UI rectangles
CAMERA1_GAIN_SLIDER_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 200, UI_HEIGHT_OFFSET + 30, 150, 20)  # Track for slider
CAMERA1_GAIN_SLIDER_HANDLE_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 200, UI_HEIGHT_OFFSET + 30, 20, 20)  # Slider handle
CAMERA2_GAIN_SLIDER_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 1070, UI_HEIGHT_OFFSET + 30, 150, 20)  # Track for slider
CAMERA2_GAIN_SLIDER_HANDLE_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 1070, UI_HEIGHT_OFFSET + 30, 20, 20)  # Slider handle

# Exposure control UI rectangles (positioned to the right of gain sliders)
CAMERA1_EXPOSURE_SLIDER_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 200 + 200, UI_HEIGHT_OFFSET + 30, 150, 20)  # Track for slider - 200px right of camera 1 gain
CAMERA1_EXPOSURE_SLIDER_HANDLE_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 200 + 200, UI_HEIGHT_OFFSET + 30, 20, 20)  # Slider handle
CAMERA2_EXPOSURE_SLIDER_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 1070 + 200, UI_HEIGHT_OFFSET + 30, 150, 20)  # Track for slider - 200px right of camera 2 gain
CAMERA2_EXPOSURE_SLIDER_HANDLE_RECT = pygame.Rect(MENU_WIDTH + UI_MARGIN + 1070 + 200, UI_HEIGHT_OFFSET + 30, 20, 20)  # Slider handle

# Gain slider segments (for visual feedback)
CAMERA1_GAIN_TRACK_COLOR = (139, 139, 139)  # Dark grey for track
CAMERA1_GAIN_ACTIVE_COLOR = (100, 100, 100)  # Grey for active portion
CAMERA2_GAIN_TRACK_COLOR = CAMERA1_GAIN_TRACK_COLOR
CAMERA2_GAIN_ACTIVE_COLOR = CAMERA1_GAIN_ACTIVE_COLOR

# Camera connection variables
camera1_connected = False
camera2_connected = False
camera1_cap = None
camera2_cap = None
camera1_prop = None
camera2_prop = None
camera1_index = 0  # ASI462MM camera index
camera2_index = 1  # ASI178MC camera index

# Dynamic camera resolution variables (set at connection time)
camera1_width_res = CAMERA_WIDTH  # Default fallback, updated on connection
camera1_height_res = CAMERA_HEIGHT
camera2_width_res = CAMERA2_WIDTH
camera2_height_res = CAMERA_HEIGHT
camera1_last_update = 0
camera2_last_update = 0
camera1_last_frame_time = 0
camera2_last_frame_time = 0

# Camera FPS and timestamp variables
camera1_fps = 0.0
camera2_fps = 0.0
camera1_utc_ts = ""
camera2_utc_ts = ""
camera1_local_ts = ""
camera2_local_ts = ""

# Camera state variables
camera_error_message = ""
camera1_frame = None
camera2_frame = None

# Camera gain variables (ASI cameras typically range from 0-300 or similar, adjust as needed)
camera1_gain = 1
camera2_gain = 1
camera1_gain_min = 0
camera1_gain_max = 300  # Conservative range, can be adjusted based on camera capabilities
camera2_gain_min = 0
camera2_gain_max = 300

# Combined view UI controls (bottom center of screen) - defined after pygame initialization

# Combined camera view variables (for sensor calibration mode)
combined_view_toggle = False  # Toggle between split screen and combined view
camera1_opacity = 0.5  # Opacity for camera 1 (0.0-1.0), camera 2 gets (1.0 - opacity)
camera1_opacity_min = 0.0
camera1_opacity_max = 1.0

# Camera exposure variables (ASI cameras typically in microseconds, 100us to 2000s range)
camera1_exposure = 10000  # 10ms default
camera2_exposure = 10000
camera1_exposure_min = 10  # 10 us minimum
camera1_exposure_max = 500000  # 0.5 maximum (conservative range)
camera2_exposure_min = 10
camera2_exposure_max = 500000

# ROI selection variables
camera1_roi_size = 0  # Index into roi_sizes list (0 = no ROI, 1+ = roi_sizes[i-1])
camera2_roi_size = 0
camera1_roi_x = 0.5  # Fraction of frame width (0.0-1.0)
camera2_roi_x = 0.5
camera1_roi_y = 0.5  # Fraction of frame height (0.0-1.0)
camera2_roi_y = 0.5

# ROI size presets (as fraction of full frame, diminishing sizes)
# Note: These should be conservative to avoid exceeding binned sensor resolutions
roi_sizes = [
    (0.03125, 0.03125),  # Smallest: 3.125% size
    (0.0625, 0.0625),    # Small: 6.25% size (index 1)
    (0.125, 0.125),      # 12.5% size (index 2)
    (0.25, 0.25),        # 25% size (index 3)
    (0.5, 0.5),          # 50% size (index 4)
    (1.0, 1.0),          # Full frame (index 5)
]

# Camera UI buttons state
camera_button_states = {}
camera_button_states["camera1_connect"] = {"hover": False, "clicked": False}
camera_button_states["camera2_connect"] = {"hover": False, "clicked": False}
camera_button_states["camera1_disconnect"] = {"hover": False, "clicked": False}
camera_button_states["camera2_disconnect"] = {"hover": False, "clicked": False}

# Gain control UI button states
camera_button_states["camera1_gain_slider"] = {"hover": False, "dragging": False}
camera_button_states["camera2_gain_slider"] = {"hover": False, "dragging": False}

# Exposure control UI button states
camera_button_states["camera1_exposure_slider"] = {"hover": False, "dragging": False}
camera_button_states["camera2_exposure_slider"] = {"hover": False, "dragging": False}

# ==============================================================================
# CAMERA THREADING VARIABLES (new threaded system)
# ==============================================================================

# Camera thread objects
camera1_thread = None
camera2_thread = None

# Buffer size for circular buffers (30 frames = ~1 second at 30 FPS)
BUFFER_SIZE = 30

# Thread-safe flags for thread lifecycle management
camera_threads_running = False
camera1_threads_running = False
camera2_threads_running = False

# Initialize camera threads
camera1_thread = None
camera2_thread = None

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

# Enumerate cameras if available
camera_infos = []
camera1_name = "Camera 1"
camera2_name = "Camera 2"

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

# Load or update TLEs from cache or Celestrak
tle_loaded = False
satellites = []
ts = load.timescale()
try:
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        current_time = time.time()
        if current_time - cache_time > cache_age_limit:
            status_messages.append("Downloading TLEs from Celestrak...")
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
            pygame.time.wait(100)  # Allow brief time for display update
            print(f"Debug: Status - {status_messages[-1]}")
            # Update from Celestrak
            url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
            response = requests.get(url)
            response.raise_for_status()
            tle_text = response.text
            with open(cache_file, 'w') as f:
                f.write(tle_text)
        else:
            status_messages.append("Loading TLEs from cache...")
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
            pygame.time.wait(100)  # Allow brief time for display update
            print(f"Debug: Status - {status_messages[-1]}")
            # Load from cache
            with open(cache_file, 'r') as f:
                tle_text = f.read()
    else:
        status_messages.append("Downloading TLEs from Celestrak...")
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
        pygame.time.wait(100)  # Allow brief time for display update
        print(f"Debug: Status - {status_messages[-1]}")
        # Initial download from Celestrak
        url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
        response = requests.get(url)
        response.raise_for_status()
        tle_text = response.text
        with open(cache_file, 'w') as f:
            f.write(tle_text)

    status_messages.append("Creating satellite objects...")
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
    pygame.time.wait(100)  # Allow brief time for display update
    print(f"Debug: Status - {status_messages[-1]}")

    # Process TLE text into satellites
    satellites = load.tle_file(cache_file)
    status_messages.append("TLEs ready")
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
    pygame.time.wait(100)  # Allow brief time for display update
    print(f"Debug: Status - {status_messages[-1]}")
    tle_loaded = True
except Exception as e:
    print(f"Debug: Error loading TLEs in text format: {e}")

# Pre-compute satellite labels and mean altitudes
satellite_labels = {}
satellite_mean_altitudes = {}
satellite_perigee = {}
satellite_apogee = {}
MU = 3.986004418e14  # Earth's gravitational parameter in m^3/s^2
R_EARTH = 6371  # Earth radius in km
for sat in satellites:
    name = sat.name.strip()
    norad_id = sat.model.satnum_str
    label_text = f"{norad_id} - {name}"
    satellite_labels[sat] = small_font.render(label_text, True, (255, 255, 255))
    # Compute mean altitude using Keplerian math
    n = sat.model.no_kozai / 60  # Mean motion in rad/s (convert from rad/min)
    a = (MU / (n**2))**(1/3) / 1000  # Semi-major axis in km
    e = sat.model.ecco  # Eccentricity
    perigee = a * (1 - e) - R_EARTH  # Perigee altitude in km
    apogee = a * (1 + e) - R_EARTH  # Apogee altitude in km
    mean_altitude = (perigee + apogee) / 2
    satellite_mean_altitudes[sat] = mean_altitude
    satellite_perigee[sat] = perigee
    satellite_apogee[sat] = apogee
    #print(f"Debug: Satellite {label_text} mean altitude: {mean_altitude:.1f} km perigee: {perigee:.1f} km apogee: {apogee:.1f} km")

last_update_time = 0
update_interval = 0.1  # Target 10 Hz
last_trajectory_update = 0
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
# CAMERA THREAD MANAGEMENT FUNCTIONS
# ==============================================================================

def start_camera1_thread():
    """Start camera 1 capture thread if not already running"""
    global camera1_thread, camera1_threads_running

    if camera1_threads_running or not camera1_connected or camera1_cap is None:
        return

    camera1_threads_running = True
    camera1_thread = CameraThread(camera1_index, camera1_cap, BUFFER_SIZE, CAMERA_TARGET_FPS)
    camera1_thread.start()
    update_status_callback("Camera 1 thread started")

def start_camera2_thread():
    """Start camera 2 capture thread if not already running"""
    global camera2_thread, camera2_threads_running

    if camera2_threads_running or not camera2_connected or camera2_cap is None:
        return

    camera2_threads_running = True
    camera2_thread = CameraThread(camera2_index, camera2_cap, BUFFER_SIZE, CAMERA_TARGET_FPS)
    camera2_thread.start()
    update_status_callback("Camera 2 thread started")

def stop_camera1_thread():
    """Stop camera 1 capture thread"""
    global camera1_thread, camera1_threads_running

    camera1_threads_running = False

    if camera1_thread is not None:
        camera1_thread.stop()
        camera1_thread = None
        update_status_callback("Camera 1 thread stopped")

def stop_camera2_thread():
    """Stop camera 2 capture thread"""
    global camera2_thread, camera2_threads_running

    camera2_threads_running = False

    if camera2_thread is not None:
        camera2_thread.stop()
        camera2_thread = None
        update_status_callback("Camera 2 thread stopped")

def stop_camera_threads():
    """Stop all camera capture threads"""
    stop_camera1_thread()
    stop_camera2_thread()

def update_camera_frames_from_buffers():
    """Update camera frames from thread buffers"""
    global camera1_frame, camera2_frame, camera1_fps, camera2_fps
    global camera1_utc_ts, camera2_utc_ts, camera1_local_ts, camera2_local_ts

    # Update camera 1 frame from buffer
    if camera1_thread is not None:
        latest_frame = camera1_thread.buffer.get_latest_frame()
        if latest_frame is not None:
            camera1_frame = latest_frame['frame']
            camera1_fps = camera1_thread.get_buffer_fps()
            camera1_utc_ts = camera1_thread.get_utc_timestamp()
            camera1_local_ts = camera1_thread.get_local_timestamp()

    # Update camera 2 frame from buffer
    if camera2_thread is not None:
        latest_frame = camera2_thread.buffer.get_latest_frame()
        if latest_frame is not None:
            camera2_frame = latest_frame['frame']
            camera2_fps = camera2_thread.get_buffer_fps()
            camera2_utc_ts = camera2_thread.get_utc_timestamp()
            camera2_local_ts = camera2_thread.get_local_timestamp()

def update_todo_progress():
    """Update task progress checklist"""
    # This would be called periodically to mark tasks as complete
    pass

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
            # Apply filters to pass table
            filtered_pass_table = []
            for pass_entry in satellite_pass_table:
                sat = pass_entry['satellite']
                include_sat = True

                # Apply name/NORAD filter
                if filter_text:
                    include_sat = include_sat and (
                        filter_text.lower() in sat.name.lower() or
                        filter_text in sat.model.satnum_str
                    )

                # Apply altitude above filter
                if filter_above_alt_text and include_sat:
                    try:
                        alt_filter = float(filter_above_alt_text)
                        sat_alt = satellite_mean_altitudes.get(sat, 0.0)
                        include_sat = include_sat and sat_alt >= alt_filter
                    except ValueError:
                        include_sat = False

                # Apply altitude below filter
                if filter_below_alt_text and include_sat:
                    try:
                        alt_filter = float(filter_below_alt_text)
                        sat_alt = satellite_mean_altitudes.get(sat, 0.0)
                        include_sat = include_sat and sat_alt <= alt_filter
                    except ValueError:
                        include_sat = False

                if include_sat:
                    filtered_pass_table.append(pass_entry)

            satellite_pass_table = sort_pass_table(filtered_pass_table, table_sort_keys, table_sort_reverse)

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
        # Apply filters to pass table
        filtered_pass_table = []
        for pass_entry in satellite_pass_table:
            sat = pass_entry['satellite']
            include_sat = True

            # Apply name/NORAD filter
            if filter_text:
                include_sat = include_sat and (
                    filter_text.lower() in sat.name.lower() or
                    filter_text in sat.model.satnum_str
                )

            # Apply altitude above filter
            if filter_above_alt_text and include_sat:
                try:
                    alt_filter = float(filter_above_alt_text)
                    sat_alt = satellite_mean_altitudes.get(sat, 0.0)
                    include_sat = include_sat and sat_alt >= alt_filter
                except ValueError:
                    include_sat = False

            # Apply altitude below filter
            if filter_below_alt_text and include_sat:
                try:
                    alt_filter = float(filter_below_alt_text)
                    sat_alt = satellite_mean_altitudes.get(sat, 0.0)
                    include_sat = include_sat and sat_alt <= alt_filter
                except ValueError:
                    include_sat = False

            if include_sat:
                filtered_pass_table.append(pass_entry)

        satellite_pass_table = sort_pass_table(filtered_pass_table, table_sort_keys, table_sort_reverse)

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

    # Handle events
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for btn in buttons:
                    if btn["rect"].collidepoint(pos):
                        if btn["mode"] == "exit":
                            running = False
                        else:
                            for b in buttons:
                                button_states[b["mode"]]["clicked"] = False
                            current_mode = btn["mode"]
                            button_states[current_mode]["clicked"] = True
                            focused_field = None
                            cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0, "filter": 0, "filter_above_alt": 0, "filter_below_alt": 0, "center_time": 0, "duration": 0}
                            selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None, "filter": None, "filter_above_alt": None, "filter_below_alt": None, "center_time": None, "duration": None}
                if current_mode == "config_options":
                    if save_button.collidepoint(pos):
                        button_states["save"]["clicked"] = True
                        config = {"lat": lat_str, "lon": lon_str, "alt": alt_str, "elevation_mask": elevation_mask_str}
                        save_config(config)
                        status_messages.append("Configuration saved")
                        print("Debug: Configuration saved")
                    elif load_button.collidepoint(pos):
                        button_states["load"]["clicked"] = True
                        root = Tk()
                        root.withdraw()
                        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
                        if file_path:
                            config = load_config(file_path)
                            lat_str, lon_str, alt_str, elevation_mask_str = config["lat"], config["lon"], config["alt"], config["elevation_mask"]
                            status_messages.append(f"Configuration loaded from {file_path}")
                            print(f"Debug: Configuration loaded from {file_path}")
                    elif input_rects['lat'].collidepoint(pos):
                        focused_field = 'lat'
                        cursor_pos['lat'] = len(lat_str)
                        selection_start['lat'] = None
                    elif input_rects['lon'].collidepoint(pos):
                        focused_field = 'lon'
                        cursor_pos['lon'] = len(lon_str)
                        selection_start['lon'] = None
                    elif input_rects['alt'].collidepoint(pos):
                        focused_field = 'alt'
                        cursor_pos['alt'] = len(alt_str)
                        selection_start['alt'] = None
                    elif input_rects['elevation_mask'].collidepoint(pos):
                        focused_field = 'elevation_mask'
                        cursor_pos['elevation_mask'] = len(elevation_mask_str)
                        selection_start['elevation_mask'] = None
                elif current_mode == "sensor_calib":
                    # Detect number of cameras connected
                    num_cameras = asi.get_num_cameras()
                    update_status_callback(f"Detected {num_cameras} cameras")
                    camera_infos = asi.list_cameras() 
                    update_status_callback(f"Camera names: {camera_infos}")
                    if len(camera_infos) > 0:
                        camera1_name = camera_infos[0]
                    if len(camera_infos) > 1:
                        camera2_name = camera_infos[1]
                    
                    # Handle camera connection/disconnection buttons
                    if CAMERA1_CONNECT_BUTTON.collidepoint(pos):
                        camera_button_states["camera1_connect"]["clicked"] = True
                        if ASI_AVAILABLE and camera1_index is not None:
                            try:
                                if camera1_connected:
                                    update_status_callback("Camera 1 already connected")
                                else:
                                    camera1_cap = asi.Camera(camera1_index)
                                    camera1_prop = camera1_cap.get_camera_property()
                                    if camera1_cap:
                                        # Set video format
                                        if camera1_prop['IsColorCam']:
                                            camera1_cap.set_image_type(asi.ASI_IMG_RGB24)
                                        else:
                                            camera1_cap.set_image_type(asi.ASI_IMG_RAW8)

                                        # Set camera controls - optimized for higher framerate
                                        camera1_cap.set_control_value(asi.ASI_EXPOSURE, camera1_exposure)  # Use exposure variable
                                        camera1_cap.set_control_value(asi.ASI_GAIN, camera1_gain)  # Set initial gain
                                        camera1_connected = True
                                        camera1_last_update = current_time
                                        update_status_callback("Camera 1 connected successfully")
                                        camera_error_message = ""

                                        # Set dynamic camera resolution from camera properties
                                        camera1_width_res = camera1_prop.get('MaxWidth', CAMERA_WIDTH)
                                        camera1_height_res = camera1_prop.get('MaxHeight', CAMERA_HEIGHT)
                                        update_status_callback(f"Camera 1 resolution: {camera1_width_res}x{camera1_height_res}")

                                        # Start threaded capture for camera 1
                                        start_camera1_thread()
                                    else:
                                        update_status_callback("Failed to connect Camera 1")
                                        camera1_cap = None
                            except Exception as e:
                                update_status_callback(f"Camera 1 connection error: {str(e)}")
                                camera1_cap = None
                        elif not ASI_AVAILABLE:
                            update_status_callback("ZWO ASI SDK not available")
                        else:
                            update_status_callback(f"Camera 1 not found at index {camera1_index}")
                    elif CAMERA1_DISCONNECT_BUTTON.collidepoint(pos):
                        camera_button_states["camera1_disconnect"]["clicked"] = True
                        stop_camera1_thread()
                        if camera1_cap is not None:
                            camera1_cap = None
                        camera1_connected = False
                        camera1_frame = None
                        update_status_callback("Camera 1 disconnected")
                    elif CAMERA2_CONNECT_BUTTON.collidepoint(pos):
                        camera_button_states["camera2_connect"]["clicked"] = True
                        if ASI_AVAILABLE:
                            try:
                                if camera2_connected:
                                    update_status_callback("Camera 2 already connected")
                                else:
                                    # Check how many cameras are actually available
                                    num_cameras = asi.get_num_cameras()
                                    if num_cameras <= camera2_index:
                                        update_status_callback(f"Camera 2 (index {camera2_index}) not found. Only {num_cameras} camera(s) detected.")
                                        camera2_cap = None
                                    else:
                                        camera2_cap = asi.Camera(camera2_index)
                                        camera2_prop = camera2_cap.get_camera_property()
                                        if camera2_cap:
                                            # Set video format
                                            if camera2_prop['IsColorCam']:
                                                camera2_cap.set_image_type(asi.ASI_IMG_RGB24)
                                            else:
                                                camera2_cap.set_image_type(asi.ASI_IMG_RAW8)

                                            # Set camera controls - optimized for higher framerate
                                            camera2_cap.set_control_value(asi.ASI_EXPOSURE, camera2_exposure)  # Use exposure variable
                                            camera2_cap.set_control_value(asi.ASI_GAIN, camera2_gain)  # Set initial gain
                                            camera2_connected = True
                                            camera2_last_update = current_time
                                            update_status_callback("Camera 2 connected successfully")
                                            camera_error_message = ""

                                            # Start threaded capture for camera 2
                                            start_camera2_thread()
                                        else:
                                            update_status_callback("Failed to connect Camera 2")
                                            camera2_cap = None
                            except Exception as e:
                                update_status_callback(f"Camera 2 connection error: {str(e)}")
                                camera2_cap = None
                        elif not ASI_AVAILABLE:
                            update_status_callback("ZWO ASI SDK not available")
                        else:
                            update_status_callback("ASI library not available")
                    elif CAMERA2_DISCONNECT_BUTTON.collidepoint(pos):
                        camera_button_states["camera2_disconnect"]["clicked"] = True
                        stop_camera2_thread()
                        if camera2_cap is not None:
                            camera2_cap = None
                        camera2_connected = False
                        camera2_frame = None
                        update_status_callback("Camera 2 disconnected")

                    # Handle gain slider clicks
                    elif CAMERA1_GAIN_SLIDER_RECT.collidepoint(pos):
                        camera_button_states["camera1_gain_slider"]["hover"] = True
                        # Calculate new gain value based on click position
                        slider_fraction = (pos[0] - CAMERA1_GAIN_SLIDER_RECT.x) / CAMERA1_GAIN_SLIDER_RECT.width
                        slider_fraction = max(0.0, min(1.0, slider_fraction))  # Clamp to 0-1
                        new_gain = int(camera1_gain_min + slider_fraction * (camera1_gain_max - camera1_gain_min))
                        camera1_gain = new_gain

                        # Update camera gain if connected
                        if camera1_connected and camera1_cap and ASI_AVAILABLE:
                            try:
                                camera1_cap.set_control_value(asi.ASI_GAIN, camera1_gain)
                                update_status_callback(f"Camera 1 gain set to {camera1_gain}")
                            except Exception as e:
                                update_status_callback(f"Failed to set Camera 1 gain: {str(e)}")

                    elif CAMERA2_GAIN_SLIDER_RECT.collidepoint(pos):
                        camera_button_states["camera2_gain_slider"]["hover"] = True
                        # Calculate new gain value based on click position
                        slider_fraction = (pos[0] - CAMERA2_GAIN_SLIDER_RECT.x) / CAMERA2_GAIN_SLIDER_RECT.width
                        slider_fraction = max(0.0, min(1.0, slider_fraction))  # Clamp to 0-1
                        new_gain = int(camera2_gain_min + slider_fraction * (camera2_gain_max - camera2_gain_min))
                        camera2_gain = new_gain

                        # Update camera gain if connected
                        if camera2_connected and camera2_cap and ASI_AVAILABLE:
                            try:
                                camera2_cap.set_control_value(asi.ASI_GAIN, camera2_gain)
                                update_status_callback(f"Camera 2 gain set to {camera2_gain}")
                            except Exception as e:
                                update_status_callback(f"Failed to set Camera 2 gain: {str(e)}")

                    # Handle exposure slider clicks
                    elif CAMERA1_EXPOSURE_SLIDER_RECT.collidepoint(pos):
                        camera_button_states["camera1_exposure_slider"]["hover"] = True
                        # Calculate new exposure value based on click position using logarithmic scaling
                        slider_fraction = (pos[0] - CAMERA1_EXPOSURE_SLIDER_RECT.x) / CAMERA1_EXPOSURE_SLIDER_RECT.width
                        slider_fraction = max(0.0, min(1.0, slider_fraction))  # Clamp to 0-1
                        # Logarithmic scaling: map 0-1 to log(min)-log(max)
                        log_min = math.log10(camera1_exposure_min) if camera1_exposure_min > 0 else 0
                        log_max = math.log10(camera1_exposure_max)
                        log_value = log_min + slider_fraction * (log_max - log_min)
                        new_exposure = int(10 ** log_value)
                        camera1_exposure = new_exposure

                        # Update camera exposure if connected
                        if camera1_connected and camera1_cap and ASI_AVAILABLE:
                            try:
                                camera1_cap.set_control_value(asi.ASI_EXPOSURE, camera1_exposure)
                                update_status_callback(f"Camera 1 exposure set to {camera1_exposure} µs")
                            except Exception as e:
                                update_status_callback(f"Failed to set Camera 1 exposure: {str(e)}")

                    elif CAMERA2_EXPOSURE_SLIDER_RECT.collidepoint(pos):
                        camera_button_states["camera2_exposure_slider"]["hover"] = True
                        # Calculate new exposure value based on click position using logarithmic scaling
                        slider_fraction = (pos[0] - CAMERA2_EXPOSURE_SLIDER_RECT.x) / CAMERA2_EXPOSURE_SLIDER_RECT.width
                        slider_fraction = max(0.0, min(1.0, slider_fraction))  # Clamp to 0-1
                        # Logarithmic scaling: map 0-1 to log(min)-log(max)
                        log_min = math.log10(camera2_exposure_min) if camera2_exposure_min > 0 else 0
                        log_max = math.log10(camera2_exposure_max)
                        log_value = log_min + slider_fraction * (log_max - log_min)
                        new_exposure = int(10 ** log_value)
                        camera2_exposure = new_exposure

                        # Update camera exposure if connected
                        if camera2_connected and camera2_cap and ASI_AVAILABLE:
                            try:
                                camera2_cap.set_control_value(asi.ASI_EXPOSURE, camera2_exposure)
                                update_status_callback(f"Camera 2 exposure set to {camera2_exposure} µs")
                            except Exception as e:
                                update_status_callback(f"Failed to set Camera 2 exposure: {str(e)}")

                    # Handle combined view toggle button
                    elif COMBINED_VIEW_BUTTON_RECT.collidepoint(pos):
                        if camera1_connected and camera2_connected:
                            combined_view_toggle = not combined_view_toggle
                            if combined_view_toggle:
                                update_status_callback("Combined view enabled")
                            else:
                                update_status_callback("Split screen view restored")
                        else:
                            update_status_callback("Both cameras must be connected to use combined view")

                    # Handle opacity slider
                    elif CAMERA_OPACITY_SLIDER_RECT.collidepoint(pos):
                        if combined_view_toggle and camera1_connected and camera2_connected:
                            # Calculate new opacity value based on click position
                            slider_fraction = (pos[0] - CAMERA_OPACITY_SLIDER_RECT.x) / CAMERA_OPACITY_SLIDER_RECT.width
                            slider_fraction = max(0.0, min(1.0, slider_fraction))  # Clamp to 0-1
                            camera1_opacity = camera1_opacity_min + slider_fraction * (camera1_opacity_max - camera1_opacity_min)
                            update_status_callback(f"Camera opacity set to {camera1_opacity:.1f}")

                    # Handle ROI controls for Camera 1
                    elif camera1_connected:
                        roi1_controls_x = cam1_left + cam1_width - 200
                        roi1_controls_y = cam1_top + 30

                        # Check ROI size selection buttons for Camera 1
                        roi_button_clicked = False
                        for i, size in enumerate(roi_sizes):
                            button_rect = pygame.Rect(roi1_controls_x + (i % 3) * 55, roi1_controls_y + (i // 3) * 25, 50, 20)
                            if button_rect.collidepoint(pos):
                                camera1_roi_size = i + 1 if camera1_roi_size != i + 1 else 0  # Toggle
                                roi_button_clicked = True

                                            # Apply ROI to camera 1 hardware if connected
                                if camera1_connected and camera1_cap and ASI_AVAILABLE and camera1_roi_size > 0:
                                    try:
                                        # Get current ROI format to preserve bins and image_type
                                        try:
                                            current_roi_format = camera1_cap.get_roi_format()
                                        except:
                                            # Fallback if get_roi_format fails
                                            current_roi_format = (CAMERA_WIDTH, CAMERA_HEIGHT, 1, camera1_prop.get('ImgType', 0))

                                        bins = current_roi_format[2] if len(current_roi_format) > 2 else 1
                                        image_type = current_roi_format[3] if len(current_roi_format) > 3 else camera1_prop.get('ImgType', 0)

                                        size_fraction = roi_sizes[camera1_roi_size - 1]
                                        cam1_width_res = camera1_prop.get('MaxWidth', CAMERA_WIDTH)
                                        cam1_height_res = camera1_prop.get('MaxHeight', CAMERA_HEIGHT)
                                        roi_width = int(cam1_width_res * size_fraction[0])
                                        roi_height = int(cam1_height_res * size_fraction[1])

                                        # Ensure dimensions meet camera requirements - adjusted for different camera models
                                        roi_width = roi_width - (roi_width % 8)  # Width must be multiple of 8
                                        roi_height = roi_height - (roi_height % 2)  # Height must be multiple of 2
                                        roi_width = max(roi_width, 32)  # Minimum width
                                        roi_height = max(roi_height, 4)  # Minimum height

                                        roi_center_x = int(camera1_roi_x * cam1_width_res)
                                        roi_center_y = int(camera1_roi_y * cam1_height_res)

                                        roi_start_x = roi_center_x - roi_width // 2
                                        roi_start_y = roi_center_y - roi_height // 2

                                        # Ensure ROI stays within frame bounds
                                        roi_start_x = max(0, min(roi_start_x, cam1_width_res - roi_width))
                                        roi_start_y = max(0, min(roi_start_y, cam1_height_res - roi_height))

                                        # Set ROI format first (dimensions and type, preserving current bins)
                                        camera1_cap.set_roi_format(roi_width, roi_height, bins, image_type)

                                        # Then set ROI start position
                                        camera1_cap.set_roi_start_position(roi_start_x, roi_start_y)

                                        update_status_callback(f"Camera 1 ROI applied: {roi_width}x{roi_height} at ({roi_start_x}, {roi_start_y})")
                                    except Exception as e:
                                        update_status_callback(f"Failed to set Camera 1 ROI: {str(e)}")
                                elif camera1_roi_size == 0 and camera1_connected and camera1_cap and ASI_AVAILABLE:
                                    # Reset to full frame
                                    try:
                                        try:
                                            current_roi_format = camera1_cap.get_roi_format()
                                        except:
                                            # Fallback if get_roi_format fails
                                            current_roi_format = (CAMERA_WIDTH, CAMERA_HEIGHT, 1, camera1_prop.get('ImgType', 0))

                                        bins = current_roi_format[2] if len(current_roi_format) > 2 else 1
                                        image_type = current_roi_format[3] if len(current_roi_format) > 3 else camera1_prop.get('ImgType', 0)
                                        cam1_width_res = camera1_prop.get('MaxWidth', CAMERA_WIDTH)
                                        cam1_height_res = camera1_prop.get('MaxHeight', CAMERA_HEIGHT)
                                        # Set to full frame with proper parameters: width, height, bins, image_type
                                        camera1_cap.set_roi_format(cam1_width_res, cam1_height_res, bins, image_type)
                                        # Also reset the ROI start position to (0, 0) and center variables
                                        camera1_cap.set_roi_start_position(0, 0)
                                        camera1_roi_x = 0.5
                                        camera1_roi_y = 0.5
                                        update_status_callback("Camera 1 ROI reset to full frame")
                                    except Exception as e:
                                        update_status_callback(f"Failed to reset Camera 1 ROI: {str(e)}")

                                roi_status = "Disabled" if camera1_roi_size == 0 else f"Size {roi_sizes[i][0]:.1f}"
                                update_status_callback(f"Camera 1 ROI: {roi_status}")
                                break

                        # Check for ROI position setting by clicking on Camera 1 image
                        if not roi_button_clicked and camera1_frame:
                            cam1_image_rect = pygame.Rect(cam1_img_x, cam1_img_y, cam1_scaled_frame.get_width(), cam1_scaled_frame.get_height())
                            if cam1_image_rect.collidepoint(pos):
                                # Convert click position to ROI coordinates (normalized 0-1)
                                camera1_roi_x = (pos[0] - cam1_img_x) / cam1_scaled_frame.get_width()
                                camera1_roi_y = (pos[1] - cam1_img_y) / cam1_scaled_frame.get_height()
                                camera1_roi_x = max(0.0, min(1.0, camera1_roi_x))  # Clamp to 0-1
                                camera1_roi_y = max(0.0, min(1.0, camera1_roi_y))
                                update_status_callback(f"Camera 1 ROI center: ({camera1_roi_x:.2f}, {camera1_roi_y:.2f})")

                                # Apply new position to camera 1 hardware
                                if camera1_connected and camera1_cap and ASI_AVAILABLE and camera1_roi_size > 0:
                                    try:
                                        # Get current ROI format to preserve bins and image_type
                                        current_roi_format = camera1_cap.get_roi_format()
                                        bins = current_roi_format[2]
                                        image_type = current_roi_format[3]

                                        cam1_width_res = camera1_prop.get('MaxWidth', CAMERA_WIDTH)
                                        cam1_height_res = camera1_prop.get('MaxHeight', CAMERA_HEIGHT)

                                        size_fraction = roi_sizes[camera1_roi_size - 1]
                                        roi_width = int(cam1_width_res * size_fraction[0])
                                        roi_height = int(cam1_height_res * size_fraction[1])
                                        
                                        # Ensure dimensions meet camera requirements - adjusted for different camera models
                                        roi_width = roi_width - (roi_width % 8)  # Width must be multiple of 8
                                        roi_height = roi_height - (roi_height % 2)  # Height must be multiple of 2
                                        roi_width = max(roi_width, 32)  # Minimum width
                                        roi_height = max(roi_height, 4)  # Minimum height

                                        roi_center_x = int(camera1_roi_x * cam1_width_res)
                                        roi_center_y = int(camera1_roi_y * cam1_height_res)

                                        roi_start_x = roi_center_x - roi_width // 2
                                        roi_start_y = roi_center_y - roi_height // 2

                                        # Ensure ROI stays within frame bounds
                                        roi_start_x = max(0, min(roi_start_x, cam1_width_res - roi_width))
                                        roi_start_y = max(0, min(roi_start_y, cam1_height_res - roi_height))

                                        # Set ROI format first (dimensions and type, preserving current bins)
                                        camera1_cap.set_roi_format(roi_width, roi_height, bins, image_type)

                                        # Then set ROI start position
                                        camera1_cap.set_roi_start_position(roi_start_x, roi_start_y)

                                        update_status_callback("Camera 1 ROI position updated")
                                    except Exception as e:
                                        update_status_callback(f"Failed to update Camera 1 ROI: {str(e)}")

                    # Handle ROI controls for Camera 2
                    if camera2_connected:
                        roi2_controls_x = cam2_left + cam2_width - 200
                        roi2_controls_y = cam2_top + 30

                        # Check ROI size selection buttons for Camera 2
                        roi_button_clicked = False
                        for i, size in enumerate(roi_sizes):
                            button_rect = pygame.Rect(roi2_controls_x + (i % 3) * 55, roi2_controls_y + (i // 3) * 25, 50, 20)
                            if button_rect.collidepoint(pos):
                                camera2_roi_size = i + 1 if camera2_roi_size != i + 1 else 0  # Toggle
                                roi_button_clicked = True

                                # Apply ROI to camera 2 hardware if connected
                                if camera2_connected and camera2_cap and ASI_AVAILABLE and camera2_roi_size > 0:
                                    try:
                                        # Get current ROI format to preserve bins and image_type
                                        try:
                                            current_roi_format = camera2_cap.get_roi_format()
                                        except:
                                            # Fallback if get_roi_format fails
                                            current_roi_format = (CAMERA2_WIDTH, CAMERA_HEIGHT, 1, camera2_prop.get('ImgType', 0))

                                        bins = current_roi_format[2] if len(current_roi_format) > 2 else 1
                                        image_type = current_roi_format[3] if len(current_roi_format) > 3 else camera2_prop.get('ImgType', 0)

                                        cam2_width_res = camera2_prop.get('MaxWidth', CAMERA2_WIDTH)
                                        cam2_height_res = camera2_prop.get('MaxHeight', CAMERA_HEIGHT)

                                        size_fraction = roi_sizes[camera2_roi_size - 1]
                                        roi_width = int(cam2_width_res * size_fraction[0])
                                        roi_height = int(cam2_height_res * size_fraction[1])

                                        # Ensure dimensions meet camera requirements for different models
                                        roi_width = roi_width - (roi_width % 8)  # Width must be multiple of 8
                                        roi_height = roi_height - (roi_height % 2)  # Height must be multiple of 2
                                        roi_width = max(roi_width, 32)  # Minimum width
                                        roi_height = max(roi_height, 4)  # Minimum height

                                        roi_center_x = int(camera2_roi_x * cam2_width_res)
                                        roi_center_y = int(camera2_roi_y * cam2_height_res)

                                        roi_start_x = roi_center_x - roi_width // 2
                                        roi_start_y = roi_center_y - roi_height // 2

                                        # Ensure ROI stays within frame bounds
                                        roi_start_x = max(0, min(roi_start_x, cam2_width_res - roi_width))
                                        roi_start_y = max(0, min(roi_start_y, cam2_height_res - roi_height))

                                        # Set ROI format first (dimensions and type, preserving current bins)
                                        camera2_cap.set_roi_format(roi_width, roi_height, bins, image_type)

                                        # Then set ROI start position
                                        camera2_cap.set_roi_start_position(roi_start_x, roi_start_y)

                                        update_status_callback(f"Camera 2 ROI applied: {roi_width}x{roi_height} at ({roi_start_x}, {roi_start_y})")
                                    except Exception as e:
                                        update_status_callback(f"Failed to set Camera 2 ROI: {str(e)}")
                                elif camera2_roi_size == 0 and camera2_connected and camera2_cap and ASI_AVAILABLE:
                                    # Reset to full frame
                                    try:
                                        current_roi_format = camera2_cap.get_roi_format()
                                        bins = current_roi_format[2]
                                        image_type = current_roi_format[3]
                                        cam2_width_res = camera2_prop.get('MaxWidth', CAMERA2_WIDTH)
                                        cam2_height_res = camera2_prop.get('MaxHeight', CAMERA_HEIGHT)
                                        # Set to full frame with proper parameters: width, height, bins, image_type
                                        camera2_cap.set_roi_format(cam2_width_res, cam2_height_res, bins, image_type)
                                        # Also reset the ROI start position to (0, 0) and center variables
                                        camera2_cap.set_roi_start_position(0, 0)
                                        camera2_roi_x = 0.5
                                        camera2_roi_y = 0.5
                                        update_status_callback("Camera 2 ROI reset to full frame")
                                    except Exception as e:
                                        update_status_callback(f"Failed to reset Camera 2 ROI: {str(e)}")

                                update_status_callback(f"Camera 2 ROI: {'Disabled' if camera2_roi_size == 0 else f'{roi_sizes[i][0]:.1f}'}")
                                break

                        # Check for ROI position setting by clicking on Camera 2 image
                        if not roi_button_clicked and camera2_frame:
                            cam2_image_rect = pygame.Rect(cam2_img_x, cam2_img_y, cam2_scaled_frame.get_width(), cam2_scaled_frame.get_height())
                            if cam2_image_rect.collidepoint(pos):
                                # Convert click position to ROI coordinates (normalized 0-1)
                                camera2_roi_x = (pos[0] - cam2_img_x) / cam2_scaled_frame.get_width()
                                camera2_roi_y = (pos[1] - cam2_img_y) / cam2_scaled_frame.get_height()
                                camera2_roi_x = max(0.0, min(1.0, camera2_roi_x))  # Clamp to 0-1
                                camera2_roi_y = max(0.0, min(1.0, camera2_roi_y))
                                update_status_callback(f"Camera 2 ROI center: ({camera2_roi_x:.2f}, {camera2_roi_y:.2f})")

                                # Apply new position to camera 2 hardware
                                if camera2_connected and camera2_cap and ASI_AVAILABLE and camera2_roi_size > 0:
                                    try:
                                        cam2_width_res = camera2_prop.get('MaxWidth', CAMERA2_WIDTH)
                                        cam2_height_res = camera2_prop.get('MaxHeight', CAMERA_HEIGHT)

                                        size_fraction = roi_sizes[camera2_roi_size - 1]
                                        roi_width = int(cam2_width_res * size_fraction[0])
                                        roi_height = int(cam2_height_res * size_fraction[1])
                                        
                                        # Ensure dimensions meet camera requirements - adjusted for different camera models
                                        roi_width = roi_width - (roi_width % 8)  # Width must be multiple of 8
                                        roi_height = roi_height - (roi_height % 2)  # Height must be multiple of 2
                                        roi_width = max(roi_width, 32)  # Minimum width
                                        roi_height = max(roi_height, 4)  # Minimum height

                                        roi_center_x = int(camera2_roi_x * cam2_width_res)
                                        roi_center_y = int(camera2_roi_y * cam2_height_res)

                                        roi_start_x = roi_center_x - roi_width // 2
                                        roi_start_y = roi_center_y - roi_height // 2

                                        # Ensure ROI stays within frame bounds
                                        roi_start_x = max(0, min(roi_start_x, cam2_width_res - roi_width))
                                        roi_start_y = max(0, min(roi_start_y, cam2_height_res - roi_height))

                                        # Set ROI format first (dimensions and type, preserving current bins)
                                        camera2_cap.set_roi_format(roi_width, roi_height, bins, image_type)

                                        # Then set ROI start position
                                        camera2_cap.set_roi_start_position(roi_start_x, roi_start_y)

                                        update_status_callback("Camera 2 ROI position updated")
                                    except Exception as e:
                                        update_status_callback(f"Failed to update Camera 2 ROI: {str(e)}")
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
                        # Re-build satellite pass table to remove filters (filtered for current visibility + upcoming passes)
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
                    else:
                        # Check pass table clicks
                        table_clicked = False
                        if hasattr(pass_table_clickable_areas, '__iter__'):
                            for area_type, index, rect in pass_table_clickable_areas:
                                if rect.collidepoint(pos):
                                    table_clicked = True
                                    if area_type == 'row' and satellite_pass_table and index < len(satellite_pass_table):
                                        # Select satellite from pass table
                                        selected_satellite = satellite_pass_table[index]['satellite']
                                        status_messages.append(f"Selected: {satellite_pass_table[index]['name']}")
                                    elif area_type == 'header':
                                        # Sort by column
                                        if table_sort_keys and index < len(table_sort_keys):
                                            # Toggle sort direction if same column, otherwise set new sort
                                            if table_sort_keys[index]:
                                                table_sort_reverse[index] = not table_sort_reverse[index]
                                            else:
                                                # Reset all sort keys
                                                table_sort_keys = [False] * len(table_sort_keys)
                                                table_sort_keys[index] = True
                                                table_sort_reverse = [False] * len(table_sort_reverse)
                                                table_sort_reverse[index] = True
                                            # Re-sort the table
                                            satellite_pass_table = sort_pass_table(satellite_pass_table, table_sort_keys, table_sort_reverse)
                                        break

                        if not table_clicked:
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
                for btn in buttons:
                    button_states[btn["mode"]]["hover"] = btn["rect"].collidepoint(event.pos)
                if current_mode == "config_options":
                    button_states["save"]["hover"] = save_button.collidepoint(event.pos)
                    button_states["load"]["hover"] = load_button.collidepoint(event.pos)
                elif current_mode == "sensor_calib":
                    # Update gain slider hover states
                    camera_button_states["camera1_gain_slider"]["hover"] = CAMERA1_GAIN_SLIDER_RECT.collidepoint(event.pos)
                    camera_button_states["camera2_gain_slider"]["hover"] = CAMERA2_GAIN_SLIDER_RECT.collidepoint(event.pos)

                    # Update exposure slider hover states
                    camera_button_states["camera1_exposure_slider"]["hover"] = CAMERA1_EXPOSURE_SLIDER_RECT.collidepoint(event.pos)
                    camera_button_states["camera2_exposure_slider"]["hover"] = CAMERA2_EXPOSURE_SLIDER_RECT.collidepoint(event.pos)
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
                for btn in buttons:
                    button_states[btn["mode"]]["clicked"] = False
                button_states["save"]["clicked"] = False
                button_states["load"]["clicked"] = False
                button_states["clear_filters"]["clicked"] = False
                button_states["reset"]["clicked"] = False
                button_states["pause"]["clicked"] = False
                button_states["play"]["clicked"] = False
                camera_button_states["camera1_connect"]["clicked"] = False
                camera_button_states["camera2_connect"]["clicked"] = False
                camera_button_states["camera1_disconnect"]["clicked"] = False
                camera_button_states["camera2_disconnect"]["clicked"] = False
                if dragging_slider and paused:
                    fraction = (slider_rect.x - scroll_bar_rect.x) / (scroll_bar_rect.width - slider_rect.width)
                    paused_tt = t0.tt + fraction * (t1.tt - t0.tt)
                dragging_slider = False
                scroll_start = None
            elif event.type == pygame.KEYDOWN:
                if current_mode == "config_options" and focused_field:
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
                                cursor_pos["filter"] = 1
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
                                cursor_pos["filter_above_alt"] = 1
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
                                cursor_pos["filter_above_alt"] = 1
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
    except Exception as e:
        status_messages.append(f"Event error: {str(e)}")
        print(f"Debug: Event error - {str(e)}")

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
        # Filter pass table in realtime based on current filter settings
        filtered_pass_table = []
        for pass_entry in satellite_pass_table:
            sat = pass_entry['satellite']
            include_sat = True

            # Apply name/NORAD filter
            if filter_text:
                include_sat = include_sat and (
                    filter_text.lower() in sat.name.lower() or
                    filter_text in sat.model.satnum_str
                )

            # Apply altitude above filter
            if filter_above_alt_text and include_sat:
                try:
                    alt_filter = float(filter_above_alt_text)
                    sat_alt = satellite_mean_altitudes.get(sat, 0.0)
                    include_sat = include_sat and sat_alt >= alt_filter
                except ValueError:
                    include_sat = False

            # Apply altitude below filter
            if filter_below_alt_text and include_sat:
                try:
                    alt_filter = float(filter_below_alt_text)
                    sat_alt = satellite_mean_altitudes.get(sat, 0.0)
                    include_sat = include_sat and sat_alt <= alt_filter
                except ValueError:
                    include_sat = False

            if include_sat:
                filtered_pass_table.append(pass_entry)

        # Apply sorting to filtered data
        sorted_filtered_pass_table = sort_pass_table(filtered_pass_table, table_sort_keys, table_sort_reverse)

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
        # Sensor calibration mode - display dual ASI camera feeds
        sub_rect = (sub_x, sub_y, sub_width, sub_height)
        menu_screen.fill((20, 20, 20), sub_rect)  # Darker background

        # Update camera frames from thread buffers
        update_camera_frames_from_buffers()

        # Camera frames are now updated automatically by the CameraThread - no blocking capture in main thread

        # Draw camera feeds if available
        display_width = sub_width // 2
        display_height = sub_height

        # Camera 1 display area (left half)
        cam1_left = sub_x
        cam1_top = sub_y
        cam1_width = sub_width // 2
        cam1_height = sub_height

        # Camera 2 display area (right half)
        cam2_left = sub_x + sub_width // 2
        cam2_top = sub_y
        cam2_width = sub_width // 2
        cam2_height = sub_height

        # Handle combined view mode
        if combined_view_toggle and camera1_connected and camera2_connected and camera1_frame and camera2_frame:
            # Combined view: Blend both cameras into a single full-screen image
            try:
                # Use full available area for combined view
                combined_width = sub_width - 20
                combined_height = sub_height - 100  # Leave space for controls at bottom

                # Scale both images to the same size for blending
                cam1_combined = pygame.transform.scale(camera1_frame, (combined_width, combined_height))
                cam2_combined = pygame.transform.scale(camera2_frame, (combined_width, combined_height))

                # Create a combined surface for blending
                combined_surface = pygame.Surface((combined_width, combined_height))

                # Copy cam1 to combined surface with its opacity
                cmap1_array = pygame.surfarray.array3d(cam1_combined)
                cam2_array = pygame.surfarray.array3d(cam2_combined)

                # Blend the arrays using weighted sum
                # camera1_opacity for cam1, (1 - camera1_opacity) for cam2
                blended_array = (cmap1_array.astype(np.float32) * camera1_opacity +
                               cam2_array.astype(np.float32) * (1.0 - camera1_opacity)).astype(np.uint8)

                # Create surface from blended array
                combined_surface = pygame.surfarray.make_surface(blended_array)

                # Center the combined image on screen
                combined_x = sub_x + (sub_width - combined_width) // 2
                combined_y = sub_y + 40

                menu_screen.blit(combined_surface, (combined_x, combined_y))

                # Draw targeting crosshairs at center of combined image
                center_x = combined_x + combined_width // 2
                center_y = combined_y + combined_height // 2

                # Draw horizontal line of crosshairs
                pygame.draw.line(menu_screen, (255, 0, 0), (center_x - 50, center_y), (center_x + 50, center_y), 1)
                # Draw vertical line of crosshairs
                pygame.draw.line(menu_screen, (255, 0, 0), (center_x, center_y - 50), (center_x, center_y + 50), 1)

                # Draw center dot
                pygame.draw.circle(menu_screen, (255, 0, 0), (center_x, center_y), 2)

                # Display combined view info
                combined_status = f"Combined View - Camera 1: {camera1_opacity:.1f}, Camera 2: {1-camera1_opacity:.1f}"
                status_surface = small_font.render(combined_status, True, (255, 255, 255))
                menu_screen.blit(status_surface, (combined_x, combined_y + combined_height))

                # FPS info for both cameras
                fps_text = f"Cam1 FPS: {camera1_fps:.1f}  Cam2 FPS: {camera2_fps:.1f}"
                fps_surface = small_font.render(fps_text, True, (255, 255, 255))
                menu_screen.blit(fps_surface, (combined_x, combined_y + combined_height + 25))

            except Exception as e:
                error_msg = small_font.render(f"Combined view error: {str(e)}", True, (255, 0, 0))
                menu_screen.blit(error_msg, (sub_x + 10, sub_y + sub_height // 2))
        else:
            # Split screen view
            # Draw camera 1 feed
            if camera1_connected:
                camera1_status = f"Camera 1 ({camera1_name}): Connected"
                status_color = (0, 255, 0)
            else:
                camera1_status = f"Camera 1 ({camera1_name}): Disconnected"
                status_color = (255, 0, 0)

            # Draw status text for camera 1
            status_surface = small_font.render(camera1_status, True, status_color)
            menu_screen.blit(status_surface, (cam1_left + 10, cam1_top + 10))

            # Display camera 1 image if connected
            cam1_img_x = 0
            cam1_img_y = 0
            cam1_scaled_frame = None
            if camera1_frame and camera1_connected:
                try:
                    # Scale image to fit display area
                    cam1_scaled_frame = pygame.transform.scale(camera1_frame, (cam1_width - 20, cam1_height - 80))
                    # Center the scaled image
                    cam1_img_x = cam1_left + (cam1_width - cam1_scaled_frame.get_width()) // 2
                    cam1_img_y = cam1_top + 50
                    menu_screen.blit(cam1_scaled_frame, (cam1_img_x, cam1_img_y))

                    # Display FPS, UTC and Local time below the image
                    info_text = f"FPS: {camera1_fps:.1f}  UTC: {camera1_utc_ts}  Local: {camera1_local_ts}"
                    info_surface = small_font.render(info_text, True, (255, 255, 255))
                    text_y = cam1_img_y + cam1_scaled_frame.get_height() + 10
                    menu_screen.blit(info_surface, (cam1_img_x, text_y))
                except Exception as e:
                    error_msg = small_font.render(f"Display error: {str(e)}", True, (255, 255, 0))
                    menu_screen.blit(error_msg, (cam1_left + 10, cam1_top + 40))

            # Draw camera 2 feed
            if camera2_connected:
                camera2_status = f"Camera 2 ({camera2_name}): Connected"
                status_color = (0, 255, 0)
            else:
                camera2_status = f"Camera 2 ({camera2_name}): Disconnected"
                status_color = (255, 0, 0)

            # Draw status text for camera 2
            status_surface = small_font.render(camera2_status, True, status_color)
            menu_screen.blit(status_surface, (cam2_left + 10, cam2_top + 10))

            # Display camera 2 image if connected
            cam2_img_x = 0
            cam2_img_y = 0
            cam2_scaled_frame = None
            if camera2_frame and camera2_connected:
                try:
                    # Scale image to fit display area
                    cam2_scaled_frame = pygame.transform.scale(camera2_frame, (cam2_width - 20, cam2_height - 80))
                    # Center the scaled image
                    cam2_img_x = cam2_left + (cam2_width - cam2_scaled_frame.get_width()) // 2
                    cam2_img_y = cam2_top + 50
                    menu_screen.blit(cam2_scaled_frame, (cam2_img_x, cam2_img_y))

                    # Display FPS, UTC and Local time below the image
                    info_text = f"FPS: {camera2_fps:.1f}  UTC: {camera2_utc_ts}  Local: {camera2_local_ts}"
                    info_surface = small_font.render(info_text, True, (255, 255, 255))
                    text_y = cam2_img_y + cam2_scaled_frame.get_height() + 10
                    menu_screen.blit(info_surface, (cam2_img_x, text_y))
                except Exception as e:
                    error_msg = small_font.render(f"Display error: {str(e)}", True, (255, 255, 0))
                    menu_screen.blit(error_msg, (cam2_left + 10, cam2_top + 40))

        # Draw connect buttons with smaller font
        # Camera 1 buttons
        camera1_connect_text = tiny_font.render("Connect 1", True, (255, 255, 255) if not camera_button_states["camera1_connect"]["clicked"] else (0, 0, 255))
        camera1_disconnect_text = tiny_font.render("Disconnect 1", True, (255, 255, 255) if not camera_button_states["camera1_disconnect"]["clicked"] else (0, 0, 255))

        # Camera 2 buttons
        camera2_connect_text = tiny_font.render("Connect 2", True, (255, 255, 255) if not camera_button_states["camera2_connect"]["clicked"] else (0, 0, 255))
        camera2_disconnect_text = tiny_font.render("Disconnect 2", True, (255, 255, 255) if not camera_button_states["camera2_disconnect"]["clicked"] else (0, 0, 255))

        # Draw button backgrounds
        pygame.draw.rect(menu_screen, (100, 100, 100) if camera_button_states["camera1_connect"]["clicked"] else (150, 150, 150), CAMERA1_CONNECT_BUTTON)
        pygame.draw.rect(menu_screen, (100, 100, 100) if camera_button_states["camera1_disconnect"]["clicked"] else (150, 150, 150), CAMERA1_DISCONNECT_BUTTON)
        pygame.draw.rect(menu_screen, (100, 100, 100) if camera_button_states["camera2_connect"]["clicked"] else (150, 150, 150), CAMERA2_CONNECT_BUTTON)
        pygame.draw.rect(menu_screen, (100, 100, 100) if camera_button_states["camera2_disconnect"]["clicked"] else (150, 150, 150), CAMERA2_DISCONNECT_BUTTON)

        # Center text in buttons
        camera1_connect_rect = camera1_connect_text.get_rect(center=CAMERA1_CONNECT_BUTTON.center)
        camera1_disconnect_rect = camera1_disconnect_text.get_rect(center=CAMERA1_DISCONNECT_BUTTON.center)
        camera2_connect_rect = camera2_connect_text.get_rect(center=CAMERA2_CONNECT_BUTTON.center)
        camera2_disconnect_rect = camera2_disconnect_text.get_rect(center=CAMERA2_DISCONNECT_BUTTON.center)

        # Draw text
        menu_screen.blit(camera1_connect_text, camera1_connect_rect)
        menu_screen.blit(camera1_disconnect_text, camera1_disconnect_rect)
        menu_screen.blit(camera2_connect_text, camera2_connect_rect)
        menu_screen.blit(camera2_disconnect_text, camera2_disconnect_rect)

        # Draw gain controls for Camera 1 (left side)
        if camera1_connected:
            # Camera 1 gain label and value
            camera1_gain_label = tiny_font.render(f"Cam 1 Gain:", True, (255, 255, 255))
            camera1_gain_value = tiny_font.render(f"{camera1_gain}", True, (255, 255, 0) if camera_button_states["camera1_gain_slider"]["hover"] else (255, 255, 255))
            menu_screen.blit(camera1_gain_label, (cam1_left + 10, cam1_top + 32))
            menu_screen.blit(camera1_gain_value, (cam1_left + 10, cam1_top + 45))

            # Draw Camera 1 gain slider track
            pygame.draw.rect(menu_screen, CAMERA1_GAIN_TRACK_COLOR, CAMERA1_GAIN_SLIDER_RECT)

            # Calculate handle position based on current gain value
            gain_fraction = (camera1_gain - camera1_gain_min) / (camera1_gain_max - camera1_gain_min)
            handle_x = CAMERA1_GAIN_SLIDER_RECT.x + int(gain_fraction * CAMERA1_GAIN_SLIDER_RECT.width)
            CAMERA1_GAIN_SLIDER_HANDLE_RECT.x = handle_x - CAMERA1_GAIN_SLIDER_HANDLE_RECT.width // 2

            # Draw active part of the track (filled portion)
            active_width = handle_x - CAMERA1_GAIN_SLIDER_RECT.x
            pygame.draw.rect(menu_screen, CAMERA1_GAIN_ACTIVE_COLOR, pygame.Rect(CAMERA1_GAIN_SLIDER_RECT.x, CAMERA1_GAIN_SLIDER_RECT.y, active_width, CAMERA1_GAIN_SLIDER_RECT.height))

            # Draw handle
            handle_color = (0, 255, 255) if camera_button_states["camera1_gain_slider"]["hover"] else (255, 255, 255)
            pygame.draw.rect(menu_screen, handle_color, CAMERA1_GAIN_SLIDER_HANDLE_RECT)

            # Camera 1 exposure label and value (beside gain)
            camera1_exposure_label = tiny_font.render(f"Cam 1 Exp:", True, (255, 255, 255))
            camera1_exposure_value = tiny_font.render(f"{camera1_exposure} µs", True, (255, 255, 0) if camera_button_states["camera1_exposure_slider"]["hover"] else (255, 255, 255))
            exposure_offset_x = 90  # Position exposure labels to the right of gain labels
            menu_screen.blit(camera1_exposure_label, (cam1_left + exposure_offset_x, cam1_top + 32))
            menu_screen.blit(camera1_exposure_value, (cam1_left + exposure_offset_x, cam1_top + 45))

            # Draw Camera 1 exposure slider track
            pygame.draw.rect(menu_screen, CAMERA1_GAIN_TRACK_COLOR, CAMERA1_EXPOSURE_SLIDER_RECT)

            # Calculate handle position based on current exposure value using logarithmic scaling
            if camera1_exposure > 0:
                exposure_fraction = (math.log10(camera1_exposure) - math.log10(camera1_exposure_min)) / (math.log10(camera1_exposure_max) - math.log10(camera1_exposure_min))
            else:
                exposure_fraction = 0.0
            exposure_fraction = max(0.0, min(1.0, exposure_fraction))  # Clamp to 0-1
            handle_x = CAMERA1_EXPOSURE_SLIDER_RECT.x + int(exposure_fraction * CAMERA1_EXPOSURE_SLIDER_RECT.width)
            CAMERA1_EXPOSURE_SLIDER_HANDLE_RECT.x = handle_x - CAMERA1_EXPOSURE_SLIDER_HANDLE_RECT.width // 2

            # Draw active part of the track (filled portion)
            active_width = handle_x - CAMERA1_EXPOSURE_SLIDER_RECT.x
            pygame.draw.rect(menu_screen, CAMERA1_GAIN_ACTIVE_COLOR, pygame.Rect(CAMERA1_EXPOSURE_SLIDER_RECT.x, CAMERA1_EXPOSURE_SLIDER_RECT.y, active_width, CAMERA1_EXPOSURE_SLIDER_RECT.height))

            # Draw handle
            handle_color = (0, 255, 255) if camera_button_states["camera1_exposure_slider"]["hover"] else (255, 255, 255)
            pygame.draw.rect(menu_screen, handle_color, CAMERA1_EXPOSURE_SLIDER_HANDLE_RECT)

        # Draw gain controls for Camera 2 (right side)
        if camera2_connected:
            # Camera 2 gain label and value
            camera2_gain_label = tiny_font.render(f"Cam 2 Gain:", True, (255, 255, 255))
            camera2_gain_value = tiny_font.render(f"{camera2_gain}", True, (255, 255, 0) if camera_button_states["camera2_gain_slider"]["hover"] else (255, 255, 255))
            menu_screen.blit(camera2_gain_label, (cam2_left + 10, cam2_top + 32))
            menu_screen.blit(camera2_gain_value, (cam2_left + 10, cam2_top + 45))

            # Draw Camera 2 gain slider track
            pygame.draw.rect(menu_screen, CAMERA2_GAIN_TRACK_COLOR, CAMERA2_GAIN_SLIDER_RECT)

            # Calculate handle position based on current gain value
            gain_fraction = (camera2_gain - camera2_gain_min) / (camera2_gain_max - camera2_gain_min)
            handle_x = CAMERA2_GAIN_SLIDER_RECT.x + int(gain_fraction * CAMERA2_GAIN_SLIDER_RECT.width)
            CAMERA2_GAIN_SLIDER_HANDLE_RECT.x = handle_x - CAMERA2_GAIN_SLIDER_HANDLE_RECT.width // 2

            # Draw active part of the track (filled portion)
            active_width = handle_x - CAMERA2_GAIN_SLIDER_RECT.x
            pygame.draw.rect(menu_screen, CAMERA2_GAIN_ACTIVE_COLOR, pygame.Rect(CAMERA2_GAIN_SLIDER_RECT.x, CAMERA2_GAIN_SLIDER_RECT.y, active_width, CAMERA2_GAIN_SLIDER_RECT.height))

            # Draw handle
            handle_color = (0, 255, 255) if camera_button_states["camera2_gain_slider"]["hover"] else (255, 255, 255)
            pygame.draw.rect(menu_screen, handle_color, CAMERA2_GAIN_SLIDER_HANDLE_RECT)

            # Camera 2 exposure label and value (beside gain)
            camera2_exposure_label = tiny_font.render(f"Cam 2 Exp:", True, (255, 255, 255))
            camera2_exposure_value = tiny_font.render(f"{camera2_exposure} µs", True, (255, 255, 0) if camera_button_states["camera2_exposure_slider"]["hover"] else (255, 255, 255))
            exposure_offset_x = 90  # Position exposure labels to the right of gain labels
            menu_screen.blit(camera2_exposure_label, (cam2_left + exposure_offset_x, cam2_top + 32))
            menu_screen.blit(camera2_exposure_value, (cam2_left + exposure_offset_x, cam2_top + 45))

            # Draw Camera 2 exposure slider track
            pygame.draw.rect(menu_screen, CAMERA2_GAIN_TRACK_COLOR, CAMERA2_EXPOSURE_SLIDER_RECT)

            # Calculate handle position based on current exposure value using logarithmic scaling
            if camera2_exposure > 0:
                exposure_fraction = (math.log10(camera2_exposure) - math.log10(camera2_exposure_min)) / (math.log10(camera2_exposure_max) - math.log10(camera2_exposure_min))
            else:
                exposure_fraction = 0.0
            exposure_fraction = max(0.0, min(1.0, exposure_fraction))  # Clamp to 0-1
            handle_x = CAMERA2_EXPOSURE_SLIDER_RECT.x + int(exposure_fraction * CAMERA2_EXPOSURE_SLIDER_RECT.width)
            CAMERA2_EXPOSURE_SLIDER_HANDLE_RECT.x = handle_x - CAMERA2_EXPOSURE_SLIDER_HANDLE_RECT.width // 2

            # Draw active part of the track (filled portion)
            active_width = handle_x - CAMERA2_EXPOSURE_SLIDER_RECT.x
            pygame.draw.rect(menu_screen, CAMERA2_GAIN_ACTIVE_COLOR, pygame.Rect(CAMERA2_EXPOSURE_SLIDER_RECT.x, CAMERA2_EXPOSURE_SLIDER_RECT.y, active_width, CAMERA2_EXPOSURE_SLIDER_RECT.height))

            # Draw handle
            handle_color = (0, 255, 255) if camera_button_states["camera2_exposure_slider"]["hover"] else (255, 255, 255)
            pygame.draw.rect(menu_screen, handle_color, CAMERA2_EXPOSURE_SLIDER_HANDLE_RECT)

        # Draw ROI controls for Camera 1 (if connected)
        if camera1_connected:
            # ROI controls positioned at upper right of camera 1 display area
            roi1_controls_x = cam1_left + cam1_width - 200
            roi1_controls_y = cam1_top + 30

            # ROI size selection buttons
            for i, size in enumerate(roi_sizes):
                button_rect = pygame.Rect(roi1_controls_x + (i % 3) * 55, roi1_controls_y + (i // 3) * 25, 50, 20)
                size_text = f"{size[0]:.1f}"
                if size == (1.0, 1.0):
                    size_text = "1:1"
                button_color = (0, 255, 0) if camera1_roi_size == i + 1 else (100, 100, 100)
                pygame.draw.rect(menu_screen, button_color, button_rect)
                text = tiny_font.render(size_text, True, (255, 255, 255))
                text_rect = text.get_rect(center=button_rect.center)
                menu_screen.blit(text, text_rect)

            # ROI position label and current values
            roi1_pos_label = tiny_font.render(f"ROI Pos: ({camera1_roi_x:.2f}, {camera1_roi_y:.2f})", True, (255, 255, 255))
            menu_screen.blit(roi1_pos_label, (roi1_controls_x, roi1_controls_y + 80))

            # Instructional text
            roi_help = tiny_font.render("Click image to set ROI center", True, (200, 200, 200))
            menu_screen.blit(roi_help, (roi1_controls_x, roi1_controls_y + 95))

            # Draw ROI overlay on camera 1 image if ROI is active
            if camera1_roi_size > 0 and camera1_frame and cam1_scaled_frame:
                size_fraction = roi_sizes[camera1_roi_size - 1]
                roi_width = camera1_frame.get_width() * size_fraction[0]
                roi_height = camera1_frame.get_height() * size_fraction[1]
                roi_center_x = camera1_roi_x * camera1_frame.get_width()
                roi_center_y = camera1_roi_y * camera1_frame.get_height()
                roi_left = roi_center_x - roi_width / 2
                roi_top = roi_center_y - roi_height / 2

                # Convert to display coordinates (scaled and positioned)
                scaled_roi_x = cam1_img_x + (roi_left / camera1_frame.get_width()) * cam1_scaled_frame.get_width()
                scaled_roi_y = cam1_img_y + (roi_top / camera1_frame.get_height()) * cam1_scaled_frame.get_height()
                scaled_roi_w = (roi_width / camera1_frame.get_width()) * cam1_scaled_frame.get_width()
                scaled_roi_h = (roi_height / camera1_frame.get_height()) * cam1_scaled_frame.get_height()

                # Draw ROI rectangle
                pygame.draw.rect(menu_screen, (0, 255, 0), pygame.Rect(scaled_roi_x, scaled_roi_y, scaled_roi_w, scaled_roi_h), 2)

                # Draw center cross
                center_x = scaled_roi_x + scaled_roi_w / 2
                center_y = scaled_roi_y + scaled_roi_h / 2
                pygame.draw.line(menu_screen, (0, 255, 0), (center_x - 10, center_y), (center_x + 10, center_y), 1)
                pygame.draw.line(menu_screen, (0, 255, 0), (center_x, center_y - 10), (center_x, center_y + 10), 1)

        # Draw ROI controls for Camera 2 (if connected)
        if camera2_connected:
            # ROI controls positioned at upper right of camera 2 display area
            roi2_controls_x = cam2_left + cam2_width - 200
            roi2_controls_y = cam2_top + 30

            # ROI size selection buttons
            for i, size in enumerate(roi_sizes):
                button_rect = pygame.Rect(roi2_controls_x + (i % 3) * 55, roi2_controls_y + (i // 3) * 25, 50, 20)
                size_text = f"{size[0]:.1f}"
                if size == (1.0, 1.0):
                    size_text = "1:1"
                button_color = (0, 255, 0) if camera2_roi_size == i + 1 else (100, 100, 100)
                pygame.draw.rect(menu_screen, button_color, button_rect)
                text = tiny_font.render(size_text, True, (255, 255, 255))
                text_rect = text.get_rect(center=button_rect.center)
                menu_screen.blit(text, text_rect)

            # ROI position label and current values
            roi2_pos_label = tiny_font.render(f"ROI Pos: ({camera2_roi_x:.2f}, {camera2_roi_y:.2f})", True, (255, 255, 255))
            menu_screen.blit(roi2_pos_label, (roi2_controls_x, roi2_controls_y + 80))

            # Instructional text
            roi_help = tiny_font.render("Click image to set ROI center", True, (200, 200, 200))
            menu_screen.blit(roi_help, (roi2_controls_x, roi2_controls_y + 95))

            # Draw ROI overlay on camera 2 image if ROI is active
            if camera2_roi_size > 0 and camera2_frame and cam2_scaled_frame:
                size_fraction = roi_sizes[camera2_roi_size - 1]
                roi_width = camera2_frame.get_width() * size_fraction[0]
                roi_height = camera2_frame.get_height() * size_fraction[1]
                roi_center_x = camera2_roi_x * camera2_frame.get_width()
                roi_center_y = camera2_roi_y * camera2_frame.get_height()
                roi_left = roi_center_x - roi_width / 2
                roi_top = roi_center_y - roi_height / 2

                # Convert to display coordinates (scaled and positioned)
                scaled_roi_x = cam2_img_x + (roi_left / camera2_frame.get_width()) * cam2_scaled_frame.get_width()
                scaled_roi_y = cam2_img_y + (roi_top / camera2_frame.get_height()) * cam2_scaled_frame.get_height()
                scaled_roi_w = (roi_width / camera2_frame.get_width()) * cam2_scaled_frame.get_width()
                scaled_roi_h = (roi_height / camera2_frame.get_height()) * cam2_scaled_frame.get_height()

                # Draw ROI rectangle
                pygame.draw.rect(menu_screen, (0, 255, 0), pygame.Rect(scaled_roi_x, scaled_roi_y, scaled_roi_w, scaled_roi_h), 2)

                # Draw center cross
                center_x = scaled_roi_x + scaled_roi_w / 2
                center_y = scaled_roi_y + scaled_roi_h / 2
                pygame.draw.line(menu_screen, (0, 255, 0), (center_x - 10, center_y), (center_x + 10, center_y), 1)
                pygame.draw.line(menu_screen, (0, 255, 0), (center_x, center_y - 10), (center_x, center_y + 10), 1)

        # Display ASI SDK status
        if not ASI_AVAILABLE:
            asi_msg = small_font.render("WARNING: ZWO ASI SDK not available", True, (255, 255, 0))
            menu_screen.blit(asi_msg, (sub_x + 10, sub_y + sub_height - 30))

        # Draw combined view toggle button and opacity slider at bottom center
        # Only show if both cameras are connected
        if camera1_connected and camera2_connected:

            # Calculate positions to center the controls
            combined_controls_y = total_height - 70  # Position above the slider
            slider_y = total_height - 30

            # Draw toggle button
            button_color = (0, 128, 255) if combined_view_toggle else (128, 128, 128)
            pygame.draw.rect(menu_screen, button_color, COMBINED_VIEW_BUTTON_RECT)
            pygame.draw.rect(menu_screen, (255, 255, 255), COMBINED_VIEW_BUTTON_RECT, 2)  # Border

            # Button text
            button_text = "Combined View: ON" if combined_view_toggle else "Combined View: OFF"
            button_surface = small_font.render(button_text, True, (255, 255, 255))
            button_rect = button_surface.get_rect(center=COMBINED_VIEW_BUTTON_RECT.center)
            menu_screen.blit(button_surface, button_rect)

            # Draw opacity slider (only when combined view is on)
            if combined_view_toggle:
                # Draw slider track
                pygame.draw.rect(menu_screen, (120, 120, 120), CAMERA_OPACITY_SLIDER_RECT)

                # Calculate handle position based on current opacity value
                opacity_fraction = (camera1_opacity - camera1_opacity_min) / (camera1_opacity_max - camera1_opacity_min)
                handle_x = CAMERA_OPACITY_SLIDER_RECT.x + int(opacity_fraction * CAMERA_OPACITY_SLIDER_RECT.width)
                CAMERA_OPACITY_SLIDER_HANDLE_RECT.x = handle_x - CAMERA_OPACITY_SLIDER_HANDLE_RECT.width // 2

                # Draw active part of the track (gray fill from left)
                active_width = handle_x - CAMERA_OPACITY_SLIDER_RECT.x
                pygame.draw.rect(menu_screen, (150, 150, 150), pygame.Rect(CAMERA_OPACITY_SLIDER_RECT.x, CAMERA_OPACITY_SLIDER_RECT.y, active_width, CAMERA_OPACITY_SLIDER_RECT.height))

                # Draw handle
                handle_color = (0, 255, 255)
                pygame.draw.rect(menu_screen, handle_color, CAMERA_OPACITY_SLIDER_HANDLE_RECT)

        # Display camera error message if any
        if camera_error_message:
            error_surface = small_font.render(camera_error_message, True, (255, 0, 0))
            menu_screen.blit(error_surface, (sub_x + 10, sub_y + sub_height - 50))
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
