import pygame
import os
import sys
import webbrowser
import math
from skyfield.api import load, wgs84, EarthSatellite, utc
import time
import requests
import json
from tkinter import filedialog, Tk
import datetime
import numpy as np

# Button drawing function
def draw_button(surface, rect, text, state):
    base_color = (211, 211, 211)  # Normal state
    hover_color = (225, 225, 225)  # Lighter, sleeker hover state
    clicked_color = (190, 190, 190)  # Softer clicked/active state

    # Determine color based on state
    color = base_color
    if state["clicked"]:
        color = clicked_color
    elif state["hover"]:
        color = hover_color

    # Draw button with sleeker embossment (thinner lines, softer gradient)
    pygame.draw.rect(surface, color, rect)
    if state["clicked"]:
        pygame.draw.line(surface, (160, 160, 160), rect.topleft, rect.bottomleft, 1)  # Left shadow
        pygame.draw.line(surface, (160, 160, 160), rect.topleft, rect.topright, 1)  # Top shadow
        pygame.draw.line(surface, (240, 240, 240), rect.bottomleft, rect.bottomright, 1)  # Bottom highlight
        pygame.draw.line(surface, (240, 240, 240), rect.topright, rect.bottomright, 1)  # Right highlight
    else:
        pygame.draw.line(surface, (240, 240, 240), rect.topleft, rect.bottomleft, 1)  # Left highlight
        pygame.draw.line(surface, (240, 240, 240), rect.topleft, rect.topright, 1)  # Top highlight
        pygame.draw.line(surface, (160, 160, 160), rect.bottomleft, rect.bottomright, 1)  # Bottom shadow
        pygame.draw.line(surface, (160, 160, 160), rect.topright, rect.bottomright, 1)  # Right shadow

    text_surface = font.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)

# Function to create a negative image
def create_negative_image(original_image):
    negative = original_image.copy()
    for x in range(negative.get_width()):
        for y in range(negative.get_height()):
            r, g, b, a = negative.get_at((x, y))
            negative.set_at((x, y), (255 - r, 255 - g, 255 - b, a))
    return negative

# Function to get color based on mean altitude (for LEO only)
def get_altitude_color(mean_altitude_km):
    if not (0 <= mean_altitude_km <= 2000):  # Apply only to LEO
        return None
    max_altitude = 1000.0
    min_altitude = 0.0
    norm_alt = max(0.0, min(1.0, (mean_altitude_km - min_altitude) / (max_altitude - min_altitude)))
    if norm_alt < 0.5:
        r = 0
        g = int(255 * (norm_alt * 2))
        b = int(255 * (1 - norm_alt * 2))
    else:
        r = int(255 * ((norm_alt - 0.5) * 2))
        g = int(255 * (1 - (norm_alt - 0.5) * 2))
        b = 0
    return (r, g, b)

# Function to draw a hexagon
def draw_hexagon(surface, x, y, color, size=3):
    points = [
        (x + size * math.cos(math.radians(angle)), y + size * math.sin(math.radians(angle)))
        for angle in range(0, 360, 60)
    ]
    pygame.draw.polygon(surface, color, points)

# Function to draw a triangle
def draw_triangle(surface, x, y, color, size=3):
    points = [
        (x, y - size),
        (x - size * math.cos(math.radians(30)), y + size * math.sin(math.radians(30))),
        (x + size * math.cos(math.radians(30)), y + size * math.sin(math.radians(30)))
    ]
    pygame.draw.polygon(surface, color, points)

def precompute_trajectories(satellites, observer, ts, sub_x, sub_y, sub_width, sub_height):
    current_utc = datetime.datetime.now(utc)
    t0 = ts.utc(current_utc - datetime.timedelta(minutes=15))
    t1 = ts.utc(current_utc + datetime.timedelta(minutes=15))
    times = ts.linspace(t0, t1, 900)  # 1-second intervals over 30 minutes
    trajectories = {}
    arc_segments = {}
    cx = sub_x + sub_width // 2
    cy = sub_y + sub_height // 2
    radius = min(sub_width, sub_height) // 2 - 50
    for sat in satellites:
        if sat in satellite_labels:
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
        return None, None, None
    trajectory, times_array = trajectory_data
    # Use precomputed time array for nearest index lookup
    nearest_idx = np.argmin(np.abs(times_array - current_tt))
    # Return the precomputed pixel coordinates and altitude of the nearest point
    return trajectory[nearest_idx][4], trajectory[nearest_idx][5], trajectory[nearest_idx][1]  # Return px, py, alt

if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    pygame.init()
    display_info = pygame.display.Info()
    menu_width = 200
    total_width = display_info.current_w
    total_height = display_info.current_h
    menu_screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Main Menu")

    # Load background image for menu and icon (assume 'cli/lucky.jpg' exists)
    try:
        bg_image = pygame.image.load('cli/lucky.jpg')
        bg_image_menu = pygame.transform.scale(bg_image, (160, 160))  # For menu background
        bg_image_icon = pygame.transform.scale(bg_image, (32, 32))  # For icon
        pygame.display.set_icon(bg_image_icon)  # Set as program icon
        negative_image = create_negative_image(bg_image_menu)  # Create negative version
        rotation_angle = 0  # Initialize rotation angle
    except pygame.error:
        bg_image_menu = None  # Fallback to solid color if image not found
        bg_image_icon = None
        negative_image = None
        rotation_angle = 0
        print("Warning: 'cli/lucky.jpg' not found. Using fallback color and no icon.")

    font = pygame.font.Font(None, 24)
    large_font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 14)  # Smaller font for labels to save space
    status_font = pygame.font.Font(None, 14)  # Increased from 12 to 14 for status messages

    sub_x = menu_width
    sub_y = 0
    sub_width = total_width - menu_width
    sub_height = total_height

    buttons = [
        {"rect": pygame.Rect(10, 10, 180, 80), "text": "Tracking Vis", "mode": "tracking_vis"},
        {"rect": pygame.Rect(10, 100, 180, 80), "text": "Sensor Calib", "mode": "sensor_calib"},
        {"rect": pygame.Rect(10, 190, 180, 80), "text": "Joystick Loop", "mode": "joystick_loop"},
        {"rect": pygame.Rect(10, 280, 180, 80), "text": "Post Process", "mode": "post_process"},
        {"rect": pygame.Rect(10, 370, 180, 80), "text": "Config Options", "mode": "config_options"},
        {"rect": pygame.Rect(10, 460, 180, 80), "text": "Author Info", "mode": "author_info"},
        {"rect": pygame.Rect(10, 550, 180, 80), "text": "Exit", "mode": "exit"},
    ]

    image_y = 550 + 80 + 10  # Position underneath the buttons
    status_y_start = total_height - 14 * 4  # Adjusted for new font size, space for 4 lines

    current_mode = None
    clock = pygame.time.Clock()

    # For author info
    author_bg = None
    try:
        author_bg = pygame.image.load('cli/lucky.jpg')
    except pygame.error:
        pass

    # Configuration defaults
    config = {"lat": "34.87405877829887", "lon": "-120.44621926328121", "alt": "120.0", "elevation_mask": "0.0"}
    # Load config.json if it exists, overriding defaults
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        except Exception as e:
            print(f"Debug: Error loading config.json: {e}")
    lat_str = config["lat"]
    lon_str = config["lon"]
    alt_str = config["alt"]
    elevation_mask_str = config["elevation_mask"]
    focused_field = None  # None, 'lat', 'lon', 'alt', 'elevation_mask', 'filter', 'filter_alt'
    cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0, "filter": 0, "filter_alt": 0}  # Cursor position in each field
    selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None, "filter": None, "filter_alt": None}  # Selection start position
    filter_text = ""  # Moved outside loop to persist
    filter_alt_text = ""  # Moved outside loop to persist

    # Input rects for config
    input_rects = {
        'lat': pygame.Rect(sub_x + 20, sub_y + 60, 200, 30),
        'lon': pygame.Rect(sub_x + 20, sub_y + 150, 200, 30),
        'alt': pygame.Rect(sub_x + 20, sub_y + 240, 200, 30),
        'elevation_mask': pygame.Rect(sub_x + 20, sub_y + 330, 200, 30),
    }
    save_button = pygame.Rect(sub_x + 20, sub_y + sub_height - 50, 100, 30)  # Moved to bottom left
    load_button = pygame.Rect(sub_x + 130, sub_y + sub_height - 50, 100, 30)  # Moved to bottom left

    # Button states for embossment
    button_states = {btn["mode"]: {"hover": False, "clicked": False} for btn in buttons}
    button_states["save"] = {"hover": False, "clicked": False}
    button_states["load"] = {"hover": False, "clicked": False}
    button_states["clear_filters"] = {"hover": False, "clicked": False}

    # Initial render of main menu
    menu_screen.fill((200, 200, 200), (0, 0, menu_width, total_height))  # Menu background
    if bg_image_menu:
        menu_screen.blit(bg_image_menu, ((menu_width - 160) // 2, image_y))  # Center horizontally, adjusted for 160px width
    for btn in buttons:
        draw_button(menu_screen, btn["rect"], btn["text"], button_states[btn["mode"]])
    status_messages = ["Starting TLE process..."]
    for i, msg in enumerate(status_messages[-4:]):
        status_render = status_font.render(msg, True, (0, 0, 0))
        menu_screen.blit(status_render, (10, status_y_start + i * 14))
    pygame.display.flip()
    print(f"Debug: Status - {'Starting TLE process...'}")

    # Cache file management
    cache_file = "tle_cache.tle"
    cache_age_limit = 24 * 3600  # 24 hours in seconds

    # Load or update TLEs from cache or Celestrak
    tle_loaded = False
    satellites = []
    try:
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            if current_time - cache_time > cache_age_limit:
                status_messages.append("Downloading TLEs from Celestrak...")
                status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
                menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
                pygame.display.flip()
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
                status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
                menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
                pygame.display.flip()
                print(f"Debug: Status - {status_messages[-1]}")
                # Load from cache
                with open(cache_file, 'r') as f:
                    tle_text = f.read()
        else:
            status_messages.append("Downloading TLEs from Celestrak...")
            status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
            menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
            pygame.display.flip()
            print(f"Debug: Status - {status_messages[-1]}")
            # Initial download from Celestrak
            url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
            response = requests.get(url)
            response.raise_for_status()
            tle_text = response.text
            with open(cache_file, 'w') as f:
                f.write(tle_text)

        status_messages.append("Creating satellite objects...")
        status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
        menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
        pygame.display.flip()
        print(f"Debug: Status - {status_messages[-1]}")

        # Process TLE text into satellites
        satellites = load.tle_file(cache_file)
        status_messages.append("TLEs ready")
        status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
        menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
        pygame.display.flip()
        print(f"Debug: Status - {status_messages[-1]}")
        tle_loaded = True
    except Exception as e:
        print(f"Debug: Error loading TLEs in text format: {e}")

    # Pre-compute satellite labels and mean altitudes
    satellite_labels = {}
    satellite_mean_altitudes = {}
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

    last_update_time = 0
    update_interval = 0.1  # Target 10 Hz
    last_trajectory_update = 0
    trajectory_interval = 900  # 15 minutes in seconds
    satellite_trajectories = {}
    satellite_arc_segments = {}
    hovered_satellite = None
    selected_satellite = None

    running = True
    while running:
        current_time = time.time()
        mouse_pos = pygame.mouse.get_pos()
        # Check if mouse is over the background image
        image_rect = pygame.Rect((menu_width - 160) // 2, image_y, 160, 160) if bg_image_menu else None
        # Define filter rectangles inside the loop
        filter_rect = pygame.Rect(sub_x + 20, sub_y + 210, 200, 30)  # Filter by name box
        filter_alt_rect = pygame.Rect(sub_x + 20, sub_y + 280, 200, 30)  # Filter by altitude box

        # Precompute trajectories and arc segments every 15 minutes
        if current_time - last_trajectory_update >= trajectory_interval:
            status_messages.append("Starting trajectory precomputation...")
            status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
            menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
            pygame.display.flip()
            print(f"Debug: Status - {status_messages[-1]}")
            lat = float(lat_str)
            lon = float(lon_str)
            alt_m = float(alt_str)
            observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
            ts = load.timescale()
            satellite_trajectories, satellite_arc_segments = precompute_trajectories(satellites, observer, ts, sub_x, sub_y, sub_width, sub_height)
            last_trajectory_update = current_time
            status_messages.append("Trajectories updated")
            status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
            menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
            pygame.display.flip()
            print(f"Debug: Status - {status_messages[-1]}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
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
                            focused_field = None  # Reset focus when switching modes
                            cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0, "filter": 0, "filter_alt": 0}  # Reset cursor on mode switch
                            selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None, "filter": None, "filter_alt": None}  # Reset selection
                if current_mode == "config_options":
                    if save_button.collidepoint(pos):
                        button_states["save"]["clicked"] = True
                        config = {"lat": lat_str, "lon": lon_str, "alt": alt_str, "elevation_mask": elevation_mask_str}
                        with open("config.json", "w") as f:
                            json.dump(config, f)
                        status_messages.append("Config saved successfully")
                        status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
                        menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
                        pygame.display.flip()
                        print(f"Debug: Status - {status_messages[-1]}")
                        button_states["save"]["clicked"] = False  # Revert after action
                    elif load_button.collidepoint(pos):
                        button_states["load"]["clicked"] = True
                        root = Tk()
                        root.withdraw()
                        initial_dir = os.getcwd()
                        file_path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("JSON files", "*.json")])
                        if file_path:
                            with open(file_path, "r") as f:
                                config = json.load(f)
                                lat_str = config.get("lat", lat_str)
                                lon_str = config.get("lon", lon_str)
                                alt_str = config.get("alt", alt_str)
                                elevation_mask_str = config.get("elevation_mask", elevation_mask_str)
                            status_messages.append(f"Config loaded successfully from {os.path.basename(file_path)}")
                            status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
                            menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
                            pygame.display.flip()
                            print(f"Debug: Status - {status_messages[-1]}")
                        button_states["load"]["clicked"] = False  # Revert after action
                if current_mode == "tracking_vis" and tle_loaded and pos[0] >= sub_x:  # Only check satellite clicks in tracking area
                    for sat, (px, py) in satellite_positions.items():
                        if math.hypot(pos[0] - px, pos[1] - py) < 10:  # 10-pixel click radius
                            if selected_satellite == sat:
                                selected_satellite = None  # Deselect on second click
                            else:
                                selected_satellite = sat  # Select satellite on first click
                                filter_text = sat.name.strip()  # Set filter_text to selected satellite name
                            break
                if current_mode == "tracking_vis" and 'clear_filters_button' in locals() and clear_filters_button.collidepoint(pos):
                    button_states["clear_filters"]["clicked"] = True
                    filter_text = ""
                    filter_alt_text = ""
                    selected_satellite = None  # Clear selected satellite filter
                    status_messages.append("Filters Cleared")
                    status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
                    menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 14))
                    pygame.display.flip()
                    print(f"Debug: Status - {status_messages[-1]}")
                    button_states["clear_filters"]["clicked"] = False  # Revert after action
                # Check for clicks on filter boxes to set focus
                if current_mode == "tracking_vis" and filter_rect.collidepoint(pos):
                    focused_field = "filter"
                elif current_mode == "tracking_vis" and filter_alt_rect.collidepoint(pos):
                    focused_field = "filter_alt"
            if event.type == pygame.KEYDOWN:
                if current_mode == "tracking_vis" and focused_field:
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
                    elif focused_field == "filter_alt":
                        field_str = filter_alt_text
                        mods = pygame.key.get_mods()
                        if event.key == pygame.K_LEFT:
                            if mods & pygame.KMOD_SHIFT:
                                selection_start["filter_alt"] = cursor_pos["filter_alt"] if selection_start["filter_alt"] is None else selection_start["filter_alt"]
                                cursor_pos["filter_alt"] = max(0, cursor_pos["filter_alt"] - 1)
                            else:
                                cursor_pos["filter_alt"] = max(0, cursor_pos["filter_alt"] - 1)
                                selection_start["filter_alt"] = None
                        elif event.key == pygame.K_RIGHT:
                            if mods & pygame.KMOD_SHIFT:
                                selection_start["filter_alt"] = cursor_pos["filter_alt"] if selection_start["filter_alt"] is None else selection_start["filter_alt"]
                                cursor_pos["filter_alt"] = min(len(field_str), cursor_pos["filter_alt"] + 1)
                            else:
                                cursor_pos["filter_alt"] = min(len(field_str), cursor_pos["filter_alt"] + 1)
                                selection_start["filter_alt"] = None
                        elif event.key == pygame.K_HOME:
                            if mods & pygame.KMOD_SHIFT:
                                selection_start["filter_alt"] = cursor_pos["filter_alt"] if selection_start["filter_alt"] is None else selection_start["filter_alt"]
                            cursor_pos["filter_alt"] = 0
                            if not mods & pygame.KMOD_SHIFT:
                                selection_start["filter_alt"] = None
                        elif event.key == pygame.K_END:
                            if mods & pygame.KMOD_SHIFT:
                                selection_start["filter_alt"] = cursor_pos["filter_alt"] if selection_start["filter_alt"] is None else selection_start["filter_alt"]
                                cursor_pos["filter_alt"] = len(field_str)
                            if not mods & pygame.KMOD_SHIFT:
                                selection_start["filter_alt"] = None
                        elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                            start = min(cursor_pos["filter_alt"], selection_start["filter_alt"]) if selection_start["filter_alt"] is not None else cursor_pos["filter_alt"]
                            end = max(cursor_pos["filter_alt"], selection_start["filter_alt"]) if selection_start["filter_alt"] is not None else cursor_pos["filter_alt"] + 1 if event.key == pygame.K_DELETE else cursor_pos["filter_alt"]
                            if start < end:
                                field_str = field_str[:start] + field_str[end:]
                                cursor_pos["filter_alt"] = start
                                selection_start["filter_alt"] = None
                            elif event.key == pygame.K_BACKSPACE and cursor_pos["filter_alt"] > 0:
                                field_str = field_str[:cursor_pos["filter_alt"] - 1] + field_str[cursor_pos["filter_alt"]:]
                                cursor_pos["filter_alt"] -= 1
                                selection_start["filter_alt"] = None
                        elif event.key == pygame.K_RETURN:
                            focused_field = None
                            selection_start["filter_alt"] = None
                        else:
                            char = event.unicode
                            if char.isdigit() or char in ['.', '-']:
                                start = min(cursor_pos["filter_alt"], selection_start["filter_alt"]) if selection_start["filter_alt"] is not None else cursor_pos["filter_alt"]
                                end = max(cursor_pos["filter_alt"], selection_start["filter_alt"]) if selection_start["filter_alt"] is not None else cursor_pos["filter_alt"]
                                field_str = field_str[:start] + char + field_str[end:]
                                cursor_pos["filter_alt"] += 1
                                selection_start["filter_alt"] = None
                        filter_alt_text = field_str
                elif current_mode == "config_options" and focused_field:
                    field_str = locals()[f"{focused_field}_str"]
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
                            cursor_pos[focused_field] = max(0, cursor_pos[focused_field] - 1)
                        else:
                            cursor_pos[focused_field] = max(0, cursor_pos[focused_field] - 1)
                            selection_start[focused_field] = None
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
                            cursor_pos[focused_field] = min(len(field_str), cursor_pos[focused_field] + 1)
                        else:
                            cursor_pos[focused_field] = min(len(field_str), cursor_pos[focused_field] + 1)
                            selection_start[focused_field] = None
                    elif event.key == pygame.K_HOME:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
                        cursor_pos[focused_field] = 0
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start[focused_field] = None
                    elif event.key == pygame.K_END:
                        if mods & pygame.KMOD_SHIFT:
                            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
                            cursor_pos[focused_field] = len(field_str)
                        if not mods & pygame.KMOD_SHIFT:
                            selection_start[focused_field] = None
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        start = min(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field]
                        end = max(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field] + 1 if event.key == pygame.K_DELETE else cursor_pos[focused_field]
                        if start < end:
                            field_str = field_str[:start] + field_str[end:]
                            cursor_pos[focused_field] = start
                            selection_start[focused_field] = None
                        elif event.key == pygame.K_BACKSPACE and cursor_pos[focused_field] > 0:
                            field_str = field_str[:cursor_pos[focused_field] - 1] + field_str[cursor_pos[focused_field]:]
                            cursor_pos[focused_field] -= 1
                            selection_start[focused_field] = None
                    elif event.key == pygame.K_RETURN:
                        focused_field = None
                        selection_start[focused_field] = None
                    else:
                        char = event.unicode
                        if char.isdigit() or char in ['.', '-', '+']:
                            start = min(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field]
                            end = max(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field]
                            field_str = field_str[:start] + char + field_str[end:]
                            cursor_pos[focused_field] += 1
                            selection_start[focused_field] = None
                    locals()[f"{focused_field}_str"] = field_str
            if event.type == pygame.MOUSEMOTION:
                for btn in buttons:
                    button_states[btn["mode"]]["hover"] = btn["rect"].collidepoint(mouse_pos)
                button_states["save"]["hover"] = save_button.collidepoint(mouse_pos)
                button_states["load"]["hover"] = load_button.collidepoint(mouse_pos)
                if current_mode == "tracking_vis" and tle_loaded:
                    button_states["clear_filters"]["hover"] = clear_filters_button.collidepoint(mouse_pos)
                    hovered_satellite = None
                    mouse_x, mouse_y = mouse_pos
                    for sat, (px, py) in satellite_positions.items():
                        if math.hypot(mouse_x - px, mouse_y - py) < 10:  # 10-pixel hover radius
                            hovered_satellite = sat
                            break

        menu_screen.fill((200, 200, 200), (0, 0, menu_width, total_height))  # Menu background

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
            status_render = status_font.render(msg, True, (0, 0, 0))
            menu_screen.blit(status_render, (10, status_y_start + i * 14))
        if current_mode == "config_options":
            sub_rect = (sub_x, sub_y, sub_width, sub_height)
            # Draw gradient background from (160, 160, 160) to (155, 155, 155)
            for y in range(sub_height):
                color = (160 - (y / sub_height * 5), 160 - (y / sub_height * 5), 160 - (y / sub_height * 5))
                pygame.draw.line(menu_screen, color, (sub_x, sub_y + y), (sub_x + sub_width, sub_y + y))
            # Draw grouping box and label
            group_rect = pygame.Rect(sub_x + 10, sub_y + 0, 220, 370)  # Adjusted height for new parameter
            pygame.draw.rect(menu_screen, (0, 0, 0), group_rect, 2, border_radius=5)  # Sleek black box with rounded edges
            group_label = font.render("Observer Location", True, (0, 0, 0))
            menu_screen.blit(group_label, (sub_x + 20, sub_y + 10))
            # Draw labels and inputs with adjusted positions
            lat_label = font.render("Latitude:", True, (0, 0, 0))
            menu_screen.blit(lat_label, (sub_x + 20, sub_y + 30))
            pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['lat'])
            lat_text = font.render(lat_str, True, (0, 0, 0))
            menu_screen.blit(lat_text, (input_rects['lat'].x + 5, input_rects['lat'].y + 5))
            if focused_field == 'lat':
                pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['lat'], 2)
                text_width, _ = font.size(lat_str[:cursor_pos['lat']])
                pygame.draw.line(menu_screen, (0, 0, 255),
                                (input_rects['lat'].x + 5 + text_width, input_rects['lat'].y + 5),
                                (input_rects['lat'].x + 5 + text_width, input_rects['lat'].y + 25), 2)
                if selection_start['lat'] is not None:
                    start_width, _ = font.size(lat_str[:min(cursor_pos['lat'], selection_start['lat'])])
                    end_width, _ = font.size(lat_str[:max(cursor_pos['lat'], selection_start['lat'])])
                    pygame.draw.rect(menu_screen, (0, 120, 215),
                                    (input_rects['lat'].x + 5 + start_width, input_rects['lat'].y + 5,
                                     end_width - start_width, 20), 2)

            lon_label = font.render("Longitude:", True, (0, 0, 0))
            menu_screen.blit(lon_label, (sub_x + 20, sub_y + 120))
            pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['lon'])
            lon_text = font.render(lon_str, True, (0, 0, 0))
            menu_screen.blit(lon_text, (input_rects['lon'].x + 5, input_rects['lon'].y + 5))
            if focused_field == 'lon':
                pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['lon'], 2)
                text_width, _ = font.size(lon_str[:cursor_pos['lon']])
                pygame.draw.line(menu_screen, (0, 0, 255),
                                (input_rects['lon'].x + 5 + text_width, input_rects['lon'].y + 5),
                                (input_rects['lon'].x + 5 + text_width, input_rects['lon'].y + 25), 2)
                if selection_start['lon'] is not None:
                    start_width, _ = font.size(lon_str[:min(cursor_pos['lon'], selection_start['lon'])])
                    end_width, _ = font.size(lon_str[:max(cursor_pos['lon'], selection_start['lon'])])
                    pygame.draw.rect(menu_screen, (0, 120, 215),
                                    (input_rects['lon'].x + 5 + start_width, input_rects['lon'].y + 5,
                                     end_width - start_width, 20), 2)

            alt_label = font.render("Altitude (m):", True, (0, 0, 0))
            menu_screen.blit(alt_label, (sub_x + 20, sub_y + 210))
            pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['alt'])
            alt_text = font.render(alt_str, True, (0, 0, 0))
            menu_screen.blit(alt_text, (input_rects['alt'].x + 5, input_rects['alt'].y + 5))
            if focused_field == 'alt':
                pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['alt'], 2)
                text_width, _ = font.size(alt_str[:cursor_pos['alt']])
                pygame.draw.line(menu_screen, (0, 0, 255),
                                (input_rects['alt'].x + 5 + text_width, input_rects['alt'].y + 5),
                                (input_rects['alt'].x + 5 + text_width, input_rects['alt'].y + 25), 2)
                if selection_start['alt'] is not None:
                    start_width, _ = font.size(alt_str[:min(cursor_pos['alt'], selection_start['alt'])])
                    end_width, _ = font.size(alt_str[:max(cursor_pos['alt'], selection_start['alt'])])
                    pygame.draw.rect(menu_screen, (0, 120, 215),
                                    (input_rects['alt'].x + 5 + start_width, input_rects['alt'].y + 5,
                                     end_width - start_width, 20), 2)

            elevation_mask_label = font.render("Elevation Mask (deg):", True, (0, 0, 0))
            menu_screen.blit(elevation_mask_label, (sub_x + 20, sub_y + 300))
            pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['elevation_mask'])
            elevation_mask_text = font.render(elevation_mask_str, True, (0, 0, 0))
            menu_screen.blit(elevation_mask_text, (input_rects['elevation_mask'].x + 5, input_rects['elevation_mask'].y + 5))
            if focused_field == 'elevation_mask':
                pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['elevation_mask'], 2)
                text_width, _ = font.size(elevation_mask_str[:cursor_pos['elevation_mask']])
                pygame.draw.line(menu_screen, (0, 0, 255),
                                (input_rects['elevation_mask'].x + 5 + text_width, input_rects['elevation_mask'].y + 5),
                                (input_rects['elevation_mask'].x + 5 + text_width, input_rects['elevation_mask'].y + 25), 2)
                if selection_start['elevation_mask'] is not None:
                    start_width, _ = font.size(elevation_mask_str[:min(cursor_pos['elevation_mask'], selection_start['elevation_mask'])])
                    end_width, _ = font.size(elevation_mask_str[:max(cursor_pos['elevation_mask'], selection_start['elevation_mask'])])
                    pygame.draw.rect(menu_screen, (0, 120, 215),
                                    (input_rects['elevation_mask'].x + 5 + start_width, input_rects['elevation_mask'].y + 5,
                                     end_width - start_width, 20), 2)

            # Draw buttons and divider
            pygame.draw.line(menu_screen, (0, 0, 0), (sub_x, sub_y + sub_height - 60), (sub_x + sub_width, sub_y + sub_height - 60), 1)
            draw_button(menu_screen, save_button, "Save", button_states["save"])
            draw_button(menu_screen, load_button, "Load", button_states["load"])
        elif current_mode == "tracking_vis" and tle_loaded:
            legend_x = sub_x + 20  # Define legend_x here
            legend_y = sub_y + 20  # Define legend_y here
            clear_filters_button = pygame.Rect(legend_x + 170, legend_y, 110, 30)  # Wider button (30 pixels more)
            sub_rect = (sub_x, sub_y, sub_width, sub_height)
            menu_screen.fill((0, 0, 0), sub_rect)

            # Interpolate satellite positions
            ts = load.timescale()
            t = ts.now()
            current_tt = t.tt
            satellite_positions = {}
            lat = float(lat_str)
            lon = float(lon_str)
            alt_m = float(alt_str)
            observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
            elevation_mask = float(elevation_mask_str) if elevation_mask_str.replace('.', '').isdigit() else 0.0
            max_alt = float(filter_alt_text) if filter_alt_text.replace('.', '').isdigit() else float('inf')

            for sat in satellites:
                if sat in satellite_trajectories and sat in satellite_labels:
                    px, py, alt = interpolate_position(satellite_trajectories[sat], current_tt)
                    if px is not None and py is not None and alt is not None:
                        if alt > elevation_mask and alt > 0 and satellite_mean_altitudes.get(sat, 0.0) <= max_alt:
                            if selected_satellite is None or sat == selected_satellite:
                                satellite_positions[sat] = (int(px), int(py))

            # Draw polar plot (static elements only, no per-frame math)
            cx = sub_x + sub_width // 2
            cy = sub_y + sub_height // 2
            radius = min(sub_width, sub_height) // 2 - 50
            # Draw horizon circle
            pygame.draw.circle(menu_screen, (255, 255, 255), (cx, cy), radius, 1)
            # Draw elevation mask circle
            mask_radius = (90 - elevation_mask) / 90 * radius
            pygame.draw.circle(menu_screen, (255, 0, 0), (cx, cy), mask_radius, 2)
            # Draw elevation circles
            for el in [30, 60]:
                r = (90 - el) / 90 * radius
                pygame.draw.circle(menu_screen, (100, 100, 100), (cx, cy), int(r), 1)
                el_label = small_font.render(f"{el}°", True, (255, 255, 255))
                menu_screen.blit(el_label, (cx + r + 5, cy - 5))
            # Draw azimuth lines and labels
            for az_deg in range(0, 360, 30):
                az_rad = math.radians(az_deg)
                x1 = cx + radius * math.sin(az_rad)
                y1 = cy - radius * math.cos(az_rad)
                pygame.draw.line(menu_screen, (100, 100, 100), (cx, cy), (x1, y1), 1)
                if az_deg % 90 == 0:
                    direction = {0: "N", 90: "E", 180: "S", 270: "W"}[az_deg]
                    direction_label = small_font.render(direction, True, (255, 255, 255))
                    # Ensure all cardinal directions are outside the circle with 10-pixel clearance
                    if az_deg == 0:  # North
                        menu_screen.blit(direction_label, (cx - direction_label.get_width() // 2, cy - radius - 10))
                    elif az_deg == 90:  # East
                        menu_screen.blit(direction_label, (cx + radius + 10, cy - direction_label.get_height() // 2))
                    elif az_deg == 180:  # South
                        menu_screen.blit(direction_label, (cx - direction_label.get_width() // 2, cy + radius + 10))
                    elif az_deg == 270:  # West
                        menu_screen.blit(direction_label, (cx - radius - 10 - direction_label.get_width(), cy - direction_label.get_height() // 2))
            # Draw precomputed arc segments for selected satellite
            if selected_satellite and tle_loaded and selected_satellite in satellite_arc_segments:
                for x0, y0, x1, y1, color in satellite_arc_segments[selected_satellite]:
                    pygame.draw.line(menu_screen, color, (x0, y0), (x1, y1), 1)
            # Draw details box
            if (hovered_satellite or selected_satellite) and current_mode == "tracking_vis":
                sat = selected_satellite if selected_satellite else hovered_satellite
                details = [
                    f"NORAD ID: {sat.model.satnum_str}",
                    f"Name: {sat.name.strip()}",
                    f"Mean Altitude (km): {satellite_mean_altitudes.get(sat, 0.0):.1f}",
                    f"Eccentricity: {sat.model.ecco:.4f}"
                ]
                details_rect = pygame.Rect(sub_x + sub_width - 250, sub_y + 20, 230, 200)
                pygame.draw.rect(menu_screen, (50, 50, 50), details_rect)  # Dark grey background
                pygame.draw.rect(menu_screen, (0, 0, 0), details_rect, 2)  # Black border
                for i, line in enumerate(details):
                    text_surface = small_font.render(line, True, (255, 255, 255))
                    menu_screen.blit(text_surface, (details_rect.x + 5, details_rect.y + 5 + i * 20))
            # Plot satellites with color and shape based on orbit type
            for sat, (px, py) in satellite_positions.items():
                if not filter_text or filter_text.lower() in sat.name.lower():
                    mean_altitude = satellite_mean_altitudes.get(sat, 0.0)
                    eccentricity = sat.model.ecco
                    if 2000 < mean_altitude <= 35786:  # MEO
                        color = (255, 165, 0)  # Orange
                        draw_hexagon(menu_screen, px, py, color)
                    elif abs(mean_altitude - 35786) <= 1000:  # GEO or nearby
                        color = (128, 0, 128)  # Purple
                        draw_triangle(menu_screen, px, py, color)
                    else:  # LEO (0-2000 km)
                        color = get_altitude_color(mean_altitude) or (0, 255, 0)  # Fallback to green if out of range
                        if eccentricity > 0.01:
                            width = 6
                            height = 3
                            angle = math.degrees(math.atan2(py - cy, px - cx))
                            oval_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                            pygame.draw.ellipse(oval_surface, color, (0, 0, width, height))
                            rotated_oval = pygame.transform.rotate(oval_surface, angle)
                            rotated_rect = rotated_oval.get_rect(center=(px, py))
                            menu_screen.blit(rotated_oval, rotated_rect.topleft)
                        else:
                            pygame.draw.circle(menu_screen, color, (px, py), 3)
                    if sat == hovered_satellite or sat == selected_satellite:
                        pygame.draw.circle(menu_screen, (255, 255, 0), (px, py), 5, 1)  # Highlight on hover or select
                    menu_screen.blit(satellite_labels[sat], (px + 5, py))
            # Draw filter boxes and labels above the boxes
            filter_label = small_font.render("Name Filter:", True, (255, 255, 255))
            menu_screen.blit(filter_label, (filter_rect.x, filter_rect.y - filter_label.get_height() - 5))
            pygame.draw.rect(menu_screen, (255, 255, 255), filter_rect)
            filter_text_surface = small_font.render(filter_text, True, (0, 0, 0))
            menu_screen.blit(filter_text_surface, (filter_rect.x + 5, filter_rect.y + 5))
            if focused_field == "filter":
                pygame.draw.rect(menu_screen, (0, 0, 255), filter_rect, 2)
                text_width, _ = small_font.size(filter_text[:cursor_pos["filter"]])
                pygame.draw.line(menu_screen, (0, 0, 255),
                                (filter_rect.x + 5 + text_width, filter_rect.y + 5),
                                (filter_rect.x + 5 + text_width, filter_rect.y + 25), 2)
                if selection_start["filter"] is not None:
                    start_width, _ = small_font.size(filter_text[:min(cursor_pos["filter"], selection_start["filter"])])
                    end_width, _ = small_font.size(filter_text[:max(cursor_pos["filter"], selection_start["filter"])])
                    pygame.draw.rect(menu_screen, (0, 120, 215),
                                    (filter_rect.x + 5 + start_width, filter_rect.y + 5,
                                     end_width - start_width, 20), 2)

            filter_alt_label = small_font.render("Alt Filter (km):", True, (255, 255, 255))
            menu_screen.blit(filter_alt_label, (filter_alt_rect.x, filter_alt_rect.y - filter_alt_label.get_height() - 5))
            pygame.draw.rect(menu_screen, (255, 255, 255), filter_alt_rect)
            filter_alt_text_surface = small_font.render(filter_alt_text, True, (0, 0, 0))
            menu_screen.blit(filter_alt_text_surface, (filter_alt_rect.x + 5, filter_alt_rect.y + 5))
            if focused_field == "filter_alt":
                pygame.draw.rect(menu_screen, (0, 0, 255), filter_alt_rect, 2)
                text_width, _ = small_font.size(filter_alt_text[:cursor_pos["filter_alt"]])
                pygame.draw.line(menu_screen, (0, 0, 255),
                                (filter_alt_rect.x + 5 + text_width, filter_alt_rect.y + 5),
                                (filter_alt_rect.x + 5 + text_width, filter_alt_rect.y + 25), 2)
                if selection_start["filter_alt"] is not None:
                    start_width, _ = small_font.size(filter_alt_text[:min(cursor_pos["filter_alt"], selection_start["filter_alt"])])
                    end_width, _ = small_font.size(filter_alt_text[:max(cursor_pos["filter_alt"], selection_start["filter_alt"])])
                    pygame.draw.rect(menu_screen, (0, 120, 215),
                                    (filter_alt_rect.x + 5 + start_width, filter_alt_rect.y + 5,
                                     end_width - start_width, 20), 2)

            # Draw legend for altitude heatmap and orbit types
            pygame.draw.rect(menu_screen, (50, 50, 50), (legend_x, legend_y, 150, 140))  # Larger legend
            pygame.draw.line(menu_screen, (255, 255, 255), (legend_x, legend_y + 20), (legend_x + 150, legend_y + 20), 1)  # Title line
            legend_title = small_font.render("Orbit Legend", True, (255, 255, 255))
            menu_screen.blit(legend_title, (legend_x + 5, legend_y + 5))
            # LEO heatmap
            for i in range(150):
                alt = i / 150 * 1000  # Scale from 0 to 1000 km
                color = get_altitude_color(alt) or (0, 255, 0)
                pygame.draw.line(menu_screen, color, (legend_x + i, legend_y + 40), (legend_x + i, legend_y + 60))
            low_alt = small_font.render("0 km", True, (255, 255, 255))
            high_alt = small_font.render("1000 km", True, (255, 255, 255))
            menu_screen.blit(low_alt, (legend_x, legend_y + 70))
            menu_screen.blit(high_alt, (legend_x + 140, legend_y + 70))
            # MEO hexagon
            draw_hexagon(menu_screen, legend_x + 20, legend_y + 90, (255, 165, 0))  # Orange
            meo_label = small_font.render("MEO (Orange)", True, (255, 255, 255))
            menu_screen.blit(meo_label, (legend_x + 40, legend_y + 85))
            # GEO triangle with line break
            draw_triangle(menu_screen, legend_x + 20, legend_y + 110, (128, 0, 128))  # Purple
            geo_label = small_font.render("GEO (Purple)", True, (255, 255, 255))
            menu_screen.blit(geo_label, (legend_x + 40, legend_y + 105))
            # Draw clear filters button
            draw_button(menu_screen, clear_filters_button, "Clear Filters", button_states["clear_filters"])
            # Draw time display in lower left
            current_utc = datetime.datetime.utcnow()
            current_local = current_utc - datetime.timedelta(hours=7)  # PDT is UTC-7
            utc_time_str = current_utc.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
            local_time_str = current_local.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
            time_text = f"UTC: {utc_time_str}  Local: {local_time_str}"
            time_surface = small_font.render(time_text, True, (255, 255, 255))
            menu_screen.blit(time_surface, (sub_x + 10, sub_y + sub_height - 30))
        elif current_mode == "sensor_calib":
            sub_rect = (sub_x, sub_y, sub_width, sub_height)
            menu_screen.fill((50, 50, 50), sub_rect)
            # Add sensor-mount calibration code here later
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