import pygame
import os
import sys
import webbrowser
import math
from skyfield.api import load, wgs84, EarthSatellite
import time
import requests
import json

if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    pygame.init()
    display_info = pygame.display.Info()
    menu_width = 200
    total_width = display_info.current_w
    total_height = display_info.current_h
    menu_screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Main Menu")

    # Load background image for menu (assume 'cli/lucky.jpg' exists)
    try:
        bg_image = pygame.image.load('cli/lucky.jpg')
        bg_image = pygame.transform.scale(bg_image, (80, 80))  # Shrink to 80x80
    except pygame.error:
        bg_image = None  # Fallback to solid color if image not found
        print("Warning: 'cli/lucky.jpg' not found. Using fallback color.")

    font = pygame.font.Font(None, 24)
    large_font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 14)  # Smaller font for labels to save space
    status_font = pygame.font.Font(None, 12)  # Reduced font size for status messages

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
    status_y_start = total_height - 12 * 4  # Space for 4 lines, 12 pixels each (adjusted for smaller font)

    current_mode = None
    clock = pygame.time.Clock()

    # For author info
    author_bg = None
    try:
        author_bg = pygame.image.load('cli/lucky.jpg')
    except pygame.error:
        pass

    # Configuration defaults
    lat_str = "34.87405877829887"
    lon_str = "-120.44621926328121"
    alt_str = "120.0"
    focused_field = None  # None, 'lat', 'lon', 'alt'

    # Input rects for config
    input_rects = {
        'lat': pygame.Rect(sub_x + 20, sub_y + 50, 200, 30),
        'lon': pygame.Rect(sub_x + 20, sub_y + 100, 200, 30),
        'alt': pygame.Rect(sub_x + 20, sub_y + 150, 200, 30),
    }

    # Initial render of main menu
    menu_screen.fill((200, 200, 200), (0, 0, menu_width, total_height))  # Menu background
    if bg_image:
        menu_screen.blit(bg_image, ((menu_width - 80) // 2, image_y))  # Center horizontally underneath buttons
    for btn in buttons:
        pygame.draw.rect(menu_screen, (211, 211, 211), btn["rect"])  # Light grey button color
        text = font.render(btn["text"], True, (0, 0, 0))
        menu_screen.blit(text, (btn["rect"].x + 10, btn["rect"].y + 30))
    status_messages = ["Starting TLE process..."]
    for i, msg in enumerate(status_messages[-4:]):
        status_render = status_font.render(msg, True, (0, 0, 0))
        menu_screen.blit(status_render, (10, status_y_start + i * 12))
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
                menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 12))
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
                menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 12))
                pygame.display.flip()
                print(f"Debug: Status - {status_messages[-1]}")
                # Load from cache
                with open(cache_file, 'r') as f:
                    tle_text = f.read()
        else:
            status_messages.append("Downloading TLEs from Celestrak...")
            status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
            menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 12))
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
        menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 12))
        pygame.display.flip()
        print(f"Debug: Status - {status_messages[-1]}")

        # Process TLE text into satellites
        satellites = load.tle_file(cache_file)
        status_messages.append("TLEs ready")
        status_render = status_font.render(status_messages[-1], True, (0, 0, 0))
        menu_screen.blit(status_render, (10, status_y_start + (len(status_messages) - 1) * 12))
        pygame.display.flip()
        print(f"Debug: Status - {status_messages[-1]}")
        tle_loaded = True
    except Exception as e:
        print(f"Debug: Error loading TLEs in text format: {e}")

    # Pre-compute satellite labels
    satellite_labels = {}
    for sat in satellites:
        name = sat.name.strip()  # Use the satellite name from TLE
        norad_id = sat.model.satnum_str  # Use as-is, no zfill
        label_text = f"{norad_id} - {name}"
        satellite_labels[sat] = small_font.render(label_text, True, (255, 255, 255))

    last_update_time = 0
    update_interval = 2.0  # Update satellite positions every 2 seconds

    running = True
    while running:
        current_time = time.time()
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
                            current_mode = btn["mode"]
                            focused_field = None  # Reset focus when switching modes
                if current_mode:
                    if current_mode == "author_info":
                        contact_text = "Jonathan Nikkel - @NikkelJonathan"
                        text2 = large_font.render(contact_text, True, (255, 255, 255))
                        text_rect = text2.get_rect(topleft=(sub_x + 10, sub_y + 50))
                        if text_rect.collidepoint(pos):
                            webbrowser.open("https://x.com/NikkelJonathan")
                    elif current_mode == "config_options":
                        for field, rect in input_rects.items():
                            if rect.collidepoint(pos):
                                focused_field = field
                                break
            if event.type == pygame.KEYDOWN and current_mode == "config_options" and focused_field:
                if event.key == pygame.K_BACKSPACE:
                    if focused_field == 'lat':
                        lat_str = lat_str[:-1]
                    elif focused_field == 'lon':
                        lon_str = lon_str[:-1]
                    elif focused_field == 'alt':
                        alt_str = alt_str[:-1]
                elif event.key == pygame.K_RETURN:
                    focused_field = None
                else:
                    char = event.unicode
                    if char.isdigit() or char in ['.', '-', '+']:
                        if focused_field == 'lat':
                            lat_str += char
                        elif focused_field == 'lon':
                            lon_str += char
                        elif focused_field == 'alt':
                            alt_str += char

        menu_screen.fill((200, 200, 200), (0, 0, menu_width, total_height))  # Menu background

        if bg_image:
            menu_screen.blit(bg_image, ((menu_width - 80) // 2, image_y))  # Center horizontally underneath buttons

        for btn in buttons:
            pygame.draw.rect(menu_screen, (211, 211, 211), btn["rect"])  # Light grey button color
            text = font.render(btn["text"], True, (0, 0, 0))
            menu_screen.blit(text, (btn["rect"].x + 10, btn["rect"].y + 30))

        # Status note with scrolling stack (max 4 messages)
        status_messages = status_messages[-4:]  # Keep last 4 messages
        for i, msg in enumerate(status_messages):
            status_render = status_font.render(msg, True, (0, 0, 0))
            menu_screen.blit(status_render, (10, status_y_start + i * 12))

        if current_mode:
            sub_rect = (sub_x, sub_y, sub_width, sub_height)
            if current_mode == "tracking_vis" and tle_loaded:
                menu_screen.fill((0, 0, 0), sub_rect)
                if current_time - last_update_time >= update_interval:
                    try:
                        lat = float(lat_str)
                        lon = float(lon_str)
                        alt_m = float(alt_str)
                        observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
                        ts = load.timescale()
                        t = ts.now()
                        # Update satellite positions
                        satellite_positions = {}
                        for sat in satellites:
                            if sat in satellite_labels:  # Only process satellites with valid labels
                                difference = sat - observer
                                topocentric = difference.at(t)
                                alt, az, distance = topocentric.altaz()
                                if alt.degrees > 0:
                                    r = (90 - alt.degrees) / 90 * min(sub_width, sub_height) // 2 - 50
                                    theta = math.radians(az.degrees)
                                    px = sub_x + sub_width // 2 + r * math.sin(theta)
                                    py = sub_y + sub_height // 2 - r * math.cos(theta)
                                    satellite_positions[sat] = (int(px), int(py))
                        last_update_time = current_time
                    except ValueError:
                        pass
                # Draw polar plot
                cx = sub_x + sub_width // 2
                cy = sub_y + sub_height // 2
                radius = min(sub_width, sub_height) // 2 - 50
                # Draw horizon circle
                pygame.draw.circle(menu_screen, (255, 255, 255), (cx, cy), radius, 1)
                # Draw elevation circles
                for el in [30, 60]:
                    r = (90 - el) / 90 * radius
                    pygame.draw.circle(menu_screen, (100, 100, 100), (cx, cy), int(r), 1)
                # Draw azimuth lines
                for az_deg in range(0, 360, 30):
                    az_rad = math.radians(az_deg)
                    x1 = cx + radius * math.sin(az_rad)
                    y1 = cy - radius * math.cos(az_rad)
                    pygame.draw.line(menu_screen, (100, 100, 100), (cx, cy), (x1, y1), 1)
                # Plot satellites
                for sat, (px, py) in satellite_positions.items():
                    pygame.draw.circle(menu_screen, (0, 255, 0), (px, py), 3)
                    menu_screen.blit(satellite_labels[sat], (px + 5, py))
            elif current_mode == "sensor_calib":
                menu_screen.fill((50, 50, 50), sub_rect)
                # Add sensor-mount calibration code here later
            elif current_mode == "joystick_loop":
                menu_screen.fill((100, 100, 100), sub_rect)
                # Add manual joystick loop code here later
            elif current_mode == "post_process":
                menu_screen.fill((150, 150, 150), sub_rect)
                # Add post-processing tool code here later
            elif current_mode == "config_options":
                menu_screen.fill((200, 200, 0), sub_rect)
                # Draw labels and inputs
                lat_label = font.render("Latitude:", True, (0, 0, 0))
                menu_screen.blit(lat_label, (sub_x + 20, sub_y + 20))
                pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['lat'])
                lat_text = font.render(lat_str, True, (0, 0, 0))
                menu_screen.blit(lat_text, (sub_x + 25, sub_y + 55))
                if focused_field == 'lat':
                    pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['lat'], 2)

                lon_label = font.render("Longitude:", True, (0, 0, 0))
                menu_screen.blit(lon_label, (sub_x + 20, sub_y + 70))
                pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['lon'])
                lon_text = font.render(lon_str, True, (0, 0, 0))
                menu_screen.blit(lon_text, (sub_x + 25, sub_y + 105))
                if focused_field == 'lon':
                    pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['lon'], 2)

                alt_label = font.render("Altitude (m):", True, (0, 0, 0))
                menu_screen.blit(alt_label, (sub_x + 20, sub_y + 120))
                pygame.draw.rect(menu_screen, (255, 255, 255), input_rects['alt'])
                alt_text = font.render(alt_str, True, (0, 0, 0))
                menu_screen.blit(alt_text, (sub_x + 25, sub_y + 155))
                if focused_field == 'alt':
                    pygame.draw.rect(menu_screen, (0, 0, 255), input_rects['alt'], 2)
                # Add configuration options code here later
            elif current_mode == "author_info":
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