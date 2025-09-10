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
from utils import draw_button, create_negative_image, draw_menu_button, draw_button_with_objects
from trajectory import precompute_trajectories, interpolate_position, clear_trajectory_cache, update_satellite_positions, build_satellite_pass_table
from config import load_config, save_config, handle_input, ConfigState
from tracking_visuals import TrackingVisState, draw_polar_plot, draw_satellites, draw_legend, draw_details, draw_filters, draw_time_display, draw_satellite_count, draw_scroll_bar, draw_scroll_time_display, draw_satellite_pass_table, filter_and_sort_pass_table, PolarPlotMode
from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
from camera_manager import camera_manager, render_sensor_calibration, render_camera_sliders, render_camera_roi_controls, render_combined_view_controls, update_camera_frames_from_buffers, handle_sensor_calib_events
from joystick_controller import JoystickModeState, render_joystick_mode, handle_joystick_mode_events
# Camera button initialization is now handled internally by camera_manager
from events import *

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Update Intervals (seconds)
POSITION_UPDATE_INTERVAL = 0.1  # 10 Hz
TRAJECTORY_UPDATE_INTERVAL = 900  # 15 minutes

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

# Load configuration using ConfigState
config_state = load_config()

# Input field state for tracking vis (now handled by TrackingVisState class)

current_mode = None
clock = pygame.time.Clock()

# Author background image now handled by DisplaySetup class

# Initial render of main menu
status_messages = ["Starting TLE process..."]
display.render_initial_menu(status_messages)
print(f"Debug: Status - {'Starting TLE process...'}")

# Tracking Visualization State Management
global tracking_vis_state
tracking_vis_state = TrackingVisState()

# Joystick Mode State Management
global joystick_mode_state
joystick_mode_state = JoystickModeState()

# Define status update callback for satellite loading
def update_status_callback(message):
    status_messages.append(message)
    display.menu_screen.fill(display.COLOR_BACKGROUND_DARK, (0, 0, display.MENU_WIDTH, display.total_height))  # Dark theme menu background
    for btn in display.buttons:
        draw_menu_button(display, btn)
    if display.bg_image_menu:
        display.menu_screen.blit(display.bg_image_menu, display.image_rect.topleft if display.image_rect else ((display.MENU_WIDTH - 160) // 2, display.image_y))
    for i, msg in enumerate(status_messages[-4:]):
        status_render = display.status_font.render(msg, True, display.COLOR_TEXT_WHITE)
        display.menu_screen.blit(status_render, (10, display.status_y_start + i * 14))
    pygame.display.flip()
    pygame.time.wait(50)  # Brief pause to ensure display update
    print(f"Debug: Status - {message}")

ts = load.timescale()

try:
    from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
    load_satellite_data(tracking_vis_state, update_status_callback)

    # Pre-compute satellite labels and mean altitudes using refactored module
    if tracking_vis_state.tle_loaded and tracking_vis_state.satellites:
        create_satellite_labels_and_metadata(tracking_vis_state, update_status_callback)
    else:
        update_status_callback("WARNING: No TLEs or Satellites loaded!!!")
except Exception as e:
    status_messages.append(f"TLE loading error: {str(e)}")
    print(f"Debug: Error loading TLEs: {e}")
    # Force immediate trajectory computation after TLEs are loaded
    last_trajectory_update = 0

# Force immediate trajectory computation with no delays
recompute_triggered = True
# Force immediate timer trigger
last_trajectory_update = -1
last_update_time = 0
update_interval = 0.1  # Target 10 Hz
last_trajectory_update = -1  # Force immediate trigger on first loop
trajectory_interval = 900  # 15 minutes

# ==============================================================================
# MAIN PROGRAM LOOP
# ==============================================================================

running = True
while running:
    current_time = time.time()
    mouse_pos = pygame.mouse.get_pos()


    if tracking_vis_state.recompute_triggered and tracking_vis_state.tle_loaded:
        try:
            center_time_obj = datetime.fromisoformat(tracking_vis_state.center_time_str.replace('Z', '+00:00'))
            center_time = center_time_obj.replace(tzinfo=timezone.utc)
            duration_minutes = float(tracking_vis_state.duration_str)
            lat = float(config_state.lat_str or 0)
            lon = float(config_state.lon_str or 0)
            alt_m = float(config_state.alt_str or 0)
            observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
            tracking_vis_state.t0 = ts.utc((center_time - timedelta(minutes=duration_minutes / 2)))
            tracking_vis_state.t1 = ts.utc((center_time + timedelta(minutes=duration_minutes / 2)))

            # Update scroll bar time range labels in state
            tracking_vis_state.scroll_bar_start_label = "-" + str(duration_minutes/2) + " min"
            tracking_vis_state.scroll_bar_end_label = "+" + str(duration_minutes/2) + " min"

            update_status_callback("Recomputing trajectories...")
            precompute_trajectories(tracking_vis_state, observer, ts, display, update_status_callback, center_time, duration_minutes)

            # Build satellite pass table (filtered for current visibility + upcoming passes)
            build_satellite_pass_table(tracking_vis_state, elevation_mask_deg=float(config_state.elevation_mask_str or 10.0), ts=ts)

            last_trajectory_update = current_time
            update_status_callback("Trajectories recomputed")
            tracking_vis_state.selected_satellite = None
            tracking_vis_state.paused = False
            tracking_vis_state.paused_tt = None
            fraction = max(0.0, min(1.0, (ts.now().tt - tracking_vis_state.t0.tt) / (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)))
            display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))
        except Exception as e:
            update_status_callback(f"Error recomputing: {str(e)}")
        tracking_vis_state.recompute_triggered = False

    # Precompute trajectories and arc segments every 15 minutes
    if current_time - last_trajectory_update >= trajectory_interval:
        update_status_callback("Starting trajectory precomputation...")
        lat = float(config_state.lat_str or 0)
        lon = float(config_state.lon_str or 0)
        alt_m = float(config_state.alt_str or 0)
        observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
        # Use current time as center and user-specified duration
        duration_minutes_auto = float(tracking_vis_state.duration_str) if tracking_vis_state.duration_str else 30.0
        current_utc = datetime.now(timezone.utc)
        tracking_vis_state.t0 = ts.utc(current_utc - timedelta(minutes=duration_minutes_auto/2))
        tracking_vis_state.t1 = ts.utc(current_utc + timedelta(minutes=duration_minutes_auto/2))

        # Update scroll bar time range labels based on the duration being used
        tracking_vis_state.scroll_bar_start_label = "-" + str(duration_minutes_auto/2) + " min"
        tracking_vis_state.scroll_bar_end_label = "+" + str(duration_minutes_auto/2) + " min"

        precompute_trajectories(tracking_vis_state, observer, ts, display, update_status_callback)

        # Build satellite pass table (filtered for current visibility + upcoming passes)
        build_satellite_pass_table(tracking_vis_state, elevation_mask_deg=float(config_state.elevation_mask_str or 10.0), ts=ts)

        last_trajectory_update = current_time
        update_status_callback("Trajectories updated")
        # Initialize slider to current time
        fraction = (ts.now().tt - tracking_vis_state.t0.tt) / (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
        display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))

    # Compute satellite positions at 10 Hz
    if current_time - last_update_time >= update_interval:
        tracking_vis_state.satellite_positions = {}

        # Calculate current time for trajectory interpolation - ensure it's never None
        if tracking_vis_state.dragging_slider and tracking_vis_state.t0 is not None and tracking_vis_state.t1 is not None:
            # User is actively dragging slider - use slider position
            fraction = (display.slider_rect.x - display.scroll_bar_rect.x) / (display.scroll_bar_rect.width - display.slider_rect.width)
            current_tt = tracking_vis_state.t0.tt + fraction * (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
        elif tracking_vis_state.paused and tracking_vis_state.paused_tt is not None:
            # Simulation is paused - use paused time
            current_tt = tracking_vis_state.paused_tt
        else:
            # Default to current time
            current_tt = ts.now().tt

            # Update slider position to reflect real-time (only if trajectories are available)
            if tracking_vis_state.t0 is not None and tracking_vis_state.t1 is not None and not tracking_vis_state.dragging_slider:
                fraction = (current_tt - tracking_vis_state.t0.tt) / (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
                display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))
        # Use state-direct mutation approach for satellite position updates
        # Only update positions if trajectories are available
        if tracking_vis_state.tle_loaded and tracking_vis_state.satellites:
            update_satellite_positions(tracking_vis_state, current_tt, elevation_mask_deg=float(config_state.elevation_mask_str or 0))
        last_update_time = current_time

    # Handle events using modular event system
    # Always process joystick events first, regardless of current mode
    # Check for existing joysticks that were connected before this point
    if not hasattr(joystick_mode_state, '_initialized'):
        # Get current connected joysticks and add them to the state
        for i in range(pygame.joystick.get_count()):
            joy = pygame.joystick.Joystick(i)
            joystick_mode_state.joysticks[joy.get_instance_id()] = joy
            print(f"Existing joystick {joy.get_instance_id()} detected: {joy.get_name()}")
            # Auto-connect to first joystick
            if joystick_mode_state.connected_joystick is None:
                joystick_mode_state.connected_joystick = joy.get_instance_id()
                joystick_mode_state.reset_tare()
        joystick_mode_state._initialized = True

    for event in pygame.event.get():
        # Always handle joystick events regardless of current mode
        if event.type in [pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED, pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION]:
            joystick_mode_state.process_joystick_events(event)

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif current_mode == "config_options" and config_state.focused_field:
                handle_input(event, config_state)
            elif current_mode == "tracking_vis":
                # Handle keyboard input for tracking visualization fields
                from events import handle_tracking_vis_keyboard_events
                handle_tracking_vis_keyboard_events(event, tracking_vis_state)

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
                    tracking_vis_state.focused_field = None
                    config_state.reset_input_fields()  # Reset config input fields
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
                    tracking_vis_state.focused_field = None
                    config_state.reset_input_fields()  # Reset config input fields
                    break  # Exit inner loop to process new mode

            # Handle mode-specific events
            elif current_mode == "sensor_calib":
                # Handle sensor calibration events using modular handler
                handle_sensor_calib_events(event, pos, display, camera_manager, update_status_callback)
            elif current_mode == "config_options":
                # Handle save button click
                if display.save_button.collidepoint(pos) and str(pos) and len(str(pos).strip()) > 2:
                    display.button_states["save"]["clicked"] = True
                    config_state.save_to_file()
                    status_messages.append("Configuration saved")
                    break

                # Handle load button click
                elif display.load_button.collidepoint(pos) and str(pos) and len(str(pos).strip()) > 2:
                    display.button_states["load"]["clicked"] = True
                    try:
                        config_state.load_from_file()
                        status_messages.append("Configuration loaded")
                    except Exception as e:
                        status_messages.append(f"Error loading config: {e}")
                    break

                # Handle input field focus
                for field_name, input_rect in display.input_rects.items():
                    if input_rect.collidepoint(pos):
                        config_state.focused_field = field_name
                        config_state.cursor_pos[field_name] = 0
                        config_state.selection_start[field_name] = None
                        break

            # Handle tracking_vis events using state-based approach
            elif current_mode == "tracking_vis":
                try:
                    # Handle events using state-based handler
                    handle_tracking_vis_mouse_events_state(tracking_vis_state, event, pos, display, display.button_states)

                except Exception as e:
                    print(f"DEBUG: Error in tracking_vis mouse events: {e}")
                    print(f"DEBUG: Current mode: {current_mode}, Mouse pos: {pos}, Event type: {event.type}")
                    import traceback
                    traceback.print_exc()

            # Handle joystick mode events
            elif current_mode == "joystick_loop":
                handle_joystick_mode_events(event, joystick_mode_state, display)


        elif event.type == pygame.MOUSEMOTION:
            if current_mode is None:
                # Handle main menu hover states
                for btn in display.buttons:
                    display.button_states[btn["mode"]]["hover"] = btn["rect"].collidepoint(event.pos)
            elif current_mode == "config_options":
                display.button_states["save"]["hover"] = display.save_button.collidepoint(event.pos)
                display.button_states["load"]["hover"] = display.load_button.collidepoint(event.pos)
            elif current_mode == "sensor_calib":
                # Camera slider hover states handled by modular camera code
                handle_sensor_calib_events(event, pos, display, camera_manager, update_status_callback)
            elif current_mode == "tracking_vis":
                display.button_states["clear_filters"]["hover"] = display.clear_filters_button.collidepoint(event.pos)
                display.button_states["recompute"]["hover"] = display.recompute_button.collidepoint(event.pos)
                display.button_states["reset"]["hover"] = display.reset_button.collidepoint(event.pos)
                display.button_states["pause"]["hover"] = display.pause_button.collidepoint(event.pos)
                display.button_states["play"]["hover"] = display.play_button.collidepoint(event.pos)
                if tracking_vis_state.dragging_slider:
                    display.slider_rect.x = max(display.scroll_bar_rect.x, min(event.pos[0] - display.slider_rect.width // 2, display.scroll_bar_rect.x + display.scroll_bar_rect.width - display.slider_rect.width))
                for sat, (px, py, _, _) in tracking_vis_state.satellite_positions.items():
                    if math.hypot(px - event.pos[0], py - event.pos[1]) < 10:
                        tracking_vis_state.hovered_satellite = sat
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
                handle_sensor_calib_events(event, pos, display, camera_manager, update_status_callback)
            elif current_mode == "tracking_vis":
                display.button_states["clear_filters"]["clicked"] = False
                display.button_states["reset"]["clicked"] = False
                display.button_states["pause"]["clicked"] = False
                display.button_states["play"]["clicked"] = False
                # Reset dragging state
                if tracking_vis_state.dragging_slider and tracking_vis_state.paused:
                    fraction = (display.slider_rect.x - display.scroll_bar_rect.x) / (display.scroll_bar_rect.width - display.slider_rect.width)
                    tracking_vis_state.paused_tt = tracking_vis_state.t0.tt + fraction * (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
                tracking_vis_state.dragging_slider = False
                tracking_vis_state.scroll_start = None

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
        draw_menu_button(display, btn)
    # Render status messages each frame
    status_messages = status_messages[-4:]  # Keep last 4 messages
    for i, msg in enumerate(status_messages):
        status_render = display.status_font.render(msg, True, display.COLOR_TEXT_WHITE)  # White text for dark theme
        display.menu_screen.blit(status_render, (10, display.status_y_start + i * 14))

    cx = display.sub_x + display.sub_width // 2
    cy = display.sub_y + display.sub_height // 2
    if current_mode == "config_options":
        draw_polar_plot(display, config_state, None, None, None)
    elif current_mode == "tracking_vis" and tracking_vis_state.tle_loaded:
        display.menu_screen.fill((0, 0, 0), (display.sub_x, display.sub_y, display.sub_width, display.sub_height))  # Clear the subplot area with black
        draw_polar_plot(display, config_state, ts, current_tt, tracking_vis_state)
        draw_satellites(display, tracking_vis_state, cx, cy, PolarPlotMode.FULL_SCREEN)
        draw_filters(display, tracking_vis_state)
        draw_legend(display)
        draw_details(display, tracking_vis_state)
        draw_time_display(display)
        draw_satellite_count(display, tracking_vis_state)
        # Modular pass table filtering and sorting - replaced ~30 lines of inline filtering logic
        from tracking_visuals import filter_and_sort_pass_table
        sorted_filtered_pass_table = filter_and_sort_pass_table(tracking_vis_state)

        # Draw satellite pass table
        draw_satellite_pass_table(display, tracking_vis_state)
        draw_button_with_objects(display, "clear_filters")
        draw_button_with_objects(display, "pause")
        draw_button_with_objects(display, "play")
        draw_scroll_bar(display, tracking_vis_state)
        draw_scroll_time_display(display, current_tt, ts)
        # Draw Recompute Button
        draw_button_with_objects(display, "recompute")
        # Draw Reset Button
        draw_button_with_objects(display, "reset")
        # Draw Center Time Input
        pygame.draw.rect(display.menu_screen, (255, 255, 255), display.center_time_rect)
        if tracking_vis_state.focused_field == 'center_time':
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.center_time_rect, 2)
        text_surface = display.small_font.render(tracking_vis_state.center_time_str, True, (0, 0, 0))
        display.menu_screen.blit(text_surface, (display.center_time_rect.x + 5, display.center_time_rect.y + 5))
        if tracking_vis_state.focused_field == 'center_time':
            text_width, _ = display.small_font.size(tracking_vis_state.center_time_str[:tracking_vis_state.cursor_pos['center_time']])
            pygame.draw.line(display.menu_screen, (0, 0, 255), (display.center_time_rect.x + 5 + text_width, display.center_time_rect.y + 5), (display.center_time_rect.x + 5 + text_width, display.center_time_rect.y + 25), 2)
        # Label for Center Time
        label = display.small_font.render("Center Time:", True, (255, 255, 255))
        display.menu_screen.blit(label, (display.center_time_rect.x, display.center_time_rect.y - 15))
        # Draw Duration Input
        pygame.draw.rect(display.menu_screen, (255, 255, 255), display.duration_rect)
        if tracking_vis_state.focused_field == 'duration':
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.duration_rect, 2)
        text_surface = display.small_font.render(tracking_vis_state.duration_str, True, (0, 0, 0))
        display.menu_screen.blit(text_surface, (display.duration_rect.x + 5, display.duration_rect.y + 5))
        if tracking_vis_state.focused_field == 'duration':
            text_width, _ = display.small_font.size(tracking_vis_state.duration_str[:tracking_vis_state.cursor_pos['duration']])
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
        render_joystick_mode(display, joystick_mode_state, tracking_vis_state, config_state)
        # Update joystick rate control
        if joystick_mode_state.telescope_connected:
            joystick_mode_state.rate_control()
    elif current_mode == "post_process":
        sub_rect = (display.sub_x, display.sub_y, display.sub_width, display.sub_height)
        display.menu_screen.fill((150, 150, 150), sub_rect)
        # Add post-processing tool code here later
    elif current_mode == "author_info":
        sub_rect = (display.sub_x, display.sub_y, display.sub_width, display.sub_height)
        if display.author_bg:
            scaled_bg = pygame.transform.scale(display.author_bg, (display.sub_width, display.sub_height))
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
