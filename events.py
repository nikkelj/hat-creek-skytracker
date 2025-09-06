"""
Event handling module for SkyTracker Application.
Contains modular event handlers for different modes and UI interactions.
"""

import pygame
from pygame.locals import *
import math

# Remove duplicate status_callback function - should import from main or make global

# Combined button states dictionary for all UI elements
button_states = {
    # Main menu buttons
    "save": {"hover": False, "clicked": False},
    "load": {"hover": False, "clicked": False},
    "clear_filters": {"hover": False, "clicked": False},
    "recompute": {"hover": False, "clicked": False},
    "reset": {"hover": False, "clicked": False},
    "pause": {"hover": False, "clicked": False},
    "play": {"hover": False, "clicked": False},
    # Camera control buttons
    "camera1_connect": {"hover": False, "clicked": False},
    "camera2_connect": {"hover": False, "clicked": False},
    "camera1_disconnect": {"hover": False, "clicked": False},
    "camera2_disconnect": {"hover": False, "clicked": False},
    "camera1_gain_slider": {"hover": False, "clicked": False},
    "camera2_gain_slider": {"hover": False, "clicked": False},
    "camera1_exposure_slider": {"hover": False, "clicked": False},
    "camera2_exposure_slider": {"hover": False, "clicked": False},
}

# Define missing camera slider rectangles
CAMERA1_GAIN_SLIDER_RECT = pygame.Rect(300, 50, 200, 20)
CAMERA2_GAIN_SLIDER_RECT = pygame.Rect(300, 120, 200, 20)
CAMERA1_EXPOSURE_SLIDER_RECT = pygame.Rect(300, 200, 200, 20)
CAMERA2_EXPOSURE_SLIDER_RECT = pygame.Rect(300, 270, 200, 20)

# Camera connection status flags
camera1_connected = False
camera2_connected = False

def handle_main_menu_events(event, buttons, current_mode):
    """Handle main menu button click events."""
    pos = event.pos
    for btn in buttons:
        if btn["rect"].collidepoint(pos):
            return btn["mode"]
    return current_mode

def handle_config_events(event, pos, input_rects, save_button, load_button,
    lat_str, lon_str, alt_str, elevation_mask_str,
    focused_field, cursor_pos, selection_start,
    button_states):
    """Handle configuration mode events - extracted from main event loop."""
    if save_button.collidepoint(pos):
        button_states["save"]["clicked"] = True
        from config import save_config
        config_data = {
            "lat": lat_str,
            "lon": lon_str,
            "alt": alt_str,
            "elevation_mask": elevation_mask_str
        }
        save_config(config_data)
        return {"status": "Configuration saved"}

    elif load_button.collidepoint(pos):
        button_states["load"]["clicked"] = True
        from config import load_config
        config_data = load_config()
        return {
            "lat_str": config_data["lat"],
            "lon_str": config_data["lon"],
            "alt_str": config_data["alt"],
            "elevation_mask_str": config_data["elevation_mask"],
            "status": "Configuration loaded"
        }

    elif event.type == pygame.MOUSEBUTTONDOWN:
        # Handle input field selection
        for field_name, rect in input_rects.items():
            if rect.collidepoint(pos):
                focused_field = field_name
                cursor_pos[field_name] = len(eval(f"{field_name}_str"))
                selection_start[field_name] = None
                return {"focused_field": field_name, "status": f"Selected {field_name} input"}

    return None

def handle_sensor_calib_events(event, pos, button_states):
    """Handle sensor calibration mode events - extracted from main event loop."""
    result = None

    if event.type == pygame.MOUSEBUTTONDOWN:
        # Check camera control buttons
        if hasattr(pos, '__iter__'):  # Ensure pos is valid
            # Camera connection buttons
            camera1_connect_rect = pygame.Rect(10, 10, 120, 30)
            camera1_disconnect_rect = pygame.Rect(10, 50, 120, 30)
            camera2_connect_rect = pygame.Rect(10, 100, 120, 30)
            camera2_disconnect_rect = pygame.Rect(10, 140, 120, 30)

            if str(pos) and len(str(pos).strip()) > 2:  # Basic validation
                if camera1_connect_rect.collidepoint(pos):
                    button_states["camera1_connect"]["clicked"] = True
                    global camera1_connected
                    camera1_connected = True
                    result = {"status": "Camera 1 connected", "action": "connect_camera1"}
                elif camera1_disconnect_rect.collidepoint(pos):
                    button_states["camera1_disconnect"]["clicked"] = True
                    camera1_connected = False
                    result = {"status": "Camera 1 disconnected", "action": "disconnect_camera1"}
                elif camera2_connect_rect.collidepoint(pos):
                    button_states["camera2_connect"]["clicked"] = True
                    global camera2_connected
                    camera2_connected = True
                    result = {"status": "Camera 2 connected", "action": "connect_camera2"}
                elif camera2_disconnect_rect.collidepoint(pos):
                    button_states["camera2_disconnect"]["clicked"] = True
                    camera2_connected = False
                    result = {"status": "Camera 2 disconnected", "action": "disconnect_camera2"}

                # Handle slider interactions (simplified)
                elif CAMERA1_GAIN_SLIDER_RECT.collidepoint(pos):
                    button_states["camera1_gain_slider"]["clicked"] = True
                    result = {"status": "Camera 1 gain slider clicked", "action": "adjust_gain_camera1"}
                elif CAMERA2_GAIN_SLIDER_RECT.collidepoint(pos):
                    button_states["camera2_gain_slider"]["clicked"] = True
                    result = {"status": "Camera 2 gain slider clicked", "action": "adjust_gain_camera2"}
                elif CAMERA1_EXPOSURE_SLIDER_RECT.collidepoint(pos):
                    button_states["camera1_exposure_slider"]["clicked"] = True
                    result = {"status": "Camera 1 exposure slider clicked", "action": "adjust_exposure_camera1"}
                elif CAMERA2_EXPOSURE_SLIDER_RECT.collidepoint(pos):
                    button_states["camera2_exposure_slider"]["clicked"] = True
                    result = {"status": "Camera 2 exposure slider clicked", "action": "adjust_exposure_camera2"}

    elif event.type == pygame.MOUSEMOTION:
        # Handle hover states
        camera1_connect_rect = pygame.Rect(10, 10, 120, 30)
        camera2_connect_rect = pygame.Rect(10, 100, 120, 30)
        camera1_disconnect_rect = pygame.Rect(10, 50, 120, 30)
        camera2_disconnect_rect = pygame.Rect(10, 140, 120, 30)

        button_states["camera1_connect"]["hover"] = camera1_connect_rect.collidepoint(event.pos)
        button_states["camera1_disconnect"]["hover"] = camera1_disconnect_rect.collidepoint(event.pos)
        button_states["camera2_connect"]["hover"] = camera2_connect_rect.collidepoint(event.pos)
        button_states["camera2_disconnect"]["hover"] = camera2_disconnect_rect.collidepoint(event.pos)
        button_states["camera1_gain_slider"]["hover"] = CAMERA1_GAIN_SLIDER_RECT.collidepoint(event.pos)
        button_states["camera2_gain_slider"]["hover"] = CAMERA2_GAIN_SLIDER_RECT.collidepoint(event.pos)
        button_states["camera1_exposure_slider"]["hover"] = CAMERA1_EXPOSURE_SLIDER_RECT.collidepoint(event.pos)
        button_states["camera2_exposure_slider"]["hover"] = CAMERA2_EXPOSURE_SLIDER_RECT.collidepoint(event.pos)

    return result

def handle_tracking_vis_events(event, pos, clear_filters_button, pause_button, play_button,
    center_time_rect, duration_rect, filter_rect, filter_above_alt_rect,
    filter_below_alt_rect, scroll_bar_rect, recompute_button, reset_button,
    focused_field, cursor_pos, selection_start, paused, paused_tt,
    button_states, filter_text, filter_above_alt_text, filter_below_alt_text,
    satellite_pass_table):
    """Handle tracking visualization mode events - extracted from main event loop."""

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
        return {
            "action": "clear_filters",
            "filter_text": "",
            "filter_above_alt_text": "",
            "filter_below_alt_text": ""
        }

    elif pause_button.collidepoint(pos):
        button_states["pause"]["clicked"] = True
        paused = True
        paused_tt = None  # Will be set when ts.now().tt is available
        return {
            "action": "pause",
            "paused": True
        }

    elif play_button.collidepoint(pos):
        button_states["play"]["clicked"] = True
        paused = False
        paused_tt = None
        return {
            "action": "play",
            "paused": False
        }

    elif scroll_bar_rect.collidepoint(pos):
        # Scroll bar interaction would be handled here
        return {"action": "drag_slider_start"}

    elif filter_rect.collidepoint(pos):
        focused_field = 'filter'
        cursor_pos['filter'] = len(filter_text)
        selection_start['filter'] = None
        return {"action": "focus_field", "field": "filter"}

    elif filter_above_alt_rect.collidepoint(pos):
        focused_field = 'filter_above_alt'
        cursor_pos['filter_above_alt'] = len(filter_above_alt_text)
        selection_start['filter_above_alt'] = None
        return {"action": "focus_field", "field": "filter_above_alt"}

    elif filter_below_alt_rect.collidepoint(pos):
        focused_field = 'filter_below_alt'
        cursor_pos['filter_below_alt'] = len(filter_below_alt_text)
        selection_start['filter_below_alt'] = None
        return {"action": "focus_field", "field": "filter_below_alt"}

    elif center_time_rect.collidepoint(pos):
        focused_field = 'center_time'
        cursor_pos['center_time'] = len(center_time_str) if 'center_time_str' in globals() else 0
        selection_start['center_time'] = None
        return {"action": "focus_field", "field": "center_time"}

    elif duration_rect.collidepoint(pos):
        focused_field = 'duration'
        cursor_pos['duration'] = len(duration_str) if 'duration_str' in globals() else 0
        selection_start['duration'] = None
        return {"action": "focus_field", "field": "duration"}

    elif recompute_button.collidepoint(pos):
        button_states["recompute"]["clicked"] = True
        return {"action": "recompute"}

    elif reset_button.collidepoint(pos):
        button_states["reset"]["clicked"] = True
        return {"action": "reset"}

    return None

# Additional event handling functions can be added here
