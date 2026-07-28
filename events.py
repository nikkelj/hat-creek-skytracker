"""
Event handling module for SkyTracker Application.
Contains modular event handlers for different modes and UI interactions.
"""

import pygame
from pygame.locals import *
import math





def handle_tracking_vis_keyboard_events(event, state):
    """Handle keyboard events for tracking visualization mode text input fields.
    Modifies state object directly instead of returning updates."""

    if state.focused_field is None:
        return

    # Get current field value
    focused_field = state.focused_field

    if focused_field == "filter":
        field_str = state.filter_text
    elif focused_field == "filter_above_alt":
        field_str = state.filter_above_alt_text
    elif focused_field == "filter_below_alt":
        field_str = state.filter_below_alt_text
    elif focused_field == "center_time":
        field_str = state.center_time_str
    elif focused_field == "duration":
        field_str = state.duration_str
    else:
        return

    mods = pygame.key.get_mods()

    # Handle navigation and text editing keys
    if event.key == pygame.K_LEFT:
        if mods & pygame.KMOD_SHIFT:
            state.selection_start[focused_field] = state.cursor_pos[focused_field] if state.selection_start[focused_field] is None else state.selection_start[focused_field]
            state.cursor_pos[focused_field] = max(0, state.cursor_pos[focused_field] - 1)
        else:
            state.cursor_pos[focused_field] = max(0, state.cursor_pos[focused_field] - 1)
            state.selection_start[focused_field] = None

    elif event.key == pygame.K_RIGHT:
        if mods & pygame.KMOD_SHIFT:
            state.selection_start[focused_field] = state.cursor_pos[focused_field] if state.selection_start[focused_field] is None else state.selection_start[focused_field]
            state.cursor_pos[focused_field] = min(len(field_str), state.cursor_pos[focused_field] + 1)
        else:
            state.cursor_pos[focused_field] = min(len(field_str), state.cursor_pos[focused_field] + 1)
            state.selection_start[focused_field] = None

    elif event.key == pygame.K_HOME:
        if mods & pygame.KMOD_SHIFT:
            state.selection_start[focused_field] = state.cursor_pos[focused_field] if state.selection_start[focused_field] is None else state.selection_start[focused_field]
        state.cursor_pos[focused_field] = 0
        if not mods & pygame.KMOD_SHIFT:
            state.selection_start[focused_field] = None

    elif event.key == pygame.K_END:
        if mods & pygame.KMOD_SHIFT:
            state.selection_start[focused_field] = state.cursor_pos[focused_field] if state.selection_start[focused_field] is None else state.selection_start[focused_field]
        state.cursor_pos[focused_field] = len(field_str)
        if not mods & pygame.KMOD_SHIFT:
            state.selection_start[focused_field] = None

    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
        start = min(state.cursor_pos[focused_field], state.selection_start[focused_field]) if state.selection_start[focused_field] is not None else state.cursor_pos[focused_field]
        end = max(state.cursor_pos[focused_field], state.selection_start[focused_field]) if state.selection_start[focused_field] is not None else state.cursor_pos[focused_field] + 1 if event.key == pygame.K_DELETE else state.cursor_pos[focused_field]
        if start < end:
            field_str = field_str[:start] + field_str[end:]
            state.cursor_pos[focused_field] = start
            state.selection_start[focused_field] = None
        elif event.key == pygame.K_BACKSPACE and state.cursor_pos[focused_field] > 0:
            field_str = field_str[:state.cursor_pos[focused_field] - 1] + field_str[state.cursor_pos[focused_field]:]
            state.cursor_pos[focused_field] -= 1
            state.selection_start[focused_field] = None

        # Update the appropriate field
        if focused_field == "filter":
            state.filter_text = field_str
        elif focused_field == "filter_above_alt":
            state.filter_above_alt_text = field_str
        elif focused_field == "filter_below_alt":
            state.filter_below_alt_text = field_str
        elif focused_field == "center_time":
            state.center_time_str = field_str
        elif focused_field == "duration":
            state.duration_str = field_str

    elif event.key == pygame.K_RETURN:
        state.focused_field = None
        # Note: we don't clear selection_start for focused_field here since focused_field is now None

    # Handle character input
    else:
        char = event.unicode
        allowed_chars_map = {
            "filter": lambda c: c.isalnum() or c in [' ', '-', '_'],
            "filter_above_alt": lambda c: c.isdigit() or c in ['.', '-'],
            "filter_below_alt": lambda c: c.isdigit() or c in ['.', '-'],
            "center_time": lambda c: c.isalnum() or c in ['-', ':', 'T', 'Z', '.'],
            "duration": lambda c: c.isdigit()
        }

        if allowed_chars_map.get(focused_field, lambda c: False)(char):
            start = min(state.cursor_pos[focused_field], state.selection_start[focused_field]) if state.selection_start[focused_field] is not None else state.cursor_pos[focused_field]
            end = max(state.cursor_pos[focused_field], state.selection_start[focused_field]) if state.selection_start[focused_field] is not None else state.cursor_pos[focused_field]
            field_str = field_str[:start] + char + field_str[end:]
            state.cursor_pos[focused_field] += 1
            state.selection_start[focused_field] = None

            # Update the appropriate field
            if focused_field == "filter":
                state.filter_text = field_str
            elif focused_field == "filter_above_alt":
                state.filter_above_alt_text = field_str
            elif focused_field == "filter_below_alt":
                state.filter_below_alt_text = field_str
            elif focused_field == "center_time":
                state.center_time_str = field_str
            elif focused_field == "duration":
                state.duration_str = field_str


def handle_tracking_vis_mouse_events_state(state, event, pos, display, button_states):
    """Updated mouse event handler that works with TrackingVisState object."""

    updates = {}

    # Handle button clicks - updated to use state attributes
    if display.clear_filters_button.collidepoint(pos):
        button_states["clear_filters"]["clicked"] = True
        state.filter_text = ""
        state.filter_above_alt_text = ""
        state.filter_below_alt_text = ""
        state.cursor_pos.update({"filter": 0, "filter_above_alt": 0, "filter_below_alt": 0})
        state.selection_start.update({"filter": None, "filter_above_alt": None, "filter_below_alt": None})
        state.selected_satellite = None
        return True  # Return True to indicate state was updated

    elif display.pause_button.collidepoint(pos):
        button_states["pause"]["clicked"] = True
        state.paused = True
        # Freeze the tracking clock AT ITS CURRENT VALUE -- which is the
        # slider position if the user scrubbed. Latching wall-clock now here
        # would snap a scrubbed scene back to live time on pause.
        cur = float(getattr(state, 'current_tt', 0.0) or 0.0)
        if cur > 2400000.0:  # real TT Julian dates are ~2.46e6; 0/None = unset
            state.paused_tt = cur
        else:
            from skyfield.api import load
            state.paused_tt = load.timescale().now().tt
        return True

    elif display.play_button.collidepoint(pos):
        button_states["play"]["clicked"] = True
        state.paused = False
        state.paused_tt = None
        return True

    elif display.scroll_bar_rect.collidepoint(pos):
        state.dragging_slider = True
        state.scroll_start = pos[0]
        return True

    elif display.filter_rect.collidepoint(pos):
        state.focused_field = 'filter'
        state.cursor_pos['filter'] = len(state.filter_text)
        state.selection_start['filter'] = None
        return True

    elif display.filter_above_alt_rect.collidepoint(pos):
        state.focused_field = 'filter_above_alt'
        state.cursor_pos['filter_above_alt'] = len(state.filter_above_alt_text)
        state.selection_start['filter_above_alt'] = None
        return True

    elif display.filter_below_alt_rect.collidepoint(pos):
        state.focused_field = 'filter_below_alt'
        state.cursor_pos['filter_below_alt'] = len(state.filter_below_alt_text)
        state.selection_start['filter_below_alt'] = None
        return True

    elif display.center_time_rect.collidepoint(pos):
        state.focused_field = 'center_time'
        state.cursor_pos['center_time'] = len(state.center_time_str)
        state.selection_start['center_time'] = None
        return True

    elif display.duration_rect.collidepoint(pos):
        state.focused_field = 'duration'
        state.cursor_pos['duration'] = len(state.duration_str)
        state.selection_start['duration'] = None
        return True

    elif display.recompute_button.collidepoint(pos):
        button_states["recompute"]["clicked"] = True
        state.recompute_triggered = True
        return True

    elif display.reset_button.collidepoint(pos):
        button_states["reset"]["clicked"] = True
        state.reset_input_fields()
        return True

    elif display.launch_button.collidepoint(pos) and hasattr(state, 'selected_launch') and state.selected_launch:
        button_states["launch"]["clicked"] = True
        # Toggle launch visualization
        if state.launch_launched:
            # Turn off launch visualization and reset to start
            state.launch_launched = False
            state.launch_start_time = None
        else:
            # Start launch visualization from current time
            from skyfield.api import load
            state.launch_launched = True
            state.launch_start_time = load.timescale().now().tt
        return True

    # Check pass table clicks (shared with joystick-mode skyplot overlay)
    if handle_pass_table_click(state, pos):
        return True

    # While a launch is active it is the sole target -- don't let plot clicks
    # select/track a satellite (a launched rocket overrides any satellite target).
    if getattr(state, 'selected_launch', None) and getattr(state, 'launch_launched', False):
        return False

    # Satellite selection - if no table click
    if state.selected_satellite and state.satellite_positions.get(state.selected_satellite):
        px, py, _, _ = state.satellite_positions[state.selected_satellite]
        if math.hypot(px - pos[0], py - pos[1]) < 10:
            state.selected_satellite = None
            return True

    # Select new satellite
    for sat, (px, py, _, _) in state.satellite_positions.items():
        if math.hypot(px - pos[0], py - pos[1]) < 10:
            state.selected_satellite = sat
            state.selected_celestial = None  # mutual exclusivity
            return True

    # Celestial objects (sun/moon/planets, named stars, Messier/NGC): same
    # full-screen coordinate convention. Checked after satellites so the
    # sparser satellite layer keeps priority. No config_state here, so the
    # tracking trajectory is built by the control loop's ensure call instead.
    for ckey, cpos in list(getattr(state, 'celestial_positions', {}).items()):
        if math.hypot(cpos[0] - pos[0], cpos[1] - pos[1]) < 10:
            from celestial import select_celestial
            select_celestial(state, None, ckey)
            return True

    return False


def handle_pass_table_click(state, pos):
    """Handle a click on the satellite pass table / launch box (row select,
    column-header sort, launch select). Returns True if consumed. Shared by the
    full-screen tracking-vis handler and the joystick-mode skyplot overlay; works
    off state.pass_table_clickable_areas (absolute screen rects), so it is
    independent of where the table was drawn."""
    if hasattr(state.pass_table_clickable_areas, '__iter__') and state.pass_table_clickable_areas:
        for area_type, index, rect in state.pass_table_clickable_areas:
            if rect.collidepoint(pos):
                if area_type == 'row' and state.satellite_pass_table and index < len(state.satellite_pass_table):
                    state.selected_satellite = state.satellite_pass_table[index]['satellite']
                    if hasattr(state, 'selected_aircraft'):
                        state.selected_aircraft = None  # mutual exclusivity
                    return True
                elif area_type == 'header':
                    # Column-header sort. Plain click = single-column sort (re-click
                    # toggles direction). Shift-click = add/toggle a secondary sort
                    # tier, enabling multi-column sorting.
                    from tracking_visuals import PASS_TABLE_DEFAULT_DESC
                    col = index
                    shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    order = list(state.table_sort_order or [])
                    existing = next((k for k, (c, _) in enumerate(order) if c == col), None)
                    default_desc = PASS_TABLE_DEFAULT_DESC.get(col, True)
                    if shift:
                        if existing is not None:
                            c, r = order[existing]
                            order[existing] = (c, not r)
                        else:
                            order.append((col, default_desc))
                    else:
                        if len(order) == 1 and order[0][0] == col:
                            order = [(col, not order[0][1])]
                        else:
                            order = [(col, default_desc)]
                    state.table_sort_order = order
                    # Keep legacy single-sort fields roughly in sync (primary key)
                    if order:
                        primary_col, primary_rev = order[0]
                        state.table_sort_keys = [j == primary_col for j in range(len(state.table_sort_keys))]
                        state.table_sort_reverse = [primary_rev for _ in range(len(state.table_sort_reverse))]
                    # Reset scroll to top so the new ordering is visible from the start
                    state.pass_table_scroll_offset = 0
                    return True
                elif area_type == 'launch':
                    # Handle launch trajectory selection
                    launch_name = index
                    if state.selected_launch == launch_name:
                        # Clicking again deselects the launch
                        state.selected_launch = None
                        state.launch_launched = False
                        state.launch_start_time = None
                    else:
                        # Select new launch and deselect satellite
                        state.selected_launch = launch_name
                        state.selected_satellite = None
                        state.launch_launched = False
                        state.launch_start_time = None
                    return True
                break

    return False


def handle_tracking_vis_hover_events(event, pos, display, button_states, satellite_positions):
    """Handle mouse motion/hover events for tracking visualization mode."""

    updates = {}

    # Update button hover states
    button_hover_map = {
        "clear_filters": display.clear_filters_button,
        "recompute": display.recompute_button,
        "reset": display.reset_button,
        "pause": display.pause_button,
        "play": display.play_button
    }

    for button_name, button_rect in button_hover_map.items():
        updates[f"{button_name}_hover"] = button_rect.collidepoint(pos)

    # Update hovered satellite
    hovered_satellite = None
    for sat, (px, py, _, _) in satellite_positions.items():
        if math.hypot(px - pos[0], py - pos[1]) < 10:
            hovered_satellite = sat
            break

    updates["hovered_satellite"] = hovered_satellite
    return updates


def handle_main_menu_events(event, buttons, current_mode):
    """Handle main menu navigation events for switching between application modes.
    Returns the new mode to switch to, or "exit" to quit, or current_mode if no change."""

    # Only handle MOUSEBUTTONDOWN events
    if event.type != pygame.MOUSEBUTTONDOWN:
        return current_mode

    pos = event.pos

    # Check each button for collision
    for btn in buttons:
        if btn["rect"].collidepoint(pos):
            mode = btn["mode"]

            if mode == "exit":
                return "exit"
            else:
                return mode  # Return the new mode to switch to

    # No button clicked
    return current_mode


# Additional event handling functions can be added here
