"""
Joystick Loop panel rendering + panel mouse handlers.

Split out of joystick_controller.py (which keeps the JoystickModeState
tracking state machine, control handlers, and joystick input): the
safety-critical control loop and the paint code no longer live in one
4,700-line module. joystick_controller re-imports every public and private
name defined here, so ALL existing imports keep working -- treat
`joystick_controller` as the facade and this module as its rendering half.
"""

import math
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pygame

from camera_manager import camera_manager, update_camera_frames_from_buffers
from camera_manager import apply_gamma_correction, roi_sizes, roi_label_texts
from lib.auxstar import RATES, Targets
from trajectory import interpolate_position_data_and_rates
from utils import draw_button

# Names owned by the state-machine half. Resolved when joystick_controller
# imports this module at ITS bottom, by which point they are all defined.
from joystick_controller import (
    TrackingMode,
    BUTTON_LABELS,
    BUTTON_FUNCTIONS,
    button_function_map,
    axis_to_rate,
    joystick_panel_layout,
    joystick_center_layout,
    _draw_disabled_scrim,
    JL_LEAD_MIN,
    JL_LEAD_MAX,
    JL_ADSB_FIT_MIN,
    JL_ADSB_FIT_MAX,
)

def render_position_display(display, joystick_state):
    """Render the current mode / AZM / ALT position box.

    Anchored to the top-right corner of the joystick mode's upper-left
    quadrant. The box is always drawn (so its location is visible) but is
    greyed out when the telescope is not connected and there is no live data.
    """
    connected = joystick_state.telescope_connected

    # Box sized to envelope its text (the old 100x65 box clipped the values).
    width, height = 140, 86
    x_start = display.joystick_layout_params()['divider_x'] - width - 12
    y_start = display.sub_y + 12

    # Background color based on stop state
    if joystick_state.stopped:
        bg_color = (120, 60, 60)  # Reddish background when stopped
        border_color = (200, 100, 100)
    else:
        bg_color = (60, 60, 60)   # Normal grey when active
        border_color = (100, 100, 100)

    # Background rectangle for position display
    box_rect = pygame.Rect(x_start, y_start, width, height)
    pygame.draw.rect(display.menu_screen, bg_color, box_rect)
    pygame.draw.rect(display.menu_screen, border_color, box_rect, 1)

    # Mode/status indicator at the top
    if joystick_state.stopped:
        status_text = "STOPPED"
        status_color = (255, 100, 100)  # Red when stopped
    else:
        status_text = joystick_state.tracking_mode.name  # Show current tracking mode
        status_color = (100, 255, 100)  # Green when active
    status_render = display.small_font.render(status_text, True, status_color)
    display.menu_screen.blit(status_render, (x_start + 6, y_start + 4))

    # Add mount mode indicator
    mount_mode_text = display.tiny_font.render(f"Mount: {joystick_state.mount_mode}", True, (200, 200, 100))
    display.menu_screen.blit(mount_mode_text, (x_start + 6, y_start + 22))

    # AZM display
    azm_label_text = display.tiny_font.render("AZM", True, (255, 255, 255))
    display.menu_screen.blit(azm_label_text, (x_start + 6, y_start + 38))
    azm_value = joystick_state.azm_display_str[:14] if connected else "--"
    azm_value_text = display.tiny_font.render(azm_value, True, (0, 255, 0))
    display.menu_screen.blit(azm_value_text, (x_start + 34, y_start + 38))

    # ALT display
    alt_label_text = display.tiny_font.render("ALT", True, (255, 255, 255))
    display.menu_screen.blit(alt_label_text, (x_start + 6, y_start + 54))
    alt_value = joystick_state.alt_display_str[:14] if connected else "--"
    alt_value_text = display.tiny_font.render(alt_value, True, (0, 255, 0))
    display.menu_screen.blit(alt_value_text, (x_start + 34, y_start + 54))

    # Connection hint at the bottom
    conn_text = "Connected" if connected else "Disconnected"
    conn_color = (120, 220, 120) if connected else (200, 120, 120)
    conn_render = display.tiny_font.render(conn_text, True, conn_color)
    display.menu_screen.blit(conn_render, (x_start + 6, y_start + 70))

    # Grey the box out when there is no live telescope data
    if not connected:
        _draw_disabled_scrim(display, box_rect)

# ==============================================================================
# JOYSTICK MODE RENDERING FUNCTIONS
# ==============================================================================

def render_connection_controls(display, joystick_state):
    """Render connection controls in upper left"""
    # Update available ports
    joystick_state.get_available_serial_ports()

    # Connect/Disconnect buttons
    button_y = display.sub_y + 10
    button_width = 80
    button_height = 30

    # Connect button
    connect_rect = pygame.Rect(display.sub_x + 10, button_y, button_width, button_height)
    joystick_state.connect_button_hover = connect_rect.collidepoint(pygame.mouse.get_pos())

    if joystick_state.telescope_connected:
        button_color = (100, 100, 100)  # Grey when connected
    else:
        button_color = (100, 150, 100) if joystick_state.connect_button_hover else (100, 120, 100)

    pygame.draw.rect(display.menu_screen, button_color, connect_rect)
    connect_text = display.small_font.render("Connect", True, (255, 255, 255))
    display.menu_screen.blit(connect_text, (connect_rect.x + 5, connect_rect.y + 5))

    # Disconnect button
    disconnect_rect = pygame.Rect(display.sub_x + 100, button_y, button_width, button_height)
    joystick_state.disconnect_button_hover = disconnect_rect.collidepoint(pygame.mouse.get_pos())

    if not joystick_state.telescope_connected:
        button_color = (100, 100, 100)  # Grey when disconnected
    else:
        button_color = (150, 100, 100) if joystick_state.disconnect_button_hover else (120, 100, 100)

    pygame.draw.rect(display.menu_screen, button_color, disconnect_rect)
    disconnect_text = display.small_font.render("Disconnect", True, (255, 255, 255))
    display.menu_screen.blit(disconnect_text, (disconnect_rect.x + 5, disconnect_rect.y + 5))

    # Port selector
    port_y = button_y + 40
    port_label = display.small_font.render("Port:", True, (255, 255, 255))
    display.menu_screen.blit(port_label, (display.sub_x + 10, port_y))

    # Port dropdown box
    dropdown_width = 120
    dropdown_height = 25
    dropdown_rect = pygame.Rect(display.sub_x + 50, port_y, dropdown_width, dropdown_height)
    pygame.draw.rect(display.menu_screen, (70, 70, 70), dropdown_rect)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), dropdown_rect, 1)

    if joystick_state.selected_port:
        port_text = display.small_font.render(joystick_state.selected_port, True, (255, 255, 255))
    else:
        port_text = display.small_font.render("Select Port", True, (255, 255, 255))
    display.menu_screen.blit(port_text, (dropdown_rect.x + 5, dropdown_rect.y + 3))

    # Baud rate display (fixed)
    baud_y = port_y + 30
    baud_text = display.small_font.render("Baud: 9600 (fixed)", True, (255, 255, 255))
    display.menu_screen.blit(baud_text, (display.sub_x + 10, baud_y))

    # Connection status
    status_y = baud_y + 25
    if joystick_state.telescope_connected:
        status_text = display.small_font.render("Status: Connected", True, (0, 255, 0))
    else:
        status_text = display.small_font.render("Status: Disconnected", True, (255, 0, 0))
    display.menu_screen.blit(status_text, (display.sub_x + 10, status_y))

def render_adsb_connection_controls(display, joystick_state):
    """Render the ADS-B (RTL-SDR) connect/disconnect controls, status, and the
    linear-fit-points slider in a compact pane at the top-right of the joystick
    quadrant. Mirrors the telescope connection controls."""
    layout = joystick_panel_layout(display)
    pane = layout['adsb']
    cfg = getattr(joystick_state, 'config_state', None)
    mouse_pos = pygame.mouse.get_pos()

    # Mirror live receiver state so async failures (missing deps, device unplugged)
    # show immediately and the buttons reflect the real connection state.
    if joystick_state.adsb is not None:
        joystick_state.adsb_connected = joystick_state.adsb.connected
        joystick_state.adsb_status = joystick_state.adsb.status

    # Pane background + border.
    pygame.draw.rect(display.menu_screen, (28, 30, 38), pane)
    pygame.draw.rect(display.menu_screen, (90, 95, 110), pane, 1)

    mode = str(getattr(cfg, 'adsb_source_mode', 'rtlsdr')) if cfg else 'rtlsdr'
    title = display.small_font.render(f"ADS-B ({mode})", True, (200, 210, 230))
    display.menu_screen.blit(title, (pane.x + 6, pane.y + 2))

    # Connect / Disconnect buttons.
    bw, bh = 84, 20
    by = pane.y + 18
    connect_rect = pygame.Rect(pane.x + 6, by, bw, bh)
    disconnect_rect = pygame.Rect(pane.x + 6 + bw + 6, by, bw, bh)
    joystick_state.adsb_connect_button_hover = connect_rect.collidepoint(mouse_pos)
    joystick_state.adsb_disconnect_button_hover = disconnect_rect.collidepoint(mouse_pos)
    joystick_state.adsb_button_rects = {'connect': connect_rect, 'disconnect': disconnect_rect}

    if joystick_state.adsb_connected:
        c_color = (100, 100, 100)
    else:
        c_color = (100, 150, 100) if joystick_state.adsb_connect_button_hover else (100, 120, 100)
    pygame.draw.rect(display.menu_screen, c_color, connect_rect)
    display.menu_screen.blit(display.small_font.render("Connect", True, (255, 255, 255)),
                             (connect_rect.x + 6, connect_rect.y + 4))

    if not joystick_state.adsb_connected:
        d_color = (100, 100, 100)
    else:
        d_color = (150, 100, 100) if joystick_state.adsb_disconnect_button_hover else (120, 100, 100)
    pygame.draw.rect(display.menu_screen, d_color, disconnect_rect)
    display.menu_screen.blit(display.small_font.render("Disconnect", True, (255, 255, 255)),
                             (disconnect_rect.x + 4, disconnect_rect.y + 4))

    # Status line (connected count + receiver status).
    status_y = by + bh + 2
    n_ac = 0
    tvs = getattr(joystick_state, 'tracking_vis_state', None)
    if tvs is not None:
        n_ac = len(getattr(tvs, 'aircraft_positions', None) or {})
    if joystick_state.adsb_connected:
        stat = f"Connected - {n_ac} aircraft"
        col = (0, 220, 0)
    else:
        stat = "Disconnected"
        col = (220, 90, 90)
    display.menu_screen.blit(display.tiny_font.render(stat, True, col), (pane.x + 6, status_y))
    if joystick_state.adsb_status:
        display.menu_screen.blit(
            display.tiny_font.render(joystick_state.adsb_status[:42], True, (150, 155, 170)),
            (pane.x + 120, status_y))

    # Fit-points slider (number of recent fixes fit for linear prediction).
    fit_val = int(getattr(cfg, 'adsb_fit_points', 5)) if cfg else 5
    slider_y = status_y + 16
    display.menu_screen.blit(display.tiny_font.render("Fit pts:", True, (255, 200, 100)),
                             (pane.x + 6, slider_y))
    display.menu_screen.blit(display.tiny_font.render(f"{fit_val}", True, (255, 255, 255)),
                             (pane.x + 52, slider_y))
    track = pygame.Rect(pane.x + 78, slider_y + 6, pane.width - 90, 4)
    display.joystick_adsb_fit_slider_rect = track
    pygame.draw.rect(display.menu_screen, (150, 150, 150), track)
    ratio = 0.0
    if JL_ADSB_FIT_MAX > JL_ADSB_FIT_MIN:
        ratio = min(1.0, max(0.0, (fit_val - JL_ADSB_FIT_MIN) / (JL_ADSB_FIT_MAX - JL_ADSB_FIT_MIN)))
    handle_x = track.x + int(ratio * track.width)
    hover = pygame.Rect(handle_x - 3, track.y - 4, 6, 12).collidepoint(mouse_pos)
    pygame.draw.rect(display.menu_screen, (255, 0, 0) if hover else (200, 0, 0),
                     (handle_x - 3, track.y - 4, 6, 12))


def _adsb_fit_from_track_x(track, x):
    """Map an x pixel on the ADS-B fit slider track to an integer fit-points count."""
    rel = min(max(x - track.x, 0), track.width)
    frac = rel / track.width if track.width else 0.0
    return int(round(JL_ADSB_FIT_MIN + frac * (JL_ADSB_FIT_MAX - JL_ADSB_FIT_MIN)))


def handle_adsb_fit_slider_mouse_events(joystick_state, display, mouse_pos):
    """Click/drag on the ADS-B fit-points slider sets config.adsb_fit_points live.
    Returns True if the click hit the track."""
    if not hasattr(display, 'joystick_adsb_fit_slider_rect'):
        return False
    track = display.joystick_adsb_fit_slider_rect
    cfg = getattr(joystick_state, 'config_state', None)
    if cfg is None or not track.collidepoint(mouse_pos):
        return False
    cfg.adsb_fit_points = _adsb_fit_from_track_x(track, mouse_pos[0])
    return True


def render_joystick_target_panel(display, joystick_state, tracking_vis_state, config_state):
    """Skyplot 'Targets' overlay in the joystick upper-right quadrant.

    Always-on toggle strip (Targets panel open/close, Sats show/hide, Labels
    show/hide). When the panel is open, draws the name/alt filter boxes and the
    sortable satellite-passes + launches table over the skyplot. Selection,
    sorting and filtering reuse the tracking-vis machinery (draw_filters /
    draw_satellite_pass_table / filter_and_sort_pass_table), so behaviour matches
    the full-screen tracking visualization mode."""
    if tracking_vis_state is None:
        return
    screen = display.menu_screen
    font = display.small_font
    mouse_pos = pygame.mouse.get_pos()
    qx = display.joystick_layout_params()['divider_x']
    qy = display.sub_y
    qw = display.sub_x + display.sub_width - qx
    qh = display.sub_height // 2

    strip_x, strip_y, bh = qx + 8, qy + 6, 22

    def _btn(label, x, w, active, on_color, off_color):
        rect = pygame.Rect(x, strip_y, w, bh)
        base = on_color if active else off_color
        col = tuple(min(255, c + 25) for c in base) if rect.collidepoint(mouse_pos) else base
        pygame.draw.rect(screen, col, rect)
        pygame.draw.rect(screen, (150, 150, 160), rect, 1)
        screen.blit(font.render(label, True, (255, 255, 255)), (rect.x + 6, rect.y + 4))
        return rect

    t_rect = _btn("Targets " + ("▲" if joystick_state.targets_panel_open else "▼"),
                  strip_x, 96, joystick_state.targets_panel_open, (70, 90, 120), (60, 70, 90))
    sats_on = getattr(config_state, 'satellites_enabled', True)
    s_rect = _btn("Sats " + ("On" if sats_on else "Off"), strip_x + 102, 66, sats_on,
                  (70, 110, 70), (90, 70, 70))
    labels_on = getattr(config_state, 'satellite_labels_enabled', True)
    l_rect = _btn("Labels " + ("On" if labels_on else "Off"), strip_x + 172, 84, labels_on,
                  (70, 110, 70), (90, 70, 70))
    joystick_state.jl_target_btn_rects = {'targets': t_rect, 'sats': s_rect, 'labels': l_rect}

    if not joystick_state.targets_panel_open:
        joystick_state.jl_filter_rects = {}
        joystick_state.jl_clear_filters_rect = None
        joystick_state.jl_pass_table_rect = None
        joystick_state.jl_panel_rect = None
        return

    # Overlay panel over the skyplot.
    panel_x = qx + 6
    panel_y = strip_y + bh + 6
    panel_w = min(380, qw - 12)
    panel_h = qh - (panel_y - qy) - 10
    panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    joystick_state.jl_panel_rect = panel
    scrim = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
    scrim.fill((18, 20, 28, 235))
    screen.blit(scrim, (panel.x, panel.y))
    pygame.draw.rect(screen, (90, 95, 110), panel, 1)

    # Filter boxes (reuse draw_filters with quadrant-local rects).
    fx, fw, fhgt = panel.x + 12, 150, 28
    jl_filter_rects = {
        'filter': pygame.Rect(fx, panel.y + 24, fw, fhgt),
        'filter_above_alt': pygame.Rect(fx, panel.y + 78, fw, fhgt),
        'filter_below_alt': pygame.Rect(fx, panel.y + 132, fw, fhgt),
    }
    joystick_state.jl_filter_rects = jl_filter_rects
    from tracking_visuals import draw_filters, draw_satellite_pass_table, filter_and_sort_pass_table
    draw_filters(display, tracking_vis_state, rects=jl_filter_rects)

    # Clear-filters button.
    clr = pygame.Rect(fx + fw + 14, panel.y + 24, 90, 24)
    joystick_state.jl_clear_filters_rect = clr
    pygame.draw.rect(screen, (110, 90, 90) if clr.collidepoint(mouse_pos) else (90, 75, 75), clr)
    pygame.draw.rect(screen, (160, 150, 150), clr, 1)
    screen.blit(font.render("Clear", True, (255, 255, 255)), (clr.x + 8, clr.y + 4))

    # Sortable passes + launches table. Apply live filter/sort each frame, then
    # draw into a quadrant-local box (clickable areas are absolute -> selection
    # and header-sort clicks work via the shared handle_pass_table_click).
    tracking_vis_state.satellite_pass_table = filter_and_sort_pass_table(tracking_vis_state)
    table_top = panel.y + 172
    table_h = max(120, panel.bottom - table_top - 96)  # leave room for launch box
    table_box = pygame.Rect(panel.x + 8, table_top, panel_w - 16, table_h)
    joystick_state.jl_pass_table_rect = table_box
    draw_satellite_pass_table(display, tracking_vis_state, box=table_box)


def handle_joystick_target_panel_click(joystick_state, tracking_vis_state, config_state, pos):
    """Handle clicks on the skyplot Targets overlay (toggle strip + panel). Returns
    True if the click was consumed. Called before skyplot selection so panel
    interactions take precedence over selecting objects behind the panel."""
    btns = getattr(joystick_state, 'jl_target_btn_rects', {}) or {}
    t = btns.get('targets')
    if t and t.collidepoint(pos):
        joystick_state.targets_panel_open = not joystick_state.targets_panel_open
        return True
    if config_state is not None:
        s = btns.get('sats')
        if s and s.collidepoint(pos):
            config_state.satellites_enabled = not getattr(config_state, 'satellites_enabled', True)
            return True
        lbl = btns.get('labels')
        if lbl and lbl.collidepoint(pos):
            config_state.satellite_labels_enabled = not getattr(config_state, 'satellite_labels_enabled', True)
            return True

    if not joystick_state.targets_panel_open or tracking_vis_state is None:
        return False

    # Filter boxes -> focus for text entry (reuses tracking_vis_state.focused_field).
    for field, rect in (getattr(joystick_state, 'jl_filter_rects', {}) or {}).items():
        if rect.collidepoint(pos):
            tracking_vis_state.focused_field = field
            tracking_vis_state.cursor_pos[field] = len(_filter_field_text(tracking_vis_state, field))
            tracking_vis_state.selection_start[field] = None
            return True

    clr = getattr(joystick_state, 'jl_clear_filters_rect', None)
    if clr and clr.collidepoint(pos):
        tracking_vis_state.filter_text = ""
        tracking_vis_state.filter_above_alt_text = ""
        tracking_vis_state.filter_below_alt_text = ""
        tracking_vis_state.cursor_pos.update({"filter": 0, "filter_above_alt": 0, "filter_below_alt": 0})
        tracking_vis_state.selection_start.update({"filter": None, "filter_above_alt": None, "filter_below_alt": None})
        tracking_vis_state.focused_field = None
        return True

    # Table row / header / launch clicks (shared handler).
    from events import handle_pass_table_click
    if handle_pass_table_click(tracking_vis_state, pos):
        tracking_vis_state.focused_field = None
        return True

    # Click anywhere else inside the panel: consume so it doesn't select an object
    # on the skyplot behind the panel, and drop filter-box focus.
    panel = getattr(joystick_state, 'jl_panel_rect', None)
    if panel and panel.collidepoint(pos):
        tracking_vis_state.focused_field = None
        return True
    return False


def _filter_field_text(state, field):
    return {'filter': state.filter_text,
            'filter_above_alt': state.filter_above_alt_text,
            'filter_below_alt': state.filter_below_alt_text}.get(field, "")


def render_joystick_status(display, joystick_state):
    """Render the joystick status (button map + axes) below the connection
    controls.

    The full button map and axis displays are always drawn so their location
    is visible, with each button labelled by the functionality currently
    assigned to it. When no joystick is connected the whole block is greyed
    out (visible but inactive).
    """
    connected = (joystick_state.connected_joystick is not None and
                 joystick_state.connected_joystick in joystick_state.joysticks)
    joy = joystick_state.joysticks[joystick_state.connected_joystick] if connected else None
    num_buttons = joy.get_numbuttons() if connected else 0
    num_axes = joy.get_numaxes() if connected else 0

    base_x = display.sub_x + 10
    y_start = display.sub_y + 140

    # Joystick name (kept outside the grey scrim so its status stays readable)
    if connected:
        name_text = display.small_font.render(f"Joystick: {joy.get_name()}", True, (255, 255, 255))
    else:
        name_text = display.small_font.render("Joystick: None", True, (255, 0, 0))
    display.menu_screen.blit(name_text, (base_x, y_start))

    # ---- Buttons section ---------------------------------------------------
    region_top = y_start + 25
    buttons_label = display.small_font.render("Buttons (function):", True, (255, 255, 255))
    display.menu_screen.blit(buttons_label, (base_x, region_top))

    buttons_top = region_top + 20
    col_w = 130
    func_map = button_function_map(joystick_state)
    swatch_w, swatch_h = 26, 18
    row_h = 22
    num_slots = len(BUTTON_LABELS)
    rows_per_col = (num_slots + 1) // 2  # 8 rows over two columns for 16 buttons

    for i in range(num_slots):
        col = i // rows_per_col
        row = i % rows_per_col
        bx = base_x + col * col_w
        by = buttons_top + row * row_h

        active = connected and i < num_buttons and joy.get_button(i)
        swatch_color = (0, 255, 0) if active else (90, 90, 90)
        swatch_rect = pygame.Rect(bx, by, swatch_w, swatch_h)
        pygame.draw.rect(display.menu_screen, swatch_color, swatch_rect)
        pygame.draw.rect(display.menu_screen, (150, 150, 150), swatch_rect, 1)

        label = BUTTON_LABELS[i] if i < len(BUTTON_LABELS) else str(i)
        label_color = (0, 0, 0) if active else (230, 230, 230)
        label_text = display.tiny_font.render(label, True, label_color)
        display.menu_screen.blit(label_text, label_text.get_rect(center=swatch_rect.center))

        func = func_map.get(i, "-")
        func_color = (220, 220, 220) if i in func_map else (120, 120, 120)
        func_text = display.tiny_font.render(func, True, func_color)
        display.menu_screen.blit(func_text, (bx + swatch_w + 4, by + 4))

    current_y = buttons_top + rows_per_col * row_h + 12

    # ---- Axes section ------------------------------------------------------
    axes_label = display.small_font.render("Axes:", True, (255, 255, 255))
    display.menu_screen.blit(axes_label, (base_x, current_y))
    current_y += 20

    # First two axis pairs as 2D crosshair boxes (left/right sticks)
    box_size = 60
    crosshair_range = 20
    for pair in range(2):
        axis_x, axis_y = pair * 2, pair * 2 + 1
        box_x, box_y = base_x, current_y
        center_x = box_x + box_size // 2
        center_y = box_y + box_size // 2

        pygame.draw.rect(display.menu_screen, (80, 80, 80), (box_x, box_y, box_size, box_size))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (box_x, box_y, box_size, box_size), 1)

        x_val = joy.get_axis(axis_x) if connected and axis_x < num_axes else 0.0
        y_val = joy.get_axis(axis_y) if connected and axis_y < num_axes else 0.0
        crosshair_x = center_x + int(x_val * crosshair_range)
        crosshair_y = center_y + int(y_val * crosshair_range)

        pygame.draw.line(display.menu_screen, (255, 255, 255),
                         (crosshair_x, center_y - crosshair_range),
                         (crosshair_x, center_y + crosshair_range), 1)
        pygame.draw.line(display.menu_screen, (255, 255, 255),
                         (center_x - crosshair_range, crosshair_y),
                         (center_x + crosshair_range, crosshair_y), 1)

        pair_label = "Left Stick" if pair == 0 else "Right Stick"
        label_text = display.tiny_font.render(pair_label, True, (255, 255, 255))
        display.menu_screen.blit(label_text, (base_x + box_size + 10, current_y + 20))

        current_y += box_size + 10

    # Triggers (L2/R2) as linear sliders
    for idx, ax_label in ((4, "L2"), (5, "R2")):
        slider_width, slider_height = 100, 12
        slider_x, slider_y = base_x, current_y
        pygame.draw.rect(display.menu_screen, (80, 80, 80), (slider_x, slider_y, slider_width, slider_height))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (slider_x, slider_y, slider_width, slider_height), 1)

        axis_val = joy.get_axis(idx) if connected and idx < num_axes else -1.0
        slider_pos = int((axis_val + 1) / 2 * slider_width)
        pygame.draw.rect(display.menu_screen, (255, 255, 0),
                         (slider_x + slider_pos - 2, slider_y - 2, 4, slider_height + 4))

        val_text = display.tiny_font.render(f"{ax_label}: {axis_val:+.2f}", True, (255, 255, 255))
        display.menu_screen.blit(val_text, (slider_x + slider_width + 10, slider_y))
        current_y += slider_height + 8

    # Focus motor: the L2/R2 triggers above drive it (L2 = retract/-, R2 = extend
    # /+). Show the commanded rate as a centre-zero bar plus the live encoder
    # read-back so the operator can see focus moving.
    tele_connected = joystick_state.telescope_connected
    focus_rate = int(getattr(joystick_state, 'focus_rate', 0))
    focus_pos = int(getattr(joystick_state, 'current_focus', 0))

    focus_label = display.small_font.render("Focus (L2-/R2+):", True, (255, 255, 255))
    display.menu_screen.blit(focus_label, (base_x, current_y))
    current_y += 20

    bar_w, bar_h = 100, 12
    bar_x, bar_y = base_x, current_y
    pygame.draw.rect(display.menu_screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(display.menu_screen, (150, 150, 150), (bar_x, bar_y, bar_w, bar_h), 1)
    center_x = bar_x + bar_w // 2
    pygame.draw.line(display.menu_screen, (150, 150, 150),
                     (center_x, bar_y), (center_x, bar_y + bar_h), 1)
    pos_color, neg_color, idle_color = (0, 220, 0), (240, 140, 0), (200, 200, 200)
    if focus_rate != 0:
        fill = int(max(-1.0, min(1.0, focus_rate / 9.0)) * (bar_w // 2))
        fill_color = pos_color if focus_rate > 0 else neg_color
        x0 = center_x if fill >= 0 else center_x + fill
        pygame.draw.rect(display.menu_screen, fill_color, (x0, bar_y + 1, abs(fill), bar_h - 1))

    rate_color = pos_color if focus_rate > 0 else neg_color if focus_rate < 0 else idle_color
    rate_text = display.tiny_font.render(f"rate {focus_rate:+d}", True, rate_color)
    display.menu_screen.blit(rate_text, (bar_x + bar_w + 10, bar_y - 2))
    pos_str = f"pos {focus_pos}" if tele_connected else "pos --"
    pos_text = display.tiny_font.render(pos_str, True, (0, 255, 0) if tele_connected else (160, 160, 160))
    display.menu_screen.blit(pos_text, (bar_x + bar_w + 10, bar_y + 9))
    current_y += bar_h + 10

    # Hat information
    num_hats = joy.get_numhats() if connected else 0
    hats_label = display.small_font.render(f"Hats: {num_hats}", True, (255, 255, 255))
    display.menu_screen.blit(hats_label, (base_x, current_y))
    current_y += 20

    # Grey the whole button/axes block out when no joystick is connected
    if not connected:
        region = pygame.Rect(display.sub_x + 5, region_top - 2,
                             2 * col_w + 20, current_y - region_top + 2)
        _draw_disabled_scrim(display, region)

def render_capture_controls(display, joystick_state):
    """Render capture controls and progress indicator below polar graph"""
    try:
        # Update capture progress from capture manager
        from capture_manager import capture_manager
        progress, status = capture_manager.get_dump_progress()

        # Update joystick state
        joystick_state.capture_progress = progress
        joystick_state.capture_status = status if status else ""

        # Determine if cameras are connected and get individual camera status
        camera_connected = False
        camera_buffer_info = {}
        joystick_state.capture_active = False

        for idx in range(len(camera_manager.cameras)):
            camera = camera_manager.get_camera(idx)
            if camera and camera.connected and camera.thread:
                camera_connected = True
                buffer_info = camera.thread.get_capture_buffer_info()

                # Store individual camera info
                camera_buffer_info[idx] = buffer_info

                # Check if any camera is actively capturing
                if buffer_info.get('capture_active', False):
                    joystick_state.capture_active = True

        # Capture toggle button
        button_y = display.sub_y + display.sub_height // 2 - 80  # Between polar graph and camera feeds
        button_width = 90
        button_height = 35
        button_x = display.sub_x + display.sub_width - button_width - 20  # Right side of screen

        joystick_state.capture_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        # Button color based on capture state
        mouse_pos = pygame.mouse.get_pos()
        hover = joystick_state.capture_button_rect.collidepoint(mouse_pos)

        if joystick_state.capture_active:
            button_color = (0, 150, 0) if hover else (0, 100, 0)  # Green when active
            button_text = "Stop Capture"
        else:
            button_color = (150, 150, 150) if hover else (120, 120, 120)  # Grey when inactive
            button_text = "Start Capture"

        pygame.draw.rect(display.menu_screen, button_color, joystick_state.capture_button_rect)
        pygame.draw.rect(display.menu_screen, (200, 200, 200), joystick_state.capture_button_rect, 1)

        # Button text
        button_surface = display.small_font.render(button_text, True, (255, 255, 255))
        text_rect = button_surface.get_rect(center=joystick_state.capture_button_rect.center)
        display.menu_screen.blit(button_surface, text_rect)

        # Buffer fill progress bar (below button)
        progress_y = button_y + button_height + 10
        progress_width = button_width
        progress_height = 15

        # Progress bar background
        pygame.draw.rect(display.menu_screen, (50, 50, 50), (button_x, progress_y, progress_width, progress_height))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (button_x, progress_y, progress_width, progress_height), 1)

        # Display individual camera buffer status
        if camera_connected and camera_buffer_info:
            # Display format: C1: [frames] [percent]% | C2: [frames] [percent]%
            camera_status_lines = []
            camera_fill_ratios = []
            total_capacity = 0

            for cam_idx in sorted(camera_buffer_info.keys()):
                buffer_info = camera_buffer_info[cam_idx]
                cam_num = cam_idx + 1
                max_buffer_size = buffer_info.get('max_buffer_size', 1000)
                total_capacity += max_buffer_size

                if joystick_state.capture_active:
                    # During capture: show frame count and capture progress percentage
                    captured_frames = buffer_info.get('capture_frame_count', 0)
                    capture_progress_ratio = buffer_info.get('capture_progress_ratio', 0.0)
                    camera_fill_ratios.append(capture_progress_ratio)
                    camera_status_lines.append(f"C{cam_num}: {captured_frames} {int(capture_progress_ratio * 100)}%")
                else:
                    # When idle: show buffer capacity with zero percent
                    camera_status_lines.append(f"C{cam_num}: {max_buffer_size} 0%")

            # Draw progress bar only when capturing or dumping
            if camera_fill_ratios:
                # Use average fill ratio for progress bar color and length
                avg_fill_ratio = sum(camera_fill_ratios) / len(camera_fill_ratios)

                if avg_fill_ratio < 0.7:
                    # Green for low fill
                    fill_color = (0, int(255 * avg_fill_ratio / 0.7), 0)
                elif avg_fill_ratio < 0.9:
                    # Yellow to orange for medium fill
                    fill_level = (avg_fill_ratio - 0.7) / 0.2
                    fill_color = (int(255 * fill_level), int(200 * fill_level), 0)
                else:
                    # Red for high fill
                    fill_color = (255, int(100 * (avg_fill_ratio - 0.9) / 0.1), 0)

                # Draw progress fill based on average of both cameras
                fill_width = int(progress_width * avg_fill_ratio)
                if fill_width > 0:
                    pygame.draw.rect(display.menu_screen, fill_color,
                                    (button_x, progress_y, fill_width, progress_height))

            # Show camera status lines (C1: frames %, C2: frames %)
            if camera_status_lines:
                status_text = " | ".join(camera_status_lines)
                fill_text = display.tiny_font.render(status_text, True, (255, 255, 255))

                text_width = fill_text.get_width()
                text_y = progress_y + progress_height + 5
                text_x = button_x + (progress_width - text_width) // 2
                display.menu_screen.blit(fill_text, (text_x, text_y))

        # Status and progress information (to the left of progress bar)
        info_x = button_x - 200
        info_y = progress_y - 5

        # Capture status
        if joystick_state.capture_active:
            status_text = display.small_font.render("REC", True, (255, 0, 0))
            display.menu_screen.blit(status_text, (info_x, info_y))

            # Recording time (would need to track actual time)
            # For now, just show "Recording..." when active
            recording_text = display.tiny_font.render("Recording...", True, (255, 0, 0))
            display.menu_screen.blit(recording_text, (info_x, info_y + 15))
        else:
            status_text = display.small_font.render("Ready", True, (0, 200, 0))
            display.menu_screen.blit(status_text, (info_x, info_y))

        # Dump progress (when dumping)
        if progress > 0:
            if progress < 1.0:
                dump_progress = int(progress * 100)
                dump_text = display.tiny_font.render(f"Dumping: {dump_progress}%", True, (255, 165, 0))
            else:
                dump_text = display.tiny_font.render("Dump Complete!", True, (0, 255, 0))
            display.menu_screen.blit(dump_text, (info_x, info_y + 30))

        # Camera status
        if not camera_connected:
            no_cam_text = display.tiny_font.render("Camera not connected", True, (255, 0, 0))
            display.menu_screen.blit(no_cam_text, (info_x + 20, info_y + 15))

    except Exception as e:
        print(f"Error in render_capture_controls: {e}")
        # Fallback: render simple error message
        error_text = display.tiny_font.render("Capture Error", True, (255, 0, 0))
        display.menu_screen.blit(error_text, (display.sub_x + 10, display.sub_y + display.sub_height // 2 - 60))

# ==============================================================================
# JOYSTICK-LOOP CAMERA CONTROLS (half-height feeds)
# ==============================================================================
# The Joystick Loop view shows the two camera feeds at half height in the bottom
# camera area. The helpers below give those feeds the same camera/view controls
# as the Sensor Calibration view (gain, exposure, gamma, alignment rotation, ROI,
# combined view + opacity, reset/save). A single layout helper is shared by the
# renderer and the event handler so hit-testing never drifts from what is drawn.

# Slider/control geometry constants for the joystick-loop camera area
JL_SLIDER_W = 100          # gain / exposure / gamma slider track width
JL_ROT_SLIDER_W = 120      # alignment-rotation slider track width
JL_GAMMA_MIN = 0.01
JL_GAMMA_MAX = 2.0
JL_ROTATION_RANGE = 90.0   # -90deg .. +90deg


def _joystick_camera_layout(display):
    """Compute every draw/hit rect for the joystick-loop camera controls.

    Returns a dict shared by render_joystick_camera_controls() and
    handle_joystick_camera_control_events() so the two never diverge.
    """
    camera_area_x = display.sub_x + 10
    camera_area_y = display.sub_y + display.sub_height // 2 + 10
    camera_area_width = display.sub_width - 20
    camera_area_height = display.sub_height // 2 - 20

    cam_w = camera_area_width // 2 - 5
    cam_h = camera_area_height

    layout = {
        "camera_area": pygame.Rect(camera_area_x, camera_area_y, camera_area_width, camera_area_height),
        "cameras": [],
    }

    for idx in range(2):
        if idx == 0:
            fx = camera_area_x
        else:
            fx = camera_area_x + camera_area_width // 2 + 5
        fy = camera_area_y

        strip_y = fy + 4
        gain_x = fx + 84
        exp_x = gain_x + JL_SLIDER_W + 30
        gamma_x = exp_x + JL_SLIDER_W + 30

        # ROI size buttons: two columns of three at the top-right of the feed
        roi_rects = []
        roi_col1_x = fx + cam_w - 70
        roi_col2_x = fx + cam_w - 36
        roi_top = fy + 26
        for i in range(6):
            col_x = roi_col1_x if i < 3 else roi_col2_x
            row = i % 3
            roi_rects.append(pygame.Rect(col_x, roi_top + row * 15, 30, 12))

        rot_x = fx + (cam_w - JL_ROT_SLIDER_W) // 2
        rot_y = fy + cam_h - 26

        layout["cameras"].append({
            "fx": fx, "fy": fy, "cam_w": cam_w, "cam_h": cam_h,
            "image_rect": pygame.Rect(fx, fy, cam_w, cam_h),
            # Connect and disconnect share a slot; only the relevant one is shown.
            "connect_rect": pygame.Rect(fx + 4, strip_y, 70, 18),
            "disconnect_rect": pygame.Rect(fx + 4, strip_y, 70, 18),
            "gain_track": pygame.Rect(gain_x, strip_y, JL_SLIDER_W, 18),
            "exposure_track": pygame.Rect(exp_x, strip_y, JL_SLIDER_W, 18),
            "gamma_track": pygame.Rect(gamma_x, strip_y, JL_SLIDER_W, 18),
            "gamma_toggle": pygame.Rect(gamma_x + JL_SLIDER_W + 6, strip_y, 34, 18),
            "roi_rects": roi_rects,
            "rotation_track": pygame.Rect(rot_x, rot_y, JL_ROT_SLIDER_W, 14),
        })

    # Global view controls grouped at the bottom-center, clear of the per-camera
    # ROI buttons that sit at each feed's top-right (the left feed's ROI buttons
    # are right next to the seam, so the seam-top is not free).
    seam_x = camera_area_x + camera_area_width // 2
    bottom = camera_area_y + camera_area_height
    layout["combined_btn"] = pygame.Rect(seam_x - 115, bottom - 48, 110, 18)
    layout["opacity_track"] = pygame.Rect(seam_x + 15, bottom - 44, 100, 14)
    layout["reset_btn"] = pygame.Rect(seam_x - 105, bottom - 24, 100, 20)
    layout["save_btn"] = pygame.Rect(seam_x + 5, bottom - 24, 100, 20)
    return layout


def _process_feed_surface(camera, frame, width, height):
    """Apply gamma + scale + alignment rotation, matching the Sensor Calibration view.

    The processed surface is cached per camera and reused while the source frame
    (identified by its capture sequence) and every processing parameter are
    unchanged. The UI loop runs faster than the capture thread, so without this
    the same frame would be re-gamma'd, re-scaled and re-rotated on every render
    -- wasted GIL-holding work that starves the capture thread. Callers only blit
    (read) the result, so it is safe to hand back the shared cached surface.
    """
    seq = getattr(camera, 'frame_seq', None)
    key = (seq, width, height, camera.gamma_enabled, camera.gamma,
           camera.alignment_rotation)
    cache = getattr(camera, '_feed_cache', None)
    if seq is not None and cache is not None and cache[0] == key:
        return cache[1]

    processed = frame
    if camera.gamma_enabled:
        processed = apply_gamma_correction(frame, camera.gamma)
    surface = pygame.transform.scale(processed, (width, height))
    if camera.alignment_rotation != 0.0:
        rotated = pygame.transform.rotate(surface, camera.alignment_rotation)
        surface = pygame.Surface((width, height))
        surface.fill((0, 0, 0))
        surface.blit(rotated, (width // 2 - rotated.get_width() // 2,
                               height // 2 - rotated.get_height() // 2))
    if seq is not None:
        camera._feed_cache = (key, surface)
    return surface


def _draw_feed_roi(screen, camera, image_rect):
    """Draw the green ROI box + crosshair over a feed, matching the Sensor Calibration view."""
    if camera.roi_size is None or camera.roi_size < 0:
        return
    roi_w_pct, roi_h_pct = roi_sizes[camera.roi_size]
    roi_w = int(roi_w_pct * image_rect.width)
    roi_h = int(roi_h_pct * image_rect.height)
    roi_x = int(camera.roi_x * (image_rect.width - roi_w))
    roi_y = int(camera.roi_y * (image_rect.height - roi_h))
    rect = pygame.Rect(image_rect.x + roi_x, image_rect.y + roi_y, roi_w, roi_h)
    pygame.draw.rect(screen, (0, 255, 0), rect, 2)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(screen, (0, 255, 0), (cx - 10, cy), (cx + 10, cy), 1)
    pygame.draw.line(screen, (0, 255, 0), (cx, cy - 10), (cx, cy + 10), 1)


def _draw_feed_timestamps(display, camera, fx, fy, cam_w, cam_h, align):
    """Draw UTC / Local / FPS stacked in a feed's outer-bottom corner.

    The left feed uses its bottom-left corner and the right feed its bottom-right
    corner, so the timestamps never collide with the centered rotation slider or
    the seam-centered combined/opacity/reset/save controls.
    """
    info_font = pygame.font.Font(None, 16)
    lines = [
        f"UTC: {camera.utc_ts}",
        f"Local: {camera.local_ts}",
        f"FPS: {camera.fps:.1f}",
    ]
    line_h = 15
    bottom = fy + cam_h - 4
    for i, line in enumerate(lines):
        surf = info_font.render(line, True, (255, 255, 255))
        y = bottom - (len(lines) - i) * line_h  # stack upward; last line at the bottom
        x = fx + 8 if align == 'left' else fx + cam_w - 8 - surf.get_width()
        display.menu_screen.blit(surf, (x, y))


def _draw_jl_slider(screen, font, track, ratio, label, track_color, handle_color):
    """Draw a compact horizontal slider (track + handle + label above)."""
    line_y = track.centery
    pygame.draw.rect(screen, track_color, (track.x, line_y - 2, track.width, 4))
    handle_x = track.x + int(max(0.0, min(1.0, ratio)) * track.width)
    pygame.draw.rect(screen, handle_color, (handle_x - 4, line_y - 6, 8, 12))
    if label:
        screen.blit(font.render(label, True, (255, 255, 255)), (track.x, track.y - 12))


def _camera_fov_deg(config_state, camera, cam_key, default_focal):
    """Approximate (fov_w_deg, fov_h_deg) from config optics + camera resolution."""
    cc = config_state.camera_configs.get(cam_key, {}) if config_state else {}
    pixel_size_um = float(cc.get('pixel_size', 2.9))
    focal_length_mm = float(cc.get('focal_length', default_focal))
    if focal_length_mm <= 0:
        return 0.0, 0.0
    w = getattr(camera, 'width_res', 0) or 1920
    h = getattr(camera, 'height_res', 0) or 1080
    fov_w = math.degrees(2 * math.atan((w * pixel_size_um * 1e-3) / (2 * focal_length_mm)))
    fov_h = math.degrees(2 * math.atan((h * pixel_size_um * 1e-3) / (2 * focal_length_mm)))
    return fov_w, fov_h


def _cam_rotation(config_state, camera, cam_key):
    """Live camera alignment rotation (deg), falling back to the config value."""
    rot = getattr(camera, 'alignment_rotation', None)
    if rot is not None:
        return float(rot)
    cc = config_state.camera_configs.get(cam_key, {}) if config_state else {}
    return float(cc.get('alignment_rotation', 0.0))


def _draw_cam2_fov_in_cam1(display, joystick_state, image_rect):
    """Overlay camera2's FOV as a red box centered in camera1's frame, sized by the
    FOV ratio and rotated by cam2's alignment rotation relative to cam1. Lets us
    see how close the narrow (cam2) field is to centered in the finder (cam1)."""
    cfg = joystick_state.config_state
    if cfg is None:
        return
    c1 = camera_manager.get_camera(0)
    c2 = camera_manager.get_camera(1)
    fov1 = _camera_fov_deg(cfg, c1, 'camera1', 162.0)
    fov2 = _camera_fov_deg(cfg, c2, 'camera2', 2000.0)
    if fov1[0] <= 0 or fov1[1] <= 0 or fov2[0] <= 0:
        return
    hw = max(3.0, (fov2[0] / fov1[0]) * image_rect.width / 2.0)
    hh = max(3.0, (fov2[1] / fov1[1]) * image_rect.height / 2.0)
    rel = math.radians(_cam_rotation(cfg, c2, 'camera2') - _cam_rotation(cfg, c1, 'camera1'))
    ct, st = math.cos(rel), math.sin(rel)
    cx, cy = image_rect.center
    pts = [(int(cx + x * ct - y * st), int(cy + x * st + y * ct))
           for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    pygame.draw.polygon(display.menu_screen, (255, 0, 0), pts, 2)
    lbl = display.tiny_font.render("Cam2 FOV", True, (255, 90, 90))
    display.menu_screen.blit(lbl, (cx + hw + 3, cy - hh - 2))


def _draw_feed_axes(display, joystick_state, image_rect, camera, cam_key):
    """Draw HUD axes from the frame center: the +along-track (IT) and +cross-track
    (CT) bias-direction axes, plus a target-position axis (TGT) pointing to where
    the tracked target currently sits in the image (from the live tracking error,
    so its tip lands on the spot). A central GAP keeps the boresight/target itself
    un-obscured -- without it the converging axes hid a centered target. Only
    shown while tracking."""
    cfg = joystick_state.config_state
    if cfg is None or joystick_state.tracking_mode not in (
            TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT):
        return
    surf = display.menu_screen
    cx, cy = image_rect.center
    span = min(image_rect.width, image_rect.height)
    gap = 18  # clear radius around boresight so the target spot stays visible
    x_sign = float(getattr(cfg, 'hotspot_x_sign', 1.0)) or 1.0
    y_sign = float(getattr(cfg, 'hotspot_y_sign', -1.0)) or -1.0
    altaz = getattr(cfg, 'mount_mode', 'AltAz') == 'AltAz'
    el = getattr(joystick_state, 'target_el_deg', 0.0)
    cos_el = max(math.cos(math.radians(el)), 0.087)

    def draw_axis(dx, dy, length, color, label):
        n = math.hypot(dx, dy)
        if n < 1e-9 or length <= gap + 4:
            return
        ux, uy = dx / n, dy / n
        sx, sy = cx + ux * gap, cy + uy * gap          # start past the central gap
        ex, ey = cx + ux * length, cy + uy * length
        pygame.draw.line(surf, color, (sx, sy), (ex, ey), 2)
        ang = math.atan2(ey - cy, ex - cx)
        for da in (2.618, -2.618):  # +/-150 deg arrowhead
            pygame.draw.line(surf, color, (ex, ey),
                             (ex + 8 * math.cos(ang + da), ey + 8 * math.sin(ang + da)), 2)
        surf.blit(display.tiny_font.render(label, True, color), (ex + 3, ey - 6))

    # Bias direction axes (along/cross-track), from the target's sky velocity.
    # Feed is rotation-corrected by alignment_rotation, so only signs apply here.
    vx = getattr(joystick_state, 'target_az_rate', 0.0) * cos_el  # on-sky cross-el
    vy = getattr(joystick_state, 'target_el_rate', 0.0)           # on-sky el
    vn = math.hypot(vx, vy)
    if vn >= 1e-4:
        ux, uy = vx / vn, vy / vn
        bias_len = 0.26 * span
        draw_axis(ux / x_sign, uy / y_sign, bias_len, (255, 230, 80), "+IT")
        draw_axis(-uy / x_sign, ux / y_sign, bias_len, (80, 230, 255), "+CT")

    # Target-position axis: map the live tracking position error to the target's
    # image offset (true pixels, scaled to the displayed feed) so the arrow tip
    # lands on the spot. Shrinks to nothing as you center up.
    az_err = getattr(joystick_state, 'azm_position_error', 0.0)
    alt_err = getattr(joystick_state, 'alt_position_error', 0.0)
    sky_el_err = -alt_err if altaz else alt_err          # ALT axis runs opposite el
    cross_el = az_err * cos_el
    pix = float(cfg.get_camera_pixel_size(cam_key))
    foc = float(cfg.get_camera_focal_length(cam_key))
    ifov = math.degrees(pix * 1e-3 / foc) if foc > 0 else 0.0
    if ifov > 0:
        raw_w = getattr(camera, 'width_res', 0) or 1920
        scale = image_rect.width / float(raw_w)
        tdx = (cross_el / ifov) / x_sign * scale
        tdy = (sky_el_err / ifov) / y_sign * scale
        tlen = math.hypot(tdx, tdy)
        if tlen > gap + 4:
            draw_axis(tdx, tdy, min(tlen, 0.46 * span), (255, 90, 255), "TGT")


def render_camera_feeds(display, joystick_state=None):
    """Render camera feeds with cross-hairs using existing camera manager"""
    try:
        # Update camera frames
        update_camera_frames_from_buffers()

        # Safely get camera status
        camera1_connected = False
        camera2_connected = False
        camera1_frame = None
        camera2_frame = None

        # Safely access cameras with bounds checking and get time/fps info
        if len(camera_manager.cameras) > 0:
            camera1 = camera_manager.cameras[0]
            camera1_connected = camera1.connected
            camera1_frame = camera1.frame

        if len(camera_manager.cameras) > 1:
            camera2 = camera_manager.cameras[1]
            camera2_connected = camera2.connected
            camera2_frame = camera2.frame

        display_width = display.sub_width - 20
        display_height = display.sub_height // 2 - 20  # Bottom half minus some margin

        # Create surface for camera area (bottom half)
        camera_area_x = display.sub_x + 10
        camera_area_y = display.sub_y + display.sub_height // 2 + 10
        camera_area_width = display.sub_width - 20
        camera_area_height = display.sub_height // 2 - 20

        display.menu_screen.fill((0, 0, 0), (camera_area_x, camera_area_y,
                                            camera_area_width, camera_area_height))

        # Check if any cameras are expected to be present
        num_available = camera_manager.get_num_cameras()
        if num_available == 0:
            # No cameras detected at all
            no_cams_text = display.small_font.render("No cameras detected", True, (255, 0, 0))
            text_rect = no_cams_text.get_rect(center=(display.sub_x + display.sub_width // 2,
                                                     display.sub_y + display.sub_height * 3 // 4))
            display.menu_screen.blit(no_cams_text, text_rect)

            if joystick_state:
                # Send status update (we'll use print for now since no callback is passed)
                print("No cameras detected - joystick mode camera features will not function")
            return

        # Combined view: overlay both feeds (opacity-blended) across the full camera
        # area, matching the Sensor Calibration view's combined mode.
        combined_active = (camera_manager.combined_view_toggle and
                           camera1_connected and camera2_connected and
                           camera1_frame is not None and camera2_frame is not None)
        if combined_active:
            try:
                comb_w = camera_area_width
                comb_h = camera_area_height
                surf1 = _process_feed_surface(camera1, camera1_frame, comb_w, comb_h).convert_alpha()
                surf2 = _process_feed_surface(camera2, camera2_frame, comb_w, comb_h).convert_alpha()
                opacity = camera_manager.camera_opacities[0] if camera_manager.camera_opacities else 0.5
                surf1.set_alpha(int(opacity * 255))
                surf2.set_alpha(int((1 - opacity) * 255))
                display.menu_screen.blit(surf1, (camera_area_x, camera_area_y))
                display.menu_screen.blit(surf2, (camera_area_x, camera_area_y))

                center_x = camera_area_x + comb_w // 2
                center_y = camera_area_y + comb_h // 2
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - 20, center_y), (center_x + 20, center_y), 1)
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - 20), (center_x, center_y + 20), 1)
                combined_label = display.small_font.render(f"Combined View - Opacity: {opacity:.1f}", True, (0, 200, 200))
                display.menu_screen.blit(combined_label, (camera_area_x + 10, camera_area_y + 10))
            except Exception as e:
                print(f"Error rendering combined joystick feed: {e}")
                combined_active = False

        # Camera 1 (left half of camera area)
        cam1_width = camera_area_width // 2 - 5
        cam1_height = camera_area_height

        if not combined_active and camera1_connected and camera1_frame is not None:
            try:
                cam1_scaled = _process_feed_surface(camera1, camera1_frame, cam1_width, cam1_height)
                display.menu_screen.blit(cam1_scaled, (camera_area_x, camera_area_y))
                _draw_feed_roi(display.menu_screen, camera1,
                               pygame.Rect(camera_area_x, camera_area_y, cam1_width, cam1_height))

                # Draw cross-hair at center
                center_x = camera_area_x + cam1_width // 2
                center_y = camera_area_y + cam1_height // 2
                crosshair_length = 20
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y),
                                (center_x + crosshair_length, center_y), 1)  # Horizontal
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length),
                                (center_x, center_y + crosshair_length), 1)  # Vertical

                # Camera 1 label
                cam1_text = display.small_font.render("Camera 1", True, (255, 255, 255))
                display.menu_screen.blit(cam1_text, (camera_area_x + 10, camera_area_y + 10))

                # Time/FPS stacked in the left feed's bottom-left corner (clear of the
                # centered rotation slider and the seam-centered view controls)
                _draw_feed_timestamps(display, camera1, camera_area_x, camera_area_y, cam1_width, cam1_height, 'left')

                # Overlays: camera2's FOV box (centering aid) + bias-direction and
                # target-position HUD axes (with a central gap so they don't hide
                # a centered target).
                if joystick_state is not None:
                    cam1_rect = pygame.Rect(camera_area_x, camera_area_y, cam1_width, cam1_height)
                    _draw_cam2_fov_in_cam1(display, joystick_state, cam1_rect)
                    _draw_feed_axes(display, joystick_state, cam1_rect, camera1, 'camera1')
                    draw_solve_centroids_on_feed(display, joystick_state, cam1_rect, 0)
            except Exception as e:
                error_text = display.small_font.render("Camera 1 Error", True, (255, 0, 0))
                display.menu_screen.blit(error_text, (camera_area_x + 10, camera_area_y + 10))
                if joystick_state:
                    print(f"Error rendering Camera 1: {e}")
        elif len(camera_manager.cameras) > 0 and not camera1_connected:
            # Camera available but not connected
            not_connected_text = display.small_font.render("Not Connected", True, (255, 0, 0))
            text_rect = not_connected_text.get_rect(center=(camera_area_x + cam1_width // 2,
                                                           camera_area_y + cam1_height // 2))
            display.menu_screen.blit(not_connected_text, text_rect)

        # Camera 2 (right half of camera area)
        cam2_x = camera_area_x + camera_area_width // 2 + 5
        cam2_width = camera_area_width // 2 - 5
        cam2_height = camera_area_height

        if not combined_active and camera2_connected and camera2_frame is not None:
            try:
                cam2_scaled = _process_feed_surface(camera2, camera2_frame, cam2_width, cam2_height)
                display.menu_screen.blit(cam2_scaled, (cam2_x, camera_area_y))
                _draw_feed_roi(display.menu_screen, camera2,
                               pygame.Rect(cam2_x, camera_area_y, cam2_width, cam2_height))

                # Draw cross-hair at center
                center_x = cam2_x + cam2_width // 2
                center_y = camera_area_y + cam2_height // 2
                crosshair_length = 20
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y),
                                (center_x + crosshair_length, center_y), 1)  # Horizontal
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length),
                                (center_x, center_y + crosshair_length), 1)  # Vertical

                # Camera 2 label
                cam2_text = display.small_font.render("Camera 2", True, (255, 255, 255))
                display.menu_screen.blit(cam2_text, (cam2_x + 10, camera_area_y + 10))

                # Time/FPS stacked in the right feed's bottom-right corner (clear of the
                # centered rotation slider and the seam-centered view controls)
                _draw_feed_timestamps(display, camera2, cam2_x, camera_area_y, cam2_width, cam2_height, 'right')

                # Bias-direction + target-position HUD axes (camera2 only -- the
                # FOV box is a finder/cam1 aid).
                if joystick_state is not None:
                    _draw_feed_axes(display, joystick_state,
                                    pygame.Rect(cam2_x, camera_area_y, cam2_width, cam2_height),
                                    camera2, 'camera2')
            except Exception as e:
                error_text = display.small_font.render("Camera 2 Error", True, (255, 0, 0))
                display.menu_screen.blit(error_text, (cam2_x + 10, camera_area_y + 10))
                if joystick_state:
                    print(f"Error rendering Camera 2: {e}")
        elif len(camera_manager.cameras) > 1 and not camera2_connected:
            # Camera available but not connected
            not_connected_text = display.small_font.render("Not Connected", True, (255, 0, 0))
            text_rect = not_connected_text.get_rect(center=(cam2_x + cam2_width // 2,
                                                           camera_area_y + cam2_height // 2))
            display.menu_screen.blit(not_connected_text, text_rect)
        elif len(camera_manager.cameras) == 1 and num_available >= 2:
            # Second camera slot available but camera not present
            missing_text = display.small_font.render("Camera 2 Missing", True, (255, 165, 0))
            text_rect = missing_text.get_rect(center=(cam2_x + cam2_width // 2,
                                                     camera_area_y + cam2_height // 2))
            display.menu_screen.blit(missing_text, text_rect)

        # Camera and view controls overlaid on the half-height feeds
        render_joystick_camera_controls(display, joystick_state)

    except Exception as e:
        # Catch any unexpected errors and display gracefully
        display.menu_screen.fill((0, 0, 0), (display.sub_x + 10, display.sub_y + display.sub_height // 2 + 10,
                                            display.sub_width - 20, display.sub_height // 2 - 20))

        error_text = display.small_font.render(f"Camera Error: {str(e)[:40]}", True, (255, 0, 0))
        text_rect = error_text.get_rect(center=(display.sub_x + display.sub_width // 2,
                                               display.sub_y + display.sub_height * 3 // 4))
        display.menu_screen.blit(error_text, text_rect)
        print(f"Camera rendering error in joystick mode: {e}")


def render_joystick_camera_controls(display, joystick_state=None):
    """Render the camera/view controls overlaid on the half-height joystick-loop feeds.

    Mirrors the Sensor Calibration view's control set: per-camera connect/disconnect,
    gain, exposure, gamma (slider + toggle) and alignment-rotation sliders, ROI size
    buttons, plus the global combined-view toggle, opacity slider and reset/save config.
    """
    try:
        screen = display.menu_screen
        mouse_pos = pygame.mouse.get_pos()
        font = pygame.font.Font(None, 14)
        layout = _joystick_camera_layout(display)

        camera1 = camera_manager.get_camera(0)
        camera2 = camera_manager.get_camera(1)
        cameras = [camera1, camera2]

        for idx, camera in enumerate(cameras):
            if camera is None:
                continue
            cam = layout["cameras"][idx]

            # Connect / Disconnect button (one shown depending on state)
            if camera.connected:
                rect = cam["disconnect_rect"]
                hovered = rect.collidepoint(mouse_pos)
                color = (255, 100, 100) if hovered else (200, 70, 70)
                pygame.draw.rect(screen, color, rect)
                screen.blit(font.render("Disconnect", True, (255, 255, 255)),
                            font.render("Disconnect", True, (255, 255, 255)).get_rect(center=rect.center))
            else:
                rect = cam["connect_rect"]
                hovered = rect.collidepoint(mouse_pos)
                color = (100, 100, 255) if hovered else (70, 70, 200)
                pygame.draw.rect(screen, color, rect)
                screen.blit(font.render("Connect", True, (255, 255, 255)),
                            font.render("Connect", True, (255, 255, 255)).get_rect(center=rect.center))
                # No further per-camera controls when disconnected
                continue

            # Gain slider
            max_gain = camera.prop.get('MaxGain', 500) if camera.prop else 500
            gain_ratio = min(1.0, camera.gain / max_gain) if max_gain else 0.0
            _draw_jl_slider(screen, font, cam["gain_track"], gain_ratio,
                            f"Gain {camera.gain}", (100, 100, 100), (200, 0, 0))

            # Exposure slider (logarithmic)
            max_exp = camera.prop.get('MaxExposure', 500000) if camera.prop else 500000
            min_exp = 1
            if camera.exposure > 0 and max_exp > min_exp:
                exp_ratio = math.log10(camera.exposure / min_exp) / math.log10(max_exp / min_exp)
                exp_ratio = min(1.0, max(0.0, exp_ratio))
            else:
                exp_ratio = 0.0
            exp_us = camera.exposure
            if exp_us < 1000:
                exp_val = f"{exp_us:g}us"
            elif exp_us < 1000000:
                exp_val = f"{exp_us / 1000.0:.1f}ms"
            else:
                exp_val = f"{exp_us / 1000000.0:.1f}s"
            _draw_jl_slider(screen, font, cam["exposure_track"], exp_ratio,
                            f"Exp {exp_val}", (100, 100, 100), (0, 200, 0))

            # Gamma slider + toggle
            gamma_ratio = (camera.gamma - JL_GAMMA_MIN) / (JL_GAMMA_MAX - JL_GAMMA_MIN)
            _draw_jl_slider(screen, font, cam["gamma_track"], gamma_ratio,
                            f"Gamma {camera.gamma:.2f}", (150, 150, 150), (200, 200, 255))
            toggle = cam["gamma_toggle"]
            toggle_color = (0, 150, 0) if camera.gamma_enabled else (150, 0, 0)
            if toggle.collidepoint(mouse_pos):
                toggle_color = tuple(min(255, c + 50) for c in toggle_color)
            pygame.draw.rect(screen, toggle_color, toggle)
            t_surf = font.render("ON" if camera.gamma_enabled else "OFF", True, (255, 255, 255))
            screen.blit(t_surf, t_surf.get_rect(center=toggle.center))

            # Alignment rotation slider (bottom-center)
            rot = cam["rotation_track"]
            rot_ratio = (camera.alignment_rotation + JL_ROTATION_RANGE) / (2 * JL_ROTATION_RANGE)
            _draw_jl_slider(screen, font, rot, rot_ratio,
                            f"Rot {camera.alignment_rotation:+.1f}", (50, 50, 150), (150, 150, 255))
            # Center marker at 0 degrees
            center_marker_x = rot.x + rot.width // 2
            pygame.draw.line(screen, (255, 255, 255),
                             (center_marker_x, rot.centery - 5), (center_marker_x, rot.centery + 5), 1)

            # ROI size buttons
            for i, roi_rect in enumerate(cam["roi_rects"]):
                is_selected = (camera.roi_size == i)
                is_hovered = roi_rect.collidepoint(mouse_pos)
                if is_selected:
                    pygame.draw.rect(screen, (0, 100, 0), roi_rect)
                    pygame.draw.rect(screen, (0, 255, 0), roi_rect, 1)
                    text_color = (255, 255, 255)
                elif is_hovered:
                    pygame.draw.rect(screen, (100, 100, 100), roi_rect)
                    pygame.draw.rect(screen, (200, 200, 200), roi_rect, 1)
                    text_color = (255, 255, 0)
                else:
                    pygame.draw.rect(screen, (70, 70, 70), roi_rect)
                    pygame.draw.rect(screen, (150, 150, 150), roi_rect, 1)
                    text_color = (255, 255, 255)
                r_surf = font.render(roi_label_texts[i], True, text_color)
                screen.blit(r_surf, r_surf.get_rect(center=roi_rect.center))

        # Global view controls -----------------------------------------------
        # Combined view toggle
        combined_btn = layout["combined_btn"]
        is_toggled = camera_manager.combined_view_toggle
        is_hovered = combined_btn.collidepoint(mouse_pos)
        if is_toggled:
            btn_color = (50, 150, 50) if is_hovered else (0, 100, 0)
        else:
            btn_color = (100, 100, 100) if is_hovered else (70, 70, 70)
        pygame.draw.rect(screen, btn_color, combined_btn)
        pygame.draw.rect(screen, (150, 150, 150), combined_btn, 1)
        c_surf = font.render("Combined View", True, (255, 255, 255))
        screen.blit(c_surf, c_surf.get_rect(center=combined_btn.center))

        # Opacity slider (only when both cameras connected)
        if camera1 and camera2 and camera1.connected and camera2.connected:
            opacity = camera_manager.camera_opacities[0] if camera_manager.camera_opacities else 0.5
            _draw_jl_slider(screen, font, layout["opacity_track"], opacity,
                            f"Opacity {opacity:.1f}", (100, 100, 100), (255, 255, 0))

        # Reset / Save config buttons
        reset_btn = layout["reset_btn"]
        pygame.draw.rect(screen, (100, 70, 70) if reset_btn.collidepoint(mouse_pos) else (70, 50, 50), reset_btn)
        pygame.draw.rect(screen, (150, 150, 150), reset_btn, 1)
        rs_surf = font.render("Reset", True, (255, 255, 255))
        screen.blit(rs_surf, rs_surf.get_rect(center=reset_btn.center))

        save_btn = layout["save_btn"]
        pygame.draw.rect(screen, (70, 100, 70) if save_btn.collidepoint(mouse_pos) else (50, 70, 50), save_btn)
        pygame.draw.rect(screen, (150, 150, 150), save_btn, 1)
        sv_surf = font.render("Save", True, (255, 255, 255))
        screen.blit(sv_surf, sv_surf.get_rect(center=save_btn.center))

    except Exception as e:
        print(f"Error rendering joystick camera controls: {e}")


def _apply_jl_camera_drag(current_pos, display):
    """Apply slider drags for the joystick-loop camera controls (gain/exposure/gamma/rotation/opacity)."""
    layout = _joystick_camera_layout(display)
    for idx in range(2):
        camera = camera_manager.get_camera(idx)
        if camera is None or not camera.connected:
            continue
        cam = layout["cameras"][idx]

        # Gain
        track = cam["gain_track"]
        if track.collidepoint(current_pos):
            max_gain = camera.prop.get('MaxGain', 500) if camera.prop else 500
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            camera_manager.set_camera_gain(idx, int((rel / track.width) * max_gain))

        # Exposure (logarithmic)
        track = cam["exposure_track"]
        if track.collidepoint(current_pos):
            max_exp = camera.prop.get('MaxExposure', 500000) if camera.prop else 500000
            min_exp = 1
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            slider_pos = rel / track.width
            if max_exp > min_exp:
                new_exp = int(min_exp * (10 ** (slider_pos * math.log10(max_exp / min_exp))))
                new_exp = max(min_exp, min(new_exp, max_exp))
            else:
                new_exp = max_exp
            camera_manager.set_camera_exposure(idx, new_exp)

        # Gamma
        track = cam["gamma_track"]
        if track.collidepoint(current_pos):
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            new_gamma = JL_GAMMA_MIN + (rel / track.width) * (JL_GAMMA_MAX - JL_GAMMA_MIN)
            camera.gamma = round(max(JL_GAMMA_MIN, min(JL_GAMMA_MAX, new_gamma)), 2)

        # Alignment rotation
        track = cam["rotation_track"]
        if track.collidepoint(current_pos):
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            new_rot = ((rel / track.width) - 0.5) * (2 * JL_ROTATION_RANGE)
            camera.alignment_rotation = max(-JL_ROTATION_RANGE, min(JL_ROTATION_RANGE, new_rot))

    # Opacity (global, only when both cameras connected)
    c1 = camera_manager.get_camera(0)
    c2 = camera_manager.get_camera(1)
    if c1 and c2 and c1.connected and c2.connected:
        track = layout["opacity_track"]
        if track.collidepoint(current_pos):
            rel = min(max(current_pos[0] - track.x, 0), track.width)
            camera_manager.camera_opacities[0] = max(0.0, min(1.0, rel / track.width))


def _jl_pos_over_control(layout, idx, pos):
    """True if pos hits any control overlaying the feed (so an image click is not an ROI-origin set)."""
    cam = layout["cameras"][idx]
    rects = [cam["connect_rect"], cam["disconnect_rect"], cam["gain_track"],
             cam["exposure_track"], cam["gamma_track"], cam["gamma_toggle"],
             cam["rotation_track"]] + cam["roi_rects"]
    rects += [layout["combined_btn"], layout["opacity_track"], layout["reset_btn"], layout["save_btn"]]
    return any(r.collidepoint(pos) for r in rects)


def _jl_reset_camera_config(config_state, update_status_callback):
    """Reset both cameras' settings to config-file defaults (mirrors Sensor Calibration reset)."""
    if not config_state:
        update_status_callback("Error: Config state not available")
        return
    c0 = camera_manager.get_camera(0)
    c1 = camera_manager.get_camera(1)
    c0.alignment_rotation = float(config_state.get_camera_alignment_rotation("camera1") or 0.0)
    c0.gain = int(float(config_state.get_camera_gain("camera1") or 1))
    c0.exposure = int(float(config_state.get_camera_exposure("camera1") or 10000))
    c0.gamma = float(config_state.camera_configs["camera1"].get("gamma", 0.1))
    c0.gamma_enabled = bool(config_state.camera_configs["camera1"].get("gamma_enabled", False))
    c1.alignment_rotation = float(config_state.get_camera_alignment_rotation("camera2") or 0.0)
    c1.gain = int(float(config_state.get_camera_gain("camera2") or 1))
    c1.exposure = int(float(config_state.get_camera_exposure("camera2") or 10000))
    c1.gamma = float(config_state.camera_configs["camera2"].get("gamma", 0.1))
    c1.gamma_enabled = bool(config_state.camera_configs["camera2"].get("gamma_enabled", False))
    if c0.connected and c0.cap:
        camera_manager.set_camera_gain(0, c0.gain, update_status_callback)
        camera_manager.set_camera_exposure(0, c0.exposure, update_status_callback)
    if c1.connected and c1.cap:
        camera_manager.set_camera_gain(1, c1.gain, update_status_callback)
        camera_manager.set_camera_exposure(1, c1.exposure, update_status_callback)
    update_status_callback("Camera settings reset to config file defaults")


def _jl_save_camera_config(config_state, update_status_callback):
    """Save both cameras' current settings to config.json (mirrors Sensor Calibration save)."""
    if not config_state:
        update_status_callback("Error: Config state not available")
        return
    c0 = camera_manager.get_camera(0)
    c1 = camera_manager.get_camera(1)
    config_state.camera_configs["camera1"]["alignment_rotation"] = c0.alignment_rotation
    config_state.camera_configs["camera1"]["gain"] = c0.gain
    config_state.camera_configs["camera1"]["exposure"] = c0.exposure
    config_state.camera_configs["camera1"]["gamma"] = c0.gamma
    config_state.camera_configs["camera1"]["gamma_enabled"] = c0.gamma_enabled
    config_state.camera_configs["camera2"]["alignment_rotation"] = c1.alignment_rotation
    config_state.camera_configs["camera2"]["gain"] = c1.gain
    config_state.camera_configs["camera2"]["exposure"] = c1.exposure
    config_state.camera_configs["camera2"]["gamma"] = c1.gamma
    config_state.camera_configs["camera2"]["gamma_enabled"] = c1.gamma_enabled
    config_state.save_to_file()
    update_status_callback("Camera settings saved to config.json")


def handle_joystick_camera_control_events(event, display, config_state=None, update_status_callback=None):
    """Handle mouse events for the joystick-loop half-height camera controls.

    Returns True if the event was consumed by a camera/view control."""
    if update_status_callback is None:
        update_status_callback = print

    if event.type == pygame.MOUSEMOTION:
        if event.buttons[0]:
            _apply_jl_camera_drag(event.pos, display)
        return False

    if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
        return False

    pos = event.pos
    layout = _joystick_camera_layout(display)

    for idx in range(2):
        camera = camera_manager.get_camera(idx)
        if camera is None:
            continue
        cam = layout["cameras"][idx]

        # Connect / Disconnect
        if not camera.connected:
            if cam["connect_rect"].collidepoint(pos):
                camera_manager.connect_camera(idx, update_status_callback)
                return True
            continue  # no other per-camera controls while disconnected
        if cam["disconnect_rect"].collidepoint(pos):
            camera_manager.disconnect_camera(idx, update_status_callback)
            return True

        # Gamma toggle
        if cam["gamma_toggle"].collidepoint(pos):
            camera.gamma_enabled = not camera.gamma_enabled
            return True

        # Slider clicks set the value immediately (then drag continues to adjust)
        if (cam["gain_track"].collidepoint(pos) or cam["exposure_track"].collidepoint(pos) or
                cam["gamma_track"].collidepoint(pos) or cam["rotation_track"].collidepoint(pos)):
            _apply_jl_camera_drag(pos, display)
            return True

        # ROI size buttons
        for i, roi_rect in enumerate(cam["roi_rects"]):
            if roi_rect.collidepoint(pos):
                if camera.roi_size == i:
                    camera.roi_size = -1
                    camera.roi_x = 0.5
                    camera.roi_y = 0.5
                    camera_manager.set_camera_roi(idx, update_status_callback, None, None, -1)
                else:
                    was_first = (camera.roi_size == -1)
                    camera.roi_size = i
                    if was_first:
                        camera.roi_x = 0.5
                        camera.roi_y = 0.5
                    camera_manager.set_camera_roi(idx, update_status_callback, camera.roi_x, camera.roi_y, i)
                return True

    # Global view controls
    if layout["combined_btn"].collidepoint(pos):
        camera_manager.combined_view_toggle = not camera_manager.combined_view_toggle
        return True

    c1 = camera_manager.get_camera(0)
    c2 = camera_manager.get_camera(1)
    if c1 and c2 and c1.connected and c2.connected and layout["opacity_track"].collidepoint(pos):
        track = layout["opacity_track"]
        rel = min(max(pos[0] - track.x, 0), track.width)
        camera_manager.camera_opacities[0] = max(0.0, min(1.0, rel / track.width))
        return True

    if layout["reset_btn"].collidepoint(pos):
        _jl_reset_camera_config(config_state, update_status_callback)
        return True
    if layout["save_btn"].collidepoint(pos):
        _jl_save_camera_config(config_state, update_status_callback)
        return True

    # ROI origin selection by clicking on a connected feed (skip if over any control)
    for idx in range(2):
        camera = camera_manager.get_camera(idx)
        if camera is None or not camera.connected or camera.frame is None:
            continue
        cam = layout["cameras"][idx]
        image_rect = cam["image_rect"]
        if not image_rect.collidepoint(pos) or _jl_pos_over_control(layout, idx, pos):
            continue
        camera.roi_x = max(0.0, min(1.0, (pos[0] - image_rect.x) / image_rect.width))
        camera.roi_y = max(0.0, min(1.0, (pos[1] - image_rect.y) / image_rect.height))
        if camera.roi_size >= 0:
            camera_manager.set_camera_roi(idx, update_status_callback, camera.roi_x, camera.roi_y, camera.roi_size)
        return True

    return False

# ==============================================================================
# JOYSTICK MODE EVENT HANDLING
# ==============================================================================

def handle_joystick_mode_mouse_events(event, joystick_state, display, tracking_vis_state, config_state, current_tracking_surface):
    """Handle mouse events specific to joystick mode"""
    mouse_pos = pygame.mouse.get_pos()
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = event.pos

        # Connect button
        connect_rect = pygame.Rect(display.sub_x + 10, display.sub_y + 10, 80, 30)
        if connect_rect.collidepoint(pos) and not joystick_state.telescope_connected:
            success = joystick_state.connect_telescope()
            if success:
                print("Telescope connected")

        # Disconnect button
        disconnect_rect = pygame.Rect(display.sub_x + 100, display.sub_y + 10, 80, 30)
        if disconnect_rect.collidepoint(pos) and joystick_state.telescope_connected:
            joystick_state.disconnect_telescope()
            print("Telescope disconnected")

        # ADS-B connect/disconnect buttons (rects stored by the renderer).
        adsb_rects = getattr(joystick_state, 'adsb_button_rects', {}) or {}
        adsb_connect = adsb_rects.get('connect')
        adsb_disconnect = adsb_rects.get('disconnect')
        if adsb_connect and adsb_connect.collidepoint(pos) and not joystick_state.adsb_connected:
            if joystick_state.connect_adsb():
                print("ADS-B receiver connected")
            else:
                print(f"ADS-B connect failed: {joystick_state.adsb_status}")
        if adsb_disconnect and adsb_disconnect.collidepoint(pos) and joystick_state.adsb_connected:
            joystick_state.disconnect_adsb()
            print("ADS-B receiver disconnected")

        # Port dropdown (simplified - click to cycle through ports)
        dropdown_rect = pygame.Rect(display.sub_x + 50, display.sub_y + 50, 120, 25)
        if dropdown_rect.collidepoint(pos):
            joystick_state.get_available_serial_ports()
            if joystick_state.available_ports:
                # Force refresh and cycle to next port
                current_index = 0
                if joystick_state.selected_port:
                    for i, port in enumerate(joystick_state.available_ports):
                        if port['device'] == joystick_state.selected_port:
                            current_index = i
                            break

                next_index = (current_index + 1) % len(joystick_state.available_ports)
                joystick_state.selected_port = joystick_state.available_ports[next_index]['device']
                print(f"Selected port: {joystick_state.selected_port}")

        # Skyplot "Targets" overlay (toggle strip + filters/passes/launches panel).
        # Checked before skyplot selection so panel clicks take precedence over
        # selecting objects behind the panel.
        if handle_joystick_target_panel_click(joystick_state, tracking_vis_state, config_state, pos):
            return True

        # Handle satellite selection/hover in polar plot area
        quadrant_x = display.joystick_layout_params()['divider_x']
        quadrant_y = display.sub_y
        quadrant_width = display.sub_x + display.sub_width - quadrant_x
        quadrant_height = display.sub_height // 2

        quadrant_rect = pygame.Rect(quadrant_x, quadrant_y, quadrant_width, quadrant_height)
        if quadrant_rect.collidepoint(pos):
            # Mouse is over polar plot quadrant - check for satellite hover/selection
            hovered_sat = None

            # Debug: print mouse position on click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                print(f"Click at ({pos[0]}, {pos[1]}) in polar plot quadrant")

            # Pre-calculate centers and scale factor to match
            # draw_satellites_on_surface exactly: positions are stored in
            # full-subarea coordinates, scaled by 0.45 about the full-subarea
            # center, then placed about the quadrant surface's center (which
            # sits at the blit position + half the quadrant size on screen).
            full_screen_center_x = display.sub_x + display.sub_width // 2
            full_screen_center_y = display.sub_y + display.sub_height // 2
            quadrant_center_x = quadrant_x + quadrant_width // 2
            quadrant_center_y = quadrant_y + quadrant_height // 2
            scale_factor = 0.45

            for sat, (px, py, alt, _) in tracking_vis_state.satellite_positions.items():
                # Exact coordinate transformation matching draw_satellites
                rel_x = px - full_screen_center_x
                rel_y = py - full_screen_center_y
                trans_x = quadrant_center_x + rel_x * scale_factor
                trans_y = quadrant_center_y + rel_y * scale_factor

                if alt > 0:  # Only consider satellites above horizon
                    # Debug satellite positions on click
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        print(f"Satellite {sat.name}: orig({px},{py}) -> trans({trans_x},{trans_y})")

                    # Check if mouse is over satellite (larger hit area for easier clicking)
                    dist_to_sat = math.sqrt((pos[0] - trans_x)**2 + (pos[1] - trans_y)**2)
                    if dist_to_sat <= 15:  # 15 pixel radius for easier clicking/hovering
                        hovered_sat = sat
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            print(f"  -> Hit! Distance: {dist_to_sat}")
                        break

            # Same hit-test for ADS-B aircraft (positions stored as full-screen
            # coords like satellites). Aircraft take selection priority over
            # satellites since their markers are sparser/explicitly chosen.
            hovered_ac = None
            for icao, acpos in list(tracking_vis_state.aircraft_positions.items()):
                try:
                    apx, apy, ael, aaz, arng = acpos
                except (TypeError, ValueError):
                    continue
                if ael is None or ael <= 0:
                    continue
                rel_x = apx - full_screen_center_x
                rel_y = apy - full_screen_center_y
                trans_x = quadrant_center_x + rel_x * scale_factor
                trans_y = quadrant_center_y + rel_y * scale_factor
                if math.hypot(pos[0] - trans_x, pos[1] - trans_y) <= 15:
                    hovered_ac = icao
                    break

            # Update hover state on motion
            if event.type == pygame.MOUSEMOTION:
                tracking_vis_state.hovered_satellite = hovered_sat
                tracking_vis_state.hovered_aircraft = hovered_ac

            # Handle satellite/aircraft selection on click. While a launch is
            # active it is the sole target, so plot clicks must not select/track
            # anything (this also covers a launch-button press over the plot).
            launch_active = bool(getattr(tracking_vis_state, 'selected_launch', None)
                                 and getattr(tracking_vis_state, 'launch_launched', False))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not launch_active:
                if hovered_ac is not None:
                    if tracking_vis_state.selected_aircraft == hovered_ac:
                        tracking_vis_state.selected_aircraft = None  # toggle off
                        print("Deselected aircraft")
                    else:
                        tracking_vis_state.selected_aircraft = hovered_ac
                        tracking_vis_state.selected_satellite = None  # mutual exclusivity
                        print(f"Selected aircraft: {hovered_ac}")
                elif hovered_sat is not None:
                    if tracking_vis_state.selected_satellite == hovered_sat:
                        tracking_vis_state.selected_satellite = None  # Deselect if clicking same
                        print("Deselected satellite")
                    else:
                        tracking_vis_state.selected_satellite = hovered_sat  # Select new satellite
                        tracking_vis_state.selected_aircraft = None  # mutual exclusivity
                        print(f"Selected satellite: {hovered_sat.name}")
                else:
                    print("  -> Clicked empty area")
                    # Click in empty area - deselect current selection
                    tracking_vis_state.selected_satellite = None
                    tracking_vis_state.selected_aircraft = None
                    print("Deselected target (empty area clicked)")
        else:
            # Mouse not over polar plot area - clear hover state
            if event.type == pygame.MOUSEMOTION:
                tracking_vis_state.hovered_satellite = None

        # Capture button (mouse click)
        if (hasattr(joystick_state, 'capture_button_rect') and
            joystick_state.capture_button_rect and
            joystick_state.capture_button_rect.collidepoint(pos)):
            _handle_capture_toggle(joystick_state, tracking_vis_state, config_state, current_tracking_surface)

        # Handle bias control button clicks
        if handle_bias_control_mouse_events(joystick_state, mouse_pos):
            return True

        # Handle feed-forward toggle button clicks
        if handle_ff_toggle_mouse_events(joystick_state, mouse_pos):
            return True

        # Handle the star-filter toggle (PID Diagnostics pane)
        if handle_star_filter_mouse_events(joystick_state, mouse_pos):
            return True

        # Handle plate-solve pane clicks (toggle / apply alignment)
        if handle_plate_solve_mouse_events(joystick_state, mouse_pos, config_state):
            return True

        # Handle joystick launch button clicks
        if (hasattr(display, 'joystick_launch_button') and
            display.joystick_launch_button and
            display.joystick_launch_button.collidepoint(pos)):
            # Handle launch button click - same logic as main tracking mode
            if hasattr(tracking_vis_state, 'selected_launch') and tracking_vis_state.selected_launch:
                if tracking_vis_state.launch_launched:
                    # Turn off launch visualization and reset to start
                    tracking_vis_state.launch_launched = False
                    tracking_vis_state.launch_start_time = None
                    print("Launch visualization stopped")
                else:
                    # Start launch visualization from current time
                    from skyfield.api import load
                    tracking_vis_state.launch_launched = True
                    tracking_vis_state.launch_start_time = load.timescale().now().tt
                    print(f"Launch visualization started for {tracking_vis_state.selected_launch}")
                return True

        # Handle PID slider clicks
        if handle_pid_sliders_mouse_events(joystick_state, display, mouse_pos):
            return True

        # Handle Lead-time slider clicks
        if handle_lead_slider_mouse_events(joystick_state, display, mouse_pos):
            return True

        # Handle ADS-B fit-points slider clicks
        if handle_adsb_fit_slider_mouse_events(joystick_state, display, mouse_pos):
            return True

def _handle_capture_toggle(joystick_state, tracking_vis_state, config_state, tracking_surface=None):
    """Handle capture toggle from UI or joystick"""
    from camera_manager import camera_manager
    from capture_manager import capture_manager

    if joystick_state.capture_active:
        # Stop capture and begin dump process for all cameras
        capture_manager.stop_capture(None, tracking_vis_state, tracking_vis_state.selected_satellite, config_state, tracking_surface)
        print("Capture stopped on all cameras, dump process started")
        joystick_state.capture_active = False
    else:
        # Start capture on all connected cameras
        # Check if any camera is available and connected
        any_camera_available = False
        for idx in range(len(camera_manager.cameras)):
            camera = camera_manager.get_camera(idx)
            if camera and camera.connected:
                any_camera_available = True
                break

        if any_camera_available:
            capture_manager.start_capture()  # Start capture on all cameras
            joystick_state.capture_active = True
            print("Capture started on all connected cameras")
        else:
            print("No cameras available for capture")

def _plate_solve_worker(joystick_state):
    """Background loop: solve the latest camera frame and derive the instantaneous
    alignment. Pairs each solve with the mount-reported encoder position and the frame
    time so the az/el conversion and alignment are self-consistent."""
    import time as _t
    from skyfield.api import load as _sf_load
    from camera_manager import camera_manager
    import plate_solver as ps_mod

    cfg = joystick_state.config_state
    tvs = joystick_state.tracking_vis_state
    ts = joystick_state.ts or _sf_load.timescale()
    cam_index = int(getattr(cfg, 'plate_solve_camera_index', 0))
    cam_name = f"camera{cam_index + 1}"
    eph = getattr(tvs, 'ephemeris', None)

    try:
        solver = ps_mod.PlateSolver(cfg, cam_name)
        joystick_state.plate_solve_status = f"loading DB {solver.db_name}..."
        solver.ensure_loaded()
        joystick_state.plate_solver = solver
    except Exception as e:
        joystick_state.plate_solve_status = f"solver init failed: {e}"
        joystick_state.plate_solve_running = False
        return

    while joystick_state.plate_solve_running:
        try:
            camera = camera_manager.get_camera(cam_index)
            raw = (camera.thread.get_latest_raw()
                   if camera is not None and getattr(camera, 'thread', None) is not None else None)
            if raw is None:
                joystick_state.plate_solve_status = "no camera frame"
                _t.sleep(0.5)
                continue

            # Snapshot the encoder position and time paired with this frame.
            mount_azm = joystick_state.current_azm_raw
            mount_alt = joystick_state.current_alt_raw
            t = ts.now()

            result = solver.solve(raw)
            if result is None or not result.solved:
                joystick_state.plate_solve_status = "no solution"
                joystick_state.last_solve = None
                if tvs is not None:
                    tvs.last_solve = None
                _t.sleep(1.0)
                continue

            az = el = align_az = None
            if eph is not None:
                lat = float(cfg.lat_str or 0.0)
                lon = float(cfg.lon_str or 0.0)
                elev = float(cfg.alt_str or 0.0)
                az, el = ps_mod.solved_azel(result.ra_deg, result.dec_deg, lat, lon, elev, eph, ts, t)
                align_az = ps_mod.instantaneous_alignment_azimuth(az, mount_azm)

            solve_info = {
                'result': result, 'az': az, 'el': el, 'align_az': align_az,
                'mount_azm': mount_azm, 'mount_alt': mount_alt, 't': t,
                'src_shape': tuple(raw.shape[:2]), 'cam_index': cam_index,
            }
            joystick_state.last_solve = solve_info
            if tvs is not None:
                tvs.last_solve = solve_info  # consumed by the skyplot overlay
            joystick_state.plate_solve_status = f"OK {result.n_matches}m FOV {result.fov_deg:.2f}"
        except Exception as e:
            joystick_state.plate_solve_status = f"err: {e}"
        _t.sleep(1.0)


def toggle_plate_solve(joystick_state):
    """Start/stop the background plate-solve worker."""
    import plate_solver as ps_mod
    if joystick_state.plate_solve_running:
        joystick_state.plate_solve_running = False
        joystick_state.plate_solve_status = "stopped"
        if joystick_state.config_state is not None:
            joystick_state.config_state.plate_solve_enabled = False
        return
    if not ps_mod.tetra3_available():
        joystick_state.plate_solve_status = "tetra3 not installed"
        return
    joystick_state.plate_solve_running = True
    if joystick_state.config_state is not None:
        joystick_state.config_state.plate_solve_enabled = True
    th = threading.Thread(target=_plate_solve_worker, args=(joystick_state,), daemon=True)
    joystick_state.plate_solve_thread = th
    th.start()


def apply_instantaneous_alignment(joystick_state, config_state):
    """Write the most recent solved alignment azimuth into config (AltAz mode)."""
    ls = joystick_state.last_solve
    if ls is None or ls.get('align_az') is None or config_state is None:
        joystick_state.plate_solve_status = "no solution to apply"
        return
    align_az = ls['align_az']
    config_state.alignment_azimuth_str = f"{align_az:.4f}"
    try:
        config_state.save_to_file()
    except Exception as e:
        print(f"Apply-alignment save failed: {e}")
    tvs = joystick_state.tracking_vis_state
    if tvs is not None:
        tvs.alignment_azimuth = align_az
    joystick_state.plate_solve_status = f"applied align az {align_az:+.3f}"


def draw_solve_centroids_on_feed(display, joystick_state, cam_rect, cam_index):
    """Overlay the plate solver's matched star centroids on a camera feed.

    matched_centroids are in solved-frame pixel coords (row, col); scale them into the
    on-screen feed rect. Drawn only on the camera the solver is reading.
    """
    ls = getattr(joystick_state, 'last_solve', None)
    if ls is None or ls.get('cam_index') != cam_index:
        return
    result = ls.get('result')
    src = ls.get('src_shape')
    if result is None or src is None:
        return
    cents = getattr(result, 'matched_centroids', None)
    if cents is None or len(cents) == 0:
        return
    src_h, src_w = src[0], src[1]
    if src_h <= 0 or src_w <= 0:
        return
    sx_scale = cam_rect.width / float(src_w)
    sy_scale = cam_rect.height / float(src_h)
    for c in cents:
        row, col = float(c[0]), float(c[1])
        px = int(cam_rect.x + col * sx_scale)
        py = int(cam_rect.y + row * sy_scale)
        pygame.draw.circle(display.menu_screen, (0, 230, 230), (px, py), 5, 1)


def render_plate_solve_panel(display, joystick_state):
    """Plate-solve pane: on/off toggle, Apply-Alignment button, and solve status."""
    pane = joystick_panel_layout(display)['plate']
    pygame.draw.rect(display.menu_screen, (35, 40, 50), pane)
    pygame.draw.rect(display.menu_screen, (120, 140, 180), pane, 2)
    display.menu_screen.blit(display.small_font.render("Plate Solve", True, (200, 210, 235)),
                             (pane.x + 8, pane.y + 4))

    mouse_pos = pygame.mouse.get_pos()
    btn_w, btn_h = 78, 22
    toggle_rect = pygame.Rect(pane.x + 8, pane.y + 26, btn_w, btn_h)
    on = joystick_state.plate_solve_running
    tcol = (0, 150, 0) if on else (100, 100, 100)
    if toggle_rect.collidepoint(mouse_pos):
        tcol = tuple(min(255, c + 40) for c in tcol)
    pygame.draw.rect(display.menu_screen, tcol, toggle_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), toggle_rect, 1)
    tlabel = display.tiny_font.render("Solve ON" if on else "Solve OFF", True, (255, 255, 255))
    display.menu_screen.blit(tlabel, tlabel.get_rect(center=toggle_rect.center))

    have = joystick_state.last_solve is not None and joystick_state.last_solve.get('align_az') is not None
    apply_rect = pygame.Rect(pane.x + 8 + btn_w + 8, pane.y + 26, 110, btn_h)
    acol = (60, 90, 140) if have else (70, 70, 70)
    if have and apply_rect.collidepoint(mouse_pos):
        acol = tuple(min(255, c + 40) for c in acol)
    pygame.draw.rect(display.menu_screen, acol, apply_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), apply_rect, 1)
    alabel = display.tiny_font.render("Apply Align", True, (255, 255, 255))
    display.menu_screen.blit(alabel, alabel.get_rect(center=apply_rect.center))

    joystick_state.ps_button_rects = [('ps_toggle', toggle_rect), ('ps_apply', apply_rect)]

    ls = joystick_state.last_solve
    y = pane.y + 50
    if ls is not None and ls.get('az') is not None:
        r = ls['result']
        rmse = f"{r.rmse:.1f}\"" if r.rmse == r.rmse else "--"  # NaN-safe
        display.menu_screen.blit(
            display.tiny_font.render(f"SOLVED  {r.n_matches} stars  RMSE {rmse}", True, (110, 220, 140)),
            (pane.x + 8, y)); y += 12
        lines = [f"sky az {ls['az']:.2f}  el {ls['el']:.2f}",
                 f"align az -> {ls['align_az']:+.3f}°"]
        col = (200, 200, 200)
    else:
        st = joystick_state.plate_solve_status or "idle"
        col = (220, 205, 120) if joystick_state.plate_solve_running else (150, 150, 150)
        display.menu_screen.blit(display.tiny_font.render(
            ("searching... " + st) if joystick_state.plate_solve_running else st, True, col),
            (pane.x + 8, y)); y += 12
        lines = []
    for ln in lines:
        display.menu_screen.blit(display.tiny_font.render(ln, True, col), (pane.x + 8, y))
        y += 12


def handle_plate_solve_mouse_events(joystick_state, mouse_pos, config_state):
    """Route clicks on the plate-solve pane's toggle / apply buttons."""
    if not getattr(joystick_state, 'ps_button_rects', None):
        return False
    for name, rect in joystick_state.ps_button_rects:
        if rect.collidepoint(mouse_pos):
            if name == 'ps_toggle':
                toggle_plate_solve(joystick_state)
            elif name == 'ps_apply':
                apply_instantaneous_alignment(joystick_state, config_state)
            return True
    return False


def render_pid_diagnostics(display, joystick_state):
    """
    Live PID tracking diagnostics for both axes: position error and commanded
    rate, plus the active rate-command mode. Always drawn so its location is
    visible (greyed out when not actively tracking), matching the PID Gain pane.
    Reads only fields both the Python and Rust loops populate (position error and
    pid output), so it works regardless of which control loop is running.
    """
    pane = joystick_panel_layout(display)['diag']
    x_start, y_start, width, height = pane.x, pane.y, pane.width, pane.height

    # "Needed" = actively tracking (PROGRAM / HANDOFF / HOTSPOT); otherwise greyed.
    active = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    # Palette: dim when idle, lit when active.
    bg = (70, 70, 90) if active else (45, 45, 55)
    border = (140, 140, 170) if active else (80, 80, 95)
    label_c = (255, 200, 0) if active else (110, 100, 70)
    val_c = (255, 255, 255) if active else (110, 110, 120)
    rate_c = (160, 255, 160) if active else (90, 110, 90)

    pygame.draw.rect(display.menu_screen, bg, (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, border, (x_start, y_start, width, height), 1)
    display.menu_screen.blit(
        display.small_font.render("PID Diagnostics", True, val_c), (x_start + 10, y_start + 5))

    # Star-filter toggle (top-right): the HANDOFF/HOTSPOT detector rejects
    # detections whose angular rate doesn't match the target (i.e. stars).
    # Toggle OFF to deliberately track a star. Persisted to config.
    cfg_sf = getattr(joystick_state, 'config_state', None)
    sf_on = bool(getattr(cfg_sf, 'hotspot_star_filter_enabled', True))
    sf_rect = pygame.Rect(x_start + width - 78, y_start + 4, 70, 16)
    sf_color = ((0, 130, 0) if sf_on else (110, 90, 40)) if active else (60, 60, 70)
    if active and sf_rect.collidepoint(pygame.mouse.get_pos()):
        sf_color = tuple(min(255, c + 40) for c in sf_color)
    pygame.draw.rect(display.menu_screen, sf_color, sf_rect)
    pygame.draw.rect(display.menu_screen, (170, 170, 190), sf_rect, 1)
    sf_text = display.tiny_font.render(
        "no stars" if sf_on else "stars OK", True, (255, 255, 255))
    display.menu_screen.blit(sf_text, sf_text.get_rect(center=sf_rect.center))
    joystick_state.star_filter_button_rect = sf_rect

    az_err = getattr(joystick_state, 'azm_position_error', 0.0)
    el_err = getattr(joystick_state, 'alt_position_error', 0.0)
    # pid_output is in rev/sec (the command scale); show it as deg/sec.
    az_rate = getattr(joystick_state, 'azm_pid_output', 0.0) * 360.0
    el_rate = getattr(joystick_state, 'alt_pid_output', 0.0) * 360.0

    cx = [x_start + 10, x_start + 130]   # AZ column, EL column
    display.menu_screen.blit(display.tiny_font.render("AZIMUTH", True, label_c), (cx[0], y_start + 26))
    display.menu_screen.blit(display.tiny_font.render("ELEVATION", True, label_c), (cx[1], y_start + 26))
    for col, err, rate in ((0, az_err, az_rate), (1, el_err, el_rate)):
        display.menu_screen.blit(
            display.tiny_font.render(f"err {err:+.2f}°", True, val_c), (cx[col], y_start + 42))
        display.menu_screen.blit(
            display.tiny_font.render(f"rate {rate:+.2f}°/s", True, rate_c), (cx[col], y_start + 58))

    # Active rate-command primitive (the control-theory lesson at a glance).
    cfg = getattr(joystick_state, 'config_state', None)
    continuous = bool(getattr(cfg, 'continuous_rate_tracking', False)) if cfg else False
    if joystick_state.tracking_mode == TrackingMode.HANDOFF:
        # In HANDOFF this line reports the parallel-detection progress instead.
        hs = getattr(joystick_state, 'handoff_status', '') or 'armed'
        mode_label = f"HANDOFF: {hs}"
        mode_c = (255, 200, 120)
    else:
        mode_label = "rate cmd: CONTINUOUS (guide-rate)" if continuous else "rate cmd: DISCRETE (MC_MOVE)"
        mode_c = (120, 200, 255) if (active and continuous) else (val_c if active else (90, 90, 100))
    display.menu_screen.blit(display.tiny_font.render(mode_label, True, mode_c), (x_start + 10, y_start + 80))
    # Bias line: show along/cross if either is set, else Az/El.
    it = getattr(joystick_state, 'bias_intrack_deg', 0.0)
    xt = getattr(joystick_state, 'bias_crosstrack_deg', 0.0)
    if it or xt:
        bias_text = f"bias InTk {it:+.1f}° XTk {xt:+.1f}°"
    else:
        bias_text = f"bias AZ {getattr(joystick_state,'bias_azm_deg',0.0):+.1f}° EL {getattr(joystick_state,'bias_alt_deg',0.0):+.1f}°"
    bias_c = val_c if active else (90, 90, 100)
    display.menu_screen.blit(display.tiny_font.render(bias_text, True, bias_c), (x_start + 10, y_start + 94))

def _draw_strip_chart(display, rect, title, series, active, scale=None):
    """Draw a single zero-centered strip chart. `series` is a list of
    (label, values, color); all series share one symmetric vertical scale.
    `scale` sets the ±vertical fullscale (auto-ranged by the caller); if None it
    falls back to the peak of the supplied data. Greyed when not tracking."""
    surf = display.menu_screen
    bg = (25, 25, 35) if active else (30, 30, 36)
    pygame.draw.rect(surf, bg, rect)
    pygame.draw.rect(surf, (90, 90, 110) if active else (60, 60, 70), rect, 1)

    title_c = (210, 210, 220) if active else (120, 120, 130)
    surf.blit(display.tiny_font.render(title, True, title_c), (rect.x + 4, rect.y + 2))

    # Legend at the top-right.
    lx = rect.right - 6
    for label, _vals, color in reversed(series):
        t = display.tiny_font.render(label, True, color if active else (110, 110, 120))
        lx -= t.get_width() + 8
        surf.blit(t, (lx, rect.y + 2))

    plot = pygame.Rect(rect.x + 4, rect.y + 15, rect.width - 8, rect.height - 28)
    zero_y = plot.centery
    pygame.draw.line(surf, (70, 70, 80), (plot.x, zero_y), (plot.right, zero_y), 1)

    if scale is not None:
        amax = max(scale, 1e-6)
    else:
        allvals = [v for _l, vals, _c in series for v in vals]
        amax = max(max((abs(v) for v in allvals), default=1.0), 1e-6)

    for _label, vals, color in series:
        n = len(vals)
        if n < 2:
            continue
        pts = []
        for i, v in enumerate(vals):
            x = plot.x + int(i / (n - 1) * plot.width)
            y = zero_y - int(v / amax * (plot.height / 2 - 1))
            y = max(plot.y, min(plot.bottom, y))
            pts.append((x, y))
        pygame.draw.lines(surf, color if active else (90, 90, 100), False, pts, 1)

    scale_c = (130, 130, 140) if active else (90, 90, 100)
    surf.blit(display.tiny_font.render(f"±{amax:.3g}", True, scale_c),
              (plot.x, rect.bottom - 12))


def render_tracking_strip_charts(display, joystick_state):
    """Render the az/el tracking-rate and position-error strip charts in the
    center column of the upper-left quadrant, for live PID tuning. Always drawn
    (greyed when not tracking) so their location is stable."""
    layout = joystick_center_layout(display)
    if not layout['valid']:
        return

    # Sample fresh data each frame (loop-agnostic: Python thread or Rust loop).
    joystick_state.sample_tracking_history()

    active = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    az_c, el_c = (120, 255, 120), (120, 200, 255)
    az_rate, el_rate = list(joystick_state.az_rate_history), list(joystick_state.el_rate_history)
    az_err, el_err = list(joystick_state.az_err_history), list(joystick_state.el_err_history)
    # Auto-range each chart from a recent window so the axis shrinks promptly when
    # the signal collapses (floors keep it from over-zooming on noise).
    rate_scale = joystick_state.chart_axis_scale('rate', az_rate + el_rate, floor=0.05)
    err_scale = joystick_state.chart_axis_scale('err', az_err + el_err, floor=0.02)
    _draw_strip_chart(display, layout['chart_rate'], "Tracking Rate °/s", [
        ("AZ", az_rate, az_c),
        ("EL", el_rate, el_c),
    ], active, scale=rate_scale)
    _draw_strip_chart(display, layout['chart_err'], "Pos Error °", [
        ("AZ", az_err, az_c),
        ("EL", el_err, el_c),
    ], active, scale=err_scale)


_NAVBALL_GRID_CACHE = {}

# One-deep cache of the rendered navball "base" (hemisphere fill + grid + static
# labels), keyed on (R, quantized az, quantized el). The base is the expensive
# part -- a full per-pixel hemisphere fill plus ~1700 pure-Python trig
# projections for the grid -- and it only changes when the pointing or size
# changes. Reusing it whenever the mount is stationary keeps the main render
# thread from saturating the GIL and starving the camera capture thread.
_NAVBALL_BASE_CACHE = {}


def _navball_grid(R):
    """Per-pixel orthographic backprojection grid for a navball of radius R,
    cached because R only changes on window resize. For each pixel inside the
    disc, (NX, NY) are its camera-plane coords in [-1, 1] and NZ is the implied
    front-hemisphere depth (sqrt(1 - NX^2 - NY^2)); MASK marks in-disc pixels.
    The hemisphere fill is then a single vectorized sign test per frame."""
    g = _NAVBALL_GRID_CACHE.get(R)
    if g is None:
        size = 2 * R
        ix = np.arange(size, dtype=np.float32)
        PX, PY = np.meshgrid(ix, ix, indexing='ij')   # [screen_x, screen_y]
        NX = (PX - R + 0.5) / R
        NY = (R - PY - 0.5) / R                        # +Y is screen-up
        RR = NX * NX + NY * NY
        MASK = RR <= 1.0
        NZ = np.sqrt(np.clip(1.0 - RR, 0.0, 1.0))
        g = (NX, NY, NZ, MASK)
        _NAVBALL_GRID_CACHE[R] = g
    return g


def active_program_trajectory(tvs):
    """Resolve the trajectory the PROGRAM loop should track: a selected satellite
    first, then a selected aircraft (ADS-B). Launch tracking is handled separately
    by the launch override, so it is not considered here. Returns
    (traj_tuple, kind, key) or (None, None, None); traj_tuple is the
    (rows, times_array) pair in the canonical 8-column format. Shared by the
    Python program_track loop and the Rust adapter so both agree on the target."""
    if tvs is None:
        return None, None, None
    sat = getattr(tvs, 'selected_satellite', None)
    sat_trajs = getattr(tvs, 'satellite_trajectories', None) or {}
    if sat is not None and sat_trajs.get(sat):
        return sat_trajs[sat], 'satellite', sat
    icao = getattr(tvs, 'selected_aircraft', None)
    ac_trajs = getattr(tvs, 'aircraft_trajectories', None) or {}
    if icao is not None and ac_trajs.get(icao):
        return ac_trajs[icao], 'aircraft', icao
    return None, None, None


def _navball_active_target(tvs):
    """Resolve the target the navball should overlay, mirroring how PROGRAM
    tracking picks one: a selected satellite first, then a selected aircraft,
    then a selected launch.

    Returns (traj_data, cur_tt, sunlit_list, tgt_az, tgt_el) or None. traj_data
    is the (rows, times_array) pair whose rows carry [1]=elevation, [2]=azimuth
    (true sky frame, same as the navball). sunlit_list is the per-point sunlit
    cache when available (satellites), else None."""
    if tvs is None:
        return None
    cur_tt = getattr(tvs, 'current_tt', None)
    if cur_tt is None:
        # No shared timestamp yet (startup / mode switch): the ball still
        # renders, just without a target overlay.
        return None

    def _interp(*args):
        # A malformed trajectory (None rows around cur_tt) raises out of
        # numpy's searchsorted; that must cost the target marker for a frame,
        # not the whole navball.
        try:
            return interpolate_position_data_and_rates(*args)
        except Exception:
            return None

    sat = getattr(tvs, 'selected_satellite', None)
    sat_trajs = getattr(tvs, 'satellite_trajectories', None) or {}
    if sat is not None and sat_trajs.get(sat):
        traj = sat_trajs[sat]
        try:
            sunlit = getattr(tvs, 'sunlit_status_cache', {}).get(sat.name)
        except Exception:
            sunlit = None
        res = _interp(traj, cur_tt)
        tgt = (res[4], res[2]) if res and res[0] is not None else (None, None)
        return traj, cur_tt, sunlit, tgt[0], tgt[1]

    icao = getattr(tvs, 'selected_aircraft', None)
    ac_trajs = getattr(tvs, 'aircraft_trajectories', None) or {}
    if icao is not None and ac_trajs.get(icao):
        traj = ac_trajs[icao]
        res = _interp(traj, cur_tt)
        tgt = (res[4], res[2]) if res and res[0] is not None else (None, None)
        return traj, cur_tt, None, tgt[0], tgt[1]

    lname = getattr(tvs, 'selected_launch', None)
    launch_trajs = getattr(tvs, 'launch_trajectories', None) or {}
    if lname and launch_trajs.get(lname):
        traj = launch_trajs[lname]
        res = _interp(
            traj, cur_tt,
            getattr(tvs, 'launch_start_time', 0) or 0,
            getattr(tvs, 'launch_launched', False))
        tgt = (res[4], res[2]) if res and res[0] is not None else (None, None)
        return traj, cur_tt, None, tgt[0], tgt[1]

    return None


def render_navball(display, joystick_state):
    """Render a KSP-style navball in the center of the upper-left quadrant,
    showing the mount's current attitude (azimuth = heading tape, elevation =
    pitch). A fixed boresight reticle marks where the scope points; the ball
    moves under it. Greyed/blanked when the telescope is not connected."""
    layout = joystick_center_layout(display)
    if not layout['valid']:
        return
    rect = layout['navball']
    cx, cy = rect.center
    R = rect.width // 2 - 2
    if R < 30:
        return

    connected = joystick_state.telescope_connected
    # Show the same SKY az/el the skyplot draws, not raw mount coordinates. In
    # AltAz the mount ALT axis runs opposite sky elevation (sky el = 90 - ALT)
    # and azimuth carries the alignment offset, so using current_azm/current_alt
    # directly made the navball read inverted/rotated vs the camera pointing.
    cfg = getattr(joystick_state, 'config_state', None)
    if connected and cfg is not None:
        try:
            align_az = float(getattr(cfg, 'alignment_azimuth_str', 0.0) or 0.0)
            align_el = float(getattr(cfg, 'alignment_elevation_str', 0.0) or 0.0)
        except (TypeError, ValueError):
            align_az = align_el = 0.0
        mount_mode = getattr(cfg, 'mount_mode', 'AltAz')
        if mount_mode == 'AltAz':
            from transformations import AzAlt2AzEl_AltAz
            az, el = AzAlt2AzEl_AltAz(joystick_state.current_azm,
                                     joystick_state.current_alt, align_az)
        elif mount_mode == 'AltAz-Side':
            # Side-mounted rig: equatorial forward transform, pole on the
            # horizon at alignment_azimuth (index-mark home)
            from transformations import AzAlt2AzEl_AltAzSide
            az, el = AzAlt2AzEl_AltAzSide(joystick_state.current_azm,
                                          joystick_state.current_alt, align_az,
                                          flip=bool(getattr(cfg, 'altaz_side_flip', False)))
        else:
            from transformations import AzAlt2AzEl
            az, el = AzAlt2AzEl(joystick_state.current_azm,
                                joystick_state.current_alt, align_az, align_el)
        az = az % 360.0
        # Do NOT clamp elevation. Past-zenith / past-nadir mount angles are
        # valid orientations (e.g. ALT=350 -> el=100, i.e. 10 deg past zenith on
        # the far side). The spherical projection below rotates correctly through
        # the poles; the readout normalizes a copy for human-readable Az/El.
    else:
        az = el = 0.0

    # Title above the ball.
    display.menu_screen.blit(display.small_font.render("Navball", True, (220, 220, 230)),
                             (rect.x, rect.y - 20))

    # KSP-style palette.
    SKY = (86, 150, 220)          # upper hemisphere (light blue)
    GROUND = (150, 110, 70)       # lower hemisphere (tan/brown)
    HORIZON = (250, 250, 252)     # equator (horizon) great circle
    SKY_LINE = (225, 234, 246)    # parallels above the horizon
    GND_LINE = (212, 196, 176)    # parallels below the horizon
    MERIDIAN = (206, 210, 222)    # azimuth meridians (vertical tines)
    CARDINAL = (255, 230, 110)    # N / E / S / W
    INTER = (214, 218, 228)       # NE / SE / SW / NW
    BEZEL = (188, 190, 200)
    YELLOW = (255, 214, 0)        # boresight aircraft reticle
    COLORKEY = (1, 2, 3)          # transparent sentinel for the square fill

    bcx, bcy = R, R

    # --- Camera basis from the current pointing direction (az, el) -----------
    # World axes: X=east, Y=up, Z=north. A sky point (a, e) is the unit vector
    # v = (cos e sin a, sin e, cos e cos a). The ball is oriented so the current
    # pointing sits at the front center; screen-up follows increasing elevation.
    # Geometry is snapped to a small angular grid (QSTEP) so the expensive base
    # render below can be cached and reused frame-to-frame; the snap is sub-pixel
    # on screen but lets a stationary mount skip the rebuild entirely.
    QSTEP = 0.5
    gaz = round(az / QSTEP) * QSTEP
    gel = round(el / QSTEP) * QSTEP
    a0, e0 = math.radians(gaz), math.radians(gel)
    ce0, se0 = math.cos(e0), math.sin(e0)
    sa0, ca0 = math.sin(a0), math.cos(a0)
    Zc = (ce0 * sa0, se0, ce0 * ca0)                       # out of screen (front)
    Yc = (-se0 * sa0, ce0, -se0 * ca0)                     # screen up (+pitch)
    Xc = (Yc[1] * Zc[2] - Yc[2] * Zc[1],                   # screen right = Yc x Zc
          Yc[2] * Zc[0] - Yc[0] * Zc[2],
          Yc[0] * Zc[1] - Yc[1] * Zc[0])

    def proj(az_deg, el_deg):
        """Orthographic projection of a sky point to ball-local pixels.
        Returns (sx, sy, z); the point is on the visible front hemisphere when
        z > 0 (and is then guaranteed to fall inside the disc)."""
        ar, er = math.radians(az_deg), math.radians(el_deg)
        ce = math.cos(er)
        vx, vy, vz = ce * math.sin(ar), math.sin(er), ce * math.cos(ar)
        z = vx * Zc[0] + vy * Zc[1] + vz * Zc[2]
        x = vx * Xc[0] + vy * Xc[1] + vz * Xc[2]
        y = vx * Yc[0] + vy * Yc[1] + vz * Yc[2]
        return bcx + x * R, bcy - y * R, z

    def draw_curve(target, samples, color, width):
        """Draw a sky curve (list of (az, el)), broken into runs at the limb."""
        run = []
        for a_deg, e_deg in samples:
            sx, sy, z = proj(a_deg, e_deg)
            if z > 0.0:
                run.append((int(sx), int(sy)))
            elif len(run) >= 2:
                pygame.draw.lines(target, color, False, run, width)
                run = []
            else:
                run = []
        if len(run) >= 2:
            pygame.draw.lines(target, color, False, run, width)

    # --- Base ball (hemisphere fill + grid + static labels), cached -----------
    # Everything here depends only on (R, gaz, gel), so a one-deep cache makes a
    # stationary navball nearly free: on a hit we copy the cached surface instead
    # of rebuilding the per-pixel hemisphere and ~1700 trig projections. Only the
    # dynamic overlays (trajectory, target, aircraft) are redrawn every frame.
    base_key = (R, gaz, gel)
    if _NAVBALL_BASE_CACHE.get('key') == base_key and _NAVBALL_BASE_CACHE.get('surface') is not None:
        base = _NAVBALL_BASE_CACHE['surface']
    else:
        # Hemisphere fill (vectorized backprojection). A pixel's world-up
        # component is up.(NX*Xc + NY*Yc + NZ*Zc); its sign tells sky from
        # ground. (Xc[1], Yc[1], Zc[1]) is world-up in the camera frame, so the
        # whole disc resolves in one numpy expression.
        NX, NY, NZ, MASK = _navball_grid(R)
        vy_grid = Xc[1] * NX + Yc[1] * NY + Zc[1] * NZ
        size = 2 * R
        rgb = np.empty((size, size, 3), dtype=np.uint8)
        rgb[:] = COLORKEY
        rgb[MASK & (vy_grid >= 0.0)] = SKY
        rgb[MASK & (vy_grid < 0.0)] = GROUND
        base = pygame.surfarray.make_surface(rgb)
        base.set_colorkey(COLORKEY)

        # Grid: alt parallels (rings) + azimuth meridians (tines).
        az_samples = range(0, 361, 5)
        el_samples = range(-85, 86, 5)
        for az_line in range(0, 360, 30):                  # vertical az tines
            draw_curve(base, [(az_line, e) for e in el_samples], MERIDIAN, 1)
        for el_line in range(-80, 81, 10):                 # horizontal alt rings
            if el_line == 0:
                continue
            col = SKY_LINE if el_line > 0 else GND_LINE
            draw_curve(base, [(a, el_line) for a in az_samples],
                       col, 2 if el_line % 30 == 0 else 1)
        draw_curve(base, [(a, 0) for a in az_samples], HORIZON, 3)  # horizon last

        # Pitch numerals: green-on-dark chips parked in a gap between two azimuth
        # tines (off the central column where the reticle/meridian hid them).
        LABEL_FG, LABEL_BG = (130, 255, 140), (16, 26, 18)
        gap_az = math.floor(gaz / 30.0) * 30.0 + 15.0      # midway between tines
        for el_line in range(-60, 61, 30):
            if el_line == 0:
                continue
            sx, sy, z = proj(gap_az, el_line)
            if z > 0.0:
                t = display.tiny_font.render(f"{el_line:+d}", True, LABEL_FG, LABEL_BG)
                base.blit(t, (int(sx) - t.get_width() // 2, int(sy) - t.get_height() // 2))

        # Cardinal / intercardinal letters where their meridian meets the horizon.
        HEADINGS = {0: "N", 45: "NE", 90: "E", 135: "SE",
                    180: "S", 225: "SW", 270: "W", 315: "NW"}
        for h, lbl in HEADINGS.items():
            sx, sy, z = proj(h, 0)
            if z <= 0.0:
                continue
            col = CARDINAL if len(lbl) == 1 else INTER
            t = display.tiny_font.render(lbl, True, col)
            base.blit(t, (int(sx) - t.get_width() // 2,
                          int(sy) - t.get_height() - 2))

        _NAVBALL_BASE_CACHE['key'] = base_key
        _NAVBALL_BASE_CACHE['surface'] = base

    # Dynamic overlays paint onto a throwaway copy so the cached base stays clean.
    ball = base.copy()

    # --- Target trajectory + current target crosshair (mirrors the skyplot) --
    # Painted onto the ball surface so it is clipped to the disc and rides under
    # the boresight reticle. Trajectory: grey=past, yellow=sunlit future,
    # red=shadowed future. Target: purple crosshair at the live target az/el.
    PURPLE = (205, 95, 255)
    target = _navball_active_target(getattr(joystick_state, 'tracking_vis_state', None)) \
        if connected else None
    if target is not None:
        traj, cur_tt, sunlit, tgt_az, tgt_el = target
        rows, times_array = traj
        prev = None                                    # previous visible point
        for i, row in enumerate(rows):
            t_el, t_az = float(row[1]), float(row[2])
            sx, sy, z = proj(t_az, t_el)
            if t_el <= 0.0 or z <= 0.0:                 # below horizon / back side
                prev = None
                continue
            pt = (int(sx), int(sy))
            if prev is not None:
                future = (cur_tt is None) or (times_array[i] > cur_tt)
                if not future:
                    col = (130, 130, 130)
                elif sunlit is not None and i < len(sunlit):
                    col = (255, 255, 0) if sunlit[i] else (255, 80, 80)
                else:
                    col = (255, 255, 0)
                pygame.draw.line(ball, col, prev, pt, 2)
            prev = pt

        if tgt_az is not None:
            sx, sy, z = proj(tgt_az, tgt_el)
            if z > 0.0:
                tx, ty = int(sx), int(sy)
                rr = max(6, int(R * 0.075))
                pygame.draw.circle(ball, PURPLE, (tx, ty), rr, 2)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    pygame.draw.line(ball, PURPLE, (tx + dx * rr, ty + dy * rr),
                                     (tx + dx * (rr + 6), ty + dy * (rr + 6)), 2)
                pygame.draw.circle(ball, PURPLE, (tx, ty), 1)

    # --- ADS-B aircraft markers (all tracked aircraft, on the visible hemisphere)
    # Small orange diamonds; the selected aircraft gets a white ring. Drawn onto
    # the ball so they clip to the disc like the satellite/target overlays.
    if connected:
        tvs = getattr(joystick_state, 'tracking_vis_state', None)
        ac_positions = getattr(tvs, 'aircraft_positions', None) or {} if tvs else {}
        sel_ac = getattr(tvs, 'selected_aircraft', None) if tvs else None
        AC_COLOR = (255, 170, 60)
        for icao, pos in list(ac_positions.items()):
            try:
                _, _, ac_el, ac_az, _ = pos
            except (TypeError, ValueError):
                continue
            if ac_el is None or ac_el <= 0.0:
                continue
            sx, sy, z = proj(ac_az, ac_el)
            if z <= 0.0:
                continue
            ax, ay = int(sx), int(sy)
            d = max(3, int(R * 0.03))
            pygame.draw.polygon(ball, AC_COLOR,
                                [(ax, ay - d), (ax + d, ay), (ax, ay + d), (ax - d, ay)])
            if icao == sel_ac:
                pygame.draw.circle(ball, (255, 255, 255), (ax, ay), d + 3, 2)

    display.menu_screen.blit(ball, (cx - R, cy - R))

    # Bezel ring with 30-deg tick marks and a fixed heading index at the top.
    pygame.draw.circle(display.menu_screen, BEZEL, (cx, cy), R, 2)
    for deg in range(0, 360, 30):
        ang = math.radians(deg - 90)
        ca, sa = math.cos(ang), math.sin(ang)
        pygame.draw.line(display.menu_screen, BEZEL,
                         (int(cx + (R + 1) * ca), int(cy + (R + 1) * sa)),
                         (int(cx + (R + 7) * ca), int(cy + (R + 7) * sa)), 2)
    pygame.draw.polygon(display.menu_screen, CARDINAL,
                        [(cx, cy - R + 1), (cx - 6, cy - R - 9), (cx + 6, cy - R - 9)])

    # Fixed boresight reticle: KSP-style yellow aircraft "waterline" marker.
    r0 = max(4, int(R * 0.055))
    wing = max(10, int(R * 0.20))
    drop = max(3, wing // 3)
    pygame.draw.circle(display.menu_screen, YELLOW, (cx, cy), r0, 2)
    for sx in (-1, 1):
        x_in = cx + sx * (r0 + 2)
        x_out = cx + sx * (r0 + wing)
        pygame.draw.line(display.menu_screen, YELLOW, (x_in, cy), (x_out, cy), 3)
        pygame.draw.line(display.menu_screen, YELLOW, (x_out, cy), (x_out, cy + drop), 3)
    pygame.draw.line(display.menu_screen, YELLOW,
                     (cx, cy - r0 - 2), (cx, cy - r0 - max(5, wing // 2)), 3)

    # Green HDG / PITCH readout box below the ball (KSP instrument styling).
    box_w, box_h = int(2 * R * 0.92), 20
    box_x, box_y = cx - box_w // 2, cy + R + 8
    pygame.draw.rect(display.menu_screen, (16, 26, 18), (box_x, box_y, box_w, box_h))
    pygame.draw.rect(display.menu_screen, (70, 110, 80), (box_x, box_y, box_w, box_h), 1)
    if connected:
        # Normalize for display: fold past-pole orientations back into a valid
        # azimuth/elevation pair (el in [-90, 90], az flipped 180 over a pole).
        el_disp = ((el + 180.0) % 360.0) - 180.0
        az_disp = az
        if el_disp > 90.0:
            el_disp, az_disp = 180.0 - el_disp, az_disp + 180.0
        elif el_disp < -90.0:
            el_disp, az_disp = -180.0 - el_disp, az_disp + 180.0
        az_disp %= 360.0
        readout = f"HDG {az_disp:05.1f}°   PITCH {el_disp:+5.1f}°"
        rc = (130, 255, 140)
    else:
        readout = "HDG ---.-°   PITCH --.-°"
        rc = (110, 130, 110)
    t = display.small_font.render(readout, True, rc)
    display.menu_screen.blit(t, (cx - t.get_width() // 2,
                                 box_y + (box_h - t.get_height()) // 2))

    if not connected:
        scrim = pygame.Surface((2 * R, 2 * R), pygame.SRCALPHA)
        pygame.draw.circle(scrim, (25, 25, 30, 170), (R, R), R)
        display.menu_screen.blit(scrim, (cx - R, cy - R))


def render_bias_control_grid(display, joystick_state):
    """
    Render the manual bias control grid (D-pad mirror) with the active frame /
    resolution and current values. Anchored above the PID pane in the bottom-
    right of the upper-left quadrant. Always drawn so its location is visible;
    greyed out and non-interactable unless in a tracking mode that uses bias.
    """
    enabled = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    frame = getattr(joystick_state, 'bias_frame', 'azel')
    res = getattr(joystick_state, 'bias_resolution', 'coarse')
    alongcross = (frame == 'alongcross')
    h_lbl, v_lbl = ("InTk", "XTk") if alongcross else ("Az", "El")

    # Position above the PID pane, hugging the bottom-right of the quadrant
    pane = joystick_panel_layout(display)['bias']
    x_start, y_start = pane.x, pane.y
    width, height = pane.width, pane.height

    # Background rectangle
    pygame.draw.rect(display.menu_screen, (60, 60, 80),
                     (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, (120, 120, 150),
                     (x_start, y_start, width, height), 1)

    # Title (note the Op button cycles the mode)
    title_text = display.small_font.render("Bias Control", True, (255, 255, 255))
    display.menu_screen.blit(title_text, (x_start + 10, y_start + 5))
    hint = display.tiny_font.render("Op: mode", True, (150, 150, 180))
    display.menu_screen.blit(hint, (x_start + width - hint.get_width() - 8, y_start + 8))

    # Current bias mode indicator: resolution + frame
    res_color = (100, 255, 100) if res == "coarse" else (255, 180, 100)
    frame_color = (120, 200, 255) if alongcross else (200, 200, 120)
    res_text = display.tiny_font.render(f"{res.upper()}", True, res_color)
    display.menu_screen.blit(res_text, (x_start + 10, y_start + 25))
    frame_text = display.tiny_font.render(
        "ALONG/CROSS-TRACK" if alongcross else "AZ/EL", True, frame_color)
    display.menu_screen.blit(frame_text, (x_start + 55, y_start + 25))

    # Current values: show the active frame's pair brightly, the other dimmed.
    azel_c = (140, 140, 150) if alongcross else (255, 255, 255)
    ac_c = (255, 255, 255) if alongcross else (140, 140, 150)
    azel_str = f"Az {joystick_state.bias_azm_deg:+.2f}° El {joystick_state.bias_alt_deg:+.2f}°"
    ac_str = f"InTk {joystick_state.bias_intrack_deg:+.2f}° XTk {joystick_state.bias_crosstrack_deg:+.2f}°"
    display.menu_screen.blit(display.tiny_font.render(azel_str, True, azel_c), (x_start + 10, y_start + 40))
    display.menu_screen.blit(display.tiny_font.render(ac_str, True, ac_c), (x_start + 10, y_start + 52))

    # Button grid for manual adjustment (mirrors the D-pad). Each entry carries
    # its (horizontal, vertical) direction so the handler calls adjust_bias().
    button_size = 25
    button_spacing = 8
    grid_start_x = x_start + 20
    grid_start_y = y_start + 72

    h_color, v_color = (255, 150, 150), (150, 255, 150)
    button_positions = [
        (f"-{h_lbl}", -1, 0, grid_start_x, grid_start_y, h_color),
        (f"+{v_lbl}", 0, 1, grid_start_x + button_size + button_spacing, grid_start_y, v_color),
        (f"+{h_lbl}", 1, 0, grid_start_x + 2 * (button_size + button_spacing), grid_start_y, h_color),
        (f"-{v_lbl}", 0, -1, grid_start_x + button_size + button_spacing,
         grid_start_y + button_size + button_spacing, v_color),
    ]

    button_rects = []
    mouse_pos = pygame.mouse.get_pos()
    for label, hdir, vdir, bx, by, color in button_positions:
        button_rect = pygame.Rect(bx, by, button_size, button_size)
        button_rects.append((hdir, vdir, button_rect))

        hover = button_rect.collidepoint(mouse_pos)
        bg_color = tuple(min(255, c + 40) for c in color) if hover else color
        pygame.draw.rect(display.menu_screen, bg_color, button_rect)
        pygame.draw.rect(display.menu_screen, (200, 200, 200), button_rect, 1)

        label_text = display.tiny_font.render(label, True, (0, 0, 0))
        text_rect = label_text.get_rect(center=button_rect.center)
        display.menu_screen.blit(label_text, text_rect)

    # Store button rects in joystick state for mouse handling
    joystick_state.bias_button_rects = button_rects

    # Grey out when not in a bias-using tracking mode (visible but not interactable)
    if not enabled:
        _draw_disabled_scrim(display, pane)

def render_feed_forward_toggle_buttons(display, joystick_state):
    """
    Render feed-forward toggle buttons (FF AZ / FF EL / FF OFF) as a row inside
    the bottom of the PID Gain Control pane. Always drawn so the controls are
    visible; greyed out and non-interactable unless in a tracking mode that
    applies feed-forward (PROGRAM/HANDOFF program track; HOTSPOT rides its
    optical correction on the same trajectory rates).
    """
    enabled = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    # Lay the buttons out along the bottom strip of the PID pane
    pane = joystick_panel_layout(display)['pid']
    button_width, button_height = 72, 22
    button_spacing = 6
    x_start = pane.x + 12
    row_y = pane.bottom - button_height - 8

    mouse_pos = pygame.mouse.get_pos()

    # Section label
    ff_label = display.tiny_font.render("Feed-Forward:", True, (220, 220, 220))
    display.menu_screen.blit(ff_label, (x_start, row_y - 13))

    # AZ Feed-forward button
    az_ff_rect = pygame.Rect(x_start, row_y, button_width, button_height)
    az_hover = enabled and az_ff_rect.collidepoint(mouse_pos)

    az_color = (0, 150, 0) if joystick_state.feed_forward_azm_enabled else (100, 100, 100)
    if az_hover:
        az_color = tuple(min(255, c + 40) for c in az_color)

    pygame.draw.rect(display.menu_screen, az_color, az_ff_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), az_ff_rect, 1)

    az_text = display.tiny_font.render("FF AZ", True, (255, 255, 255))
    text_rect = az_text.get_rect(center=az_ff_rect.center)
    display.menu_screen.blit(az_text, text_rect)

    # EL Feed-forward button
    el_ff_rect = pygame.Rect(x_start + button_width + button_spacing, row_y, button_width, button_height)
    el_hover = enabled and el_ff_rect.collidepoint(mouse_pos)

    el_color = (0, 150, 0) if joystick_state.feed_forward_alt_enabled else (100, 100, 100)
    if el_hover:
        el_color = tuple(min(255, c + 40) for c in el_color)

    pygame.draw.rect(display.menu_screen, el_color, el_ff_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), el_ff_rect, 1)

    el_text = display.tiny_font.render("FF EL", True, (255, 255, 255))
    text_rect = el_text.get_rect(center=el_ff_rect.center)
    display.menu_screen.blit(el_text, text_rect)

    # Disable All button
    disable_rect = pygame.Rect(x_start + 2 * (button_width + button_spacing), row_y, button_width, button_height)
    disable_hover = enabled and disable_rect.collidepoint(mouse_pos)

    disable_color = (150, 100, 100) if disable_hover else (120, 100, 100)
    pygame.draw.rect(display.menu_screen, disable_color, disable_rect)
    pygame.draw.rect(display.menu_screen, (200, 200, 200), disable_rect, 1)

    disable_all_text = display.tiny_font.render("FF OFF", True, (255, 255, 255))
    text_rect = disable_all_text.get_rect(center=disable_rect.center)
    display.menu_screen.blit(disable_all_text, text_rect)

    # Store button rects for mouse click handling
    joystick_state.ff_button_rects = [
        (" ff_az", az_ff_rect),
        (" ff_el", el_ff_rect),
        (" ff_off", disable_rect)
    ]

    # Grey out the feed-forward strip when not in PROGRAM mode
    if not enabled:
        strip = pygame.Rect(pane.x + 1, row_y - 15, pane.width - 2, button_height + 19)
        _draw_disabled_scrim(display, strip)

def handle_bias_control_mouse_events(joystick_state, mouse_pos):
    """
    Handle mouse clicks on the bias control grid buttons (on-screen mirror of the
    D-pad). Routes through adjust_bias() so it honors the active frame/resolution.
    """
    if not hasattr(joystick_state, 'bias_button_rects'):
        return False

    # Interactable only in tracking modes that use bias (greyed out otherwise).
    if joystick_state.tracking_mode not in (
            TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT):
        return False

    for hdir, vdir, rect in joystick_state.bias_button_rects:
        if rect.collidepoint(mouse_pos):
            joystick_state.adjust_bias(hdir, vdir)
            print(f"Bias [{joystick_state.bias_frame}/{joystick_state.bias_resolution}]: "
                  f"Az {joystick_state.bias_azm_deg:+.2f}° El {joystick_state.bias_alt_deg:+.2f}° "
                  f"InTk {joystick_state.bias_intrack_deg:+.2f}° XTk {joystick_state.bias_crosstrack_deg:+.2f}°")
            return True
    return False

def render_pid_gain_sliders(display, joystick_state):
    """
    Render PID gain sliders for adjusting P, I, D gains in joystick mode.
    Anchored to the bottom-right of the upper-left quadrant (below the Bias
    pane). The feed-forward toggle buttons are drawn inside the bottom of this
    pane by render_feed_forward_toggle_buttons(). Always drawn so its location
    is visible; greyed out and non-interactable unless in a tracking mode that
    runs the shared PID loop (PROGRAM/HANDOFF/HOTSPOT).
    """
    # Only render when there's an active config_state
    if not hasattr(joystick_state, 'config_state') or joystick_state.config_state is None:
        return

    config_state = joystick_state.config_state
    enabled = joystick_state.tracking_mode in (
        TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT)

    # Pane hugging the bottom-right of the quadrant
    pane = joystick_panel_layout(display)['pid']
    x_start, y_start = pane.x, pane.y
    width, height = pane.width, pane.height

    # Background rectangle
    pygame.draw.rect(display.menu_screen, (70, 70, 90),
                     (x_start, y_start, width, height))
    pygame.draw.rect(display.menu_screen, (140, 140, 170),
                     (x_start, y_start, width, height), 1)

    # Title
    title_text = display.small_font.render("PID Gain Control", True, (255, 255, 255))
    display.menu_screen.blit(title_text, (x_start + 10, y_start + 5))

    # Define PID gain ranges (similar to camera settings)
    PID_GAIN_RANGE = (0.0, 2.0)  # From 0 to 2.0
    SLIDER_WIDTH = 80  # Width for each PID slider

    # Mouse position for hover detection
    mouse_pos = pygame.mouse.get_pos()

    # PID gain range for logarithmic sliders spanning 5 orders of magnitude (0.00002 to 2.0)
    PID_MAX_VALUE = 2.0
    PID_MIN_VALUE = 2.0 / 100000.0  # 5 orders of magnitude: 2.0e-5
    LOG_SCALE_FACTOR = 5.0  # 5 orders of magnitude
    LOG_SCALE_OFFSET = PID_MIN_VALUE  # Start just above zero

    # Recompute slider input and track rectangles every frame so they track the
    # pane's (quadrant-relative) position rather than being frozen at first render.
    display.joystick_pid_rects = {
        'pid_azm_p_gain': pygame.Rect(x_start + 40 - 10, y_start + 30, 60, 20),
        'pid_azm_i_gain': pygame.Rect(x_start + 40 - 10, y_start + 65, 60, 20),
        'pid_azm_d_gain': pygame.Rect(x_start + 40 - 10, y_start + 100, 60, 20),
        'pid_alt_p_gain': pygame.Rect(x_start + 160 - 10, y_start + 30, 60, 20),
        'pid_alt_i_gain': pygame.Rect(x_start + 160 - 10, y_start + 65, 60, 20),
        'pid_alt_d_gain': pygame.Rect(x_start + 160 - 10, y_start + 100, 60, 20),
    }
    display.joystick_pid_slider_rects = {
        'pid_azm_p_gain': pygame.Rect(x_start + 40 - 10, y_start + 55, SLIDER_WIDTH, 5),
        'pid_azm_i_gain': pygame.Rect(x_start + 40 - 10, y_start + 90, SLIDER_WIDTH, 5),
        'pid_azm_d_gain': pygame.Rect(x_start + 40 - 10, y_start + 125, SLIDER_WIDTH, 5),
        'pid_alt_p_gain': pygame.Rect(x_start + 160 - 10, y_start + 55, SLIDER_WIDTH, 5),
        'pid_alt_i_gain': pygame.Rect(x_start + 160 - 10, y_start + 90, SLIDER_WIDTH, 5),
        'pid_alt_d_gain': pygame.Rect(x_start + 160 - 10, y_start + 125, SLIDER_WIDTH, 5),
    }

    # AZM PID labels and inputs (left column)
    azm_title = display.tiny_font.render("AZM:", True, (255, 200, 100))
    display.menu_screen.blit(azm_title, (x_start, y_start + 20))

    # P gain
    p_label = display.tiny_font.render("P:", True, (255, 255, 255))
    display.menu_screen.blit(p_label, (x_start, y_start + 30))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_azm_p_gain'])
    p_value = getattr(config_state, 'pid_azm_p_gain', 0.0)
    p_text = display.tiny_font.render(f"{p_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(p_text, (display.joystick_pid_rects['pid_azm_p_gain'].x + 3, display.joystick_pid_rects['pid_azm_p_gain'].y + 2))

    # P slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_azm_p_gain'])
    # Position calculation: log10(value / min_val) / log10(max_val / min_val)
    log_position = math.log10(max(p_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_azm_p_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_azm_p_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_azm_p_gain'].y - 3, 6, 11))

    # I gain
    i_label = display.tiny_font.render("I:", True, (255, 255, 255))
    display.menu_screen.blit(i_label, (x_start, y_start + 65))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_azm_i_gain'])
    i_value = getattr(config_state, 'pid_azm_i_gain', 0.0)
    i_text = display.tiny_font.render(f"{i_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(i_text, (display.joystick_pid_rects['pid_azm_i_gain'].x + 3, display.joystick_pid_rects['pid_azm_i_gain'].y + 2))

    # I slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_azm_i_gain'])
    log_position = math.log10(max(i_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_azm_i_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_azm_i_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_azm_i_gain'].y - 3, 6, 11))

    # D gain
    d_label = display.tiny_font.render("D:", True, (255, 255, 255))
    display.menu_screen.blit(d_label, (x_start, y_start + 105))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_azm_d_gain'])
    d_value = getattr(config_state, 'pid_azm_d_gain', 0.0)
    d_text = display.tiny_font.render(f"{d_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(d_text, (display.joystick_pid_rects['pid_azm_d_gain'].x + 3, display.joystick_pid_rects['pid_azm_d_gain'].y + 2))

    # D slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_azm_d_gain'])
    log_position = math.log10(max(d_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_azm_d_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_azm_d_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_azm_d_gain'].y - 3, 6, 11))

    # ALT PID labels and inputs (right column)
    alt_title = display.tiny_font.render("ALT:", True, (255, 200, 100))
    display.menu_screen.blit(alt_title, (x_start + 120, y_start + 20))

    # P gain
    alt_p_label = display.tiny_font.render("P:", True, (255, 255, 255))
    display.menu_screen.blit(alt_p_label, (x_start + 120, y_start + 30))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_alt_p_gain'])
    alt_p_value = getattr(config_state, 'pid_alt_p_gain', 0.0)
    alt_p_text = display.tiny_font.render(f"{alt_p_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_p_text, (display.joystick_pid_rects['pid_alt_p_gain'].x + 3, display.joystick_pid_rects['pid_alt_p_gain'].y + 2))

    # P slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_alt_p_gain'])
    log_position = math.log10(max(alt_p_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_alt_p_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_alt_p_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_alt_p_gain'].y - 3, 6, 11))

    # I gain
    alt_i_label = display.tiny_font.render("I:", True, (255, 255, 255))
    display.menu_screen.blit(alt_i_label, (x_start + 120, y_start + 65))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_alt_i_gain'])
    alt_i_value = getattr(config_state, 'pid_alt_i_gain', 0.0)
    alt_i_text = display.tiny_font.render(f"{alt_i_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_i_text, (display.joystick_pid_rects['pid_alt_i_gain'].x + 3, display.joystick_pid_rects['pid_alt_i_gain'].y + 2))

    # I slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_alt_i_gain'])
    log_position = math.log10(max(alt_i_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_alt_i_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_alt_i_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_alt_i_gain'].y - 3, 6, 11))

    # D gain
    alt_d_label = display.tiny_font.render("D:", True, (255, 255, 255))
    display.menu_screen.blit(alt_d_label, (x_start + 120, y_start + 105))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.joystick_pid_rects['pid_alt_d_gain'])
    alt_d_value = getattr(config_state, 'pid_alt_d_gain', 0.0)
    alt_d_text = display.tiny_font.render(f"{alt_d_value:.5f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_d_text, (display.joystick_pid_rects['pid_alt_d_gain'].x + 3, display.joystick_pid_rects['pid_alt_d_gain'].y + 2))

    # D slider - LOGARITHMIC SCALING (5 orders of magnitude)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), display.joystick_pid_slider_rects['pid_alt_d_gain'])
    log_position = math.log10(max(alt_d_value, PID_MIN_VALUE) / PID_MIN_VALUE) / LOG_SCALE_FACTOR
    slider_ratio = min(1.0, max(0.0, log_position))
    handle_x = display.joystick_pid_slider_rects['pid_alt_d_gain'].x + int(slider_ratio * SLIDER_WIDTH)
    hover = pygame.Rect(handle_x - 3, display.joystick_pid_slider_rects['pid_alt_d_gain'].y - 3, 6, 11).collidepoint(mouse_pos)
    handle_color = (255, 0, 0) if hover else (200, 0, 0)
    pygame.draw.rect(display.menu_screen, handle_color, (handle_x - 3, display.joystick_pid_slider_rects['pid_alt_d_gain'].y - 3, 6, 11))

    # Focus highlights
    for field, rect in display.joystick_pid_rects.items():
        if hasattr(config_state, 'focused_field') and config_state.focused_field == field:
            pygame.draw.rect(display.menu_screen, (0, 0, 255), rect, 2)
            if hasattr(config_state, 'cursor_pos'):
                field_display_str = ""
                if field == 'pid_azm_p_gain':
                    field_display_str = f"{p_value:.5f}"
                elif field == 'pid_azm_i_gain':
                    field_display_str = f"{i_value:.5f}"
                elif field == 'pid_azm_d_gain':
                    field_display_str = f"{d_value:.5f}"
                elif field == 'pid_alt_p_gain':
                    field_display_str = f"{alt_p_value:.5f}"
                elif field == 'pid_alt_i_gain':
                    field_display_str = f"{alt_i_value:.5f}"
                elif field == 'pid_alt_d_gain':
                    field_display_str = f"{alt_d_value:.5f}"

                if field_display_str and field in config_state.cursor_pos:
                    text_width, _ = display.tiny_font.size(field_display_str[:config_state.cursor_pos[field]])
                    pygame.draw.line(display.menu_screen, (0, 0, 255),
                                   (rect.x + 5 + text_width, rect.y + 5),
                                   (rect.x + 5 + text_width, rect.y + 20), 2)

    # Lead-time slider (transport-latency compensation). Live-tunable like the
    # gains; takes effect immediately in both the Python and Rust loops, which
    # re-read pid_lead_time_sec each cycle. Range JL_LEAD_MIN..JL_LEAD_MAX.
    lead_val = float(getattr(config_state, 'pid_lead_time_sec', 0.0) or 0.0)
    lead_y = y_start + 143
    display.menu_screen.blit(display.tiny_font.render("Lead s:", True, (255, 200, 100)),
                             (x_start, lead_y))
    display.menu_screen.blit(display.tiny_font.render(f"{lead_val:.2f}", True, (255, 255, 255)),
                             (x_start + 50, lead_y))
    lead_track = pygame.Rect(x_start + 85, lead_y + 6, width - 97, 4)
    display.joystick_lead_slider_rect = lead_track
    pygame.draw.rect(display.menu_screen, (150, 150, 150), lead_track)
    lead_ratio = 0.0
    if JL_LEAD_MAX > JL_LEAD_MIN:
        lead_ratio = min(1.0, max(0.0, (lead_val - JL_LEAD_MIN) / (JL_LEAD_MAX - JL_LEAD_MIN)))
    lead_handle_x = lead_track.x + int(lead_ratio * lead_track.width)
    lead_hover = pygame.Rect(lead_handle_x - 3, lead_track.y - 4, 6, 12).collidepoint(mouse_pos)
    pygame.draw.rect(display.menu_screen, (255, 0, 0) if lead_hover else (200, 0, 0),
                     (lead_handle_x - 3, lead_track.y - 4, 6, 12))

    # Grey out the slider area when not in PROGRAM mode (the feed-forward strip
    # below is greyed separately by render_feed_forward_toggle_buttons()).
    if not enabled:
        slider_region = pygame.Rect(x_start + 1, y_start + 18, width - 2, 151)
        _draw_disabled_scrim(display, slider_region)

def _lead_from_track_x(track, x):
    """Map an x pixel on the lead slider track to a lead time (seconds)."""
    rel = min(max(x - track.x, 0), track.width)
    frac = rel / track.width if track.width else 0.0
    return round(JL_LEAD_MIN + frac * (JL_LEAD_MAX - JL_LEAD_MIN), 3)


def handle_lead_slider_mouse_events(joystick_state, display, mouse_pos):
    """Click on the joystick PID pane's Lead slider sets pid_lead_time_sec live.
    Dragging is handled in main.py's MOUSEMOTION path (same as the PID gain
    sliders). Returns True if the click hit the lead track."""
    if not hasattr(display, 'joystick_lead_slider_rect'):
        return False
    track = display.joystick_lead_slider_rect
    cfg = getattr(joystick_state, 'config_state', None)
    if cfg is None or not track.collidepoint(mouse_pos):
        return False
    cfg.pid_lead_time_sec = _lead_from_track_x(track, mouse_pos[0])
    print(f"Lead time: {cfg.pid_lead_time_sec:.3f}s")
    return True


def handle_pid_sliders_mouse_events(joystick_state, display, mouse_pos):
    """
    Handle mouse clicks on PID gain slider input fields.
    """
    if not hasattr(display, 'joystick_pid_rects') or not hasattr(joystick_state, 'config_state'):
        return False

    # PID gain sliders are only interactable in PROGRAM mode (greyed otherwise)
    if joystick_state.tracking_mode != TrackingMode.PROGRAM:
        return False

    config_state = joystick_state.config_state

    for field, rect in display.joystick_pid_rects.items():
        if rect.collidepoint(mouse_pos):
            config_state.focused_field = field
            if hasattr(config_state, 'cursor_pos'):
                config_state.cursor_pos[field] = 0
            if hasattr(config_state, 'selection_start'):
                config_state.selection_start[field] = None
            return True

    return False

def handle_star_filter_mouse_events(joystick_state, mouse_pos):
    """
    Handle clicks on the star-filter toggle in the PID Diagnostics pane.
    Toggles config.hotspot_star_filter_enabled (both control loops read it)
    and auto-saves so the choice persists across restarts.
    """
    rect = getattr(joystick_state, 'star_filter_button_rect', None)
    cfg = getattr(joystick_state, 'config_state', None)
    if rect is None or cfg is None or not rect.collidepoint(mouse_pos):
        return False
    if joystick_state.tracking_mode not in (
            TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT):
        return False
    cfg.hotspot_star_filter_enabled = not bool(
        getattr(cfg, 'hotspot_star_filter_enabled', True))
    state = "ON (stars rejected)" if cfg.hotspot_star_filter_enabled else "OFF (stars trackable)"
    print(f"Star filter: {state}")
    try:
        cfg.save_to_file()
    except Exception as e:
        print(f"Star filter: could not save config: {e}")
    return True


def handle_ff_toggle_mouse_events(joystick_state, mouse_pos):
    """
    Handle mouse clicks on feed-forward toggle buttons.
    Called from main event loop when buttons are clicked.
    """
    if not hasattr(joystick_state, 'ff_button_rects'):
        return False

    # Interactable while program-tracking (PROGRAM, or HANDOFF/HOTSPOT which
    # run the same feed-forward underneath; greyed otherwise).
    if joystick_state.tracking_mode not in (
            TrackingMode.PROGRAM, TrackingMode.HANDOFF, TrackingMode.HOTSPOT):
        return False

    for label, rect in joystick_state.ff_button_rects:
        if rect.collidepoint(mouse_pos):
            if " ff_az" in label:
                # Toggle AZ feed-forward
                joystick_state.feed_forward_azm_enabled = not joystick_state.feed_forward_azm_enabled
                if joystick_state.azm_pid:
                    joystick_state.azm_pid.set_feed_forward_enabled(joystick_state.feed_forward_azm_enabled)
                print(f"FF AZ toggled: {joystick_state.feed_forward_azm_enabled}")
            elif " ff_el" in label:
                # Toggle EL feed-forward
                joystick_state.feed_forward_alt_enabled = not joystick_state.feed_forward_alt_enabled
                if joystick_state.alt_pid:
                    joystick_state.alt_pid.set_feed_forward_enabled(joystick_state.feed_forward_alt_enabled)
                print(f"FF EL toggled: {joystick_state.feed_forward_alt_enabled}")
            elif " ff_off" in label:
                # Disable all feed-forward
                joystick_state.feed_forward_azm_enabled = False
                joystick_state.feed_forward_alt_enabled = False
                if joystick_state.azm_pid:
                    joystick_state.azm_pid.set_feed_forward_enabled(False)
                if joystick_state.alt_pid:
                    joystick_state.alt_pid.set_feed_forward_enabled(False)
                print("All feed-forward disabled")
            return True
    return False

