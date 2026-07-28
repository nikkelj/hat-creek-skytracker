"""
Explanatory hover tooltips for the UIs, with a global on/off toggle.

Design: rather than threading a tooltip call through every draw site, the
renderer here COLLECTS hoverable regions each frame from the rect registries
the screens already publish for click handling (jl_target_btn_rects,
object_toggle_rects, m3d.ui_rects, pane layout rects, display button rects...)
and pairs them with the static explanations below. Two granularities:
buttons/controls get specific tips; whole panes get an "about this pane" tip
that shows anywhere on the pane not covered by a more specific control (the
smallest matching rect wins).

main.py calls render(...) once per frame AFTER everything else (topmost), and
handle_click(...) first in the mouse dispatch. The Tips On/Off chip in the
bottom-right corner is drawn by render() itself on every screen; the state
persists via config_state.tooltips_enabled.
"""

import pygame

# ---- explanation texts -------------------------------------------------------

OBJECT_TOGGLE_TIPS = {
    'satellites_enabled': "Show/hide satellite markers on the sky plot. Shapes: circle = LEO, hexagon = MEO, triangle = GEO.",
    'satellite_labels_enabled': "Show/hide satellite name labels (the dots stay).",
    'aircraft_enabled': "Show/hide live ADS-B aircraft (orange diamonds).",
    'starfield_enabled': "Show/hide the Hipparcos star field. The 100 brightest named stars (gold spikes) are always shown.",
    'messier_enabled': "Show/hide the 110 Messier deep-sky objects (violet squares). Click one to target it.",
    'ngc_enabled': "Show/hide the NGC catalogue (teal circles), magnitude-limited in config. Noisy - off by default.",
}

JL_STRIP_TIPS = {
    'targets': "Open/close the target browser: filter and pick satellites and launches from a sortable pass table.",
    'sats': "Show/hide satellites on this sky plot.",
    'labels': "Show/hide satellite name labels.",
    'messier': "Show/hide Messier deep-sky objects.",
    'ngc': "Show/hide the NGC catalogue.",
}

PANE_TIPS = {
    'pid': "PID Gains: how aggressively tracking chases the target. P reacts to current error, I trims steady lag, D damps overshoot. Log sliders tune live; AUTOTUNE searches automatically while tracking.",
    'bias': "Bias: nudge the aim point in small steps (D-pad on the controller). Az/El or along/cross-track frame - for centering a target that sits slightly off.",
    'diag': "PID Diagnostics: live position error and commanded rates for each axis, plus the star-rejection filter toggle for optical tracking.",
    'plate': "Plate Solve: match a camera image against the star catalogue to find where the telescope is REALLY pointing, and optionally apply the alignment.",
    'adsb': "ADS-B: receive live aircraft positions with an RTL-SDR dongle. Click an aircraft on the sky plot to track it like a satellite.",
    'navball': "Navball: the mount's attitude like a flight game - the ball turns under the fixed crosshair as the telescope moves. Magenta marker = current target.",
    'chart_rate': "Strip chart: commanded tracking rates over time. Smooth lines = healthy tracking; sawtooth = discrete rate steps.",
    'chart_err': "Strip chart: tracking position error over time. Closer to zero is better.",
}

ADSB_TIPS = {
    'connect': "Start the RTL-SDR receiver and begin plotting live aircraft.",
    'disconnect': "Stop the ADS-B receiver (aircraft fade out).",
}

M3D_TIPS = {
    'view_toggle': "Switch between free orbit view (drag/wheel) and the operator view rendered from your configured seat position.",
    'follow_toggle': "FOLLOW LIVE poses the 3D mount from the real encoder angles; MANUAL POSE frees the AZM/ALT sliders.",
    'slider_azm': "Manually pose the model's azimuth axis (when not following live).",
    'slider_alt': "Manually pose the model's ALT axis (when not following live).",
    'seat_bearing_minus': "Move your operator seat marker around the mount.",
    'seat_bearing_plus': "Move your operator seat marker around the mount.",
    'seat_dist_minus': "Move the operator seat closer to the mount.",
    'seat_dist_plus': "Move the operator seat farther from the mount.",
    'seat_eye_minus': "Lower the operator eye height.",
    'seat_eye_plus': "Raise the operator eye height.",
}
M3D_TIPS.update({f'toggle_{a}': t for a, t in OBJECT_TOGGLE_TIPS.items()})

FILTER_TIPS = {
    'filter': "Type part of a name to filter the satellite list (e.g. 'ISS' or 'STARLINK').",
    'filter_above_alt': "Only show satellites with mean altitude ABOVE this (km).",
    'filter_below_alt': "Only show satellites with mean altitude BELOW this (km).",
}

CONN_TIPS = [
    ("connect", "Connect to the telescope mount (or the simulator when HW Sim is enabled)."),
    ("disconnect", "Disconnect from the mount."),
    ("sync", "Tell the mount that its current position is its home/zero reference."),
]

LEGEND_TIPS = {
    'gradient': "LEO altitude color scale. LEFT-click an altitude to show only satellites ABOVE it; RIGHT-click to show only ones BELOW it. Tick marks show the active filters; clear them in the filter boxes.",
    'meo': "Click to show/hide MEO satellites (orange hexagons - GPS, GLONASS, Galileo...).",
    'geo': "Click to show/hide GEO satellites (purple triangles - the geostationary belt).",
}

CONFIG_TIPS = {
    'alignment_azimuth': "TRUE north, not magnetic! If you pre-align with a phone compass, enable Settings > Compass > 'Use True North' on iPhone first (or add your local magnetic declination). A magnetic alignment is off by ~12 deg here - plenty to miss the ISS.",
    'lat': "Observer latitude in decimal degrees (north positive). Drives every pass prediction and pointing solution.",
    'lon': "Observer longitude in decimal degrees (east positive; US west coast is negative).",
    'alt': "Observer altitude above sea level, meters.",
    'elevation_mask': "Ignore sky below this elevation (deg): trees, rooftops, haze. Pass predictions and the red ring on the sky plot use it.",
    'azm_offset': "Encoder azimuth offset captured by Sync Home - the difference between the mount's raw zero and true north.",
    'alt_offset': "Encoder altitude offset captured by Sync Home.",
    'azm_limit_min': "Mount safety limit: slewing stops outside these azimuth bounds.",
    'azm_limit_max': "Mount safety limit: slewing stops outside these azimuth bounds.",
    'alt_limit_min': "Mount safety limit: keeps the tube off the tripod and the horizon.",
    'alt_limit_max': "Mount safety limit: maximum ALT-axis angle.",
}

MISC_TIPS = {
    'capture': "Record camera frames to disk while tracking - stop to save the labeled dataset.",
    'clear_filters': "Clear all satellite name/altitude filters.",
    'recompute': "Recompute satellite trajectories around the current time window.",
    'scrollbar': "Time slider: scrub the sky plot backward/forward in time. Visualization only - the mount always tracks at live time.",
    'gear': "Adaptive slew speed: full stick is capped at the green base range; hold the stick pinned to earn the orange boost gears. Release to wind back down.",
}

# ---- state / API --------------------------------------------------------------

_toggle_rect = None            # the Tips chip, set each render
_WRAP = 46                     # tooltip wrap width, chars


def enabled(config_state):
    return bool(getattr(config_state, 'tooltips_enabled', True))


def handle_click(pos, config_state):
    """Toggle chip click. Call first in the mouse dispatch; returns True if
    the click was consumed."""
    if _toggle_rect is not None and _toggle_rect.collidepoint(pos):
        config_state.tooltips_enabled = not enabled(config_state)
        return True
    return False


def _collect(display, config_state, mode, tvs, jms, m3d):
    """[(rect, text)] for the current screen from the published registries."""
    out = []

    def add_map(rects, texts):
        for key, rect in (rects or {}).items():
            t = texts.get(key)
            if t and rect:
                out.append((rect, t))

    if mode == "config_options":
        for field, rect in (getattr(display, 'input_rects', {}) or {}).items():
            t = CONFIG_TIPS.get(field)
            if t and rect:
                # Include the label line above the input box in the hover zone.
                out.append((rect.union(pygame.Rect(rect.x, rect.y - 22,
                                                   rect.width, 22)), t))

    elif mode == "tracking_vis" and tvs is not None:
        add_map(getattr(tvs, 'legend_rects', None), LEGEND_TIPS)
        add_map(getattr(tvs, 'object_toggle_rects', None), OBJECT_TOGGLE_TIPS)
        for key, tip in FILTER_TIPS.items():
            rect = getattr(display, {'filter': 'filter_rect',
                                     'filter_above_alt': 'filter_above_alt_rect',
                                     'filter_below_alt': 'filter_below_alt_rect'}[key], None)
            if rect:
                out.append((rect, tip))
        for attr, key in (('clear_filters_button', 'clear_filters'),
                          ('recompute_button', 'recompute'),
                          ('scroll_bar_rect', 'scrollbar')):
            rect = getattr(display, attr, None)
            if rect:
                out.append((rect, MISC_TIPS[key]))

    elif mode == "joystick" and jms is not None:
        add_map(getattr(jms, 'jl_target_btn_rects', None), JL_STRIP_TIPS)
        add_map(getattr(jms, 'adsb_button_rects', None), ADSB_TIPS)
        add_map(getattr(jms, 'jl_filter_rects', None), FILTER_TIPS)
        cap = getattr(jms, 'capture_button_rect', None)
        if cap:
            out.append((cap, MISC_TIPS['capture']))
        # Connection buttons (fixed geometry, mirrors the click handler).
        for i, (_k, tip) in enumerate(CONN_TIPS):
            out.append((pygame.Rect(display.sub_x + 10 + 90 * i,
                                    display.sub_y + 10, 80, 30), tip))
        # Pane-level tips (specific controls above win via smallest-rect rule).
        try:
            from joystick_controller import joystick_panel_layout, joystick_center_layout
            panes = joystick_panel_layout(display)
            for key in ('pid', 'bias', 'diag', 'plate', 'adsb'):
                if key in panes:
                    out.append((panes[key], PANE_TIPS[key]))
            center = joystick_center_layout(display)
            if center.get('valid'):
                for key in ('navball', 'chart_rate', 'chart_err'):
                    out.append((center[key], PANE_TIPS[key]))
        except Exception:
            pass

    elif mode == "mount3d" and m3d is not None:
        # m3d.ui_rects are surface-local; shift to screen coords.
        for key, rect in (getattr(m3d, 'ui_rects', {}) or {}).items():
            t = M3D_TIPS.get(key)
            if t and rect:
                out.append((rect.move(display.sub_x, display.sub_y), t))

    return out


def _wrap(text):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > _WRAP and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def render(display, config_state, mode, tracking_vis_state=None,
           joystick_state=None, m3d=None, mouse_pos=None):
    """Draw the Tips toggle chip and, when enabled, the tooltip for whatever
    the mouse is over. Call once per frame after all other drawing.
    mouse_pos overrides the live cursor (headless screenshot rigs)."""
    global _toggle_rect
    screen = display.menu_screen
    on = enabled(config_state)

    # Toggle chip, bottom-right corner of the window.
    w, h = screen.get_size()
    label = "Tips: ON" if on else "Tips: OFF"
    font = pygame.font.Font(None, 16)
    t = font.render(label, True, (255, 255, 255))
    _toggle_rect = pygame.Rect(w - t.get_width() - 22, h - 24,
                               t.get_width() + 14, 19)
    pos = mouse_pos if mouse_pos is not None else pygame.mouse.get_pos()
    hover_chip = _toggle_rect.collidepoint(pos)
    base = (60, 95, 70) if on else (70, 70, 78)
    pygame.draw.rect(screen, tuple(min(255, c + 25) for c in base) if hover_chip else base,
                     _toggle_rect)
    pygame.draw.rect(screen, (150, 150, 160), _toggle_rect, 1)
    screen.blit(t, (_toggle_rect.x + 7, _toggle_rect.y + 3))

    if not on:
        return
    if hover_chip:
        best = (_toggle_rect, "Turn explanatory hover tooltips on/off everywhere.")
    else:
        best = None
        for rect, text in _collect(display, config_state, mode,
                                   tracking_vis_state, joystick_state, m3d):
            if rect.collidepoint(pos):
                if best is None or rect.width * rect.height < best[0].width * best[0].height:
                    best = (rect, text)   # smallest region wins (most specific)
        if best is None:
            return

    lines = _wrap(best[1])
    tfont = pygame.font.Font(None, 17)
    surfs = [tfont.render(ln, True, (235, 235, 240)) for ln in lines]
    bw = max(s.get_width() for s in surfs) + 14
    bh = len(surfs) * 16 + 10
    x = min(pos[0] + 14, w - bw - 4)
    y = pos[1] + 18
    if y + bh > h - 4:
        y = pos[1] - bh - 8
    box = pygame.Surface((bw, bh), pygame.SRCALPHA)
    box.fill((22, 26, 34, 242))
    screen.blit(box, (x, y))
    pygame.draw.rect(screen, (120, 140, 170), pygame.Rect(x, y, bw, bh), 1)
    for i, s in enumerate(surfs):
        screen.blit(s, (x + 7, y + 5 + i * 16))
