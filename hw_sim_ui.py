"""
"Hardware Sim" UI screen: a master enable toggle plus [-] value [+] steppers for
every simulator parameter, and Save/Load. Uses simple rect hit-testing (no
text-entry machinery), and persists via the existing config save/load since
sim_config already round-trips through ConfigState.

draw_hw_sim_options() renders into display.menu_screen and stashes the control
rects on `display` for handle_hw_sim_click() to hit-test.
"""

import pygame

# (label, sim_config key, step, format)
ROWS = [
    ("Mount misalign AZ (deg)", "mount_misalignment_az_deg", 0.1, "{:.2f}"),
    ("Mount misalign EL (deg)", "mount_misalignment_el_deg", 0.1, "{:.2f}"),
    ("Encoder noise (deg)",     "mount_encoder_noise_deg",   0.01, "{:.3f}"),
    ("Rate jitter (deg/s)",     "mount_rate_noise_dps",      0.05, "{:.2f}"),
    ("Backlash (deg)",          "mount_backlash_deg",        0.002, "{:.3f}"),
    ("Periodic err amp (deg)",  "mount_pe_amplitude_deg",    0.001, "{:.3f}"),
    ("Periodic err period (s)", "mount_pe_period_sec",       30,   "{:.0f}"),
    ("Cam2 rotation (deg)",     "cam2_offset_rotation_deg",  0.5, "{:.1f}"),
    ("Cam2 offset X (px)",      "cam2_offset_x_px",          1,   "{:.0f}"),
    ("Cam2 offset Y (px)",      "cam2_offset_y_px",          1,   "{:.0f}"),
    ("Star density",            "star_density",              25,  "{:.0f}"),
    ("Background level",        "background_level",          1,   "{:.0f}"),
    ("Read noise",              "read_noise",                0.5, "{:.1f}"),
    ("Target brightness",       "target_brightness",         10,  "{:.0f}"),
    ("Seed",                    "seed",                      1,   "{:.0f}"),
]

NON_NEGATIVE = {"mount_encoder_noise_deg", "mount_rate_noise_dps", "mount_backlash_deg",
                "mount_pe_amplitude_deg", "mount_pe_period_sec", "star_density",
                "background_level", "read_noise", "target_brightness", "seed"}
INTEGER_KEYS = {"star_density", "seed", "cam2_offset_x_px", "cam2_offset_y_px"}


def _btn(surface, rect, label, font, bg=(70, 70, 70), fg=(255, 255, 255)):
    pygame.draw.rect(surface, bg, rect, border_radius=4)
    pygame.draw.rect(surface, (0, 0, 0), rect, 1, border_radius=4)
    t = font.render(label, True, fg)
    surface.blit(t, (rect.x + rect.width // 2 - t.get_width() // 2,
                     rect.y + rect.height // 2 - t.get_height() // 2))


def draw_hw_sim_options(display, config_state, hardware_sim=None):
    s = config_state.sim_config
    screen = display.menu_screen
    x0 = display.sub_x + 20
    y0 = display.sub_y + 20

    screen.fill((30, 30, 40), (display.sub_x, display.sub_y, display.sub_width, display.sub_height))
    screen.blit(display.large_font.render("Hardware Simulator", True, (255, 255, 255)), (x0, y0))

    enabled = bool(s.get("enabled", False))
    en_rect = pygame.Rect(x0, y0 + 40, 220, 34)
    _btn(screen, en_rect, f"Simulation: {'ON' if enabled else 'OFF'}", display.font,
         bg=(20, 140, 40) if enabled else (140, 40, 40))
    display.hw_sim_enable_rect = en_rect

    save_rect = pygame.Rect(x0 + 240, y0 + 40, 90, 34)
    load_rect = pygame.Rect(x0 + 340, y0 + 40, 90, 34)
    _btn(screen, save_rect, "Save", display.font)
    _btn(screen, load_rect, "Load", display.font)
    display.hw_sim_save_rect = save_rect
    display.hw_sim_load_rect = load_rect

    # Experimental: choose the control-loop implementation (Rust vs Python).
    # Takes effect on restart; auto-saved on toggle for convenience.
    rust_on = bool(getattr(config_state, "use_rust_core_loop", False))
    rust_rect = pygame.Rect(x0 + 440, y0 + 40, 230, 34)
    _btn(screen, rust_rect, f"Loop: {'RUST' if rust_on else 'PYTHON'}", display.font,
         bg=(20, 110, 150) if rust_on else (70, 70, 70))
    display.hw_sim_rust_loop_rect = rust_rect
    screen.blit(display.small_font.render("(restart app to apply)", True, (150, 150, 150)),
                (x0 + 440, y0 + 76))

    # Live status: what the sim thinks it is tracking right now.
    status = "target: --"
    if hardware_sim is not None:
        try:
            az, el, kind, vis = hardware_sim.current_target_azel()
            status = (f"target: {kind} @ AZ {az:.1f}  EL {el:.1f}" if vis
                      else "target: none in view")
        except Exception:
            pass
    screen.blit(display.small_font.render(status, True, (200, 200, 200)), (x0, y0 + 82))
    screen.blit(display.small_font.render(
        "(reconnect telescope + cameras after toggling to apply)", True, (150, 150, 150)),
        (x0, y0 + 100))

    # Continuous variable-rate tracking toggle (the control-theory lesson):
    # smooth guide-rate command vs the coarse 10-step MC_MOVE sawtooth. Applies
    # live (no restart); auto-saved on toggle.
    cont_on = bool(getattr(config_state, "continuous_rate_tracking", False))
    cont_rect = pygame.Rect(x0, y0 + 120, 360, 30)
    _btn(screen, cont_rect,
         f"Rate cmd: {'CONTINUOUS (guide-rate)' if cont_on else 'DISCRETE (MC_MOVE)'}",
         display.font, bg=(20, 110, 150) if cont_on else (70, 70, 70))
    display.hw_sim_continuous_rect = cont_rect
    screen.blit(display.small_font.render(
        "(smooth vs sawtooth -- a control-theory lesson; applies live)", True, (150, 150, 150)),
        (x0 + 370, y0 + 126))

    display.hw_sim_rects = {}
    ry = y0 + 162
    for label, key, step, fmt in ROWS:
        screen.blit(display.small_font.render(label, True, (230, 230, 230)), (x0, ry + 6))
        minus = pygame.Rect(x0 + 230, ry, 28, 26)
        valr = pygame.Rect(x0 + 262, ry, 90, 26)
        plus = pygame.Rect(x0 + 356, ry, 28, 26)
        _btn(screen, minus, "-", display.font)
        pygame.draw.rect(screen, (255, 255, 255), valr)
        pygame.draw.rect(screen, (0, 0, 0), valr, 1)
        screen.blit(display.small_font.render(fmt.format(float(s.get(key, 0))), True, (0, 0, 0)),
                    (valr.x + 6, valr.y + 5))
        _btn(screen, plus, "+", display.font)
        display.hw_sim_rects[key] = (minus, plus, step)
        ry += 36


def handle_hw_sim_click(pos, display, config_state, hardware_sim, status_messages):
    """Hit-test the HW Sim controls. Returns True if a control was clicked."""
    s = config_state.sim_config

    if getattr(display, 'hw_sim_enable_rect', None) and display.hw_sim_enable_rect.collidepoint(pos):
        s["enabled"] = not bool(s.get("enabled", False))
        status_messages.append(
            f"Simulation {'ENABLED' if s['enabled'] else 'DISABLED'} - reconnect devices to apply")
        return True

    if getattr(display, 'hw_sim_save_rect', None) and display.hw_sim_save_rect.collidepoint(pos):
        config_state.save_to_file()
        status_messages.append("Sim config saved")
        return True

    if getattr(display, 'hw_sim_load_rect', None) and display.hw_sim_load_rect.collidepoint(pos):
        config_state.load_from_file()
        status_messages.append("Sim config loaded")
        return True

    if getattr(display, 'hw_sim_rust_loop_rect', None) and display.hw_sim_rust_loop_rect.collidepoint(pos):
        config_state.use_rust_core_loop = not bool(getattr(config_state, "use_rust_core_loop", False))
        config_state.save_to_file()  # persist so it applies on restart
        status_messages.append(
            f"Control loop set to {'RUST' if config_state.use_rust_core_loop else 'PYTHON'} "
            "- restart app to apply (saved)")
        return True

    if getattr(display, 'hw_sim_continuous_rect', None) and display.hw_sim_continuous_rect.collidepoint(pos):
        config_state.continuous_rate_tracking = not bool(getattr(config_state, "continuous_rate_tracking", False))
        config_state.save_to_file()
        status_messages.append(
            "Rate command: "
            + ("CONTINUOUS guide-rate (smooth)" if config_state.continuous_rate_tracking
               else "DISCRETE MC_MOVE (sawtooth)"))
        return True

    for key, (minus, plus, step) in getattr(display, 'hw_sim_rects', {}).items():
        on_plus = plus.collidepoint(pos)
        if on_plus or minus.collidepoint(pos):
            newval = float(s.get(key, 0)) + (step if on_plus else -step)
            if key in NON_NEGATIVE:
                newval = max(0.0, newval)
            if key in INTEGER_KEYS:
                newval = int(round(newval))
            s[key] = newval
            return True

    return False
