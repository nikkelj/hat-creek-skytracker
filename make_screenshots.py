#!/usr/bin/env python
"""
Offscreen screenshot generator for the README.

Renders the real UI draw code and simulator camera output to PNGs without a
visible window (SDL dummy driver), so the captures match what the app draws.
Output: doc/screenshots/*.png

Run: python make_screenshots.py
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
pygame.init()

# Force a roomy canvas regardless of the (headless) desktop size.
class _Info:
    current_w = 1600
    current_h = 1000
pygame.display.Info = lambda: _Info()

OUT = os.path.join("doc", "screenshots")
os.makedirs(OUT, exist_ok=True)


def save(surface, name):
    path = os.path.join(OUT, name)
    pygame.image.save(surface, path)
    print("wrote", path)


def gray_to_surface(frame, upscale=1, gamma=0.5):
    # Display stretch (gamma) so faint stars/targets are legible, like a real
    # viewer's auto-stretch. The underlying frame the tracker sees is unmodified.
    f = np.clip(frame.astype(np.float32) / 255.0, 0, 1) ** gamma
    img = (f * 255.0).astype(np.uint8)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    if upscale != 1:
        surf = pygame.transform.scale(
            surf, (surf.get_width() * upscale, surf.get_height() * upscale))
    return surf


def main():
    from display import DisplaySetup
    from config import load_config, draw_config_options
    from hw_sim_ui import draw_hw_sim_options
    from simulator import HardwareSimulator

    display = DisplaySetup()
    cfg = load_config()

    # ---- Main menu ----
    try:
        display.render_initial_menu(["Hat Creek Skytracker ready.",
                                     "TLEs loaded.", "Select a mode."])
        save(display.menu_screen, "main_menu.png")
    except Exception as e:
        print("menu failed:", e)

    # ---- Config Options ----
    try:
        display.menu_screen.fill((30, 30, 30))
        draw_config_options(display, cfg)
        save(display.menu_screen, "config_options.png")
    except Exception as e:
        print("config failed:", e)

    # ---- Mount 3D screen ----
    # AltAz-Side is the money shot: the cyan AZM-axis arrow lies ON the
    # horizon at the alignment azimuth. Save/restore the mode so the later
    # blocks keep the user's configured mode.
    try:
        from mount3d import Mount3DState, draw_mount3d
        from skyfield.api import load as _sf_load
        _ts = _sf_load.timescale()
        _saved = (cfg.mount_mode, cfg.alignment_azimuth_str)
        cfg.mount_mode = "AltAz-Side"
        cfg.alignment_azimuth_str = "259.0"
        m3d = Mount3DState()
        m3d.follow_live = False
        # Pose chosen to point well above the horizon (sky el ~ +40 deg with
        # this pole geometry) so the FOV cones sweep visible sky.
        m3d.manual_azm, m3d.manual_alt = 312.0, 60.0
        # Camera on the opposite side of the boresight (sky az ~210) so the
        # FOV cones sweep across the visible star field.
        m3d.view_az, m3d.view_el = 40.0, 16.0
        display.menu_screen.fill((30, 30, 30))
        draw_mount3d(display, cfg, None, None, m3d, _ts)
        save(display.menu_screen, "mount3d.png")
        cfg.mount_mode, cfg.alignment_azimuth_str = _saved
    except Exception as e:
        print("mount3d failed:", e)

    # ---- HW Sim screen ----
    try:
        cfg.sim_config["enabled"] = True
        cfg.sim_config["mount_misalignment_az_deg"] = 1.5
        cfg.sim_config["mount_misalignment_el_deg"] = -0.8
        cfg.sim_config["cam2_offset_rotation_deg"] = 3.0
        sim = HardwareSimulator(cfg, None, None)
        sim.current_target_azel = lambda: (137.4, 42.1, 'satellite', True)
        display.menu_screen.fill((30, 30, 30))
        draw_hw_sim_options(display, cfg, sim)
        save(display.menu_screen, "hw_sim.png")
    except Exception as e:
        print("hw_sim failed:", e)

    # ---- Simulator camera frames ----
    # Larger exposure so star streaks are visible in the stills.
    cfg.sim_config["background_level"] = 8
    cfg.sim_config["read_noise"] = 3
    cfg.sim_config["target_brightness"] = 255
    cfg.sim_config["star_density"] = 120000   # spread over the whole sky; ~tens land in a wide FOV
    cfg.sim_config["cam2_offset_rotation_deg"] = 8.0
    for nm in ("camera1", "camera2"):
        cfg.camera_configs[nm]["exposure"] = 25000  # 0.025 s (tighter, brighter streaks)

    sim = HardwareSimulator(cfg, None, None)
    sim.mount.az_true_deg, sim.mount.el_true_deg = 100.0, 30.0
    sim.mount._az_rate_dps = 6.0   # moving -> stars streak
    sim.mount._el_rate_dps = 1.0

    try:
        # Wide/guide cam (FOV ~8 deg): satellite as a small dot among streaking stars.
        sim.current_target_azel = lambda: (100.4, 30.2, 'satellite', True)
        save(gray_to_surface(sim.render_frame(0)), "sim_wide_satellite.png")
    except Exception as e:
        print("sim wide frame failed:", e)

    try:
        # Narrow cam (FOV ~0.07 deg): target must be near boresight to be in frame.
        sim.current_target_azel = lambda: (100.0, 30.0, 'satellite', True)
        save(gray_to_surface(sim.render_frame(1)), "sim_narrow_v2mini.png")
    except Exception as e:
        print("sim narrow frame failed:", e)

    try:
        sim.mount._az_rate_dps = 2.0
        sim.current_target_azel = lambda: (100.3, 30.15, 'plume', True)
        save(gray_to_surface(sim.render_frame(0)), "sim_launch_plume.png")
    except Exception as e:
        print("sim plume frame failed:", e)


if __name__ == "__main__":
    main()
    print("done")
