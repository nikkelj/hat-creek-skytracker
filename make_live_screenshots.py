#!/usr/bin/env python
"""
Offscreen capture of the live, multi-thread UI screens (tracking vis, sensor
calibration, joystick loop) by bringing up the real subsystems in SIM mode and
calling the same render code main.py uses. SDL dummy driver -> no visible window.

Output: doc/screenshots/{tracking_vis,sensor_calib,joystick_loop}.png
Run: python make_live_screenshots.py
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pygame
pygame.init()


class _Info:
    current_w = 1600
    current_h = 1000
pygame.display.Info = lambda: _Info()

OUT = os.path.join("doc", "screenshots")
os.makedirs(OUT, exist_ok=True)


def save(surface, name):
    pygame.image.save(surface, os.path.join(OUT, name))
    print("wrote", name)


def main():
    from skyfield.api import wgs84, load
    from display import DisplaySetup
    from config import load_config
    from tracking_visuals import (TrackingVisState, draw_filters, draw_legend, draw_details,
                                  draw_camera_fov_details, draw_time_display, draw_satellite_count,
                                  draw_satellite_pass_table, draw_scroll_bar)
    from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
    from trajectory import precompute_trajectories, update_satellite_positions
    from rendering_threads import TrackingVisualizationThread, JoystickVisualizationThread
    from simulator import HardwareSimulator
    from camera_manager import (camera_manager, update_camera_frames_from_buffers,
                                render_sensor_calibration, render_camera_sliders,
                                render_alignment_rotation_sliders, render_camera_roi_controls,
                                render_combined_view_controls)
    from joystick_controller import (JoystickModeState, render_connection_controls,
                                     render_joystick_status, render_position_display,
                                     render_pid_diagnostics, render_camera_feeds,
                                     render_navball, render_tracking_strip_charts,
                                     render_bias_control_grid, render_pid_gain_sliders,
                                     render_feed_forward_toggle_buttons, TrackingMode)
    from utils import draw_button_with_objects

    msgs = []
    cb = msgs.append

    display = DisplaySetup()
    cfg = load_config()
    cfg.sim_config["enabled"] = True
    ts = load.timescale()
    tvs = TrackingVisState()

    # ---- Load catalog + trajectories ----
    try:
        load_satellite_data(tvs, cb)
        if tvs.tle_loaded and tvs.satellites:
            create_satellite_labels_and_metadata(tvs, cb)
        lat, lon, alt = float(cfg.lat_str), float(cfg.lon_str), float(cfg.alt_str)
        observer = wgs84.latlon(lat, lon, elevation_m=alt)
        now = datetime.now(timezone.utc)
        tvs.center_time_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        tvs.duration_str = "60"
        tvs.t0 = ts.utc(now - timedelta(minutes=30))
        tvs.t1 = ts.utc(now + timedelta(minutes=30))
        precompute_trajectories(tvs, observer, ts, display, cb, now, 60)
        update_satellite_positions(tvs, ts.now().tt, elevation_mask_deg=float(cfg.elevation_mask_str or 0))
        print(f"catalog: tle_loaded={tvs.tle_loaded} visible={len(getattr(tvs,'satellite_positions',{}) or {})}")
    except Exception as e:
        import traceback; traceback.print_exc()

    # Pick a visible satellite to select / point at.
    sel = None
    sats = getattr(tvs, "satellite_positions", {}) or {}
    for name in sats:
        if name in getattr(tvs, "satellite_trajectories", {}):
            sel = name
            break
    if sel:
        tvs.selected_satellite = sel
        print("selected:", sel)

    # ---- Hardware sim + cameras ----
    hw = HardwareSimulator(cfg, tvs, ts)
    camera_manager.simulator = hw
    try:
        az, el, kind, vis = hw.current_target_azel()
        if vis:
            # Aim the sim mount at the target in MOUNT coords so render_frame
            # (which converts mount->sky) actually centers it. In AltAz the ALT
            # axis runs opposite sky elevation (ALT = 90 - el).
            if getattr(cfg, 'mount_mode', 'AltAz') == 'AltAz':
                align = float(getattr(cfg, 'alignment_azimuth_str', 0.0) or 0.0)
                hw.mount.az_true_deg, hw.mount.el_true_deg = az - align, 90.0 - el
            else:
                hw.mount.az_true_deg, hw.mount.el_true_deg = az, el
    except Exception:
        pass
    cfg.sim_config["star_density"] = 90000
    cfg.sim_config["target_brightness"] = 255
    cfg.sim_config["background_level"] = 14
    cfg.sim_config["read_noise"] = 4
    try:
        camera_manager.connect_camera(0, cb)
        camera_manager.connect_camera(1, cb)
        time.sleep(1.0)            # let the camera threads render a few frames
        update_camera_frames_from_buffers()
    except Exception as e:
        import traceback; traceback.print_exc()

    # ---- Render threads ----
    tviz = TrackingVisualizationThread(display, cfg, tvs, target_fps=10)
    jviz = JoystickVisualizationThread(display, cfg, tvs, target_fps=10)
    tviz.set_timescale(ts); jviz.set_timescale(ts)
    tviz.start(); jviz.start()
    # Wait until both threads have published a real surface.
    for _ in range(80):
        if tviz.get_latest_surface() is not None and jviz.get_latest_surface() is not None:
            break
        time.sleep(0.1)
    print("surfaces ready: tviz=%s jviz=%s" % (
        tviz.get_latest_surface() is not None, jviz.get_latest_surface() is not None))
    time.sleep(0.5)

    # ---- Joystick state (telescope connected via sim) ----
    jms = JoystickModeState(tvs, cfg, cb)
    jms.hardware_sim = hw
    jms.connect_telescope()
    # current_azm/current_alt are MOUNT coordinates (what the encoders read). In
    # AltAz the mount ALT axis runs opposite sky elevation (ALT = 90 - el), so
    # feed mount coords here -- otherwise the navball (which converts mount->sky)
    # would read inverted in the screenshot.
    jms.current_azm = hw.mount.az_true_deg
    jms.current_alt = 90.0 - hw.mount.el_true_deg
    jms.azm_display_str = f"{jms.current_azm:.1f}"
    jms.alt_display_str = f"{jms.current_alt:.1f}"
    # Focus motor: show it mid-travel with a small extend command so the focus
    # readout in joystick mode isn't blank in the screenshot.
    jms.focus_rate = 3
    jms.current_focus = int(round(hw.mount.focus_true_deg / 360.0 * (2 ** 24)))

    cur_tt = ts.now().tt

    # ============ Tracking Vis ============
    # Render the full-screen polar plot directly via the same draw functions the
    # thread uses (deterministic; no thread-timing dependence).
    from rendering_threads import (draw_polar_plot_on_surface, draw_fov_on_surface,
                                   draw_satellites_on_surface)
    from tracking_visuals import PolarPlotMode
    try:
        display.menu_screen.fill((0, 0, 0))
        plot = pygame.Surface((display.sub_width, display.sub_height))
        plot.fill((0, 0, 0))
        db = {'sub_x': display.sub_x, 'sub_y': display.sub_y,
              'sub_width': plot.get_width(), 'sub_height': plot.get_height()}
        ccx, ccy = plot.get_width() // 2, plot.get_height() // 2
        try:
            tviz.compute_camera_fov_data()
        except Exception:
            pass
        for fn in (lambda: draw_polar_plot_on_surface(plot, cfg, ts, cur_tt, tvs, db, PolarPlotMode.FULL_SCREEN),
                   lambda: draw_fov_on_surface(plot, tvs, ccx, ccy, db, PolarPlotMode.FULL_SCREEN, None),
                   lambda: draw_satellites_on_surface(plot, tvs, ccx, ccy, db, PolarPlotMode.FULL_SCREEN, cfg)):
            try:
                fn()
            except Exception:
                import traceback; traceback.print_exc()
        display.menu_screen.blit(plot, (display.sub_x, display.sub_y))
        for fn in (lambda: draw_filters(display, tvs),
                   lambda: draw_legend(display),
                   lambda: draw_details(display, tvs),
                   lambda: draw_camera_fov_details(display, tvs, 290),
                   lambda: draw_time_display(display),
                   lambda: draw_satellite_count(display, tvs),
                   lambda: draw_satellite_pass_table(display, tvs),
                   lambda: draw_button_with_objects(display, "clear_filters"),
                   lambda: draw_button_with_objects(display, "recompute"),
                   lambda: draw_scroll_bar(display, tvs)):
            try:
                fn()
            except Exception:
                pass
        save(display.menu_screen, "tracking_vis.png")
    except Exception:
        import traceback; traceback.print_exc()

    # ============ Sensor Calibration ============
    try:
        update_camera_frames_from_buffers()
        display.menu_screen.fill((30, 30, 30))
        c0 = camera_manager.get_camera(0).connected
        c1 = camera_manager.get_camera(1).connected
        render_sensor_calibration(display.menu_screen, display.sub_x, display.sub_y,
                                  display.sub_width, display.sub_height, c0, c1,
                                  "SimCam1 (finder)", "SimCam2 (narrow)")
        for fn in (lambda: render_camera_sliders(display.menu_screen, display.tiny_font, display.sub_x, display.sub_y, display.sub_width, display.sub_height),
                   lambda: render_alignment_rotation_sliders(display.menu_screen, display.tiny_font, display.sub_x, display.sub_y, display.sub_width, display.sub_height),
                   lambda: render_camera_roi_controls(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height),
                   lambda: (camera_manager._initialize_control_rects(display.menu_screen, display.total_width, display.total_height),
                            render_combined_view_controls(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height, display.small_font, display.tiny_font))):
            try:
                fn()
            except Exception:
                pass
        save(display.menu_screen, "sensor_calib.png")
    except Exception:
        import traceback; traceback.print_exc()

    # ============ Joystick Loop ============
    # Exercise the new center-column panels: PROGRAM mode so the bias/PID panes
    # light up, a bit of bias set, and some synthetic diagnostics history so the
    # strip charts show traces.
    import math as _math
    jms.tracking_mode = TrackingMode.PROGRAM
    jms.bias_frame, jms.bias_resolution = "alongcross", "fine"
    jms.bias_intrack_deg, jms.bias_crosstrack_deg = 0.35, -0.20
    for i in range(jms.diag_history_len):
        jms.az_rate_history.append(0.8 * _math.sin(i * 0.08))
        jms.el_rate_history.append(0.5 * _math.cos(i * 0.06))
        jms.az_err_history.append(0.15 * _math.sin(i * 0.05 + 1))
        jms.el_err_history.append(0.10 * _math.cos(i * 0.04))
    jms.azm_position_error, jms.alt_position_error = 0.12, -0.07
    jms.azm_pid_output, jms.alt_pid_output = 0.002, -0.001
    # Target sky-velocity so the camera along/cross-track bias axes render.
    jms.target_az_rate, jms.target_el_rate, jms.target_el_deg = 0.6, 0.3, 40.0
    try:
        update_camera_frames_from_buffers()
        display.menu_screen.fill((30, 30, 30), (display.sub_x, display.sub_y, display.sub_width, display.sub_height))
        jsurf = jviz.get_latest_surface()
        if jsurf:
            display.menu_screen.blit(jsurf, (display.sub_x + display.sub_width // 2, display.sub_y))
        for fn in (lambda: render_connection_controls(display, jms),
                   lambda: render_joystick_status(display, jms),
                   lambda: render_position_display(display, jms),
                   lambda: render_pid_diagnostics(display, jms),
                   lambda: render_pid_gain_sliders(display, jms),
                   lambda: render_bias_control_grid(display, jms),
                   lambda: render_feed_forward_toggle_buttons(display, jms),
                   lambda: render_navball(display, jms),
                   lambda: render_tracking_strip_charts(display, jms),
                   lambda: render_camera_feeds(display, jms)):
            try:
                fn()
            except Exception:
                import traceback; traceback.print_exc()
        save(display.menu_screen, "joystick_loop.png")
    except Exception:
        import traceback; traceback.print_exc()

    tviz.stop(); jviz.stop()
    try:
        camera_manager.disconnect_camera(0); camera_manager.disconnect_camera(1)
    except Exception:
        pass
    print("done")


if __name__ == "__main__":
    main()
