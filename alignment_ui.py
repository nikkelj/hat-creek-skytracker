"""
"Alignment" UI screen: drive an automated pointing-model alignment and inspect the result.

Mirrors hw_sim_ui.py's simple rect-hit-test pattern. draw_alignment_options() renders into
display.menu_screen and stashes control rects on `display`; handle_alignment_click()
hit-tests them. The heavy lifting (slew, capture, plate-solve, fit, backtest) runs on an
AlignmentRunner worker thread so the render loop stays responsive.
"""

import math

import pygame

from pointing_model import TERM_NAMES
import alignment as al


def _btn(surface, rect, label, font, bg=(70, 70, 70), fg=(255, 255, 255)):
    pygame.draw.rect(surface, bg, rect, border_radius=4)
    pygame.draw.rect(surface, (0, 0, 0), rect, 1, border_radius=4)
    t = font.render(label, True, fg)
    surface.blit(t, (rect.x + rect.width // 2 - t.get_width() // 2,
                     rect.y + rect.height // 2 - t.get_height() // 2))


def _draw_alignment_skymap(display, rect, config_state, align_state):
    """Polar sky map of the alignment run: the planned sample grid, which points are
    done vs pending, the point currently being slewed to, the held-out backtest points,
    and (after the fit) the per-sample residual vectors. This is the operator's live
    'where is it slewing / how is it converging' view."""
    screen = display.menu_screen
    pygame.draw.rect(screen, (14, 16, 24), rect)
    pygame.draw.rect(screen, (90, 90, 110), rect, 1)
    cx, cy = rect.centerx, rect.centery + 6
    radius = min(rect.width, rect.height) // 2 - 16

    def to_xy(az, el):
        r = (90.0 - el) / 90.0 * radius
        return int(cx + r * math.sin(math.radians(az))), int(cy - r * math.cos(math.radians(az)))

    # grid: horizon, elevation mask, az spokes, cardinals
    try:
        mask = float(getattr(config_state, 'elevation_mask_str', 0.0) or 0.0)
    except ValueError:
        mask = 0.0
    pygame.draw.circle(screen, (200, 200, 210), (cx, cy), radius, 1)          # horizon
    if mask > 0:
        pygame.draw.circle(screen, (150, 60, 60), (cx, cy), int((90 - mask) / 90 * radius), 1)
    for el in (30, 60):
        pygame.draw.circle(screen, (45, 45, 55), (cx, cy), int((90 - el) / 90 * radius), 1)
    for az_deg in range(0, 360, 45):
        ex, ey = to_xy(az_deg, 0)
        pygame.draw.line(screen, (40, 40, 50), (cx, cy), (ex, ey), 1)
    for az_deg, lbl in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        lx, ly = to_xy(az_deg, -6)
        screen.blit(display.tiny_font.render(lbl, True, (150, 150, 170)),
                    (lx - 4, ly - 6))
    screen.blit(display.small_font.render("Sky sampling map", True, (170, 180, 210)),
                (rect.x + 6, rect.y + 4))

    done_fit = {(round(a, 3), round(e, 3)) for a, e, _, _ in align_state.samples}
    done_bt = {(round(a, 3), round(e, 3)) for a, e, _, _ in align_state.backtest_samples}

    # planned fit points: green=done, gray=pending
    for az, el in align_state.fit_points:
        x, y = to_xy(az, el)
        if (round(az, 3), round(el, 3)) in done_fit:
            pygame.draw.circle(screen, (60, 210, 110), (x, y), 4)
        else:
            pygame.draw.circle(screen, (90, 90, 100), (x, y), 4, 1)
    # held-out backtest points: blue (filled if visited)
    for az, el in align_state.holdout_points:
        x, y = to_xy(az, el)
        filled = (round(az, 3), round(el, 3)) in done_bt
        pygame.draw.circle(screen, (80, 150, 240), (x, y), 4, 0 if filled else 1)

    # residual vectors (after fit): orange, exaggerated, tip shows pointing error
    model = align_state.model
    if model is not None:
        scale = 300.0
        for az_cmd, el_cmd, az_obs, el_obs in align_state.samples:
            x, y = to_xy(az_cmd, el_cmd)
            pr_az, pr_el = model.error(az_cmd, el_cmd)
            d_az = ((az_obs - az_cmd + 180) % 360 - 180) - pr_az
            d_el = (el_obs - el_cmd) - pr_el
            ex = x + d_az * math.cos(math.radians(el_cmd)) * scale
            ey = y - d_el * scale
            pygame.draw.line(screen, (230, 140, 30), (x, y), (int(ex), int(ey)), 1)

    # current slew target: yellow ring + crosshair
    tgt = getattr(align_state, 'current_target', None)
    if tgt is not None:
        x, y = to_xy(tgt[0], tgt[1])
        pygame.draw.circle(screen, (255, 230, 60), (x, y), 8, 2)
        pygame.draw.line(screen, (255, 230, 60), (x - 11, y), (x + 11, y), 1)
        pygame.draw.line(screen, (255, 230, 60), (x, y - 11), (x, y + 11), 1)
        screen.blit(display.tiny_font.render("slewing", True, (255, 230, 60)), (x + 10, y - 14))

    # current camera FOV / mount pointing: red box + crosshair (matches the other
    # skyplots' cam1 FOV colour) -- moves live as the mount slews.
    mp = getattr(align_state, 'current_mount_azel', None)
    if mp is not None and mp[1] >= -2:
        x, y = to_xy(mp[0], mp[1])
        pygame.draw.rect(screen, (255, 70, 70), pygame.Rect(x - 6, y - 5, 12, 10), 1)
        pygame.draw.line(screen, (255, 70, 70), (x - 9, y), (x + 9, y), 1)
        pygame.draw.line(screen, (255, 70, 70), (x, y - 8), (x, y + 8), 1)
        screen.blit(display.tiny_font.render("FOV", True, (255, 120, 120)), (x + 9, y + 2))

    # legend
    ly = rect.bottom - 16
    for col, txt, dx in (((60, 210, 110), "done", 0), ((90, 90, 100), "pending", 50),
                         ((80, 150, 240), "backtest", 115), ((255, 230, 60), "slewing", 190),
                         ((255, 70, 70), "FOV", 252)):
        pygame.draw.circle(screen, col, (rect.x + 8 + dx, ly), 4)
        screen.blit(display.tiny_font.render(txt, True, (150, 150, 160)), (rect.x + 14 + dx, ly - 6))


def _draw_prereqs(display, config_state, joystick_state, x0, y0):
    """Render a readiness checklist so the operator can see what 'Start' needs."""
    import plate_solver as ps_mod
    import camera_manager

    cam_index = int(getattr(config_state, 'plate_solve_camera_index', 0))
    cam_name = f"camera{cam_index + 1}"
    sim = getattr(config_state, 'sim_config', {}) or {}
    sim_on = bool(sim.get('enabled', False))

    controller = getattr(joystick_state, 'telescope_controller', None) if joystick_state else None
    cam = camera_manager.camera_manager.get_camera(cam_index)
    cam_ready = cam is not None and getattr(cam, 'connected', False) and getattr(cam, 'thread', None) is not None
    try:
        fov = ps_mod.camera_fov_deg(config_state, cam_name)
        db_name = ps_mod._resolve_db_name(config_state, cam_name, fov)
        db_match = (4.0 <= fov <= 14.0 and db_name != 'default_database') or fov >= 10.0 \
            or (config_state.get_camera_config(cam_name) or {}).get('tetra3_db')
        db_txt = f"DB '{db_name}' for {fov:.2f}deg FOV" + ("" if db_match else "  (FOV/DB MISMATCH)")
    except Exception:
        db_match = False
        db_txt = "DB: unknown"

    checks = [
        (ps_mod.tetra3_available(), "tetra3 installed"),
        (controller is not None or sim_on, "mount connected" + ("" if controller is not None else " (auto in sim)")),
        (cam_ready or sim_on, f"cam{cam_index + 1} streaming" + ("" if cam_ready else " (auto in sim)")),
        (bool(sim.get('sim_use_real_stars', False)) or not sim_on, "sim REAL stars" + (" on" if sim.get('sim_use_real_stars') else " (auto-on at start)")),
        (db_match, db_txt),
    ]
    screen = display.menu_screen
    screen.blit(display.small_font.render("Prerequisites:", True, (200, 200, 220)), (x0, y0))
    yy = y0 + 18
    for ok, label in checks:
        color = (120, 220, 140) if ok else (240, 160, 120)
        mark = "OK " if ok else "!! "
        screen.blit(display.small_font.render(mark + label, True, color), (x0, yy))
        yy += 16


def draw_alignment_options(display, config_state, align_state, joystick_state=None):
    screen = display.menu_screen
    x0 = display.sub_x + 20
    y0 = display.sub_y + 20
    mid = display.sub_x + display.sub_width // 2   # panel split (skymap left / camera|tables right)
    screen.fill((28, 30, 38), (display.sub_x, display.sub_y, display.sub_width, display.sub_height))
    screen.blit(display.large_font.render("Pointing-Model Alignment", True, (255, 255, 255)), (x0, y0))

    mount_mode = getattr(config_state, 'mount_mode', 'AltAz')
    if mount_mode != 'AltAz':
        screen.blit(display.font.render(
            f"Alignment supports AltAz mount mode only (current: {mount_mode}).",
            True, (255, 180, 120)), (x0, y0 + 44))

    running = align_state.phase in (al.RUNNING, al.BACKTEST)

    # Control buttons.
    start_rect = pygame.Rect(x0, y0 + 44, 150, 34)
    abort_rect = pygame.Rect(x0 + 160, y0 + 44, 110, 34)
    accept_rect = pygame.Rect(x0 + 280, y0 + 44, 170, 34)
    _btn(screen, start_rect, "Running..." if running else "Start Alignment", display.font,
         bg=(120, 90, 20) if running else (20, 120, 60))
    _btn(screen, abort_rect, "Abort", display.font, bg=(120, 40, 40))
    can_accept = align_state.model is not None
    _btn(screen, accept_rect, "Apply to Config", display.font,
         bg=(40, 90, 150) if can_accept else (60, 60, 60))
    display.align_start_rect = start_rect
    display.align_abort_rect = abort_rect
    display.align_accept_rect = accept_rect

    # Mount controls: connect/disconnect (sim-aware) + live az/el readout.
    connected = bool(getattr(joystick_state, 'telescope_connected', False)) if joystick_state else False
    conn_rect = pygame.Rect(x0 + 460, y0 + 44, 90, 34)
    disc_rect = pygame.Rect(x0 + 556, y0 + 44, 110, 34)
    _btn(screen, conn_rect, "Connect", display.font,
         bg=(60, 60, 60) if connected else (40, 110, 70))
    _btn(screen, disc_rect, "Disconnect", display.font,
         bg=(110, 50, 50) if connected else (60, 60, 60))
    display.align_connect_rect = conn_rect
    display.align_disconnect_rect = disc_rect
    readout = "mount: not connected"
    if connected:
        try:
            from transformations import AzAlt2AzEl_AltAz
            azm = float(getattr(joystick_state, 'current_azm_raw', 0.0))
            alt = float(getattr(joystick_state, 'current_alt_raw', 0.0))
            align_az = float(getattr(config_state, 'alignment_azimuth_str', 0.0) or 0.0)
            saz, sel = AzAlt2AzEl_AltAz(azm, alt, align_az)
            readout = f"mount  Az {saz:6.1f}°  El {sel:5.1f}°"
        except Exception:
            readout = "mount: connected"
    screen.blit(display.small_font.render(readout, True, (170, 200, 230)), (x0 + 676, y0 + 54))

    # n_points stepper.
    npts = int(getattr(align_state, 'requested_points', 18))
    screen.blit(display.small_font.render("Sample points:", True, (220, 220, 220)), (x0, y0 + 90))
    minus = pygame.Rect(x0 + 120, y0 + 86, 28, 26)
    valr = pygame.Rect(x0 + 152, y0 + 86, 56, 26)
    plus = pygame.Rect(x0 + 212, y0 + 86, 28, 26)
    _btn(screen, minus, "-", display.font)
    pygame.draw.rect(screen, (255, 255, 255), valr); pygame.draw.rect(screen, (0, 0, 0), valr, 1)
    screen.blit(display.small_font.render(str(npts), True, (0, 0, 0)), (valr.x + 8, valr.y + 5))
    _btn(screen, plus, "+", display.font)
    display.align_points_minus = minus
    display.align_points_plus = plus

    # Pointing-model master toggle.
    pm_on = bool(getattr(config_state, 'pointing_model_enabled', False))
    pm_rect = pygame.Rect(x0 + 250, y0 + 86, 130, 26)
    _btn(screen, pm_rect, f"Model: {'ON' if pm_on else 'OFF'}", display.small_font,
         bg=(20, 110, 60) if pm_on else (90, 60, 60))
    display.align_model_toggle_rect = pm_rect

    # View toggle (right panel: camera <-> tables).
    view = getattr(align_state, 'view_mode', 'camera')
    view_rect = pygame.Rect(x0 + 388, y0 + 86, 130, 26)
    _btn(screen, view_rect, f"View: {'Tables' if view == 'tables' else 'Camera'}", display.small_font,
         bg=(60, 80, 120))
    display.align_view_rect = view_rect

    # Pause-on-fail toggle (manual intervention when auto+grid-search fail).
    pof = bool(getattr(align_state, 'pause_on_fail', True))
    pof_rect = pygame.Rect(x0 + 526, y0 + 86, 150, 26)
    _btn(screen, pof_rect, f"Pause-on-fail: {'ON' if pof else 'OFF'}", display.small_font,
         bg=(110, 80, 30) if pof else (70, 70, 70))
    display.align_pausefail_rect = pof_rect

    # Retry-failed (re-run the points that never solved, then refit).
    nfail = len(getattr(align_state, 'failed_points', []))
    retry_rect = pygame.Rect(x0 + 684, y0 + 86, 150, 26)
    can_retry = nfail > 0 and not running
    _btn(screen, retry_rect, f"Retry Failed ({nfail})", display.small_font,
         bg=(120, 80, 40) if can_retry else (60, 60, 60))
    display.align_retry_rect = retry_rect

    # Manual-intervention banner + controls (shown while paused for a hand-jogged point).
    display.align_manual_skip_rect = None
    display.align_manual_abort_rect = None
    if getattr(align_state, 'manual_active', False):
        mt = getattr(align_state, 'manual_target', None) or (0.0, 0.0)
        bx, by = mid + 10, y0 + 40
        pygame.draw.rect(screen, (60, 40, 20), (bx - 4, by - 4, display.sub_x + display.sub_width - bx - 16, 96))
        pygame.draw.rect(screen, (220, 170, 60), (bx - 4, by - 4, display.sub_x + display.sub_width - bx - 16, 96), 1)
        screen.blit(display.font.render("MANUAL INTERVENTION", True, (255, 210, 120)), (bx, by))
        screen.blit(display.small_font.render(
            f"Jog with the joystick to frame stars near az {mt[0]:.0f}° el {mt[1]:.0f}°; "
            "auto-captures on the next solve.", True, (235, 220, 190)), (bx, by + 26))
        skip_rect = pygame.Rect(bx, by + 50, 120, 30)
        ab_rect = pygame.Rect(bx + 132, by + 50, 120, 30)
        _btn(screen, skip_rect, "Skip Point", display.font, bg=(90, 80, 40))
        _btn(screen, ab_rect, "Abort Run", display.font, bg=(120, 40, 40))
        display.align_manual_skip_rect = skip_rect
        display.align_manual_abort_rect = ab_rect

    # Status + progress + live convergence readout.
    done, total = align_state.progress
    status = align_state.status or "idle"
    screen.blit(display.font.render(f"Status: {status}", True, (200, 220, 255)), (x0, y0 + 122))
    if total:
        bar = pygame.Rect(x0, y0 + 148, 460, 14)
        pygame.draw.rect(screen, (60, 60, 70), bar)
        pygame.draw.rect(screen, (60, 160, 90), (bar.x, bar.y, int(bar.width * done / max(1, total)), bar.height))
        pygame.draw.rect(screen, (120, 120, 140), bar, 1)
    # good-solve count + live slew target (progress toward convergence)
    good = len(align_state.samples)
    bt = len(align_state.backtest_samples)
    line2 = f"sampled {done}/{total}   good solves: {good}" + (f"   backtest: {bt}" if bt else "")
    ct = getattr(align_state, 'current_target', None)
    if ct is not None:
        line2 += f"   ->  slewing to az {ct[0]:.0f}° el {ct[1]:.0f}°"
    screen.blit(display.small_font.render(line2, True, (200, 220, 200)), (x0, y0 + 166))

    # Below the control header the panel splits in two: the live sky sampling map
    # fills the left half and the right half shows either the plate-solve camera or the
    # data tables (View toggle). The right-header band above the split shows -- in
    # priority order -- the manual banner (drawn above), the fit results, or the
    # prerequisite checklist while idle.
    split_top = y0 + 200
    panel_bottom = display.sub_y + display.sub_height - 40
    rhx = max(mid + 10, x0 + 845)   # right-header content x, kept clear of the toggle row

    if getattr(align_state, 'manual_active', False):
        pass  # the manual banner already owns the right-header band
    elif align_state.stats is not None:
        st = align_state.stats
        screen.blit(display.font.render(
            f"Fit RMS: {st['rms_before_arcmin']:.2f}'  ->  {st['rms_after_arcmin']:.2f}'  "
            f"(n={st['n_samples']})", True, (200, 255, 200)), (rhx, y0 + 40))
        if align_state.backtest_rms_deg is not None:
            screen.blit(display.font.render(
                f"Backtest RMS (held-out): {align_state.backtest_rms_deg * 60.0:.2f}'",
                True, (180, 230, 255)), (rhx, y0 + 64))
        terms = align_state.model.terms
        for i, k in enumerate(TERM_NAMES):
            col = rhx + (i % 3) * 150
            row = (y0 + 94) + (i // 3) * 22
            screen.blit(display.small_font.render(f"{k}: {terms[k]:+.4f}°", True, (230, 230, 200)),
                        (col, row))
    else:
        _draw_prereqs(display, config_state, joystick_state, rhx, y0 + 40)

    # Live sky sampling map: left half, below the controls.
    skymap = pygame.Rect(x0, split_top, (mid - 10) - x0, panel_bottom - split_top)
    _draw_alignment_skymap(display, skymap, config_state, align_state)

    # Right half: camera live view (+ solved-star overlay) or the data tables.
    if view == 'tables':
        _draw_alignment_tables(display, config_state, align_state)
    else:
        _draw_alignment_cam_panel(display, config_state, align_state)

    # Hint.
    screen.blit(display.small_font.render(
        "Connect the mount here (sim auto-connects), Start to slew+solve a sky grid, then fit. "
        "On a failed point it grid-searches, then pauses for a manual joystick jog (auto-captures on solve).",
        True, (150, 150, 160)), (x0, display.sub_y + display.sub_height - 26))


def handle_alignment_click(pos, display, config_state, align_state, joystick_state,
                           tracking_vis_state, ts, status_messages):
    """Hit-test alignment controls. Returns True if a control was clicked."""
    # n_points steppers
    if getattr(display, 'align_points_minus', None) and display.align_points_minus.collidepoint(pos):
        align_state.requested_points = max(8, int(getattr(align_state, 'requested_points', 18)) - 2)
        return True
    if getattr(display, 'align_points_plus', None) and display.align_points_plus.collidepoint(pos):
        align_state.requested_points = min(60, int(getattr(align_state, 'requested_points', 18)) + 2)
        return True

    if getattr(display, 'align_model_toggle_rect', None) and display.align_model_toggle_rect.collidepoint(pos):
        config_state.pointing_model_enabled = not bool(getattr(config_state, 'pointing_model_enabled', False))
        config_state.save_to_file()
        status_messages.append(
            f"Pointing model {'ENABLED' if config_state.pointing_model_enabled else 'DISABLED'}")
        return True

    # View toggle (camera <-> tables) for the right panel.
    if getattr(display, 'align_view_rect', None) and display.align_view_rect.collidepoint(pos):
        align_state.view_mode = 'tables' if getattr(align_state, 'view_mode', 'camera') == 'camera' else 'camera'
        return True

    # Pause-on-fail (manual intervention) toggle.
    if getattr(display, 'align_pausefail_rect', None) and display.align_pausefail_rect.collidepoint(pos):
        align_state.pause_on_fail = not bool(getattr(align_state, 'pause_on_fail', True))
        status_messages.append(
            f"Manual pause-on-fail {'ON' if align_state.pause_on_fail else 'OFF'}")
        return True

    # Mount connect / disconnect (sim-aware; reuses the joystick connection path).
    if getattr(display, 'align_connect_rect', None) and display.align_connect_rect.collidepoint(pos):
        if joystick_state is not None and not getattr(joystick_state, 'telescope_connected', False):
            ok = joystick_state.connect_telescope()
            status_messages.append("Mount connected" if ok else
                                   "Mount connect failed (select a port in Joystick mode, or enable the sim)")
        return True
    if getattr(display, 'align_disconnect_rect', None) and display.align_disconnect_rect.collidepoint(pos):
        if joystick_state is not None and getattr(joystick_state, 'telescope_connected', False):
            joystick_state.disconnect_telescope()
            status_messages.append("Mount disconnected")
        return True

    # Manual-intervention buttons (shown only while paused for a hand-jogged point).
    if getattr(display, 'align_manual_skip_rect', None) and display.align_manual_skip_rect.collidepoint(pos):
        runner = getattr(align_state, 'runner', None)
        if runner is not None:
            runner.skip()
            status_messages.append("Manual: skipping this point")
        return True
    if getattr(display, 'align_manual_abort_rect', None) and display.align_manual_abort_rect.collidepoint(pos):
        runner = getattr(align_state, 'runner', None)
        if runner is not None:
            runner.abort()
            status_messages.append("Alignment abort requested")
        return True

    if getattr(display, 'align_abort_rect', None) and display.align_abort_rect.collidepoint(pos):
        runner = getattr(align_state, 'runner', None)
        if runner is not None:
            runner.abort()
            status_messages.append("Alignment abort requested")
        return True

    if getattr(display, 'align_accept_rect', None) and display.align_accept_rect.collidepoint(pos):
        if al.accept_alignment(align_state, config_state):
            status_messages.append("Pointing model applied to config and enabled (saved)")
        else:
            status_messages.append("No fitted model to apply")
        return True

    # Retry the points that never solved (re-run them, append, refit).
    if getattr(display, 'align_retry_rect', None) and display.align_retry_rect.collidepoint(pos):
        _retry_failed(config_state, align_state, joystick_state, tracking_vis_state, ts, status_messages)
        return True

    if getattr(display, 'align_start_rect', None) and display.align_start_rect.collidepoint(pos):
        _start_alignment(config_state, align_state, joystick_state, tracking_vis_state, ts, status_messages)
        return True

    return False


def _start_alignment(config_state, align_state, joystick_state, tracking_vis_state, ts, status_messages):
    if align_state.phase in (al.RUNNING, al.BACKTEST):
        status_messages.append("Alignment already running")
        return
    if getattr(config_state, 'mount_mode', 'AltAz') != 'AltAz':
        status_messages.append("Alignment requires AltAz mount mode")
        return

    sampler = _prepare_alignment_run(config_state, align_state, joystick_state,
                                     tracking_vis_state, ts, status_messages)
    if sampler is None:
        return

    npts = int(getattr(align_state, 'requested_points', 18))
    align_state.reset()
    align_state.requested_points = npts
    runner = al.AlignmentRunner(align_state, config_state, sampler, n_points=npts)
    align_state.runner = runner
    runner.start()
    status_messages.append(f"Alignment started ({npts} points, cam{sampler.cam_index + 1}, "
                           f"DB {sampler.solver.db_name})")


def _prepare_alignment_run(config_state, align_state, joystick_state, tracking_vis_state,
                           ts, status_messages):
    """Shared setup for a run: tetra3 check, mount + camera connect, sim stars, solver,
    park the control loop, and build the GuiSampler. Returns the sampler or None."""
    import plate_solver as ps_mod
    if not ps_mod.tetra3_available():
        status_messages.append("tetra3 not installed - cannot plate solve")
        return None

    sim_enabled = bool(getattr(joystick_state, 'hardware_sim', None) and
                       joystick_state.hardware_sim.sim_enabled())
    cam_index = int(getattr(config_state, 'plate_solve_camera_index', 0))
    cam_name = f"camera{cam_index + 1}"

    # Mount: auto-connect in sim so the operator doesn't have to visit Joystick mode.
    controller = getattr(joystick_state, 'telescope_controller', None) if joystick_state else None
    if controller is None:
        if sim_enabled and joystick_state is not None:
            try:
                joystick_state.connect_telescope()
                controller = joystick_state.telescope_controller
            except Exception as e:
                status_messages.append(f"Sim mount connect failed: {e}")
        if controller is None:
            status_messages.append("Connect the mount first (button above, or Joystick mode)")
            return None

    # Camera: make sure the plate-solve camera is streaming frames.
    import camera_manager
    cam = camera_manager.camera_manager.get_camera(cam_index)
    if cam is None or not getattr(cam, 'connected', False) or getattr(cam, 'thread', None) is None:
        try:
            camera_manager.camera_manager.connect_camera(cam_index)
        except Exception as e:
            status_messages.append(f"Camera {cam_index + 1} connect failed: {e}")
        cam = camera_manager.camera_manager.get_camera(cam_index)
        if cam is None or getattr(cam, 'thread', None) is None:
            status_messages.append(f"Camera {cam_index + 1} not available - cannot plate solve")
            return None

    # Sim frames need the real star catalogue rendered or there's nothing to solve.
    if sim_enabled and not config_state.sim_config.get('sim_use_real_stars', False):
        config_state.sim_config['sim_use_real_stars'] = True
        status_messages.append("Enabled sim REAL star rendering for alignment")

    try:
        solver = ps_mod.PlateSolver(config_state, cam_name)
        solver.ensure_loaded()
    except Exception as e:
        status_messages.append(f"Plate solver init failed: {e}")
        return None

    # Park the control loop so it doesn't fight the alignment's direct slews.
    try:
        from joystick_controller import TrackingMode
        joystick_state.tracking_mode = TrackingMode.STANDBY
    except Exception:
        pass

    eph = getattr(tracking_vis_state, 'ephemeris', None)
    return al.make_default_sample_fn(joystick_state, config_state, solver, ts, eph,
                                     cam_index, align_state=align_state)


def _retry_failed(config_state, align_state, joystick_state, tracking_vis_state, ts, status_messages):
    """Re-run the points that never solved, appending to the existing samples and refitting."""
    if align_state.phase in (al.RUNNING, al.BACKTEST):
        status_messages.append("Alignment already running")
        return
    pending = list(getattr(align_state, 'failed_points', []))
    if not pending:
        status_messages.append("No failed points to retry")
        return
    sampler = _prepare_alignment_run(config_state, align_state, joystick_state,
                                     tracking_vis_state, ts, status_messages)
    if sampler is None:
        return
    # Re-run only the failed points; keep existing samples/grid (no reset), then refit.
    align_state.failed_points = []
    align_state.phase = al.RUNNING
    align_state.error = None
    runner = al.AlignmentRunner(align_state, config_state, sampler,
                                pending_points=pending, append=True)
    align_state.runner = runner
    runner.start()
    status_messages.append(f"Retrying {len(pending)} failed point(s)")


# ==============================================================================
# PLATE-SOLVE CAMERA PANEL (right half of the alignment screen)
# ==============================================================================
# A single-camera live view with the same controls as the Sensor Calibration /
# Joystick-loop feeds (connect, gain, exposure, gamma, alignment rotation, ROI),
# so the operator can frame/focus and tune the plate-solve camera without leaving
# the alignment screen. Geometry lives in _alignment_cam_layout() so the renderer
# and the event handler never drift. The low-level draw helpers are reused from
# joystick_controller to keep the look/behaviour identical.

_ALIGN_CAM_SLIDER_W = 100
_ALIGN_CAM_ROT_W = 120


def _alignment_cam_index(config_state):
    """The camera alignment drives: the configured plate-solve camera (default cam1)."""
    return int(getattr(config_state, 'plate_solve_camera_index', 0) or 0)


def _alignment_cam_panel_rect(display):
    """Right-half panel rect, below the control header (mirrors the skymap on the left)."""
    y0 = display.sub_y + 20
    split_top = y0 + 200
    panel_bottom = display.sub_y + display.sub_height - 40
    mid = display.sub_x + display.sub_width // 2
    px = mid + 10
    return pygame.Rect(px, split_top, (display.sub_x + display.sub_width - 20) - px,
                       panel_bottom - split_top)


def _alignment_cam_layout(display):
    """Every draw/hit rect for the camera panel, shared by renderer and event handler."""
    P = _alignment_cam_panel_rect(display)
    sw = _ALIGN_CAM_SLIDER_W
    strip_y = P.y + 4
    gain_x = P.x + 90
    exp_x = gain_x + sw + 40
    gamma_x = exp_x + sw + 40
    img_top = strip_y + 32
    img_rect = pygame.Rect(P.x, img_top, P.width, P.bottom - img_top)

    # ROI size buttons: two columns of three at the top-right of the image.
    roi_rects = []
    roi_col1_x = img_rect.right - 70
    roi_col2_x = img_rect.right - 36
    roi_top = img_rect.y + 6
    for i in range(6):
        col_x = roi_col1_x if i < 3 else roi_col2_x
        roi_rects.append(pygame.Rect(col_x, roi_top + (i % 3) * 15, 30, 12))

    rot_x = img_rect.x + (img_rect.width - _ALIGN_CAM_ROT_W) // 2
    return {
        "panel": P,
        "image_rect": img_rect,
        # Connect and disconnect share a slot; only the relevant one is shown.
        "connect_rect": pygame.Rect(P.x + 4, strip_y, 74, 18),
        "disconnect_rect": pygame.Rect(P.x + 4, strip_y, 74, 18),
        "gain_track": pygame.Rect(gain_x, strip_y, sw, 18),
        "exposure_track": pygame.Rect(exp_x, strip_y, sw, 18),
        "gamma_track": pygame.Rect(gamma_x, strip_y, sw, 18),
        "gamma_toggle": pygame.Rect(gamma_x + sw + 6, strip_y, 34, 18),
        "save_rect": pygame.Rect(P.right - 76, strip_y, 72, 18),
        "roi_rects": roi_rects,
        "rotation_track": pygame.Rect(rot_x, img_rect.bottom - 22, _ALIGN_CAM_ROT_W, 14),
    }


def _draw_alignment_tables(display, config_state, align_state):
    """Right-half data view: the sample-accumulation table (what feeds the fit) plus a
    solution box with the fitted terms, RMS, and counts. Replaces the camera view when
    the View toggle is set to Tables."""
    screen = display.menu_screen
    P = _alignment_cam_panel_rect(display)
    pygame.draw.rect(screen, (14, 16, 24), P)
    pygame.draw.rect(screen, (90, 90, 110), P, 1)

    tf = display.tiny_font
    sf = display.small_font
    model = align_state.model

    def w180(d):
        return (d + 180.0) % 360.0 - 180.0

    # Column layout (x offsets from P.x). dAz'/dEl' are raw pointing error (arcmin);
    # rAz'/rEl' are post-fit residuals; N/RMSE are solve quality.
    cols = [("#", 22), ("azCmd", 52), ("elCmd", 46), ("azObs", 52), ("elObs", 46),
            ("dAz'", 46), ("dEl'", 46), ("rAz'", 46), ("rEl'", 46), ("N", 28), ("RMSE", 46)]
    cx = []
    x = P.x + 10
    for _, wgt in cols:
        cx.append(x)
        x += wgt
    screen.blit(sf.render("Sample accumulation (data feeding the fit)", True, (200, 210, 235)), (P.x + 10, P.y + 4))
    hy = P.y + 24
    for (name, _), xx in zip(cols, cx):
        screen.blit(tf.render(name, True, (150, 160, 185)), (xx, hy))
    pygame.draw.line(screen, (70, 70, 90), (P.x + 6, hy + 14), (P.right - 6, hy + 14), 1)

    # Build a unified row list: fit samples, then backtest, then failed points.
    rows = []
    for i, (s4, meta) in enumerate(zip(align_state.samples, align_state.sample_meta)):
        rows.append(("fit", i + 1, s4, meta))
    for i, s4 in enumerate(align_state.backtest_samples):
        rows.append(("bt", len(align_state.samples) + i + 1, s4, None))
    for i, (az, el) in enumerate(align_state.failed_points):
        rows.append(("fail", None, (az, el, None, None), None))

    row_h = 15
    rows_top = hy + 18
    box_h = 150  # reserved for the solution box at the bottom
    max_rows = max(1, (P.bottom - box_h - rows_top) // row_h)
    shown = rows[-max_rows:] if len(rows) > max_rows else rows
    if len(rows) > max_rows:
        screen.blit(tf.render(f"... showing last {len(shown)} of {len(rows)}", True, (150, 150, 160)),
                    (P.right - 150, hy))

    colors = {"fit": (200, 230, 200), "bt": (170, 200, 240), "fail": (235, 150, 130)}
    for r, (kind, num, s4, meta) in enumerate(shown):
        yy = rows_top + r * row_h
        col = colors[kind]
        az_cmd, el_cmd = s4[0], s4[1]
        if kind == "fail":
            screen.blit(tf.render("--", True, col), (cx[0], yy))
            screen.blit(tf.render(f"{az_cmd:.1f}", True, col), (cx[1], yy))
            screen.blit(tf.render(f"{el_cmd:.1f}", True, col), (cx[2], yy))
            screen.blit(tf.render("FAILED (no solve)", True, col), (cx[3], yy))
            continue
        az_obs, el_obs = s4[2], s4[3]
        d_az = w180(az_obs - az_cmd) * 60.0
        d_el = (el_obs - el_cmd) * 60.0
        vals = [("" if num is None else str(num)), f"{az_cmd:.1f}", f"{el_cmd:.1f}",
                f"{az_obs:.1f}", f"{el_obs:.1f}", f"{d_az:+.1f}", f"{d_el:+.1f}"]
        # Post-fit residuals.
        if model is not None:
            pr_az, pr_el = model.error(az_cmd, el_cmd)
            vals += [f"{w180(d_az / 60.0 - pr_az) * 60.0:+.1f}", f"{(d_el / 60.0 - pr_el) * 60.0:+.1f}"]
        else:
            vals += ["-", "-"]
        if meta is not None:
            vals += [str(meta.get("n_matches", "-")), f"{meta.get('rmse', float('nan')):.2f}"]
        else:
            vals += ["-", "-"]
        for v, xx in zip(vals, cx):
            screen.blit(tf.render(v, True, col), (xx, yy))

    # Solution box.
    by = P.bottom - box_h + 6
    pygame.draw.line(screen, (70, 70, 90), (P.x + 6, by - 4), (P.right - 6, by - 4), 1)
    screen.blit(sf.render("Solution", True, (200, 210, 235)), (P.x + 10, by))
    nfit = len(align_state.samples)
    nfail = len(align_state.failed_points)
    nbt = len(align_state.backtest_samples)
    screen.blit(sf.render(f"fit:{nfit}  backtest:{nbt}  failed:{nfail}", True, (200, 220, 200)),
                (P.x + 90, by))
    if model is not None and align_state.stats is not None:
        st = align_state.stats
        screen.blit(sf.render(
            f"RMS {st['rms_before_arcmin']:.2f}' -> {st['rms_after_arcmin']:.2f}'"
            + (f"   backtest {align_state.backtest_rms_deg * 60.0:.2f}'"
               if align_state.backtest_rms_deg is not None else ""),
            True, (200, 255, 200)), (P.x + 10, by + 22))
        for i, k in enumerate(TERM_NAMES):
            col = P.x + 10 + (i % 4) * 130
            row = by + 44 + (i // 4) * 20
            screen.blit(sf.render(f"{k}: {model.terms[k]:+.4f}°", True, (230, 230, 200)), (col, row))
    else:
        screen.blit(sf.render("(no fit yet)", True, (150, 150, 160)), (P.x + 10, by + 22))


def _draw_solved_centroids(screen, align_state, idx, img_rect):
    """Overlay the plate solver's matched-star centroids on the camera image.

    matched_centroids are in solved-frame pixel coords (row, col); scale into img_rect.
    Mirrors joystick_controller.draw_solve_centroids_on_feed but reads align_state."""
    ls = getattr(align_state, 'last_solve', None) if align_state is not None else None
    if not ls or ls.get('cam_index') != idx:
        return
    result = ls.get('result')
    src = ls.get('src_shape')
    cents = getattr(result, 'matched_centroids', None) if result is not None else None
    if src is None or cents is None or len(cents) == 0:
        return
    src_h, src_w = src[0], src[1]
    if src_h <= 0 or src_w <= 0:
        return
    sx = img_rect.width / float(src_w)
    sy = img_rect.height / float(src_h)
    for c in cents:
        px = int(img_rect.x + float(c[1]) * sx)
        py = int(img_rect.y + float(c[0]) * sy)
        pygame.draw.circle(screen, (0, 230, 230), (px, py), 6, 1)


def _draw_alignment_cam_panel(display, config_state, align_state=None):
    """Render the plate-solve camera's live frame + controls in the right-half panel."""
    from camera_manager import (camera_manager as cm, update_camera_frames_from_buffers,
                                roi_label_texts)
    from joystick_controller import (_process_feed_surface, _draw_feed_roi, _draw_jl_slider,
                                     JL_GAMMA_MIN, JL_GAMMA_MAX, JL_ROTATION_RANGE)

    screen = display.menu_screen
    font = display.small_font
    mouse = pygame.mouse.get_pos()
    lay = _alignment_cam_layout(display)
    P, img = lay["panel"], lay["image_rect"]
    idx = _alignment_cam_index(config_state)
    camera = cm.get_camera(idx)

    pygame.draw.rect(screen, (14, 16, 24), P)
    pygame.draw.rect(screen, (90, 90, 110), P, 1)
    pygame.draw.rect(screen, (0, 0, 0), img)

    update_camera_frames_from_buffers()
    connected = camera is not None and getattr(camera, 'connected', False)
    frame = getattr(camera, 'frame', None) if camera is not None else None

    if connected and frame is not None:
        try:
            screen.blit(_process_feed_surface(camera, frame, img.width, img.height), (img.x, img.y))
            _draw_feed_roi(screen, camera, img)
            _draw_solved_centroids(screen, align_state, idx, img)
            cx, cy = img.centerx, img.centery
            pygame.draw.line(screen, (255, 0, 0), (cx - 18, cy), (cx + 18, cy), 1)
            pygame.draw.line(screen, (255, 0, 0), (cx, cy - 18), (cx, cy + 18), 1)
        except Exception as e:
            screen.blit(font.render(f"Camera error: {e}", True, (255, 80, 80)), (img.x + 8, img.y + 26))
    else:
        msg = "Not Connected" if camera is not None else "No camera"
        t = display.font.render(msg, True, (255, 90, 90))
        screen.blit(t, t.get_rect(center=img.center))

    screen.blit(font.render(f"Camera {idx + 1} (plate-solve)", True, (220, 220, 235)),
                (img.x + 8, img.y + 6))

    # Last-solve caption (matches / FOV / RMSE) so the operator can gauge solve quality.
    ls = getattr(align_state, 'last_solve', None) if align_state is not None else None
    if ls and ls.get('cam_index') == idx:
        cap = f"solve: {ls.get('n_matches', 0)} stars  FOV {ls.get('fov', 0.0):.2f}°  RMSE {ls.get('rmse', float('nan')):.2f}"
        s = font.render(cap, True, (120, 235, 235))
        screen.blit(s, (img.x + 8, img.bottom - 18))

    # Connect / Disconnect (only one shown).
    if connected:
        r = lay["disconnect_rect"]
        pygame.draw.rect(screen, (255, 100, 100) if r.collidepoint(mouse) else (200, 70, 70), r)
        s = font.render("Disconnect", True, (255, 255, 255))
        screen.blit(s, s.get_rect(center=r.center))
    else:
        r = lay["connect_rect"]
        pygame.draw.rect(screen, (100, 100, 255) if r.collidepoint(mouse) else (70, 70, 200), r)
        s = font.render("Connect", True, (255, 255, 255))
        screen.blit(s, s.get_rect(center=r.center))
        return  # no further per-camera controls when disconnected

    # Gain slider.
    max_gain = camera.prop.get('MaxGain', 500) if camera.prop else 500
    gain_ratio = min(1.0, camera.gain / max_gain) if max_gain else 0.0
    _draw_jl_slider(screen, font, lay["gain_track"], gain_ratio,
                    f"Gain {camera.gain}", (100, 100, 100), (200, 0, 0))

    # Exposure slider (logarithmic).
    max_exp = camera.prop.get('MaxExposure', 500000) if camera.prop else 500000
    min_exp = 1
    if camera.exposure > 0 and max_exp > min_exp:
        exp_ratio = min(1.0, max(0.0, math.log10(camera.exposure / min_exp) / math.log10(max_exp / min_exp)))
    else:
        exp_ratio = 0.0
    e = camera.exposure
    exp_val = f"{e:g}us" if e < 1000 else (f"{e / 1000.0:.1f}ms" if e < 1000000 else f"{e / 1000000.0:.1f}s")
    _draw_jl_slider(screen, font, lay["exposure_track"], exp_ratio,
                    f"Exp {exp_val}", (100, 100, 100), (0, 200, 0))

    # Gamma slider + toggle.
    gamma_ratio = (camera.gamma - JL_GAMMA_MIN) / (JL_GAMMA_MAX - JL_GAMMA_MIN)
    _draw_jl_slider(screen, font, lay["gamma_track"], gamma_ratio,
                    f"Gamma {camera.gamma:.2f}", (150, 150, 150), (200, 200, 255))
    tg = lay["gamma_toggle"]
    tcol = (0, 150, 0) if camera.gamma_enabled else (150, 0, 0)
    if tg.collidepoint(mouse):
        tcol = tuple(min(255, c + 50) for c in tcol)
    pygame.draw.rect(screen, tcol, tg)
    s = font.render("ON" if camera.gamma_enabled else "OFF", True, (255, 255, 255))
    screen.blit(s, s.get_rect(center=tg.center))

    # Save (persist this camera's settings to config.json).
    sv = lay["save_rect"]
    pygame.draw.rect(screen, (70, 100, 70) if sv.collidepoint(mouse) else (50, 70, 50), sv)
    pygame.draw.rect(screen, (150, 150, 150), sv, 1)
    s = font.render("Save", True, (255, 255, 255))
    screen.blit(s, s.get_rect(center=sv.center))

    # ROI size buttons.
    for i, rr in enumerate(lay["roi_rects"]):
        if camera.roi_size == i:
            pygame.draw.rect(screen, (0, 100, 0), rr)
            pygame.draw.rect(screen, (0, 255, 0), rr, 1)
            tc = (255, 255, 255)
        elif rr.collidepoint(mouse):
            pygame.draw.rect(screen, (100, 100, 100), rr)
            pygame.draw.rect(screen, (200, 200, 200), rr, 1)
            tc = (255, 255, 0)
        else:
            pygame.draw.rect(screen, (70, 70, 70), rr)
            pygame.draw.rect(screen, (150, 150, 150), rr, 1)
            tc = (255, 255, 255)
        s = display.tiny_font.render(roi_label_texts[i], True, tc)
        screen.blit(s, s.get_rect(center=rr.center))

    # Alignment rotation slider (bottom-center) with a 0-deg center marker.
    rot = lay["rotation_track"]
    rot_ratio = (camera.alignment_rotation + JL_ROTATION_RANGE) / (2 * JL_ROTATION_RANGE)
    _draw_jl_slider(screen, font, rot, rot_ratio,
                    f"Rot {camera.alignment_rotation:+.1f}", (50, 50, 150), (150, 150, 255))
    cmx = rot.x + rot.width // 2
    pygame.draw.line(screen, (255, 255, 255), (cmx, rot.centery - 5), (cmx, rot.centery + 5), 1)


def _save_alignment_cam_config(config_state, idx, camera, cb):
    """Persist this camera's gain/exposure/gamma/rotation to config.json."""
    if not config_state:
        cb("Error: config not available")
        return
    cc = config_state.camera_configs.setdefault(f"camera{idx + 1}", {})
    cc["gain"] = camera.gain
    cc["exposure"] = camera.exposure
    cc["gamma"] = camera.gamma
    cc["gamma_enabled"] = camera.gamma_enabled
    cc["alignment_rotation"] = camera.alignment_rotation
    config_state.save_to_file()
    cb(f"Camera {idx + 1} settings saved to config.json")


def _apply_alignment_cam_drag(pos, display, config_state, cb=None):
    """Apply slider drags (gain / exposure / gamma / rotation) for the camera panel."""
    from camera_manager import camera_manager as cm
    from joystick_controller import JL_GAMMA_MIN, JL_GAMMA_MAX, JL_ROTATION_RANGE
    idx = _alignment_cam_index(config_state)
    camera = cm.get_camera(idx)
    if camera is None or not camera.connected:
        return
    lay = _alignment_cam_layout(display)

    t = lay["gain_track"]
    if t.collidepoint(pos):
        max_gain = camera.prop.get('MaxGain', 500) if camera.prop else 500
        rel = min(max(pos[0] - t.x, 0), t.width)
        cm.set_camera_gain(idx, int((rel / t.width) * max_gain), cb)

    t = lay["exposure_track"]
    if t.collidepoint(pos):
        max_exp = camera.prop.get('MaxExposure', 500000) if camera.prop else 500000
        min_exp = 1
        rel = min(max(pos[0] - t.x, 0), t.width)
        slider_pos = rel / t.width
        if max_exp > min_exp:
            new_exp = int(min_exp * (10 ** (slider_pos * math.log10(max_exp / min_exp))))
            new_exp = max(min_exp, min(new_exp, max_exp))
        else:
            new_exp = max_exp
        cm.set_camera_exposure(idx, new_exp, cb)

    t = lay["gamma_track"]
    if t.collidepoint(pos):
        rel = min(max(pos[0] - t.x, 0), t.width)
        new_gamma = JL_GAMMA_MIN + (rel / t.width) * (JL_GAMMA_MAX - JL_GAMMA_MIN)
        camera.gamma = round(max(JL_GAMMA_MIN, min(JL_GAMMA_MAX, new_gamma)), 2)

    t = lay["rotation_track"]
    if t.collidepoint(pos):
        rel = min(max(pos[0] - t.x, 0), t.width)
        new_rot = ((rel / t.width) - 0.5) * (2 * JL_ROTATION_RANGE)
        camera.alignment_rotation = max(-JL_ROTATION_RANGE, min(JL_ROTATION_RANGE, new_rot))


def handle_alignment_camera_events(event, display, config_state, update_status_cb=None):
    """Handle mouse events for the alignment-screen camera panel.

    Returns True if the event was consumed by a camera control."""
    from camera_manager import camera_manager as cm
    if update_status_cb is None:
        update_status_cb = print

    if event.type == pygame.MOUSEMOTION:
        if event.buttons[0]:
            _apply_alignment_cam_drag(event.pos, display, config_state, update_status_cb)
        return False

    if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
        return False

    pos = event.pos
    idx = _alignment_cam_index(config_state)
    camera = cm.get_camera(idx)
    if camera is None:
        return False
    lay = _alignment_cam_layout(display)

    # Connect / Disconnect.
    if not camera.connected:
        if lay["connect_rect"].collidepoint(pos):
            cm.connect_camera(idx, update_status_cb)
            return True
        return False
    if lay["disconnect_rect"].collidepoint(pos):
        cm.disconnect_camera(idx, update_status_cb)
        return True

    if lay["gamma_toggle"].collidepoint(pos):
        camera.gamma_enabled = not camera.gamma_enabled
        return True

    if lay["save_rect"].collidepoint(pos):
        _save_alignment_cam_config(config_state, idx, camera, update_status_cb)
        return True

    # Slider clicks set the value immediately (then drag continues to adjust).
    if (lay["gain_track"].collidepoint(pos) or lay["exposure_track"].collidepoint(pos) or
            lay["gamma_track"].collidepoint(pos) or lay["rotation_track"].collidepoint(pos)):
        _apply_alignment_cam_drag(pos, display, config_state, update_status_cb)
        return True

    # ROI size buttons.
    for i, rr in enumerate(lay["roi_rects"]):
        if rr.collidepoint(pos):
            if camera.roi_size == i:
                camera.roi_size = -1
                camera.roi_x = 0.5
                camera.roi_y = 0.5
                cm.set_camera_roi(idx, update_status_cb, None, None, -1)
            else:
                was_first = (camera.roi_size == -1)
                camera.roi_size = i
                if was_first:
                    camera.roi_x = 0.5
                    camera.roi_y = 0.5
                cm.set_camera_roi(idx, update_status_cb, camera.roi_x, camera.roi_y, i)
            return True

    # ROI origin by clicking on the image (skip if over any control).
    img = lay["image_rect"]
    if img.collidepoint(pos) and camera.frame is not None:
        controls = [lay["connect_rect"], lay["disconnect_rect"], lay["gain_track"],
                    lay["exposure_track"], lay["gamma_track"], lay["gamma_toggle"],
                    lay["save_rect"], lay["rotation_track"]] + lay["roi_rects"]
        if not any(r.collidepoint(pos) for r in controls):
            camera.roi_x = max(0.0, min(1.0, (pos[0] - img.x) / img.width))
            camera.roi_y = max(0.0, min(1.0, (pos[1] - img.y) / img.height))
            if camera.roi_size >= 0:
                cm.set_camera_roi(idx, update_status_cb, camera.roi_x, camera.roi_y, camera.roi_size)
            return True

    return False
