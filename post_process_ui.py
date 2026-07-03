"""Pygame UI for the post-processing / replay screen.

Two views share the right-hand sub-pane (everything is offset by
``display.sub_x`` / ``display.sub_y``):

* **Library** - browse runs grouped by target, manage them (rename / favorite /
  tag / notes / delete), and load one for replay.
* **Replay** - side-by-side synced camera panes with a scrub slider + transport,
  per-pane gamma / brightness / contrast, stabilize & overlay toggles, a simple
  annotation tool, and MP4 export (clip / stabilized movie / whole run).

Follows the existing screen pattern (``hw_sim_ui`` / ``alignment_ui``):
``draw_post_process`` renders and stashes hit-test rects on the state object;
``handle_pp_*`` functions are dispatched from main.py's event loop.

All heavy lifting (decode / process / stabilize / export) lives in
``post_process.py``; this module is presentation + input only.
"""

import os
import time

import pygame

from utils import draw_button
from post_process import (
    RunLibrary, ReplayEngine, Mp4Exporter, StackExporter, fmt_utc,
    compute_track_vectors,
)

# Colors
_BG = (28, 28, 32)
_PANEL = (44, 44, 52)
_PANEL_HI = (60, 60, 72)
_TEXT = (230, 230, 235)
_MUTED = (160, 160, 170)
_ACCENT = (90, 170, 255)
_GOLD = (255, 205, 70)
_GREEN = (90, 210, 110)
_RED = (230, 90, 90)
_TRACK = (70, 70, 84)
_KNOB = (200, 200, 210)

# Image-adjustment slider ranges (lo, hi, default)
_RANGES = {
    "gamma": (0.2, 5.0, 1.0),
    "brightness": (-128.0, 128.0, 0.0),
    "contrast": (0.2, 3.0, 1.0),
}
_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
_EXPORT_DIR = "exports"


class PostProcessState:
    """All mutable state for the post-processing screen."""

    def __init__(self, data_dir="data"):
        self.library = RunLibrary(data_dir)
        self.library.scan()
        self.engine = None
        self.run = None
        self.view = "library"

        # library
        self.favorites_only = False
        self.text_filter = ""
        self.scroll = 0
        self.collapsed = set()
        self.selected_run = None
        self.confirm_delete = False

        # text editing: focus is one of None|'filter'|'rename'|'tag'|'notes'|'annot'
        self.focus = None
        self.text_buffer = ""

        # replay interaction
        self.dragging = None          # 'timeline' | ('img', cam, kind)
        self.active_cam = 0

        # annotation: mode None|'text'|'arrow'|'box'; draft holds an in-progress shape
        self.annot_mode = None
        self.annot_draft = None       # dict with cam + normalized coords
        self.annot_pending_pos = None # (cam, nx, ny) awaiting typed text

        # export
        self.exporter = None
        self.export_msg = ""

        # hit-test rects, rebuilt every frame
        self.rects = {}
        self.group_rows = []
        self.run_rows = []
        self.pane_rects = {}          # cam -> screen Rect

    # ----------------------------------------------------------- lifecycle
    def load_run(self, run):
        self.stop_engine()
        self.run = run
        self.selected_run = run
        self.engine = ReplayEngine(run)
        self.engine.start()
        self.view = "replay"
        self.focus = None
        self.annot_mode = None
        self.annot_draft = None

    def stop_engine(self):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None

    def shutdown(self):
        self.stop_engine()

    def rescan(self):
        self.library.scan()


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def _button(display, rect, text, *, hover=False, active=False, color=None):
    state = {"hover": hover, "clicked": active}
    if color is not None:
        pygame.draw.rect(display.menu_screen, color, rect)
        pygame.draw.rect(display.menu_screen, (20, 20, 24), rect, 1)
        ts = display.font.render(text, True, (15, 15, 18))
        display.menu_screen.blit(ts, ts.get_rect(center=rect.center))
    else:
        draw_button(display.menu_screen, rect, text, state)


def _text(display, text, x, y, font=None, color=_TEXT):
    font = font or display.font
    display.menu_screen.blit(font.render(str(text), True, color), (x, y))


def _field(display, rect, text, focused):
    pygame.draw.rect(display.menu_screen, (245, 245, 248), rect)
    pygame.draw.rect(display.menu_screen, _ACCENT if focused else (90, 90, 100), rect, 2)
    shown = text + ("|" if focused else "")
    ts = display.font.render(shown, True, (15, 15, 18))
    prev = display.menu_screen.get_clip()
    display.menu_screen.set_clip(rect)
    display.menu_screen.blit(ts, (rect.x + 5, rect.y + (rect.height - ts.get_height()) // 2))
    display.menu_screen.set_clip(prev)


def _slider(display, rect, frac, knob_w=10):
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(display.menu_screen, _TRACK, rect, border_radius=3)
    kx = rect.x + int(frac * (rect.width - knob_w))
    knob = pygame.Rect(kx, rect.y - 3, knob_w, rect.height + 6)
    pygame.draw.rect(display.menu_screen, _KNOB, knob, border_radius=3)
    return knob


def _val_to_frac(v, lo, hi):
    return 0.0 if hi <= lo else (v - lo) / (hi - lo)


def _frac_to_val(f, lo, hi):
    return lo + max(0.0, min(1.0, f)) * (hi - lo)


# ---------------------------------------------------------------------------
# LIBRARY VIEW
# ---------------------------------------------------------------------------
def _draw_library(display, state):
    sx, sy = display.sub_x, display.sub_y
    sw, sh = display.sub_width, display.sub_height
    screen = display.menu_screen
    screen.fill(_BG, (sx, sy, sw, sh))
    state.rects = {}
    state.group_rows = []
    state.run_rows = []

    # Header
    _text(display, "Post Processing  -  Run Library", sx + 14, sy + 12, display.large_font)
    rescan = pygame.Rect(sx + sw - 120, sy + 12, 100, 30)
    _button(display, rescan, "Rescan", hover=rescan.collidepoint(pygame.mouse.get_pos()))
    state.rects["rescan"] = rescan

    fav = pygame.Rect(sx + sw - 330, sy + 12, 190, 30)
    _button(display, fav, ("Favorites: ON" if state.favorites_only else "Favorites: off"),
            active=state.favorites_only)
    state.rects["fav_filter"] = fav

    # Filter field
    _text(display, "Filter:", sx + 14, sy + 54, display.font, _MUTED)
    filt = pygame.Rect(sx + 80, sy + 50, 320, 28)
    _field(display, filt, state.text_filter, state.focus == "filter")
    state.rects["filter"] = filt

    # Split: list (left) / details (right)
    list_x = sx + 14
    list_y = sy + 92
    list_w = int(sw * 0.52)
    list_h = sh - (list_y - sy) - 14
    list_rect = pygame.Rect(list_x, list_y, list_w, list_h)
    pygame.draw.rect(screen, _PANEL, list_rect)
    state.rects["list"] = list_rect

    prev_clip = screen.get_clip()
    screen.set_clip(list_rect)
    row_h = 26
    y = list_y + 6 - state.scroll
    groups = state.library.groups(state.favorites_only, state.text_filter)
    for key, runs in groups:
        collapsed = key in state.collapsed
        ghdr = pygame.Rect(list_x + 4, y, list_w - 8, row_h)
        if y + row_h > list_y and y < list_y + list_h:
            pygame.draw.rect(screen, _PANEL_HI, ghdr)
            arrow = "+" if collapsed else "-"
            _text(display, f"[{arrow}] {key}   ({len(runs)})", ghdr.x + 6, ghdr.y + 4,
                  display.font, _GOLD)
        state.group_rows.append((ghdr, key))
        y += row_h
        if collapsed:
            continue
        for run in runs:
            rr = pygame.Rect(list_x + 18, y, list_w - 24, row_h)
            if y + row_h > list_y and y < list_y + list_h:
                if run is state.selected_run:
                    pygame.draw.rect(screen, (70, 90, 120), rr)
                star = "*" if run.sidecar["favorite"] else " "
                dur = run.duration
                label = (f"{star} {run.display_name}  |  {dur:.1f}s  |  "
                         f"C1:{run.frame_count(0)} C2:{run.frame_count(1)}")
                _text(display, label, rr.x + 4, rr.y + 4, display.small_font)
            state.run_rows.append((rr, run))
            y += row_h
    content_h = y - (list_y + 6) + state.scroll
    state._list_content_h = content_h
    state._list_view_h = list_h
    screen.set_clip(prev_clip)

    # Details panel
    dx = list_x + list_w + 16
    dy = list_y
    dw = sx + sw - dx - 14
    drect = pygame.Rect(dx, dy, dw, list_h)
    pygame.draw.rect(screen, _PANEL, drect)
    _draw_details(display, state, drect)


def _draw_details(display, state, drect):
    run = state.selected_run
    x, y = drect.x + 12, drect.y + 12
    if run is None:
        _text(display, "Select a run to inspect / manage.", x, y, display.font, _MUTED)
        return
    sc = run.sidecar
    _text(display, run.display_name, x, y, display.large_font)
    y += 36
    _text(display, f"Folder: {run.folder}", x, y, display.small_font, _MUTED); y += 20
    _text(display, f"Target: {run.target}   NORAD: {run.norad or '-'}", x, y, display.small_font, _MUTED); y += 20
    _text(display, f"Duration: {run.duration:.1f}s   Cam1: {run.frame_count(0)}   Cam2: {run.frame_count(1)}",
          x, y, display.small_font, _MUTED); y += 20
    _text(display, f"Trajectory: {'yes' if run.trajectory.valid else 'no'}", x, y, display.small_font, _MUTED); y += 28

    # Load & replay
    load = pygame.Rect(x, y, 200, 38)
    _button(display, load, "Load & Replay", color=_GREEN); state.rects["load"] = load
    favb = pygame.Rect(x + 212, y, 160, 38)
    _button(display, favb, ("* Favorited" if sc["favorite"] else "Favorite"), active=sc["favorite"])
    state.rects["detail_fav"] = favb
    y += 52

    # Rename
    _text(display, "Display name:", x, y, display.small_font, _MUTED); y += 20
    rn = pygame.Rect(x, y, drect.width - 130, 28)
    _field(display, rn, state.text_buffer if state.focus == "rename" else sc["display_name"],
           state.focus == "rename")
    state.rects["rename"] = rn
    rnb = pygame.Rect(rn.right + 8, y, 100, 28)
    _button(display, rnb, "Set"); state.rects["rename_set"] = rnb
    y += 40

    # Tags
    _text(display, "Tags: " + (", ".join(sc["tags"]) or "(none)"), x, y, display.small_font, _MUTED); y += 22
    tg = pygame.Rect(x, y, drect.width - 130, 28)
    _field(display, tg, state.text_buffer if state.focus == "tag" else "", state.focus == "tag")
    state.rects["tag"] = tg
    tgb = pygame.Rect(tg.right + 8, y, 100, 28)
    _button(display, tgb, "Add Tag"); state.rects["tag_add"] = tgb
    y += 40

    # Notes
    _text(display, "Notes:", x, y, display.small_font, _MUTED); y += 20
    nt = pygame.Rect(x, y, drect.width - 24, 60)
    _field(display, nt, state.text_buffer if state.focus == "notes" else sc["notes"],
           state.focus == "notes")
    state.rects["notes"] = nt
    y += 72

    # Delete (with confirm)
    if state.confirm_delete:
        _text(display, "Delete this run folder permanently?", x, y, display.font, _RED); y += 26
        yes = pygame.Rect(x, y, 120, 34); no = pygame.Rect(x + 132, y, 120, 34)
        _button(display, yes, "Yes, delete", color=_RED)
        _button(display, no, "Cancel")
        state.rects["delete_yes"] = yes
        state.rects["delete_no"] = no
    else:
        db = pygame.Rect(x, y, 200, 34)
        _button(display, db, "Delete Run...", color=(170, 70, 70))
        state.rects["delete"] = db


# ---------------------------------------------------------------------------
# REPLAY VIEW
# ---------------------------------------------------------------------------
def _fit_box(max_w, max_h, aspect):
    """Largest (w, h) fitting in (max_w, max_h) preserving ``aspect`` (w/h)."""
    if aspect <= 0:
        return max_w, max_h
    w = max_w
    h = int(w / aspect)
    if h > max_h:
        h = max_h
        w = int(h * aspect)
    return max(1, w), max(1, h)


def _draw_replay(display, state):
    sx, sy = display.sub_x, display.sub_y
    sw, sh = display.sub_width, display.sub_height
    screen = display.menu_screen
    screen.fill(_BG, (sx, sy, sw, sh))
    state.rects = {}
    state.pane_rects = {}
    eng = state.engine
    run = state.run

    # Header / back
    back = pygame.Rect(sx + 10, sy + 8, 90, 28)
    _button(display, back, "< Runs"); state.rects["back"] = back
    _text(display, run.display_name, sx + 112, sy + 12, display.font, _ACCENT)

    ctrl_w = 300
    ctrl_x = sx + sw - ctrl_w - 10
    panes_x = sx + 12
    panes_y = sy + 46
    panes_w = ctrl_x - panes_x - 12
    panes_h = sh - (panes_y - sy) - 150

    # Two camera panes
    cams = run.camera_indices[:2] or [0]
    gap = 10
    half_w = (panes_w - gap) // 2 if len(cams) > 1 else panes_w
    for i, cam in enumerate(cams):
        box_x = panes_x + i * (half_w + gap)
        shape = run.frame_shape(cam)
        aspect = (shape[0] / shape[1]) if shape else (16 / 9)
        pw, ph = _fit_box(half_w, panes_h, aspect)
        pane = pygame.Rect(box_x, panes_y, pw, ph)
        pygame.draw.rect(screen, (0, 0, 0), pane)
        eng.set_pane_size(cam, pw, ph)
        surf = eng.get_surface(cam)
        if surf is not None:
            screen.blit(surf, pane.topleft)
        else:
            _text(display, "decoding...", pane.x + 8, pane.y + 8, display.small_font, _MUTED)
        # Active-pane highlight
        col = _ACCENT if cam == state.active_cam else (80, 80, 90)
        pygame.draw.rect(screen, col, pane, 2)
        _text(display, f"Cam{cam + 1}", pane.x + 6, pane.bottom - 22, display.small_font, _GOLD)
        state.pane_rects[cam] = pane

    _draw_annotation_preview(display, state)

    # Bottom: timeline + transport
    _draw_transport(display, state, sx, sy, sw, sh, panes_y + panes_h + 10)

    # Right control column
    _draw_controls(display, state, ctrl_x, panes_y, ctrl_w)


def _draw_annotation_preview(display, state):
    draft = state.annot_draft
    if not draft:
        return
    pane = state.pane_rects.get(draft["cam"])
    if not pane:
        return
    screen = display.menu_screen
    if draft["type"] == "box":
        r = pygame.Rect(
            pane.x + int(min(draft["x0"], draft["x1"]) * pane.width),
            pane.y + int(min(draft["y0"], draft["y1"]) * pane.height),
            int(abs(draft["x1"] - draft["x0"]) * pane.width),
            int(abs(draft["y1"] - draft["y0"]) * pane.height))
        pygame.draw.rect(screen, _RED, r, 2)
    elif draft["type"] == "arrow":
        p0 = (pane.x + int(draft["x0"] * pane.width), pane.y + int(draft["y0"] * pane.height))
        p1 = (pane.x + int(draft["x1"] * pane.width), pane.y + int(draft["y1"] * pane.height))
        pygame.draw.line(screen, _RED, p0, p1, 2)


def _draw_transport(display, state, sx, sy, sw, sh, y):
    eng = state.engine
    screen = display.menu_screen
    bar_x = sx + 14
    bar_w = sw - 28 - 0
    # Timeline slider
    tl = pygame.Rect(bar_x, y, bar_w, 16)
    pygame.draw.rect(screen, _TRACK, tl, border_radius=4)
    # in/out markers
    span = eng.t1 - eng.t0
    if span > 0:
        for mk, col in ((eng.in_marker, _GREEN), (eng.out_marker, _RED)):
            mx = bar_x + int((mk - eng.t0) / span * bar_w)
            pygame.draw.line(screen, col, (mx, y - 4), (mx, y + 20), 2)
    kx = bar_x + int(eng.fraction * (bar_w - 10))
    pygame.draw.rect(screen, _KNOB, pygame.Rect(kx, y - 4, 10, 24), border_radius=3)
    state.rects["timeline"] = tl

    # Transport buttons row
    ty = y + 32
    defs = [
        ("step_back", "|<"), ("playpause", "Pause" if eng.playing else "Play"),
        ("step_fwd", ">|"), ("speed", f"x{eng.speed:g}"),
        ("loop", f"Loop:{'on' if eng.loop else 'off'}"),
        ("set_in", "Set In"), ("set_out", "Set Out"),
    ]
    bx = bar_x
    for key, label in defs:
        w = 80 if key in ("playpause", "loop") else 70
        r = pygame.Rect(bx, ty, w, 30)
        _button(display, r, label, active=(key == "loop" and eng.loop))
        state.rects[key] = r
        bx += w + 8

    # Time / range readout (timestamps are stored/displayed in UTC)
    _text(display, f"{fmt_utc(eng.t)} UTC   ({eng.fraction * 100:4.1f}%)",
          bx + 10, ty + 6, display.font, _MUTED)
    in_f = (eng.in_marker - eng.t0)
    out_f = (eng.out_marker - eng.t0)
    _text(display, f"Clip: {in_f:.1f}s -> {out_f:.1f}s", bar_x, ty + 38, display.small_font, _MUTED)


def _draw_controls(display, state, x, y, w):
    eng = state.engine
    screen = display.menu_screen
    panel = pygame.Rect(x, y, w, display.sub_height - (y - display.sub_y) - 14)
    pygame.draw.rect(screen, _PANEL, panel)
    cx = x + 12
    cy = y + 10

    # Active pane selector
    _text(display, "Image adjust - active pane:", cx, cy, display.small_font, _MUTED); cy += 22
    for i, cam in enumerate(state.run.camera_indices[:2]):
        r = pygame.Rect(cx + i * 90, cy, 84, 28)
        _button(display, r, f"Cam{cam + 1}", active=(cam == state.active_cam))
        state.rects[f"sel_cam{cam}"] = r
    cy += 40

    cs = eng.cams.get(state.active_cam)
    if cs:
        for kind, label, cur in (
            ("gamma", "Gamma (faint)", cs.gamma),
            ("brightness", "Brightness", cs.brightness),
            ("contrast", "Contrast", cs.contrast),
        ):
            lo, hi, _ = _RANGES[kind]
            _text(display, f"{label}: {cur:.2f}", cx, cy, display.small_font); cy += 18
            sr = pygame.Rect(cx, cy, w - 24, 12)
            _slider(display, sr, _val_to_frac(cur, lo, hi))
            state.rects[("img", kind)] = sr
            cy += 26
        rb = pygame.Rect(cx, cy, w - 24, 26)
        _button(display, rb, "Reset adjust"); state.rects["reset_adjust"] = rb
        cy += 38

    # Toggles
    pygame.draw.line(screen, (80, 80, 90), (cx, cy), (x + w - 12, cy)); cy += 8
    for key, label, on in (
        ("stab", "Stabilize", eng.stabilize_on),
        ("overlays", "Overlays (vectors/meta)", eng.overlays_on),
    ):
        r = pygame.Rect(cx, cy, w - 24, 28)
        _button(display, r, f"{label}: {'ON' if on else 'off'}", active=on)
        state.rects[key] = r
        cy += 34
    ref = pygame.Rect(cx, cy, w - 24, 26)
    _button(display, ref, "Set stab reference = here"); state.rects["set_ref"] = ref
    cy += 38

    # Annotation tools
    pygame.draw.line(screen, (80, 80, 90), (cx, cy), (x + w - 12, cy)); cy += 8
    _text(display, "Annotate (active pane):", cx, cy, display.small_font, _MUTED); cy += 22
    for i, (key, label) in enumerate((("text", "Text"), ("arrow", "Arrow"), ("box", "Box"))):
        r = pygame.Rect(cx + i * 90, cy, 84, 28)
        _button(display, r, label, active=(state.annot_mode == key))
        state.rects[f"annot_{key}"] = r
    cy += 34
    clr = pygame.Rect(cx, cy, w - 24, 26)
    _button(display, clr, "Clear annotations"); state.rects["annot_clear"] = clr
    cy += 30
    if state.annot_pending_pos is not None:
        _text(display, "Type label, Enter:", cx, cy, display.small_font, _GOLD); cy += 18
        af = pygame.Rect(cx, cy, w - 24, 26)
        _field(display, af, state.text_buffer, state.focus == "annot")
        cy += 32

    # Export
    pygame.draw.line(screen, (80, 80, 90), (cx, cy), (x + w - 12, cy)); cy += 8
    _text(display, "Export MP4:", cx, cy, display.small_font, _MUTED); cy += 22
    for key, label in (("exp_clip", "Save clip (In->Out)"),
                       ("exp_stab", "Save stabilized movie"),
                       ("exp_run", "Save whole run")):
        r = pygame.Rect(cx, cy, w - 24, 28)
        _button(display, r, label)
        state.rects[key] = r
        cy += 34

    # Stack (lucky imaging: cull sharpest frames -> align -> average to a master)
    pygame.draw.line(screen, (80, 80, 90), (cx, cy), (x + w - 12, cy)); cy += 8
    _text(display, "Stack best frames (In->Out):", cx, cy, display.small_font, _MUTED)
    cy += 22
    for key, label in (("stack_25", "Stack best 25%"),
                       ("stack_50", "Stack best 50%"),
                       ("stack_50_local", "Stack 50% (align points)")):
        r = pygame.Rect(cx, cy, w - 24, 28)
        _button(display, r, label)
        state.rects[key] = r
        cy += 34
    if state.exporter is not None:
        msg = (f"Exporting... {state.exporter.progress * 100:.0f}%"
               if not state.exporter.done else state.export_msg)
        _text(display, msg, cx, cy, display.small_font, _GREEN); cy += 18
    elif state.export_msg:
        _text(display, state.export_msg, cx, cy, display.small_font, _GREEN)


# ---------------------------------------------------------------------------
# Public entry points (called from main.py)
# ---------------------------------------------------------------------------
def draw_post_process(display, state):
    # advance playback clock
    if state.engine is not None:
        now = time.perf_counter()
        last = getattr(state, "_last_tick", now)
        state.engine.update(now - last)
        state._last_tick = now
    # poll exporter
    if state.exporter is not None and state.exporter.done:
        if state.exporter.error:
            state.export_msg = f"Export failed: {state.exporter.error}"
        else:
            state.export_msg = f"Saved {state.exporter.frames_written} frames -> {state.exporter.out_path}"
        state.exporter = None

    if state.view == "replay" and state.run is not None:
        _draw_replay(display, state)
    else:
        _draw_library(display, state)


def handle_pp_click(pos, display, state, status_messages):
    if state.view == "replay":
        _click_replay(pos, display, state, status_messages)
    else:
        _click_library(pos, display, state, status_messages)


def _click_library(pos, display, state, status_messages):
    R = state.rects
    if R.get("rescan") and R["rescan"].collidepoint(pos):
        state.rescan(); status_messages.append("Post: library rescanned"); return
    if R.get("fav_filter") and R["fav_filter"].collidepoint(pos):
        state.favorites_only = not state.favorites_only; state.scroll = 0; return
    if R.get("filter") and R["filter"].collidepoint(pos):
        state.focus = "filter"; state.text_buffer = state.text_filter; return

    run = state.selected_run
    if run is not None:
        if R.get("load") and R["load"].collidepoint(pos):
            state.load_run(run); status_messages.append(f"Post: replaying {run.display_name}"); return
        if R.get("detail_fav") and R["detail_fav"].collidepoint(pos):
            run.toggle_favorite(); return
        if R.get("rename") and R["rename"].collidepoint(pos):
            state.focus = "rename"; state.text_buffer = run.sidecar["display_name"]; return
        if R.get("rename_set") and R["rename_set"].collidepoint(pos):
            run.set_display_name(state.text_buffer); state.focus = None; return
        if R.get("tag") and R["tag"].collidepoint(pos):
            state.focus = "tag"; state.text_buffer = ""; return
        if R.get("tag_add") and R["tag_add"].collidepoint(pos):
            run.add_tag(state.text_buffer); state.focus = None; state.text_buffer = ""; return
        if R.get("notes") and R["notes"].collidepoint(pos):
            state.focus = "notes"; state.text_buffer = run.sidecar["notes"]; return
        if R.get("delete") and R["delete"].collidepoint(pos):
            state.confirm_delete = True; return
        if R.get("delete_no") and R["delete_no"].collidepoint(pos):
            state.confirm_delete = False; return
        if R.get("delete_yes") and R["delete_yes"].collidepoint(pos):
            try:
                run.delete()
                status_messages.append(f"Post: deleted {run.folder}")
            except OSError as e:
                status_messages.append(f"Post: delete failed: {e}")
            state.confirm_delete = False
            state.selected_run = None
            state.rescan()
            return

    # group headers (collapse/expand)
    for rect, key in state.group_rows:
        if rect.collidepoint(pos):
            state.collapsed.symmetric_difference_update({key})
            return
    # run rows (select)
    for rect, r in state.run_rows:
        if rect.collidepoint(pos):
            state.selected_run = r
            state.confirm_delete = False
            state.focus = None
            return
    state.focus = None


def _click_replay(pos, display, state, status_messages):
    R = state.rects
    eng = state.engine
    if R.get("back") and R["back"].collidepoint(pos):
        state.stop_engine(); state.run = None; state.view = "library"; state.rescan(); return

    # Image-adjust sliders (start drag)
    for kind in ("gamma", "brightness", "contrast"):
        sr = R.get(("img", kind))
        if sr and sr.collidepoint(pos):
            state.dragging = ("img", kind)
            _apply_img_slider(state, kind, pos[0], sr)
            return
    if R.get("timeline") and R["timeline"].collidepoint(pos):
        state.dragging = "timeline"
        _apply_timeline(state, pos[0], R["timeline"])
        return

    # Transport
    if _hit(R, "step_back", pos): eng.step_frames(-1); return
    if _hit(R, "step_fwd", pos): eng.step_frames(1); return
    if _hit(R, "playpause", pos): eng.playing = not eng.playing; return
    if _hit(R, "loop", pos): eng.loop = not eng.loop; return
    if _hit(R, "set_in", pos): eng.set_in_marker(); return
    if _hit(R, "set_out", pos): eng.set_out_marker(); return
    if _hit(R, "speed", pos):
        i = (_SPEEDS.index(eng.speed) + 1) % len(_SPEEDS) if eng.speed in _SPEEDS else 2
        eng.speed = _SPEEDS[i]; return

    # Pane select
    for cam, pane in state.pane_rects.items():
        if pane.collidepoint(pos):
            if state.annot_mode:
                _start_annotation(state, cam, pos, pane)
            else:
                state.active_cam = cam
            return

    # Active-cam selector + adjust reset
    for cam in state.run.camera_indices[:2]:
        if _hit(R, f"sel_cam{cam}", pos):
            state.active_cam = cam; return
    if _hit(R, "reset_adjust", pos):
        eng.set_image_params(state.active_cam, gamma=1.0, brightness=0.0, contrast=1.0)
        return

    # Toggles
    if _hit(R, "stab", pos): eng.set_stabilize(not eng.stabilize_on); return
    if _hit(R, "overlays", pos): eng.set_overlays(not eng.overlays_on); return
    if _hit(R, "set_ref", pos):
        eng.set_reference_here(state.active_cam)
        status_messages.append("Post: stabilization reference set"); return

    # Annotation mode buttons
    for key in ("text", "arrow", "box"):
        if _hit(R, f"annot_{key}", pos):
            state.annot_mode = None if state.annot_mode == key else key
            state.annot_pending_pos = None
            return
    if _hit(R, "annot_clear", pos):
        state.run.sidecar["annotations"] = []
        state.run.save_sidecar()
        if eng:
            eng._bump()
        return

    # Export
    if _hit(R, "exp_clip", pos):
        _start_export(state, status_messages, eng.in_marker, eng.out_marker,
                      stabilize=eng.stabilize_on, tag="clip"); return
    if _hit(R, "exp_stab", pos):
        _start_export(state, status_messages, eng.in_marker, eng.out_marker,
                      stabilize=True, tag="stab"); return
    if _hit(R, "exp_run", pos):
        _start_export(state, status_messages, eng.t0, eng.t1,
                      stabilize=eng.stabilize_on, tag="full"); return

    # Stack (In->Out range)
    if _hit(R, "stack_25", pos):
        _start_stack_export(state, status_messages, 0.25, local=False, tag="stack25"); return
    if _hit(R, "stack_50", pos):
        _start_stack_export(state, status_messages, 0.50, local=False, tag="stack50"); return
    if _hit(R, "stack_50_local", pos):
        _start_stack_export(state, status_messages, 0.50, local=True, tag="stack50ap"); return


def _hit(R, key, pos):
    r = R.get(key)
    return r is not None and r.collidepoint(pos)


def _apply_img_slider(state, kind, mx, sr):
    lo, hi, _ = _RANGES[kind]
    frac = (mx - sr.x) / max(1, sr.width)
    val = _frac_to_val(frac, lo, hi)
    state.engine.set_image_params(state.active_cam, **{kind: val})


def _apply_timeline(state, mx, tl):
    frac = (mx - tl.x) / max(1, tl.width)
    state.engine.seek_fraction(frac)


def _start_annotation(state, cam, pos, pane):
    nx = (pos[0] - pane.x) / pane.width
    ny = (pos[1] - pane.y) / pane.height
    if state.annot_mode == "text":
        state.annot_pending_pos = (cam, nx, ny)
        state.focus = "annot"
        state.text_buffer = ""
    else:  # arrow/box drag
        state.annot_draft = {"type": state.annot_mode, "cam": cam,
                             "x0": nx, "y0": ny, "x1": nx, "y1": ny}
        state.dragging = ("annot", cam)


def _start_export(state, status_messages, t_start, t_end, stabilize, tag):
    if state.exporter is not None:
        status_messages.append("Post: export already running"); return
    os.makedirs(_EXPORT_DIR, exist_ok=True)
    cam = state.active_cam
    cs = state.engine.cams.get(cam)
    safe = state.run.folder.replace(os.sep, "_")
    out = os.path.join(_EXPORT_DIR, f"{safe}_cam{cam + 1}_{tag}.mp4")
    state.exporter = Mp4Exporter(
        state.run, cam, t_start, t_end, out,
        gamma=cs.gamma, brightness=cs.brightness, contrast=cs.contrast,
        stabilize=stabilize, overlays=state.engine.overlays_on,
        stab_method=state.engine.stab_method)
    state.exporter.start()
    state.export_msg = ""
    status_messages.append(f"Post: exporting {os.path.basename(out)}")


def _start_stack_export(state, status_messages, keep_fraction, local, tag):
    if state.exporter is not None:
        status_messages.append("Post: export already running"); return
    os.makedirs(_EXPORT_DIR, exist_ok=True)
    eng = state.engine
    cam = state.active_cam
    safe = state.run.folder.replace(os.sep, "_")
    out = os.path.join(_EXPORT_DIR, f"{safe}_cam{cam + 1}_{tag}.png")
    # Stack the marked In->Out range; the master is intentionally linear (no
    # gamma/brightness/contrast baked in) so it feeds a later sharpening step.
    state.exporter = StackExporter(
        state.run, cam, eng.in_marker, eng.out_marker, out,
        keep_fraction=keep_fraction, local=local, method=eng.stab_method)
    state.exporter.start()
    state.export_msg = ""
    status_messages.append(f"Post: stacking best {int(keep_fraction * 100)}% "
                           f"-> {os.path.basename(out)}")


def handle_pp_motion(pos, display, state, buttons):
    if not buttons[0] or state.dragging is None:
        return
    if state.dragging == "timeline":
        tl = state.rects.get("timeline")
        if tl:
            _apply_timeline(state, pos[0], tl)
    elif isinstance(state.dragging, tuple) and state.dragging[0] == "img":
        kind = state.dragging[1]
        sr = state.rects.get(("img", kind))
        if sr:
            _apply_img_slider(state, kind, pos[0], sr)
    elif isinstance(state.dragging, tuple) and state.dragging[0] == "annot":
        cam = state.dragging[1]
        pane = state.pane_rects.get(cam)
        if pane and state.annot_draft:
            state.annot_draft["x1"] = (pos[0] - pane.x) / pane.width
            state.annot_draft["y1"] = (pos[1] - pane.y) / pane.height


def handle_pp_mouseup(pos, display, state):
    if isinstance(state.dragging, tuple) and state.dragging[0] == "annot":
        if state.annot_draft:
            state.run.sidecar["annotations"].append(dict(state.annot_draft))
            state.run.save_sidecar()
            state.annot_draft = None
            if state.engine:
                state.engine._bump()
    state.dragging = None


def handle_pp_wheel(pos, display, state, dy):
    if state.view != "library":
        return
    lst = state.rects.get("list")
    if lst and lst.collidepoint(pos):
        max_scroll = max(0, getattr(state, "_list_content_h", 0) - getattr(state, "_list_view_h", 0))
        state.scroll = max(0, min(state.scroll - dy * 30, max_scroll))


def handle_pp_key(event, display, state, status_messages):
    if state.focus is None:
        # Replay hotkeys when not typing
        if state.view == "replay" and state.engine is not None:
            if event.key == pygame.K_SPACE:
                state.engine.playing = not state.engine.playing
            elif event.key == pygame.K_LEFT:
                state.engine.step_frames(-1)
            elif event.key == pygame.K_RIGHT:
                state.engine.step_frames(1)
        return

    if event.key == pygame.K_RETURN:
        _commit_text(state, status_messages)
        return
    if event.key == pygame.K_ESCAPE:
        state.focus = None
        state.annot_pending_pos = None
        return
    if event.key == pygame.K_BACKSPACE:
        state.text_buffer = state.text_buffer[:-1]
    elif event.unicode and event.unicode.isprintable():
        state.text_buffer += event.unicode

    # Live-apply the filter as you type.
    if state.focus == "filter":
        state.text_filter = state.text_buffer
        state.scroll = 0


def _commit_text(state, status_messages):
    run = state.selected_run or state.run
    if state.focus == "filter":
        state.text_filter = state.text_buffer
    elif state.focus == "rename" and run:
        run.set_display_name(state.text_buffer)
    elif state.focus == "tag" and run:
        run.add_tag(state.text_buffer)
    elif state.focus == "notes" and run:
        run.set_notes(state.text_buffer)
    elif state.focus == "annot" and state.annot_pending_pos and state.run:
        cam, nx, ny = state.annot_pending_pos
        state.run.sidecar["annotations"].append(
            {"type": "text", "cam": cam, "x": nx, "y": ny, "text": state.text_buffer})
        state.run.save_sidecar()
        state.annot_pending_pos = None
        if state.engine:
            state.engine._bump()
    state.focus = None
    state.text_buffer = ""
