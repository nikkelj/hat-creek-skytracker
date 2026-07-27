"""Headless core for the post-processing / replay tool.

This module knows how tracking runs are laid out on disk and turns them into
something replayable. It has **no pygame UI** so it can be unit-tested and reused
for batch export.

Layout of a run folder (produced by ``capture_manager.py``)::

    data/<TARGET>_<NORAD>_<YYYY_MM_DD_HH_MM_SS>/   (or  manual_<timestamp>/)
        Camera1_000000__<YYYY_MM_DD_HH_MM_SS.ffffff>[Z].bmp
        Camera2_000000__...bmp
        trajectory.csv
        <name>_trackingvis_<...>.png
        postproc.json          <- sidecar this tool writes (favorite/tags/notes/...)

Pieces
------
* :class:`RunLibrary` - scans ``data/``, groups runs by target, owns sidecars.
* :class:`Run` - indexes camera frames + parses ``trajectory.csv`` into a synced
  timeline; resolves which frame each camera shows at a given instant.
* :class:`TrajectorySeries` - interpolatable az/el/range/pixel series + rates,
  used for the in-track / cross-track overlay vectors.
* :class:`FrameProcessor` - gamma stretch + brightness/contrast via a cached LUT.
* :func:`draw_overlays` - bakes vectors / meta text / annotations onto a frame.
* :class:`ReplayEngine` - threaded decode+process pipeline driving live replay.
* :class:`Mp4Exporter` - writes clips / stabilized movies / whole runs to MP4.
"""

import os
import re
import csv
import json
import bisect
import shutil
import threading
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:  # pragma: no cover
    pygame = None
    PYGAME_AVAILABLE = False

from stabilizer import Stabilizer, apply_affine_point

SIDECAR_NAME = "postproc.json"

# Folder timestamp suffix: _YYYY_MM_DD_HH_MM_SS  (always 6 trailing numeric groups)
_FOLDER_RE = re.compile(r"^(?P<prefix>.+?)_(?P<ts>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")
# Frame file: Camera<N>_<seq>__<YYYY_MM_DD_HH_MM_SS[.ffffff]>[Z].(bmp|png)
# The fractional part is optional (a timestamp landing on exactly .000000
# microseconds is written without a fraction) and both capture formats are
# accepted (config image_format supports BMP and PNG) -- a PNG run must not be
# silently invisible to post-processing.
_FRAME_RE = re.compile(
    r"^Camera(?P<cam>\d+)_(?P<seq>\d+)__(?P<ts>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}(?:\.\d+)?)Z?\.(?:bmp|png)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _parse_frame_time(ts):
    """Parse a frame-filename timestamp 'YYYY_MM_DD_HH_MM_SS[.ffffff]' -> epoch sec (UTC)."""
    if "." in ts:
        date_part, frac = ts.split(".")
    else:
        date_part, frac = ts, "0"
    dt = datetime.strptime(date_part, "%Y_%m_%d_%H_%M_%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() + float("0." + frac)


def _parse_iso_time(s):
    """Parse a trajectory.csv ISO timestamp (handles a trailing 'Z' after +00:00)."""
    s = s.strip()
    if s.endswith("Z") and ("+" in s or s.count(":") >= 3):
        s = s[:-1]  # strip the redundant trailing Z (already has +00:00 offset)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fall back: plain 'Z' UTC marker.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def fmt_utc(epoch):
    """Format an epoch second as a compact UTC HH:MM:SS.mmm string."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


# ---------------------------------------------------------------------------
# Trajectory series
# ---------------------------------------------------------------------------
class TrajectorySeries:
    """Interpolatable target state over time, parsed from trajectory.csv.

    Uses every row (pre-computed trajectory + per-frame) sorted by time, so az/el
    can be interpolated -- and differentiated for rate -- at any instant.
    """

    def __init__(self, t, az, el, dist, px, py):
        order = np.argsort(t)
        self.t = np.asarray(t)[order]
        self.az = np.unwrap(np.asarray(az)[order], period=360.0) if len(az) else np.asarray(az)
        self.el = np.asarray(el)[order]
        self.dist = np.asarray(dist)[order]
        self.px = np.asarray(px)[order]
        self.py = np.asarray(py)[order]

    @property
    def valid(self):
        return len(self.t) >= 2

    def interp(self, t):
        """Return dict(az, el, dist, px, py) interpolated at epoch ``t``."""
        if not len(self.t):
            return None
        az = float(np.interp(t, self.t, self.az)) % 360.0
        return {
            "az": az,
            "el": float(np.interp(t, self.t, self.el)),
            "dist": float(np.interp(t, self.t, self.dist)),
            "px": float(np.interp(t, self.t, self.px)),
            "py": float(np.interp(t, self.t, self.py)),
        }

    def rate(self, t, dt=0.5):
        """Central-difference (az_rate, el_rate) in deg/sec around ``t``."""
        if not self.valid:
            return 0.0, 0.0
        a0 = np.interp(t - dt, self.t, self.az)
        a1 = np.interp(t + dt, self.t, self.az)
        e0 = np.interp(t - dt, self.t, self.el)
        e1 = np.interp(t + dt, self.t, self.el)
        return float((a1 - a0) / (2 * dt)), float((e1 - e0) / (2 * dt))


def _load_trajectory(csv_path):
    if not os.path.isfile(csv_path):
        return TrajectorySeries([], [], [], [], [], [])
    t, az, el, dist, px, py = [], [], [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t.append(_parse_iso_time(row["timestamp"]))
                az.append(float(row["azimuth_deg"]))
                el.append(float(row["altitude_deg"]))
                dist.append(float(row.get("distance_km", "nan") or "nan"))
                px.append(float(row.get("pixel_x", "nan") or "nan"))
                py.append(float(row.get("pixel_y", "nan") or "nan"))
            except (KeyError, ValueError):
                continue
    return TrajectorySeries(t, az, el, dist, px, py)


# ---------------------------------------------------------------------------
# Run + library
# ---------------------------------------------------------------------------
class Run:
    """A single tracking run on disk: camera frame sequences + trajectory."""

    def __init__(self, path):
        self.path = path
        self.folder = os.path.basename(path.rstrip("/\\"))
        self.is_manual = self.folder.lower().startswith("manual_")
        self.target = "Manual"
        self.norad = None
        self.group_key = "Manual"
        self.start_dt = None

        m = _FOLDER_RE.match(self.folder)
        if m:
            prefix = m.group("prefix")
            try:
                self.start_dt = datetime.strptime(m.group("ts"), "%Y_%m_%d_%H_%M_%S")
            except ValueError:
                self.start_dt = None
            if self.is_manual:
                self.group_key = "Manual"
                self.target = "Manual"
            else:
                self.group_key = prefix  # e.g. STARLINK-31879_59937
                # NORAD = last all-digit token of the prefix.
                tokens = prefix.split("_")
                if tokens and tokens[-1].isdigit():
                    self.norad = tokens[-1]
                    self.target = "_".join(tokens[:-1])
                else:
                    self.target = prefix

        self._cameras = None     # {cam_index: [ {path, t, seq} ... ]}
        self._traj = None
        self._sidecar = None

    # --------------------------------------------------------------- frames
    def _index_cameras(self):
        if self._cameras is not None:
            return
        cams = {}
        try:
            names = os.listdir(self.path)
        except OSError:
            names = []
        for name in names:
            m = _FRAME_RE.match(name)
            if not m:
                continue
            cam_file = int(m.group("cam"))           # 1-based in filename
            cam_index = cam_file - 1                  # 0-based internal
            try:
                t = _parse_frame_time(m.group("ts"))
            except ValueError:
                continue
            cams.setdefault(cam_index, []).append({
                "path": os.path.join(self.path, name),
                "t": t,
                "seq": int(m.group("seq")),
            })
        for lst in cams.values():
            lst.sort(key=lambda d: d["t"])
        self._cameras = cams

    @property
    def camera_indices(self):
        self._index_cameras()
        return sorted(self._cameras.keys())

    def frames(self, cam_index):
        self._index_cameras()
        return self._cameras.get(cam_index, [])

    def frame_count(self, cam_index):
        return len(self.frames(cam_index))

    def frame_shape(self, cam_index):
        """(width, height) of this camera's frames, decoded once and cached."""
        self._index_cameras()
        if not hasattr(self, "_shapes"):
            self._shapes = {}
        if cam_index in self._shapes:
            return self._shapes[cam_index]
        frames = self.frames(cam_index)
        shape = None
        if frames:
            arr = decode_frame(frames[0]["path"])
            if arr is not None:
                shape = (arr.shape[1], arr.shape[0])
        self._shapes[cam_index] = shape
        return shape

    @property
    def trajectory(self):
        if self._traj is None:
            self._traj = _load_trajectory(os.path.join(self.path, "trajectory.csv"))
        return self._traj

    # ------------------------------------------------------------- timeline
    @property
    def t0(self):
        """Earliest frame time across all cameras (epoch sec), or None."""
        starts = [self.frames(c)[0]["t"] for c in self.camera_indices if self.frames(c)]
        return min(starts) if starts else None

    @property
    def t1(self):
        ends = [self.frames(c)[-1]["t"] for c in self.camera_indices if self.frames(c)]
        return max(ends) if ends else None

    @property
    def duration(self):
        if self.t0 is None or self.t1 is None:
            return 0.0
        return self.t1 - self.t0

    def frame_index_at(self, cam_index, t):
        """Index of the frame this camera shows at epoch ``t`` (last frame <= t)."""
        frames = self.frames(cam_index)
        if not frames:
            return None
        times = [f["t"] for f in frames]
        i = bisect.bisect_right(times, t) - 1
        return max(0, min(i, len(frames) - 1))

    # -------------------------------------------------------------- sidecar
    @property
    def sidecar_path(self):
        return os.path.join(self.path, SIDECAR_NAME)

    @property
    def sidecar(self):
        if self._sidecar is None:
            data = {}
            if os.path.isfile(self.sidecar_path):
                try:
                    with open(self.sidecar_path, "r") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    data = {}
            self._sidecar = {
                "display_name": data.get("display_name", ""),
                "favorite": bool(data.get("favorite", False)),
                "tags": list(data.get("tags", [])),
                "notes": data.get("notes", ""),
                "annotations": list(data.get("annotations", [])),
            }
        return self._sidecar

    def save_sidecar(self):
        try:
            with open(self.sidecar_path, "w") as f:
                json.dump(self.sidecar, f, indent=2)
            return True
        except OSError:
            return False

    @property
    def display_name(self):
        return self.sidecar["display_name"] or self.folder

    def set_display_name(self, name):
        self.sidecar["display_name"] = name.strip()
        self.save_sidecar()

    def toggle_favorite(self):
        self.sidecar["favorite"] = not self.sidecar["favorite"]
        self.save_sidecar()
        return self.sidecar["favorite"]

    def set_notes(self, notes):
        self.sidecar["notes"] = notes
        self.save_sidecar()

    def add_tag(self, tag):
        tag = tag.strip()
        if tag and tag not in self.sidecar["tags"]:
            self.sidecar["tags"].append(tag)
            self.save_sidecar()

    def remove_tag(self, tag):
        if tag in self.sidecar["tags"]:
            self.sidecar["tags"].remove(tag)
            self.save_sidecar()

    def delete(self):
        """Permanently delete the run folder from disk. Irreversible."""
        shutil.rmtree(self.path)


class RunLibrary:
    """Scans the data directory and groups runs by target for the browser UI."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.runs = []

    def scan(self):
        self.runs = []
        if not os.path.isdir(self.data_dir):
            return self.runs
        for name in sorted(os.listdir(self.data_dir)):
            full = os.path.join(self.data_dir, name)
            if not os.path.isdir(full):
                continue
            run = Run(full)
            # Only surface folders that actually contain camera frames.
            if run.camera_indices:
                self.runs.append(run)
        return self.runs

    def groups(self, favorites_only=False, text_filter=""):
        """Return ordered [(group_key, [runs])], newest run first within a group."""
        text = text_filter.lower().strip()
        groups = {}
        for run in self.runs:
            if favorites_only and not run.sidecar["favorite"]:
                continue
            if text and text not in run.folder.lower() and text not in run.display_name.lower():
                continue
            groups.setdefault(run.group_key, []).append(run)
        for key in groups:
            groups[key].sort(key=lambda r: (r.start_dt or datetime.min), reverse=True)
        # Order groups by their most-recent run.
        ordered = sorted(
            groups.items(),
            key=lambda kv: max((r.start_dt or datetime.min) for r in kv[1]),
            reverse=True,
        )
        return ordered


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------
class FrameProcessor:
    """Gamma stretch + brightness/contrast on an RGB uint8 frame via a 256-LUT.

    gamma     : >1 brightens faint detail (out = 255*(i/255)**(1/gamma)); 1 = off
    brightness: additive, [-128, 128]
    contrast  : multiplicative about mid-grey, [0, 3]; 1 = off
    """

    def __init__(self):
        self._lut = None
        self._key = None

    def _build_lut(self, gamma, brightness, contrast):
        i = np.arange(256, dtype=np.float64)
        g = max(gamma, 1e-3)
        stretched = 255.0 * np.power(i / 255.0, 1.0 / g)
        out = contrast * (stretched - 128.0) + 128.0 + brightness
        return np.clip(out, 0, 255).astype(np.uint8)

    def lut(self, gamma=1.0, brightness=0.0, contrast=1.0):
        key = (round(gamma, 4), round(brightness, 2), round(contrast, 4))
        if key != self._key:
            self._lut = self._build_lut(gamma, brightness, contrast)
            self._key = key
        return self._lut

    def apply(self, frame, gamma=1.0, brightness=0.0, contrast=1.0):
        if gamma == 1.0 and brightness == 0.0 and contrast == 1.0:
            return frame
        if CV2_AVAILABLE:
            return cv2.LUT(frame, self.lut(gamma, brightness, contrast))
        return self.lut(gamma, brightness, contrast)[frame]


def is_default_params(gamma, brightness, contrast):
    return gamma == 1.0 and brightness == 0.0 and contrast == 1.0


# ---------------------------------------------------------------------------
# Overlays (drawn with cv2 onto the final-size RGB frame -> DRY for display+export)
# ---------------------------------------------------------------------------
_COL_INTRACK = (80, 220, 90)     # green
_COL_CROSSTRACK = (90, 170, 255)  # blue
_COL_META = (255, 235, 120)       # amber
_COL_ANNOT = (255, 90, 90)        # red


def compute_track_vectors(traj, t, out_w, out_h, length_frac=0.18):
    """Anchor + in-track / cross-track end points for the overlay, in output px.

    The on-sky velocity direction is (az_rate*cos(el), el_rate) -- mirroring the
    bias projection in joystick_controller.py. We anchor at the frame centre (a
    well-tracked target sits near sensor centre) and draw the in-track direction
    plus a perpendicular cross-track axis. Image +y is down, so the elevation
    component is negated.
    """
    if traj is None or not traj.valid:
        return None
    az_rate, el_rate = traj.rate(t)
    state = traj.interp(t)
    cos_el = max(np.cos(np.radians(state["el"])) if state else 1.0, 0.087)
    vx = az_rate * cos_el
    vy = el_rate
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return None
    ux, uy = vx / norm, vy / norm           # on-sky unit vector
    ix, iy = ux, -uy                        # to image coords (y down)
    cx, cy = -iy, ix                        # perpendicular (cross-track)
    L = length_frac * min(out_w, out_h)
    ax, ay = out_w / 2.0, out_h / 2.0
    return {
        "anchor": (ax, ay),
        "intrack": (ax + ix * L, ay + iy * L),
        "crosstrack_p": (ax + cx * L * 0.6, ay + cy * L * 0.6),
        "crosstrack_n": (ax - cx * L * 0.6, ay - cy * L * 0.6),
    }


def draw_overlays(frame, vectors=None, meta_lines=None, annotations=None,
                  out_size=None, font_scale=1.0, meta_scale=None):
    """Bake overlays onto an RGB uint8 frame (in place-ish; returns the frame).

    annotations are stored in NORMALIZED [0,1] coords so they map to any output
    size; ``out_size`` defaults to the frame's own size.

    ``meta_scale`` is the *absolute* cv2 fontScale for the meta/label text and is
    decoupled from ``font_scale`` (which sizes the vector arrows) so meta text can
    stay small and readable regardless of how large the pane is. Defaults to a
    small fixed size.
    """
    if not CV2_AVAILABLE:
        return frame
    h, w = frame.shape[:2]
    out_w, out_h = (out_size or (w, h))
    th = max(1, int(round(font_scale)))
    ms = 0.4 if meta_scale is None else float(meta_scale)
    mt = 1  # meta/label text is always thin (1px) so it reads cleanly when small
    line_h = int(round(ms * 34)) + 4

    if vectors:
        a = tuple(int(round(v)) for v in vectors["anchor"])
        it = tuple(int(round(v)) for v in vectors["intrack"])
        cv2.arrowedLine(frame, a, it, _COL_INTRACK, max(1, th), tipLength=0.18)
        cp = tuple(int(round(v)) for v in vectors["crosstrack_p"])
        cn = tuple(int(round(v)) for v in vectors["crosstrack_n"])
        cv2.line(frame, cp, cn, _COL_CROSSTRACK, max(1, th))
        cv2.putText(frame, "IN", it, cv2.FONT_HERSHEY_SIMPLEX, ms, _COL_INTRACK, mt, cv2.LINE_AA)
        cv2.putText(frame, "CROSS", cp, cv2.FONT_HERSHEY_SIMPLEX, ms, _COL_CROSSTRACK, mt, cv2.LINE_AA)

    if meta_lines:
        y = line_h
        for line in meta_lines:
            cv2.putText(frame, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, ms, _COL_META, mt, cv2.LINE_AA)
            y += line_h

    for ann in (annotations or []):
        col = tuple(ann.get("color", _COL_ANNOT))
        kind = ann.get("type")
        if kind == "text":
            x = int(ann["x"] * w); y = int(ann["y"] * h)
            cv2.putText(frame, ann.get("text", ""), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, ms * 1.15, col, mt, cv2.LINE_AA)
        elif kind == "arrow":
            p0 = (int(ann["x0"] * w), int(ann["y0"] * h))
            p1 = (int(ann["x1"] * w), int(ann["y1"] * h))
            cv2.arrowedLine(frame, p0, p1, col, th + 1, tipLength=0.2)
        elif kind == "box":
            p0 = (int(ann["x0"] * w), int(ann["y0"] * h))
            p1 = (int(ann["x1"] * w), int(ann["y1"] * h))
            cv2.rectangle(frame, p0, p1, col, th + 1)
    return frame


_REDUCED_FLAGS = {
    1: cv2.IMREAD_COLOR if CV2_AVAILABLE else 1,
    2: cv2.IMREAD_REDUCED_COLOR_2 if CV2_AVAILABLE else 1,
    4: cv2.IMREAD_REDUCED_COLOR_4 if CV2_AVAILABLE else 1,
    8: cv2.IMREAD_REDUCED_COLOR_8 if CV2_AVAILABLE else 1,
}


def decode_frame(path, reduce=1):
    """Decode a BMP to an RGB uint8 array (None on failure).

    ``reduce`` (1, 2, 4 or 8) uses OpenCV's reduced-scale decode to read the image
    at 1/N resolution -- decoding ~N^2 less data, which is the dominant cost when
    showing a 1936x1096 frame in a small replay pane.
    """
    if not CV2_AVAILABLE:
        return None
    flag = _REDUCED_FLAGS.get(reduce, cv2.IMREAD_COLOR)
    bgr = cv2.imread(path, flag)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def reduce_for(full_w, pane_w):
    """Largest power-of-two decode reduction that still yields >= pane width."""
    if not full_w or pane_w <= 0:
        return 1
    r = 1
    while r < 8 and full_w // (r * 2) >= pane_w:
        r *= 2
    return r


def build_meta_lines(run, cam_index, t, frame_idx, n_frames, gamma, stabilize_on):
    state = run.trajectory.interp(t) if run.trajectory.valid else None
    lines = [f"Cam{cam_index + 1}  {fmt_utc(t)} UTC  [{frame_idx + 1}/{n_frames}]"]
    if state:
        lines.append(f"Az {state['az']:.2f}  El {state['el']:.2f}  Rng {state['dist']:.0f} km")
    flags = []
    if gamma != 1.0:
        flags.append(f"gamma {gamma:.2f}")
    if stabilize_on:
        flags.append("STAB")
    if flags:
        lines.append("  ".join(flags))
    return lines


# ---------------------------------------------------------------------------
# Live replay engine
# ---------------------------------------------------------------------------
class _CamState:
    def __init__(self, cam_index):
        self.cam_index = cam_index
        self.pane_size = (640, 360)
        self.gamma = 1.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.reference_idx = 0
        self.stabilizer = None
        self.stab_reduce = None       # reduce factor the reference was built at
        self.reduce = 1               # current decode reduction for display
        # worker bookkeeping
        self.target_idx = 0
        self.rendered_key = None
        self.surface = None
        self.cur_frame_idx = 0
        self.thread = None
        self.wake = threading.Event()


class ReplayEngine:
    """Threaded decode+process pipeline that produces display surfaces.

    The main loop calls :meth:`update` each tick (advancing the playhead when
    playing) and :meth:`get_surface` to blit. Each camera gets its **own** worker
    thread so both panes decode in parallel, and frames are decoded at a reduced
    resolution matched to the pane size (OpenCV ``IMREAD_REDUCED_*``) -- together
    these are the big wins over reading full 6-19 MB BMPs on the UI thread. Each
    worker also prefetches the next few frames so playback stays smooth.
    """

    PREFETCH = 3

    def __init__(self, run, cache_size=48):
        self.run = run
        self.cams = {c: _CamState(c) for c in run.camera_indices}
        self.processor = FrameProcessor()

        self.t0 = run.t0 or 0.0
        self.t1 = run.t1 or 0.0
        self.t = self.t0
        self.playing = False
        self.speed = 1.0
        self.loop = True

        self.stabilize_on = False
        self.overlays_on = True
        self.stab_method = "orb"

        self.in_marker = self.t0
        self.out_marker = self.t1

        self._params_version = 0
        self._raw_cache = OrderedDict()   # (path, reduce) -> RGB array
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle
    def start(self):
        for cs in self.cams.values():
            if cs.thread is None:
                cs.thread = threading.Thread(
                    target=self._cam_worker, args=(cs,), daemon=True)
                cs.thread.start()

    def stop(self):
        self._stop.set()
        for cs in self.cams.values():
            cs.wake.set()
        for cs in self.cams.values():
            if cs.thread is not None:
                cs.thread.join(timeout=2.0)
                cs.thread = None

    def _bump(self):
        """Mark params changed and wake every camera worker."""
        self._params_version += 1
        for cs in self.cams.values():
            cs.wake.set()

    # --------------------------------------------------------------- params
    def set_pane_size(self, cam_index, w, h):
        cs = self.cams.get(cam_index)
        if not cs or w <= 0 or h <= 0 or cs.pane_size == (w, h):
            return
        cs.pane_size = (int(w), int(h))
        shape = self.run.frame_shape(cam_index)
        new_reduce = reduce_for(shape[0], w) if shape else 1
        if new_reduce != cs.reduce:
            cs.reduce = new_reduce
            cs.stabilizer = None      # reference must match the new reduce factor
        self._params_version += 1
        cs.wake.set()

    def set_image_params(self, cam_index, gamma=None, brightness=None, contrast=None):
        cs = self.cams.get(cam_index)
        if not cs:
            return
        if gamma is not None:
            cs.gamma = float(gamma)
        if brightness is not None:
            cs.brightness = float(brightness)
        if contrast is not None:
            cs.contrast = float(contrast)
        self._params_version += 1
        cs.wake.set()

    def set_stabilize(self, on):
        self.stabilize_on = bool(on)
        for cs in self.cams.values():
            cs.stabilizer = None  # rebuilt lazily with current reference
        self._bump()

    def set_overlays(self, on):
        self.overlays_on = bool(on)
        self._bump()

    def set_reference_here(self, cam_index):
        """Use the camera's currently displayed frame as the stabilization anchor."""
        cs = self.cams.get(cam_index)
        if cs:
            cs.reference_idx = cs.cur_frame_idx
            cs.stabilizer = None
            self._params_version += 1
            cs.wake.set()

    # ------------------------------------------------------------- playhead
    @property
    def fraction(self):
        span = self.t1 - self.t0
        return 0.0 if span <= 0 else (self.t - self.t0) / span

    def seek_fraction(self, f):
        f = max(0.0, min(1.0, f))
        self.t = self.t0 + f * (self.t1 - self.t0)
        self._wake_all()

    def step_frames(self, n):
        """Step the playhead by n frames on the first camera (rough nudge)."""
        cams = self.run.camera_indices
        if not cams:
            return
        cam = cams[0]
        idx = self.run.frame_index_at(cam, self.t) or 0
        frames = self.run.frames(cam)
        idx = max(0, min(idx + n, len(frames) - 1))
        self.t = frames[idx]["t"]
        self.playing = False
        self._wake_all()

    def _wake_all(self):
        for cs in self.cams.values():
            cs.wake.set()

    def set_in_marker(self):
        self.in_marker = self.t

    def set_out_marker(self):
        self.out_marker = self.t

    def update(self, dt):
        if self.playing and self.t1 > self.t0:
            self.t += dt * self.speed
            if self.t >= self.t1:
                if self.loop:
                    self.t = self.t0
                else:
                    self.t = self.t1
                    self.playing = False
        # Compute per-camera target frame index; only wake a worker when its
        # frame actually changes (avoids spinning the workers every 60 Hz tick).
        for cam, cs in self.cams.items():
            idx = self.run.frame_index_at(cam, self.t)
            if idx is not None and idx != cs.target_idx:
                cs.target_idx = idx
                cs.cur_frame_idx = idx
                cs.wake.set()

    def get_surface(self, cam_index):
        cs = self.cams.get(cam_index)
        return cs.surface if cs else None

    # --------------------------------------------------------------- worker
    def _decode_cached(self, path, reduce):
        ckey = (path, reduce)
        with self._lock:
            if ckey in self._raw_cache:
                self._raw_cache.move_to_end(ckey)
                return self._raw_cache[ckey]
        arr = decode_frame(path, reduce)
        if arr is not None:
            with self._lock:
                self._raw_cache[ckey] = arr
                while len(self._raw_cache) > self._cache_size:
                    self._raw_cache.popitem(last=False)
        return arr

    def _ensure_reference(self, cs):
        if (cs.stabilizer is not None and cs.stabilizer.has_reference
                and cs.stab_reduce == cs.reduce):
            return
        frames = self.run.frames(cs.cam_index)
        if not frames:
            return
        ref_idx = max(0, min(cs.reference_idx, len(frames) - 1))
        ref_raw = self._decode_cached(frames[ref_idx]["path"], cs.reduce)
        if ref_raw is None:
            return
        ref_proc = self.processor.apply(ref_raw, cs.gamma, cs.brightness, cs.contrast)
        cs.stabilizer = Stabilizer(method=self.stab_method)
        cs.stabilizer.set_reference(ref_proc)
        cs.stab_reduce = cs.reduce

    def _build_surface(self, cs):
        frames = self.run.frames(cs.cam_index)
        if not frames:
            return
        idx = max(0, min(cs.target_idx, len(frames) - 1))
        key = (idx, self._params_version, cs.pane_size, self.stabilize_on, self.overlays_on)
        if key == cs.rendered_key:
            return
        raw = self._decode_cached(frames[idx]["path"], cs.reduce)
        if raw is None:
            return
        proc = self.processor.apply(raw, cs.gamma, cs.brightness, cs.contrast)

        if self.stabilize_on:
            self._ensure_reference(cs)
            if cs.stabilizer is not None:
                proc, _ = cs.stabilizer.stabilize(proc)

        pw, ph = cs.pane_size
        if CV2_AVAILABLE:
            scaled = cv2.resize(proc, (pw, ph), interpolation=cv2.INTER_AREA)
        else:  # pragma: no cover
            scaled = proc

        if self.overlays_on:
            t = frames[idx]["t"]
            vectors = compute_track_vectors(self.run.trajectory, t, pw, ph)
            meta = build_meta_lines(self.run, cs.cam_index, t, idx, len(frames),
                                    cs.gamma, self.stabilize_on)
            draw_overlays(scaled, vectors=vectors, meta_lines=meta,
                          annotations=self.run.sidecar["annotations"])

        surf = None
        if PYGAME_AVAILABLE:
            buf = np.ascontiguousarray(scaled)
            surf = pygame.image.frombuffer(buf.tobytes(), (pw, ph), "RGB")
        cs.surface = surf
        cs.rendered_key = key

    def _prefetch(self, cs):
        """Warm the cache with the next few frames for this camera (cheap when idle)."""
        frames = self.run.frames(cs.cam_index)
        n = len(frames)
        for d in range(1, self.PREFETCH + 1):
            if self._stop.is_set() or cs.wake.is_set():
                return  # a real request arrived; abandon prefetch
            j = cs.target_idx + d
            if 0 <= j < n:
                self._decode_cached(frames[j]["path"], cs.reduce)

    def _cam_worker(self, cs):
        while not self._stop.is_set():
            cs.wake.wait(timeout=0.2)
            cs.wake.clear()
            if self._stop.is_set():
                break
            try:
                self._build_surface(cs)
            except Exception as exc:  # keep replay alive on a bad frame
                print(f"[post_process] frame build error cam{cs.cam_index}: {exc}")
            if not cs.wake.is_set():
                self._prefetch(cs)


# ---------------------------------------------------------------------------
# MP4 export
# ---------------------------------------------------------------------------
class Mp4Exporter:
    """Export a camera's frame range to MP4, baking in the same pipeline.

    Reuses gamma/brightness/contrast, optional stabilization (fresh anchor at the
    range start) and overlays, at full resolution. Runs on its own thread (call
    :meth:`start`); poll :attr:`progress` / :attr:`done` / :attr:`error` from the UI.

    (Owns a thread rather than subclassing Thread so the Run object can live on
    ``self.run`` without shadowing ``Thread.run``.)
    """

    def __init__(self, run, cam_index, t_start, t_end, out_path,
                 gamma=1.0, brightness=0.0, contrast=1.0,
                 stabilize=False, overlays=True, stab_method="orb", fps=None):
        self.run = run
        self.cam_index = cam_index
        self.t_start = t_start
        self.t_end = t_end
        self.out_path = out_path
        self.gamma = gamma
        self.brightness = brightness
        self.contrast = contrast
        self.stabilize = stabilize
        self.overlays = overlays
        self.stab_method = stab_method
        self.fps = fps

        self.progress = 0.0
        self.done = False
        self.error = None
        self.frames_written = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _frames_in_range(self):
        frames = self.run.frames(self.cam_index)
        lo, hi = sorted((self.t_start, self.t_end))
        return [f for f in frames if lo <= f["t"] <= hi]

    def _worker(self):
        try:
            self._export()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done = True

    def _export(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is required for MP4 export")
        frames = self._frames_in_range()
        if len(frames) < 1:
            raise RuntimeError("No frames in the selected range")

        first = decode_frame(frames[0]["path"])
        if first is None:
            raise RuntimeError("Could not decode first frame")
        h, w = first.shape[:2]

        # FPS: explicit, else from median capture interval (clamped to a sane range).
        # KNOWN LIMITATION: the export writes one video frame per captured frame
        # at this constant fps, so captures with dropped frames or a mid-run
        # rate change are time-warped relative to the timestamp-paced replay UI.
        # The per-frame overlay text stays correct; don't measure rates off the
        # exported video.
        fps = self.fps
        if fps is None:
            times = [f["t"] for f in frames]
            if len(times) > 1:
                diffs = np.diff(times)
                med = float(np.median(diffs)) if len(diffs) else 0.2
                fps = 1.0 / med if med > 1e-3 else 5.0
            else:
                fps = 5.0
            fps = float(min(60.0, max(1.0, fps)))

        processor = FrameProcessor()
        stab = Stabilizer(method=self.stab_method) if self.stabilize else None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {self.out_path}")

        try:
            total = len(frames)
            for i, f in enumerate(frames):
                raw = decode_frame(f["path"])
                if raw is None:
                    continue
                proc = processor.apply(raw, self.gamma, self.brightness, self.contrast)
                if stab is not None:
                    proc, _ = stab.stabilize(proc)
                if self.overlays:
                    vectors = compute_track_vectors(self.run.trajectory, f["t"], w, h)
                    meta = build_meta_lines(self.run, self.cam_index, f["t"], i, total,
                                            self.gamma, self.stabilize)
                    draw_overlays(proc, vectors=vectors, meta_lines=meta,
                                  annotations=self.run.sidecar["annotations"],
                                  font_scale=max(1.0, h / 720.0),
                                  meta_scale=max(0.5, h / 1400.0))
                writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                self.frames_written += 1
                self.progress = (i + 1) / total
        finally:
            writer.release()


# ---------------------------------------------------------------------------
# Lucky-imaging stack export (PIPP cull/centre + AutoStakkert align/average)
# ---------------------------------------------------------------------------
class StackExporter:
    """Export a high-SNR stacked master from a camera's frame range.

    This is the AutoStakkert end of the pipeline wired to the replay UI: grade
    the frames in ``[t_start, t_end]`` by sharpness, keep the sharpest
    ``keep_fraction``, align them to the sharpest frame and average them into one
    low-noise image (optionally with alignment-point local warping). Image
    adjustments are intentionally *not* baked in -- a stacked master is the
    linear input to a later wavelet/deconvolution sharpening step.

    Mirrors :class:`Mp4Exporter`'s threading contract: call :meth:`start`, then
    poll :attr:`progress` / :attr:`done` / :attr:`error` and read
    :attr:`out_path` / :attr:`stats` when finished.
    """

    def __init__(self, run, cam_index, t_start, t_end, out_path,
                 keep_fraction=0.5, quality="laplacian", method="orb",
                 local=False, roi=None, max_frames=None, workers=None,
                 grade_scale=1.0, prefilter=True, align_scale=1.0,
                 center_size=None, finish=True):
        self.run = run
        self.cam_index = cam_index
        self.t_start = t_start
        self.t_end = t_end
        self.out_path = out_path
        self.keep_fraction = keep_fraction
        self.quality = quality
        self.method = method
        self.local = local
        self.roi = roi
        self.max_frames = max_frames
        self.workers = workers          # None -> all cores
        self.grade_scale = grade_scale  # <1 grades at reduced res for speed
        self.prefilter = prefilter      # conservative garbage pre-cull
        self.align_scale = align_scale  # <1 estimates alignment at reduced res
        self.center_size = center_size  # PIPP centring crop (px), None = off
        self.finish = finish            # also emit a sharpened/stretched final

        self.progress = 0.0
        self.done = False
        self.error = None
        self.stats = None
        self.dropped = 0
        self.final_path = None
        self.frames_written = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _worker(self):
        try:
            self._export()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done = True

    def _export(self):
        # Imported lazily so post_process stays importable without the stacker.
        from stacking import stack_run, save_master

        # The align pass is roughly the back 80% of wall time; reserve the
        # front 20% for grading (which previously reported no progress at all).
        def _progress(done, total):
            self.progress = 0.2 + 0.8 * (done / total if total else 1.0)

        self.progress = 0.05  # grading...
        # 16-bit linear master: keeps the stacked SNR through the finishing
        # stretch instead of quantizing it away at 8 bits.
        result = stack_run(
            self.run, self.cam_index, keep_fraction=self.keep_fraction,
            quality=self.quality, method=self.method, local=self.local,
            roi=self.roi, t_start=self.t_start, t_end=self.t_end,
            max_frames=self.max_frames, workers=self.workers,
            grade_scale=self.grade_scale, prefilter=self.prefilter,
            align_scale=self.align_scale, center_size=self.center_size,
            bits=16, progress=_progress)
        if result.master is None:
            raise RuntimeError("No alignable frames in the selected range to stack")
        self.out_path = save_master(result.master, self.out_path)
        self.stats = result.stats
        self.dropped = len(result.dropped)
        self.frames_written = result.stats.n_stacked
        if self.finish:
            from sharpen import finish as _finish, save_final
            base, _ = os.path.splitext(self.out_path)
            self.final_path = save_final(_finish(result.master),
                                         base + "_final.png")
        self.progress = 1.0
