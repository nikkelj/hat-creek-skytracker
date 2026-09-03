"""Flag-gated bridge from stacking.py / stabilizer.py / sharpen.py numeric
kernels to skytracker-imaging (Phase 3b of the Rust port). Same contract as
the other adapters: every entry point returns None on ANY failure so the
cv2/numpy implementations remain the fallback. Enabled by config
`use_rust_imaging` (configure(config_state) at startup) or env
SKYTRACKER_RUST_IMAGING=1 (0 force-disables).

Gray (2-D) kernels only: color frames and all orchestration (grading loops,
stack accumulation, reference management) stay in Python.
"""

from __future__ import annotations

import os

import numpy as np

_enabled_flag = False
_mod = None
_failed = False


def configure(config_state):
    global _enabled_flag
    _enabled_flag = bool(getattr(config_state, "use_rust_imaging", False))


def enabled():
    env = os.environ.get("SKYTRACKER_RUST_IMAGING")
    if env == "1":
        return True
    if env == "0":
        return False
    return _enabled_flag


def _core():
    global _mod, _failed
    if _mod is None and not _failed:
        try:
            import skytracker_core as sc

            if not getattr(sc, "IMAGING_AVAILABLE", False):
                raise ImportError("wheel predates imaging kernels")
            _mod = sc
        except Exception as e:
            print(f"Rust imaging kernels unavailable ({e}); using cv2/numpy.")
            _failed = True
    return _mod


def _gray32(arr):
    a = np.asarray(arr)
    if a.ndim != 2:
        return None
    return np.ascontiguousarray(a, dtype=np.float32)


def sharpness(gray, method, scale=1.0):
    sc = _core()
    g = _gray32(gray)
    if sc is None or g is None or method not in ("laplacian", "tenengrad"):
        return None
    was_uint8 = np.asarray(gray).dtype == np.uint8
    try:
        if scale != 1.0 and g.size:
            gh, gw = g.shape
            nw, nh = max(1, int(gw * scale)), max(1, int(gh * scale))
            g = np.asarray(sc.imaging_resize_area(g, nw, nh))
            if was_uint8:
                # cv2 resizes uint8 input to uint8 (banker's rounding);
                # reproduce that quantization before scoring.
                g = np.ascontiguousarray(
                    np.clip(np.rint(g), 0, 255), dtype=np.float32)
        if g.size == 0:
            return 0.0
        return float(sc.imaging_sharpness(g, method))
    except Exception:
        return None


def brightness_centroid(gray, threshold=None):
    sc = _core()
    g = _gray32(gray)
    if sc is None or g is None:
        return None
    try:
        res = sc.imaging_brightness_centroid(
            g, threshold=None if threshold is None else float(threshold))
        # Distinguish "no target" (legit result) from adapter failure by
        # wrapping in a tuple.
        return (res,)
    except Exception:
        return None


def measure_local_shifts(ref_gray, cur_gray, points, patch, min_response, max_shift):
    sc = _core()
    r = _gray32(ref_gray)
    c = _gray32(cur_gray)
    if sc is None or r is None or c is None:
        return None
    try:
        return np.asarray(sc.imaging_measure_local_shifts(
            r, c, [[float(p[0]), float(p[1])] for p in points], int(patch),
            min_response=float(min_response),
            max_shift=None if max_shift is None else float(max_shift)))
    except Exception:
        return None


def warp_by_grid(frame, rows, cols, shifts):
    """Gray or color frame; color goes through per-channel."""
    sc = _core()
    if sc is None:
        return None
    try:
        shifts_list = [[float(s[0]), float(s[1])] for s in shifts]
        arr = np.asarray(frame)
        if arr.ndim == 2:
            out = sc.imaging_warp_by_grid(
                np.ascontiguousarray(arr, dtype=np.float32), rows, cols, shifts_list)
            return np.asarray(out).astype(arr.dtype) if arr.dtype != np.float32 else np.asarray(out)
        chans = [np.asarray(sc.imaging_warp_by_grid(
            np.ascontiguousarray(arr[..., i], dtype=np.float32), rows, cols, shifts_list))
            for i in range(arr.shape[2])]
        out = np.stack(chans, axis=-1)
        return out.astype(arr.dtype) if arr.dtype != np.float32 else out
    except Exception:
        return None


def detect_flow_reference(ref_gray, max_features):
    sc = _core()
    g = _gray32(ref_gray)
    if sc is None or g is None:
        return None
    try:
        return np.asarray(sc.imaging_detect_flow_reference(g, int(max_features)))
    except Exception:
        return None


def estimate_flow(ref_gray, ref_points, cur_gray, min_inliers, min_inlier_ratio,
                  scale_tol, max_rotation_deg, max_translation_frac):
    sc = _core()
    r = _gray32(ref_gray)
    c = _gray32(cur_gray)
    if sc is None or r is None or c is None:
        return None
    try:
        pts = [[float(p[0]), float(p[1])] for p in np.asarray(ref_points).reshape(-1, 2)]
        m, inliers, reason = sc.imaging_estimate_flow(
            r, pts, c, min_inliers=int(min_inliers),
            min_inlier_ratio=float(min_inlier_ratio), scale_tol=float(scale_tol),
            max_rotation_deg=float(max_rotation_deg),
            max_translation_frac=float(max_translation_frac))
        return (None if m is None else np.asarray(m, dtype=np.float64), inliers, reason)
    except Exception:
        return None


class _RustVideoWriter:
    """cv2.VideoWriter-compatible surface over the Rust H.264 encoder.

    write() takes BGR frames (like cv2.VideoWriter) and flips to RGB."""

    def __init__(self, encoder):
        self._enc = encoder
        self._open = True

    def isOpened(self):
        return self._open

    def write(self, bgr):
        frame = np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1])
        self._enc.write(frame)

    def release(self):
        if self._open:
            self._open = False
            try:
                self._enc.finish()
            except Exception as e:
                print(f"Rust mp4 finish failed: {e}")


def make_video_writer(out_path, width, height, fps):
    """A cv2.VideoWriter-like H.264 writer, or None (caller falls back)."""
    sc = _core()
    if sc is None:
        return None
    try:
        return _RustVideoWriter(
            sc.Mp4Encoder(str(out_path), int(width), int(height), float(fps)))
    except Exception as e:
        print(f"Rust mp4 encoder unavailable ({e}); using cv2.VideoWriter.")
        return None


def finish_gray(img01, layers, stretch, black_pct, white_pct, target_median):
    sc = _core()
    g = _gray32(img01)
    if sc is None or g is None:
        return None
    try:
        return np.asarray(sc.imaging_finish_gray(
            g, [(float(s), float(a)) for s, a in layers], stretch=bool(stretch),
            black_pct=float(black_pct), white_pct=float(white_pct),
            target_median=float(target_median)))
    except Exception:
        return None
