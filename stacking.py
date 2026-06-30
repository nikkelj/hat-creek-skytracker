"""Lucky-imaging post-processing: PIPP-style prep + AutoStakkert-style stacking.

This module adds the planetary / lunar / satellite "lucky imaging" pipeline to
the post-processing core. It follows the same two-tool mental model the rest of
the project uses (see ``post_process.py``):

PIPP stage -- *prep & conditioning* (tidy and align)
    * :func:`sharpness` / :class:`QualityGrader`
        estimate per-frame sharpness and rank frames worst-to-best.
    * :func:`select_best`
        cull down to the sharpest top X% (or top N) -- "lucky" frame selection.
    * :func:`brightness_centroid` / :func:`bounding_box`
        find the target (a bright blob on a dark sky).
    * :func:`crop_centered` / :func:`recenter_frame`
        re-centre the target on the canvas and/or crop a small box around it,
        which is the big win for a fast-moving target like the ISS.

AutoStakkert stage -- *stacking* (combine the best frames)
    * :class:`AlignmentPointGrid` + :func:`measure_local_shifts`
        a regular grid of alignment points and the per-point sub-pixel shift of
        a frame vs a reference -- i.e. the local distortion from seeing.
    * :class:`LuckyStacker`
        align the selected frames to a reference and average them into one
        high-SNR master, optionally correcting local distortion via the
        alignment-point grid. Averaging beats down random noise so detail buried
        in any single frame emerges.
    * :func:`stack_run`
        glue that runs the whole PIPP->AutoStakkert flow over an on-disk
        :class:`post_process.Run`.

Everything is pure numpy / OpenCV so it runs headless and is unit-testable, and
it composes with the existing :class:`stabilizer.Stabilizer` (global rigid
alignment) and the on-disk frame indexing in :class:`post_process.Run`.
"""

import os
from collections import namedtuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - cv2 is expected in the track env
    cv2 = None
    CV2_AVAILABLE = False

from stabilizer import Stabilizer, _to_gray


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _require_cv2():
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for stacking")


def _decode(path):
    """Decode an image file to an RGB uint8 array (None on failure)."""
    if not CV2_AVAILABLE:
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _as_array(frame):
    """Coerce a frame spec (ndarray | path str | {'path': ...}) to an ndarray."""
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return frame
    if isinstance(frame, str):
        return _decode(frame)
    if isinstance(frame, dict) and "path" in frame:
        return _decode(frame["path"])
    raise TypeError(f"cannot coerce {type(frame)!r} to a frame array")


def _clamp_roi(shape, roi):
    """Clamp an (x, y, w, h) ROI to the image, returning a valid sub-box."""
    h, w = shape[:2]
    x, y, rw, rh = (int(v) for v in roi)
    x0 = max(0, min(x, w - 1))
    y0 = max(0, min(y, h - 1))
    x1 = max(x0 + 1, min(x + rw, w))
    y1 = max(y0 + 1, min(y + rh, h))
    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# PIPP stage: quality estimation + frame grading / culling
# ---------------------------------------------------------------------------
SHARPNESS_METHODS = ("laplacian", "tenengrad", "fft")


def sharpness(frame, method="laplacian", roi=None):
    """Estimate a frame's sharpness as a single non-negative score.

    Higher means sharper. The score is only meaningful *relative* to other
    frames of the same scene, which is exactly how it is used for grading.

    method
        ``"laplacian"`` -- variance of the Laplacian (cheap, robust default).
        ``"tenengrad"`` -- mean squared Sobel gradient magnitude (edge energy).
        ``"fft"``       -- fraction of spectral energy in the high-frequency
                           band; less sensitive to overall brightness/contrast.
    roi
        optional (x, y, w, h) sub-box (full-frame px) to score only the target
        region -- keeps a bright, sharp target from being averaged out by a
        large empty sky.
    """
    _require_cv2()
    gray = _to_gray(frame)
    if roi is not None:
        x, y, w, h = _clamp_roi(gray.shape, roi)
        gray = gray[y:y + h, x:x + w]
    if gray.size == 0:
        return 0.0
    if method == "laplacian":
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if method == "tenengrad":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx * gx + gy * gy))
    if method == "fft":
        return _fft_sharpness(gray)
    raise ValueError(f"unknown sharpness method {method!r}; choose from {SHARPNESS_METHODS}")


def _fft_sharpness(gray):
    """High-frequency energy fraction of the magnitude spectrum (0..1-ish)."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    rmax = np.hypot(cy, cx) + 1e-9
    high = float(mag[r > 0.25 * rmax].sum())
    total = float(mag.sum()) + 1e-9
    return high / total


#: One graded frame. ``index`` is the position in the input sequence, ``score``
#: its sharpness, and ``source`` the original path/dict (None for raw arrays).
FrameGrade = namedtuple("FrameGrade", ["index", "score", "source"])


class QualityGrader:
    """Score and rank a sequence of frames by sharpness (PIPP-style sorting)."""

    def __init__(self, method="laplacian", roi=None):
        if method not in SHARPNESS_METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.roi = roi

    def score(self, frame):
        arr = _as_array(frame)
        return 0.0 if arr is None else sharpness(arr, self.method, self.roi)

    def grade(self, frames):
        """Grade an iterable of frames, returning ``FrameGrade`` in input order."""
        grades = []
        for i, f in enumerate(frames):
            arr = _as_array(f)
            s = 0.0 if arr is None else sharpness(arr, self.method, self.roi)
            src = f if not isinstance(f, np.ndarray) else None
            grades.append(FrameGrade(i, s, src))
        return grades


def select_best(grades, fraction=None, count=None, min_keep=1):
    """Pick the highest-scoring grades, sharpest first (the "lucky" frames).

    Provide ``fraction`` (e.g. 0.25 keeps the best 25%) or ``count`` (keep N).
    With neither, all grades are returned sorted best-first. ``min_keep`` floors
    the result so an aggressive fraction never yields zero frames.
    """
    ordered = sorted(grades, key=lambda g: g.score, reverse=True)
    n = len(ordered)
    if n == 0:
        return []
    if count is not None:
        k = int(count)
    elif fraction is not None:
        k = int(round(fraction * n))
    else:
        k = n
    k = max(int(min_keep), min(k, n))
    return ordered[:k]


# ---------------------------------------------------------------------------
# PIPP stage: target finding, centering, cropping
# ---------------------------------------------------------------------------
def _foreground(gray, threshold):
    """Background-subtracted foreground weights (>=0) and the threshold used."""
    g = gray.astype(np.float64)
    if threshold is None:
        threshold = float(g.mean() + 2.0 * g.std())
    return np.clip(g - threshold, 0.0, None), threshold


def brightness_centroid(frame, threshold=None, roi=None):
    """Intensity-weighted centroid (cx, cy) of the bright target, or None.

    Pixels at or below ``threshold`` (default ``mean + 2*std``) are treated as
    sky and ignored, so the centroid locks onto the target rather than drifting
    toward the frame centre. Returns float pixel coords in full-frame space.
    """
    _require_cv2()
    gray = _to_gray(frame)
    x0 = y0 = 0
    if roi is not None:
        x0, y0, rw, rh = _clamp_roi(gray.shape, roi)
        gray = gray[y0:y0 + rh, x0:x0 + rw]
    weights, _ = _foreground(gray, threshold)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * weights).sum() / total) + x0
    cy = float((ys * weights).sum() / total) + y0
    return cx, cy


def bounding_box(frame, threshold=None, pad=0):
    """Tight (x, y, w, h) box around the bright target, padded, or None.

    Useful for PIPP-style cropping: shrink each frame to a small box around the
    target to cut file size and speed up everything downstream.
    """
    _require_cv2()
    gray = _to_gray(frame).astype(np.float64)
    if threshold is None:
        threshold = float(gray.mean() + 2.0 * gray.std())
    mask = gray > threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = gray.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def crop_centered(frame, center, size, pad_value=0):
    """Crop a fixed ``size`` box centred on ``center``, padding off-frame edges.

    ``size`` may be an int (square) or (w, h). The output is always exactly
    ``size`` regardless of how close ``center`` sits to an edge -- revealed area
    is filled with ``pad_value`` -- so cropped frames stack without resizing.
    """
    arr = np.asarray(frame)
    if isinstance(size, (int, float)):
        out_w = out_h = int(size)
    else:
        out_w, out_h = int(size[0]), int(size[1])
    cx, cy = float(center[0]), float(center[1])

    shape = (out_h, out_w) + arr.shape[2:]
    canvas = np.full(shape, pad_value, dtype=arr.dtype)

    # Source top-left so that ``center`` lands at the output centre.
    sx0 = int(round(cx - out_w / 2.0))
    sy0 = int(round(cy - out_h / 2.0))
    src_h, src_w = arr.shape[:2]

    # Overlap of the requested source window with the actual image.
    ix0 = max(0, sx0)
    iy0 = max(0, sy0)
    ix1 = min(src_w, sx0 + out_w)
    iy1 = min(src_h, sy0 + out_h)
    if ix1 <= ix0 or iy1 <= iy0:
        return canvas  # target window entirely off-frame
    # Where that overlap lands in the output canvas.
    ox0 = ix0 - sx0
    oy0 = iy0 - sy0
    canvas[oy0:oy0 + (iy1 - iy0), ox0:ox0 + (ix1 - ix0)] = arr[iy0:iy1, ix0:ix1]
    return canvas


def recenter_frame(frame, center=None, out_size=None, threshold=None, pad_value=0):
    """Shift the target to the canvas centre (PIPP centring / stabilising).

    ``center`` defaults to :func:`brightness_centroid`; if the target cannot be
    found the frame is returned (cropped to ``out_size`` about its geometric
    centre) so a blank frame never derails a batch. ``out_size`` defaults to the
    input size.
    """
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    if out_size is None:
        out_size = (w, h)
    if center is None:
        center = brightness_centroid(arr, threshold=threshold)
    if center is None:
        center = (w / 2.0, h / 2.0)
    return crop_centered(arr, center, out_size, pad_value=pad_value)


# ---------------------------------------------------------------------------
# AutoStakkert stage: alignment points (local distortion from seeing)
# ---------------------------------------------------------------------------
class AlignmentPointGrid:
    """A regular grid of alignment points laid over a frame.

    AutoStakkert places many alignment points across the target and tracks how
    each moves frame-to-frame, correcting *local* distortion from seeing rather
    than just a whole-frame shift. Keeping the points on a regular grid lets the
    measured per-point shifts be reshaped to ``(rows, cols)`` and upsampled into
    a dense displacement field with a single :func:`cv2.resize` -- no scattered
    interpolation dependency.
    """

    def __init__(self, points, rows, cols, shape):
        self.points = np.asarray(points, dtype=np.float64)  # (rows*cols, 2) x,y
        self.rows = int(rows)
        self.cols = int(cols)
        self.shape = tuple(shape[:2])

    def __len__(self):
        return len(self.points)

    @classmethod
    def over(cls, shape, spacing=80, margin=None):
        """Build a grid spanning ``shape`` (h, w) at roughly ``spacing`` px apart.

        At least a 2x2 grid is produced so the displacement field can be
        bilinearly upsampled.
        """
        h, w = shape[:2]
        spacing = max(1, int(spacing))
        if margin is None:
            margin = spacing // 2
        margin = int(max(0, min(margin, (min(h, w) - 1) // 2)))
        cols = max(2, 1 + (w - 2 * margin) // spacing)
        rows = max(2, 1 + (h - 2 * margin) // spacing)
        xs = np.linspace(margin, w - 1 - margin, cols)
        ys = np.linspace(margin, h - 1 - margin, rows)
        gx, gy = np.meshgrid(xs, ys)
        points = np.column_stack([gx.ravel(), gy.ravel()])
        return cls(points, rows, cols, shape)


def measure_local_shifts(ref_gray, frame_gray, points, patch=48, min_response=0.0):
    """Per-point sub-pixel shift of ``frame`` vs ``ref`` at each alignment point.

    For each point a square patch (side ``patch``) is cut from both images and
    registered with phase correlation. The returned ``(N, 2)`` array holds the
    ``(dx, dy)`` that maps a reference location to where that content sits in the
    frame -- i.e. ``frame`` sampled at ``ref_xy + shift`` lands back on ``ref``.
    Points whose correlation response is below ``min_response`` (or that fall too
    near an edge) report a zero shift.
    """
    _require_cv2()
    ref = ref_gray if ref_gray.ndim == 2 else _to_gray(ref_gray)
    cur = frame_gray if frame_gray.ndim == 2 else _to_gray(frame_gray)
    ref = ref.astype(np.float32)
    cur = cur.astype(np.float32)
    h, w = ref.shape
    half = int(patch) // 2
    win = cv2.createHanningWindow((2 * half, 2 * half), cv2.CV_32F)

    side = 2 * half
    if side > w or side > h:
        return np.zeros((len(points), 2), dtype=np.float64)  # patch bigger than image

    shifts = np.zeros((len(points), 2), dtype=np.float64)
    for i, (px, py) in enumerate(points):
        # Clamp the patch fully inside the image (points near the edge sample a
        # slightly off-centre window rather than being dropped, so every grid
        # node still contributes to the displacement field).
        x0 = int(np.clip(round(px) - half, 0, w - side))
        y0 = int(np.clip(round(py) - half, 0, h - side))
        a = ref[y0:y0 + 2 * half, x0:x0 + 2 * half]
        b = cur[y0:y0 + 2 * half, x0:x0 + 2 * half]
        (dx, dy), response = cv2.phaseCorrelate(a, b, win)
        if response < min_response:
            continue
        shifts[i] = (dx, dy)
    return shifts


def warp_by_grid(frame, grid, shifts, border=None):
    """Remap ``frame`` by a dense field interpolated from per-point ``shifts``.

    The coarse ``(rows, cols)`` shift grid is upsampled to full resolution and
    used as a displacement field: output pixel ``(x, y)`` samples the input at
    ``(x + dx, y + dy)``. This undoes the local distortion measured by
    :func:`measure_local_shifts` so the warped frame lines up with the reference
    before it is added to the stack.
    """
    _require_cv2()
    if border is None:
        border = cv2.BORDER_REFLECT
    h, w = frame.shape[:2]
    dx = shifts[:, 0].reshape(grid.rows, grid.cols).astype(np.float32)
    dy = shifts[:, 1].reshape(grid.rows, grid.cols).astype(np.float32)
    dx_full = cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
    dy_full = cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
    base_x, base_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = base_x + dx_full
    map_y = base_y + dy_full
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=border)


# ---------------------------------------------------------------------------
# AutoStakkert stage: the stacker
# ---------------------------------------------------------------------------
StackStats = namedtuple("StackStats", ["n_total", "n_stacked", "n_rejected"])


class LuckyStacker:
    """Align frames to a reference and average them into one high-SNR master.

    This is the AutoStakkert core. Each incoming frame is globally aligned to a
    fixed reference (via :class:`stabilizer.Stabilizer`, RANSAC similarity), and
    -- when ``local=True`` -- additionally corrected for seeing distortion with
    an :class:`AlignmentPointGrid`. Aligned frames are summed in float; the mean
    is the low-noise master. Frames that fail to align are dropped (counted in
    :attr:`stats`) rather than smeared into the result.

    Typical use is to feed only the sharpest top X% (see :func:`select_best`),
    with the sharpest frame as the reference.
    """

    def __init__(self, method="orb", local=False, ap_spacing=80, ap_patch=48,
                 full_affine=False, min_local_response=0.0):
        _require_cv2()
        self.method = method
        self.local = bool(local)
        self.ap_spacing = int(ap_spacing)
        self.ap_patch = int(ap_patch)
        self.full_affine = bool(full_affine)
        self.min_local_response = float(min_local_response)

        self._stab = None
        self._ref_gray = None
        self._grid = None
        self._accum = None
        self._count = 0
        self.n_total = 0
        self.n_rejected = 0

    # ------------------------------------------------------------- reference
    def set_reference(self, frame):
        """Anchor the stack on ``frame`` (also the first stacked frame)."""
        arr = _as_array(frame)
        if arr is None:
            raise ValueError("reference frame could not be decoded")
        self._stab = Stabilizer(method=self.method, full_affine=self.full_affine)
        self._stab.set_reference(arr)
        self._ref_gray = _to_gray(arr).astype(np.float32)
        self._accum = arr.astype(np.float64).copy()
        self._count = 1
        if self.local:
            self._grid = AlignmentPointGrid.over(arr.shape, self.ap_spacing)

    @property
    def has_reference(self):
        return self._stab is not None and self._stab.has_reference

    # --------------------------------------------------------------- adding
    def add(self, frame):
        """Align ``frame`` to the reference and add it. Returns True if stacked.

        The first ``add`` with no reference set adopts the frame as reference.
        """
        arr = _as_array(frame)
        self.n_total += 1
        if arr is None:
            self.n_rejected += 1
            return False
        if not self.has_reference:
            self.set_reference(arr)
            return True

        warped, _ = self._stab.stabilize(arr)
        if not self._stab.last_ok:
            self.n_rejected += 1
            return False
        if self.local and self._grid is not None:
            shifts = measure_local_shifts(
                self._ref_gray, _to_gray(warped), self._grid.points,
                patch=self.ap_patch, min_response=self.min_local_response)
            warped = warp_by_grid(warped, self._grid, shifts)
        self._accum += warped.astype(np.float64)
        self._count += 1
        return True

    def stack(self, frames):
        """Convenience: ``add`` every frame in ``frames`` and return the master."""
        for f in frames:
            self.add(f)
        return self.result()

    # --------------------------------------------------------------- output
    def result(self):
        """The averaged master frame (uint8 RGB), or None if nothing stacked."""
        if self._count == 0 or self._accum is None:
            return None
        return np.clip(self._accum / self._count, 0, 255).astype(np.uint8)

    @property
    def stats(self):
        return StackStats(self.n_total, self._count, self.n_rejected)


# ---------------------------------------------------------------------------
# Run-level glue: PIPP -> AutoStakkert over an on-disk Run
# ---------------------------------------------------------------------------
StackResult = namedtuple(
    "StackResult", ["master", "stats", "grades", "kept", "reference_index"])


def stack_run(run, cam_index, keep_fraction=0.5, keep_count=None,
              quality="laplacian", method="orb", local=False, roi=None,
              t_start=None, t_end=None, max_frames=None, progress=None):
    """Run the full lucky-imaging pipeline over one camera of a :class:`Run`.

    Steps: gather frames in the (optional) ``[t_start, t_end]`` window, grade
    them by ``quality`` sharpness (optionally within ``roi``), keep the sharpest
    ``keep_fraction`` (or ``keep_count``), use the single sharpest frame as the
    alignment reference, then align + average the kept set with
    :class:`LuckyStacker`.

    ``progress`` is an optional ``callback(done, total)`` for UIs. Returns a
    :class:`StackResult` whose ``master`` is the stacked image (None if no usable
    frames).
    """
    frames = run.frames(cam_index)
    if t_start is not None or t_end is not None:
        lo = -np.inf if t_start is None else t_start
        hi = np.inf if t_end is None else t_end
        lo, hi = min(lo, hi), max(lo, hi)
        frames = [f for f in frames if lo <= f["t"] <= hi]
    if max_frames is not None and len(frames) > max_frames:
        # Evenly subsample so the graded set still spans the whole capture.
        idx = np.linspace(0, len(frames) - 1, int(max_frames)).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]

    grader = QualityGrader(method=quality, roi=roi)
    grades = grader.grade(frames)
    if not grades:
        return StackResult(None, StackStats(0, 0, 0), [], [], None)

    kept = select_best(grades, fraction=None if keep_count else keep_fraction,
                       count=keep_count)
    reference_index = kept[0].index  # sharpest frame anchors the stack

    stacker = LuckyStacker(method=method, local=local)
    stacker.set_reference(frames[reference_index]["path"])
    total = len(kept)
    if progress:
        progress(1, total)
    for n, g in enumerate(kept[1:], start=2):
        stacker.add(frames[g.index]["path"])
        if progress:
            progress(n, total)

    return StackResult(stacker.result(), stacker.stats, grades, kept, reference_index)


def save_master(master, out_path):
    """Write a stacked master (RGB uint8) to disk as PNG/TIFF/etc. by extension."""
    _require_cv2()
    if master is None:
        raise ValueError("no master image to save")
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
        out_path = out_path + ".png"
    bgr = cv2.cvtColor(master, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(out_path, bgr):
        raise RuntimeError(f"could not write master image to {out_path}")
    return out_path
