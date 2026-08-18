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
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

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


def sharpness(frame, method="laplacian", roi=None, scale=1.0):
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
    scale
        <1.0 downsamples the (gray, ROI-cropped) frame before scoring. Since the
        score is used only to *rank* frames, a reduced-resolution pass is far
        cheaper and preserves ordering; use 1.0 for the most discriminating pass.

    Uses single-precision (``CV_32F``) accumulators -- the variance/energy is
    identical to double precision for ranking but roughly halves the cost on a
    full-frame image.
    """
    _require_cv2()
    gray = _to_gray(frame)
    if roi is not None:
        x, y, w, h = _clamp_roi(gray.shape, roi)
        gray = gray[y:y + h, x:x + w]
    if scale != 1.0 and gray.size:
        gh, gw = gray.shape
        nw, nh = max(1, int(gw * scale)), max(1, int(gh * scale))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    if gray.size == 0:
        return 0.0
    if method == "laplacian":
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if method == "tenengrad":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
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


def content_score(frame, scale=0.25, blur=2.0):
    """A noise-robust "is there real signal here" score (higher = more signal).

    Sharpness metrics are *fooled by noise*: a pure sensor-noise or hot-pixel
    frame has a huge Laplacian variance and would be ranked as the sharpest,
    while a faint but real target scores near zero -- so a naive top-X% cull
    would keep the junk and discard the good frames. This score instead measures
    the *prominence* of the brightest structure after denoising: the frame is
    reduced (``scale``, area-averaging beats down noise) and Gaussian-blurred
    (``blur``), then scored as ``max - median``. A real target/starfield keeps a
    prominent peak; uniform sky and structureless noise collapse toward zero.

    It is deliberately cheap (runs at reduced resolution) and is used only as a
    *conservative* first-pass gate to drop obviously empty/garbage frames, never
    to rank the good ones -- that is what :func:`sharpness` is for.
    """
    _require_cv2()
    gray = _to_gray(frame)
    if gray.size == 0:
        return 0.0
    gh, gw = gray.shape
    nw, nh = max(1, int(gw * scale)), max(1, int(gh * scale))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    if blur > 0:
        small = cv2.GaussianBlur(small, (0, 0), blur)
    return float(small.max() - np.median(small))


#: One graded frame. ``index`` is the position in the input sequence, ``score``
#: its sharpness, ``source`` the original path/dict (None for raw arrays), and
#: ``content`` its :func:`content_score` (None when not computed).
FrameGrade = namedtuple("FrameGrade", ["index", "score", "source", "content"])


class QualityGrader:
    """Score and rank a sequence of frames by sharpness (PIPP-style sorting).

    ``scale`` (<1.0) grades at reduced resolution for speed; the ordering is
    preserved so it is safe for ranking. When ``with_content`` is set, each grade
    also carries a :func:`content_score` for the garbage pre-cull.
    """

    def __init__(self, method="laplacian", roi=None, scale=1.0, with_content=False):
        if method not in SHARPNESS_METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.roi = roi
        self.scale = float(scale)
        self.with_content = bool(with_content)

    def score(self, frame):
        arr = _as_array(frame)
        return 0.0 if arr is None else sharpness(arr, self.method, self.roi, self.scale)

    def _grade_one(self, i, frame):
        arr = _as_array(frame)
        src = frame if not isinstance(frame, np.ndarray) else None
        if arr is None:
            return FrameGrade(i, 0.0, src, 0.0 if self.with_content else None)
        s = sharpness(arr, self.method, self.roi, self.scale)
        c = content_score(arr) if self.with_content else None
        return FrameGrade(i, s, src, c)

    def grade(self, frames, workers=1):
        """Grade frames, returning ``FrameGrade`` in input order.

        ``workers`` >1 decodes+scores frames on a thread pool. OpenCV releases
        the GIL during decode and the metric filters, so this scales close to
        linearly with cores on the decode-bound full-frame workload.
        """
        frames = list(frames)
        if workers and workers > 1 and len(frames) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(lambda p: self._grade_one(*p), enumerate(frames)))
        return [self._grade_one(i, f) for i, f in enumerate(frames)]


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


def prefilter_garbage(grades, ratio=0.12, min_keep=1):
    """Drop only *obviously* empty/garbage frames by :func:`content_score`.

    Returns ``(survivors, dropped)``. A frame is dropped when its content score
    falls below ``ratio`` times a robust "good frame" reference (the median of
    the top half of scores), so the cut adapts to the capture's signal level and
    is intentionally lenient -- a faint-but-real target sits well above the floor
    while structureless noise and blank sky fall below it. This runs *before* the
    precise sharpness ranking so noise frames (which score high on sharpness) can
    never sneak into the kept set. Grades lacking a content score are all kept.

    ``min_keep`` guarantees at least that many survivors (the highest-content
    frames) so a mostly-garbage capture still yields something to stack.
    """
    scored = [g for g in grades if g.content is not None]
    if not scored:
        return list(grades), []
    vals = np.sort([g.content for g in scored])
    ref = float(np.median(vals[len(vals) // 2:]))  # median of the top half
    floor = ratio * ref
    survivors = [g for g in grades if g.content is None or g.content >= floor]
    dropped = [g for g in grades if g.content is not None and g.content < floor]
    if len(survivors) < min_keep:  # keep the best-content frames regardless
        keep = sorted(scored, key=lambda g: g.content, reverse=True)[:min_keep]
        keep_ids = {id(g) for g in keep}
        survivors = [g for g in grades if id(g) in keep_ids]
        dropped = [g for g in grades if id(g) not in keep_ids]
    return survivors, dropped


# ---------------------------------------------------------------------------
# PIPP stage: target finding, centering, cropping
# ---------------------------------------------------------------------------
def _default_threshold(gray):
    """The shared 'sky' cut: pixels above ``mean + 2*std`` count as target."""
    g = gray.astype(np.float64)
    return float(g.mean() + 2.0 * g.std())


def _foreground(gray, threshold):
    """Background-subtracted foreground weights (>=0) and the threshold used."""
    g = gray.astype(np.float64)
    if threshold is None:
        threshold = _default_threshold(g)
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
        threshold = _default_threshold(gray)
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

    # Source top-left so that ``center`` lands on the output's centre pixel
    # (index ``out//2`` -- correct for odd sizes, not biased by a 0.5 round).
    sx0 = int(round(cx)) - out_w // 2
    sy0 = int(round(cy)) - out_h // 2
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
    def over(cls, shape, spacing=80):
        """Build a grid spanning ``shape`` (h, w) at roughly ``spacing`` px apart.

        Nodes run edge-to-edge (the first/last sit on pixel 0 and ``size-1``) so
        that upsampling the coarse shift grid with :func:`cv2.resize` lands each
        node's displacement back on the pixel it was measured at -- an inset grid
        would be stretched to the borders and apply every shift ~inset px off.
        Patches near an edge are clamped inward by :func:`measure_local_shifts`,
        so edge nodes are safe. At least a 2x2 grid is produced so the field can
        be bilinearly upsampled.
        """
        h, w = shape[:2]
        spacing = max(1, int(spacing))
        cols = max(2, 1 + (w - 1) // spacing)
        rows = max(2, 1 + (h - 1) // spacing)
        xs = np.linspace(0, w - 1, cols)
        ys = np.linspace(0, h - 1, rows)
        gx, gy = np.meshgrid(xs, ys)
        points = np.column_stack([gx.ravel(), gy.ravel()])
        return cls(points, rows, cols, shape)


def measure_local_shifts(ref_gray, frame_gray, points, patch=48, min_response=0.0,
                         max_shift=None):
    """Per-point sub-pixel shift of ``frame`` vs ``ref`` at each alignment point.

    For each point a square patch (side ``patch``) is cut from both images and
    registered with phase correlation. The returned ``(N, 2)`` array holds the
    ``(dx, dy)`` that maps a reference location to where that content sits in the
    frame -- i.e. ``frame`` sampled at ``ref_xy + shift`` lands back on ``ref``.
    Points whose correlation response is below ``min_response`` report a zero
    shift. ``max_shift`` (px) rejects implausibly large per-point shifts -- a
    structureless sky patch can phase-correlate to garbage, and without this
    guard one bad node would smear a whole band of the master via the upsampled
    field; such points fall back to a zero shift, mirroring how
    :class:`stabilizer.Stabilizer` rejects implausible global fits.
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
        # .copy() is load-bearing: cv2.phaseCorrelate MUTATES its inputs in
        # place (OpenCV 4.8 applies the window into the caller's buffers).
        # These slices are only safe today because non-contiguous views get
        # copied by the numpy bridge -- a full-width window would pass a
        # contiguous view and corrupt the frame. See LEARNINGS 2026-08-17.
        a = ref[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        b = cur[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        (dx, dy), response = cv2.phaseCorrelate(a, b, win)
        if response < min_response:
            continue
        if max_shift is not None and (abs(dx) > max_shift or abs(dy) > max_shift):
            continue  # implausible -> treat as no local correction at this node
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

    Averaging is **per-pixel coverage weighted**: aligning a frame shifts real
    pixels in and leaves black (BORDER_CONSTANT) at the revealed edge, so each
    contribution is masked and a per-pixel weight is accumulated alongside the
    sum. Dividing sum by weight (rather than a single frame count) keeps the
    shifted black borders from darkening the master's edges -- the reference
    covers the whole frame, so every pixel keeps a valid average.

    Typical use is to feed only the sharpest top X% (see :func:`select_best`),
    with the sharpest frame as the reference.
    """

    def __init__(self, method="orb", local=False, ap_spacing=80, ap_patch=48,
                 full_affine=False, min_local_response=0.0, max_local_shift=None,
                 align_scale=1.0):
        _require_cv2()
        self.method = method
        self.local = bool(local)
        self.ap_spacing = int(ap_spacing)
        self.ap_patch = int(ap_patch)
        self.full_affine = bool(full_affine)
        self.min_local_response = float(min_local_response)
        # Default cap on a single alignment point's shift: a quarter of the patch.
        self.max_local_shift = (self.ap_patch / 4.0 if max_local_shift is None
                                else float(max_local_shift))
        # <1.0 estimates the global transform on a downscaled frame (feature
        # detection is the dominant cost and scales ~quadratically), then warps
        # the FULL-res frame -- so the master stays full resolution. Only the
        # global-shift *estimate* loses precision; measured quality loss is small
        # (see tests), but 1.0 is the default when maximum sharpness matters.
        self.align_scale = float(align_scale)

        self._stab = None
        self._ref_gray = None
        self._grid = None
        self._accum = None       # float64 HxWx3 weighted sum
        self._weight = None      # float64 HxW per-pixel coverage
        self._count = 0
        self.n_rejected = 0

    # ------------------------------------------------------------- reference
    def set_reference(self, frame, seed_stack=True):
        """Anchor the stack on ``frame``.

        With ``seed_stack`` (default) the reference is also the first stacked
        frame. Pass ``seed_stack=False`` to build only the alignment anchor with
        an empty accumulator -- used by the parallel map-reduce path so several
        worker stackers can share one reference without counting it N times; one
        worker seeds it, the rest start empty and are :meth:`merge`\\ d in.
        """
        arr = _as_array(frame)
        if arr is None:
            raise ValueError("reference frame could not be decoded")
        self._stab = Stabilizer(method=self.method, full_affine=self.full_affine)
        # Detect features at align_scale so incoming frames match; keep the
        # full-res gray for local alignment-point measurement.
        self._stab.set_reference(self._align_gray(arr))
        self._ref_gray = _to_gray(arr).astype(np.float32)
        if seed_stack:
            self._accum = arr.astype(np.float64).copy()
            self._weight = np.ones(arr.shape[:2], dtype=np.float64)  # full coverage
            self._count = 1
        else:
            self._accum = np.zeros(arr.shape, dtype=np.float64)
            self._weight = np.zeros(arr.shape[:2], dtype=np.float64)
            self._count = 0
        if self.local:
            self._grid = AlignmentPointGrid.over(arr.shape, self.ap_spacing)

    def merge(self, other):
        """Fold another stacker's partial sums into this one (map-reduce join).

        Both must share the same reference frame shape. Used to recombine the
        per-worker partial stacks produced by the parallel path.
        """
        if other._accum is None:
            return self
        if self._accum is None:
            self._accum = other._accum.copy()
            self._weight = other._weight.copy()
        else:
            self._accum += other._accum
            self._weight += other._weight
        self._count += other._count
        self.n_rejected += other.n_rejected
        return self

    @property
    def has_reference(self):
        return self._stab is not None and self._stab.has_reference

    def _align_gray(self, arr):
        """Gray frame at the alignment scale (identity RGB when align_scale==1)."""
        if self.align_scale == 1.0:
            return arr
        g = _to_gray(arr)
        nh = max(1, int(g.shape[0] * self.align_scale))
        nw = max(1, int(g.shape[1] * self.align_scale))
        return cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)

    def _align(self, arr):
        """Return (warped_full_res, M_full_res, ok) aligning ``arr`` to the ref."""
        if self.align_scale == 1.0:
            warped, M = self._stab.stabilize(arr)
            return warped, M, self._stab.last_ok
        # Estimate on the downscaled frame, then apply to the full-res frame.
        M, _ = self._stab.estimate_transform(self._align_gray(arr))
        if M is None:
            return None, None, False
        M = M.copy()
        M[0, 2] /= self.align_scale     # only translation is resolution-dependent
        M[1, 2] /= self.align_scale
        if not self._stab._validate(M, arr.shape):
            return None, None, False
        h, w = arr.shape[:2]
        warped = cv2.warpAffine(
            arr, M.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return warped, M, True

    # --------------------------------------------------------------- adding
    def add(self, frame):
        """Align ``frame`` to the reference and add it. Returns True if stacked.

        The first ``add`` with no reference set adopts the frame as reference.
        """
        arr = _as_array(frame)
        if not self.has_reference:
            if arr is None:
                self.n_rejected += 1
                return False
            self.set_reference(arr)
            return True
        if arr is None:
            self.n_rejected += 1
            return False

        warped, M, ok = self._align(arr)
        if not ok:
            self.n_rejected += 1
            return False

        # Coverage mask: where the aligning warp brought in real (non-border) px.
        h, w = arr.shape[:2]
        cover = cv2.warpAffine(
            np.ones((h, w), np.float32), M.astype(np.float32), (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if self.local and self._grid is not None:
            shifts = measure_local_shifts(
                self._ref_gray, _to_gray(warped), self._grid.points,
                patch=self.ap_patch, min_response=self.min_local_response,
                max_shift=self.max_local_shift)
            warped = warp_by_grid(warped, self._grid, shifts)
            cover = warp_by_grid(cover, self._grid, shifts, border=cv2.BORDER_CONSTANT)
        cover = np.clip(cover, 0.0, 1.0)
        self._accum += warped.astype(np.float64) * cover[..., None]
        self._weight += cover
        self._count += 1
        return True

    def stack(self, frames):
        """Convenience: ``add`` every frame in ``frames`` and return the master."""
        for f in frames:
            self.add(f)
        return self.result()

    # --------------------------------------------------------------- output
    def result(self, bits=8):
        """The averaged master frame (RGB), or None if nothing stacked.

        ``bits=8`` (default) returns uint8. ``bits=16`` returns uint16 scaled
        by 257 (255 -> 65535): a stack of N frames carries ~sqrt(N) more SNR
        than one frame, and quantizing the mean back to 8 bits throws away
        everything below 1 LSB -- the 16-bit master is the linear input the
        sharpening/stretch stage (sharpen.py) is designed to consume.
        """
        if self._count == 0 or self._accum is None:
            return None
        denom = np.maximum(self._weight, 1e-6)[..., None]
        mean = self._accum / denom
        if bits == 16:
            return np.clip(mean * 257.0, 0, 65535).astype(np.uint16)
        return np.clip(mean, 0, 255).astype(np.uint8)

    @property
    def stats(self):
        # n_total is derived so it always reconciles: every frame offered to the
        # stacker is either stacked (incl. the reference) or rejected.
        return StackStats(self._count + self.n_rejected, self._count, self.n_rejected)


# ---------------------------------------------------------------------------
# Run-level glue: PIPP -> AutoStakkert over an on-disk Run
# ---------------------------------------------------------------------------
StackResult = namedtuple(
    "StackResult",
    ["master", "stats", "grades", "kept", "reference_index", "dropped"])


def _stack_parallel(ref_arr, frame_specs, method, local, workers, on_advance=None,
                    align_scale=1.0):
    """Map-reduce stack: K workers each align+accumulate a slice, then merge.

    All workers share one alignment reference (built K times, a one-off cost);
    only worker 0 seeds the reference into its accumulator so the merged master
    counts it exactly once. Accumulation is per-worker (no shared mutable state),
    so there is no lock on the hot path -- the partial float sums are added at
    the end. OpenCV releases the GIL during detect/warp, so this scales with
    cores on the align-bound stack pass.
    """
    n = len(frame_specs)
    k = max(1, min(int(workers or 1), n if n else 1))
    stackers = []
    for i in range(k):
        s = LuckyStacker(method=method, local=local, align_scale=align_scale)
        s.set_reference(ref_arr, seed_stack=(i == 0))
        stackers.append(s)
    chunks = [frame_specs[i::k] for i in range(k)]

    def work(i):
        s = stackers[i]
        for spec in chunks[i]:
            s.add(spec)
            if on_advance:
                on_advance()
        return s

    if k == 1:
        work(0)
    else:
        # Trim OpenCV's own thread pool so K Python workers don't oversubscribe.
        old = cv2.getNumThreads()
        cv2.setNumThreads(max(1, old // k))
        try:
            with ThreadPoolExecutor(max_workers=k) as ex:
                list(ex.map(work, range(k)))
        finally:
            cv2.setNumThreads(old)

    master = stackers[0]
    for s in stackers[1:]:
        master.merge(s)
    return master


def stack_run(run, cam_index, keep_fraction=0.5, keep_count=None,
              quality="laplacian", method="orb", local=False, roi=None,
              t_start=None, t_end=None, max_frames=None, progress=None,
              workers=None, grade_scale=1.0, prefilter=True, prefilter_ratio=0.12,
              align_scale=1.0, bits=8, center_size=None):
    """Run the full lucky-imaging pipeline over one camera of a :class:`Run`.

    Two-tier culling keeps a fast first pass from throwing away good data:

    1. **Grade** every frame in the ``[t_start, t_end]`` window in parallel --
       a precise ``quality`` sharpness score (optionally at ``grade_scale`` for
       speed, within ``roi``) plus a noise-robust :func:`content_score`.
    2. **Pre-cull garbage** (``prefilter``): drop only frames with essentially no
       signal (blank sky, structureless noise) via :func:`prefilter_garbage`.
       This must precede ranking because sharpness *rewards* noise, so a naive
       top-X% would otherwise keep hot-pixel/noise frames and discard faint but
       real ones.
    3. **Rank & keep** the sharpest ``keep_fraction`` (or ``keep_count``) of the
       survivors; the single sharpest anchors the alignment.
    4. **Align + average** the kept set into a high-SNR master, fanned out across
       ``workers`` cores (defaults to all).

    ``progress`` is an optional ``callback(done, total)`` for UIs. Returns a
    :class:`StackResult`; ``dropped`` lists the frames the pre-cull removed.

    ``bits`` selects the master depth (8 or 16 -- see LuckyStacker.result).

    ``center_size`` (int, e.g. 512) enables PIPP-style target centring: each
    kept frame is recentred on its brightness centroid and cropped to a
    ``center_size`` square before alignment. For a fast-moving target this is
    both a quality win (alignment sees the target, not the star field the
    mount is slewing across) and a large speed win (ORB cost scales with
    area). Kept frames are decoded up-front in this mode, so budget
    ``center_size^2 * 3 * n_kept`` bytes of RAM.
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

    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    grader = QualityGrader(method=quality, roi=roi, scale=grade_scale,
                           with_content=prefilter)
    grades = grader.grade(frames, workers=workers)
    if not grades:
        return StackResult(None, StackStats(0, 0, 0), [], [], None, [])

    if prefilter:
        survivors, dropped = prefilter_garbage(grades, ratio=prefilter_ratio)
    else:
        survivors, dropped = list(grades), []

    kept = select_best(survivors,
                       fraction=keep_fraction if keep_count is None else None,
                       count=keep_count)
    reference_index = kept[0].index  # sharpest survivor anchors the stack
    ref_arr = _as_array(frames[reference_index])  # decode the reference once

    if center_size:
        size = int(center_size)

        def _centred(arr):
            return recenter_frame(arr, out_size=(size, size))

        ref_arr = _centred(ref_arr)
        to_stack = [_centred(_as_array(frames[g.index])) for g in kept[1:]]
    else:
        to_stack = [frames[g.index] for g in kept[1:]]

    total = len(kept)
    counter = {"n": 1}
    lock = threading.Lock()
    if progress:
        progress(1, total)

    def advance():
        if not progress:
            return
        with lock:
            counter["n"] += 1
            progress(counter["n"], total)

    stacker = _stack_parallel(ref_arr, to_stack, method, local, workers, advance,
                              align_scale=align_scale)

    return StackResult(stacker.result(bits=bits), stacker.stats, grades, kept,
                       reference_index, dropped)


def save_master(master, out_path):
    """Write a stacked master (RGB uint8 or uint16) to disk by extension.

    uint16 masters require a 16-bit-capable format (PNG/TIFF); the extension
    is coerced to .png if the caller asked for one that can't hold 16 bits.
    """
    _require_cv2()
    if master is None:
        raise ValueError("no master image to save")
    ext = os.path.splitext(out_path)[1].lower()
    if master.dtype == np.uint16:
        if ext not in (".png", ".tif", ".tiff"):
            out_path = os.path.splitext(out_path)[0] + ".png"
    elif ext not in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
        out_path = out_path + ".png"
    bgr = cv2.cvtColor(master, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(out_path, bgr):
        raise RuntimeError(f"could not write master image to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI: headless batch stacking ("good shot posted fast" from a terminal)
# ---------------------------------------------------------------------------
def _cli(argv=None):
    """Stack one run directory -- or every run in a library directory -- to
    16-bit linear masters plus sharpened/stretched share-ready finals.

        python stacking.py data/MYSAT_12345_2026_07_03_01_02_03
        python stacking.py data --all --keep 0.25 --center 512 --local
    """
    import argparse

    from post_process import Run, RunLibrary
    from sharpen import finish as _finish, save_final

    p = argparse.ArgumentParser(
        description="Lucky-imaging stack: cull -> align -> average -> sharpen.")
    p.add_argument("path", help="a run directory, or a library dir with --all")
    p.add_argument("--all", action="store_true",
                   help="treat PATH as a library dir and stack every run in it")
    p.add_argument("--cam", type=int, default=None,
                   help="camera index (default: every camera in the run)")
    p.add_argument("--keep", type=float, default=0.5,
                   help="fraction of (non-garbage) frames to keep (default 0.5)")
    p.add_argument("--local", action="store_true",
                   help="alignment-point local warping (extended targets)")
    p.add_argument("--center", type=int, default=None, metavar="PX",
                   help="PIPP centring: recenter+crop each frame to PX square")
    p.add_argument("--grade-scale", type=float, default=0.5,
                   help="grade sharpness at this scale (default 0.5; 1.0 = full res)")
    p.add_argument("--align-scale", type=float, default=1.0,
                   help="estimate global alignment at this scale (default 1.0)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel workers (default: all cores)")
    p.add_argument("--no-finish", action="store_true",
                   help="skip the sharpened/stretched 8-bit final")
    p.add_argument("--out", default="exports", help="output directory")
    args = p.parse_args(argv)

    if args.all:
        lib = RunLibrary(args.path)
        lib.scan()
        runs = lib.runs
    else:
        runs = [Run(args.path)]
    os.makedirs(args.out, exist_ok=True)

    n_ok = 0
    for run in runs:
        cams = [args.cam] if args.cam is not None else list(run.camera_indices)
        for cam in cams:
            name = os.path.basename(os.path.normpath(run.path))
            tag = f"{name}_cam{cam + 1}"
            try:
                result = stack_run(
                    run, cam, keep_fraction=args.keep, local=args.local,
                    center_size=args.center, grade_scale=args.grade_scale,
                    align_scale=args.align_scale, workers=args.workers,
                    bits=16)
                if result.master is None:
                    print(f"[skip] {tag}: nothing stackable")
                    continue
                master_path = save_master(
                    result.master, os.path.join(args.out, f"{tag}_master16.png"))
                line = (f"[ok]   {tag}: stacked {result.stats.n_stacked}"
                        f"/{result.stats.n_total} (pre-culled {len(result.dropped)})"
                        f" -> {master_path}")
                if not args.no_finish:
                    final_path = save_final(
                        _finish(result.master),
                        os.path.join(args.out, f"{tag}_final.png"))
                    line += f" + {final_path}"
                print(line)
                n_ok += 1
            except Exception as e:
                print(f"[FAIL] {tag}: {e}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
