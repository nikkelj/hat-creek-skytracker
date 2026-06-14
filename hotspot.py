"""
Hot-spot detection and pixel->angle geometry for the closed-loop optical tracker.

These are pure functions over numpy intensity images (no pygame / camera /
hardware dependencies) so they can be unit-tested with synthetic frames. The
HOTSPOT tracking mode in joystick_controller uses them to lock onto a bright
("hot") object - a rocket plume, an aircraft, etc. - and drive the mount to keep
it centered.

Detection model: the target is assumed to be the brightest compact source in the
search region. That holds for hot objects; dim satellites among streaking stars
need a different approach and are out of scope here.
"""

import math
import numpy as np


class Detection:
    """Result of a successful hot-spot detection (image pixel coordinates)."""

    __slots__ = ("cx", "cy", "peak", "background", "noise", "snr", "n_pixels")

    def __init__(self, cx, cy, peak, background, noise, snr, n_pixels):
        self.cx = cx              # centroid x (column), full-image pixels
        self.cy = cy              # centroid y (row), full-image pixels
        self.peak = peak          # peak intensity in the blob
        self.background = background
        self.noise = noise        # robust noise estimate (~1 sigma)
        self.snr = snr            # (peak - background) / noise
        self.n_pixels = n_pixels  # pixels above the half-max threshold

    def __repr__(self):
        return (f"Detection(cx={self.cx:.2f}, cy={self.cy:.2f}, "
                f"snr={self.snr:.1f}, n={self.n_pixels})")


def to_intensity(image):
    """Reduce an HxW (mono) or HxWx3 (color) array to a 2D float32 intensity map."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a.astype(np.float32)


def _smooth3(a):
    """Cheap 3x3 box blur (edge-padded) to keep single hot pixels from winning
    the peak search. Used only for peak localization, not for the centroid."""
    p = np.pad(a, 1, mode="edge")
    return (p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
            p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def detect_hotspot(image, gate_center=None, gate_radius=None,
                   snr_threshold=5.0, centroid_radius=12,
                   half_max_fraction=0.5, min_pixels=3):
    """
    Detect the brightest compact hot spot in an image.

    Args:
        image: HxW or HxWx3 array.
        gate_center: optional (cx, cy) in full-image pixels to restrict the
            search to a window around the last known target (tracking gate).
        gate_radius: half-size of that window in pixels (required if gate_center).
        snr_threshold: minimum (peak-background)/noise to accept a detection.
        centroid_radius: half-size of the window (around the peak) used for the
            intensity-weighted centroid.
        half_max_fraction: pixels above background + frac*(peak-background) are
            included in the centroid.
        min_pixels: reject detections with fewer than this many bright pixels
            (filters single hot/defective pixels).

    Returns:
        Detection in full-image pixel coordinates, or None if nothing qualifies.
    """
    img = to_intensity(image)
    H, W = img.shape

    # Restrict to the tracking gate if provided.
    if gate_center is not None and gate_radius is not None:
        gx, gy = gate_center
        x0 = max(0, int(gx - gate_radius)); x1 = min(W, int(gx + gate_radius) + 1)
        y0 = max(0, int(gy - gate_radius)); y1 = min(H, int(gy + gate_radius) + 1)
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return None

    # Robust background and noise (median / MAD) over the search region.
    background = float(np.median(region))
    mad = float(np.median(np.abs(region - background)))
    noise = 1.4826 * mad if mad > 0 else (float(region.std()) or 1.0)

    # Locate the peak on a lightly smoothed copy so a lone hot pixel can't win.
    smooth = _smooth3(region)
    ry, rx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    peak = float(region[ry, rx])

    snr = (peak - background) / noise
    if snr < snr_threshold:
        return None

    # Intensity-weighted centroid in a small window around the peak.
    wy0 = max(0, ry - centroid_radius); wy1 = min(region.shape[0], ry + centroid_radius + 1)
    wx0 = max(0, rx - centroid_radius); wx1 = min(region.shape[1], rx + centroid_radius + 1)
    win = region[wy0:wy1, wx0:wx1]

    thresh = background + half_max_fraction * (peak - background)
    mask = win >= thresh
    n = int(mask.sum())
    if n < min_pixels:
        return None

    ys, xs = np.nonzero(mask)
    weights = (win[ys, xs] - background).astype(np.float64)
    wsum = weights.sum()
    if wsum <= 0:
        return None

    cx = float((xs * weights).sum() / wsum) + wx0 + x0
    cy = float((ys * weights).sum() / wsum) + wy0 + y0

    return Detection(cx, cy, peak, background, noise, float(snr), n)


def pixel_offset_to_angles(dx_pix, dy_pix, pixel_size_um, focal_length_mm,
                           rotation_deg=0.0, el_deg=0.0,
                           x_sign=1.0, y_sign=-1.0, apply_cos_el=True):
    """
    Convert an object's pixel offset from boresight into the mount-coordinate
    angular correction (az_error, el_error) in degrees needed to center it.

    Args:
        dx_pix, dy_pix: object centroid minus image center, in pixels
            (image x to the right, y downward).
        pixel_size_um: detector pixel pitch in microns.
        focal_length_mm: effective focal length in mm.
        rotation_deg: sensor rotation relative to the mount el/cross-el axes
            (the camera alignment_rotation).
        el_deg: current elevation, used to convert cross-elevation to azimuth.
        x_sign, y_sign: per-axis sign calibration (set on hardware so a positive
            error drives the mount the right way). Default y_sign=-1 because image
            rows increase downward while elevation increases upward.
        apply_cos_el: divide the azimuth term by cos(el) so az motion matches the
            on-sky cross-elevation rate.

    Returns:
        (az_error_deg, el_error_deg)
    """
    if focal_length_mm == 0:
        return 0.0, 0.0

    ifov_deg = math.degrees((pixel_size_um * 1e-3) / focal_length_mm)  # deg/pixel

    ax = dx_pix * ifov_deg * x_sign
    ay = dy_pix * ifov_deg * y_sign

    th = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    cross_el = ax * cos_t - ay * sin_t
    el_err = ax * sin_t + ay * cos_t

    if apply_cos_el:
        c = math.cos(math.radians(el_deg))
        az_err = cross_el / c if abs(c) > 1e-3 else cross_el
    else:
        az_err = cross_el

    return az_err, el_err
