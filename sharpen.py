"""
Finishing stage for stacked masters: multi-scale sharpening + auto-stretch.

The stacker deliberately produces a LINEAR master (no gamma/brightness baked
in) because averaging must happen on linear data -- but a linear master looks
soft and dark. This module is the missing second half of "get a good shot
posted fast": wavelet-style multi-scale unsharp sharpening (the ~AutoStakkert/
Registax step) followed by a deterministic histogram stretch, producing a
share-ready 8-bit image.

Everything works in float32 [0, 1] internally and accepts uint8 or uint16
masters (16-bit input preserves the stacked SNR through the stretch, which is
exactly why LuckyStacker.result(bits=16) exists).

    from stacking import stack_run, save_master
    from sharpen import finish, save_final

    result = stack_run(run, 0, bits=16)
    save_master(result.master, "master16.tif")     # archival linear master
    save_final(finish(result.master), "final.png") # share-ready image
"""

import os

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False


def _require_cv2():
    if not CV2_AVAILABLE:
        raise RuntimeError("sharpen requires opencv-python (cv2)")


def _to_float01(img):
    """uint8/uint16/float image -> float32 in [0, 1]."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _to_uint8(img01):
    return np.clip(np.asarray(img01) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# Default sharpening layers: (gaussian sigma px, amount). Small sigma bites on
# fine detail, larger sigmas add local contrast -- the classic wavelet-layer
# recipe, conservative enough not to ring on point sources.
DEFAULT_LAYERS = ((1.0, 0.6), (2.5, 0.35), (6.0, 0.15))


def unsharp_layers(img, layers=DEFAULT_LAYERS):
    """Multi-scale unsharp mask on a linear image.

    For each ``(sigma, amount)`` layer the detail at that scale
    (``img - blur(img, sigma)``) is boosted by ``amount``. Returns float32
    [0, 1]; input may be uint8/uint16/float.
    """
    _require_cv2()
    out = _to_float01(img)
    for sigma, amount in layers:
        if amount <= 0.0 or sigma <= 0.0:
            continue
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=float(sigma))
        out = out + float(amount) * (out - blurred)
    return np.clip(out, 0.0, 1.0)


def auto_stretch(img, black_pct=0.25, white_pct=99.9, target_median=0.25,
                 max_gamma=5.0):
    """Deterministic display stretch for a linear image.

    1. Black/white points from luminance percentiles (``black_pct`` /
       ``white_pct``) rescale the data to full range.
    2. A gamma is solved so the (non-background) median lands at
       ``target_median`` -- the "make the faint stuff visible" step --
       clamped to ``max_gamma`` so a nearly-empty frame isn't stretched
       into pure noise.

    Returns float32 [0, 1].
    """
    out = _to_float01(img)
    lum = out.mean(axis=2) if out.ndim == 3 else out
    lo = float(np.percentile(lum, black_pct))
    hi = float(np.percentile(lum, white_pct))
    if hi <= lo:
        return out
    out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)

    lum = out.mean(axis=2) if out.ndim == 3 else out
    fg = lum[lum > 0.001]
    med = float(np.median(fg)) if fg.size else 0.0
    if 0.0 < med < target_median:
        # x**g with g<1 brightens; solve med**g == target and clamp so a
        # nearly-empty frame isn't stretched into pure noise.
        g = np.log(target_median) / np.log(med)
        g = max(float(g), 1.0 / float(max_gamma))
        out = np.power(out, g)
    return np.clip(out, 0.0, 1.0)


def finish(master, layers=DEFAULT_LAYERS, stretch=True, black_pct=0.25,
           white_pct=99.9, target_median=0.25):
    """Sharpen + stretch a linear master into a share-ready uint8 RGB image."""
    out = unsharp_layers(master, layers=layers)
    if stretch:
        out = auto_stretch(out, black_pct=black_pct, white_pct=white_pct,
                           target_median=target_median)
    return _to_uint8(out)


def save_final(img8, out_path):
    """Write a finished uint8 RGB image (PNG/JPG by extension)."""
    _require_cv2()
    if img8 is None:
        raise ValueError("no image to save")
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        out_path = out_path + ".png"
    if not cv2.imwrite(out_path, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write image to {out_path}")
    return out_path
