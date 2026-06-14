#!/usr/bin/env python
"""
Cross-language parity for the Rust hotspot port (skytracker_core).

Runs the real numpy detector (hotspot.detect_hotspot) and the Rust port on the
SAME frames and asserts they agree:
  * detection vs no-detection (None) on every case,
  * sub-pixel centroid agreement (Python's own tests only require <1px vs
    ground truth; we require Rust within 0.05px of Python),
  * peak / background / noise / snr agreement.

Both sides receive a float32 2D intensity map (the Rust binding takes the
post-to_intensity frame; Python's detect_hotspot calls to_intensity internally,
which is idempotent on a 2D float32 array), so this isolates the detection
algorithm. Small float32-vs-f64 reduction differences are absorbed by tolerance;
n_pixels is allowed to differ by at most 1 borderline pixel.

Build first, then run:
    cd rust/skytracker_core && maturin develop --release
    python test_rust_hotspot_parity.py
"""

import math
import unittest

import numpy as np

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

from hotspot import detect_hotspot, pixel_offset_to_angles, to_intensity


def gaussian_image(w, h, cx, cy, amp, sigma, bg=10.0, noise=0.0, seed=0, channels=1):
    yy, xx = np.mgrid[0:h, 0:w]
    img = bg + amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise, img.shape)
    img = np.clip(img, 0, None)
    if channels == 3:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img.astype(np.float32)


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class RustHotspotParity(unittest.TestCase):

    def _assert_detections_match(self, py_d, rs_d, ctx=""):
        if py_d is None:
            self.assertIsNone(rs_d, f"Rust detected where Python did not [{ctx}]")
            return
        self.assertIsNotNone(rs_d, f"Rust missed a detection Python found [{ctx}]")
        self.assertAlmostEqual(py_d.cx, rs_d.cx, delta=0.05, msg=f"cx [{ctx}]")
        self.assertAlmostEqual(py_d.cy, rs_d.cy, delta=0.05, msg=f"cy [{ctx}]")
        self.assertAlmostEqual(py_d.peak, rs_d.peak, places=3, msg=f"peak [{ctx}]")
        self.assertAlmostEqual(py_d.background, rs_d.background, delta=1e-2,
                               msg=f"background [{ctx}]")
        rel = abs(py_d.snr - rs_d.snr) / max(1e-9, abs(py_d.snr))
        self.assertLess(rel, 0.01, f"snr rel diff {rel} [{ctx}]")
        self.assertLessEqual(abs(py_d.n_pixels - rs_d.n_pixels), 1, f"n_pixels [{ctx}]")

    def _run(self, img, ctx, **kw):
        intensity = to_intensity(img).astype(np.float32)
        py_d = detect_hotspot(img, **kw)
        # Translate gate kwargs for the Rust signature.
        gc = kw.get("gate_center")
        rs_d = rc.detect_hotspot(
            intensity,
            gate_center=(tuple(map(float, gc)) if gc is not None else None),
            gate_radius=(float(kw["gate_radius"]) if kw.get("gate_radius") is not None else None),
            snr_threshold=kw.get("snr_threshold", 5.0),
            centroid_radius=kw.get("centroid_radius", 12),
            half_max_fraction=kw.get("half_max_fraction", 0.5),
            min_pixels=kw.get("min_pixels", 3),
        )
        self._assert_detections_match(py_d, rs_d, ctx)

    def test_centroid_cases(self):
        cases = [
            (256, 200, 130.3, 90.7, 200, 3.0, 1.0, 1),
            (128, 128, 64.0, 40.0, 180, 2.5, 1.0, 1),
            (300, 220, 210.6, 33.2, 120, 4.0, 2.0, 1),
        ]
        for i, (w, h, cx, cy, amp, sig, ns, ch) in enumerate(cases):
            img = gaussian_image(w, h, cx, cy, amp, sig, noise=ns, seed=i + 1, channels=ch)
            self._run(img, f"centroid[{i}]", snr_threshold=5.0)

    def test_color_frame(self):
        img = gaussian_image(128, 128, 64, 40, 180, 2.5, noise=1.0, seed=2, channels=3)
        self._run(img, "color", snr_threshold=5.0)

    def test_pure_noise_none(self):
        rng = np.random.default_rng(3)
        img = (10 + rng.normal(0, 2.0, (200, 200))).astype(np.float32)
        self._run(img, "pure_noise", snr_threshold=8.0)

    def test_single_hot_pixel_rejected(self):
        rng = np.random.default_rng(5)
        img = (10 + rng.normal(0, 1.0, (128, 128))).astype(np.float32)
        img[64, 80] = 5000.0
        self._run(img, "hot_pixel", snr_threshold=5.0, min_pixels=3)

    def test_gate(self):
        img = gaussian_image(200, 200, 50, 50, 150, 3.0, noise=0.5, seed=4)
        bright = gaussian_image(200, 200, 150, 150, 255, 3.0, bg=0.0)
        img = (img + bright).astype(np.float32)
        self._run(img, "no_gate", snr_threshold=5.0)
        self._run(img, "gate", gate_center=(50, 50), gate_radius=30, snr_threshold=5.0)

    def test_geometry_parity(self):
        cases = [
            (100, 100, 4.0, 1000.0, 0.0, 0.0, 1.0, -1.0, True),
            (100, 0, 4.0, 1000.0, 90.0, 0.0, 1.0, -1.0, True),
            (100, 0, 4.0, 1000.0, 0.0, 60.0, 1.0, -1.0, True),
            (37, -52, 3.45, 800.0, 12.5, 35.0, -1.0, -1.0, True),
            (10, 10, 4.0, 0.0, 0.0, 0.0, 1.0, -1.0, True),  # zero focal length
        ]
        for c in cases:
            py = pixel_offset_to_angles(c[0], c[1], c[2], c[3], rotation_deg=c[4],
                                        el_deg=c[5], x_sign=c[6], y_sign=c[7],
                                        apply_cos_el=c[8])
            rs = rc.pixel_offset_to_angles(c[0], c[1], c[2], c[3], rotation_deg=c[4],
                                           el_deg=c[5], x_sign=c[6], y_sign=c[7],
                                           apply_cos_el=c[8])
            self.assertAlmostEqual(py[0], rs[0], places=9, msg=f"az {c}")
            self.assertAlmostEqual(py[1], rs[1], places=9, msg=f"el {c}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
