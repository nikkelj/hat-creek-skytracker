#!/usr/bin/env python
"""
Unit tests for the hot-spot detector and pixel->angle geometry (hotspot.py).

Pure numpy; no camera/pygame. Run: python test_hotspot.py
"""

import math
import unittest

import numpy as np

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


class IntensityTests(unittest.TestCase):

    def test_mono_passthrough_shape(self):
        a = np.zeros((10, 20), dtype=np.uint8)
        self.assertEqual(to_intensity(a).shape, (10, 20))

    def test_color_reduced_to_2d(self):
        a = np.zeros((10, 20, 3), dtype=np.uint8)
        self.assertEqual(to_intensity(a).ndim, 2)


class DetectTests(unittest.TestCase):

    def test_centroid_near_truth(self):
        img = gaussian_image(256, 200, cx=130.3, cy=90.7, amp=200, sigma=3.0,
                             noise=1.0, seed=1)
        d = detect_hotspot(img, snr_threshold=5.0)
        self.assertIsNotNone(d)
        self.assertLess(abs(d.cx - 130.3), 1.0)
        self.assertLess(abs(d.cy - 90.7), 1.0)

    def test_color_image_detected(self):
        img = gaussian_image(128, 128, cx=64, cy=40, amp=180, sigma=2.5,
                             noise=1.0, seed=2, channels=3)
        d = detect_hotspot(img, snr_threshold=5.0)
        self.assertIsNotNone(d)
        self.assertLess(abs(d.cx - 64), 1.5)
        self.assertLess(abs(d.cy - 40), 1.5)

    def test_pure_noise_returns_none(self):
        rng = np.random.default_rng(3)
        img = (10 + rng.normal(0, 2.0, (200, 200))).astype(np.float32)
        self.assertIsNone(detect_hotspot(img, snr_threshold=8.0))

    def test_gate_selects_dimmer_nearby_blob(self):
        img = gaussian_image(200, 200, cx=50, cy=50, amp=150, sigma=3.0, noise=0.5, seed=4)
        # Add a brighter blob far away.
        bright = gaussian_image(200, 200, cx=150, cy=150, amp=255, sigma=3.0, bg=0.0)
        img = (img + bright).astype(np.float32)

        # No gate -> locks the brightest (near 150,150).
        d_all = detect_hotspot(img, snr_threshold=5.0)
        self.assertIsNotNone(d_all)
        self.assertLess(abs(d_all.cx - 150), 2.0)

        # Gate around the dimmer blob -> locks it instead.
        d_gate = detect_hotspot(img, gate_center=(50, 50), gate_radius=30,
                                snr_threshold=5.0)
        self.assertIsNotNone(d_gate)
        self.assertLess(abs(d_gate.cx - 50), 2.0)
        self.assertLess(abs(d_gate.cy - 50), 2.0)

    def test_single_hot_pixel_rejected(self):
        rng = np.random.default_rng(5)
        img = (10 + rng.normal(0, 1.0, (128, 128))).astype(np.float32)
        img[64, 80] = 5000.0  # lone hot/defective pixel
        self.assertIsNone(detect_hotspot(img, snr_threshold=5.0, min_pixels=3))


class GeometryTests(unittest.TestCase):

    def setUp(self):
        self.px = 4.0       # microns
        self.fl = 1000.0    # mm
        self.ifov = math.degrees((self.px * 1e-3) / self.fl)  # deg/pixel

    def test_magnitude_and_default_signs(self):
        # +x (object right) -> +az; +y (object below) -> -el (default y_sign=-1)
        az, el = pixel_offset_to_angles(100, 100, self.px, self.fl,
                                        rotation_deg=0.0, el_deg=0.0)
        self.assertAlmostEqual(az, 100 * self.ifov, places=6)
        self.assertAlmostEqual(el, -100 * self.ifov, places=6)

    def test_rotation_90(self):
        # With y_sign=-1 and a 90 deg sensor rotation, a +x offset maps to +el.
        az, el = pixel_offset_to_angles(100, 0, self.px, self.fl,
                                        rotation_deg=90.0, el_deg=0.0)
        self.assertAlmostEqual(el, 100 * self.ifov, places=6)
        self.assertAlmostEqual(az, 0.0, places=6)

    def test_cos_el_scaling(self):
        az0, _ = pixel_offset_to_angles(100, 0, self.px, self.fl, el_deg=0.0)
        az60, _ = pixel_offset_to_angles(100, 0, self.px, self.fl, el_deg=60.0)
        self.assertAlmostEqual(az60, az0 / math.cos(math.radians(60.0)), places=6)

    def test_zero_focal_length_safe(self):
        self.assertEqual(pixel_offset_to_angles(10, 10, 4.0, 0.0), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
