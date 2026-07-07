#!/usr/bin/env python
"""
Finishing-stage tests: multi-scale unsharp, auto-stretch, 16-bit flow.
Run: python test_sharpen.py
"""

import unittest

import cv2
import numpy as np

from sharpen import (DEFAULT_LAYERS, auto_stretch, finish, save_final,
                     unsharp_layers, _to_float01)
from stacking import sharpness


def _soft_scene(w=160, h=120, seed=5):
    """A blurred, dim scene: what a linear stacked master looks like."""
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 60).astype(np.uint8)
    cv2.circle(img, (80, 60), 18, (140, 140, 140), -1)
    return cv2.GaussianBlur(img, (7, 7), 2.0)


class UnsharpTests(unittest.TestCase):

    def test_sharpening_increases_sharpness_metric(self):
        soft = _soft_scene()
        sharpened = (unsharp_layers(soft) * 255).astype(np.uint8)
        self.assertGreater(sharpness(sharpened), sharpness(soft) * 1.2)

    def test_zero_amount_is_identity(self):
        img = _soft_scene()
        out = unsharp_layers(img, layers=((1.0, 0.0),))
        np.testing.assert_allclose(out, _to_float01(img), atol=1e-6)

    def test_output_stays_in_range(self):
        out = unsharp_layers(_soft_scene(), layers=((1.0, 3.0), (4.0, 2.0)))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)


class StretchTests(unittest.TestCase):

    def test_stretch_brightens_dim_image(self):
        dim = (_to_float01(_soft_scene()) * 0.2).astype(np.float32)
        out = auto_stretch(dim)
        self.assertGreater(np.median(out), np.median(dim) * 2.0)

    def test_flat_image_survives(self):
        flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = auto_stretch(flat)
        self.assertTrue(np.isfinite(out).all())

    def test_uint16_input_preserves_sub8bit_detail(self):
        # Two levels that are identical at 8 bits (e.g. 100.2 vs 100.6 of 255)
        # but distinct at 16 bits must remain distinct after the stretch --
        # the entire reason the master is kept at 16 bits.
        a16 = np.full((8, 8, 3), int(100.2 * 257), dtype=np.uint16)
        b16 = np.full((8, 8, 3), int(100.6 * 257), dtype=np.uint16)
        sa = auto_stretch(np.concatenate([a16, b16], axis=1))
        left, right = sa[:, :8].mean(), sa[:, 8:].mean()
        self.assertGreater(right, left)


class FinishTests(unittest.TestCase):

    def test_finish_returns_uint8_same_shape(self):
        img = _soft_scene()
        out = finish(img)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, img.shape)

    def test_finish_accepts_uint16_master(self):
        img16 = (_to_float01(_soft_scene()) * 65535).astype(np.uint16)
        out = finish(img16)
        self.assertEqual(out.dtype, np.uint8)

    def test_save_final_roundtrip(self):
        import os
        import tempfile
        out = finish(_soft_scene())
        with tempfile.TemporaryDirectory() as d:
            p = save_final(out, os.path.join(d, "final.png"))
            back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            self.assertEqual(back.dtype, np.uint8)
            self.assertEqual(back.shape, out.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
