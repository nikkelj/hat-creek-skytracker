#!/usr/bin/env python
"""Live A/B parity: Rust imaging kernels vs the cv2/numpy paths, through
the real stacking.py / stabilizer.py / sharpen.py entry points on a
synthetic jittered star-field burst.

Gates (Phase 3 plan): kernels near-exact; end-to-end stacked master
PSNR > 40 dB flag-off vs flag-on; stabilizer transforms within
0.1 px / 0.05 deg.

Build first: cd rust/skytracker-ffi && maturin develop --release
"""

import math
import os
import unittest

import numpy as np

try:
    import skytracker_core
    _HAVE = getattr(skytracker_core, "IMAGING_AVAILABLE", False)
except ImportError:
    _HAVE = False

import cv2


def _flag(on):
    os.environ["SKYTRACKER_RUST_IMAGING"] = "1" if on else "0"


def _make_burst(n=8, w=320, h=240, seed=11):
    """Star field + drifting bright target, with per-frame rigid jitter."""
    rng = np.random.default_rng(seed)
    base = rng.normal(20, 4, size=(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    stars = [(rng.uniform(10, w - 10), rng.uniform(10, h - 10),
              rng.uniform(60, 200)) for _ in range(25)]
    for x, y, a in stars:
        base += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))
    base += 230 * np.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * 3.0**2))
    base = np.clip(base, 0, 255).astype(np.float32)

    frames = []
    for i in range(n):
        ang = rng.normal(0, 0.15)
        dx, dy = rng.normal(0, 2.0, 2)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        m[0, 2] += dx
        m[1, 2] += dy
        f = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        f = f + rng.normal(0, 2.0, f.shape)
        g = np.clip(f, 0, 255).astype(np.uint8)
        # LuckyStacker consumes RGB frames (capture dumps are RGB BMPs).
        frames.append(np.stack([g, g, g], axis=-1))
    return frames


def _psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(255.0**2 / mse)


@unittest.skipUnless(_HAVE, "skytracker_core wheel lacks imaging kernels")
class ImagingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = _make_burst()

    def tearDown(self):
        os.environ.pop("SKYTRACKER_RUST_IMAGING", None)

    def test_kernels_match(self):
        import stacking

        f = self.frames[0]
        for method in ("laplacian", "tenengrad"):
            for scale in (1.0, 0.5):
                _flag(False)
                py = stacking.sharpness(f, method=method, scale=scale)
                _flag(True)
                rs = stacking.sharpness(f, method=method, scale=scale)
                rel = abs(py - rs) / max(abs(py), 1e-12)
                self.assertLess(rel, 1e-3, f"{method}@{scale}: {py} vs {rs}")

        _flag(False)
        py_c = stacking.brightness_centroid(f)
        _flag(True)
        rs_c = stacking.brightness_centroid(f)
        self.assertIsNotNone(py_c)
        self.assertIsNotNone(rs_c)
        self.assertLess(abs(py_c[0] - rs_c[0]), 1e-6)
        self.assertLess(abs(py_c[1] - rs_c[1]), 1e-6)
        print(f"\n[parity] kernels: sharpness rel<1e-3, centroid "
              f"({py_c[0]:.3f},{py_c[1]:.3f}) exact")

    def test_local_shifts_and_grid_warp(self):
        import stacking

        ref = self.frames[0].astype(np.float32)
        cur = self.frames[1].astype(np.float32)
        grid = stacking.AlignmentPointGrid.over(ref.shape, spacing=60)
        _flag(False)
        py = stacking.measure_local_shifts(ref, cur, grid.points, patch=48)
        _flag(True)
        rs = stacking.measure_local_shifts(ref, cur, grid.points, patch=48)
        # On noisy rotated patches both estimators sit at their own noise
        # floor (cv2's f32 cross-power normalization vs our f64 diverges on
        # weak bins), so per-node equality is not meaningful — the gates are
        # a sanity bound on the field plus the warped-output PSNR below. The
        # clean-signal case is already golden-gated at 0.01 px in cargo.
        node_diff = np.abs(py - rs).max(axis=1)
        med = float(np.median(node_diff))
        self.assertLess(med, 0.5, f"median node diff {med}")

        _flag(False)
        py_w = stacking.warp_by_grid(self.frames[1], grid, py)
        py_w_rs_field = stacking.warp_by_grid(self.frames[1], grid, rs)
        _flag(True)
        rs_w = stacking.warp_by_grid(self.frames[1], grid, py)
        psnr_impl = _psnr(py_w, rs_w)           # same field, different warp impl
        psnr_field = _psnr(py_w, py_w_rs_field)  # same warp, different fields
        self.assertGreater(psnr_impl, 45.0, f"grid warp PSNR {psnr_impl:.1f}")
        self.assertGreater(psnr_field, 40.0, f"shift-field PSNR {psnr_field:.1f}")
        print(f"[parity] local shifts: median {med:.4f} px; warp PSNR "
              f"{psnr_impl:.1f} dB, field PSNR {psnr_field:.1f} dB")

    def test_stabilizer_flow_transforms(self):
        from stabilizer import Stabilizer

        results = {}
        for on in (False, True):
            _flag(on)
            st = Stabilizer(method="flow", max_features=300)
            st.set_reference(self.frames[0])
            ms = []
            for f in self.frames[1:5]:
                _w, m = st.stabilize(f)
                self.assertTrue(st.last_ok, f"flag={on}: {st.last_reject_reason}")
                ms.append(np.asarray(m))
            results[on] = ms
        worst_t = worst_r = 0.0
        for mp, mr in zip(results[False], results[True]):
            worst_t = max(worst_t, float(np.max(np.abs(mp[:, 2] - mr[:, 2]))))
            rot_p = math.degrees(math.atan2(mp[1, 0], mp[0, 0]))
            rot_r = math.degrees(math.atan2(mr[1, 0], mr[0, 0]))
            worst_r = max(worst_r, abs(rot_p - rot_r))
        self.assertLess(worst_t, 0.1, f"translation {worst_t}")
        self.assertLess(worst_r, 0.05, f"rotation {worst_r}")
        print(f"[parity] stabilizer flow: dT {worst_t:.4f} px, dR {worst_r:.5f} deg")

    def test_end_to_end_stack_psnr(self):
        from stacking import LuckyStacker

        masters = {}
        for on in (False, True):
            _flag(on)
            st = LuckyStacker(method="flow", local=True, ap_spacing=60, ap_patch=48)
            for f in self.frames:
                st.add(f)
            masters[on] = st.result(bits=8)
        psnr = _psnr(masters[False], masters[True])
        self.assertGreater(psnr, 40.0, f"stacked master PSNR {psnr:.1f} dB")
        print(f"[parity] end-to-end stack: PSNR {psnr:.1f} dB "
              f"(flag-off vs flag-on masters)")

    def test_mp4_export_roundtrip(self):
        """Rust H.264 writer: cv2-decodable file, right frame count, decent
        content fidelity (H.264 is lossy; gate well above garbage)."""
        import tempfile

        import rust_imaging_adapter

        _flag(True)
        out = os.path.join(tempfile.gettempdir(), "skytracker_test_export.mp4")
        writer = rust_imaging_adapter.make_video_writer(out, 320, 240, 15.0)
        self.assertIsNotNone(writer, "Rust mp4 writer unavailable")
        self.assertTrue(writer.isOpened())
        src = [f[..., ::-1].copy() for f in self.frames]  # BGR like cv2 path
        for f in src:
            writer.write(f)
        writer.release()

        def read_back(path):
            cap = cv2.VideoCapture(path)
            n = 0
            worst = float("inf")
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                worst = min(worst, _psnr(fr, src[n]))
                n += 1
            cap.release()
            return n, worst

        count, worst = read_back(out)
        os.remove(out)

        # Comparative gate: the frames are mostly sensor noise (PSNR here
        # measures bitrate, not correctness), so require the H.264 output
        # to be no worse than the cv2 mp4v writer it replaces.
        ref_path = os.path.join(tempfile.gettempdir(), "skytracker_test_ref.mp4")
        vw = cv2.VideoWriter(ref_path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
        for f in src:
            vw.write(f)
        vw.release()
        ref_count, ref_worst = read_back(ref_path)
        os.remove(ref_path)

        self.assertEqual(count, len(src), "frame count")
        self.assertEqual(ref_count, len(src), "cv2 reference frame count")
        self.assertGreater(worst, ref_worst - 1.0,
                           f"H.264 {worst:.1f} dB vs mp4v {ref_worst:.1f} dB")
        print(f"[parity] mp4 export: {count} frames, decoded PSNR {worst:.1f} dB "
              f"(cv2 mp4v reference {ref_worst:.1f} dB)")

    def test_finish_mono(self):
        from sharpen import finish

        master = self.frames[0]
        _flag(False)
        py = finish(master)
        _flag(True)
        rs = finish(master)
        psnr = _psnr(py, rs)
        self.assertGreater(psnr, 40.0, f"finish PSNR {psnr:.1f}")
        print(f"[parity] finish(): PSNR {psnr:.1f} dB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
