#!/usr/bin/env python
"""Phase 4a: the Rust camera pipeline rate-proof, driven by hardware-sim
frames (the user's requirement: sim it and ensure we hit camera-native
rates — Python's capture path capped at 4-10 FPS; cameras do 50-100).

- Frames come from the real HardwareSimulator renderer (star field +
  target), pre-rendered then replayed at metered 50/100 FPS so the test
  measures the PIPELINE, not the sim renderer.
- Gates: sustained pump rate within 5% of the 50 and 100 FPS targets
  with zero drops, while a concurrent 30 Hz display consumer pulls
  frames; a micro-A/B against the Python per-frame path
  (pygame surface conversion + CircularBuffer + timestamping); armed
  capture dumps correct BMPs with monotonic midpoint-backdated stamps.

Build first: cd rust/skytracker-ffi && maturin develop --release
"""

import json
import math
import os
import tempfile
import threading
import time
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import skytracker_core
    _HAVE = getattr(skytracker_core, "CAMERA_AVAILABLE", False)
except ImportError:
    _HAVE = False


def _render_sim_frames(n=24, w=640, h=480):
    """Real HardwareSimulator frames (star field + target), pre-rendered."""
    import pygame
    pygame.init()
    from config import ConfigState
    from simulator import HardwareSimulator

    cfg = ConfigState()
    cfg.load_from_dict(json.load(open("config.example.json")))
    cfg.sim_config["enabled"] = True
    cfg.sim_config["cam_width"] = w
    cfg.sim_config["cam_height"] = h
    cfg.sim_config["sim_use_real_stars"] = False  # fast deterministic field
    sim = HardwareSimulator(cfg, None, None)
    sim.mount.az_true_deg = 100.0
    sim.mount.el_true_deg = 50.0
    frames = []
    for i in range(n):
        sim.mount.az_true_deg = 100.0 + i * 0.02
        f = sim.render_frame(0, exposure_us=10000)
        arr = np.asarray(f)
        if arr.ndim == 3:
            arr = arr[..., 0]
        frames.append(np.ascontiguousarray(arr, dtype=np.uint8))
    return frames


@unittest.skipUnless(_HAVE, "skytracker_core wheel lacks camera pipeline")
class CameraPipelineRates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = _render_sim_frames()

    def _metered_pump(self, fps, seconds=2.5, with_display_consumer=True):
        pipe = skytracker_core.CameraPipeline.push_source(ring_capacity=1000)
        pulls = [0]
        stop = threading.Event()

        def display_consumer():
            while not stop.is_set():
                got = pipe.latest_frame()
                if got is not None:
                    pulls[0] += 1
                time.sleep(1.0 / 30.0)

        consumer = None
        if with_display_consumer:
            consumer = threading.Thread(target=display_consumer, daemon=True)
            consumer.start()

        period = 1.0 / fps
        n_target = int(seconds * fps)
        capture_time = 1.0 / fps
        t0 = time.perf_counter()
        for i in range(n_target):
            # Metered release: wait until this frame's slot.
            while time.perf_counter() - t0 < i * period:
                pass
            pipe.push_frame_mono(self.frames[i % len(self.frames)], capture_time)
        # Wait for the pump to drain.
        deadline = time.time() + 2.0
        while pipe.frames_pumped() < n_target and time.time() < deadline:
            time.sleep(0.005)
        elapsed = time.perf_counter() - t0
        stop.set()
        if consumer:
            consumer.join(timeout=1.0)
        pumped = pipe.frames_pumped()
        dropped = pipe.frames_dropped()
        pipe.close()
        rate = pumped / elapsed
        return rate, pumped, n_target, dropped, pulls[0]

    def test_sustains_50_fps(self):
        rate, pumped, target, dropped, pulls = self._metered_pump(50.0)
        self.assertEqual(pumped, target, f"pumped {pumped}/{target}")
        self.assertEqual(dropped, 0)
        self.assertGreater(rate, 50.0 * 0.95, f"rate {rate:.1f} FPS")
        self.assertGreater(pulls, 30, "display consumer starved")
        print(f"\n[rate] 50 FPS target: {rate:.1f} FPS sustained, 0 dropped, "
              f"{pulls} display pulls")

    def test_sustains_100_fps(self):
        rate, pumped, target, dropped, pulls = self._metered_pump(100.0)
        self.assertEqual(pumped, target, f"pumped {pumped}/{target}")
        self.assertEqual(dropped, 0)
        self.assertGreater(rate, 100.0 * 0.95, f"rate {rate:.1f} FPS")
        print(f"[rate] 100 FPS target: {rate:.1f} FPS sustained, 0 dropped, "
              f"{pulls} display pulls")

    def test_unthrottled_headroom_vs_python_path(self):
        """Push as fast as possible; compare with the Python per-frame path
        (pygame surface + CircularBuffer + midpoint stamp under the GIL)."""
        import pygame
        from camera_buffer import CircularBuffer, exposure_midpoint_utc

        n = 300
        pipe = skytracker_core.CameraPipeline.push_source(ring_capacity=1000)
        t0 = time.perf_counter()
        for i in range(n):
            pipe.push_frame_mono(self.frames[i % len(self.frames)], 0.01)
        while pipe.frames_pumped() < n - pipe.frames_dropped():
            time.sleep(0.001)
        rust_fps = n / (time.perf_counter() - t0)
        pipe.close()

        buf = CircularBuffer(size=1000)
        rgb = np.stack([self.frames[0]] * 3, axis=-1)
        t0 = time.perf_counter()
        for i in range(n):
            f = self.frames[i % len(self.frames)]
            surf = pygame.surfarray.make_surface(
                np.stack([f] * 3, axis=-1).swapaxes(0, 1))
            ts = exposure_midpoint_utc(0.01)
            buf.append((surf, ts))
        py_fps = n / (time.perf_counter() - t0)
        del rgb

        print(f"[rate] unthrottled: rust pipeline {rust_fps:.0f} FPS vs "
              f"python per-frame path {py_fps:.0f} FPS (x{rust_fps / py_fps:.1f})")
        self.assertGreater(rust_fps, 200.0,
                           f"rust pipeline headroom too low: {rust_fps:.0f}")

    def test_capture_dump_and_timestamps(self):
        pipe = skytracker_core.CameraPipeline.push_source(ring_capacity=1000)
        pipe.arm_capture()
        for i in range(40):
            pipe.push_frame_mono(self.frames[i % len(self.frames)], 0.02)
            time.sleep(0.002)
        while pipe.frames_pumped() < 40:
            time.sleep(0.005)
        with tempfile.TemporaryDirectory() as d:
            count, stamps = pipe.disarm_and_dump(d)
            self.assertEqual(count, 40)
            bmps = sorted(f for f in os.listdir(d) if f.endswith(".bmp"))
            self.assertEqual(len(bmps), 40)
            import cv2
            img = cv2.imread(os.path.join(d, bmps[0]))
            self.assertEqual(img.shape[:2], self.frames[0].shape)
            # BMP content round-trips (gray replicated into BGR).
            self.assertTrue(np.array_equal(img[..., 0], self.frames[0]))
        # Monotonic, midpoint-backdated (stamp < push time).
        self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))
        self.assertLess(abs(time.time() - stamps[-1]), 5.0)
        pipe.close()
        print(f"[capture] 40 frames dumped, stamps monotonic, "
              f"span {stamps[-1] - stamps[0]:.2f}s")

    def test_midpoint_backdating_semantics(self):
        """Parity with camera_buffer.exposure_midpoint_utc: stamp = now - t/2."""
        pipe = skytracker_core.CameraPipeline.push_source(ring_capacity=10)
        before = time.time()
        pipe.push_frame_mono(self.frames[0], 0.5)
        while pipe.frames_pumped() < 1:
            time.sleep(0.001)
        _, stamp, _ = pipe.latest_frame()
        after = time.time()
        pipe.close()
        # Stamp should be ~0.25 s before "now at pump time".
        self.assertLess(stamp, after - 0.20)
        self.assertGreater(stamp, before - 0.30)
        offset = (before + after) / 2 - stamp
        self.assertLess(abs(offset - 0.25), 0.1, f"midpoint offset {offset:.3f}")
        print(f"[stamps] midpoint backdating: {offset * 1000:.0f} ms (expect ~250)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
