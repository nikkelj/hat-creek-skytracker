#!/usr/bin/env python
"""
Regression test for tracking-visualization flicker.

The render thread clears its surface to black and redraws it every frame. The
flicker bug was publishing a *reference* to that same surface, so the main loop
could blit it mid-clear (a black/torn frame). The fix (_publish_surface) hands
consumers a copy taken only after drawing completes.

This test drives a writer (clear-to-black -> fill-with-color -> publish) against
a concurrent reader, and flags any observed frame that is black or non-uniform
(i.e. caught mid-update). It asserts:
  * the real _publish_surface path yields zero torn reads, and
  * the old "publish the live surface" behavior DOES produce torn reads, proving
    the detector actually catches the bug.

Headless; needs only pygame. Run: python test_render_buffer.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import threading
import time
import types
import unittest

import pygame
pygame.init()

from rendering_threads import VisualizationRenderingThread

W, H = 200, 150
SAMPLE_PTS = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, H // 2)]
N_FRAMES = 400
GAP = 0.0005  # widen the mid-update window so a race is observable


def _make_stub():
    """A minimal object bound to the real publish/get methods (no display)."""
    stub = type("Stub", (), {})()
    stub.surface = pygame.Surface((W, H))
    stub.latest_surface = None
    stub.surface_lock = threading.Condition()
    stub._publish_surface = types.MethodType(
        VisualizationRenderingThread._publish_surface, stub)
    stub.get_latest_surface = types.MethodType(
        VisualizationRenderingThread.get_latest_surface, stub)
    return stub


def _run_scenario(stub, publish_fn):
    """Return the number of torn/black frames a reader observed."""
    stop = threading.Event()
    torn = [0]

    def reader():
        while not stop.is_set():
            s = stub.get_latest_surface()
            if s is None:
                continue
            cols = [s.get_at(p)[:3] for p in SAMPLE_PTS]
            # Bad if not a single uniform color, or uniformly black (caught
            # right after the clear, before the color fill).
            if len(set(cols)) > 1 or cols[0] == (0, 0, 0):
                torn[0] += 1

    r = threading.Thread(target=reader, daemon=True)
    r.start()
    try:
        for i in range(N_FRAMES):
            stub.surface.fill((0, 0, 0))      # clear
            time.sleep(GAP)
            c = 50 + (i % 200)                # always non-black
            stub.surface.fill((c, c, c))      # draw
            publish_fn()
            time.sleep(GAP)
    finally:
        stop.set()
        r.join()
    return torn[0]


class RenderBufferTests(unittest.TestCase):

    def test_publish_snapshot_never_tears(self):
        stub = _make_stub()
        torn = _run_scenario(stub, stub._publish_surface)
        self.assertEqual(torn, 0, f"snapshot publish produced {torn} torn reads")

    def test_detector_catches_live_surface_publish(self):
        # The old behavior: publish a reference to the live surface.
        stub = _make_stub()

        def buggy_publish():
            with stub.surface_lock:
                stub.latest_surface = stub.surface

        torn = _run_scenario(stub, buggy_publish)
        self.assertGreater(torn, 0,
                           "expected torn reads from live-surface publish")


if __name__ == '__main__':
    unittest.main(verbosity=2)
