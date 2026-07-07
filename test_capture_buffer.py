#!/usr/bin/env python
"""
Regression tests for the capture buffer's stop_capture snapshot.

stop_capture must hand the dump thread a SNAPSHOT (plain list), not the live
deque: the capture thread keeps appending, and once the deque is full each
append evicts the oldest entry and shifts every index -- so the indices
computed at stop time drift under a long dump and the wrong frames get saved.

Headless, no hardware. Run: python test_capture_buffer.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest

from camera_buffer import CameraThread


class _FakeCap:
    def get_camera_property(self):
        return {'MaxWidth': 64, 'MaxHeight': 48}


def _make_thread(buffer_size=5):
    # Constructed but never start()ed -- we drive the buffer directly.
    return CameraThread(0, _FakeCap(), buffer_size=buffer_size)


class StopCaptureSnapshotTests(unittest.TestCase):

    def test_returns_list_snapshot(self):
        t = _make_thread(buffer_size=5)
        t.start_capture()
        for i in range(8):  # overfill: deque keeps the last 5
            t.circular_buffer.append({'i': i})
            t.capture_frame_count += 1

        info, frames = t.stop_capture()
        self.assertIsInstance(frames, list)
        self.assertEqual(len(frames), 5)
        self.assertEqual([f['i'] for f in frames], [3, 4, 5, 6, 7])
        self.assertEqual(info['end_idx'], len(frames) - 1)
        self.assertEqual(info['captured_frame_count'], 8)

    def test_snapshot_immune_to_further_capture(self):
        t = _make_thread(buffer_size=5)
        t.start_capture()
        for i in range(5):
            t.circular_buffer.append({'i': i})
            t.capture_frame_count += 1
        info, frames = t.stop_capture()

        # The capture thread keeps running after stop; the snapshot the dump
        # thread holds must not shift underneath it.
        for i in range(100, 110):
            t.circular_buffer.append({'i': i})
        self.assertEqual([f['i'] for f in frames], [0, 1, 2, 3, 4])
        # end_idx still addresses the last captured frame within the snapshot.
        self.assertEqual(frames[info['end_idx']]['i'], 4)

    def test_no_frames_returns_none(self):
        t = _make_thread()
        t.start_capture()
        info, frames = t.stop_capture()
        self.assertIsNone(info)
        self.assertIsNone(frames)


if __name__ == '__main__':
    unittest.main(verbosity=2)
