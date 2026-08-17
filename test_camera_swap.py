"""Tests for CameraManager.swap_cameras: exchanging which physical camera
backs logical Camera 1 vs Camera 2 when USB enumeration order flips."""
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import numpy as np

from camera_manager import CameraManager


class _StubSimulator:
    """Just enough of HardwareSimulator for SimCap + CameraThread to run."""

    def sim_enabled(self):
        return True

    def _sim(self):
        return {'cam_width': 64, 'cam_height': 48, 'sim_exposure_timing': False}

    def render_frame(self, cam_index, width, height):
        return np.zeros((height, width), dtype=np.uint8)


def _make_manager(connected=False):
    mgr = CameraManager()
    if connected:
        mgr.simulator = _StubSimulator()
        assert mgr.connect_camera(0)
        assert mgr.connect_camera(1)
    return mgr


def _teardown(mgr):
    for i in range(len(mgr.cameras)):
        mgr.disconnect_camera(i)


def test_swap_while_disconnected_exchanges_hw_indices():
    mgr = _make_manager()
    c0, c1 = mgr.cameras
    assert (c0.asi_index, c1.asi_index) == (0, 1)

    mgr.swap_cameras()
    assert (c0.asi_index, c1.asi_index) == (1, 0)
    # Logical slot identity is untouched -- settings stay with the slot.
    assert (c0.index, c1.index) == (0, 1)

    mgr.swap_cameras()
    assert (c0.asi_index, c1.asi_index) == (0, 1)


def test_swap_while_connected_reconnects_swapped():
    mgr = _make_manager(connected=True)
    try:
        c0, c1 = mgr.cameras
        assert (c0.cap.cam_index, c1.cap.cam_index) == (0, 1)

        assert mgr.swap_cameras()
        assert c0.connected and c1.connected
        assert (c0.cap.cam_index, c1.cap.cam_index) == (1, 0)
    finally:
        _teardown(mgr)


def test_swap_reconnects_only_previously_connected_slots():
    mgr = _make_manager()
    mgr.simulator = _StubSimulator()
    assert mgr.connect_camera(0)
    try:
        assert mgr.swap_cameras()
        c0, c1 = mgr.cameras
        assert c0.connected
        assert not c1.connected
        assert c0.cap.cam_index == 1
    finally:
        _teardown(mgr)


def test_swap_settings_stay_with_logical_slot():
    mgr = _make_manager()
    c0, c1 = mgr.cameras
    c0.gain, c1.gain = 100, 200
    c0.alignment_rotation, c1.alignment_rotation = 5.0, -5.0

    mgr.swap_cameras()
    assert (c0.gain, c1.gain) == (100, 200)
    assert (c0.alignment_rotation, c1.alignment_rotation) == (5.0, -5.0)
