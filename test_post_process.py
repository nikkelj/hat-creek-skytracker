"""Headless tests for the post-processing core (post_process.py + stabilizer.py).

Run with the track env:
    C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe test_post_process.py

These exercise run discovery/grouping, the synced timeline, the image pipeline,
stabilization, sidecar round-trips, and a real MP4 export -- all without the UI.
"""

import os
import tempfile
import numpy as np
import cv2

from post_process import (
    RunLibrary, Run, FrameProcessor, TrajectorySeries, decode_frame,
    compute_track_vectors, Mp4Exporter, _parse_frame_time, _parse_iso_time,
)
from stabilizer import Stabilizer

DATA_DIR = "data"


def _first_real_run(lib):
    for r in lib.runs:
        if r.camera_indices and r.frame_count(r.camera_indices[0]) >= 2:
            return r
    return None


def test_time_parsing():
    t = _parse_frame_time("2025_09_16_03_47_22.342929")
    assert abs(t - 1757994442.342929) < 1.0, t
    a = _parse_iso_time("2025-09-16T17:05:55.450644+00:00Z")
    b = _parse_iso_time("2025-09-16T17:05:55.450644+00:00")
    assert abs(a - b) < 1e-6
    print("ok  time parsing")


def test_library_scan_and_group():
    lib = RunLibrary(DATA_DIR)
    lib.scan()
    assert len(lib.runs) > 0, "no runs found in data/"
    groups = lib.groups()
    assert len(groups) > 0
    # every grouped run carries its files
    for key, runs in groups:
        for r in runs:
            assert r.camera_indices
    # a manual run groups under "Manual"
    assert any(k == "Manual" for k, _ in groups) or all(not r.is_manual for r in lib.runs)
    print(f"ok  library scan/group ({len(lib.runs)} runs, {len(groups)} groups)")


def test_timeline_and_frame_at():
    lib = RunLibrary(DATA_DIR)
    lib.scan()
    run = _first_real_run(lib)
    assert run is not None
    assert run.t0 is not None and run.t1 >= run.t0
    assert run.duration >= 0
    cam = run.camera_indices[0]
    # frame_index_at is monotonic and clamped
    i0 = run.frame_index_at(cam, run.t0 - 100)
    i_mid = run.frame_index_at(cam, run.t0 + run.duration * 0.5)
    i1 = run.frame_index_at(cam, run.t1 + 100)
    assert i0 == 0
    assert i1 == run.frame_count(cam) - 1
    assert 0 <= i_mid <= i1
    # two cameras stay synced by time even with different frame counts
    if len(run.camera_indices) > 1:
        t = run.t0 + run.duration * 0.5
        for c in run.camera_indices:
            idx = run.frame_index_at(c, t)
            assert run.frames(c)[idx]["t"] <= t + 1e-6
    print("ok  timeline + frame_at sync")


def test_frame_processor_lut():
    proc = FrameProcessor()
    img = np.full((4, 4, 3), 50, dtype=np.uint8)
    # gamma > 1 brightens
    bright = proc.apply(img, gamma=2.2)
    assert bright.mean() > img.mean()
    # contrast spreads about mid-grey; brightness shifts up
    up = proc.apply(img, brightness=40)
    assert up.mean() > img.mean()
    # identity is a no-op (returns same array)
    same = proc.apply(img, 1.0, 0.0, 1.0)
    assert np.array_equal(same, img)
    print("ok  frame processor LUT")


def test_stabilizer_reduces_shift():
    rng = np.random.default_rng(1)
    base = (rng.random((240, 240, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    M = np.float32([[1, 0, 9], [0, 1, -6]])
    shifted = cv2.warpAffine(base, M, (240, 240))
    for method in ("orb", "flow"):
        st = Stabilizer(method=method)
        st.set_reference(base)
        warped, _ = st.stabilize(shifted)
        before = np.abs(base.astype(int) - shifted.astype(int)).mean()
        after = np.abs(base.astype(int) - warped.astype(int)).mean()
        assert after < before * 0.6, (method, before, after)
    print("ok  stabilizer reduces shift (orb + flow)")


def test_sidecar_roundtrip():
    lib = RunLibrary(DATA_DIR)
    lib.scan()
    run = _first_real_run(lib)
    orig = dict(run.sidecar)
    try:
        run.set_display_name("UNIT TEST NAME")
        run.add_tag("unit-test-tag")
        run.set_notes("hello notes")
        fav = run.toggle_favorite()
        # reload from disk via a fresh Run object
        fresh = Run(run.path)
        assert fresh.sidecar["display_name"] == "UNIT TEST NAME"
        assert "unit-test-tag" in fresh.sidecar["tags"]
        assert fresh.sidecar["notes"] == "hello notes"
        assert fresh.sidecar["favorite"] == fav
        print("ok  sidecar round-trip")
    finally:
        # restore so we don't leave test cruft on a real run
        run._sidecar = dict(orig)
        run.save_sidecar()
        if not orig and os.path.exists(run.sidecar_path):
            os.remove(run.sidecar_path)


def test_track_vectors():
    lib = RunLibrary(DATA_DIR)
    lib.scan()
    run = next((r for r in lib.runs if r.trajectory.valid), None)
    if run is None:
        print("skip track vectors (no run with trajectory.csv)")
        return
    t = run.t0 + run.duration * 0.5
    v = compute_track_vectors(run.trajectory, t, 640, 360)
    assert v is not None
    assert "anchor" in v and "intrack" in v
    print("ok  track vectors")


def test_mp4_export():
    lib = RunLibrary(DATA_DIR)
    lib.scan()
    run = _first_real_run(lib)
    cam = run.camera_indices[0]
    frames = run.frames(cam)
    t_end = frames[min(4, len(frames) - 1)]["t"]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "clip.mp4")
        exp = Mp4Exporter(run, cam, run.t0, t_end, out,
                          gamma=2.0, stabilize=True, overlays=True)
        exp.start()
        import time
        for _ in range(300):
            if exp.done:
                break
            time.sleep(0.05)
        assert exp.done and exp.error is None, exp.error
        assert os.path.exists(out) and os.path.getsize(out) > 0
        cap = cv2.VideoCapture(out)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n >= 1, n
    print(f"ok  mp4 export ({exp.frames_written} frames)")


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("-" * 40)
    print("ALL PASSED" if failed == 0 else f"{failed} test(s) failed")
    raise SystemExit(1 if failed else 0)
