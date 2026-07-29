"""Headless tests for the post-processing core (post_process.py + stabilizer.py).

Run: python test_post_process.py

These exercise run discovery/grouping, the synced timeline, the image pipeline,
stabilization, sidecar round-trips, and a real MP4 export -- all without the UI.

HERMETIC: the tests build synthetic capture runs (BMP frames named exactly as
capture_manager writes them + a trajectory.csv) in a temp directory, so they
run on a fresh checkout / CI with no real `data/` captures -- and never mutate
real run sidecars.
"""

import atexit
import csv as _csv
import os
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import cv2

from post_process import (
    RunLibrary, Run, FrameProcessor, TrajectorySeries, decode_frame,
    compute_track_vectors, Mp4Exporter, _parse_frame_time, _parse_iso_time,
    FULL_VIEW, clamp_view, view_crop_px, remap_annotations,
)
from stabilizer import Stabilizer


# ---------------------------------------------------------------------------
# Synthetic run fixtures
# ---------------------------------------------------------------------------
_T0 = datetime(2026, 7, 3, 1, 2, 3, tzinfo=timezone.utc).timestamp()


def _frame_name(cam, seq, t):
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    stamp = dt.strftime("%Y_%m_%d_%H_%M_%S") + f".{dt.microsecond:06d}"
    return f"Camera{cam}_{seq:06d}__{stamp}.bmp"


def _write_frames(run_dir, cam, n, fps=5.0, w=96, h=64, seed=7):
    """Textured frames with a small per-frame shift (so ORB/stabilize works)."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h * 2, w * 2, 3)) * 255).astype(np.uint8),
                            (3, 3), 0)
    for i in range(n):
        t = _T0 + i / fps
        ox, oy = 4 + i, 6 + (i % 2)  # jitter: crop window drifts
        img = base[oy:oy + h, ox:ox + w]
        cv2.imwrite(os.path.join(run_dir, _frame_name(cam, i + 1, t)), img)


def _write_trajectory(run_dir, n=8, fps=2.0):
    path = os.path.join(run_dir, "trajectory.csv")
    fields = ["timestamp", "satellite_name", "altitude_deg", "azimuth_deg",
              "distance_km", "pixel_x", "pixel_y", "sequence_in_capture",
              "camera_index"]
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            t = datetime.fromtimestamp(_T0 - 1 + i / fps, tz=timezone.utc)
            w.writerow({
                "timestamp": t.isoformat(),
                "satellite_name": "TESTSAT",
                "altitude_deg": 40.0 + i * 0.5,
                "azimuth_deg": 120.0 + i * 1.0,
                "distance_km": 800.0 - i * 5.0,
                "pixel_x": 100.0 + i * 2.0,
                "pixel_y": 80.0 - i * 1.0,
                "sequence_in_capture": -1,
                "camera_index": -1,
            })
    return path


def _build_synthetic_data():
    root = tempfile.mkdtemp(prefix="pp_test_data_")
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    sat_run = os.path.join(root, "TESTSAT_99999_2026_07_03_01_02_03")
    os.makedirs(sat_run)
    _write_frames(sat_run, cam=1, n=6)
    _write_frames(sat_run, cam=2, n=4, fps=3.0, seed=11)
    _write_trajectory(sat_run)

    manual_run = os.path.join(root, "manual_2026_07_03_02_00_00")
    os.makedirs(manual_run)
    _write_frames(manual_run, cam=1, n=2, seed=3)
    return root


DATA_DIR = _build_synthetic_data()


def _first_real_run(lib):
    for r in lib.runs:
        if r.camera_indices and r.frame_count(r.camera_indices[0]) >= 2:
            return r
    return None


def test_png_run_and_fractionless_timestamps_ingest():
    # A PNG-format capture (config image_format=PNG / the capture fallback)
    # and a frame whose timestamp landed on exactly .000000 microseconds
    # (isoformat emits no fraction) must both be visible to the run indexer.
    root = tempfile.mkdtemp(prefix="pp_png_run_")
    try:
        run_dir = os.path.join(root, "PNGSAT_1_2026_07_03_03_00_00")
        os.makedirs(run_dir)
        img = np.full((16, 16, 3), 40, dtype=np.uint8)
        cv2.imwrite(os.path.join(
            run_dir, "Camera1_000001__2026_07_03_03_00_00.png"), img)      # no fraction
        cv2.imwrite(os.path.join(
            run_dir, "Camera1_000002__2026_07_03_03_00_00.250000.png"), img)
        run = Run(run_dir)
        assert run.camera_indices, "PNG frames must be indexed"
        cam = run.camera_indices[0]
        assert run.frame_count(cam) == 2
        ts = [f["t"] for f in run.frames(cam)]
        assert ts[1] - ts[0] == 0.25, ts
        assert decode_frame(run.frames(cam)[0]["path"]) is not None
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ok  png + fractionless-timestamp ingest")


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


def test_view_helpers():
    # clamp_view: orders corners, clamps to [0,1], enforces the minimum span
    assert clamp_view((0.8, 0.9, 0.2, 0.1)) == (0.2, 0.1, 0.8, 0.9)
    v = clamp_view((-0.5, 0.3, 0.5, 2.0))
    assert v[0] == 0.0 and v[3] == 1.0
    v = clamp_view((0.5, 0.5, 0.5001, 0.5001))  # micro-box -> min span, centred
    assert v[2] - v[0] >= 0.019 and v[3] - v[1] >= 0.019
    v = clamp_view((0.999, 0.999, 1.0, 1.0))    # min span at the border stays inside
    assert v[2] <= 1.0 and v[3] <= 1.0 and v[0] >= 0.0

    # view_crop_px: at least 1px; even=True yields even w/h for the encoder
    assert view_crop_px(FULL_VIEW, 100, 60) == (0, 0, 100, 60)
    x0, y0, x1, y1 = view_crop_px((0.1, 0.2, 0.5, 0.7), 101, 61, even=True)
    assert (x1 - x0) % 2 == 0 and (y1 - y0) % 2 == 0
    assert x1 > x0 and y1 > y0 and x1 <= 101 and y1 <= 61
    x0, y0, x1, y1 = view_crop_px((0.5, 0.5, 0.5005, 0.5005), 100, 60)
    assert x1 - x0 >= 1 and y1 - y0 >= 1

    # remap_annotations: identity for the full view, exact mapping for a crop
    anns = [{"type": "text", "cam": 0, "x": 0.5, "y": 0.5, "text": "t"},
            {"type": "box", "cam": 0, "x0": 0.25, "y0": 0.25, "x1": 0.5, "y1": 0.5}]
    assert remap_annotations(anns, FULL_VIEW) is anns
    out = remap_annotations(anns, (0.25, 0.25, 0.75, 0.75))
    assert abs(out[0]["x"] - 0.5) < 1e-9 and abs(out[0]["y"] - 0.5) < 1e-9
    assert abs(out[1]["x0"] - 0.0) < 1e-9 and abs(out[1]["x1"] - 0.5) < 1e-9
    assert anns[1]["x0"] == 0.25, "input list must not be mutated"
    print("ok  view helpers (clamp/crop/remap)")


def test_mp4_export_cropped():
    lib = RunLibrary(DATA_DIR)
    lib.scan()
    run = _first_real_run(lib)
    cam = run.camera_indices[0]
    frames = run.frames(cam)
    t_end = frames[min(4, len(frames) - 1)]["t"]
    w, h = run.frame_shape(cam)
    crop = (0.25, 0.25, 0.75, 0.75)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "clip_crop.mp4")
        exp = Mp4Exporter(run, cam, run.t0, t_end, out, overlays=True, crop=crop)
        exp.start()
        import time
        for _ in range(300):
            if exp.done:
                break
            time.sleep(0.05)
        assert exp.done and exp.error is None, exp.error
        assert os.path.exists(out) and os.path.getsize(out) > 0
        cap = cv2.VideoCapture(out)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n >= 1, n
        # Output is the crop region (half the frame), even-rounded
        assert vw == (view_crop_px(crop, w, h, even=True)[2]
                      - view_crop_px(crop, w, h, even=True)[0])
        assert vh == (view_crop_px(crop, w, h, even=True)[3]
                      - view_crop_px(crop, w, h, even=True)[1])
        assert vw <= w // 2 + 2 and vh <= h // 2 + 2
    print(f"ok  mp4 export cropped ({vw}x{vh})")


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
