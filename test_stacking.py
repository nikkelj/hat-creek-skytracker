"""Headless tests for the lucky-imaging pipeline (stacking.py).

PIPP-style prep (sharpness grading, culling, target centring/cropping) and
AutoStakkert-style stacking (global + alignment-point alignment, averaging into
a high-SNR master). Everything is synthesised in-memory so these run with no
real ``data/`` capture and no UI.

Run with the track env:
    C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe test_stacking.py
or just ``python test_stacking.py`` / ``pytest test_stacking.py``.
"""

import os
import tempfile

import numpy as np
import cv2

from stacking import (
    sharpness, SHARPNESS_METHODS, QualityGrader, FrameGrade, select_best,
    brightness_centroid, bounding_box, crop_centered, recenter_frame,
    AlignmentPointGrid, measure_local_shifts, warp_by_grid,
    LuckyStacker, stack_run, save_master, _decode,
)


# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------
def _blob(size=200, center=None, radius=14, amp=200, bg=18, blur=0.0, seed=0):
    """A bright Gaussian blob (a 'satellite') on a dim noisy sky."""
    h = w = size
    if center is None:
        center = (w / 2.0, h / 2.0)
    yy, xx = np.mgrid[0:h, 0:w]
    g = amp * np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) /
                       (2.0 * radius ** 2)))
    rng = np.random.default_rng(seed)
    img = bg + g + rng.normal(0, 4, (h, w))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blur > 0:
        k = max(3, int(blur) | 1)
        img = cv2.GaussianBlur(img, (k, k), blur)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _textured(size=200, seed=0, blur=0.0):
    """A high-frequency textured frame, optionally blurred (less sharp)."""
    rng = np.random.default_rng(seed)
    img = (rng.random((size, size)) * 255).astype(np.uint8)
    if blur > 0:
        k = max(3, int(blur) | 1)
        img = cv2.GaussianBlur(img, (k, k), blur)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _starfield(size=160, n_stars=45, target=True, seed=0):
    """A noiseless starfield (+ central target blob) -- ground truth to stack.

    Many small bright dots give ORB/optical-flow real features to lock onto, so
    global alignment behaves like it does on a real sky, while the fixed dot
    positions act as ground truth for measuring stack error.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 14.0)
    for _ in range(n_stars):
        sx = rng.integers(14, size - 14)
        sy = rng.integers(14, size - 14)
        amp = rng.uniform(120, 255)
        r = rng.uniform(1.1, 2.2)
        img += amp * np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * r ** 2)))
    if target:
        img += 180.0 * np.exp(-(((xx - size / 2) ** 2 + (yy - size / 2) ** 2) /
                                (2.0 * 9.0 ** 2)))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# PIPP: sharpness + grading
# ---------------------------------------------------------------------------
def test_sharpness_orders_blur():
    """Every metric must rank a crisp frame above a blurred copy."""
    crisp = _textured(seed=1, blur=0.0)
    soft = _textured(seed=1, blur=7.0)
    for m in SHARPNESS_METHODS:
        assert sharpness(crisp, m) > sharpness(soft, m), m
    print("ok  sharpness orders crisp > blurred (all metrics)")


def test_sharpness_roi():
    """ROI scoring ignores pixels outside the box."""
    img = _blob(center=(60, 60), radius=10)
    # ROI on the blob vs ROI on empty sky -> blob region is sharper (more edges).
    on = sharpness(img, "tenengrad", roi=(40, 40, 40, 40))
    off = sharpness(img, "tenengrad", roi=(150, 150, 40, 40))
    assert on > off
    print("ok  sharpness ROI focuses on target")


def test_grader_and_select():
    frames = [
        _textured(seed=2, blur=9.0),   # 0 worst
        _textured(seed=2, blur=0.0),   # 1 best
        _textured(seed=2, blur=3.0),   # 2 middle
    ]
    grades = QualityGrader("laplacian").grade(frames)
    assert [g.index for g in grades] == [0, 1, 2]  # input order preserved
    best = select_best(grades, fraction=0.5)
    assert len(best) == 2 and best[0].index == 1   # sharpest first
    one = select_best(grades, count=1)
    assert len(one) == 1 and one[0].index == 1
    # min_keep floors an over-aggressive fraction.
    assert len(select_best(grades, fraction=0.0)) == 1
    assert select_best([]) == []
    print("ok  grader sorts + select_best culls (fraction/count/min_keep)")


def test_grade_paths():
    """Grader accepts file paths and dicts, not just arrays."""
    with tempfile.TemporaryDirectory() as d:
        p_sharp = os.path.join(d, "a.png")
        p_soft = os.path.join(d, "b.png")
        cv2.imwrite(p_sharp, _textured(seed=3, blur=0.0))
        cv2.imwrite(p_soft, _textured(seed=3, blur=9.0))
        grades = QualityGrader().grade([p_soft, {"path": p_sharp}])
        best = select_best(grades, count=1)[0]
        assert best.index == 1                       # the sharp one
        assert best.source == {"path": p_sharp}
    print("ok  grader handles paths + dict sources")


# ---------------------------------------------------------------------------
# PIPP: centring + cropping
# ---------------------------------------------------------------------------
def test_centroid_locks_target():
    cx, cy = 132.0, 58.0
    img = _blob(center=(cx, cy), radius=9, seed=5)
    c = brightness_centroid(img)
    assert c is not None
    assert abs(c[0] - cx) < 2.0 and abs(c[1] - cy) < 2.0, c
    # Empty sky -> no target.
    flat = np.full((100, 100, 3), 20, np.uint8)
    assert brightness_centroid(flat) is None
    print(f"ok  brightness_centroid locks target ({c[0]:.1f},{c[1]:.1f})")


def test_bounding_box():
    img = _blob(center=(100, 100), radius=12, seed=6)
    box = bounding_box(img, pad=2)
    assert box is not None
    x, y, w, h = box
    assert x < 100 < x + w and y < 100 < y + h    # contains the centre
    assert w < 120 and h < 120                     # and is a small box, not full frame
    print(f"ok  bounding_box tight around target {box}")


def test_crop_centered_size_and_padding():
    img = _blob(center=(100, 100), radius=10, seed=7)
    out = crop_centered(img, (100, 100), 50)
    assert out.shape == (50, 50, 3)
    # Centre of crop is bright (the blob), corners are sky.
    assert int(out[25, 25].mean()) > int(out[0, 0].mean()) + 30
    # Crop straddling a corner stays full-size, zero-padded outside the image.
    edge = crop_centered(img, (2, 2), 40, pad_value=0)
    assert edge.shape == (40, 40, 3)
    assert edge[0, 0].sum() == 0                   # off-frame corner padded
    print("ok  crop_centered fixed-size + edge padding")


def test_crop_centered_odd_size_is_centered():
    """The target must land on the centre pixel (out//2) for even AND odd sizes."""
    img = np.zeros((60, 60, 3), np.uint8)
    img[33, 21] = 255                                # single bright px at (x=21,y=33)
    for size in (40, 41):                            # even and odd
        out = crop_centered(img, (21, 33), size)
        g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        peak_y, peak_x = np.unravel_index(int(g.argmax()), g.shape)
        assert (peak_x, peak_y) == (size // 2, size // 2), (size, peak_x, peak_y)
    print("ok  crop_centered centres even + odd sizes on out//2")


def test_recenter_moves_target_to_middle():
    img = _blob(center=(150, 40), radius=10, seed=8)
    out = recenter_frame(img)
    c = brightness_centroid(out)
    h, w = out.shape[:2]
    assert abs(c[0] - w / 2) < 3 and abs(c[1] - h / 2) < 3, c
    print("ok  recenter_frame centres the target")


# ---------------------------------------------------------------------------
# AutoStakkert: alignment points
# ---------------------------------------------------------------------------
def test_alignment_grid_shape():
    grid = AlignmentPointGrid.over((400, 600), spacing=100)
    assert grid.rows >= 2 and grid.cols >= 2
    assert len(grid) == grid.rows * grid.cols
    # Nodes span the full frame edge-to-edge (so cv2.resize lands each shift on
    # the pixel it was measured at), not an inset sub-region.
    assert grid.points[:, 0].min() == 0 and grid.points[:, 1].min() == 0
    assert grid.points[:, 0].max() == 599 and grid.points[:, 1].max() == 399
    # A tiny frame still yields a non-degenerate >=2x2 grid (no collapsed axis).
    tiny = AlignmentPointGrid.over((3, 3), spacing=80)
    assert tiny.rows >= 2 and tiny.cols >= 2
    assert tiny.points[:, 0].min() != tiny.points[:, 0].max()
    print(f"ok  alignment grid {grid.rows}x{grid.cols} = {len(grid)} pts, spans edges")


def test_warp_by_grid_constant_field_is_translation():
    """A uniform shift field must reproduce a plain integer translation."""
    img = _textured(size=120, seed=9)
    grid = AlignmentPointGrid.over(img.shape, spacing=40)
    shift = np.zeros((len(grid), 2))
    shift[:, 0] = 5.0   # sample 5px to the right everywhere -> content moves left
    out = warp_by_grid(img, grid, shift)
    ref = np.roll(img, -5, axis=1)
    inner = slice(10, -10)
    diff = np.abs(out[inner, inner].astype(int) - ref[inner, inner].astype(int)).mean()
    assert diff < 3.0, diff
    print("ok  warp_by_grid constant field == translation")


def test_measure_local_shifts_recovers_translation():
    """Phase-correlation shift sign is consistent with warp_by_grid's convention."""
    ref = _textured(size=200, seed=11)
    dx, dy = 4, -3
    M = np.float32([[1, 0, dx], [0, 1, dy]])         # warp (not roll) -> no wrap
    moved = cv2.warpAffine(ref, M, (200, 200), borderMode=cv2.BORDER_REFLECT)
    grid = AlignmentPointGrid.over(ref.shape, spacing=50)
    shifts = measure_local_shifts(ref, moved, grid.points, patch=64)
    med = np.median(shifts, axis=0)
    # Re-warping 'moved' by the measured field should land back on 'ref'.
    fixed = warp_by_grid(moved, grid, shifts)
    inner = slice(30, -30)
    before = np.abs(moved[inner, inner].astype(int) - ref[inner, inner].astype(int)).mean()
    after = np.abs(fixed[inner, inner].astype(int) - ref[inner, inner].astype(int)).mean()
    assert after < before * 0.4, (before, after, med)
    print(f"ok  local shifts recover translation (err {before:.1f} -> {after:.1f})")


# ---------------------------------------------------------------------------
# AutoStakkert: the stacker
# ---------------------------------------------------------------------------
def _jittered_noisy_set(n=24, size=160, noise=30, seed=0):
    """A static starfield captured n times with random sub-pixel jitter + noise."""
    rng = np.random.default_rng(seed)
    truth = _starfield(size=size, seed=seed)
    frames = []
    for i in range(n):
        jx, jy = rng.normal(0, 3, 2)
        M = np.float32([[1, 0, jx], [0, 1, jy]])
        shifted = cv2.warpAffine(truth, M, (size, size), borderMode=cv2.BORDER_REFLECT)
        noisy = np.clip(shifted.astype(np.float64) +
                        rng.normal(0, noise, shifted.shape), 0, 255).astype(np.uint8)
        frames.append(noisy)
    return truth, frames


def test_stacker_improves_snr():
    """Averaging aligned frames must beat a single frame's noise."""
    truth, frames = _jittered_noisy_set(n=24, noise=30, seed=21)
    stacker = LuckyStacker(method="orb")
    # Use a clean reference (jitter=0) so we measure against ground truth.
    stacker.set_reference(truth)
    for f in frames:
        stacker.add(f)
    master = stacker.result()
    assert master is not None

    # Compare noise (std of the flat sky corner) and error vs truth.
    def corner_std(im):
        return float(im[:30, :30].astype(np.float64).std())
    single_err = np.abs(frames[0].astype(int) - truth.astype(int)).mean()
    stack_err = np.abs(master.astype(int) - truth.astype(int)).mean()
    assert stack_err < single_err * 0.6, (single_err, stack_err)
    assert corner_std(master) < corner_std(frames[0]) * 0.7
    st = stacker.stats
    # Reference + all added frames are accounted for, and the counts reconcile.
    assert st.n_total == len(frames) + 1, st
    assert st.n_stacked + st.n_rejected == st.n_total, st
    assert st.n_stacked >= int(0.7 * st.n_total), st
    print(f"ok  stacker SNR: err {single_err:.1f} -> {stack_err:.1f}, "
          f"stacked {st.n_stacked}/{st.n_total}")


def test_stacker_local_runs_and_stacks():
    """Local (alignment-point) stacking produces a valid master and improves SNR."""
    truth, frames = _jittered_noisy_set(n=18, noise=28, seed=22)
    stacker = LuckyStacker(method="orb", local=True, ap_spacing=50, ap_patch=48)
    stacker.set_reference(truth)
    for f in frames:
        stacker.add(f)
    master = stacker.result()
    assert master is not None and master.shape == truth.shape
    stack_err = np.abs(master.astype(int) - truth.astype(int)).mean()
    single_err = np.abs(frames[0].astype(int) - truth.astype(int)).mean()
    assert stack_err < single_err * 0.7, (single_err, stack_err)
    print(f"ok  local stacker master err {single_err:.1f} -> {stack_err:.1f}")


def test_stacker_rejects_unalignable():
    """A frame that can't be aligned is dropped, not smeared into the master."""
    truth, frames = _jittered_noisy_set(n=6, noise=20, seed=23)
    stacker = LuckyStacker(method="orb")
    stacker.set_reference(truth)
    for f in frames:
        stacker.add(f)
    good = stacker.stats.n_stacked
    # Pure noise shares no features with the reference -> rejected.
    junk = (np.random.default_rng(99).random(truth.shape) * 255).astype(np.uint8)
    stacker.add(junk)
    assert stacker.stats.n_stacked == good       # not added
    assert stacker.stats.n_rejected >= 1
    print(f"ok  stacker rejects unalignable frame ({stacker.stats.n_rejected} rejected)")


def test_stacker_no_border_vignette():
    """Coverage weighting must keep shifted black borders out of the master.

    Every frame is translated by a large, same-direction shift, so a naive
    sum/count average would darken one whole border band. Per-pixel coverage
    weighting (the reference covers everything) must keep edges near the truth.
    """
    truth = _starfield(size=160, seed=40)
    shift = 16
    frames = []
    for i in range(8):
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        frames.append(cv2.warpAffine(truth, M, (160, 160),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    stacker = LuckyStacker(method="orb")
    stacker.set_reference(truth)
    for f in frames:
        stacker.add(f)
    master = stacker.result()
    assert master is not None
    # The top-left band is where every shifted frame contributed black; with
    # coverage weighting it should still track the reference, not collapse to 0.
    band_master = master[:shift, :shift].astype(np.float64).mean()
    band_truth = truth[:shift, :shift].astype(np.float64).mean()
    assert abs(band_master - band_truth) < 12.0, (band_master, band_truth)
    # And no all-black pixels anywhere (a naive average would zero the corner).
    assert master.sum(axis=2).min() > 0
    print(f"ok  stacker coverage avoids border vignette "
          f"(band {band_master:.0f} vs truth {band_truth:.0f})")


def test_stacker_empty_result_is_none():
    s = LuckyStacker()
    assert s.result() is None
    assert s.stack([]) is None
    print("ok  empty stacker -> None")


# ---------------------------------------------------------------------------
# Run-level glue
# ---------------------------------------------------------------------------
class _FakeRun:
    """Minimal Run stand-in: writes frames to disk and indexes them like Run."""

    def __init__(self, frames, times, directory):
        self._frames = []
        for i, (arr, t) in enumerate(zip(frames, times)):
            p = os.path.join(directory, f"Camera1_{i:06d}.bmp")
            cv2.imwrite(p, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            self._frames.append({"path": p, "t": float(t), "seq": i})

    def frames(self, cam_index):
        return self._frames


def test_stack_run_pipeline():
    truth, frames = _jittered_noisy_set(n=20, noise=30, seed=24)
    # Make a few frames much blurrier so culling has something to drop.
    for i in (0, 5, 11):
        frames[i] = cv2.GaussianBlur(frames[i], (9, 9), 6)
    with tempfile.TemporaryDirectory() as d:
        run = _FakeRun(frames, times=range(len(frames)), directory=d)
        seen = []
        res = stack_run(run, 0, keep_fraction=0.5, quality="laplacian",
                        progress=lambda done, total: seen.append((done, total)))
        assert res.master is not None
        assert res.stats.n_stacked >= 1
        # Sharpest frame anchors the stack, and the blurred ones are culled.
        kept_idx = {g.index for g in res.kept}
        assert res.reference_index == res.kept[0].index
        assert 0 not in kept_idx and 5 not in kept_idx   # blurred frames dropped
        assert len(res.kept) == round(0.5 * len(frames))
        assert seen and seen[-1][0] == seen[-1][1]       # progress reached 100%

        out = os.path.join(d, "master.png")
        save_master(res.master, out)
        assert os.path.exists(out)
        loaded = _decode(out)
        assert loaded is not None and loaded.shape == res.master.shape
    print(f"ok  stack_run pipeline (kept {len(res.kept)}, "
          f"stacked {res.stats.n_stacked}, ref #{res.reference_index})")


def test_stack_run_keep_count_and_window():
    truth, frames = _jittered_noisy_set(n=12, noise=20, seed=25)
    with tempfile.TemporaryDirectory() as d:
        run = _FakeRun(frames, times=range(len(frames)), directory=d)
        # keep_count overrides fraction.
        res = stack_run(run, 0, keep_count=3)
        assert len(res.kept) == 3
        # time window restricts the candidate set; FrameGrade.source carries the
        # original frame dict so callers can map a grade back to its capture.
        res2 = stack_run(run, 0, keep_fraction=1.0, t_start=2, t_end=5)
        assert all(2 <= g.source["t"] <= 5 for g in res2.grades)
        assert len(res2.grades) == 4
    print("ok  stack_run keep_count + time window")


def test_stack_run_max_frames_subsample():
    truth, frames = _jittered_noisy_set(n=30, noise=15, seed=26)
    with tempfile.TemporaryDirectory() as d:
        run = _FakeRun(frames, times=range(len(frames)), directory=d)
        res = stack_run(run, 0, keep_fraction=1.0, max_frames=10)
        assert len(res.grades) <= 10 and res.master is not None
    print("ok  stack_run max_frames subsampling")


def test_stack_exporter_thread():
    """The threaded StackExporter writes a master PNG and reports stats."""
    import time
    from post_process import StackExporter

    truth, frames = _jittered_noisy_set(n=16, noise=25, seed=27)
    with tempfile.TemporaryDirectory() as d:
        run = _FakeRun(frames, times=range(len(frames)), directory=d)
        out = os.path.join(d, "master.png")
        exp = StackExporter(run, 0, t_start=None, t_end=None, out_path=out,
                            keep_fraction=0.5, local=False)
        exp.start()
        for _ in range(400):
            if exp.done:
                break
            time.sleep(0.02)
        assert exp.done and exp.error is None, exp.error
        assert os.path.exists(exp.out_path) and os.path.getsize(exp.out_path) > 0
        assert exp.stats is not None and exp.stats.n_stacked >= 1
        assert abs(exp.progress - 1.0) < 1e-6
    print(f"ok  StackExporter thread (stacked {exp.stats.n_stacked})")


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("-" * 40)
    print("ALL PASSED" if failed == 0 else f"{failed} test(s) failed")
    raise SystemExit(1 if failed else 0)
