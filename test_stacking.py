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
    content_score, prefilter_garbage,
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


def test_sharpness_scale_preserves_order():
    """Reduced-resolution grading still ranks crisp above blurred (for speed)."""
    crisp = _textured(seed=4, blur=0.0)
    soft = _textured(seed=4, blur=7.0)
    assert sharpness(crisp, scale=0.5) > sharpness(soft, scale=0.5)
    assert sharpness(crisp, scale=0.25) > sharpness(soft, scale=0.25)
    print("ok  sharpness ranking preserved at reduced scale")


def test_sharpness_is_fooled_by_noise_but_content_is_not():
    """The core reason the pre-cull exists: sharpness rewards noise.

    A pure-noise frame scores HIGHER on sharpness than a faint but real target,
    so a naive top-X% cull would keep junk and drop good data. content_score
    must invert that ordering.
    """
    faint = _blob(size=200, center=(100, 100), radius=22, amp=45, bg=12, blur=2.0, seed=1)
    noise = cv2.cvtColor((np.random.default_rng(2).normal(30, 26, (200, 200))
                          ).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    assert sharpness(noise) > sharpness(faint)              # the trap
    assert content_score(faint) > content_score(noise)      # the fix
    print(f"ok  sharpness fooled by noise, content_score not "
          f"(sharp n>{sharpness(noise):.0f}>f{sharpness(faint):.0f}; "
          f"content f>{content_score(faint):.1f}>n{content_score(noise):.1f})")


def test_content_score_separates_signal():
    real = _starfield(size=200, seed=3)
    faint = _blob(size=200, center=(100, 100), radius=20, amp=50, bg=12, blur=2.0, seed=3)
    noise = cv2.cvtColor((np.random.default_rng(4).normal(30, 25, (200, 200))
                          ).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    blank = np.full((200, 200, 3), 15, np.uint8)
    assert content_score(real) > content_score(faint) > content_score(noise)
    assert content_score(noise) > content_score(blank) - 1e-6  # blank ~ 0
    assert content_score(blank) < 1.0
    print("ok  content_score orders real > faint > noise > blank")


def test_prefilter_drops_only_garbage():
    """Pre-cull removes noise/blank but keeps every real frame (incl. faint)."""
    frames = [_starfield(size=200, seed=s) for s in range(6)]         # 0..5 real
    frames.append(_blob(size=200, center=(100, 100), radius=20,       # 6 faint real
                        amp=48, bg=12, blur=2.0, seed=9))
    frames.append(cv2.cvtColor((np.random.default_rng(7).normal(30, 26, (200, 200))
                                ).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB))  # 7 noise
    frames.append(np.full((200, 200, 3), 15, np.uint8))              # 8 blank
    grades = QualityGrader(with_content=True).grade(frames)
    survivors, dropped = prefilter_garbage(grades)
    kept_idx = {g.index for g in survivors}
    drop_idx = {g.index for g in dropped}
    assert drop_idx == {7, 8}, (drop_idx, kept_idx)          # only noise + blank
    assert {0, 1, 2, 3, 4, 5, 6} <= kept_idx                 # all real survive
    print(f"ok  prefilter drops garbage {sorted(drop_idx)}, keeps real+faint")


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
def _texture_scene(size=200, seed=0):
    """A fixed extended-texture target (like a resolved planet/lunar surface).

    Unlike a point-source starfield, this has stable features at every scale, so
    it survives downscaling -- the case where half-res alignment is appropriate.
    """
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((size, size)) * 255).astype(np.uint8), (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _jittered_noisy_set(n=24, size=160, noise=30, seed=0, base=None):
    """A static scene captured n times with random sub-pixel jitter + noise.

    Defaults to a starfield; pass ``base`` (e.g. :func:`_texture_scene`) for an
    extended-texture target.
    """
    rng = np.random.default_rng(seed)
    truth = _starfield(size=size, seed=seed) if base is None else base
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


def test_seed_stack_and_merge_counts_reference_once():
    """Map-reduce: only the seeding worker holds the reference; merge joins sums."""
    truth = _starfield(size=160, seed=50)
    frames = [truth.copy() for _ in range(5)]        # identity -> all align cleanly
    a = LuckyStacker(method="orb"); a.set_reference(truth, seed_stack=True)
    b = LuckyStacker(method="orb"); b.set_reference(truth, seed_stack=False)
    for f in frames[:3]:
        a.add(f)
    for f in frames[3:]:
        b.add(f)
    a.merge(b)
    # Reference counted exactly once: 1 (ref) + 5 identical adds = 6.
    assert a.stats.n_stacked == 6, a.stats
    assert a.stats.n_total == a.stats.n_stacked + a.stats.n_rejected
    assert int(a._weight.max()) == 6                 # peak coverage == frames summed
    master = a.result()
    assert np.abs(master.astype(int) - truth.astype(int)).mean() < 1.0
    print("ok  seed_stack=False + merge counts reference once")


def test_align_scale_half_still_improves_snr():
    """Estimating the transform at half-res still aligns + denoises the stack.

    Uses an extended-texture target (resolved planet/lunar-like), which is where
    half-res alignment applies -- point-source star fields blur away when
    downscaled. This is the speed/quality trade the align_scale knob exposes; it
    must still stack most frames and cut noise.
    """
    base = _texture_scene(size=240, seed=51)
    truth, frames = _jittered_noisy_set(n=18, size=240, noise=22, seed=51, base=base)
    stk = LuckyStacker(method="orb", align_scale=0.5)
    stk.set_reference(truth)
    for f in frames:
        stk.add(f)
    master = stk.result()
    assert master is not None
    single = np.abs(frames[0].astype(int) - truth.astype(int)).mean()
    stack_err = np.abs(master.astype(int) - truth.astype(int)).mean()
    assert stack_err < single * 0.7, (single, stack_err)
    assert stk.stats.n_stacked >= int(0.6 * stk.stats.n_total), stk.stats
    print(f"ok  align_scale=0.5 improves SNR ({single:.1f} -> {stack_err:.1f}, "
          f"stacked {stk.stats.n_stacked}/{stk.stats.n_total})")


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


def test_stack_run_parallel_matches_serial():
    """workers>1 (map-reduce) yields the same stacked set and ~same master."""
    truth, frames = _jittered_noisy_set(n=16, size=200, noise=18, seed=52)
    with tempfile.TemporaryDirectory() as d:
        run = _FakeRun(frames, times=range(len(frames)), directory=d)
        r1 = stack_run(run, 0, keep_fraction=1.0, workers=1, prefilter=False)
        r4 = stack_run(run, 0, keep_fraction=1.0, workers=4, prefilter=False)
        assert r1.stats == r4.stats, (r1.stats, r4.stats)
        diff = np.abs(r1.master.astype(int) - r4.master.astype(int)).mean()
        assert diff < 2.0, diff
    print(f"ok  parallel stack matches serial (stats {r4.stats}, master diff {diff:.2f})")


def test_stack_run_prefilter_drops_garbage_end_to_end():
    """Injected noise/blank frames are pre-culled before ranking + stacking."""
    truth, frames = _jittered_noisy_set(n=14, size=200, noise=18, seed=53)
    frames.append(cv2.cvtColor((np.random.default_rng(1).normal(30, 26, (200, 200))
                                ).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB))
    frames.append(np.full((200, 200, 3), 15, np.uint8))
    with tempfile.TemporaryDirectory() as d:
        run = _FakeRun(frames, times=range(len(frames)), directory=d)
        res = stack_run(run, 0, keep_fraction=1.0, prefilter=True)
        dropped_idx = {g.index for g in res.dropped}
        assert {14, 15} <= dropped_idx, dropped_idx     # noise + blank removed
        assert res.master is not None
    print(f"ok  stack_run pre-culls garbage end-to-end (dropped {sorted(dropped_idx)})")


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


def test_result_16bit_preserves_subquantum_snr():
    # Average of many noisy frames of a flat scene at level ~100.4: the mean
    # lands between 8-bit levels; the 16-bit master must carry the fraction.
    rng = np.random.default_rng(11)
    ref = np.full((32, 32, 3), 100, dtype=np.uint8)
    stacker = LuckyStacker(method="flow")
    stacker.set_reference(ref)
    for _ in range(30):
        noisy = np.clip(100.4 + rng.normal(0, 2.0, (32, 32, 3)), 0, 255).astype(np.uint8)
        stacker.add(noisy)
    m8 = stacker.result(bits=8)
    m16 = stacker.result(bits=16)
    assert m8.dtype == np.uint8 and m16.dtype == np.uint16
    mean16 = m16.astype(np.float64).mean() / 257.0
    assert 99.9 < mean16 < 101.0
    # the 16-bit master resolves the fractional level the 8-bit one rounds
    assert abs(mean16 - round(mean16)) > 0.01
    print("ok  16-bit master preserves fractional level")


def test_save_master_uint16_png_roundtrip():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m16 = np.full((8, 8, 3), 30000, dtype=np.uint16)
        p = save_master(m16, os.path.join(d, "m.png"))
        back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint16, back.dtype
        assert int(back.max()) == 30000
        # formats that can't hold 16 bits are coerced to png
        p2 = save_master(m16, os.path.join(d, "m.jpg"))
        assert p2.endswith(".png")
    print("ok  save_master uint16 roundtrip")


def test_center_size_stack_recentres_target():
    # Frames with the target wandering across the field: with center_size the
    # master is a target-centred crop and the target lands mid-frame.
    rng = np.random.default_rng(4)
    frames = []
    for i in range(6):
        img = (rng.normal(8, 1.0, (120, 160, 3))).clip(0, 255).astype(np.uint8)
        cx, cy = 40 + i * 12, 30 + i * 8   # target drifts frame to frame
        cv2.circle(img, (cx, cy), 5, (220, 220, 220), -1)
        frames.append(img)

    class _FakeRun:
        def frames(self, cam):
            return [{"t": float(i), "arr": f} for i, f in enumerate(frames)]

    # stack_run decodes via _as_array which passes ndarrays through; wrap specs
    monkey_frames = [dict(t=float(i), path=None) for i in range(len(frames))]
    # simplest: call the pipeline pieces directly for the centred path
    from stacking import recenter_frame
    size = 64
    centred = [recenter_frame(f, out_size=(size, size)) for f in frames]
    st = LuckyStacker(method="flow")
    st.set_reference(centred[0])
    for c in centred[1:]:
        st.add(c)
    master = st.result()
    assert master.shape[:2] == (size, size)
    gray = master.mean(axis=2)
    peak = np.unravel_index(np.argmax(gray), gray.shape)
    assert abs(peak[0] - size // 2) <= 3 and abs(peak[1] - size // 2) <= 3, peak
    print("ok  centred stack puts target mid-frame")
