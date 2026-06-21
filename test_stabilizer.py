"""Robust test suite for the motion stabilizer (stabilizer.py).

Motivated by the stabilizer occasionally producing "whacky" output (mirrored /
flipped / zoomed frames). These tests assert not just that error is reduced, but
that the recovered transform is a *plausible rigid jitter* -- positive
determinant (no reflection), scale ~= 1 (no zoom), small rotation, and that
revealed borders are filled black (not mirror-reflected). They also confirm the
stabilizer *refuses* to transform when matches are bad, rather than warping the
frame into garbage.

Run with the track env:
    C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe test_stabilizer.py
"""

import numpy as np
import cv2

from stabilizer import Stabilizer

W, H = 320, 240


def make_starfield(n=140, seed=0, w=W, h=H, margin=24):
    """Dark frame with n Gaussian 'stars' -- a realistic stabilization input."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xs = rng.integers(margin, w - margin, n)
    ys = rng.integers(margin, h - margin, n)
    bright = rng.integers(120, 256, n)
    for x, y, b in zip(xs, ys, bright):
        img[y, x] = (b, b, b)
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return np.clip(img.astype(int) * 3, 0, 255).astype(np.uint8)


def apply_affine(img, M):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M.astype(np.float32), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def shift_M(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)


def rot_scale_M(deg, scale, w=W, h=H):
    return cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)


def decompose(M):
    """Return (scale, rotation_deg, det, tx, ty) of a 2x3 affine."""
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return scale, rot, det, float(M[0, 2]), float(M[1, 2])


def residual(a, b):
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


# ---------------------------------------------------------------------------
def test_pure_translation_recovers_and_is_sane():
    base = make_starfield(seed=1)
    for method in ("orb", "flow"):
        for dx, dy in [(8, 0), (-6, 5), (12, -9), (3, 14)]:
            shifted = apply_affine(base, shift_M(dx, dy))
            st = Stabilizer(method=method)
            st.set_reference(base)
            warped, M = st.stabilize(shifted)
            assert st.last_ok, (method, dx, dy, "rejected")
            scale, rot, det, tx, ty = decompose(M)
            # plausible rigid transform
            assert det > 0, (method, dx, dy, "reflection!")
            assert abs(scale - 1.0) < 0.05, (method, scale)
            assert abs(rot) < 3.0, (method, rot)
            # maps shifted back: should translate by ~(-dx, -dy)
            assert abs(tx + dx) < 2.5 and abs(ty + dy) < 2.5, (method, dx, dy, tx, ty)
            # and it actually reduces error in the overlap region
            assert residual(base, warped) < residual(base, shifted) * 0.7
    print("ok  pure translation: recovered, sane, no reflection (orb+flow)")


def test_rotation_and_scale_recovered():
    base = make_starfield(seed=2)
    for deg, scl in [(2.0, 1.0), (-3.0, 1.0), (0.0, 1.03), (1.5, 0.98)]:
        moved = apply_affine(base, rot_scale_M(deg, scl))
        st = Stabilizer(method="orb")
        st.set_reference(base)
        warped, M = st.stabilize(moved)
        assert st.last_ok, (deg, scl, "rejected")
        scale, rot, det, _, _ = decompose(M)
        assert det > 0
        # M is the inverse (moved->base). With decompose()'s sign convention
        # (rot = atan2(M[1,0], M[0,0]) = -angle for getRotationMatrix2D), the
        # inverse of a +deg rotation reads back as +deg; scale reads back as 1/scl.
        assert abs(rot - deg) < 1.5, (deg, rot)
        assert abs(scale - 1.0 / scl) < 0.05, (scl, scale)
        assert residual(base, warped) < residual(base, moved) * 0.8
    print("ok  rotation + scale recovered")


def test_robust_to_noise():
    rng = np.random.default_rng(3)
    base = make_starfield(seed=3)
    noisy_base = np.clip(base.astype(int) + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    shifted = apply_affine(base, shift_M(10, -7))
    shifted = np.clip(shifted.astype(int) + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    st = Stabilizer(method="orb")
    st.set_reference(noisy_base)
    warped, M = st.stabilize(shifted)
    assert st.last_ok
    _, _, det, tx, ty = decompose(M)
    assert det > 0 and abs(tx + 10) < 3 and abs(ty - 7) < 3
    print("ok  robust to additive noise")


def test_revealed_border_is_black_not_mirrored():
    """The key bug fix: shifting a frame back must letterbox with black, not a
    mirror of the image content."""
    base = make_starfield(seed=4, margin=40)
    dx = 22
    shifted = apply_affine(base, shift_M(dx, 0))  # content moved right
    st = Stabilizer(method="orb")
    st.set_reference(base)
    warped, M = st.stabilize(shifted)
    assert st.last_ok
    # stabilizer shifts content back left by ~dx, revealing the right edge.
    right_cols = warped[:, -(dx - 6):, :]
    assert right_cols.mean() < 4.0, f"revealed border not black: mean={right_cols.mean():.2f}"
    print("ok  revealed border filled black (no mirror artefact)")


def test_rejects_horizontal_mirror():
    """A genuinely mirrored frame cannot be a similarity transform -> must be
    rejected (pass-through), never 'corrected' into a flip."""
    base = make_starfield(seed=5)
    mirrored = base[:, ::-1, :].copy()
    st = Stabilizer(method="orb")
    st.set_reference(base)
    warped, M = st.stabilize(mirrored)
    assert not st.last_ok, "mirror was accepted!"
    assert np.array_equal(warped, mirrored), "rejected fit should pass frame through unchanged"
    print(f"ok  rejects horizontal mirror ({st.last_reject_reason})")


def test_rejects_unrelated_frame():
    """Two independent star fields share no real correspondence -> reject."""
    base = make_starfield(seed=6)
    other = make_starfield(seed=999)
    st = Stabilizer(method="orb")
    st.set_reference(base)
    warped, M = st.stabilize(other)
    # Either too-few inliers or validation must trip; output is unchanged.
    assert not st.last_ok
    assert np.array_equal(warped, other)
    print(f"ok  rejects unrelated frame ({st.last_reject_reason})")


def test_rejects_huge_translation():
    base = make_starfield(seed=7)
    huge = apply_affine(base, shift_M(W * 0.8, 0))  # shifted almost entirely off
    st = Stabilizer(method="orb", max_translation_frac=0.6)
    st.set_reference(base)
    warped, M = st.stabilize(huge)
    # If a fit is found it must be rejected by the translation bound.
    if st.last_ok:
        _, _, _, tx, ty = decompose(M)
        assert abs(tx) <= 0.6 * W
    print("ok  huge translation rejected / bounded")


def test_blank_frame_is_identity():
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    st = Stabilizer(method="orb")
    st.set_reference(blank)            # no features
    warped, M = st.stabilize(blank)
    assert np.array_equal(warped, blank)
    assert np.allclose(M, np.array([[1, 0, 0], [0, 1, 0]]))
    print("ok  blank frame -> identity (no crash)")


def test_output_shape_and_dtype_preserved():
    base = make_starfield(seed=8)
    shifted = apply_affine(base, shift_M(5, 5))
    st = Stabilizer(method="orb")
    st.set_reference(base)
    warped, _ = st.stabilize(shifted)
    assert warped.shape == base.shape and warped.dtype == np.uint8
    print("ok  output shape/dtype preserved")


def test_roi_restricts_detection():
    """With an ROI on a quiet corner (no stars), the fit should reject rather
    than warp on noise."""
    base = make_starfield(seed=9, margin=60)
    shifted = apply_affine(base, shift_M(7, 0))
    st = Stabilizer(method="orb", roi=(0, 0, 30, 30))  # tiny empty corner
    st.set_reference(base)
    warped, M = st.stabilize(shifted)
    assert not st.last_ok
    print("ok  ROI restricts detection region")


if __name__ == "__main__":
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
