"""ASTAP-inspired motion stabilization for replay / post-processing.

The goal is to remove mount jitter so a tracked target (or the surrounding star
field) stays put frame-to-frame during replay. The approach is inspired by
ASTAP's star-alignment: detect bright/feature points, match them against a single
*reference* frame, fit a similarity transform (translation + rotation + uniform
scale) with RANSAC, and warp the frame onto the reference.

Design notes
------------
* **Stateful, single-anchor.** The reference frame's keypoints/descriptors are
  kept so every incoming frame is aligned to one fixed anchor. This gives
  *absolute* stabilization with no drift accumulation, and -- importantly --
  makes random access (scrubbing the replay slider) work: any frame can be
  aligned independently. Call :meth:`set_reference` / :meth:`reset` to re-anchor.
* **Sub-ROI.** Feature detection can be restricted to a region of interest (in
  full-frame pixel coords) so a fast-moving target does not corrupt the
  background lock -- or, by placing the ROI on the target, so the lock follows
  the target instead. The estimated transform always applies to the whole frame.
* **Pluggable estimator.** :meth:`estimate_transform` is the single seam where
  the alignment algorithm lives, so alternatives (phase-correlation, ORB +
  homography, optical flow, ...) can be swapped in for A/B testing without
  touching the rest of the pipeline. Two methods ship here: ``"orb"`` (feature
  matching, default) and ``"flow"`` (good-features + Lucas-Kanade optical flow).

Everything is pure numpy / OpenCV so it runs headless and is unit-testable.
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - cv2 is expected in the track env
    cv2 = None
    CV2_AVAILABLE = False


# Identity affine (2x3) used whenever alignment cannot be estimated.
IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _to_gray(frame):
    """Return an 8-bit single-channel view of an RGB/gray uint8 array."""
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _roi_mask(shape, roi):
    """Build a uint8 detection mask for ``roi`` (x, y, w, h) over ``shape`` (h, w).

    Returns None when there is no ROI (detect everywhere), which lets OpenCV skip
    masking entirely.
    """
    if roi is None:
        return None
    h, w = shape[:2]
    x, y, rw, rh = (int(v) for v in roi)
    x0 = max(0, min(x, w - 1))
    y0 = max(0, min(y, h - 1))
    x1 = max(x0 + 1, min(x + rw, w))
    y1 = max(y0 + 1, min(y + rh, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


class Stabilizer:
    """Align frames to a fixed reference via star/feature matching.

    Parameters
    ----------
    method : {"orb", "flow"}
        Alignment algorithm. ``"orb"`` matches ORB descriptors (robust to large
        jumps, ASTAP-like). ``"flow"`` tracks good-features with Lucas-Kanade
        optical flow (fast, best for small jitter).
    max_features : int
        Upper bound on detected points.
    roi : tuple or None
        (x, y, w, h) sub-region, in full-frame pixels, to restrict detection to.
    full_affine : bool
        When True, estimate a full affine (allows shear); otherwise a partial
        (similarity) transform -- translation + rotation + uniform scale, which
        is the physically correct model for a rigid sky scene.
    """

    def __init__(self, method="orb", max_features=600, roi=None, full_affine=False,
                 scale_tol=0.12, max_rotation_deg=20.0, max_translation_frac=0.6,
                 min_inliers=8, min_inlier_ratio=0.4):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is required for stabilization")
        self.method = method
        self.max_features = int(max_features)
        self.roi = roi
        self.full_affine = bool(full_affine)

        # Sanity bounds: a star field between consecutive frames should differ by
        # only a small rigid jitter. Anything outside these bounds is a bad RANSAC
        # fit (repetitive star matches), so we reject it and pass the frame through
        # unchanged rather than warp/mirror/zoom it into garbage.
        self.scale_tol = float(scale_tol)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_translation_frac = float(max_translation_frac)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.last_reject_reason = None

        self._orb = None
        self._matcher = None
        if method == "orb":
            self._orb = cv2.ORB_create(nfeatures=self.max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Reference state.
        self._ref_gray = None
        self._ref_kp = None
        self._ref_des = None
        self._ref_pts = None  # for optical-flow method

        # Diagnostics from the most recent stabilize() call.
        self.last_num_inliers = 0
        self.last_ok = False

    # ------------------------------------------------------------------ refs
    def reset(self):
        """Forget the reference frame so the next frame becomes the new anchor."""
        self._ref_gray = None
        self._ref_kp = None
        self._ref_des = None
        self._ref_pts = None

    def set_roi(self, roi):
        """Update the detection ROI and drop cached reference features."""
        self.roi = roi
        # Reference image stays, but its features were detected with the old ROI.
        if self._ref_gray is not None:
            self.set_reference(self._ref_gray)

    def set_reference(self, frame):
        """Set the anchor frame all subsequent frames are aligned to."""
        gray = _to_gray(frame)
        self._ref_gray = gray
        mask = _roi_mask(gray.shape, self.roi)
        if self.method == "orb":
            self._ref_kp, self._ref_des = self._orb.detectAndCompute(gray, mask)
        else:  # flow
            # Rust fast path (Phase 3b): same detector parameters; cv2
            # keeps the ROI-masked path (mask unsupported in Rust).
            pts = None
            if mask is None:
                import rust_imaging_adapter
                if rust_imaging_adapter.enabled():
                    got = rust_imaging_adapter.detect_flow_reference(
                        gray.astype(np.float32), self.max_features)
                    if got is not None and len(got):
                        pts = got.reshape(-1, 1, 2).astype(np.float32)
            if pts is None:
                pts = cv2.goodFeaturesToTrack(
                    gray, maxCorners=self.max_features, qualityLevel=0.01,
                    minDistance=8, mask=mask,
                )
            self._ref_pts = pts

    @property
    def has_reference(self):
        return self._ref_gray is not None

    # ------------------------------------------------------------- estimate
    def estimate_transform(self, frame):
        """Estimate the 2x3 affine mapping ``frame`` onto the reference frame.

        Returns (M, num_inliers). M is None when estimation fails (too few
        matches). This is the single pluggable seam for alignment algorithms.
        """
        gray = _to_gray(frame)
        if self.method == "orb":
            return self._estimate_orb(gray)
        return self._estimate_flow(gray)

    def _estimate_orb(self, gray):
        if self._ref_des is None or len(self._ref_kp) < 3:
            return None, 0
        mask = _roi_mask(gray.shape, self.roi)
        kp, des = self._orb.detectAndCompute(gray, mask)
        if des is None or len(kp) < 3:
            return None, 0

        # Lowe ratio test on the 2 nearest descriptor matches.
        matches = self._matcher.knnMatch(des, self._ref_des, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 3:
            return None, 0

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self._fit(src, dst)

    def _estimate_flow(self, gray):
        if self._ref_pts is None or len(self._ref_pts) < 3:
            return None, 0
        # Rust fast path (Phase 3b): LK + RANSAC + plausibility bounds in
        # one call (similarity model, no ROI); cv2 fallback otherwise.
        if not self.full_affine and self.roi is None:
            import rust_imaging_adapter
            if rust_imaging_adapter.enabled():
                res = rust_imaging_adapter.estimate_flow(
                    self._ref_gray.astype(np.float32), self._ref_pts,
                    gray.astype(np.float32), self.min_inliers,
                    self.min_inlier_ratio, self.scale_tol,
                    self.max_rotation_deg, self.max_translation_frac)
                if res is not None:
                    M, n, reason = res
                    if reason:
                        self.last_reject_reason = reason
                    return M, n
        # Track reference points forward into the new frame.
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._ref_gray, gray, self._ref_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None:
            return None, 0
        status = status.reshape(-1).astype(bool)
        if status.sum() < 3:
            return None, 0
        src = nxt[status].reshape(-1, 1, 2)          # points in the current frame
        dst = self._ref_pts[status].reshape(-1, 1, 2)  # where they sit in the reference
        return self._fit(src, dst)

    def _fit(self, src, dst):
        """RANSAC-fit the transform mapping src (current) -> dst (reference).

        Rejects fits with too few / too low a fraction of inliers, which is the
        common failure mode on repetitive star fields.
        """
        if self.full_affine:
            M, inliers = cv2.estimateAffine2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        else:
            M, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return None, 0
        n = int(inliers.sum()) if inliers is not None else 0
        total = len(src)
        if n < self.min_inliers or (total and n / total < self.min_inlier_ratio):
            self.last_reject_reason = f"inliers {n}/{total}"
            return None, n
        return M, n

    def _validate(self, M, shape):
        """True if the transform is a plausible small rigid jitter (no flip/zoom).

        Decomposes the 2x3 affine into scale / rotation / translation and checks
        each against the configured bounds. Catches reflections (negative
        determinant), runaway zoom, large rotations, and huge shifts -- the
        artefacts that show up as a mirrored or warped image.
        """
        h, w = shape[:2]
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if det <= 0:
            self.last_reject_reason = "reflection (det<=0)"
            return False
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        if not (1.0 - self.scale_tol <= scale <= 1.0 + self.scale_tol):
            self.last_reject_reason = f"scale {scale:.3f}"
            return False
        rot = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if rot > self.max_rotation_deg:
            self.last_reject_reason = f"rotation {rot:.1f}deg"
            return False
        if abs(M[0, 2]) > self.max_translation_frac * w or \
                abs(M[1, 2]) > self.max_translation_frac * h:
            self.last_reject_reason = f"translation ({M[0, 2]:.0f},{M[1, 2]:.0f})"
            return False
        return True

    # ------------------------------------------------------------- stabilize
    def stabilize(self, frame, border=None):
        """Return (warped_frame, M) aligning ``frame`` to the reference.

        If no reference is set yet, this frame becomes the reference and is
        returned unchanged with the identity transform. On estimation failure or
        an implausible (rejected) transform the original frame is returned with
        identity (no warp), so playback never stalls and we never show a
        mirrored/warped frame.

        Revealed edges are filled with black (BORDER_CONSTANT) -- the expected
        "stabilized with letterbox" look. (The old BORDER_REFLECT default mirrored
        the image into the revealed border, which read as the frame being flipped.)
        """
        if border is None:
            border = cv2.BORDER_CONSTANT
        self.last_reject_reason = None
        if not self.has_reference:
            self.set_reference(frame)
            self.last_ok = True
            self.last_num_inliers = 0
            return frame, IDENTITY_AFFINE.copy()

        M, n = self.estimate_transform(frame)
        self.last_num_inliers = n
        if M is None or not self._validate(M, frame.shape):
            self.last_ok = False
            return frame, IDENTITY_AFFINE.copy()

        self.last_ok = True
        h, w = frame.shape[:2]
        warped = cv2.warpAffine(
            frame, M.astype(np.float32), (w, h),
            flags=cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0))
        return warped, M


def apply_affine_point(M, x, y):
    """Map a single (x, y) point through a 2x3 affine matrix."""
    M = np.asarray(M, dtype=np.float64)
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return nx, ny
