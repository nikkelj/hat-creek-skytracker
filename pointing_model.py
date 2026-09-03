"""
7-term alt-az TPOINT pointing model.

Relates the mount's nominal (commanded) sky position to where the boresight actually
lands, capturing the repeatable mechanical/optical errors of an alt-az mount:

    IA   - azimuth index (zero-point) error
    IE   - elevation index (zero-point) error
    AN   - azimuth axis tilt toward North
    AW   - azimuth axis tilt toward East/West
    NPAE - non-perpendicularity of the alt and az axes (cone error)
    CA   - collimation (optical axis not perpendicular to the alt axis)
    TF   - tube/OTA flexure with altitude

All coefficients are in degrees. The pointing *error* (observed minus commanded), as a
function of the commanded sky position (az=A, el=E), is:

    dAz = IA - AN*sin(A)*tan(E) - AW*cos(A)*tan(E) + NPAE*tan(E) + CA/cos(E)
    dEl = IE - AN*cos(A)        + AW*sin(A)                       - TF*cos(E)

The model is linear in the seven coefficients, so fitting plate-solved samples is a
single linear least-squares solve. Sampling is restricted below ~80 deg elevation so the
tan(E)/sec(E) terms stay well-conditioned (and to avoid the alt-az zenith singularity).

This module is pure math (numpy) -- it deliberately does NOT touch the coordinate
transforms, which are mirrored in Rust and parity-tested. The correction is applied as a
Python pre-step at target-setting time (see control.py / rust_loop_adapter.py).
"""

import math

import numpy as np

TERM_NAMES = ("IA", "IE", "AN", "AW", "NPAE", "CA", "TF")


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def bennett_refraction_deg(el_deg, pressure_mbar=1010.0, temperature_c=10.0):
    """Apparent-minus-true elevation from atmospheric refraction (Bennett), in degrees.

    Used to strip refraction from plate-solved elevations before the geometric fit so the
    geometric terms aren't polluted. Near/below the horizon the formula is clamped.
    """
    if el_deg < -1.0:
        return 0.0
    # Bennett 1982: R (arcmin) = cot(el + 7.31/(el+4.4)), el in degrees.
    r_arcmin = 1.0 / math.tan(math.radians(el_deg + 7.31 / (el_deg + 4.4)))
    # Scale for non-standard atmosphere.
    r_arcmin *= (pressure_mbar / 1010.0) * (283.0 / (273.0 + temperature_c))
    return r_arcmin / 60.0


def _design_rows(az_deg, el_deg):
    """Return (row_az, row_el): the 7-term basis for the az and el error equations."""
    A = math.radians(az_deg)
    E = math.radians(el_deg)
    tanE = math.tan(E)
    secE = 1.0 / math.cos(E)
    # order: IA, IE, AN, AW, NPAE, CA, TF
    row_az = [1.0, 0.0, -math.sin(A) * tanE, -math.cos(A) * tanE, tanE, secE, 0.0]
    row_el = [0.0, 1.0, -math.cos(A), math.sin(A), 0.0, 0.0, -math.cos(E)]
    return row_az, row_el


class PointingModel:
    """Holds the seven coefficients and applies/inverts the correction."""

    def __init__(self, terms=None):
        self.terms = {k: 0.0 for k in TERM_NAMES}
        if terms:
            self.terms.update({k: float(v) for k, v in terms.items() if k in self.terms})

    # ---- vector of coefficients in canonical order ----
    @property
    def _p(self):
        return np.array([self.terms[k] for k in TERM_NAMES])

    def error(self, az_deg, el_deg):
        """Pointing error (dAz, dEl) in degrees at a commanded sky position."""
        row_az, row_el = _design_rows(az_deg, el_deg)
        p = self._p
        return float(np.dot(row_az, p)), float(np.dot(row_el, p))

    def predict_observed(self, az_cmd, el_cmd):
        """Where the boresight lands if the mount is commanded to (az_cmd, el_cmd)."""
        d_az, d_el = self.error(az_cmd, el_cmd)
        return (az_cmd + d_az) % 360.0, el_cmd + d_el

    def correct(self, az_desired, el_desired):
        """Command to issue so the boresight lands on (az_desired, el_desired).

        First-order inverse (the terms are small): command = desired - error(desired).
        """
        d_az, d_el = self.error(az_desired, el_desired)
        return (az_desired - d_az) % 360.0, el_desired - d_el

    def to_config(self):
        return dict(self.terms)

    # ---- fitting -----------------------------------------------------------
    @classmethod
    def _solve(cls, samples, remove_refraction, seed_terms, free_terms):
        """Core least-squares solve for a fixed sample set. Returns
        (model, cmd, design_cond) where ``cmd`` is the per-sample (az, el, d_az, d_el)
        raw-error list and ``design_cond`` is the condition number of the (free) design
        matrix -- a high value flags an ill-posed grid (terms the points can't separate)."""
        rows, b, cmd = [], [], []
        for az_cmd, el_cmd, az_obs, el_obs in samples:
            el_obs_geo = el_obs - bennett_refraction_deg(el_obs) if remove_refraction else el_obs
            d_az = _wrap180(az_obs - az_cmd)
            d_el = el_obs_geo - el_cmd
            row_az, row_el = _design_rows(az_cmd, el_cmd)
            rows.append(row_az); b.append(d_az)
            rows.append(row_el); b.append(d_el)
            cmd.append((az_cmd, el_cmd, d_az, d_el))

        A = np.array(rows)
        b = np.array(b)

        free_idx = None
        if free_terms is not None:
            free_idx = [i for i, k in enumerate(TERM_NAMES) if k in set(free_terms)]
        if not free_idx:  # full fit (free_terms None, empty, or unrecognised)
            A_used = A
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            model = cls({k: coeffs[i] for i, k in enumerate(TERM_NAMES)})
        else:
            # Partial fit: hold the non-free terms at their seed values and solve only the
            # free columns against the residual error they don't explain.
            seed = {k: 0.0 for k in TERM_NAMES}
            if seed_terms:
                seed.update({k: float(v) for k, v in seed_terms.items() if k in seed})
            p_fixed = np.array([0.0 if i in free_idx else seed[TERM_NAMES[i]] for i in range(len(TERM_NAMES))])
            A_used = A[:, free_idx]
            c_free, _, _, _ = np.linalg.lstsq(A_used, b - A @ p_fixed, rcond=None)
            terms = dict(seed)
            for j, i in enumerate(free_idx):
                terms[TERM_NAMES[i]] = float(c_free[j])
            model = cls(terms)

        try:
            design_cond = float(np.linalg.cond(A_used)) if A_used.size else float('inf')
        except np.linalg.LinAlgError:
            design_cond = float('inf')
        return model, cmd, design_cond

    @classmethod
    def fit(cls, samples, remove_refraction=False, seed_terms=None, free_terms=None,
            robust=False, robust_sigma=4.0, robust_floor_deg=30.0 / 3600.0):
        """Fit a model from samples.

        samples: iterable of (az_cmd, el_cmd, az_obs, el_obs) in degrees, where *_cmd is
        the nominal commanded sky position and *_obs is the plate-solved truth.

        ``free_terms`` enables a *partial* (nightly-refresh) fit: only the named terms are
        solved for, the rest are held fixed at ``seed_terms`` (the previous night's model).
        The mount's mechanical terms (AN/AW/NPAE/CA/TF) are stable across nights -- only the
        index errors IA/IE drift with re-leveling/re-homing -- so seeding from last night
        and refitting just ``["IA", "IE"]`` reaches a good model with a handful of points.
        When ``free_terms`` is None all seven terms are fit (the full alignment).

        ``robust`` does a second pass that rejects samples whose post-fit sky residual
        exceeds BOTH ``robust_sigma`` * (MAD-scaled) sigma AND ``robust_floor_deg`` (an
        absolute floor so tight, clean fits never reject good points), then re-fits on the
        survivors -- so a single bad plate solve (a mis-match, a satellite/plane in the field)
        can't drag the whole model. Skipped if it would drop below the minimum fittable count.

        Returns (model, stats). stats includes per-term coefficients, the sample count, the
        design-matrix condition number, the number of rejected samples (robust), and sky RMS
        before/after the fit (great-circle, az weighted by cos(el)).
        """
        samples = list(samples)

        # Rust fast path (Phase 2b): coefficient parity 1e-6, numpy fallback.
        import rust_pointing_adapter
        if rust_pointing_adapter.enabled():
            stats = rust_pointing_adapter.fit_altaz(
                samples, remove_refraction, seed_terms, free_terms,
                robust, robust_sigma, robust_floor_deg)
            if stats is not None:
                return cls(stats["terms"]), stats

        def sky_rms(pairs):
            sq = 0.0
            for az_cmd, el_cmd, d_az, d_el in pairs:
                cE = math.cos(math.radians(el_cmd))
                sq += (d_az * cE) ** 2 + d_el ** 2
            return math.sqrt(sq / max(1, len(pairs)))

        def residuals(model, cmd):
            out = []
            for az_cmd, el_cmd, d_az, d_el in cmd:
                pr_az, pr_el = model.error(az_cmd, el_cmd)
                out.append((az_cmd, el_cmd, _wrap180(d_az - pr_az), d_el - pr_el))
            return out

        model, cmd, design_cond = cls._solve(samples, remove_refraction, seed_terms, free_terms)
        n_rejected = 0
        MIN_FIT = 4  # below this a 7-term (or even 2-term) fit is meaningless
        if robust and len(samples) > MIN_FIT:
            resid = residuals(model, cmd)
            # Per-sample sky residual magnitude (same great-circle metric as sky_rms).
            mags = np.array([math.hypot(d_az * math.cos(math.radians(el_cmd)), d_el)
                             for az_cmd, el_cmd, d_az, d_el in resid])
            med = float(np.median(mags))
            mad = float(np.median(np.abs(mags - med))) or 1e-9
            thresh = max(med + robust_sigma * 1.4826 * mad, robust_floor_deg)
            keep = [s for s, m in zip(samples, mags) if m <= thresh]
            if MIN_FIT < len(keep) < len(samples):
                n_rejected = len(samples) - len(keep)
                samples = keep
                model, cmd, design_cond = cls._solve(samples, remove_refraction, seed_terms, free_terms)

        rms_before = sky_rms(cmd)
        rms_after = sky_rms(residuals(model, cmd))

        stats = {
            "terms": model.to_config(),
            "n_samples": len(cmd),
            "n_rejected": n_rejected,
            "design_cond": design_cond,
            "rms_before_deg": rms_before,
            "rms_after_deg": rms_after,
            "rms_before_arcmin": rms_before * 60.0,
            "rms_after_arcmin": rms_after * 60.0,
        }
        return model, stats

    def backtest(self, samples, remove_refraction=False):
        """Sky RMS (deg) of this model against held-out samples."""
        sq = 0.0
        n = 0
        for az_cmd, el_cmd, az_obs, el_obs in samples:
            el_obs_geo = el_obs - bennett_refraction_deg(el_obs) if remove_refraction else el_obs
            pr_az, pr_el = self.predict_observed(az_cmd, el_cmd)
            d_az = _wrap180(az_obs - pr_az)
            d_el = el_obs_geo - pr_el
            cE = math.cos(math.radians(el_cmd))
            sq += (d_az * cE) ** 2 + d_el ** 2
            n += 1
        return math.sqrt(sq / max(1, n))


def fibonacci_sky_grid(n, el_min, el_max):
    """~Uniform (az, el) sample points on the hemisphere between el_min and el_max.

    Uses a Fibonacci-sphere distribution restricted to the elevation band, avoiding the
    zenith (alt-az singularity) and points below the elevation mask.
    """
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    # Oversample then filter to the band so we end up near n points.
    m = max(n * 3, 12)
    for i in range(m):
        z = 1.0 - (i + 0.5) / m       # z in (0,1): upper hemisphere
        el = math.degrees(math.asin(z))
        if el < el_min or el > el_max:
            continue
        az = math.degrees((i * golden) % (2.0 * math.pi))
        pts.append((az % 360.0, el))
        if len(pts) >= n:
            break
    return pts
