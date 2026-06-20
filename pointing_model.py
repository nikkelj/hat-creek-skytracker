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
    def fit(cls, samples, remove_refraction=False):
        """Fit a model from samples.

        samples: iterable of (az_cmd, el_cmd, az_obs, el_obs) in degrees, where *_cmd is
        the nominal commanded sky position and *_obs is the plate-solved truth.

        Returns (model, stats). stats includes per-term coefficients, the sample count,
        and sky RMS before/after the fit (great-circle, az weighted by cos(el)).
        """
        rows = []
        b = []
        cmd = []
        for az_cmd, el_cmd, az_obs, el_obs in samples:
            el_obs_geo = el_obs
            if remove_refraction:
                el_obs_geo = el_obs - bennett_refraction_deg(el_obs)
            d_az = _wrap180(az_obs - az_cmd)
            d_el = el_obs_geo - el_cmd
            row_az, row_el = _design_rows(az_cmd, el_cmd)
            rows.append(row_az); b.append(d_az)
            rows.append(row_el); b.append(d_el)
            cmd.append((az_cmd, el_cmd, d_az, d_el))

        A = np.array(rows)
        b = np.array(b)
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        model = cls({k: coeffs[i] for i, k in enumerate(TERM_NAMES)})

        # RMS before (raw error) and after (residual), great-circle.
        def sky_rms(pairs):
            sq = 0.0
            for az_cmd, el_cmd, d_az, d_el in pairs:
                cE = math.cos(math.radians(el_cmd))
                sq += (d_az * cE) ** 2 + d_el ** 2
            return math.sqrt(sq / max(1, len(pairs)))

        rms_before = sky_rms(cmd)
        resid = []
        for az_cmd, el_cmd, d_az, d_el in cmd:
            pr_az, pr_el = model.error(az_cmd, el_cmd)
            resid.append((az_cmd, el_cmd, _wrap180(d_az - pr_az), d_el - pr_el))
        rms_after = sky_rms(resid)

        stats = {
            "terms": model.to_config(),
            "n_samples": len(cmd),
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
