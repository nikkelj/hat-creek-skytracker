"""
7-term equatorial TPOINT pointing model (the Eq-mode counterpart of pointing_model.py).

Polar alignment removes the gross polar-axis misalignment mechanically, but the mount is
never perfect: residual axis tilt, axis non-perpendicularity, collimation, index zero-points
and tube flexure all remain. This model captures them so go-to lands the target after a
polar-aligned mount is synced. It operates in the mount's hour-angle / declination axes:

    IH - hour-angle index (zero-point) error
    ID - declination index (zero-point) error
    NP - HA/Dec axis non-perpendicularity
    CH - collimation (optical axis vs the Dec axis)
    ME - polar-axis elevation error (residual after polar align)
    MA - polar-axis azimuth error    (residual after polar align)
    TF - tube flexure (gravity sag, toward the horizon)

All coefficients are in degrees. The error (observed minus commanded) as a function of the
commanded mount position (h, d) and the site latitude phi:

    dHA  = IH + NP*tan(d) + CH*sec(d) + ME*sin(h)*tan(d) - MA*cos(h)*tan(d) + TF*flex_h
    dDec = ID                          + ME*cos(h)        + MA*sin(h)        + TF*flex_d

The IH/ID/NP/CH/ME/MA basis mirrors the alt-az model (the polar axis plays the role of the
azimuth axis). Only flexure differs: it is gravity-driven, so it depends on the true altitude
(zenith distance) and parallactic angle q -- NOT on declination -- and therefore needs the
latitude. flex_h = cos(el)*sin(q)*sec(d), flex_d = -cos(el)*cos(q), so the great-circle
flexure magnitude is TF*cos(el) (max near the horizon), exactly like the alt-az TF*cos(E).

Linear in the seven coefficients, so fitting plate-solved samples is one linear least-squares
solve -- the fit/backtest/robust machinery parallels PointingModel.
"""

import math

import numpy as np

TERM_NAMES = ("IH", "ID", "NP", "CH", "ME", "MA", "TF")


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _design_rows(h_deg, d_deg, lat_deg):
    """(row_ha, row_dec): the 7-term basis for the HA and Dec error equations at latitude."""
    H = math.radians(h_deg)
    D = math.radians(d_deg)
    phi = math.radians(lat_deg)
    tanD = math.tan(D)
    secD = 1.0 / math.cos(D)
    # Tube flexure: droop toward the horizon, magnitude cos(el), projected into (HA, Dec)
    # through the parallactic angle q (needs the true altitude el and q from h, d, phi).
    sin_el = math.sin(phi) * math.sin(D) + math.cos(phi) * math.cos(D) * math.cos(H)
    sin_el = max(-1.0, min(1.0, sin_el))
    cos_el = math.cos(math.asin(sin_el))
    q = math.atan2(math.sin(H), math.tan(phi) * math.cos(D) - math.sin(D) * math.cos(H))
    flex_h = cos_el * math.sin(q) * secD
    flex_d = -cos_el * math.cos(q)
    # order: IH, ID, NP, CH, ME, MA, TF
    row_ha = [1.0, 0.0, tanD, secD, math.sin(H) * tanD, -math.cos(H) * tanD, flex_h]
    row_dec = [0.0, 1.0, 0.0, 0.0, math.cos(H), math.sin(H), flex_d]
    return row_ha, row_dec


class EquatorialPointingModel:
    """Holds the seven coefficients (and the site latitude for the flexure term)."""

    def __init__(self, terms=None, lat_deg=0.0):
        self.terms = {k: 0.0 for k in TERM_NAMES}
        if terms:
            self.terms.update({k: float(v) for k, v in terms.items() if k in self.terms})
        self.lat_deg = float(lat_deg)

    @property
    def _p(self):
        return np.array([self.terms[k] for k in TERM_NAMES])

    def error(self, h_deg, d_deg):
        """Pointing error (dHA, dDec) in degrees at a commanded mount position."""
        row_ha, row_dec = _design_rows(h_deg, d_deg, self.lat_deg)
        p = self._p
        return float(np.dot(row_ha, p)), float(np.dot(row_dec, p))

    def predict_observed(self, h_cmd, d_cmd):
        """Where the boresight lands (HA, Dec) if the mount is commanded to (h_cmd, d_cmd)."""
        d_ha, d_dec = self.error(h_cmd, d_cmd)
        return _wrap180(h_cmd + d_ha), d_cmd + d_dec

    def correct(self, h_desired, d_desired):
        """Command (HA, Dec) to issue so the boresight lands on (h_desired, d_desired).

        First-order inverse (the terms are small): command = desired - error(desired).
        """
        d_ha, d_dec = self.error(h_desired, d_desired)
        return _wrap180(h_desired - d_ha), d_desired - d_dec

    def to_config(self):
        return dict(self.terms)

    # ---- fitting (parallels pointing_model.PointingModel) -------------------
    @classmethod
    def _solve(cls, samples, lat_deg, seed_terms, free_terms):
        rows, b, cmd = [], [], []
        for h_cmd, d_cmd, h_obs, d_obs in samples:
            d_ha = _wrap180(h_obs - h_cmd)
            d_dec = d_obs - d_cmd
            row_ha, row_dec = _design_rows(h_cmd, d_cmd, lat_deg)
            rows.append(row_ha); b.append(d_ha)
            rows.append(row_dec); b.append(d_dec)
            cmd.append((h_cmd, d_cmd, d_ha, d_dec))

        A = np.array(rows)
        b = np.array(b)
        free_idx = None
        if free_terms is not None:
            free_idx = [i for i, k in enumerate(TERM_NAMES) if k in set(free_terms)]
        if not free_idx:
            A_used = A
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            model = cls({k: coeffs[i] for i, k in enumerate(TERM_NAMES)}, lat_deg=lat_deg)
        else:
            seed = {k: 0.0 for k in TERM_NAMES}
            if seed_terms:
                seed.update({k: float(v) for k, v in seed_terms.items() if k in seed})
            p_fixed = np.array([0.0 if i in free_idx else seed[TERM_NAMES[i]]
                                for i in range(len(TERM_NAMES))])
            A_used = A[:, free_idx]
            c_free, _, _, _ = np.linalg.lstsq(A_used, b - A @ p_fixed, rcond=None)
            terms = dict(seed)
            for j, i in enumerate(free_idx):
                terms[TERM_NAMES[i]] = float(c_free[j])
            model = cls(terms, lat_deg=lat_deg)

        try:
            design_cond = float(np.linalg.cond(A_used)) if A_used.size else float('inf')
        except np.linalg.LinAlgError:
            design_cond = float('inf')
        return model, cmd, design_cond

    @classmethod
    def fit(cls, samples, lat_deg, seed_terms=None, free_terms=None,
            robust=False, robust_sigma=4.0, robust_floor_deg=30.0 / 3600.0):
        """Fit a model from samples (h_cmd, d_cmd, h_obs, d_obs) at the given latitude.

        ``free_terms`` enables a partial fit (hold the rest at ``seed_terms``); ``robust``
        rejects gross-outlier solves (MAD with an absolute floor). Returns (model, stats)
        with per-term coefficients, n_samples, n_rejected, design_cond, and great-circle sky
        RMS before/after (HA weighted by cos(dec)).
        """
        samples = list(samples)

        def sky_rms(pairs):
            sq = 0.0
            for h_cmd, d_cmd, d_ha, d_dec in pairs:
                cD = math.cos(math.radians(d_cmd))
                sq += (d_ha * cD) ** 2 + d_dec ** 2
            return math.sqrt(sq / max(1, len(pairs)))

        def residuals(model, cmd):
            out = []
            for h_cmd, d_cmd, d_ha, d_dec in cmd:
                pr_ha, pr_dec = model.error(h_cmd, d_cmd)
                out.append((h_cmd, d_cmd, _wrap180(d_ha - pr_ha), d_dec - pr_dec))
            return out

        model, cmd, design_cond = cls._solve(samples, lat_deg, seed_terms, free_terms)
        n_rejected = 0
        MIN_FIT = 4
        if robust and len(samples) > MIN_FIT:
            resid = residuals(model, cmd)
            mags = np.array([math.hypot(d_ha * math.cos(math.radians(d_cmd)), d_dec)
                             for h_cmd, d_cmd, d_ha, d_dec in resid])
            med = float(np.median(mags))
            mad = float(np.median(np.abs(mags - med))) or 1e-9
            thresh = max(med + robust_sigma * 1.4826 * mad, robust_floor_deg)
            keep = [s for s, m in zip(samples, mags) if m <= thresh]
            if MIN_FIT < len(keep) < len(samples):
                n_rejected = len(samples) - len(keep)
                samples = keep
                model, cmd, design_cond = cls._solve(samples, lat_deg, seed_terms, free_terms)

        rms_before = sky_rms(cmd)
        rms_after = sky_rms(residuals(model, cmd))
        stats = {
            "terms": model.to_config(),
            "lat_deg": lat_deg,
            "n_samples": len(cmd),
            "n_rejected": n_rejected,
            "design_cond": design_cond,
            "rms_before_deg": rms_before,
            "rms_after_deg": rms_after,
            "rms_before_arcmin": rms_before * 60.0,
            "rms_after_arcmin": rms_after * 60.0,
        }
        return model, stats

    def backtest(self, samples):
        """Great-circle sky RMS (deg) of this model against held-out (h,d,h_obs,d_obs)."""
        sq = 0.0
        n = 0
        for h_cmd, d_cmd, h_obs, d_obs in samples:
            pr_ha, pr_dec = self.predict_observed(h_cmd, d_cmd)
            d_ha = _wrap180(h_obs - pr_ha)
            d_dec = d_obs - pr_dec
            cD = math.cos(math.radians(d_cmd))
            sq += (d_ha * cD) ** 2 + d_dec ** 2
            n += 1
        return math.sqrt(sq / max(1, n))
