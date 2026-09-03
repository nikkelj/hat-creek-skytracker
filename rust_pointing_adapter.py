"""Flag-gated bridge from the pointing-model fits to skytracker-pointing
(Phase 2b of the Rust port). Same contract as the other adapters: every
entry point returns None on ANY failure and the numpy implementations
remain the fallback. Enabled by config `use_rust_pointing` (call
configure(config_state) at startup) or env SKYTRACKER_RUST_POINTING=1
(0 force-disables). Coefficient parity 1e-6: test_rust_pointing_parity.py.
"""

from __future__ import annotations

import os

_enabled_flag = False
_mod = None
_failed = False


def configure(config_state):
    global _enabled_flag
    _enabled_flag = bool(getattr(config_state, "use_rust_pointing", False))


def enabled():
    env = os.environ.get("SKYTRACKER_RUST_POINTING")
    if env == "1":
        return True
    if env == "0":
        return False
    return _enabled_flag


def _core():
    global _mod, _failed
    if _mod is None and not _failed:
        try:
            import skytracker_core as sc

            if not getattr(sc, "POINTING_AVAILABLE", False):
                raise ImportError("wheel predates pointing fits")
            _mod = sc
        except Exception as e:
            print(f"Rust pointing fits unavailable ({e}); using numpy.")
            _failed = True
    return _mod


def _clean_samples(samples):
    return [[float(v) for v in s] for s in samples]


def fit_altaz(samples, remove_refraction, seed_terms, free_terms,
              robust, robust_sigma, robust_floor_deg):
    sc = _core()
    if sc is None:
        return None
    try:
        return sc.fit_pointing_model(
            _clean_samples(samples), remove_refraction=bool(remove_refraction),
            seed_terms=dict(seed_terms) if seed_terms else None,
            free_terms=list(free_terms) if free_terms is not None else None,
            robust=bool(robust), robust_sigma=float(robust_sigma),
            robust_floor_deg=float(robust_floor_deg))
    except Exception:
        return None


def fit_eq(samples, lat_deg, seed_terms, free_terms,
           robust, robust_sigma, robust_floor_deg):
    sc = _core()
    if sc is None:
        return None
    try:
        stats = sc.fit_eq_pointing_model(
            _clean_samples(samples), float(lat_deg),
            seed_terms=dict(seed_terms) if seed_terms else None,
            free_terms=list(free_terms) if free_terms is not None else None,
            robust=bool(robust), robust_sigma=float(robust_sigma),
            robust_floor_deg=float(robust_floor_deg))
        stats["lat_deg"] = float(lat_deg)
        return stats
    except Exception:
        return None


def fit_polar_axis(samples_azel, toward_az_deg, toward_alt_deg):
    sc = _core()
    if sc is None:
        return None
    try:
        return sc.fit_polar_axis(
            [[float(a), float(e)] for a, e in samples_azel],
            toward_az_deg=float(toward_az_deg), toward_alt_deg=float(toward_alt_deg))
    except Exception:
        return None
