"""Flag-gated bridge from trajectory.py / celestial.py to the Rust astro
engine (skytracker-astro via the skytracker_core extension module).

Phase 1 of the Rust port. Enabled by `config_state.use_rust_astro` (call
`configure(config_state)` once at startup, as main.py does) or by env
`SKYTRACKER_RUST_ASTRO=1` (env `0` force-disables). Every entry point
returns None on ANY failure so callers fall back to the skyfield path --
the Python implementation stays the safety net for the whole strangler
period.

Parity: the engine matches skyfield to 0.03 arcsec (satellites) /
<1 arcsec (bodies, stars); see VALIDATION.md 2026-08-15 entries and
test_rust_astro_parity.py for the live A/B gates.
"""

from __future__ import annotations

import os
import threading

_lock = threading.Lock()
_engine = None
_engine_failed = False
_tle_mtime = None
_enabled_flag = False

DE421_PATH = "de421.bsp"


def configure(config_state):
    """Latch the config flag (called once at startup from main.py)."""
    global _enabled_flag
    _enabled_flag = bool(getattr(config_state, "use_rust_astro", False))


def enabled():
    env = os.environ.get("SKYTRACKER_RUST_ASTRO")
    if env == "1":
        return True
    if env == "0":
        return False
    return _enabled_flag


def _get_engine(tle_path=None):
    """The lazily-built singleton engine, TLE catalog refreshed on file
    mtime change. Returns None (and stays None) if the wheel lacks the
    astro engine."""
    global _engine, _engine_failed, _tle_mtime
    with _lock:
        if _engine is None:
            if _engine_failed:
                return None
            try:
                import skytracker_core as sc

                if not getattr(sc, "ASTRO_ENGINE_AVAILABLE", False):
                    raise ImportError("skytracker_core wheel predates AstroEngine")
                _engine = sc.AstroEngine(
                    DE421_PATH if os.path.exists(DE421_PATH) else None
                )
            except Exception as e:
                print(f"Rust astro engine unavailable ({e}); using skyfield.")
                _engine_failed = True
                return None
        if tle_path is not None:
            try:
                mtime = os.path.getmtime(tle_path)
            except OSError:
                return _engine
            if mtime != _tle_mtime:
                try:
                    n = _engine.load_tle_file(tle_path)
                    _tle_mtime = mtime
                    print(f"Rust astro engine: {n} TLEs loaded from {tle_path}")
                except Exception as e:
                    print(f"Rust astro engine TLE load failed ({e})")
                    return None
        return _engine


def _satnum(sat):
    model = getattr(sat, "model", None)
    return getattr(model, "satnum_str", None) if model is not None else None


def _site(observer):
    return (
        float(observer.latitude.degrees),
        float(observer.longitude.degrees),
        float(observer.elevation.m),
    )


def _tle_path(default="tle_cache.tle"):
    if os.path.exists(default):
        return default
    # CI fallback: the committed golden TLE set (6 satellites), so the
    # parity suite still exercises the satellite path without a cache.
    golden = os.path.join("tests", "golden", "sat_tles.txt")
    return golden if os.path.exists(golden) else None


# ---- satellite paths (trajectory.py) --------------------------------------


def satellite_rows(sat, times, observer, cx, cy, radius):
    """(n, 8) ndarray for one skyfield EarthSatellite on a skyfield Time
    array, or None."""
    satnum = _satnum(sat)
    if satnum is None:
        return None
    engine = _get_engine(_tle_path())
    if engine is None:
        return None
    try:
        lat, lon, elev = _site(observer)
        rows = engine.satellite_rows(
            satnum, list(times.tt), lat, lon, elev, float(cx), float(cy), float(radius)
        )
        return rows if rows.shape[0] == len(times.tt) else None
    except Exception:
        return None


def bulk_rows(sats, times, observer, cx, cy, radius):
    """{satnum_str: (n, 8) ndarray} for many satellites, or None."""
    satnums = [s for s in (_satnum(sat) for sat in sats) if s is not None]
    if not satnums:
        return None
    engine = _get_engine(_tle_path())
    if engine is None:
        return None
    try:
        lat, lon, elev = _site(observer)
        return engine.precompute_trajectories(
            satnums, list(times.tt), lat, lon, elev, float(cx), float(cy), float(radius)
        )
    except Exception:
        return None


def visible_satnums(sats, times, observer, min_alt_deg):
    """Set of satnum_str that clear min_alt_deg, or None."""
    satnums = [s for s in (_satnum(sat) for sat in sats) if s is not None]
    if not satnums:
        return None
    engine = _get_engine(_tle_path())
    if engine is None:
        return None
    try:
        lat, lon, elev = _site(observer)
        return set(
            engine.visible_satnums(
                satnums, list(times.tt), lat, lon, elev, float(min_alt_deg)
            )
        )
    except Exception:
        return None


# ---- celestial paths (celestial.py) ---------------------------------------


def body_altaz_dist(key, t_tt, lat_deg, lon_deg, elev_m):
    """(alt_deg, az_deg, dist_km) for a named body, or None."""
    engine = _get_engine()
    if engine is None:
        return None
    try:
        return engine.body_altaz_dist(
            key, float(t_tt), float(lat_deg), float(lon_deg), float(elev_m)
        )
    except Exception:
        return None


def body_rows(key, times_tt, lat_deg, lon_deg, elev_m):
    """(n, 8) tracking rows for a named body, or None."""
    engine = _get_engine()
    if engine is None:
        return None
    try:
        return engine.body_rows(
            key, [float(t) for t in times_tt], float(lat_deg), float(lon_deg), float(elev_m)
        )
    except Exception:
        return None


def fixed_rows(ra_deg, dec_deg, times_tt, lat_deg, lon_deg, elev_m):
    """(n, 8) tracking rows for a fixed ICRS RA/Dec target, or None."""
    engine = _get_engine()
    if engine is None:
        return None
    try:
        return engine.fixed_rows(
            float(ra_deg),
            float(dec_deg),
            [float(t) for t in times_tt],
            float(lat_deg),
            float(lon_deg),
            float(elev_m),
        )
    except Exception:
        return None
