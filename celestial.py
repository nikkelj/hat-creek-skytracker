"""
Celestial targets for the skyplot and PROGRAM tracking: the sun, moon and
planets (from the app's de421 ephemeris), and the Messier + NGC deep-sky
catalogues (OpenNGC, catalogs/openngc.csv).

Everything is exposed through one CelestialCatalog singleton (get_celestial),
mirroring star_catalog. Two accuracy tiers, deliberately:

  * Plot positions (compute_plot_objects) -- the solar-system bodies get the
    full Skyfield topocentric apparent treatment (11 bodies, cheap); the
    fixed-RA/Dec deep-sky objects use a vectorised hour-angle transform on
    apparent sidereal time (J2000 coords, no precession: ~0.4 deg -- invisible
    on a polar plot, thousands of objects per frame for free). Results are
    TTL-cached so both render threads share one computation.

  * Tracking trajectories (build_trajectory) -- ALWAYS full Skyfield apparent
    positions, sampled over a sliding window in the standard 8-column format
    [time_tt, alt, az, dist_km, px, py, az_rate, el_rate] consumed by
    interpolate_position_data_and_rates. px/py are left 0: the control path
    never reads them and the selected-object polyline is projected from az/el
    at draw time (plot geometry lives in the render thread, not here).

Selection keys are stable strings ('sun', 'planet:Jupiter', 'M031',
'NGC0224', 'star:HIP32349') -- they are the PROGRAM-track PID-reset identity.
"""

import csv
import math
import threading

import numpy as np

from skyfield.api import Star, load, wgs84

from trajectory import _unwrap_az_diff, live_tt

OPENNGC_PATH = "catalogs/openngc.csv"

# Sliding tracking-trajectory window: rebuild when live time is within
# TRAJ_REBUILD_MARGIN_MIN of the end. Celestial motion is smooth, so coarse
# 15 s samples interpolate to well under an arcsecond.
TRAJ_DURATION_MIN = 90.0
TRAJ_STEP_SEC = 15.0
TRAJ_REBUILD_MARGIN_MIN = 10.0

# Plot-position cache quantum (seconds of tracking-clock time). The sky moves
# ~15 arcsec in 1 s -- far below a polar-plot pixel.
PLOT_CACHE_QUANTUM_SEC = 1.0

# Solar-system body table: key -> (display name, de421 segment name candidates,
# draw color, draw radius px). Pluto is not a planet, but every kid asks.
SOLAR_BODIES = [
    ("sun",     "Sun",     ("sun",),                                (255, 220, 60),  7),
    ("moon",    "Moon",    ("moon",),                               (210, 210, 215), 6),
    ("planet:Mercury", "Mercury", ("mercury",),                     (170, 160, 150), 3),
    ("planet:Venus",   "Venus",   ("venus",),                       (245, 235, 200), 4),
    ("planet:Mars",    "Mars",    ("mars", "mars barycenter"),      (230, 110, 70),  4),
    ("planet:Jupiter", "Jupiter", ("jupiter barycenter",),          (225, 190, 140), 5),
    ("planet:Saturn",  "Saturn",  ("saturn barycenter",),           (235, 215, 160), 5),
    ("planet:Uranus",  "Uranus",  ("uranus barycenter",),           (150, 220, 225), 3),
    ("planet:Neptune", "Neptune", ("neptune barycenter",),          (110, 150, 245), 3),
    ("planet:Pluto",   "Pluto",   ("pluto barycenter",),            (200, 180, 170), 2),
]

# M45 (Pleiades) has no NGC number, so OpenNGC's NGC-keyed rows miss it.
MESSIER_PATCH = [
    {"key": "M045", "name": "M45 Pleiades", "ra_deg": 56.75, "dec_deg": 24.1167,
     "mag": 1.6, "kind": "messier", "type": "OCl"},
]


def _parse_hms(s):
    """OpenNGC 'HH:MM:SS.ss' -> degrees."""
    h, m, sec = s.split(":")
    return (int(h) + int(m) / 60.0 + float(sec) / 3600.0) * 15.0


def _parse_dms(s):
    """OpenNGC '+DD:MM:SS.s' -> degrees."""
    sign = -1.0 if s.strip().startswith("-") else 1.0
    d, m, sec = s.strip().lstrip("+-").split(":")
    return sign * (int(d) + int(m) / 60.0 + float(sec) / 3600.0)


class CelestialCatalog:
    """Solar-system bodies + Messier/NGC deep-sky objects for one observer."""

    def __init__(self, ephemeris=None, ngc_path=OPENNGC_PATH):
        eph = ephemeris if ephemeris is not None else load("de421.bsp")
        self._earth = eph["earth"]
        self._bodies = {}       # key -> (name, skyfield segment, color, radius)
        for key, name, candidates, color, rad in SOLAR_BODIES:
            body = None
            for cand in candidates:
                try:
                    body = eph[cand]
                    break
                except (KeyError, ValueError):
                    continue
            if body is not None:
                self._bodies[key] = (name, body, color, rad)

        # Deep-sky catalogues (parallel numpy arrays per catalogue).
        self.messier = self._empty_dso()
        self.ngc = self._empty_dso()
        self._load_openngc(ngc_path)

        self._observers = {}
        self._lock = threading.Lock()
        self._plot_cache = None   # (cache_key, objects list)
        self._named_stars = None  # lazy: top-100 bright star arrays

    @staticmethod
    def _empty_dso():
        return {"key": [], "name": [], "ra_deg": np.empty(0), "dec_deg": np.empty(0),
                "mag": np.empty(0)}

    def _load_openngc(self, path):
        """Parse OpenNGC: Messier objects (via the M cross-ref column, plus the
        M45 patch) and all NGC-numbered objects with a finite V (or B) mag."""
        mrows, nrows = [], []
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    ra_s, dec_s = row.get("RA", ""), row.get("Dec", "")
                    if not ra_s or not dec_s:
                        continue
                    try:
                        ra, dec = _parse_hms(ra_s), _parse_dms(dec_s)
                    except (ValueError, IndexError):
                        continue
                    mag_s = row.get("V-Mag") or row.get("B-Mag") or ""
                    try:
                        mag = float(mag_s)
                    except ValueError:
                        mag = float("nan")
                    common = (row.get("Common names") or "").split(",")[0].strip()
                    m_num = (row.get("M") or "").strip()
                    if m_num:
                        label = f"M{int(m_num)}"
                        name = f"{label} {common}" if common else label
                        mrows.append((f"M{m_num}", name, ra, dec,
                                      mag if math.isfinite(mag) else 9.9))
                    if row.get("Name", "").startswith("NGC") and math.isfinite(mag):
                        name = row["Name"]
                        disp = f"{name} {common}" if common else name
                        nrows.append((name, disp, ra, dec, mag))
        except OSError as e:
            print(f"Celestial: OpenNGC catalogue not available ({e}) - "
                  f"Messier/NGC overlays disabled")
            return
        for patch in MESSIER_PATCH:
            if not any(r[0] == patch["key"] for r in mrows):
                mrows.append((patch["key"], patch["name"], patch["ra_deg"],
                              patch["dec_deg"], patch["mag"]))
        for rows, dest in ((mrows, "messier"), (nrows, "ngc")):
            if rows:
                setattr(self, dest, {
                    "key": [r[0] for r in rows],
                    "name": [r[1] for r in rows],
                    "ra_deg": np.array([r[2] for r in rows]),
                    "dec_deg": np.array([r[3] for r in rows]),
                    "mag": np.array([r[4] for r in rows]),
                })
        print(f"Celestial: {len(self._bodies)} solar-system bodies, "
              f"{len(mrows)} Messier, {len(nrows)} NGC objects loaded")

    # ---- lookup --------------------------------------------------------------

    def display_name(self, key):
        """Human-friendly name for any selection key ('moon' -> 'Moon',
        'M031' -> 'M31 Andromeda Galaxy', 'star:HIP32349' -> 'Sirius')."""
        if key in self._bodies:
            return self._bodies[key][0]
        for cat in (self.messier, self.ngc):
            if key in cat["key"]:
                return cat["name"][cat["key"].index(key)]
        if key.startswith("star:HIP"):
            from star_catalog import get_catalog
            return get_catalog().name_for(int(key[8:]))
        return key

    def _fixed_radec(self, key):
        """(ra_deg, dec_deg) for a fixed-position key, or None."""
        for cat in (self.messier, self.ngc):
            if key in cat["key"]:
                i = cat["key"].index(key)
                return float(cat["ra_deg"][i]), float(cat["dec_deg"][i])
        if key.startswith("star:HIP"):
            from star_catalog import get_catalog
            sc = get_catalog()
            hip = int(key[8:])
            idx = np.where(sc.hip == hip)[0]
            if idx.size:
                return float(sc.ra_deg[idx[0]]), float(sc.dec_deg[idx[0]])
        return None

    def _named_star_arrays(self):
        """Lazy arrays for the top-100 bright stars (the click-selectable,
        always-labelled set): {'key', 'name', 'ra_deg', 'dec_deg'}. Empty dict
        of arrays if the Hipparcos catalogue is unavailable."""
        if self._named_stars is not None:
            return self._named_stars
        out = {"key": [], "name": [], "ra_deg": np.empty(0), "dec_deg": np.empty(0),
               "mag": np.empty(0)}
        try:
            from star_catalog import get_catalog
            sc = get_catalog()
            named = sc.top_named_hips(100)
            idx = np.where(np.isin(sc.hip, list(named)))[0]
            out = {
                "key": [f"star:HIP{int(sc.hip[i])}" for i in idx],
                "name": [sc.name_for(int(sc.hip[i])) for i in idx],
                "ra_deg": sc.ra_deg[idx].astype(float),
                "dec_deg": sc.dec_deg[idx].astype(float),
                "mag": sc.mag[idx].astype(float),
            }
        except Exception as e:
            print(f"Celestial: named-star set unavailable ({e})")
        self._named_stars = out
        return out

    def _observer(self, lat_deg, lon_deg, elev_m):
        okey = (round(lat_deg, 6), round(lon_deg, 6), round(elev_m, 1))
        o = self._observers.get(okey)
        if o is None:
            o = self._earth + wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)
            self._observers[okey] = o
        return o

    # ---- plot positions -------------------------------------------------------

    def solar_system_altaz(self, lat_deg, lon_deg, elev_m, ts, t_tt):
        """[(key, name, az, el, dist_km, color, radius_px)] for all bodies.
        Full Skyfield apparent topocentric positions (11 observes, ~ms)."""
        observer = self._observer(lat_deg, lon_deg, elev_m).at(ts.tt_jd(t_tt))
        out = []
        for key, (name, body, color, rad) in self._bodies.items():
            alt, az, dist = observer.observe(body).apparent().altaz()
            out.append((key, name, az.degrees, alt.degrees, dist.km, color, rad))
        return out

    def fixed_altaz(self, ra_deg, dec_deg, lat_deg, lon_deg, ts, t_tt):
        """Vectorised J2000 RA/Dec -> (az_deg, el_deg) via the hour angle at
        apparent sidereal time. Visualisation-accurate (no precession)."""
        t = ts.tt_jd(t_tt)
        lst_deg = ((t.gast + lon_deg / 15.0) % 24.0) * 15.0
        H = np.radians(lst_deg - np.asarray(ra_deg))
        dec = np.radians(np.asarray(dec_deg))
        phi = math.radians(lat_deg)
        sin_el = (math.sin(phi) * np.sin(dec)
                  + math.cos(phi) * np.cos(dec) * np.cos(H))
        el = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
        az = np.degrees(np.arctan2(
            -np.cos(dec) * np.sin(H),
            np.sin(dec) * math.cos(phi) - np.cos(dec) * np.cos(H) * math.sin(phi),
        )) % 360.0
        return az, el

    def compute_plot_objects(self, config_state, ts, t_tt, elevation_mask=0.0):
        """Drawable object list for the skyplot at the tracking-clock time:
        [{key, kind, name, az, el, color, radius, mag}] -- solar bodies always;
        Messier / NGC per the config toggles (NGC magnitude-limited). TTL-cached
        on (quantised time, toggles, observer) so both render threads share."""
        lat = float(config_state.lat_str)
        lon = float(config_state.lon_str)
        elev = float(config_state.alt_str)
        messier_on = bool(getattr(config_state, "messier_enabled", True))
        ngc_on = bool(getattr(config_state, "ngc_enabled", False))
        ngc_lim = float(getattr(config_state, "ngc_limiting_magnitude", 10.0))
        ckey = (round(t_tt * 86400.0 / PLOT_CACHE_QUANTUM_SEC), messier_on,
                ngc_on, round(ngc_lim, 2), round(lat, 6), round(lon, 6))
        with self._lock:
            if self._plot_cache and self._plot_cache[0] == ckey:
                return self._plot_cache[1]

            objs = []
            for key, name, az, el, dist_km, color, rad in self.solar_system_altaz(
                    lat, lon, elev, ts, t_tt):
                kind = "sun" if key == "sun" else ("moon" if key == "moon" else "planet")
                objs.append({"key": key, "kind": kind, "name": name, "az": az,
                             "el": el, "color": color, "radius": rad, "mag": 0.0,
                             "dist_km": float(dist_km)})

            def add_dso(cat, kind, mag_limit=None):
                if not cat["key"]:
                    return
                az, el = self.fixed_altaz(cat["ra_deg"], cat["dec_deg"],
                                          lat, lon, ts, t_tt)
                vis = el >= elevation_mask
                if mag_limit is not None:
                    vis &= cat["mag"] <= mag_limit
                for i in np.where(vis)[0]:
                    objs.append({"key": cat["key"][i], "kind": kind,
                                 "name": cat["name"][i], "az": float(az[i]),
                                 "el": float(el[i]), "color": None, "radius": 3,
                                 "mag": float(cat["mag"][i]),
                                 "ra_deg": float(cat["ra_deg"][i]),
                                 "dec_deg": float(cat["dec_deg"][i])})

            # Always-selectable named-star anchors (drawn by the starfield;
            # these entries exist for hit-testing and PROGRAM targeting).
            add_dso(self._named_star_arrays(), "star")

            if messier_on:
                add_dso(self.messier, "messier")
            if ngc_on:
                add_dso(self.ngc, "ngc", mag_limit=ngc_lim)

            self._plot_cache = (ckey, objs)
            return objs

    # ---- tracking trajectories ------------------------------------------------

    def build_trajectory(self, key, lat_deg, lon_deg, elev_m, ts, center_tt,
                         duration_min=TRAJ_DURATION_MIN, step_sec=TRAJ_STEP_SEC):
        """8-column tracking trajectory for any selection key, full Skyfield
        apparent positions (precession/nutation/aberration + topocentric
        parallax -- the moon needs the latter at the ~1 deg level).
        Returns (rows_list, times_array) or None for an unknown key."""
        if key in self._bodies:
            target = self._bodies[key][1]
        else:
            radec = self._fixed_radec(key)
            if radec is None:
                return None
            target = Star(ra_hours=radec[0] / 15.0, dec_degrees=radec[1])

        half = duration_min * 60.0 / 2.0
        n = max(3, int(round(duration_min * 60.0 / step_sec)) + 1)
        offsets = np.linspace(-half, half, n)
        times = ts.tt_jd(center_tt + offsets / 86400.0)
        observer = self._observer(lat_deg, lon_deg, elev_m)
        alt, az, dist = observer.at(times).observe(target).apparent().altaz()

        alt_deg = np.asarray(alt.degrees)
        az_deg = np.asarray(az.degrees)
        dist_km = np.asarray(dist.km)
        time_vals = np.asarray(times.tt)

        dt = step_sec
        az_rates = np.zeros(n)
        el_rates = np.zeros(n)
        for i in range(1, n - 1):
            az_rates[i] = _unwrap_az_diff(az_deg[i + 1] - az_deg[i - 1]) / (2 * dt)
            el_rates[i] = (alt_deg[i + 1] - alt_deg[i - 1]) / (2 * dt)
        az_rates[0] = _unwrap_az_diff(az_deg[1] - az_deg[0]) / dt
        az_rates[-1] = _unwrap_az_diff(az_deg[-1] - az_deg[-2]) / dt
        el_rates[0] = (alt_deg[1] - alt_deg[0]) / dt
        el_rates[-1] = (alt_deg[-1] - alt_deg[-2]) / dt

        zeros = np.zeros(n)
        rows = np.column_stack([time_vals, alt_deg, az_deg, dist_km,
                                zeros, zeros, az_rates, el_rates]).tolist()
        return rows, time_vals.copy()


KIND_LABELS = {"sun": "Sun", "moon": "Moon", "planet": "Planet",
               "star": "Bright star", "messier": "Messier object",
               "ngc": "NGC object"}


def selected_object_info_lines(tvs):
    """(label, value) lines describing the selected celestial object, for the
    details box the sky screens used to reserve for satellites. None when no
    celestial object is selected. Reads state.celestial_plot_objects (the
    render thread's per-frame lookup) so it costs nothing to build."""
    key = getattr(tvs, "selected_celestial", None)
    if not key:
        return None
    try:
        cat = get_celestial(getattr(tvs, "ephemeris", None))
        name = cat.display_name(key)
    except Exception:
        name = key
    lines = [("Target", name)]
    obj = (getattr(tvs, "celestial_plot_objects", {}) or {}).get(key)
    if obj is None:
        lines.append(("Status", "below horizon / hidden"))
        return lines
    lines.append(("Type", KIND_LABELS.get(obj["kind"], obj["kind"])))
    lines.append(("Azimuth", f"{obj['az']:.2f} deg"))
    lines.append(("Elevation", f"{obj['el']:.2f} deg"))
    if obj.get("ra_deg") is not None:
        ra_h = obj["ra_deg"] / 15.0
        lines.append(("RA", f"{int(ra_h):02d}h {(ra_h % 1) * 60:04.1f}m"))
        lines.append(("Dec", f"{obj['dec_deg']:+.2f} deg"))
    if obj["kind"] in ("star", "messier", "ngc"):
        lines.append(("V mag", f"{obj['mag']:.1f}"))
    if obj.get("dist_km"):
        d = obj["dist_km"]
        lines.append(("Distance", f"{d:,.0f} km" if d < 5.0e7
                      else f"{d / 1.496e8:.2f} AU"))
    # Sky rates from the live tracking trajectory when one is built.
    traj = (getattr(tvs, "celestial_trajectories", {}) or {}).get(key)
    cur_tt = getattr(tvs, "current_tt", None)
    if traj and cur_tt:
        try:
            from trajectory import interpolate_position_data_and_rates
            r = interpolate_position_data_and_rates(traj, cur_tt)
            if r[0] is not None:
                lines.append(("Az rate", f"{r[5] * 3600.0:+.1f} arcsec/s"))
                lines.append(("El rate", f"{r[6] * 3600.0:+.1f} arcsec/s"))
        except Exception:
            pass
    return lines


# ---- selected-target trajectory maintenance ----------------------------------

_TS = None


def _timescale():
    """Lazy module timescale (mirrors trajectory.live_tt's private one)."""
    global _TS
    if _TS is None:
        _TS = load.timescale()
    return _TS


def select_celestial(tvs, config_state, key, status_cb=None):
    """Select (or toggle off) a celestial target, with mutual exclusivity
    against satellite/aircraft selection, an immediate trajectory build when
    config is available (so PROGRAM tracking starts without waiting a control
    cycle), and a loud solar-safety warning."""
    if getattr(tvs, "selected_celestial", None) == key:
        tvs.selected_celestial = None
        print(f"Deselected celestial target {key}")
        return
    tvs.selected_celestial = key
    tvs.selected_satellite = None
    tvs.selected_aircraft = None
    try:
        name = get_celestial(getattr(tvs, "ephemeris", None)).display_name(key)
    except Exception:
        name = key
    print(f"Selected celestial target: {name} ({key})")
    if key == "sun":
        msg = "SUN selected - NEVER observe without a proper solar filter!"
        print(f"WARNING: {msg}")
        if status_cb:
            status_cb(msg)
    if config_state is not None:
        try:
            ensure_selected_trajectory(tvs, config_state)
        except Exception as e:
            print(f"Celestial trajectory build failed: {e}")


def ensure_selected_trajectory(tvs, config_state, ts=None):
    """Keep tvs.celestial_trajectories[selected] covering live time.

    Called at click-selection (so tracking can start immediately) and from both
    control loops each PROGRAM cycle (so a long session slides the window
    forward). Rebuilds are a few hundred vectorised Skyfield samples -- cheap
    enough to run inline on whichever thread notices the window is stale."""
    key = getattr(tvs, "selected_celestial", None)
    if not key or config_state is None:
        return
    if ts is None:
        ts = _timescale()
    trajs = getattr(tvs, "celestial_trajectories", None)
    if trajs is None:
        trajs = tvs.celestial_trajectories = {}
    now = live_tt()
    entry = trajs.get(key)
    margin_jd = TRAJ_REBUILD_MARGIN_MIN * 60.0 / 86400.0
    if entry is not None:
        times = entry[1]
        if times[0] <= now <= times[-1] - margin_jd:
            return
    cat = get_celestial(getattr(tvs, "ephemeris", None))
    center = now + (TRAJ_DURATION_MIN / 2.0 - TRAJ_REBUILD_MARGIN_MIN) * 60.0 / 86400.0
    traj = cat.build_trajectory(key, float(config_state.lat_str),
                                float(config_state.lon_str),
                                float(config_state.alt_str), ts, center)
    if traj is not None:
        # Atomic rebind; drop other keys so the dict can't grow unboundedly.
        tvs.celestial_trajectories = {key: traj}


# ---- module singleton ----------------------------------------------------------

_CELESTIAL = None
_CELESTIAL_LOCK = threading.Lock()


def get_celestial(ephemeris=None):
    """Lazily build and return the shared CelestialCatalog singleton."""
    global _CELESTIAL
    if _CELESTIAL is None:
        with _CELESTIAL_LOCK:
            if _CELESTIAL is None:
                _CELESTIAL = CelestialCatalog(ephemeris=ephemeris)
    return _CELESTIAL
