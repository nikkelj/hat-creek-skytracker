"""
ADS-B aircraft tracking subsystem (Nooelec / RTL-SDR).

This mirrors the satellite pipeline so aircraft drop into the SAME UI / selection
/ slew / track machinery:

  * `AdsbTracker` keeps a dict of `Aircraft` keyed by 24-bit ICAO address, ingests
    decoded position/velocity/identification reports, prunes stale aircraft, and
    publishes per-frame `aircraft_positions` plus predicted `aircraft_trajectories`
    onto a `TrackingVisState` -- exactly the structures the renderers and the
    PROGRAM-track control loops already consume for satellites.

  * Trajectories are built by fitting a simple linear motion model (constant
    velocity) to the last N observed fixes in a local ENU (east/north/up) frame
    and propagating it forward `horizon` seconds. N is `config.adsb_fit_points`
    (UI slider). The result is emitted in the canonical 8-column trajectory row
    format `[tt, el_deg, az_deg, range_km, px, py, az_rate_dps, el_rate_dps]` so
    `trajectory.interpolate_position_data_and_rates` works unchanged.

  * Message sourcing is pluggable behind `AdsbSource`:
      - `RtlSdrAdsbSource`   native sampling from the SDR via pyModeS' RtlReader
                             (the real hardware path; deps imported lazily).
      - `Dump1090Source`     connect to a running dump1090 SBS/BaseStation feed.
      - `SimAdsbSource`      synthetic aircraft for testing without hardware.

Geometry note: az/el/range are computed from an EXACT WGS84 geodetic->ECEF->ENU
conversion (no flat-earth approximation -- earth curvature matters at the >100 km
ranges aircraft are seen), validated against skyfield in test_adsb.py. Azimuth is
0=North increasing East and elevation is degrees above the horizon, matching the
satellite/star convention used everywhere else.
"""

import math
import os
import socket
import threading
import time
from collections import deque

import numpy as np

# WGS84 ellipsoid constants.
_WGS84_A = 6378137.0                      # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563           # flattening
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)  # first eccentricity squared


# ==============================================================================
# COORDINATE HELPERS (geodetic -> ENU -> az/el/range, exact WGS84)
# ==============================================================================

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """WGS84 geodetic latitude/longitude/height -> ECEF (X, Y, Z) metres."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(dx, dy, dz, ref_lat_deg, ref_lon_deg):
    """Rotate an ECEF difference vector into the local East/North/Up frame at the
    reference geodetic latitude/longitude."""
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def enu_to_azel_range(e, n, u):
    """Local ENU vector -> (azimuth deg [0,360), elevation deg, range km)."""
    horiz = math.hypot(e, n)
    rng = math.sqrt(e * e + n * n + u * u)
    az = math.degrees(math.atan2(e, n)) % 360.0
    el = math.degrees(math.atan2(u, horiz)) if horiz > 0 or u != 0 else 0.0
    return az, el, rng / 1000.0


def geodetic_to_azel_range(lat_deg, lon_deg, alt_m, obs_lat, obs_lon, obs_alt_m):
    """Observer-relative topocentric az/el/range for a geodetic target point."""
    tx, ty, tz = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    ox, oy, oz = geodetic_to_ecef(obs_lat, obs_lon, obs_alt_m)
    e, n, u = ecef_to_enu(tx - ox, ty - oy, tz - oz, obs_lat, obs_lon)
    return enu_to_azel_range(e, n, u)


def _polar_xy(az_deg, el_deg, cx, cy, radius):
    """Map az/el to skyplot pixel coords, matching trajectory._compute_one_trajectory."""
    az_rad = math.radians(az_deg % 360.0)
    polar_radius = (90.0 - el_deg) / 90.0 * radius
    return cx + polar_radius * math.sin(az_rad), cy - polar_radius * math.cos(az_rad)


# ==============================================================================
# AIRCRAFT MODEL
# ==============================================================================

class Aircraft:
    """A single tracked aircraft, keyed by its 24-bit ICAO address (hex str)."""

    def __init__(self, icao):
        self.icao = icao
        self.callsign = ""
        # Rolling history of observed fixes: dicts {tt, lat, lon, alt_m}.
        self.history = deque()
        self.last_seen = 0.0          # perf_counter time of last update (for
                                      # pruning; monotonic so an NTP wall-clock
                                      # step can't mass-prune, and high-res --
                                      # Windows time.monotonic() has a 15.6 ms
                                      # quantum on CPython<=3.12)
        self.last_fix_tt = None       # TT of last positional fix
        # Latest decoded scalar fields (for the info panel).
        self.lat = None
        self.lon = None
        self.alt_m = None
        self.speed_kt = None
        self.track_deg = None
        self.vert_rate = None
        # True when a new fix arrived and the predicted trajectory needs rebuilding.
        self.dirty = False

    @property
    def label(self):
        return self.callsign.strip() or self.icao.upper()


# ==============================================================================
# TRACKER
# ==============================================================================

class AdsbTracker:
    """Owns the aircraft set and publishes positions/trajectories onto a
    TrackingVisState, mirroring how satellite_data + trajectory populate it.

    `ingest_*` methods are called from the source thread; `update_positions` runs
    on the main loop each frame. We snapshot dicts before iterating so the two
    threads never trip over a concurrent insert (benign-race model used elsewhere
    in this app)."""

    def __init__(self, config_state, tracking_vis_state, ts=None):
        self.config_state = config_state
        self.tvs = tracking_vis_state
        if ts is None:
            from skyfield.api import load
            ts = load.timescale()
        self.ts = ts
        self.aircraft = {}            # icao -> Aircraft
        self._geom = None             # (cx, cy, radius) last used for px/py baking

    # ---- ingest (source thread) -------------------------------------------
    def _now_tt(self):
        return self.ts.now().tt

    def _get(self, icao):
        ac = self.aircraft.get(icao)
        if ac is None:
            ac = Aircraft(icao)
            self.aircraft[icao] = ac
        return ac

    def ingest_position(self, icao, lat, lon, alt_m, tt=None):
        """Record a positional fix. alt_m may be None (then last known is reused).
        `tt` (Terrestrial Time) defaults to now; tests inject it for determinism."""
        if lat is None or lon is None:
            return
        ac = self._get(icao)
        if alt_m is None:
            alt_m = ac.alt_m if ac.alt_m is not None else 0.0
        if tt is None:
            tt = self._now_tt()
        ac.lat, ac.lon, ac.alt_m = lat, lon, alt_m
        ac.history.append({"tt": tt, "lat": lat, "lon": lon, "alt_m": alt_m})
        ac.last_fix_tt = tt
        ac.last_seen = time.perf_counter()
        ac.dirty = True
        self._trim_history(ac)

    def ingest_velocity(self, icao, speed_kt=None, track_deg=None, vert_rate=None):
        ac = self._get(icao)
        if speed_kt is not None:
            ac.speed_kt = speed_kt
        if track_deg is not None:
            ac.track_deg = track_deg
        if vert_rate is not None:
            ac.vert_rate = vert_rate
        ac.last_seen = time.perf_counter()

    def ingest_identification(self, icao, callsign):
        ac = self._get(icao)
        if callsign:
            ac.callsign = callsign
        ac.last_seen = time.perf_counter()

    def _trim_history(self, ac):
        keep = self._history_seconds()
        if keep <= 0 or ac.last_fix_tt is None:
            return
        cutoff = ac.last_fix_tt - keep / 86400.0
        while len(ac.history) > 2 and ac.history[0]["tt"] < cutoff:
            ac.history.popleft()

    # ---- config accessors --------------------------------------------------
    def _cfg(self, name, default):
        try:
            return getattr(self.config_state, name)
        except AttributeError:
            return default

    def _fit_points(self):
        return max(2, int(self._cfg("adsb_fit_points", 5)))

    def _history_seconds(self):
        return float(self._cfg("adsb_history_seconds", 120.0))

    def _horizon(self):
        return float(self._cfg("adsb_predict_horizon_sec", 60.0))

    def _step(self):
        return max(0.5, float(self._cfg("adsb_predict_step_sec", 2.0)))

    def _stale_timeout(self):
        return float(self._cfg("adsb_stale_timeout_sec", 30.0))

    def _observer(self):
        cfg = self.config_state
        try:
            return (float(cfg.lat_str), float(cfg.lon_str), float(cfg.alt_str))
        except (AttributeError, TypeError, ValueError):
            return (0.0, 0.0, 0.0)

    # ---- per-frame update (main thread) ------------------------------------
    def _geom_from_display(self, display):
        if display is None:
            return (400, 300, 300)
        cx = display.sub_x + display.sub_width // 2
        cy = display.sub_y + display.sub_height // 2
        radius = min(display.sub_width, display.sub_height) // 2 - 50
        return (cx, cy, radius)

    def update_positions(self, display, current_tt):
        """Prune stale aircraft, (re)build dirty trajectories, and republish the
        per-frame position dict. Writes `aircraft`, `aircraft_trajectories` and
        `aircraft_positions` onto the TrackingVisState."""
        if self.tvs is None:
            return
        geom = self._geom_from_display(display)
        if geom != self._geom:
            self._geom = geom
            for ac in list(self.aircraft.values()):
                ac.dirty = True

        obs = self._observer()
        now_mono = time.perf_counter()
        timeout = self._stale_timeout()

        trajectories = dict(getattr(self.tvs, "aircraft_trajectories", {}) or {})
        positions = {}

        for icao, ac in list(self.aircraft.items()):
            if timeout > 0 and (now_mono - ac.last_seen) > timeout:
                self.aircraft.pop(icao, None)
                trajectories.pop(icao, None)
                if getattr(self.tvs, "selected_aircraft", None) == icao:
                    self.tvs.selected_aircraft = None
                continue

            if ac.dirty or icao not in trajectories:
                traj = self._build_trajectory(ac, obs, geom)
                if traj is not None:
                    trajectories[icao] = traj
                ac.dirty = False

            traj = trajectories.get(icao)
            if not traj:
                continue
            from trajectory import interpolate_position_data_and_rates
            px, py, el, rng, az, az_rate, el_rate = interpolate_position_data_and_rates(
                traj, current_tt)
            if px is None or el is None:
                continue
            positions[icao] = (px, py, el, az, rng)

        self.tvs.aircraft = self.aircraft
        self.tvs.aircraft_trajectories = trajectories
        self.tvs.aircraft_positions = positions

    def _build_trajectory(self, ac, obs, geom):
        """Fit a constant-velocity model to the last N fixes (ENU) and propagate
        it forward, returning (rows, times_array) in the 8-col trajectory format.
        Observed fixes form the past (grey) tail; propagated points the future."""
        hist = list(ac.history)
        if not hist:
            return None
        obs_lat, obs_lon, obs_alt = obs
        cx, cy, radius = geom

        # ENU for every observed fix.
        ox, oy, oz = geodetic_to_ecef(obs_lat, obs_lon, obs_alt)
        enu = []
        for h in hist:
            tx, ty, tz = geodetic_to_ecef(h["lat"], h["lon"], h["alt_m"])
            e, n, u = ecef_to_enu(tx - ox, ty - oy, tz - oz, obs_lat, obs_lon)
            enu.append((h["tt"], e, n, u))

        t_last = enu[-1][0]
        rows = []  # (tt, el, az, range_km)

        # Past tail: the observed fixes themselves.
        for tt, e, n, u in enu:
            az, el, rng = enu_to_azel_range(e, n, u)
            rows.append((tt, el, az, rng))

        # Linear fit over the last N points -> velocity, then propagate forward.
        n_fit = min(self._fit_points(), len(enu))
        if n_fit >= 2:
            fit = enu[-n_fit:]
            t0 = fit[-1][0]
            ts_s = np.array([(p[0] - t0) * 86400.0 for p in fit])
            es = np.array([p[1] for p in fit])
            ns = np.array([p[2] for p in fit])
            us = np.array([p[3] for p in fit])
            ce = np.polyfit(ts_s, es, 1)   # [slope, intercept]
            cn = np.polyfit(ts_s, ns, 1)
            cu = np.polyfit(ts_s, us, 1)
            horizon = self._horizon()
            step = self._step()
            t = step
            while t <= horizon + 1e-6:
                e = ce[0] * t + ce[1]
                n = cn[0] * t + cn[1]
                u = cu[0] * t + cu[1]
                az, el, rng = enu_to_azel_range(e, n, u)
                rows.append((t_last + t / 86400.0, el, az, rng))
                t += step

        rows.sort(key=lambda r: r[0])

        # Assemble the 8-col rows with px/py + finite-difference sky rates.
        out = []
        times = []
        nrows = len(rows)
        for i, (tt, el, az, rng) in enumerate(rows):
            px, py = _polar_xy(az, el, cx, cy, radius)
            if i < nrows - 1:
                tt1, el1, az1, _ = rows[i + 1]
                dt = (tt1 - tt) * 86400.0
                if dt > 1e-6:
                    d_az = ((az1 - az + 180.0) % 360.0) - 180.0
                    az_rate = d_az / dt
                    el_rate = (el1 - el) / dt
                else:
                    az_rate = el_rate = 0.0
            elif out:
                az_rate, el_rate = out[-1][6], out[-1][7]
            else:
                az_rate = el_rate = 0.0
            out.append([tt, el, az, rng, px, py, az_rate, el_rate])
            times.append(tt)

        return (out, np.array(times))


# ==============================================================================
# MESSAGE DECODE (pyModeS)
# ==============================================================================

_RUST_ADSB_FLAG = False


def configure_rust_adsb(config_state):
    """Latch the config flag (called at startup from main.py)."""
    global _RUST_ADSB_FLAG
    _RUST_ADSB_FLAG = bool(getattr(config_state, "use_rust_adsb", False))


def rust_adsb_enabled():
    env = os.environ.get("SKYTRACKER_RUST_ADSB")
    if env == "1":
        return True
    if env == "0":
        return False
    return _RUST_ADSB_FLAG



def decode_adsb_message(msg, ref_lat, ref_lon):
    """Decode one raw Mode-S hex message into an update dict, or None if it is not
    a usable ADS-B (DF17/18) extended-squitter. Uses pyModeS' single-message
    `position_with_ref` (valid within ~180 NM of the reference -- always true for
    a ground receiver), so no even/odd CPR frame pairing is needed.

    Returns dicts like {'icao', 'kind': 'position'|'velocity'|'ident', ...}."""
    if msg is None or len(msg) != 28:
        return None

    # Rust fast path (Phase 5 of the Rust port): same decode, pyModeS
    # fallback. Enabled by env SKYTRACKER_RUST_ADSB=1 or config
    # use_rust_adsb (latched via rust_adsb_enabled below).
    if rust_adsb_enabled():
        try:
            import skytracker_core as _sc
            return _sc.adsb_decode_message(msg, ref_lat, ref_lon)
        except Exception:
            pass  # fall through to pyModeS

    import pyModeS as pms
    try:
        df = pms.df(msg)
        if df not in (17, 18):
            return None
        if pms.crc(msg) != 0:
            return None
        icao = pms.adsb.icao(msg)
        tc = pms.adsb.typecode(msg)
        if icao is None or tc is None:
            return None
        if 1 <= tc <= 4:
            cs = pms.adsb.callsign(msg)
            return {"icao": icao, "kind": "ident",
                    "callsign": cs.replace("_", "").strip() if cs else ""}
        if (9 <= tc <= 18) or (20 <= tc <= 22):
            alt_ft = pms.adsb.altitude(msg)
            pos = pms.adsb.position_with_ref(msg, ref_lat, ref_lon)
            if pos is None:
                return None
            lat, lon = pos
            alt_m = alt_ft * 0.3048 if alt_ft is not None else None
            return {"icao": icao, "kind": "position",
                    "lat": lat, "lon": lon, "alt_m": alt_m}
        if tc == 19:
            v = pms.adsb.velocity(msg)
            if v is None:
                return None
            spd, trk, vs, _ = v
            return {"icao": icao, "kind": "velocity",
                    "speed_kt": spd, "track_deg": trk, "vert_rate": vs}
    except Exception:
        return None
    return None


# ==============================================================================
# SOURCES
# ==============================================================================

class AdsbSource:
    """Base class for a message source. Subclasses run a background thread and
    push decoded reports into the tracker via `_emit`."""

    def __init__(self, tracker, config_state, update_status=None):
        self.tracker = tracker
        self.config_state = config_state
        self.update_status = update_status
        self._thread = None
        self._stop = threading.Event()
        self.failed = threading.Event()   # set when the source dies (deps/device)
        self.status = ""

    def preflight(self):
        """Synchronously verify prerequisites before the thread starts. Raise on
        failure so connect() can report it and stay disconnected. Default: no-op."""
        return

    def _set_status(self, msg):
        self.status = msg
        if self.update_status:
            try:
                self.update_status(msg)
            except Exception:
                pass

    def _fail(self, msg):
        """Record a terminal failure (the receiver flips to disconnected)."""
        self._set_status(msg)
        self.failed.set()

    def _emit(self, report):
        """Route a decoded report dict into the tracker."""
        if not report:
            return
        icao = report.get("icao")
        if not icao:
            return
        kind = report.get("kind")
        if kind == "position":
            self.tracker.ingest_position(icao, report.get("lat"), report.get("lon"),
                                         report.get("alt_m"))
        elif kind == "velocity":
            self.tracker.ingest_velocity(icao, report.get("speed_kt"),
                                         report.get("track_deg"), report.get("vert_rate"))
        elif kind == "ident":
            self.tracker.ingest_identification(icao, report.get("callsign"))

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__,
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        raise NotImplementedError


class SimAdsbSource(AdsbSource):
    """Synthetic aircraft for hardware-free testing/demo. Flies a handful of
    aircraft in straight lines at constant velocity near the observer, emitting
    ~1 Hz position fixes (with callsigns), so the whole pipeline can be exercised
    without an SDR."""

    def __init__(self, tracker, config_state, update_status=None, n_aircraft=3):
        super().__init__(tracker, config_state, update_status)
        self.n_aircraft = n_aircraft
        self._planes = []

    def _init_planes(self):
        try:
            lat0 = float(self.config_state.lat_str)
            lon0 = float(self.config_state.lon_str)
        except (AttributeError, TypeError, ValueError):
            lat0, lon0 = 34.874, -120.446
        self._planes = []
        for i in range(self.n_aircraft):
            ang = math.radians(120.0 * i)
            self._planes.append({
                "icao": f"SIM{i:03d}",
                "callsign": f"SIM{i:03d}",
                # Start ~30-60 km out on different bearings.
                "lat": lat0 + 0.35 * math.cos(ang) + 0.05 * i,
                "lon": lon0 + 0.45 * math.sin(ang),
                "alt_m": 9000.0 + 800.0 * i,
                # Velocity in ENU m/s (heading varies per plane), ~220 m/s.
                "ve": 220.0 * math.sin(math.radians(40.0 * i + 20.0)),
                "vn": 220.0 * math.cos(math.radians(40.0 * i + 20.0)),
                "vu": (i - 1) * 3.0,
            })

    def _run(self):
        self._init_planes()
        self._set_status("sim: generating aircraft")
        last = time.time()
        for p in self._planes:
            self.tracker.ingest_identification(p["icao"], p["callsign"])
        while not self._stop.is_set():
            now = time.time()
            dt = now - last
            last = now
            for p in self._planes:
                # Advance position (flat-earth deltas are fine for the sim path).
                dlat = (p["vn"] * dt) / 111320.0
                dlon = (p["ve"] * dt) / (111320.0 * math.cos(math.radians(p["lat"])))
                p["lat"] += dlat
                p["lon"] += dlon
                p["alt_m"] += p["vu"] * dt
                self.tracker.ingest_position(p["icao"], p["lat"], p["lon"], p["alt_m"])
            self._stop.wait(1.0)


class Dump1090Source(AdsbSource):
    """Connect to a running dump1090 SBS/BaseStation TCP feed (default port 30003)
    and parse MSG rows. Useful when dump1090 already owns the SDR."""

    def _run(self):
        host = self.tracker._cfg("adsb_dump1090_host", "127.0.0.1")
        port = int(self.tracker._cfg("adsb_dump1090_port", 30003))
        while not self._stop.is_set():
            try:
                self._set_status(f"dump1090: connecting {host}:{port}")
                sock = socket.create_connection((host, port), timeout=5.0)
                sock.settimeout(1.0)
                self._set_status(f"dump1090: connected {host}:{port}")
                buf = ""
                while not self._stop.is_set():
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data.decode("ascii", "ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        self._parse_sbs(line.strip())
                sock.close()
            except Exception as e:
                self._set_status(f"dump1090: {e}")
                self._stop.wait(3.0)

    def _parse_sbs(self, line):
        # SBS-1 CSV: MSG,type,...,HexIdent(4),...,Callsign(10),Altitude(11),...,
        #            Lat(14),Lon(15),...
        if not line.startswith("MSG"):
            return
        f = line.split(",")
        if len(f) < 16:
            return
        icao = f[4].strip().lower()
        if not icao:
            return
        cs = f[10].strip()
        if cs:
            self.tracker.ingest_identification(icao, cs)
        try:
            if f[14] and f[15]:
                lat = float(f[14])
                lon = float(f[15])
                alt_ft = float(f[11]) if f[11] else None
                alt_m = alt_ft * 0.3048 if alt_ft is not None else None
                self.tracker.ingest_position(icao, lat, lon, alt_m)
        except ValueError:
            pass


class RtlSdrAdsbSource(AdsbSource):
    """Native ADS-B sampling from the RTL-SDR via pyModeS' RtlReader (does the
    1090 MHz IQ read + Mode-S preamble detection/demodulation), decoding each
    message locally. This is the real-hardware path for the Nooelec dongle.

    Dependencies (`pyModeS`, `pyrtlsdr`/librtlsdr) are imported lazily so the app
    still runs without them; connect() surfaces a clear error if they're missing."""

    def preflight(self):
        # Verify the SDR libs are importable before we declare the receiver
        # connected, so a missing-deps click reports cleanly instead of flashing
        # "connected" then failing on the worker thread.
        try:
            import pyModeS  # noqa: F401
            from pyModeS.extra.rtlreader import RtlReader  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                f"pyModeS not installed ({e}). Run: pip install 'pyModeS<3'") from e
        # pyModeS' RtlReader imports rtlsdr lazily (only warns if missing), so
        # check it here -- this is what fails when librtlsdr.dll isn't on PATH.
        try:
            import rtlsdr  # noqa: F401  (pyrtlsdr; loads librtlsdr.dll at import)
        except Exception as e:
            raise RuntimeError(
                f"RTL-SDR library not loadable ({e}). Install pyrtlsdr and put "
                f"librtlsdr.dll (+ its deps) on PATH.") from e

    def _make_reader_class(self):
        from pyModeS.extra.rtlreader import RtlReader
        tracker = self.tracker
        source = self

        class _Reader(RtlReader):
            def handle_messages(self, messages):
                if source._stop.is_set():
                    # RtlReader.run() loops forever; closing the device makes the
                    # next read raise, which we catch below to exit the thread.
                    try:
                        self.stop()
                    except Exception:
                        pass
                    return
                lat, lon, _ = tracker._observer()
                for item in messages:
                    msg = item[0] if isinstance(item, (list, tuple)) else item
                    rep = decode_adsb_message(msg, lat, lon)
                    if rep:
                        source._emit(rep)

            def _process_buffer(self):
                # pyModeS' demodulator crashes ("max() arg is an empty sequence")
                # when a preamble is detected at the very end of the sample buffer
                # (the frame slice comes back empty). Don't let one malformed buffer
                # kill the whole receiver -- drop it and keep going.
                try:
                    return super()._process_buffer()
                except (ValueError, IndexError):
                    self.signal_buffer = []
                    return []

        return _Reader

    def _run(self):
        try:
            reader_cls = self._make_reader_class()
        except Exception as e:
            self._fail(f"rtlsdr: pyModeS/pyrtlsdr unavailable ({e})")
            return

        # Run the reader, restarting it on transient device errors so a hiccup
        # doesn't permanently kill the receiver. Bail out (fail) only after several
        # consecutive failures (e.g. the dongle was unplugged).
        consecutive_failures = 0
        while not self._stop.is_set():
            try:
                self.reader = reader_cls()
            except Exception as e:
                consecutive_failures += 1
                hint = ""
                if "access" in str(e).lower() or "permission" in str(e).lower():
                    hint = " - device may be in use by another program"
                if consecutive_failures >= 3 or self._stop.is_set():
                    self._fail(f"rtlsdr: open failed ({e}){hint}")
                    return
                self._set_status(f"rtlsdr: open retry ({e}){hint}")
                self._stop.wait(1.5)
                continue

            # Apply configured gain (pyModeS stores the device as `.sdr`).
            gain = self.tracker._cfg("adsb_gain", "auto")
            try:
                self.reader.sdr.gain = gain
            except Exception:
                pass

            self._set_status("rtlsdr: receiving 1090 MHz")
            consecutive_failures = 0
            try:
                self.reader.run()
                # run() only returns when the device was closed (our stop()).
                return
            except Exception as e:
                if self._stop.is_set():
                    return
                consecutive_failures += 1
                try:
                    self.reader.stop()   # close the device before reopening
                except Exception:
                    pass
                if consecutive_failures >= 3:
                    self._fail(f"rtlsdr: stopped ({e})")
                    return
                self._set_status(f"rtlsdr: restarting ({e})")
                self._stop.wait(1.0)

    def stop(self):
        super().stop()
        reader = getattr(self, "reader", None)
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass


_SOURCES = {
    "rtlsdr": RtlSdrAdsbSource,
    "dump1090": Dump1090Source,
    "sim": SimAdsbSource,
}


# ==============================================================================
# RECEIVER (held by JoystickModeState)
# ==============================================================================

class AdsbReceiver:
    """Front-end object the UI talks to: owns the tracker + the active source and
    exposes a telescope-style connect()/disconnect()/connected/status surface."""

    def __init__(self, config_state, tracking_vis_state, ts=None, update_status=None):
        self.config_state = config_state
        self.update_status = update_status
        self.tracker = AdsbTracker(config_state, tracking_vis_state, ts=ts)
        self.source = None
        self.connected = False
        self.status = "disconnected"

    def _set_status(self, msg):
        self.status = msg
        if self.update_status:
            try:
                self.update_status(f"ADS-B: {msg}")
            except Exception:
                pass

    def connect(self):
        if self.connected:
            return True
        mode = str(getattr(self.config_state, "adsb_source_mode", "rtlsdr")).lower()
        cls = _SOURCES.get(mode, RtlSdrAdsbSource)
        try:
            self.source = cls(self.tracker, self.config_state,
                              update_status=self._set_status)
            # Verify prerequisites BEFORE declaring connected (e.g. SDR libs), so a
            # missing dependency reports cleanly instead of a false "connected".
            self.source.preflight()
            self.source.start()
            self.connected = True
            self._set_status(f"{mode}: connected")
            return True
        except Exception as e:
            self.source = None
            self.connected = False
            self._set_status(f"{mode}: {e}")
            return False

    def disconnect(self):
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
        self.source = None
        self.connected = False
        self._set_status("disconnected")

    def update(self, display, current_tt):
        """Per-frame refresh of aircraft positions/trajectories. Cheap when no
        aircraft are present; safe to call every frame."""
        # A source that died on its worker thread (e.g. device unplugged) flips us
        # back to disconnected so the UI stops showing a false "connected".
        if self.source is not None and self.source.failed.is_set():
            self.connected = False
            self.status = self.source.status
            self.source = None
            return
        if current_tt is None:
            return
        try:
            self.tracker.update_positions(display, current_tt)
        except Exception as e:
            print(f"ADS-B update error: {e}")
