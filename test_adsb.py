"""
Tests for the ADS-B aircraft tracking subsystem (adsb_receiver.py).

Run with the `track` conda env python (see memory: dev-environment), e.g.:
    C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe test_adsb.py
or via pytest. pyModeS-dependent tests skip cleanly if pyModeS isn't installed.
"""

import math
import time
import types

import numpy as np

from adsb_receiver import (
    geodetic_to_ecef, ecef_to_enu, enu_to_azel_range, geodetic_to_azel_range,
    AdsbTracker, SimAdsbSource, decode_adsb_message,
)


# ------------------------------------------------------------------ helpers

def make_config(**overrides):
    """Minimal config object with the fields the tracker reads."""
    cfg = types.SimpleNamespace(
        lat_str="34.0", lon_str="-120.0", alt_str="100.0",
        adsb_source_mode="sim",
        adsb_fit_points=5,
        adsb_predict_horizon_sec=20.0,
        adsb_predict_step_sec=2.0,
        adsb_history_seconds=600.0,
        adsb_stale_timeout_sec=30.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_tvs():
    return types.SimpleNamespace(
        aircraft={}, aircraft_positions={}, aircraft_trajectories={},
        selected_aircraft=None, hovered_aircraft=None,
        selected_satellite=None, satellite_trajectories={},
    )


def _timescale():
    from skyfield.api import load
    return load.timescale()


def enu_to_geodetic(e, n, u, ref_lat, ref_lon, ref_alt):
    """Inverse of the module's geodetic->ENU path (test-only), via ECEF + Bowring."""
    lat = math.radians(ref_lat)
    lon = math.radians(ref_lon)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # ENU -> ECEF difference (transpose of the ecef_to_enu rotation).
    dx = -so * e - sl * co * n + cl * co * u
    dy = co * e - sl * so * n + cl * so * u
    dz = cl * n + sl * u
    ox, oy, oz = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    x, y, z = ox + dx, oy + dy, oz + dz
    # ECEF -> geodetic (Bowring).
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)
    b = a * (1 - f)
    ep2 = (a * a - b * b) / (b * b)
    p = math.hypot(x, y)
    th = math.atan2(a * z, b * p)
    lon2 = math.atan2(y, x)
    lat2 = math.atan2(z + ep2 * b * math.sin(th) ** 3,
                      p - e2 * a * math.cos(th) ** 3)
    n_rad = a / math.sqrt(1 - e2 * math.sin(lat2) ** 2)
    alt = p / math.cos(lat2) - n_rad
    return math.degrees(lat2), math.degrees(lon2), alt


# ------------------------------------------------------------------ tests

def test_coordinates_match_skyfield():
    """geodetic_to_azel_range must match skyfield's geometric topocentric altaz."""
    from skyfield.api import wgs84
    ts = _timescale()
    t = ts.now()
    obs = (34.0, -120.0, 100.0)
    targets = [
        (34.5, -120.4, 10000.0),
        (33.6, -119.7, 8000.0),
        (35.1, -120.0, 12000.0),
        (34.0, -121.2, 6000.0),
    ]
    for tgt in targets:
        az, el, rng = geodetic_to_azel_range(*tgt, *obs)
        obs_site = wgs84.latlon(obs[0], obs[1], elevation_m=obs[2])
        tgt_site = wgs84.latlon(tgt[0], tgt[1], elevation_m=tgt[2])
        alt_sf, az_sf, dist_sf = (tgt_site - obs_site).at(t).altaz()
        d_az = abs(((az - az_sf.degrees + 180.0) % 360.0) - 180.0)  # wrap-safe
        assert d_az < 0.05, (az, az_sf.degrees)
        assert abs(el - alt_sf.degrees) < 0.05, (el, alt_sf.degrees)
        assert abs(rng - dist_sf.km) < 0.5, (rng, dist_sf.km)
    print("test_coordinates_match_skyfield OK")


def test_enu_roundtrip():
    """enu_to_geodetic (test helper) inverts the module's geodetic->ENU exactly."""
    obs = (34.0, -120.0, 100.0)
    for e, n, u in [(10000, 20000, 5000), (-30000, 5000, 9000), (0, 0, 3000)]:
        lat, lon, alt = enu_to_geodetic(e, n, u, *obs)
        ox, oy, oz = geodetic_to_ecef(*obs)
        tx, ty, tz = geodetic_to_ecef(lat, lon, alt)
        e2, n2, u2 = ecef_to_enu(tx - ox, ty - oy, tz - oz, obs[0], obs[1])
        assert abs(e - e2) < 1e-3 and abs(n - n2) < 1e-3 and abs(u - u2) < 1e-3
    print("test_enu_roundtrip OK")


def test_linear_prediction_extrapolates_velocity():
    """A constant-ENU-velocity aircraft must yield a predicted trajectory whose
    future points continue that exact velocity (the core requirement)."""
    cfg = make_config(adsb_fit_points=5, adsb_predict_horizon_sec=20.0,
                      adsb_predict_step_sec=2.0)
    tvs = make_tvs()
    ts = _timescale()
    tracker = AdsbTracker(cfg, tvs, ts=ts)

    obs = (float(cfg.lat_str), float(cfg.lon_str), float(cfg.alt_str))
    # Start 40 km east, 30 km north, 10 km up; velocity (220, 60, 0) m/s ENU.
    p0 = np.array([40000.0, 30000.0, 10000.0])
    vel = np.array([220.0, 60.0, 0.0])
    dt = 1.0
    base_tt = ts.now().tt
    n_pts = 8
    for k in range(n_pts):
        enu = p0 + vel * (k * dt)
        lat, lon, alt = enu_to_geodetic(enu[0], enu[1], enu[2], *obs)
        tt = base_tt + (k * dt) / 86400.0
        tracker.ingest_position("ABC123", lat, lon, alt, tt=tt)

    last_tt = base_tt + ((n_pts - 1) * dt) / 86400.0
    tracker.update_positions(display=None, current_tt=last_tt)

    rows, times = tvs.aircraft_trajectories["ABC123"]
    # Reconstruct ENU from each future row (az/el/range) and check velocity.
    future = [(t, r) for t, r in zip(times, rows) if t > last_tt + 1e-9]
    assert len(future) >= 3, "expected several propagated points"

    def row_to_enu(r):
        el = math.radians(r[1]); az = math.radians(r[2]); rng = r[3] * 1000.0
        e = rng * math.cos(el) * math.sin(az)
        n = rng * math.cos(el) * math.cos(az)
        u = rng * math.sin(el)
        return np.array([e, n, u])

    (t_a, r_a), (t_b, r_b) = future[0], future[1]
    ea, eb = row_to_enu(r_a), row_to_enu(r_b)
    dt_s = (t_b - t_a) * 86400.0
    v_est = (eb - ea) / dt_s
    assert np.allclose(v_est, vel, atol=1.0), (v_est, vel)
    print("test_linear_prediction_extrapolates_velocity OK")


def test_update_positions_and_selection():
    """update_positions publishes aircraft_positions; active_program_trajectory
    returns the aircraft trajectory once it's selected."""
    from joystick_controller import active_program_trajectory
    cfg = make_config()
    tvs = make_tvs()
    ts = _timescale()
    tracker = AdsbTracker(cfg, tvs, ts=ts)

    obs = (float(cfg.lat_str), float(cfg.lon_str), float(cfg.alt_str))
    base_tt = ts.now().tt
    for k in range(6):
        # Move along east, well above the horizon.
        lat, lon, alt = enu_to_geodetic(20000 + 200 * k, 25000, 9000, *obs)
        tracker.ingest_position("AAA111", lat, lon, alt, tt=base_tt + k / 86400.0)

    tracker.update_positions(display=None, current_tt=base_tt + 5 / 86400.0)
    assert "AAA111" in tvs.aircraft_positions
    px, py, el, az, rng = tvs.aircraft_positions["AAA111"]
    assert el > 0 and 0 <= az < 360 and rng > 0

    # Not selected -> no program target.
    traj, kind, key = active_program_trajectory(tvs)
    assert traj is None
    # Select the aircraft -> it becomes the program target.
    tvs.selected_aircraft = "AAA111"
    traj, kind, key = active_program_trajectory(tvs)
    assert kind == "aircraft" and key == "AAA111" and traj is not None
    # A selected satellite outranks an aircraft.
    tvs.selected_satellite = "SAT"
    tvs.satellite_trajectories = {"SAT": traj}
    traj2, kind2, key2 = active_program_trajectory(tvs)
    assert kind2 == "satellite"
    print("test_update_positions_and_selection OK")


def test_stale_pruning():
    cfg = make_config(adsb_stale_timeout_sec=0.0001)
    tvs = make_tvs()
    ts = _timescale()
    tracker = AdsbTracker(cfg, tvs, ts=ts)
    obs = (float(cfg.lat_str), float(cfg.lon_str), float(cfg.alt_str))
    base_tt = ts.now().tt
    for k in range(3):
        lat, lon, alt = enu_to_geodetic(15000, 15000, 8000, *obs)
        tracker.ingest_position("OLD999", lat, lon, alt, tt=base_tt + k / 86400.0)
    time.sleep(0.01)  # exceed the tiny stale timeout
    tracker.update_positions(display=None, current_tt=base_tt)
    assert "OLD999" not in tracker.aircraft
    assert "OLD999" not in tvs.aircraft_positions
    print("test_stale_pruning OK")


def test_sim_source_produces_aircraft():
    cfg = make_config(adsb_source_mode="sim")
    tvs = make_tvs()
    ts = _timescale()
    tracker = AdsbTracker(cfg, tvs, ts=ts)
    src = SimAdsbSource(tracker, cfg, n_aircraft=2)
    src.start()
    try:
        time.sleep(1.3)  # let it emit at least one round of fixes
    finally:
        src.stop()
    assert len(tracker.aircraft) >= 2
    for ac in tracker.aircraft.values():
        assert len(ac.history) >= 1
    print("test_sim_source_produces_aircraft OK")


def test_decode_adsb_message():
    try:
        import pyModeS  # noqa: F401
    except Exception:
        print("test_decode_adsb_message SKIPPED (pyModeS not installed)")
        return
    # pyModeS reference example: an airborne position squitter decoded against a
    # nearby reference location.
    ref_lat, ref_lon = 52.258, 3.918
    rep = decode_adsb_message("8D40621D58C382D690C8AC2863A7", ref_lat, ref_lon)
    assert rep is not None and rep["kind"] == "position"
    assert abs(rep["lat"] - 52.2572) < 0.01
    assert abs(rep["lon"] - 3.9193) < 0.01
    # Identification message yields a callsign.
    ident = decode_adsb_message("8D4840D6202CC371C32CE0576098", ref_lat, ref_lon)
    assert ident is not None and ident["kind"] == "ident"
    assert ident["callsign"]
    print("test_decode_adsb_message OK")


def test_rtlsdr_process_buffer_resilient():
    """pyModeS' demodulator raises 'max() arg is an empty sequence' when a preamble
    lands at the very end of the sample buffer. RtlSdrAdsbSource's _Reader override
    must swallow that, clear the buffer, and return [] instead of killing the
    receiver thread."""
    try:
        import pyModeS  # noqa: F401
        from pyModeS.extra.rtlreader import RtlReader  # noqa: F401
    except Exception:
        print("test_rtlsdr_process_buffer_resilient SKIPPED (pyModeS not installed)")
        return
    import threading
    from adsb_receiver import RtlSdrAdsbSource

    src = RtlSdrAdsbSource.__new__(RtlSdrAdsbSource)
    src.tracker = AdsbTracker(make_config(), make_tvs(), ts=_timescale())
    src._stop = threading.Event()
    reader_cls = src._make_reader_class()

    # Build a reader instance WITHOUT opening a device (bypass RtlReader.__init__).
    r = reader_cls.__new__(reader_cls)
    r.debug = False
    r.noise_floor = 1e6
    preamble = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    # 200 low-noise samples (so _calc_noise works) then a preamble flush at the end
    # -> frame_start == len(buffer) -> empty frame slice -> max() on empty.
    r.signal_buffer = [0.01] * 200 + preamble

    # The unguarded parent must raise; our override must not.
    raised = False
    try:
        RtlReader._process_buffer(r)
    except ValueError:
        raised = True
    assert raised, "expected the pyModeS empty-sequence crash to reproduce"

    r.signal_buffer = [0.01] * 200 + preamble
    out = r._process_buffer()  # our override
    assert out == [] and r.signal_buffer == []
    print("test_rtlsdr_process_buffer_resilient OK")


if __name__ == "__main__":
    test_coordinates_match_skyfield()
    test_enu_roundtrip()
    test_linear_prediction_extrapolates_velocity()
    test_update_positions_and_selection()
    test_stale_pruning()
    test_sim_source_produces_aircraft()
    test_decode_adsb_message()
    test_rtlsdr_process_buffer_resilient()
    print("\nAll ADS-B tests passed.")
