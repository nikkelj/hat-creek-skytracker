#!/usr/bin/env python
"""
Headless profiler for the launch-time trajectory pipeline.

Times each phase that runs before the tracking-vis window becomes useful:
  1. load_satellite_data            (TLE parse -> EarthSatellite objects)
  2. create_satellite_labels...     (per-sat labels + orbital metadata)
  3. visibility filter              (coarse SGP4 over all sats)
  4. precise trajectory pass        (coarse bulk pass for survivors; spacing scales with duration)

Run:  C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe tools\\profile_launch.py
Optionally pass a duration (minutes), default 30.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
pygame.init()


class _Info:
    current_w = 1600
    current_h = 1000


pygame.display.Info = lambda: _Info()


class Timer:
    def __init__(self):
        self.marks = []

    def __call__(self, label, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        dt = time.perf_counter() - t0
        self.marks.append((label, dt))
        print(f"  {label:<42} {dt*1000:9.1f} ms")
        return out

    def total(self):
        return sum(dt for _, dt in self.marks)


def main():
    duration_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    from skyfield.api import wgs84, load
    from display import DisplaySetup
    from config import load_config
    from tracking_visuals import TrackingVisState
    from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
    import trajectory

    msgs = []
    cb = msgs.append

    display = DisplaySetup()
    cfg = load_config()
    ts = load.timescale()
    tvs = TrackingVisState()

    lat, lon, alt = float(cfg.lat_str or 0), float(cfg.lon_str or 0), float(cfg.alt_str or 0)
    observer = wgs84.latlon(lat, lon, elevation_m=alt)
    now = datetime.now(timezone.utc)

    print(f"\n=== Launch trajectory profile (duration={duration_minutes} min) ===")
    T = Timer()

    T("load_satellite_data", load_satellite_data, tvs, cb)
    n_sats = len(tvs.satellites)
    print(f"  -> {n_sats} satellites loaded")

    T("create_satellite_labels_and_metadata", create_satellite_labels_and_metadata, tvs, cb)

    # Time the visibility filter on its own (the per-sat SGP4 loop).
    visible = T(
        "_filter_satellites_by_visibility",
        trajectory._filter_satellites_by_visibility,
        tvs.satellites, observer, ts, now, duration_minutes, 10.0, None,
    )
    print(f"  -> {len(visible)}/{n_sats} potentially visible "
          f"({100.0*len(visible)/max(1,n_sats):.1f}%)")

    # Full precompute (includes a second visibility filter pass internally, plus
    # the coarse bulk trajectory pass for survivors). Cache is cold here.
    trajectory.clear_trajectory_cache()
    T("precompute_trajectories (full, cold)", trajectory.precompute_trajectories,
      tvs, observer, ts, display, None, now, duration_minutes)
    print(f"  -> {len(tvs.satellite_trajectories)} trajectories computed")

    # Warm cache re-run, to show cache-hit cost.
    T("precompute_trajectories (full, warm)", trajectory.precompute_trajectories,
      tvs, observer, ts, display, None, now, duration_minutes)

    print(f"\n  {'TOTAL (cold launch path)':<42} "
          f"{(T.marks[0][1]+T.marks[1][1]+T.marks[3][1])*1000:9.1f} ms")
    print("  (load + labels + full cold precompute)\n")


if __name__ == "__main__":
    main()
