#!/usr/bin/env python
"""
Benchmark the Rust core (skytracker_core) against the Python baseline modules.

Reports ns/op and the speedup ratio for the ported hot paths, plus a full
PROGRAM control-cycle comparison. Honest about where the win is:

  * Heavy ops (hotspot detection) and scipy-backed transforms: large Rust wins.
  * Trivial scalar ops called one-at-a-time FROM Python: the win is small or
    even negative because the Python->Rust FFI call overhead (~1 us) dominates
    the actual compute. This is expected and is exactly why the control LOOP is
    the real prize: it runs entirely inside a Rust thread with NO per-cycle FFI
    and NO GIL, so the per-call overhead disappears.

Build first, then run:
    cd rust/skytracker-ffi && maturin develop --release
    python bench_rust_vs_python.py
"""

import math
import time

import numpy as np

import skytracker_core as rc
import transformations as tf
from control import PIDController
from hotspot import detect_hotspot as py_detect


def bench(fn, n, warmup=None):
    """Return seconds-per-call for `fn` over `n` iterations."""
    if warmup is None:
        warmup = max(1, n // 20)
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def row(name, py_s, rs_s):
    ratio = py_s / rs_s if rs_s > 0 else float("inf")
    print(f"{name:<34} {py_s*1e9:>12.0f} {rs_s*1e9:>12.0f} {ratio:>8.1f}x")


def gaussian_frame(w, h, cx, cy, amp=200.0, sigma=3.0, bg=10.0, noise=1.0, seed=1):
    yy, xx = np.mgrid[0:h, 0:w]
    img = bg + amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    img = img + np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return np.clip(img, 0, None).astype(np.float32)


def main():
    print(f"{'operation':<34} {'python ns':>12} {'rust ns':>12} {'speedup':>8}")
    print("-" * 70)

    # 1) Hotspot detection (heavy: median/MAD/argmax/centroid over a frame).
    frame = gaussian_frame(640, 480, 330.0, 250.0)
    py = bench(lambda: py_detect(frame, snr_threshold=5.0), 200)
    rs = bench(lambda: rc.detect_hotspot(frame, snr_threshold=5.0), 200)
    row("detect_hotspot 640x480", py, rs)

    # 2) General transform (Python path uses scipy align_vectors).
    py = bench(lambda: tf.AzAlt2AzEl(123.0, 40.0, 10.0, 2.0), 2000)
    rs = bench(lambda: rc.AzAlt2AzEl(123.0, 40.0, 10.0, 2.0), 2000)
    row("AzAlt2AzEl (scipy vs Rust)", py, rs)

    # 3) Eq-mode sky->mount transform (numpy trig).
    py = bench(lambda: tf.local_elev_az_to_telescope(40.0, 123.0, lat_deg=45.0), 5000)
    rs = bench(lambda: rc.local_elev_az_to_telescope(40.0, 123.0, lat_deg=45.0), 5000)
    row("local_elev_az_to_telescope", py, rs)

    # 4) Single PID step, one-at-a-time from Python (FFI overhead visible).
    py_pid = PIDController(0.025, 0.01, 0.005, "AZM")
    rs_pid = rc.PidController(0.025, 0.01, 0.005, "AZM")
    py = bench(lambda: py_pid.compute_pid_output(5.0, 0.1, 100.0), 20000)
    rs = bench(lambda: rs_pid.compute_pid_output(5.0, 0.1, 100.0), 20000)
    row("PID step (per-call FFI)", py, rs)

    # 5) Full PROGRAM control cycle.
    #    Rust: SimCoreLoop.step() runs the WHOLE cycle (sim poll over the
    #    loopback + transform + 2x PID + actuate) inside Rust, one FFI call.
    #    Python: the equivalent assembled from the baseline modules.
    loop = rc.SimCoreLoop(0.0, 0.0)
    loop.set_mode("program")
    loop.set_mount_mode("altaz")
    loop.set_gains(0.025, 0.01, 0.0, 0.025, 0.01, 0.0)
    loop.set_setpoint(123.0, 40.0)
    rs = bench(lambda: loop.step(0.001), 5000)

    az_pid = PIDController(0.025, 0.01, 0.0, "AZM")
    al_pid = PIDController(0.025, 0.01, 0.0, "ALT")
    state = {"az": 0.0, "al": 0.0}

    def py_cycle():
        # poll (python attribute access stands in for the cached position)
        cur_az, cur_al = state["az"], state["al"]
        t_az, t_al = tf.AzEl2AzAlt_AltAz(123.0, 40.0, 0.0, 0.0)
        e_az = (t_az - cur_az + 180.0) % 360.0 - 180.0
        e_al = t_al - cur_al
        az_pid.get_current_rates(e_az, 0.001, measurement_degrees=cur_az)
        al_pid.get_current_rates(e_al, 0.001, measurement_degrees=cur_al)

    py = bench(py_cycle, 5000)
    row("PROGRAM cycle (compute)", py, rs)

    print("-" * 70)
    print("Note: the PID per-call row reflects FFI overhead, not compute; the")
    print("loop runs cycles inside a Rust thread with no per-cycle FFI, so the")
    print("'PROGRAM cycle' row is the figure that matters for the live loop.")


if __name__ == "__main__":
    main()
