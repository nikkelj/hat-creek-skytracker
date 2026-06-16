#!/usr/bin/env python
"""
Bench test: verify and calibrate the Celestron AVX fine variable-rate primitive
(MC_SET_POS/NEG_GUIDERATE) for smooth tracking.

Background: the 10-step MC_MOVE table is geometric, so steady tracking has a
quantization sawtooth ~ rate-gap x control-dt. The 24-bit guide-rate command
should give effectively continuous rates and remove that sawtooth -- IF the AVX
firmware honors it for sustained tracking and IF our on-wire encoding scale is
right (hc_set_rate_dps assumes the rev/sec convention; UNVERIFIED). This script
answers three questions on real hardware:

  1) Does MC_SET_POS_GUIDERATE produce smooth, sustained axis motion (not just a
     brief autoguide nudge)?
  2) What is the encoding scale?  actual_dps / commanded_dps  (ideally ~1.0).
  3) What is the max rate it will honor before it saturates/ignores?

It commands a rate, samples the encoder over a couple seconds, fits the slope to
get the ACTUAL rate, stops, and reports. Safety: small rates first, a hard
angular travel limit per step, and a stop after every step.

Usage:
    python bench_guiderate.py --port COM5 --target azm
    python bench_guiderate.py --port COM5 --target alt --max-dps 3.0

Start with the axis roughly mid-range and clear to move +/- a few degrees.
"""
import argparse
import sys
import time

from lib.auxstar import NexstarHandController, Targets, RATES


def read_deg(hc, target):
    return hc.hc_get_position(target) * 360.0


def unwrap_deg(prev, cur):
    """Shortest-arc delta so an azimuth 0<->360 wrap doesn't look like a jump."""
    return (cur - prev + 180.0) % 360.0 - 180.0


def measure_rate(hc, target, sample_s=2.5, hz=20.0):
    """Sample position vs time and least-squares fit the slope (deg/s)."""
    t0 = time.perf_counter()
    ts, xs = [], []
    base = read_deg(hc, target)
    acc = 0.0
    prev = base
    while time.perf_counter() - t0 < sample_s:
        cur = read_deg(hc, target)
        acc += unwrap_deg(prev, cur)
        prev = cur
        ts.append(time.perf_counter() - t0)
        xs.append(acc)
        time.sleep(1.0 / hz)
    n = len(ts)
    if n < 3:
        return 0.0, 0.0
    tbar = sum(ts) / n
    xbar = sum(xs) / n
    num = sum((t - tbar) * (x - xbar) for t, x in zip(ts, xs))
    den = sum((t - tbar) ** 2 for t in ts)
    slope = num / den if den else 0.0
    travel = xs[-1]  # net degrees moved during the sample
    return slope, travel


def stop(hc, target):
    # Stop via both primitives so we're safe regardless of which engaged.
    try:
        hc.hc_set_rate_dps(target, 0.0)
    except Exception:
        pass
    hc.hc_slew_fixed(target, 0)


def run(port, target_name, max_dps, travel_limit_deg):
    target = Targets.ALT if target_name == "alt" else Targets.AZM
    hc = NexstarHandController(port)
    print(f"Connected on {port}. Target axis: {target_name.upper()}")
    print(f"Start position: {read_deg(hc, target):.4f} deg")
    print("\n  cmd_dps   actual_dps   ratio   travel    note")
    print("  -------   ----------   -----   ------    ----")

    rates = [r for r in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0)
             if r <= max_dps]
    # Probe positive then negative so the axis returns toward where it started.
    for sign in (+1.0, -1.0):
        for r in rates:
            cmd = sign * r
            try:
                hc.hc_set_rate_dps(target, cmd)
            except Exception as e:
                print(f"  command error at {cmd:+.3f} dps: {e}")
                stop(hc, target)
                continue
            actual, travel = measure_rate(hc, target)
            stop(hc, target)
            time.sleep(0.3)
            ratio = actual / cmd if abs(cmd) > 1e-9 else float("nan")
            note = ""
            if abs(actual) < 0.2 * abs(cmd):
                note = "NOT MOVING (ignored?)"
            elif abs(ratio - 1.0) > 0.25:
                note = "scale off -> recalibrate encoding"
            if abs(travel) > travel_limit_deg:
                note = (note + " ").strip() + " travel-limit, stopping sweep"
            print(f"  {cmd:+7.3f}   {actual:+9.4f}   {ratio:5.2f}   {travel:+6.2f}   {note}")
            if abs(travel) > travel_limit_deg:
                break

    # MC_MOVE reference for comparison (discrete rate 4 ~ 0.25 dps).
    print("\n  Reference, discrete MC_MOVE rate 4 (table says "
          f"{RATES[4] * 360:.3f} dps):")
    hc.hc_slew_fixed(target, 4)
    actual, travel = measure_rate(hc, target)
    stop(hc, target)
    print(f"    actual = {actual:+.4f} dps (travel {travel:+.2f})")

    print("\nDone. Interpreting results:")
    print("  * ratio ~1.0 across rates  -> encoding scale is correct, command honored.")
    print("  * ratio constant but != 1  -> fix the scale in hc_set_rate_dps by that factor.")
    print("  * actual flat/zero above some rate -> that's the guide-rate max; set")
    print("    guide_rate_max_dps below it and let MC_MOVE handle faster (near-zenith).")
    print("  * smooth actual at low rates -> the sawtooth fix is real on this mount.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--target", choices=["azm", "alt"], default="azm")
    ap.add_argument("--max-dps", type=float, default=3.0,
                    help="don't command rates above this (default 3.0; raise carefully)")
    ap.add_argument("--travel-limit", type=float, default=8.0,
                    help="abort a step if the axis travels more than this many degrees")
    args = ap.parse_args()
    try:
        run(args.port, args.target, args.max_dps, args.travel_limit)
    except KeyboardInterrupt:
        print("\nInterrupted -- sending stop.")
        try:
            hc = NexstarHandController(args.port)
            hc.hc_slew_fixed(Targets.AZM, 0)
            hc.hc_slew_fixed(Targets.ALT, 0)
        except Exception:
            pass
        sys.exit(1)
