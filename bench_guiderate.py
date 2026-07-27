#!/usr/bin/env python
"""
Bench test: verify and calibrate the Celestron AVX fine variable-rate primitive
(MC_SET_POS/NEG_GUIDERATE) for smooth tracking.

Background: the 10-step MC_MOVE table is geometric, so steady tracking has a
quantization sawtooth ~ rate-gap x control-dt. The 24-bit guide-rate command
should give effectively continuous rates and remove that sawtooth. The on-wire
scale was CALIBRATED with this script on the real AVX (2026-07-25): the 24-bit
value is arcsec/s * 1024 (the original rev/sec assumption measured a dead-flat
ratio of 0.01265 = exactly 1/79.1, and full scale works out to 4.551 dps = the
AVX max slew). Re-run after firmware changes or on a new mount. This script
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
    python bench_guiderate.py --port COM5 --survey       # measure MC_MOVE 1..9
    python bench_guiderate.py --port COM5 --throughput   # link Hz / latency

--survey measures every discrete MC_MOVE step instead: the shipped RATES table
is suspect (measured rate 4 = 8x sidereal on the AVX, not the 0.25 dps listed),
and the discrete-rate PID path and the simulator both consume that table.
Paste the survey output into RATES once measured.

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


def preload(hc, target, sign, rate_dps=0.5, seconds=1.5):
    """Take up gear backlash in the direction about to be measured.

    Field data 2026-07-25: the first sample after a direction reversal read
    ratio 0.68 with net travel the WRONG way -- the gear train was taking up
    lash while the encoder sat still. Run the axis briefly in the new
    direction (~0.75 deg) so measurements start with the train loaded.
    """
    try:
        hc.hc_set_rate_dps(target, sign * rate_dps)
        time.sleep(seconds)
    finally:
        stop(hc, target)
    time.sleep(0.3)


def run(port, target_name, max_dps, travel_limit_deg, settle_s=0.75):
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
        # Load the gear train in this sweep's direction before measuring.
        preload(hc, target, sign)
        for r in rates:
            cmd = sign * r
            try:
                hc.hc_set_rate_dps(target, cmd)
            except Exception as e:
                print(f"  command error at {cmd:+.3f} dps: {e}")
                stop(hc, target)
                continue
            # Let the motor ramp to speed before fitting the slope (2026-07-25
            # run: the ramp inside the sample window read as ratio 0.99 at
            # 1 dps tapering to 0.96 at 3 dps).
            time.sleep(settle_s)
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

    # MC_MOVE reference for comparison. NOTE: the shipped RATES table is
    # suspect (2026-07-25: rate 4 measured 0.0335 dps = 8.0x sidereal, not the
    # 0.25 dps the table lists) -- run --survey to measure the real table.
    print("\n  Reference, discrete MC_MOVE rate 4 (RATES table claims "
          f"{RATES[4] * 360:.3f} dps; AVX measured 8x sidereal = 0.0334):")
    preload(hc, target, +1.0)  # reference runs positive after the negative sweep
    hc.hc_slew_fixed(target, 4)
    time.sleep(settle_s)
    actual, travel = measure_rate(hc, target)
    stop(hc, target)
    print(f"    actual = {actual:+.4f} dps (travel {travel:+.2f})")

    print("\nDone. Interpreting results:")
    print("  * ratio ~1.0 across rates  -> encoding scale is correct, command honored.")
    print("  * ratio constant but != 1  -> fix the scale in hc_set_rate_dps by that factor.")
    print("  * actual flat/zero above some rate -> that's the guide-rate max; set")
    print("    guide_rate_max_dps below it and let MC_MOVE handle faster (near-zenith).")
    print("  * smooth actual at low rates -> the sawtooth fix is real on this mount.")


SIDEREAL_DPS = 360.0 / 86164.0905  # 0.0041781 deg/s


def run_survey(port, target_name, travel_limit_deg):
    """Measure every discrete MC_MOVE step (1..9), out-and-back per step so the
    axis roughly returns to start. Prints a ready-to-paste RATES table."""
    target = Targets.ALT if target_name == "alt" else Targets.AZM
    hc = NexstarHandController(port)
    print(f"Connected on {port}. Target axis: {target_name.upper()}")
    print(f"Start position: {read_deg(hc, target):.4f} deg")
    print("\n  step   actual_dps   x sidereal   table_dps   travel")
    print("  ----   ----------   ----------   ---------   ------")

    measured = {}
    # Short dwells at high steps: 9 can run multiple deg/s.
    dwell = {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 3.0, 6: 2.5, 7: 2.0, 8: 1.5, 9: 1.2}
    for step in range(1, 10):
        fwd = rev = 0.0
        for sign in (+1, -1):
            # Brief pre-load so lash from the previous (opposite) leg doesn't
            # bleed into the slope fit, then settle past the motor ramp.
            preload(hc, target, float(sign), seconds=1.0)
            hc.hc_slew_fixed(target, sign * step)
            time.sleep(0.5)
            actual, travel = measure_rate(hc, target, sample_s=dwell[step])
            stop(hc, target)
            time.sleep(0.3)
            if sign > 0:
                fwd = actual
            else:
                rev = actual
            if abs(travel) > travel_limit_deg:
                print(f"  step {step}: travel limit hit ({travel:+.2f} deg), "
                      "skipping remaining steps")
                break
        actual = (abs(fwd) + abs(rev)) / 2.0
        measured[step] = actual
        print(f"   {step}     {actual:+9.4f}   {actual / SIDEREAL_DPS:9.2f}   "
              f"{RATES[step] * 360.0:9.4f}   ok")

    print("\nMeasured RATES table (rev/sec, paste into lib/auxstar.py):")
    print("RATES = {")
    print("    0 : 0.0,")
    for step in range(1, 10):
        if step in measured:
            print(f"    {step} : {measured[step] / 360.0:.9f},"
                  f"  # {measured[step]:.4f} dps = "
                  f"{measured[step] / SIDEREAL_DPS:.1f}x sidereal (measured)")
    print("}")


def _time_calls(fn, seconds):
    """Call fn back-to-back for `seconds`. Returns (durations, errors):
    per-call round-trip times for the calls that succeeded, and the count
    that raised (serial timeout/short read)."""
    durations, errors = [], 0
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            errors += 1
            continue
        durations.append(time.perf_counter() - t0)
    return durations, errors


def _report_timing(label, durations, errors, seconds):
    n = len(durations)
    if not n:
        print(f"  {label}: NO successful transactions in {seconds:.0f} s "
              f"({errors} errors)")
        return 0.0
    s = sorted(durations)
    ms = 1000.0
    hz = n / sum(s)
    print(f"  {label}:")
    print(f"    {n} transactions, {hz:7.1f} Hz sustained, {errors} errors")
    print(f"    round-trip ms: mean {sum(s) / n * ms:5.1f}   p50 {s[n // 2] * ms:5.1f}   "
          f"p95 {s[int(n * 0.95)] * ms:5.1f}   max {s[-1] * ms:6.1f}")
    return hz


def run_throughput(port, target_name, seconds):
    """Benchmark the serial link + firmware transaction throughput.

    Three measurements, each a saturating back-to-back loop:
      1. telemetry  -- MC_GET_POSITION on the chosen axis,
      2. command    -- MC_SET_POS_GUIDERATE 0 (a stop; exercises the command
                       path with zero motion risk),
      3. full cycle -- read AZM + read ALT + command AZM + command ALT, the
                       wire cost of one real control-loop iteration.
    The full-cycle Hz is the number that bounds the control loop cadence
    (mount_control targets 15 Hz).
    """
    target = Targets.ALT if target_name == "alt" else Targets.AZM
    hc = NexstarHandController(port)
    print(f"Connected on {port}. Link throughput bench, {seconds:.0f} s per test.")
    print(f"Start position: {read_deg(hc, target):.4f} deg\n")

    durs, errs = _time_calls(lambda: hc.hc_get_position(target), seconds)
    _report_timing(f"Telemetry (MC_GET_POSITION {target_name.upper()})",
                   durs, errs, seconds)

    durs, errs = _time_calls(lambda: hc.hc_set_rate_dps(target, 0.0), seconds)
    _report_timing(f"Command (MC_SET_POS_GUIDERATE 0, {target_name.upper()})",
                   durs, errs, seconds)

    def cycle():
        hc.hc_get_position(Targets.AZM)
        hc.hc_get_position(Targets.ALT)
        hc.hc_set_rate_dps(Targets.AZM, 0.0)
        hc.hc_set_rate_dps(Targets.ALT, 0.0)

    durs, errs = _time_calls(cycle, seconds)
    hz = _report_timing("Full control cycle (read AZM+ALT, command AZM+ALT)",
                        durs, errs, seconds)

    stop(hc, Targets.AZM)
    stop(hc, Targets.ALT)
    if hz > 0.0:
        print(f"\n  -> the wire supports a {hz:.1f} Hz control loop "
              f"(mount_control targets 15 Hz; headroom {hz / 15.0:.1f}x).")
        if hz < 20.0:
            print("     WARNING: under ~20 Hz there is little margin over the "
                  "15 Hz target -- expect skipped cycles under serial faults.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--target", choices=["azm", "alt"], default="azm")
    ap.add_argument("--max-dps", type=float, default=3.0,
                    help="don't command rates above this (default 3.0; raise carefully)")
    ap.add_argument("--travel-limit", type=float, default=8.0,
                    help="abort a step if the axis travels more than this many degrees")
    ap.add_argument("--survey", action="store_true",
                    help="measure the discrete MC_MOVE steps 1..9 instead "
                         "(the shipped RATES table is suspect)")
    ap.add_argument("--throughput", action="store_true",
                    help="benchmark link transaction throughput instead: "
                         "telemetry Hz, command Hz, and the full "
                         "read+command control-cycle Hz (no sustained motion)")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="duration of each throughput test (default 5.0)")
    ap.add_argument("--settle", type=float, default=0.75,
                    help="seconds to let the motor ramp before sampling a "
                         "rate measurement (default 0.75)")
    args = ap.parse_args()
    try:
        if args.survey:
            run_survey(args.port, args.target, args.travel_limit)
        elif args.throughput:
            run_throughput(args.port, args.target, args.seconds)
        else:
            run(args.port, args.target, args.max_dps, args.travel_limit,
                settle_s=args.settle)
    except KeyboardInterrupt:
        print("\nInterrupted -- sending stop.")
        try:
            hc = NexstarHandController(args.port)
            hc.hc_slew_fixed(Targets.AZM, 0)
            hc.hc_slew_fixed(Targets.ALT, 0)
        except Exception:
            pass
        sys.exit(1)
