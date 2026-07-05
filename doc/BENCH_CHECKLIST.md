# Bench-Day Checklist — hardware validation of the Claude-era changes

Ordered procedure for the first sessions back on the real AVX mount and ASI
cameras. Everything below exists because a specific class of bug cannot be
seen in simulation; the order is chosen so each step validates what the next
step depends on. Do not skip ahead: continuous-rate tracking, for example, is
built on a wire encoding that is *unverified on real firmware* (step 2).

Throughout: keep a hand on the power switch, set conservative
`azm/alt_limit_*` values first, and remember **STOP (Circle) cancels
everything, including an in-flight Park**.

---

## 0. Prep (at the desk, before hardware)

- [ ] `pytest` green locally; CI green on the branch you're about to run.
- [ ] Config sanity: site lat/lon/alt set (without them, per-frame trajectory
      CSV rows fall back to az=0 — they *are* written now, but with degraded
      azimuth); safety limits set to the mount frame values you actually mean
      (they gate encoder positions; in AltAz mount ALT = 90 − sky el).
- [ ] Rehearse the full sequence once in sim with the hardware seams ON:
      HW Sim → `Mount link: BYTE-LEVEL SERIAL`, exposure timing on (default),
      a realistic `Goto slew rate (deg/s)` (3–4), some serial fault
      probabilities (e.g. short-read 0.02), and backlash/PE/noise set to
      plausible AVX values. The whole checklist below should work in that
      configuration before you touch hardware.

## 1. Serial link + basic mount sanity (Python loop, flag-off)

- [ ] Connect. Firmware version reads back (`MC_GET_VER`), position poll is
      stable at 15 Hz, `actual_hz` ≈ target in the console output.
- [ ] RATE mode: slow joystick slews in all four directions; verify axis
      directions match the on-screen readouts, and safety limits abort to
      STANDBY at the configured encoder values.
- [ ] STOP button: motion halts from both the UI press and the control loop.
- [ ] Park (Triangle): mount drives to the configured offsets, completes with
      "Park complete", and STOP cancels it mid-flight. It times out (90 s)
      rather than hanging if something is wrong.
- [ ] Fault behavior: pull the serial cable during a slow slew. Expect
      console/status "consecutive faults — stopping motion" and, since the
      link is dead, the "FAILED to stop motion" operator alert. The mount
      should coast to a stop on its own inertia; reconnect and confirm
      recovery. (This validates the watchdog paths test_sim_serial exercises.)

## 2. Guide-rate wire scale calibration — DO THIS BEFORE ANY TRACKING

The continuous variable-rate path (`hc_set_rate_dps`) assumes the
`pack_int3(rate/360)` rev/sec convention. The encoding *crashed* until the
2026-07 fix and the on-wire **scale has never been verified on firmware**.

- [ ] Run `python bench_guiderate.py` per its header: command a series of
      known rates on one axis, measure actual encoder motion over time.
- [ ] Record the measured scale factor and the maximum rate the firmware
      honors before it clamps/ignores (`guide_rate_max_dps` candidate).
- [ ] If the scale is wrong: fix `hc_set_rate_dps` (and the mirrored
      `protocol.rs encode_set_guiderate`), re-run the byte-parity suite, and
      update the SimMount quantization comment. Do NOT proceed with
      `continuous_rate_tracking` enabled until this passes.
- [ ] Expected failure signatures: no motion at all (scale far too small),
      immediate max-rate slew (far too large), or motion only above a
      threshold (firmware minimum) — all safe at guide rates, which is why
      this comes before PROGRAM/HOTSPOT.

## 3. Camera pipeline

- [ ] Both ASI cameras connect; exposure/gain controls respond; frame rate is
      consistent with the exposure setting (the sim now models this — the
      real thing should look familiar).
- [ ] Capture a short labeled run and verify on disk: frames named
      `CameraN_000001__...` starting at sequence **1**, `trajectory.csv`
      contains per-frame rows, timestamps are exposure-midpoint (a frame of a
      moving target should line up with the interpolated trajectory row).
- [ ] Watch for: NaN/hot-column artifacts (the detectors now tolerate them,
      but note which sensor produces them), USB stalls (feed freezes — the
      frozen-feed error path is still print-only), dropped-frame cadence.

## 4. Sign-convention calibration (sim cannot validate these)

- [ ] With a bright star centered, nudge each axis positive from RATE mode and
      record which way the star moves in each camera → set/confirm
      `hotspot_x_sign`, `hotspot_y_sign`, and per-camera
      `alignment_rotation`. A wrong sign makes HOTSPOT drive *away* at PID
      speed — verify before engaging it.
- [ ] Confirm ALT feed-forward sign: in AltAz mode the ALT axis runs opposite
      sky elevation; PROGRAM track on a slow, high pass should show the ALT
      rate leading in the correct direction.

## 5. Tracking progression (Python loop first, flag-off)

- [ ] PROGRAM on a slow, predictable pass (high-elevation-mask ISS or similar)
      with `continuous_rate_tracking` OFF (discrete MC_MOVE). Watch error
      convergence and the sawtooth character.
- [ ] Same pass class with continuous rate ON (only after step 2 passes).
      Expect the sawtooth to smooth out.
- [ ] Let a pass run to setting: PROGRAM should drive to the mask-exit point
      (Python loop) rather than chasing below the horizon.
- [ ] HOTSPOT handoff on a bright target (aircraft at dusk is ideal): PROGRAM
      → HOTSPOT, verify lock, coast on brief dropout, fallback to PROGRAM on
      loss, and the mount-limit abort.
- [ ] Long-exposure check: set the tracking camera to an exposure longer than
      the 15 Hz control period and confirm HOTSPOT stays stable (the
      stale-frame gate) instead of oscillating.

## 6. Rust core loop (flag-on) — only after 1–5 pass

Promotion gates (see also rust/FINDINGS.md):

- [ ] The closed-loop A/B parity harness exists and matches rate-command
      traces between the loops on identical sim scenarios. (Not yet built —
      build it before this bench session.)
- [ ] Repeat RATE / PROGRAM / HOTSPOT from step 5 with `use_rust_core_loop`
      on (bridge mode). Compare position/error/rate readouts against the
      Python loop on the same pass class.
- [ ] Cable-pull test again: the Rust loop must stop motion after 3 faults
      and the adapter watchdog must latch on a dead loop.
- [ ] Satellite set: mount must STOP when the setpoint clears (not keep its
      last rate; mask-exit pre-positioning is a known gap on this path).
- [ ] Only after all of the above: consider wiring `CoreLoop.open_serial`
      (true off-GIL serial) and re-run this section against it.

## 7. Post-session

- [ ] Stack a captured run end-to-end and note where the manual steps are
      (input for the post-processing finishing-stage work).
- [ ] Write what surprised you into LEARNINGS.md — especially any place the
      simulator's behavior diverged from the hardware, and then close that
      gap in `simulator.py` so the next desk session rehearses reality.
