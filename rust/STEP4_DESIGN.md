# Step 4 design — control loop in Rust

Design only. No code yet. Grounded in the actual `JoystickModeState.tracking_control()`
dispatch and the `program_track` / `hotspot_track` handlers in
`joystick_controller.py`, and the existing `MountControlThread` in
`mount_control.py`.

## Goal and the one hard constraint

Move the fixed-rate `poll → decide → command` loop into a Rust thread that owns
the mount serial link, the PID, and the per-mode actuation math — so the control
cadence is off the GIL and the GC, with deterministic timing.

**The constraint that shapes everything:** a Rust thread that does not hold the
GIL cannot touch Python objects. The loop's current inputs are largely
Python-owned and must stay that way:

| Input | Owner | Why it stays Python |
|-------|-------|---------------------|
| Joystick axes (RATE_CONTROL) | pygame | input device lives in the UI process |
| Camera frame (HOTSPOT) | `camera_manager` / zwoasi | ASI SDK + capture thread are Python |
| Trajectory / setpoint (PROGRAM) | skyfield + selected satellite | astronomy stays Python (agreed scope) |
| Config (gains, limits, offsets, cam params) | `config_state` | edited live in the UI |

So step 4 is **not** "move `tracking_control()` into Rust." It's: the Rust loop
owns cadence + serial + PID + actuation; Python keeps owning the inputs it
already owns and *pushes copies* of them across a plain-Rust boundary. The loop
reads only Rust-owned memory and never acquires the GIL on the hot path.

## The seam

```
┌──────────────────────── Python shell (UI process) ────────────────────────┐
│  pygame UI / joystick    skyfield trajectory     camera_manager (zwoasi)   │
│         │                       │                        │                 │
│   axes + mode          mount-frame setpoint        latest frame            │
│         │                  + ff rates                    │                 │
│         ▼                       ▼                        ▼                 │
│   core.push_inputs(...)   core.push_setpoint(...)  core.push_frame(...)    │
│   core.submit_command(...)  (goto / manual slew / stop)                    │
│                                                                            │
│   snap = core.snapshot()  ──► positions, errors, hotspot lock, hz,         │
│                               requested_mode, status messages              │
└──────────────────────────────────┬─────────────────────────────────────────┘
            push (Python holds GIL) │ pull
┌──────────────────────────────────▼─────────────────────────────────────────┐
│  skytracker_core::CoreLoop   (Rust OS thread @ 15 Hz, NO GIL on hot path)   │
│                                                                            │
│   owns: Mount (real serialport)  +  PidController(AZM/ALT)                  │
│   reads: Arc<Mutex<Inputs>> / ArcSwap<Frame>  (Rust-owned copies)          │
│   writes: Arc<Mutex<Outputs>>                                               │
│                                                                            │
│   each period:                                                             │
│     1. poll position via Mount (single serial read)                        │
│     2. drain command queue (goto / manual slew / stop)                     │
│     3. run active mode → discrete rate → Mount.hc_slew_fixed               │
│     4. write Outputs snapshot                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The Mount serial port is owned **solely** by the loop. Today the UI thread also
issues commands (goto, manual slews); those now go through `submit_command` →
the loop's command queue, preserving the "one owner of the wire" invariant that
`auxstar`'s lock and `MountControlThread` were built around.

## The shared interface (fields drawn from the real handlers)

### Inputs — `Arc<Mutex<Inputs>>`, written by Python, read by the loop
```
connected: bool
stopped: bool
mode: Mode                      // STANDBY | RATE | PROGRAM | HANDOFF | HOTSPOT | MTI
offsets: { azm, alt }
limits:  { azm_min, azm_max, alt_min, alt_max }
gains:   { azm:(p,i,d), alt:(p,i,d) }

// RATE_CONTROL: Python maps joystick → desired discrete rate and pushes it
rate_cmd: { azm: i32, alt: i32 }            // -9..=9

// PROGRAM: Python computes target in MOUNT coordinates (see "deferring
// transforms" below) and pushes it; loop does error = setpoint - position.
setpoint: Option<{ azm_deg, alt_deg, ff_azm_dps, ff_alt_dps, seq }>

// HOTSPOT params (the loop does detect + geometry itself)
hotspot: { snr_threshold, gate_radius, coast_time_s, x_sign, y_sign,
           pixel_size_um, focal_length_mm, rotation_deg, cam_index }
```

### Frame — `ArcSwap<Frame>`, written by the camera shim, read by the loop
```
Frame { data: Arc<Vec<f32>>, h: usize, w: usize, seq: u64, t_capture: f64 }
```
The camera thread already allocates a fresh buffer per frame; the shim converts
it once (to_intensity → f32) under the GIL and `store`s a new Arc. The loop
`load`s the current Arc with no copy and no GIL. (A 1920×1280 mono frame is
~10 MB as f32; at 15–30 Hz that conversion is well within budget, and can be
narrowed to the gate window if ever needed.)

### Outputs — `Arc<Mutex<Outputs>>`, written by the loop, read by Python
```
position: { azm, alt, azm_raw, alt_raw, fresh }
errors:   { azm_pos_err, alt_pos_err, azm_rate_cmd, alt_rate_cmd,
            azm_pid_output, alt_pid_output }
hotspot:  { acquired, gate_center, centroid, snr, miss_count, status }
requested_mode: Option<Mode>    // loop-originated transitions (see below)
status_msgs: Vec<String>        // drained by the UI for the status line
stats: { actual_hz, cycle_ms, poll_ms }
```

## Mode-by-mode responsibility split

| Mode | Python pushes | Rust loop does |
|------|---------------|----------------|
| STANDBY | `mode` | one-shot stop on entry (the `_standby_motion_stopped` guard moves into the loop) |
| RATE_CONTROL | mapped `rate_cmd` from joystick | actuate `hc_slew_fixed` |
| PROGRAM | mount-frame `setpoint` + ff rates (from skyfield) | error = setpoint − position (AZM wrap), PID, slew |
| HOTSPOT | `frame` + `hotspot` params | detect_hotspot → pixel_offset_to_angles → PID → slew; **lock state machine** (gate, miss count, coast) |
| HANDOFF / MTI | — | stubs (as today) |

## Three thorny bits and the proposed decisions

1. **Loop-originated mode transitions.** `hotspot_track` can switch itself to
   STANDBY (safety limit) or PROGRAM (lost lock). Mode is therefore *bidirectional*:
   Python sets `Inputs.mode`, but the loop can publish `Outputs.requested_mode`,
   which the UI applies to its own enum next tick (with a seq guard). The HOTSPOT
   lock/coast state machine moves wholesale into the loop.

2. **PROGRAM needs the mount transform — defer it.** `program_track` →
   `compute_mount_position_error` uses `transformations.py` (AltAz / Eq / passthrough)
   to convert a sky target into mount coordinates. Porting those is a separate
   concern (a possible step 5). **Decision for step 4: Python pushes the target
   already in mount coordinates**, so the loop only does `error = setpoint −
   position` + AZM wrap + PID. This keeps skyfield *and* the coordinate math in
   Python and shrinks step 4 to the loop mechanics.

3. **Frame sharing across threads.** Use `ArcSwap<Frame>` so the camera shim
   publishes and the loop reads without locking or copying on the read side, and
   without the loop ever holding a reference to a live numpy buffer.

## Validation — the HW simulator is the harness

Per the memory notes, the software simulator already closes the loop: `SimMount`
integrates the rate commands into a true pointing, and `SimCap` renders frames
through the *real* camera pipeline. That makes it a perfect parity harness,
extending the steps 1–3 philosophy to the loop itself:

- Run identical sim scenarios (a launch trajectory; a TLE satellite) through
  **(a)** the existing Python `MountControlThread` and **(b)** the Rust
  `CoreLoop`, and compare the closed-loop traces: acquisition time, steady-state
  pointing error, rate-command sequence.
- Because `angles_to_pixel` is the exact inverse of `pixel_offset_to_angles`
  (already ported in step 3), the HOTSPOT loop converges by construction in sim,
  so divergence between the two implementations is the signal we watch.

Sign-convention calibration on real hardware remains out of scope for sim (the
memory notes flag this); the Rust loop inherits the same `x_sign`/`y_sign`
parameters, so it's no worse off than today.

## Migration order & flag-gating

1. `CoreLoop` skeleton: thread + Inputs/Outputs/Frame/command-queue plumbing,
   STANDBY + RATE only. Prove cadence + serial ownership against the sim.
2. Add PROGRAM (mount-frame setpoint) and HOTSPOT (detect + lock SM).
3. Wire a thin Python `CoreLoopAdapter` that mimics today's call sites
   (`push_*` from the UI/camera/joystick ticks; `snapshot` for rendering).
4. **Opt-in via config** (`use_rust_core_loop`, default off). `MountControlThread`
   stays the default until the Rust loop is proven on hardware. Both can run the
   same sim scenario for A/B comparison.

This keeps the live path untouched until you flip the flag, and every stage is
checkable against the simulator with no hardware.

## Open questions for you

- **Setpoint in mount coords (defer transforms) vs. port `transformations.py`
  now?** The doc assumes defer. Porting transforms is a clean, pure, well-testable
  step 5 if you'd rather the loop be fully self-contained.
- **RATE_CONTROL: push mapped discrete rates (simplest) vs. push raw joystick
  axes and map in Rust?** The doc assumes Python keeps the axis→rate mapping.
- **Command queue scope:** which manual operations (goto, set-position, manual
  slew, focus?) must route through the loop on day one vs. stay direct until the
  flag flips.
