# Step 4 integration — running the Rust control loop in the app

This wires the Rust control loop into the live app behind a flag. It is **off by
default**; with the flag off the app is byte-for-byte unchanged.

## Enabling it

Three ways, in order of convenience:

- **UI toggle (recommended):** Hardware Simulator screen → **"Loop: PYTHON/RUST"**
  button (next to Save/Load). It flips `use_rust_core_loop`, auto-saves, and
  takes effect on **restart** ("restart app to apply" is shown next to it).
- Config flag directly: `config_state.use_rust_core_loop = True`
  (`rust_core_loop_hz` optional, default 15).
- Environment: `set SKYTRACKER_RUST_LOOP=1` (Windows) — overrides the flag for a
  single run.

When enabled, `main.py` starts a `RustCoreLoopAdapter` instead of
`MountControlThread`. If `skytracker_core` isn't importable, it logs a warning
and falls back to `MountControlThread`, so enabling the flag can never hard-fail
startup.

## How it's wired (and why this is low-risk)

`RustCoreLoopAdapter` (`rust_loop_adapter.py`) mirrors `MountControlThread`'s
`start/stop/join` interface. On connect it builds `skytracker_core.CoreLoop`
**bridged to the already-connected controller** (`CoreLoop.wrap_mount`), which
works identically for the real `NexstarHandController` and the sim `SimMount`.
So:

- **No change to the connect flow** — the controller is created exactly as today.
- **Manual UI commands still work** — they call the controller directly; the
  controller's internal lock serializes them with the loop's calls.
- **Flag off → zero code path change** (the adapter isn't even imported).

What now runs **in Rust** (off the GIL, on the loop thread): the fixed-rate
cadence, position poll handling, PID, sky→mount transforms, hotspot detection,
and the hotspot lock/coast/loss state machine. Only the brief serial calls
acquire the GIL (via the bridge).

What stays **in Python**: the skyfield trajectory → setpoint, the joystick read
+ rate mapping (`axis_to_rate`, now shared with the adapter), camera capture,
and all UI/manual commands.

## What to validate in the UI

Enable the flag, then exercise each mode and confirm parity with flag-off:

**Hardware simulator** (HW Sim screen, sim enabled):
1. **RATE_CONTROL** — joystick slews both axes; motion matches flag-off feel.
2. **PROGRAM** — select a satellite; the mount tracks it; position/error
   readouts update.
3. **HOTSPOT** — point at the rendered plume/target; it locks (`status=locked`,
   SNR shown), stays centered, coasts on brief dropout, and falls back to
   PROGRAM on sustained loss.

**Real hardware**: same three, connected to the mount's COM port. Watch the
`x_sign`/`y_sign` HOTSPOT calibration (the sim can't validate sign conventions).

Compare the on-screen position/error/rate readouts and the tracking behavior
against the flag-off (`MountControlThread`) path.

## Known gaps (deferred — use flag-off for these until ported)

- **PROGRAM launch tracking** and **below-horizon mask-exit**: handled by
  `JoystickModeState.program_track` in Python. The adapter currently pushes the
  satellite setpoint only; for the launch/mask-exit cases it clears the setpoint
  so the loop **holds** (it won't misbehave, but it won't drive to a mask-exit
  point). Porting that logic is the natural next step.
- **HANDOFF / MTI**: stubs, as in the Python loop.

## Performance

`python bench_rust_vs_python.py` reports the speedups (e.g. transforms 58×/562×,
PID 114×, hotspot detection 1.5×, full PROGRAM cycle 1.7×).
`python test_rust_perf.py` is a regression guard. Note the current bridge holds
the GIL during each serial call; a later `CoreLoop.open_serial` path would make
the serial I/O fully off-GIL too.

## Rollback

Set the flag off (or unset the env var). No other change needed.

## Unrelated pre-existing issue

`test_transform.py` fails to import `AzEl2AzAlt` from `transformations.py` — that
bare name does not exist there (only `AzEl2AzAlt_AltAz` / `_Passthrough`). This
predates the Rust work and is independent of it.
