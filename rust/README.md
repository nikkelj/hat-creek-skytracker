# Rust core (experiment)

Exploration of a **Rust-core / Python-shell** split for the skytracker. The goal
is to move the real-time, hardware-touching parts (mount serial protocol, the
fixed-rate control loop, PID, hotspot detection) into a Rust extension module
while keeping skyfield trajectories, the pygame UI, and capture-to-disk in
Python. See branch `rust-core-experiment`.

## Status — Step 1: Mount protocol port

`skytracker_core/` ports the **used subset** of `lib/auxstar.py` (the NexStar 'P'
pass-through path) to Rust, plus a **byte-level** simulated mount the existing
Python `SimMount` lacks (it duck-types at the method level and never speaks
bytes). This lets a full `encode -> transact -> parse` round-trip run with no
hardware.

| Piece | File |
|-------|------|
| Protocol encoding (byte-faithful port) | `skytracker_core/src/protocol.rs` |
| `Mount<Transport>` + byte-level `SimResponder` + loopback | `skytracker_core/src/sim.rs` |
| `PidController` (port of `control.py`) | `skytracker_core/src/pid.rs` |
| Hotspot detect + geometry (port of `hotspot.py`) | `skytracker_core/src/hotspot.rs` |
| Coordinate transforms (port of `transformations.py`) | `skytracker_core/src/transforms.rs` |
| Control-loop decision logic (pure) | `skytracker_core/src/controller.rs` |
| Threaded fixed-rate loop (port of `mount_control.py`) | `skytracker_core/src/core_loop.rs` |
| PyO3 bindings (`extension-module` feature) | `skytracker_core/src/python_bindings.rs` |
| Rust unit + integration tests | `skytracker_core/tests/*.rs` |
| Cross-language parity tests | `../test_rust_*.py` |

## Build & test

Prerequisites: Rust (MSVC toolchain) + `maturin` in the `track` conda env.

```sh
# Pure-Rust tests (no Python needed): protocol, pid, hotspot, transforms,
# controller, and closed-loop integration against the byte-level sim.
cd rust/skytracker_core
cargo test

# Build the extension into the active env, then run cross-language parity +
# the Python-driven control loop.
maturin develop --release
cd ../..
python test_rust_mount_parity.py
python test_rust_pid_parity.py
python test_rust_hotspot_parity.py
python test_rust_transforms_parity.py
python test_rust_core_loop.py
```

## What "parity" means here

`test_rust_mount_parity.py` drives the real `NexstarHandController` through a
recording fake-serial and asserts the bytes it writes are **identical** to the
bytes the Rust encoders produce — for `get_position`, `get_version`,
`slew_fixed` (both signs), and `goto_fast`. So the port is verified against the
actual Python wire output, not a hand-copied spec.

## Status

Steps 1–5 are complete and validated in software (see `FINDINGS.md`): protocol,
PID, hotspot, transforms, and the control loop — the loop closes against the
byte-level sim from both Rust and Python. What remains is the hardware-
integration boundary (a real serial `Transport`, a real-port `CoreLoop`, and
flag-gated wiring into `joystick_controller.py`), which needs the mount on the
bench; see the "What remains" section of `FINDINGS.md`.
