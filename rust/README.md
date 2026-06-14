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
| PyO3 bindings (`extension-module` feature) | `skytracker_core/src/python_bindings.rs` |
| Rust golden-byte + closed-loop tests | `skytracker_core/tests/encoding.rs` |
| Cross-language byte-parity test | `../test_rust_mount_parity.py` |

## Build & test

Prerequisites: Rust (MSVC toolchain) + `maturin` in the `track` conda env.

```sh
# Pure-Rust tests (no Python needed): golden bytes + sim closed loop
cd rust/skytracker_core
cargo test

# Build the extension into the active env, then run cross-language parity
maturin develop --release
cd ../..
python test_rust_mount_parity.py
```

## What "parity" means here

`test_rust_mount_parity.py` drives the real `NexstarHandController` through a
recording fake-serial and asserts the bytes it writes are **identical** to the
bytes the Rust encoders produce — for `get_position`, `get_version`,
`slew_fixed` (both signs), and `goto_fast`. So the port is verified against the
actual Python wire output, not a hand-copied spec.

## Next steps (not yet done)

2. Port `control.py::PIDController`; validate against `test_pid_control.py`.
3. Port `hotspot.py::detect_hotspot`; validate against `test_hotspot.py`.
4. Move the `mount_control.py` fixed-rate loop into a Rust thread owning
   Mount+PID; Python pushes setpoints (from skyfield) and pulls snapshots.
