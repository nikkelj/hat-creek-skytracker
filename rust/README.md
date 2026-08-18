# Rust workspace (full-port program)

This directory is a **Cargo workspace** and the home of the full Rust port
(branch `rust-port`). The original Rust-core / Python-shell experiment
(branch `rust-core-experiment`) proved the pattern; the port now proceeds
in phases — engines first, egui UI last, Python retired at the end. The
master plan lives with the phase gates in the repo plan file and
`doc/BENCH_CHECKLIST.md`; golden validation vectors live in
`../tests/golden/` (recorded by `../tools/record_golden.py`).

Crates:

| Crate | Role |
|-------|------|
| `skytracker-core` | Pure-Rust engine: protocol, sim, PID, hotspot, transforms, controller, core loop (no pyo3) |
| `skytracker-astro` | Phase 1 astro engine replacing skyfield: timescales/GAST (IAU2000A tables generated from skyfield), SGP4 passes, pure-Rust SPK reader for de421.bsp, body/star apparent places, TLE catalog. Parity: sats 0.03″, GAST 1.9 ms, bodies 0.79″, stars 0.17″; bulk precompute 76× |
| `skytracker-ffi` | The ONLY pyo3 crate; builds the Python module `skytracker_core` (strangler seam, deleted at the end). Exposes the core-loop classes + `AstroEngine` |
| `skytracker-platesolve` | Phase 2 tetra3 port: pattern hash bit-exact vs the existing .npz databases, centroids 0.0 px, solutions numerically identical to Python tetra3 (12/12 A/B fields) |
| `skytracker-pointing` | Phase 2b TPOINT fits (alt-az + equatorial incl. partial/robust modes, polar-axis fit): coefficients at machine precision vs numpy (4e-16 deg) |
| `skytracker-imaging` | Phase 3a imaging primitives (filters/warps/phase-correlate/Shi-Tomasi/LK/RANSAC) at cv2 parity: filters ≤9e-5, LK 0.006 px, RANSAC identical. Phase 3b composes them into the stacking/stabilizer/sharpen pipelines |
| *(planned)* `skytracker-camera`, `-adsb`, `-app` | Phase 4–7 engines and the final eframe binary |

## Ported so far

`skytracker-core/` ports the **used subset** of `lib/auxstar.py` (the NexStar 'P'
pass-through path) to Rust, plus a **byte-level** simulated mount the existing
Python `SimMount` lacks (it duck-types at the method level and never speaks
bytes). This lets a full `encode -> transact -> parse` round-trip run with no
hardware.

| Piece | File |
|-------|------|
| Protocol encoding (byte-faithful port) | `skytracker-core/src/protocol.rs` |
| `Mount<Transport>` + byte-level `SimResponder` + loopback | `skytracker-core/src/sim.rs` |
| `PidController` (port of `control.py`) | `skytracker-core/src/pid.rs` |
| Hotspot detect + geometry (port of `hotspot.py`) | `skytracker-core/src/hotspot.rs` |
| Coordinate transforms (port of `transformations.py`) | `skytracker-core/src/transforms.rs` |
| Control-loop decision logic (pure) | `skytracker-core/src/controller.rs` |
| Threaded fixed-rate loop (port of `mount_control.py`) | `skytracker-core/src/core_loop.rs` |
| PyO3 bindings (`extension-module` feature) | `skytracker-ffi/src/bindings.rs` |
| Rust unit + integration tests | `skytracker-core/tests/*.rs` |
| Cross-language parity tests | `../test_rust_*.py` |

## Build & test

Prerequisites: Rust (MSVC toolchain) + `maturin` in the `track` conda env.

```sh
# Pure-Rust tests (no Python needed): protocol, pid, hotspot, transforms,
# controller, and closed-loop integration against the byte-level sim.
cd rust
cargo test --workspace

# Build the extension (skytracker-ffi crate; Python module name stays
# `skytracker_core`) into the active env, then run cross-language parity +
# the Python-driven control loop.
cd skytracker-ffi
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
