# Rust-core / Python-shell — experiment findings

Status as of the `rust-core-experiment` branch. Living document; updated as
steps land.

## The question

Is it worth moving the real-time, hardware-touching core of the skytracker
(mount serial protocol, fixed-rate control loop, PID, hotspot detection) into a
Rust extension module, while keeping skyfield trajectories, the pygame UI, and
capture-to-disk in Python? The win we're testing for: get the control loop off
the GIL/GC with real threads and deterministic timing, **without** losing
Python's iteration speed where it matters (UI, astronomy, experimentation).

## Approach

Incremental port behind a hard invariant: **every ported module must reproduce
the existing Python behavior, verified against the existing Python test suite**,
not a re-spec. Each step ships independently and is checkpointed by:

1. Rust unit tests mirroring the relevant `test_*.py`, and
2. a cross-language parity test that drives the *real* Python implementation and
   the Rust port through identical inputs and asserts they agree.

The codebase was already factored for this — `auxstar.py`, `control.py`, and
`hotspot.py` are pure/self-contained, and `mount_control.py` is the only place
that owns hardware.

## What's built

| Step | Module | Rust files | Verification |
|------|--------|-----------|--------------|
| 1 | NexStar `Mount` protocol (`lib/auxstar.py`) | `protocol.rs`, `sim.rs` | golden bytes + **byte-identical** to `NexstarHandController` wire output; byte-level `SimResponder` closes a slew→position loop |
| 2 | `PidController` (`control.py`) | `pid.rs` | 9 mirrored unit tests + **step-for-step** parity over 400 cycles × 6 configs (output to 1e-9, discrete rate exact) |
| 3 | Hotspot detection + geometry (`hotspot.py`) | `hotspot.rs` | 8 mirrored unit tests + **sub-pixel** parity (<0.05px) with the real numpy detector across noisy/color/gated frames; geometry to 1e-9. Crosses image data zero-copy via `PyReadonlyArray2`. |
| 5 | Coordinate transforms (`transformations.py`) | `transforms.rs` | 6 unit tests + full parity sweep incl. the scipy `AzAlt2AzEl` path (Rodrigues port) and numpy NaN-propagation; `sky_to_mount` gives the loop its per-cycle transform. |
| 4a | Control loop logic + threaded loop (`tracking_control`, `mount_control.py`) | `controller.rs`, `core_loop.rs` | pure `step()` (11 tests: all modes, hotspot lock/coast/loss/limit) + closed-loop convergence, moving-setpoint tracking, poll-fault skip, threaded spawn/stop against the byte-level sim (4 tests). |
| 4b | PyO3 loop adapter | `python_bindings.rs` (`SimCoreLoop`) | Python drives the full message-passing surface; 6 tests close the loop from Python with no hardware. |

Both also added a capability the Python side lacked: a **byte-level** simulated
mount. The Python `SimMount` duck-types the controller at the method level and
never speaks bytes, so it can't exercise a serial round-trip; `SimResponder`
can, which makes the port testable end-to-end with no hardware.

## Results & observations

- **Parity is achievable and cheap.** The two cleanest modules ported with zero
  behavioral drift. The translation was mechanical; the existing tests made
  correctness self-evident.
- **The FFI seam is comfortable.** PyO3 + maturin build cleanly against Anaconda
  CPython 3.11 (MSVC toolchain — no ABI friction). Scalars, bytes, and small
  classes cross the boundary with no ceremony.
- **The numpy boundary works (step 3).** Image frames cross zero-copy via
  `PyReadonlyArray2`; the Rust detector matches numpy to sub-pixel accuracy. The
  remaining float32-vs-f64 reduction differences are negligible (<0.05px) and
  could be eliminated entirely by reducing in f32 if ever needed.
- **`cargo test` stays Python-free.** pyo3 is an optional dependency gated behind
  the `extension-module` feature, so pure-Rust logic tests run without an
  interpreter; the cross-language tests skip cleanly if the wheel isn't built.

## Toolchain footprint (reversibility)

Installed for the experiment: VS Build Tools (C++ workload, MSVC 14.44), rustup
+ cargo 1.96, maturin 1.14. All cleanly removable (`rustup self uninstall`, VS
Build Tools uninstall) if the experiment is abandoned — and the whole thing
lives on a branch.

## Hardware integration — built, pending on-bench validation

Items 1–3 below are now done and validated in software; see
`STEP4_INTEGRATION.md`. What's left is interactive validation on the mount and
in the HW-sim UI (item 4), which by nature can't be done headlessly.

1. **Real serial `Transport`** — `serial.rs` (`SerialTransport`, serialport
   crate, mirrors auxstar's 9600 8N1 + short-timeout/flush). Done; needs a port
   to exercise.
2. **Threaded `CoreLoop` PyO3 class** — `open_serial` (real port, off-GIL) and
   `wrap_mount` (bridge to a Python mount for the HW sim). Validated via the
   bridge against a Python fake mount (`test_rust_core_loop_bridge.py`).
3. **Flag-gated wiring** — `RustCoreLoopAdapter` + `use_rust_core_loop`
   (default off). Bridges to the connected controller, so no connect-flow
   change; validated headlessly (`test_rust_loop_adapter.py`).
4. **On-bench validation (your step):** RATE / PROGRAM / HOTSPOT in the HW-sim
   UI and on hardware, plus `x_sign`/`y_sign` calibration (the sim can't
   validate sign conventions).

Deferred (updated 2026-07; use flag-off where these matter): satellite
**below-horizon mask-exit** (the adapter now commands a safe stop on a cleared
setpoint but doesn't pre-position at the mask-exit azimuth) and the **MTI**
stub. Launch tracking and HANDOFF, listed as deferred in earlier versions of
this note, have since been built (`_push_launch_setpoint`, `step_handoff`).
skyfield stays Python and feeds the setpoint. The 2026-07 P0 sprint also added
safety parity the port originally lacked: consecutive-fault safe-stop, panic
containment + adapter liveness watchdog, NaN-tolerant detection, and a
mount-frame target limit gate. See `doc/BENCH_CHECKLIST.md` for the ordered
on-mount validation procedure and the Rust promotion gates.

## Performance

`bench_rust_vs_python.py` (run on this machine): transforms **58×** (numpy) /
**562×** (scipy), PID step **114×**, hotspot detection **1.5×** (the benchmark
caught and we fixed a naive-port regression that was 0.3×), full PROGRAM cycle
**1.7×** (conservative — the Rust figure includes a simulated serial round-trip
the Python side skips, and neither captures the off-GIL threading win).
`test_rust_perf.py` guards against regressing below the Python baseline.

## Recommendation

The original hypothesis held all the way through: every real-time, hardware-
adjacent module ports to Rust faithfully and cheaply, validated against the
existing Python — including the loop itself, which closes against the simulator
from both Rust and Python. The genuine remaining cost is not Rust but the live
wiring + on-mount calibration, which is normal integration work and is cleanly
gated behind a flag so the current Python path stays the default until proven.

Net: the full Rust-core / Python-shell split is demonstrated end to end in
software. Promoting it to the live hardware path is a contained, flag-gated
integration step — not a rewrite risk.
