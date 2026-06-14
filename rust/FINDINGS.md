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
- **`cargo test` stays Python-free.** pyo3 is an optional dependency gated behind
  the `extension-module` feature, so pure-Rust logic tests run without an
  interpreter; the cross-language tests skip cleanly if the wheel isn't built.

## Toolchain footprint (reversibility)

Installed for the experiment: VS Build Tools (C++ workload, MSVC 14.44), rustup
+ cargo 1.96, maturin 1.14. All cleanly removable (`rustup self uninstall`, VS
Build Tools uninstall) if the experiment is abandoned — and the whole thing
lives on a branch.

## Open questions / what's NOT yet proven

- **The numpy boundary (step 3).** `hotspot.py` is the first port that moves
  image data across the FFI (zero-copy `PyReadonlyArray`). Different dimension
  of risk than scalars.
- **The loop refactor (step 4) is the real cost.** Porting `mount_control.py`'s
  loop means converting `tracking_control()`'s shared-mutable-`state` access
  into explicit command/snapshot message-passing. That's design work, not
  translation — and it's where the effort actually lives.
- **skyfield stays Python.** No intent to port the astronomy; the Rust loop
  consumes setpoints Python computes. Coordinate-transform correctness remains a
  Python concern.

## Recommendation (interim)

The hypothesis is holding: the high-value, hard-real-time pieces port to Rust
faithfully and cheaply, validated by the existing suite. The decision to commit
to the full split should hinge on step 4 (the loop/message-passing refactor),
since that — not the language port — is the genuine investment. Steps 1–3 are
low-regret either way: they're independently useful, fully tested, and isolated
on a branch.
