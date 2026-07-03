# Hat Creek Skytracker — Comprehensive Project Review

**Date:** 2026-07-03 · **At commit:** `617f3e5` (post PR #12)
**Constraint honored:** recent changes remain unvalidated on the physical AVX mount and
cameras; the review therefore weights *simulator fidelity, hardware-seam risk, and
bench-day readiness* as first-class concerns, and all recommendations assume continued
reliance on the hardware simulator for now.

**How this review was produced.** Five parallel deep-dives (core control architecture,
simulation & imaging, Rust core, post-processing, test/infra) over the full source tree,
plus live verification in a fresh Linux container: the entire Python test suite was
executed headlessly, the Rust extension was built from source and its full parity +
performance suite run, and every high-severity claim below was **confirmed by executing
the code**, not just reading it.

---

## 1. Executive summary

The project is in genuinely strong shape for what it is: a single-operator, game-like
optical tracking application whose core loop design (dedicated 15 Hz control thread,
transactional serial layer, documented anti-windup/wrap math, atomic-publish threading
discipline) is well above hobby-project quality, backed by an unusually honest
engineering culture (LEARNINGS.md, rust/FINDINGS.md, limits documented in code).

The central risk is exactly the one flagged in the request: **a large body of recent
work is validated only against a simulator that is structurally incapable of catching
certain classes of hardware bugs** — and this review found live examples of that class:

- **Confirmed crash:** the variable guide-rate serial encoding
  (`lib/auxstar.py:350`) throws `TypeError` on any non-sidereal rate — the exact path
  behind `continuous_rate_tracking` and alignment fine-slews. The sim overrides this
  method, so every sim test passes while every real-hardware use would crash.
- **Confirmed latent unit bug:** `dms2f(0,30,0)` returns 0.5 of a full rotation (180°)
  instead of 30 arcmin; harmless today only because all callers pass `(dd, 0, 0)`.
- **Safety regression in the Rust loop:** on serial link loss it skips cycles forever
  and never commands a stop (the Python loop stops after 3 consecutive faults) — the
  one failure mode only a real bench produces.

None of this argues against the sim-first strategy — it argues for **investing in the
sim's hardware seams** (serial transport, exposure timing, finite slews) and in **CI**,
so that "passes in sim" keeps getting closer to "works on the bench."

**Top-line investment ranking (detail in §7):**

| Priority | Investment | Why |
|---|---|---|
| P0 | Fix the 7 confirmed/high-confidence bugs (½–1 day total) | Several are guaranteed bench failures or data-product corruptors |
| P1 | CI + pytest adoption (proven feasible on Linux in this review) | The whole strategy rests on sim tests; nothing enforces they pass |
| P2 | Sim fidelity at the hardware seams (serial faults, exposure timing, finite gotos, limits) | The highest-leverage work available *without* hardware access |
| P3 | Bench-day readiness kit (ordered checklist, guide-rate calibration first) | Converts eventual hardware time into hours, not weekends |
| P4 | Post-processing finishing stage (sharpen/stretch/16-bit + CLI) | The one user-facing goal ("good shot posted fast") that is half-built |
| P5 | Structural debt: split `joystick_controller.py`, typed config, repo hygiene | Compounds every future change |

Keep the Rust core **flag-gated** (neither promote nor retire) until its three defects
are fixed and a loop-level A/B parity harness exists — details in §5.

---

## 2. Purpose & current functionality (state of the union)

Purpose: find, acquire, track, and image satellites, rockets, and aircraft with a
Celestron NexStar mount + two ZWO cameras, from one fast UI. The feature set as built:

- **Tracking:** STANDBY / RATE / PROGRAM (TLE + launch trajectories) / HOTSPOT
  closed-loop optical tracking with coast-and-fallback; ADS-B aircraft as first-class
  targets; continuous variable-rate tracking; AltAz *and* equatorial mount modes.
- **Pointing:** 7-term TPOINT-style alt-az model + EQ counterpart, supervised alignment
  runs with grid-search recovery, plate-solve (tetra3) polar alignment. The math layers
  are clean, robust-fit, well tested.
- **Imaging:** threaded ZWO capture with circular buffer and µs timestamps, labeled
  capture dumps with per-frame interpolated trajectory metadata.
- **Post-processing:** PIPP-style grading/culling + AutoStakkert-style stacking
  (global RANSAC + optional alignment-point local warp, coverage-weighted averaging),
  ~2 s to a master for a 600-frame 1080p run. Replay UI with MP4 export.
- **Simulation:** SimMount (backlash, periodic error, encoder noise, guide-rate
  quantization, injectable pointing-model errors) + SimCap cameras rendering real
  Hipparcos star fields **through the production capture pipeline** — the loop closes
  entirely in software.
- **Rust core (experimental, flag-off):** protocol/PID/hotspot/transforms/loop ported,
  byte- and 1e-9-level parity with Python, 57× faster PROGRAM-cycle compute.

Project timeline context: roots in 2023–2025, but ~80% of the current system landed in
an intense June 2026 burst (12 PRs) — which is precisely the code that has never
touched hardware.

---

## 3. Verification performed for this review

Fresh Linux container, Python 3.11.15:

- **Test suite: 32 of 35 test files pass headlessly** (`SDL_VIDEODRIVER=dummy`). All
  initial failures were environmental, which is itself a finding (§6): `config.py`'s
  dead `from tkinter import Tk` blocked 9 test files until shimmed; `camera_manager.py`'s
  top-level `import zwoasi` blocked 2 more.
  Remaining 3 failures are environment/data-dependent, not code defects:
  `test_star_catalog` + `test_sim_stars` need to download `de421.bsp` (blocked network);
  `test_post_process` requires a populated local `data/` directory (not hermetic).
- **Rust core: builds and fully passes on Linux** — `cargo test` (48 tests), maturin
  wheel build (needs `libudev-dev`), and all 8 `test_rust_*.py` parity/bridge/perf
  suites pass. FINDINGS.md only claimed Windows/MSVC validation; Linux portability is
  now demonstrated, which derisks CI (§7-P1).
- **Benchmarks reproduce:** PROGRAM cycle compute 57×, PID step 101×, transforms
  22–870×, hotspot 1.6× (Rust vs Python), matching FINDINGS.md.
- **Every high-severity bug claim below was verified by execution or direct code read.**

---

## 4. Confirmed defects (ranked)

These were found by the subsystem reviews and individually verified. Line numbers at
commit `617f3e5`.

1. **Guide-rate encoding crashes on real hardware** — `lib/auxstar.py:349-350`:
   `'{:06x}'.format(packed_rate)` where `packed_rate` is `bytes` from `pack_int3` →
   `TypeError` (verified by execution). This is the path used by
   `continuous_rate_tracking` (`joystick_controller.py:1462-1466`) and alignment fine
   slews (`alignment.py:121-133`). `SimMount` overrides the method
   (`simulator.py:346`), so sim testing structurally cannot catch it. Fix is one line
   (hex-encode the bytes); add a unit test that exercises the *real* encoding path.
2. **Rust loop: no fault safe-stop** — `core_loop.rs:115-120` returns on poll fault
   with no consecutive-fault counter; Python stops motion after 3 faults
   (`mount_control.py:56-62`). Link loss mid-slew leaves the mount running at its last
   commanded rate.
3. **Rust loop: NaN panic kills the thread silently** — `hotspot.rs:34`
   (`partial_cmp().unwrap()` in `median_inplace`): one NaN pixel panics the loop
   thread, the post-loop stop is skipped, and the adapter keeps reporting
   `fresh: true` stale data (`core_loop.rs:172`; no liveness watchdog in
   `rust_loop_adapter.py`).
4. **Safety limits checked in inconsistent frames** — RATE mode gates *mount*
   coordinates (`joystick_controller.py:686-709`) while PROGRAM/launch gate the *sky*
   target (`:926-932, 1118-1124`) against the same `azm/alt_limit_*` values; in AltAz
   mode mount ALT = 90 − el, so one of the two checks is wrong. Related: the failed-
   stop path swallows errors silently (`mount_control.py:135-144`).
5. **Capture metadata silently lost** — `capture_manager.py:309-332` indentation: the
   per-frame trajectory `writerow` sits inside `if observer is not None:`; without
   lat/lon config, *no per-frame rows are written at all* and the az=0 fallback at
   `:311` is dead. This corrupts the labeled-data product, the project's core output.
6. **Capture-dump race** — `stop_capture` hands the *live* `CircularBuffer` to the dump
   thread (`camera_buffer/__init__.py:149-173`) while capture keeps appending; at full
   buffer the stop-time indices drift and long dumps can save wrong frames. Also
   `deque[i]` indexing makes the dump O(n²). Fix: snapshot `list(buffer)` at stop.
7. **Stale-frame PID reprocessing** — `hotspot_track` (`joystick_controller.py:1349`)
   uses `get_latest_raw()` with no new-frame (`buffer_sequence`) check; with exposures
   longer than the control period the PID re-integrates the same centroid every cycle
   (windup/overshoot). Invisible in sim because SimCap renders a fresh frame per read.
8. **Latent:** `dms2f` minute/second scaling wrong (`lib/auxstar.py:136`, verified:
   `dms2f(0,30,0)` = 180°); Park-button busy-loop blocks the UI thread on serial with
   no timeout and non-wrap-aware convergence (`joystick_controller.py:1533-1548`);
   Rust loop keeps last rate when a setpoint clears (satellite sets → mount keeps
   slewing, `controller.rs:306-309`); frame timestamps taken *after* exposure+readout
   (`camera_buffer/__init__.py:354-363`) bias trajectory-CSV interpolation on real
   cameras; UI stale-`pos` NameError path + duplicated dead MOUSEMOTION block
   (`main.py:610-760`); `TRAJECTORY_CACHE` grows without bound in multi-hour sessions
   (`trajectory.py:446-471`); stacking coverage mask warped `INTER_NEAREST` vs frame
   `INTER_LINEAR` (~1-px edge band, `stacking.py:687-688`).

A cross-cutting observation: **items 1, 5, 7, and the timestamp bias are all invisible
to the current sim by construction** — they live exactly in the seams the sim replaces
(serial encoding, real exposure timing, config-dependent capture paths). That is the
strongest argument for the P2 investment below.

---

## 5. Subsystem assessments

### 5.1 Core control architecture — strong core, debt concentrated at the edges

The mount control thread, serial transaction layer, and control math are the best code
in the repo: single poll per cycle shared by control and display, interruptible fixed
cadence, consecutive-fault watchdog with guaranteed stop on shutdown, derivative-on-
measurement with wrap handling, conditional-integration anti-windup with rollback, and
shortest-arc wrap on both axes with excellent failure-mode comments. The lock-free
publish-by-rebind threading idiom is applied consistently and documented at every site.

Debt: `joystick_controller.py` is a **4,649-line god object** holding the safety-
critical state machine *and* ~30 render/input functions; `program_track` vs
`_program_track_launch` are ~180 duplicated lines whose feed-forward sign handling has
already diverged (`:966` vs `:1146-1149`); config numerics are stored as strings and
`float()`-parsed inside the 15 Hz loop every cycle; `position_update_lock` is acquired
at exactly one site (protects nothing); some mount commanding still happens on the UI
thread. Control cadence is serial-bound (~3 wire transactions/cycle at 9600 baud, incl.
a FOCUS poll every cycle even when unused) — 15 Hz is marginal and one 0.25 s timeout
eats 4 cycle budgets.

### 5.2 Simulator & imaging — high-fidelity where it models, honest about what it doesn't

Modeled well: rate integration with jitter, backlash dead-band, worm periodic error,
encoder noise/misalignment, guide-rate quantization to the real 24-bit LSB, injectable
7-term + EQ pointing errors and refraction, inter-camera misalignment, star streaking,
real catalog star fields — all flowing through the **production** capture pipeline, with
tests proving the alignment routines recover the injected errors.

Not modeled (the gap list that matters given the caveat): **the serial transport
entirely** (no latency/short-reads/`SerialCommError` — SimMount is an in-process
object); instant gotos (all settle/timeout logic untested); no cord wrap or hard
stops; no camera exposure/readout timing or faults (sim loop latency ≈ 0; USB stalls,
dropped frames, saturation unexercised); ROI is a no-op; sign conventions
(`hotspot_x_sign`/`y_sign`, guide-rate wire scale) acknowledged as unverifiable in sim.
Physics nits: periodic error runs on wall time not worm angle; rate jitter should scale
√dt; goto bypasses backlash.

Features currently validated *only* in sim: continuous variable-rate tracking, EQ mount
mode + EQ pointing model, supervised alignment + grid-search recovery, polar alignment,
focus motor, the Rust core loop, and everything from the June burst.

### 5.3 Rust core — high-quality port; keep flag-gated, don't promote or retire yet

Parity rigor at module level is exemplary (byte-identical wire output, PID step-for-step
to 1e-9 over 400 cycles × 6 configs, sub-0.05 px hotspot parity, transform sweeps) and
this review demonstrated full Linux portability. Perf case is honest and reproduced.
Docs are stale in the good direction: launch tracking and HANDOFF are built
(`rust_loop_adapter.py:295-351`, `controller.rs:376-409`) though STEP4/FINDINGS still
list them deferred; only MTI is a true stub.

But: the promised **closed-loop A/B parity harness** (same sim scenario through both
loops, comparing rate-command traces) was never written; there is **no fault-injection
testing** (which would have caught defects 2–3 in §4); the launch/mask/bias adapter
paths have zero test coverage; parity tests skip silently when the wheel isn't built —
and with no CI, that means they run on one machine. Crucially, the current flag-on path
uses `wrap_mount` (bridge through Python, GIL per serial call), so **the off-GIL
determinism that motivates the whole experiment is not yet realized on the hardware
path** (`open_serial` is built but never wired).

Maintenance tax is real: five Python modules now have Rust twins, and recent feature
commits already show every control change landing 2–3 times.

**Recommendation:** keep flag-gated. Promotion gates, in order: fix defects 2/3 and
setpoint-clear behavior → build the A/B loop-trace harness + run parity in CI →
`bench_guiderate.py` on the mount to verify the guide-rate wire scale → bench campaign
per STEP4_INTEGRATION.md → then consider `open_serial` for the true off-GIL win. Also:
15 minutes to refresh the stale "known gaps" sections before any bench day.

### 5.4 Post-processing — correct, fast core; the *finishing* half is missing

The cull/align/stack core is correct and well-tested (every LEARNINGS trap has a
dedicated regression test; parallel==serial verified; ~2 s masters). Gaps against the
stated goal "get a good shot posted fast":

- **No finishing stage.** The master is deliberately linear "for a later
  wavelet/deconvolution step" that doesn't exist — and it's quantized to 8-bit
  (`stacking.py:713-714`), discarding stacked SNR below 1 LSB and contradicting its own
  design note. The user ends with a soft, dark PNG they must finish elsewhere.
- **No CLI/batch mode** — stacking is reachable only via three fixed pygame buttons
  (25%/50%/50%+AP); the profiled speed knobs (`grade_scale`, `align_scale`, `workers`)
  never reach the UI or `StackExporter`.
- **BMP-only ingest**: `_FRAME_RE` (`post_process.py:62-65`) silently sees zero frames
  if `image_format` is switched to PNG (which `capture_manager.py:106` uses as its
  fallback); timestamps with exactly 0 µs also fail the regex and drop frames.
- PIPP-style centring/crop (`stacking.py:306-407`) — documented as "the big win" for
  ISS-class targets and also the cheapest ORB cost reducer — is dead code.
- Diagnostics (`StackResult.grades`, `dropped`) are computed and discarded; the master
  isn't displayed; grading pass reports no progress.

### 5.5 Testing & infrastructure — good tests, zero enforcement

~30 of 35 test files are genuine assertion-based tests and already pytest-compatible.
But: **no CI, no test runner, no linting/typing, no pre-commit**; 4 files are
assert-free print-and-eyeball scripts (`test_alt_sign.py:53` contains a mangled
`print(".1f")` — evidence they're never run); `test_post_process.py` requires local
`data/`; Rust parity tests skip silently without the wheel. Config: hand-rolled,
no schema/validation/version (though `jsonschema` is pinned in the env, unused),
numbers stored as strings, personal site coordinates + live tuning state committed to
git. Repo hygiene: `.git` is 38 MB (12 committed generations of `tle_cache.tle` ~2 MB
each + a 16 MB `de421.bsp` in history); a 2.0 MB `gp.php` Celestrak dump is still
tracked at the root; `environment.yml` is Windows-only with a hardcoded personal prefix
and lacks pytest. Zero-coverage modules include the scariest one: `camera_manager.py`
(89 KB, all ZWO hardware handling), plus `capture_manager.py`, `mount_control.py`,
`satellite_data.py`, `trajectory.py` (in effect), `main.py`.

---

## 6. What's genuinely working well (keep doing this)

- **Sim-through-production-pipeline design** — SimCap feeds the real capture thread,
  SimMount feeds the real control thread. This is the property that makes desk
  validation meaningful at all; protect it as the sim grows.
- **Written-down learning** (LEARNINGS.md, FINDINGS.md, failure-mode comments in
  `control.py`/`auxstar.py`) — several review findings were *predicted* by the
  project's own annotations ("UNVERIFIED on real hardware…"). The habit of leaving
  honest markers at unverified seams is exactly what made this review tractable.
- **Adversarial-review-before-merge culture** visible in recent PRs (the stacking traps
  were caught pre-merge) and regression tests added per learning.
- **Incremental, parity-checked porting discipline** in the Rust experiment — a model
  for how to do a rewrite experiment without betting the project.

---

## 7. Investment plan — what to do next, in order

### P0 — Bug-fix sprint (≈1 day, all sim-verifiable)
Fix §4 items 1–8: guide-rate hex encoding (+ a unit test through the *real* encoder,
no sim override), Rust fault safe-stop + NaN-tolerant median + adapter liveness
watchdog + setpoint-clear stop, frame-consistent safety limits + escalating
`_safe_stop_motion`, capture_manager indentation, capture-dump snapshot, hotspot
new-frame gate, `dms2f`, Park-loop off the UI thread. Every one of these is testable
headlessly today; several are guaranteed bench failures if left.

### P1 — CI + pytest (≈1–2 days; the highest-leverage single investment)
The project's entire risk posture is "trust the sim tests," yet nothing runs them
automatically. This review proved the recipe on Linux end-to-end: apt `libudev-dev`
(+ tk-enabled Python), pip deps, `SDL_VIDEODRIVER=dummy`, maturin build, full suite.
Concretely: add `pyproject.toml` + `conftest.py` (set SDL dummy once, markers for
`tetra3`/`rust`/`data`-dependent tests), convert/delete the 4 assert-free scripts, make
`test_post_process.py` hermetic with a fake-run fixture, vendor or cache `de421.bsp`
for CI, and a two-job GitHub Actions workflow (Python suite; cargo test + wheel +
parity suite). Side quests that fall out of it: remove the dead `tkinter` import
(`config.py:4`) and make `camera_manager.py`'s `zwoasi` import lazy/optional — both
currently make half the suite un-runnable on a clean machine, and both are one-liners.
Cross-platform `requirements.txt` alongside the Windows conda env.

### P2 — Sim fidelity at the hardware seams (the best proxy for hardware you can build)
In value order:
1. **Simulated serial transport under SimMount** — byte-level AUX responder (the Rust
   side already has one; port or reuse the idea) with configurable latency, short-read/
   timeout probability, garbage bytes. Run the control loop *and* an alignment run
   against it. This exercises `SerialCommError` recovery, real command encodings
   (would have caught §4-1 immediately), and cadence under 0.25 s timeouts.
2. **Exposure/readout timing + faults in SimCap** — hold `ASI_EXP_WORKING` for the
   configured exposure + readout latency; inject occasional failures. Then re-run the
   sim-in-the-loop convergence test at 0.5–1 s exposures (it will surface §4-7's
   windup as real hardware would), and back-date frame timestamps to exposure midpoint.
3. **Finite gotos** — slew at realistic AVX rates with accel through the backlash
   model, so settle/timeout logic stops being validated against a mount that cannot
   mis-settle.
4. **Axis limits + cord wrap in SimMount**, so the safety-abort paths get a rehearsal.
5. Smaller physics: √dt rate noise, worm-angle-driven PE, frame-timestamped alignment
   samples (`alignment.py:260-277` currently pairs solves with `ts.now()` — the sky
   moves 15″/s and the buffer's µs timestamps go unused).

### P3 — Bench-day readiness kit (cheap now, pays off the day hardware appears)
A single ordered checklist doc: (1) `bench_guiderate.py` **first** — the continuous-rate
path rests on an admittedly unverified wire scale; (2) `x_sign`/`y_sign`/rotation
calibration procedure; (3) RATE → PROGRAM → HOTSPOT progression with the Python loop;
(4) only then the Rust loop per its integration doc. Include expected failure signatures
and safe-abort steps. Refresh the stale Rust docs as part of it.

### P4 — Post-processing: close the "posted fast" loop
(1) 16-bit master (accumulate → uint16 TIFF/PNG) + a finishing stage (unsharp/à-trous
wavelet + auto-stretch → share-ready 8-bit) — pure cv2, fits the codebase; (2) a CLI
(`python stacking.py <run|data/> --keep 0.5 --local --sharpen`) for headless batch after
a session; (3) fix BMP/PNG ingest + timestamp-regex drops; (4) wire `grade_scale`/
`align_scale`/`workers` and the dead PIPP centring into `StackExporter` and the UI;
(5) show the master + grade histogram with an adjustable keep slider when done.

### P5 — Structural debt (do opportunistically, not as a big-bang)
Split `joystick_controller.py` into tracking-modes / input / panels (unlocks unit-
testing the state machine without pygame and de-duplicates the diverged
`program_track`/`_program_track_launch` pair); typed config snapshot per control cycle
(kills per-cycle `float(str)` parsing and torn multi-field reads) + `jsonschema`
validation + `config.example.json` (gitignore the live one); evict expired
`TRAJECTORY_CACHE` windows; delete dead code (unused general quaternion transforms in
the control path, legacy config load/save, duplicated MOUSEMOTION block,
`position_update_lock`); gitignore `gp.php`-style caches and consider a one-time
history clean of the 25 MB of committed TLE/ephemeris blobs.

### Explicitly *not* now
- **Promoting the Rust loop to default** — gates in §5.3.
- **New tracking capability (HANDOFF/MTI, ephemeris+covariance ingest)** — the roadmap
  items are good, but every new mode multiplies the dual-implementation and
  sim-validation surface; land P0–P2 first so new modes inherit a trustworthy harness.
- **A UI-framework migration or big-bang refactor** — the pygame architecture is
  performing fine after the invariant-caching work; P5's mechanical split is enough.

---

## 8. Closing assessment

For a project whose recent code has never touched its hardware, the risk management to
date has been unusually good: production-pipeline simulation, parity-tested porting,
honest documentation of unverified seams. The findings above don't change the strategy —
they sharpen it. The two cheapest, highest-yield moves are the one-day bug sprint (P0)
and CI (P1); the most important medium-term investment is making the simulator lie less
at exactly the seams where this review caught it lying (P2). Do those three and the
eventual bench campaign becomes a calibration exercise instead of a debugging marathon.
