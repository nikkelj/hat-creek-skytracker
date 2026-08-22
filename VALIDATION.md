# Validation

Independent cross-checks of the tracker's predictions against external
references. Newest entries first.

---

## 2026-08-18 — Rust PID auto-tuner identical to Python (Phase 6b)

**What it is.** `autotune.rs` in skytracker-core ports PIDAutoTuner: the
shared-schedule P/D/I twiddle in log-gain space (settle/eval windows,
>3% acceptance margin, step expand/shrink, live divergence gate,
pause/resume on tracking loss, sweep convergence). Exposed as
RustPIDAutoTuner; the PIDAutoTuner-compatible Python wrapper mirrors the
applied gains into the live config fields both loops read, so
joystick_controller / joystick_panels are untouched. Selected by
`make_autotuner` behind `use_rust_autotune` / SKYTRACKER_RUST_AUTOTUNE=1.

**Validation** (test_rust_autotune_parity.py, in CI): both tuners drive
identical synthetic plants (error depends on the gains each applied, so
decision divergence cascades) at 15 Hz — **5,230 cycles identical**
through full convergence (9 sweeps; phase, stage, every applied gain to
1e-9, every status message); 3,600 cycles identical through a 6 s
tracking dropout + divergence spike; stop(revert=True) identical and
restores initial gains.

Alignment-run sequencing (AlignmentRunner) is thread/hardware-coupled
and moves with the Phase 7 app's state redesign; its pure math
(Fibonacci grid, fits) is already ported.

---

## 2026-08-18 — Rust-loop gaps closed: mask-exit + rate gearbox (Phase 6a)

**Mask-exit pre-positioning**: the flag-on Rust loop's documented gap
(below-horizon satellite → safe stop) is closed: the adapter now holds
the setpoint at the mask-exit point (target azimuth at the elevation
mask, zero rates), mirroring the Python program_track drive and the
launch path's mask hold. Loop/adapter/A-B suites green (32 tests).

**Adaptive rate gearbox**: `axis_to_rate` + `AdaptiveRateMapper` ported
to skytracker-core (rate.rs) with the exact constants and absolute-
deadline semantics; exposed as RustAdaptiveRateMapper.
test_rust_rate_parity.py (in CI): 16,004 axis samples identical, and 20
random deterministic stick timelines (pins, releases, reversals, stale
gaps at 15 Hz service) match the Python gearbox on EVERY step and
ceiling. One porting catch: a guessed JOY_RELEASE_DEFLECTION (0.5 vs
the real 0.70) was flushed out immediately by the timeline A/B.

MTI remains a stub on both sides by design; alignment-run sequencing +
autotune port in Phase 6b; Rust-loop promotion to default awaits the rig.

---

## 2026-08-18 — Rust camera pipeline sustains camera-native rates (Phase 4a)

**What it is.** `skytracker-camera`: the capture pipeline rebuilt in
Rust — frame pump thread, ring buffer with exposure-midpoint stamping
(camera_buffer semantics), armed-capture retention + rayon-parallel BMP
dump, and the ZWO ASI SDK binding (ASICamera2.dll via libloading,
rig-ready but timing-truth pending hardware). Python touches frames only
on display pull. Motivation: the Python capture path (CameraThread +
pygame conversion + CircularBuffer, all under the GIL) capped at
**4-10 FPS** against cameras capable of 50-100.

**Validation** (test_rust_camera_pipeline.py, in CI; frames rendered by
the real HardwareSimulator per the sim-first directive, replayed
metered):

- **100.2 FPS sustained** at the 100 FPS target and **50.3** at 50, zero
  dropped frames, while a concurrent 30 Hz display consumer pulls.
- Unthrottled pipeline headroom **2,244 FPS** (VGA mono) vs 992 FPS for
  the isolated Python per-frame path (2.3×) — and the Rust path holds
  under load since no stage touches the GIL, where the Python 992
  degrades to the observed 4-10 in the live app.
- Armed capture: 40 sim frames dumped as byte-correct BMPs with
  monotonic stamps; midpoint backdating measured at exactly 250 ms for
  a 0.5 s exposure (parity with exposure_midpoint_utc).

**Phase 4b (wiring)**: `RustCameraThread` (rust_camera_adapter.py)
subclasses CameraThread and reroutes the per-frame path: raw frames go to
the Rust ring, the display Surface is built lazily at UI-pull rate
instead of per frame, and armed capture keeps the exact frame-dict
contract capture_manager consumes. Selected in
camera_manager._start_camera_thread behind `use_rust_camera` /
SKYTRACKER_RUST_CAMERA=1 (Python CameraThread on any failure). Verified:
camera_manager sim connect runs the Rust thread at the paced sim target
(15.4 FPS actual vs 15 target), display + detection + capture round-trip
correct, and the full-wiring closed-loop tracking gate holds on the Rust
camera (**PROGRAM rms 74.4″** vs the 67-73″ Python baseline band).
Remaining: rig-session timing truth + native open_asi promotion.

---

## 2026-08-18 — Rust ADS-B decode identical to pyModeS (Phase 5)

**What it is.** `skytracker-adsb` ports the Mode-S DF17/18 decode subset
the app uses (CRC-24, ICAO, typecode, TC 1-4 callsign, TC 9-18/20-22
CPR position-with-reference + altitude, TC 19 velocity — faithful to
pyModeS's algorithms including its integer-truncation quirks) plus the
WGS84 geodetic→ECEF→ENU→az/el chain. `decode_adsb_message` routes
through it behind `use_rust_adsb` / SKYTRACKER_RUST_ADSB=1 with pyModeS
as fallback. Deliberate divergence: Gillham (Q=0) altitudes above
50,187 ft decode to None. SBS source + tracker orchestration stay
Python; native RTL demod arrives with the Phase 7 app.

**Validation** (test_rust_adsb_parity.py, in CI): a generated corpus of
242 CRC-valid DF17 frames (random ident/position/velocity payloads,
CRCs from pyModeS's own encoder) decodes **identically** in every field
(ident 81, position 81, velocity 80); geometry matches numpy to 7e-15;
the classic pyModeS reference messages decode to the documented values
in pure-Rust unit tests. Existing test_adsb.py suite green.

---

## 2026-08-17 — Rust imaging kernels live in the pipelines (Phase 3b)

**What it is.** The stacking/stabilizer/sharpen numeric kernels now route
through `skytracker-imaging` behind `use_rust_imaging` /
SKYTRACKER_RUST_IMAGING=1 (cv2/numpy fallback): sharpness grading
(laplacian/tenengrad incl. INTER_AREA downscale with uint8 rounding
semantics), brightness centroid, patchwise local shifts + dense grid
warp, the flow-method stabilizer estimate (detect + LK + RANSAC +
plausibility bounds in one Rust call), and the mono finish()
(multi-scale unsharp + auto-stretch). Orchestration and color paths stay
Python.

**Validation** (test_rust_imaging_parity.py, in CI, real entry points on
a synthetic jittered burst): end-to-end LuckyStacker master (flow align
+ local grid + accumulate) **51.2 dB PSNR** flag-off vs flag-on (gate
40); finish() **identical**; sharpness <1e-3 rel; centroid exact;
stabilizer transforms 0.0055 px / 0.0012°; grid-warp 51.1 dB, shift-field
42.2 dB. Per-node local shifts on noisy rotated patches differ at the
estimators' shared noise floor (median 0.37 px) — the clean-signal
phase-correlation case is golden-gated at 0.01 px in cargo.

**Phase 3c (MP4 export)**: post_process's Mp4Exporter can now write
H.264 via openh264 + a pure-Rust MP4 muxer behind the same flag
(cv2.VideoWriter mp4v fallback; odd dimensions fall back). Round-trip
gate is comparative: the H.264 output must decode no worse than the
mp4v writer it replaces — measured **37.2 dB vs mp4v's 35.1 dB** on the
noisy synthetic burst, all frames cv2-decodable. (Two traps documented
in LEARNINGS: limited-range YUV, and openh264's per-GOP QP startup.)

---

## 2026-08-17 — Rust imaging primitives vs cv2 goldens (Phase 3a)

**What it is.** `skytracker-imaging`: the OpenCV numeric cluster the
stacking/stabilization pipeline composes — filters (Gaussian/Laplacian/
Sobel, reflect-101 borders), warpAffine/remap (incl. cv2's 1/32-px
fixed-point coordinate quantization), rustfft phase correlation with
OpenCV's fftShift/minMaxLoc/weighted-centroid subpixel chain, Shi-Tomasi
corners + pyramidal Lucas-Kanade (win 21, 3 levels, Scharr gradients),
and RANSAC similarity with exact-LSQ refine.

**Validation** (cargo test -p skytracker-imaging vs regenerated
cv2_ops.npz): filters ≤ 9e-5; warp 4.6e-5; LK 0.0055 px; RANSAC
**identical** to cv2 (rotation/translation/scale 0.000, same 54 inliers);
phase correlation 0.01 px with a mirror-tolerant gate at exact
half-pixel shifts (cv2's f32 pipeline and our f64 pipeline break the
two-row peak tie oppositely; the test also asserts we are never less
accurate than cv2 against truth).

**Golden integrity bug found and fixed**: cv2.phaseCorrelate mutates its
inputs in place; the original cv2_ops.npz was internally inconsistent
(see LEARNINGS 2026-08-17). Recorder fixed, goldens regenerated, and a
defensive copy added to stacking.py's live phase-correlate call.

---

## 2026-08-16 — Rust pointing-model fits at machine precision

**What it is.** Phase 2b: `skytracker-pointing` ports the 7-term TPOINT
fits (alt-az IA/IE/AN/AW/NPAE/CA/TF and equatorial IH/ID/NP/CH/ME/MA/TF
with parallactic-angle flexure), including the partial (seeded) nightly
refit, MAD-robust outlier rejection, Bennett refraction stripping, and
the polar-axis SVD plane fit. Least squares uses numpy's
`lstsq(rcond=None)` semantics (SVD minimum-norm with the same cutoff).
Fit classmethods route via `use_rust_pointing` / SKYTRACKER_RUST_POINTING
with the numpy implementations as fallback.

**Validation** (test_rust_pointing_parity.py, in CI; synthesized truth
models + noisy samples + planted outliers through the real classmethod
entry points): worst coefficient difference **4.4e-16 deg** (gate 1e-6),
identical robust rejection counts (3/3), identical n_samples/design_cond/
RMS stats, polar-axis fit exact to 2.8e-14 deg.

---

## 2026-08-16 — Rust plate solver: numerically identical to tetra3

**What it is.** Phase 2a of the Rust port: `skytracker-platesolve` is a
pure-Rust line-by-line port of ESA tetra3's lost-in-space solver
(centroider + edge-ratio pattern hash + Kabsch verify), reading the app's
existing `.npz` pattern databases unchanged. Wired into
`plate_solver.PlateSolver` behind `use_rust_platesolve` /
`SKYTRACKER_RUST_PLATESOLVE=1` with tetra3 as the automatic fallback.

**Validation:**

- **Pattern hash**: 1,000 golden keys **bit-exact** on the live
  db_cam1_tyc geometry (bins=50, catalog=2,351,182) and an alternate
  geometry — the existing databases work as-is (numpy uint64 wraparound
  reproduced with Rust wrapping arithmetic).
- **Centroids**: within **0.0000 px** of Python tetra3 on golden frames
  and freshly rendered fields (gate 0.3 px).
- **End-to-end** (test_rust_platesolve_parity.py, in CI): 12/12 synthetic
  star fields (stars projected from the DB's own catalog at random
  pointings) solved by both solvers with **0.00 arcsec** centre
  difference, **0.0000°** roll difference, and identical match counts —
  the same accepted pattern, the same refined rotation, to f64 precision.
- Wrapper flag-on path verified (identical SolveResult); solve speed
  comparable (~260 ms first-call, dominated by DB load on both sides).
- Existing test_plate_solve.py / test_alignment.py suites green.

---

## 2026-08-16 — Rust astro engine wired into the app (flag-gated) at 76×

**What it is.** Phase 1 integration: the `AstroEngine` PyO3 class exposes
the Rust astro engine, and `trajectory.py` / `celestial.py` route through
it behind `use_rust_astro` (or env `SKYTRACKER_RUST_ASTRO=1`), skyfield
remaining the automatic fallback on any error.

**Live A/B parity** (test_rust_astro_parity.py, real tle_cache + app entry
points, both implementations side by side):

- `_compute_one_trajectory` rows: 40 sats × 31 samples, worst **0.041″**
  (gate 20″); px/py sub-pixel; rates within 2e-3 °/s.
- Visibility gate: 250 sats, **zero** disagreements (boundary cases
  verified within 0.2° of the threshold when they occur).
- `solar_system_altaz` + `build_trajectory` (moon, planet:Jupiter, star
  anchor): worst **0.03″** — with an explicit guard that the Rust path
  actually engaged (see LEARNINGS: silent fallback made an early version
  of this test compare skyfield to itself).

**Measured speedup** (bench_rust_vs_python.py, 16,085-sat catalog,
31 samples): visibility gate 176.6 → 57.3 ms (**3.1×**); bulk trajectory
rows for the 970 visible satellites 285.7 → **3.8 ms (75.9×)** — one FFI
call, rayon-parallel, GIL released. TLE catalog parse: 16,085 sats in
40 ms.

---

## 2026-08-15 — Rust astro engine: bodies, stars, and a pure-Rust SPK reader

**What it is.** Completion of the Phase 1 engine math: `skytracker-astro`
now covers solar-system bodies (sun/moon/planets from `de421.bsp`) and
Hipparcos star apparent places, replacing skyfield's observe/apparent chain.

- **SPK reader** (`spk.rs`): minimal pure-Rust DAF/Type-2 Chebyshev reader
  (~250 LOC, no external deps) instead of a heavy ephemeris crate.
  Validated against jplephem on all 15 de421 segments × 25 epochs:
  worst position diff **1 cm**, velocity 1e-11 km/s.
- **Bodies** (`ephemeris.rs` + `apparent.rs`): light-time iteration +
  NOVAS aberration (ported from skyfield.relativity) + IAU2000A frame
  chain. Worst alt/az and RA/Dec separation vs skyfield: **0.79 arcsec**
  (gate 60"); the omitted relativistic light deflection is the residual.
- **Stars** (`stars.rs`): hip_main.dat parser (117,955 rows) + the
  starlib position/proper-motion/parallax model. Worst separation
  **0.17 arcsec** across 50 stars (incl. high-proper-motion set) × 10
  epochs over 2024–2027.

---

## 2026-08-15 — Rust astro engine: satellite/time parity vs skyfield

**What it is.** Phase 1 of the Rust port: `rust/skytracker-astro` replaces
the skyfield satellite pipeline. Validated against the Phase 0 golden
vectors (pure-Rust `cargo test -p skytracker-astro`):

- **Satellite topocentric alt/az/range**: worst sky separation
  **0.03 arcsec** over 6 TLEs (ISS→polar→Molniya-class) × 200 epochs;
  relative range error ≤ 1.1e-7 (gate: 20 arcsec).
- **GAST/GMST/ΔT**: worst 1.9 ms of time (gate: 5 ms), using the full
  IAU2000A nutation series generated verbatim from skyfield's tables.
- **Az/el rates**: worst sky-projected difference 8.5e-8 deg/s against the
  golden finite-difference scheme.

Two convention traps found and documented in LEARNINGS.md: the Rust sgp4
crate defaults to WGS84 (skyfield uses WGS72 — use the AFSPC-compat
constructor), and skyfield runs SGP4 on UTC while rotating TEME→PEF with
UT1.

---

## 2026-08-15 — Rust port Phase 0: golden vectors + closed-loop baseline

**What it is.** The full-Rust port (branch `rust-port`) validates every
future phase against frozen reference outputs of the *current* Python
implementations, recorded before any port work touches them.

**What was recorded** (`tools/record_golden.py` → `tests/golden/`, committed):

- **skyfield**: 6 TLEs spanning inclinations (ISS 51.6° → polar) × 200
  epochs → topocentric alt/az/range/rates; sun/moon/5 planets and 50
  Hipparcos stars (40 brightest + 10 highest-PM) at 2024–2027 epochs;
  GAST/GMST/ΔT series. Tolerances for the Rust astro engine: sats 20″,
  bodies/stars 60″, GAST 5 ms.
- **tetra3**: 1,000 pattern-hash keys → `_key_to_index` outputs against the
  live `db_cam1_tyc` geometry (bins=50, catalog=2,351,182) plus an alternate
  geometry — the Rust solver must match **bit-exact** or the existing
  database is unusable. Centroids on 5 synthetic star fields (0.3 px gate).
- **OpenCV**: phase-correlate on known sub-pixel shifts, Gaussian/Laplacian/
  Sobel kernels, RANSAC similarity estimation with planted outliers, and
  pyramidal LK on a known shift — the numeric contract for the Phase 3
  imaging port.
- **Closed-loop baseline** (`tools/record_loop_baseline.py` →
  `tests/golden/loop_baseline.json`): full-wiring sim rig
  (test_tracking_quality.py). PROGRAM rms ≈ 67–73″, HOTSPOT hold
  rms ≈ 138″, HANDOFF latency ≈ 1.0 s, zero false rejects/losses.
  Phases 1/4/6/7 must not regress these.

**Structural change validated:** `rust/` is now a Cargo workspace —
`skytracker-core` (pure Rust) + `skytracker-ffi` (the only pyo3 crate;
Python module name unchanged). All 49 cross-language parity tests pass
against the relocated wheel; `cargo test --workspace` links no libpython.

---

## 2026-07-28 — Pass-table visual magnitude: model basis and validity bounds

**What it is.** The pass table's **Mag** column is an *estimate* of apparent
visual magnitude at culmination: the McCants standard-magnitude formula
`m = stdmag − 15.75 + 2.5·log10(range_km² / frac_illum)` with a Lambertian
phase term `frac_illum = (1 + cos φ)/2` (φ = sun–satellite–observer angle),
plus a cylindrical Earth-shadow test — eclipsed passes show `ecl` instead of
a number. Implemented in `trajectory._estimate_pass_magnitude`.

**What was validated.**

- The eclipse (shadow-cylinder) test was checked against Skyfield's
  `is_sunlit` at 31 samples over a full ISS orbit: agreement at every sample.
- Sanity of the formula: at the 1000 km / 50%-illuminated reference geometry
  the output equals `stdmag` exactly (the convention's definition).
- Physical plausibility in the live table: near local midnight only high-LEO
  constellations (OneWeb ~1200 km, Globalstar ~1400 km) carry numeric
  magnitudes (~6.0–6.5) while lower Starlink shells read `ecl` — matching the
  expected shadow geometry.

**Known limitation (by design).** TLEs carry no per-satellite brightness, so
every satellite uses one default intrinsic magnitude (`stdmag = 6.0`, roughly
Starlink-class). Relative rankings (which pass is brighter, sorting) are
driven by range/phase/shadow and are meaningful; absolute values are not
calibrated per satellite — the real ISS (stdmag ≈ −0.5 to −1.8) is several
magnitudes brighter than this estimate reports. Do not compare the column's
absolute numbers against heavens-above brightness predictions, which use
per-satellite intrinsic magnitudes.

---

## 2026-07-27 — Pass predictions vs heavens-above.com (4 satellites, 4 orbit classes)

**Question.** Do the app's TLE → SGP4 → topocentric pass predictions agree with
an independent reference, and is there any orbit-class-dependent bias?

**Method.** Passes computed with the repo's own data path — `tle_cache.tle`
(refreshed via `satellite_data.download_tle_data` immediately before the test),
the observer from `config.json` (lat 34.8740289, lon −120.4461237, alt 100 m),
Skyfield `find_events` at the 10° threshold heavens-above uses — over the next
5 days, in UTC. Reference: `heavens-above.com/PassSummary.aspx` with the same
coordinates (`satid=<norad>&lat=34.8740289&lng=-120.4461237&alt=100&tz=UCT`),
which lists visible passes only. Visibility classification for comparison:
satellite sunlit and observer sun altitude < −6°.

**Satellites chosen to span orbit classes:**

| Satellite | NORAD | Inclination | Altitude | Notes |
|---|---|---|---|---|
| STARLINK-31495 | 59313 | 43.0° | ~480 km | high drag (decaying) |
| ISS (ZARYA) | 25544 | 51.6° | ~420 km | large, frequent maneuvers |
| HST | 20580 | 28.5° | ~530 km | low inclination |
| TERRA | 25994 | 98.2° | ~700 km | sun-synchronous, stable |

**Results (overlapping visible passes, culmination time / max elevation):**

| Satellite | Passes compared | Time agreement | Elevation agreement |
|---|---|---|---|
| STARLINK-31495 | 8 | within 0–3 s | within 1° |
| ISS | 2 of 2 (both flagged by both) | exact / 1 s | exact |
| HST | 6 | within 1 s | within 2° (shadow-truncation cases) |
| TERRA | 7 | within 0–1 s | exact, incl. an 89° zenith pass |

Rise/set azimuth sectors matched in every case. Every pass we classified
eclipsed or daylight was correctly absent from heavens-above's visible list.

**Verdict: agreement, no bias.** The spread of inclinations, altitudes, and
drag regimes rules out the systematic errors that would matter — a
latitude/longitude or units slip, a UTC/TT or timezone offset (would appear
as a constant ~32 s or minutes-scale shift), an altitude-datum error, or an
SGP4/epoch-handling difference.

**Known, benign differences:**

- heavens-above truncates its listed segment at Earth-shadow entry/exit; we
  report the full geometric >10° pass. Its "highest point" is the highest
  *visible* point, so it can read lower than our geometric culmination (e.g.
  a Starlink pass: 69° visible max vs 87° geometric, times still matching).
  Same physics, different reporting convention.
- Passes with the sun near −5° to −6° at culmination sit on the visibility
  threshold and can be classified either way (heavens-above included one such
  pass we called twilight, and excluded another); pass *times* still matched.

**Operational caveat.** TLE freshness dominates real-world accuracy for low,
high-drag satellites (Starlinks): with the cached TLE ~1.6 days old, pass
times can drift by tens of seconds to minutes. The cache was refreshed for
this test; refresh before a tracking session if pass timing looks off.

Repro: `scratchpad` script pattern — load `tle_cache.tle` + `config.json`
observer, `sat.find_events(topos, t0, t0+5d, altitude_degrees=10)`, classify
with `sat.at(t).is_sunlit(eph)` and observer sun altitude, print UTC; compare
against the PassSummary URL above with the matching NORAD id.
