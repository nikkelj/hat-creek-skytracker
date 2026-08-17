# Validation

Independent cross-checks of the tracker's predictions against external
references. Newest entries first.

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
