# Validation

Independent cross-checks of the tracker's predictions against external
references. Newest entries first.

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
