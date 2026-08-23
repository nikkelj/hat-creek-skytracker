# Phase 8 readiness — what the native app covers, what the rig must confirm

Phase 8 (retiring the Python/pygame app) is gated on a hardware session and
sign-off. This is the inventory to walk through on the rig, written as of
2026-08-23 (branch `rust-port`, commit f9171c2).

## Feature coverage: Python app → `skytracker-app`

| Python | Native | Notes |
|---|---|---|
| Tracking Vis skyplot (stars, sats, planets, named stars, Messier/NGC, keepout wash, ADS-B, launch trajectories, pass table) | Track + Passes | pass predictor validated to 0.00 s / 0.0003 mag vs skyfield |
| PROGRAM / RATE / HANDOFF / HOTSPOT / STANDBY, STOP, park, operator bias, PID mode profiles, autotune, feed-forward toggle | Track (mount panel) + gamepad | controller geometry generalized for AltAz-Side / Eq (LEARNINGS 2026-08-22) |
| Gamepad map (Cross capture, Circle stop, Triangle park, Share bias mode, Options mount-mode toggle, R3/pad mode cycle, L1 FF, D-pad bias) | all except **Options mount-mode toggle** and **Square tare** | tare = AUX encoder zero; decide on the rig whether it is still wanted |
| Sensor calibration (2 cameras) | Cameras (3 cameras) | gain / exposure / gamma / rotation / ROI / swap / combined view; hardware ROI via `ASISetStartPos` (untested) |
| Filter wheels | Cameras | new: EFW via `EFW_filter.dll` (untested — DLL not on the dev PC), manual wheel assignments persisted |
| Alignment (plate solve + runner) | Align | tetra3 port identical; runner bit-matches Python grid/holdout; Eq mode fit available, Eq alignment UI minimal |
| Post-process / replay / stacking / MP4 | Replay | proxies + buffering; OneDrive online-only runs detected |
| Mount 3D | Mount 3D | + sky objects |
| HW Sim | Sim | encoder/rate noise + backlash not injected yet (the Rust responder owns its physics) |
| Config editor | Config | native-app keys only; round-trips unknown keys; Python-only keys (tooltips, ngc/messier toggles live in the Track toggles now) |
| ADS-B (RTL-SDR / dump1090 / sim) | native | **real Mode S frames received on this PC** |
| TLE fetch (gp.php) + disk cache | native | `skytracker-astro::tle::download_to_cache` (ureq); auto-refresh by age + a button; first run downloaded 16,069 TLEs |
| Live screenshots script, tooltips, pygame tests | n/a | `SKYTRACKER_SCREENSHOT_DIR` tour replaces make_live_screenshots |

## Must be confirmed on the rig before deleting Python

1. **Serial transport**: `mount_transport: "serial"`, `mount_serial_port`, 9600 baud — PROGRAM track on a real pass; compare loop Hz and rms to the Python baseline (67–73″).
2. **ASI cameras** by hardware index for all three slots (`camera_source: "asi"`), frame rates at the configured bins, hardware ROI, gain/exposure live changes, capture timestamps (exposure-midpoint backdating).
3. **HOTSPOT x/y signs + rotation** on the guide camera; HANDOFF → HOTSPOT on a bright pass; lock robustness against stars crossing the gate (sim shows occasional captures — tune `hotspot_rate_gate_dps`, gate radius).
4. **AltAz-Side geometry** end-to-end: PROGRAM residuals, HOTSPOT corrections in the right axis directions (the `sky_delta_to_axis` generalization).
5. **EFW**: slot count, names, goto, position polling; `EFW_INFO` struct layout.
6. **Bubble cam**: fisheye focal length (1.55 mm assumed) and orientation (north up / rotation) — compare the sim overlay with a real frame.
7. **Gamepad**: button indices on the actual controller (gilrs names vs pygame numbers).
8. Flip the eight Python-side Rust flags and run the Python app once more as the reference for the session.

## Deletion plan (after sign-off)

- Move `tests/golden` parity tests to pure-Rust integration tests (npyz readers exist in platesolve/astro), drop maturin/pytest from CI, add the Windows release package (exe + ASICamera2.dll + EFW_filter.dll + rtlsdr.dll + de421.bsp + catalogs).
- Delete `skytracker-ffi`, `*.py`, `environment.yml`, `requirements.txt`, `pyproject.toml`; keep `tools/` scripts that generated goldens as documentation.
