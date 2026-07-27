# Learnings

A running log of non-obvious things we hit and how we resolved them — a general
companion to the Rust-experiment-specific [`rust/FINDINGS.md`](rust/FINDINGS.md).
Newest entries first.

---

## 2026-07-27 — Project-wide timing audit: three clocks, and who may read which

A full audit ("something is off" — sky objects disagreeing in time) found the
root pattern behind a dozen bugs: the app has THREE clock domains and
consumers were picking freely. The rules now enforced in code:

* **Tracking clock** (`tracking_vis_state.current_tt`: slider / paused / live)
  is for VISUALIZATION ONLY — starfield, overlays, arc colors, T+ readouts,
  Mount 3D. The control loops must never read it: they interpolate setpoints
  at `trajectory.live_tt()` (the mount cannot time-travel; pausing the UI
  used to freeze the setpoint mid-track with feed-forward still running).
  Pause now latches `current_tt` (not wall now), so pausing a scrubbed scene
  doesn't snap it back to live.
* **Wall clock** (`time.time()` / `ts.now()`) is for absolute sky time only.
  Every dt/elapsed/timeout now runs on **`time.perf_counter()`** — the PID
  cycle dt (which also silently DISCARDED the dt its callers passed:
  shadowed parameter), park timeout, ADS-B pruning, sim mount physics,
  hotspot coast. NOT `time.monotonic()`: on Windows/CPython ≤ 3.12 that is
  GetTickCount64 with a **15.625 ms quantum** (measured here: `sleep(0.01)`
  reads as 0.0 elapsed) — ±25% dt noise at a 15 Hz loop, and it silently
  broke a stale-pruning test. perf_counter is QPC: equally monotonic,
  sub-microsecond. dt is clamped to [DT_MIN, DT_MAX] in control.py AND
  pid.rs, so the first cycle after a standby can't integrate the whole idle
  gap (that pinned the integrator at its clip in one cycle → max-rate burst
  on resume).
* **Frames carry their own time**: the capture thread stamps a monotonic
  exposure-midpoint (`latest_raw_time`, seq/payload/seq tear-free protocol;
  Rust `Frame.time` on the shared loop epoch, back-dated by `age_s` at
  push). Everything image-derived — hotspot frame interval, gate prediction,
  star-filter rates, plate-solve pairing — uses frame time; pickup-time
  stamping aliased the control period into those measurements (correction
  cap jittered ±50%; frame-age jitter ≈ the star-filter gate itself).

Repeat offenders worth remembering: 0.0 as a timestamp sentinel is safe
under a wall clock (epoch ~1.8e9) and WRONG under a monotonic/loop clock
(Rust "coasting on a lock never acquired") — use Option/None. Azimuth
differences must take the short arc EVERYWHERE (rates fed a ±360°/dt
feed-forward spike on due-north crossings). And caches keyed by less than
what the value depends on (sunlit lists vs their time grid, trajectories vs
observer+plot geometry) rot silently — key them fully or clear at the source.

## 2026-07-26 — The stars WERE frozen — under the time slider, not the live clock

Correction to the entry below: after "verified NOT the case," the user
reported the precise repro that was the case — scrub the tracking-vis time
slider and the stars don't move while the satellites do. Both render
threads computed their own `current_tt = ts.now().tt` (wall clock) and drew
the starfield + launch overlays at it, while the satellites were positioned
in main.py from `tracking_vis_state.current_tt` (the app's tracking clock:
slider position while scrubbing, `paused_tt` while paused, live otherwise).
Two clocks, one scene — the halves disagree exactly when the user scrubs.
Fix: `render_time_tt()` in rendering_threads.py resolves the tracking clock
(with a live-now fallback before the first 10 Hz tick) and both threads use
it; pinned by a scrubbed-starfield pixel-diff test in test_star_catalog.py.

Lesson: "does X update with time?" has TWO answers per surface — live time
and scrubbed time — and a test that only advances the live clock proves
nothing about the slider. The earlier verification tested the wrong one.

## 2026-07-26 — Mount 3D: below-horizon orbit, and "is the star sky frozen?"

Two follow-ups on the Mount 3D tab. (1) The orbit camera's elevation clamp
was floored at -5°; it now runs the full ±89°, so you can dive under the
translucent ground disc and look straight up through the mount at the sky
dome. That exposed a layering bug worth remembering: everything translucent
lived on ONE overlay blitted after the mount model, so the ground/keepout
tint washed over the hardware. Translucent layers need to composite in
scene order — sky-level tint under the mount, boresight ray + FOV cones
over it — which means two overlays, not one.

(2) "The star catalogue seems fixed at one epoch" — verified NOT the case,
and now pinned by a test instead of a code-reading argument:
`tracking_vis_state.current_tt` advances at 10 Hz **unconditionally** in the
main loop (not gated by the active screen), and `star_catalog.current_altaz`
applies exact LST rotation per query around a 600-s re-anchor. The
regression test renders the 3D scene twice, 2 h of tracking-time apart, with
everything but the stars held constant, and asserts the frames diverge. The
first version of that test "confirmed" the freeze with 0 changed pixels —
because it had passed `ts` into the wrong positional slot and the star pass
was silently excepting. A silent `except -> print` in a render path turns a
test bug into a false confirmation of the very fear being tested; check the
console line, not just the assertion.

## 2026-07-26 — Online PID auto-tune: twiddle on the live error stream works

Added [`autotune.py`](autotune.py): one button tunes all six PID gains *while
tracking*, no injected test signals — a coordinate-descent ("twiddle")
optimizer in **log-gain space** (the gains span 5 decades, matching the UI's
log sliders) that probes each gain up/down, scores a settle-then-measure RMS
window of the live mount-axis error, keeps >3% improvements, and expands/
shrinks its per-gain step. Both control loops re-read gains from config every
cycle, so the tuner is loop-agnostic: it just writes candidates into
`config_state` and watches `azm/alt_position_error` on the shared state
(serviced from the mount-control cycle *and* the Rust adapter pump). On the
live-fidelity sim rig (noise, backlash, periodic error, ~10 fps camera,
15 Hz loop) a ~2.5-minute tune took the field-tuned example gains from
**81″ → 38″ sky RMS** (encoder-error RMS 30″/13.5″ → 5.7″/4.4″ az/el).

Non-obvious bits:

- **Re-baseline every sweep.** Windowed RMS drifts with the pass geometry;
  scoring candidates against a best-cost measured minutes ago either blocks
  all acceptance or accepts regressions. Each sweep starts with a fresh
  baseline window at the current best gains.
- **The acceptance margin is what makes it a descent.** With ~40 samples per
  window the RMS estimate is noisy; a plain `<` comparison random-walks the
  gains. Requiring >3% improvement (and reverting otherwise) makes progress
  monotone in expectation.
- **Pause ≠ stop.** Mode change, STOP, or a lost optical lock mid-probe must
  revert to the best-known gains and *restart the interrupted probe* on
  resume — a half-measured window scored across the gap is garbage.
- **Measure the plant you armed on.** PROGRAM (encoder loop) and HOTSPOT
  (optical loop) are different plants; the tuner refuses to mix their samples.
  Follow-through: each plant keeps its own **gain profile**
  (`config.pid_mode_profiles`, stamped with the target it was tuned on),
  swapped into the live gain fields automatically on mode transitions by
  `service_gain_profiles()`. Keeping the six live fields as "the active set"
  meant the sliders, both loops, and the tuner needed zero changes — but the
  swap must run BEFORE the tuner's servicing each cycle, and a plant-changing
  transition must STOP a running tune (keeping its best) so the departing
  profile saves tuned gains, not a half-tested probe candidate.
- **Sim-rig gotcha that looked like a tuner disaster:** the tracking-quality
  `Rig`'s analytic target climbs at 0.5°/s elevation, so past ~120 s it
  crosses the zenith and the rig's sky-error metric (built for el ≤ 90°)
  reports hundreds of thousands of arcsec while the loop is actually fine.
  A first "tune made it worse" verdict (38″ → 143″, then 430 000″) was
  entirely this artifact; with a 0.08°/s target the same tune measured
  cleanly. Any long-window use of `Rig` needs rates that keep el < 90°.

## 2026-07-26 — The 9600-baud wire caps the control loop at ~8 Hz; dedup rate commands

`bench_guiderate.py --throughput` on the real AVX: every AUX transaction costs
~30 ms round-trip (mean 30.5, p95 41.6 — that is 9600-baud wire time, not
software overhead), so the control cycle's 4 transactions (read AZM+ALT,
command AZM+ALT) support only **7.7 Hz against the 15 Hz target**. Reads alone
support ~16 Hz. The fix shipped in both loops: **rate-command deduplication** —
the firmware *holds* the last guide rate (and MC_MOVE step), so an unchanged
command is pure wire waste; commands are re-sent only when the wire-quantized
value changes, with a 1 s keepalive and cache invalidation on every
out-of-band stop path. Steady tracking now costs 2 transactions/cycle.

Rehearse this reality in sim: set `sim_serial_latency_s` to `0.03` in the HW
Sim screen — gains tuned against an instant-serial sim meet ~130 ms of
transport lag on hardware and lose phase margin (a contributor to the first
side-mode session's PID oscillation).

Also: the mount has no software encoder tare — the "Tare axes" button tares
the *game controller sticks*. Mount homing = power on at the index marks
(firmware boots encoders to 0) or the joystick screen's **Sync Home** button
(captures current raw encoder readings into azm/alt offsets; Park drives back
to them).

## 2026-07-26 — AltAz-Side home is the AVX index marks, not the zenith

First PROGRAM track in the new side-mount mode drove the scope into the
ground while the UI self-consistently displayed el +49°, and jog polarity
*appeared* wrong on both axes. Root cause: the mode's transforms assumed
encoder (0, 0) = zenith, but the natural (and mechanically meaningful) home
on the AVX is the **index marks — where the scope points ALONG the polar
axis**. In the side rig that's at the *horizon*, at the pole azimuth, with
the dec axis vertical. The decisive field datum: at the index marks the
scope sat on the horizon at az 259° while the model read el +17.7° — exactly
asin(cos ALT · cos AZM) of the encoder values, confirming a coherent model
with the wrong home.

Fix: H = AZM + 90°, dec = 90° − ALT (`ALTAZ_SIDE_H0_DEG`), which also makes
the axes behave as originally described from home (+ALT walks the horizon —
identically 0 elevation, a strong test — and AZM sweeps the vertical circle
once off the pole). `altaz_side_flip` mirrors the H origin for a rig laid
down on its other side. Field procedure: index marks → Sync Home (the
connection-panel button that captures raw encoder readings into the
azm/alt offsets — the joystick "Tare axes" button is for the game
controller sticks, not the mount) → enter the azimuth the scope points at
as Alignment Azimuth.

Meta-lessons: (1) an apparent *polarity* error on both axes was actually an
*index* error — near H = 90° the sign of ∂el/∂ALT flips, so a 90° home
offset masquerades as reversed axes; (2) "the UI agrees with itself" proves
nothing — the navball, skyplot and FOV boxes all read from the same wrong
transform; only an independent physical reference (the horizon) broke the
tie; (3) define a mode's home as something the operator can *find* with the
hardware's own markings, not an attitude they must eyeball.

## 2026-07-25 — AVX guide-rate wire unit is arcsec/s × 1024, not rev/s × 2²⁴

First hardware run of `bench_guiderate.py` (checklist step 2): every commanded
continuous rate produced "NOT MOVING" with a dead-constant ratio of **0.01265 =
1/79.1** — commanded 1.0 °/s actually moved at 45.5″/s. 79.1 is exactly the
ratio between our assumed encoding (revolutions/sec × 2²⁴) and the firmware's
real unit: **the 24-bit MC_SET_POS/NEG_GUIDERATE value is arcseconds/second in
Q10 fixed point (value = arcsec/s × 1024)**. Corroboration: 24-bit full scale
works out to 4.551 °/s — precisely the AVX max slew — and the encoder readback
was independently validated in the same run (see below). Fixed in
`hc_set_rate_dps` + `protocol.rs encode_set_guiderate` + both sim decoders
(count-space rounding so full scale encodes exactly 0xFFFFFF), and
`guide_rate_max_dps` default lowered 5.0 → 4.5 to sit under full scale.

Second finding from the same run: the discrete **`RATES` table is wrong on
this mount** — MC_MOVE rate 4 measured 0.0335 °/s = 8.0× sidereal (the classic
Celestron HC progression), not the 0.25 °/s the table lists. That table feeds
the discrete-rate PID path AND the simulated mount, so desk rehearsals were
self-consistent but ~7.5× optimistic about discrete slew authority. Left
as-is pending a full measurement: `bench_guiderate.py --survey` now measures
all nine steps and prints a paste-ready table.

Meta-lesson: the bench ordering in `doc/BENCH_CHECKLIST.md` was right — this
was caught at guide rates with a hand on the power switch, before any
tracking mode trusted the encoding. And a single reference measurement of a
*known* discrete rate in the same run was what separated "encoder scale
wrong" from "command scale wrong."

## 2026-07-07 — HOTSPOT/HANDOFF stair-steps: four coupled failure modes

Field report: HANDOFF centered briefly then walked off in near-FOV-sized
stair-steps; manual HOTSPOT reverted to PROGRAM immediately. A live-fidelity
harness ([`test_mode_machine.py`](test_mode_machine.py)) — real
`tracking_control()` state machine against the simulated mount, a paced
camera thread, and the *tuned* `config.example.json` gains — reproduced the
walk-off on the first run. Four distinct bugs compounded; any one of them
alone would have been survivable:

1. **Wrong elevation convention in the optical geometry.** The pixel↔angle
   transforms need the SKY elevation for the cos(el) azimuth compression, but
   `hotspot_track` passed the mount ALT axis (which is `90 − el` in AltAz).
   At el=30° that overstates the azimuth error by cos(30)/cos(60) ≈ 1.73× —
   a *divergent* per-step overshoot. This is the same convention trap as the
   sky-vs-mount entry below: any code touching angles must say which frame
   it's in.
2. **Uncapped correction rates.** The PID output was clamped only by
   `guide_rate_max_dps` (5°/s — a slew rate). One high-SNR detection at the
   FOV edge commanded a lunge that swept the whole FOV before the next frame.
   New `hotspot_max_rate_dps` (default 2.0) plus a measurement-cadence cap:
   never cover more than ~90% of the remaining error per measured frame
   interval — a centering loop must not outrun its own measurements.
3. **Static tracking gate.** The search gate stayed where the target *was*;
   the loop's own commanded slew streams the target across the frame, so a
   legitimate correction pushed the target out of its own gate → instant
   "lost". The gate is now predicted forward by the loop's own commanded
   rates (anchored at the last detection, not the last frame) and grows
   1.5×/miss (cap 4×) so residual error can't strand it.
4. **Held rates never decayed on miss.** On a missed frame the mount kept
   the last commanded rate indefinitely ("coast"), so one overshoot coasted
   into the next stair-step. Held rates now halve per missed fresh frame —
   coast through a one-frame dropout survives, sustained loss bleeds to zero.

All four are mirrored in the Rust loop (`step_hotspot`), with
`hotspot::angles_to_pixel_offset` added as the exact inverse of the forward
transform for the gate prediction.

General principle: closed-loop optical tracking bugs are invisible to
open-loop unit tests — every piece (detector, PID, geometry) tested fine in
isolation. The failures only appear when the loop's own actuation feeds back
into its next measurement. Keep an end-to-end harness (real state machine +
simulated physics + real config) as the regression net for any control-loop
change.

---

## 2026-07-07 — Per-axis shortest-arc wrap is not shortest spherical path

Field report from the sim: mount pointed west, target in the east — PROGRAM
tracking drove the azimuth axis ~180° the long way around instead of ~90° of
ALT motion straight over the zenith.

The per-axis modular wrap (the earlier "360 lap" / ALT-seam fixes) reduces
each axis error to ±180°, but it only ever sees the **canonical** mount
solution. An alt-az style mount reaches every sky pointing in **two** axis
configurations — canonical, and over-the-zenith via the mirrored sky
representation `(az+180°, 180−el)` (in the AltAz convention that is
`(AZM+180°, −ALT)`). When the target is on the far side of the sky, the
flipped solution is often far shorter, and no amount of per-axis wrapping
can discover it.

Fix: `control.choose_mount_target` evaluates both solutions each cycle
(minimax wrapped axis error, 0.5° hysteresis so near-ties don't flap
mid-slew, safety-limit aware — an out-of-limits solution is never chosen),
mirrored in the Rust `step_program`. Two knock-on sign rules matter: the
ALT **feed-forward sign inverts** in the flipped configuration (the axis
runs opposite), and the limit gate must gate the solution actually being
driven to. Fixing the FF sign here also resolved a long-standing divergence
where the launch path had no AltAz negation at all.

General principle: on a two-axis mount, "shortest path" is a choice between
*configurations* first and per-axis arcs second. Any future mode (Eq
meridian flips, HANDOFF) that computes axis errors from a sky target needs
to ask "which of the mount's solutions am I wrapping toward?"

---

## 2026-07-03 — Stacking performance + the "sharpness rewards noise" trap

Profiling the stacking pipeline ([`stacking.py`](stacking.py)) on realistic
1920×1080 frames overturned two assumptions and surfaced a correctness trap that
matters more than the speed.

**Where the time actually goes.** Decode of the uncompressed BMPs is ~1.6 ms/frame
(and OpenCV's `IMREAD_REDUCED_*` does *not* help BMP — it's already memory-bound),
so the earlier "double-decode" worry was a non-issue. The real costs are the
**sharpness metric** (`cv2.Laplacian` at `CV_64F` ≈ 11 ms/frame) and **ORB
feature detection** (≈ 25 ms/frame, the dominant stack cost). Fixes: `CV_64F →
CV_32F` (~2×, variance identical for ranking); optional reduced-res grading
(5–10×, ranking-preserving); a `ThreadPoolExecutor` over frames (OpenCV releases
the GIL, so ~1.7× on the align-bound pass across 4 cores even though cv2 already
threads internally); and an opt-in half-res *alignment* (`align_scale`, ~2× on
detect for **extended/textured** targets — point-source star fields blur away
when downscaled, so it's target-dependent and defaults off).

**The trap: Laplacian variance rewards noise.** A pure sensor-noise / hot-pixel
frame scores a *huge* sharpness (measured ~10 000) while a faint but real target
scores ~0. So a naive "keep the top X% by sharpness" first pass does the opposite
of its job — it **keeps the junk and discards the good frames**, and then the
stacker can't align the noise so it stacks almost nothing (measured 1/60). The
fix is a two-tier cull: a cheap, noise-robust `content_score` (reduce with
area-averaging → Gaussian blur → `max − median` prominence) gates out frames with
*no signal* first, *then* the precise sharpness metric ranks the survivors.
Structure survives denoising; noise collapses — so faint real frames are kept and
noise/blank frames are dropped before ranking ever sees them.

**Parallel stacking without locks.** The align+average pass is parallelized as a
map-reduce: K worker stackers share one read-only alignment reference (only one
*seeds* it into its accumulator, so the merged master counts it once), each
accumulates its own slice, and the partial float sums are added at the end — no
lock on the hot path. Trim `cv2.setNumThreads` while the Python pool runs so the
two thread pools don't oversubscribe the cores.

---

## 2026-06-30 — Lucky-imaging stacking (PIPP/AutoStakkert) gotchas

Adding the stacking pipeline ([`stacking.py`](stacking.py)) surfaced three
non-obvious traps, all caught by an adversarial review pass before merge:

- **Alignment-point grid must span edge-to-edge.** AutoStakkert-style local
  warping measures a per-point shift on a coarse grid, then upsamples it to a
  dense field with `cv2.resize`. `resize` maps grid node 0 → pixel 0 and the
  last node → pixel `size-1`, so if the grid is *inset* by a margin, every
  measured shift is re-applied ~margin px away from where it was measured — it
  actively warps detail to the wrong place near the borders. Fix: place nodes on
  `linspace(0, size-1, n)` and let `measure_local_shifts` clamp each patch
  inward at the edges instead of insetting the nodes.
- **Stacking must be per-pixel coverage weighted, not sum/count.** Aligning a
  jittered frame shifts real pixels in and leaves `BORDER_CONSTANT` black at the
  revealed edge. A naive `sum / frame_count` average then darkens a whole border
  band proportional to the jitter. Accumulate a per-pixel coverage mask (warp an
  all-ones image by the same transform) alongside the sum and divide by it; the
  reference covers the whole frame so every pixel keeps a valid average.
- **Derive reconciling stats.** Tracking `n_total` as its own counter drifted
  out of sync with `n_stacked`/`n_rejected` (the directly-set reference frame
  was never counted), yielding "stacked 10 of 9". Derive
  `n_total = n_stacked + n_rejected` so the counts can't disagree.

General principle reaffirmed: any "coarse grid → dense field via resize" step has
an off-by-the-margin registration trap, and any align-then-average step needs a
coverage denominator, not a frame count.

---

## 2026-06-24 — Recovering joystick-mode render & sim-imaging FPS

**Symptom.** After the navball rebuild and the ADS-B additions, the Joystick Loop
felt sluggish and the simulated imaging frame rate fell to 3–6 FPS.

**Root cause: GIL starvation.** The whole UI is redrawn synchronously on one
thread, and the simulated camera captures on its own thread. Several per-frame
costs on the render thread were holding the GIL away from the capture thread:

- `render_navball` (since the "true sphere" rebuild) did a full per-pixel
  hemisphere fill **plus ~1,700 pure-Python trig projections** for the grid —
  *every frame*, even with the mount stationary. Pure-Python work doesn't release
  the GIL, so the capture thread couldn't get a time slice.
- `_process_feed_surface` ran gamma + scale + rotate on **both feeds every
  frame**, even when re-displaying the same captured frame (the UI renders faster
  than the camera produces frames).
- The main loop ran at `FPS_TARGET = 60`, doubling all of the above vs 30.

**Fixes.**

1. **Cache the navball "base" surface** (hemisphere + grid + static labels) keyed
   on `(radius, quantized az, quantized el)`; only the dynamic overlays
   (trajectory, target, aircraft) redraw each frame. Pointing is snapped to 0.5°
   (sub-pixel on screen). `render_navball`: **3.37 → 0.10 ms/frame (33×)** when
   stationary. Cache misses during a slew, so there's no regression while moving;
   it amortizes the moment pointing settles.
   ([`joystick_controller.py`](joystick_controller.py) `render_navball`)
2. **Cache the processed feed surface** per camera, keyed on the capture
   `frame_seq` (the monotonic `buffer_sequence` from the capture thread) plus all
   processing params. Same frame → cache hit, no reprocessing. Saves ~0.84 ms per
   feed per redundant frame (more with gamma on). Falls back to always-recompute
   if `frame_seq` is unavailable, so it's safe by construction.
   ([`joystick_controller.py`](joystick_controller.py) `_process_feed_surface`,
   [`camera_manager.py`](camera_manager.py) `update_camera_frames_from_buffers`)
3. **Cap the main loop at 30 Hz** (`FPS_TARGET = 60 → 30`) — halves the frequency
   of every per-frame cost; imperceptible for a control UI.
   ([`display.py`](display.py))

**Verification.** Headless harnesses confirmed both caches are pixel-identical to
the recompute path and invalidate correctly (new pointing / new frame / param
change). `test_render_buffer`, `test_simulator`, `test_hw_sim_ui` all pass.

**General principle.** Anything redrawn every frame on the render thread should be
*invariant-cached*: identify what the output actually depends on, key a cached
surface on that, and rebuild only on change. Pure-Python per-frame work is
especially costly here because it blocks the GIL the capture thread needs — push
it off the hot path or cache it. The control loop already lives on its own thread
([`mount_control.py`](mount_control.py)); the render thread needs the same
discipline about what it recomputes.
