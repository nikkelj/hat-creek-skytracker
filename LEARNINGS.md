# Learnings

A running log of non-obvious things we hit and how we resolved them — a general
companion to the Rust-experiment-specific [`rust/FINDINGS.md`](rust/FINDINGS.md).
Newest entries first.

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
