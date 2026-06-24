# Learnings

A running log of non-obvious things we hit and how we resolved them — a general
companion to the Rust-experiment-specific [`rust/FINDINGS.md`](rust/FINDINGS.md).
Newest entries first.

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
