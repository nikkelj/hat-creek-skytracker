"""Flag-gated Rust camera pipeline behind the CameraThread interface
(Phase 4b of the Rust port).

`RustCameraThread` subclasses camera_buffer.CameraThread and overrides only
the hot loop: raw frames go straight into the Rust CameraPipeline (ring +
exposure-midpoint stamping, zero per-frame GIL work beyond the device read
itself), and the pygame display Surface is built LAZILY on display pull
(<= UI rate) instead of per frame -- the two costs that capped the Python
path at 4-10 FPS. The armed-capture path keeps building the exact
frame-dict ring capture_manager consumes, so capture semantics are
unchanged (and, as before, full frames are retained only while armed).

Device I/O still runs through camera_cap (SimCap or zwoasi) in this phase;
the native `CameraPipeline.open_asi` path replaces it at rig time once the
SDK timing truth is validated.

Enabled by config `use_rust_camera` (configure(config_state) at startup)
or env SKYTRACKER_RUST_CAMERA=1 (0 force-disables). On any pipeline
failure the constructor raises and camera_manager falls back to the
Python CameraThread.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from camera_buffer import CameraThread, exposure_midpoint_utc

_enabled_flag = False


def configure(config_state):
    global _enabled_flag
    _enabled_flag = bool(getattr(config_state, "use_rust_camera", False))


def enabled():
    env = os.environ.get("SKYTRACKER_RUST_CAMERA")
    if env == "1":
        return True
    if env == "0":
        return False
    return _enabled_flag


class RustCameraThread(CameraThread):
    """CameraThread with the per-frame path routed through Rust."""

    def __init__(self, camera_index, camera_cap, buffer_size=1000, target_fps=30):
        super().__init__(camera_index, camera_cap, buffer_size=buffer_size,
                         target_fps=target_fps)
        import skytracker_core as sc

        if not getattr(sc, "CAMERA_AVAILABLE", False):
            raise RuntimeError("skytracker_core wheel predates CameraPipeline")
        self._pipe = sc.CameraPipeline.push_source(ring_capacity=buffer_size)
        self._display_cache_seq = -1
        self._display_cache = None

    # ------------------------------------------------------------- hot loop
    def run(self):
        print(f"Rust camera thread {self.camera_index} starting...")
        self.frame_counter = 0
        self.fps_timer = time.time()
        self.capture_sequence_counter = 0

        while self.running:
            try:
                if not self.camera_cap:
                    self.running = False
                    continue

                capture_start_time = time.perf_counter()
                raw_frame = self._ultra_fast_capture()
                capture_process_time = time.perf_counter() - capture_start_time
                utc_after_capture = datetime.now(timezone.utc)
                mono_midpoint = time.perf_counter() - max(0.0, capture_process_time) / 2.0

                if raw_frame is not None and raw_frame.size > 0 \
                        and capture_process_time < self.capture_timeout:
                    # Publish raw for the hot-spot loop (same torn-read
                    # discipline as the parent: payload first, seq last).
                    self.latest_raw = raw_frame
                    self.latest_raw_time = mono_midpoint
                    self.latest_raw_seq += 1

                    # Rust ring + midpoint stamp (Arc'd, no copy on reads).
                    if raw_frame.ndim == 2:
                        self._pipe.push_frame_mono(raw_frame, capture_process_time)
                    else:
                        self._pipe.push_frame_rgb(raw_frame, capture_process_time)

                    if self.capture_active:
                        # Armed: build the full frame dict (incl. Surface)
                        # exactly like the Python path -- capture flushes
                        # depend on it and sessions are bounded.
                        surface = self._process_raw_frame(raw_frame)
                        if surface is not None:
                            utc_now = exposure_midpoint_utc(
                                capture_process_time, now=utc_after_capture)
                            local_now = utc_now.astimezone().replace(tzinfo=None)
                            frame_data = {
                                'frame': surface,
                                'monotonic_midpoint': mono_midpoint,
                                'datetime_utc': utc_now.isoformat(),
                                'datetime_utc_microseconds':
                                    f"{utc_now.strftime('%Y-%m-%dT%H:%M:%S')}"
                                    f".{utc_now.microsecond:06d}Z",
                                'datetime_local': local_now.isoformat(),
                                'camera_index': self.camera_index,
                                'capture_time': capture_process_time,
                                'sequence_in_capture': self.capture_sequence_counter + 1,
                                'buffer_sequence': self.frame_count,
                            }
                            self.circular_buffer.append(frame_data)
                            self.latest_frame = frame_data
                            self.capture_sequence_counter += 1
                            self.capture_frame_count += 1
                    else:
                        # Idle: NO per-frame Surface work; display converts
                        # lazily in get_latest_frame at the UI rate.
                        self._last_meta = (capture_process_time, utc_after_capture,
                                           mono_midpoint)

                    self.frame_count += 1
                    self.frame_counter += 1
                    now = time.time()
                    if now - self.fps_timer >= 1.0:
                        self.actual_fps = self.frame_counter / (now - self.fps_timer)
                        self.frame_counter = 0
                        self.fps_timer = now

                # Pace to the target rate like the parent loop.
                elapsed = time.perf_counter() - capture_start_time
                delay = max(0.0, (1.0 / self.target_fps) - elapsed)
                if delay > 0:
                    time.sleep(delay)
            except Exception as e:
                print(f"Rust camera thread {self.camera_index} error: {e}")
                time.sleep(0.1)

        print(f"Rust camera thread {self.camera_index} stopped.")

    # ------------------------------------------------------------- display
    def get_latest_frame(self):
        """Lazy display dict: the Surface is built only when a NEW frame is
        actually pulled (UI rate), not per capture."""
        if self.capture_active and self.latest_frame is not None:
            return self.latest_frame
        seq = self.latest_raw_seq
        if seq == self._display_cache_seq:
            return self._display_cache
        raw, seq, mono = self.get_latest_raw_with_meta()
        if raw is None:
            return None
        surface = self._process_raw_frame(raw)
        if surface is None:
            return None
        meta = getattr(self, "_last_meta", None)
        capture_time = meta[0] if meta else 0.0
        utc_now = exposure_midpoint_utc(capture_time,
                                        now=(meta[1] if meta else None))
        local_now = utc_now.astimezone().replace(tzinfo=None)
        self._display_cache = {
            'frame': surface,
            'monotonic_midpoint': mono,
            'datetime_utc': utc_now.isoformat(),
            'datetime_utc_microseconds':
                f"{utc_now.strftime('%Y-%m-%dT%H:%M:%S')}"
                f".{utc_now.microsecond:06d}Z",
            'datetime_local': local_now.isoformat(),
            'camera_index': self.camera_index,
            'capture_time': capture_time,
            'sequence_in_capture': 0,
            'buffer_sequence': self.frame_count,
        }
        self._display_cache_seq = seq
        return self._display_cache

    # ------------------------------------------------------------- teardown
    def stop(self):
        super().stop()
        try:
            self._pipe.close()
        except Exception:
            pass
