"""
Threaded camera capture system with large circular buffer for Hat Creek Skytracker
Enhanced to support 10+ minute buffering with microsecond-precision timestamps
"""

import threading
import time
import numpy as np
import pygame
from collections import deque
from datetime import datetime, timedelta, timezone


def exposure_midpoint_utc(capture_time_s, now=None):
    """UTC timestamp back-dated to the approximate exposure midpoint.

    Frames used to be stamped with now() AFTER exposure + readout + numpy
    conversion completed, so every timestamp lagged reality by the full
    capture duration -- a bias that flows straight into the per-frame
    trajectory-CSV interpolation (a LEO target moves ~1 deg/s; a 0.5 s
    exposure stamped at readout end mislabels the frame by ~0.25-0.5 deg).
    Back-dating by half the measured capture time lands on the exposure
    midpoint to within the readout latency.
    """
    now = now if now is not None else datetime.now(timezone.utc)
    return now - timedelta(seconds=max(0.0, capture_time_s) / 2.0)

class CircularBuffer:
    """True circular buffer implementation with pre-allocated memory"""
    def __init__(self, size=1000):  # 1000 frames = ~1 minute
        self.buffer = deque(maxlen=size)
        self.size = size

    def append(self, item):
        """Add item to buffer - will automatically remove oldest when full"""
        self.buffer.append(item)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def clear(self):
        self.buffer.clear()

    def is_full(self):
        return len(self.buffer) == self.size

    def get_fill_ratio(self):
        return len(self.buffer) / self.size

class CameraThread(threading.Thread):
    """Dedicated thread for camera capture with large circular buffer support"""

    def __init__(self, camera_index, camera_cap, buffer_size=1000, target_fps=30):
        super().__init__()
        self.camera_index = camera_index
        self.camera_cap = camera_cap
        self.running = True
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_capture = 0
        self.capture_timeout = 5.0
        self.error_count = 0
        self.max_errors = 5
        self.frame_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()
        self.frame_counter = 0

        # Enhanced circular buffer for capture features
        self.circular_buffer = CircularBuffer(buffer_size)

        # Capture state tracking
        self.capture_active = False
        self.capture_start_idx = -1
        self.capture_start_time = None

        # Provide direct buffer access for UI
        self.latest_frame = None

        # Latest raw numpy frame (mono HxW or color HxWx3), kept for the
        # hot-spot detector which needs full-bit-depth intensity, not the
        # RGB pygame Surface used for display. latest_raw_seq increments with
        # every new raw frame so detection loops can tell a fresh frame from
        # a stale one (real exposures outlast the control period).
        self.latest_raw = None
        self.latest_raw_seq = 0

        # MEMORY OPTIMIZATION: Pre-allocate frame data structure
        self._frame_data_template = {
            'frame': None,
            'timestamp': 0,
            'datetime_utc': '',
            'datetime_local': '',
            'camera_index': camera_index,
            'capture_time': 0,
            'sequence_in_capture': 0
        }

        # Get camera properties
        try:
            self.camera_props = camera_cap.get_camera_property()
        except Exception as e:
            self.camera_props = {'MaxWidth': 1920, 'MaxHeight': 1280}

        print(f"Camera {camera_index}: initialized with {buffer_size} frame buffer")

    def get_latest_frame(self):
        """Get the most recent frame - no locks, just return the reference"""
        return self.latest_frame

    def get_latest_raw(self):
        """Get the most recent raw numpy frame for detection (or None)."""
        return self.latest_raw

    def get_buffer_fps(self):
        """Get current buffer FPS"""
        return self.fps

    def get_utc_timestamp(self):
        """Get latest frame UTC timestamp"""
        if self.latest_frame:
            return self.latest_frame.get('datetime_utc', "")
        return ""

    def get_local_timestamp(self):
        """Get latest frame local timestamp"""
        if self.latest_frame:
            return self.latest_frame.get('datetime_local', "")
        return ""

    def get_buffer_size(self):
        """Get current buffer size for UI compatibility"""
        return len(self.circular_buffer)

    def get_capture_buffer_info(self):
        """Get capture buffer information"""
        info = {
            'buffer_size': len(self.circular_buffer),
            'max_buffer_size': self.circular_buffer.size,
            'fill_ratio': self.circular_buffer.get_fill_ratio(),
            'capture_active': self.capture_active,
            'capture_start_idx': self.capture_start_idx,
            'capture_start_time': self.capture_start_time,
            'capture_frame_count': self.capture_frame_count if hasattr(self, 'capture_frame_count') else 0,
            'is_full': self.circular_buffer.is_full()
        }

        # Calculate capture progress percentage (frames added during current capture / max buffer size)
        if self.capture_active and hasattr(self, 'capture_frame_count'):
            # If buffer hasn't wrapped yet, capture frames represent what's been added since start
            info['capture_progress_ratio'] = min(1.0, self.capture_frame_count / self.circular_buffer.size)

        return info

    def start_capture(self):
        """Start capture - mark beginning of capture in buffer"""
        self.capture_active = True
        self.capture_start_idx = len(self.circular_buffer) - 1  # Last frame in buffer
        self.capture_start_time = datetime.now(timezone.utc)
        self.capture_sequence_counter = 0
        self.capture_frame_count = 0  # Track how many frames we've captured
        print(f"Camera {self.camera_index}: Capture started at buffer index {self.capture_start_idx}")

    def stop_capture(self):
        """Stop capture - return a SNAPSHOT of the frames to dump.

        The snapshot (a plain list) is taken at stop time, because the capture
        thread keeps appending to the live deque: once the deque is full each
        append evicts the oldest entry and shifts every index, so the
        start/end indices computed here would drift under the dump thread and
        a long dump could save the wrong frames. A list also gives the dump
        O(1) indexing instead of deque's O(n).
        """
        if self.capture_active:
            self.capture_active = False

            if self.capture_frame_count == 0:
                # No frames were captured during this session
                print(f"Camera {self.camera_index}: No frames captured")
                return None, None

            frames = list(self.circular_buffer.buffer)
            capture_end_idx = len(frames) - 1

            # Return buffer range and metadata for dump
            capture_info = {
                'start_idx': self.capture_start_idx,
                'end_idx': capture_end_idx,
                'buffer_length': len(frames),
                'capture_start_time': self.capture_start_time,
                'capture_end_time': datetime.now(timezone.utc),
                'captured_frame_count': self.capture_frame_count
            }

            print(f"Camera {self.camera_index}: Capture stopped - captured {self.capture_frame_count} frames, indices {self.capture_start_idx} to {capture_end_idx}")
            return capture_info, frames

        return None, None

    def get_buffer_fill_ratio(self):
        """Get buffer fill ratio for UI progress indicators"""
        return self.circular_buffer.get_fill_ratio()

    def get_utc_timestamp_microseconds(self):
        """Get latest frame UTC timestamp with microsecond precision"""
        if self.latest_frame:
            return self.latest_frame.get('datetime_utc_microseconds', "")
        return ""

    def _ultra_fast_capture(self):
        """Ultra-fast capture - bypass zwoasi's slow polling capture

        This implements the capture logic without the fixed polling delays
        in the original zwoasi capture() method.
        """
        import zwoasi as asi

        try:
            # Step 1: Start exposure immediately
            self.camera_cap.start_exposure()

            # Step 2: Poll exposure status with minimal delays
            # Ultra-fast polling - no initial sleep, minimal poll delay
            while True:
                exposure_status = self.camera_cap.get_exposure_status()
                if exposure_status == asi.ASI_EXP_SUCCESS:
                    break
                elif exposure_status != asi.ASI_EXP_WORKING:
                    # Error condition - return None to handle silently
                    return None
                # Minimal CPU-friendly yield instead of sleep
                time.sleep(0.0001)  # 100 microseconds - much faster than 0.01s

            # Step 3: Get data immediately once ready
            data = self.camera_cap.get_data_after_exposure()

            # Step 4: Handle data conversion - FIXED MEMORY ISSUE
            whbi = self.camera_cap.get_roi_format()
            shape = [whbi[1], whbi[0]]  # height, width

            if whbi[3] == asi.ASI_IMG_RAW8 or whbi[3] == asi.ASI_IMG_Y8:
                # CRITICAL FIX: Use .copy() to avoid buffer lifetime issues
                img = np.frombuffer(data, dtype=np.uint8).copy()
            elif whbi[3] == asi.ASI_IMG_RAW16:
                # CRITICAL FIX: Use .copy() to avoid buffer lifetime issues
                img = np.frombuffer(data, dtype=np.uint16).copy()
            elif whbi[3] == asi.ASI_IMG_RGB24:
                # CRITICAL FIX: Use .copy() to avoid buffer lifetime issues
                img = np.frombuffer(data, dtype=np.uint8).copy()
                shape.append(3)
            else:
                # Unsupported image type - handle silently
                return None

            img = img.reshape(shape)
            return img

        except Exception:
            # Silent failure to maintain speed
            return None

    def calculate_fps(self):
        """Calculate FPS - called externally to avoid main loop overhead"""
        if time.time() - self.fps_timer >= 1.0:
            self.fps = self.frame_counter / (time.time() - self.fps_timer)
            self.fps_timer = time.time()
            self.frame_counter = 0
            return True
        return False

    def _init_frame_buffers(self):
        """Pre-allocate fixed-size buffers for different ROI configurations"""
        max_width, max_height = self.camera_props.get('MaxWidth', 1920), self.camera_props.get('MaxHeight', 1280)

        # Pre-allocate arrays for common ROI sizes (from camera_manager.py roi_sizes)
        self.frame_buffers = {
            'full': {
                'mono_rgb': np.zeros((max_height, max_width, 3), dtype=np.uint8),
                'surface': pygame.Surface((max_width, max_height))
            }
        }

        # Pre-allocate for scaled ROIs (1/2, 1/4, 1/8, 1/16, 1/32)
        roi_scales = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        for scale in roi_scales:
            width = int(max_width * scale)
            height = int(max_height * scale)
            if width > 0 and height > 0:
                self.frame_buffers[f'{scale}'] = {
                    'mono_rgb': np.zeros((height, width, 3), dtype=np.uint8),
                    'surface': pygame.Surface((width, height))
                }

        # Current frame buffer key - will be updated when ROI changes
        self.current_buffer_key = 'full'

        print(f"Pre-allocated {len(self.frame_buffers)} frame buffers for camera {self.camera_index}")

    def set_roi_scale(self, roi_index):
        """Update current buffer key based on ROI setting (called from camera_manager)"""
        roi_scales = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        if 0 <= roi_index < len(roi_scales):
            self.current_buffer_key = f'{roi_scales[roi_index]}'
        else:
            self.current_buffer_key = 'full'

    def _process_raw_frame(self, raw_frame):
        """Process raw frame data with MAXIMUM speed - no prints, minimal operations"""
        try:
            # Get actual frame dimensions - critical for performance, no other way around it
            frame_height, frame_width = raw_frame.shape[:2]
            frame_channels = raw_frame.shape[2] if raw_frame.ndim == 3 else 1

            # Handle different camera types and data formats - optimized
            if self.camera_props.get('IsColorCam', False):
                # Color camera - BGR to RGB conversion, optimized
                if frame_channels == 3:
                    # Ultra-fast: view of raw data with channel reversal (no copy)
                    rgb_frame = raw_frame[:, :, ::-1]
                else:
                    return None
            else:
                # Monochrome camera - expand to RGB using fastest method
                if frame_channels == 1:
                    # No copying needed - can we skip pygame surface creation?
                    rgb_frame = np.repeat(raw_frame[:, :, np.newaxis], 3, axis=2)
                else:
                    return None

            # Try multiple methods to create pygame surface
            try:
                surface = pygame.image.frombuffer(rgb_frame.data, (frame_width, frame_height), 'RGB')
                return surface
            except:
                # If frombuffer fails, try creating from tobytes()
                try:
                    surface = pygame.image.frombuffer(rgb_frame.tobytes(), (frame_width, frame_height), 'RGB')
                    return surface
                except:
                    # If still fails, create minimal surface and let it fail gracefully
                    # Create a black surface as fallback to avoid crashing
                    return None  # Let's handle errors silently rather than showing bad frames

        except Exception:
            # Silent failure - don't slow down with error reporting
            return None

    def run(self):
        """Main thread execution loop with circular buffer support"""
        print(f"Camera thread {self.camera_index} starting...")

        # Reset FPS tracking variables for current session
        self.frame_counter = 0
        self.fps_timer = time.time()
        self.capture_sequence_counter = 0

        while self.running:
            current_time = time.perf_counter()

            try:
                if not self.camera_cap:
                    self.running = False
                    continue

                # ULTRA-FAST CAPTURE - bypass zwoasi's slow polling capture method
                capture_start_time = time.perf_counter()
                raw_frame = self._ultra_fast_capture()
                capture_process_time = time.perf_counter() - capture_start_time

                # Only handle frames that exist and are not timed out
                if raw_frame is not None and raw_frame.size > 0 and capture_process_time < self.capture_timeout:
                    # Keep the raw frame available for the hot-spot detector.
                    self.latest_raw = raw_frame
                    self.latest_raw_seq += 1
                    # Process frame - minimal error handling
                    surface = self._process_raw_frame(raw_frame)
                    if surface is not None:
                        # Microsecond-precision UTC timestamp, back-dated to
                        # the exposure midpoint (see exposure_midpoint_utc).
                        utc_now = exposure_midpoint_utc(capture_process_time)
                        local_now = datetime.now() - timedelta(
                            seconds=capture_process_time / 2.0)

                        # Prepare frame data for both latest frame and buffer
                        frame_data = {
                            'frame': surface,
                            'timestamp': current_time,
                            'datetime_utc': utc_now.isoformat(),
                            'datetime_utc_microseconds': f"{utc_now.strftime('%Y-%m-%dT%H:%M:%S')}.{utc_now.microsecond:06d}Z",
                            'datetime_local': local_now.isoformat(),
                            'camera_index': self.camera_index,
                            'capture_time': capture_process_time,
                            # 1-based: consumers use `sequence_in_capture > 0`
                            # to mean "captured during the active session", so
                            # a 0-based counter silently dropped the first
                            # captured frame from the trajectory CSV.
                            'sequence_in_capture': self.capture_sequence_counter + 1 if self.capture_active else 0,
                            'buffer_sequence': self.frame_count
                        }

                        # Always buffer the frame (circular buffer for capture)
                        self.circular_buffer.append(frame_data)

                        # Keep reference to latest frame for UI/display
                        self.latest_frame = frame_data

                        # Update frame counters
                        self.frame_count += 1
                        self.frame_counter += 1  # FPS calculation counter

                        # Update capture sequence counter if capturing
                        if self.capture_active:
                            self.capture_sequence_counter += 1
                            self.capture_frame_count += 1

                        self.error_count = 0  # Reset error count on success

            except Exception as e:
                self.error_count += 1
                if self.error_count > self.max_errors:
                    print(f"Camera {self.camera_index}: Too many errors, stopping thread")
                    self.running = False
                    continue

            # Minimal delay for thread safety (adjust if needed)
            time.sleep(0.0001)  # 100 microseconds - CPU friendly

            # REMOVED all debug prints for speed
            # REMOVED FPS calculation overhead from main loop

    def calculate_fps(self):
        """Calculate FPS - called externally to avoid main loop overhead"""
        if time.time() - self.fps_timer >= 1.0:
            self.fps = self.frame_counter / (time.time() - self.fps_timer)
            self.fps_timer = time.time()
            self.frame_counter = 0
            return True
        return False

    def stop(self):
        """Stop the camera thread"""
        print(f"Stopping camera thread {self.camera_index}")
        self.running = False
        self.join(timeout=2.0)

    def get_buffer_fps(self):
        """Get current buffer FPS"""
        return self.fps

    def get_utc_timestamp(self):
        """Get latest frame UTC timestamp"""
        if self.latest_frame:
            return self.latest_frame.get('datetime_utc', "")
        return ""

    def get_local_timestamp(self):
        """Get latest frame local timestamp"""
        if self.latest_frame:
            return self.latest_frame.get('datetime_local', "")
        return ""

    def get_buffer_size(self):
        """Get current buffer size"""
        return 1 if self.latest_frame else 0
