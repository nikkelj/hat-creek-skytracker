"""
Threaded camera capture system with circular buffer for Hat Creek Skytracker
"""

import threading
import time
import numpy as np
import pygame
from collections import deque
from datetime import datetime, timezone

class CircularBuffer:
    """Thread-safe circular buffer for camera frames"""

    def __init__(self, size=30, max_frame_size=(1920, 1280, 3)):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.Lock()
        self.max_frame_size = max_frame_size
        # Pre-allocate reusable frame data objects
        self.reusable_frame_data = [{
            'frame': None,
            'timestamp': 0,
            'datetime_utc': '',
            'datetime_local': '',
            'camera_index': 0,
            'camera_props': {'MaxWidth': 1920, 'MaxHeight': 1280},
            'capture_time': 0,
            'raw_buffer': None
        } for _ in range(size)]

    def add_frame(self, frame_data):
        """Add a new frame to the buffer"""
        try:
            with self.lock:
                self.buffer.append(frame_data)
        except Exception as e:
            print(f"Buffer add error: {str(e)}")

    def get_latest_frame(self):
        """Get the most recent frame from the buffer"""
        try:
            with self.lock:
                if len(self.buffer) > 0:
                    return self.buffer[-1]
                return None
        except Exception as e:
            print(f"Buffer get latest error: {str(e)}")
            return None

    def get_buffer_size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

class CameraThread(threading.Thread):
    """Dedicated thread for camera capture"""

    def __init__(self, camera_index, camera_cap, buffer_size=30, target_fps=10):
        super().__init__()
        self.camera_index = camera_index
        self.camera_cap = camera_cap
        self.buffer = CircularBuffer(buffer_size)
        self.running = True
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_capture = 0
        self.capture_timeout = 5.0  # Increased timeout to 5 seconds
        self.error_count = 0
        self.max_errors = 5
        self.frame_count = 0
        self.fps = 0.0
        self.last_successful_capture = 0
        self.capture_attempts = 0
        # Cache timestamp strings to avoid flicker
        self.last_local_timestamp = ""
        self.last_utc_timestamp = ""

        # Get camera properties for error handling
        try:
            self.camera_props = camera_cap.get_camera_property()
        except Exception as e:
            print(f"Camera {camera_index} property error: {str(e)}")
            self.camera_props = {'MaxWidth': 1920, 'MaxHeight': 1280}

        # Pre-allocate memory buffers for different ROI sizes
        self._init_frame_buffers()

        print(f"Camera thread {camera_index} initialized with target FPS: {target_fps}, timeout: {self.capture_timeout}s")

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
        """Process raw frame data using pre-allocated buffers (no dynamic memory allocation)"""
        try:
            # Get actual frame dimensions
            if raw_frame.ndim == 3:
                frame_height, frame_width = raw_frame.shape[:2]
                frame_channels = raw_frame.shape[2]
            elif raw_frame.ndim == 2:
                frame_height, frame_width = raw_frame.shape
                frame_channels = 1
            else:
                print(f"Unsupported frame dimensionality: {raw_frame.ndim}")
                return None

            # Handle different camera types and data formats
            if self.camera_props.get('IsColorCam', False):
                # Color camera - handle BGR to RGB conversion
                if frame_channels == 3:
                    # Optimized: use numpy's transpose for BGR->RGB
                    rgb_frame = np.ascontiguousarray(raw_frame[:, :, ::-1])  # Faster than manual assignment
                else:
                    print(f"Unexpected color frame channels: {frame_channels}")
                    return None
            else:
                # Monochrome camera - expand to RGB using broadcasting (fastest method)
                if frame_channels == 1:
                    # Stack the mono channel into RGB - optimized broadcasting
                    rgb_frame = np.repeat(raw_frame[:, :, np.newaxis], 3, axis=2)
                    rgb_frame = rgb_frame.astype(np.uint8)
                else:
                    print(f"Unexpected mono frame channels: {frame_channels}")
                    return None

            # Create pygame surface directly from the RGB array (skip tobytes() if possible)
            # Try direct creation first, fallback to buffer method
            try:
                surface = pygame.image.frombuffer(rgb_frame.data, (frame_width, frame_height), 'RGB')
            except:
                # Fallback to tobytes method
                surface = pygame.image.frombuffer(rgb_frame.tobytes(), (frame_width, frame_height), 'RGB')

            return surface

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            return None

    def run(self):
        """Main thread execution loop"""
        print(f"Camera thread {self.camera_index} starting...")

        fps_timer = time.perf_counter()  # Use higher precision timer
        frame_counter = 0

        while self.running:
            current_time = time.perf_counter()  # Use higher precision timer for frame timing

            # Reduce throttling for high FPS - only throttle if we're ahead by more than 1ms
            # At 300 FPS, target interval is ~0.0033s = 3.3ms
            if current_time - self.last_capture < (self.frame_interval - 0.001):  # Allow 1ms ahead
                time.sleep(0.0001)  # Minimal sleep to prevent CPU hogging
                continue

            try:
                if not self.camera_cap:
                    print(f"Camera {self.camera_index} capture object is None")
                    self.running = False
                    continue

                # Simple blocking capture for now (may need to be adjusted based on camera API)
                start_time = time.perf_counter()
                try:
                    raw_frame = self.camera_cap.capture()
                    capture_time = time.perf_counter() - start_time
                except Exception as e:
                    print(f"Camera {self.camera_index} capture failed: {str(e)}")
                    self.error_count += 1
                    self.last_capture = current_time
                    time.sleep(0.1)
                    continue

                # Check for timeout - bailout early if no data to reduce overhead
                if capture_time > self.capture_timeout:
                    print(f"Camera {self.camera_index} capture timeout ({capture_time:.2f}s)")
                    self.error_count += 1
                    self.last_capture = current_time
                    continue

                # Check if capture returned data - early exit for null/empty frames
                if raw_frame is None or raw_frame.size == 0:
                    # No console output for this common case to avoid spam
                    self.error_count += 1
                    self.last_capture = current_time
                    continue

                # Process and store frame using pre-allocated buffers
                try:
                    surface = self._process_raw_frame(raw_frame)

                    if surface is not None:
                        frame_data = {
                            'frame': surface,
                            'timestamp': current_time,
                            'datetime_utc': datetime.now(timezone.utc).isoformat(),  # Real-time UTC timestamp
                            'datetime_local': datetime.now().isoformat(),  # Real-time local timestamp
                            'camera_index': self.camera_index,
                            'capture_time': capture_time
                        }

                        # Add to buffer
                        self.buffer.add_frame(frame_data)

                        # Reset error count on successful capture
                        self.error_count = 0
                        self.frame_count += 1
                        frame_counter += 1
                    else:
                        self.error_count += 1
                        print(f"Camera {self.camera_index} frame processing failed")

                except Exception as e:
                    self.error_count += 1
                    print(f"Camera {self.camera_index} frame processing error: {str(e)}")

                self.last_capture = current_time

            except Exception as e:
                self.error_count += 1
                print(f"Camera {self.camera_index} capture error: {str(e)}")
                time.sleep(0.1)  # Brief pause on error

            # Check if we have too many consecutive errors
            if self.error_count > self.max_errors:
                print(f"Camera {self.camera_index} has {self.error_count} consecutive errors, pausing capture")
                time.sleep(1.0)  # Longer pause when many errors
                self.error_count = 0

            # Calculate FPS every second - optimized timing consistency
            current_wall_time = time.time()  # Use wall time for FPS display calculations
            if current_wall_time - fps_timer >= 1.0:
                self.fps = frame_counter / (current_wall_time - fps_timer)
                fps_timer = current_wall_time
                frame_counter = 0

            # Remove sleep for maximum FPS - only yield briefly if too far ahead
            if current_time - self.last_capture > self.frame_interval + 0.01:  # More than 10ms ahead
                time.sleep(0.00001)  # Microsecond sleep to yield CPU

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
        latest = self.buffer.get_latest_frame()
        if latest:
            return latest.get('datetime_utc', "")
        return ""

    def get_local_timestamp(self):
        """Get latest frame local timestamp"""
        latest = self.buffer.get_latest_frame()
        if latest:
            return latest.get('datetime_local', "")
        return ""

    def get_buffer_size(self):
        """Get current buffer size"""
        return self.buffer.get_buffer_size()
