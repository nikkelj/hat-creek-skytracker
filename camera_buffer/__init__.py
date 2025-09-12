"""
Threaded camera capture system with circular buffer for Hat Creek Skytracker
"""

import threading
import time
import numpy as np
import pygame
from collections import deque
from datetime import datetime, timezone

class SimpleBuffer:
    """Simple lock-free buffer for latest frame only"""

    def __init__(self):
        self.latest_frame = None
        self.frame_count = 0

    def add_frame(self, frame_data):
        """Replace the latest frame (lock-free)"""
        self.latest_frame = frame_data
        self.frame_count += 1

    def get_latest_frame(self):
        """Get the latest frame (lock-free)"""
        return self.latest_frame

    def get_buffer_size(self):
        """Return frame count"""
        return self.frame_count

class CameraThread(threading.Thread):
    """Dedicated thread for camera capture"""

    def __init__(self, camera_index, camera_cap, buffer_size=30, target_fps=10):
        super().__init__()
        self.camera_index = camera_index
        self.camera_cap = camera_cap
        self.buffer = SimpleBuffer()
        self.running = True
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.last_capture = 0
        # Dynamic timeout based on expected capture speeds
        if target_fps == 0:  # Unlimited FPS (small ROI)
            self.capture_timeout = 0.5  # Short timeout for fast captures
        elif target_fps >= 60:
            self.capture_timeout = 1.0  # Medium timeout for fast captures
        else:
            self.capture_timeout = 5.0  # Long timeout for slow captures
        self.error_count = 0
        self.max_errors = 5
        self.frame_count = 0
        self.fps = 0.0
        self.last_successful_capture = 0
        self.capture_attempts = 0

        # Get camera properties for error handling
        try:
            self.camera_props = camera_cap.get_camera_property()
        except Exception as e:
            print(f"Camera {camera_index} property error: {str(e)}")
            self.camera_props = {'MaxWidth': 1920, 'MaxHeight': 1280}

        self.is_color = self.camera_props.get('IsColorCam', False)

        # Pre-allocated rotating buffers to eliminate array copying
        self.last_frame_shape = None
        self.prealloc_rgb = np.zeros((self.camera_props.get('MaxHeight', 1280), self.camera_props.get('MaxWidth', 1920), 3), dtype=np.uint8)
        self.prealloc_rgb2 = np.zeros((self.camera_props.get('MaxHeight', 1280), self.camera_props.get('MaxWidth', 1920), 3), dtype=np.uint8)
        self.buffer_index = 0  # Toggle between buffers
        self.current_roi_size = -1  # Will be updated when ROI changes

        # Surface creation will be done per frame to ensure each frame gets its own surface

        print(f"Camera thread {camera_index} initialized with target FPS: {target_fps}, timeout: {self.capture_timeout}s")

    def run(self):
        """Main thread execution loop"""
        print(f"Camera thread {self.camera_index} starting...")

        fps_timer = time.time()
        frame_counter = 0

        while self.running:
            current_time = time.time()

            # Control capture rate to prevent timeouts - skip for unlimited FPS (target_fps=0)
            if self.target_fps > 0 and current_time - self.last_capture < self.frame_interval:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue

            # Fast path for unlimited FPS - optimized capture method
            if self.target_fps == 0:
                # Ultra-fast capture with minimal polling
                try:
                    current_time = time.time()

                    # Call the standard capture but override polling parameters
                    # This bypasses the default 0.01s sleep polling
                    self.camera_cap.start_exposure()

                    # Ultra-fast polling - check status in tight loop instead of sleeping
                    start_poll = time.perf_counter()
                    while self.camera_cap.get_exposure_status() == 1:  # ASI_EXP_WORKING
                        # Spin-wait instead of sleep for minimal latency
                        if time.perf_counter() - start_poll > 0.1:  # 100ms timeout max
                            break

                    # Get data immediately
                    data = self.camera_cap.get_data_after_exposure()

                    # Minimal processing with existing buffers
                    if data and len(data) > 0:
                        # Use the standard buffer processing but mark as fast capture
                        frame_data = {
                            'raw_data': data,  # Keep as raw to avoid processing
                            'timestamp': current_time,
                            'camera_index': self.camera_index,
                            'fast_capture': True
                        }
                        self.buffer.add_frame(frame_data)
                        self.error_count = 0
                        self.frame_count += 1
                    continue
                except Exception:
                    continue


            else:
                # Standard path with full error handling
                if not self.camera_cap:
                    print(f"Camera {self.camera_index} capture object is None")
                    self.running = False
                    continue

                start_time = time.time()
                try:
                    raw_frame = self.camera_cap.capture()
                    capture_time = time.time() - start_time
                except Exception as e:
                    print(f"Camera {self.camera_index} capture failed: {str(e)}")
                    self.error_count += 1
                    self.last_capture = current_time
                    time.sleep(0.1)
                    continue

                # Check for timeout
                if capture_time > self.capture_timeout:
                    print(f"Camera {self.camera_index} capture timeout ({capture_time:.2f}s)")
                    self.error_count += 1
                    self.last_capture = current_time
                    continue

                # Check if capture returned data
                if raw_frame is None or raw_frame.size == 0:
                    self.error_count += 1
                    print(f"Camera {self.camera_index} no frame data")
                    self.last_capture = current_time
                    continue

            # Process and store frame (common for both paths)
            try:
                # Handle buffer resizing for ROI changes - resize both rotating buffers
                new_shape = (raw_frame.shape[0], raw_frame.shape[1], 3)
                if raw_frame.shape != self.last_frame_shape:
                    # Resize both buffers when ROI changes
                    self.prealloc_rgb = np.zeros(new_shape, dtype=np.uint8)
                    self.prealloc_rgb2 = np.zeros(new_shape, dtype=np.uint8)
                    self.last_frame_shape = raw_frame.shape

                # Toggle between pre-allocated buffers to avoid copying
                if self.buffer_index == 0:
                    active_buffer = self.prealloc_rgb
                    self.buffer_index = 1
                else:
                    active_buffer = self.prealloc_rgb2
                    self.buffer_index = 0

                # Handle different camera types and data formats using active buffer
                if self.is_color:
                    # Color camera - BGR to RGB conversion
                    if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
                        active_buffer[:, :, 0] = raw_frame[:, :, 2]  # R = B
                        active_buffer[:, :, 1] = raw_frame[:, :, 1]  # G = G
                        active_buffer[:, :, 2] = raw_frame[:, :, 0]  # B = R
                else:
                    # Monochrome camera - replicate channel
                    if raw_frame.ndim == 2:
                        active_buffer[:, :, 0] = raw_frame
                        active_buffer[:, :, 1] = raw_frame
                        active_buffer[:, :, 2] = raw_frame

                # Create frame data with minimal dictionary (no copy needed - buffer rotation provides thread safety)
                frame_data = {
                    'rgb_array': active_buffer,
                    'timestamp': current_time,
                    'camera_index': self.camera_index
                }



                # Add to buffer
                self.buffer.add_frame(frame_data)

                # Update counters
                self.error_count = 0
                self.frame_count += 1
                frame_counter += 1
                self.last_capture = current_time

            except Exception as e:
                self.error_count += 1
                if self.target_fps > 0:  # Only print errors for standard path
                    print(f"Camera {self.camera_index} frame processing error: {str(e)}")
                continue

            # Check if we have too many consecutive errors
            if self.error_count > self.max_errors:
                print(f"Camera {self.camera_index} has {self.error_count} consecutive errors, pausing capture")
                time.sleep(1.0)  # Longer pause when many errors
                self.error_count = 0

            # Calculate FPS every second
            if time.time() - fps_timer >= 1.0:
                self.fps = frame_counter / (time.time() - fps_timer)
                fps_timer = time.time()
                frame_counter = 0

            # Small sleep to prevent busy waiting - skip for unlimited FPS
            if self.target_fps > 0:
                time.sleep(0.001)

    def set_target_fps(self, new_target_fps):
        """Update the target FPS and frame interval"""
        self.target_fps = new_target_fps
        self.frame_interval = 1.0 / new_target_fps if new_target_fps > 0 else 0
        # Update timeout based on new FPS
        if new_target_fps == 0:  # Unlimited FPS (small ROI)
            self.capture_timeout = 0.5
        elif new_target_fps >= 60:
            self.capture_timeout = 1.0
        else:
            self.capture_timeout = 5.0
        print(f"Camera thread {self.camera_index} target FPS updated to: {new_target_fps}, timeout: {self.capture_timeout}s")

    def update_roi_size(self, roi_size):
        """Update current ROI size for buffer management"""
        self.current_roi_size = roi_size

    def stop(self):
        """Stop the camera thread"""
        print(f"Stopping camera thread {self.camera_index}")
        self.running = False
        self.join(timeout=2.0)

    def get_buffer_fps(self):
        """Get current buffer FPS"""
        return self.fps

    def get_utc_timestamp(self):
        """Generate UTC timestamp for latest frame"""
        latest = self.buffer.get_latest_frame()
        if latest:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(latest.get('timestamp', 0), tz=timezone.utc).isoformat()
        return ""

    def get_local_timestamp(self):
        """Generate local timestamp for latest frame"""
        latest = self.buffer.get_latest_frame()
        if latest:
            from datetime import datetime
            return datetime.fromtimestamp(latest.get('timestamp', 0)).isoformat()
        return ""

    def get_buffer_size(self):
        """Get current buffer size"""
        return self.buffer.get_buffer_size()
