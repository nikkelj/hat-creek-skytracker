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

    def __init__(self, size=30):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.Lock()

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

        # Get camera properties for error handling
        try:
            self.camera_props = camera_cap.get_camera_property()
        except Exception as e:
            print(f"Camera {camera_index} property error: {str(e)}")
            self.camera_props = {'MaxWidth': 1920, 'MaxHeight': 1280}

        print(f"Camera thread {camera_index} initialized with target FPS: {target_fps}, timeout: {self.capture_timeout}s")

    def run(self):
        """Main thread execution loop"""
        print(f"Camera thread {self.camera_index} starting...")

        fps_timer = time.time()
        frame_counter = 0

        while self.running:
            current_time = time.time()

            # Control capture rate to prevent timeouts
            if current_time - self.last_capture < self.frame_interval:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue

            try:
                if not self.camera_cap:
                    print(f"Camera {self.camera_index} capture object is None")
                    self.running = False
                    continue

                # Simple blocking capture for now (may need to be adjusted based on camera API)
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

                # Process and store frame
                try:
                    # Handle different camera types and data formats
                    if self.camera_props.get('IsColorCam', False):
                        # Color camera - handle BGR to RGB conversion
                        if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
                            # Reverse BGR to RGB
                            rgb_frame = raw_frame[..., ::-1].copy()
                        else:
                            rgb_frame = raw_frame
                    else:
                        # Monochrome camera
                        if raw_frame.ndim == 2:
                            # Replicate single channel to RGB for display
                            rgb_frame = np.stack([raw_frame, raw_frame, raw_frame], axis=-1)
                        else:
                            rgb_frame = raw_frame

                    # Ensure we have a valid numpy array
                    if isinstance(rgb_frame, np.ndarray) and rgb_frame.size > 0:
                        # Create pygame surface from numpy array
                        try:
                            surface = pygame.image.frombuffer(
                                rgb_frame.tobytes(),
                                rgb_frame.shape[1::-1],  # Width, Height
                                'RGB'
                            )

                            # Create frame data with timestamp
                            frame_data = {
                                'frame': surface,
                                'timestamp': current_time,
                                'datetime_utc': datetime.now(timezone.utc).isoformat(),
                                'datetime_local': datetime.now().isoformat(),
                                'camera_index': self.camera_index,
                                'capture_time': capture_time
                            }

                            # Add to buffer
                            self.buffer.add_frame(frame_data)

                            # Reset error count on successful capture
                            self.error_count = 0
                            self.frame_count += 1
                            frame_counter += 1

                        except Exception as e:
                            self.error_count += 1
                            print(f"Camera {self.camera_index} surface creation error: {str(e)}")
                    else:
                        self.error_count += 1
                        print(f"Camera {self.camera_index} invalid frame data")

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

            # Calculate FPS every second
            if time.time() - fps_timer >= 1.0:
                self.fps = frame_counter / (time.time() - fps_timer)
                fps_timer = time.time()
                frame_counter = 0

            # Small sleep to prevent busy waiting
            time.sleep(0.001)

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
