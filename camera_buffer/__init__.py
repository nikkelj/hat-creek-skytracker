"""
Threaded camera capture system with circular buffer for Hat Creek Skytracker
"""

import threading
import time
import numpy as np
import pygame
from collections import deque
from datetime import datetime, timezone

# Maintain import compatibility - CircularBuffer now replaced with simple latest_frame approach
class CircularBuffer:
    """Empty placeholder class for import compatibility"""
    def __init__(self, size=30, max_frame_size=(1920, 1280, 3)):
        pass

class CameraThread(threading.Thread):
    """Dedicated thread for camera capture - MAXIMUM SPEED VERSION"""

    def __init__(self, camera_index, camera_cap, buffer_size=30, target_fps=10):
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
        self.latest_frame = None  # Simple single frame buffer - no locks!

        # Get camera properties
        try:
            self.camera_props = camera_cap.get_camera_property()
        except Exception as e:
            self.camera_props = {'MaxWidth': 1920, 'MaxHeight': 1280}

    def get_latest_frame(self):
        """Get the most recent frame - no locks, just return the reference"""
        return self.latest_frame

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
        return 1

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

            # Step 4: Handle data conversion (copied from zwoasi capture method)
            whbi = self.camera_cap.get_roi_format()
            shape = [whbi[1], whbi[0]]  # height, width

            if whbi[3] == asi.ASI_IMG_RAW8 or whbi[3] == asi.ASI_IMG_Y8:
                img = np.frombuffer(data, dtype=np.uint8)
            elif whbi[3] == asi.ASI_IMG_RAW16:
                img = np.frombuffer(data, dtype=np.uint16)
            elif whbi[3] == asi.ASI_IMG_RGB24:
                img = np.frombuffer(data, dtype=np.uint8)
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
        """Main thread execution loop - MAXIMUM SPEED VERSION"""
        print(f"Camera thread {self.camera_index} starting...")

        # Reset FPS tracking variables for current session
        self.frame_counter = 0
        self.fps_timer = time.time()

        while self.running:
            # SKIP ALL THROTTLING FOR MAXIMUM SPEED
            # Remove frame interval checking entirely - let camera run at max speed
            current_time = time.perf_counter()

            # REMOVED: Frame interval throttling
            # REMOVED: Sleep calls in main loop
            # This is critical for 300fps - we don't want ANY artificial delays

            try:
                if not self.camera_cap:
                    self.running = False
                    continue

                # ULTRA-FAST CAPTURE - bypass zwoasi's slow polling capture method
                start_time = time.perf_counter()

                # Custom high-speed capture implementation
                raw_frame = self._ultra_fast_capture()
                capture_time = time.perf_counter() - start_time

                # Only handle frames that exist and are not timed out
                if raw_frame is not None and raw_frame.size > 0 and capture_time < self.capture_timeout:
                    # Process frame - minimal error handling
                    try:
                        surface = self._process_raw_frame(raw_frame)
                        if surface is not None:
                            # Create new frame data - direct assignment, no buffer overhead
                            self.latest_frame = {
                                'frame': surface,
                                'timestamp': current_time,
                                'datetime_utc': datetime.now(timezone.utc).isoformat(),  # Real-time UTC timestamp
                                'datetime_local': datetime.now().isoformat(),  # Real-time local timestamp
                                'camera_index': self.camera_index,
                                'capture_time': capture_time
                            }
                            # Direct counters - no buffer locking overhead!
                            self.frame_count += 1
                            self.frame_counter += 1  # Use instance variable for FPS calculation
                            self.error_count = 0  # Reset on success

                    except Exception:
                        self.error_count += 1
                        continue

                self.last_capture = current_time

            except Exception:
                self.error_count += 1
                continue

            # REMOVED all error handling and most debug prints for speed
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
