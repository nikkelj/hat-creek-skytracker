import pygame
import zwoasi as asi
import numpy as np
import json
from camera_buffer import CameraThread, CircularBuffer

# Create global capture manager instance
from capture_manager import CaptureManager
capture_manager = CaptureManager()
import config


class CameraState:
    """Class to encapsulate camera state and settings"""
    def __init__(self, index=0, alignment_rotation=0.0):
        # Connection state
        self.connected = False
        self.cap = None
        self.prop = None
        self.index = index
        self.width_res = 1920  # Default resolution
        self.height_res = 1280

        # Camera frame
        self.frame = None
        # Monotonic sequence of the latest displayed frame (from the capture
        # thread's buffer_sequence) and a one-deep cache of its processed
        # display surface, so the render loop can skip re-gamma/scale/rotate
        # when the same frame is shown across multiple UI frames.
        self.frame_seq = None
        self._feed_cache = None

        # ROI settings
        self.roi_size = -1  # -1 = no ROI selected
        self.roi_x = 0.5
        self.roi_y = 0.5

        # Camera controls
        self.gain = 1
        self.exposure = 10000  # 10ms default
        self.alignment_rotation = alignment_rotation  # Degrees, positive = counter-clockwise
        self.gamma = 0.1  # Gamma correction value
        self.gamma_enabled = False  # Gamma correction toggle

        # Performance tracking
        self.fps = 0.0
        self.utc_ts = ""
        self.local_ts = ""

        # Threads
        self.thread = None
        self.threads_running = False

    def __del__(self):
        """Safe cleanup method to prevent AttributeError during garbage collection"""
        try:
            if self.thread is not None:
                self.thread.stop()
                self.thread = None
        except:
            pass

        try:
            if self.cap is not None:
                self.cap.close()
                self.cap = None
        except:
            pass

        # Clear other attributes to prevent any attribute errors
        self.prop = None
        self.frame = None


class CameraManager:
    """Manages camera operations and state - supports n cameras"""
    def __init__(self, num_cameras=2):
        self.cameras = []  # Dynamic list to support any number of cameras
        self.initialize_cameras(num_cameras)

        # Hardware simulator (set by main.py). When set and enabled, connect_camera
        # produces synthetic frames through a duck-typed SimCap instead of ASI.
        self.simulator = None

        # Screening/layout settings
        self.CAMERA_CONNECT_BUTTONS = [None] * num_cameras
        self.CAMERA_DISCONNECT_BUTTONS = [None] * num_cameras
        self.COMBINED_VIEW_BUTTON_RECT = None
        self.CAMERA_OPACITY_SLIDER_RECT = None
        self.CAMERA_OPACITY_SLIDER_HANDLE_RECT = None

        # Camera button states (dynamically generated)
        self.button_states = {}
        for i in range(num_cameras):
            prefix = f"camera{i}_"
            self.button_states.update({
                f"{prefix}connect": {"hover": False, "clicked": False},
                f"{prefix}disconnect": {"hover": False, "clicked": False},
                f"{prefix}gain_slider": {"hover": False, "dragging": False},
                f"{prefix}exposure_slider": {"hover": False, "dragging": False},
                f"{prefix}alignment_rotation_slider": {"hover": False, "dragging": False},
            })

        # Config control button states
        self.button_states.update({
            "reset_config": {"hover": False, "clicked": False},
            "save_config": {"hover": False, "clicked": False},
        })

        # Combined view settings
        self.combined_view_toggle = False
        self.camera_opacities = [0.5] * num_cameras

        # UI control rects
        self.COMBINED_VIEW_BUTTON_RECT = None
        self.CAMERA_OPACITY_SLIDER_RECT = None
        self.CAMERA_OPACITY_SLIDER_HANDLE_RECT = None

        # Camera connect/disconnect button rects
        self.CAMERA1_CONNECT_BUTTON_RECT = None
        self.CAMERA1_DISCONNECT_BUTTON_RECT = None
        self.CAMERA2_CONNECT_BUTTON_RECT = None
        self.CAMERA2_DISCONNECT_BUTTON_RECT = None



    def _initialize_control_rects(self, menu_screen, total_width, total_height):
        """Initialize UI control rectangles with proper dimensions"""
        # Initialize combined view button and controls with display dimensions
        self.COMBINED_VIEW_BUTTON_RECT = pygame.Rect(total_width // 2 + 50, total_height - 60, 100, 20)
        self.CAMERA_OPACITY_SLIDER_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 300, 20)
        self.CAMERA_OPACITY_SLIDER_HANDLE_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 20, 20)

        # Initialize camera connect/disconnect button rectangles
        self.CAMERA1_CONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 20, total_height - 60, 60, 20)
        self.CAMERA1_DISCONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 80, total_height - 60, 60, 20)
        self.CAMERA2_CONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 20 - 200, total_height - 60, 60, 20)
        self.CAMERA2_DISCONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 80 - 200, total_height - 60, 60, 20)

        # Update dictionary with actual rects
        self.CAMERA_CONNECT_BUTTONS = [self.CAMERA1_CONNECT_BUTTON_RECT, self.CAMERA2_CONNECT_BUTTON_RECT]
        self.CAMERA_DISCONNECT_BUTTONS = [self.CAMERA1_DISCONNECT_BUTTON_RECT, self.CAMERA2_DISCONNECT_BUTTON_RECT]

    def initialize_cameras(self, count):
        """Initialize n cameras dynamically"""
        for i in range(count):
            camera = CameraState()
            camera.index = i
            camera.name = f"Camera {i+1}"
            self.cameras.append(camera)

    def get_camera(self, index):
        """Get camera by index with bounds checking"""
        if 0 <= index < len(self.cameras):
            return self.cameras[index]
        return None



    def connect_camera(self, camera_index, update_status_callback=None):
        """Connect camera by index"""
        camera = self.get_camera(camera_index)
        if not camera:
            return False

        if camera.connected:
            if update_status_callback:
                update_status_callback(f"Camera {camera_index+1} already connected")
            return True

        # Simulation: back the camera with a duck-typed SimCap so the normal
        # CameraThread pipeline renders synthetic frames.
        if self.simulator is not None and self.simulator.sim_enabled():
            from simulator import SimCap
            camera.cap = SimCap(camera.index, self.simulator)
            camera.prop = camera.cap.get_camera_property()
            camera.cap.set_image_type(asi.ASI_IMG_RAW8)
            camera.connected = True
            camera.width_res = camera.prop.get('MaxWidth', 960)
            camera.height_res = camera.prop.get('MaxHeight', 720)
            self._start_camera_thread(camera)
            if update_status_callback:
                update_status_callback(f"Camera {camera_index+1} connected (SIMULATED): {camera.width_res}x{camera.height_res}")
            return True

        try:
            camera.cap = asi.Camera(camera.index)
            camera.prop = camera.cap.get_camera_property()

            if camera.cap:
                # Set video format
                if camera.prop['IsColorCam']:
                    camera.cap.set_image_type(asi.ASI_IMG_RGB24)
                else:
                    camera.cap.set_image_type(asi.ASI_IMG_RAW8)

                # Set camera controls
                camera.cap.set_control_value(asi.ASI_EXPOSURE, camera.exposure)
                camera.cap.set_control_value(asi.ASI_GAIN, camera.gain)
                camera.connected = True

                # Set dynamic camera resolution
                camera.width_res = camera.prop.get('MaxWidth', 1920)
                camera.height_res = camera.prop.get('MaxHeight', 1280)

            # Start camera thread for continuous capture
            self._start_camera_thread(camera)

            if update_status_callback:
                update_status_callback(f"Camera {camera_index+1} connected successfully: {camera.width_res}x{camera.height_res} at {camera.thread.target_fps} FPS")
            return True

        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Camera {camera_index+1} connection error: {str(e)}")
            # Properly clean up any partially initialized camera object
            camera.cap = None
            camera.prop = None
            camera.connected = False
            return False

    def disconnect_camera(self, camera_index, update_status_callback=None):
        """Disconnect camera by index"""
        camera = self.get_camera(camera_index)
        if not camera:
            return

        self._stop_camera_thread(camera)
        if camera.cap is not None:
            camera.cap = None
        camera.connected = False
        camera.frame = None
        # Reset ROI settings to defaults
        camera.roi_size = -1
        camera.roi_x = 0.5
        camera.roi_y = 0.5
        if update_status_callback:
            update_status_callback(f"Camera {camera_index+1} disconnected")

    def _start_camera_thread(self, camera, config_state=None):
        """Start capture thread for camera with configurable circular buffer"""
        if camera.threads_running or not camera.connected or camera.cap is None:
            return

        # Use configurable buffer size from config_state if provided, else use default
        buffer_size = 1000  # Default fallback
        if config_state:
            buffer_size = getattr(config_state, 'buffer_size', 1000)

        camera.threads_running = True
        camera.thread = CameraThread(camera.index, camera.cap, buffer_size=buffer_size, target_fps=30)
        camera.thread.start()

        if config_state:
            # Update camera settings from config after thread start
            camera.gain = int(float(config_state.camera_configs[f"camera{camera.index + 1}"]["gain"]))
            camera.exposure = int(float(config_state.camera_configs[f"camera{camera.index + 1}"]["exposure"]))
            camera.gamma = float(config_state.camera_configs[f"camera{camera.index + 1}"]["gamma"])
            camera.gamma_enabled = bool(config_state.camera_configs[f"camera{camera.index + 1}"]["gamma_enabled"])

            # Apply settings to camera if connected
            if camera.cap:
                self.set_camera_gain(camera.index, camera.gain)
                self.set_camera_exposure(camera.index, camera.exposure)

    def _stop_camera_thread(self, camera):
        """Stop capture thread for camera"""
        camera.threads_running = False
        if camera.thread is not None:
            camera.thread.stop()
            camera.thread = None

    def get_num_cameras(self):
        """Get number of cameras available"""
        # In simulation, report the sim cameras so UI features that gate on
        # camera availability (e.g. the joystick-loop camera feeds) work.
        if self.simulator is not None and self.simulator.sim_enabled():
            return len(self.cameras)
        try:
            return asi.get_num_cameras()
        except:
            return 0

    def get_camera_names(self):
        """Get list of camera names"""
        try:
            return asi.list_cameras()
        except:
            return []

    def set_camera_gain(self, camera_index, gain_value, update_status_callback=None):
        """Set camera gain by index"""
        camera = self.get_camera(camera_index)
        if not camera:
            return

        camera.gain = gain_value
        if camera.connected and camera.cap:
            try:
                camera.cap.set_control_value(asi.ASI_GAIN, camera.gain)
                if update_status_callback:
                    update_status_callback(f"Camera {camera_index+1} gain set to {camera.gain}")
            except Exception as e:
                if update_status_callback:
                    update_status_callback(f"Failed to set Camera {camera_index+1} gain: {str(e)}")

    def set_camera_exposure(self, camera_index, exposure_value, update_status_callback=None):
        """Set camera exposure by index"""
        camera = self.get_camera(camera_index)
        if not camera:
            return

        camera.exposure = exposure_value
        if camera.connected and camera.cap:
            try:
                camera.cap.set_control_value(asi.ASI_EXPOSURE, camera.exposure)
                if update_status_callback:
                    update_status_callback(f"Camera {camera_index+1} exposure set to {camera.exposure} µs")
            except Exception as e:
                if update_status_callback:
                    update_status_callback(f"Failed to set Camera {camera_index+1} exposure: {str(e)}")

    def set_camera_roi(self, camera_index, update_status_callback, roi_x, roi_y, roi_size):
        """Set camera ROI by index using the proper API"""
        camera = self.get_camera(camera_index)
        if not camera or not camera.connected or camera.cap is None or camera.prop is None:
            return False

        try:
            # Handle deselected ROI (set to -1) - reset to full resolution
            if roi_size == -1:
                max_width = camera.prop.get('MaxWidth', camera.width_res)
                max_height = camera.prop.get('MaxHeight', camera.height_res)
                # Get current ROI format to preserve bins and image_type
                current_roi_format = camera.cap.get_roi_format()
                # Set to full resolution
                camera.cap.set_roi_format(max_width, max_height, current_roi_format[2], current_roi_format[3])
                camera.cap.set_roi_start_position(0, 0)

                # Update camera object state
                camera.roi_size = -1
                camera.roi_x = 0.5
                camera.roi_y = 0.5

                if update_status_callback:
                    update_status_callback(f"Camera {camera_index+1} ROI reset to full: (0, 0) {max_width}x{max_height}")
                return True

            # Get current ROI size
            roi_width, roi_height = roi_sizes[roi_size]

            # Calculate pixel coordinates - use actual camera resolution from properties
            max_width = camera.prop.get('MaxWidth', camera.width_res)
            max_height = camera.prop.get('MaxHeight', camera.height_res)

            roi_w = int(roi_width * max_width)
            roi_h = int(roi_height * max_height)

            # Convert center point to top-left corner for camera API
            center_x_pixel = roi_x * max_width
            center_y_pixel = roi_y * max_height
            start_x = int(center_x_pixel - (roi_w / 2))
            start_y = int(center_y_pixel - (roi_h / 2))

            # Ensure ROI stays within bounds
            start_x = max(0, min(start_x, max_width - roi_w))
            start_y = max(0, min(start_y, max_height - roi_h))
            roi_w = min(roi_w, max_width - start_x)
            roi_h = min(roi_h, max_height - start_y)

            # Apply camera-specific ROI alignment requirements
            # Height must be multiple of 2, Width must be multiple of 8
            roi_h = (roi_h // 2) * 2  # Round down to nearest multiple of 2
            if roi_h == 0:
                roi_h = 2  # Minimum height of 2
            roi_w = (roi_w // 8) * 8  # Round down to nearest multiple of 8
            if roi_w == 0:
                roi_w = 8  # Minimum width of 8

            # Ensure ROI still fits within camera bounds after alignment
            roi_w = min(roi_w, max_width - start_x)
            roi_h = min(roi_h, max_height - start_y)

            # Get current ROI format to preserve bins and image_type (returned as list)
            current_roi_format = camera.cap.get_roi_format()

            # Apply ROI to camera using correct API
            try:
                camera.cap.set_roi_format(roi_w, roi_h, current_roi_format[2], current_roi_format[3])
                camera.cap.set_roi_start_position(start_x, start_y)

                # Update camera object state
                camera.roi_size = roi_size
                camera.roi_x = roi_x
                camera.roi_y = roi_y

                if update_status_callback:
                    update_status_callback(f"Camera {camera_index+1} ROI set: ({start_x}, {start_y}) {roi_w}x{roi_h}")
                return True
            except AttributeError:
                # ROI setting not supported by this camera/driver
                if update_status_callback:
                    update_status_callback(f"Camera {camera_index+1} ROI setting not supported by hardware")
                return False
        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Error setting Camera {camera_index+1} ROI: {str(e)}")
            return False



# Create global camera manager instance
camera_manager = CameraManager()

# ROI constants moved to global scope for easy access
roi_sizes = [
    (1.0, 1.0),
    (0.5, 0.5),
    (0.25, 0.25),
    (0.125, 0.125),
    (0.0625, 0.0625),
    (0.03125, 0.03125),
]
roi_label_texts = ["1.0", ".5", ".25", ".125", ".063", ".032"]

# ==============================================================================
# GAMMA CORRECTION FUNCTIONS
# ==============================================================================

def apply_gamma_correction(surface, gamma):
    """Apply gamma correction to a pygame surface and return the corrected surface"""
    if surface is None or gamma <= 0:
        return surface

    try:
        # Convert pygame surface to numpy array
        # Use pygame.surfarray to get the pixel array
        pixel_array = pygame.surfarray.array3d(surface)

        # Apply gamma correction to each channel
        # Create lookup table for gamma correction
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)

        # Apply lookup table to each color channel
        corrected_array = lookUpTable[0][pixel_array]

        # Create new surface from corrected array
        corrected_surface = pygame.surfarray.make_surface(corrected_array)

        return corrected_surface

    except Exception as e:
        # If gamma correction fails, return original surface
        print(f"Gamma correction failed: {e}")
        return surface

# ==============================================================================
# CAMERA INITIALIZATION
# ==============================================================================

def initialize_camera_buttons(menu_screen, total_width, total_height):
    """Initialize camera control button positions after pygame is set up"""
    # Store camera connect/disconnect button rectangles in camera_manager
    camera_manager.CAMERA1_CONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 20, total_height - 60, 60, 20)
    camera_manager.CAMERA1_DISCONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 80, total_height - 60, 60, 20)
    camera_manager.CAMERA2_CONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 20 - 200, total_height - 60, 60, 20)
    camera_manager.CAMERA2_DISCONNECT_BUTTON_RECT = pygame.Rect(total_width - 200 - 80 - 200, total_height - 60, 60, 20)

    # Initialize combined view controls in camera_manager
    camera_manager.COMBINED_VIEW_BUTTON_RECT = pygame.Rect(total_width // 2 + 50, total_height - 60, 100, 20)
    camera_manager.CAMERA_OPACITY_SLIDER_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 300, 20)
    camera_manager.CAMERA_OPACITY_SLIDER_HANDLE_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 20, 20)

def update_camera_frames_from_buffers():
    """Update camera frames from thread buffers"""
    for camera in camera_manager.cameras:
        if camera.thread is not None:
            latest_frame = camera.thread.get_latest_frame()  # Use direct method instead of buffer
            if latest_frame is not None:
                camera.frame = latest_frame['frame']
                camera.frame_seq = latest_frame.get('buffer_sequence')
                camera.thread.calculate_fps()  # Calculate and update FPS internally
                camera.fps = camera.thread.fps  # Get the updated FPS value
                camera.utc_ts = camera.thread.get_utc_timestamp()
                camera.local_ts = camera.thread.get_local_timestamp()

# ==============================================================================
# CAMERA RENDERING FUNCTIONS (refactored from main.py)
# ==============================================================================

def render_sensor_calibration(menu_screen, sub_x, sub_y, sub_width, sub_height, camera1_connected, camera2_connected, camera1_name, camera2_name):
    """Render sensor calibration interface - refactored for camera_manager"""
    # Get camera references from camera_manager
    camera1 = camera_manager.get_camera(0)
    camera2 = camera_manager.get_camera(1)
    button_states = camera_manager.button_states

    menu_screen.fill((0, 0, 0), (sub_x, sub_y, sub_width, sub_height))

    # Update camera frames from buffers
    update_camera_frames_from_buffers()

    # Camera 1 display (left half)
    cam_display_width = (sub_width - 30) // 2  # Leave space between cameras
    cam_display_height = sub_height - 30

    # Handle combined view mode
    combined_view_active = (camera_manager.combined_view_toggle and
                           camera1.connected and camera2.connected and
                           camera1.frame is not None and camera2.frame is not None)

    if combined_view_active:
        try:
            # Render combined view in full view pane
            combined_width = sub_width - 20
            combined_height = sub_height - 20

            # Get camera frames and apply rotations
            camera1_frame_display = camera1.frame
            camera2_frame_display = camera2.frame

            # Apply alignment rotation to camera 1 image for combined view - keep centered and crop corners
            if camera1.alignment_rotation != 0.0:
                rotated_surface = pygame.transform.rotate(camera1_frame_display, camera1.alignment_rotation)
                # Create new surface same size as original
                desired_width = camera1_frame_display.get_width()
                desired_height = camera1_frame_display.get_height()
                camera1_frame_display = pygame.Surface((desired_width, desired_height))
                camera1_frame_display.fill((0, 0, 0))  # Fill with black background

                # Calculate offset to center the rotation (copy center portion of rotated image)
                rotated_center_x = rotated_surface.get_width() // 2
                rotated_center_y = rotated_surface.get_height() // 2
                display_center_x = desired_width // 2
                display_center_y = desired_height // 2

                # Copy center portion of rotated image to display surface
                camera1_frame_display.blit(rotated_surface,
                                         (display_center_x - rotated_center_x,
                                          display_center_y - rotated_center_y))

            # Apply alignment rotation to camera 2 image for combined view - keep centered and crop corners
            if camera2.alignment_rotation != 0.0:
                rotated_surface = pygame.transform.rotate(camera2_frame_display, camera2.alignment_rotation)
                # Create new surface same size as original
                desired_width = camera2_frame_display.get_width()
                desired_height = camera2_frame_display.get_height()
                camera2_frame_display = pygame.Surface((desired_width, desired_height))
                camera2_frame_display.fill((0, 0, 0))  # Fill with black background

                # Calculate offset to center the rotation (copy center portion of rotated image)
                rotated_center_x = rotated_surface.get_width() // 2
                rotated_center_y = rotated_surface.get_height() // 2
                display_center_x = desired_width // 2
                display_center_y = desired_height // 2

                # Copy center portion of rotated image to display surface
                camera2_frame_display.blit(rotated_surface,
                                         (display_center_x - rotated_center_x,
                                          display_center_y - rotated_center_y))

            # Now scale the (possibly rotated) frames to combined view size
            camera1_frame_display = pygame.transform.scale(camera1_frame_display, (combined_width, combined_height))
            camera2_frame_display = pygame.transform.scale(camera2_frame_display, (combined_width, combined_height))

            # Combine frames with opacity - camera1_opacity controls camera1, (1-opacity) controls camera2
            camera_opacity = camera_manager.camera_opacities[0] if camera_manager.camera_opacities else 0.5

            # Create combined surface
            combined_surface = camera1_frame_display.convert_alpha()
            combined_surface.set_alpha(int(camera_opacity * 255))

            combined_surface2 = camera2_frame_display.convert_alpha()
            combined_surface2.set_alpha(int((1 - camera_opacity) * 255))

            # Blit both frames to create combined view
            menu_screen.blit(combined_surface, (sub_x + 10, sub_y + 10))
            menu_screen.blit(combined_surface2, (sub_x + 10, sub_y + 10))

            # Draw thin red crosshair at center of combined view
            center_x = sub_x + 10 + combined_width // 2
            center_y = sub_y + 10 + combined_height // 2
            crosshair_length = 20
            pygame.draw.line(menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), 1)  # Horizontal line
            pygame.draw.line(menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), 1)  # Vertical line

            # Camera status info (same as separate view to maintain consistent status display)
            status_font = pygame.font.Font(None, 20)
            # Camera 1 status
            camera1_status_text = status_font.render(f"Camera 1: {camera1_name}", True, (0, 255, 0))
            menu_screen.blit(camera1_status_text, (sub_x + 30, sub_y + 10))

            # Camera 2 status (next to Camera 1 status)
            camera2_status_text = status_font.render(f"Camera 2: {camera2_name}", True, (0, 255, 0))
            menu_screen.blit(camera2_status_text, (sub_x + 30, sub_y + 35))

            # Combined view status (moved below camera status)
            small_font = pygame.font.Font(None, 16)
            combined_text = small_font.render(f"Combined View - Opacity: {camera_opacity:.1f}", True, (0, 200, 200))
            menu_screen.blit(combined_text, (sub_x + 30, sub_y + 60))

        except Exception as e:
            # Fallback to separate camera display if combined view fails
            combined_view_active = False

    if not combined_view_active:
        # Original separate camera display logic
        if camera1.connected and camera1.frame is not None:
            try:
                # Apply gamma correction if enabled
                camera1_frame_processed = camera1.frame
                if camera1.gamma_enabled:
                    camera1_frame_processed = apply_gamma_correction(camera1.frame, camera1.gamma)

                # Resize camera frame to fit left half
                camera1_frame_display = pygame.transform.scale(camera1_frame_processed, (cam_display_width, cam_display_height))

                # Apply alignment rotation to camera 1 image - keep centered and crop corners
                if camera1.alignment_rotation != 0.0:
                    rotated_surface = pygame.transform.rotate(camera1_frame_display, camera1.alignment_rotation)
                    # Create new surface same size as display area
                    camera1_frame_display = pygame.Surface((cam_display_width, cam_display_height))
                    camera1_frame_display.fill((0, 0, 0))  # Fill with black background

                    # Calculate offset to center the rotation (copy center portion of rotated image)
                    rotated_center_x = rotated_surface.get_width() // 2
                    rotated_center_y = rotated_surface.get_height() // 2
                    display_center_x = cam_display_width // 2
                    display_center_y = cam_display_height // 2

                    # Copy center portion of rotated image to display surface
                    camera1_frame_display.blit(rotated_surface,
                                             (display_center_x - rotated_center_x,
                                              display_center_y - rotated_center_y))

                menu_screen.blit(camera1_frame_display, (sub_x + 10, sub_y + 10))

                # Draw ROI overlay for Camera 1 using encapsulated camera state
                if camera1 and camera1.roi_size >= 0:  # Only draw overlay if ROI is active (not -1 = deselected)
                    roi_width_pct, roi_height_pct = roi_sizes[camera1.roi_size]
                    roi_width_px = int(roi_width_pct * cam_display_width)
                    roi_height_px = int(roi_height_pct * cam_display_height)
                    roi_x_px = int(camera1.roi_x * (cam_display_width - roi_width_px))
                    roi_y_px = int(camera1.roi_y * (cam_display_height - roi_height_px))

                    # Draw green ROI rectangle overlay
                    roi_rect = pygame.Rect(sub_x + 10 + roi_x_px, sub_y + 10 + roi_y_px, roi_width_px, roi_height_px)
                    pygame.draw.rect(menu_screen, (0, 255, 0), roi_rect, 2)  # Green border, 2px thickness

                    # Draw green cross-hair in center of ROI box
                    center_x = roi_rect.centerx
                    center_y = roi_rect.centery
                    crosshair_length = 10  # Length of cross-hair arms
                    pygame.draw.line(menu_screen, (0, 255, 0), (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), 1)  # Horizontal line
                    pygame.draw.line(menu_screen, (0, 255, 0), (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), 1)  # Vertical line

                # Camera 1 info (right of camera frame)
                status_font = pygame.font.Font(None, 20)
                name_text = status_font.render(f"Camera 1: {camera1_name}", True, (0, 255, 0))
                menu_screen.blit(name_text, (sub_x + 30, sub_y + 10))

                info_font = pygame.font.Font(None, 16)
                fps_text = info_font.render(f"FPS: {camera1.fps:.1f}", True, (255, 255, 255))
                menu_screen.blit(fps_text, (sub_x + 630, sub_height - 10))

                utc_text = info_font.render(f"UTC: {camera1.utc_ts}", True, (255, 255, 255))
                menu_screen.blit(utc_text, (sub_x + 30, sub_height - 10))

                local_text = info_font.render(f"Local: {camera1.local_ts}", True, (255, 255, 255))
                menu_screen.blit(local_text, (sub_x + 330, sub_height - 10))

            except Exception as e:
                status_text = status_font.render(f"Camera 1 Error: {str(e)}", True, (255, 0, 0))
                menu_screen.blit(status_text, (sub_x + 10, sub_y + 10))
        else:
            # Camera 1 not connected
            font = pygame.font.Font(None, 20)
            not_connected_text = font.render(f"Camera 1: {camera1_name}", True, (255, 0, 0))
            menu_screen.blit(not_connected_text, (sub_x + 10, sub_y + 10))

            font_small = pygame.font.Font(None, 20)
            not_connected_subtext = font_small.render("Not Connected", True, (255, 0, 0))
            menu_screen.blit(not_connected_subtext, (sub_x + 10, sub_y + 50))

    # Camera 1 Connect button (next to Camera 1 status)
    tiny_font = pygame.font.Font(None, 12)
    mouse_pos = pygame.mouse.get_pos()

    camera1_connect_rect = pygame.Rect(sub_x + 230, sub_y + 10, 60, 20)
    if not camera1.connected:
        button_states["camera0_connect"]["hover"] = camera1_connect_rect.collidepoint(mouse_pos)
        button_color = (100, 100, 255) if button_states["camera0_connect"]["hover"] else (70, 70, 200)
        pygame.draw.rect(menu_screen, button_color, camera1_connect_rect)
        text_surface = tiny_font.render("Connect", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=camera1_connect_rect.center)
        menu_screen.blit(text_surface, text_rect)

    # Camera 1 Disconnect button (next to status)
    camera1_disconnect_rect = pygame.Rect(sub_x + 330, sub_y + 10, 60, 20)
    if camera1.connected:
        button_states["camera0_disconnect"]["hover"] = camera1_disconnect_rect.collidepoint(mouse_pos)
        button_color = (255, 100, 100) if button_states["camera0_disconnect"]["hover"] else (200, 70, 70)
        pygame.draw.rect(menu_screen, button_color, camera1_disconnect_rect)
        text_surface = tiny_font.render("Disconnect", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=camera1_disconnect_rect.center)
        menu_screen.blit(text_surface, text_rect)

    # Camera 2 display (right half) - Only render in separate view mode
    if not combined_view_active:
        if camera2_connected and camera2.frame is not None:
            try:
                # Apply gamma correction if enabled
                camera2_frame_processed = camera2.frame
                if camera2.gamma_enabled:
                    camera2_frame_processed = apply_gamma_correction(camera2.frame, camera2.gamma)

                # Resize camera frame to fit right half
                camera2_frame_display = pygame.transform.scale(camera2_frame_processed, (cam_display_width, cam_display_height))

                # Apply alignment rotation to camera 2 image - keep centered and crop corners
                if camera2.alignment_rotation != 0.0:
                    rotated_surface = pygame.transform.rotate(camera2_frame_display, camera2.alignment_rotation)
                    # Create new surface same size as display area
                    camera2_frame_display = pygame.Surface((cam_display_width, cam_display_height))
                    camera2_frame_display.fill((0, 0, 0))  # Fill with black background

                    # Calculate offset to center the rotation (copy center portion of rotated image)
                    rotated_center_x = rotated_surface.get_width() // 2
                    rotated_center_y = rotated_surface.get_height() // 2
                    display_center_x = cam_display_width // 2
                    display_center_y = cam_display_height // 2

                    # Copy center portion of rotated image to display surface
                    camera2_frame_display.blit(rotated_surface,
                                             (display_center_x - rotated_center_x,
                                              display_center_y - rotated_center_y))

                menu_screen.blit(camera2_frame_display, (sub_x + cam_display_width + 20, sub_y + 10))

                # Draw ROI overlay for Camera 2 using encapsulated camera state
                if camera2 and camera2.roi_size >= 0:  # Only draw overlay if ROI is active (not -1 = deselected)
                    roi_width_pct, roi_height_pct = roi_sizes[camera2.roi_size]
                    roi_width_px = int(roi_width_pct * cam_display_width)
                    roi_height_px = int(roi_height_pct * cam_display_height)
                    roi_x_px = int(camera2.roi_x * (cam_display_width - roi_width_px))
                    roi_y_px = int(camera2.roi_y * (cam_display_height - roi_height_px))

                    # Draw green ROI rectangle overlay
                    roi_rect = pygame.Rect(sub_x + cam_display_width + 20 + roi_x_px, sub_y + 10 + roi_y_px, roi_width_px, roi_height_px)
                    pygame.draw.rect(menu_screen, (0, 255, 0), roi_rect, 2)  # Green border, 2px thickness

                    # Draw green cross-hair in center of ROI box
                    center_x = roi_rect.centerx
                    center_y = roi_rect.centery
                    crosshair_length = 10  # Length of cross-hair arms
                    pygame.draw.line(menu_screen, (0, 255, 0), (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), 1)  # Horizontal line
                    pygame.draw.line(menu_screen, (0, 255, 0), (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), 1)  # Vertical line

                # Camera 2 info (right of camera frame)
                status_font = pygame.font.Font(None, 20)
                name_text = status_font.render(f"Camera 2: {camera2_name}", True, (0, 255, 0))
                menu_screen.blit(name_text, (sub_x + cam_display_width + 30, sub_y + 10))

                info_font = pygame.font.Font(None, 16)
                fps_text = info_font.render(f"FPS: {camera2.fps:.1f}", True, (255, 255, 255))
                menu_screen.blit(fps_text, (sub_x + cam_display_width + 630, sub_height - 10))

                utc_text = info_font.render(f"UTC: {camera2.utc_ts}", True, (255, 255, 255))
                menu_screen.blit(utc_text, (sub_x + cam_display_width + 30, sub_height - 10))

                local_text = info_font.render(f"Local: {camera2.local_ts}", True, (255, 255, 255))
                menu_screen.blit(local_text, (sub_x + cam_display_width + 330, sub_height - 10))

            except Exception as e:
                status_font = pygame.font.Font(None, 16)
                status_text = status_font.render(f"Camera 2 Error: {str(e)}", True, (255, 0, 0))
                menu_screen.blit(status_text, (sub_x + cam_display_width + 20, sub_y + 10))
        else:
            # Camera 2 not connected
            font = pygame.font.Font(None, 20)
            not_connected_text = font.render(f"Camera 2: {camera2_name}", True, (255, 0, 0))
            menu_screen.blit(not_connected_text, (sub_x + cam_display_width + 20, sub_y + 10))

            font_small = pygame.font.Font(None, 20)
            not_connected_subtext = font_small.render("Not Connected", True, (255, 0, 0))
            menu_screen.blit(not_connected_subtext, (sub_x + cam_display_width + 20, sub_y + 50))

    # Camera 2 Connect/Disconnect buttons - Only show in separate view mode
    if not combined_view_active:
        # Camera 2 Connect button (next to status)
        camera2_connect_rect = pygame.Rect(sub_x + cam_display_width + 230, sub_y + 10, 60, 20)
        if not camera2.connected:
            button_states["camera1_connect"]["hover"] = camera2_connect_rect.collidepoint(mouse_pos)
            button_color = (100, 100, 255) if button_states["camera1_connect"]["hover"] else (70, 70, 200)
            pygame.draw.rect(menu_screen, button_color, camera2_connect_rect)
            text_surface = tiny_font.render("Connect", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=camera2_connect_rect.center)
            menu_screen.blit(text_surface, text_rect)

        # Camera 2 Disconnect button (next to status)
        camera2_disconnect_rect = pygame.Rect(sub_x + cam_display_width + 330, sub_y + 10, 60, 20)
        if camera2.connected:
            button_states["camera1_disconnect"]["hover"] = camera2_disconnect_rect.collidepoint(mouse_pos)
            button_color = (255, 100, 100) if button_states["camera1_disconnect"]["hover"] else (200, 70, 70)
            pygame.draw.rect(menu_screen, button_color, camera2_disconnect_rect)
            text_surface = tiny_font.render("Disconnect", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=camera2_disconnect_rect.center)
            menu_screen.blit(text_surface, text_rect)

    # Reset and Save buttons (always visible for easy access) - Positioned at the bottom right of display area
    button_y = sub_y + sub_height - 35  # 35px from bottom of display area
    reset_button_rect = pygame.Rect(sub_x + sub_width - 240, button_y, 100, 25)
    button_states["reset_config"]["hover"] = reset_button_rect.collidepoint(mouse_pos)
    button_color = (150, 100, 100) if button_states["reset_config"]["hover"] else (100, 70, 70)
    pygame.draw.rect(menu_screen, button_color, reset_button_rect)
    pygame.draw.rect(menu_screen, (200, 200, 200), reset_button_rect, 1)  # Add border
    text_surface = tiny_font.render("Reset Config", True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=reset_button_rect.center)
    menu_screen.blit(text_surface, text_rect)

    save_button_rect = pygame.Rect(sub_x + sub_width - 120, button_y, 100, 25)
    button_states["save_config"]["hover"] = save_button_rect.collidepoint(mouse_pos)
    button_color = (100, 150, 100) if button_states["save_config"]["hover"] else (70, 100, 70)
    pygame.draw.rect(menu_screen, button_color, save_button_rect)
    pygame.draw.rect(menu_screen, (200, 200, 200), save_button_rect, 1)  # Add border
    text_surface = tiny_font.render("Save Config", True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=save_button_rect.center)
    menu_screen.blit(text_surface, text_rect)

def render_alignment_rotation_sliders(menu_screen, tiny_font, sub_x, sub_y, sub_width, sub_height):
    """Render alignment rotation sliders for both cameras - positioned at bottom-center of each image"""
    tiny_font = pygame.font.Font(None, 12)
    mouse_pos = pygame.mouse.get_pos()

    cam_display_width = (sub_width - 30) // 2  # Match the display function calculation
    cam_display_height = sub_height - 30

    # Get camera references from camera_manager
    camera1 = camera_manager.get_camera(0)
    camera2 = camera_manager.get_camera(1)

    # Define rotation slider ranges
    ROTATION_RANGE = 90.0  # -90° to +90° degrees
    SLIDER_WIDTH = 160  # Wider slider for better precision

    if camera1.connected:
        # Camera 1 Alignment Rotation Slider - positioned at bottom-center of camera 1 display
        slider_x = sub_x + 10 + (cam_display_width - SLIDER_WIDTH) // 2
        slider_y = sub_y + cam_display_height - 10  # 10px above bottom of camera display
        slider_color = (50, 50, 150)  # Blue-ish color for rotation sliders
        pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, SLIDER_WIDTH, 5))

        # Handle position - center at 0°
        rotation_ratio = (camera1.alignment_rotation + ROTATION_RANGE) / (2 * ROTATION_RANGE)
        rotation_ratio = max(0.0, min(1.0, rotation_ratio))  # Clamp to 0-1
        handle_x = slider_x + int(rotation_ratio * SLIDER_WIDTH)
        camera_manager.button_states["camera0_alignment_rotation_slider"]["hover"] = pygame.Rect(handle_x - 6, slider_y - 6, 12, 17).collidepoint(mouse_pos)
        handle_color = (100, 100, 255) if camera_manager.button_states["camera0_alignment_rotation_slider"]["hover"] else (150, 150, 255)
        pygame.draw.rect(menu_screen, handle_color, (handle_x - 6, slider_y - 6, 12, 17))

        # Label - show rotation value
        rotation_text = f"{camera1.alignment_rotation:+.1f}°"
        label_text = tiny_font.render(rotation_text, True, (255, 255, 255))
        menu_screen.blit(label_text, (slider_x, slider_y - 20))

        # Center marker at 0° position
        center_x = slider_x + int(0.5 * SLIDER_WIDTH)
        pygame.draw.line(menu_screen, (255, 255, 255), (center_x, slider_y - 3), (center_x, slider_y + 8), 2)

    if camera2.connected:
        # Camera 2 Alignment Rotation Slider - positioned at bottom-center of camera 2 display
        # Keep same position in both combined and separate view modes for consistency
        slider_x = sub_x + cam_display_width + 20 + (cam_display_width - SLIDER_WIDTH) // 2

        slider_y = sub_y + cam_display_height - 10  # 10px above bottom of camera display
        slider_color = (50, 50, 150)  # Blue-ish color for rotation sliders
        pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, SLIDER_WIDTH, 5))

        # Handle position - center at 0°
        rotation_ratio = (camera2.alignment_rotation + ROTATION_RANGE) / (2 * ROTATION_RANGE)
        rotation_ratio = max(0.0, min(1.0, rotation_ratio))  # Clamp to 0-1
        handle_x = slider_x + int(rotation_ratio * SLIDER_WIDTH)
        camera_manager.button_states["camera1_alignment_rotation_slider"]["hover"] = pygame.Rect(handle_x - 6, slider_y - 6, 12, 17).collidepoint(mouse_pos)
        handle_color = (100, 100, 255) if camera_manager.button_states["camera1_alignment_rotation_slider"]["hover"] else (150, 150, 255)
        pygame.draw.rect(menu_screen, handle_color, (handle_x - 6, slider_y - 6, 12, 17))

        # Label - show rotation value
        rotation_text = f"{camera2.alignment_rotation:+.1f}°"
        label_text = tiny_font.render(rotation_text, True, (255, 255, 255))
        menu_screen.blit(label_text, (slider_x, slider_y - 20))

        # Center marker at 0° position
        center_x = slider_x + int(0.5 * SLIDER_WIDTH)
        pygame.draw.line(menu_screen, (255, 255, 255), (center_x, slider_y - 3), (center_x, slider_y + 8), 2)

def render_camera_sliders(menu_screen, tiny_font, sub_x, sub_y, sub_width, sub_height):
    """Render gain and exposure sliders - positioned within camera display area"""
    tiny_font = pygame.font.Font(None, 12)
    mouse_pos = pygame.mouse.get_pos()

    cam_display_width = (sub_width - 30) // 2  # Match the display function calculation

    # Get camera references from camera_manager
    camera1 = camera_manager.get_camera(0)
    camera2 = camera_manager.get_camera(1)

    if camera1.connected:
        # Camera 1 Gain Slider - Next to connect / disconnect buttons
        if camera1.prop:
            max_gain = camera1.prop.get('MaxGain', 500) if camera1.prop else 500
            slider_x = sub_x + 450
            slider_y = sub_y + 20  # Next to connect / disconnect buttons
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position
            gain_ratio = min(1.0, camera1.gain / max_gain)
            handle_x = slider_x + int(gain_ratio * 120)
            camera_manager.button_states["camera1_gain_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (150, 0, 0) if camera_manager.button_states["camera1_gain_slider"]["hover"] else (200, 0, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label
            label_text = tiny_font.render(f"Gain: {camera1.gain}", True, (255, 255, 255))
            menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Camera 1 Exposure Slider - positioned next to gain slider
        if camera1.prop:
            max_exp = camera1.prop.get('MaxExposure', 2000000) if camera1.prop else 2000000
            slider_x = sub_x + 600
            slider_y = sub_y + 20 # Next to connect / disconnect buttons
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position - logarithmic scale for exposure
            import math
            min_exposure = 1  # 1 microsecond minimum
            max_log = math.log10(max_exp / min_exposure)
            if camera1.exposure > 0:
                log_ratio = math.log10(camera1.exposure / min_exposure) / max_log
                exp_ratio = min(1.0, max(0.0, log_ratio))
            else:
                exp_ratio = 0.0
            handle_x = slider_x + int(exp_ratio * 120)
            camera_manager.button_states["camera1_exposure_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (0, 150, 0) if camera_manager.button_states["camera1_exposure_slider"]["hover"] else (0, 200, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label - show appropriate units based on magnitude
            exp_us = camera1.exposure
            if exp_us < 1000:
                # Display in microseconds
                exp_val = f"{exp_us:g}µs"
            elif exp_us < 1000000:
                # Display in milliseconds
                exp_ms = exp_us / 1000.0
                exp_val = f"{exp_ms:.1f}ms"
            else:
                # Display in seconds
                exp_s = exp_us / 1000000.0
                exp_val = f"{exp_s:.1f}s"
            label_text = tiny_font.render(f"Exp: {exp_val}", True, (255, 255, 255))
            menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Camera 1 Gamma Slider - positioned below exposure slider
        slider_x = sub_x + 750
        slider_y = sub_y + 20
        slider_color = (150, 150, 150)  # Different color for gamma slider
        pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

        # Handle position - gamma range 0.01 to 2.0
        gamma_ratio = (camera1.gamma - 0.01) / (2.0 - 0.01)
        gamma_ratio = max(0.0, min(1.0, gamma_ratio))
        handle_x = slider_x + int(gamma_ratio * 120)
        camera_manager.button_states["camera1_gamma_slider"] = camera_manager.button_states.get("camera1_gamma_slider", {"hover": False, "dragging": False})
        camera_manager.button_states["camera1_gamma_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
        handle_color = (150, 150, 255) if camera_manager.button_states["camera1_gamma_slider"]["hover"] else (200, 200, 255)
        pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

        # Label
        label_text = tiny_font.render(f"Gamma: {camera1.gamma:.2f}", True, (255, 255, 255))
        menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Gamma toggle button next to gamma slider
        gamma_toggle_rect = pygame.Rect(slider_x + 130, slider_y - 5, 40, 15)
        camera_manager.button_states["camera1_gamma_toggle"] = camera_manager.button_states.get("camera1_gamma_toggle", {"hover": False, "clicked": False})
        camera_manager.button_states["camera1_gamma_toggle"]["hover"] = gamma_toggle_rect.collidepoint(mouse_pos)
        toggle_color = (0, 150, 0) if camera1.gamma_enabled else (150, 0, 0)  # Green if enabled, red if disabled
        if camera_manager.button_states["camera1_gamma_toggle"]["hover"]:
            toggle_color = tuple(min(255, c + 50) for c in toggle_color)  # Brighten on hover
        pygame.draw.rect(menu_screen, toggle_color, gamma_toggle_rect)
        toggle_text = tiny_font.render("ON" if camera1.gamma_enabled else "OFF", True, (255, 255, 255))
        text_rect = toggle_text.get_rect(center=gamma_toggle_rect.center)
        menu_screen.blit(toggle_text, text_rect)

    if camera2.connected:
        # Camera 2 Gain Slider - positioned below camera display on right side
        if camera2.prop:
            max_gain = camera2.prop.get('MaxGain', 500)
            slider_x = sub_x + cam_display_width + 450  # Right side, below camera display
            slider_y = sub_y + 20
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position
            gain_ratio = min(1.0, camera2.gain / max_gain)
            handle_x = slider_x + int(gain_ratio * 120)
            camera_manager.button_states["camera1_gain_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (150, 0, 0) if camera_manager.button_states["camera1_gain_slider"]["hover"] else (200, 0, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label
            label_text = tiny_font.render(f"Gain: {camera2.gain}", True, (255, 255, 255))
            menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Camera 2 Exposure Slider - positioned below gain slider on right side
        if camera2.prop:
            max_exp = camera2.prop.get('MaxExposure', 500000)
            slider_x = sub_x + cam_display_width + 600  # Right side, below gain slider
            slider_y = sub_y + 20
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position - logarithmic scale for exposure
            import math
            min_exposure = 1  # 1 microsecond minimum
            max_log = math.log10(max_exp / min_exposure)
            if camera2.exposure > 0:
                log_ratio = math.log10(camera2.exposure / min_exposure) / max_log
                exp_ratio = min(1.0, max(0.0, log_ratio))
            else:
                exp_ratio = 0.0
            handle_x = slider_x + int(exp_ratio * 120)
            camera_manager.button_states["camera1_exposure_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (0, 150, 0) if camera_manager.button_states["camera1_exposure_slider"]["hover"] else (0, 200, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label - show appropriate units based on magnitude
            exp_us = camera2.exposure
            if exp_us < 1000:
                # Display in microseconds
                exp_val = f"{exp_us:g}µs"
            elif exp_us < 1000000:
                # Display in milliseconds
                exp_ms = exp_us / 1000.0
                exp_val = f"{exp_ms:.1f}ms"
            else:
                # Display in seconds
                exp_s = exp_us / 1000000.0
                exp_val = f"{exp_s:.1f}s"
            label_text = tiny_font.render(f"Exp: {exp_val}", True, (255, 255, 255))
            menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Camera 2 Gamma Slider - positioned next to exposure slider on right side
        slider_x = sub_x + cam_display_width + 750
        slider_y = sub_y + 20
        slider_color = (150, 150, 150)  # Different color for gamma slider
        pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

        # Handle position - gamma range 0.01 to 2.0
        gamma_ratio = (camera2.gamma - 0.01) / (2.0 - 0.01)
        gamma_ratio = max(0.0, min(1.0, gamma_ratio))
        handle_x = slider_x + int(gamma_ratio * 120)
        camera_manager.button_states["camera2_gamma_slider"] = camera_manager.button_states.get("camera2_gamma_slider", {"hover": False, "dragging": False})
        camera_manager.button_states["camera2_gamma_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
        handle_color = (150, 150, 255) if camera_manager.button_states["camera2_gamma_slider"]["hover"] else (200, 200, 255)
        pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

        # Label
        label_text = tiny_font.render(f"Gamma: {camera2.gamma:.2f}", True, (255, 255, 255))
        menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Gamma toggle button next to gamma slider
        gamma_toggle_rect = pygame.Rect(slider_x + 130, slider_y - 5, 40, 15)
        camera_manager.button_states["camera2_gamma_toggle"] = camera_manager.button_states.get("camera2_gamma_toggle", {"hover": False, "clicked": False})
        camera_manager.button_states["camera2_gamma_toggle"]["hover"] = gamma_toggle_rect.collidepoint(mouse_pos)
        toggle_color = (0, 150, 0) if camera2.gamma_enabled else (150, 0, 0)  # Green if enabled, red if disabled
        if camera_manager.button_states["camera2_gamma_toggle"]["hover"]:
            toggle_color = tuple(min(255, c + 50) for c in toggle_color)  # Brighten on hover
        pygame.draw.rect(menu_screen, toggle_color, gamma_toggle_rect)
        toggle_text = tiny_font.render("ON" if camera2.gamma_enabled else "OFF", True, (255, 255, 255))
        text_rect = toggle_text.get_rect(center=gamma_toggle_rect.center)
        menu_screen.blit(toggle_text, text_rect)


def render_camera_roi_controls(menu_screen, sub_x, sub_y, sub_width, sub_height):
    """Render ROI controls for both cameras - positioned near each camera image"""
    mouse_pos = pygame.mouse.get_pos()
    tiny_font = pygame.font.Font(None, 12)

    # Calculate camera display dimensions
    cam_display_width = (sub_width - 30) // 2

    # Get camera references
    camera1 = camera_manager.get_camera(0)
    camera2 = camera_manager.get_camera(1)

    # ROI sizes and labels

    if camera1.connected:
        # Camera 1 ROI Size Controls - positioned in upper right of camera 1 image
        for i, size_text in enumerate(roi_label_texts):
            # Position buttons vertically near upper right edge of camera 1 - moved 100 pixels left
            if i < 3:  # First column
                rect = pygame.Rect(sub_x + 10 + cam_display_width + 15 - 100, sub_y + 10 + i * 15, 28, 12)
            else:  # Second column
                rect = pygame.Rect(sub_x + 10 + cam_display_width + 55 - 100, sub_y + 10 + (i - 3) * 15, 28, 12)

            is_selected = (camera1.roi_size == i)
            is_hovered = rect.collidepoint(mouse_pos)

            # Visual button appearance
            if is_selected:
                # Selected - green background, white text
                pygame.draw.rect(menu_screen, (0, 100, 0), rect)  # Dark green background
                pygame.draw.rect(menu_screen, (0, 255, 0), rect, 1)  # Bright green border
                text_color = (255, 255, 255)
            elif is_hovered:
                # Hover - lighter background
                pygame.draw.rect(menu_screen, (100, 100, 100), rect)
                pygame.draw.rect(menu_screen, (200, 200, 200), rect, 1)
                text_color = (255, 255, 0)  # Yellow text for hover
            else:
                # Normal - darker background
                pygame.draw.rect(menu_screen, (70, 70, 70), rect)
                pygame.draw.rect(menu_screen, (150, 150, 150), rect, 1)
                text_color = (255, 255, 255)

            roi_text = tiny_font.render(size_text, True, text_color)
            text_rect = roi_text.get_rect(center=rect.center)
            menu_screen.blit(roi_text, text_rect)

    if camera2.connected:
        # Camera 2 ROI Size Controls - positioned in upper right of camera 2 image
        # (mirrors Camera 1: snug against the right edge of the camera 2 image so
        # the buttons clear the gain/exposure/gamma sliders along the top)
        for i, size_text in enumerate(roi_label_texts):
            if i < 3:  # First column
                rect = pygame.Rect(sub_x + 2 * cam_display_width - 65, sub_y + 10 + i * 15, 28, 12)
            else:  # Second column
                rect = pygame.Rect(sub_x + 2 * cam_display_width - 25, sub_y + 10 + (i - 3) * 15, 28, 12)

            is_selected = (camera2.roi_size == i)
            is_hovered = rect.collidepoint(mouse_pos)

            # Visual button appearance
            if is_selected:
                # Selected - green background, white text
                pygame.draw.rect(menu_screen, (0, 100, 0), rect)  # Dark green background
                pygame.draw.rect(menu_screen, (0, 255, 0), rect, 1)  # Bright green border
                text_color = (255, 255, 255)
            elif is_hovered:
                # Hover - lighter background
                pygame.draw.rect(menu_screen, (100, 100, 100), rect)
                pygame.draw.rect(menu_screen, (200, 200, 200), rect, 1)
                text_color = (255, 255, 0)  # Yellow text for hover
            else:
                # Normal - darker background
                pygame.draw.rect(menu_screen, (70, 70, 70), rect)
                pygame.draw.rect(menu_screen, (150, 150, 150), rect, 1)
                text_color = (255, 255, 255)

            roi_text = tiny_font.render(size_text, True, text_color)
            text_rect = roi_text.get_rect(center=rect.center)
            menu_screen.blit(roi_text, text_rect)


def render_combined_view_controls(menu_screen, sub_x, sub_y, sub_width, sub_height, small_font, tiny_font):
    """Complete camera interface rendering including combined view controls"""
    mouse_pos = pygame.mouse.get_pos()

    # Get camera references
    camera1 = camera_manager.get_camera(0)
    camera2 = camera_manager.get_camera(1)

    # Combined view toggle button
    if camera_manager.COMBINED_VIEW_BUTTON_RECT:
        button_rect = camera_manager.COMBINED_VIEW_BUTTON_RECT
        is_toggled = camera_manager.combined_view_toggle
        is_hovered = button_rect.collidepoint(mouse_pos)

        # Button color based on state
        if is_toggled:
            button_color = (50, 150, 50) if is_hovered else (0, 100, 0)
        else:
            button_color = (100, 100, 100) if is_hovered else (70, 70, 70)

        pygame.draw.rect(menu_screen, button_color, button_rect)
        pygame.draw.rect(menu_screen, (150, 150, 150), button_rect, 1)

        # Button text
        combined_text = tiny_font.render("Combined View", True, (255, 255, 255))
        text_rect = combined_text.get_rect(center=button_rect.center)
        menu_screen.blit(combined_text, text_rect)

    # Camera opacity slider (only show when both cameras are connected or in combined view)
    if camera_manager.CAMERA_OPACITY_SLIDER_RECT and camera1.connected and camera2.connected:
        slider_rect = camera_manager.CAMERA_OPACITY_SLIDER_RECT
        handle_rect = camera_manager.CAMERA_OPACITY_SLIDER_HANDLE_RECT

        # Draw slider background
        pygame.draw.rect(menu_screen, (100, 100, 100), slider_rect)

        # Get camera opacity values (use the first camera's opacity for now)
        camera_opacity = camera_manager.camera_opacities[0]

        # Draw handle at current opacity position
        handle_pos = slider_rect.x + int(camera_opacity * slider_rect.width)
        handle_rect.centerx = handle_pos
        handle_rect.centery = slider_rect.centery
        pygame.draw.rect(menu_screen, (255, 255, 0), handle_rect)

        # Label
        opacity_text = tiny_font.render(f"Camera Opacity: {camera_opacity:.1f}", True, (255, 255, 255))
        menu_screen.blit(opacity_text, (slider_rect.x, slider_rect.y - 20))

def handle_sensor_calib_events(event, pos, display, camera_manager, update_status_callback=None, config_state=None):
    """Handle sensor calibration mode events - modular event handler matching new layout"""
    # Get camera references - these will be modified directly by set_camera_roi calls
    camera1 = camera_manager.get_camera(0)
    camera2 = camera_manager.get_camera(1)

    # Calculate layout metrics to match render_sensor_calibration
    cam_display_width = (display.sub_width - 30) // 2  # Same calculation as in render function
    cam_display_height = display.sub_height - 30  # Define at function scope to avoid UnboundLocalError

    if event.type == pygame.MOUSEBUTTONDOWN:
        # Camera 1 Connect button (next to status info)
        camera1_connect_rect = pygame.Rect(display.sub_x + 230, display.sub_y + 10, 60, 20)
        camera1_disconnect_rect = pygame.Rect(display.sub_x + 330, display.sub_y + 10, 60, 20)

        # Camera 2 Connect button (next to status info)
        camera2_connect_rect = pygame.Rect(display.sub_x + cam_display_width + 230, display.sub_y + 10, 60, 20)
        camera2_disconnect_rect = pygame.Rect(display.sub_x + cam_display_width + 330, display.sub_y + 10, 60, 20)

        if camera1_connect_rect.collidepoint(pos) and not camera1.connected:
            camera_manager.connect_camera(0, update_status_callback)
        elif camera1_disconnect_rect.collidepoint(pos) and camera1.connected:
            camera_manager.disconnect_camera(0, update_status_callback)
        elif camera2_connect_rect.collidepoint(pos) and not camera2.connected:
            camera_manager.connect_camera(1, update_status_callback)
        elif camera2_disconnect_rect.collidepoint(pos) and camera2.connected:
            camera_manager.disconnect_camera(1, update_status_callback)
        # Check gain/exposure/gamma sliders - entire slider track is clickable for easier interaction
        if camera1.connected:
            # Camera 1 Gain Slider track detection
            camera1_gain_track_rect = pygame.Rect(display.sub_x + 450 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            if camera1_gain_track_rect.collidepoint(pos):
                camera_manager.button_states["camera1_gain_slider"]["dragging"] = True

            # Camera 1 Exposure Slider track detection
            camera1_exposure_track_rect = pygame.Rect(display.sub_x + 600 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            if camera1_exposure_track_rect.collidepoint(pos):
                camera_manager.button_states["camera1_exposure_slider"]["dragging"] = True

            # Camera 1 Gamma Slider track detection
            camera1_gamma_track_rect = pygame.Rect(display.sub_x + 750 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            if camera1_gamma_track_rect.collidepoint(pos):
                camera_manager.button_states["camera1_gamma_slider"]["dragging"] = True

            # Camera 1 Gamma Toggle button detection
            camera1_gamma_toggle_rect = pygame.Rect(display.sub_x + 880, display.sub_y + 10, 40, 15)
            if camera1_gamma_toggle_rect.collidepoint(pos):
                camera_manager.cameras[0].gamma_enabled = not camera_manager.cameras[0].gamma_enabled

        if camera2.connected:
            # Camera 2 Gain Slider track detection
            camera2_gain_track_rect = pygame.Rect(display.sub_x + cam_display_width + 450 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            if camera2_gain_track_rect.collidepoint(pos):
                camera_manager.button_states["camera1_gain_slider"]["dragging"] = True

            # Camera 2 Exposure Slider track detection
            camera2_exposure_track_rect = pygame.Rect(display.sub_x + cam_display_width + 600 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            if camera2_exposure_track_rect.collidepoint(pos):
                camera_manager.button_states["camera1_exposure_slider"]["dragging"] = True

            # Camera 2 Gamma Slider track detection
            camera2_gamma_track_rect = pygame.Rect(display.sub_x + cam_display_width + 750 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            if camera2_gamma_track_rect.collidepoint(pos):
                camera_manager.button_states["camera2_gamma_slider"] = camera_manager.button_states.get("camera2_gamma_slider", {"hover": False, "dragging": False})
                camera_manager.button_states["camera2_gamma_slider"]["dragging"] = True

            # Camera 2 Gamma Toggle button detection
            camera2_gamma_toggle_rect = pygame.Rect(display.sub_x + cam_display_width + 880, display.sub_y + 10, 40, 15)
            if camera2_gamma_toggle_rect.collidepoint(pos):
                camera_manager.cameras[1].gamma_enabled = not camera_manager.cameras[1].gamma_enabled

        # Camera 1/2 Alignment Rotation Slider track detection - bottom-center of camera display
        cam_display_height = display.sub_height - 30
        ROTATION_SLIDER_WIDTH = 160
        ROTATION_RANGE = 90.0  # -90° to +90° degrees

        if camera1.connected:
            camera1_rotation_track_rect = pygame.Rect(display.sub_x + 10 + (cam_display_width - ROTATION_SLIDER_WIDTH) // 2 - 5, display.sub_y + cam_display_height - 10 - 5, ROTATION_SLIDER_WIDTH + 10, 15)
            if camera1_rotation_track_rect.collidepoint(pos):
                camera_manager.button_states["camera0_alignment_rotation_slider"]["dragging"] = True

        if camera2.connected:
            # Calculate Camera2 slider position - same as in render function (bottom-center of camera 2 area)
            camera2_rotation_slider_x = display.sub_x + cam_display_width + 20 + (cam_display_width - ROTATION_SLIDER_WIDTH) // 2

            camera2_rotation_track_rect = pygame.Rect(camera2_rotation_slider_x - 5, display.sub_y + cam_display_height - 10 - 5, ROTATION_SLIDER_WIDTH + 10, 15)
            if camera2_rotation_track_rect.collidepoint(pos):
                camera_manager.button_states["camera1_alignment_rotation_slider"]["dragging"] = True

        # Combined view button click
        is_combined_view_controls_click = False
        if camera_manager.COMBINED_VIEW_BUTTON_RECT and camera_manager.COMBINED_VIEW_BUTTON_RECT.collidepoint(pos):
            is_combined_view_controls_click = True
            camera_manager.combined_view_toggle = not camera_manager.combined_view_toggle

        # Camera opacity slider click (if both cameras connected)
        if camera1.connected and camera2.connected:
            if camera_manager.CAMERA_OPACITY_SLIDER_RECT and camera_manager.CAMERA_OPACITY_SLIDER_RECT.collidepoint(pos):
                is_combined_view_controls_click = True
                # Calculate new opacity based on click position
                slider_rect = camera_manager.CAMERA_OPACITY_SLIDER_RECT
                relative_x = min(max(pos[0] - slider_rect.x, 0), slider_rect.width)
                new_opacity = relative_x / slider_rect.width
                new_opacity = max(0.0, min(1.0, new_opacity))
                camera_manager.camera_opacities[0] = new_opacity  # Update first camera opacity

        # ROI Size and Origin Selection for Camera 1
        if camera1.connected:
            cam_display_width = (display.sub_width - 30) // 2
            cam_display_height = display.sub_height - 30

            # Camera 1 ROI Size Control buttons - positions must match render function
            is_roi_selection = False
            for i in range(6):
                if i < 3:  # First column
                    roi_rect = pygame.Rect(display.sub_x + 10 + cam_display_width + 15 - 100, display.sub_y + 10 + i * 15, 28, 12)
                else:  # Second column
                    roi_rect = pygame.Rect(display.sub_x + 10 + cam_display_width + 55 - 100, display.sub_y + 10 + (i - 3) * 15, 28, 12)

                if roi_rect.collidepoint(pos):
                    is_roi_selection = True
                    # If clicking the currently selected size, deselect ROI
                    if camera1.roi_size == i:
                        camera1.roi_size = -1
                        camera1.roi_x = 0.5  # Reset to center
                        camera1.roi_y = 0.5
                        # Actually reset camera to full resolution when deselecting
                        camera_manager.set_camera_roi(0, update_status_callback, None, None, -1)
                    else:
                        was_first_selection = (camera1.roi_size == -1)
                        camera1.roi_size = i
                        if was_first_selection:
                            camera1.roi_x = 0.5
                            camera1.roi_y = 0.5
                        # Retain current position for subsequent selections
                        camera_manager.set_camera_roi(0, update_status_callback, camera1.roi_x, camera1.roi_y, i)
                    break

            # ROI Origin Selection by clicking on Camera 1 image
            camera1_image_rect = pygame.Rect(display.sub_x + 10, display.sub_y + 10, cam_display_width, cam_display_height)

            # Skip camera image click if it's within a slider track area
            camera1_gain_track_rect = pygame.Rect(display.sub_x + 450 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            camera1_exposure_track_rect = pygame.Rect(display.sub_x + 600 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            camera1_gamma_track_rect = pygame.Rect(display.sub_x + 750 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            skip_camera1_click = camera1_gain_track_rect.collidepoint(pos) or camera1_exposure_track_rect.collidepoint(pos) or camera1_gamma_track_rect.collidepoint(pos)

            if camera1_image_rect.collidepoint(pos) and camera1.frame is not None and not skip_camera1_click and not is_roi_selection and not is_combined_view_controls_click:
                # Convert click position to normalized coordinates (0.0 to 1.0)
                rel_x = (pos[0] - (display.sub_x + 10)) / cam_display_width
                rel_y = (pos[1] - (display.sub_y + 10)) / cam_display_height
                camera1.roi_x = max(0.0, min(1.0, rel_x))
                camera1.roi_y = max(0.0, min(1.0, rel_y))

                # Apply ROI to camera if supported (pass current values to avoid globals issue)
                if camera1.roi_size >= 0 and camera_manager.set_camera_roi(0, update_status_callback, camera1.roi_x, camera1.roi_y, camera1.roi_size):
                    print(f"Camera 1 ROI origin set to ({camera1.roi_x:.3f}, {camera1.roi_y:.3f})")
                else:
                    print("Camera 1 ROI setting not supported or failed")

        # ROI Size and Origin Selection for Camera 2
        if camera2.connected:
            cam_display_width = (display.sub_width - 30) // 2
            cam_display_height = display.sub_height - 30

            # Camera 2 ROI Size Control buttons - positions must match render function
            is_roi_selection = False
            for i in range(6):
                if i < 3:  # First column
                    roi_rect = pygame.Rect(display.sub_x + 2 * cam_display_width - 65, display.sub_y + 10 + i * 15, 28, 12)
                else:  # Second column
                    roi_rect = pygame.Rect(display.sub_x + 2 * cam_display_width - 25, display.sub_y + 10 + (i - 3) * 15, 28, 12)

                if roi_rect.collidepoint(pos):
                    is_roi_selection = True
                    # If clicking the currently selected size, deselect ROI
                    if camera2.roi_size == i:
                        camera2.roi_size = -1
                        camera2.roi_x = 0.5  # Reset to center
                        camera2.roi_y = 0.5
                        # Actually reset camera to full resolution when deselecting
                        camera_manager.set_camera_roi(1, update_status_callback, None, None, -1)
                    else:
                        was_first_selection = (camera2.roi_size == -1)
                        camera2.roi_size = i
                        if was_first_selection:
                            camera2.roi_x = 0.5
                            camera2.roi_y = 0.5
                        # Retain current position for subsequent selections
                        camera_manager.set_camera_roi(1, update_status_callback, camera2.roi_x, camera2.roi_y, i)
                    break

            # ROI Origin Selection by clicking on Camera 2 image
            camera2_image_rect = pygame.Rect(display.sub_x + cam_display_width + 20, display.sub_y + 10, cam_display_width, cam_display_height)

            # Skip camera image click if it's within a slider track area
            camera2_gain_track_rect = pygame.Rect(display.sub_x + cam_display_width + 450 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            camera2_exposure_track_rect = pygame.Rect(display.sub_x + cam_display_width + 600 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            camera2_gamma_track_rect = pygame.Rect(display.sub_x + cam_display_width + 750 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
            skip_camera2_click = camera2_gain_track_rect.collidepoint(pos) or camera2_exposure_track_rect.collidepoint(pos) or camera2_gamma_track_rect.collidepoint(pos)

            if camera2_image_rect.collidepoint(pos) and camera2.frame is not None and not skip_camera2_click and not is_roi_selection and not is_combined_view_controls_click:
                # Convert click position to normalized coordinates (0.0 to 1.0)
                rel_x = (pos[0] - (display.sub_x + cam_display_width + 20)) / cam_display_width
                rel_y = (pos[1] - (display.sub_y + 10)) / cam_display_height
                camera2.roi_x = max(0.0, min(1.0, rel_x))
                camera2.roi_y = max(0.0, min(1.0, rel_y))
                # Apply ROI to camera if supported (pass current values to avoid globals issue)
                camera_manager.set_camera_roi(1, update_status_callback, camera2.roi_x, camera2.roi_y, camera2.roi_size)

        # Reset and Save Config buttons
        button_y = display.sub_y + display.sub_height - 35  # Match the rendering position
        reset_button_rect = pygame.Rect(display.sub_x + display.sub_width - 240, button_y, 100, 25)
        save_button_rect = pygame.Rect(display.sub_x + display.sub_width - 120, button_y, 100, 25)

        if reset_button_rect.collidepoint(pos):
            # Reset alignment_rotation, gain, exposure, gamma, and gamma_enabled to config file defaults using passed config_state
            if config_state:
                # Reset Camera 1 settings to config defaults
                camera_manager.cameras[0].alignment_rotation = float(config_state.get_camera_alignment_rotation("camera1") or 0.0)
                camera_manager.cameras[0].gain = int(float(config_state.get_camera_gain("camera1") or 1))
                camera_manager.cameras[0].exposure = int(float(config_state.get_camera_exposure("camera1") or 10000))
                camera_manager.cameras[0].gamma = float(config_state.camera_configs["camera1"].get("gamma", 0.1))
                camera_manager.cameras[0].gamma_enabled = bool(config_state.camera_configs["camera1"].get("gamma_enabled", False))

                # Reset Camera 2 settings to config defaults
                camera_manager.cameras[1].alignment_rotation = float(config_state.get_camera_alignment_rotation("camera2") or 0.0)
                camera_manager.cameras[1].gain = int(float(config_state.get_camera_gain("camera2") or 1))
                camera_manager.cameras[1].exposure = int(float(config_state.get_camera_exposure("camera2") or 10000))
                camera_manager.cameras[1].gamma = float(config_state.camera_configs["camera2"].get("gamma", 0.1))
                camera_manager.cameras[1].gamma_enabled = bool(config_state.camera_configs["camera2"].get("gamma_enabled", False))

                # Apply settings to connected cameras
                if camera1.connected and camera1.cap:
                    camera_manager.set_camera_gain(0, camera_manager.cameras[0].gain, update_status_callback)
                    camera_manager.set_camera_exposure(0, camera_manager.cameras[0].exposure, update_status_callback)

                if camera2.connected and camera2.cap:
                    camera_manager.set_camera_gain(1, camera_manager.cameras[1].gain, update_status_callback)
                    camera_manager.set_camera_exposure(1, camera_manager.cameras[1].exposure, update_status_callback)

                if update_status_callback:
                    update_status_callback("Camera settings reset to config file defaults")
            else:
                if update_status_callback:
                    update_status_callback("Error: Config state not available")

        elif save_button_rect.collidepoint(pos):
            # Save current alignment_rotation, gain, exposure, gamma, and gamma_enabled values to config file using passed config_state
            if config_state:
                # Update config with current values
                config_state.camera_configs["camera1"]["alignment_rotation"] = camera_manager.cameras[0].alignment_rotation
                config_state.camera_configs["camera1"]["gain"] = camera_manager.cameras[0].gain
                config_state.camera_configs["camera1"]["exposure"] = camera_manager.cameras[0].exposure
                config_state.camera_configs["camera1"]["gamma"] = camera_manager.cameras[0].gamma
                config_state.camera_configs["camera1"]["gamma_enabled"] = camera_manager.cameras[0].gamma_enabled

                config_state.camera_configs["camera2"]["alignment_rotation"] = camera_manager.cameras[1].alignment_rotation
                config_state.camera_configs["camera2"]["gain"] = camera_manager.cameras[1].gain
                config_state.camera_configs["camera2"]["exposure"] = camera_manager.cameras[1].exposure
                config_state.camera_configs["camera2"]["gamma"] = camera_manager.cameras[1].gamma
                config_state.camera_configs["camera2"]["gamma_enabled"] = camera_manager.cameras[1].gamma_enabled

                # Save to file
                config_state.save_to_file()

                if update_status_callback:
                    update_status_callback("Camera settings saved to config.json")
            else:
                if update_status_callback:
                    update_status_callback("Error: Config state not available")

    elif event.type == pygame.MOUSEMOTION:
        # Handle slider dragging based on mouse button state, not stored state
        if event.buttons[0]:  # Left mouse button is pressed
            current_pos = event.pos  # Use current mouse position for motion events
            # Calculate positions to match render function
            cam_display_width = (display.sub_width - 30) // 2

            if camera1.connected:
                # Check if mouse is over Camera 1 Gain Slider track
                camera1_gain_track_rect = pygame.Rect(display.sub_x + 450 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
                if camera1_gain_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + 450
                    slider_width = 120
                    max_gain = camera1.prop.get('MaxGain', 500) if camera1.prop else 500
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    new_gain = int((relative_x / slider_width) * max_gain)
                    camera_manager.set_camera_gain(0, new_gain, lambda msg: print(f"Gain: {new_gain}"))

                # Check if mouse is over Camera 1 Exposure Slider track
                camera1_exposure_track_rect = pygame.Rect(display.sub_x + 600 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
                if camera1_exposure_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + 600
                    slider_width = 120
                    max_exp = camera1.prop.get('MaxExposure', 500000) if camera1.prop else 500000
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    slider_pos = relative_x / slider_width  # 0-1 position along slider

                    # Logarithmic scale for exposure
                    min_exposure = 1  # 1 microsecond minimum
                    if max_exp > min_exposure:
                        import math
                        logarithmic_value = slider_pos * math.log10(max_exp / min_exposure)
                        new_exposure = int(min_exposure * (10 ** logarithmic_value))
                        new_exposure = max(min_exposure, min(new_exposure, max_exp))
                    else:
                        new_exposure = max_exp
                    camera_manager.set_camera_exposure(0, new_exposure, lambda msg: print(f"Exposure: {new_exposure} μs"))

                # Check if mouse is over Camera 1 Gamma Slider track
                camera1_gamma_track_rect = pygame.Rect(display.sub_x + 750 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
                if camera1_gamma_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + 750
                    slider_width = 120
                    gamma_min = 0.01
                    gamma_max = 2.0
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    slider_pos = relative_x / slider_width  # 0-1 position along slider
                    new_gamma = gamma_min + slider_pos * (gamma_max - gamma_min)
                    new_gamma = max(gamma_min, min(gamma_max, new_gamma))
                    camera_manager.cameras[0].gamma = round(new_gamma, 2)  # Round to 2 decimal places

            if camera2.connected:
                # Check if mouse is over Camera 2 Gain Slider track
                camera2_gain_track_rect = pygame.Rect(display.sub_x + cam_display_width + 450 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
                if camera2_gain_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + cam_display_width + 450
                    slider_width = 120
                    max_gain = camera2.prop.get('MaxGain', 500) if camera2.prop else 500
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    new_gain = int((relative_x / slider_width) * max_gain)
                    camera_manager.set_camera_gain(1, new_gain, lambda msg: print(f"Gain: {new_gain}"))

                # Check if mouse is over Camera 2 Exposure Slider track
                camera2_exposure_track_rect = pygame.Rect(display.sub_x + cam_display_width + 600 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
                if camera2_exposure_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + cam_display_width + 600
                    slider_width = 120
                    max_exp = camera2.prop.get('MaxExposure', 500000) if camera2.prop else 500000
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    slider_pos = relative_x / slider_width  # 0-1 position along slider

                    # Logarithmic scale for exposure
                    min_exposure = 1  # 1 microsecond minimum
                    if max_exp > min_exposure:
                        import math
                        logarithmic_value = slider_pos * math.log10(max_exp / min_exposure)
                        new_exposure = int(min_exposure * (10 ** logarithmic_value))
                        new_exposure = max(min_exposure, min(new_exposure, max_exp))
                    else:
                        new_exposure = max_exp
                    camera_manager.set_camera_exposure(1, new_exposure, lambda msg: print(f"Exposure: {new_exposure} μs"))

                # Check if mouse is over Camera 2 Gamma Slider track
                camera2_gamma_track_rect = pygame.Rect(display.sub_x + cam_display_width + 750 - 10, display.sub_y + 20 - 10, 120 + 20, 25)
                if camera2_gamma_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + cam_display_width + 750
                    slider_width = 120
                    gamma_min = 0.01
                    gamma_max = 2.0
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    slider_pos = relative_x / slider_width  # 0-1 position along slider
                    new_gamma = gamma_min + slider_pos * (gamma_max - gamma_min)
                    new_gamma = max(gamma_min, min(gamma_max, new_gamma))
                    camera_manager.cameras[1].gamma = round(new_gamma, 2)  # Round to 2 decimal places

            # Handle Camera 1 Alignment Rotation Slider dragging
            if camera1.connected:
                ROTATION_SLIDER_WIDTH = 160
                ROTATION_RANGE = 90.0  # -90° to +90° degrees
                camera1_rotation_track_rect = pygame.Rect(display.sub_x + 10 + (cam_display_width - ROTATION_SLIDER_WIDTH) // 2 - 5, display.sub_y + cam_display_height - 10 - 5, ROTATION_SLIDER_WIDTH + 10, 15)
                if camera1_rotation_track_rect.collidepoint(current_pos):
                    slider_x = display.sub_x + 10 + (cam_display_width - ROTATION_SLIDER_WIDTH) // 2
                    relative_x = min(max(current_pos[0] - slider_x, 0), ROTATION_SLIDER_WIDTH)
                    slider_pos = relative_x / ROTATION_SLIDER_WIDTH  # 0-1 position along slider
                    new_rotation = (slider_pos - 0.5) * (2 * ROTATION_RANGE)  # -90° to +90°
                    new_rotation = max(-ROTATION_RANGE, min(ROTATION_RANGE, new_rotation))
                    camera_manager.cameras[0].alignment_rotation = new_rotation

            # Handle Camera 2 Alignment Rotation Slider dragging
            if camera2.connected:
                ROTATION_SLIDER_WIDTH = 160
                ROTATION_RANGE = 90.0  # -90° to +90° degrees

                # Calculate Camera2 slider position - same as in render function (bottom-center of camera 2 area)
                camera2_slider_x = display.sub_x + cam_display_width + 20 + (cam_display_width - ROTATION_SLIDER_WIDTH) // 2

                camera2_rotation_track_rect = pygame.Rect(camera2_slider_x - 5, display.sub_y + cam_display_height - 10 - 5, ROTATION_SLIDER_WIDTH + 10, 15)
                if camera2_rotation_track_rect.collidepoint(current_pos):
                    slider_x = camera2_slider_x
                    relative_x = min(max(current_pos[0] - slider_x, 0), ROTATION_SLIDER_WIDTH)
                    slider_pos = relative_x / ROTATION_SLIDER_WIDTH  # 0-1 position along slider
                    new_rotation = (slider_pos - 0.5) * (2 * ROTATION_RANGE)  # -90° to +90°
                    new_rotation = max(-ROTATION_RANGE, min(ROTATION_RANGE, new_rotation))
                    camera_manager.cameras[1].alignment_rotation = new_rotation
