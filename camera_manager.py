import pygame
import zwoasi as asi
import numpy as np
from camera_buffer import CameraThread, CircularBuffer

# Camera connection status
camera1_connected = False
camera2_connected = False
camera1_cap = None
camera2_cap = None
camera1_prop = None
camera2_prop = None
camera1_index = 0
camera2_index = 1

# Camera frame data
camera1_frame = None
camera2_frame = None

# Camera settings
camera1_gain = 1
camera2_gain = 1
camera1_exposure = 10000  # 10ms default
camera2_exposure = 10000
camera1_width_res = 1920
camera1_height_res = 1280
camera2_width_res = 1936
camera2_height_res = 1216

# Screen/layout settings
CAMERA1_CONNECT_BUTTON = None  # Will be set after pygame initialization
CAMERA1_DISCONNECT_BUTTON = None
CAMERA2_CONNECT_BUTTON = None
CAMERA2_DISCONNECT_BUTTON = None
COMBINED_VIEW_BUTTON_RECT = None
CAMERA_OPACITY_SLIDER_RECT = None
CAMERA_OPACITY_SLIDER_HANDLE_RECT = None

# Camera threads
camera1_thread = None
camera2_thread = None
camera_threads_running = False
camera1_threads_running = False
camera2_threads_running = False

# Camera performance tracking
camera1_fps = 0.0
camera2_fps = 0.0
camera1_utc_ts = ""
camera2_utc_ts = ""
camera1_local_ts = ""
camera2_local_ts = ""

# ROI settings
camera1_roi_size = 0
camera2_roi_size = 0
camera1_roi_x = 0.5
camera2_roi_x = 0.5
camera1_roi_y = 0.5
camera2_roi_y = 0.5
roi_sizes = [
    (1.0, 1.0),
    (0.5, 0.5),
    (0.25, 0.25),
    (0.125, 0.125),
    (0.0625, 0.0625),
    (0.03125, 0.03125),
]

roi_label_texts = ["1.0", ".5", ".25", ".125", ".063", ".032"]

# Combined view settings
combined_view_toggle = False
camera1_opacity = 0.5

# Camera button states
camera_button_states = {
    "camera1_connect": {"hover": False, "clicked": False},
    "camera2_connect": {"hover": False, "clicked": False},
    "camera1_disconnect": {"hover": False, "clicked": False},
    "camera2_disconnect": {"hover": False, "clicked": False},
    "camera1_gain_slider": {"hover": False, "dragging": False},
    "camera2_gain_slider": {"hover": False, "dragging": False},
    "camera1_exposure_slider": {"hover": False, "dragging": False},
    "camera2_exposure_slider": {"hover": False, "dragging": False},
}

# ==============================================================================
# CAMERA INITIALIZATION
# ==============================================================================

def initialize_camera_buttons(menu_screen, total_width, total_height):
    """Initialize camera control button positions after pygame is set up"""
    global CAMERA1_CONNECT_BUTTON, CAMERA1_DISCONNECT_BUTTON, CAMERA2_CONNECT_BUTTON, CAMERA2_DISCONNECT_BUTTON

    CAMERA1_CONNECT_BUTTON = pygame.Rect(total_width - 200 - 20, total_height - 60, 60, 20)
    CAMERA1_DISCONNECT_BUTTON = pygame.Rect(total_width - 200 - 80, total_height - 60, 60, 20)
    CAMERA2_CONNECT_BUTTON = pygame.Rect(total_width - 200 - 20 - 200, total_height - 60, 60, 20)
    CAMERA2_DISCONNECT_BUTTON = pygame.Rect(total_width - 200 - 80 - 200, total_height - 60, 60, 20)

    # Initialize combined view controls
    COMBINED_VIEW_BUTTON_RECT = pygame.Rect(total_width // 2 + 50, total_height - 60, 100, 20)
    CAMERA_OPACITY_SLIDER_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 300, 20)
    CAMERA_OPACITY_SLIDER_HANDLE_RECT = pygame.Rect(total_width // 2 - 40, total_height - 30, 20, 20)

def connect_camera1(update_status_callback=None):
    """Connect to camera 1"""
    global camera1_connected, camera1_cap, camera1_prop, camera1_width_res, camera1_height_res

    if camera1_connected:
        if update_status_callback:
            update_status_callback("Camera 1 already connected")
        return True

    try:
        camera1_cap = asi.Camera(camera1_index)
        camera1_prop = camera1_cap.get_camera_property()

        if camera1_cap:
            # Set video format
            if camera1_prop['IsColorCam']:
                camera1_cap.set_image_type(asi.ASI_IMG_RGB24)
            else:
                camera1_cap.set_image_type(asi.ASI_IMG_RAW8)

            # Set camera controls
            camera1_cap.set_control_value(asi.ASI_EXPOSURE, camera1_exposure)
            camera1_cap.set_control_value(asi.ASI_GAIN, camera1_gain)
            camera1_connected = True

            # Set dynamic camera resolution
            camera1_width_res = camera1_prop.get('MaxWidth', 1920)
            camera1_height_res = camera1_prop.get('MaxHeight', 1280)

            # Start camera thread for continuous capture
            start_camera1_thread()

            if update_status_callback:
                update_status_callback(f"Camera 1 connected successfully: {camera1_width_res}x{camera1_height_res}")
            return True
    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Camera 1 connection error: {str(e)}")
        camera1_cap = None

    return False

def connect_camera2(update_status_callback=None):
    """Connect to camera 2"""
    global camera2_connected, camera2_cap, camera2_prop, camera2_width_res, camera2_height_res

    if camera2_connected:
        if update_status_callback:
            update_status_callback("Camera 2 already connected")
        return True

    try:
        # Check available cameras
        num_cameras = asi.get_num_cameras()
        if num_cameras <= camera2_index:
            if update_status_callback:
                update_status_callback(f"Camera 2 (index {camera2_index}) not found. Only {num_cameras} camera(s) detected.")
            return False

        camera2_cap = asi.Camera(camera2_index)
        camera2_prop = camera2_cap.get_camera_property()

        if camera2_cap:
            # Set video format
            if camera2_prop['IsColorCam']:
                camera2_cap.set_image_type(asi.ASI_IMG_RGB24)
            else:
                camera2_cap.set_image_type(asi.ASI_IMG_RAW8)

            # Set camera controls
            camera2_cap.set_control_value(asi.ASI_EXPOSURE, camera2_exposure)
            camera2_cap.set_control_value(asi.ASI_GAIN, camera2_gain)
            camera2_connected = True

            # Start threaded capture
            start_camera2_thread()

            if update_status_callback:
                update_status_callback("Camera 2 connected successfully")
            return True
    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Camera 2 connection error: {str(e)}")
        camera2_cap = None

    return False

def disconnect_camera1(update_status_callback=None):
    """Disconnect camera 1"""
    global camera1_connected, camera1_cap, camera1_frame

    stop_camera1_thread()
    if camera1_cap is not None:
        camera1_cap = None
    camera1_connected = False
    camera1_frame = None
    if update_status_callback:
        update_status_callback("Camera 1 disconnected")

def disconnect_camera2(update_status_callback=None):
    """Disconnect camera 2"""
    global camera2_connected, camera2_cap, camera2_frame

    stop_camera2_thread()
    if camera2_cap is not None:
        camera2_cap = None
    camera2_connected = False
    camera2_frame = None
    if update_status_callback:
        update_status_callback("Camera 2 disconnected")

# ==============================================================================
# THREAD MANAGEMENT
# ==============================================================================

def start_camera1_thread():
    """Start camera 1 capture thread"""
    global camera1_thread, camera1_threads_running

    if camera1_threads_running or not camera1_connected or camera1_cap is None:
        return

    camera1_threads_running = True
    camera1_thread = CameraThread(camera1_index, camera1_cap, 60, 15)  # BUFFER_SIZE=60, TARGET_FPS=15
    camera1_thread.start()

def start_camera2_thread():
    """Start camera 2 capture thread"""
    global camera2_thread, camera2_threads_running

    if camera2_threads_running or not camera2_connected or camera2_cap is None:
        return

    camera2_threads_running = True
    camera2_thread = CameraThread(camera2_index, camera2_cap, 60, 15)  # BUFFER_SIZE=60, TARGET_FPS=15
    camera2_thread.start()

def stop_camera1_thread():
    """Stop camera 1 capture thread"""
    global camera1_thread, camera1_threads_running

    camera1_threads_running = False
    if camera1_thread is not None:
        camera1_thread.stop()
        camera1_thread = None

def stop_camera2_thread():
    """Stop camera 2 capture thread"""
    global camera2_thread, camera2_threads_running

    camera2_threads_running = False
    if camera2_thread is not None:
        camera2_thread.stop()
        camera2_thread = None

def stop_all_camera_threads():
    """Stop all camera threads"""
    stop_camera1_thread()
    stop_camera2_thread()

def update_camera_frames_from_buffers():
    """Update camera frames from thread buffers"""
    global camera1_frame, camera2_frame, camera1_fps, camera2_fps
    global camera1_utc_ts, camera2_utc_ts, camera1_local_ts, camera2_local_ts

    # Update camera 1 frame from buffer
    if camera1_thread is not None:
        latest_frame = camera1_thread.buffer.get_latest_frame()
        if latest_frame is not None:
            camera1_frame = latest_frame['frame']
            camera1_fps = camera1_thread.get_buffer_fps()
            camera1_utc_ts = camera1_thread.get_utc_timestamp()
            camera1_local_ts = camera1_thread.get_local_timestamp()

    # Update camera 2 frame from buffer
    if camera2_thread is not None:
        latest_frame = camera2_thread.buffer.get_latest_frame()
        if latest_frame is not None:
            camera2_frame = latest_frame['frame']
            camera2_fps = camera2_thread.get_buffer_fps()
            camera2_utc_ts = camera2_thread.get_utc_timestamp()
            camera2_local_ts = camera2_thread.get_local_timestamp()

# ==============================================================================
# SETTINGS CONTROL
# ==============================================================================

def set_camera1_gain(gain_value, update_status_callback=None):
    """Set camera 1 gain"""
    global camera1_gain
    camera1_gain = gain_value
    if camera1_connected and camera1_cap:
        try:
            camera1_cap.set_control_value(asi.ASI_GAIN, camera1_gain)
            if update_status_callback:
                update_status_callback(f"Camera 1 gain set to {camera1_gain}")
        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Failed to set Camera 2 gain: {str(e)}")

def set_camera2_gain(gain_value, update_status_callback=None):
    """Set camera 2 gain"""
    global camera2_gain
    camera2_gain = gain_value
    if camera2_connected and camera2_cap:
        try:
            camera2_cap.set_control_value(asi.ASI_GAIN, camera2_gain)
            if update_status_callback:
                update_status_callback(f"Camera 2 gain set to {camera2_gain}")
        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Failed to set Camera 1 gain: {str(e)}")

def set_camera1_exposure(exposure_value, update_status_callback=None):
    """Set camera 1 exposure"""
    global camera1_exposure
    camera1_exposure = exposure_value
    if camera1_connected and camera1_cap:
        try:
            camera1_cap.set_control_value(asi.ASI_EXPOSURE, camera1_exposure)
            if update_status_callback:
                update_status_callback(f"Camera 1 exposure set to {camera1_exposure} µs")
        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Failed to set Camera 1 exposure: {str(e)}")

def set_camera2_exposure(exposure_value, update_status_callback=None):
    """Set camera 2 exposure"""
    global camera2_exposure
    camera2_exposure = exposure_value
    if camera2_connected and camera2_cap:
        try:
            camera2_cap.set_control_value(asi.ASI_EXPOSURE, camera2_exposure)
            if update_status_callback:
                update_status_callback(f"Camera 2 exposure set to {camera2_exposure} µs")
        except Exception as e:
            if update_status_callback:
                update_status_callback(f"Failed to set Camera 2 exposure: {str(e)}")

def set_camera1_roi():
    """Set camera 1 ROI based on current settings"""
    global camera1_connected, camera1_cap, camera1_prop
    if not camera1_connected or camera1_cap is None or camera1_prop is None:
        return False

    try:
        # Get current ROI size
        roi_width, roi_height = roi_sizes[camera1_roi_size]

        # Calculate pixel coordinates
        max_width = camera1_prop.get('MaxWidth', 1920)
        max_height = camera1_prop.get('MaxHeight', 1280)

        start_x = int(camera1_roi_x * max_width)
        start_y = int(camera1_roi_y * max_height)
        roi_w = int(roi_width * max_width)
        roi_h = int(roi_height * max_height)

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

        # Get current ROI format to preserve bins and image_type
        current_roi_format = camera1_cap.get_roi_format()

        # Apply ROI to camera using correct API
        try:
            camera1_cap.set_roi_format(roi_w, roi_h, current_roi_format.bins, current_roi_format.image_type)
            camera1_cap.set_roi_start_position(start_x, start_y)
            print(f"Camera 1 ROI set: ({start_x}, {start_y}) {roi_w}x{roi_h}")
            return True
        except AttributeError:
            # ROI setting not supported by this camera/driver
            print(f"Camera 1 ROI setting not supported by hardware")
            return False
    except Exception as e:
        print(f"Error setting Camera 1 ROI: {str(e)}")
        return False

def set_camera2_roi():
    """Set camera 2 ROI based on current settings"""
    global camera2_connected, camera2_cap, camera2_prop
    if not camera2_connected or camera2_cap is None or camera2_prop is None:
        return False

    try:
        # Get current ROI size
        roi_width, roi_height = roi_sizes[camera2_roi_size]

        # Calculate pixel coordinates
        max_width = camera2_prop.get('MaxWidth', 1936)
        max_height = camera2_prop.get('MaxHeight', 1216)

        start_x = int(camera2_roi_x * max_width)
        start_y = int(camera2_roi_y * max_height)
        roi_w = int(roi_width * max_width)
        roi_h = int(roi_height * max_height)

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

        # Get current ROI format to preserve bins and image_type
        current_roi_format = camera2_cap.get_roi_format()

        # Apply ROI to camera using correct API
        try:
            camera2_cap.set_roi_format(roi_w, roi_h, current_roi_format.bins, current_roi_format.image_type)
            camera2_cap.set_roi_start_position(start_x, start_y)
            print(f"Camera 2 ROI set: ({start_x}, {start_y}) {roi_w}x{roi_h}")
            return True
        except AttributeError:
            # ROI setting not supported by this camera/driver
            print(f"Camera 2 ROI setting not supported by hardware")
            return False
    except Exception as e:
        print(f"Error setting Camera 2 ROI: {str(e)}")
        return False

def get_num_cameras():
    """Get number of cameras available"""
    try:
        return asi.get_num_cameras()
    except:
        return 0

def get_camera_names():
    """Get list of camera names"""
    try:
        return asi.list_cameras()
    except:
        return []

# ==============================================================================
# CAMERA RENDERING FUNCTIONS (refactored from main.py)
# ==============================================================================

def render_sensor_calibration(menu_screen, sub_x, sub_y, sub_width, sub_height, camera1_connected, camera2_connected, camera1_name, camera2_name):
    """Render sensor calibration interface - original side-by-side layout restored"""
    from camera_buffer import CameraThread
    from utils import draw_button
    import threading

    menu_screen.fill((0, 0, 0), (sub_x, sub_y, sub_width, sub_height))

    # Update camera frames from buffers
    update_camera_frames_from_buffers()

    # Camera 1 display (left half)
    cam_display_width = (sub_width - 30) // 2  # Leave space between cameras
    cam_display_height = sub_height - 30

    if camera1_connected and camera1_frame is not None:
        try:
            # Resize camera frame to fit left half
            camera1_frame_display = pygame.transform.scale(camera1_frame, (cam_display_width, cam_display_height))
            menu_screen.blit(camera1_frame_display, (sub_x + 10, sub_y + 10))

            # Draw ROI overlay for camera 1 if ROI size is not full (index 5)
            if camera1_roi_size < 5:  # Only draw overlay if ROI is active (not full image)
                roi_width_pct, roi_height_pct = roi_sizes[camera1_roi_size]
                roi_width_px = int(roi_width_pct * cam_display_width)
                roi_height_px = int(roi_height_pct * cam_display_height)
                roi_x_px = int(camera1_roi_x * (cam_display_width - roi_width_px))
                roi_y_px = int(camera1_roi_y * (cam_display_height - roi_height_px))

                # Draw green ROI rectangle overlay
                roi_rect = pygame.Rect(sub_x + 10 + roi_x_px, sub_y + 10 + roi_y_px, roi_width_px, roi_height_px)
                pygame.draw.rect(menu_screen, (0, 255, 0), roi_rect, 2)  # Green border, 2px thickness

            # Camera 1 info (right of camera frame)
            status_font = pygame.font.Font(None, 20)
            name_text = status_font.render(f"Camera 1: {camera1_name}", True, (0, 255, 0))
            menu_screen.blit(name_text, (sub_x + 30, sub_y + 10))

            info_font = pygame.font.Font(None, 16)
            fps_text = info_font.render(f"FPS: {camera1_fps:.1f}", True, (255, 255, 255))
            menu_screen.blit(fps_text, (sub_x + 630, sub_height - 10))

            utc_text = info_font.render(f"UTC: {camera1_utc_ts}", True, (255, 255, 255))
            menu_screen.blit(utc_text, (sub_x + 30, sub_height - 10))

            local_text = info_font.render(f"Local: {camera1_local_ts}", True, (255, 255, 255))
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
    if not camera1_connected:
        camera_button_states["camera1_connect"]["hover"] = camera1_connect_rect.collidepoint(mouse_pos)
        button_color = (100, 100, 255) if camera_button_states["camera1_connect"]["hover"] else (70, 70, 200)
        pygame.draw.rect(menu_screen, button_color, camera1_connect_rect)
        text_surface = tiny_font.render("Connect", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=camera1_connect_rect.center)
        menu_screen.blit(text_surface, text_rect)

    # Camera 1 Disconnect button (next to status)
    camera1_disconnect_rect = pygame.Rect(sub_x + 330, sub_y + 10, 60, 20)
    if camera1_connected:
        camera_button_states["camera1_disconnect"]["hover"] = camera1_disconnect_rect.collidepoint(mouse_pos)
        button_color = (255, 100, 100) if camera_button_states["camera1_disconnect"]["hover"] else (200, 70, 70)
        pygame.draw.rect(menu_screen, button_color, camera1_disconnect_rect)
        text_surface = tiny_font.render("Disconnect", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=camera1_disconnect_rect.center)
        menu_screen.blit(text_surface, text_rect)

    # Camera 2 display (right half)
    if camera2_connected and camera2_frame is not None:
        try:
            # Resize camera frame to fit right half
            camera2_frame_display = pygame.transform.scale(camera2_frame, (cam_display_width, cam_display_height))
            menu_screen.blit(camera2_frame_display, (sub_x + cam_display_width + 20, sub_y + 10))

            # Draw ROI overlay for camera 2 if ROI size is not full (index 5)
            if camera2_roi_size < 5:  # Only draw overlay if ROI is active (not full image)
                roi_width_pct, roi_height_pct = roi_sizes[camera2_roi_size]
                roi_width_px = int(roi_width_pct * cam_display_width)
                roi_height_px = int(roi_height_pct * cam_display_height)
                roi_x_px = int(camera2_roi_x * (cam_display_width - roi_width_px))
                roi_y_px = int(camera2_roi_y * (cam_display_height - roi_height_px))

                # Draw green ROI rectangle overlay
                roi_rect = pygame.Rect(sub_x + cam_display_width + 20 + roi_x_px, sub_y + 10 + roi_y_px, roi_width_px, roi_height_px)
                pygame.draw.rect(menu_screen, (0, 255, 0), roi_rect, 2)  # Green border, 2px thickness

            # Camera 2 info (right of camera frame)
            status_font = pygame.font.Font(None, 20)
            name_text = status_font.render(f"Camera 2: {camera2_name}", True, (0, 255, 0))
            menu_screen.blit(name_text, (sub_x + cam_display_width + 30, sub_y + 10))

            info_font = pygame.font.Font(None, 16)
            fps_text = info_font.render(f"FPS: {camera2_fps:.1f}", True, (255, 255, 255))
            menu_screen.blit(fps_text, (sub_x + cam_display_width + 630, sub_height - 10))

            utc_text = info_font.render(f"UTC: {camera2_utc_ts}", True, (255, 255, 255))
            menu_screen.blit(utc_text, (sub_x + cam_display_width + 30, sub_height - 10))

            local_text = info_font.render(f"Local: {camera2_local_ts}", True, (255, 255, 255))
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

    # Camera 2 Connect button (next to status)
    camera2_connect_rect = pygame.Rect(sub_x + cam_display_width + 230, sub_y + 10, 60, 20)
    if not camera2_connected:
        camera_button_states["camera2_connect"]["hover"] = camera2_connect_rect.collidepoint(mouse_pos)
        button_color = (100, 100, 255) if camera_button_states["camera2_connect"]["hover"] else (70, 70, 200)
        pygame.draw.rect(menu_screen, button_color, camera2_connect_rect)
        text_surface = tiny_font.render("Connect", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=camera2_connect_rect.center)
        menu_screen.blit(text_surface, text_rect)

    # Camera 2 Disconnect button (next to status)
    camera2_disconnect_rect = pygame.Rect(sub_x + cam_display_width + 330, sub_y + 10, 60, 20)
    if camera2_connected:
        camera_button_states["camera2_disconnect"]["hover"] = camera2_disconnect_rect.collidepoint(mouse_pos)
        button_color = (255, 100, 100) if camera_button_states["camera2_disconnect"]["hover"] else (200, 70, 70)
        pygame.draw.rect(menu_screen, button_color, camera2_disconnect_rect)
        text_surface = tiny_font.render("Disconnect", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=camera2_disconnect_rect.center)
        menu_screen.blit(text_surface, text_rect)

def render_camera_sliders(menu_screen, tiny_font, sub_x, sub_y, sub_width, sub_height):
    """Render gain and exposure sliders - positioned within camera display area"""
    tiny_font = pygame.font.Font(None, 12)
    mouse_pos = pygame.mouse.get_pos()

    cam_display_width = (sub_width - 30) // 2  # Match the display function calculation

    if camera1_connected:
        # Camera 1 Gain Slider - Next to connect / disconnect buttons
        if camera1_prop:
            max_gain = camera1_prop.get('MaxGain', 500)
            slider_x = sub_x + 450
            slider_y = sub_y + 20  # Next to connect / disconnect buttons
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position
            gain_ratio = min(1.0, camera1_gain / max_gain)
            handle_x = slider_x + int(gain_ratio * 120)
            camera_button_states["camera1_gain_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (150, 0, 0) if camera_button_states["camera1_gain_slider"]["hover"] else (200, 0, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label
            label_text = tiny_font.render(f"Gain: {camera1_gain}", True, (255, 255, 255))
            menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Camera 1 Exposure Slider - positioned next to gain slider
        if camera1_prop:
            max_exp = camera1_prop.get('MaxExposure', 2000000)
            slider_x = sub_x + 600
            slider_y = sub_y + 20 # Next to connect / disconnect buttons
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position - logarithmic scale for exposure
            import math
            min_exposure = 1  # 1 microsecond minimum
            max_log = math.log10(max_exp / min_exposure)
            if camera1_exposure > 0:
                log_ratio = math.log10(camera1_exposure / min_exposure) / max_log
                exp_ratio = min(1.0, max(0.0, log_ratio))
            else:
                exp_ratio = 0.0
            handle_x = slider_x + int(exp_ratio * 120)
            camera_button_states["camera1_exposure_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (0, 150, 0) if camera_button_states["camera1_exposure_slider"]["hover"] else (0, 200, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label - show appropriate units based on magnitude
            exp_us = camera1_exposure
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

    if camera2_connected:
        # Camera 2 Gain Slider - positioned below camera display on right side
        if camera2_prop:
            max_gain = camera2_prop.get('MaxGain', 500)
            slider_x = sub_x + cam_display_width + 450  # Right side, below camera display
            slider_y = sub_y + 20
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position
            gain_ratio = min(1.0, camera2_gain / max_gain)
            handle_x = slider_x + int(gain_ratio * 120)
            camera_button_states["camera2_gain_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (150, 0, 0) if camera_button_states["camera2_gain_slider"]["hover"] else (200, 0, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label
            label_text = tiny_font.render(f"Gain: {camera2_gain}", True, (255, 255, 255))
            menu_screen.blit(label_text, (slider_x, slider_y + 20))

        # Camera 2 Exposure Slider - positioned below gain slider on right side
        if camera2_prop:
            max_exp = camera2_prop.get('MaxExposure', 500000)
            slider_x = sub_x + cam_display_width + 600  # Right side, below gain slider
            slider_y = sub_y + 20
            slider_color = (100, 100, 100)
            pygame.draw.rect(menu_screen, slider_color, (slider_x, slider_y, 120, 5))

            # Handle position - logarithmic scale for exposure
            import math
            min_exposure = 1  # 1 microsecond minimum
            max_log = math.log10(max_exp / min_exposure)
            if camera2_exposure > 0:
                log_ratio = math.log10(camera2_exposure / min_exposure) / max_log
                exp_ratio = min(1.0, max(0.0, log_ratio))
            else:
                exp_ratio = 0.0
            handle_x = slider_x + int(exp_ratio * 120)
            camera_button_states["camera2_exposure_slider"]["hover"] = pygame.Rect(handle_x - 5, slider_y - 5, 10, 15).collidepoint(mouse_pos)
            handle_color = (0, 150, 0) if camera_button_states["camera2_exposure_slider"]["hover"] else (0, 200, 0)
            pygame.draw.rect(menu_screen, handle_color, (handle_x - 5, slider_y - 5, 10, 15))

            # Label - show appropriate units based on magnitude
            exp_us = camera2_exposure
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


def render_camera_roi_controls(menu_screen, sub_x, sub_y, sub_width, sub_height):
    """Render ROI controls for both cameras - positioned near each camera image"""
    mouse_pos = pygame.mouse.get_pos()
    tiny_font = pygame.font.Font(None, 12)

    # Calculate camera display dimensions
    cam_display_width = (sub_width - 30) // 2
    cam_display_height = sub_height - 30

    # ROI sizes and labels

    if camera1_connected:
            # Camera 1 ROI Size Controls - positioned in upper right of camera 1 image
        for i, size_text in enumerate(roi_label_texts):
            # Position buttons vertically near upper right edge of camera 1 - moved 100 pixels left
            if i < 3:  # First column
                rect = pygame.Rect(sub_x + 10 + cam_display_width + 15 - 100, sub_y + 10 + i * 15, 28, 12)
            else:  # Second column
                rect = pygame.Rect(sub_x + 10 + cam_display_width + 55 - 100, sub_y + 10 + (i - 3) * 15, 28, 12)

            is_selected = (camera1_roi_size == i)
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

    if camera2_connected:
        # Camera 2 ROI Size Controls - positioned in upper right of camera 2 image
        for i, size_text in enumerate(roi_label_texts):
            # Position buttons vertically near upper right edge of camera 2 - moved 800 pixels right
            if i < 3:  # First column
                rect = pygame.Rect(sub_x + cam_display_width + 15 + 800, sub_y + 10 + i * 15, 28, 12)
            else:  # Second column
                rect = pygame.Rect(sub_x + cam_display_width + 55 + 800, sub_y + 10 + (i - 3) * 15, 28, 12)

            is_selected = (camera2_roi_size == i)
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


def render_camera_interface_completion(menu_screen, sub_x, sub_y, sub_width, sub_height, small_font, tiny_font):
    """Complete camera interface rendering"""
    # Combined view toggle button
    if COMBINED_VIEW_BUTTON_RECT:
        combined_text = tiny_font.render("Combined View", True, (255, 255, 255))
        menu_screen.blit(combined_text, (COMBINED_VIEW_BUTTON_RECT.x + 5, COMBINED_VIEW_BUTTON_RECT.y + 5))

    # Camera opacity slider
    if CAMERA_OPACITY_SLIDER_RECT and CAMERA_OPACITY_SLIDER_HANDLE_RECT:
        # Draw slider background
        pygame.draw.rect(menu_screen, (100, 100, 100), CAMERA_OPACITY_SLIDER_RECT)
        # Draw handle
        handle_pos = CAMERA_OPACITY_SLIDER_RECT.x + int(camera1_opacity * CAMERA_OPACITY_SLIDER_RECT.width)
        CAMERA_OPACITY_SLIDER_HANDLE_RECT.centerx = handle_pos
        CAMERA_OPACITY_SLIDER_HANDLE_RECT.centery = CAMERA_OPACITY_SLIDER_RECT.centery
        pygame.draw.rect(menu_screen, (255, 255, 0), CAMERA_OPACITY_SLIDER_HANDLE_RECT)

        # Label
        opacity_text = tiny_font.render(f"Opacity: {camera1_opacity:.1f}", True, (255, 255, 255))
        menu_screen.blit(opacity_text, (CAMERA_OPACITY_SLIDER_RECT.x, CAMERA_OPACITY_SLIDER_RECT.y - 20))

def handle_sensor_calib_events(event, pos, sub_x, sub_y, sub_width, sub_height, camera1_connected, camera2_connected):
    """Handle sensor calibration mode events - modular event handler matching new layout"""
    result = None



    if event.type == pygame.MOUSEBUTTONDOWN:

        # Calculate layout metrics to match render_sensor_calibration
        cam_display_width = (sub_width - 30) // 2  # Same calculation as in render function

        # Camera 1 Connect button (next to status info)
        camera1_connect_rect = pygame.Rect(sub_x + 230, sub_y + 10, 60, 20)
        camera1_disconnect_rect = pygame.Rect(sub_x + 330, sub_y + 10, 60, 20)

        # Camera 2 Connect button (next to status info)
        camera2_connect_rect = pygame.Rect(sub_x + cam_display_width + 230, sub_y + 10, 60, 20)
        camera2_disconnect_rect = pygame.Rect(sub_x + cam_display_width + 330, sub_y + 10, 60, 20)

        # Camera 1 Gain Slider - coordinates must match render function exactly
        camera1_gain_rect = pygame.Rect(sub_x + 450, sub_y + 20, 120, 25)  # Match: slider_x = sub_x + 450, slider_y = sub_y + 20
        camera1_exposure_rect = pygame.Rect(sub_x + 600, sub_y + 20, 120, 25)  # Match: slider_x = sub_x + 600, slider_y = sub_y + 20

        # Camera 2 Gain Slider - coordinates must match render function exactly
        camera2_gain_rect = pygame.Rect(sub_x + cam_display_width + 450, sub_y + 20, 120, 25)  # Match: slider_x = sub_x + cam_display_width + 450, slider_y = sub_y + 20
        camera2_exposure_rect = pygame.Rect(sub_x + cam_display_width + 600, sub_y + 20, 120, 25)  # Match: slider_x = sub_x + cam_display_width + 600, slider_y = sub_y + 20

        if camera1_connect_rect.collidepoint(pos) and not camera1_connected:
            if connect_camera1():
                result = {"action": "connect_camera1", "status": "Camera 1 connected"}
        elif camera1_disconnect_rect.collidepoint(pos) and camera1_connected:
            disconnect_camera1()
            result = {"action": "disconnect_camera1", "status": "Camera 1 disconnected"}
        elif camera2_connect_rect.collidepoint(pos) and not camera2_connected:
            if connect_camera2():
                result = {"action": "connect_camera2", "status": "Camera 2 connected"}
        elif camera2_disconnect_rect.collidepoint(pos) and camera2_connected:
            disconnect_camera2()
            result = {"action": "disconnect_camera2", "status": "Camera 2 disconnected"}
        # Check gain/exposure sliders - entire slider track is clickable for easier interaction
        if camera1_connected:
            # Camera 1 Gain Slider track detection
            camera1_gain_track_rect = pygame.Rect(sub_x + 450 - 10, sub_y + 20 - 10, 120 + 20, 25)
            if camera1_gain_track_rect.collidepoint(pos):
                camera_button_states["camera1_gain_slider"]["dragging"] = True
                print(f"Camera 1 gain slider clicked - track mode enabled")

            # Camera 1 Exposure Slider track detection
            camera1_exposure_track_rect = pygame.Rect(sub_x + 600 - 10, sub_y + 20 - 10, 120 + 20, 25)
            if camera1_exposure_track_rect.collidepoint(pos):
                camera_button_states["camera1_exposure_slider"]["dragging"] = True
                print(f"Camera 1 exposure slider clicked - track mode enabled")

        if camera2_connected:
            # Camera 2 Gain Slider track detection
            camera2_gain_track_rect = pygame.Rect(sub_x + cam_display_width + 450 - 10, sub_y + 20 - 10, 120 + 20, 25)
            if camera2_gain_track_rect.collidepoint(pos):
                camera_button_states["camera2_gain_slider"]["dragging"] = True
                print(f"Camera 2 gain slider clicked - track mode enabled")

            # Camera 2 Exposure Slider track detection
            camera2_exposure_track_rect = pygame.Rect(sub_x + cam_display_width + 600 - 10, sub_y + 20 - 10, 120 + 20, 25)
            if camera2_exposure_track_rect.collidepoint(pos):
                camera_button_states["camera2_exposure_slider"]["dragging"] = True
                print(f"Camera 2 exposure slider clicked - track mode enabled")

        # ROI Size and Origin Selection for Camera 1
        if camera1_connected:
            cam_display_width = (sub_width - 30) // 2
            cam_display_height = sub_height - 30

            # Camera 1 ROI Size Control buttons - positions must match render function
            for i in range(6):
                if i < 3:  # First column
                    roi_rect = pygame.Rect(sub_x + 10 + cam_display_width + 15 - 100, sub_y + 10 + i * 15, 28, 12)
                else:  # Second column
                    roi_rect = pygame.Rect(sub_x + 10 + cam_display_width + 55 - 100, sub_y + 10 + (i - 3) * 15, 28, 12)

                if roi_rect.collidepoint(pos):
                    camera1_roi_size = i
                    print(f"Camera 1 ROI size set to {roi_sizes[i]}")
                    # Apply ROI settings to camera
                    if i == 5:  # "1.0" - Full image reset
                        camera1_roi_x = 0.0
                        camera1_roi_y = 0.0
                        set_camera1_roi()  # Reset to full image
                        print("Camera 1 ROI reset to full image")
                    else:
                        set_camera1_roi()  # Apply new ROI size
                    result = {"action": f"camera1_roi_{roi_sizes[i][0]}", "status": f"Camera 1 ROI set to {roi_sizes[i]}"}
                    break

            # ROI Origin Selection by clicking on Camera 1 image
            camera1_image_rect = pygame.Rect(sub_x + 10, sub_y + 10, cam_display_width, cam_display_height)

            # Skip camera image click if it's within a slider track area
            camera1_gain_track_rect = pygame.Rect(sub_x + 450 - 10, sub_y + 20 - 10, 120 + 20, 25)
            camera1_exposure_track_rect = pygame.Rect(sub_x + 600 - 10, sub_y + 20 - 10, 120 + 20, 25)
            skip_camera1_click = camera1_gain_track_rect.collidepoint(pos) or camera1_exposure_track_rect.collidepoint(pos)

            if camera1_image_rect.collidepoint(pos) and camera1_frame is not None and not skip_camera1_click:
                # Convert click position to normalized coordinates (0.0 to 1.0)
                rel_x = (pos[0] - (sub_x + 10)) / cam_display_width
                rel_y = (pos[1] - (sub_y + 10)) / cam_display_height
                camera1_roi_x = max(0.0, min(1.0, rel_x))
                camera1_roi_y = max(0.0, min(1.0, rel_y))
                print(f"Camera 1 ROI origin set to ({camera1_roi_x:.3f}, {camera1_roi_y:.3f})")
                # Apply ROI to camera if supported
                set_camera1_roi()

        # ROI Size and Origin Selection for Camera 2
        if camera2_connected:
            cam_display_width = (sub_width - 30) // 2
            cam_display_height = sub_height - 30

            # Camera 2 ROI Size Control buttons - positions must match render function
            for i in range(6):
                if i < 3:  # First column
                    roi_rect = pygame.Rect(sub_x + cam_display_width + 15 + 800, sub_y + 10 + i * 15, 28, 12)
                else:  # Second column
                    roi_rect = pygame.Rect(sub_x + cam_display_width + 55 + 800, sub_y + 10 + (i - 3) * 15, 28, 12)

                if roi_rect.collidepoint(pos):
                    camera2_roi_size = i
                    print(f"Camera 2 ROI size set to {roi_sizes[i]}")
                    # Apply ROI settings to camera
                    if i == 5:  # "1.0" - Full image reset
                        camera2_roi_x = 0.0
                        camera2_roi_y = 0.0
                        set_camera2_roi()  # Reset to full image
                        print("Camera 2 ROI reset to full image")
                    else:
                        set_camera2_roi()  # Apply new ROI size
                    result = {"action": f"camera2_roi_{roi_sizes[i][0]}", "status": f"Camera 2 ROI set to {roi_sizes[i]}"}
                    break

            # ROI Origin Selection by clicking on Camera 2 image
            camera2_image_rect = pygame.Rect(sub_x + cam_display_width + 20, sub_y + 10, cam_display_width, cam_display_height)

            # Skip camera image click if it's within a slider track area
            camera2_gain_track_rect = pygame.Rect(sub_x + cam_display_width + 450 - 10, sub_y + 20 - 10, 120 + 20, 25)
            camera2_exposure_track_rect = pygame.Rect(sub_x + cam_display_width + 600 - 10, sub_y + 20 - 10, 120 + 20, 25)
            skip_camera2_click = camera2_gain_track_rect.collidepoint(pos) or camera2_exposure_track_rect.collidepoint(pos)

            if camera2_image_rect.collidepoint(pos) and camera2_frame is not None and not skip_camera2_click:
                # Convert click position to normalized coordinates (0.0 to 1.0)
                rel_x = (pos[0] - (sub_x + cam_display_width + 20)) / cam_display_width
                rel_y = (pos[1] - (sub_y + 10)) / cam_display_height
                camera2_roi_x = max(0.0, min(1.0, rel_x))
                camera2_roi_y = max(0.0, min(1.0, rel_y))
                print(f"Camera 2 ROI origin set to ({camera2_roi_x:.3f}, {camera2_roi_y:.3f})")
                # Apply ROI to camera if supported
                set_camera2_roi()

    elif event.type == pygame.MOUSEMOTION:
        # Handle slider dragging based on mouse button state, not stored state
        if event.buttons[0]:  # Left mouse button is pressed
            current_pos = event.pos  # Use current mouse position for motion events
            print(f"[DEBUG] MOUSEMOTION with left button: pos={current_pos}")  # Debug mouse position
            # Calculate positions to match render function
            cam_display_width = (sub_width - 30) // 2

            if camera1_connected:
                # Check if mouse is over Camera 1 Gain Slider track
                camera1_gain_track_rect = pygame.Rect(sub_x + 450 - 10, sub_y + 20 - 10, 120 + 20, 25)
                print(f"[DEBUG] Camera1 gain track rect: {camera1_gain_track_rect}")  # Debug track rect
                if camera1_gain_track_rect.collidepoint(current_pos):
                    print(f"[DEBUG] Mouse is over camera1 gain track!")  # Debug collision
                    slider_x = sub_x + 450
                    slider_width = 120
                    max_gain = camera1_prop.get('MaxGain', 500) if camera1_prop else 500
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    new_gain = int((relative_x / slider_width) * max_gain)
                    set_camera1_gain(new_gain, lambda msg: print(f"Gain: {new_gain}"))
                    return result  # Prevent other processing

                # Check if mouse is over Camera 1 Exposure Slider track
                camera1_exposure_track_rect = pygame.Rect(sub_x + 600 - 10, sub_y + 20 - 10, 120 + 20, 25)
                if camera1_exposure_track_rect.collidepoint(current_pos):
                    slider_x = sub_x + 600
                    slider_width = 120
                    max_exp = camera1_prop.get('MaxExposure', 500000) if camera1_prop else 500000
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
                    set_camera1_exposure(new_exposure, lambda msg: print(f"Exposure: {new_exposure} μs"))
                    return result  # Prevent other processing

            if camera2_connected:
                # Check if mouse is over Camera 2 Gain Slider track
                camera2_gain_track_rect = pygame.Rect(sub_x + cam_display_width + 450 - 10, sub_y + 20 - 10, 120 + 20, 25)
                if camera2_gain_track_rect.collidepoint(current_pos):
                    slider_x = sub_x + cam_display_width + 450
                    slider_width = 120
                    max_gain = camera2_prop.get('MaxGain', 500) if camera2_prop else 500
                    relative_x = min(max(current_pos[0] - slider_x, 0), slider_width)
                    new_gain = int((relative_x / slider_width) * max_gain)
                    set_camera2_gain(new_gain, lambda msg: print(f"Gain: {new_gain}"))
                    return result  # Prevent other processing

                # Check if mouse is over Camera 2 Exposure Slider track
                camera2_exposure_track_rect = pygame.Rect(sub_x + cam_display_width + 600 - 10, sub_y + 20 - 10, 120 + 20, 25)
                if camera2_exposure_track_rect.collidepoint(current_pos):
                    slider_x = sub_x + cam_display_width + 600
                    slider_width = 120
                    max_exp = camera2_prop.get('MaxExposure', 500000) if camera2_prop else 500000
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
                    set_camera2_exposure(new_exposure, lambda msg: print(f"Exposure: {new_exposure} μs"))
                    return result  # Prevent other processing

    return result
