import os
import json
import pygame
from tkinter import Tk

# ==============================================================================
# CONFIG STATE CLASS
# ==============================================================================

class ConfigState:
    """Encapsulates all configuration state for the application.
    Follows state-direct mutation architecture to reduce parameter bloat.
    """

    def __init__(self):
        # Location configuration
        self.lat_str = "34.87405877829887"
        self.lon_str = "-120.44621926328121"
        self.alt_str = "120.0"
        self.elevation_mask_str = "0.0"

        # System configuration
        self.buffer_size = 1000  # Circular buffer size (frames)
        self.image_format = "BMP"  # Capture image format (BMP/PNG)

        # Camera configuration (units: pixel_size=um, array_size_diagonal=mm, focal_length=mm, alignment_rotation=deg, gain=unitless, exposure=us)
        self.camera_configs = {
            "camera1": {
                "pixel_size": 3.75,  # μm
                "array_size_diagonal": 11.0,  # mm
                "focal_length": 25.0,  # mm
                "alignment_rotation": 0.0,  # degrees
                "gain": 1.0,  # unitless
                "exposure": 10000.0  # microseconds
            },
            "camera2": {
                "pixel_size": 3.75,  # μm
                "array_size_diagonal": 11.0,  # mm
                "focal_length": 25.0,  # mm
                "alignment_rotation": 0.0,  # degrees
                "gain": 1.0,  # unitless
                "exposure": 10000.0  # microseconds
            }
        }

        # Input field state for config mode
        self.focused_field = None
        self.cursor_pos = {
            "lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0,
            "camera1_pixel_size": 0, "camera1_array_size_diagonal": 0, "camera1_focal_length": 0, "camera1_alignment_rotation": 0,
            "camera1_gain": 0, "camera1_exposure": 0,
            "camera2_pixel_size": 0, "camera2_array_size_diagonal": 0, "camera2_focal_length": 0, "camera2_alignment_rotation": 0,
            "camera2_gain": 0, "camera2_exposure": 0
        }
        self.selection_start = {
            "lat": None, "lon": None, "alt": None, "elevation_mask": None,
            "camera1_pixel_size": None, "camera1_array_size_diagonal": None, "camera1_focal_length": None, "camera1_alignment_rotation": None,
            "camera1_gain": None, "camera1_exposure": None,
            "camera2_pixel_size": None, "camera2_array_size_diagonal": None, "camera2_focal_length": None, "camera2_alignment_rotation": None,
            "camera2_gain": None, "camera2_exposure": None
        }

    def get_config_dict(self):
        """Get configuration as dictionary for saving."""
        return {
            "lat": self.lat_str,
            "lon": self.lon_str,
            "alt": self.alt_str,
            "elevation_mask": self.elevation_mask_str,
            "camera_configs": self.camera_configs
        }

    def load_from_dict(self, config_dict):
        """Load configuration from dictionary."""
        self.lat_str = config_dict.get("lat", self.lat_str)
        self.lon_str = config_dict.get("lon", self.lon_str)
        self.alt_str = config_dict.get("alt", self.alt_str)
        self.elevation_mask_str = config_dict.get("elevation_mask", self.elevation_mask_str)

        # Load system configuration
        self.buffer_size = config_dict.get("buffer_size", self.buffer_size)
        self.image_format = config_dict.get("image_format", self.image_format)

        # Load camera configurations if present
        if "camera_configs" in config_dict:
            for camera_name, config_data in config_dict["camera_configs"].items():
                if camera_name not in self.camera_configs:
                    self.camera_configs[camera_name] = {}
                self.camera_configs[camera_name].update(config_data)

        # Ensure default values for new config fields
        defaults = {
            "pixel_size": 3.75,
            "array_size_diagonal": 11.0,
            "focal_length": 25.0,
            "alignment_rotation": 0.0,
            "gain": 1.0,
            "exposure": 10000.0
        }

        for camera_name in self.camera_configs:
            for key, default_value in defaults.items():
                if key not in self.camera_configs[camera_name]:
                    self.camera_configs[camera_name][key] = default_value

    def reset_input_fields(self):
        """Reset input field positions when switching modes."""
        self.cursor_pos = {
            "lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0,
            "camera1_pixel_size": 0, "camera1_array_size_diagonal": 0, "camera1_focal_length": 0, "camera1_alignment_rotation": 0,
            "camera1_gain": 0, "camera1_exposure": 0,
            "camera2_pixel_size": 0, "camera2_array_size_diagonal": 0, "camera2_focal_length": 0, "camera2_alignment_rotation": 0,
            "camera2_gain": 0, "camera2_exposure": 0
        }
        self.selection_start = {
            "lat": None, "lon": None, "alt": None, "elevation_mask": None,
            "camera1_pixel_size": None, "camera1_array_size_diagonal": None, "camera1_focal_length": None, "camera1_alignment_rotation": None,
            "camera1_gain": None, "camera1_exposure": None,
            "camera2_pixel_size": None, "camera2_array_size_diagonal": None, "camera2_focal_length": None, "camera2_alignment_rotation": None,
            "camera2_gain": None, "camera2_exposure": None
        }
        self.focused_field = None

    def load_from_file(self, file_path=None):
        """Load configuration from file and update state directly."""
        if file_path is None and os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    loaded_config = json.load(f)
                    self.load_from_dict(loaded_config)
            except Exception as e:
                print(f"Debug: Error loading config.json: {e}")
        elif file_path:
            try:
                with open(file_path, "r") as f:
                    loaded_config = json.load(f)
                    self.load_from_dict(loaded_config)
            except Exception as e:
                print(f"Debug: Error loading {file_path}: {e}")

    def save_to_file(self):
        """Save configuration to file."""
        config_dict = self.get_config_dict()
        with open("config.json", "w") as f:
            json.dump(config_dict, f)

    def get_camera_config(self, camera_name):
        """Get configuration for a specific camera."""
        return self.camera_configs.get(camera_name, {})

    def get_camera_pixel_size(self, camera_name):
        """Get pixel size for a camera (μm)."""
        return self.camera_configs.get(camera_name, {}).get("pixel_size", 3.75)

    def get_camera_array_size_diagonal(self, camera_name):
        """Get array size diagonal for a camera (mm)."""
        return self.camera_configs.get(camera_name, {}).get("array_size_diagonal", 11.0)

    def get_camera_focal_length(self, camera_name):
        """Get focal length for a camera (mm)."""
        return self.camera_configs.get(camera_name, {}).get("focal_length", 25.0)

    def get_camera_alignment_rotation(self, camera_name):
        """Get alignment rotation for a camera (degrees)."""
        return self.camera_configs.get(camera_name, {}).get("alignment_rotation", 0.0)

    def get_camera_gain(self, camera_name):
        """Get gain for a camera (unitless)."""
        return self.camera_configs.get(camera_name, {}).get("gain", 1.0)

    def get_camera_exposure(self, camera_name):
        """Get exposure for a camera (microseconds)."""
        return self.camera_configs.get(camera_name, {}).get("exposure", 10000.0)

    def set_camera_config(self, camera_name, config_dict):
        """Update configuration for a specific camera."""
        if camera_name not in self.camera_configs:
            self.camera_configs[camera_name] = {}
        self.camera_configs[camera_name].update(config_dict)


def load_config(file_path=None):
    """Create and initialize ConfigState from file (backward compatibility)."""
    config_state = ConfigState()
    config_state.load_from_file(file_path)
    return config_state


def load_config_legacy(file_path=None):
    """Legacy function for backward compatibility - returns dict."""
    config_state = ConfigState()
    config_state.load_from_file(file_path)
    return config_state.get_config_dict()


def save_config(config):
    """Legacy function for backward compatibility."""
    if isinstance(config, ConfigState):
        config.save_to_file()
    else:
        # Handle legacy dictionary format
        with open("config.json", "w") as f:
            json.dump(config, f)


def draw_config_options(display, config_state):
    """
    Draw the configuration options UI when in config mode.
    This includes location fields and both camera configuration groups.
    """
    import pygame
    from utils import draw_button_with_objects

    # Draw gradient background for config options
    for y in range(display.sub_height):
        color = (160 - (y / display.sub_height * 5), 160 - (y / display.sub_height * 5), 160 - (y / display.sub_height * 5))
        pygame.draw.line(display.menu_screen, color, (display.sub_x, display.sub_y + y), (display.sub_x + display.sub_width, display.sub_y + y))

    # Draw grouping box and label
    group_rect = pygame.Rect(display.sub_x + 10, display.sub_y + 0, 220, 370)
    pygame.draw.rect(display.menu_screen, (0, 0, 0), group_rect, 2, border_radius=5)
    group_label = display.font.render("Observer Location", True, (0, 0, 0))
    display.menu_screen.blit(group_label, (display.sub_x + 20, display.sub_y + 10))

    # Draw labels and inputs
    lat_label = display.font.render("Latitude:", True, (0, 0, 0))
    display.menu_screen.blit(lat_label, (display.sub_x + 20, display.sub_y + 30))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['lat'])
    lat_text = display.font.render(config_state.lat_str, True, (0, 0, 0))
    display.menu_screen.blit(lat_text, (display.input_rects['lat'].x + 5, display.input_rects['lat'].y + 5))

    lon_label = display.font.render("Longitude:", True, (0, 0, 0))
    display.menu_screen.blit(lon_label, (display.sub_x + 20, display.sub_y + 120))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['lon'])
    lon_text = display.font.render(config_state.lon_str, True, (0, 0, 0))
    display.menu_screen.blit(lon_text, (display.input_rects['lon'].x + 5, display.input_rects['lon'].y + 5))

    alt_label = display.font.render("Altitude (m):", True, (0, 0, 0))
    display.menu_screen.blit(alt_label, (display.sub_x + 20, display.sub_y + 210))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['alt'])
    alt_text = display.font.render(config_state.alt_str, True, (0, 0, 0))
    display.menu_screen.blit(alt_text, (display.input_rects['alt'].x + 5, display.input_rects['alt'].y + 5))

    elevation_mask_label = display.font.render("Elevation Mask (deg):", True, (0, 0, 0))
    display.menu_screen.blit(elevation_mask_label, (display.sub_x + 20, display.sub_y + 300))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['elevation_mask'])
    elevation_mask_text = display.font.render(config_state.elevation_mask_str, True, (0, 0, 0))
    display.menu_screen.blit(elevation_mask_text, (display.input_rects['elevation_mask'].x + 5, display.input_rects['elevation_mask'].y + 5))

    # Focus highlights for location fields
    if config_state.focused_field == "lat":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['lat'], 2)
        if config_state.focused_field == "lat":
            text_width, _ = display.font.size(config_state.lat_str[:config_state.cursor_pos["lat"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['lat'].x + 5 + text_width, display.input_rects['lat'].y + 5),
                            (display.input_rects['lat'].x + 5 + text_width, display.input_rects['lat'].y + 25), 2)

    if config_state.focused_field == "lon":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['lon'], 2)
        if config_state.focused_field == "lon":
            text_width, _ = display.font.size(config_state.lon_str[:config_state.cursor_pos["lon"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['lon'].x + 5 + text_width, display.input_rects['lon'].y + 5),
                            (display.input_rects['lon'].x + 5 + text_width, display.input_rects['lon'].y + 25), 2)

    if config_state.focused_field == "alt":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['alt'], 2)
        if config_state.focused_field == "alt":
            text_width, _ = display.font.size(config_state.alt_str[:config_state.cursor_pos["alt"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['alt'].x + 5 + text_width, display.input_rects['alt'].y + 5),
                            (display.input_rects['alt'].x + 5 + text_width, display.input_rects['alt'].y + 25), 2)

    if config_state.focused_field == "elevation_mask":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['elevation_mask'], 2)
        if config_state.focused_field == "elevation_mask":
            text_width, _ = display.font.size(config_state.elevation_mask_str[:config_state.cursor_pos["elevation_mask"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['elevation_mask'].x + 5 + text_width, display.input_rects['elevation_mask'].y + 5),
                            (display.input_rects['elevation_mask'].x + 5 + text_width, display.input_rects['elevation_mask'].y + 25), 2)

    # Camera 1 configuration
    camera1_group_rect = pygame.Rect(display.sub_x + 235, display.sub_y + 0, 240, 380)
    pygame.draw.rect(display.menu_screen, (0, 0, 0), camera1_group_rect, 2, border_radius=5)
    camera1_label = display.font.render("Camera 1", True, (0, 0, 0))
    display.menu_screen.blit(camera1_label, (display.sub_x + 245, display.sub_y + 10))

    pixel_size1_label = display.font.render("Pixel Size (μm):", True, (0, 0, 0))
    display.menu_screen.blit(pixel_size1_label, (display.sub_x + 245, display.sub_y + 35))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera1_pixel_size'])
    pixel_size1_text = display.font.render(f"{config_state.camera_configs['camera1']['pixel_size']:.2f}", True, (0, 0, 0))
    display.menu_screen.blit(pixel_size1_text, (display.input_rects['camera1_pixel_size'].x + 5, display.input_rects['camera1_pixel_size'].y + 5))

    array_diag1_label = display.font.render("Array Diagonal (mm):", True, (0, 0, 0))
    display.menu_screen.blit(array_diag1_label, (display.sub_x + 245, display.sub_y + 90))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera1_array_size_diagonal'])
    array_diag1_text = display.font.render(f"{config_state.camera_configs['camera1']['array_size_diagonal']:.1f}", True, (0, 0, 0))
    display.menu_screen.blit(array_diag1_text, (display.input_rects['camera1_array_size_diagonal'].x + 5, display.input_rects['camera1_array_size_diagonal'].y + 5))

    focal_length1_label = display.font.render("Focal Length (mm):", True, (0, 0, 0))
    display.menu_screen.blit(focal_length1_label, (display.sub_x + 245, display.sub_y + 145))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera1_focal_length'])
    focal_length1_text = display.font.render(f"{config_state.camera_configs['camera1']['focal_length']:.1f}", True, (0, 0, 0))
    display.menu_screen.blit(focal_length1_text, (display.input_rects['camera1_focal_length'].x + 5, display.input_rects['camera1_focal_length'].y + 5))

    alignment_rotation1_label = display.font.render("Alignment Rotation (deg):", True, (0, 0, 0))
    display.menu_screen.blit(alignment_rotation1_label, (display.sub_x + 245, display.sub_y + 205))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera1_alignment_rotation'])
    alignment_rotation1_text = display.font.render(f"{config_state.camera_configs['camera1']['alignment_rotation']:.1f}", True, (0, 0, 0))
    display.menu_screen.blit(alignment_rotation1_text, (display.input_rects['camera1_alignment_rotation'].x + 5, display.input_rects['camera1_alignment_rotation'].y + 5))

    gain1_label = display.font.render("Gain:", True, (0, 0, 0))
    display.menu_screen.blit(gain1_label, (display.sub_x + 245, display.sub_y + 265))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera1_gain'])
    gain1_text = display.font.render(f"{config_state.camera_configs['camera1']['gain']:.2f}", True, (0, 0, 0))
    display.menu_screen.blit(gain1_text, (display.input_rects['camera1_gain'].x + 5, display.input_rects['camera1_gain'].y + 5))

    exposure1_label = display.font.render("Exposure (µs):", True, (0, 0, 0))
    display.menu_screen.blit(exposure1_label, (display.sub_x + 245, display.sub_y + 325))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera1_exposure'])
    exposure1_text = display.font.render(f"{config_state.camera_configs['camera1']['exposure']:.0f}", True, (0, 0, 0))
    display.menu_screen.blit(exposure1_text, (display.input_rects['camera1_exposure'].x + 5, display.input_rects['camera1_exposure'].y + 5))

    # Camera 2 configuration
    camera2_group_rect = pygame.Rect(display.sub_x + 235, display.sub_y + 390, 240, 360)
    pygame.draw.rect(display.menu_screen, (0, 0, 0), camera2_group_rect, 2, border_radius=5)
    camera2_label = display.font.render("Camera 2", True, (0, 0, 0))
    display.menu_screen.blit(camera2_label, (display.sub_x + 245, display.sub_y + 400))

    pixel_size2_label = display.font.render("Pixel Size (μm):", True, (0, 0, 0))
    display.menu_screen.blit(pixel_size2_label, (display.sub_x + 245, display.sub_y + 415))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera2_pixel_size'])
    pixel_size2_text = display.font.render(f"{config_state.camera_configs['camera2']['pixel_size']:.2f}", True, (0, 0, 0))
    display.menu_screen.blit(pixel_size2_text, (display.input_rects['camera2_pixel_size'].x + 5, display.input_rects['camera2_pixel_size'].y + 5))

    array_diag2_label = display.font.render("Array Diagonal (mm):", True, (0, 0, 0))
    display.menu_screen.blit(array_diag2_label, (display.sub_x + 245, display.sub_y + 470))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera2_array_size_diagonal'])
    array_diag2_text = display.font.render(f"{config_state.camera_configs['camera2']['array_size_diagonal']:.1f}", True, (0, 0, 0))
    display.menu_screen.blit(array_diag2_text, (display.input_rects['camera2_array_size_diagonal'].x + 5, display.input_rects['camera2_array_size_diagonal'].y + 5))

    focal_length2_label = display.font.render("Focal Length (mm):", True, (0, 0, 0))
    display.menu_screen.blit(focal_length2_label, (display.sub_x + 245, display.sub_y + 525))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera2_focal_length'])
    focal_length2_text = display.font.render(f"{config_state.camera_configs['camera2']['focal_length']:.1f}", True, (0, 0, 0))
    display.menu_screen.blit(focal_length2_text, (display.input_rects['camera2_focal_length'].x + 5, display.input_rects['camera2_focal_length'].y + 5))

    alignment_rotation2_label = display.font.render("Alignment Rotation (deg):", True, (0, 0, 0))
    display.menu_screen.blit(alignment_rotation2_label, (display.sub_x + 245, display.sub_y + 585))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera2_alignment_rotation'])
    alignment_rotation2_text = display.font.render(f"{config_state.camera_configs['camera2']['alignment_rotation']:.1f}", True, (0, 0, 0))
    display.menu_screen.blit(alignment_rotation2_text, (display.input_rects['camera2_alignment_rotation'].x + 5, display.input_rects['camera2_alignment_rotation'].y + 5))

    gain2_label = display.font.render("Gain:", True, (0, 0, 0))
    display.menu_screen.blit(gain2_label, (display.sub_x + 245, display.sub_y + 645))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera2_gain'])
    gain2_text = display.font.render(f"{config_state.camera_configs['camera2']['gain']:.2f}", True, (0, 0, 0))
    display.menu_screen.blit(gain2_text, (display.input_rects['camera2_gain'].x + 5, display.input_rects['camera2_gain'].y + 5))

    exposure2_label = display.font.render("Exposure (µs):", True, (0, 0, 0))
    display.menu_screen.blit(exposure2_label, (display.sub_x + 245, display.sub_y + 695))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['camera2_exposure'])
    exposure2_text = display.font.render(f"{config_state.camera_configs['camera2']['exposure']:.0f}", True, (0, 0, 0))
    display.menu_screen.blit(exposure2_text, (display.input_rects['camera2_exposure'].x + 5, display.input_rects['camera2_exposure'].y + 5))

    # Camera field focus highlights
    # Camera 1 fields
    if config_state.focused_field == "camera1_pixel_size":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera1_pixel_size'], 2)
        pixel_size1_str = f"{config_state.camera_configs['camera1']['pixel_size']:.2f}"
        text_width, _ = display.font.size(pixel_size1_str[:config_state.cursor_pos["camera1_pixel_size"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera1_pixel_size'].x + 5 + text_width, display.input_rects['camera1_pixel_size'].y + 5),
                        (display.input_rects['camera1_pixel_size'].x + 5 + text_width, display.input_rects['camera1_pixel_size'].y + 25), 2)

    if config_state.focused_field == "camera1_array_size_diagonal":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera1_array_size_diagonal'], 2)
        array_diag1_str = f"{config_state.camera_configs['camera1']['array_size_diagonal']:.1f}"
        text_width, _ = display.font.size(array_diag1_str[:config_state.cursor_pos["camera1_array_size_diagonal"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera1_array_size_diagonal'].x + 5 + text_width, display.input_rects['camera1_array_size_diagonal'].y + 5),
                        (display.input_rects['camera1_array_size_diagonal'].x + 5 + text_width, display.input_rects['camera1_array_size_diagonal'].y + 25), 2)

    if config_state.focused_field == "camera1_focal_length":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera1_focal_length'], 2)
        focal_length1_str = f"{config_state.camera_configs['camera1']['focal_length']:.1f}"
        text_width, _ = display.font.size(focal_length1_str[:config_state.cursor_pos["camera1_focal_length"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera1_focal_length'].x + 5 + text_width, display.input_rects['camera1_focal_length'].y + 5),
                        (display.input_rects['camera1_focal_length'].x + 5 + text_width, display.input_rects['camera1_focal_length'].y + 25), 2)

    if config_state.focused_field == "camera1_alignment_rotation":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera1_alignment_rotation'], 2)
        alignment_rotation1_str = f"{config_state.camera_configs['camera1']['alignment_rotation']:.1f}"
        text_width, _ = display.font.size(alignment_rotation1_str[:config_state.cursor_pos["camera1_alignment_rotation"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera1_alignment_rotation'].x + 5 + text_width, display.input_rects['camera1_alignment_rotation'].y + 5),
                        (display.input_rects['camera1_alignment_rotation'].x + 5 + text_width, display.input_rects['camera1_alignment_rotation'].y + 25), 2)

    if config_state.focused_field == "camera1_gain":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera1_gain'], 2)
        gain1_str = f"{config_state.camera_configs['camera1']['gain']:.2f}"
        text_width, _ = display.font.size(gain1_str[:config_state.cursor_pos["camera1_gain"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera1_gain'].x + 5 + text_width, display.input_rects['camera1_gain'].y + 5),
                        (display.input_rects['camera1_gain'].x + 5 + text_width, display.input_rects['camera1_gain'].y + 25), 2)

    if config_state.focused_field == "camera1_exposure":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera1_exposure'], 2)
        exposure1_str = f"{config_state.camera_configs['camera1']['exposure']:.0f}"
        text_width, _ = display.font.size(exposure1_str[:config_state.cursor_pos["camera1_exposure"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera1_exposure'].x + 5 + text_width, display.input_rects['camera1_exposure'].y + 5),
                        (display.input_rects['camera1_exposure'].x + 5 + text_width, display.input_rects['camera1_exposure'].y + 25), 2)

    # Camera 2 fields
    if config_state.focused_field == "camera2_pixel_size":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera2_pixel_size'], 2)
        pixel_size2_str = f"{config_state.camera_configs['camera2']['pixel_size']:.2f}"
        text_width, _ = display.font.size(pixel_size2_str[:config_state.cursor_pos["camera2_pixel_size"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera2_pixel_size'].x + 5 + text_width, display.input_rects['camera2_pixel_size'].y + 5),
                        (display.input_rects['camera2_pixel_size'].x + 5 + text_width, display.input_rects['camera2_pixel_size'].y + 25), 2)

    if config_state.focused_field == "camera2_array_size_diagonal":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera2_array_size_diagonal'], 2)
        array_diag2_str = f"{config_state.camera_configs['camera2']['array_size_diagonal']:.1f}"
        text_width, _ = display.font.size(array_diag2_str[:config_state.cursor_pos["camera2_array_size_diagonal"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera2_array_size_diagonal'].x + 5 + text_width, display.input_rects['camera2_array_size_diagonal'].y + 5),
                        (display.input_rects['camera2_array_size_diagonal'].x + 5 + text_width, display.input_rects['camera2_array_size_diagonal'].y + 25), 2)

    if config_state.focused_field == "camera2_focal_length":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera2_focal_length'], 2)
        focal_length2_str = f"{config_state.camera_configs['camera2']['focal_length']:.1f}"
        text_width, _ = display.font.size(focal_length2_str[:config_state.cursor_pos["camera2_focal_length"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera2_focal_length'].x + 5 + text_width, display.input_rects['camera2_focal_length'].y + 5),
                        (display.input_rects['camera2_focal_length'].x + 5 + text_width, display.input_rects['camera2_focal_length'].y + 25), 2)

    if config_state.focused_field == "camera2_alignment_rotation":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera2_alignment_rotation'], 2)
        alignment_rotation2_str = f"{config_state.camera_configs['camera2']['alignment_rotation']:.1f}"
        text_width, _ = display.font.size(alignment_rotation2_str[:config_state.cursor_pos["camera2_alignment_rotation"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera2_alignment_rotation'].x + 5 + text_width, display.input_rects['camera2_alignment_rotation'].y + 5),
                        (display.input_rects['camera2_alignment_rotation'].x + 5 + text_width, display.input_rects['camera2_alignment_rotation'].y + 25), 2)

    if config_state.focused_field == "camera2_gain":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera2_gain'], 2)
        gain2_str = f"{config_state.camera_configs['camera2']['gain']:.2f}"
        text_width, _ = display.font.size(gain2_str[:config_state.cursor_pos["camera2_gain"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera2_gain'].x + 5 + text_width, display.input_rects['camera2_gain'].y + 5),
                        (display.input_rects['camera2_gain'].x + 5 + text_width, display.input_rects['camera2_gain'].y + 25), 2)

    if config_state.focused_field == "camera2_exposure":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['camera2_exposure'], 2)
        exposure2_str = f"{config_state.camera_configs['camera2']['exposure']:.0f}"
        text_width, _ = display.font.size(exposure2_str[:config_state.cursor_pos["camera2_exposure"]])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                        (display.input_rects['camera2_exposure'].x + 5 + text_width, display.input_rects['camera2_exposure'].y + 5),
                        (display.input_rects['camera2_exposure'].x + 5 + text_width, display.input_rects['camera2_exposure'].y + 25), 2)

    # Draw buttons and divider
    pygame.draw.line(display.menu_screen, (0, 0, 0), (display.sub_x, display.sub_y + display.sub_height - 60), (display.sub_x + display.sub_width, display.sub_y + display.sub_height - 60), 1)
    # Config mode buttons use display object properties
    draw_button_with_objects(display, "save")
    draw_button_with_objects(display, "load")


def handle_input(event, config_state):
    """Updated handle_input that works directly with ConfigState object."""
    if config_state.focused_field is None:
        return

    focused_field = config_state.focused_field

    # Get field value from config_state
    if focused_field == "lat":
        field_str = config_state.lat_str
    elif focused_field == "lon":
        field_str = config_state.lon_str
    elif focused_field == "alt":
        field_str = config_state.alt_str
    elif focused_field == "elevation_mask":
        field_str = config_state.elevation_mask_str
    elif focused_field == "camera1_pixel_size":
        field_str = str(config_state.camera_configs["camera1"]["pixel_size"])
    elif focused_field == "camera1_array_size_diagonal":
        field_str = str(config_state.camera_configs["camera1"]["array_size_diagonal"])
    elif focused_field == "camera1_focal_length":
        field_str = str(config_state.camera_configs["camera1"]["focal_length"])
    elif focused_field == "camera1_alignment_rotation":
        field_str = str(config_state.camera_configs["camera1"]["alignment_rotation"])
    elif focused_field == "camera2_pixel_size":
        field_str = str(config_state.camera_configs["camera2"]["pixel_size"])
    elif focused_field == "camera2_array_size_diagonal":
        field_str = str(config_state.camera_configs["camera2"]["array_size_diagonal"])
    elif focused_field == "camera2_focal_length":
        field_str = str(config_state.camera_configs["camera2"]["focal_length"])
    elif focused_field == "camera1_gain":
        field_str = str(config_state.camera_configs["camera1"]["gain"])
    elif focused_field == "camera1_exposure":
        field_str = str(config_state.camera_configs["camera1"]["exposure"])
    elif focused_field == "camera2_gain":
        field_str = str(config_state.camera_configs["camera2"]["gain"])
    elif focused_field == "camera2_exposure":
        field_str = str(config_state.camera_configs["camera2"]["exposure"])
    elif focused_field == "camera2_alignment_rotation":
        field_str = str(config_state.camera_configs["camera2"]["alignment_rotation"])
    else:
        return

    mods = pygame.key.get_mods()

    # Handle navigation and text editing keys
    if event.key == pygame.K_LEFT:
        if mods & pygame.KMOD_SHIFT:
            config_state.selection_start[focused_field] = config_state.cursor_pos[focused_field] if config_state.selection_start[focused_field] is None else config_state.selection_start[focused_field]
            config_state.cursor_pos[focused_field] = max(0, config_state.cursor_pos[focused_field] - 1)
        else:
            config_state.cursor_pos[focused_field] = max(0, config_state.cursor_pos[focused_field] - 1)
            config_state.selection_start[focused_field] = None
    elif event.key == pygame.K_RIGHT:
        if mods & pygame.KMOD_SHIFT:
            config_state.selection_start[focused_field] = config_state.cursor_pos[focused_field] if config_state.selection_start[focused_field] is None else config_state.selection_start[focused_field]
            config_state.cursor_pos[focused_field] = min(len(field_str), config_state.cursor_pos[focused_field] + 1)
        else:
            config_state.cursor_pos[focused_field] = min(len(field_str), config_state.cursor_pos[focused_field] + 1)
            config_state.selection_start[focused_field] = None
    elif event.key == pygame.K_HOME:
        if mods & pygame.KMOD_SHIFT:
            config_state.selection_start[focused_field] = config_state.cursor_pos[focused_field] if config_state.selection_start[focused_field] is None else config_state.selection_start[focused_field]
        config_state.cursor_pos[focused_field] = 0
        if not mods & pygame.KMOD_SHIFT:
            config_state.selection_start[focused_field] = None
    elif event.key == pygame.K_END:
        if mods & pygame.KMOD_SHIFT:
            config_state.selection_start[focused_field] = config_state.cursor_pos[focused_field] if config_state.selection_start[focused_field] is None else config_state.selection_start[focused_field]
            config_state.cursor_pos[focused_field] = len(field_str)
        if not mods & pygame.KMOD_SHIFT:
            config_state.selection_start[focused_field] = None
    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
        start = min(config_state.cursor_pos[focused_field], config_state.selection_start[focused_field]) if config_state.selection_start[focused_field] is not None else config_state.cursor_pos[focused_field]
        end = max(config_state.cursor_pos[focused_field], config_state.selection_start[focused_field]) if config_state.selection_start[focused_field] is not None else config_state.cursor_pos[focused_field] + 1 if event.key == pygame.K_DELETE else config_state.cursor_pos[focused_field]
        if start < end:
            field_str = field_str[:start] + field_str[end:]
            config_state.cursor_pos[focused_field] = start
            config_state.selection_start[focused_field] = None
        elif event.key == pygame.K_BACKSPACE and config_state.cursor_pos[focused_field] > 0:
            field_str = field_str[:config_state.cursor_pos[focused_field] - 1] + field_str[config_state.cursor_pos[focused_field]:]
            config_state.cursor_pos[focused_field] -= 1
            config_state.selection_start[focused_field] = None
    elif event.key == pygame.K_RETURN:
        config_state.focused_field = None
    else:
        char = event.unicode
        if char.isdigit() or char in ['.', '-', '+']:
            start = min(config_state.cursor_pos[focused_field], config_state.selection_start[focused_field]) if config_state.selection_start[focused_field] is not None else config_state.cursor_pos[focused_field]
            end = max(config_state.cursor_pos[focused_field], config_state.selection_start[focused_field]) if config_state.selection_start[focused_field] is not None else config_state.cursor_pos[focused_field]
            field_str = field_str[:start] + char + field_str[end:]
            config_state.cursor_pos[focused_field] += 1
            config_state.selection_start[focused_field] = None

    # Update the appropriate field in config_state
    if focused_field == "lat":
        config_state.lat_str = field_str
    elif focused_field == "lon":
        config_state.lon_str = field_str
    elif focused_field == "alt":
        config_state.alt_str = field_str
    elif focused_field == "elevation_mask":
        config_state.elevation_mask_str = field_str
    elif focused_field == "camera1_pixel_size":
        try:
            config_state.camera_configs["camera1"]["pixel_size"] = float(field_str) if field_str else 3.75
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera1_array_size_diagonal":
        try:
            config_state.camera_configs["camera1"]["array_size_diagonal"] = float(field_str) if field_str else 11.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera1_focal_length":
        try:
            config_state.camera_configs["camera1"]["focal_length"] = float(field_str) if field_str else 25.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_pixel_size":
        try:
            config_state.camera_configs["camera2"]["pixel_size"] = float(field_str) if field_str else 3.75
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_array_size_diagonal":
        try:
            config_state.camera_configs["camera2"]["array_size_diagonal"] = float(field_str) if field_str else 11.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera1_alignment_rotation":
        try:
            config_state.camera_configs["camera1"]["alignment_rotation"] = float(field_str) if field_str else 0.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_pixel_size":
        try:
            config_state.camera_configs["camera2"]["pixel_size"] = float(field_str) if field_str else 3.75
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_array_size_diagonal":
        try:
            config_state.camera_configs["camera2"]["array_size_diagonal"] = float(field_str) if field_str else 11.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_focal_length":
        try:
            config_state.camera_configs["camera2"]["focal_length"] = float(field_str) if field_str else 25.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera1_gain":
        try:
            config_state.camera_configs["camera1"]["gain"] = float(field_str) if field_str else 1.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera1_exposure":
        try:
            config_state.camera_configs["camera1"]["exposure"] = float(field_str) if field_str else 10000.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_gain":
        try:
            config_state.camera_configs["camera2"]["gain"] = float(field_str) if field_str else 1.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_exposure":
        try:
            config_state.camera_configs["camera2"]["exposure"] = float(field_str) if field_str else 10000.0
        except ValueError:
            pass  # Keep current value if invalid
    elif focused_field == "camera2_alignment_rotation":
        try:
            config_state.camera_configs["camera2"]["alignment_rotation"] = float(field_str) if field_str else 0.0
        except ValueError:
            pass  # Keep current value if invalid
