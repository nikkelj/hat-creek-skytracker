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

        # Input field state for config mode
        self.focused_field = None
        self.cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0}
        self.selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None}

    def get_config_dict(self):
        """Get configuration as dictionary for saving."""
        return {
            "lat": self.lat_str,
            "lon": self.lon_str,
            "alt": self.alt_str,
            "elevation_mask": self.elevation_mask_str
        }

    def load_from_dict(self, config_dict):
        """Load configuration from dictionary."""
        self.lat_str = config_dict.get("lat", self.lat_str)
        self.lon_str = config_dict.get("lon", self.lon_str)
        self.alt_str = config_dict.get("alt", self.alt_str)
        self.elevation_mask_str = config_dict.get("elevation_mask", self.elevation_mask_str)

    def reset_input_fields(self):
        """Reset input field positions when switching modes."""
        self.cursor_pos = {"lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0}
        self.selection_start = {"lat": None, "lon": None, "alt": None, "elevation_mask": None}
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
