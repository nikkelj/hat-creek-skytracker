import os
import json
import pygame
from tkinter import Tk

def load_config(file_path=None):
    config = {"lat": "34.87405877829887", "lon": "-120.44621926328121", "alt": "120.0", "elevation_mask": "0.0"}
    if file_path is None and os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        except Exception as e:
            print(f"Debug: Error loading config.json: {e}")
    elif file_path:
        try:
            with open(file_path, "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        except Exception as e:
            print(f"Debug: Error loading {file_path}: {e}")
    return config

def save_config(config):
    with open("config.json", "w") as f:
        json.dump(config, f)

def handle_input(event, focused_field, lat_str, lon_str, alt_str, elevation_mask_str, cursor_pos, selection_start):
    field_str = locals()[f"{focused_field}_str"]
    mods = pygame.key.get_mods()
    if event.key == pygame.K_LEFT:
        if mods & pygame.KMOD_SHIFT:
            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
            cursor_pos[focused_field] = max(0, cursor_pos[focused_field] - 1)
        else:
            cursor_pos[focused_field] = max(0, cursor_pos[focused_field] - 1)
            selection_start[focused_field] = None
    elif event.key == pygame.K_RIGHT:
        if mods & pygame.KMOD_SHIFT:
            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
            cursor_pos[focused_field] = min(len(field_str), cursor_pos[focused_field] + 1)
        else:
            cursor_pos[focused_field] = min(len(field_str), cursor_pos[focused_field] + 1)
            selection_start[focused_field] = None
    elif event.key == pygame.K_HOME:
        if mods & pygame.KMOD_SHIFT:
            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
        cursor_pos[focused_field] = 0
        if not mods & pygame.KMOD_SHIFT:
            selection_start[focused_field] = None
    elif event.key == pygame.K_END:
        if mods & pygame.KMOD_SHIFT:
            selection_start[focused_field] = cursor_pos[focused_field] if selection_start[focused_field] is None else selection_start[focused_field]
            cursor_pos[focused_field] = len(field_str)
        if not mods & pygame.KMOD_SHIFT:
            selection_start[focused_field] = None
    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
        start = min(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field]
        end = max(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field] + 1 if event.key == pygame.K_DELETE else cursor_pos[focused_field]
        if start < end:
            field_str = field_str[:start] + field_str[end:]
            cursor_pos[focused_field] = start
            selection_start[focused_field] = None
        elif event.key == pygame.K_BACKSPACE and cursor_pos[focused_field] > 0:
            field_str = field_str[:cursor_pos[focused_field] - 1] + field_str[cursor_pos[focused_field]:]
            cursor_pos[focused_field] -= 1
            selection_start[focused_field] = None
    elif event.key == pygame.K_RETURN:
        focused_field = None
        selection_start[focused_field] = None
    else:
        char = event.unicode
        if char.isdigit() or char in ['.', '-', '+']:
            start = min(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field]
            end = max(cursor_pos[focused_field], selection_start[focused_field]) if selection_start[focused_field] is not None else cursor_pos[focused_field]
            field_str = field_str[:start] + char + field_str[end:]
            cursor_pos[focused_field] += 1
            selection_start[focused_field] = None
    if focused_field == 'lat':
        lat_str = field_str
    elif focused_field == 'lon':
        lon_str = field_str
    elif focused_field == 'alt':
        alt_str = field_str
    elif focused_field == 'elevation_mask':
        elevation_mask_str = field_str
    return lat_str, lon_str, alt_str, elevation_mask_str