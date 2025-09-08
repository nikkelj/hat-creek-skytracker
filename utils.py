import pygame

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Button Colors
BUTTON_BASE_COLOR = (211, 211, 211)      # Normal state
BUTTON_HOVER_COLOR = (225, 225, 225)     # Lighter, sleeker hover state
BUTTON_CLICKED_COLOR = (190, 190, 190)   # Softer clicked/active state

# Embossment Colors
SHADOW_COLOR = (160, 160, 160)           # Shadow for embossment
HIGHLIGHT_COLOR = (240, 240, 240)        # Highlight for embossment

# Text and UI
BUTTON_TEXT_COLOR = (0, 0, 0)            # Black text
BUTTON_FONT_SIZE = 24                    # Default button font size
LINE_THICKNESS = 1                       # Default line thickness

# Altitude ranges and thresholds (in km)
LEO_ALTITUDE_MIN = 0.0                   # Minimum LEO altitude
LEO_ALTITUDE_MAX = 2000.0                # Maximum LEO altitude
ALTITUDE_COLOR_NORMALIZATION_MIN = 0.0   # Min altitude for color normalization
ALTITUDE_COLOR_NORMALIZATION_MAX = 1000.0  # Max altitude for color normalization

# Color transition points
ALTITUDE_COLOR_TRANSITION_POINT = 0.5    # Point where color transitions from green to red

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def draw_button(surface, rect, text, state):
    # Determine color based on state
    color = BUTTON_BASE_COLOR
    if state["clicked"]:
        color = BUTTON_CLICKED_COLOR
    elif state["hover"]:
        color = BUTTON_HOVER_COLOR

    # Draw button with sleeker embossment
    pygame.draw.rect(surface, color, rect)
    if state["clicked"]:
        pygame.draw.line(surface, SHADOW_COLOR, rect.topleft, rect.bottomleft, LINE_THICKNESS)  # Left shadow
        pygame.draw.line(surface, SHADOW_COLOR, rect.topleft, rect.topright, LINE_THICKNESS)  # Top shadow
        pygame.draw.line(surface, HIGHLIGHT_COLOR, rect.bottomleft, rect.bottomright, LINE_THICKNESS)  # Bottom highlight
        pygame.draw.line(surface, HIGHLIGHT_COLOR, rect.topright, rect.bottomright, LINE_THICKNESS)  # Right highlight
    else:
        pygame.draw.line(surface, HIGHLIGHT_COLOR, rect.topleft, rect.bottomleft, LINE_THICKNESS)  # Left highlight
        pygame.draw.line(surface, HIGHLIGHT_COLOR, rect.topleft, rect.topright, LINE_THICKNESS)  # Top highlight
        pygame.draw.line(surface, SHADOW_COLOR, rect.bottomleft, rect.bottomright, LINE_THICKNESS)  # Bottom shadow
        pygame.draw.line(surface, SHADOW_COLOR, rect.topright, rect.bottomright, LINE_THICKNESS)  # Right shadow

    text_surface = pygame.font.Font(None, BUTTON_FONT_SIZE).render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)

def create_negative_image(original_image):
    negative = original_image.copy()
    for x in range(negative.get_width()):
        for y in range(negative.get_height()):
            r, g, b, a = negative.get_at((x, y))
            negative.set_at((x, y), (255 - r, 255 - g, 255 - b, a))
    return negative

def draw_button_with_objects(display, button_type):
    """
    State-direct mutation function for drawing buttons using display object properties.
    Takes display object and button_type string to access appropriate properties.
    """
    # Get button properties based on type
    if button_type == "save":
        rect = display.save_button
        text = "Save"
        state = display.button_states["save"]
    elif button_type == "load":
        rect = display.load_button
        text = "Load"
        state = display.button_states["load"]
    elif button_type == "clear_filters":
        rect = display.clear_filters_button
        text = "Clear Filters"
        state = display.button_states["clear_filters"]
    elif button_type == "pause":
        rect = display.pause_button
        text = "Pause"
        state = display.button_states["pause"]
    elif button_type == "play":
        rect = display.play_button
        text = "Play"
        state = display.button_states["play"]
    elif button_type == "recompute":
        rect = display.recompute_button
        text = "Update Traj"
        state = display.button_states["recompute"]
    elif button_type == "reset":
        rect = display.reset_button
        text = "Reset"
        state = display.button_states["reset"]
    else:
        return  # Unknown button type

    # Determine color based on state
    color = BUTTON_BASE_COLOR
    if state["clicked"]:
        color = BUTTON_CLICKED_COLOR
    elif state["hover"]:
        color = BUTTON_HOVER_COLOR

    # Draw button with sleeker embossment
    pygame.draw.rect(display.menu_screen, color, rect)
    if state["clicked"]:
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.topleft, rect.bottomleft, LINE_THICKNESS)  # Left shadow
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.topleft, rect.topright, LINE_THICKNESS)  # Top shadow
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.bottomleft, rect.bottomright, LINE_THICKNESS)  # Bottom highlight
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.topright, rect.bottomright, LINE_THICKNESS)  # Right highlight
    else:
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.topleft, rect.bottomleft, LINE_THICKNESS)  # Left highlight
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.topleft, rect.topright, LINE_THICKNESS)  # Top highlight
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.bottomleft, rect.bottomright, LINE_THICKNESS)  # Bottom shadow
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.topright, rect.bottomright, LINE_THICKNESS)  # Right shadow

    text_surface = pygame.font.Font(None, BUTTON_FONT_SIZE).render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=rect.center)
    display.menu_screen.blit(text_surface, text_rect)

def draw_menu_button(display, button):
    """
    State-direct mutation function for drawing main menu buttons.
    Takes display object and button dict to access appropriate properties.
    """
    rect = button["rect"]
    text = button["text"]
    mode = button["mode"]
    state = display.button_states[mode]

    # Determine color based on state
    color = BUTTON_BASE_COLOR
    if state["clicked"]:
        color = BUTTON_CLICKED_COLOR
    elif state["hover"]:
        color = BUTTON_HOVER_COLOR

    # Draw button with sleeker embossment
    pygame.draw.rect(display.menu_screen, color, rect)
    if state["clicked"]:
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.topleft, rect.bottomleft, LINE_THICKNESS)  # Left shadow
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.topleft, rect.topright, LINE_THICKNESS)  # Top shadow
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.bottomleft, rect.bottomright, LINE_THICKNESS)  # Bottom highlight
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.topright, rect.bottomright, LINE_THICKNESS)  # Right highlight
    else:
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.topleft, rect.bottomleft, LINE_THICKNESS)  # Left highlight
        pygame.draw.line(display.menu_screen, HIGHLIGHT_COLOR, rect.topleft, rect.topright, LINE_THICKNESS)  # Top highlight
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.bottomleft, rect.bottomright, LINE_THICKNESS)  # Bottom shadow
        pygame.draw.line(display.menu_screen, SHADOW_COLOR, rect.topright, rect.bottomright, LINE_THICKNESS)  # Right shadow

    text_surface = pygame.font.Font(None, BUTTON_FONT_SIZE).render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=rect.center)
    display.menu_screen.blit(text_surface, text_rect)

def get_altitude_color(mean_altitude_km):
    # Only apply altitude coloring for Low Earth Orbit (LEO) satellites
    if not (LEO_ALTITUDE_MIN <= mean_altitude_km <= LEO_ALTITUDE_MAX):
        return None

    # Normalize altitude within displayable range
    norm_alt = max(0.0, min(1.0, (mean_altitude_km - ALTITUDE_COLOR_NORMALIZATION_MIN) /
                               (ALTITUDE_COLOR_NORMALIZATION_MAX - ALTITUDE_COLOR_NORMALIZATION_MIN)))

    # Create color gradient: green (low alt) -> yellow -> red (high alt)
    if norm_alt < ALTITUDE_COLOR_TRANSITION_POINT:
        r = 0
        g = int(255 * (norm_alt * 2))
        b = int(255 * (1 - norm_alt * 2))
    else:
        r = int(255 * ((norm_alt - ALTITUDE_COLOR_TRANSITION_POINT) * 2))
        g = int(255 * (1 - (norm_alt - ALTITUDE_COLOR_TRANSITION_POINT) * 2))
        b = 0
    return (r, g, b)
