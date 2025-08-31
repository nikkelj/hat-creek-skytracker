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
