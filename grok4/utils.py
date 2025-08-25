import pygame

def draw_button(surface, rect, text, state):
    base_color = (211, 211, 211)  # Normal state
    hover_color = (225, 225, 225)  # Lighter, sleeker hover state
    clicked_color = (190, 190, 190)  # Softer clicked/active state

    # Determine color based on state
    color = base_color
    if state["clicked"]:
        color = clicked_color
    elif state["hover"]:
        color = hover_color

    # Draw button with sleeker embossment (thinner lines, softer gradient)
    pygame.draw.rect(surface, color, rect)
    if state["clicked"]:
        pygame.draw.line(surface, (160, 160, 160), rect.topleft, rect.bottomleft, 1)  # Left shadow
        pygame.draw.line(surface, (160, 160, 160), rect.topleft, rect.topright, 1)  # Top shadow
        pygame.draw.line(surface, (240, 240, 240), rect.bottomleft, rect.bottomright, 1)  # Bottom highlight
        pygame.draw.line(surface, (240, 240, 240), rect.topright, rect.bottomright, 1)  # Right highlight
    else:
        pygame.draw.line(surface, (240, 240, 240), rect.topleft, rect.bottomleft, 1)  # Left highlight
        pygame.draw.line(surface, (240, 240, 240), rect.topleft, rect.topright, 1)  # Top highlight
        pygame.draw.line(surface, (160, 160, 160), rect.bottomleft, rect.bottomright, 1)  # Bottom shadow
        pygame.draw.line(surface, (160, 160, 160), rect.topright, rect.bottomright, 1)  # Right shadow

    text_surface = pygame.font.Font(None, 24).render(text, True, (0, 0, 0))
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
    if not (0 <= mean_altitude_km <= 2000):  # Apply only to LEO
        return None
    max_altitude = 1000.0
    min_altitude = 0.0
    norm_alt = max(0.0, min(1.0, (mean_altitude_km - min_altitude) / (max_altitude - min_altitude)))
    if norm_alt < 0.5:
        r = 0
        g = int(255 * (norm_alt * 2))
        b = int(255 * (1 - norm_alt * 2))
    else:
        r = int(255 * ((norm_alt - 0.5) * 2))
        g = int(255 * (1 - (norm_alt - 0.5) * 2))
        b = 0
    return (r, g, b)