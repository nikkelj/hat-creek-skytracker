import pygame
import os
from utils import create_negative_image

class DisplaySetup:
    # UI Dimensions
    MENU_WIDTH = 200
    BUTTON_WIDTH = 180
    BUTTON_HEIGHT = 40
    INPUT_WIDTH = 200
    INPUT_HEIGHT = 30
    CENTER_TIME_WIDTH = 130
    DURATION_WIDTH = 130
    FILTER_WIDTH = 100

    # UI Spacing and Layout
    BUTTON_GAP = 20
    INPUT_GAP = 70
    FILTER_GAP = 50
    UI_MARGIN = 10
    UI_HEIGHT_OFFSET = 5

    # Background and Icon Images
    BG_IMAGE_SIZE = (160, 160)
    ICON_SIZE = (32, 32)
    BACKGROUND_FILENAME = 'lucky.jpg'

    # Font Sizes
    LARGE_FONT_SIZE = 36
    NORMAL_FONT_SIZE = 24
    SMALL_FONT_SIZE = 14

    # Colors (RGB tuples)
    COLOR_BACKGROUND_DARK = (30, 30, 30)
    COLOR_TEXT_WHITE = (255, 255, 255)
    COLOR_INPUT_BACKGROUND = (255, 255, 255)
    COLOR_INPUT_TEXT = (0, 0, 0)
    COLOR_FOCUS_BLUE = (0, 0, 255)

    # Frame rate and window settings
    FPS_TARGET = 60
    WINDOW_POSITION = "0,0"

    def __init__(self):
        pygame.init()
        display_info = pygame.display.Info()
        self.total_width = display_info.current_w
        self.total_height = display_info.current_h
        self.menu_screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption("Main Menu")

        # Calculate layout
        self.sub_x = self.MENU_WIDTH
        self.sub_y = 0
        self.sub_width = self.total_width - self.MENU_WIDTH
        self.sub_height = self.total_height
        self.radius = min(self.sub_width, self.sub_height) // 2 - 50  # For scroll bar width

        # Load fonts
        self.font = pygame.font.Font(None, self.NORMAL_FONT_SIZE)
        self.large_font = pygame.font.Font(None, self.LARGE_FONT_SIZE)
        self.small_font = pygame.font.Font(None, self.SMALL_FONT_SIZE)
        self.tiny_font = pygame.font.Font(None, 12)  # Smaller font for compact camera buttons
        self.status_font = pygame.font.Font(None, self.SMALL_FONT_SIZE)

        # Load background images
        self.bg_image = None
        self.bg_image_menu = None
        self.bg_image_icon = None
        self.negative_image = None
        self.rotation_angle = 0
        try:
            self.bg_image = pygame.image.load(self.BACKGROUND_FILENAME)
            self.bg_image_menu = pygame.transform.scale(self.bg_image, self.BG_IMAGE_SIZE)
            self.bg_image_icon = pygame.transform.scale(self.bg_image, self.ICON_SIZE)
            pygame.display.set_icon(self.bg_image_icon)
            self.negative_image = create_negative_image(self.bg_image_menu)
        except pygame.error:
            print(f"Warning: '{self.BACKGROUND_FILENAME}' not found. Using fallback color and no icon.")

        # Author background image for author_info mode
        self.author_bg = None
        try:
            self.author_bg = pygame.image.load(self.BACKGROUND_FILENAME)
        except pygame.error:
            print(f"Warning: '{self.BACKGROUND_FILENAME}' not found for author background. Using fallback color.")

        # Define input rectangles
        self.input_rects = {
            'lat': pygame.Rect(self.sub_x + 20, self.sub_y + 60, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'lon': pygame.Rect(self.sub_x + 20, self.sub_y + 150, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'alt': pygame.Rect(self.sub_x + 20, self.sub_y + 240, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'elevation_mask': pygame.Rect(self.sub_x + 20, self.sub_y + 330, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'alignment_azimuth': pygame.Rect(self.sub_x + 20, self.sub_y + 410, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'alignment_elevation': pygame.Rect(self.sub_x + 20, self.sub_y + 490, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'azm_offset': pygame.Rect(self.sub_x + 20, self.sub_y + 570, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'alt_offset': pygame.Rect(self.sub_x + 20, self.sub_y + 650, self.INPUT_WIDTH, self.INPUT_HEIGHT),

            # Hardware safety limits input rectangles
            'azm_limit_min': pygame.Rect(self.sub_x + 20, self.sub_y + 720, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'azm_limit_max': pygame.Rect(self.sub_x + 20, self.sub_y + 780, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'alt_limit_min': pygame.Rect(self.sub_x + 20, self.sub_y + 840, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'alt_limit_max': pygame.Rect(self.sub_x + 20, self.sub_y + 900, self.INPUT_WIDTH, self.INPUT_HEIGHT),

            # PID gain configuration input rectangles
            'pid_azm_p_gain': pygame.Rect(self.sub_x + 480, self.sub_y + 70, 90, self.INPUT_HEIGHT),
            'pid_azm_i_gain': pygame.Rect(self.sub_x + 590, self.sub_y + 70, 90, self.INPUT_HEIGHT),
            'pid_azm_d_gain': pygame.Rect(self.sub_x + 700, self.sub_y + 70, 90, self.INPUT_HEIGHT),
            'pid_alt_p_gain': pygame.Rect(self.sub_x + 480, self.sub_y + 140, 90, self.INPUT_HEIGHT),
            'pid_alt_i_gain': pygame.Rect(self.sub_x + 590, self.sub_y + 140, 90, self.INPUT_HEIGHT),
            'pid_alt_d_gain': pygame.Rect(self.sub_x + 700, self.sub_y + 140, 90, self.INPUT_HEIGHT),

            # Camera configuration input rectangles
            'camera1_pixel_size': pygame.Rect(self.sub_x + 250, self.sub_y + 55, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera1_array_size_diagonal': pygame.Rect(self.sub_x + 250, self.sub_y + 110, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera1_focal_length': pygame.Rect(self.sub_x + 250, self.sub_y + 165, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera1_alignment_rotation': pygame.Rect(self.sub_x + 250, self.sub_y + 225, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera1_gain': pygame.Rect(self.sub_x + 250, self.sub_y + 285, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera1_exposure': pygame.Rect(self.sub_x + 250, self.sub_y + 345, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera2_pixel_size': pygame.Rect(self.sub_x + 250, self.sub_y + 435, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera2_array_size_diagonal': pygame.Rect(self.sub_x + 250, self.sub_y + 490, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera2_focal_length': pygame.Rect(self.sub_x + 250, self.sub_y + 545, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera2_alignment_rotation': pygame.Rect(self.sub_x + 250, self.sub_y + 605, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera2_gain': pygame.Rect(self.sub_x + 250, self.sub_y + 665, self.INPUT_WIDTH, self.INPUT_HEIGHT),
            'camera2_exposure': pygame.Rect(self.sub_x + 250, self.sub_y + 715, self.INPUT_WIDTH, self.INPUT_HEIGHT),
        }

        # Define button rectangles
        self.save_button = pygame.Rect(self.sub_x + 20, self.sub_y + self.sub_height - 50, 100, 30)
        self.load_button = pygame.Rect(self.sub_x + 130, self.sub_y + self.sub_height - 50, 100, 30)
        self.clear_filters_button = pygame.Rect(self.sub_x + 10, self.sub_y + 10, 100, 30)
        self.recompute_button = pygame.Rect(self.sub_x + 230, self.sub_y + 10, 100, 30)
        self.reset_button = pygame.Rect(self.sub_x + 140, self.sub_y + 10, 80, 30)
        self.center_time_rect = pygame.Rect(self.sub_x + 140, self.sub_y + 90, self.CENTER_TIME_WIDTH, 30)
        self.duration_rect = pygame.Rect(self.sub_x + 140, self.sub_y + 140, self.DURATION_WIDTH, 30)
        self.filter_rect = pygame.Rect(self.sub_x + 10, self.sub_y + 70, self.FILTER_WIDTH, 30)
        self.filter_above_alt_rect = pygame.Rect(self.sub_x + 10, self.sub_y + 130, self.FILTER_WIDTH, 30)
        self.filter_below_alt_rect = pygame.Rect(self.sub_x + 10, self.sub_y + 185, self.FILTER_WIDTH, 30)
        self.scroll_bar_rect = pygame.Rect(self.sub_x + self.sub_width // 2 - self.radius, self.sub_y + self.sub_height - 35, 2 * self.radius, 10)
        self.slider_rect = pygame.Rect(self.sub_x + self.sub_width // 2 - self.radius, self.sub_y + self.sub_height - 35, 20, 10)
        self.pause_button = pygame.Rect(self.sub_x + self.sub_width // 2 + self.radius + 10, self.sub_y + self.sub_height - 45, 60, 30)
        self.play_button = pygame.Rect(self.sub_x + self.sub_width // 2 + self.radius + 80, self.sub_y + self.sub_height - 45, 60, 30)
        self.launch_button = pygame.Rect(self.sub_x + self.sub_width // 2 - self.radius - 80, self.sub_y + self.sub_height - 45, 70, 30)
        self.legend_x = self.sub_x + self.sub_width - 170
        self.legend_y = self.sub_y + self.sub_height - 160

        # Status display position
        self.image_y = self.total_height // 2  # Adjusted from original hard-coded position
        self.status_y_start = self.total_height - 14 * 5  # Space for 5 lines

        # Status log scrollback state (lines scrolled up from the newest message)
        self.status_scroll_offset = 0
        self.status_max_scroll = 0

        # Button states
        self.button_states = {}

        # Main menu buttons
        self.buttons = [
            {"rect": pygame.Rect(10, 10, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Tracking Vis", "mode": "tracking_vis"},
            {"rect": pygame.Rect(10, 60, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Sensor Calib", "mode": "sensor_calib"},
            {"rect": pygame.Rect(10, 110, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Joystick Loop", "mode": "joystick_loop"},
            {"rect": pygame.Rect(10, 160, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Post Process", "mode": "post_process"},
            {"rect": pygame.Rect(10, 210, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Config Options", "mode": "config_options"},
            {"rect": pygame.Rect(10, 260, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "HW Sim", "mode": "hw_sim"},
            {"rect": pygame.Rect(10, 310, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Alignment", "mode": "alignment"},
            {"rect": pygame.Rect(10, 360, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Author Info", "mode": "author_info"},
            {"rect": pygame.Rect(10, 410, self.BUTTON_WIDTH, self.BUTTON_HEIGHT), "text": "Exit", "mode": "exit"},
        ]

        # Initialize button states for all buttons
        for btn in self.buttons:
            self.button_states[btn["mode"]] = {"hover": False, "clicked": False}
        for mode in ["save", "load", "clear_filters", "recompute", "reset", "pause", "play", "launch"]:
            self.button_states[mode] = {"hover": False, "clicked": False}

        # Initialize camera-specific button states
        camera_button_modes = [
            "camera1_connect", "camera1_disconnect", "camera2_connect", "camera2_disconnect",
            "camera1_gain_slider", "camera2_gain_slider",
            "camera1_exposure_slider", "camera2_exposure_slider"
        ]
        for mode in camera_button_modes:
            self.button_states[mode] = {"hover": False, "clicked": False}

    @property
    def image_rect(self):
        """Rectangle for the background image."""
        return pygame.Rect((self.MENU_WIDTH - 160) // 2, self.image_y, 160, 160) if self.bg_image_menu else None

    def update_background_rotation(self):
        """Update the rotation angle for the background image."""
        self.rotation_angle = (self.rotation_angle + 1) % 360

    def get_rotated_image(self, image_rect):
        """Get the appropriate image for the background (rotated negative if over image)."""
        if image_rect and image_rect.collidepoint(pygame.mouse.get_pos()):
            return pygame.transform.rotate(self.negative_image, self.rotation_angle)
        return self.bg_image_menu

    def get_status_panel_rect(self):
        """Rect of the scrollable status log, confined to the left mode-select bar.

        Sits below the rotating logo and extends to the bottom of the bar so it is
        as tall as the available space allows (and never spills into the right pane).
        """
        margin = 6
        top = self.image_y + self.BG_IMAGE_SIZE[1] + 10  # just below the logo image
        bottom = self.total_height - margin
        if bottom - top < 60:  # guarantee a usable minimum height on short screens
            top = max(margin, bottom - 60)
        return pygame.Rect(margin, top, self.MENU_WIDTH - 2 * margin, bottom - top)

    def _wrap_text(self, text, font, max_width):
        """Greedy word-wrap a string to max_width px, hard-breaking over-long words."""
        lines = []
        cur = ""
        for word in text.split(" "):
            # Hard-break a single word that is wider than the panel
            while font.size(word)[0] > max_width and len(word) > 1:
                cut = len(word)
                while cut > 1 and font.size(word[:cut])[0] > max_width:
                    cut -= 1
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(word[:cut])
                word = word[cut:]
            trial = word if not cur else cur + " " + word
            if font.size(trial)[0] <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines if lines else [""]

    def render_status_log(self, status_messages):
        """Render the status log as a tall, word-wrapped, back-scrollable panel.

        Newest message sits at the bottom; mouse-wheel scrolling (handled in the
        main event loop via status_scroll_offset) reveals older messages above.
        """
        panel = self.get_status_panel_rect()
        font = self.status_font
        line_height = font.get_height() + 2
        pad = 4
        max_width = panel.width - 2 * pad

        # Build the full wrapped-line list (oldest -> newest)
        all_lines = []
        for msg in status_messages:
            all_lines.extend(self._wrap_text(str(msg), font, max_width))

        max_visible = max(1, (panel.height - 2 * pad) // line_height)
        total = len(all_lines)
        self.status_max_scroll = max(0, total - max_visible)
        self.status_scroll_offset = max(0, min(self.status_scroll_offset, self.status_max_scroll))

        end = total - self.status_scroll_offset
        start = max(0, end - max_visible)
        visible = all_lines[start:end]

        # Background + border so the log reads cleanly over the rotating logo
        self.menu_screen.fill(self.COLOR_BACKGROUND_DARK, panel)
        pygame.draw.rect(self.menu_screen, (70, 70, 70), panel, 1)

        # Clip so nothing can bleed past the left bar into the right pane
        prev_clip = self.menu_screen.get_clip()
        self.menu_screen.set_clip(panel)
        y = panel.y + pad
        for line in visible:
            self.menu_screen.blit(font.render(line, True, self.COLOR_TEXT_WHITE), (panel.x + pad, y))
            y += line_height
        # Hint that newer messages exist below the current scroll position
        if self.status_scroll_offset > 0:
            hint = font.render("v newer", True, (150, 150, 150))
            self.menu_screen.blit(hint, (panel.right - hint.get_width() - pad, panel.bottom - line_height))
        self.menu_screen.set_clip(prev_clip)

    def render_initial_menu(self, status_messages):
        """Render initial menu state."""
        self.menu_screen.fill(self.COLOR_BACKGROUND_DARK, (0, 0, self.MENU_WIDTH, self.total_height))
        if self.bg_image_menu:
            self.menu_screen.blit(self.bg_image_menu, (image_rect.topleft if (image_rect := self.image_rect) else ((self.MENU_WIDTH - 160) // 2, self.image_y)))

        from utils import draw_button  # Import here to avoid circular imports
        for btn in self.buttons:
            draw_button(self.menu_screen, btn["rect"], btn["text"], self.button_states[btn["mode"]])

        # Render status messages (scrollable log confined to the left bar)
        self.render_status_log(status_messages)
        pygame.display.flip()
