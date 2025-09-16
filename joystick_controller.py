import pygame
import serial
import serial.tools.list_ports
import math
from datetime import datetime, timezone
from skyfield.api import load
from enum import Enum

# Import existing components
from lib.auxstar import NexstarHandController, RATES, Targets
from tracking_visuals import draw_polar_plot, PolarPlotMode
from camera_manager import camera_manager, update_camera_frames_from_buffers
from camera_manager import render_sensor_calibration
from utils import draw_button

# PS4 Controller Button Labels (zero-indexed)
BUTTON_LABELS = ["X", "O", "[]", "/\\", "Sh", "PS5", "Op", "LS", "RS", "L1", "R1", "D/\\", "D\\/", "D<", "D>", "Pad"]

# ==============================================================================
# JOYSTICK MODE STATE CLASS
# ==============================================================================

class JoystickModeState:
    """
    Encapsulates all state for joystick mode, following state-direct object mutation pattern.
    Similar to TrackingVisState, this manages all joystick mode specific state.
    """

    def __init__(self):
        # Initialize Pygame joystick subsystem
        pygame.joystick.init()
        print(f"Pygame joystick initialized: {pygame.joystick.get_count()} joysticks detected")

        # Joystick state
        self.joysticks = {}  # Dict of active joysticks
        self.connected_joystick = None  # Currently active joystick
        self.joystick_tare = {}  # Tare values for deadzone calibration
        self.stopped = False  # Stop button state

        # Telescope connection state
        self.telescope_connected = False
        self.telescope_controller = None
        self.selected_port = None
        self.available_ports = []

        # UI state
        self.connect_button_hover = False
        self.disconnect_button_hover = False
        self.port_dropdown_open = False
        self.port_options_rects = []

        # Polar graph integration
        self.ts = None  # Loaded timescope for polar plot
        self.current_tt = None  # Current time for polar plot

        # Capture integration
        self.capture_active = False
        self.capture_progress = 0.0
        self.capture_status = ""
        self.capture_button_rect = None

    def reset_tare(self):
        """Reset tare values for all connected joysticks"""
        self.joystick_tare = {}
        for joy in self.joysticks.values():
            self.joystick_tare[joy.get_instance_id()] = [0] * joy.get_numaxes()

    def tare_current_joystick(self):
        """Tare the currently connected joystick"""
        if self.connected_joystick is not None:
            joy = self.joysticks[self.connected_joystick]
            tare_values = []
            for i in range(joy.get_numaxes()):
                axis_value = joy.get_axis(i)
                tare_values.append(axis_value)
                print(f"Tared Axis {i} value: {axis_value:>6.3f}")
            self.joystick_tare[self.connected_joystick] = tare_values

    def get_available_serial_ports(self):
        """Get list of available serial ports"""
        self.available_ports = []
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.available_ports.append({
                'device': port.device,
                'description': port.description,
                'name': port.name or port.device
            })

        # If no port selected, pick first one available
        if not self.selected_port and self.available_ports:
            self.selected_port = self.available_ports[0]['device']

    def connect_telescope(self):
        """Connect to telescope via serial port"""
        if not self.selected_port:
            return False

        try:
            self.telescope_controller = NexstarHandController(self.selected_port)
            self.telescope_connected = True
            print(f"Connected to telescope on {self.selected_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to telescope: {e}")
            self.telescope_controller = None
            self.telescope_connected = False
            return False

    def disconnect_telescope(self):
        """Disconnect from telescope"""
        if self.telescope_controller:
            try:
                self.telescope_controller.close()
            except:
                pass
        self.telescope_controller = None
        self.telescope_connected = False
        print("Disconnected from telescope")

    def rate_control(self):
        """Handle rate control based on connected joystick"""
        if not self.telescope_connected or self.connected_joystick is None:
            return

        if self.connected_joystick not in self.joysticks:
            return

        joy = self.joysticks[self.connected_joystick]

        # Reset stopped state if needed
        if self.stopped:
            self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
            self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)
            return

        # Process axes 2 and 3 (PlayStation right stick)
        for i in [2, 3]:  # AZM and ALT axes
            if i >= joy.get_numaxes():
                continue

            axis_value = joy.get_axis(i)

            # Apply tare if available
            if self.connected_joystick in self.joystick_tare:
                tare_value = self.joystick_tare[self.connected_joystick][i]
                axis_value -= tare_value

            # Map to telescope rates (-9 to 9)
            # Clamp values to avoid extreme movements
            axis_value = max(-1.0, min(1.0, axis_value))

            # Convert to rate (0 = stop, 1-9 positive, -1 to -9 negative)
            if abs(axis_value) < 0.1:  # Deadzone
                rate = 0
            else:
                # Scale to available rates (assuming list(RATES.keys()) has 9 rates: 1-9)
                rate = int(math.floor(axis_value * 9))
                # Clamp to max rate (9)
                rate = max(-9, min(9, rate))

            # Send command
            if i == 2:  # AZM
                if rate != 0:
                    self.telescope_controller.hc_slew_fixed(Targets.AZM, rate)
                    print(f"AZM rate: {rate}")
            elif i == 3:  # ALT
                if rate != 0:
                    self.telescope_controller.hc_slew_fixed(Targets.ALT, rate)
                    print(f"ALT rate: {rate}")

    def process_joystick_events(self, event, current_mode=None, current_tracking_surface=None, tracking_vis_state=None, config_state=None):
        """Process pygame joystick events (non-mode-specific)"""
        if event.type == pygame.JOYDEVICEADDED:
            joy = pygame.joystick.Joystick(event.device_index)
            self.joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connected: {joy.get_name()}")

            # Auto-connect to first joystick
            if self.connected_joystick is None:
                self.connected_joystick = joy.get_instance_id()
                # Initialize tare values
                self.reset_tare()

        elif event.type == pygame.JOYDEVICEREMOVED:
            if event.instance_id in self.joysticks:
                del self.joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")

                # If this was the connected joystick, disconnect it
                if self.connected_joystick == event.instance_id:
                    self.connected_joystick = None

                    # Connect to remaining joystick if any
                    if self.joysticks:
                        self.connected_joystick = next(iter(self.joysticks.keys()))

        elif event.type == pygame.JOYBUTTONDOWN:
            # Handle button events based on mode
            print(f"process_joystick_events: Button {event.button} pressed in mode {current_mode}")

            # Capture button (X button) - works in ANY mode
            if event.button == 0:  # X button
                print("process_joystick_events: X button pressed - calling _handle_capture_toggle")
                self._handle_capture_toggle(current_tracking_surface, tracking_vis_state, config_state)
                return  # Don't process any other buttons after handling capture

            # Handle stop button (Circle button) - this is universal
            if event.button == 1:  # Circle button
                self.stopped = not self.stopped
                print(f"Stop toggled: {self.stopped}")

                # Stop movement immediately when stopped
                if self.stopped and self.telescope_connected:
                    self.telescope_controller.hc_slew_fixed(Targets.AZM, 0)
                    self.telescope_controller.hc_slew_fixed(Targets.ALT, 0)

            elif event.button == 2:  # Square button for tare
                print(f"Taring joystick axes")
                self.tare_current_joystick()

        elif event.type == pygame.JOYAXISMOTION:
            # Update joystick axis state for display (this stays in process for all modes)
            if self.connected_joystick is not None and event.joy == self.connected_joystick:
                # Could store axis state here if needed for display updates
                pass

    def _handle_capture_toggle(self, tracking_surface, tracking_vis_state, config_state):
        """Handle capture toggle for joystick"""
        from camera_manager import camera_manager
        from capture_manager import capture_manager

        if self.capture_active:
            # Stop capture and begin dump process for all cameras
            capture_manager.stop_capture(None, tracking_vis_state, tracking_vis_state.selected_satellite, config_state, tracking_surface)
            print("Capture stopped on all cameras, dump process started")
            self.capture_active = False
        else:
            # Start capture on all connected cameras
            # Check if any camera is available and connected
            any_camera_available = False
            for idx in range(len(camera_manager.cameras)):
                camera = camera_manager.get_camera(idx)
                if camera and camera.connected:
                    any_camera_available = True
                    break

            if any_camera_available:
                capture_manager.start_capture()  # Start capture on all cameras
                self.capture_active = True
                print("Capture started on all connected cameras")
            else:
                print("No cameras available for capture")

    # Helper methods to get states from global scope - removed to avoid circular import

    def update_polar_plot_time(self):
        """Update current time for polar plot"""
        if not self.ts:
            self.ts = load.timescale()
        self.current_tt = self.ts.now().tt

# ==============================================================================
# JOYSTICK MODE RENDERING FUNCTIONS
# ==============================================================================

def render_joystick_mode(display, joystick_state, tracking_vis_state, config_state):
    """
    Render the complete joystick mode UI
    """
    # Clear the main area
    display.menu_screen.fill((30, 30, 30), (display.sub_x, display.sub_y,
                                            display.sub_width, display.sub_height))

    # Top section - Connection controls and joystick status
    render_connection_controls(display, joystick_state)
    render_joystick_status(display, joystick_state)

    # Top right - Polar graph
    render_polar_graph(display, joystick_state, tracking_vis_state, config_state)

    # Bottom half - Capture controls and camera feeds
    render_capture_controls(display, joystick_state)
    render_camera_feeds(display, joystick_state)

def render_connection_controls(display, joystick_state):
    """Render connection controls in upper left"""
    # Update available ports
    joystick_state.get_available_serial_ports()

    # Connect/Disconnect buttons
    button_y = display.sub_y + 10
    button_width = 80
    button_height = 30

    # Connect button
    connect_rect = pygame.Rect(display.sub_x + 10, button_y, button_width, button_height)
    joystick_state.connect_button_hover = connect_rect.collidepoint(pygame.mouse.get_pos())

    if joystick_state.telescope_connected:
        button_color = (100, 100, 100)  # Grey when connected
    else:
        button_color = (100, 150, 100) if joystick_state.connect_button_hover else (100, 120, 100)

    pygame.draw.rect(display.menu_screen, button_color, connect_rect)
    connect_text = display.small_font.render("Connect", True, (255, 255, 255))
    display.menu_screen.blit(connect_text, (connect_rect.x + 5, connect_rect.y + 5))

    # Disconnect button
    disconnect_rect = pygame.Rect(display.sub_x + 100, button_y, button_width, button_height)
    joystick_state.disconnect_button_hover = disconnect_rect.collidepoint(pygame.mouse.get_pos())

    if not joystick_state.telescope_connected:
        button_color = (100, 100, 100)  # Grey when disconnected
    else:
        button_color = (150, 100, 100) if joystick_state.disconnect_button_hover else (120, 100, 100)

    pygame.draw.rect(display.menu_screen, button_color, disconnect_rect)
    disconnect_text = display.small_font.render("Disconnect", True, (255, 255, 255))
    display.menu_screen.blit(disconnect_text, (disconnect_rect.x + 5, disconnect_rect.y + 5))

    # Port selector
    port_y = button_y + 40
    port_label = display.small_font.render("Port:", True, (255, 255, 255))
    display.menu_screen.blit(port_label, (display.sub_x + 10, port_y))

    # Port dropdown box
    dropdown_width = 120
    dropdown_height = 25
    dropdown_rect = pygame.Rect(display.sub_x + 50, port_y, dropdown_width, dropdown_height)
    pygame.draw.rect(display.menu_screen, (70, 70, 70), dropdown_rect)
    pygame.draw.rect(display.menu_screen, (150, 150, 150), dropdown_rect, 1)

    if joystick_state.selected_port:
        port_text = display.small_font.render(joystick_state.selected_port, True, (255, 255, 255))
    else:
        port_text = display.small_font.render("Select Port", True, (255, 255, 255))
    display.menu_screen.blit(port_text, (dropdown_rect.x + 5, dropdown_rect.y + 3))

    # Baud rate display (fixed)
    baud_y = port_y + 30
    baud_text = display.small_font.render("Baud: 9600 (fixed)", True, (255, 255, 255))
    display.menu_screen.blit(baud_text, (display.sub_x + 10, baud_y))

    # Connection status
    status_y = baud_y + 25
    if joystick_state.telescope_connected:
        status_text = display.small_font.render("Status: Connected", True, (0, 255, 0))
    else:
        status_text = display.small_font.render("Status: Disconnected", True, (255, 0, 0))
    display.menu_screen.blit(status_text, (display.sub_x + 10, status_y))

def render_joystick_status(display, joystick_state):
    """Render joystick status below connection controls"""
    y_start = display.sub_y + 140

    # Joystick name
    if joystick_state.connected_joystick is not None and joystick_state.connected_joystick in joystick_state.joysticks:
        joy = joystick_state.joysticks[joystick_state.connected_joystick]
        name_text = display.small_font.render(f"Joystick: {joy.get_name()}", True, (255, 255, 255))
    else:
        name_text = display.small_font.render("Joystick: None", True, (255, 0, 0))
    display.menu_screen.blit(name_text, (display.sub_x + 10, y_start))

    current_y = y_start + 25

    # Button states
    if joystick_state.connected_joystick is not None:
        joy = joystick_state.joysticks[joystick_state.connected_joystick]

        # Buttons section
        buttons_label = display.small_font.render("Buttons:", True, (255, 255, 255))
        display.menu_screen.blit(buttons_label, (display.sub_x + 10, current_y))
        current_y += 20

        # Display buttons dynamically
        num_buttons = joy.get_numbuttons()
        if num_buttons > 0:
            for i in range(num_buttons):
                button_state = joy.get_button(i)
                button_color = (0, 255, 0) if button_state else (100, 100, 100)

                col = i % 4
                row = i // 4
                button_rect = pygame.Rect(display.sub_x + 10 + col * 40, current_y + row * 25, 30, 20)
                pygame.draw.rect(display.menu_screen, button_color, button_rect)

                # Use button labels if available, otherwise fall back to numbers
                if i < len(BUTTON_LABELS):
                    button_label = BUTTON_LABELS[i]
                else:
                    button_label = str(i)

                button_text = display.tiny_font.render(button_label, True, (255, 255, 255))
                text_rect = button_text.get_rect(center=button_rect.center)
                display.menu_screen.blit(button_text, text_rect)

            current_y += ((num_buttons - 1) // 4 + 1) * 25 + 15

        # Axes section
        axes_label = display.small_font.render("Axes:", True, (255, 255, 255))
        display.menu_screen.blit(axes_label, (display.sub_x + 10, current_y))
        current_y += 20

        # Display axes
        num_axes = joy.get_numaxes()
        axes_displayed = 0

        # Display first two pairs of axes as 2D boxes with crosshairs
        for pair in range(2):
            axis_x = pair * 2
            axis_y = pair * 2 + 1

            if axis_x < num_axes and axis_y < num_axes:
                # 2D controller display as square
                box_size = 60

                # Position the square box
                box_x = display.sub_x + 10
                box_y = current_y
                center_x = box_x + box_size // 2
                center_y = box_y + box_size // 2

                # Draw 2D square box background
                pygame.draw.rect(display.menu_screen, (80, 80, 80),
                               (box_x, box_y, box_size, box_size))
                pygame.draw.rect(display.menu_screen, (150, 150, 150),
                               (box_x, box_y, box_size, box_size), 1)

                # Get axis values (-1 to 1 range)
                x_val = joy.get_axis(axis_x)
                y_val = joy.get_axis(axis_y)

                # Draw crosshairs (inverted Y interpretation)
                crosshair_range = 20  # 20 pixels in each direction
                crosshair_x = center_x + int(x_val * crosshair_range)
                crosshair_y = center_y + int(y_val * crosshair_range)  # Inverted Y interpretation

                # Vertical line
                pygame.draw.line(display.menu_screen, (255, 255, 255),
                               (crosshair_x, center_y - crosshair_range),
                               (crosshair_x, center_y + crosshair_range), 1)
                # Horizontal line
                pygame.draw.line(display.menu_screen, (255, 255, 255),
                               (center_x - crosshair_range, crosshair_y),
                               (center_x + crosshair_range, crosshair_y), 1)

                # Label
                if pair == 0:
                    pair_label = "Left Stick"
                else:
                    pair_label = "Right Stick"
                label_text = display.tiny_font.render(pair_label, True, (255, 255, 255))
                display.menu_screen.blit(label_text, (display.sub_x + 10 + box_size + 10, current_y + 20))

                current_y += box_size + 10
                axes_displayed += 2

        # Display remaining axes as linear sliders
        remaining_axes = num_axes - axes_displayed
        if remaining_axes > 0:
            for i in range(axes_displayed, num_axes):
                # Linear slider
                slider_width = 100
                slider_height = 12
                slider_x = display.sub_x + 10
                slider_y = current_y

                # Draw slider background
                pygame.draw.rect(display.menu_screen, (80, 80, 80),
                               (slider_x, slider_y, slider_width, slider_height))
                pygame.draw.rect(display.menu_screen, (150, 150, 150),
                               (slider_x, slider_y, slider_width, slider_height), 1)

                # Get axis value and display position
                axis_val = joy.get_axis(i)
                slider_pos = int((axis_val + 1) / 2 * slider_width)  # Convert -1..1 to 0..width

                # Draw slider position
                pygame.draw.rect(display.menu_screen, (255, 255, 0),
                               (slider_x + slider_pos - 2, slider_y - 2, 4, slider_height + 4))

                # Slider value text - use special labels for L2/R2
                if i == 4:
                    axis_label = "L2"
                elif i == 5:
                    axis_label = "R2"
                else:
                    axis_label = f"A{i}"
                val_text = display.tiny_font.render(f"{axis_label}: {axis_val:+.2f}", True, (255, 255, 255))
                display.menu_screen.blit(val_text, (slider_x + slider_width + 10, slider_y))

                current_y += slider_height + 8

        # Hat information
        num_hats = joy.get_numhats()
        hats_label = display.small_font.render(f"Hats: {num_hats}", True, (255, 255, 255))
        display.menu_screen.blit(hats_label, (display.sub_x + 10, current_y))

def render_polar_graph(display, joystick_state, tracking_vis_state, config_state):
    """Render polar graph using existing tracking visuals function"""
    # Update current time
    joystick_state.update_polar_plot_time()

    # Use tracking visualization state for polar plot data
    # Use UPPER_RIGHT_QUADRANT mode for joystick mode
    draw_polar_plot(display, config_state, joystick_state.ts, joystick_state.current_tt, tracking_vis_state, PolarPlotMode.UPPER_RIGHT_QUADRANT)

    # Draw satellites on the polar plot
    # Calculate center coordinates for the quadrant
    quadrant_center_x = display.sub_x + display.sub_width // 2
    quadrant_center_y = display.sub_y + display.sub_height // 2

    # Import draw_satellites - need to get the existing function
    from tracking_visuals import draw_satellites
    draw_satellites(display, tracking_vis_state, quadrant_center_x, quadrant_center_y, PolarPlotMode.UPPER_RIGHT_QUADRANT)

    # Draw satellite info pane to the right of the polar plot
    from tracking_visuals import draw_details
    draw_details(display, tracking_vis_state, PolarPlotMode.UPPER_RIGHT_QUADRANT)

def render_capture_controls(display, joystick_state):
    """Render capture controls and progress indicator below polar graph"""
    try:
        # Update capture progress from capture manager
        from capture_manager import capture_manager
        progress, status = capture_manager.get_dump_progress()

        # Update joystick state
        joystick_state.capture_progress = progress
        joystick_state.capture_status = status if status else ""

        # Determine if cameras are connected and get individual camera status
        camera_connected = False
        camera_buffer_info = {}
        joystick_state.capture_active = False

        for idx in range(len(camera_manager.cameras)):
            camera = camera_manager.get_camera(idx)
            if camera and camera.connected and camera.thread:
                camera_connected = True
                buffer_info = camera.thread.get_capture_buffer_info()

                # Store individual camera info
                camera_buffer_info[idx] = buffer_info

                # Check if any camera is actively capturing
                if buffer_info.get('capture_active', False):
                    joystick_state.capture_active = True

        # Capture toggle button
        button_y = display.sub_y + display.sub_height // 2 - 80  # Between polar graph and camera feeds
        button_width = 90
        button_height = 35
        button_x = display.sub_x + display.sub_width - button_width - 20  # Right side of screen

        joystick_state.capture_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        # Button color based on capture state
        mouse_pos = pygame.mouse.get_pos()
        hover = joystick_state.capture_button_rect.collidepoint(mouse_pos)

        if joystick_state.capture_active:
            button_color = (0, 150, 0) if hover else (0, 100, 0)  # Green when active
            button_text = "Stop Capture"
        else:
            button_color = (150, 150, 150) if hover else (120, 120, 120)  # Grey when inactive
            button_text = "Start Capture"

        pygame.draw.rect(display.menu_screen, button_color, joystick_state.capture_button_rect)
        pygame.draw.rect(display.menu_screen, (200, 200, 200), joystick_state.capture_button_rect, 1)

        # Button text
        button_surface = display.small_font.render(button_text, True, (255, 255, 255))
        text_rect = button_surface.get_rect(center=joystick_state.capture_button_rect.center)
        display.menu_screen.blit(button_surface, text_rect)

        # Buffer fill progress bar (below button)
        progress_y = button_y + button_height + 10
        progress_width = button_width
        progress_height = 15

        # Progress bar background
        pygame.draw.rect(display.menu_screen, (50, 50, 50), (button_x, progress_y, progress_width, progress_height))
        pygame.draw.rect(display.menu_screen, (150, 150, 150), (button_x, progress_y, progress_width, progress_height), 1)

        # Display individual camera buffer status
        if camera_connected and camera_buffer_info:
            # Display format: C1: [frames] [percent]% | C2: [frames] [percent]%
            camera_status_lines = []
            camera_fill_ratios = []
            total_capacity = 0

            for cam_idx in sorted(camera_buffer_info.keys()):
                buffer_info = camera_buffer_info[cam_idx]
                cam_num = cam_idx + 1
                max_buffer_size = buffer_info.get('max_buffer_size', 1000)
                total_capacity += max_buffer_size

                if joystick_state.capture_active:
                    # During capture: show frame count and capture progress percentage
                    captured_frames = buffer_info.get('capture_frame_count', 0)
                    capture_progress_ratio = buffer_info.get('capture_progress_ratio', 0.0)
                    camera_fill_ratios.append(capture_progress_ratio)
                    camera_status_lines.append(f"C{cam_num}: {captured_frames} {int(capture_progress_ratio * 100)}%")
                else:
                    # When idle: show buffer capacity with zero percent
                    camera_status_lines.append(f"C{cam_num}: {max_buffer_size} 0%")

            # Draw progress bar only when capturing or dumping
            if camera_fill_ratios:
                # Use average fill ratio for progress bar color and length
                avg_fill_ratio = sum(camera_fill_ratios) / len(camera_fill_ratios)

                if avg_fill_ratio < 0.7:
                    # Green for low fill
                    fill_color = (0, int(255 * avg_fill_ratio / 0.7), 0)
                elif avg_fill_ratio < 0.9:
                    # Yellow to orange for medium fill
                    fill_level = (avg_fill_ratio - 0.7) / 0.2
                    fill_color = (int(255 * fill_level), int(200 * fill_level), 0)
                else:
                    # Red for high fill
                    fill_color = (255, int(100 * (avg_fill_ratio - 0.9) / 0.1), 0)

                # Draw progress fill based on average of both cameras
                fill_width = int(progress_width * avg_fill_ratio)
                if fill_width > 0:
                    pygame.draw.rect(display.menu_screen, fill_color,
                                    (button_x, progress_y, fill_width, progress_height))

            # Show camera status lines (C1: frames %, C2: frames %)
            if camera_status_lines:
                status_text = " | ".join(camera_status_lines)
                fill_text = display.tiny_font.render(status_text, True, (255, 255, 255))

                text_width = fill_text.get_width()
                text_y = progress_y + progress_height + 5
                text_x = button_x + (progress_width - text_width) // 2
                display.menu_screen.blit(fill_text, (text_x, text_y))

        # Status and progress information (to the left of progress bar)
        info_x = button_x - 200
        info_y = progress_y - 5

        # Capture status
        if joystick_state.capture_active:
            status_text = display.small_font.render("REC", True, (255, 0, 0))
            display.menu_screen.blit(status_text, (info_x, info_y))

            # Recording time (would need to track actual time)
            # For now, just show "Recording..." when active
            recording_text = display.tiny_font.render("Recording...", True, (255, 0, 0))
            display.menu_screen.blit(recording_text, (info_x, info_y + 15))
        else:
            status_text = display.small_font.render("Ready", True, (0, 200, 0))
            display.menu_screen.blit(status_text, (info_x, info_y))

        # Dump progress (when dumping)
        if progress > 0:
            if progress < 1.0:
                dump_progress = int(progress * 100)
                dump_text = display.tiny_font.render(f"Dumping: {dump_progress}%", True, (255, 165, 0))
            else:
                dump_text = display.tiny_font.render("Dump Complete!", True, (0, 255, 0))
            display.menu_screen.blit(dump_text, (info_x, info_y + 30))

        # Camera status
        if not camera_connected:
            no_cam_text = display.tiny_font.render("Camera not connected", True, (255, 0, 0))
            display.menu_screen.blit(no_cam_text, (info_x + 20, info_y + 15))

    except Exception as e:
        print(f"Error in render_capture_controls: {e}")
        # Fallback: render simple error message
        error_text = display.tiny_font.render("Capture Error", True, (255, 0, 0))
        display.menu_screen.blit(error_text, (display.sub_x + 10, display.sub_y + display.sub_height // 2 - 60))

def render_camera_feeds(display, joystick_state=None):
    """Render camera feeds with cross-hairs using existing camera manager"""
    try:
        # Update camera frames
        update_camera_frames_from_buffers()

        # Safely get camera status
        camera1_connected = False
        camera2_connected = False
        camera1_frame = None
        camera2_frame = None

        # Safely access cameras with bounds checking and get time/fps info
        if len(camera_manager.cameras) > 0:
            camera1 = camera_manager.cameras[0]
            camera1_connected = camera1.connected
            camera1_frame = camera1.frame

        if len(camera_manager.cameras) > 1:
            camera2 = camera_manager.cameras[1]
            camera2_connected = camera2.connected
            camera2_frame = camera2.frame

        display_width = display.sub_width - 20
        display_height = display.sub_height // 2 - 20  # Bottom half minus some margin

        # Create surface for camera area (bottom half)
        camera_area_x = display.sub_x + 10
        camera_area_y = display.sub_y + display.sub_height // 2 + 10
        camera_area_width = display.sub_width - 20
        camera_area_height = display.sub_height // 2 - 20

        display.menu_screen.fill((0, 0, 0), (camera_area_x, camera_area_y,
                                            camera_area_width, camera_area_height))

        # Check if any cameras are expected to be present
        num_available = camera_manager.get_num_cameras()
        if num_available == 0:
            # No cameras detected at all
            no_cams_text = display.small_font.render("No cameras detected", True, (255, 0, 0))
            text_rect = no_cams_text.get_rect(center=(display.sub_x + display.sub_width // 2,
                                                     display.sub_y + display.sub_height * 3 // 4))
            display.menu_screen.blit(no_cams_text, text_rect)

            if joystick_state:
                # Send status update (we'll use print for now since no callback is passed)
                print("No cameras detected - joystick mode camera features will not function")
            return

        # Camera 1 (left half of camera area)
        cam1_width = camera_area_width // 2 - 5
        cam1_height = camera_area_height

        if camera1_connected and camera1_frame is not None:
            try:
                cam1_scaled = pygame.transform.scale(camera1_frame, (cam1_width, cam1_height))
                display.menu_screen.blit(cam1_scaled, (camera_area_x, camera_area_y))

                # Draw cross-hair at center
                center_x = camera_area_x + cam1_width // 2
                center_y = camera_area_y + cam1_height // 2
                crosshair_length = 20
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y),
                                (center_x + crosshair_length, center_y), 1)  # Horizontal
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length),
                                (center_x, center_y + crosshair_length), 1)  # Vertical

                # Camera 1 label
                cam1_text = display.small_font.render("Camera 1", True, (255, 255, 255))
                display.menu_screen.blit(cam1_text, (camera_area_x + 10, camera_area_y + 10))

                # Display time and FPS info below camera 1 (similar to sensor calibration mode)
                info_font = pygame.font.Font(None, 16)
                fps_text = info_font.render(f"FPS: {camera1.fps:.1f}", True, (255, 255, 255))
                display.menu_screen.blit(fps_text, (camera_area_x + cam1_width - 80, camera_area_y + cam1_height - 25))

                utc_text = info_font.render(f"UTC: {camera1.utc_ts}", True, (255, 255, 255))
                display.menu_screen.blit(utc_text, (camera_area_x + 10, camera_area_y + cam1_height - 25))

                local_text = info_font.render(f"Local: {camera1.local_ts}", True, (255, 255, 255))
                display.menu_screen.blit(local_text, (camera_area_x + cam1_width // 2, camera_area_y + cam1_height - 25))
            except Exception as e:
                error_text = display.small_font.render("Camera 1 Error", True, (255, 0, 0))
                display.menu_screen.blit(error_text, (camera_area_x + 10, camera_area_y + 10))
                if joystick_state:
                    print(f"Error rendering Camera 1: {e}")
        elif len(camera_manager.cameras) > 0 and not camera1_connected:
            # Camera available but not connected
            not_connected_text = display.small_font.render("Not Connected", True, (255, 0, 0))
            text_rect = not_connected_text.get_rect(center=(camera_area_x + cam1_width // 2,
                                                           camera_area_y + cam1_height // 2))
            display.menu_screen.blit(not_connected_text, text_rect)

        # Camera 2 (right half of camera area)
        cam2_x = camera_area_x + camera_area_width // 2 + 5
        cam2_width = camera_area_width // 2 - 5
        cam2_height = camera_area_height

        if camera2_connected and camera2_frame is not None:
            try:
                cam2_scaled = pygame.transform.scale(camera2_frame, (cam2_width, cam2_height))
                display.menu_screen.blit(cam2_scaled, (cam2_x, camera_area_y))

                # Draw cross-hair at center
                center_x = cam2_x + cam2_width // 2
                center_y = camera_area_y + cam2_height // 2
                crosshair_length = 20
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x - crosshair_length, center_y),
                                (center_x + crosshair_length, center_y), 1)  # Horizontal
                pygame.draw.line(display.menu_screen, (255, 0, 0), (center_x, center_y - crosshair_length),
                                (center_x, center_y + crosshair_length), 1)  # Vertical

                # Camera 2 label
                cam2_text = display.small_font.render("Camera 2", True, (255, 255, 255))
                display.menu_screen.blit(cam2_text, (cam2_x + 10, camera_area_y + 10))

                # Display time and FPS info below camera 2 (similar to sensor calibration mode)
                info_font = pygame.font.Font(None, 16)
                fps_text = info_font.render(f"FPS: {camera2.fps:.1f}", True, (255, 255, 255))
                display.menu_screen.blit(fps_text, (cam2_x + cam2_width - 80, camera_area_y + cam2_height - 25))

                utc_text = info_font.render(f"UTC: {camera2.utc_ts}", True, (255, 255, 255))
                display.menu_screen.blit(utc_text, (cam2_x + 10, camera_area_y + cam2_height - 25))

                local_text = info_font.render(f"Local: {camera2.local_ts}", True, (255, 255, 255))
                display.menu_screen.blit(local_text, (cam2_x + cam2_width // 2, camera_area_y + cam2_height - 25))
            except Exception as e:
                error_text = display.small_font.render("Camera 2 Error", True, (255, 0, 0))
                display.menu_screen.blit(error_text, (cam2_x + 10, camera_area_y + 10))
                if joystick_state:
                    print(f"Error rendering Camera 2: {e}")
        elif len(camera_manager.cameras) > 1 and not camera2_connected:
            # Camera available but not connected
            not_connected_text = display.small_font.render("Not Connected", True, (255, 0, 0))
            text_rect = not_connected_text.get_rect(center=(cam2_x + cam2_width // 2,
                                                           camera_area_y + cam2_height // 2))
            display.menu_screen.blit(not_connected_text, text_rect)
        elif len(camera_manager.cameras) == 1 and num_available >= 2:
            # Second camera slot available but camera not present
            missing_text = display.small_font.render("Camera 2 Missing", True, (255, 165, 0))
            text_rect = missing_text.get_rect(center=(cam2_x + cam2_width // 2,
                                                     camera_area_y + cam2_height // 2))
            display.menu_screen.blit(missing_text, text_rect)

    except Exception as e:
        # Catch any unexpected errors and display gracefully
        display.menu_screen.fill((0, 0, 0), (display.sub_x + 10, display.sub_y + display.sub_height // 2 + 10,
                                            display.sub_width - 20, display.sub_height // 2 - 20))

        error_text = display.small_font.render(f"Camera Error: {str(e)[:40]}", True, (255, 0, 0))
        text_rect = error_text.get_rect(center=(display.sub_x + display.sub_width // 2,
                                               display.sub_y + display.sub_height * 3 // 4))
        display.menu_screen.blit(error_text, text_rect)
        print(f"Camera rendering error in joystick mode: {e}")

# ==============================================================================
# JOYSTICK MODE EVENT HANDLING
# ==============================================================================

def handle_joystick_mode_mouse_events(event, joystick_state, display, tracking_vis_state, config_state, current_tracking_surface):
    """Handle mouse events specific to joystick mode"""
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = event.pos

        # Connect button
        connect_rect = pygame.Rect(display.sub_x + 10, display.sub_y + 10, 80, 30)
        if connect_rect.collidepoint(pos) and not joystick_state.telescope_connected:
            success = joystick_state.connect_telescope()
            if success:
                print("Telescope connected")

        # Disconnect button
        disconnect_rect = pygame.Rect(display.sub_x + 100, display.sub_y + 10, 80, 30)
        if disconnect_rect.collidepoint(pos) and joystick_state.telescope_connected:
            joystick_state.disconnect_telescope()
            print("Telescope disconnected")

        # Port dropdown (simplified - click to cycle through ports)
        dropdown_rect = pygame.Rect(display.sub_x + 50, display.sub_y + 50, 120, 25)
        if dropdown_rect.collidepoint(pos):
            joystick_state.get_available_serial_ports()
            if joystick_state.available_ports:
                # Force refresh and cycle to next port
                current_index = 0
                if joystick_state.selected_port:
                    for i, port in enumerate(joystick_state.available_ports):
                        if port['device'] == joystick_state.selected_port:
                            current_index = i
                            break

                next_index = (current_index + 1) % len(joystick_state.available_ports)
                joystick_state.selected_port = joystick_state.available_ports[next_index]['device']
                print(f"Selected port: {joystick_state.selected_port}")

        # Handle satellite selection/hover in polar plot area
        quadrant_x = display.sub_x + display.sub_width // 2
        quadrant_y = display.sub_y
        quadrant_width = display.sub_width // 2
        quadrant_height = display.sub_height // 2

        quadrant_rect = pygame.Rect(quadrant_x, quadrant_y, quadrant_width, quadrant_height)
        if quadrant_rect.collidepoint(pos):
            # Mouse is over polar plot quadrant - check for satellite hover/selection
            hovered_sat = None

            # Debug: print mouse position on click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                print(f"Click at ({pos[0]}, {pos[1]}) in polar plot quadrant")

            # Pre-calculate centers and scale factor to match draw_satellites exactly
            full_screen_center_x = display.sub_x + display.sub_width // 2
            full_screen_center_y = display.sub_y + display.sub_height // 2
            quadrant_center_x = display.sub_x + display.sub_width // 2
            quadrant_center_y = display.sub_y + display.sub_height // 2
            scale_factor = 0.45

            for sat, (px, py, alt, _) in tracking_vis_state.satellite_positions.items():
                # Exact coordinate transformation matching draw_satellites
                rel_x = px - full_screen_center_x
                rel_y = py - full_screen_center_y
                trans_x = quadrant_center_x + rel_x * scale_factor
                trans_y = quadrant_center_y + rel_y * scale_factor

                if alt > 0:  # Only consider satellites above horizon
                    # Debug satellite positions on click
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        print(f"Satellite {sat.name}: orig({px},{py}) -> trans({trans_x},{trans_y})")

                    # Check if mouse is over satellite (larger hit area for easier clicking)
                    dist_to_sat = math.sqrt((pos[0] - trans_x)**2 + (pos[1] - trans_y)**2)
                    if dist_to_sat <= 15:  # 15 pixel radius for easier clicking/hovering
                        hovered_sat = sat
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            print(f"  -> Hit! Distance: {dist_to_sat}")
                        break

            # Update hover state on motion
            if event.type == pygame.MOUSEMOTION:
                tracking_vis_state.hovered_satellite = hovered_sat

            # Handle satellite selection on click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hovered_sat is not None:
                    if tracking_vis_state.selected_satellite == hovered_sat:
                        tracking_vis_state.selected_satellite = None  # Deselect if clicking same
                        print("Deselected satellite")
                    else:
                        tracking_vis_state.selected_satellite = hovered_sat  # Select new satellite
                        print(f"Selected satellite: {hovered_sat.name}")
                else:
                    print("  -> Clicked empty area")
                    # Click in empty area - deselect current selection
                    tracking_vis_state.selected_satellite = None
                    print("Deselected satellite (empty area clicked)")
        else:
            # Mouse not over polar plot area - clear hover state
            if event.type == pygame.MOUSEMOTION:
                tracking_vis_state.hovered_satellite = None

        # Capture button (mouse click)
        if (hasattr(joystick_state, 'capture_button_rect') and
            joystick_state.capture_button_rect and
            joystick_state.capture_button_rect.collidepoint(pos)):
            _handle_capture_toggle(joystick_state, tracking_vis_state, config_state, current_tracking_surface)

def _handle_capture_toggle(joystick_state, tracking_vis_state, config_state, tracking_surface=None):
    """Handle capture toggle from UI or joystick"""
    from camera_manager import camera_manager
    from capture_manager import capture_manager

    if joystick_state.capture_active:
        # Stop capture and begin dump process for all cameras
        capture_manager.stop_capture(None, tracking_vis_state, tracking_vis_state.selected_satellite, config_state, tracking_surface)
        print("Capture stopped on all cameras, dump process started")
        joystick_state.capture_active = False
    else:
        # Start capture on all connected cameras
        # Check if any camera is available and connected
        any_camera_available = False
        for idx in range(len(camera_manager.cameras)):
            camera = camera_manager.get_camera(idx)
            if camera and camera.connected:
                any_camera_available = True
                break

        if any_camera_available:
            capture_manager.start_capture()  # Start capture on all cameras
            joystick_state.capture_active = True
            print("Capture started on all connected cameras")
        else:
            print("No cameras available for capture")
