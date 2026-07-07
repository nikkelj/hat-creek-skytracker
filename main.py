import pygame
import time
from time import perf_counter
import math
import threading
from skyfield.api import wgs84, load
from datetime import datetime, timedelta, timezone

# Camera imports (for ASI cameras)
try:
    import zwoasi as asi
    import numpy as np
    asi.init('lib/ASICamera2.dll')
    print("ZWO ASI SDK available for camera integration")
    ASI_AVAILABLE = True
except ImportError:
    print("Warning: ZWO ASI SDK not available - camera functionality will be disabled")
    ASI_AVAILABLE = False
    asi = None
except Exception as e:
    print(f"Error initializing ASI SDK: {e}")
    ASI_AVAILABLE = False
    asi = None
from utils import draw_menu_button, draw_button_with_objects
from trajectory import precompute_trajectories, update_satellite_positions, build_satellite_pass_table, read_launch_trajectories, update_launch_positions, load_trajectory_disk_cache, save_trajectory_disk_cache, quantized_now, compute_fine_selected_trajectory
from config import load_config, handle_input, draw_config_options
from hw_sim_ui import draw_hw_sim_options, handle_hw_sim_click
from post_process_ui import (
    PostProcessState, draw_post_process, handle_pp_click, handle_pp_motion,
    handle_pp_mouseup, handle_pp_wheel, handle_pp_key,
)
from alignment_ui import draw_alignment_options, handle_alignment_click, handle_alignment_camera_events
from alignment import AlignmentState
from tracking_visuals import TrackingVisState, draw_legend, draw_details, draw_camera_fov_details, draw_filters, draw_time_display, draw_satellite_count, draw_scroll_bar, draw_scroll_time_display, draw_satellite_pass_table, filter_and_sort_pass_table
from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
from camera_manager import camera_manager, render_sensor_calibration, render_camera_sliders, render_camera_roi_controls, render_combined_view_controls, handle_sensor_calib_events
from joystick_controller import JoystickModeState, handle_joystick_mode_mouse_events, render_bias_control_grid, render_feed_forward_toggle_buttons, render_pid_diagnostics, render_connection_controls, render_joystick_status, render_position_display, render_capture_controls, render_camera_feeds, render_pid_gain_sliders, handle_pid_sliders_mouse_events, handle_joystick_camera_control_events, render_navball, render_tracking_strip_charts, handle_lead_slider_mouse_events, render_plate_solve_panel, render_adsb_connection_controls, handle_adsb_fit_slider_mouse_events, render_joystick_target_panel
from lib.auxstar import Targets
from rendering_threads import TrackingVisualizationThread, JoystickVisualizationThread
from mount_control import MountControlThread
from simulator import HardwareSimulator
# Camera button initialization is now handled internally by camera_manager
from events import *

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Update Intervals (seconds)
POSITION_UPDATE_INTERVAL = 0.1  # 10 Hz
TRAJECTORY_UPDATE_INTERVAL = 900  # 15 minutes

# TLE API Configuration
TLE_API_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'

# Astrodynamics Constants
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14  # m^3/s^2
EARTH_RADIUS_KM = 6371 # Mean radius
SECONDS_PER_HOUR = 3600
WINDOW_POSITION = "0,0"



# ==============================================================================
# END CONSTANTS
# ==============================================================================

# Initialize Pygame and set up the display
from display import DisplaySetup
display = DisplaySetup()

# Enumerate cameras if available
camera_infos = []
camera1_name = "Camera 1"
camera2_name = "Camera 2"

# Update camera names with actual camera names if available
if ASI_AVAILABLE:
    try:
        camera_names = camera_manager.get_camera_names()
        if len(camera_names) > 0:
            camera1_name = camera_names[0] if len(camera_names) > 0 else "Camera 1"
        if len(camera_names) > 1:
            camera2_name = camera_names[1] if len(camera_names) > 1 else "Camera 2"
        print(f"Camera names updated: Camera 1='{camera1_name}', Camera 2='{camera2_name}'")
    except Exception as e:
        print(f"Failed to get camera names: {e}")

# Load configuration using ConfigState
config_state = load_config()

# Update camera manager with config values after loading (convert to numeric types)
if camera_manager.cameras[0]:
    camera_manager.cameras[0].gain = int(float(config_state.get_camera_gain("camera1")))
    camera_manager.cameras[0].exposure = int(float(config_state.get_camera_exposure("camera1")))
    camera_manager.cameras[0].alignment_rotation = float(config_state.get_camera_alignment_rotation("camera1"))
if camera_manager.cameras[1]:
    camera_manager.cameras[1].gain = int(float(config_state.get_camera_gain("camera2")))
    camera_manager.cameras[1].exposure = int(float(config_state.get_camera_exposure("camera2")))
    camera_manager.cameras[1].alignment_rotation = float(config_state.get_camera_alignment_rotation("camera2"))

# Input field state for tracking vis (now handled by TrackingVisState class)

current_mode = None
# Lazily created when the Post Process screen is first opened (it scans data/ and
# spins up decode worker threads, so we avoid the cost unless the user goes there).
post_process_state = None
clock = pygame.time.Clock()

# Author background image now handled by DisplaySetup class

# Initial render of main menu
status_messages = ["Starting TLE process..."]
display.render_initial_menu(status_messages)
print(f"Debug: Status - {'Starting TLE process...'}")

# Tracking Visualization State Management
global tracking_vis_state
tracking_vis_state = TrackingVisState()

# Current tracking visualization surface for capture
global current_tracking_surface
current_tracking_surface = None

# Position polling frequency measurement
import time
position_poll_count = 0
position_poll_start_time = time.time()
position_poll_frequency = 0.0
last_azm_duration = 0.0
last_alt_duration = 0.0

# Rendering Thread Management
global tracking_viz_thread
global joystick_viz_thread
# Initialize rendering threads
print("Initializing rendering threads...")
tracking_viz_thread = TrackingVisualizationThread(display, config_state, tracking_vis_state, target_fps=30)
joystick_viz_thread = JoystickVisualizationThread(display, config_state, tracking_vis_state, target_fps=30)

# Share the timescale reference with rendering threads for synchronization
ts = load.timescale()
tracking_viz_thread.set_timescale(ts)
joystick_viz_thread.set_timescale(ts)

# Start rendering threads
tracking_viz_thread.start()
joystick_viz_thread.start()
print("Rendering threads started.")

# Define status update callback for satellite loading
def update_status_callback(message):
    status_messages.append(message)
    display.menu_screen.fill(display.COLOR_BACKGROUND_DARK, (0, 0, display.MENU_WIDTH, display.total_height))  # Dark theme menu background
    for btn in display.buttons:
        draw_menu_button(display, btn)
    if display.bg_image_menu:
        display.menu_screen.blit(display.bg_image_menu, display.image_rect.topleft if display.image_rect else ((display.MENU_WIDTH - 160) // 2, display.image_y))
    display.render_status_log(status_messages)
    pygame.display.flip()
    pygame.time.wait(50)  # Brief pause to ensure display update
    print(f"Debug: Status - {message}")

# Joystick Mode State Management (initialize after update_status_callback is defined)
global joystick_mode_state
joystick_mode_state = JoystickModeState(tracking_vis_state, config_state, update_status_callback)
joystick_mode_state.ts = ts  # share the app timescale (used by the ADS-B receiver)

# Hardware simulator (mount + cameras). Inert unless sim_config.enabled; when
# enabled, connect_telescope / connect_camera hand back simulated devices so the
# whole tracking loop can run without physical hardware.
global hardware_sim
hardware_sim = HardwareSimulator(config_state, tracking_vis_state, ts)
joystick_mode_state.hardware_sim = hardware_sim
camera_manager.simulator = hardware_sim

# Automated pointing-model alignment state (Alignment sub-UI). The runner thread is
# created on demand when the operator presses Start.
global alignment_state
alignment_state = AlignmentState()
print(f"Hardware simulator ready (enabled={hardware_sim.sim_enabled()}).")

# Dedicated real-time mount control thread. Owns all serial traffic to the
# mount and runs the read -> PID -> command cycle at a fixed cadence, decoupled
# from this render loop. It no-ops while the telescope is disconnected.
global mount_control_thread
# Experiment: optionally run the control loop in Rust (skytracker_core) via the
# RustCoreLoopAdapter, behind a flag (default off). Enable with
# config_state.use_rust_core_loop = True or env SKYTRACKER_RUST_LOOP=1.
import os as _os
_use_rust_loop = bool(getattr(config_state, "use_rust_core_loop", False)) or (
    _os.environ.get("SKYTRACKER_RUST_LOOP", "") in ("1", "true", "True")
)
if _use_rust_loop:
    try:
        from rust_loop_adapter import RustCoreLoopAdapter
        _hz = int(getattr(config_state, "rust_core_loop_hz", 15))
        mount_control_thread = RustCoreLoopAdapter(joystick_mode_state, config_state, target_hz=_hz)
        print("Using RUST core control loop (skytracker_core).")
    except Exception as _e:
        print(f"Rust core loop unavailable ({_e}); falling back to MountControlThread.")
        mount_control_thread = MountControlThread(joystick_mode_state, config_state, target_hz=15)
else:
    mount_control_thread = MountControlThread(joystick_mode_state, config_state, target_hz=15)
mount_control_thread.start()
print("Mount control thread started.")

try:
    from satellite_data import load_satellite_data, create_satellite_labels_and_metadata
    load_satellite_data(tracking_vis_state, update_status_callback)

    # Pre-compute satellite labels and mean altitudes using refactored module
    if tracking_vis_state.tle_loaded and tracking_vis_state.satellites:
        create_satellite_labels_and_metadata(tracking_vis_state, update_status_callback)
    else:
        update_status_callback("WARNING: No TLEs or Satellites loaded!!!")
except Exception as e:
    status_messages.append(f"TLE loading error: {str(e)}")
    print(f"Debug: Error loading TLEs: {e}")
    # Force immediate trajectory computation after TLEs are loaded
    last_trajectory_update = 0

# Load launch trajectories eagerly at startup so the "Launch Trajectories" box
# (and the Launch button, which only shows once a launch is selected) is available
# immediately. Previously this only happened inside the background precompute
# worker, so launches didn't appear until several seconds after launch/recompute.
# Loading is cheap (binary-cached); the worker still refreshes it each cycle to
# pick up any launch files added during the session.
try:
    tracking_vis_state.launch_trajectories = read_launch_trajectories("./launches", display, update_status_callback)
except Exception as e:
    print(f"Debug: Error loading launch trajectories at startup: {e}")

# Force immediate trajectory computation with no delays
recompute_triggered = True
# Force immediate timer trigger
last_trajectory_update = -1
last_update_time = 0
update_interval = 1.0 / 30.0  # Target 30 Hz (vectorized update is ~0.6 ms, so cheap)
last_trajectory_update = -1  # Force immediate trigger on first loop
trajectory_interval = 900  # 15 minutes

# ==============================================================================
# MAIN PROGRAM LOOP
# ==============================================================================

def start_precompute_worker(center_time, duration_minutes, observer):
    """Run the heavy trajectory pipeline (visibility filter + precise pass +
    launch trajectories + pass table) on a background daemon thread so the main
    loop keeps drawing and handling input while it runs -- this is what makes the
    window appear instantly at launch instead of freezing for a couple seconds.

    Single-flight: callers must check tracking_vis_state.precompute_in_progress
    first. The worker only appends to status_messages (the main loop renders them
    every frame); it must NOT touch pygame/display drawing from this thread.
    precompute_trajectories publishes its result dicts atomically, so the render
    thread always sees a complete trajectory set."""
    def _worker():
        def cb(msg):
            status_messages.append(msg)  # list.append is atomic in CPython
        try:
            # Disk cache: try to serve this exact window from a previous run so the
            # precise pass is skipped, then refresh it after computing. Window
            # bounds must match precompute_trajectories' internal t0/t1 exactly.
            t0_tt = ts.utc(center_time - timedelta(minutes=duration_minutes / 2)).tt
            t1_tt = ts.utc(center_time + timedelta(minutes=duration_minutes / 2)).tt
            tle_path = tracking_vis_state.cache_file
            hits = load_trajectory_disk_cache(t0_tt, t1_tt, tle_path=tle_path)
            if hits:
                cb(f"Loaded {hits} trajectories from disk cache")

            precompute_trajectories(tracking_vis_state, observer, ts, display, cb,
                                    center_time, duration_minutes)
            launch_trajectories = read_launch_trajectories("./launches", display, cb)
            tracking_vis_state.launch_trajectories = launch_trajectories
            build_satellite_pass_table(tracking_vis_state,
                                       elevation_mask_deg=float(config_state.elevation_mask_str or 10.0),
                                       ts=ts)
            tracking_vis_state.trajectories_ready = True
            cb("Trajectories ready")

            # Persist this window for the next launch (best-effort; off critical path).
            if not hits:
                save_trajectory_disk_cache(tracking_vis_state, t0_tt, t1_tt, tle_path=tle_path)
        except Exception as e:
            cb(f"Error computing trajectories: {str(e)}")
        finally:
            tracking_vis_state.precompute_in_progress = False

    tracking_vis_state.precompute_in_progress = True
    threading.Thread(target=_worker, name="precompute", daemon=True).start()

# Tracks the (selected satellite, trajectory window) the fine-resolution recompute
# last ran for, so we only refine on an actual change. prev_precompute_in_progress
# lets us force a refine after each bulk republish (which overwrites the fine entry).
last_fine_key = None
prev_precompute_in_progress = False

running = True
while running:
    current_time = time.time()
    mouse_pos = pygame.mouse.get_pos()


    if (tracking_vis_state.recompute_triggered and tracking_vis_state.tle_loaded
            and not tracking_vis_state.precompute_in_progress):
        try:
            center_time_obj = datetime.fromisoformat(tracking_vis_state.center_time_str.replace('Z', '+00:00'))
            center_time = center_time_obj.replace(tzinfo=timezone.utc)
            duration_minutes = float(tracking_vis_state.duration_str)
            lat = float(config_state.lat_str or 0)
            lon = float(config_state.lon_str or 0)
            alt_m = float(config_state.alt_str or 0)
            observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
            tracking_vis_state.t0 = ts.utc((center_time - timedelta(minutes=duration_minutes / 2)))
            tracking_vis_state.t1 = ts.utc((center_time + timedelta(minutes=duration_minutes / 2)))

            # Update scroll bar time range labels in state
            tracking_vis_state.scroll_bar_start_label = "-" + str(duration_minutes/2) + " min"
            tracking_vis_state.scroll_bar_end_label = "+" + str(duration_minutes/2) + " min"

            # Reset view state immediately; the heavy compute runs in the background
            # and publishes trajectories when ready (the slider self-corrects to the
            # live time every frame once t0/t1 are set above).
            update_status_callback("Recomputing trajectories...")
            tracking_vis_state.selected_satellite = None
            tracking_vis_state.paused = False
            tracking_vis_state.paused_tt = None
            start_precompute_worker(center_time, duration_minutes, observer)
            last_trajectory_update = current_time
            tracking_vis_state.recompute_triggered = False
        except Exception as e:
            update_status_callback(f"Error recomputing: {str(e)}")
            tracking_vis_state.recompute_triggered = False

    # Precompute trajectories and arc segments every 15 minutes (background).
    if (current_time - last_trajectory_update >= trajectory_interval
            and not tracking_vis_state.precompute_in_progress):
        update_status_callback("Starting trajectory precomputation...")
        lat = float(config_state.lat_str or 0)
        lon = float(config_state.lon_str or 0)
        alt_m = float(config_state.alt_str or 0)
        observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
        # Use current time as center and user-specified duration. Snap the center
        # to the quantization grid so the window matches across relaunches (disk
        # cache reuse); the live "now" marker still tracks true time via the slider.
        duration_minutes_auto = float(tracking_vis_state.duration_str) if tracking_vis_state.duration_str else 120.0
        current_utc = quantized_now()
        tracking_vis_state.t0 = ts.utc(current_utc - timedelta(minutes=duration_minutes_auto/2))
        tracking_vis_state.t1 = ts.utc(current_utc + timedelta(minutes=duration_minutes_auto/2))

        # Update the center time UI field to reflect the new center time
        tracking_vis_state.center_time_str = current_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Update scroll bar time range labels based on the duration being used
        tracking_vis_state.scroll_bar_start_label = "-" + str(duration_minutes_auto/2) + " min"
        tracking_vis_state.scroll_bar_end_label = "+" + str(duration_minutes_auto/2) + " min"

        # Pass current_utc as an explicit center so the worker's internal time
        # window matches the t0/t1 we set above. Runs in the background; the
        # slider self-corrects to live time every frame (see the position block).
        start_precompute_worker(current_utc, duration_minutes_auto, observer)
        last_trajectory_update = current_time

    # Compute satellite positions at 10 Hz
    if current_time - last_update_time >= update_interval:
        # Thread-safety here comes from atomic dict rebinds (see NOTE below),
        # not from a lock: the position_update_lock that used to wrap this
        # block was acquired at exactly one site in the codebase, so it
        # serialized nothing and only suggested false safety. Removed.
        # NOTE: do not clear satellite_positions here. The rendering thread
        # reads it live every frame; clearing it now and only repopulating
        # at update_satellite_positions() below leaves an empty-dict window
        # during the current_tt/slider math, and a render landing in that
        # window paints a fully black frame -> visualization flicker. The
        # update path below replaces the dict in a single atomic rebind, and
        # the no-data branch clears it atomically, so no empty window exists.

        # Calculate current time for trajectory interpolation - ensure it's never None
        if tracking_vis_state.dragging_slider and tracking_vis_state.t0 is not None and tracking_vis_state.t1 is not None:
            # User is actively dragging slider - use slider position
            fraction = (display.slider_rect.x - display.scroll_bar_rect.x) / (display.scroll_bar_rect.width - display.slider_rect.width)
            current_tt = tracking_vis_state.t0.tt + fraction * (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
        elif tracking_vis_state.paused and tracking_vis_state.paused_tt is not None:
            # Simulation is paused - use paused time
            current_tt = tracking_vis_state.paused_tt
        else:
            # Default to current time
            current_tt = ts.now().tt

            # Update slider position to reflect real-time (only if trajectories are available)
            if tracking_vis_state.t0 is not None and tracking_vis_state.t1 is not None and not tracking_vis_state.dragging_slider:
                fraction = (current_tt - tracking_vis_state.t0.tt) / (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
                display.slider_rect.x = display.scroll_bar_rect.x + int(fraction * (display.scroll_bar_rect.width - display.slider_rect.width))

        # Update the current time in the state for use by other components (like PROGRAM tracking)
        tracking_vis_state.current_tt = current_tt

        # Use state-direct mutation approach for satellite position updates
        # Only update positions if trajectories are available
        if tracking_vis_state.tle_loaded and tracking_vis_state.satellites:
            update_satellite_positions(tracking_vis_state, current_tt, elevation_mask_deg=float(config_state.elevation_mask_str or 0))
        else:
            # No trajectory data available: clear positions in a single
            # atomic rebind (never leaves a partially-built/empty window).
            tracking_vis_state.satellite_positions = {}

        # Refresh ADS-B aircraft positions/trajectories (prune stale, rebuild
        # dirty predictions) when the receiver is connected. Cheap when idle.
        if joystick_mode_state.adsb is not None and joystick_mode_state.adsb_connected:
            joystick_mode_state.adsb.update(display, current_tt)

        # Update launch positions if any launch trajectories are loaded
        if hasattr(tracking_vis_state, 'launch_trajectories') and tracking_vis_state.launch_trajectories:
            update_launch_positions(tracking_vis_state, current_tt)

        last_update_time = current_time

    # Fine-resolution trajectory for the actively tracked satellite. Bulk
    # trajectories are sampled coarsely (display only); the selected satellite is
    # what PROGRAM tracking interpolates, so recompute just that one at tight
    # spacing. Retrigger on selection change, window change, or after a bulk
    # republish (which overwrites the fine entry with a coarse one).
    if (not tracking_vis_state.precompute_in_progress) and prev_precompute_in_progress:
        last_fine_key = None  # bulk just republished -> refine the selected sat again
    prev_precompute_in_progress = tracking_vis_state.precompute_in_progress

    sel = tracking_vis_state.selected_satellite
    if sel is None:
        last_fine_key = None
    elif (tracking_vis_state.tle_loaded and tracking_vis_state.t0 is not None
          and tracking_vis_state.t1 is not None and not tracking_vis_state.precompute_in_progress):
        fine_key = (sel, tracking_vis_state.t0.tt, tracking_vis_state.t1.tt)
        if fine_key != last_fine_key:
            try:
                obs_fine = wgs84.latlon(float(config_state.lat_str or 0),
                                        float(config_state.lon_str or 0),
                                        elevation_m=float(config_state.alt_str or 0))
                if compute_fine_selected_trajectory(tracking_vis_state, sel, obs_fine, ts,
                                                    display, tracking_vis_state.t0, tracking_vis_state.t1):
                    last_fine_key = fine_key
            except Exception as e:
                print(f"Debug: fine trajectory recompute failed: {e}")

    # Handle events using modular event system
    # Always process joystick events first, regardless of current mode
    # Check for existing joysticks that were connected before this point
    if not hasattr(joystick_mode_state, '_initialized'):
        # Get current connected joysticks and add them to the state
        for i in range(pygame.joystick.get_count()):
            joy = pygame.joystick.Joystick(i)
            joystick_mode_state.joysticks[joy.get_instance_id()] = joy
            print(f"Existing joystick {joy.get_instance_id()} detected: {joy.get_name()}")
            # Auto-connect to first joystick
            if joystick_mode_state.connected_joystick is None:
                joystick_mode_state.connected_joystick = joy.get_instance_id()
                joystick_mode_state.reset_tare()
        joystick_mode_state._initialized = True

    for event in pygame.event.get():
        # Handle joystick connection events first regardless of current mode
        if event.type in [pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED, pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION]:
            joystick_mode_state.process_joystick_events(event, current_mode, current_tracking_surface, tracking_vis_state, config_state)

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if current_mode == "post_process" and post_process_state is not None:
                # Post-process screen owns the keyboard (text fields + hotkeys);
                # only ESC with nothing focused falls through to quit.
                if event.key == pygame.K_ESCAPE and post_process_state.focus is None:
                    running = False
                else:
                    handle_pp_key(event, display, post_process_state, status_messages)
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif current_mode == "config_options" and config_state.focused_field:
                handle_input(event, config_state)
            elif current_mode == "joystick_loop" and tracking_vis_state.focused_field in ("filter", "filter_above_alt", "filter_below_alt"):
                # Targets-overlay filter boxes (shares tracking-vis text editing).
                from events import handle_tracking_vis_keyboard_events
                handle_tracking_vis_keyboard_events(event, tracking_vis_state)
            elif current_mode == "joystick_loop" and (hasattr(joystick_mode_state, 'config_state') and joystick_mode_state.config_state.focused_field):
                handle_input(event, joystick_mode_state.config_state)
            elif current_mode == "tracking_vis":
                # Handle keyboard input for tracking visualization fields
                from events import handle_tracking_vis_keyboard_events
                handle_tracking_vis_keyboard_events(event, tracking_vis_state)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            # Always check main menu buttons first (allows switching back to menu or between modes)
            menu_result = None
            for btn in display.buttons:
                if btn["rect"].collidepoint(pos):
                    menu_result = btn["mode"]
                    break

            if menu_result:
                from events import handle_main_menu_events
                result = handle_main_menu_events(event, display.buttons, current_mode)
                if result == "exit":
                    running = False
                    break
                elif result and result != current_mode:
                    current_mode = result
                    # Reset fields when switching modes
                    tracking_vis_state.focused_field = None
                    config_state.reset_input_fields()  # Reset config input fields
                    break  # Exit inner loop to process new mode
            elif current_mode is None:
                # Handle main menu button clicks when already in main menu mode
                from events import handle_main_menu_events
                result = handle_main_menu_events(event, display.buttons, current_mode)
                if result == "exit":
                    running = False
                    break
                elif result and result != current_mode:
                    current_mode = result
                    # Reset fields when switching modes
                    tracking_vis_state.focused_field = None
                    config_state.reset_input_fields()  # Reset config input fields
                    break  # Exit inner loop to process new mode

            # Handle mode-specific events
            elif current_mode == "sensor_calib":
                # Handle sensor calibration events using modular handler
                handle_sensor_calib_events(event, pos, display, camera_manager, update_status_callback, config_state)
            elif current_mode == "config_options":
                # Handle save button click
                if display.save_button.collidepoint(pos) and str(pos) and len(str(pos).strip()) > 2:
                    display.button_states["save"]["clicked"] = True
                    config_state.save_to_file()
                    status_messages.append("Configuration saved")
                    break

                # Handle load button click
                elif display.load_button.collidepoint(pos) and str(pos) and len(str(pos).strip()) > 2:
                    display.button_states["load"]["clicked"] = True
                    try:
                        config_state.load_from_file()
                        status_messages.append("Configuration loaded")

                        # Update camera manager gain and exposure for connected cameras after loading config
                        if camera_manager.cameras[0].connected:
                            gain1 = int(float(config_state.get_camera_gain("camera1")))
                            exposure1 = int(float(config_state.get_camera_exposure("camera1")))
                            camera_manager.cameras[0].gain = gain1
                            camera_manager.cameras[0].exposure = exposure1
                            camera_manager.set_camera_gain(0, gain1)
                            camera_manager.set_camera_exposure(0, exposure1)
                        if camera_manager.cameras[1].connected:
                            gain2 = int(float(config_state.get_camera_gain("camera2")))
                            exposure2 = int(float(config_state.get_camera_exposure("camera2")))
                            camera_manager.cameras[1].gain = gain2
                            camera_manager.cameras[1].exposure = exposure2
                            camera_manager.set_camera_gain(1, gain2)
                            camera_manager.set_camera_exposure(1, exposure2)
                    except Exception as e:
                        status_messages.append(f"Error loading config: {e}")
                    break

                # Handle input field focus
                for field_name, input_rect in display.input_rects.items():
                    if input_rect.collidepoint(pos):
                        config_state.focused_field = field_name
                        config_state.cursor_pos[field_name] = 0
                        config_state.selection_start[field_name] = None
                        break

            # Hardware Sim screen: toggle / steppers / save-load
            elif current_mode == "hw_sim":
                handle_hw_sim_click(pos, display, config_state, hardware_sim, status_messages)
            elif current_mode == "post_process":
                if post_process_state is None:
                    post_process_state = PostProcessState()
                handle_pp_click(pos, display, post_process_state, status_messages)
            elif current_mode == "alignment":
                # Camera-panel controls (right half) take precedence, but only when the
                # camera view is shown; fall through to the alignment controls otherwise.
                cam_view = getattr(alignment_state, 'view_mode', 'camera') == 'camera'
                if not (cam_view and handle_alignment_camera_events(event, display, config_state, update_status_callback)):
                    handle_alignment_click(pos, display, config_state, alignment_state,
                                           joystick_mode_state, tracking_vis_state, ts, status_messages)

            # Handle tracking_vis events using state-based approach
            elif current_mode == "tracking_vis":
                try:
                    # Handle events using state-based handler
                    handle_tracking_vis_mouse_events_state(tracking_vis_state, event, pos, display, display.button_states)

                except Exception as e:
                    print(f"DEBUG: Error in tracking_vis mouse events: {e}")
                    print(f"DEBUG: Current mode: {current_mode}, Mouse pos: {pos}, Event type: {event.type}")
                    import traceback
                    traceback.print_exc()

            # Handle joystick mode events
            elif current_mode == "joystick_loop":
                # Camera/view controls on the half-height feeds take precedence; if the
                # click was not consumed by a camera control, fall through to the
                # normal joystick-mode handler (satellite selection, launch, etc.).
                if not handle_joystick_camera_control_events(event, display, config_state, update_status_callback):
                    handle_joystick_mode_mouse_events(event, joystick_mode_state, display, tracking_vis_state, config_state, current_tracking_surface)


        elif event.type == pygame.MOUSEMOTION:
            if current_mode is None:
                # Handle main menu hover states
                for btn in display.buttons:
                    display.button_states[btn["mode"]]["hover"] = btn["rect"].collidepoint(event.pos)
            elif current_mode == "post_process" and post_process_state is not None:
                handle_pp_motion(event.pos, display, post_process_state, event.buttons)
            elif current_mode == "config_options":
                display.button_states["save"]["hover"] = display.save_button.collidepoint(event.pos)
                display.button_states["load"]["hover"] = display.load_button.collidepoint(event.pos)
            elif current_mode == "sensor_calib":
                # Camera slider hover states handled by modular camera code.
                # event.pos, NOT the stale `pos` bound by a previous
                # MOUSEBUTTONDOWN iteration (NameError before any click,
                # silently wrong coordinates after one).
                handle_sensor_calib_events(event, event.pos, display, camera_manager, update_status_callback)
            elif current_mode == "tracking_vis":
                display.button_states["clear_filters"]["hover"] = display.clear_filters_button.collidepoint(event.pos)
                display.button_states["recompute"]["hover"] = display.recompute_button.collidepoint(event.pos)
                display.button_states["reset"]["hover"] = display.reset_button.collidepoint(event.pos)
                display.button_states["pause"]["hover"] = display.pause_button.collidepoint(event.pos)
                display.button_states["play"]["hover"] = display.play_button.collidepoint(event.pos)
                display.button_states["launch"]["hover"] = display.launch_button.collidepoint(event.pos)
                if tracking_vis_state.dragging_slider:
                    display.slider_rect.x = max(display.scroll_bar_rect.x, min(event.pos[0] - display.slider_rect.width // 2, display.scroll_bar_rect.x + display.scroll_bar_rect.width - display.slider_rect.width))
                for sat, (px, py, _, _) in tracking_vis_state.satellite_positions.items():
                    if math.hypot(px - event.pos[0], py - event.pos[1]) < 10:
                        tracking_vis_state.hovered_satellite = sat
                        break
                # Starfield hover: star_screen_positions are in surface coords; the
                # surface is blitted at (display.sub_x, display.sub_y).
                tracking_vis_state.hovered_star = None
                for sx, sy, hip in (tracking_vis_state.star_screen_positions or []):
                    if math.hypot(sx + display.sub_x - event.pos[0],
                                  sy + display.sub_y - event.pos[1]) < 8:
                        tracking_vis_state.hovered_star = hip
                        break
            elif current_mode == "alignment":
                # Handle camera control slider dragging on the right-half camera panel
                if getattr(alignment_state, 'view_mode', 'camera') == 'camera':
                    handle_alignment_camera_events(event, display, config_state, update_status_callback)
            elif current_mode == "joystick_loop":
                # Handle camera control slider dragging on the half-height feeds
                handle_joystick_camera_control_events(event, display, config_state, update_status_callback)
                # Handle PID slider dragging in joystick mode
                if event.buttons[0]:  # Left mouse button is pressed
                    current_pos = event.pos

                    # Check PID sliders for dragging
                    if (hasattr(display, 'joystick_pid_slider_rects') and
                        joystick_mode_state is not None and
                        hasattr(joystick_mode_state, 'config_state') and
                        joystick_mode_state.config_state is not None):

                        # PID slider ranges and logarithmic scaling (5 orders of magnitude)
                        PID_MAX_VALUE = 2.0
                        PID_MIN_VALUE = 2.0 / 100000.0
                        LOG_SCALE_FACTOR = 5.0  # 5 orders of magnitude

                        for field, slider_rect in display.joystick_pid_slider_rects.items():
                            if slider_rect.collidepoint(current_pos):
                                SLIDER_WIDTH = 80
                                relative_x = min(max(current_pos[0] - slider_rect.x, 0), SLIDER_WIDTH)
                                slider_pos = relative_x / SLIDER_WIDTH  # 0-1 position

                                # Calculate new value using 5 orders of magnitude scaling: value = min_val * 10^(pos * factor)
                                new_value = PID_MIN_VALUE * math.pow(10, slider_pos * LOG_SCALE_FACTOR)
                                new_value = max(PID_MIN_VALUE, min(PID_MAX_VALUE, new_value))

                                # Update config value
                                if field == 'pid_azm_p_gain':
                                    joystick_mode_state.config_state.pid_azm_p_gain = round(new_value, 5)
                                elif field == 'pid_azm_i_gain':
                                    joystick_mode_state.config_state.pid_azm_i_gain = round(new_value, 5)
                                elif field == 'pid_azm_d_gain':
                                    joystick_mode_state.config_state.pid_azm_d_gain = round(new_value, 5)
                                elif field == 'pid_alt_p_gain':
                                    joystick_mode_state.config_state.pid_alt_p_gain = round(new_value, 5)
                                elif field == 'pid_alt_i_gain':
                                    joystick_mode_state.config_state.pid_alt_i_gain = round(new_value, 5)
                                elif field == 'pid_alt_d_gain':
                                    joystick_mode_state.config_state.pid_alt_d_gain = round(new_value, 5)

                                # Update PID controllers if they exist
                                if joystick_mode_state.azm_pid:
                                    joystick_mode_state.azm_pid.update_gains(
                                        joystick_mode_state.config_state.pid_azm_p_gain,
                                        joystick_mode_state.config_state.pid_azm_i_gain,
                                        joystick_mode_state.config_state.pid_azm_d_gain
                                    )
                                if joystick_mode_state.alt_pid:
                                    joystick_mode_state.alt_pid.update_gains(
                                        joystick_mode_state.config_state.pid_alt_p_gain,
                                        joystick_mode_state.config_state.pid_alt_i_gain,
                                        joystick_mode_state.config_state.pid_alt_d_gain
                                    )
                                break  # Only handle one slider at a time

                    # Lead-time slider drag (reuse the click handler so the lead
                    # range logic stays in one place)
                    if joystick_mode_state is not None:
                        handle_lead_slider_mouse_events(joystick_mode_state, display, current_pos)
                        # ADS-B fit-points slider drag (same click-handler reuse)
                        handle_adsb_fit_slider_mouse_events(joystick_mode_state, display, current_pos)


        # (A second, unreachable MOUSEMOTION branch used to sit here -- the
        # dispatch above already consumes the event type. Removed.)
        elif event.type == pygame.MOUSEBUTTONUP:
            # Reset all button clicked states - essential for UI feedback
            for btn in display.buttons:
                display.button_states[btn["mode"]]["clicked"] = False
            if current_mode is None:
                pass
            elif current_mode == "post_process" and post_process_state is not None:
                handle_pp_mouseup(event.pos, display, post_process_state)
            elif current_mode == "config_options":
                display.button_states["save"]["clicked"] = False
                display.button_states["load"]["clicked"] = False
            elif current_mode == "sensor_calib":
                # Handle sensor calibration events (event.pos, not the stale
                # `pos` bound by a previous MOUSEBUTTONDOWN iteration)
                handle_sensor_calib_events(event, event.pos, display, camera_manager, update_status_callback)
            elif current_mode == "tracking_vis":
                display.button_states["clear_filters"]["clicked"] = False
                display.button_states["reset"]["clicked"] = False
                display.button_states["pause"]["clicked"] = False
                display.button_states["play"]["clicked"] = False
                display.button_states["launch"]["clicked"] = False
                # Reset dragging state
                if tracking_vis_state.dragging_slider and tracking_vis_state.paused:
                    fraction = (display.slider_rect.x - display.scroll_bar_rect.x) / (display.scroll_bar_rect.width - display.slider_rect.width)
                    tracking_vis_state.paused_tt = tracking_vis_state.t0.tt + fraction * (tracking_vis_state.t1.tt - tracking_vis_state.t0.tt)
                tracking_vis_state.dragging_slider = False
                tracking_vis_state.scroll_start = None

        elif event.type == pygame.MOUSEWHEEL:
            wheel_pos = pygame.mouse.get_pos()
            # Scroll the left-bar status log when the cursor is over it
            if display.get_status_panel_rect().collidepoint(wheel_pos):
                display.status_scroll_offset = max(
                    0, min(display.status_scroll_offset + event.y, display.status_max_scroll))
            # Scroll the run list in the post-process library
            elif current_mode == "post_process" and post_process_state is not None:
                handle_pp_wheel(wheel_pos, display, post_process_state, event.y)
            # Scroll the satellite passes table when hovering it in tracking-vis mode
            elif current_mode == "tracking_vis":
                from tracking_visuals import get_pass_table_rect
                if get_pass_table_rect(display).collidepoint(wheel_pos):
                    max_scroll = getattr(tracking_vis_state, 'pass_table_max_scroll', 0)
                    tracking_vis_state.pass_table_scroll_offset = max(
                        0, min(tracking_vis_state.pass_table_scroll_offset - event.y, max_scroll))
            # Scroll the joystick-mode Targets overlay pass table when hovering it
            elif current_mode == "joystick_loop":
                jl_box = getattr(joystick_mode_state, 'jl_pass_table_rect', None)
                if jl_box is not None and jl_box.collidepoint(wheel_pos):
                    max_scroll = getattr(tracking_vis_state, 'pass_table_max_scroll', 0)
                    tracking_vis_state.pass_table_scroll_offset = max(
                        0, min(tracking_vis_state.pass_table_scroll_offset - event.y, max_scroll))

    # Render continuously
    # (Global current_tracking_surface already declared at top)

    display.menu_screen.fill(display.COLOR_BACKGROUND_DARK, (0, 0, display.MENU_WIDTH, display.total_height))  # Dark theme menu background
    if display.bg_image_menu:
        # Update background rotation
        display.update_background_rotation()
        # Get the appropriate image
        image_tool_use = display.get_rotated_image(display.image_rect)
        if display.image_rect:
            display.menu_screen.blit(image_tool_use, display.image_rect.topleft)

    for btn in display.buttons:
        draw_menu_button(display, btn)
    # Render status messages each frame (scrollable log confined to the left bar).
    # Keep a bounded history so older messages remain available for scrollback.
    status_messages = status_messages[-500:]
    display.render_status_log(status_messages)

    cx = display.sub_x + display.sub_width // 2
    cy = display.sub_y + display.sub_height // 2
    viz_surface = None
    if current_mode == "config_options":
        draw_config_options(display, config_state)
    elif current_mode == "hw_sim":
        draw_hw_sim_options(display, config_state, hardware_sim)
    elif current_mode == "alignment":
        draw_alignment_options(display, config_state, alignment_state, joystick_mode_state)
    elif current_mode == "tracking_vis" and tracking_vis_state.tle_loaded:
        # Only clear if we don't have a thread surface (initial loading)
        # This prevents washing away the displayed surface while thread is rendering
        if not hasattr(tracking_viz_thread, 'latest_surface') or tracking_viz_thread.latest_surface is None:
            display.menu_screen.fill((0, 0, 0), (display.sub_x, display.sub_y, display.sub_width, display.sub_height))

        # Calculate center coordinates for satellite rendering calculations
        cx = display.sub_x + display.sub_width // 2
        cy = display.sub_y + display.sub_height // 2

        # Get completed surface from thread and blit it once when available
        viz_surface = tracking_viz_thread.get_latest_surface()
        if viz_surface:
            display.menu_screen.blit(viz_surface, (display.sub_x, display.sub_y))
            # Store for capture functionality
            current_tracking_surface = viz_surface

        # Draw UI elements on top of the polar plot
        draw_filters(display, tracking_vis_state)
        draw_legend(display)
        draw_details(display, tracking_vis_state)
        draw_camera_fov_details(display, tracking_vis_state, 290)  # Start below satellite details (20+250+20)
        draw_time_display(display)
        draw_satellite_count(display, tracking_vis_state)

        # Apply live name/altitude filtering + multi-column sort each frame so the
        # table reacts immediately to filter edits and column-header clicks.
        from tracking_visuals import filter_and_sort_pass_table
        tracking_vis_state.satellite_pass_table = filter_and_sort_pass_table(tracking_vis_state)

        # Draw satellite pass table
        draw_satellite_pass_table(display, tracking_vis_state)
        draw_button_with_objects(display, "clear_filters")
        draw_button_with_objects(display, "pause")
        draw_button_with_objects(display, "play")
        draw_scroll_bar(display, tracking_vis_state)
        draw_scroll_time_display(display, current_tt, ts)
        # Draw Recompute Button
        draw_button_with_objects(display, "recompute")
        # Draw Reset Button
        draw_button_with_objects(display, "reset")
        # Draw Launch Button (only if a launch is selected)
        if hasattr(tracking_vis_state, 'selected_launch') and tracking_vis_state.selected_launch:
            # Display launch elapsed time above button
            if hasattr(tracking_vis_state, 'launch_start_time') and tracking_vis_state.launch_start_time and tracking_vis_state.launch_launched:
                # Convert TT difference to seconds for display
                elapsed_seconds = (ts.now().tt - tracking_vis_state.launch_start_time) * 86400  # TT is in days, convert to seconds
                elapsed_text = f"{elapsed_seconds:.1f}s"
                elapsed_surface = display.small_font.render(elapsed_text, True, (0, 255, 0))  # Green text
                display.menu_screen.blit(elapsed_surface, (display.launch_button.x + display.launch_button.width // 2 - elapsed_surface.get_width() // 2, display.launch_button.y - 20))
            else:
                # Show T-0 if launch selected but not launched
                elapsed_text = "T-0"
                elapsed_surface = display.small_font.render(elapsed_text, True, (255, 255, 255))  # White text
                display.menu_screen.blit(elapsed_surface, (display.launch_button.x + display.launch_button.width // 2 - elapsed_surface.get_width() // 2, display.launch_button.y - 20))
            draw_button_with_objects(display, "launch", tracking_vis_state.launch_launched)
        # Draw Center Time Input
        pygame.draw.rect(display.menu_screen, (255, 255, 255), display.center_time_rect)
        if tracking_vis_state.focused_field == 'center_time':
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.center_time_rect, 2)
        text_surface = display.small_font.render(tracking_vis_state.center_time_str, True, (0, 0, 0))
        display.menu_screen.blit(text_surface, (display.center_time_rect.x + 5, display.center_time_rect.y + 5))
        if tracking_vis_state.focused_field == 'center_time':
            text_width, _ = display.small_font.size(tracking_vis_state.center_time_str[:tracking_vis_state.cursor_pos['center_time']])
            pygame.draw.line(display.menu_screen, (0, 0, 255), (display.center_time_rect.x + 5 + text_width, display.center_time_rect.y + 5), (display.center_time_rect.x + 5 + text_width, display.center_time_rect.y + 25), 2)
        # Label for Center Time
        label = display.small_font.render("Center Time:", True, (255, 255, 255))
        display.menu_screen.blit(label, (display.center_time_rect.x, display.center_time_rect.y - 15))
        # Draw Duration Input
        pygame.draw.rect(display.menu_screen, (255, 255, 255), display.duration_rect)
        if tracking_vis_state.focused_field == 'duration':
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.duration_rect, 2)
        text_surface = display.small_font.render(tracking_vis_state.duration_str, True, (0, 0, 0))
        display.menu_screen.blit(text_surface, (display.duration_rect.x + 5, display.duration_rect.y + 5))
        if tracking_vis_state.focused_field == 'duration':
            text_width, _ = display.small_font.size(tracking_vis_state.duration_str[:tracking_vis_state.cursor_pos['duration']])
            pygame.draw.line(display.menu_screen, (0, 0, 255), (display.duration_rect.x + 5 + text_width, display.duration_rect.y + 5), (display.duration_rect.x + 5 + text_width, display.duration_rect.y + 25), 2)
        # Label for Duration
        label = display.small_font.render("Duration (min.):", True, (255, 255, 255))
        display.menu_screen.blit(label, (display.duration_rect.x, display.duration_rect.y - 15))
    elif current_mode == "sensor_calib":
        # Modular camera display rendering - replaced ~200 lines of inline camera rendering code
        from camera_manager import render_sensor_calibration
        render_sensor_calibration(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                               camera_manager.get_camera(0).connected, camera_manager.get_camera(1).connected,
                               camera1_name, camera2_name)

        # Modular camera slider controls - positioned within camera display area
        from camera_manager import render_camera_sliders
        render_camera_sliders(display.menu_screen, display.tiny_font, display.sub_x, display.sub_y, display.sub_width, display.sub_height)

        # Modular alignment rotation sliders - positioned at bottom-center of each camera image
        from camera_manager import render_alignment_rotation_sliders
        render_alignment_rotation_sliders(display.menu_screen, display.tiny_font, display.sub_x, display.sub_y, display.sub_width, display.sub_height)

        # Modular ROI controls for both cameras - replaced ~100+ lines of inline ROI control code
        from camera_manager import render_camera_roi_controls
        render_camera_roi_controls(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height)

        # Modular camera interface completion - replaced ~40 lines of inline combined view and status code
        from camera_manager import render_combined_view_controls
        # Initialize camera control rects with proper dimensions
        camera_manager._initialize_control_rects(display.menu_screen, display.total_width, display.total_height)
        render_combined_view_controls(display.menu_screen, display.sub_x, display.sub_y, display.sub_width, display.sub_height,
                                        display.small_font, display.tiny_font)
    elif current_mode == "joystick_loop":
        # Clear the main area
        display.menu_screen.fill((30, 30, 30), (display.sub_x, display.sub_y,
                                        display.sub_width, display.sub_height))

        # Blit pre-rendered polar plot from joystick visual thread
        joystick_surface = joystick_viz_thread.get_latest_surface()
        if joystick_surface:
            display.menu_screen.blit(joystick_surface, (display.sub_x + display.sub_width // 2, display.sub_y))
            # Update tracking surface for capture functionality in joystick mode
            current_tracking_surface = joystick_surface
        else:
            # If no joystick surface available, keep existing tracking surface
            # This handles timing when switching modes or if thread isn't ready yet
            pass

        # Draw Launch Button in joystick mode (only if a launch is selected)
        # Position at lower left of the upper right quadrant
        if hasattr(tracking_vis_state, 'selected_launch') and tracking_vis_state.selected_launch:
            # Create joystick launch button rectangle (lower left of upper right quadrant)
            joystick_launch_button = pygame.Rect(
                display.sub_x + display.sub_width // 2 + 10,  # Left edge of quadrant + margin
                display.sub_y + display.sub_height // 2 - 45,  # Bottom of quadrant - button height - margin
                70, 30  # Same size as main launch button
            )

            # Display launch elapsed time above button
            if hasattr(tracking_vis_state, 'launch_start_time') and tracking_vis_state.launch_start_time and tracking_vis_state.launch_launched:
                # Convert TT difference to seconds for display
                elapsed_seconds = (ts.now().tt - tracking_vis_state.launch_start_time) * 86400  # TT is in days, convert to seconds
                elapsed_text = f"{elapsed_seconds:.1f}s"
                elapsed_surface = display.small_font.render(elapsed_text, True, (0, 255, 0))  # Green text
                display.menu_screen.blit(elapsed_surface, (joystick_launch_button.x + joystick_launch_button.width // 2 - elapsed_surface.get_width() // 2, joystick_launch_button.y - 20))
            else:
                # Show T-0 if launch selected but not launched
                elapsed_text = "T-0"
                elapsed_surface = display.small_font.render(elapsed_text, True, (255, 255, 255))  # White text
                display.menu_screen.blit(elapsed_surface, (joystick_launch_button.x + joystick_launch_button.width // 2 - elapsed_surface.get_width() // 2, joystick_launch_button.y - 20))

            # Store the joystick launch button for event handling
            display.joystick_launch_button = joystick_launch_button
            draw_button_with_objects(display, "launch", tracking_vis_state.launch_launched, joystick_launch_button)

        # Render connection controls and joystick status (not threaded)
        render_connection_controls(display, joystick_mode_state)
        render_adsb_connection_controls(display, joystick_mode_state)
        render_joystick_status(display, joystick_mode_state)
        render_position_display(display, joystick_mode_state)

        # Render capture controls (progress indicator, capture button)
        render_capture_controls(display, joystick_mode_state)

        # Render PID diagnostics
        render_pid_diagnostics(display, joystick_mode_state)

        # Render PID gain sliders (for real-time tuning)
        render_pid_gain_sliders(display, joystick_mode_state)

        # Render bias control grid
        render_bias_control_grid(display, joystick_mode_state)

        # Render feed-forward toggle buttons
        render_feed_forward_toggle_buttons(display, joystick_mode_state)

        # Render plate-solve pane (toggle + apply-alignment)
        render_plate_solve_panel(display, joystick_mode_state)

        # Render the navball + PID-tuning strip charts in the center column
        render_navball(display, joystick_mode_state)
        render_tracking_strip_charts(display, joystick_mode_state)

        # Render camera feeds
        render_camera_feeds(display, joystick_mode_state)

        # Skyplot "Targets" overlay (filters + sortable passes/launches table +
        # show/hide toggles) over the upper-right quadrant skyplot. Drawn last so
        # the panel overlays the skyplot.
        render_joystick_target_panel(display, joystick_mode_state, tracking_vis_state, config_state)

        # Mount control and position polling run on MountControlThread
        # (see mount_control.py), decoupled from this render loop. The thread
        # populates joystick_mode_state.current_azm/current_alt and the display
        # strings, which the rendering code above reads.

    elif current_mode == "post_process":
        if post_process_state is None:
            post_process_state = PostProcessState()
        draw_post_process(display, post_process_state)
    elif current_mode == "author_info":
        sub_rect = (display.sub_x, display.sub_y, display.sub_width, display.sub_height)
        if display.author_bg:
            scaled_bg = pygame.transform.scale(display.author_bg, (display.sub_width, display.sub_height))
            display.menu_screen.blit(scaled_bg, (display.sub_x, display.sub_y))
        else:
            display.menu_screen.fill((0, 0, 0), sub_rect)
        text1 = display.large_font.render("Starlink-1060", True, (255, 255, 255))
        display.menu_screen.blit(text1, (display.sub_x + 10, display.sub_y + 10))
        contact_text = "Jonathan Nikkel - @NikkelJonathan"
        text2 = display.large_font.render(contact_text, True, (255, 255, 255))
        display.menu_screen.blit(text2, (display.sub_x + 10, display.sub_y + 50))

    pygame.display.flip()
    clock.tick(display.FPS_TARGET)  # Limit to FPS_TARGET FPS for better responsiveness

pygame.quit()

# Stop the mount control thread first so the mount is commanded to a stop
# before we tear anything else down.
print("Shutting down mount control thread...")
if mount_control_thread:
    mount_control_thread.stop()
    mount_control_thread.join(timeout=2.0)

# Stop the ADS-B receiver (SDR source thread) if it was started.
if joystick_mode_state is not None and getattr(joystick_mode_state, 'adsb', None) is not None:
    print("Shutting down ADS-B receiver...")
    joystick_mode_state.adsb.disconnect()

# Stop post-process decode/export worker threads if the screen was opened
if post_process_state is not None:
    print("Shutting down post-process workers...")
    post_process_state.shutdown()

# Clean up rendering threads
print("Shutting down rendering threads...")
if tracking_viz_thread:
    tracking_viz_thread.stop()
if joystick_viz_thread:
    joystick_viz_thread.stop()
print("Rendering threads stopped. All cleanup complete.")

print("Program exit complete.")
