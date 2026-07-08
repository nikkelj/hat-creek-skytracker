import os
import json
import pygame

# Bumped when the config file format changes incompatibly; saved into
# config.json so an older app can warn instead of silently part-loading.
CONFIG_VERSION = 1

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

        # Mount alignment configuration
        self.alignment_azimuth_str = "0.0"
        self.alignment_elevation_str = "0.0"

        # Skyplot satellite display toggles (hide/show satellites + their labels;
        # surfaced as toggle buttons in the joystick-mode skyplot overlay).
        self.satellites_enabled = True          # draw satellite markers on the skyplot
        self.satellite_labels_enabled = True    # draw satellite name labels

        # Star catalogue / starfield configuration
        self.starfield_enabled = True          # draw catalogue stars on the skyplot
        self.max_rendered_star_count = 2000    # brightest-N cap (skyplot + sim images)
        self.star_limiting_magnitude = 6.5     # faintest star ever considered

        # Plate solver (tetra3) configuration
        self.plate_solve_enabled = False       # background plate-solve worker on/off
        self.plate_solve_camera_index = 0      # which camera the solver reads (0=wide/finder)

        # ADS-B (RTL-SDR aircraft tracking) configuration
        self.adsb_source_mode = "rtlsdr"       # "rtlsdr" (native SDR) | "dump1090" | "sim"
        self.adsb_device_index = 0             # RTL-SDR device index when multiple dongles
        self.adsb_gain = "auto"                # RTL-SDR tuner gain ("auto" or dB)
        self.adsb_dump1090_host = "127.0.0.1"  # SBS feed host (dump1090 mode)
        self.adsb_dump1090_port = 30003        # SBS/BaseStation TCP port
        self.adsb_fit_points = 5               # # of recent fixes fit for linear prediction (UI slider)
        self.adsb_predict_horizon_sec = 60.0   # how far ahead to propagate the trajectory
        self.adsb_predict_step_sec = 2.0       # spacing of propagated trajectory points
        self.adsb_history_seconds = 120.0      # observed-fix tail kept per aircraft
        self.adsb_stale_timeout_sec = 30.0     # drop aircraft not heard from in this long

        # Automated alignment: when a sample point won't plate-solve, the runner spirals
        # outward over this many grid cells (~0.5x FOV each) searching for a solve before
        # giving up / pausing for a manual jog.
        self.alignment_grid_search_cells = 100

        # Alignment slew settle: the plate-solve pairs the encoder reading AT SOLVE TIME
        # with the solved sky position, so the sample is correct no matter how far from the
        # grid point we land -- the mount only needs to be *near* a star-rich field and to
        # have stopped moving enough that stars don't streak. A loose tolerance + few settle
        # cycles is therefore the single biggest dusk-time saver (a tight 0.02 deg asymptotic
        # settle wastes seconds per point for zero accuracy gain). Degrees / cycles.
        self.alignment_settle_tol_deg = 0.3
        self.alignment_settle_cycles = 2

        # Adaptive early-stop: once the running fit's post-fit sky RMS (arcmin) drops to/below
        # this for two consecutive checks (and enough points are covered), stop sampling the
        # remaining fit grid early. Disabled by default (0.0 -> always sample the full grid):
        # with few, azimuth-correlated points the per-term accuracy (esp. NPAE/CA) softens, so
        # this is opt-in. Set e.g. 0.75 to trade a little term accuracy for a faster run.
        self.alignment_target_rms_arcmin = 0.0

        # Pointing model (7-term alt-az TPOINT). Coefficients in degrees; applied as a
        # Python pre-step at target-setting time when pointing_model_enabled is True.
        self.pointing_model_enabled = False
        self.pointing_model_terms = {"IA": 0.0, "IE": 0.0, "AN": 0.0, "AW": 0.0,
                                     "NPAE": 0.0, "CA": 0.0, "TF": 0.0}

        # Equatorial (Eq-mode) 7-term pointing model: residual mount errors after polar
        # alignment (eq_pointing_model.py). Applied in hour-angle/dec space in the Eq branch.
        self.eq_pointing_model_enabled = False
        self.eq_pointing_model_terms = {"IH": 0.0, "ID": 0.0, "NP": 0.0, "CH": 0.0,
                                        "ME": 0.0, "MA": 0.0, "TF": 0.0}

        # Position offset configuration
        self.azm_offset_str = "0.0"
        self.alt_offset_str = "0.0"

        # PID gain configuration
        self.pid_azm_p_gain = 1.0
        self.pid_azm_i_gain = 0.0
        self.pid_azm_d_gain = 0.0
        self.pid_alt_p_gain = 1.0
        self.pid_alt_i_gain = 0.0
        self.pid_alt_d_gain = 0.0

        # Seconds to lead the trajectory target by, compensating read/command
        # transport latency. 0 disables leading. Tune on hardware.
        self.pid_lead_time_sec = 0.0

        # Feed-forward configuration
        self.feed_forward_azm_enabled = False
        self.feed_forward_alt_enabled = False

        # Bias control configuration
        self.bias_azm_deg = 0.0  # Bias in azimuth direction (degrees)
        self.bias_alt_deg = 0.0  # Bias in elevation direction (degrees)
        self.bias_control_mode = "coarse"  # "coarse" or "fine" movement mode

        # Hotspot (closed-loop optical) tracker configuration
        self.hotspot_camera_index = 0       # which camera feeds the loop (0 = finder/wide)
        self.hotspot_snr_threshold = 5.0    # min (peak-bg)/noise to accept a detection
        self.hotspot_gate_radius = 120      # tracking-gate half-size in pixels once locked
        self.hotspot_coast_time_sec = 1.0   # coast this long on loss before falling back
        # HOTSPOT is a CENTERING loop, not a slew: cap its commanded rate so a
        # large pixel error can't lunge the mount and yank the target out of
        # its own tracking gate (the acquire->yank->lose->fallback stair-step).
        self.hotspot_max_rate_dps = 2.0
        self.hotspot_x_sign = 1.0           # per-axis sign calibration (set on hardware)
        self.hotspot_y_sign = -1.0

        # HANDOFF mode: PROGRAM track runs the hotspot detector in parallel and
        # hands the loop to HOTSPOT after this many consecutive solid detections.
        self.handoff_min_frames = 5

        # Hardware simulator configuration (mount + camera sim). enabled=False
        # leaves all real-hardware behavior unchanged.
        self.sim_config = {
            "enabled": False,
            "cam_width": 960,            # sim sensor resolution (px)
            "cam_height": 720,
            "mount_misalignment_az_deg": 0.0,   # encoder bias vs true sky
            "mount_misalignment_el_deg": 0.0,
            "mount_encoder_noise_deg": 0.0,     # per-read uniform noise bound
            "mount_rate_noise_dps": 0.0,        # slew rate jitter (1-sigma)
            "mount_backlash_deg": 0.0,          # gear lost-motion on reversal (AVX ~0.006 = 22")
            "mount_pe_amplitude_deg": 0.0,      # periodic error, zero-to-peak (AVX ~0.003 = 11")
            "mount_pe_period_sec": 600.0,       # worm period (AVX ~10 min)
            # Injected 7-term alt-az pointing model: the repeatable mount/optical errors the
            # alignment routine must recover (encoder bias above is only an IA/IE-like offset).
            # Empty/zero = no distortion. Used by the sim render so the rendered boresight lands
            # where predict_observed() says -- letting an alignment run recover all seven terms.
            "mount_pointing_model": {},         # e.g. {"IA":0.4,"IE":-0.2,"AN":0.05,...} (degrees)
            "sim_refraction": False,            # add atmospheric refraction to the apparent elevation
            "sim_refraction_pressure_mbar": 1010.0,
            "sim_refraction_temperature_c": 10.0,
            # Equatorial-mode polar-axis misalignment (degrees) the polar-alignment routine
            # must measure: the sim's TRUE polar axis = (north + az_err, latitude + alt_err).
            "sim_pole_az_err_deg": 0.0,
            "sim_pole_alt_err_deg": 0.0,
            # Injected equatorial residual pointing model (IH/ID/NP/CH/ME/MA/TF, degrees) the
            # Eq-mode alignment must recover -- applied in HA/Dec before the pole geometry.
            "mount_eq_pointing_model": {},
            "cam2_offset_rotation_deg": 0.0,    # inter-camera misalignment
            "cam2_offset_x_px": 0.0,
            "cam2_offset_y_px": 0.0,
            "star_density": 300,                # stars sprinkled over the sky (legacy random field)
            "sim_use_real_stars": False,        # render the real star catalogue instead of the random field
            "sim_star_limit_mag": 10.0,         # faintest catalogue star rendered into sim images (deep for narrow FOV)
            "sim_use_deep_catalog": True,       # use Tycho (deep) when present; False forces Hipparcos
            "sim_debug_target": False,          # print per-frame target-vs-boresight diagnostics (cam1, 1 Hz)
            "background_level": 6.0,
            "read_noise": 2.0,
            "target_brightness": 200.0,
            "seed": 1234,
        }

        # Mount mode configuration
        self.mount_mode = "AltAz"  # "AltAz" or "Eq" - mount coordinate system

        # Continuous-rate tracking: issue the fine 24-bit variable-rate
        # (MC_SET_POS/NEG_GUIDERATE) command for smooth tracking instead of the
        # coarse 10-step MC_MOVE, falling back to MC_MOVE above guide_rate_max_dps
        # (e.g. the near-zenith keyhole). Off by default until verified on hardware
        # with bench_guiderate.py.
        self.continuous_rate_tracking = False
        self.guide_rate_max_dps = 5.0  # max rate sent via the guide-rate command

        # Experimental: run the control loop in Rust (skytracker_core) instead of
        # the Python MountControlThread. Read at startup in main.py; toggled from
        # the Hardware Simulator screen. Takes effect on restart.
        self.use_rust_core_loop = False
        self.rust_core_loop_hz = 15

        # Hardware safety limits configuration (degrees)
        self.azm_limit_min_str = "-180.0"
        self.azm_limit_max_str = "180.0"
        self.alt_limit_min_str = "-100.0"
        self.alt_limit_max_str = "100.0"

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
                "exposure": 10000.0,  # microseconds
                "gamma": 0.1,  # gamma correction value
                "gamma_enabled": False  # gamma correction toggle
            },
            "camera2": {
                "pixel_size": 3.75,  # μm
                "array_size_diagonal": 11.0,  # mm
                "focal_length": 25.0,  # mm
                "alignment_rotation": 0.0,  # degrees
                "gain": 1.0,  # unitless
                "exposure": 10000.0,  # microseconds
                "gamma": 0.1,  # gamma correction value
                "gamma_enabled": False  # gamma correction toggle
            }
        }

        # Input field state for config mode
        self.focused_field = None
        self.cursor_pos = {
            "lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0,
            "alignment_azimuth": 0, "alignment_elevation": 0,
            "azm_offset": 0, "alt_offset": 0,
            "azm_limit_min": 0, "azm_limit_max": 0,
            "alt_limit_min": 0, "alt_limit_max": 0,
            "pid_azm_p_gain": 0, "pid_azm_i_gain": 0, "pid_azm_d_gain": 0,
            "pid_alt_p_gain": 0, "pid_alt_i_gain": 0, "pid_alt_d_gain": 0,
            "camera1_pixel_size": 0, "camera1_array_size_diagonal": 0, "camera1_focal_length": 0, "camera1_alignment_rotation": 0,
            "camera1_gain": 0, "camera1_exposure": 0,
            "camera2_pixel_size": 0, "camera2_array_size_diagonal": 0, "camera2_focal_length": 0, "camera2_alignment_rotation": 0,
            "camera2_gain": 0, "camera2_exposure": 0
        }
        self.selection_start = {
            "lat": None, "lon": None, "alt": None, "elevation_mask": None,
            "alignment_azimuth": None, "alignment_elevation": None,
            "azm_offset": None, "alt_offset": None,
            "azm_limit_min": None, "azm_limit_max": None,
            "alt_limit_min": None, "alt_limit_max": None,
            "pid_azm_p_gain": None, "pid_azm_i_gain": None, "pid_azm_d_gain": None,
            "pid_alt_p_gain": None, "pid_alt_i_gain": None, "pid_alt_d_gain": None,
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
            "alignment_azimuth": self.alignment_azimuth_str,
            "alignment_elevation": self.alignment_elevation_str,
            "satellites_enabled": self.satellites_enabled,
            "satellite_labels_enabled": self.satellite_labels_enabled,
            "starfield_enabled": self.starfield_enabled,
            "max_rendered_star_count": self.max_rendered_star_count,
            "star_limiting_magnitude": self.star_limiting_magnitude,
            "plate_solve_enabled": self.plate_solve_enabled,
            "plate_solve_camera_index": self.plate_solve_camera_index,
            "adsb_source_mode": self.adsb_source_mode,
            "adsb_device_index": self.adsb_device_index,
            "adsb_gain": self.adsb_gain,
            "adsb_dump1090_host": self.adsb_dump1090_host,
            "adsb_dump1090_port": self.adsb_dump1090_port,
            "adsb_fit_points": self.adsb_fit_points,
            "adsb_predict_horizon_sec": self.adsb_predict_horizon_sec,
            "adsb_predict_step_sec": self.adsb_predict_step_sec,
            "adsb_history_seconds": self.adsb_history_seconds,
            "adsb_stale_timeout_sec": self.adsb_stale_timeout_sec,
            "alignment_grid_search_cells": self.alignment_grid_search_cells,
            "alignment_settle_tol_deg": self.alignment_settle_tol_deg,
            "alignment_settle_cycles": self.alignment_settle_cycles,
            "alignment_target_rms_arcmin": self.alignment_target_rms_arcmin,
            "pointing_model_enabled": self.pointing_model_enabled,
            "pointing_model_terms": self.pointing_model_terms,
            "eq_pointing_model_enabled": self.eq_pointing_model_enabled,
            "eq_pointing_model_terms": self.eq_pointing_model_terms,
            "azm_offset": self.azm_offset_str,
            "alt_offset": self.alt_offset_str,
            "azm_limit_min": self.azm_limit_min_str,
            "azm_limit_max": self.azm_limit_max_str,
            "alt_limit_min": self.alt_limit_min_str,
            "alt_limit_max": self.alt_limit_max_str,
            "camera_configs": self.camera_configs,
            "pid_azm_p_gain": self.pid_azm_p_gain,
            "pid_azm_i_gain": self.pid_azm_i_gain,
            "pid_azm_d_gain": self.pid_azm_d_gain,
            "pid_alt_p_gain": self.pid_alt_p_gain,
            "pid_alt_i_gain": self.pid_alt_i_gain,
            "pid_alt_d_gain": self.pid_alt_d_gain,
            "pid_lead_time_sec": self.pid_lead_time_sec,
            "feed_forward_azm_enabled": self.feed_forward_azm_enabled,
            "feed_forward_alt_enabled": self.feed_forward_alt_enabled,
            "bias_azm_deg": self.bias_azm_deg,
            "bias_alt_deg": self.bias_alt_deg,
            "bias_control_mode": self.bias_control_mode,
            "hotspot_camera_index": self.hotspot_camera_index,
            "hotspot_snr_threshold": self.hotspot_snr_threshold,
            "hotspot_gate_radius": self.hotspot_gate_radius,
            "hotspot_max_rate_dps": self.hotspot_max_rate_dps,
            "hotspot_coast_time_sec": self.hotspot_coast_time_sec,
            "handoff_min_frames": self.handoff_min_frames,
            "hotspot_x_sign": self.hotspot_x_sign,
            "hotspot_y_sign": self.hotspot_y_sign,
            "sim_config": self.sim_config,
            "mount_mode": self.mount_mode,
            "use_rust_core_loop": self.use_rust_core_loop,
            "rust_core_loop_hz": self.rust_core_loop_hz,
            "continuous_rate_tracking": self.continuous_rate_tracking,
            "guide_rate_max_dps": self.guide_rate_max_dps
        }

    def load_from_dict(self, config_dict):
        """Load configuration from dictionary.

        Unknown keys are reported (a typo'd key otherwise silently falls back
        to the default, which reads as "my setting doesn't work"), and a file
        from a newer config_version gets a warning instead of quiet partial
        loading.
        """
        ver = config_dict.get("config_version", 0)
        if isinstance(ver, (int, float)) and ver > CONFIG_VERSION:
            print(f"Config: file is version {ver}, this app knows version "
                  f"{CONFIG_VERSION} - unknown settings will be ignored")
        known = set(self.get_config_dict().keys()) | {"config_version"}
        unknown = sorted(set(config_dict) - known)
        if unknown:
            print(f"Config: ignoring unknown keys (typo'd, or from a newer "
                  f"version?): {unknown}")
        self.lat_str = config_dict.get("lat", self.lat_str)
        self.lon_str = config_dict.get("lon", self.lon_str)
        self.alt_str = config_dict.get("alt", self.alt_str)
        self.elevation_mask_str = config_dict.get("elevation_mask", self.elevation_mask_str)
        self.alignment_azimuth_str = config_dict.get("alignment_azimuth", self.alignment_azimuth_str)
        self.alignment_elevation_str = config_dict.get("alignment_elevation", self.alignment_elevation_str)
        self.satellites_enabled = config_dict.get("satellites_enabled", self.satellites_enabled)
        self.satellite_labels_enabled = config_dict.get("satellite_labels_enabled", self.satellite_labels_enabled)
        self.starfield_enabled = config_dict.get("starfield_enabled", self.starfield_enabled)
        self.max_rendered_star_count = config_dict.get("max_rendered_star_count", self.max_rendered_star_count)
        self.star_limiting_magnitude = config_dict.get("star_limiting_magnitude", self.star_limiting_magnitude)
        self.plate_solve_enabled = config_dict.get("plate_solve_enabled", self.plate_solve_enabled)
        self.plate_solve_camera_index = config_dict.get("plate_solve_camera_index", self.plate_solve_camera_index)
        self.adsb_source_mode = config_dict.get("adsb_source_mode", self.adsb_source_mode)
        self.adsb_device_index = config_dict.get("adsb_device_index", self.adsb_device_index)
        self.adsb_gain = config_dict.get("adsb_gain", self.adsb_gain)
        self.adsb_dump1090_host = config_dict.get("adsb_dump1090_host", self.adsb_dump1090_host)
        self.adsb_dump1090_port = config_dict.get("adsb_dump1090_port", self.adsb_dump1090_port)
        self.adsb_fit_points = config_dict.get("adsb_fit_points", self.adsb_fit_points)
        self.adsb_predict_horizon_sec = config_dict.get("adsb_predict_horizon_sec", self.adsb_predict_horizon_sec)
        self.adsb_predict_step_sec = config_dict.get("adsb_predict_step_sec", self.adsb_predict_step_sec)
        self.adsb_history_seconds = config_dict.get("adsb_history_seconds", self.adsb_history_seconds)
        self.adsb_stale_timeout_sec = config_dict.get("adsb_stale_timeout_sec", self.adsb_stale_timeout_sec)
        self.alignment_grid_search_cells = config_dict.get("alignment_grid_search_cells", self.alignment_grid_search_cells)
        self.alignment_settle_tol_deg = config_dict.get("alignment_settle_tol_deg", self.alignment_settle_tol_deg)
        self.alignment_settle_cycles = config_dict.get("alignment_settle_cycles", self.alignment_settle_cycles)
        self.alignment_target_rms_arcmin = config_dict.get("alignment_target_rms_arcmin", self.alignment_target_rms_arcmin)
        self.pointing_model_enabled = config_dict.get("pointing_model_enabled", self.pointing_model_enabled)
        if isinstance(config_dict.get("pointing_model_terms"), dict):
            self.pointing_model_terms.update(config_dict["pointing_model_terms"])
        self.eq_pointing_model_enabled = config_dict.get("eq_pointing_model_enabled", self.eq_pointing_model_enabled)
        if isinstance(config_dict.get("eq_pointing_model_terms"), dict):
            self.eq_pointing_model_terms.update(config_dict["eq_pointing_model_terms"])
        self.azm_offset_str = config_dict.get("azm_offset", self.azm_offset_str)
        self.alt_offset_str = config_dict.get("alt_offset", self.alt_offset_str)

        # Load system configuration
        self.buffer_size = config_dict.get("buffer_size", self.buffer_size)
        self.image_format = config_dict.get("image_format", self.image_format)

        # Load PID gains
        self.pid_azm_p_gain = config_dict.get("pid_azm_p_gain", self.pid_azm_p_gain)
        self.pid_azm_i_gain = config_dict.get("pid_azm_i_gain", self.pid_azm_i_gain)
        self.pid_azm_d_gain = config_dict.get("pid_azm_d_gain", self.pid_azm_d_gain)
        self.pid_alt_p_gain = config_dict.get("pid_alt_p_gain", self.pid_alt_p_gain)
        self.pid_alt_i_gain = config_dict.get("pid_alt_i_gain", self.pid_alt_i_gain)
        self.pid_alt_d_gain = config_dict.get("pid_alt_d_gain", self.pid_alt_d_gain)
        self.pid_lead_time_sec = config_dict.get("pid_lead_time_sec", self.pid_lead_time_sec)

        # Load feed-forward and bias control settings
        self.feed_forward_azm_enabled = config_dict.get("feed_forward_azm_enabled", self.feed_forward_azm_enabled)
        self.feed_forward_alt_enabled = config_dict.get("feed_forward_alt_enabled", self.feed_forward_alt_enabled)
        self.bias_azm_deg = config_dict.get("bias_azm_deg", self.bias_azm_deg)
        self.bias_alt_deg = config_dict.get("bias_alt_deg", self.bias_alt_deg)

        # Load hotspot tracker settings
        self.hotspot_camera_index = config_dict.get("hotspot_camera_index", self.hotspot_camera_index)
        self.hotspot_snr_threshold = config_dict.get("hotspot_snr_threshold", self.hotspot_snr_threshold)
        self.hotspot_gate_radius = config_dict.get("hotspot_gate_radius", self.hotspot_gate_radius)
        self.hotspot_max_rate_dps = config_dict.get("hotspot_max_rate_dps", self.hotspot_max_rate_dps)
        self.hotspot_coast_time_sec = config_dict.get("hotspot_coast_time_sec", self.hotspot_coast_time_sec)
        self.handoff_min_frames = config_dict.get("handoff_min_frames", self.handoff_min_frames)
        self.hotspot_x_sign = config_dict.get("hotspot_x_sign", self.hotspot_x_sign)
        self.hotspot_y_sign = config_dict.get("hotspot_y_sign", self.hotspot_y_sign)

        # Merge hardware simulator settings (keep defaults for missing keys)
        if "sim_config" in config_dict and isinstance(config_dict["sim_config"], dict):
            self.sim_config.update(config_dict["sim_config"])
        self.bias_control_mode = config_dict.get("bias_control_mode", self.bias_control_mode)

        # Load mount mode
        self.mount_mode = config_dict.get("mount_mode", self.mount_mode)

        # Load experimental Rust core loop toggle
        self.use_rust_core_loop = config_dict.get("use_rust_core_loop", self.use_rust_core_loop)
        self.rust_core_loop_hz = config_dict.get("rust_core_loop_hz", self.rust_core_loop_hz)

        # Continuous variable-rate tracking
        self.continuous_rate_tracking = config_dict.get("continuous_rate_tracking", self.continuous_rate_tracking)
        self.guide_rate_max_dps = config_dict.get("guide_rate_max_dps", self.guide_rate_max_dps)

        # Load hardware safety limits
        self.azm_limit_min_str = config_dict.get("azm_limit_min", self.azm_limit_min_str)
        self.azm_limit_max_str = config_dict.get("azm_limit_max", self.azm_limit_max_str)
        self.alt_limit_min_str = config_dict.get("alt_limit_min", self.alt_limit_min_str)
        self.alt_limit_max_str = config_dict.get("alt_limit_max", self.alt_limit_max_str)

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
            "exposure": 10000.0,
            "gamma": 0.1,
            "gamma_enabled": False
        }

        for camera_name in self.camera_configs:
            for key, default_value in defaults.items():
                if key not in self.camera_configs[camera_name]:
                    self.camera_configs[camera_name][key] = default_value

    def reset_input_fields(self):
        """Reset input field positions when switching modes."""
        self.cursor_pos = {
            "lat": 0, "lon": 0, "alt": 0, "elevation_mask": 0,
            "alignment_azimuth": 0, "alignment_elevation": 0,
            "azm_offset": 0, "alt_offset": 0,
            "azm_limit_min": 0, "azm_limit_max": 0,
            "alt_limit_min": 0, "alt_limit_max": 0,
            "pid_azm_p_gain": 0, "pid_azm_i_gain": 0, "pid_azm_d_gain": 0,
            "pid_alt_p_gain": 0, "pid_alt_i_gain": 0, "pid_alt_d_gain": 0,
            "camera1_pixel_size": 0, "camera1_array_size_diagonal": 0, "camera1_focal_length": 0, "camera1_alignment_rotation": 0,
            "camera1_gain": 0, "camera1_exposure": 0,
            "camera2_pixel_size": 0, "camera2_array_size_diagonal": 0, "camera2_focal_length": 0, "camera2_alignment_rotation": 0,
            "camera2_gain": 0, "camera2_exposure": 0
        }
        self.selection_start = {
            "lat": None, "lon": None, "alt": None, "elevation_mask": None,
            "alignment_azimuth": None, "alignment_elevation": None,
            "azm_offset": None, "alt_offset": None,
            "pid_azm_p_gain": None, "pid_azm_i_gain": None, "pid_azm_d_gain": None,
            "pid_alt_p_gain": None, "pid_alt_i_gain": None, "pid_alt_d_gain": None,
            "camera1_pixel_size": None, "camera1_array_size_diagonal": None, "camera1_focal_length": None, "camera1_alignment_rotation": None,
            "camera1_gain": None, "camera1_exposure": None,
            "camera2_pixel_size": None, "camera2_array_size_diagonal": None, "camera2_focal_length": None, "camera2_alignment_rotation": None,
            "camera2_gain": None, "camera2_exposure": None
        }
        self.focused_field = None

    def load_from_file(self, file_path=None):
        """Load configuration from file and update state directly.

        With no explicit path: config.json if present, else
        config.example.json (a fresh checkout ships only the example; the
        first Save writes a real config.json).
        """
        if file_path is None:
            if os.path.exists("config.json"):
                file_path = "config.json"
            elif os.path.exists("config.example.json"):
                file_path = "config.example.json"
            else:
                return
        try:
            with open(file_path, "r") as f:
                loaded_config = json.load(f)
                self.load_from_dict(loaded_config)
        except Exception as e:
            print(f"Debug: Error loading {file_path}: {e}")

    def save_to_file(self):
        """Save configuration to file, keeping a one-deep backup so an accidental
        overwrite (e.g. a default-valued ConfigState saving over a tuned config)
        is recoverable from config.json.bak."""
        config_dict = self.get_config_dict()
        config_dict["config_version"] = CONFIG_VERSION
        if os.path.exists("config.json"):
            try:
                import shutil
                shutil.copyfile("config.json", "config.json.bak")
            except Exception as e:
                print(f"Debug: could not back up config.json: {e}")
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


def draw_angle_groups(display, config_state):
    """
    Draw PID and alignment configuration UI groups.
    This extends draw_config_options with PID parameter controls.
    """
    import pygame
    from utils import draw_button_with_objects

    # Draw PID configuration group (right side)
    pid_group_rect = pygame.Rect(display.sub_x + 480, display.sub_y + 0, 320, 280)
    pygame.draw.rect(display.menu_screen, (0, 0, 0), pid_group_rect, 2, border_radius=5)
    pid_label = display.font.render("PID Control", True, (0, 0, 0))
    display.menu_screen.blit(pid_label, (display.sub_x + 490, display.sub_y + 10))

    current_y = display.sub_y + 40

    # AZM PID gains
    azm_p_label = display.small_font.render("AZM P:", True, (0, 0, 0))
    display.menu_screen.blit(azm_p_label, (display.sub_x + 490, current_y))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['pid_azm_p_gain'])
    azm_p_text = display.small_font.render(f"{config_state.pid_azm_p_gain:.3f}", True, (0, 0, 0))
    display.menu_screen.blit(azm_p_text, (display.input_rects['pid_azm_p_gain'].x + 5, display.input_rects['pid_azm_p_gain'].y + 5))

    azm_i_label = display.small_font.render("I:", True, (0, 0, 0))
    display.menu_screen.blit(azm_i_label, (display.sub_x + 600, current_y))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['pid_azm_i_gain'])
    azm_i_text = display.small_font.render(f"{config_state.pid_azm_i_gain:.3f}", True, (0, 0, 0))
    display.menu_screen.blit(azm_i_text, (display.input_rects['pid_azm_i_gain'].x + 5, display.input_rects['pid_azm_i_gain'].y + 5))

    azm_d_label = display.small_font.render("D:", True, (0, 0, 0))
    display.menu_screen.blit(azm_d_label, (display.sub_x + 710, current_y))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['pid_azm_d_gain'])
    azm_d_text = display.small_font.render(f"{config_state.pid_azm_d_gain:.3f}", True, (0, 0, 0))
    display.menu_screen.blit(azm_d_text, (display.input_rects['pid_azm_d_gain'].x + 5, display.input_rects['pid_azm_d_gain'].y + 5))

    current_y += 70

    # ALT PID gains
    alt_p_label = display.small_font.render("ALT P:", True, (0, 0, 0))
    display.menu_screen.blit(alt_p_label, (display.sub_x + 490, current_y))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['pid_alt_p_gain'])
    alt_p_text = display.small_font.render(f"{config_state.pid_alt_p_gain:.3f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_p_text, (display.input_rects['pid_alt_p_gain'].x + 5, display.input_rects['pid_alt_p_gain'].y + 5))

    alt_i_label = display.small_font.render("I:", True, (0, 0, 0))
    display.menu_screen.blit(alt_i_label, (display.sub_x + 600, current_y))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['pid_alt_i_gain'])
    alt_i_text = display.small_font.render(f"{config_state.pid_alt_i_gain:.3f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_i_text, (display.input_rects['pid_alt_i_gain'].x + 5, display.input_rects['pid_alt_i_gain'].y + 5))

    alt_d_label = display.small_font.render("D:", True, (0, 0, 0))
    display.menu_screen.blit(alt_d_label, (display.sub_x + 710, current_y))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), display.input_rects['pid_alt_d_gain'])
    alt_d_text = display.small_font.render(f"{config_state.pid_alt_d_gain:.3f}", True, (0, 0, 0))
    display.menu_screen.blit(alt_d_text, (display.input_rects['pid_alt_d_gain'].x + 5, display.input_rects['pid_alt_d_gain'].y + 5))

    current_y += 70

    # Feed-forward status display
    ff_azm_label = display.small_font.render("AZM FF:", True, (0, 0, 0))
    display.menu_screen.blit(ff_azm_label, (display.sub_x + 480, current_y))
    ff_azm_status = "ON" if config_state.feed_forward_azm_enabled else "OFF"
    ff_azm_color = (0, 150, 0) if config_state.feed_forward_azm_enabled else (150, 0, 0)
    ff_azm_status_text = display.small_font.render(ff_azm_status, True, ff_azm_color)
    display.menu_screen.blit(ff_azm_status_text, (display.sub_x + 550, current_y))

    ff_alt_label = display.small_font.render("ALT FF:", True, (0, 0, 0))
    display.menu_screen.blit(ff_alt_label, (display.sub_x + 620, current_y))
    ff_alt_status = "ON" if config_state.feed_forward_alt_enabled else "OFF"
    ff_alt_color = (0, 150, 0) if config_state.feed_forward_alt_enabled else (150, 0, 0)
    ff_alt_status_text = display.small_font.render(ff_alt_status, True, ff_alt_color)
    display.menu_screen.blit(ff_alt_status_text, (display.sub_x + 690, current_y))

    current_y += 30

    # Focus highlights for PID fields
    pid_focus_fields = ['pid_azm_p_gain', 'pid_azm_i_gain', 'pid_azm_d_gain',
                       'pid_alt_p_gain', 'pid_alt_i_gain', 'pid_alt_d_gain']

    for field in pid_focus_fields:
        if config_state.focused_field == field and field in display.input_rects:
            pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects[field], 2)
            field_str = ""
            if field == 'pid_azm_p_gain':
                field_str = f"{config_state.pid_azm_p_gain:.3f}"
            elif field == 'pid_azm_i_gain':
                field_str = f"{config_state.pid_azm_i_gain:.3f}"
            elif field == 'pid_azm_d_gain':
                field_str = f"{config_state.pid_azm_d_gain:.3f}"
            elif field == 'pid_alt_p_gain':
                field_str = f"{config_state.pid_alt_p_gain:.3f}"
            elif field == 'pid_alt_i_gain':
                field_str = f"{config_state.pid_alt_i_gain:.3f}"
            elif field == 'pid_alt_d_gain':
                field_str = f"{config_state.pid_alt_d_gain:.3f}"

            if field_str and config_state.focused_field in config_state.cursor_pos:
                text_width, _ = display.small_font.size(field_str[:config_state.cursor_pos[config_state.focused_field]])
                rect = display.input_rects[config_state.focused_field]
                pygame.draw.line(display.menu_screen, (0, 0, 255),
                               (rect.x + 5 + text_width, rect.y + 5),
                               (rect.x + 5 + text_width, rect.y + 25), 2)


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
    elif focused_field == "alignment_azimuth":
        field_str = config_state.alignment_azimuth_str
    elif focused_field == "alignment_elevation":
        field_str = config_state.alignment_elevation_str
    elif focused_field == "azm_offset":
        field_str = config_state.azm_offset_str
    elif focused_field == "alt_offset":
        field_str = config_state.alt_offset_str
    elif focused_field == "pid_azm_p_gain":
        field_str = f"{config_state.pid_azm_p_gain:.3f}"
    elif focused_field == "pid_azm_i_gain":
        field_str = f"{config_state.pid_azm_i_gain:.3f}"
    elif focused_field == "pid_azm_d_gain":
        field_str = f"{config_state.pid_azm_d_gain:.3f}"
    elif focused_field == "pid_alt_p_gain":
        field_str = f"{config_state.pid_alt_p_gain:.3f}"
    elif focused_field == "pid_alt_i_gain":
        field_str = f"{config_state.pid_alt_i_gain:.3f}"
    elif focused_field == "pid_alt_d_gain":
        field_str = f"{config_state.pid_alt_d_gain:.3f}"
    elif focused_field == "azm_limit_min":
        field_str = config_state.azm_limit_min_str
    elif focused_field == "azm_limit_max":
        field_str = config_state.azm_limit_max_str
    elif focused_field == "alt_limit_min":
        field_str = config_state.alt_limit_min_str
    elif focused_field == "alt_limit_max":
        field_str = config_state.alt_limit_max_str
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
    elif focused_field == "alignment_azimuth":
        config_state.alignment_azimuth_str = field_str
    elif focused_field == "alignment_elevation":
        config_state.alignment_elevation_str = field_str
    elif focused_field == "azm_offset":
        config_state.azm_offset_str = field_str
    elif focused_field == "alt_offset":
        config_state.alt_offset_str = field_str
    elif focused_field == "pid_azm_p_gain":
        try:
            config_state.pid_azm_p_gain = float(field_str) if field_str else 1.0
        except ValueError:
            pass
    elif focused_field == "pid_azm_i_gain":
        try:
            config_state.pid_azm_i_gain = float(field_str) if field_str else 0.0
        except ValueError:
            pass
    elif focused_field == "pid_azm_d_gain":
        try:
            config_state.pid_azm_d_gain = float(field_str) if field_str else 0.0
        except ValueError:
            pass
    elif focused_field == "pid_alt_p_gain":
        try:
            config_state.pid_alt_p_gain = float(field_str) if field_str else 1.0
        except ValueError:
            pass
    elif focused_field == "pid_alt_i_gain":
        try:
            config_state.pid_alt_i_gain = float(field_str) if field_str else 0.0
        except ValueError:
            pass
    elif focused_field == "pid_alt_d_gain":
        try:
            config_state.pid_alt_d_gain = float(field_str) if field_str else 0.0
        except ValueError:
            pass
    elif focused_field == "azm_limit_min":
        config_state.azm_limit_min_str = field_str
    elif focused_field == "azm_limit_max":
        config_state.azm_limit_max_str = field_str
    elif focused_field == "alt_limit_min":
        config_state.alt_limit_min_str = field_str
    elif focused_field == "alt_limit_max":
        config_state.alt_limit_max_str = field_str


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
    group_rect = pygame.Rect(display.sub_x + 10, display.sub_y + 0, 220, 990)
    pygame.draw.rect(display.menu_screen, (0, 0, 0), group_rect, 2, border_radius=5)
    group_label = display.font.render("Site / Mount / Limits", True, (0, 0, 0))
    display.menu_screen.blit(group_label, (display.sub_x + 20, display.sub_y + 10))

    # Left column fields. Each label is drawn a consistent gap ABOVE its input
    # box, with the y derived from display.input_rects so the column stays
    # aligned regardless of box spacing. (Previously the label offsets were
    # hand-tuned and drifted out of register -- the azimuth-alignment label even
    # overlapped its own box, and a duplicate header was drawn above it.)
    left_fields = [
        ('lat', "Latitude:", config_state.lat_str),
        ('lon', "Longitude:", config_state.lon_str),
        ('alt', "Altitude (m):", config_state.alt_str),
        ('elevation_mask', "Elevation Mask (deg):", config_state.elevation_mask_str),
        ('alignment_azimuth', "Azimuth Alignment (deg):", config_state.alignment_azimuth_str),
        ('alignment_elevation', "Elevation Align (deg, Eq):", config_state.alignment_elevation_str),
        ('azm_offset', "Azm Offset (deg):", config_state.azm_offset_str),
        ('alt_offset', "Alt Offset (deg):", config_state.alt_offset_str),
        ('azm_limit_min', "AZM Limit Min (deg):", config_state.azm_limit_min_str),
        ('azm_limit_max', "AZM Limit Max (deg):", config_state.azm_limit_max_str),
        ('alt_limit_min', "ALT Limit Min (deg):", config_state.alt_limit_min_str),
        ('alt_limit_max', "ALT Limit Max (deg):", config_state.alt_limit_max_str),
    ]
    for field, label_text, value_str in left_fields:
        rect = display.input_rects[field]
        label = display.font.render(label_text, True, (0, 0, 0))
        display.menu_screen.blit(label, (rect.x, rect.y - 22))
        pygame.draw.rect(display.menu_screen, (255, 255, 255), rect)
        value = display.font.render(value_str, True, (0, 0, 0))
        display.menu_screen.blit(value, (rect.x + 5, rect.y + 5))

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

    if config_state.focused_field == "alignment_azimuth":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['alignment_azimuth'], 2)
        if config_state.focused_field == "alignment_azimuth":
            text_width, _ = display.font.size(config_state.alignment_azimuth_str[:config_state.cursor_pos["alignment_azimuth"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['alignment_azimuth'].x + 5 + text_width, display.input_rects['alignment_azimuth'].y + 5),
                            (display.input_rects['alignment_azimuth'].x + 5 + text_width, display.input_rects['alignment_azimuth'].y + 25), 2)

    if config_state.focused_field == "alignment_elevation":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['alignment_elevation'], 2)
        if config_state.focused_field == "alignment_elevation":
            text_width, _ = display.font.size(config_state.alignment_elevation_str[:config_state.cursor_pos["alignment_elevation"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['alignment_elevation'].x + 5 + text_width, display.input_rects['alignment_elevation'].y + 5),
                            (display.input_rects['alignment_elevation'].x + 5 + text_width, display.input_rects['alignment_elevation'].y + 25), 2)

    if config_state.focused_field == "azm_offset":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['azm_offset'], 2)
        if config_state.focused_field == "azm_offset":
            text_width, _ = display.font.size(config_state.azm_offset_str[:config_state.cursor_pos["azm_offset"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['azm_offset'].x + 5 + text_width, display.input_rects['azm_offset'].y + 5),
                            (display.input_rects['azm_offset'].x + 5 + text_width, display.input_rects['azm_offset'].y + 25), 2)

    if config_state.focused_field == "alt_offset":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['alt_offset'], 2)
        if config_state.focused_field == "alt_offset":
            text_width, _ = display.font.size(config_state.alt_offset_str[:config_state.cursor_pos["alt_offset"]])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['alt_offset'].x + 5 + text_width, display.input_rects['alt_offset'].y + 5),
                            (display.input_rects['alt_offset'].x + 5 + text_width, display.input_rects['alt_offset'].y + 25), 2)

    # Hardware safety limits focus highlights
    if config_state.focused_field == "azm_limit_min":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['azm_limit_min'], 2)
        if config_state.focused_field == "azm_limit_min":
            text_width, _ = display.font.size(config_state.azm_limit_min_str[:config_state.cursor_pos.get("azm_limit_min", 0)])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['azm_limit_min'].x + 5 + text_width, display.input_rects['azm_limit_min'].y + 5),
                            (display.input_rects['azm_limit_min'].x + 5 + text_width, display.input_rects['azm_limit_min'].y + 25), 2)

    if config_state.focused_field == "azm_limit_max":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['azm_limit_max'], 2)
        if config_state.focused_field == "azm_limit_max":
            text_width, _ = display.font.size(config_state.azm_limit_max_str[:config_state.cursor_pos.get("azm_limit_max", 0)])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['azm_limit_max'].x + 5 + text_width, display.input_rects['azm_limit_max'].y + 5),
                            (display.input_rects['azm_limit_max'].x + 5 + text_width, display.input_rects['azm_limit_max'].y + 25), 2)

    if config_state.focused_field == "alt_limit_min":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['alt_limit_min'], 2)
        if config_state.focused_field == "alt_limit_min":
            text_width, _ = display.font.size(config_state.alt_limit_min_str[:config_state.cursor_pos.get("alt_limit_min", 0)])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['alt_limit_min'].x + 5 + text_width, display.input_rects['alt_limit_min'].y + 5),
                            (display.input_rects['alt_limit_min'].x + 5 + text_width, display.input_rects['alt_limit_min'].y + 25), 2)

    if config_state.focused_field == "alt_limit_max":
        pygame.draw.rect(display.menu_screen, (0, 0, 255), display.input_rects['alt_limit_max'], 2)
        if config_state.focused_field == "alt_limit_max":
            text_width, _ = display.font.size(config_state.alt_limit_max_str[:config_state.cursor_pos.get("alt_limit_max", 0)])
            pygame.draw.line(display.menu_screen, (0, 0, 255),
                            (display.input_rects['alt_limit_max'].x + 5 + text_width, display.input_rects['alt_limit_max'].y + 5),
                            (display.input_rects['alt_limit_max'].x + 5 + text_width, display.input_rects['alt_limit_max'].y + 25), 2)

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

    # Draw PID configuration group (extends config options with PID controls)
    draw_angle_groups(display, config_state)

    # Draw buttons and divider
    pygame.draw.line(display.menu_screen, (0, 0, 0), (display.sub_x, display.sub_y + display.sub_height - 60), (display.sub_x + display.sub_width, display.sub_y + display.sub_height - 60), 1)
    # Config mode buttons use display object properties
    draw_button_with_objects(display, "save")
    draw_button_with_objects(display, "load")


def load_config(file_path=None):
    """Create and initialize ConfigState from file."""
    config_state = ConfigState()
    config_state.load_from_file(file_path)
    return config_state


