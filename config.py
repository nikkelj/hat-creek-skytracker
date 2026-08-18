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
        self.aircraft_enabled = True            # draw ADS-B aircraft markers

        # Star catalogue / starfield configuration
        self.starfield_enabled = True          # draw catalogue stars on the skyplot
        self.max_rendered_star_count = 2000    # brightest-N cap (skyplot + sim images)
        self.star_limiting_magnitude = 6.5     # faintest star ever considered

        # Explanatory hover tooltips (tooltips.py) - global on/off, toggled by
        # the Tips chip in the bottom-right corner of every screen.
        self.tooltips_enabled = True

        # Deep-sky overlays (celestial.py). Sun/moon/planets and the top-100
        # named stars are always drawn; these gate the denser catalogues.
        self.messier_enabled = True            # Messier markers on the skyplot
        self.ngc_enabled = False               # NGC markers (noisy; off by default)
        self.ngc_limiting_magnitude = 10.0     # faintest NGC object shown
        self.meo_enabled = True                # MEO satellites (legend-click toggle)
        self.geo_enabled = True                # GEO satellites (legend-click toggle)

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

        # Mount 3D screen: operator seat relative to the mount (bearing from
        # the mount, distance, eye height) -- drives the "operator view"
        # camera so the rendered perspective matches where the user sits.
        self.mount3d_observer_bearing_deg = 200.0
        self.mount3d_observer_distance_m = 2.5
        self.mount3d_eye_height_m = 1.2

        # Per-tracking-mode PID gain profiles, keyed "PROGRAM" / "HOTSPOT".
        # The six pid_*_gain fields above are the LIVE set both control loops
        # (and the UI sliders and auto-tuner) read; JoystickModeState.
        # service_gain_profiles() swaps the live set from these profiles
        # automatically on mode transitions -- the encoder loop and the
        # optical loop are different plants and want different gains.
        self.pid_mode_profiles = {}

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

        # Star-rejection rate filter: a detection is only accepted when its
        # implied sky angular rate (boresight motion + pixel drift between
        # fresh frames) matches the program-track trajectory rate within
        # hotspot_rate_gate_dps. Without a trajectory (bare HOTSPOT), reject
        # near-sidereal (star-like) rates below the gate instead. Toggle off
        # to deliberately track a star.
        self.hotspot_star_filter_enabled = True
        self.hotspot_rate_gate_dps = 0.15

        # Low-pass (EMA) time constant for the PID FEEDBACK term, seconds.
        # Smooths encoder-noise/backlash jitter out of the rate commands while
        # feed-forward passes through unfiltered (trajectory response stays
        # crisp). 0 disables.
        self.pid_output_filter_tau_sec = 0.0

        # HANDOFF mode: PROGRAM track runs the hotspot detector in parallel and
        # hands the loop to HOTSPOT after this many consecutive solid detections.
        self.handoff_min_frames = 5

        # RATE_CONTROL adaptive gearbox (joystick_controller.AdaptiveRateMapper):
        # full stick deflection commands at most base_ceiling (MC_MOVE step,
        # 1-9) until the stick has been held pinned for windup_delay seconds,
        # after which the ceiling steps toward 9. Tune the feel here.
        self.joy_rate_base_ceiling = 5
        self.joy_rate_windup_delay_s = 0.8

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

        # Mount mode configuration. "AltAz" (az axis vertical), "AltAz-Side"
        # (mount lying on its side: the az axis is HORIZONTAL at
        # alignment_azimuth -- equatorial geometry with the pole on the
        # horizon), "Eq" (polar-aligned wedge), or "Passthrough".
        # AltAz-Side field setup: put both axes on the AVX index marks
        # (scope points along the polar axis, at the horizon), press
        # Sync Home in the joystick connection panel (captures the raw
        # encoder readings into azm/alt_offset below), and enter the azimuth
        # the scope points at as Alignment Azimuth.
        self.mount_mode = "AltAz"
        # AltAz-Side tip side: False = first +ALT jog from the index marks
        # sweeps toward alignment_azimuth - 90; True = mirrored rig (laid
        # down on its other side), sweeps toward alignment_azimuth + 90.
        self.altaz_side_flip = False

        # Continuous-rate tracking: issue the fine 24-bit variable-rate
        # (MC_SET_POS/NEG_GUIDERATE) command for smooth tracking instead of the
        # coarse 10-step MC_MOVE, falling back to MC_MOVE above guide_rate_max_dps
        # (e.g. the near-zenith keyhole). Off by default until verified on hardware
        # with bench_guiderate.py.
        self.continuous_rate_tracking = False
        # Max rate sent via the guide-rate command. The 24-bit wire value is
        # arcsec/s * 1024 (calibrated on the AVX 2026-07-25), so full scale is
        # 4.551 dps -- stay below it; MC_MOVE handles faster slews.
        self.guide_rate_max_dps = 4.5

        # Experimental: run the control loop in Rust (skytracker_core) instead of
        # the Python MountControlThread. Read at startup in main.py; toggled from
        # the Hardware Simulator screen. Takes effect on restart.
        self.use_rust_core_loop = False
        self.rust_core_loop_hz = 15
        # Phase 1 of the Rust port: route trajectory/celestial math through
        # the skytracker-astro engine (skyfield stays as the fallback).
        self.use_rust_astro = False
        # Phase 2: plate-solve with skytracker-platesolve (tetra3 fallback).
        self.use_rust_platesolve = False
        # Phase 2b: pointing-model fits via skytracker-pointing (numpy fallback).
        self.use_rust_pointing = False
        # Phase 3b: stacking/stabilizer/sharpen kernels via skytracker-imaging.
        self.use_rust_imaging = False
        # Phase 5: ADS-B Mode-S decode via skytracker-adsb (pyModeS fallback).
        self.use_rust_adsb = False
        # Phase 4b: camera capture via skytracker-camera (CameraThread fallback).
        self.use_rust_camera = False

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
            "aircraft_enabled": self.aircraft_enabled,
            "tooltips_enabled": self.tooltips_enabled,
            "messier_enabled": self.messier_enabled,
            "ngc_enabled": self.ngc_enabled,
            "ngc_limiting_magnitude": self.ngc_limiting_magnitude,
            "meo_enabled": self.meo_enabled,
            "geo_enabled": self.geo_enabled,
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
            "pid_mode_profiles": self.pid_mode_profiles,
            "pid_lead_time_sec": self.pid_lead_time_sec,
            "mount3d_observer_bearing_deg": self.mount3d_observer_bearing_deg,
            "mount3d_observer_distance_m": self.mount3d_observer_distance_m,
            "mount3d_eye_height_m": self.mount3d_eye_height_m,
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
            "hotspot_star_filter_enabled": self.hotspot_star_filter_enabled,
            "hotspot_rate_gate_dps": self.hotspot_rate_gate_dps,
            "pid_output_filter_tau_sec": self.pid_output_filter_tau_sec,
            "joy_rate_base_ceiling": self.joy_rate_base_ceiling,
            "joy_rate_windup_delay_s": self.joy_rate_windup_delay_s,
            "sim_config": self.sim_config,
            "mount_mode": self.mount_mode,
            "altaz_side_flip": self.altaz_side_flip,
            "use_rust_core_loop": self.use_rust_core_loop,
            "rust_core_loop_hz": self.rust_core_loop_hz,
            "use_rust_astro": self.use_rust_astro,
            "use_rust_platesolve": self.use_rust_platesolve,
            "use_rust_pointing": self.use_rust_pointing,
            "use_rust_imaging": self.use_rust_imaging,
            "use_rust_adsb": self.use_rust_adsb,
            "use_rust_camera": self.use_rust_camera,
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
        self.aircraft_enabled = config_dict.get("aircraft_enabled", self.aircraft_enabled)
        self.tooltips_enabled = config_dict.get("tooltips_enabled", self.tooltips_enabled)
        self.messier_enabled = config_dict.get("messier_enabled", self.messier_enabled)
        self.ngc_enabled = config_dict.get("ngc_enabled", self.ngc_enabled)
        self.ngc_limiting_magnitude = config_dict.get(
            "ngc_limiting_magnitude", self.ngc_limiting_magnitude)
        self.meo_enabled = config_dict.get("meo_enabled", self.meo_enabled)
        self.geo_enabled = config_dict.get("geo_enabled", self.geo_enabled)
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
        if isinstance(config_dict.get("pid_mode_profiles"), dict):
            self.pid_mode_profiles = config_dict["pid_mode_profiles"]
        self.pid_lead_time_sec = config_dict.get("pid_lead_time_sec", self.pid_lead_time_sec)
        self.mount3d_observer_bearing_deg = config_dict.get(
            "mount3d_observer_bearing_deg", self.mount3d_observer_bearing_deg)
        self.mount3d_observer_distance_m = config_dict.get(
            "mount3d_observer_distance_m", self.mount3d_observer_distance_m)
        self.mount3d_eye_height_m = config_dict.get(
            "mount3d_eye_height_m", self.mount3d_eye_height_m)

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
        self.hotspot_star_filter_enabled = config_dict.get(
            "hotspot_star_filter_enabled", self.hotspot_star_filter_enabled)
        self.hotspot_rate_gate_dps = config_dict.get(
            "hotspot_rate_gate_dps", self.hotspot_rate_gate_dps)
        self.pid_output_filter_tau_sec = config_dict.get(
            "pid_output_filter_tau_sec", self.pid_output_filter_tau_sec)
        self.joy_rate_base_ceiling = config_dict.get(
            "joy_rate_base_ceiling", self.joy_rate_base_ceiling)
        self.joy_rate_windup_delay_s = config_dict.get(
            "joy_rate_windup_delay_s", self.joy_rate_windup_delay_s)

        # Merge hardware simulator settings (keep defaults for missing keys)
        if "sim_config" in config_dict and isinstance(config_dict["sim_config"], dict):
            self.sim_config.update(config_dict["sim_config"])
        self.bias_control_mode = config_dict.get("bias_control_mode", self.bias_control_mode)

        # Load mount mode
        self.mount_mode = config_dict.get("mount_mode", self.mount_mode)
        self.altaz_side_flip = bool(config_dict.get("altaz_side_flip", self.altaz_side_flip))

        # Load experimental Rust core loop toggle
        self.use_rust_core_loop = config_dict.get("use_rust_core_loop", self.use_rust_core_loop)
        self.rust_core_loop_hz = config_dict.get("rust_core_loop_hz", self.rust_core_loop_hz)
        self.use_rust_astro = config_dict.get("use_rust_astro", self.use_rust_astro)
        self.use_rust_platesolve = config_dict.get("use_rust_platesolve", self.use_rust_platesolve)
        self.use_rust_pointing = config_dict.get("use_rust_pointing", self.use_rust_pointing)
        self.use_rust_imaging = config_dict.get("use_rust_imaging", self.use_rust_imaging)
        self.use_rust_adsb = config_dict.get("use_rust_adsb", self.use_rust_adsb)
        self.use_rust_camera = config_dict.get("use_rust_camera", self.use_rust_camera)

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


def _draw_config_field(display, config_state, field, label_text, value_str):
    """Draw one config input: label above the box, white box, current value,
    and (when focused) the blue focus ring + text cursor. One code path for
    every field on the config screen -- the per-field copies this replaced had
    drifted apart in fonts, offsets, and cursor handling."""
    import pygame
    rect = display.input_rects[field]
    if label_text:
        label = display.font.render(label_text, True, (0, 0, 0))
        display.menu_screen.blit(label, (rect.x, rect.y - 22))
    pygame.draw.rect(display.menu_screen, (255, 255, 255), rect)
    value = display.font.render(value_str, True, (0, 0, 0))
    display.menu_screen.blit(value, (rect.x + 5, rect.y + 5))
    if config_state.focused_field == field:
        pygame.draw.rect(display.menu_screen, (0, 0, 255), rect, 2)
        cursor = config_state.cursor_pos.get(field, 0)
        text_width, _ = display.font.size(value_str[:cursor])
        pygame.draw.line(display.menu_screen, (0, 0, 255),
                         (rect.x + 5 + text_width, rect.y + 5),
                         (rect.x + 5 + text_width, rect.y + rect.height - 5), 2)


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
    # 5 decimals: tuned gains (auto-tune, log sliders) live at 1e-4..1e-3
    # scale; the old %.3f displayed e.g. an I gain of 0.00025 as "0.000".
    elif focused_field == "pid_azm_p_gain":
        field_str = f"{config_state.pid_azm_p_gain:.5f}"
    elif focused_field == "pid_azm_i_gain":
        field_str = f"{config_state.pid_azm_i_gain:.5f}"
    elif focused_field == "pid_azm_d_gain":
        field_str = f"{config_state.pid_azm_d_gain:.5f}"
    elif focused_field == "pid_alt_p_gain":
        field_str = f"{config_state.pid_alt_p_gain:.5f}"
    elif focused_field == "pid_alt_i_gain":
        field_str = f"{config_state.pid_alt_i_gain:.5f}"
    elif focused_field == "pid_alt_d_gain":
        field_str = f"{config_state.pid_alt_d_gain:.5f}"
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
    """Draw the configuration screen: five aligned group columns (Site &
    Alignment, Mount & Limits, Camera 1, Camera 2, PID Gains), rendered
    data-driven from display.input_rects / display.config_group_rects so
    labels, boxes, and group frames stay in register. _draw_config_field
    draws label/box/value/focus identically for every field."""
    import pygame
    from utils import draw_button_with_objects

    # Gradient background
    for y in range(display.sub_height):
        shade = int(160 - (y / display.sub_height * 5))
        pygame.draw.line(display.menu_screen, (shade, shade, shade),
                         (display.sub_x, display.sub_y + y),
                         (display.sub_x + display.sub_width, display.sub_y + y))

    def camera_fields(prefix, cam):
        return [
            (f'{prefix}_pixel_size', "Pixel Size (μm):", f"{cam['pixel_size']:.2f}"),
            (f'{prefix}_array_size_diagonal', "Array Diagonal (mm):", f"{cam['array_size_diagonal']:.1f}"),
            (f'{prefix}_focal_length', "Focal Length (mm):", f"{cam['focal_length']:.1f}"),
            (f'{prefix}_alignment_rotation', "Align Rotation (deg):", f"{cam['alignment_rotation']:.1f}"),
            (f'{prefix}_gain', "Gain:", f"{cam['gain']:.2f}"),
            (f'{prefix}_exposure', "Exposure (μs):", f"{cam['exposure']:.0f}"),
        ]

    groups = [
        ('site', "Site & Alignment", [
            ('lat', "Latitude:", config_state.lat_str),
            ('lon', "Longitude:", config_state.lon_str),
            ('alt', "Altitude (m):", config_state.alt_str),
            ('elevation_mask', "Elevation Mask (deg):", config_state.elevation_mask_str),
            # TRUE north, not magnetic: a compass-aligned mount misses passes
            # by the local declination (~12 deg here) -- see the tooltip.
            ('alignment_azimuth', "Azimuth Align (deg, TRUE N):", config_state.alignment_azimuth_str),
            ('alignment_elevation', "Elevation Align (deg, Eq):", config_state.alignment_elevation_str),
        ]),
        ('mount', "Mount & Limits", [
            ('azm_offset', "Azm Offset (deg):", config_state.azm_offset_str),
            ('alt_offset', "Alt Offset (deg):", config_state.alt_offset_str),
            ('azm_limit_min', "AZM Limit Min (deg):", config_state.azm_limit_min_str),
            ('azm_limit_max', "AZM Limit Max (deg):", config_state.azm_limit_max_str),
            ('alt_limit_min', "ALT Limit Min (deg):", config_state.alt_limit_min_str),
            ('alt_limit_max', "ALT Limit Max (deg):", config_state.alt_limit_max_str),
        ]),
        ('camera1', "Camera 1", camera_fields('camera1', config_state.camera_configs['camera1'])),
        ('camera2', "Camera 2", camera_fields('camera2', config_state.camera_configs['camera2'])),
        # 5 decimals: tuned gains live at 1e-4..1e-3 scale; %.3f showed an
        # I gain of 0.00025 as "0.000". These are the LIVE gains -- the
        # per-mode profiles behind them swap on tracking-mode changes.
        ('pid', "PID Gains (live set)", [
            ('pid_azm_p_gain', "AZM P:", f"{config_state.pid_azm_p_gain:.5f}"),
            ('pid_azm_i_gain', "I:", f"{config_state.pid_azm_i_gain:.5f}"),
            ('pid_azm_d_gain', "D:", f"{config_state.pid_azm_d_gain:.5f}"),
            ('pid_alt_p_gain', "ALT P:", f"{config_state.pid_alt_p_gain:.5f}"),
            ('pid_alt_i_gain', "I:", f"{config_state.pid_alt_i_gain:.5f}"),
            ('pid_alt_d_gain', "D:", f"{config_state.pid_alt_d_gain:.5f}"),
        ]),
    ]

    for key, title, fields in groups:
        box = display.config_group_rects[key]
        pygame.draw.rect(display.menu_screen, (0, 0, 0), box, 2, border_radius=5)
        display.menu_screen.blit(display.font.render(title, True, (0, 0, 0)),
                                 (box.x + 10, box.y + 8))
        for field, label, value in fields:
            _draw_config_field(display, config_state, field, label, value)

    # Feed-forward status row at the bottom of the PID group (editable from
    # the joystick screen's FF buttons; shown here read-only for reference).
    pid_box = display.config_group_rects['pid']
    ff_y = pid_box.bottom - 30
    for i, (ff_label, enabled) in enumerate((
            ("AZM FF:", config_state.feed_forward_azm_enabled),
            ("ALT FF:", config_state.feed_forward_alt_enabled))):
        x = pid_box.x + 12 + i * 130
        display.menu_screen.blit(
            display.small_font.render(ff_label, True, (0, 0, 0)), (x, ff_y))
        display.menu_screen.blit(
            display.small_font.render("ON" if enabled else "OFF", True,
                                      (0, 150, 0) if enabled else (150, 0, 0)),
            (x + 62, ff_y))

    # Footer: divider + Save/Load buttons.
    pygame.draw.line(display.menu_screen, (0, 0, 0),
                     (display.sub_x, display.sub_y + display.sub_height - 60),
                     (display.sub_x + display.sub_width,
                      display.sub_y + display.sub_height - 60), 1)
    draw_button_with_objects(display, "save")
    draw_button_with_objects(display, "load")


def load_config(file_path=None):
    """Create and initialize ConfigState from file."""
    config_state = ConfigState()
    config_state.load_from_file(file_path)
    return config_state


