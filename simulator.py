"""
Software hardware simulator: a sim mount and sim cameras that close the tracking
loop without any physical hardware present.

Goals (see plan): let the operator hone the UX and exercise the control/acquisition
algorithms by
  * integrating the same rate commands the control loop issues into a true mount
    pointing, while reporting an *encoder* position with injectable initial
    misalignment and noise (so offset calibration + the closed loop have something
    to fight), and
  * rendering synthetic camera imagery of whatever is being tracked (launch plume,
    or a live-TLE satellite as a dot in the wide cam / a rough Starlink V2 Mini in
    the narrow cam) over a star field that streaks with mount motion, with
    injectable inter-camera rotation + image-plane misalignment.

Design notes:
  * SimMount implements the subset of the NexstarHandController API the code uses,
    so connect_telescope() can hand it back in place of the real controller.
  * Sim frames flow through the REAL camera pipeline: SimCap is a duck-typed stand-in
    for the ASI camera object, so the existing camera_buffer.CameraThread captures,
    buffers, timestamps and surfaces them unchanged.
  * The geometry helper angles_to_pixel() is the exact inverse of
    hotspot.pixel_offset_to_angles(), and the sim uses the SAME sign conventions the
    tracker uses, so the closed loop converges by construction. (Real-hardware sign
    calibration is therefore NOT something the sim can validate.)
"""

import math
import threading
import time

import numpy as np

from lib.auxstar import RATES, Targets
from trajectory import interpolate_position_data_and_rates


def wrap180(deg):
    """Wrap an angle (deg) to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Geometry (pure)
# ---------------------------------------------------------------------------

def angles_to_pixel(d_az_deg, d_el_deg, pixel_size_um, focal_length_mm,
                    rotation_deg=0.0, el_deg=0.0, x_sign=1.0, y_sign=-1.0,
                    apply_cos_el=True):
    """Inverse of hotspot.pixel_offset_to_angles.

    Given a target's angular offset from the camera boresight (d_az, d_el in deg),
    return its pixel offset (dx, dy) from the image center. Uses the same IFOV and
    sign conventions as the tracker so render->detect->command round-trips.
    """
    if focal_length_mm == 0:
        return 0.0, 0.0
    ifov_deg = math.degrees((pixel_size_um * 1e-3) / focal_length_mm)

    cross_el = d_az_deg * math.cos(math.radians(el_deg)) if apply_cos_el else d_az_deg
    el_err = d_el_deg

    th = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    # inverse rotation R(-th)
    ax = cross_el * cos_t + el_err * sin_t
    ay = -cross_el * sin_t + el_err * cos_t

    dx = ax / (ifov_deg * x_sign)
    dy = ay / (ifov_deg * y_sign)
    return dx, dy


def ifov_deg(pixel_size_um, focal_length_mm):
    if focal_length_mm == 0:
        return 0.0
    return math.degrees((pixel_size_um * 1e-3) / focal_length_mm)


# ---------------------------------------------------------------------------
# Rendering (pure numpy, operate in-place on a float HxW frame)
# ---------------------------------------------------------------------------

def render_blob(frame, px, py, amp, sigma):
    """Add a gaussian spot centered at (px, py)."""
    h, w = frame.shape
    r = max(1, int(3 * sigma))
    x0 = max(0, int(px) - r); x1 = min(w, int(px) + r + 1)
    y0 = max(0, int(py) - r); y1 = min(h, int(py) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] += amp * np.exp(
        -(((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma ** 2)))


def render_streak(frame, px, py, vx, vy, amp, sigma):
    """Draw a star as a short streak from (px,py) along (vx,vy)."""
    length = math.hypot(vx, vy)
    n = max(1, int(length))
    for i in range(n + 1):
        t = i / max(1, n)
        render_blob(frame, px + vx * t, py + vy * t, amp, sigma)


def render_v2mini(frame, px, py, amp, rotation_deg=0.0, scale=1.0):
    """Rough Starlink V2 Mini: a bright bus + a long dim solar panel."""
    h, w = frame.shape
    bus_w, bus_h = max(2, int(6 * scale)), max(2, int(3 * scale))
    panel_w, panel_h = max(4, int(28 * scale)), max(1, int(3 * scale))
    th = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)

    def paint(half_w, half_h, level):
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                rx = dx * cos_t - dy * sin_t
                ry = dx * sin_t + dy * cos_t
                ix, iy = int(round(px + rx)), int(round(py + ry))
                if 0 <= ix < w and 0 <= iy < h:
                    frame[iy, ix] = max(frame[iy, ix], level)

    paint(panel_w // 2, panel_h // 2, amp * 0.45)   # solar panel (dimmer)
    paint(bus_w // 2, bus_h // 2, amp)               # bus (bright)


def add_noise(frame, read_noise, rng):
    """Add shot-ish + read noise, in place."""
    if read_noise > 0:
        frame += rng.normal(0.0, read_noise, frame.shape)
    np.clip(frame, 0, None, out=frame)


def injected_boresight(config_state, nominal_az_deg, nominal_el_deg):
    """Where the boresight ACTUALLY lands on the sky (what a plate solve sees) when the
    mount's encoders nominally read sky (az, el), given the sim's injected error sources.

    Models, in order: the repeatable 7-term alt-az pointing model
    (``sim_config['mount_pointing_model']``) via PointingModel.predict_observed, then
    atmospheric refraction (``sim_config['sim_refraction']``) which lifts the apparent
    elevation. A no-op (returns the input) when neither is configured. This is the inverse
    problem the alignment routine solves -- injecting it here lets a sim alignment run
    recover all seven terms end-to-end (not just the encoder-bias IA/IE)."""
    s = getattr(config_state, 'sim_config', {}) or {}
    az, el = float(nominal_az_deg), float(nominal_el_deg)
    terms = s.get('mount_pointing_model') or None
    if terms and any(abs(float(v)) > 0.0 for v in terms.values()):
        from pointing_model import PointingModel
        az, el = PointingModel(terms).predict_observed(az, el)
    if s.get('sim_refraction', False):
        from pointing_model import bennett_refraction_deg
        p = float(s.get('sim_refraction_pressure_mbar', 1010.0))
        t = float(s.get('sim_refraction_temperature_c', 10.0))
        el = el + bennett_refraction_deg(el, p, t)  # apparent elevation is higher than geometric
    return az % 360.0, el


# ---------------------------------------------------------------------------
# Sim mount
# ---------------------------------------------------------------------------

class SimMount:
    """Drop-in for NexstarHandController in simulation.

    Integrates rate commands into a true pointing and reports an encoder position
    (true + misalignment + noise) as a fraction of a revolution, matching the real
    controller's unpack_int3 convention.
    """

    def __init__(self, config_state, az0_deg=0.0, el0_deg=0.0, rng=None):
        self.config_state = config_state
        self.az_true_deg = az0_deg
        self.el_true_deg = el0_deg
        self._az_rate_dps = 0.0   # signed deg/sec
        self._el_rate_dps = 0.0
        # Focus motor: a third rate-commanded axis on the same encoder/rate
        # convention as az/el. It carries no misalignment/backlash/periodic
        # error (those model the two mechanical mount axes), just a clean
        # integrator so the focus read-back tracks the trigger commands.
        self.focus_true_deg = 0.0
        self._focus_rate_dps = 0.0
        self._last_t = time.time()
        self._t0 = self._last_t
        self._lock = threading.RLock()
        self._rng = rng if rng is not None else np.random.default_rng()
        # Gear backlash state: input position within the lost-motion band per axis.
        self._az_play = 0.0
        self._el_play = 0.0
        # Periodic-error state (lazily seeded on the first advance).
        self._pe_prev_az = 0.0
        self._pe_prev_el = 0.0
        self._pe_init = False

    # --- sim params (read live from config so UI edits take effect) ---
    def _sim(self):
        return getattr(self.config_state, 'sim_config', {}) or {}

    @property
    def az_rate_dps(self):
        return self._az_rate_dps

    @property
    def el_rate_dps(self):
        return self._el_rate_dps

    def total_rate_dps(self):
        return math.hypot(self._az_rate_dps, self._el_rate_dps)

    def _advance(self):
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            return
        s = self._sim()
        jitter = float(s.get('mount_rate_noise_dps', 0.0))
        az_rate = self._az_rate_dps + (self._rng.normal(0, jitter) if jitter > 0 else 0.0)
        el_rate = self._el_rate_dps + (self._rng.normal(0, jitter) if jitter > 0 else 0.0)
        # Gear backlash: on a direction reversal the load (optical axis) stalls
        # until the lost motion is taken up. This is what makes discrete
        # rate-stepping hunt -- every time the rate command flips sign the mount
        # sits in a dead-zone before it starts moving back.
        b = float(s.get('mount_backlash_deg', 0.0))
        d_az = self._apply_backlash('_az_play', az_rate * dt, b)
        d_el = self._apply_backlash('_el_play', el_rate * dt, b)
        # Periodic (worm) error: a slow sinusoid riding on the true pointing.
        # Inject the per-step delta so it adds to tracking instead of replacing it.
        pe_az, pe_el = self._periodic_error(now)
        if not self._pe_init:
            self._pe_prev_az, self._pe_prev_el = pe_az, pe_el
            self._pe_init = True
        d_az += pe_az - self._pe_prev_az
        d_el += pe_el - self._pe_prev_el
        self._pe_prev_az, self._pe_prev_el = pe_az, pe_el
        self.az_true_deg = (self.az_true_deg + d_az) % 360.0
        self.el_true_deg = self.el_true_deg + d_el
        # Focus integrates straight from its commanded rate (no backlash/PE).
        self.focus_true_deg = (self.focus_true_deg + self._focus_rate_dps * dt) % 360.0

    def _apply_backlash(self, attr, delta, band):
        """Pass a commanded delta through a backlash dead-zone of total width
        `band` (deg). Returns the load (output) delta. The state held in `attr`
        is the input position within the [-band/2, band/2] gap; the load only
        moves once the input is pushed against an edge of the gap."""
        if band <= 0.0:
            return delta
        play = getattr(self, attr) + delta
        half = band / 2.0
        if play > half:
            out = play - half
            play = half
        elif play < -half:
            out = play + half
            play = -half
        else:
            out = 0.0
        setattr(self, attr, play)
        return out

    def _periodic_error(self, now):
        """Worm periodic error as a sinusoid (zero-to-peak amplitude, worm
        period). The two axes run a quarter cycle apart so they don't move
        identically. Returns (pe_az_deg, pe_el_deg)."""
        s = self._sim()
        amp = float(s.get('mount_pe_amplitude_deg', 0.0))
        if amp <= 0.0:
            return 0.0, 0.0
        period = float(s.get('mount_pe_period_sec', 600.0)) or 600.0
        ph = 2.0 * math.pi * (now - self._t0) / period
        return amp * math.sin(ph), amp * math.sin(ph + math.pi / 2.0)

    def _encoder(self, true_deg, mis_deg):
        noise = float(self._sim().get('mount_encoder_noise_deg', 0.0))
        n = self._rng.uniform(-noise, noise) if noise > 0 else 0.0
        return true_deg + mis_deg + n

    # --- NexstarHandController API subset ---
    def hc_get_position(self, target):
        with self._lock:
            self._advance()
            s = self._sim()
            if target == Targets.ALT:
                val = self._encoder(self.el_true_deg, float(s.get('mount_misalignment_el_deg', 0.0)))
            elif target == Targets.FOCUS:
                val = self.focus_true_deg
            else:
                val = self._encoder(self.az_true_deg, float(s.get('mount_misalignment_az_deg', 0.0)))
            return (val % 360.0) / 360.0

    def hc_slew_fixed(self, target, rate):
        with self._lock:
            self._advance()
            sign = 1.0 if rate >= 0 else -1.0
            dps = sign * RATES.get(abs(int(rate)), 0.0) * 360.0
            if target == Targets.ALT:
                self._el_rate_dps = dps
            elif target == Targets.FOCUS:
                self._focus_rate_dps = dps
            else:
                self._az_rate_dps = dps
            return True

    def hc_goto_fast(self, target, dd, mm, ss):
        with self._lock:
            self._advance()
            deg = abs(dd) + mm / 60.0 + ss / 3600.0
            deg = deg if dd >= 0 else -deg
            if target == Targets.ALT:
                self.el_true_deg = deg
                self._el_rate_dps = 0.0
            elif target == Targets.FOCUS:
                self.focus_true_deg = deg % 360.0
                self._focus_rate_dps = 0.0
            else:
                self.az_true_deg = deg % 360.0
                self._az_rate_dps = 0.0
            return True

    def hc_set_position(self, target, dd, mm, ss):
        return self.hc_goto_fast(target, dd, mm, ss)

    def hc_set_guide_rate(self, target, rate, **kwargs):
        return True

    # Fine continuous-rate primitive (MC_SET_POS/NEG_GUIDERATE). Sets the axis
    # rate directly in deg/s, quantized to the real 24-bit guide-rate LSB so the
    # sim faithfully shows "effectively continuous" tracking (vs the 10-step
    # MC_MOVE used by hc_slew_fixed).
    _GUIDE_LSB_DPS = 360.0 / 2 ** 24  # ~0.077 arcsec/s

    def hc_set_rate_dps(self, target, dps):
        with self._lock:
            self._advance()
            q = round(dps / self._GUIDE_LSB_DPS) * self._GUIDE_LSB_DPS
            if target == Targets.ALT:
                self._el_rate_dps = q
            else:
                self._az_rate_dps = q
            return True

    def close(self):
        with self._lock:
            self._az_rate_dps = 0.0
            self._el_rate_dps = 0.0


# ---------------------------------------------------------------------------
# Simulator hub + target resolution + frame assembly
# ---------------------------------------------------------------------------

class HardwareSimulator:
    """Ties the sim mount, config, tracking state and time together, and renders
    camera frames on demand."""

    def __init__(self, config_state, tracking_vis_state, ts, star_catalog=None):
        self.config_state = config_state
        self.tracking_vis_state = tracking_vis_state
        self.ts = ts
        self._rng = np.random.default_rng(int(self._sim().get('seed', 1234)))
        self.mount = SimMount(config_state, rng=self._rng)
        self._stars = None
        self._stars_seed = None
        # Shared Hipparcos catalogue (lazy). False = tried and failed; None = untried.
        self._star_catalog = star_catalog

    def _sim(self):
        return getattr(self.config_state, 'sim_config', {}) or {}

    def sim_enabled(self):
        return bool(self._sim().get('enabled', False))

    # --- star field (fixed in sky, generated once per seed) ---
    def _ensure_stars(self):
        s = self._sim()
        seed = int(s.get('seed', 1234))
        density = float(s.get('star_density', 300))
        if self._stars is not None and self._stars_seed == seed:
            return
        rng = np.random.default_rng(seed + 7)
        n = max(0, int(density))
        az = rng.uniform(0, 360, n)
        el = rng.uniform(0, 90, n)
        mag = rng.uniform(0.2, 1.0, n)  # relative brightness factor
        self._stars = (az, el, mag)
        self._stars_seed = seed

    # --- real star catalogue (Hipparcos) ---
    def _get_star_catalog(self):
        """Lazily resolve the shared StarCatalog; returns None if unavailable."""
        if self._star_catalog is None:
            try:
                from star_catalog import get_catalog
                eph = getattr(self.tracking_vis_state, 'ephemeris', None)
                self._star_catalog = get_catalog(ephemeris=eph)
            except Exception as e:
                print(f"Sim star catalog unavailable: {e}")
                self._star_catalog = False  # sentinel: tried and failed
        return self._star_catalog or None

    def _catalog_fov_stars(self, cam_az, cam_el, fov_x, fov_y, brightness):
        """Real catalogue stars in the camera FOV as (az, el, amp) arrays, or None.

        Prefers the deep Tycho catalogue (needed for the narrow real-camera FOVs that
        Hipparcos can't populate), falling back to Hipparcos. amp is scaled so the
        faintest rendered star clears the read noise (so it's plate-solvable) and the
        brightest saturate rather than blow up.
        """
        if self.ts is None:
            return None
        cfg = self.config_state
        s = self._sim()
        lat = float(cfg.lat_str or 0.0)
        lon = float(cfg.lon_str or 0.0)
        elev = float(cfg.alt_str or 0.0)
        t_tt = self.ts.now().tt
        eph = getattr(self.tracking_vis_state, 'ephemeris', None)

        # Deep catalogue first (Tycho, ~mag 10) for narrow FOVs; else Hipparcos.
        # sim_use_deep_catalog lets callers force the Hipparcos path (deterministic,
        # and what the wide-FOV demo DBs are built from).
        limit = float(s.get('sim_star_limit_mag', 10.0))
        az = el = mag = None
        deep = None
        if s.get('sim_use_deep_catalog', True):
            try:
                from star_catalog import get_deep_catalog
                deep = get_deep_catalog(ephemeris=eph)
            except Exception:
                deep = None
        if deep is not None:
            az, el, mag = deep.stars_in_fov(lat, lon, elev, self.ts, t_tt,
                                            cam_az, cam_el, fov_x, fov_y, limiting_mag=limit)
        else:
            cat = self._get_star_catalog()
            if cat is None:
                return None
            limit = min(limit, float(getattr(cfg, 'star_limiting_magnitude', 6.5)) + 1.0)
            az, el, mag = cat.stars_in_fov(lat, lon, elev, self.ts, t_tt,
                                           cam_az, cam_el, fov_x, fov_y, limiting_mag=limit)
        if az is None or len(az) == 0:
            return (np.empty(0), np.empty(0), np.empty(0))

        # Detectability floor: the faintest star (mag == limit) renders at a few read
        # noise sigma above background so plate solving has real sources to centroid.
        read_noise = float(s.get('read_noise', 2.0))
        floor_amp = max(10.0, 5.0 * read_noise)  # faintest star ~5 sigma -> centroidable
        # Cap brightest stars BELOW the tracked target so the satellite/plume stays the
        # most prominent object (so it's pickable by eye and by the HOTSPOT tracker).
        star_cap = max(floor_amp + 1.0, 0.85 * brightness)
        amp = np.clip(floor_amp * np.power(10.0, -0.4 * (mag - limit)), 0.0, star_cap)
        return az, el, amp

    # --- current target ---
    def current_target_azel(self):
        """Return (az_deg, el_deg, kind, visible). kind in {'plume','satellite',None}."""
        tvs = self.tracking_vis_state
        if tvs is None:
            return 0.0, 0.0, None, False
        current_tt = self.ts.now().tt if self.ts is not None else 0.0

        # Launch takes priority when launched (mirrors program_track).
        if (getattr(tvs, 'selected_launch', None) and getattr(tvs, 'launch_launched', False)
                and getattr(tvs, 'launch_trajectories', None)
                and tvs.selected_launch in tvs.launch_trajectories):
            px, py, alt, dist, az, azr, elr = interpolate_position_data_and_rates(
                tvs.launch_trajectories[tvs.selected_launch], current_tt,
                getattr(tvs, 'launch_start_time', 0), True)
            if az is not None and alt is not None and alt > 0:
                return az, alt, 'plume', True
            return 0.0, 0.0, None, False

        sel = getattr(tvs, 'selected_satellite', None)
        if sel and getattr(tvs, 'satellite_trajectories', None) and sel in tvs.satellite_trajectories:
            px, py, alt, dist, az, azr, elr = interpolate_position_data_and_rates(
                tvs.satellite_trajectories[sel], current_tt)
            if az is not None and alt is not None and alt > 0:
                return az, alt, 'satellite', True
        return 0.0, 0.0, None, False

    # --- render a full sensor frame for a camera (mono uint8) ---
    def render_frame(self, cam_index):
        cfg = self.config_state
        s = self._sim()
        w = int(s.get('cam_width', 960))
        h = int(s.get('cam_height', 720))
        frame = np.zeros((h, w), dtype=np.float32)
        frame += float(s.get('background_level', 6.0))

        cam_name = f"camera{cam_index + 1}"
        pix = float(cfg.get_camera_pixel_size(cam_name))
        foc = float(cfg.get_camera_focal_length(cam_name))
        rot = float(cfg.get_camera_alignment_rotation(cam_name))
        off_x = off_y = 0.0
        if cam_index == 1:  # narrow cam carries the inter-camera misalignment
            rot += float(s.get('cam2_offset_rotation_deg', 0.0))
            off_x = float(s.get('cam2_offset_x_px', 0.0))
            off_y = float(s.get('cam2_offset_y_px', 0.0))

        x_sign = float(getattr(cfg, 'hotspot_x_sign', 1.0))
        y_sign = float(getattr(cfg, 'hotspot_y_sign', -1.0))

        # The camera is bolted to the mount, so what it sees on the sky is the
        # mount's mechanical pointing run through the SAME forward transform the
        # rest of the app uses (mount AZM/ALT -> sky az/el). Earlier this used the
        # mount angles directly (identity); that only looked right while the sky->
        # mount inverse was also wrong, and broke PROGRAM-track rendering once that
        # was fixed (boresight landed at 90 - el).
        mount_mode = getattr(cfg, 'mount_mode', 'Eq')
        align_az = float(getattr(cfg, 'alignment_azimuth_str', 0.0) or 0.0)
        align_el = float(getattr(cfg, 'alignment_elevation_str', 0.0) or 0.0)
        if mount_mode == 'AltAz':
            from transformations import AzAlt2AzEl_AltAz
            cam_az, cam_el = AzAlt2AzEl_AltAz(
                self.mount.az_true_deg, self.mount.el_true_deg, align_az)
            # ALT axis runs opposite sky elevation (el = 90 - ALT): the sky el rate
            # is the negated mount rate; azimuth tracks the mount directly.
            sky_az_rate, sky_el_rate = self.mount.az_rate_dps, -self.mount.el_rate_dps
        elif mount_mode == 'Passthrough':
            cam_az, cam_el = self.mount.az_true_deg, self.mount.el_true_deg
            sky_az_rate, sky_el_rate = self.mount.az_rate_dps, self.mount.el_rate_dps
        else:
            # Equatorial: mount AZM=hour angle, ALT=declination, rotating about the TRUE
            # polar axis. The sim's true pole can be mis-set (sim_pole_az/alt_err) so a
            # polar-alignment run has a real error to measure; it defaults to the assumed
            # pole (north, latitude) -> perfect alignment.
            from transformations import eq_mount_to_azel
            lat = float(getattr(cfg, 'lat_str', 0.0) or 0.0)
            true_pole_az = align_az + float(s.get('sim_pole_az_err_deg', 0.0))
            base_alt = align_el if align_el != 0.0 else lat
            true_pole_alt = base_alt + float(s.get('sim_pole_alt_err_deg', 0.0))
            # Residual mount errors (cone/index/flexure) on top of the polar misalignment:
            # distort the mount HA/Dec by the injected equatorial model before the pole geometry.
            h_m, d_m = self.mount.az_true_deg, self.mount.el_true_deg
            eqterms = s.get('mount_eq_pointing_model') or None
            if eqterms and any(abs(float(v)) > 0.0 for v in eqterms.values()):
                from eq_pointing_model import EquatorialPointingModel
                h_m, d_m = EquatorialPointingModel(eqterms, lat_deg=lat).predict_observed(h_m, d_m)
            cam_az, cam_el = eq_mount_to_azel(h_m, d_m, true_pole_az, true_pole_alt)
            sky_az_rate, sky_el_rate = self.mount.az_rate_dps, self.mount.el_rate_dps

        # Inject the repeatable pointing-model + refraction error (alt-az only): the boresight
        # actually lands where predict_observed() says, so the whole rendered field shifts by
        # the pointing error and an alignment run can recover the seven terms end-to-end.
        if mount_mode == 'AltAz':
            cam_az, cam_el = injected_boresight(cfg, cam_az, cam_el)

        cx, cy = w / 2.0 + off_x, h / 2.0 + off_y

        brightness = float(s.get('target_brightness', 200.0))
        exposure_s = max(1e-4, float(cfg.get_camera_exposure(cam_name)) / 1e6)

        # Star field (streaked by mount motion during the exposure).
        ifd = ifov_deg(pix, foc)
        fov_x = w * ifd / 2.0   # half-FOV (deg) from boresight to edge
        fov_y = h * ifd / 2.0

        # Star field. Isolated in try/except so a catalogue hiccup can NEVER stop the
        # target (satellite/plume) from rendering -- get_data_after_exposure has no
        # error handling, so a throw here would otherwise blank the whole frame.
        try:
            star_data = None
            if bool(s.get('sim_use_real_stars', False)):
                star_data = self._catalog_fov_stars(cam_az, cam_el, fov_x, fov_y, brightness)
            if star_data is not None:
                saz, sel, samp = star_data
            else:
                self._ensure_stars()
                saz, sel, smag = self._stars
                samp = brightness * 0.6 * np.asarray(smag)

            # angular motion during exposure -> pixel streak vector
            sdx, sdy = angles_to_pixel(sky_az_rate * exposure_s,
                                       sky_el_rate * exposure_s,
                                       pix, foc, rot, cam_el, x_sign, y_sign)
            # Cheap angular pre-reject, then the exact pixel-bounds test decides
            # visibility. The az axis compresses by cos(el) into pixels (see
            # angles_to_pixel), so a raw |d_az| > fov_x gate would carve a phantom
            # vertical wall at column (w/2)*cos(el) -- stars wink out well inside the
            # frame, badly so near the zenith. Compare cross-elevation instead, with a
            # generous diagonal radius that is also safe under image rotation.
            cos_el = max(math.cos(math.radians(cam_el)), 1e-3)
            gate = math.hypot(fov_x, fov_y) + 0.5
            for i in range(len(saz)):
                d_az = wrap180(saz[i] - cam_az)
                d_el = sel[i] - cam_el
                if abs(d_az) * cos_el > gate or abs(d_el) > gate:
                    continue
                dx, dy = angles_to_pixel(d_az, d_el, pix, foc, rot, cam_el, x_sign, y_sign)
                sx, sy = cx + dx, cy + dy
                if -10 <= sx < w + 10 and -10 <= sy < h + 10:
                    render_streak(frame, sx, sy, -sdx, -sdy,
                                  amp=float(samp[i]), sigma=1.0)
        except Exception as e:
            print(f"Sim star-field render skipped: {e}")

        # Target
        az, el, kind, visible = self.current_target_azel()
        if s.get('sim_debug_target', False) and cam_index == 0:
            _now = time.time()
            if _now - getattr(self, '_dbg_last', 0.0) > 1.0:
                self._dbg_last = _now
                if visible:
                    _daz = wrap180(az - cam_az)
                    _del = el - cam_el
                    _in = abs(_daz) <= fov_x and abs(_del) <= fov_y
                    print(f"[simdbg] sat=({az:.3f},{el:.3f}) {kind} | boresight=({cam_az:.3f},{cam_el:.3f}) "
                          f"true_mount=({self.mount.az_true_deg:.3f},{self.mount.el_true_deg:.3f}) | "
                          f"dAz={_daz:+.3f} dEl={_del:+.3f} fov=+-({fov_x:.3f},{fov_y:.3f}) inFOV={_in}", flush=True)
                else:
                    tvs = self.tracking_vis_state
                    print(f"[simdbg] current_target_azel -> NOT VISIBLE | selected="
                          f"{getattr(tvs, 'selected_satellite', None)} "
                          f"in_trajectories={getattr(tvs, 'selected_satellite', None) in (getattr(tvs, 'satellite_trajectories', {}) or {})}",
                          flush=True)
        if visible:
            d_az = wrap180(az - cam_az)
            d_el = el - cam_el
            # Cross-el corrected gate (same cos(el) compression as the star field).
            cos_el = max(math.cos(math.radians(cam_el)), 1e-3)
            if abs(d_az) * cos_el <= fov_x and abs(d_el) <= fov_y:
                dx, dy = angles_to_pixel(d_az, d_el, pix, foc, rot, cam_el, x_sign, y_sign)
                tx, ty = cx + dx, cy + dy
                if kind == 'plume':
                    render_blob(frame, tx, ty, amp=brightness * 1.5, sigma=max(2.0, 4.0 / max(ifd, 1e-6) * 0.0005))
                elif kind == 'satellite':
                    if cam_index == 0:   # wide / guide cam -> small dot
                        render_blob(frame, tx, ty, amp=brightness, sigma=1.5)
                    else:                # narrow cam -> rough V2 Mini sprite
                        render_v2mini(frame, tx, ty, amp=brightness, rotation_deg=rot, scale=1.5)

        add_noise(frame, float(s.get('read_noise', 2.0)), self._rng)
        return np.clip(frame, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Duck-typed ASI camera so the real CameraThread pipeline renders sim frames
# ---------------------------------------------------------------------------

class SimCap:
    """Minimal stand-in for a zwoasi Camera object backed by the simulator.

    camera_manager.connect_camera assigns this as camera.cap and starts the normal
    CameraThread, which captures/buffers/surfaces these frames unchanged.
    """

    def __init__(self, cam_index, simulator):
        self.cam_index = cam_index
        self.simulator = simulator
        s = simulator._sim()
        self._w = int(s.get('cam_width', 960))
        self._h = int(s.get('cam_height', 720))
        self._roi = [self._w, self._h, 1, None]  # img_type filled lazily

    def get_camera_property(self):
        return {'IsColorCam': False, 'MaxWidth': self._w, 'MaxHeight': self._h,
                'Name': f'SimCam{self.cam_index + 1}'}

    def set_image_type(self, img_type):
        self._roi[3] = img_type

    def set_control_value(self, *a, **k):
        return None

    def get_roi_format(self):
        import zwoasi as asi
        if self._roi[3] is None:
            self._roi[3] = asi.ASI_IMG_RAW8
        return [self._w, self._h, self._roi[2], self._roi[3]]

    def set_roi_format(self, w, h, binning, img_type):
        self._w, self._h, self._roi = w, h, [w, h, binning, img_type]

    def set_roi_start_position(self, *a, **k):
        return None

    def start_exposure(self, *a, **k):
        return None

    def get_exposure_status(self):
        import zwoasi as asi
        return asi.ASI_EXP_SUCCESS

    def get_data_after_exposure(self, *a, **k):
        # Render defensively: the capture loop (camera_buffer) swallows exceptions and
        # returns None on error, which silently FREEZES the feed on the last frame. So
        # any render failure here is logged (so the real cause is visible) and we still
        # return a valid background frame, keeping the feed alive.
        try:
            frame = self.simulator.render_frame(self.cam_index)
        except Exception as e:
            import traceback
            print(f"SimCap render_frame error (cam {self.cam_index}): {e}")
            traceback.print_exc()
            s = self.simulator._sim()
            bg = float(s.get('background_level', 6.0))
            frame = np.full((self._h, self._w), bg, dtype=np.uint8)
        # honor a non-full ROI by cropping the center (boresight stays centered)
        if frame.shape != (self._h, self._w):
            fh, fw = frame.shape
            y0 = max(0, (fh - self._h) // 2)
            x0 = max(0, (fw - self._w) // 2)
            frame = frame[y0:y0 + self._h, x0:x0 + self._w]
        return np.ascontiguousarray(frame).tobytes()

    def close(self):
        return None
