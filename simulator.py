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
        self._last_t = time.time()
        self._lock = threading.RLock()
        self._rng = rng if rng is not None else np.random.default_rng()

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
        jitter = float(self._sim().get('mount_rate_noise_dps', 0.0))
        az_rate = self._az_rate_dps + (self._rng.normal(0, jitter) if jitter > 0 else 0.0)
        el_rate = self._el_rate_dps + (self._rng.normal(0, jitter) if jitter > 0 else 0.0)
        self.az_true_deg = (self.az_true_deg + az_rate * dt) % 360.0
        self.el_true_deg = self.el_true_deg + el_rate * dt

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
            else:
                self.az_true_deg = deg % 360.0
                self._az_rate_dps = 0.0
            return True

    def hc_set_position(self, target, dd, mm, ss):
        return self.hc_goto_fast(target, dd, mm, ss)

    def hc_set_guide_rate(self, target, rate, **kwargs):
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

    def __init__(self, config_state, tracking_vis_state, ts):
        self.config_state = config_state
        self.tracking_vis_state = tracking_vis_state
        self.ts = ts
        self._rng = np.random.default_rng(int(self._sim().get('seed', 1234)))
        self.mount = SimMount(config_state, rng=self._rng)
        self._stars = None
        self._stars_seed = None

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

        cam_az = self.mount.az_true_deg
        cam_el = self.mount.el_true_deg
        cx, cy = w / 2.0 + off_x, h / 2.0 + off_y

        brightness = float(s.get('target_brightness', 200.0))
        exposure_s = max(1e-4, float(cfg.get_camera_exposure(cam_name)) / 1e6)

        # Star field (streaked by mount motion during the exposure).
        self._ensure_stars()
        saz, sel, smag = self._stars
        ifd = ifov_deg(pix, foc)
        fov_x = w * ifd / 2.0   # half-FOV (deg) from boresight to edge
        fov_y = h * ifd / 2.0
        # angular motion during exposure -> pixel streak vector
        sdx, sdy = angles_to_pixel(self.mount.az_rate_dps * exposure_s,
                                   self.mount.el_rate_dps * exposure_s,
                                   pix, foc, rot, cam_el, x_sign, y_sign)
        for i in range(len(saz)):
            d_az = wrap180(saz[i] - cam_az)
            d_el = sel[i] - cam_el
            if abs(d_az) > fov_x or abs(d_el) > fov_y:
                continue
            dx, dy = angles_to_pixel(d_az, d_el, pix, foc, rot, cam_el, x_sign, y_sign)
            sx, sy = cx + dx, cy + dy
            if -10 <= sx < w + 10 and -10 <= sy < h + 10:
                render_streak(frame, sx, sy, -sdx, -sdy,
                              amp=brightness * 0.6 * smag[i], sigma=1.0)

        # Target
        az, el, kind, visible = self.current_target_azel()
        if visible:
            d_az = wrap180(az - cam_az)
            d_el = el - cam_el
            if abs(d_az) <= fov_x and abs(d_el) <= fov_y:
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
        frame = self.simulator.render_frame(self.cam_index)
        # honor a non-full ROI by cropping the center (boresight stays centered)
        if frame.shape != (self._h, self._w):
            fh, fw = frame.shape
            y0 = max(0, (fh - self._h) // 2)
            x0 = max(0, (fw - self._w) // 2)
            frame = frame[y0:y0 + self._h, x0:x0 + self._w]
        return frame.tobytes()

    def close(self):
        return None
