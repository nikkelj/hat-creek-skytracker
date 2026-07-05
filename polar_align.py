"""
Plate-solve polar alignment for the equatorial (Eq) mount mode.

The fast dusk-time tool for field-rotation-free imaging: instead of a full pointing-model
grid, rotate ONLY the RA (hour-angle / AZM) axis through a few positions, plate-solve each,
and fit the physical RA-axis direction from where the boresight lands. The angle between
that axis and the true celestial pole is the polar-alignment error, reported as the
azimuth/altitude knob corrections the operator dials out -- the same method SharpCap/NINA use.

Why it works: with the Dec axis held fixed and only the RA axis turning, the boresight sweeps
a circle about the physical RA axis. Converting each plate solve (RA/Dec) to horizontal
(az, el) AT ITS SOLVE TIME gives the tube's instantaneous physical direction (sky rotation
doesn't move an idle tube), so the samples lie on that circle and their best-fit plane normal
is the RA-axis direction. No precise knowledge of the rotation amounts is needed.

The routine only MEASURES the error (for the operator to null mechanically); it does not feed
the tracking transform. Sampling is decoupled from hardware through an injected sampler so it
runs headlessly against the sim (sim_pole_*_err_deg) and in the GUI against the real
slew + plate-solve path.
"""

import math
import threading
import time

import numpy as np

from transformations import cartesian_from_az_el, az_el_from_cartesian

IDLE = "idle"
RUNNING = "running"
DONE = "done"
ERROR = "error"

MIN_SAMPLES = 3  # need >=3 non-collinear points to define the circle / its axis


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def fit_polar_axis(samples_azel, toward_az_deg=0.0, toward_alt_deg=45.0):
    """Best-fit RA-axis direction (az, el) from boresight samples on a circle.

    ``samples_azel`` is a list of (az, el) horizontal directions the boresight swept while
    only the RA axis turned. Returns (axis_az_deg, axis_el_deg). The axis is the normal to
    the best-fit plane through the points (SVD), oriented toward (toward_az, toward_alt) so
    it names the pole-side end of the axis rather than its antipode.
    """
    pts = np.array([cartesian_from_az_el(az, el) for az, el in samples_azel])
    if len(pts) < MIN_SAMPLES:
        raise ValueError("need at least 3 samples to fit the polar axis")
    centroid = pts.mean(axis=0)
    # Smallest right-singular vector of the centered points = best-fit plane normal.
    _, _, vt = np.linalg.svd(pts - centroid)
    axis = vt[-1]
    axis = axis / np.linalg.norm(axis)
    if np.dot(axis, cartesian_from_az_el(toward_az_deg, toward_alt_deg)) < 0:
        axis = -axis
    return az_el_from_cartesian(axis)


def polar_error(measured_az, measured_alt, target_az, target_alt):
    """Polar-axis error and the correction to apply.

    Returns a dict: az_error/alt_error (measured minus target, degrees; az wrapped to
    +-180), total_deg (great-circle angle between the measured axis and the true pole), and
    az_move/alt_move (the correction = -error) with human-readable directions. Positive
    az_error means the polar axis points too far EAST; positive alt_error means too HIGH.
    """
    az_err = _wrap180(measured_az - target_az)
    alt_err = measured_alt - target_alt
    a = cartesian_from_az_el(measured_az, measured_alt)
    b = cartesian_from_az_el(target_az, target_alt)
    total = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))
    return {
        "az_error": az_err,
        "alt_error": alt_err,
        "total_deg": total,
        "az_move": -az_err,
        "alt_move": -alt_err,
        "az_dir": "west" if az_err > 0 else "east",
        "alt_dir": "down" if alt_err > 0 else "up",
    }


class PolarAlignState:
    """Mutable state for a polar-alignment run, read by the UI each frame."""

    def __init__(self):
        self.phase = IDLE
        self.samples = []          # (az_obs, el_obs) boresight directions
        self.sample_meta = []      # per-sample dict (azm command, solve quality)
        self.measured_pole = None  # (az, el) of the fitted RA axis
        self.result = None         # polar_error() dict
        self.status = ""
        self.error = None
        self.progress = (0, 0)
        self.current_target = None  # (azm, alt) the runner is slewing to
        self.last_solve = None

    def reset(self):
        self.__init__()


class PolarAlignRunner(threading.Thread):
    """Rotate the RA (AZM) axis through a span, plate-solve each step, fit the axis.

    ``sampler`` exposes ``slew(azm_deg, alt_deg) -> bool`` (drive the mount axes) and
    ``capture() -> (az_obs, el_obs, meta) | None`` (plate-solve the current frame, sky az/el
    at solve time). ``target_az``/``target_alt`` are the true celestial-pole direction
    (north/south azimuth, latitude). The Dec axis is parked at ``dec_deg`` and the RA axis
    stepped across ``ha_span_deg`` (centred on 0) in ``n_points`` steps.
    """

    def __init__(self, state, config_state, sampler, target_az, target_alt,
                 n_points=5, dec_deg=20.0, ha_span_deg=80.0, status_cb=None):
        super().__init__(daemon=True)
        self.s = state
        self.config_state = config_state
        self.sampler = sampler
        self.target_az = float(target_az)
        self.target_alt = float(target_alt)
        self.n_points = max(MIN_SAMPLES, int(n_points))
        self.dec_deg = float(dec_deg)
        self.ha_span_deg = float(ha_span_deg)
        self.status_cb = status_cb
        self._abort = threading.Event()

    def abort(self):
        self._abort.set()

    def _status(self, msg):
        self.s.status = msg
        if self.status_cb:
            self.status_cb(msg)

    def _ha_steps(self):
        half = self.ha_span_deg / 2.0
        if self.n_points == 1:
            return [0.0]
        return [(-half + i * self.ha_span_deg / (self.n_points - 1)) for i in range(self.n_points)]

    def run(self):
        s = self.s
        try:
            s.phase = RUNNING
            steps = self._ha_steps()
            for i, ha in enumerate(steps):
                if self._abort.is_set():
                    s.phase = IDLE
                    self._status("aborted")
                    return
                s.progress = (i, len(steps))
                s.current_target = (ha, self.dec_deg)
                self._status(f"step {i + 1}/{len(steps)}  HA {ha:+.0f}°")
                self.sampler.slew(ha, self.dec_deg)
                smp = self.sampler.capture()
                if smp is not None:
                    s.samples.append((smp[0] % 360.0, smp[1]))
                    s.sample_meta.append(dict(smp[2] or {}, ha=ha))
                    s.last_solve = smp[2]
            s.progress = (len(steps), len(steps))

            if len(s.samples) < MIN_SAMPLES:
                s.phase = ERROR
                s.error = (f"only {len(s.samples)} solves (need {MIN_SAMPLES}); "
                           "check stars/clouds/DB and retry")
                self._status(s.error)
                return

            az_m, alt_m = fit_polar_axis(s.samples, self.target_az, self.target_alt)
            s.measured_pole = (az_m, alt_m)
            s.result = polar_error(az_m, alt_m, self.target_az, self.target_alt)
            s.current_target = None
            s.phase = DONE
            r = s.result
            self._status(
                f"done: polar error {r['total_deg']:.2f}°  "
                f"(az {abs(r['az_error']):.2f}° {r['az_dir']}, "
                f"alt {abs(r['alt_error']):.2f}° {r['alt_dir']})")
        except Exception as e:
            s.current_target = None
            s.phase = ERROR
            s.error = str(e)
            self._status(f"error: {e}")


class PolarGuiSampler:
    """Live sampler: drive the mount AZM/ALT axes directly and plate-solve the result.

    Mirrors alignment.GuiSampler but commands mount axes (HA/Dec) instead of sky az/el, and
    returns the solved sky az/el at solve time (the boresight's instantaneous direction).
    """

    def __init__(self, joystick_state, config_state, solver, ts, ephemeris,
                 cam_index=0, settle_pause=0.4, state=None):
        self.joystick_state = joystick_state
        self.config_state = config_state
        self.solver = solver
        self.ts = ts
        self.ephemeris = ephemeris
        self.cam_index = cam_index
        self.settle_pause = settle_pause
        self.state = state

    def slew(self, azm_deg, alt_deg):
        from alignment import _wrap180 as _w  # reuse the settle helper's wrap
        from lib.auxstar import Targets
        controller = getattr(self.joystick_state, 'telescope_controller', None)
        if controller is None:
            return False
        cfg = self.config_state
        tol = float(getattr(cfg, 'alignment_settle_tol_deg', 0.3) or 0.3)
        cycles = int(getattr(cfg, 'alignment_settle_cycles', 2) or 2)
        kp, max_dps, timeout = 4.0, 5.0, 20.0
        use_rate = hasattr(controller, 'hc_set_rate_dps')
        try:
            controller.hc_goto_fast(Targets.AZM, azm_deg, 0, 0)
            controller.hc_goto_fast(Targets.ALT, alt_deg, 0, 0)
        except Exception:
            pass

        def cmd(target, err):
            if use_rate:
                controller.hc_set_rate_dps(target, max(-max_dps, min(max_dps, kp * err)))

        def stop():
            if use_rate:
                controller.hc_set_rate_dps(Targets.AZM, 0.0)
                controller.hc_set_rate_dps(Targets.ALT, 0.0)

        t0 = time.time(); stable = 0; settled = False
        while time.time() - t0 < timeout:
            e_az = _w(azm_deg - controller.hc_get_position(Targets.AZM) * 360.0)
            e_alt = _w(alt_deg - controller.hc_get_position(Targets.ALT) * 360.0)
            if abs(e_az) < tol and abs(e_alt) < tol:
                stable += 1
                if stable >= cycles:
                    stop(); settled = True; break
            else:
                stable = 0
                cmd(Targets.AZM, e_az); cmd(Targets.ALT, e_alt)
            time.sleep(0.05)
        else:
            stop()
        time.sleep(self.settle_pause)
        # Report the truth: a timed-out settle is not a settle. (Callers may
        # still choose to sample -- capture() pairs the solve with the actual
        # encoder position -- but they can now tell the difference.)
        return settled

    def capture(self):
        import camera_manager
        import plate_solver as ps_mod
        camera = camera_manager.camera_manager.get_camera(self.cam_index)
        raw = (camera.thread.get_latest_raw()
               if camera is not None and getattr(camera, 'thread', None) is not None else None)
        if raw is None:
            return None
        from alignment import frame_skyfield_time
        t = frame_skyfield_time(camera, self.ts)
        result = self.solver.solve(raw)
        if result is None or not result.solved:
            return None
        cfg = self.config_state
        lat = float(cfg.lat_str or 0.0); lon = float(cfg.lon_str or 0.0)
        elev = float(cfg.alt_str or 0.0)
        az_obs, el_obs = ps_mod.solved_azel(result.ra_deg, result.dec_deg, lat, lon, elev,
                                            self.ephemeris, self.ts, t)
        meta = {'result': result, 'n_matches': result.n_matches, 'rmse': result.rmse,
                'fov': result.fov_deg, 't': t}
        if self.state is not None:
            self.state.last_solve = meta
        return (az_obs % 360.0, el_obs, meta)


def make_polar_sampler(joystick_state, config_state, solver, ts, ephemeris,
                       cam_index=0, state=None):
    return PolarGuiSampler(joystick_state, config_state, solver, ts, ephemeris,
                           cam_index, state=state)
