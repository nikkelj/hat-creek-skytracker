"""
Automated pointing-model alignment.

Drives the mount to a spread of sky points, plate-solves each, fits the 7-term alt-az
pointing model (pointing_model.py), then backtests it on held-out points -- a richer,
in-software replacement for an external StarSense alignment.

The orchestration is decoupled from hardware through an injected ``sample_fn(az, el) ->
(az_obs, el_obs) | None`` so it can be exercised headlessly with a synthetic mount and in
the GUI with the real slew + capture + plate-solve path (make_default_sample_fn).

AltAz mount mode only -- the Eq/Passthrough paths are not modelled here.
"""

import math
import threading
import time

from pointing_model import PointingModel, fibonacci_sky_grid

# Phases of the alignment run (surfaced to the UI).
IDLE = "idle"
RUNNING = "running"
BACKTEST = "backtest"
DONE = "done"
ERROR = "error"

MIN_SAMPLES = 6  # need at least this many good solves to fit a 7-term model


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _spiral_offsets(n):
    """First ``n`` integer (col, row) grid offsets in an outward square spiral, nearest
    first, excluding the origin. Used to order the grid search so it tries the cells
    closest to the failed point before reaching further out."""
    out = []
    r = 1
    while len(out) < n:
        x, y = -r, -r
        for _ in range(2 * r):  # bottom edge, left -> right
            out.append((x, y)); x += 1
        for _ in range(2 * r):  # right edge, bottom -> top
            out.append((x, y)); y += 1
        for _ in range(2 * r):  # top edge, right -> left
            out.append((x, y)); x -= 1
        for _ in range(2 * r):  # left edge, top -> bottom
            out.append((x, y)); y -= 1
        r += 1
    return out[:n]


class AlignmentState:
    """Mutable state for an alignment run, read by the UI each frame."""

    def __init__(self):
        self.phase = IDLE
        self.grid = []                 # all (az, el) sample points
        self.fit_points = []           # points used for the fit
        self.holdout_points = []       # points held out for backtest
        self.samples = []              # (az_cmd, el_cmd, az_obs, el_obs) feeding the fit
        self.sample_meta = []          # per-sample dict (matches/rmse/fov/kind), aligned with samples
        self.backtest_samples = []     # held-out (az_cmd, el_cmd, az_obs, el_obs)
        self.failed_points = []        # (az, el) points that never produced a solve (retryable)
        self.model = None              # fitted PointingModel
        self.stats = None              # fit stats dict
        self.backtest_rms_deg = None   # held-out sky RMS
        self.progress = (0, 0)         # (done, total) sample points
        self.status = ""
        self.error = None
        self.current_target = None     # (az, el) the runner is slewing to right now (UI marker)
        self.current_mount_azel = None # (az, el) the mount/camera FOV is pointing right now
        self.requested_points = 18     # UI-selected sample count (preserved across resets)
        self.runner = None             # active AlignmentRunner, if any
        # --- supervised-run additions ---
        self.last_solve = None         # most recent solve dict (overlay + tables + caption)
        self.paused = False            # runner is parked awaiting manual intervention
        self.manual_active = False     # operator is jogging this point by hand
        self.manual_target = None      # (az, el) nominal point being recovered manually
        self.pause_on_fail = True      # pause for manual jog when auto+grid-search fail
        self.view_mode = 'camera'      # right-panel view: 'camera' | 'tables'

    def reset(self):
        """Clear results for a fresh run, preserving UI selections."""
        npts = getattr(self, 'requested_points', 18)
        pause = getattr(self, 'pause_on_fail', True)
        view = getattr(self, 'view_mode', 'camera')
        self.__init__()
        self.requested_points = npts
        self.pause_on_fail = pause
        self.view_mode = view


def slew_to_azel(controller, config_state, az_deg, el_deg, timeout=20.0,
                 tol_deg=0.02, settle_cycles=3, kp=4.0, max_dps=5.0, position_cb=None):
    """Slew the mount so the boresight points at sky (az, el) and wait for it to settle.

    Coarse goto for the bulk move, then a proportional continuous-rate (guide-rate)
    settle onto the target. Using hc_set_rate_dps (not the coarse discrete MC_MOVE
    table, whose mid rates are far too slow for fine settling) converges smoothly
    without overshoot. The closed-loop settle is what makes the encoder bias show up as
    a real pointing error in the sim, and matches real hardware where goto alone isn't
    accurate. Uses the BASE transform (no pointing model) to measure raw mount
    behaviour. Returns True if it settled within tolerance.
    """
    from transformations import AzEl2AzAlt_AltAz, AzAlt2AzEl_AltAz
    from lib.auxstar import Targets

    align_az = float(getattr(config_state, 'alignment_azimuth_str', 0.0) or 0.0)
    align_el = float(getattr(config_state, 'alignment_elevation_str', 0.0) or 0.0)
    azm_t, alt_t = AzEl2AzAlt_AltAz(az_deg, el_deg, align_az, align_el)

    # Coarse jump to get within the settle window quickly.
    try:
        controller.hc_goto_fast(Targets.AZM, azm_t, 0, 0)
        controller.hc_goto_fast(Targets.ALT, alt_t, 0, 0)
    except Exception:
        pass

    use_rate = hasattr(controller, 'hc_set_rate_dps')

    def _clamp(v):
        return max(-max_dps, min(max_dps, v))

    def _command(target, err):
        if use_rate:
            controller.hc_set_rate_dps(target, _clamp(kp * err))
        else:
            # Fallback: fastest discrete rate scaled to the error (coarse).
            r = 9 if abs(err) > 2 else 7 if abs(err) > 0.5 else 5 if abs(err) > 0.1 else 3
            controller.hc_slew_fixed(target, r * (1 if err > 0 else -1))

    def _stop():
        if use_rate:
            controller.hc_set_rate_dps(Targets.AZM, 0.0)
            controller.hc_set_rate_dps(Targets.ALT, 0.0)
        else:
            controller.hc_slew_fixed(Targets.AZM, 0)
            controller.hc_slew_fixed(Targets.ALT, 0)

    t0 = time.time()
    stable = 0
    e_az = e_alt = 999.0
    while time.time() - t0 < timeout:
        cur_azm = controller.hc_get_position(Targets.AZM) * 360.0
        cur_alt = controller.hc_get_position(Targets.ALT) * 360.0
        if position_cb is not None:
            try:
                position_cb(*AzAlt2AzEl_AltAz(cur_azm, cur_alt, align_az))
            except Exception:
                pass
        e_az = _wrap180(azm_t - cur_azm)
        e_alt = alt_t - cur_alt
        if abs(e_az) < tol_deg and abs(e_alt) < tol_deg:
            stable += 1
            if stable >= settle_cycles:
                _stop()
                return True
        else:
            stable = 0
            _command(Targets.AZM, e_az)
            _command(Targets.ALT, e_alt)
        time.sleep(0.05)

    _stop()
    return abs(e_az) < tol_deg * 5 and abs(e_alt) < tol_deg * 5


class GuiSampler:
    """Live slew + capture + plate-solve sampler for the GUI alignment run.

    Separates ``slew`` from ``capture`` so the runner can drive a grid search around a
    point and a manual-jog recovery without re-implementing the camera/solve path. A
    *sample* is ``(az_cmd, el_cmd, az_obs, el_obs, meta)`` where the commanded sky point
    (``az_cmd, el_cmd``) is derived from the mount **encoders at solve time** (base
    transform inverse), so it is correct for nominal points, grid-search offsets, and
    hand-jogged points alike. ``meta`` carries the SolveResult + source shape for the
    UI overlay/tables. Manual jogging is delegated to the existing joystick rate-control
    by toggling the tracking mode (the persistent MountControlThread does the work).
    """

    def __init__(self, joystick_state, config_state, solver, ts, ephemeris,
                 cam_index=0, settle_pause=0.4, align_state=None):
        self.joystick_state = joystick_state
        self.config_state = config_state
        self.solver = solver
        self.ts = ts
        self.ephemeris = ephemeris
        self.cam_index = cam_index
        self.settle_pause = settle_pause
        self.align_state = align_state

    @property
    def fov_deg(self):
        return float(getattr(self.solver, 'fov_deg', 1.0) or 1.0)

    def _publish_pos(self, az, el):
        if self.align_state is not None:
            self.align_state.current_mount_azel = (az, el)

    def slew(self, az, el):
        """Coarse goto + closed-loop settle onto (az, el). Returns True if it settled."""
        controller = getattr(self.joystick_state, 'telescope_controller', None)
        if controller is None:
            return False
        ok = slew_to_azel(controller, self.config_state, az, el, position_cb=self._publish_pos)
        time.sleep(self.settle_pause)  # let the camera grab a fresh post-slew frame
        return ok

    def capture(self):
        """Grab a frame, plate-solve it, and pair it with the current encoder position.

        Returns ``(az_cmd, el_cmd, az_obs, el_obs, meta)`` or ``None`` if there is no
        frame or no solution.
        """
        import camera_manager
        import plate_solver as ps_mod
        from transformations import AzAlt2AzEl_AltAz
        from lib.auxstar import Targets

        controller = getattr(self.joystick_state, 'telescope_controller', None)
        if controller is None:
            return None
        camera = camera_manager.camera_manager.get_camera(self.cam_index)
        raw = (camera.thread.get_latest_raw()
               if camera is not None and getattr(camera, 'thread', None) is not None else None)
        if raw is None:
            return None
        t = self.ts.now()
        result = self.solver.solve(raw)
        if result is None or not result.solved:
            return None
        lat = float(self.config_state.lat_str or 0.0)
        lon = float(self.config_state.lon_str or 0.0)
        elev = float(self.config_state.alt_str or 0.0)
        az_obs, el_obs = ps_mod.solved_azel(result.ra_deg, result.dec_deg, lat, lon, elev,
                                            self.ephemeris, self.ts, t)
        align_az = float(getattr(self.config_state, 'alignment_azimuth_str', 0.0) or 0.0)
        cur_azm = controller.hc_get_position(Targets.AZM) * 360.0
        cur_alt = controller.hc_get_position(Targets.ALT) * 360.0
        az_cmd, el_cmd = AzAlt2AzEl_AltAz(cur_azm, cur_alt, align_az)
        meta = {
            'result': result, 'az_obs': az_obs % 360.0, 'el_obs': el_obs,
            'az_cmd': az_cmd % 360.0, 'el_cmd': el_cmd, 'src_shape': tuple(raw.shape[:2]),
            'cam_index': self.cam_index, 't': t, 'n_matches': result.n_matches,
            'fov': result.fov_deg, 'rmse': result.rmse,
        }
        if self.align_state is not None:
            self.align_state.last_solve = meta
            self.align_state.current_mount_azel = (az_cmd % 360.0, el_cmd)
        return (az_cmd % 360.0, el_cmd, az_obs % 360.0, el_obs, meta)

    def set_manual(self, on):
        """Hand the mount to (on) / take it back from (off) the joystick rate-control."""
        js = self.joystick_state
        if js is None:
            return
        try:
            from joystick_controller import TrackingMode
            js.tracking_mode = TrackingMode.RATE_CONTROL if on else TrackingMode.STANDBY
        except Exception:
            pass


def make_default_sample_fn(joystick_state, config_state, solver, ts, ephemeris,
                           cam_index=0, settle_pause=0.4, align_state=None):
    """Build the GUI sampler (slew + capture + plate-solve). See GuiSampler."""
    return GuiSampler(joystick_state, config_state, solver, ts, ephemeris,
                      cam_index, settle_pause, align_state)


class AlignmentRunner(threading.Thread):
    """Runs the alignment state machine on a worker thread.

    ``sampler`` is either a GuiSampler (live slew+capture, supports grid-search and
    manual recovery) or a plain callable ``sampler(az, el) -> (az_obs, el_obs) | None``
    used by the headless tests. ``pending_points`` + ``append`` let the same runner do
    the initial run (build the grid, fresh state) or a resume/retry pass (re-run a subset
    of points, append to the existing samples, then refit).
    """

    def __init__(self, align_state, config_state, sampler,
                 n_points=18, holdout_frac=0.25, el_min=None, el_max=80.0,
                 remove_refraction=False, status_cb=None,
                 pending_points=None, append=False, grid_cells=None):
        super().__init__(daemon=True)
        self.s = align_state
        self.config_state = config_state
        self.sampler = sampler
        self.n_points = n_points
        self.holdout_frac = holdout_frac
        self.el_max = el_max
        if el_min is None:
            mask = float(getattr(config_state, 'elevation_mask_str', 0.0) or 0.0)
            el_min = max(15.0, mask + 5.0)
        self.el_min = el_min
        self.remove_refraction = remove_refraction
        self.status_cb = status_cb
        self.pending_points = pending_points
        self.append = append
        if grid_cells is None:
            grid_cells = int(getattr(config_state, 'alignment_grid_search_cells', 100) or 0)
        self.grid_cells = max(0, int(grid_cells))
        self._live = hasattr(sampler, 'capture')   # GuiSampler vs plain callable
        self._abort = threading.Event()
        self._skip = threading.Event()             # set by the UI to abandon the current point

    def abort(self):
        self._abort.set()
        self._skip.set()  # unblock a manual wait promptly

    def skip(self):
        """Abandon the current point (manual recovery / slow grid search) and move on."""
        self._skip.set()

    def _status(self, msg):
        self.s.status = msg
        if self.status_cb:
            self.status_cb(msg)

    # ---- per-point acquisition ------------------------------------------------
    def _acquire(self, az, el, kind):
        """Slew to (az, el) and return a sample tuple (az_cmd, el_cmd, az_obs, el_obs,
        meta) or None. For the live sampler this adds a grid search and, if enabled, a
        manual-jog recovery; for the plain test callable it just calls it."""
        s = self.s
        s.current_target = (az, el)
        if not self._live:
            obs = self.sampler(az, el)
            return None if obs is None else (az, el, obs[0], obs[1], {'kind': kind})

        self.sampler.slew(az, el)
        smp = self.sampler.capture()
        if smp is None and not self._abort.is_set():
            smp = self._grid_search(az, el)
        if smp is None and s.pause_on_fail and not self._abort.is_set():
            smp = self._manual(az, el)
        return smp

    def _grid_search(self, az, el):
        """Spiral outward over up to ``grid_cells`` small offsets (~0.5x FOV each) around a
        point, returning the first solve. This is far faster than a manual jog, so we
        search a wide neighbourhood (default 100 cells, configurable via
        config.alignment_grid_search_cells) before giving up / pausing for manual.

        Offsets are in cross-elevation / elevation so the on-sky step is uniform; the az
        offset is divided by cos(el) to keep it a true angular step near the pole."""
        n = self.grid_cells
        if n <= 0:
            return None
        step = max(0.2, 0.5 * self.sampler.fov_deg)
        cos_el = max(math.cos(math.radians(el)), 0.1)
        offsets = _spiral_offsets(n)
        for k, (cx, cy) in enumerate(offsets):
            if self._abort.is_set() or self._skip.is_set():
                self._skip.clear()
                return None
            t_az = (az + (cx * step) / cos_el) % 360.0
            t_el = min(self.el_max, max(self.el_min, el + cy * step))
            self.s.current_target = (t_az, t_el)
            self._status(f"grid search {k + 1}/{n} around az {az:.0f} el {el:.0f}")
            self.sampler.slew(t_az, t_el)
            smp = self.sampler.capture()
            if smp is not None:
                return smp
        return None

    def _manual(self, az, el):
        """Pause and let the operator jog by hand; auto-capture the first solve.

        The mount is handed to the joystick rate-control (set_manual). We re-solve every
        ~1 s; the first solution is captured and the run continues. Skip/Abort unblock."""
        s = self.s
        s.manual_active = True
        s.paused = True
        s.manual_target = (az, el)
        self.sampler.set_manual(True)
        self._status(f"MANUAL: jog to stars (az {az:.0f} el {el:.0f}) - auto-captures on solve")
        try:
            while not self._abort.is_set():
                if self._skip.is_set():
                    self._skip.clear()
                    return None
                smp = self.sampler.capture()
                if smp is not None:
                    return smp
                if self._skip.wait(1.0):  # interruptible re-solve cadence
                    self._skip.clear()
                    return None
            return None
        finally:
            s.manual_active = False
            s.paused = False
            s.manual_target = None
            self.sampler.set_manual(False)

    def _record(self, smp, kind):
        """Append a sample (and its meta) to the right bucket."""
        s = self.s
        meta = dict(smp[4] or {})
        meta['kind'] = kind
        if kind == 'backtest':
            s.backtest_samples.append(smp[:4])
        else:
            s.samples.append(smp[:4])
            s.sample_meta.append(meta)

    # ---- main loop ------------------------------------------------------------
    def run(self):
        s = self.s
        try:
            if self.append:
                pending = list(self.pending_points or [])
            else:
                grid = fibonacci_sky_grid(self.n_points, self.el_min, self.el_max)
                if len(grid) < MIN_SAMPLES + 2:
                    s.phase = ERROR
                    s.error = "sky grid too small for this elevation band"
                    self._status(s.error)
                    return
                # Deterministic holdout split: every Nth point is held out for backtest.
                n_hold = max(2, int(round(len(grid) * self.holdout_frac)))
                hold_idx = set(round(i * len(grid) / n_hold) % len(grid) for i in range(n_hold))
                s.fit_points = [p for i, p in enumerate(grid) if i not in hold_idx]
                s.holdout_points = [p for i, p in enumerate(grid) if i in hold_idx]
                s.grid = grid
                pending = list(s.fit_points)

            # --- collect fit samples ---
            s.phase = RUNNING
            total = len(pending)
            for i, (az, el) in enumerate(pending):
                if self._abort.is_set():
                    s.phase = IDLE
                    self._status("aborted")
                    return
                s.progress = (i, total)
                self._status(f"sampling {i + 1}/{total}  az {az:.0f} el {el:.0f}")
                smp = self._acquire(az, el, 'fit')
                if smp is not None:
                    self._record(smp, 'fit')
                else:
                    s.failed_points.append((az, el))
            s.progress = (total, total)

            if len(s.samples) < MIN_SAMPLES:
                s.phase = ERROR
                s.error = (f"only {len(s.samples)} good solves (need {MIN_SAMPLES}). "
                           f"{len(s.failed_points)} point(s) failed - check cameras/DB/stars, "
                           f"then Retry Failed.")
                self._status(s.error)
                return

            # --- fit ---
            self._status("fitting pointing model...")
            model, stats = PointingModel.fit(s.samples, remove_refraction=self.remove_refraction)
            s.model = model
            s.stats = stats

            # --- backtest on held-out points (initial run only; retry just refits) ---
            if not self.append:
                s.phase = BACKTEST
                for j, (az, el) in enumerate(s.holdout_points):
                    if self._abort.is_set():
                        break
                    self._status(f"backtest {j + 1}/{len(s.holdout_points)}")
                    smp = self._acquire(az, el, 'backtest')
                    if smp is not None:
                        self._record(smp, 'backtest')
                    else:
                        s.failed_points.append((az, el))
            if s.backtest_samples:
                s.backtest_rms_deg = model.backtest(s.backtest_samples,
                                                    remove_refraction=self.remove_refraction)

            s.current_target = None
            s.phase = DONE
            self._status(f"done: RMS {stats['rms_after_arcmin']:.2f}' "
                         f"(was {stats['rms_before_arcmin']:.2f}')"
                         + (f", {len(s.failed_points)} failed" if s.failed_points else ""))
        except Exception as e:
            s.current_target = None
            s.phase = ERROR
            s.error = str(e)
            self._status(f"error: {e}")
        finally:
            # Never leave the mount under joystick control after the run ends.
            if self._live:
                try:
                    self.sampler.set_manual(False)
                except Exception:
                    pass


def accept_alignment(align_state, config_state, save=True):
    """Write the fitted model into config and enable it."""
    if align_state.model is None:
        return False
    config_state.pointing_model_terms = align_state.model.to_config()
    config_state.pointing_model_enabled = True
    if save:
        try:
            config_state.save_to_file()
        except Exception as e:
            print(f"accept_alignment save failed: {e}")
    return True
