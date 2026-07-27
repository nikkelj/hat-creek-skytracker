"""Online PID auto-tuner for the tracking loops.

Tunes the six pid_*_gain values IN PLACE on the live ConfigState while a
target is being tracked. Both control paths (the Python loop and the Rust
core loop) re-read the gains from config every cycle, so writing a candidate
gain takes effect on the next control cycle with no loop-specific wiring --
and the PID pane's sliders visibly follow along.

Method: per-axis coordinate descent ("twiddle") in log-gain space, driven by
the natural tracking excitation (trajectory motion + measurement noise)
rather than injected test signals, so it is safe to run on a live pass:

  * each sweep starts with a measurement window at the current best gains to
    re-baseline the cost (conditions drift over a pass; a stale best-cost
    would block all later acceptance),
  * each parameter (P, then D, then I) is probed up then down by a step in
    decades; a candidate that improves the cost by more than the acceptance
    margin is kept and expands the step, otherwise it is reverted and the
    step shrinks,
  * both axes share the measurement schedule -- their errors are independent
    signals sampled from the same window -- so a full sweep costs 7 windows
    (~40 s at the defaults) instead of 13.

Cost is the RMS mount-axis position error over the window. Guards: gains are
clamped to the UI slider range, a diverging candidate is reverted mid-window,
and the tuner pauses (holding the best-known gains) whenever tracking stops
being live -- STOP, target below the mask, etc. -- resuming the interrupted
probe when tracking returns. A mode change that switches the PID gain
PROFILE (PROGRAM <-> HOTSPOT; see JoystickModeState.service_gain_profiles)
instead STOPS the tune, keeping the best gains: the two modes are different
plants with separately stored tunings, so a tune never outlives its plant.
"""

import math
import time

# Gain clamp: the PID pane's log sliders span exactly this range.
GAIN_MIN = 2.0e-5
GAIN_MAX = 2.0

# Measurement schedule. Settle discards the transient from the gain change;
# eval is the scoring window. At the ~8-15 Hz control cadence a 4 s window
# yields 30-60 error samples.
SETTLE_SEC = 1.5
EVAL_SEC = 4.0
MIN_SAMPLES = 8

# A candidate must beat the best cost by >3% -- windowed RMS is noisy, and a
# plain "<" acceptance random-walks the gains instead of descending.
IMPROVE_FRACTION = 0.97

# Twiddle steps, in decades of gain (multiplicative: 0.3 decades ~= x2).
# Expand on success so a badly-seeded gain crosses its decades quickly;
# shrink on a failed up+down pair. A parameter is converged when its step
# falls under STEP_DONE_DECADES (~ x1.1).
STEP_INIT_DECADES = 0.30
STEP_EXPAND = 1.4
STEP_SHRINK = 0.5
STEP_DONE_DECADES = 0.04
MAX_SWEEPS = 10

# Divergence guard: revert a candidate the moment an axis error exceeds
# max(absolute floor, factor x baseline RMS) -- don't wait out the window.
DIVERGE_ABS_DEG = 5.0
DIVERGE_FACTOR = 4.0

# Probe order: P sets the response, D damps what P excites, I trims bias.
PARAM_ORDER = ('p', 'd', 'i')

_INF = float('inf')


def _clamp(gain):
    return min(GAIN_MAX, max(GAIN_MIN, gain))


class _AxisTuner:
    """Twiddle state for one mount axis (fields pid_<axis>_<p|i|d>_gain)."""

    def __init__(self, config_state, axis):
        self.cfg = config_state
        self.axis = axis
        self.steps = {k: STEP_INIT_DECADES for k in PARAM_ORDER}
        self.initial = self.read_gains()
        self.best = dict(self.initial)
        self.best_cost = None      # windowed RMS (deg) at self.best
        self.baseline_rms = None   # sweep-start RMS, sizes the divergence gate
        self.aborted = False       # current window tripped the divergence gate

    def _field(self, param):
        return f"pid_{self.axis}_{param}_gain"

    def read_gains(self):
        return {k: float(getattr(self.cfg, self._field(k), 0.0) or 0.0)
                for k in PARAM_ORDER}

    def apply(self, gains):
        for k, v in gains.items():
            setattr(self.cfg, self._field(k), round(_clamp(v), 6))

    def candidate(self, param, direction):
        """Best gains with one parameter stepped up (+1) or down (-1)."""
        gains = dict(self.best)
        factor = 10.0 ** self.steps[param]
        base = _clamp(gains[param])
        gains[param] = _clamp(base * factor if direction > 0 else base / factor)
        return gains

    @property
    def done(self):
        return all(s < STEP_DONE_DECADES for s in self.steps.values())

    def diverge_limit(self):
        if self.baseline_rms is None:
            return DIVERGE_ABS_DEG
        return max(DIVERGE_ABS_DEG, DIVERGE_FACTOR * self.baseline_rms)


class PIDAutoTuner:
    """Shared-schedule twiddle over both axes. Drive it by calling update()
    once per control cycle (either loop); read status_text() for the UI."""

    def __init__(self, config_state, mode=None):
        self.cfg = config_state
        # Tracking mode this tuner was armed in. The caller gates samples on
        # it: PROGRAM and HOTSPOT are different plants (encoder loop vs.
        # optical loop), so a tune only ever measures the mode it started in.
        self.armed_mode = mode
        # Set by the caller at arm time: what we're tuning against (target
        # name). Stamped into the mode's gain profile when the tune finishes.
        self.target_label = None
        self._label_stamped = False
        self.axes = {'azm': _AxisTuner(config_state, 'azm'),
                     'alt': _AxisTuner(config_state, 'alt')}
        self.active = False
        self.phase = 'idle'        # idle | settle | measure | paused | done | stopped
        self.stage = None          # ('baseline', 0) or (param, +1/-1)
        self.sweep = 0
        self._param_idx = 0
        self._pending = {}         # axis -> candidate gains under test (None = holding best)
        self._up_accepted = {}     # axis -> up-probe accepted (skip the down-probe)
        self._phase_t0 = 0.0
        self._acc = {}             # axis -> [sum_sq, n]
        self._messages = []

    # ---------------------------------------------------------------- control
    def start(self, now=None):
        """Arm the tuner. `now` is injectable so tests can run a simulated
        clock; the live callers omit it."""
        self.sweep = 0
        self._param_idx = 0
        self.active = True
        self._begin_stage(('baseline', 0), time.time() if now is None else now)
        self._say("Auto-tune started (P/D/I twiddle, both axes)")

    def stop(self, revert=False):
        """Stop tuning. Keeps the best-known gains unless revert=True, which
        restores the gains from before start()."""
        for ax in self.axes.values():
            ax.apply(ax.initial if revert else ax.best)
        self.active = False
        self.phase = 'stopped'
        self._say("Auto-tune stopped - gains "
                  + ("reverted" if revert else f"kept ({self.summary()})"))

    def take_messages(self):
        msgs, self._messages = self._messages, []
        return msgs

    def _say(self, msg):
        print(f"AUTOTUNE: {msg}")
        self._messages.append(msg)

    # ---------------------------------------------------------------- sampling
    def update(self, now, tracking_active, azm_error_deg, alt_error_deg):
        """Feed one control-cycle sample. tracking_active must be True only
        while the armed mode is genuinely closed-loop on the target."""
        if not self.active:
            return

        if not tracking_active:
            if self.phase != 'paused':
                # Hold the best-known gains while paused; the interrupted
                # probe restarts from its settle phase on resume.
                for ax in self.axes.values():
                    ax.apply(ax.best)
                self.phase = 'paused'
                self._say("Auto-tune paused (tracking not live)")
            return

        if self.phase == 'paused':
            self._begin_stage(self.stage, now)
            self._say("Auto-tune resumed")
            return

        errors = {'azm': azm_error_deg, 'alt': alt_error_deg}

        # Divergence gate, live in both settle and measure: a runaway
        # candidate is reverted immediately, not at the end of the window.
        for name, ax in self.axes.items():
            if not ax.aborted and abs(errors[name]) > ax.diverge_limit():
                ax.apply(ax.best)
                ax.aborted = True

        if self.phase == 'settle':
            if now - self._phase_t0 >= SETTLE_SEC:
                self.phase = 'measure'
                self._phase_t0 = now
                self._acc = {name: [0.0, 0] for name in self.axes}
        elif self.phase == 'measure':
            for name, ax in self.axes.items():
                if not ax.aborted:
                    acc = self._acc[name]
                    acc[0] += errors[name] * errors[name]
                    acc[1] += 1
            if now - self._phase_t0 >= EVAL_SEC:
                self._finish_window(now)

    # ---------------------------------------------------------------- stages
    def _begin_stage(self, stage, now):
        self.stage = stage
        self.phase = 'settle'
        self._phase_t0 = now
        self._pending = {}
        kind, direction = stage
        for name, ax in self.axes.items():
            ax.aborted = False
            probe = None
            if kind != 'baseline' and not ax.done:
                if direction > 0:
                    probe = ax.candidate(kind, +1)
                elif not self._up_accepted.get(name, False):
                    probe = ax.candidate(kind, -1)
            ax.apply(probe if probe is not None else ax.best)
            self._pending[name] = probe

    def _finish_window(self, now):
        kind, _direction = self.stage

        # Too few samples with nothing aborted means the loop cadence hiccuped
        # (not the candidate's fault): re-run the same stage.
        if (not any(ax.aborted for ax in self.axes.values())
                and all(acc[1] < MIN_SAMPLES for acc in self._acc.values())):
            self._begin_stage(self.stage, now)
            return

        costs = {}
        for name, ax in self.axes.items():
            sum_sq, n = self._acc[name]
            if ax.aborted or n < MIN_SAMPLES:
                costs[name] = _INF
            else:
                costs[name] = math.sqrt(sum_sq / n)

        if kind == 'baseline':
            for name, ax in self.axes.items():
                if math.isfinite(costs[name]):
                    ax.best_cost = costs[name]
                    ax.baseline_rms = costs[name]
            self._up_accepted = {}
            self._begin_stage((PARAM_ORDER[self._param_idx], +1), now)
            return

        param, direction = kind, _direction
        for name, ax in self.axes.items():
            probe = self._pending.get(name)
            if probe is None:
                continue  # held best gains this window (done, or up accepted)
            accepted = (ax.best_cost is not None
                        and costs[name] < ax.best_cost * IMPROVE_FRACTION)
            if accepted:
                ax.best = probe
                ax.best_cost = costs[name]
                ax.steps[param] = min(2.0, ax.steps[param] * STEP_EXPAND)
                if direction > 0:
                    self._up_accepted[name] = True
            else:
                ax.apply(ax.best)
                if direction < 0:
                    # Both directions failed: tighten this parameter's step.
                    ax.steps[param] *= STEP_SHRINK

        if direction > 0:
            self._begin_stage((param, -1), now)
            return

        # Down-probe finished: advance to the next parameter / sweep.
        self._up_accepted = {}
        self._param_idx += 1
        if self._param_idx < len(PARAM_ORDER):
            self._begin_stage((PARAM_ORDER[self._param_idx], +1), now)
            return
        self._param_idx = 0
        self.sweep += 1
        if all(ax.done for ax in self.axes.values()) or self.sweep >= MAX_SWEEPS:
            for ax in self.axes.values():
                ax.apply(ax.best)
            self.active = False
            self.phase = 'done'
            self._say(f"Auto-tune converged after {self.sweep} sweep(s): {self.summary()}")
            return
        self._begin_stage(('baseline', 0), now)

    # ---------------------------------------------------------------- display
    def summary(self):
        parts = []
        for name, ax in self.axes.items():
            if ax.best_cost is not None:
                parts.append(f'{name} rms {ax.best_cost * 3600.0:.0f}"')
        return ", ".join(parts) if parts else "no measurement yet"

    def status_text(self):
        """One short line for the PID pane."""
        if self.phase == 'paused':
            return "autotune: paused (not tracking)"
        if self.phase == 'done':
            return f"autotune: converged - {self.summary()}"
        if self.phase == 'stopped':
            return f"autotune: stopped - {self.summary()}"
        if self.stage is None:
            return "autotune: idle"
        kind, direction = self.stage
        if kind == 'baseline':
            what = "baseline"
        else:
            what = f"{kind.upper()}{'+' if direction > 0 else '-'}"
        left = ""
        if self.phase == 'measure':
            left = f" {max(0.0, EVAL_SEC - (time.time() - self._phase_t0)):.0f}s"
        return f"autotune S{self.sweep + 1} {what}{left} | {self.summary()}"
