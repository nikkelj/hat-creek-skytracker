//! Online PID auto-tuner — faithful port of autotune.py (Phase 6b of the
//! Rust port): per-axis coordinate descent ("twiddle") in log-gain space
//! on a shared measurement schedule, with settle/eval windows, a live
//! divergence gate, pause/resume, and sweep convergence. The tuner owns
//! the gains it applies; the caller mirrors `applied_gains()` into its
//! live config each cycle (the Python wrapper does exactly that).
//! Parameter order is P, D, I.

pub const GAIN_MIN: f64 = 2.0e-5;
pub const GAIN_MAX: f64 = 2.0;
pub const SETTLE_SEC: f64 = 1.5;
pub const EVAL_SEC: f64 = 4.0;
pub const MIN_SAMPLES: u32 = 8;
pub const IMPROVE_FRACTION: f64 = 0.97;
pub const STEP_INIT_DECADES: f64 = 0.30;
pub const STEP_EXPAND: f64 = 1.4;
pub const STEP_SHRINK: f64 = 0.5;
pub const STEP_DONE_DECADES: f64 = 0.04;
pub const MAX_SWEEPS: u32 = 10;
pub const DIVERGE_ABS_DEG: f64 = 5.0;
pub const DIVERGE_FACTOR: f64 = 4.0;
pub const PARAM_NAMES: [&str; 3] = ["p", "d", "i"];

fn clamp(g: f64) -> f64 {
    g.clamp(GAIN_MIN, GAIN_MAX)
}

/// Python's round(x, 6) for the gain magnitudes in play.
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

#[derive(Clone, Debug)]
pub struct AxisTuner {
    pub steps: [f64; 3],
    pub initial: [f64; 3],
    pub best: [f64; 3],
    pub applied: [f64; 3],
    pub best_cost: Option<f64>,
    pub baseline_rms: Option<f64>,
    pub aborted: bool,
}

impl AxisTuner {
    fn new(initial: [f64; 3]) -> Self {
        AxisTuner {
            steps: [STEP_INIT_DECADES; 3],
            initial,
            best: initial,
            applied: initial,
            best_cost: None,
            baseline_rms: None,
            aborted: false,
        }
    }

    fn apply(&mut self, gains: [f64; 3]) {
        for k in 0..3 {
            self.applied[k] = round6(clamp(gains[k]));
        }
    }

    fn candidate(&self, param: usize, direction: i32) -> [f64; 3] {
        let mut gains = self.best;
        let factor = 10f64.powf(self.steps[param]);
        let base = clamp(gains[param]);
        gains[param] = clamp(if direction > 0 { base * factor } else { base / factor });
        gains
    }

    fn done(&self) -> bool {
        self.steps.iter().all(|&s| s < STEP_DONE_DECADES)
    }

    fn diverge_limit(&self) -> f64 {
        match self.baseline_rms {
            None => DIVERGE_ABS_DEG,
            Some(b) => DIVERGE_ABS_DEG.max(DIVERGE_FACTOR * b),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Settle,
    Measure,
    Paused,
    Done,
    Stopped,
}

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Settle => "settle",
            Phase::Measure => "measure",
            Phase::Paused => "paused",
            Phase::Done => "done",
            Phase::Stopped => "stopped",
        }
    }
}

/// A probe stage: None = baseline window; Some((param_idx, direction)).
pub type Stage = Option<(usize, i32)>;

pub struct PidAutoTuner {
    pub axes: [AxisTuner; 2], // [azm, alt]
    pub active: bool,
    pub phase: Phase,
    /// None until start(); Some(None) = baseline window.
    pub stage: Option<Stage>,
    pub sweep: u32,
    param_idx: usize,
    pending: [Option<[f64; 3]>; 2],
    up_accepted: [bool; 2],
    phase_t0: f64,
    acc: [(f64, u32); 2],
    messages: Vec<String>,
}

impl PidAutoTuner {
    pub fn new(azm_initial: [f64; 3], alt_initial: [f64; 3]) -> Self {
        PidAutoTuner {
            axes: [AxisTuner::new(azm_initial), AxisTuner::new(alt_initial)],
            active: false,
            phase: Phase::Idle,
            stage: None,
            sweep: 0,
            param_idx: 0,
            pending: [None, None],
            up_accepted: [false, false],
            phase_t0: 0.0,
            acc: [(0.0, 0), (0.0, 0)],
            messages: Vec::new(),
        }
    }

    fn say(&mut self, msg: String) {
        self.messages.push(msg);
    }

    pub fn take_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.messages)
    }

    pub fn applied_gains(&self) -> [[f64; 3]; 2] {
        [self.axes[0].applied, self.axes[1].applied]
    }

    pub fn start(&mut self, now: f64) {
        self.sweep = 0;
        self.param_idx = 0;
        self.active = true;
        self.begin_stage(None, now);
        self.say("Auto-tune started (P/D/I twiddle, both axes)".into());
    }

    pub fn stop(&mut self, revert: bool) {
        for ax in self.axes.iter_mut() {
            let g = if revert { ax.initial } else { ax.best };
            ax.apply(g);
        }
        self.active = false;
        self.phase = Phase::Stopped;
        let msg = if revert {
            "Auto-tune stopped - gains reverted".to_string()
        } else {
            format!("Auto-tune stopped - gains kept ({})", self.summary())
        };
        self.say(msg);
    }

    pub fn update(&mut self, now: f64, tracking_active: bool, azm_err: f64, alt_err: f64) {
        if !self.active {
            return;
        }
        if !tracking_active {
            if self.phase != Phase::Paused {
                for ax in self.axes.iter_mut() {
                    let b = ax.best;
                    ax.apply(b);
                }
                self.phase = Phase::Paused;
                self.say("Auto-tune paused (tracking not live)".into());
            }
            return;
        }
        if self.phase == Phase::Paused {
            let stage = self.stage.unwrap_or(None);
            self.begin_stage(stage, now);
            self.say("Auto-tune resumed".into());
            return;
        }

        let errors = [azm_err, alt_err];
        for (i, ax) in self.axes.iter_mut().enumerate() {
            if !ax.aborted && errors[i].abs() > ax.diverge_limit() {
                let b = ax.best;
                ax.apply(b);
                ax.aborted = true;
            }
        }

        match self.phase {
            Phase::Settle => {
                if now - self.phase_t0 >= SETTLE_SEC {
                    self.phase = Phase::Measure;
                    self.phase_t0 = now;
                    self.acc = [(0.0, 0), (0.0, 0)];
                }
            }
            Phase::Measure => {
                for i in 0..2 {
                    if !self.axes[i].aborted {
                        self.acc[i].0 += errors[i] * errors[i];
                        self.acc[i].1 += 1;
                    }
                }
                if now - self.phase_t0 >= EVAL_SEC {
                    self.finish_window(now);
                }
            }
            _ => {}
        }
    }

    fn begin_stage(&mut self, stage: Stage, now: f64) {
        self.stage = Some(stage);
        self.phase = Phase::Settle;
        self.phase_t0 = now;
        self.pending = [None, None];
        for i in 0..2 {
            let up_accepted = self.up_accepted[i];
            let ax = &mut self.axes[i];
            ax.aborted = false;
            let mut probe = None;
            if let Some((param, direction)) = stage {
                if !ax.done() {
                    if direction > 0 {
                        probe = Some(ax.candidate(param, 1));
                    } else if !up_accepted {
                        probe = Some(ax.candidate(param, -1));
                    }
                }
            }
            let g = probe.unwrap_or(ax.best);
            ax.apply(g);
            self.pending[i] = probe;
        }
    }

    fn finish_window(&mut self, now: f64) {
        let stage = self.stage.unwrap_or(None);
        let any_aborted = self.axes.iter().any(|a| a.aborted);
        if !any_aborted && self.acc.iter().all(|a| a.1 < MIN_SAMPLES) {
            self.begin_stage(stage, now);
            return;
        }
        let mut costs = [f64::INFINITY; 2];
        for i in 0..2 {
            let (sum_sq, n) = self.acc[i];
            if !self.axes[i].aborted && n >= MIN_SAMPLES {
                costs[i] = (sum_sq / n as f64).sqrt();
            }
        }

        let Some((param, direction)) = stage else {
            for i in 0..2 {
                if costs[i].is_finite() {
                    self.axes[i].best_cost = Some(costs[i]);
                    self.axes[i].baseline_rms = Some(costs[i]);
                }
            }
            self.up_accepted = [false, false];
            let p = self.param_idx;
            self.begin_stage(Some((p, 1)), now);
            return;
        };

        for i in 0..2 {
            let Some(probe) = self.pending[i] else { continue };
            let ax = &mut self.axes[i];
            let accepted = match ax.best_cost {
                Some(bc) => costs[i] < bc * IMPROVE_FRACTION,
                None => false,
            };
            if accepted {
                ax.best = probe;
                ax.best_cost = Some(costs[i]);
                ax.steps[param] = (ax.steps[param] * STEP_EXPAND).min(2.0);
                if direction > 0 {
                    self.up_accepted[i] = true;
                }
            } else {
                let b = ax.best;
                ax.apply(b);
                if direction < 0 {
                    ax.steps[param] *= STEP_SHRINK;
                }
            }
        }

        if direction > 0 {
            self.begin_stage(Some((param, -1)), now);
            return;
        }

        self.up_accepted = [false, false];
        self.param_idx += 1;
        if self.param_idx < 3 {
            let p = self.param_idx;
            self.begin_stage(Some((p, 1)), now);
            return;
        }
        self.param_idx = 0;
        self.sweep += 1;
        if self.axes.iter().all(|a| a.done()) || self.sweep >= MAX_SWEEPS {
            for ax in self.axes.iter_mut() {
                let b = ax.best;
                ax.apply(b);
            }
            self.active = false;
            self.phase = Phase::Done;
            let msg = format!(
                "Auto-tune converged after {} sweep(s): {}",
                self.sweep,
                self.summary()
            );
            self.say(msg);
            return;
        }
        self.begin_stage(None, now);
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (name, ax) in ["azm", "alt"].iter().zip(self.axes.iter()) {
            if let Some(c) = ax.best_cost {
                parts.push(format!("{name} rms {:.0}\"", c * 3600.0));
            }
        }
        if parts.is_empty() {
            "no measurement yet".into()
        } else {
            parts.join(", ")
        }
    }

    /// One status line (time-remaining uses the caller's `now`).
    pub fn status_text(&self, now: f64) -> String {
        match self.phase {
            Phase::Paused => return "autotune: paused (not tracking)".into(),
            Phase::Done => return format!("autotune: converged - {}", self.summary()),
            Phase::Stopped => return format!("autotune: stopped - {}", self.summary()),
            _ => {}
        }
        let Some(stage) = self.stage else {
            return "autotune: idle".into();
        };
        let what = match stage {
            None => "baseline".to_string(),
            Some((p, d)) => format!(
                "{}{}",
                PARAM_NAMES[p].to_uppercase(),
                if d > 0 { "+" } else { "-" }
            ),
        };
        let left = if self.phase == Phase::Measure {
            format!(" {:.0}s", (EVAL_SEC - (now - self.phase_t0)).max(0.0))
        } else {
            String::new()
        };
        format!("autotune S{} {}{} | {}", self.sweep + 1, what, left, self.summary())
    }

    pub fn stage_label(&self) -> String {
        match self.stage {
            None => "none".into(),
            Some(None) => "baseline".into(),
            Some(Some((p, d))) => format!("{}{}", PARAM_NAMES[p], if d > 0 { "+" } else { "-" }),
        }
    }
}
