//! Alignment-run sequencing — port of `alignment.py` (`AlignmentRunner`,
//! `fibonacci_sky_grid`, `_spiral_offsets`, the holdout split, early-stop,
//! grid-search / manual-recovery / retry-failed policies) as a PURE state
//! machine.
//!
//! Nothing here sleeps, spawns threads, touches hardware or does astrometry.
//! The app owns the clock, the mount and the camera/plate-solver; it asks
//! [`AlignmentRunner::next_action`] what to do, does it, and reports back
//! through `on_arrived` / `on_solve` / `on_user` / `skip` / `abort`.
//!
//! Coordinates are coordinate-frame-agnostic "c1/c2" pairs exactly like the
//! Python samplers: in [`MountMode::AltAz`] they are sky (az, el) read back
//! from the encoders through the BASE transform; in [`MountMode::Eq`] they are
//! mount (HA, Dec) about the assumed pole (the app does the conversion, as
//! `EqGuiSampler` does). Targets handed out by `Action::Slew` are always sky
//! (az, el) in degrees. Samples are `(c1_cmd, c2_cmd, c1_obs, c2_obs)`: the
//! commanded coords come from the encoders AT SOLVE TIME (so grid-search
//! offsets and hand-jogged points are recorded correctly) and the observed
//! coords are the plate solution mapped to the same frame.

use std::f64::consts::PI;

use crate::altaz;
use crate::eq;
use crate::fit::{wrap180, FitStats, N_TERMS};

/// Need at least this many good solves to fit a 7-term model (alignment.MIN_SAMPLES).
pub const MIN_SAMPLES: usize = 6;
/// A fresh run needs at least this many grid points (Python: `MIN_SAMPLES + 2`).
pub const MIN_GRID_POINTS: usize = MIN_SAMPLES + 2;
/// Adaptive early-stop may not trigger before this many good samples
/// (Python: `max(MIN_SAMPLES + 6, 12)`).
pub const EARLY_STOP_FLOOR: usize = 12;

/// A sky direction `(az_deg, el_deg)`.
pub type SkyPoint = (f64, f64);

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Python 3 `round()` for the holdout split: round half to EVEN (Rust's
/// `f64::round` rounds half away from zero, which changes the held-out set).
pub fn python_round(x: f64) -> f64 {
    let frac = x - x.trunc();
    if frac.abs() == 0.5 {
        let f = x.floor();
        if f % 2.0 == 0.0 {
            f
        } else {
            f + 1.0
        }
    } else {
        x.round()
    }
}

/// Great-circle separation between two (az, el) directions, degrees.
pub fn angular_separation_deg(az1: f64, el1: f64, az2: f64, el2: f64) -> f64 {
    let (a1, e1, a2, e2) = (
        az1.to_radians(),
        el1.to_radians(),
        az2.to_radians(),
        el2.to_radians(),
    );
    let c = (e1.sin() * e2.sin() + e1.cos() * e2.cos() * (a1 - a2).cos()).clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

/// A circular sky zone the target grid must avoid (e.g. the Moon, a tree, a
/// dome slit edge). Extension beyond alignment.py (which has no keep-outs);
/// with an empty keep-out list the grid is bit-identical to the Python one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeepOut {
    pub az_deg: f64,
    pub el_deg: f64,
    pub radius_deg: f64,
}

impl KeepOut {
    pub fn contains(&self, az_deg: f64, el_deg: f64) -> bool {
        angular_separation_deg(self.az_deg, self.el_deg, az_deg, el_deg) < self.radius_deg
    }
}

/// ~Uniform (az, el) sample points on the hemisphere between `el_min` and
/// `el_max` — exact port of `pointing_model.fibonacci_sky_grid` (Fibonacci
/// sphere, oversampled x3 then filtered to the band, golden-angle azimuth
/// ordering). May return fewer than `n` points for a narrow band.
pub fn fibonacci_sky_grid(n: usize, el_min: f64, el_max: f64) -> Vec<(f64, f64)> {
    fibonacci_targets(n, el_min, el_max, &[])
}

/// [`fibonacci_sky_grid`] with keep-out zones filtered out in the same pass
/// (so the surviving points keep the Python order and the count still
/// targets `n`).
pub fn fibonacci_targets(
    n: usize,
    el_min: f64,
    el_max: f64,
    keepout: &[KeepOut],
) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(n);
    if n == 0 {
        return pts;
    }
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let m = (n * 3).max(12);
    for i in 0..m {
        let z = 1.0 - (i as f64 + 0.5) / m as f64; // z in (0,1): upper hemisphere
        let el = z.asin().to_degrees();
        if el < el_min || el > el_max {
            continue;
        }
        let az = (i as f64 * golden)
            .rem_euclid(2.0 * PI)
            .to_degrees()
            .rem_euclid(360.0);
        if keepout.iter().any(|k| k.contains(az, el)) {
            continue;
        }
        pts.push((az, el));
        if pts.len() >= n {
            break;
        }
    }
    pts
}

/// First `n` integer (col, row) offsets in an outward square spiral, nearest
/// ring first, origin excluded — exact port of `alignment._spiral_offsets`.
pub fn spiral_offsets(n: usize) -> Vec<(i32, i32)> {
    let mut out = Vec::with_capacity(n);
    let mut r: i32 = 1;
    while out.len() < n {
        let (mut x, mut y) = (-r, -r);
        for _ in 0..(2 * r) {
            out.push((x, y));
            x += 1;
        }
        for _ in 0..(2 * r) {
            out.push((x, y));
            y += 1;
        }
        for _ in 0..(2 * r) {
            out.push((x, y));
            x -= 1;
        }
        for _ in 0..(2 * r) {
            out.push((x, y));
            y -= 1;
        }
        r += 1;
    }
    out.truncate(n);
    out
}

/// Deterministic holdout split: `n_hold = max(2, round(len * frac))` points,
/// every `len/n_hold`-th index (Python rounding), are held out for the
/// backtest. Returns `(fit_points, holdout_points)` preserving grid order.
pub fn holdout_split(grid: &[SkyPoint], holdout_frac: f64) -> (Vec<SkyPoint>, Vec<SkyPoint>) {
    let len = grid.len();
    if len == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_hold = (python_round(len as f64 * holdout_frac) as usize).max(2);
    let hold_idx: Vec<usize> = (0..n_hold)
        .map(|i| (python_round(i as f64 * len as f64 / n_hold as f64) as usize) % len)
        .collect();
    let mut fit = Vec::new();
    let mut hold = Vec::new();
    for (i, p) in grid.iter().enumerate() {
        if hold_idx.contains(&i) {
            hold.push(*p);
        } else {
            fit.push(*p);
        }
    }
    (fit, hold)
}

// ---------------------------------------------------------------------------
// Parameters / data types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountMode {
    /// Fit the 7-term alt-az model; samples are sky (az, el).
    AltAz,
    /// Fit the 7-term equatorial model; samples are mount (HA, Dec).
    Eq,
}

#[derive(Clone, Debug)]
pub struct AlignmentParams {
    pub mount_mode: MountMode,
    /// Site latitude (Eq-mode design rows only).
    pub lat_deg: f64,
    /// Requested grid size (Python default 18).
    pub n_points: usize,
    /// Fraction of the grid held out for the backtest (default 0.25).
    pub holdout_frac: f64,
    /// Elevation band; Python derives el_min as `max(15, mask + 5)` —
    /// see [`AlignmentParams::el_min_from_mask`]. el_max default 80.
    pub el_min_deg: f64,
    pub el_max_deg: f64,
    /// Sky zones excluded from the target grid (extension; empty = Python).
    pub keepout: Vec<KeepOut>,
    /// Post-slew pause before capture so the camera has a fresh, still frame
    /// (GuiSampler.settle_pause, default 0.4 s). 0 = capture immediately.
    pub settle_s: f64,
    /// Solver FOV (deg) — grid-search step is `max(0.2, 0.5 * fov)`.
    pub fov_deg: f64,
    /// Spiral grid-search cells tried around a point that failed to solve
    /// before giving up / pausing for manual (config.alignment_grid_search_cells,
    /// default 100; 0 disables the search).
    pub grid_search_cells: usize,
    /// After the grid search also fails, pause for a manual joystick jog
    /// (the app keeps re-solving ~1 s and calling `on_solve`; first solution
    /// is captured). Python `AlignmentState.pause_on_fail`, default true.
    pub pause_on_fail: bool,
    /// Supervised mode extension: after every successful solve pause in
    /// `Action::AwaitUser` until `on_user(accept)`; a rejected sample is
    /// treated exactly like a solve failure (grid search / manual / failed).
    pub confirm_each_point: bool,
    /// Strip Bennett refraction from observed elevations before the fit (alt-az).
    pub remove_refraction: bool,
    /// MAD-robust outlier rejection in the fit (Python default True).
    pub robust: bool,
    /// Partial (nightly-refresh) fit: hold non-free terms at `seed_terms`.
    pub seed_terms: [f64; N_TERMS],
    pub free_idx: Option<Vec<usize>>,
    /// Adaptive early-stop target (arcmin); 0 disables.
    pub target_rms_arcmin: f64,
}

impl Default for AlignmentParams {
    fn default() -> Self {
        AlignmentParams {
            mount_mode: MountMode::AltAz,
            lat_deg: 0.0,
            n_points: 18,
            holdout_frac: 0.25,
            el_min_deg: 15.0,
            el_max_deg: 80.0,
            keepout: Vec::new(),
            settle_s: 0.4,
            fov_deg: 1.0,
            grid_search_cells: 100,
            pause_on_fail: true,
            confirm_each_point: false,
            remove_refraction: false,
            robust: true,
            seed_terms: [0.0; N_TERMS],
            free_idx: None,
            target_rms_arcmin: 0.0,
        }
    }
}

impl AlignmentParams {
    /// Python's default lower bound: `max(15, elevation_mask + 5)`.
    pub fn el_min_from_mask(mask_deg: f64) -> f64 {
        (mask_deg + 5.0).max(15.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    Fit,
    Backtest,
}

/// A plate solution already mapped into the runner's coordinate frame
/// (true sky az/el for AltAz, HA/Dec about the assumed pole for Eq), at the
/// frame's exposure time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolvedDirection {
    /// True az (AltAz) or HA (Eq), degrees.
    pub c1_deg: f64,
    /// True el (AltAz) or Dec (Eq), degrees.
    pub c2_deg: f64,
    pub rms_arcsec: f64,
    pub n_matches: u32,
    pub fov_deg: f64,
}

/// One recorded alignment sample (Python `samples[i]` + `sample_meta[i]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignSample {
    /// Commanded coords = encoders at solve time (az in [0,360) / HA in [-180,180)).
    pub c1_cmd: f64,
    pub c2_cmd: f64,
    /// Observed (plate-solved) coords in the same frame.
    pub c1_obs: f64,
    pub c2_obs: f64,
    /// Nominal grid point (sky az, el) this sample was acquired for.
    pub target_az: f64,
    pub target_el: f64,
    /// App clock at the solve.
    pub t_s: f64,
    pub n_matches: u32,
    pub rms_arcsec: f64,
    pub fov_deg: f64,
    pub kind: SampleKind,
}

impl AlignSample {
    /// `(c1_cmd, c2_cmd, c1_obs, c2_obs)` as consumed by `altaz::fit_altaz` / `eq::fit_eq`.
    pub fn as_row(&self) -> [f64; 4] {
        [self.c1_cmd, self.c2_cmd, self.c1_obs, self.c2_obs]
    }
}

/// The fitted model plus its stats, backtest and failure count.
#[derive(Clone, Debug)]
pub struct FitResult {
    pub mount_mode: MountMode,
    /// Coefficients in `altaz::TERM_NAMES` / `eq::TERM_NAMES` order (degrees).
    pub terms: [f64; N_TERMS],
    pub stats: FitStats,
    /// Held-out sky RMS (deg), if any backtest samples were acquired.
    pub backtest_rms_deg: Option<f64>,
    pub n_failed: usize,
}

impl FitResult {
    pub fn term_names(&self) -> [&'static str; N_TERMS] {
        match self.mount_mode {
            MountMode::AltAz => altaz::TERM_NAMES,
            MountMode::Eq => eq::TERM_NAMES,
        }
    }
    pub fn rms_before_arcmin(&self) -> f64 {
        self.stats.rms_before_deg * 60.0
    }
    pub fn rms_after_arcmin(&self) -> f64 {
        self.stats.rms_after_deg * 60.0
    }
}

/// Run phase (alignment.IDLE/RUNNING/BACKTEST/DONE/ERROR, plus an explicit
/// Aborted — Python drops back to IDLE with status "aborted").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Running,
    Backtest,
    Done,
    Error,
    Aborted,
}

/// What the app should do next.
#[derive(Clone, Debug)]
pub enum Action {
    /// Nothing to do (not started).
    Idle,
    /// Slew (coarse goto + closed-loop settle) so the boresight points at sky
    /// (az, el); call `on_arrived` when the mount has settled (or timed out —
    /// Python captures regardless of the settle verdict).
    Slew { az_deg: f64, el_deg: f64 },
    /// Post-slew camera settle; poll again at/after `until_s`.
    WaitSettle { until_s: f64 },
    /// Grab the latest frame, plate-solve it, and call `on_solve` with the
    /// result (or `None`) plus the encoder position at solve time.
    Capture,
    /// Paused for manual recovery of the nominal point (az, el): hand the
    /// mount to the joystick, keep re-solving (~1 s cadence) and calling
    /// `on_solve`; the first solution is captured. `skip` / `abort` unblock.
    Manual { az_deg: f64, el_deg: f64 },
    /// Supervised mode: a solve is waiting for `on_user(accept)`.
    AwaitUser { point: usize, sample: AlignSample },
    /// Finished with a fit (accept/reject it).
    Done(FitResult),
    Error(String),
    Aborted(String),
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Per-point acquisition stage (Python `_acquire`: nominal -> grid search -> manual).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Nominal,
    Grid(usize),
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    Idle,
    Slewing { az: f64, el: f64 },
    Settling { until_s: f64 },
    Capturing,
    Manual,
    Confirm(AlignSample),
    Finished,
}

pub struct AlignmentRunner {
    params: AlignmentParams,
    offsets: Vec<(i32, i32)>,
    phase: Phase,
    status: String,
    error: Option<String>,
    grid: Vec<(f64, f64)>,
    fit_points: Vec<(f64, f64)>,
    holdout_points: Vec<(f64, f64)>,
    pending: Vec<(f64, f64)>,
    append: bool,
    idx: usize,
    point_target: (f64, f64),
    stage: Stage,
    step: Step,
    samples: Vec<AlignSample>,
    backtest_samples: Vec<AlignSample>,
    failed_points: Vec<(f64, f64)>,
    fit: Option<FitResult>,
    consec_good: usize,
    progress: (usize, usize),
    current_target: Option<(f64, f64)>,
    paused: bool,
    last_solve: Option<AlignSample>,
    accepted: bool,
    started_s: Option<f64>,
}

impl AlignmentRunner {
    pub fn new(params: AlignmentParams) -> Self {
        let offsets = spiral_offsets(params.grid_search_cells);
        AlignmentRunner {
            params,
            offsets,
            phase: Phase::Idle,
            status: String::new(),
            error: None,
            grid: Vec::new(),
            fit_points: Vec::new(),
            holdout_points: Vec::new(),
            pending: Vec::new(),
            append: false,
            idx: 0,
            point_target: (0.0, 0.0),
            stage: Stage::Nominal,
            step: Step::Idle,
            samples: Vec::new(),
            backtest_samples: Vec::new(),
            failed_points: Vec::new(),
            fit: None,
            consec_good: 0,
            progress: (0, 0),
            current_target: None,
            paused: false,
            last_solve: None,
            accepted: false,
            started_s: None,
        }
    }

    // ---- accessors --------------------------------------------------------

    pub fn params(&self) -> &AlignmentParams {
        &self.params
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn status(&self) -> &str {
        &self.status
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    /// `(done, total)` fit-grid points (Python `AlignmentState.progress`).
    pub fn progress(&self) -> (usize, usize) {
        self.progress
    }
    pub fn is_running(&self) -> bool {
        matches!(self.phase, Phase::Running | Phase::Backtest)
    }
    /// Parked in manual recovery (Python `paused` / `manual_active`).
    pub fn paused(&self) -> bool {
        self.paused
    }
    pub fn grid(&self) -> &[(f64, f64)] {
        &self.grid
    }
    pub fn fit_points(&self) -> &[(f64, f64)] {
        &self.fit_points
    }
    pub fn holdout_points(&self) -> &[(f64, f64)] {
        &self.holdout_points
    }
    pub fn samples(&self) -> &[AlignSample] {
        &self.samples
    }
    pub fn backtest_samples(&self) -> &[AlignSample] {
        &self.backtest_samples
    }
    /// Points that never produced a solve (retryable via `start_retry`).
    pub fn failed_points(&self) -> &[(f64, f64)] {
        &self.failed_points
    }
    pub fn fit(&self) -> Option<&FitResult> {
        self.fit.as_ref()
    }
    /// Where the runner is slewing/searching right now (UI marker).
    pub fn current_target(&self) -> Option<(f64, f64)> {
        self.current_target
    }
    pub fn last_solve(&self) -> Option<&AlignSample> {
        self.last_solve.as_ref()
    }
    pub fn accepted(&self) -> bool {
        self.accepted
    }
    pub fn started_s(&self) -> Option<f64> {
        self.started_s
    }

    // ---- lifecycle --------------------------------------------------------

    /// Begin a fresh run: build the grid + holdout split, clear results.
    pub fn start(&mut self, now_s: f64) {
        self.reset_results();
        self.started_s = Some(now_s);
        let p = &self.params;
        let grid = fibonacci_targets(p.n_points, p.el_min_deg, p.el_max_deg, &p.keepout);
        if grid.len() < MIN_GRID_POINTS {
            self.fail_run("sky grid too small for this elevation band".to_string());
            return;
        }
        let (fit_pts, hold_pts) = holdout_split(&grid, p.holdout_frac);
        self.fit_points = fit_pts;
        self.holdout_points = hold_pts;
        self.pending = self.fit_points.clone();
        self.grid = grid;
        self.append = false;
        self.phase = Phase::Running;
        self.idx = 0;
        self.begin_point();
    }

    /// Re-run only the failed points, appending to the existing samples and
    /// refitting (no new grid, no backtest pass; the held-out RMS is
    /// re-evaluated against the refit). Returns false if nothing to retry or
    /// a run is in progress.
    pub fn start_retry(&mut self, now_s: f64) -> bool {
        if self.is_running() || self.failed_points.is_empty() {
            return false;
        }
        self.started_s = Some(now_s);
        self.pending = std::mem::take(&mut self.failed_points);
        self.append = true;
        self.error = None;
        self.accepted = false;
        self.consec_good = 0;
        self.phase = Phase::Running;
        self.idx = 0;
        self.begin_point();
        true
    }

    /// What to do now (pure; idempotent until an `on_*` event).
    pub fn next_action(&self, now_s: f64) -> Action {
        match self.phase {
            Phase::Idle => Action::Idle,
            Phase::Done => match &self.fit {
                Some(f) => Action::Done(f.clone()),
                None => Action::Error("done without a fit".to_string()),
            },
            Phase::Error => Action::Error(self.error.clone().unwrap_or_default()),
            Phase::Aborted => Action::Aborted(self.status.clone()),
            Phase::Running | Phase::Backtest => match self.step {
                Step::Slewing { az, el } => Action::Slew {
                    az_deg: az,
                    el_deg: el,
                },
                Step::Settling { until_s } => {
                    if now_s >= until_s {
                        Action::Capture
                    } else {
                        Action::WaitSettle { until_s }
                    }
                }
                Step::Capturing => Action::Capture,
                Step::Manual => Action::Manual {
                    az_deg: self.point_target.0,
                    el_deg: self.point_target.1,
                },
                Step::Confirm(sample) => Action::AwaitUser {
                    point: self.idx,
                    sample,
                },
                Step::Idle | Step::Finished => Action::Idle,
            },
        }
    }

    /// The mount has settled on the last `Action::Slew` target.
    pub fn on_arrived(&mut self, now_s: f64) {
        if !self.is_running() {
            return;
        }
        if let Step::Slewing { .. } = self.step {
            self.step = if self.params.settle_s > 0.0 {
                Step::Settling {
                    until_s: now_s + self.params.settle_s,
                }
            } else {
                Step::Capturing
            };
        }
    }

    /// Report a capture + plate-solve attempt. `c1_cmd, c2_cmd` = encoder
    /// position at solve time in the runner's frame (sky az/el for AltAz,
    /// HA/Dec for Eq). `None` = no frame / no solution.
    pub fn on_solve(
        &mut self,
        now_s: f64,
        result: Option<SolvedDirection>,
        c1_cmd: f64,
        c2_cmd: f64,
    ) {
        if !self.is_running() {
            return;
        }
        if !matches!(
            self.step,
            Step::Settling { .. } | Step::Capturing | Step::Manual
        ) {
            return;
        }
        match result {
            None => self.fail_stage(),
            Some(r) => {
                let (c1_cmd, c1_obs) = match self.params.mount_mode {
                    MountMode::AltAz => (c1_cmd.rem_euclid(360.0), r.c1_deg.rem_euclid(360.0)),
                    MountMode::Eq => (wrap180(c1_cmd), wrap180(r.c1_deg)),
                };
                let sample = AlignSample {
                    c1_cmd,
                    c2_cmd,
                    c1_obs,
                    c2_obs: r.c2_deg,
                    target_az: self.point_target.0,
                    target_el: self.point_target.1,
                    t_s: now_s,
                    n_matches: r.n_matches,
                    rms_arcsec: r.rms_arcsec,
                    fov_deg: r.fov_deg,
                    kind: if self.phase == Phase::Backtest {
                        SampleKind::Backtest
                    } else {
                        SampleKind::Fit
                    },
                };
                self.last_solve = Some(sample);
                if self.params.confirm_each_point {
                    self.step = Step::Confirm(sample);
                } else {
                    self.accept_sample(sample);
                }
            }
        }
    }

    /// Supervised mode: operator verdict on the sample in `Action::AwaitUser`.
    /// Reject behaves like a solve failure (grid search / manual / failed).
    pub fn on_user(&mut self, accept: bool) {
        if let Step::Confirm(sample) = self.step {
            if accept {
                self.accept_sample(sample);
            } else {
                self.fail_stage();
            }
        }
    }

    /// Abandon the current point (manual recovery / slow grid search) and move on.
    pub fn skip(&mut self) {
        if self.is_running()
            && !matches!(self.step, Step::Idle | Step::Finished)
        {
            self.point_failed();
        }
    }

    /// Abort. During sampling the run ends in `Phase::Aborted`; during the
    /// backtest pass the fit already exists and the run completes as Done
    /// with whatever backtest samples were collected (Python semantics).
    pub fn abort(&mut self) {
        match self.phase {
            Phase::Running => {
                self.leave_manual();
                self.current_target = None;
                self.step = Step::Finished;
                self.phase = Phase::Aborted;
                self.status = "aborted".to_string();
            }
            Phase::Backtest => {
                self.leave_manual();
                self.finish();
            }
            _ => {}
        }
    }

    /// Accept the final fit (Python `accept_alignment`): returns the result
    /// for the app to write into config / enable. None if no fit.
    pub fn accept(&mut self) -> Option<FitResult> {
        if self.phase != Phase::Done {
            return None;
        }
        let f = self.fit.clone()?;
        self.accepted = true;
        Some(f)
    }

    /// Discard the final fit and return to Idle (grid/samples are kept so a
    /// `start_retry` of failed points is still possible).
    pub fn reject(&mut self) {
        if self.phase == Phase::Done {
            self.fit = None;
            self.accepted = false;
            self.phase = Phase::Idle;
            self.step = Step::Idle;
            self.status = "fit rejected".to_string();
        }
    }

    // ---- internals --------------------------------------------------------

    fn reset_results(&mut self) {
        self.phase = Phase::Idle;
        self.status.clear();
        self.error = None;
        self.grid.clear();
        self.fit_points.clear();
        self.holdout_points.clear();
        self.pending.clear();
        self.append = false;
        self.idx = 0;
        self.point_target = (0.0, 0.0);
        self.stage = Stage::Nominal;
        self.step = Step::Idle;
        self.samples.clear();
        self.backtest_samples.clear();
        self.failed_points.clear();
        self.fit = None;
        self.consec_good = 0;
        self.progress = (0, 0);
        self.current_target = None;
        self.paused = false;
        self.last_solve = None;
        self.accepted = false;
    }

    fn total_points(&self) -> usize {
        match self.phase {
            Phase::Backtest => self.holdout_points.len(),
            _ => self.pending.len(),
        }
    }

    fn point_at(&self, i: usize) -> (f64, f64) {
        match self.phase {
            Phase::Backtest => self.holdout_points[i],
            _ => self.pending[i],
        }
    }

    fn slew_to(&mut self, az: f64, el: f64) {
        self.current_target = Some((az, el));
        self.step = Step::Slewing { az, el };
    }

    fn begin_point(&mut self) {
        let total = self.total_points();
        if self.idx >= total {
            self.finish_sampling();
            return;
        }
        let (az, el) = self.point_at(self.idx);
        self.point_target = (az, el);
        self.stage = Stage::Nominal;
        match self.phase {
            Phase::Running => {
                self.progress = (self.idx, total);
                self.status = format!(
                    "sampling {}/{}  az {:.0} el {:.0}",
                    self.idx + 1,
                    total,
                    az,
                    el
                );
            }
            Phase::Backtest => {
                self.status = format!("backtest {}/{}", self.idx + 1, total);
            }
            _ => {}
        }
        self.slew_to(az, el);
    }

    fn fail_stage(&mut self) {
        match self.stage {
            Stage::Nominal => self.try_grid(0),
            Stage::Grid(k) => self.try_grid(k + 1),
            Stage::Manual => self.step = Step::Manual, // keep waiting for the operator
        }
    }

    fn try_grid(&mut self, k: usize) {
        let n = self.params.grid_search_cells;
        let (az, el) = self.point_target;
        if k < n {
            let step = (0.5 * self.params.fov_deg).max(0.2);
            let cos_el = el.to_radians().cos().max(0.1);
            let (cx, cy) = self.offsets[k];
            let t_az = (az + (cx as f64 * step) / cos_el).rem_euclid(360.0);
            let t_el = (el + cy as f64 * step)
                .max(self.params.el_min_deg)
                .min(self.params.el_max_deg);
            self.stage = Stage::Grid(k);
            self.status = format!(
                "grid search {}/{} around az {:.0} el {:.0}",
                k + 1,
                n,
                az,
                el
            );
            self.slew_to(t_az, t_el);
        } else if self.params.pause_on_fail {
            self.stage = Stage::Manual;
            self.paused = true;
            self.current_target = Some((az, el));
            self.status = format!(
                "MANUAL: jog to stars (az {:.0} el {:.0}) - auto-captures on solve",
                az, el
            );
            self.step = Step::Manual;
        } else {
            self.point_failed();
        }
    }

    fn leave_manual(&mut self) {
        self.paused = false;
    }

    fn accept_sample(&mut self, sample: AlignSample) {
        self.leave_manual();
        match sample.kind {
            SampleKind::Backtest => self.backtest_samples.push(sample),
            SampleKind::Fit => self.samples.push(sample),
        }
        self.advance_point();
    }

    fn point_failed(&mut self) {
        self.leave_manual();
        self.failed_points.push(self.point_target);
        self.advance_point();
    }

    fn advance_point(&mut self) {
        if self.phase == Phase::Running {
            let total = self.pending.len();
            let target = self.params.target_rms_arcmin;
            if target > 0.0
                && !self.append
                && self.samples.len() >= EARLY_STOP_FLOOR
                && self.idx + 1 < total
            {
                let st = self.run_fit(&self.samples);
                let rms = st.rms_after_deg * 60.0;
                if rms <= target {
                    self.consec_good += 1;
                } else {
                    self.consec_good = 0;
                }
                if self.consec_good >= 2 {
                    self.status = format!(
                        "early-stop: fit RMS {:.2}' <= {:.2}' after {}/{}",
                        rms,
                        target,
                        self.samples.len(),
                        total
                    );
                    self.idx = total;
                    self.begin_point();
                    return;
                }
            }
        }
        self.idx += 1;
        self.begin_point();
    }

    fn finish_sampling(&mut self) {
        match self.phase {
            Phase::Running => {
                let total = self.pending.len();
                self.progress = (total, total);
                if self.samples.len() < MIN_SAMPLES {
                    let msg = format!(
                        "only {} good solves (need {}). {} point(s) failed - check cameras/DB/stars, then Retry Failed.",
                        self.samples.len(),
                        MIN_SAMPLES,
                        self.failed_points.len()
                    );
                    self.fail_run(msg);
                    return;
                }
                self.status = "fitting pointing model...".to_string();
                let stats = self.run_fit(&self.samples);
                self.fit = Some(FitResult {
                    mount_mode: self.params.mount_mode,
                    terms: stats.terms,
                    stats,
                    backtest_rms_deg: None,
                    n_failed: 0,
                });
                if !self.append {
                    self.phase = Phase::Backtest;
                    self.idx = 0;
                    self.begin_point();
                } else {
                    self.finish();
                }
            }
            Phase::Backtest => self.finish(),
            _ => {}
        }
    }

    fn finish(&mut self) {
        let bt_rms = if self.backtest_samples.is_empty() {
            None
        } else {
            self.fit
                .as_ref()
                .map(|f| self.run_backtest(&f.terms, &self.backtest_samples))
        };
        let n_failed = self.failed_points.len();
        let (before, after) = match self.fit.as_mut() {
            Some(f) => {
                f.backtest_rms_deg = bt_rms;
                f.n_failed = n_failed;
                (f.rms_before_arcmin(), f.rms_after_arcmin())
            }
            None => (0.0, 0.0),
        };
        self.leave_manual();
        self.current_target = None;
        self.step = Step::Finished;
        self.phase = Phase::Done;
        self.status = format!(
            "done: RMS {:.2}' (was {:.2}'){}",
            after,
            before,
            if n_failed > 0 {
                format!(", {} failed", n_failed)
            } else {
                String::new()
            }
        );
    }

    fn fail_run(&mut self, msg: String) {
        self.leave_manual();
        self.current_target = None;
        self.step = Step::Finished;
        self.phase = Phase::Error;
        self.status = msg.clone();
        self.error = Some(msg);
    }

    fn run_fit(&self, samples: &[AlignSample]) -> FitStats {
        let rows: Vec<[f64; 4]> = samples.iter().map(|s| s.as_row()).collect();
        let p = &self.params;
        match p.mount_mode {
            MountMode::AltAz => altaz::fit_altaz(
                &rows,
                p.remove_refraction,
                p.seed_terms,
                p.free_idx.clone(),
                p.robust,
                4.0,
                30.0 / 3600.0,
            ),
            MountMode::Eq => eq::fit_eq(
                &rows,
                p.lat_deg,
                p.seed_terms,
                p.free_idx.clone(),
                p.robust,
                4.0,
                30.0 / 3600.0,
            ),
        }
    }

    fn run_backtest(&self, terms: &[f64; N_TERMS], samples: &[AlignSample]) -> f64 {
        let rows: Vec<[f64; 4]> = samples.iter().map(|s| s.as_row()).collect();
        match self.params.mount_mode {
            MountMode::AltAz => altaz::backtest_altaz(terms, &rows, self.params.remove_refraction),
            MountMode::Eq => eq::backtest_eq(terms, &rows, self.params.lat_deg),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (ported intent of test_alignment.py / test_alignment_supervised.py)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // IA, IE, AN, AW, NPAE, CA, TF (same TRUTH as the Python tests)
    const TRUTH: [f64; 7] = [1.5, -0.6, 0.06, -0.04, 0.05, 0.03, 0.04];

    /// Tiny deterministic Gaussian source (xorshift64 + Box-Muller).
    struct Rng(u64);
    impl Rng {
        fn next_u(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_u().max(1e-12);
            let u2 = self.next_u();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    fn params(n: usize) -> AlignmentParams {
        AlignmentParams {
            n_points: n,
            el_min_deg: AlignmentParams::el_min_from_mask(10.0),
            fov_deg: 2.0,
            pause_on_fail: false,
            ..Default::default()
        }
    }

    /// The synthetic misaligned mount: where the boresight lands for a commanded (az, el).
    fn truth_solve(az: f64, el: f64) -> SolvedDirection {
        let (daz, del) = altaz::error(&TRUTH, az, el);
        SolvedDirection {
            c1_deg: (az + daz).rem_euclid(360.0),
            c2_deg: el + del,
            rms_arcsec: 0.4,
            n_matches: 12,
            fov_deg: 2.0,
        }
    }

    /// Headless "app": mount + clock + counters, driving the runner until it
    /// stops or needs the operator (Manual / AwaitUser) or ends.
    struct Harness {
        mount: (f64, f64),
        now: f64,
        captures: usize,
        slews: usize,
        slew_targets: Vec<(f64, f64)>,
    }

    impl Harness {
        fn new() -> Self {
            Harness {
                mount: (0.0, 0.0),
                now: 1000.0,
                captures: 0,
                slews: 0,
                slew_targets: Vec::new(),
            }
        }

        /// `solve(capture_no, mount) -> Option<SolvedDirection>`
        fn drive<F>(&mut self, r: &mut AlignmentRunner, mut solve: F) -> Action
        where
            F: FnMut(usize, (f64, f64)) -> Option<SolvedDirection>,
        {
            for _ in 0..100_000 {
                let a = r.next_action(self.now);
                match a {
                    Action::Slew { az_deg, el_deg } => {
                        self.mount = (az_deg, el_deg);
                        self.slews += 1;
                        self.slew_targets.push((az_deg, el_deg));
                        self.now += 2.0;
                        r.on_arrived(self.now);
                    }
                    Action::WaitSettle { until_s } => {
                        assert!(until_s > self.now);
                        self.now = until_s;
                    }
                    Action::Capture => {
                        self.captures += 1;
                        let res = solve(self.captures, self.mount);
                        self.now += 0.1;
                        r.on_solve(self.now, res, self.mount.0, self.mount.1);
                    }
                    other => return other,
                }
            }
            panic!("harness: runner did not finish");
        }
    }

    fn always_solves(_: usize, m: (f64, f64)) -> Option<SolvedDirection> {
        Some(truth_solve(m.0, m.1))
    }

    fn assert_terms_close(terms: &[f64; 7], truth: &[f64; 7], tol: f64) {
        for (i, (t, v)) in terms.iter().zip(truth.iter()).enumerate() {
            assert!(
                (t - v).abs() < tol,
                "term {} = {} vs truth {} (tol {})",
                i,
                t,
                v,
                tol
            );
        }
    }

    // ---- grid / helpers ---------------------------------------------------

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(python_round(4.5), 4.0);
        assert_eq!(python_round(5.5), 6.0);
        assert_eq!(python_round(13.5), 14.0);
        assert_eq!(python_round(-4.5), -4.0);
        assert_eq!(python_round(4.4), 4.0);
        assert_eq!(python_round(4.6), 5.0);
    }

    #[test]
    fn fibonacci_grid_matches_python_golden() {
        // Values generated with pointing_model.fibonacci_sky_grid (Python).
        let g = fibonacci_sky_grid(18, 15.0, 80.0);
        assert_eq!(g.len(), 18);
        let close = |a: (f64, f64), b: (f64, f64)| {
            (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
        };
        assert!(close(g[0], (137.507764, 76.463797)), "{:?}", g[0]);
        assert!(close(g[1], (275.015528, 72.497476)), "{:?}", g[1]);
        assert!(close(g[2], (52.523292, 69.258084)), "{:?}", g[2]);
        assert!(close(g[17], (315.139753, 41.102445)), "{:?}", g[17]);
        let (fit, hold) = holdout_split(&g, 0.25);
        // Python: n_hold = max(2, round(4.5)) = 4 (banker's), hold idx {0,4,9,14}
        assert_eq!(hold.len(), 4);
        assert_eq!(fit.len(), 14);
        assert!(close(hold[0], g[0]) && close(hold[1], g[4]) && close(hold[2], g[9]) && close(hold[3], g[14]));

        let g24 = fibonacci_sky_grid(24, 15.0, 80.0);
        assert_eq!(g24.len(), 24);
        assert!(close(g24[0], (137.507764, 78.284148)));
        assert!(close(g24[23], (60.186337, 41.278691)));
        let (_, hold24) = holdout_split(&g24, 0.25);
        assert_eq!(hold24.len(), 6);
        for (h, i) in hold24.iter().zip([0usize, 4, 8, 12, 16, 20]) {
            assert!(close(*h, g24[i]));
        }
        let g5 = fibonacci_sky_grid(5, 15.0, 80.0);
        assert_eq!(g5.len(), 5);
        assert!(close(g5[0], (0.0, 75.164888)));
    }

    #[test]
    fn fibonacci_grid_count_and_bounds() {
        for n in [8usize, 12, 18, 24, 40] {
            let g = fibonacci_sky_grid(n, 20.0, 75.0);
            assert!(g.len() <= n);
            assert!(g.len() >= n - 2, "n={} got {}", n, g.len());
            for &(az, el) in &g {
                assert!((0.0..360.0).contains(&az));
                assert!((20.0..=75.0).contains(&el));
            }
        }
        // Narrow band: fewer than n points, never panics.
        let g = fibonacci_sky_grid(30, 60.0, 62.0);
        assert!(g.len() < 30);
        assert!(fibonacci_sky_grid(0, 15.0, 80.0).is_empty());
    }

    #[test]
    fn fibonacci_keepout_excludes_zone() {
        let base = fibonacci_sky_grid(24, 15.0, 80.0);
        let ko = KeepOut {
            az_deg: base[3].0,
            el_deg: base[3].1,
            radius_deg: 10.0,
        };
        let g = fibonacci_targets(24, 15.0, 80.0, &[ko]);
        assert!(!g.iter().any(|&(az, el)| ko.contains(az, el)));
        // The zone removed at least the point it was centred on; the rest keep Python order.
        assert!(g.len() >= 20);
        assert_eq!(g[0], base[0]);
        assert_eq!(g[3], base[4]);
        // Empty keep-out list == Python grid.
        assert_eq!(fibonacci_targets(24, 15.0, 80.0, &[]), base);
    }

    #[test]
    fn spiral_offsets_count_unique_and_nearest_first() {
        let pts = spiral_offsets(100);
        assert_eq!(pts.len(), 100);
        let mut uniq = pts.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 100, "offsets must be unique");
        assert!(!pts.contains(&(0, 0)), "origin excluded");
        let mut ring1: Vec<(i32, i32)> = pts[..8].to_vec();
        ring1.sort();
        let mut expect: Vec<(i32, i32)> = (-1..=1)
            .flat_map(|x| (-1..=1).map(move |y| (x, y)))
            .filter(|p| *p != (0, 0))
            .collect();
        expect.sort();
        assert_eq!(ring1, expect);
        assert!(spiral_offsets(0).is_empty());
    }

    // ---- happy path ---------------------------------------------------------

    #[test]
    fn full_run_recovers_model_and_backtests() {
        let mut r = AlignmentRunner::new(params(24));
        let mut h = Harness::new();
        let mut rng = Rng(0x9E3779B97F4A7C15);
        r.start(h.now);
        let a = h.drive(&mut r, |_, m| {
            let mut s = truth_solve(m.0, m.1);
            let n = 8.0 / 3600.0;
            s.c1_deg = (s.c1_deg + rng.normal() * n).rem_euclid(360.0);
            s.c2_deg += rng.normal() * n;
            Some(s)
        });
        let fit = match a {
            Action::Done(f) => f,
            other => panic!("expected Done, got {:?} ({})", other, r.status()),
        };
        assert_eq!(r.phase(), Phase::Done);
        assert!(fit.rms_after_arcmin() < 0.5, "rms after {}", fit.rms_after_arcmin());
        assert!(fit.rms_before_arcmin() > fit.rms_after_arcmin());
        assert_terms_close(&fit.terms, &TRUTH, 0.05);
        let bt = fit.backtest_rms_deg.expect("backtest rms");
        assert!(bt * 60.0 < 1.0, "backtest {}", bt * 60.0);
        assert_eq!(fit.term_names(), altaz::TERM_NAMES);
        assert_eq!(r.samples().len(), r.fit_points().len());
        assert_eq!(r.backtest_samples().len(), r.holdout_points().len());
        assert!(r.failed_points().is_empty());
        assert_eq!(r.progress(), (18, 18));
        assert!(r.status().starts_with("done: RMS"), "{}", r.status());
        assert!(r.current_target().is_none());
        // Sample format: meta aligned with the sample, kind tagged, az wrapped.
        let s0 = r.samples()[0];
        assert_eq!(s0.n_matches, 12);
        assert_eq!(s0.kind, SampleKind::Fit);
        assert!((0.0..360.0).contains(&s0.c1_cmd) && (0.0..360.0).contains(&s0.c1_obs));
        assert_eq!(r.backtest_samples()[0].kind, SampleKind::Backtest);
        // Accept hands the fit back for the app to persist.
        let acc = r.accept().expect("accept");
        assert!(r.accepted());
        assert_terms_close(&acc.terms, &TRUTH, 0.05);
    }

    #[test]
    fn holdout_disjoint_from_fit() {
        let mut r = AlignmentRunner::new(params(20));
        let mut h = Harness::new();
        r.start(h.now);
        h.drive(&mut r, always_solves);
        assert!(!r.holdout_points().is_empty());
        for hp in r.holdout_points() {
            assert!(!r.fit_points().contains(hp));
        }
        assert_eq!(
            r.fit_points().len() + r.holdout_points().len(),
            r.grid().len()
        );
    }

    #[test]
    fn early_stop_halts_before_full_grid() {
        let mut p = params(40);
        p.target_rms_arcmin = 0.5;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, always_solves);
        let fit = match a {
            Action::Done(f) => f,
            other => panic!("expected Done, got {:?}", other),
        };
        assert!(r.samples().len() < r.fit_points().len());
        assert!(r.samples().len() <= 20, "{}", r.samples().len());
        assert_terms_close(&fit.terms, &TRUTH, 0.05);
        assert_eq!(r.progress(), (r.fit_points().len(), r.fit_points().len()));
    }

    #[test]
    fn settle_wait_then_capture() {
        let mut p = params(18);
        p.settle_s = 0.4;
        let mut r = AlignmentRunner::new(p);
        r.start(0.0);
        let (az, el) = match r.next_action(0.0) {
            Action::Slew { az_deg, el_deg } => (az_deg, el_deg),
            a => panic!("{:?}", a),
        };
        assert_eq!((az, el), r.fit_points()[0]);
        assert_eq!(r.current_target(), Some((az, el)));
        r.on_arrived(10.0);
        match r.next_action(10.0) {
            Action::WaitSettle { until_s } => assert!((until_s - 10.4).abs() < 1e-9),
            a => panic!("{:?}", a),
        }
        assert!(matches!(r.next_action(10.3), Action::WaitSettle { .. }));
        assert!(matches!(r.next_action(10.4), Action::Capture));
        // settle_s = 0 -> capture straight away
        let mut p0 = params(18);
        p0.settle_s = 0.0;
        let mut r0 = AlignmentRunner::new(p0);
        r0.start(0.0);
        r0.on_arrived(1.0);
        assert!(matches!(r0.next_action(1.0), Action::Capture));
        // Events out of order are ignored (on_solve before arrival).
        let mut r1 = AlignmentRunner::new(params(18));
        r1.start(0.0);
        r1.on_solve(0.0, Some(truth_solve(10.0, 40.0)), 10.0, 40.0);
        assert!(r1.samples().is_empty());
        assert!(matches!(r1.next_action(0.0), Action::Slew { .. }));
    }

    // ---- failures / grid search / retry -----------------------------------

    #[test]
    fn grid_search_recovers_first_point() {
        // Nominal capture of point 1 fails once; the grid search then solves,
        // so no point is lost and c1_cmd reflects the (offset) encoder position.
        let mut r = AlignmentRunner::new(params(24));
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, |n, m| if n <= 1 { None } else { always_solves(n, m) });
        assert!(matches!(a, Action::Done(_)), "{}", r.status());
        assert!(r.failed_points().is_empty(), "grid search should recover the point");
        assert_eq!(r.samples().len(), r.fit_points().len());
        let s0 = r.samples()[0];
        assert_eq!((s0.target_az, s0.target_el), r.fit_points()[0]);
        assert_ne!((s0.c1_cmd, s0.c2_cmd), r.fit_points()[0], "recorded at the offset cell");
        assert_eq!((s0.c1_cmd, s0.c2_cmd), h.slew_targets[1]);
    }

    #[test]
    fn total_solve_failure_errors_gracefully() {
        let mut p = params(20);
        p.grid_search_cells = 0;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, |_, _| None);
        match a {
            Action::Error(msg) => assert!(msg.contains("failed"), "{}", msg),
            other => panic!("expected Error, got {:?}", other),
        }
        assert_eq!(r.phase(), Phase::Error);
        assert!(r.fit().is_none());
        assert!(r.samples().len() < MIN_SAMPLES);
        assert_eq!(r.failed_points().len(), r.fit_points().len());
        assert!(r.error().unwrap().contains("failed"));
        // One capture per point (no grid search), no manual pause.
        assert_eq!(h.captures, r.fit_points().len());
    }

    #[test]
    fn failed_point_then_retry_appends_and_refits() {
        // 12-cell grid search: fail the nominal + all 12 offsets of point 1
        // (13 captures) so it lands in failed_points; everything after solves.
        let mut p = params(24);
        p.grid_search_cells = 12;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, |n, m| if n <= 13 { None } else { always_solves(n, m) });
        assert!(matches!(a, Action::Done(_)), "{}", r.status());
        assert_eq!(r.failed_points().len(), 1);
        assert_eq!(r.failed_points()[0], r.fit_points()[0]);
        assert!(r.samples().len() >= MIN_SAMPLES);
        assert!(r.status().ends_with(", 1 failed"), "{}", r.status());
        let n_before = r.samples().len();
        let bt_before = r.fit().unwrap().backtest_rms_deg;
        let n_bt = r.backtest_samples().len();

        // Retry pass: only the failed point, appended, refit, no new backtest pass.
        assert!(r.start_retry(h.now));
        assert_eq!(r.phase(), Phase::Running);
        assert_eq!(r.progress(), (0, 1));
        let a = h.drive(&mut r, always_solves);
        assert!(matches!(a, Action::Done(_)), "{}", r.status());
        assert!(r.failed_points().is_empty());
        assert_eq!(r.samples().len(), n_before + 1);
        assert_eq!(r.backtest_samples().len(), n_bt);
        let f = r.fit().unwrap();
        assert_terms_close(&f.terms, &TRUTH, 0.05);
        assert!(f.backtest_rms_deg.is_some() == bt_before.is_some());
        assert_eq!(f.n_failed, 0);
        // Nothing left to retry.
        assert!(!r.start_retry(h.now));
    }

    #[test]
    fn grid_cells_bound_capture_attempts_and_stay_in_band() {
        let mut p = params(18);
        p.grid_search_cells = 7;
        p.el_max_deg = 70.0;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        // Drive only until the first point fails (never solves).
        let mut fired = 0usize;
        loop {
            match r.next_action(h.now) {
                Action::Slew { az_deg, el_deg } => {
                    assert!((0.0..360.0).contains(&az_deg));
                    assert!((r.params().el_min_deg..=70.0).contains(&el_deg));
                    h.slew_targets.push((az_deg, el_deg));
                    h.mount = (az_deg, el_deg);
                    r.on_arrived(h.now);
                }
                Action::WaitSettle { until_s } => h.now = until_s,
                Action::Capture => {
                    fired += 1;
                    r.on_solve(h.now, None, h.mount.0, h.mount.1);
                    if !r.failed_points().is_empty() {
                        break;
                    }
                }
                a => panic!("{:?}", a),
            }
        }
        assert_eq!(fired, 8, "nominal + exactly grid_cells attempts");
        assert_eq!(h.slew_targets.len(), 8);
        // Offsets spiral around the nominal point; the first ring is ~0.5*fov/cos(el) away.
        let (az0, el0) = r.fit_points()[0];
        assert_eq!(h.slew_targets[0], (az0, el0));
        let step = (0.5 * r.params().fov_deg).max(0.2);
        for &(az, el) in &h.slew_targets[1..] {
            let daz = wrap180(az - az0).abs() * el0.to_radians().cos();
            assert!(daz <= step * 1.0001 + 1e-9, "az offset {}", daz);
            assert!((el - el0).abs() <= step + 1e-9);
        }
        assert!(r.status().starts_with("sampling 2/"), "{}", r.status());
    }

    // ---- supervised / manual recovery ------------------------------------

    #[test]
    fn manual_pause_auto_captures_on_solve() {
        let mut p = params(18);
        p.grid_search_cells = 0;
        p.pause_on_fail = true;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, |_, _| None);
        let (paz, pel) = match a {
            Action::Manual { az_deg, el_deg } => (az_deg, el_deg),
            other => panic!("expected Manual, got {:?}", other),
        };
        assert_eq!((paz, pel), r.fit_points()[0]);
        assert!(r.paused());
        assert!(r.status().starts_with("MANUAL:"), "{}", r.status());
        assert_eq!(r.phase(), Phase::Running);
        // Operator jogs; the app keeps re-solving — still parked while nothing solves.
        for _ in 0..3 {
            r.on_solve(h.now, None, paz + 1.0, pel - 0.5);
            assert!(matches!(r.next_action(h.now), Action::Manual { .. }));
        }
        // First solution is captured at the JOGGED encoder position and the run moves on.
        let jog = (paz + 1.3, pel - 0.7);
        r.on_solve(h.now, Some(truth_solve(jog.0, jog.1)), jog.0, jog.1);
        assert!(!r.paused());
        assert_eq!(r.samples().len(), 1);
        let s = r.samples()[0];
        assert!((s.c1_cmd - jog.0).abs() < 1e-12 && (s.c2_cmd - jog.1).abs() < 1e-12);
        assert_eq!((s.target_az, s.target_el), (paz, pel));
        assert!(r.failed_points().is_empty());
        match r.next_action(h.now) {
            Action::Slew { az_deg, el_deg } => assert_eq!((az_deg, el_deg), r.fit_points()[1]),
            a => panic!("{:?}", a),
        }
        // Let the rest solve normally: the run completes.
        let a = h.drive(&mut r, always_solves);
        assert!(matches!(a, Action::Done(_)), "{}", r.status());
    }

    #[test]
    fn manual_skip_marks_point_failed_and_continues() {
        let mut p = params(18);
        p.grid_search_cells = 0;
        p.pause_on_fail = true;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, |_, _| None);
        assert!(matches!(a, Action::Manual { .. }));
        r.skip();
        assert!(!r.paused());
        assert_eq!(r.failed_points().len(), 1);
        assert_eq!(r.failed_points()[0], r.fit_points()[0]);
        assert!(matches!(r.next_action(h.now), Action::Slew { .. }));
        let a = h.drive(&mut r, always_solves);
        assert!(matches!(a, Action::Done(_)), "{}", r.status());
        assert_eq!(r.fit().unwrap().n_failed, 1);
        // Skip is a no-op once finished.
        r.skip();
        assert_eq!(r.failed_points().len(), 1);
    }

    #[test]
    fn abort_during_sampling_and_in_manual() {
        let mut r = AlignmentRunner::new(params(18));
        let mut h = Harness::new();
        r.start(h.now);
        // A few points in, abort.
        let mut count = 0;
        let a = h.drive(&mut r, |n, m| {
            count = n;
            always_solves(n, m)
        });
        assert!(matches!(a, Action::Done(_)));
        assert!(count > 3);

        let mut r = AlignmentRunner::new(params(18));
        let mut h = Harness::new();
        r.start(h.now);
        h.now += 1.0;
        assert!(matches!(r.next_action(h.now), Action::Slew { .. }));
        r.on_arrived(h.now);
        r.abort();
        assert_eq!(r.phase(), Phase::Aborted);
        assert_eq!(r.status(), "aborted");
        assert!(matches!(r.next_action(h.now), Action::Aborted(_)));
        assert!(r.fit().is_none());
        assert!(r.current_target().is_none());
        // Further events are ignored.
        r.on_solve(h.now, Some(truth_solve(1.0, 40.0)), 1.0, 40.0);
        assert!(r.samples().is_empty());
        assert!(r.accept().is_none());

        // Abort while parked in manual recovery.
        let mut p = params(18);
        p.grid_search_cells = 0;
        p.pause_on_fail = true;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        assert!(matches!(h.drive(&mut r, |_, _| None), Action::Manual { .. }));
        r.abort();
        assert!(!r.paused());
        assert_eq!(r.phase(), Phase::Aborted);
    }

    #[test]
    fn abort_during_backtest_keeps_fit_and_finishes_done() {
        // Python: abort during the backtest pass breaks the loop and the run
        // still completes DONE with the fit (and whatever backtest samples exist).
        let mut r = AlignmentRunner::new(params(24));
        let mut h = Harness::new();
        r.start(h.now);
        // Drive fit points only: stop as soon as the phase flips to Backtest.
        loop {
            match r.next_action(h.now) {
                Action::Slew { az_deg, el_deg } => {
                    h.mount = (az_deg, el_deg);
                    r.on_arrived(h.now);
                }
                Action::WaitSettle { until_s } => h.now = until_s,
                Action::Capture => {
                    let s = truth_solve(h.mount.0, h.mount.1);
                    r.on_solve(h.now, Some(s), h.mount.0, h.mount.1);
                }
                a => panic!("{:?}", a),
            }
            if r.phase() == Phase::Backtest {
                break;
            }
        }
        assert!(r.fit().is_some());
        assert!(r.status().starts_with("backtest 1/"), "{}", r.status());
        r.abort();
        assert_eq!(r.phase(), Phase::Done);
        let f = r.fit().unwrap();
        assert!(f.backtest_rms_deg.is_none(), "no backtest samples were acquired");
        assert!(matches!(r.next_action(h.now), Action::Done(_)));
    }

    #[test]
    fn confirm_each_point_pauses_for_user_and_reject_acts_like_failure() {
        let mut p = params(18);
        p.confirm_each_point = true;
        p.grid_search_cells = 0;
        p.pause_on_fail = false;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let mut n_await = 0usize;
        let mut rejected_once = false;
        loop {
            let a = h.drive(&mut r, always_solves);
            match a {
                Action::AwaitUser { point, sample } => {
                    n_await += 1;
                    assert_eq!(sample.n_matches, 12);
                    assert!(matches!(r.next_action(h.now), Action::AwaitUser { .. }), "idempotent");
                    if point == 2 && !rejected_once {
                        rejected_once = true;
                        r.on_user(false); // -> failure path: no grid, no manual -> failed point
                        assert_eq!(r.failed_points().len(), 1);
                    } else {
                        let before = r.samples().len() + r.backtest_samples().len();
                        r.on_user(true);
                        assert_eq!(r.samples().len() + r.backtest_samples().len(), before + 1);
                    }
                }
                Action::Done(f) => {
                    assert_terms_close(&f.terms, &TRUTH, 0.05);
                    break;
                }
                other => panic!("{:?}", other),
            }
        }
        assert_eq!(n_await, r.fit_points().len() + r.holdout_points().len());
        assert_eq!(r.samples().len(), r.fit_points().len() - 1);
        assert_eq!(r.failed_points().len(), 1);
        // Reject the final fit: back to Idle, retry still possible.
        r.reject();
        assert_eq!(r.phase(), Phase::Idle);
        assert!(r.fit().is_none());
        assert!(matches!(r.next_action(h.now), Action::Idle));
        assert!(r.start_retry(h.now));
    }

    // ---- modes / options --------------------------------------------------

    #[test]
    fn eq_mode_recovers_eq_model() {
        // IH, ID, NP, CH, ME, MA, TF
        let truth = [0.5, -0.3, 0.04, 0.06, 0.03, -0.05, 0.08];
        let lat = 42.0;
        let mut p = params(30);
        p.mount_mode = MountMode::Eq;
        p.lat_deg = lat;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        // The "app" maps each sky target to a mount (HA, Dec) — any smooth,
        // well-spread mapping will do for the synthetic run — and reports the
        // solve in the same frame, distorted by the truth model.
        let to_mount = |az: f64, el: f64| (wrap180(az), el - 20.0);
        let mut a;
        loop {
            a = r.next_action(h.now);
            match a {
                Action::Slew { az_deg, el_deg } => {
                    h.mount = to_mount(az_deg, el_deg);
                    r.on_arrived(h.now);
                }
                Action::WaitSettle { until_s } => h.now = until_s,
                Action::Capture => {
                    let (hc, dc) = h.mount;
                    let (dh, dd) = eq::error(&truth, hc, dc, lat);
                    let sol = SolvedDirection {
                        c1_deg: hc + dh,
                        c2_deg: dc + dd,
                        rms_arcsec: 0.3,
                        n_matches: 15,
                        fov_deg: 2.0,
                    };
                    r.on_solve(h.now, Some(sol), hc, dc);
                }
                _ => break,
            }
        }
        let f = match a {
            Action::Done(f) => f,
            other => panic!("{:?} ({})", other, r.status()),
        };
        assert_eq!(f.mount_mode, MountMode::Eq);
        assert_eq!(f.term_names(), eq::TERM_NAMES);
        assert_terms_close(&f.terms, &truth, 0.05);
        assert!(f.rms_after_arcmin() < 0.5);
        assert!(f.backtest_rms_deg.unwrap() * 60.0 < 1.0);
        // HA stored wrapped to [-180, 180).
        for s in r.samples() {
            assert!((-180.0..180.0).contains(&s.c1_cmd));
        }
    }

    #[test]
    fn remove_refraction_recovers_identity_model() {
        // Identity mount + refracted observations, el_min lifted to 30 so the
        // refraction stays modest: stripping it recovers ~zero terms, and the
        // same samples fit WITHOUT stripping leave a larger residual.
        let mut p = params(18);
        p.el_min_deg = 30.0;
        p.remove_refraction = true;
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let a = h.drive(&mut r, |_, m| {
            Some(SolvedDirection {
                c1_deg: m.0,
                c2_deg: m.1 + altaz::bennett_refraction_deg(m.1, 1010.0, 10.0),
                rms_arcsec: 0.3,
                n_matches: 10,
                fov_deg: 2.0,
            })
        });
        let f = match a {
            Action::Done(f) => f,
            other => panic!("{:?}", other),
        };
        assert!(f.rms_after_arcmin() < 0.5, "{}", f.rms_after_arcmin());
        let rows: Vec<[f64; 4]> = r.samples().iter().map(|s| s.as_row()).collect();
        let keep = altaz::fit_altaz(&rows, false, [0.0; 7], None, true, 4.0, 30.0 / 3600.0);
        assert!(keep.rms_after_deg * 60.0 > f.rms_after_arcmin());
    }

    #[test]
    fn partial_fit_refits_only_free_terms() {
        // Quick nightly refit: seed the mechanical terms, solve IA/IE only from 8 points.
        let mut p = params(8);
        p.seed_terms = TRUTH;
        p.seed_terms[0] = 0.0;
        p.seed_terms[1] = 0.0;
        p.free_idx = Some(vec![0, 1]);
        let mut r = AlignmentRunner::new(p);
        let mut h = Harness::new();
        r.start(h.now);
        let f = match h.drive(&mut r, always_solves) {
            Action::Done(f) => f,
            other => panic!("{:?} ({})", other, r.status()),
        };
        assert_terms_close(&f.terms, &TRUTH, 1e-6);
    }

    #[test]
    fn grid_too_small_errors_before_slewing() {
        let mut r = AlignmentRunner::new(params(5));
        r.start(0.0);
        assert_eq!(r.phase(), Phase::Error);
        assert!(r.error().unwrap().contains("too small"));
        assert!(matches!(r.next_action(0.0), Action::Error(_)));
        // Restart with a proper grid works from the same runner.
        let mut r = AlignmentRunner::new(params(18));
        r.start(0.0);
        assert_eq!(r.phase(), Phase::Running);
        assert!(matches!(r.next_action(0.0), Action::Slew { .. }));
        assert_eq!(r.progress(), (0, r.fit_points().len()));
        assert!(r.status().starts_with("sampling 1/"), "{}", r.status());
    }

    #[test]
    fn idle_runner_ignores_events() {
        let mut r = AlignmentRunner::new(params(18));
        assert!(matches!(r.next_action(0.0), Action::Idle));
        r.on_arrived(0.0);
        r.on_solve(0.0, Some(truth_solve(1.0, 2.0)), 1.0, 2.0);
        r.on_user(true);
        r.skip();
        r.abort();
        assert_eq!(r.phase(), Phase::Idle);
        assert!(r.accept().is_none());
        assert!(!r.start_retry(0.0));
    }
}
