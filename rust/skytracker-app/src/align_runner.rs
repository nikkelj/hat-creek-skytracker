//! Adapter between the align worker and skytracker-pointing's alignment
//! runner (the faithful port of alignment.py: Fibonacci grid with holdout,
//! slew -> settle -> capture/solve, spiral grid search on failure, running
//! fit with early stop, backtest pass, supervised confirm). The runner is a
//! pure state machine; this adapter turns its Actions into worker steps
//! (goto, wait, solve) and reports a flat snapshot for the UI.

use crate::state::Config;
use skytracker_pointing::alignment::{Action, AlignmentParams, AlignmentRunner, MountMode, Phase, SolvedDirection};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    Idle,
    Slew { az: f64, el: f64 },
    Wait,
    Solve,
}

pub struct RunnerSnapshot {
    pub running: bool,
    pub manual: bool,
    pub failed: Vec<(f64, f64)>,
    pub status: String,
    pub action: String,
    pub point: usize,
    pub n_points: usize,
    pub targets: Vec<(f64, f64)>,
    /// (mount_az, mount_el, true_az, true_el, residual_arcsec)
    pub samples: Vec<(f64, f64, f64, f64, f64)>,
    pub terms: Option<[f64; 7]>,
    pub rms_arcsec: Option<f64>,
    pub new_log: Vec<String>,
}

pub struct Runner {
    cfg: Config,
    inner: AlignmentRunner,
    log: Vec<String>,
    last_goto: Instant,
    last_manual_solve: Instant,
    slew_target: Option<(f64, f64)>,
    slew_started: f64,
    last_pos: (f64, f64),
    last_status: String,
    settle_tol: f64,
    settle_cycles: usize,
    arrive_count: usize,
    n_samples_logged: usize,
    done_logged: bool,
}

fn params(cfg: &Config, n_points: usize, supervised: bool) -> AlignmentParams {
    let mut p = AlignmentParams::default();
    p.mount_mode = if cfg.mount_mode.eq_ignore_ascii_case("eq") { MountMode::Eq } else { MountMode::AltAz };
    p.lat_deg = cfg.lat_deg;
    p.n_points = n_points.clamp(4, 60);
    p.el_min_deg = AlignmentParams::el_min_from_mask(cfg.elevation_mask_deg);
    p.settle_s = cfg.alignment_settle_s.max(0.0);
    let sc = &cfg.cam[cfg.plate_solve_camera_index.min(cfg.cam.len().saturating_sub(1))];
    p.fov_deg = sc.fov_deg(crate::camera::sim_frame_size(sc).0 as f64).max(0.2);
    p.grid_search_cells = cfg.alignment_grid_search_cells.max(1);
    p.target_rms_arcmin = cfg.alignment_target_rms_arcmin.max(0.0);
    // pause_on_fail parks in Manual for a paddle rescue (Align screen paddles /
    // gamepad); off = failed points are skipped (unattended runs).
    p.pause_on_fail = cfg.alignment_pause_on_fail;
    p.confirm_each_point = supervised;
    p
}

impl Runner {
    pub fn new(cfg: &Config) -> Self {
        Runner {
            cfg: cfg.clone(),
            inner: AlignmentRunner::new(params(cfg, 12, false)),
            log: Vec::new(),
            last_goto: Instant::now() - Duration::from_secs(10),
            last_manual_solve: Instant::now() - Duration::from_secs(10),
            slew_target: None,
            slew_started: 0.0,
            last_pos: (0.0, 0.0),
            last_status: String::new(),
            settle_tol: cfg.alignment_settle_tol_deg.max(0.005),
            settle_cycles: cfg.alignment_settle_cycles.max(1),
            arrive_count: 0,
            n_samples_logged: 0,
            done_logged: false,
        }
    }

    pub fn start(&mut self, n_points: usize, supervised: bool, az: f64, el: f64, now: f64) {
        let p = params(&self.cfg, n_points, supervised);
        self.log.push(format!(
            "alignment started: {} points (holdout {:.0}%), el {:.0}-{:.0}°, settle {:.1} s{}",
            p.n_points,
            p.holdout_frac * 100.0,
            p.el_min_deg,
            p.el_max_deg,
            p.settle_s,
            if supervised { ", confirm each point" } else { "" }
        ));
        self.inner = AlignmentRunner::new(p);
        self.inner.start(now);
        self.last_pos = (az, el);
        self.slew_target = None;
        self.n_samples_logged = 0;
        self.done_logged = false;
        if let Some(e) = self.inner.error() {
            self.log.push(format!("alignment error: {e}"));
        }
    }

    pub fn retry_failed(&mut self, now: f64) -> bool {
        let n = self.inner.failed_points().len();
        if self.inner.start_retry(now) {
            self.log.push(format!("retrying {n} failed point(s)"));
            true
        } else {
            false
        }
    }

    pub fn skip(&mut self) {
        self.inner.skip();
        self.log.push("point skipped".into());
    }

    /// Quick Refit: few points, seed with the live model, fit IA/IE only.
    pub fn start_quick(&mut self, seed: [f64; 7], az: f64, el: f64, now: f64) {
        let mut p = params(&self.cfg, 6, false);
        p.seed_terms = seed;
        p.free_idx = Some(vec![0, 1]);
        self.log.push("quick refit: 6 points, IA/IE free, other terms frozen".into());
        self.inner = AlignmentRunner::new(p);
        self.inner.start(now);
        self.last_pos = (az, el);
        self.slew_target = None;
        self.n_samples_logged = 0;
        self.done_logged = false;
    }

    pub fn abort(&mut self) {
        if self.inner.is_running() {
            self.inner.abort();
            self.log.push("alignment aborted".into());
        }
    }

    pub fn user(&mut self, accept: bool) {
        self.inner.on_user(accept);
        self.log.push(format!("point {}: {}", self.inner.progress().0 + 1, if accept { "accepted" } else { "rejected" }));
    }

    pub fn terms(&self) -> Option<[f64; 7]> {
        self.inner.fit().map(|f| f.terms)
    }

    /// Per-tick: what should the worker do now? `az/el` = the mount's sky
    /// pointing (what the runner calls the commanded/encoder position).
    pub fn step(&mut self, now: f64, az: f64, el: f64) -> Step {
        let moved = (az - self.last_pos.0).abs() > 1e-4 || (el - self.last_pos.1).abs() > 1e-4;
        self.last_pos = (az, el);
        match self.inner.next_action(now) {
            Action::Idle => Step::Idle,
            Action::Slew { az_deg, el_deg } => {
                if self.slew_target != Some((az_deg, el_deg)) {
                    self.slew_target = Some((az_deg, el_deg));
                    self.slew_started = now;
                    self.last_goto = Instant::now() - Duration::from_secs(10);
                }
                let daz = ((az - az_deg + 540.0).rem_euclid(360.0) - 180.0).abs();
                let del = (el - el_deg).abs();
                // Asymptotic settle (alignment.py): within tol AND not moving
                // for `settle_cycles` consecutive reads.
                if daz < self.settle_tol && del < self.settle_tol && !moved {
                    self.arrive_count += 1;
                } else {
                    self.arrive_count = 0;
                }
                if self.arrive_count >= self.settle_cycles || now - self.slew_started > 90.0 {
                    if now - self.slew_started > 90.0 {
                        self.log.push(format!("slew to {az_deg:.1}/{el_deg:.1} timed out -- capturing anyway"));
                    }
                    self.arrive_count = 0;
                    self.inner.on_arrived(now);
                    Step::Wait
                } else if self.last_goto.elapsed() > Duration::from_millis(1500) {
                    self.last_goto = Instant::now();
                    Step::Slew { az: az_deg, el: el_deg }
                } else {
                    Step::Wait
                }
            }
            Action::WaitSettle { .. } => Step::Wait,
            Action::Capture => Step::Solve,
            Action::Manual { .. } => {
                if self.last_manual_solve.elapsed() > Duration::from_secs(1) {
                    self.last_manual_solve = Instant::now();
                    Step::Solve
                } else {
                    Step::Wait
                }
            }
            Action::AwaitUser { .. } => Step::Wait,
            Action::Done(_) | Action::Error(_) | Action::Aborted(_) => Step::Idle,
        }
    }

    /// Solve result for the latest capture: (true_az, true_el, rms_arcsec).
    pub fn on_solve(&mut self, now: f64, result: Option<(f64, f64, f64)>, mount_az: f64, mount_el: f64) {
        let sd = result.map(|(a, e, rms)| SolvedDirection {
            c1_deg: a,
            c2_deg: e,
            rms_arcsec: rms,
            n_matches: 0,
            fov_deg: self.inner.params().fov_deg,
        });
        self.inner.on_solve(now, sd, mount_az, mount_el);
    }

    pub fn snapshot(&mut self) -> RunnerSnapshot {
        // Log transitions the runner reports through its status line.
        let status = self.inner.status().to_string();
        if status != self.last_status && !status.is_empty() {
            self.log.push(status.clone());
            self.last_status = status.clone();
        }
        let samples: Vec<(f64, f64, f64, f64, f64)> = self
            .inner
            .samples()
            .iter()
            .chain(self.inner.backtest_samples().iter())
            .map(|s| {
                let daz = ((s.c1_obs - s.c1_cmd + 540.0).rem_euclid(360.0) - 180.0) * s.c2_cmd.to_radians().cos();
                let del = s.c2_obs - s.c2_cmd;
                (s.c1_cmd, s.c2_cmd, s.c1_obs, s.c2_obs, (daz * daz + del * del).sqrt() * 3600.0)
            })
            .collect();
        if samples.len() > self.n_samples_logged {
            for s in &samples[self.n_samples_logged..] {
                self.log.push(format!("sample {}: mount {:.3}/{:.3}  true {:.3}/{:.3}  Δ {:.0}″", samples.len(), s.0, s.1, s.2, s.3, s.4));
            }
            self.n_samples_logged = samples.len();
        }
        let fit = self.inner.fit();
        if let (Some(f), false) = (fit, self.done_logged) {
            if matches!(self.inner.phase(), Phase::Done) {
                self.log.push(format!(
                    "fit: {} samples, rms {:.2}′ -> {:.2}′{}{}",
                    f.stats.n_samples,
                    f.rms_before_arcmin(),
                    f.rms_after_arcmin(),
                    f.backtest_rms_deg.map(|b| format!(", backtest {:.2}′", b * 60.0)).unwrap_or_default(),
                    if f.n_failed > 0 { format!(", {} failed", f.n_failed) } else { String::new() }
                ));
                self.done_logged = true;
            }
        }
        let (done, total) = self.inner.progress();
        let manual = matches!(self.inner.next_action(0.0), Action::Manual { .. });
        RunnerSnapshot {
            manual,
            failed: self.inner.failed_points().to_vec(),
            running: self.inner.is_running(),
            status: if status.contains("confirm") || matches!(self.inner.next_action(0.0), Action::AwaitUser { .. }) {
                format!("{status} · accept?")
            } else {
                status
            },
            action: format!("{:?}", self.inner.phase()).to_lowercase(),
            point: done,
            n_points: total,
            targets: self.inner.grid().to_vec(),
            samples,
            terms: fit.map(|f| f.terms),
            rms_arcsec: fit.map(|f| f.stats.rms_after_deg * 3600.0),
            new_log: std::mem::take(&mut self.log),
        }
    }
}
