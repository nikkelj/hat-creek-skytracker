//! Hat Creek Skytracker — native app (Phase 7 of the Rust port).
//!
//! eframe + egui on wgpu at a 120 Hz display target; workers publish
//! snapshots (sky, passes, mount, camera, solve, alignment), the UI renders
//! them and sends commands. Reads config.json, tle_cache.tle, de421.bsp,
//! hip_main.dat, tyc_main.dat from the repo root (found from SKYTRACKER_ROOT,
//! the working directory, or the checkout the binary was built from):
//!
//!   cargo run --release -p skytracker-app
//!
//! SKYTRACKER_SCREENSHOT_DIR=<dir> cycles every screen and saves PNGs.

mod adsb;
mod align;
mod align_runner;
mod camera;
mod catalogs;
mod deepsky;
mod filterwheel;
mod launches;
mod mount;
mod mount3d;
mod passes_bridge;
mod replay;
mod screens;
mod sky;
mod state;
mod theme;
mod ui;

use state::{AlignCmd, CamCmd, MountCmd, Shared};
use std::sync::Arc;
use std::time::{Duration, Instant};
use theme::{ACCENT, DIM, GREEN, RED, TEXT_2};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Screen {
    Track,
    Cameras,
    Passes,
    Align,
    Replay,
    Mount3d,
    Sim,
    Config,
}

impl Screen {
    const ALL: [Screen; 8] = [Screen::Track, Screen::Cameras, Screen::Passes, Screen::Align, Screen::Replay, Screen::Mount3d, Screen::Sim, Screen::Config];
    fn label(self) -> &'static str {
        match self {
            Screen::Track => "Track",
            Screen::Cameras => "Cameras",
            Screen::Passes => "Passes",
            Screen::Align => "Align",
            Screen::Replay => "Replay",
            Screen::Mount3d => "Mount 3D",
            Screen::Sim => "Sim",
            Screen::Config => "Config",
        }
    }
    fn slug(self) -> &'static str {
        match self {
            Screen::Track => "track",
            Screen::Cameras => "cameras",
            Screen::Passes => "passes",
            Screen::Align => "align",
            Screen::Replay => "replay",
            Screen::Mount3d => "mount3d",
            Screen::Sim => "sim",
            Screen::Config => "config",
        }
    }
}

struct Screenshots {
    dir: std::path::PathBuf,
    idx: usize,
    switched_at: Instant,
    requested: bool,
    solve_sent: bool,
}

struct App {
    shared: Arc<Shared>,
    tx: crossbeam_channel::Sender<MountCmd>,
    tx_cam: crossbeam_channel::Sender<CamCmd>,
    tx_align: crossbeam_channel::Sender<AlignCmd>,
    tx_fw: crossbeam_channel::Sender<state::FwCmd>,
    ui: ui::UiState,
    cameras: screens::CamerasState,
    passes: screens::PassesState,
    align: screens::AlignState,
    config: screens::ConfigState,
    mount3d: mount3d::Mount3dView,
    replay: replay::ReplayState,
    screen: Screen,
    started: Instant,
    shots: Option<Screenshots>,
    autotest: Option<Autotest>,
    replay_test: Option<ReplayTest>,
}

struct ReplayTest {
    needle: String,
    loaded: bool,
    last_log: f64,
    started: Option<Instant>,
    shown_hist: Vec<(usize, usize)>,
}

struct Autotest {
    duration: f64,
    armed: bool,
    captured: bool,
    dumped: bool,
    solved: bool,
    aligned: bool,
    last_log: f64,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        shared: Arc<Shared>,
        tx: crossbeam_channel::Sender<MountCmd>,
        tx_cam: crossbeam_channel::Sender<CamCmd>,
        tx_align: crossbeam_channel::Sender<AlignCmd>,
        tx_fw: crossbeam_channel::Sender<state::FwCmd>,
    ) -> Self {
        theme::install(&cc.egui_ctx);
        let shots = std::env::var("SKYTRACKER_SCREENSHOT_DIR").ok().map(|d| Screenshots {
            dir: std::path::PathBuf::from(d),
            idx: 0,
            switched_at: Instant::now(),
            requested: false,
            solve_sent: false,
        });
        App {
            shared,
            tx,
            tx_cam,
            tx_align,
            tx_fw,
            ui: ui::UiState::default(),
            cameras: screens::CamerasState::default(),
            passes: screens::PassesState::default(),
            align: screens::AlignState::default(),
            config: screens::ConfigState::default(),
            mount3d: mount3d::Mount3dView::default(),
            replay: replay::ReplayState::default(),
            screen: Screen::Track,
            started: Instant::now(),
            shots,
            autotest: std::env::var("SKYTRACKER_AUTOTEST").ok().and_then(|v| v.parse::<f64>().ok()).map(|d| Autotest { duration: d, armed: false, captured: false, dumped: false, solved: false, aligned: false, last_log: 0.0 }),
            replay_test: std::env::var("SKYTRACKER_REPLAY_TEST").ok().map(|n| ReplayTest { needle: n, loaded: false, last_log: 0.0, started: None, shown_hist: Vec::new() }),
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context, ui_fps: f64) {
        egui::TopBottomPanel::top("top").exact_height(34.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Skytracker").font(theme::sans(13.0)).color(TEXT_2));
                ui.add_space(10.0);
                for s in Screen::ALL {
                    let active = self.screen == s;
                    let text = egui::RichText::new(s.label()).font(theme::sans(12.5)).color(if active { theme::TEXT } else { TEXT_2 });
                    let b = egui::Button::new(text)
                        .fill(if active { theme::with_alpha(ACCENT, 40) } else { egui::Color32::TRANSPARENT })
                        .stroke(egui::Stroke::NONE)
                        .rounding(egui::Rounding::same(4.0));
                    if ui.add(b).clicked() {
                        self.screen = s;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(format!("{ui_fps:5.1} Hz")).font(theme::mono(10.5)).color(DIM));
                    ui.label(egui::RichText::new(format!("up {:.0} s", self.started.elapsed().as_secs_f64())).font(theme::mono(10.5)).color(DIM));
                    let m = self.shared.mount.load();
                    let col = match m.mode.as_str() {
                        "RATE" => theme::AMBER,
                        "PROGRAM" | "HANDOFF" => ACCENT,
                        "HOTSPOT" => GREEN,
                        _ => TEXT_2,
                    };
                    theme::tag(ui, &m.mode, col);
                    if m.loop_dead {
                        theme::tag(ui, "LOOP DEAD", RED);
                    }
                    let sky = self.shared.sky.load();
                    ui.label(egui::RichText::new(&sky.utc_iso).font(theme::mono(11.5)).color(theme::TEXT));
                });
            });
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 120 Hz display target: ask for the next repaint immediately after
        // this one; the frame-time ring shows what we actually achieve.
        let now = Instant::now();
        let dt = now.duration_since(self.ui.last_frame).as_secs_f64();
        self.ui.last_frame = now;
        if self.ui.frame_times.len() >= 240 {
            self.ui.frame_times.pop_front();
        }
        self.ui.frame_times.push_back(dt);
        let avg = self.ui.frame_times.iter().sum::<f64>() / self.ui.frame_times.len().max(1) as f64;
        let ui_fps = if avg > 0.0 { 1.0 / avg } else { 0.0 };
        ctx.request_repaint_after(Duration::from_micros(8_300));
        // Pin the dark theme: egui follows the OS theme by default.
        ctx.options_mut(|o| o.theme_preference = egui::ThemePreference::Dark);

        // Screenshot tour.
        if let Some(shots) = self.shots.as_mut() {
            for ev in ctx.input(|i| i.raw.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = ev {
                    let path = shots.dir.join(format!("rust_app_{}.png", self.screen.slug()));
                    let _ = std::fs::create_dir_all(&shots.dir);
                    let w = image.width() as u32;
                    let h = image.height() as u32;
                    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
                    for p in &image.pixels {
                        rgba.extend_from_slice(&[p.r(), p.g(), p.b(), 255]);
                    }
                    if let Err(e) = image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
                        eprintln!("screenshot failed: {e}");
                    } else {
                        eprintln!("screenshot -> {}", path.display());
                    }
                    shots.idx += 1;
                    shots.requested = false;
                    shots.solve_sent = false;
                    shots.switched_at = Instant::now();
                    if shots.idx >= Screen::ALL.len() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    } else {
                        self.screen = Screen::ALL[shots.idx];
                    }
                }
            }
            if shots.idx < Screen::ALL.len() {
                self.screen = Screen::ALL[shots.idx];
                let dwell = match shots.idx { 0 => 12.0, 1 => 5.0, 3 => 8.0, 4 => 9.0, _ => 4.0 };
                if shots.idx == 0 && self.ui.selected.is_none() && shots.switched_at.elapsed().as_secs_f64() > 3.0 {
                    // Select something for the track screen so arcs + labels show.
                    if let Some(s) = pick_demo_target(&self.shared) {
                        self.ui.selected = Some(s.clone());
                        let _ = self.tx.send(MountCmd::SelectTarget(Some(s)));
                        let _ = self.tx.send(MountCmd::SetMode("PROGRAM".into()));
                    }
                }
                if Screen::ALL[shots.idx] == Screen::Cameras && !shots.solve_sent && shots.switched_at.elapsed().as_secs_f64() > 1.0 {
                    // Show the weighted combined overlay in the tour.
                    self.cameras.combined = true;
                    self.cameras.match_scale = true;
                    self.cameras.weights = vec![1.0, 0.6, 0.0];
                    shots.solve_sent = true;
                }
                if Screen::ALL[shots.idx] == Screen::Replay && !shots.solve_sent && shots.switched_at.elapsed().as_secs_f64() > 1.0 {
                    // Load the newest real run and play it for the screenshot.
                    if let Some(idx) = self.replay.find_run("YAOGAN").or_else(|| if self.replay.library_len().unwrap_or(0) > 0 { Some(0) } else { None }) {
                        self.replay.load_run(idx, Some(ctx.clone()));
                        self.replay.playing = true;
                        shots.solve_sent = true;
                    }
                }
                if Screen::ALL[shots.idx] == Screen::Align && shots.switched_at.elapsed().as_secs_f64() > 0.5 {
                    // Keep solving once a second until one sticks (sparse sim fields).
                    let sv = self.shared.solve.load();
                    if !sv.last_ok && !sv.busy && (!shots.solve_sent || shots.switched_at.elapsed().as_secs_f64() % 1.0 < 0.03) {
                        let _ = self.tx_align.send(AlignCmd::SolveNow);
                        shots.solve_sent = true;
                    }
                }
                if !shots.requested && shots.switched_at.elapsed().as_secs_f64() > dwell {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                    shots.requested = true;
                }
            }
        }

        // Headless closed-loop check: SKYTRACKER_AUTOTEST=<seconds> injects a
        // sim misalignment, selects a LEO target, arms HANDOFF and logs the
        // loop once a second, then exits.
        if let Some(at) = self.autotest.as_mut() {
            let t = self.started.elapsed().as_secs_f64();
            if !at.armed && t > 4.0 {
                // SKYTRACKER_AUTOTEST_TARGET=body:moon|star:HIP32349|dso:M031|adsb:<icao>
                // tracks that key in PROGRAM instead of a satellite in HANDOFF.
                if let Ok(key) = std::env::var("SKYTRACKER_AUTOTEST_TARGET") {
                    self.ui.selected = Some(key.clone());
                    let _ = self.tx.send(MountCmd::SelectTarget(Some(key.clone())));
                    let _ = self.tx.send(MountCmd::SetMode("PROGRAM".into()));
                    eprintln!("autotest: PROGRAM on {key} at t={t:.1}s");
                    at.armed = true;
                } else if let Some(s) = pick_demo_target(&self.shared) {
                    let mut sim = (**self.shared.sim.load()).clone();
                    sim.misalign_az_deg = 0.04;
                    sim.misalign_el_deg = -0.03;
                    self.shared.sim.store(Arc::new(sim));
                    self.ui.selected = Some(s.clone());
                    let _ = self.tx.send(MountCmd::SelectTarget(Some(s.clone())));
                    let _ = self.tx.send(MountCmd::SetMode("HANDOFF".into()));
                    eprintln!("autotest: target {s}, misalignment +0.040/-0.030 deg, HANDOFF armed at t={t:.1}s");
                    at.armed = true;
                }
            }
            if at.armed && !at.captured && t > 9.0 {
                let _ = self.tx_cam.send(CamCmd::Arm);
                at.captured = true;
            }
            if at.captured && !at.dumped && t > 11.0 {
                let _ = self.tx_cam.send(CamCmd::Dump { name: "autotest".into() });
                at.dumped = true;
            }
            if at.armed && !at.solved && t > 20.0 {
                let _ = self.tx_align.send(AlignCmd::SolveNow);
                at.solved = true;
            }
            if at.armed && !at.aligned && t > 26.0 {
                let _ = self.tx.send(MountCmd::SetMode("STANDBY".into()));
                let _ = self.tx_align.send(AlignCmd::Start { n_points: 8, supervised: false });
                at.aligned = true;
            }
            if t - at.last_log >= 1.0 {
                at.last_log = t;
                if at.solved {
                    let sv = self.shared.solve.load();
                    let al = self.shared.align.load();
                    eprintln!("autotest   solve: {} | align: {} ({}/{}) rms={:?}", sv.message, al.status, al.point, al.n_points, al.rms_arcsec.map(|r| (r * 10.0).round() / 10.0));
                    for l in al.log.iter().rev().take(1) {
                        eprintln!("autotest   align-log: {l}");
                    }
                }
                let m = self.shared.mount.load();
                let c = self.shared.cam(self.shared.hotspot_slot());
                if let Some(tt) = self.shared.sky.load().target.as_ref() {
                    eprintln!("autotest   target {} ({}): az {:.3} el {:.3} rates {:+.4}/{:+.4} °/s  setpoint {:?}", tt.name, tt.key, tt.az, tt.el, tt.az_rate, tt.el_rate, m.setpoint.map(|(a, e)| ((a * 1000.0).round() / 1000.0, (e * 1000.0).round() / 1000.0)));
                }
                eprintln!(
                    "autotest t={t:5.1}s mode={:<8} az={:8.3} el={:7.3} err={:+.4}/{:+.4} rate={:+}/{:+} hs={} snr={:.1} cen={:?} handoff={} loop={:.1}Hz cam={:.0}fps",
                    m.mode, m.az, m.el, m.az_error, m.el_error, m.rate_cmd.0, m.rate_cmd.1, m.hotspot_status, m.hotspot_snr,
                    m.hotspot_centroid.map(|(x, y)| ((x * 10.0).round() / 10.0, (y * 10.0).round() / 10.0)), m.handoff_count, m.actual_hz,
                    c.as_ref().map(|c| c.fps).unwrap_or(0.0)
                );
                if let Some(s) = m.status.last() {
                    eprintln!("autotest   last: {s}");
                }
            }
            if t > at.duration {
                eprintln!("autotest: done");
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        // Headless replay benchmark: SKYTRACKER_REPLAY_TEST=<run-name-substring>
        // loads that run, plays it at 1x and logs displayed-vs-wanted frames
        // once a second; exits at the end of the run.
        if let Some(rt) = self.replay_test.as_mut() {
            self.screen = Screen::Replay;
            let t = self.started.elapsed().as_secs_f64();
            if !rt.loaded {
                if let Some(idx) = self.replay.find_run(&rt.needle) {
                    self.replay.load_run(idx, Some(ctx.clone()));
                    self.replay.looping = false;
                    self.replay.playing = true;
                    rt.loaded = true;
                    rt.started = Some(Instant::now());
                    eprintln!("replaytest: loaded run #{idx} ({})", rt.needle);
                } else if t > 8.0 {
                    eprintln!("replaytest: run '{}' not found in library ({:?} entries)", rt.needle, self.replay.library_len());
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            } else {
                let a = self.replay.shown_frame_index(0).unwrap_or(0);
                let b = self.replay.shown_frame_index(1).unwrap_or(0);
                if rt.shown_hist.last() != Some(&(a, b)) {
                    rt.shown_hist.push((a, b));
                }
                if t - rt.last_log >= 1.0 {
                    rt.last_log = t;
                    eprintln!("replaytest {:5.1}s {}  distinct frames so far: {}", rt.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0), self.replay.debug_status(), rt.shown_hist.len());
                }
                if !self.replay.playing && rt.started.map_or(false, |s| s.elapsed().as_secs_f64() > 3.0) {
                    let el = rt.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
                    eprintln!("replaytest: done in {el:.1}s, {} distinct (cam0,cam1) frame pairs displayed", rt.shown_hist.len());
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }

        self.top_bar(ctx, ui_fps);

        match self.screen {
            Screen::Track => {
                // Layout selector rides in the toggles row; the choice persists.
                let layout = self.ui.track_layout.clone();
                match layout.as_str() {
                    // ---- stacked: controls column + a full camera column ----
                    "stack" => {
                        egui::SidePanel::right("right").default_width(400.0).min_width(330.0).show(ctx, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                sats_rollup(ui, &self.shared, &mut self.ui, &self.tx, false);
                                ui.separator();
                                ui::mount_panel(ui, &self.shared, &mut self.ui, &self.tx);
                            });
                        });
                        egui::SidePanel::right("cams_col").default_width(360.0).min_width(280.0).show(ctx, |ui| {
                            let n = self.shared.cams.len().max(1);
                            let h = (ui.available_height() - 46.0 * n as f32) / n as f32;
                            for slot in 0..n {
                                ui::camera_view(ui, &self.shared, &mut self.ui, slot, false, Some(h));
                                ui::camera_quick_controls(ui, &self.shared, slot, &self.tx_cam);
                                ui.add_space(4.0);
                            }
                        });
                        egui::CentralPanel::default().frame(egui::Frame::none().fill(theme::BG)).show(ctx, |ui| {
                            track_toggles(ui, &mut self.ui, &self.shared);
                            ui::skyplot(ui, &self.shared, &mut self.ui, &self.tx);
                        });
                    }
                    // ---- quad: controls | skyplot on top, cameras across the bottom ----
                    "quad" => {
                        // Explicit stored height + our own drag handle: egui's
                        // bottom-panel resize memory kept sliding back.
                        let max_h = (ctx.screen_rect().height() - 240.0).max(260.0);
                        self.ui.quad_cam_h = self.ui.quad_cam_h.clamp(200.0, max_h);
                        let cam_h = self.ui.quad_cam_h;
                        egui::TopBottomPanel::bottom("quad_cams").resizable(false).exact_height(cam_h).show(ctx, |ui| {
                            ui::vdrag_handle(ui, "quad_split", &mut self.ui.quad_cam_h, 200.0, max_h, true);
                            let k = self.ui.quad_cams.clamp(2, 3);
                            let h = ui.available_height() - 36.0;
                            ui.columns(k, |cols| {
                                for (slot, col) in cols.iter_mut().enumerate() {
                                    ui::camera_view(col, &self.shared, &mut self.ui, slot, false, Some(h));
                                    ui::camera_quick_controls(col, &self.shared, slot, &self.tx_cam);
                                }
                            });
                        });
                        egui::CentralPanel::default().frame(egui::Frame::none().fill(theme::BG)).show(ctx, |ui| {
                            ui.columns(2, |cols| {
                                {
                                    let ui = &mut cols[0];
                                    sats_rollup(ui, &self.shared, &mut self.ui, &self.tx, false);
                                    egui::ScrollArea::vertical().id_salt("quad_ctl").show(ui, |ui| {
                                        ui::mount_panel(ui, &self.shared, &mut self.ui, &self.tx);
                                    });
                                }
                                let ui = &mut cols[1];
                                track_toggles(ui, &mut self.ui, &self.shared);
                                ui::skyplot(ui, &self.shared, &mut self.ui, &self.tx);
                            });
                        });
                    }
                    // ---- scope: guide + main dominate; skyplot + controls peripheral ----
                    "scope" => {
                        egui::SidePanel::right("scope_side").resizable(true).default_width(330.0).min_width(280.0).show(ctx, |ui| {
                            // Widen the bar, then drag the handles: each
                            // section's height is stored, the bubble cam
                            // takes whatever is left.
                            let w = ui.available_width();
                            let sky_h = self.ui.scope_sky_h;
                            ui.allocate_ui(egui::Vec2::new(w, sky_h), |ui| {
                                ui.set_min_size(egui::Vec2::new(w, sky_h));
                                ui::skyplot(ui, &self.shared, &mut self.ui, &self.tx);
                            });
                            ui::vdrag_handle(ui, "scope_sky", &mut self.ui.scope_sky_h, 140.0, 760.0, false);
                            let ctl_h = self.ui.scope_ctl_h;
                            ui.allocate_ui(egui::Vec2::new(w, ctl_h), |ui| {
                                ui.set_min_size(egui::Vec2::new(w, ctl_h));
                                egui::ScrollArea::vertical().id_salt("scope_ctl").max_height(ctl_h).show(ui, |ui| {
                                    ui::compact_mount(ui, &self.shared, &self.tx);
                                });
                            });
                            ui::vdrag_handle(ui, "scope_ctl", &mut self.ui.scope_ctl_h, 80.0, 500.0, false);
                            sats_rollup(ui, &self.shared, &mut self.ui, &self.tx, false);
                            let h = (ui.available_height() - 40.0).max(120.0);
                            ui::camera_view(ui, &self.shared, &mut self.ui, 2, false, Some(h));
                            ui::camera_quick_controls(ui, &self.shared, 2, &self.tx_cam);
                        });
                        egui::CentralPanel::default().frame(egui::Frame::none().fill(theme::BG)).show(ctx, |ui| {
                            track_toggles(ui, &mut self.ui, &self.shared);
                            let h = ui.available_height() - 4.0;
                            if self.ui.scope_combined {
                                let (r, _) = ui.allocate_painter(egui::Vec2::new(ui.available_width(), h), egui::Sense::hover());
                                let w = self.ui.scope_weights;
                                ui::weighted_overlay(ui, &self.shared, &mut self.ui, r.rect, &[(0, w[0]), (1, w[1])], true);
                            } else {
                                let h = h - 30.0;
                                ui.columns(2, |cols| {
                                    ui::camera_view(&mut cols[0], &self.shared, &mut self.ui, 0, false, Some(h));
                                    ui::camera_quick_controls(&mut cols[0], &self.shared, 0, &self.tx_cam);
                                    ui::camera_view(&mut cols[1], &self.shared, &mut self.ui, 1, false, Some(h));
                                    ui::camera_quick_controls(&mut cols[1], &self.shared, 1, &self.tx_cam);
                                });
                            }
                        });
                    }
                    // ---- tabs (default): the original layout ----
                    _ => {
                        egui::SidePanel::right("right").default_width(430.0).min_width(360.0).show(ctx, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui::camera_panel(ui, &self.shared, &mut self.ui, &self.tx_cam, false);
                                ui.add_space(8.0);
                                ui.separator();
                                ui::mount_panel(ui, &self.shared, &mut self.ui, &self.tx);
                            });
                        });
                        egui::TopBottomPanel::bottom("bottom").default_height(230.0).resizable(true).show(ctx, |ui| {
                            ui::sky_table(ui, &self.shared, &mut self.ui, &self.tx);
                        });
                        egui::CentralPanel::default().frame(egui::Frame::none().fill(theme::BG)).show(ctx, |ui| {
                            track_toggles(ui, &mut self.ui, &self.shared);
                            ui::skyplot(ui, &self.shared, &mut self.ui, &self.tx);
                        });
                    }
                }
            }
            Screen::Cameras => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    screens::cameras_screen(ui, &self.shared, &mut self.ui, &mut self.cameras, &self.tx_cam, &self.tx_fw);
                });
            }
            Screen::Passes => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    screens::passes_screen(ui, &self.shared, &mut self.ui, &mut self.passes, &self.tx);
                });
            }
            Screen::Align => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    screens::align_screen(ui, &self.shared, &mut self.ui, &mut self.align, &self.tx_align, &self.tx, &self.tx_cam);
                });
            }
            Screen::Replay => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let dir = self.shared.config.repo_root().join(&self.shared.config.captures_dir);
                    replay::screen(ui, &mut self.replay, &dir);
                });
            }
            Screen::Mount3d => {
                egui::CentralPanel::default().frame(egui::Frame::none().fill(theme::BG)).show(ctx, |ui| {
                    let m = self.shared.mount.load();
                    let cfg = &self.shared.config;
                    let cam_fov = self.shared.cam(self.shared.hotspot_slot()).map(|c| c.fov_deg).unwrap_or(1.0);
                    // Sky objects for the dome: bright stars, bodies, the
                    // trackable satellites (dead-reckoned), the selection.
                    let sky = self.shared.sky.load();
                    let age_s = ((crate::sky::now_jd_tt() - sky.jd_tt) * 86400.0).clamp(0.0, 5.0);
                    let mask = cfg.elevation_mask_deg;
                    let mut marks: Vec<mount3d::SkyMark> = Vec::with_capacity(1200);
                    for s in sky.stars.iter().filter(|s| s.mag <= 4.5) {
                        marks.push(mount3d::SkyMark { az: s.az, el: s.el, kind: mount3d::SkyKind::Star { mag: s.mag as f32 } });
                    }
                    for b in &sky.bodies {
                        marks.push(mount3d::SkyMark { az: b.az, el: b.el, kind: mount3d::SkyKind::Body { name: b.name.clone() } });
                    }
                    for s in &sky.sats {
                        let el = s.el + s.el_rate * age_s;
                        if el < mask {
                            continue;
                        }
                        let selected = self.ui.selected.as_deref() == Some(s.satnum.as_str());
                        marks.push(mount3d::SkyMark {
                            az: s.az + s.az_rate * age_s,
                            el,
                            kind: mount3d::SkyKind::Sat { selected, geo: s.range_km > 20_000.0, name: s.name.clone() },
                        });
                    }
                    let pose = mount3d::MountPose {
                        sky: &marks,
                        az_deg: m.azm,
                        el_deg: m.alt,
                        mount_mode: &cfg.mount_mode,
                        lat_deg: cfg.lat_deg,
                        target: m.setpoint,
                        tracking: matches!(m.mode.as_str(), "PROGRAM" | "HANDOFF" | "HOTSPOT"),
                        fov_deg: cam_fov,
                        az_limits: Some(cfg.azm_limit),
                        el_limits: Some(cfg.alt_limit),
                    };
                    let size = ui.available_size();
                    self.mount3d.ui(ui, size, &pose);
                });
            }
            Screen::Sim => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    screens::sim_screen(ui, &self.shared);
                });
            }
            Screen::Config => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    screens::config_screen(ui, &self.shared, &mut self.config, &self.tx);
                });
            }
        }
    }
}


/// The visible-satellites table as a collapsible rollup (stack right bar,
/// quad top-left, scope right bar). Height-capped so it can nest anywhere.
fn sats_rollup(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut ui::UiState, tx: &crossbeam_channel::Sender<MountCmd>, default_open: bool) {
    egui::CollapsingHeader::new(egui::RichText::new("VISIBLE SATELLITES").font(theme::sans(11.0)).color(theme::TEXT_2))
        .id_salt("sats_rollup")
        .default_open(default_open)
        .show(ui, |ui| {
            let h = 300.0_f32.min(ui.available_height().max(180.0));
            ui.allocate_ui(egui::Vec2::new(ui.available_width(), h), |ui| {
                ui.set_min_height(h);
                ui::sky_table(ui, shared, st, tx);
            });
        });
}

/// The Track screen's toggle strip: skyplot layers + the layout selector.
fn track_toggles(ui: &mut egui::Ui, st: &mut ui::UiState, shared: &Arc<Shared>) {
    ui.horizontal(|ui| {
        ui.add_space(6.0);
        let before = (st.track_layout.clone(), st.quad_cams, st.scope_combined);
        egui::ComboBox::from_id_salt("track_layout").selected_text(st.track_layout.clone()).width(70.0).show_ui(ui, |ui| {
            for l in ["tabs", "stack", "quad", "scope"] {
                ui.selectable_value(&mut st.track_layout, l.to_string(), l);
            }
        });
        if st.track_layout == "quad" {
            for k in [2usize, 3] {
                if ui.selectable_label(st.quad_cams == k, format!("{k} cams")).clicked() {
                    st.quad_cams = k;
                }
            }
        }
        if st.track_layout == "scope" {
            ui.checkbox(&mut st.scope_combined, "combined");
            if st.scope_combined {
                ui.add(egui::Slider::new(&mut st.scope_weights[0], 0.0..=1.0).show_value(false));
                ui.label(egui::RichText::new("guide/main").font(theme::sans(10.0)).color(DIM));
                ui.add(egui::Slider::new(&mut st.scope_weights[1], 0.0..=1.0).show_value(false));
            }
        }
        if before != (st.track_layout.clone(), st.quad_cams, st.scope_combined) {
            // Persist the layout choice.
            if let Ok(text) = std::fs::read_to_string(&shared.config.path) {
                if let Ok(mut raw) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(o) = raw.as_object_mut() {
                        o.insert("track_layout".into(), serde_json::json!(st.track_layout));
                        o.insert("track_quad_cams".into(), serde_json::json!(st.quad_cams));
                        o.insert("track_scope_combined".into(), serde_json::json!(st.scope_combined));
                        if let Ok(s) = serde_json::to_string_pretty(&raw) {
                            let _ = std::fs::write(&shared.config.path, s);
                        }
                    }
                }
            }
        }
        ui.separator();
        ui.checkbox(&mut st.show_stars, "stars");
        ui.checkbox(&mut st.show_sats, "satellites");
        ui.checkbox(&mut st.show_labels, "labels");
        ui.checkbox(&mut st.show_below_mask, "below mask");
        ui.checkbox(&mut st.show_names, "star names");
        ui.checkbox(&mut st.show_messier, "Messier");
        ui.checkbox(&mut st.show_ngc, "NGC");
        ui.checkbox(&mut st.show_aircraft, "aircraft");
        ui.checkbox(&mut st.show_keepout, "keepout");
        ui.checkbox(&mut st.show_meo, "MEO");
        ui.checkbox(&mut st.show_geo, "GEO");
    });
}

/// A good demo/track target: a LEO satellite well above the mask that
/// stays up for a while (highest elevation among range < 2500 km).
fn pick_demo_target(shared: &Shared) -> Option<String> {
    let sky = shared.sky.load();
    let mask = shared.config.elevation_mask_deg;
    sky.sats
        .iter()
        .filter(|s| s.range_km < 2500.0 && s.el > mask + 15.0 && s.el < 60.0 && s.az_rate.abs() < 0.6 && s.el_rate > -0.05)
        .max_by(|a, b| a.el.partial_cmp(&b.el).unwrap())
        .map(|s| s.satnum.clone())
}

/// Locate the repo root (config.json + tle_cache.tle) from SKYTRACKER_ROOT,
/// the working directory or any parent of it, or the checkout this binary
/// was compiled from -- so `cargo run` from rust/ and a double-clicked exe
/// both work.
fn find_repo_root() -> std::path::PathBuf {
    if let Ok(r) = std::env::var("SKYTRACKER_ROOT") {
        return std::path::PathBuf::from(r);
    }
    let cwd = std::env::current_dir().expect("cwd");
    let mut dir = cwd.clone();
    for _ in 0..5 {
        if dir.join("tle_cache.tle").exists() || dir.join("config.json").exists() {
            return dir;
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    let compiled_from = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if compiled_from.join("tle_cache.tle").exists() || compiled_from.join("config.json").exists() {
        return compiled_from.canonicalize().unwrap_or(compiled_from);
    }
    cwd
}

fn main() -> eframe::Result<()> {
    let root = find_repo_root();
    eprintln!("skytracker: repo root = {}", root.display());
    let config = state::Config::load(&root.join("config.json"));
    let vsync = match std::env::var("SKYTRACKER_VSYNC") {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => config.ui_vsync,
    };
    let core = skytracker_core::core_loop::Shared::new(mount::make_inputs(&config));
    let shared = Shared::new(config, core);
    let (tx, rx) = crossbeam_channel::unbounded::<MountCmd>();
    let (tx_cam, rx_cam) = crossbeam_channel::unbounded::<CamCmd>();
    let (tx_align, rx_align) = crossbeam_channel::unbounded::<AlignCmd>();
    let (tx_fw, rx_fw) = crossbeam_channel::unbounded::<state::FwCmd>();

    sky::spawn(shared.clone(), root.clone());
    mount::spawn(shared.clone(), rx, root.clone(), tx_cam.clone());
    camera::spawn(shared.clone(), rx_cam, root.clone());
    adsb::spawn(shared.clone());
    align::spawn(shared.clone(), rx_align, tx.clone());
    filterwheel::spawn(shared.clone(), rx_fw, root.clone());

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("Hat Creek Skytracker")
            .with_inner_size([1760.0, 1060.0])
            .with_min_inner_size([1100.0, 700.0]),
        vsync,
        ..Default::default()
    };
    let (tx2, tx_cam2, tx_align2, tx_fw2) = (tx.clone(), tx_cam.clone(), tx_align.clone(), tx_fw.clone());
    eframe::run_native(
        "Hat Creek Skytracker",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, shared, tx2, tx_cam2, tx_align2, tx_fw2)))),
    )
}
