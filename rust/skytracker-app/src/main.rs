//! Hat Creek Skytracker — native app (Phase 7 of the Rust port).
//!
//! eframe + egui on wgpu at a 120 Hz display target; workers publish
//! snapshots (sky, mount, camera), the UI renders them and sends commands.
//! Run from the repo root (it reads config.json, tle_cache.tle, de421.bsp,
//! hip_main.dat from the working directory):
//!
//!   cargo run --release -p skytracker-app

mod camera;
mod mount;
mod sky;
mod state;
mod ui;

use state::{MountCmd, Shared};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct App {
    shared: Arc<Shared>,
    tx: crossbeam_channel::Sender<MountCmd>,
    ui: ui::UiState,
    started: Instant,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, shared: Arc<Shared>, tx: crossbeam_channel::Sender<MountCmd>) -> Self {
        ui::apply_theme(&cc.egui_ctx);
        App {
            shared,
            tx,
            ui: ui::UiState::default(),
            started: Instant::now(),
        }
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

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("HAT CREEK SKYTRACKER").color(ui::ACCENT).strong());
                ui.separator();
                let m = self.shared.mount.load();
                ui.label(egui::RichText::new(format!("MODE {}", m.mode)).color(ui::AMBER).monospace());
                ui.separator();
                ui.checkbox(&mut self.ui.show_stars, "stars");
                ui.checkbox(&mut self.ui.show_sats, "satellites");
                ui.checkbox(&mut self.ui.show_labels, "labels");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(format!("UI {ui_fps:5.1} Hz")).monospace().color(ui::DIM));
                    ui.separator();
                    ui.label(egui::RichText::new(format!("up {:.0}s", self.started.elapsed().as_secs_f64())).monospace().color(ui::DIM));
                });
            });
        });

        egui::SidePanel::right("right").default_width(420.0).show(ctx, |ui| {
            ui::camera_panel(ui, &self.shared, &mut self.ui);
            ui.separator();
            ui::mount_panel(ui, &self.shared, &mut self.ui, &self.tx);
        });

        egui::TopBottomPanel::bottom("bottom").default_height(220.0).resizable(true).show(ctx, |ui| {
            ui::sky_table(ui, &self.shared, &mut self.ui, &self.tx);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui::skyplot(ui, &self.shared, &mut self.ui, &self.tx);
        });
    }
}

fn main() -> eframe::Result<()> {
    let root = std::env::current_dir().expect("cwd");
    let config = state::Config::load(&root.join("config.json"));
    let shared = Shared::new(config);
    let (tx, rx) = crossbeam_channel::unbounded::<MountCmd>();

    sky::spawn(shared.clone(), root.clone());
    mount::spawn(shared.clone(), rx, root.clone());
    camera::spawn(shared.clone());

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("Hat Creek Skytracker")
            .with_inner_size([1700.0, 1050.0])
            .with_min_inner_size([1100.0, 700.0]),
        vsync: true,
        ..Default::default()
    };
    let tx2 = tx.clone();
    eframe::run_native(
        "Hat Creek Skytracker",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, shared, tx2)))),
    )
}
