//! UI: skyplot (egui Painter), camera view (wgpu texture), live sky table,
//! mount panel, HUD — dark modern theme, rendered from snapshots only.

use crate::state::{MountCmd, Shared};
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use std::sync::Arc;

pub struct UiState {
    pub selected: Option<String>,
    pub hover: Option<String>,
    pub show_stars: bool,
    pub show_sats: bool,
    pub show_labels: bool,
    pub cam_tex: Option<egui::TextureHandle>,
    pub cam_seq: u64,
    pub frame_times: std::collections::VecDeque<f64>,
    pub last_frame: std::time::Instant,
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            selected: None,
            hover: None,
            show_stars: true,
            show_sats: true,
            show_labels: true,
            cam_tex: None,
            cam_seq: u64::MAX,
            frame_times: std::collections::VecDeque::with_capacity(240),
            last_frame: std::time::Instant::now(),
        }
    }
}

pub const ACCENT: Color32 = Color32::from_rgb(56, 189, 248);
pub const AMBER: Color32 = Color32::from_rgb(251, 191, 36);
pub const DIM: Color32 = Color32::from_rgb(110, 120, 135);
pub const PANEL: Color32 = Color32::from_rgb(17, 20, 26);
pub const SKY: Color32 = Color32::from_rgb(8, 10, 14);

pub fn apply_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = SKY;
    v.widgets.noninteractive.bg_fill = PANEL;
    v.selection.bg_fill = ACCENT.linear_multiply(0.35);
    v.hyperlink_color = ACCENT;
    ctx.set_visuals(v);
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 6.0);
    style.spacing.button_padding = Vec2::new(10.0, 5.0);
    ctx.set_style(style);
}

fn alt_color(range_km: f64) -> Color32 {
    // Legend: 0..1000 km blue->green->red like the orbit legend; beyond 1000 km
    // MEO orange, GEO purple.
    if range_km > 20000.0 {
        Color32::from_rgb(190, 80, 220)
    } else if range_km > 3000.0 {
        Color32::from_rgb(255, 160, 40)
    } else {
        let t = (range_km / 1000.0).clamp(0.0, 1.0) as f32;
        let (r, g, b) = if t < 0.5 {
            (0.0, t * 2.0, 1.0 - t * 2.0)
        } else {
            ((t - 0.5) * 2.0, 1.0 - (t - 0.5) * 2.0, 0.0)
        };
        Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }
}

/// Polar projection: az clockwise from north (up), el 90 at centre.
fn polar(center: Pos2, radius: f32, az: f64, el: f64) -> Pos2 {
    let r = radius * ((90.0 - el.clamp(-10.0, 90.0)) / 90.0) as f32;
    let a = az.to_radians() as f32;
    Pos2::new(center.x + r * a.sin(), center.y - r * a.cos())
}

pub fn skyplot(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx: &crossbeam_channel::Sender<MountCmd>) {
    let sky = shared.sky.load();
    let mount = shared.mount.load();
    let cfg = &shared.config;
    let avail = ui.available_size();
    let (resp, painter) = ui.allocate_painter(avail, Sense::click());
    let rect = resp.rect;
    painter.rect_filled(rect, 0.0, SKY);
    let radius = (rect.width().min(rect.height()) / 2.0 - 28.0).max(50.0);
    let center = rect.center();

    // Elevation rings + mask.
    for el in [0.0, 30.0, 60.0] {
        let r = radius * ((90.0 - el) / 90.0) as f32;
        painter.circle_stroke(center, r, Stroke::new(1.0, Color32::from_gray(55)));
    }
    let mask_r = radius * ((90.0 - cfg.elevation_mask_deg) / 90.0) as f32;
    painter.circle_stroke(center, mask_r, Stroke::new(1.2, Color32::from_rgb(160, 40, 40)));
    // Azimuth spokes + labels.
    for az in (0..360).step_by(30) {
        let p = polar(center, radius, az as f64, 0.0);
        painter.line_segment([center, p], Stroke::new(0.6, Color32::from_gray(45)));
        let lp = polar(center, radius + 14.0, az as f64, 0.0);
        let label = match az {
            0 => "N".to_string(),
            90 => "E".to_string(),
            180 => "S".to_string(),
            270 => "W".to_string(),
            a => a.to_string(),
        };
        painter.text(lp, Align2::CENTER_CENTER, label, FontId::proportional(12.0), DIM);
    }

    // Stars (magnitude-sized).
    if st.show_stars {
        for s in &sky.stars {
            if s.el < 0.0 {
                continue;
            }
            let p = polar(center, radius, s.az, s.el);
            let size = (3.2 - s.mag * 0.45).clamp(0.6, 3.2) as f32;
            painter.circle_filled(p, size, Color32::from_gray((200.0 - s.mag * 18.0).clamp(90.0, 230.0) as u8));
        }
    }
    // Bodies.
    for b in &sky.bodies {
        if b.el < -2.0 {
            continue;
        }
        let p = polar(center, radius, b.az, b.el);
        let (col, r) = match b.name.as_str() {
            "sun" => (Color32::from_rgb(255, 230, 120), 7.0),
            "moon" => (Color32::from_rgb(220, 220, 230), 6.0),
            _ => (Color32::from_rgb(240, 200, 120), 4.0),
        };
        painter.circle_filled(p, r, col);
        painter.text(p + Vec2::new(8.0, -6.0), Align2::LEFT_CENTER, &b.name, FontId::proportional(11.0), col);
    }

    // Satellites + hit-testing.
    let pointer = resp.hover_pos();
    let mut best: Option<(f32, String, String)> = None;
    if st.show_sats {
        for s in &sky.sats {
            if s.el < 0.0 {
                continue;
            }
            let p = polar(center, radius, s.az, s.el);
            let selected = st.selected.as_deref() == Some(s.satnum.as_str());
            let col = alt_color(s.range_km);
            if selected {
                painter.circle_stroke(p, 9.0, Stroke::new(2.0, ACCENT));
            }
            painter.rect_filled(Rect::from_center_size(p, Vec2::splat(5.0)), 1.0, col);
            if st.show_labels && (selected || s.el > 40.0) {
                painter.text(p + Vec2::new(6.0, 0.0), Align2::LEFT_CENTER, &s.name, FontId::proportional(10.0), col.linear_multiply(0.9));
            }
            if let Some(ptr) = pointer {
                let d = ptr.distance(p);
                if d < 10.0 && best.as_ref().map_or(true, |b| d < b.0) {
                    best = Some((d, s.satnum.clone(), format!("{}  az {:.1}° el {:.1}°  {:.0} km  {:+.2}°/s", s.name, s.az, s.el, s.range_km, s.az_rate)));
                }
            }
        }
    }

    // Mount boresight + camera FOV footprint.
    let mp = polar(center, radius, mount.az, mount.el);
    painter.circle_stroke(mp, 7.0, Stroke::new(1.5, AMBER));
    painter.line_segment([mp + Vec2::new(-11.0, 0.0), mp + Vec2::new(11.0, 0.0)], Stroke::new(1.0, AMBER));
    painter.line_segment([mp + Vec2::new(0.0, -11.0), mp + Vec2::new(0.0, 11.0)], Stroke::new(1.0, AMBER));
    if let Some(cam) = shared.cam.load().as_ref() {
        let fov_r = radius * (cam.fov_deg / 90.0) as f32 / 2.0;
        painter.circle_stroke(mp, fov_r.max(3.0), Stroke::new(1.0, AMBER.linear_multiply(0.6)));
    }

    // Hover tooltip + click select.
    st.hover = None;
    if let Some((_, satnum, text)) = best {
        if let Some(ptr) = pointer {
            painter.text(ptr + Vec2::new(12.0, -14.0), Align2::LEFT_CENTER, text, FontId::proportional(12.0), Color32::WHITE);
        }
        st.hover = Some(satnum.clone());
        if resp.clicked() {
            st.selected = Some(satnum.clone());
            let _ = tx.send(MountCmd::SelectTarget(Some(satnum)));
        }
    } else if resp.clicked() && resp.hover_pos().map_or(false, |p| p.distance(center) > radius) {
        st.selected = None;
        let _ = tx.send(MountCmd::SelectTarget(None));
    }

    // Corner readouts.
    painter.text(rect.left_top() + Vec2::new(8.0, 8.0), Align2::LEFT_TOP,
        format!("{}\n{} visible · compute {:.1} ms · {}", sky.utc_iso, sky.n_visible, sky.compute_ms, sky.status),
        FontId::monospace(11.0), DIM);
}

pub fn camera_panel(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState) {
    let cam = shared.cam.load();
    let Some(cam) = cam.as_ref() else {
        ui.label("camera: waiting for frames…");
        return;
    };
    if cam.seq != st.cam_seq {
        let img = egui::ColorImage::from_gray([cam.width, cam.height], &cam.data);
        match st.cam_tex.as_mut() {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => st.cam_tex = Some(ui.ctx().load_texture("cam1", img, egui::TextureOptions::LINEAR)),
        }
        st.cam_seq = cam.seq;
    }
    if let Some(tex) = &st.cam_tex {
        let avail = ui.available_width();
        let h = avail * cam.height as f32 / cam.width as f32;
        let (r, p) = ui.allocate_painter(Vec2::new(avail, h), Sense::hover());
        p.image(tex.id(), r.rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        // Reticle.
        let c = r.rect.center();
        p.line_segment([c + Vec2::new(-14.0, 0.0), c + Vec2::new(14.0, 0.0)], Stroke::new(1.0, AMBER));
        p.line_segment([c + Vec2::new(0.0, -14.0), c + Vec2::new(0.0, 14.0)], Stroke::new(1.0, AMBER));
        p.text(r.rect.left_top() + Vec2::new(6.0, 6.0), Align2::LEFT_TOP,
            format!("CAM1 sim · {:.0} FPS · FOV {:.2}° · seq {}", cam.fps, cam.fov_deg, cam.seq),
            FontId::monospace(11.0), AMBER);
    }
}

pub fn mount_panel(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx: &crossbeam_channel::Sender<MountCmd>) {
    let m = shared.mount.load();
    ui.heading("Mount");
    ui.horizontal(|ui| {
        for mode in ["STANDBY", "RATE", "PROGRAM"] {
            let on = m.mode == mode;
            if ui.selectable_label(on, mode).clicked() {
                let _ = tx.send(MountCmd::SetMode(mode.to_string()));
            }
        }
        if ui.button("STOP").clicked() {
            let _ = tx.send(MountCmd::Stop);
        }
    });
    ui.add_space(4.0);
    egui::Grid::new("mount_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        ui.label("AZ"); ui.monospace(format!("{:8.3}°", m.az)); ui.end_row();
        ui.label("EL"); ui.monospace(format!("{:8.3}°", m.el)); ui.end_row();
        ui.label("err"); ui.monospace(format!("{:+.3} / {:+.3}", m.az_error, m.el_error)); ui.end_row();
        ui.label("rate cmd"); ui.monospace(format!("{:+} / {:+}  (gear {})", m.rate_cmd.0, m.rate_cmd.1, m.gear_ceiling)); ui.end_row();
        ui.label("loop"); ui.monospace(format!("{:.1} Hz", m.actual_hz)); ui.end_row();
        ui.label("joystick"); ui.monospace(m.joystick.clone().unwrap_or_else(|| "none".into())); ui.end_row();
        ui.label("stick"); ui.monospace(format!("{:+.2} / {:+.2}", m.stick.0, m.stick.1)); ui.end_row();
        ui.label("target"); ui.monospace(st.selected.clone().unwrap_or_else(|| "—".into())); ui.end_row();
    });
    ui.add_space(6.0);
    for s in &m.status {
        ui.small(s);
    }
}

pub fn sky_table(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx: &crossbeam_channel::Sender<MountCmd>) {
    use egui_extras::{Column, TableBuilder};
    let sky = shared.sky.load();
    ui.heading(format!("Visible satellites ({})", sky.n_visible));
    TableBuilder::new(ui)
        .striped(true)
        .column(Column::initial(170.0).at_least(120.0))
        .column(Column::initial(70.0))
        .column(Column::initial(60.0))
        .column(Column::initial(60.0))
        .column(Column::initial(80.0))
        .column(Column::initial(80.0))
        .header(18.0, |mut h| {
            for t in ["Name", "NORAD", "Az", "El", "Range km", "Az rate °/s"] {
                h.col(|ui| { ui.strong(t); });
            }
        })
        .body(|body| {
            let rows: Vec<_> = sky.sats.iter().filter(|s| s.el > 0.0).collect();
            body.rows(16.0, rows.len(), |mut row| {
                let s = rows[row.index()];
                let sel = st.selected.as_deref() == Some(s.satnum.as_str());
                row.col(|ui| {
                    if ui.selectable_label(sel, &s.name).clicked() {
                        st.selected = Some(s.satnum.clone());
                        let _ = tx.send(MountCmd::SelectTarget(Some(s.satnum.clone())));
                    }
                });
                row.col(|ui| { ui.monospace(&s.satnum); });
                row.col(|ui| { ui.monospace(format!("{:.1}", s.az)); });
                row.col(|ui| { ui.monospace(format!("{:.1}", s.el)); });
                row.col(|ui| { ui.monospace(format!("{:.0}", s.range_km)); });
                row.col(|ui| { ui.monospace(format!("{:+.3}", s.az_rate)); });
            });
        });
}
