//! Track screen widgets: skyplot (egui Painter), camera view (texture +
//! tracking overlay), mount instrument panel, live sky table. Rendered from
//! snapshots only; every interaction is a command on a channel.

use crate::state::{CamCmd, MountCmd, Shared};
use crate::theme::{self, ACCENT, AMBER, BG, DIM, GREEN, HAIRLINE, RED, TEXT, TEXT_2, VIOLET};
use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::sync::Arc;

pub struct UiState {
    pub selected: Option<String>,
    pub hover: Option<String>,
    pub show_stars: bool,
    pub show_sats: bool,
    pub show_labels: bool,
    pub show_below_mask: bool,
    pub cam_tex: Vec<Option<egui::TextureHandle>>,
    pub cam_seq: Vec<u64>,
    /// Camera slot shown in the Track panel.
    pub cam_view: usize,
    pub frame_times: std::collections::VecDeque<f64>,
    pub last_frame: std::time::Instant,
    pub capture_name: String,
    pub gains_edit: Option<[[f64; 3]; 2]>,
    pub table_sort: (usize, bool),
    pub show_names: bool,
    pub show_messier: bool,
    pub show_ngc: bool,
    pub show_aircraft: bool,
    pub show_keepout: bool,
    pub show_meo: bool,
    pub show_geo: bool,
    pub toggles_init: bool,
    pub keepout_tex: Option<(egui::TextureHandle, String)>,
    pub conn_transport: String,
    pub conn_port: String,
    pub conn_baud: u32,
    pub conn_init: bool,
    pub adsb_mode: String,
    /// Tracking history for the strip charts: (t_s, az_err, el_err, az_rate, el_rate, hz).
    pub history: std::collections::VecDeque<(f64, f64, f64, f64, f64)>,
    pub history_t0: std::time::Instant,
    pub show_charts: bool,
    pub show_navball: bool,
    /// Track screen layout: "tabs" | "stack" | "quad" | "scope" (persisted).
    pub track_layout: String,
    pub quad_cams: usize,
    /// Scope layout: weighted-combined guide+main instead of a split.
    pub scope_combined: bool,
    pub scope_weights: [f32; 2],
    /// Skyplot mousewheel zoom (1 = whole dome) and pan (px, screen frame).
    pub sky_zoom: f32,
    pub sky_pan: Vec2,
    /// Deterministic layout heights (the egui panel-resize memory proved
    /// unreliable for bottom panels, so these are ours).
    pub quad_cam_h: f32,
    pub scope_sky_h: f32,
    pub scope_ctl_h: f32,
    /// Navball hemisphere texture, cached on quantized elevation.
    pub navball_tex: Option<(egui::TextureHandle, i32)>,
    /// Temporary per-camera pixel zoom (mousewheel) + pan — display only,
    /// independent of the hardware ROI.
    pub cam_zoom: Vec<f32>,
    pub cam_pan: Vec<Vec2>,
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            selected: None,
            hover: None,
            show_stars: true,
            show_sats: true,
            show_labels: true,
            show_below_mask: true,
            cam_tex: Vec::new(),
            cam_seq: Vec::new(),
            cam_view: 0,
            frame_times: std::collections::VecDeque::with_capacity(240),
            last_frame: std::time::Instant::now(),
            capture_name: "manual".into(),
            gains_edit: None,
            table_sort: (2, false),
            show_names: true,
            show_messier: true,
            show_ngc: false,
            show_aircraft: true,
            show_keepout: true,
            show_meo: true,
            show_geo: true,
            toggles_init: false,
            keepout_tex: None,
            conn_transport: String::new(),
            conn_port: String::new(),
            conn_baud: 9600,
            conn_init: false,
            adsb_mode: String::new(),
            history: std::collections::VecDeque::with_capacity(2000),
            history_t0: std::time::Instant::now(),
            show_charts: true,
            show_navball: true,
            track_layout: "tabs".into(),
            quad_cams: 3,
            scope_combined: false,
            scope_weights: [1.0, 1.0],
            sky_zoom: 1.0,
            sky_pan: Vec2::ZERO,
            quad_cam_h: 470.0,
            scope_sky_h: 320.0,
            scope_ctl_h: 150.0,
            navball_tex: None,
            cam_zoom: Vec::new(),
            cam_pan: Vec::new(),
        }
    }
}

/// Polar projection: az clockwise from north (up), el 90 at centre.
pub fn polar(center: Pos2, radius: f32, az: f64, el: f64) -> Pos2 {
    let r = radius * ((90.0 - el.clamp(-10.0, 90.0)) / 90.0) as f32;
    let a = az.to_radians() as f32;
    Pos2::new(center.x + r * a.sin(), center.y - r * a.cos())
}

/// Per-camera accent colors: skyplot FOV footprints + the parameter cards.
pub const CAM_COLORS: [Color32; 3] = [AMBER, Color32::from_rgb(96, 200, 235), VIOLET];

/// "2026-08-23 04:10" from a unix timestamp (UTC), no chrono needed.
fn fmt_unix(t: f64) -> String {
    if !t.is_finite() || t <= 0.0 {
        return "—".into();
    }
    let secs = t as i64;
    let (days, sod) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", sod / 3600, (sod / 60) % 60)
}

/// A small painter-drawn key/value card (info + camera panes on the skyplot).
/// Returns the card height.
fn info_pane(painter: &egui::Painter, at: Pos2, w: f32, title: &str, title_col: Color32, lines: &[(String, String, Color32)]) -> f32 {
    let lh = 13.0;
    let h = 21.0 + lines.len() as f32 * lh + 5.0;
    let r = Rect::from_min_size(at, Vec2::new(w, h));
    painter.rect(r, 4.0, theme::with_alpha(theme::RAISED, 216), Stroke::new(1.0, HAIRLINE));
    painter.text(r.min + Vec2::new(8.0, 4.0), Align2::LEFT_TOP, title, theme::sans(11.0), title_col);
    let mut y = r.min.y + 21.0;
    for (k, v, c) in lines {
        painter.text(Pos2::new(r.min.x + 8.0, y), Align2::LEFT_TOP, k, theme::mono(9.5), DIM);
        painter.text(Pos2::new(r.max.x - 8.0, y), Align2::RIGHT_TOP, v, theme::mono(9.5), *c);
        y += lh;
    }
    h
}

/// A horizontal grab-bar splitter: dragging it resizes the section above
/// (or below, with `invert`) deterministically — no panel-memory surprises.
pub fn vdrag_handle(ui: &mut egui::Ui, id_salt: &str, val: &mut f32, min: f32, max: f32, invert: bool) {
    let (r, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 8.0), Sense::drag());
    let resp = resp.on_hover_cursor(egui::CursorIcon::ResizeVertical);
    let c = if resp.dragged() {
        ACCENT
    } else if resp.hovered() {
        theme::with_alpha(ACCENT, 150)
    } else {
        HAIRLINE
    };
    ui.painter().line_segment([Pos2::new(r.center().x - 26.0, r.center().y), Pos2::new(r.center().x + 26.0, r.center().y)], Stroke::new(2.0, c));
    if resp.dragged() {
        let d = resp.drag_delta().y;
        *val = (*val + if invert { -d } else { d }).clamp(min, max);
    }
    let _ = ui.interact(r, ui.id().with((id_salt, "h")), Sense::hover());
}

/// Compact per-camera gain / exposure / gamma / rotation / ROI controls for
/// the Track-screen camera views: one always-visible row (full controls live
/// on the Cameras screen).
pub fn camera_quick_controls(ui: &mut egui::Ui, shared: &Arc<Shared>, slot: usize, tx_cam: &crossbeam_channel::Sender<CamCmd>) {
    if slot >= shared.cam_settings.len() {
        return;
    }
    let mut s = (**shared.cam_settings[slot].load()).clone();
    let before = s.clone();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let lbl = |ui: &mut egui::Ui, t: &str| ui.label(egui::RichText::new(t).font(theme::sans(10.0)).color(DIM));
        lbl(ui, "gain");
        ui.add(egui::DragValue::new(&mut s.gain).speed(2).range(0..=600));
        lbl(ui, "exp");
        ui.add(
            egui::DragValue::new(&mut s.exposure_us)
                .speed(40)
                .range(1..=2_000_000)
                .custom_formatter(|v, _| crate::screens::fmt_exposure(v as i64))
                .custom_parser(crate::screens::parse_exposure),
        );
        ui.checkbox(&mut s.gamma_enabled, egui::RichText::new("γ").font(theme::sans(11.0)).color(DIM));
        ui.add(egui::DragValue::new(&mut s.gamma).speed(0.01).range(0.05..=2.0).max_decimals(2));
        lbl(ui, "rot");
        ui.add(egui::DragValue::new(&mut s.rotation_deg).speed(0.5).suffix("°").range(-180.0..=180.0));
        lbl(ui, "roi");
        for (f, label) in [(1.0, "1"), (0.5, "½"), (0.25, "¼"), (0.125, "⅛")] {
            if theme::mode_button(ui, label, (s.roi_frac - f).abs() < 1e-6, ACCENT) {
                s.roi_frac = f;
                if f >= 0.999 {
                    s.roi_cx = 0.5;
                    s.roi_cy = 0.5;
                }
            }
        }
    });
    if s != before {
        shared.cam_settings[slot].store(Arc::new(s));
        let _ = tx_cam.send(CamCmd::Apply { slot });
    }
}

/// Simple label de-confliction: a coarse occupancy grid over the plot.
struct LabelGrid {
    cell: f32,
    cols: usize,
    used: Vec<bool>,
    origin: Pos2,
}

impl LabelGrid {
    fn new(rect: Rect, cell: f32) -> Self {
        let cols = (rect.width() / cell).ceil() as usize + 1;
        let rows = (rect.height() / cell).ceil() as usize + 1;
        LabelGrid { cell, cols, used: vec![false; cols * rows], origin: rect.min }
    }
    /// Claim the cells a label of `w`x`h` at `p` would cover; false if any is taken.
    fn claim(&mut self, p: Pos2, w: f32, h: f32) -> bool {
        let c0 = ((p.x - self.origin.x) / self.cell).floor().max(0.0) as usize;
        let c1 = ((p.x + w - self.origin.x) / self.cell).floor().max(0.0) as usize;
        let r0 = ((p.y - h / 2.0 - self.origin.y) / self.cell).floor().max(0.0) as usize;
        let r1 = ((p.y + h / 2.0 - self.origin.y) / self.cell).floor().max(0.0) as usize;
        let rows = self.used.len() / self.cols;
        for r in r0..=r1.min(rows.saturating_sub(1)) {
            for c in c0..=c1.min(self.cols - 1) {
                if self.used[r * self.cols + c] {
                    return false;
                }
            }
        }
        for r in r0..=r1.min(rows.saturating_sub(1)) {
            for c in c0..=c1.min(self.cols - 1) {
                self.used[r * self.cols + c] = true;
            }
        }
        true
    }
}

pub fn skyplot(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx: &crossbeam_channel::Sender<MountCmd>) {
    let sky = shared.sky.load();
    let mount = shared.mount.load();
    let passes = shared.passes.load();
    let cfg = &shared.config;
    let avail = ui.available_size();
    let (resp, painter) = ui.allocate_painter(avail, Sense::click_and_drag());
    let rect = resp.rect;
    painter.rect_filled(rect, 0.0, BG);
    let painter = painter.with_clip_rect(rect);
    let base_radius = (rect.width().min(rect.height()) / 2.0 - 30.0).max(60.0);
    // Peripheral/mini form: drop the corner readouts, legend and automatic
    // labels so the plot itself stays readable at small sizes.
    let compact = base_radius < 240.0;
    // Mousewheel zoom about the cursor; drag pans when zoomed in.
    if let Some(ptr) = resp.hover_pos() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let old = st.sky_zoom;
            let new = (old * (scroll * 0.0022).exp()).clamp(1.0, 60.0);
            if new != old {
                let c = rect.center() + st.sky_pan;
                st.sky_pan = (ptr + (c - ptr) * (new / old)) - rect.center();
                st.sky_zoom = new;
            }
        }
    }
    if resp.dragged() && st.sky_zoom > 1.001 {
        st.sky_pan += resp.drag_delta();
    }
    if st.sky_zoom <= 1.001 {
        st.sky_zoom = 1.0;
        st.sky_pan = Vec2::ZERO;
    } else {
        let lim = st.sky_zoom * base_radius;
        st.sky_pan = st.sky_pan.clamp(Vec2::splat(-lim), Vec2::splat(lim));
    }
    let radius = base_radius * st.sky_zoom;
    let center = rect.center() + st.sky_pan;
    let mask = cfg.elevation_mask_deg;
    if !st.toggles_init {
        st.toggles_init = true;
        st.show_meo = cfg.meo_enabled;
        st.show_geo = cfg.geo_enabled;
        st.show_labels = cfg.satellite_labels_enabled;
        st.show_messier = cfg.messier_enabled;
        st.show_ngc = cfg.ngc_enabled;
        st.track_layout = std::env::var("SKYTRACKER_LAYOUT").unwrap_or_else(|_| cfg.raw["track_layout"].as_str().unwrap_or("tabs").to_string());
        st.quad_cams = cfg.raw["track_quad_cams"].as_u64().unwrap_or(3).clamp(2, 3) as usize;
        st.scope_combined = cfg.raw["track_scope_combined"].as_bool().unwrap_or(false);
    }

    // Dome shading: a few concentric fills, lighter toward the zenith.
    for (i, el) in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0].iter().enumerate() {
        let r = radius * ((90.0 - el) / 90.0) as f32;
        let shade = 12 + i as u8 * 2;
        painter.circle_filled(center, r, Color32::from_rgb(shade, shade + 2, shade + 6));
    }
    // Mount keepout wash: sky directions with NO axis solution inside the
    // limits (canonical + the over-the-zenith flip in AltAz/Passthrough),
    // the same transform PROGRAM track uses. Cached as a texture keyed on
    // everything that shapes it.
    if st.show_keepout {
        let key = format!(
            "{}|{}|{:?}|{:?}|{}|{}",
            cfg.mount_mode, cfg.altaz_side_flip, cfg.azm_limit, cfg.alt_limit, cfg.alignment_az, cfg.alignment_el
        );
        if st.keepout_tex.as_ref().map_or(true, |(_, k)| *k != key) {
            let img = keepout_image(cfg, 161);
            let tex = ui.ctx().load_texture("keepout", img, egui::TextureOptions::LINEAR);
            st.keepout_tex = Some((tex, key));
        }
        if let Some((tex, _)) = &st.keepout_tex {
            painter.image(
                tex.id(),
                Rect::from_center_size(center, Vec2::splat(2.0 * radius)),
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }
    // Rings + spokes.
    for el in [0.0, 30.0, 60.0] {
        let r = radius * ((90.0 - el) / 90.0) as f32;
        painter.circle_stroke(center, r, Stroke::new(1.0, HAIRLINE));
        if el > 0.0 {
            painter.text(
                Pos2::new(center.x + 3.0, center.y - r + 2.0),
                Align2::LEFT_TOP,
                format!("{el:.0}°"),
                theme::mono(9.5),
                DIM,
            );
        }
    }
    let mask_r = radius * ((90.0 - mask) / 90.0) as f32;
    painter.circle_stroke(center, mask_r, Stroke::new(1.0, theme::with_alpha(RED, 110)));
    for az in (0..360).step_by(30) {
        let p = polar(center, radius, az as f64, 0.0);
        painter.line_segment([center, p], Stroke::new(0.6, theme::with_alpha(HAIRLINE, 160)));
        let lp = polar(center, radius + 15.0, az as f64, 0.0);
        let (label, col, font) = match az {
            0 => ("N".to_string(), TEXT, theme::sans(13.0)),
            90 => ("E".to_string(), TEXT_2, theme::sans(13.0)),
            180 => ("S".to_string(), TEXT_2, theme::sans(13.0)),
            270 => ("W".to_string(), TEXT_2, theme::sans(13.0)),
            a => (a.to_string(), DIM, theme::mono(10.0)),
        };
        painter.text(lp, Align2::CENTER_CENTER, label, font, col);
    }

    // Stars (magnitude-sized, quiet).
    if st.show_stars {
        for s in &sky.stars {
            if s.el < 0.0 {
                continue;
            }
            let p = polar(center, radius, s.az, s.el);
            let size = (2.6 - s.mag * 0.38).clamp(0.5, 2.6) as f32;
            let g = (190.0 - s.mag * 16.0).clamp(80.0, 215.0) as u8;
            painter.circle_filled(p, size, Color32::from_rgb(g, g, (g as u16 + 12).min(255) as u8));
        }
    }
    let pointer = resp.hover_pos();
    // (distance, selection key, hover text, screen pos)
    let mut best: Option<(f32, String, String, Pos2)> = None;
    let age_s = ((crate::sky::now_jd_tt() - sky.jd_tt) * 86400.0).clamp(0.0, 5.0);
    let mut consider = |best: &mut Option<(f32, String, String, Pos2)>, p: Pos2, key: String, text: String, reach: f32| {
        if let Some(ptr) = pointer {
            let d = ptr.distance(p);
            if d < reach && best.as_ref().map_or(true, |b| d < b.0) {
                *best = Some((d, key, text, p));
            }
        }
    };
    let sel_ring = |p: Pos2| {
        painter.circle_stroke(p, 8.0, Stroke::new(1.6, ACCENT));
        painter.circle_stroke(p, 12.0, Stroke::new(0.8, theme::with_alpha(ACCENT, 90)));
    };

    // Bodies.
    let mut grid = LabelGrid::new(rect, 12.0);
    for b in &sky.bodies {
        if b.el < -2.0 {
            continue;
        }
        let p = polar(center, radius, b.az, b.el);
        let (col, r) = match b.name.as_str() {
            "sun" => (Color32::from_rgb(255, 225, 130), 6.5),
            "moon" => (Color32::from_rgb(215, 218, 228), 5.5),
            _ => (Color32::from_rgb(235, 205, 140), 3.2),
        };
        painter.circle_filled(p, r, col);
        let key = format!("body:{}", b.name);
        if st.selected.as_deref() == Some(key.as_str()) {
            sel_ring(p);
        }
        if grid.claim(p + Vec2::new(8.0, -6.0), 44.0, 12.0) {
            painter.text(p + Vec2::new(8.0, -6.0), Align2::LEFT_CENTER, &b.name, theme::sans(11.0), theme::with_alpha(col, 220));
        }
        consider(&mut best, p, key, format!("{}\naz {:.1}°  el {:.1}°   {:.0} km", b.name, b.az, b.el, b.dist_km), 9.0);
    }
    // Named stars (IAU-CSN): gold spike + name, click-selectable.
    if st.show_stars && st.show_names {
        for s in &sky.stars {
            if s.el < 0.0 {
                continue;
            }
            let Some(name) = sky.star_names.get(&s.hip) else { continue };
            let p = polar(center, radius, s.az, s.el);
            let gold = Color32::from_rgb(232, 196, 110);
            painter.line_segment([p + Vec2::new(-5.0, 0.0), p + Vec2::new(5.0, 0.0)], Stroke::new(0.8, theme::with_alpha(gold, 160)));
            painter.line_segment([p + Vec2::new(0.0, -5.0), p + Vec2::new(0.0, 5.0)], Stroke::new(0.8, theme::with_alpha(gold, 160)));
            let key = format!("star:HIP{}", s.hip);
            if st.selected.as_deref() == Some(key.as_str()) {
                sel_ring(p);
            }
            if s.mag < 2.5 || st.selected.as_deref() == Some(key.as_str()) {
                let lp = p + Vec2::new(7.0, -6.0);
                if grid.claim(lp, 6.0 * name.len() as f32, 12.0) {
                    painter.text(lp, Align2::LEFT_CENTER, name, theme::sans(10.5), theme::with_alpha(gold, 210));
                }
            }
            consider(&mut best, p, key, format!("{name}\nHIP {}  mag {:.1}\naz {:.1}°  el {:.1}°", s.hip, s.mag, s.az, s.el), 8.0);
        }
    }
    // Deep-sky objects: Messier violet squares, NGC teal circles.
    if st.show_messier || st.show_ngc {
        for d in &sky.dsos {
            if d.el < 0.0 || (d.messier && !st.show_messier) || (!d.messier && !st.show_ngc) {
                continue;
            }
            let p = polar(center, radius, d.az, d.el);
            if d.messier {
                painter.rect_stroke(Rect::from_center_size(p, Vec2::splat(6.0)), 1.0, Stroke::new(1.0, VIOLET));
            } else {
                painter.circle_stroke(p, 2.6, Stroke::new(0.9, Color32::from_rgb(80, 190, 190)));
            }
            if st.selected.as_deref() == Some(d.key.as_str()) {
                sel_ring(p);
            }
            if d.messier || st.selected.as_deref() == Some(d.key.as_str()) {
                let short = d.name.split(' ').next().unwrap_or(&d.name).to_string();
                let lp = p + Vec2::new(7.0, 6.0);
                if grid.claim(lp, 6.0 * short.len() as f32, 12.0) {
                    painter.text(lp, Align2::LEFT_CENTER, short, theme::sans(10.0), theme::with_alpha(VIOLET, 200));
                }
            }
            consider(&mut best, p, d.key.clone(), format!("{}\nmag {:.1}   az {:.1}°  el {:.1}°", d.name, d.mag, d.az, d.el), 8.0);
        }
    }
    // Rocket launches: the trajectory arc + the rocket's current position.
    {
        let now_u = crate::sky::now_unix();
        for l in sky.launches.iter() {
            let col = Color32::from_rgb(0, 220, 230);
            let mut prev: Option<Pos2> = None;
            for &(t, az, el, _) in l.rows.iter().step_by(5) {
                if el < -1.0 {
                    prev = None;
                    continue;
                }
                let p = polar(center, radius, az, el);
                if let Some(pp) = prev {
                    let a = if t < now_u { 70 } else { 170 };
                    painter.line_segment([pp, p], Stroke::new(1.2, theme::with_alpha(col, a)));
                }
                prev = Some(p);
            }
            if let Some((az, el, _)) = crate::launches::interp(l, now_u) {
                if el > -1.0 {
                    let p = polar(center, radius, az, el);
                    painter.circle_filled(p, 4.0, col);
                    painter.circle_stroke(p, 7.0, Stroke::new(1.0, theme::with_alpha(col, 160)));
                    if st.selected.as_deref() == Some(l.key.as_str()) {
                        sel_ring(p);
                    }
                    painter.text(p + Vec2::new(9.0, -6.0), Align2::LEFT_CENTER, &l.name, theme::sans(10.5), col);
                    consider(&mut best, p, l.key.clone(), format!("launch {}\naz {:.1}°  el {:.1}°", l.name, az, el), 10.0);
                }
            } else if let Some(&(t0, az, el, _)) = l.rows.first() {
                // Before T0: mark the pad / first point with a countdown.
                if el > -1.0 && t0 > now_u {
                    let p = polar(center, radius, az, el);
                    painter.circle_stroke(p, 4.0, Stroke::new(1.0, theme::with_alpha(col, 150)));
                    painter.text(p + Vec2::new(8.0, 0.0), Align2::LEFT_CENTER, format!("{} T-{:.0}s", l.name, t0 - now_u), theme::sans(10.0), theme::with_alpha(col, 180));
                    consider(&mut best, p, l.key.clone(), format!("launch {} in {:.0} s", l.name, t0 - now_u), 9.0);
                }
            }
        }
    }
    // Aircraft (ADS-B): cyan chevrons, predicted track when selected.
    if st.show_aircraft {
        let adsb = shared.adsb.load();
        let cyan = Color32::from_rgb(90, 220, 230);
        let now_u = crate::sky::now_unix();
        for a in &adsb.aircraft {
            let age = (now_u - a.fit_t_unix).clamp(0.0, 120.0);
            let (az, el) = (a.fit_az + a.az_rate * age, a.fit_el + a.el_rate * age);
            if el < -1.0 {
                continue;
            }
            let p = polar(center, radius, az, el);
            let key = format!("adsb:{}", a.icao);
            let selected = st.selected.as_deref() == Some(key.as_str());
            // Heading chevron along the track direction on the plot.
            let p2 = polar(center, radius, az + a.az_rate * 20.0, el + a.el_rate * 20.0);
            let dir = (p2 - p).normalized();
            let dir = if dir.x.is_nan() { Vec2::new(0.0, -1.0) } else { dir };
            let side = Vec2::new(-dir.y, dir.x);
            let tri = vec![p + dir * 6.0, p - dir * 4.0 + side * 4.0, p - dir * 4.0 - side * 4.0];
            painter.add(egui::Shape::convex_polygon(tri, theme::with_alpha(cyan, 220), Stroke::new(0.8, cyan)));
            if selected {
                sel_ring(p);
                let mut prev: Option<Pos2> = None;
                for (_, paz, pel) in &a.predicted {
                    let q = polar(center, radius, *paz, *pel);
                    if let Some(pp) = prev {
                        painter.line_segment([pp, q], Stroke::new(1.2, theme::with_alpha(cyan, 170)));
                    }
                    prev = Some(q);
                }
                let mut prev: Option<Pos2> = None;
                for (_, haz, hel) in &a.history {
                    let q = polar(center, radius, *haz, *hel);
                    if let Some(pp) = prev {
                        painter.line_segment([pp, q], Stroke::new(1.0, theme::with_alpha(cyan, 80)));
                    }
                    prev = Some(q);
                }
            }
            if selected || el > 20.0 {
                let lp = p + Vec2::new(8.0, 0.0);
                if selected || grid.claim(lp, 6.0 * a.label.len() as f32, 12.0) {
                    painter.text(lp, Align2::LEFT_CENTER, &a.label, theme::sans(10.5), theme::with_alpha(cyan, 220));
                }
            }
            consider(
                &mut best,
                p,
                key,
                format!(
                    "{}  ({})\nalt {:.0} m  range {:.0} km  {}\naz {:.1}°  el {:.1}°   fix {:.0} s ago",
                    a.label,
                    a.icao.to_uppercase(),
                    a.alt_m,
                    a.range_km,
                    a.speed_kt.map(|v| format!("{v:.0} kt")).unwrap_or_default(),
                    az,
                    el,
                    a.age_s
                ),
                9.0,
            );
        }
    }

    // Selected satellite's track: past dim, future bright, minute ticks.
    if let (Some(sn), false) = (passes.arc_satnum.as_ref(), passes.arc.is_empty()) {
        if st.selected.as_deref() == Some(sn.as_str()) {
            let mut prev: Option<(Pos2, f64)> = None;
            for a in &passes.arc {
                if a.el < -1.0 {
                    prev = None;
                    continue;
                }
                let p = polar(center, radius, a.az, a.el);
                if let Some((pp, _)) = prev {
                    let (col, w) = if a.t_rel_s <= 0.0 {
                        (theme::with_alpha(ACCENT, 70), 1.2)
                    } else {
                        (theme::with_alpha(ACCENT, 200), 1.6)
                    };
                    painter.line_segment([pp, p], Stroke::new(w, col));
                }
                if (a.t_rel_s % 60.0).abs() < 1e-6 && a.t_rel_s != 0.0 {
                    painter.circle_filled(p, 1.8, theme::with_alpha(ACCENT, if a.t_rel_s > 0.0 { 230 } else { 110 }));
                    if (a.t_rel_s % 300.0).abs() < 1e-6 {
                        painter.text(
                            p + Vec2::new(5.0, 0.0),
                            Align2::LEFT_CENTER,
                            format!("{:+.0}m", a.t_rel_s / 60.0),
                            theme::mono(9.5),
                            theme::with_alpha(ACCENT, 200),
                        );
                    }
                }
                prev = Some((p, a.t_rel_s));
            }
        }
    }

    // Satellites + hit-testing. Marks are dead-reckoned forward with their
    // rates so motion is smooth at the display rate.
    let mut n_labels = 0;
    if st.show_sats {
        // Draw in two passes so the trackable set sits on top of the chaff.
        for pass in 0..2 {
            for s in &sky.sats {
                let el = s.el + s.el_rate * age_s;
                if el < 0.0 {
                    continue;
                }
                let above = el >= mask;
                if (pass == 0) == above {
                    continue;
                }
                if (!st.show_geo && s.range_km > 20_000.0) || (!st.show_meo && s.range_km > 3_000.0 && s.range_km <= 20_000.0) {
                    continue;
                }
                if !above && !st.show_below_mask {
                    continue;
                }
                let az = s.az + s.az_rate * age_s;
                let p = polar(center, radius, az, el);
                let selected = st.selected.as_deref() == Some(s.satnum.as_str());
                let hovered = st.hover.as_deref() == Some(s.satnum.as_str());
                let col = theme::sat_color(s.range_km);
                if above {
                    let r = if s.range_km > 20_000.0 { 2.2 } else { 2.6 };
                    painter.circle_filled(p, r, col);
                } else {
                    painter.circle_filled(p, 1.4, theme::with_alpha(col, 70));
                }
                if selected {
                    painter.circle_stroke(p, 8.0, Stroke::new(1.6, ACCENT));
                    painter.circle_stroke(p, 12.0, Stroke::new(0.8, theme::with_alpha(ACCENT, 90)));
                } else if hovered {
                    painter.circle_stroke(p, 7.0, Stroke::new(1.0, theme::with_alpha(TEXT, 180)));
                }
                let want_label = st.show_labels && (selected || hovered || (!compact && above && el > 35.0 && s.range_km < 20_000.0 && n_labels < 36));
                if want_label {
                    let lp = p + Vec2::new(7.0, 0.0);
                    let w = 6.0 * s.name.len() as f32;
                    if selected || hovered || grid.claim(lp, w, 12.0) {
                        let lc = if selected { TEXT } else { theme::with_alpha(col, 200) };
                        painter.text(lp, Align2::LEFT_CENTER, &s.name, theme::sans(10.5), lc);
                        n_labels += 1;
                    }
                }
                if let Some(ptr) = pointer {
                    let d = ptr.distance(p);
                    if d < 9.0 && best.as_ref().map_or(true, |b| d < b.0) {
                        let cls = if s.range_km > 20_000.0 {
                            "GEO"
                        } else if s.range_km > 3_000.0 {
                            "MEO"
                        } else {
                            "LEO"
                        };
                        best = Some((
                            d,
                            s.satnum.clone(),
                            format!(
                                "{}\n{} · {}   az {:.1}°  el {:.1}°\nrange {:.0} km   rate {:+.3}°/s",
                                s.name, s.satnum, cls, az, el, s.range_km, s.az_rate
                            ),
                            p,
                        ));
                    }
                }
            }
        }
    }

    // Mount boresight + camera FOV footprint + setpoint vector.
    let mp = polar(center, radius, mount.az, mount.el);
    if let Some((saz, sel)) = mount.setpoint {
        let sp = polar(center, radius, saz, sel);
        painter.line_segment([mp, sp], Stroke::new(1.0, theme::with_alpha(GREEN, 150)));
        painter.line_segment([sp + Vec2::new(-4.0, -4.0), sp + Vec2::new(4.0, 4.0)], Stroke::new(1.0, GREEN));
        painter.line_segment([sp + Vec2::new(-4.0, 4.0), sp + Vec2::new(4.0, -4.0)], Stroke::new(1.0, GREEN));
    }
    painter.circle_stroke(mp, 7.0, Stroke::new(1.4, AMBER));
    painter.line_segment([mp + Vec2::new(-12.0, 0.0), mp + Vec2::new(-4.0, 0.0)], Stroke::new(1.0, AMBER));
    painter.line_segment([mp + Vec2::new(4.0, 0.0), mp + Vec2::new(12.0, 0.0)], Stroke::new(1.0, AMBER));
    painter.line_segment([mp + Vec2::new(0.0, -12.0), mp + Vec2::new(0.0, -4.0)], Stroke::new(1.0, AMBER));
    painter.line_segment([mp + Vec2::new(0.0, 4.0), mp + Vec2::new(0.0, 12.0)], Stroke::new(1.0, AMBER));
    for (i, c) in shared.cams.iter().enumerate() {
        if let Some(cam) = c.load().as_ref() {
            if !cam.connected {
                continue;
            }
            let col = CAM_COLORS[i % CAM_COLORS.len()];
            if cam.fisheye {
                // Fixed-zenith bubble: its coverage circle is centred on the
                // zenith, radius = half the fisheye FOV.
                let r = (radius * (cam.fov_deg / 2.0 / 90.0) as f32).min(radius);
                painter.circle_stroke(center, r, Stroke::new(1.0, theme::with_alpha(col, 80)));
            } else {
                let fov_r = radius * (cam.fov_deg / 90.0) as f32 / 2.0;
                let alpha = if i == shared.hotspot_slot() { 170 } else { 110 };
                painter.circle_stroke(mp, fov_r.max(2.5), Stroke::new(1.2, theme::with_alpha(col, alpha)));
            }
        }
    }
    // Reset-zoom control (bottom-right of the plot).
    let mut zoom_btn_hovered = false;
    if st.sky_zoom > 1.0 {
        let br = Rect::from_min_size(rect.right_bottom() + Vec2::new(-96.0, -26.0), Vec2::new(88.0, 20.0));
        let rb = ui.interact(br, ui.id().with("sky_zoom_reset"), Sense::click());
        zoom_btn_hovered = rb.hovered();
        painter.rect(br, 4.0, theme::with_alpha(theme::RAISED, 230), Stroke::new(1.0, if rb.hovered() { ACCENT } else { HAIRLINE }));
        painter.text(br.center(), Align2::CENTER_CENTER, format!("{:.1}× · reset", st.sky_zoom), theme::mono(10.0), if rb.hovered() { TEXT } else { TEXT_2 });
        if rb.clicked() {
            st.sky_zoom = 1.0;
            st.sky_pan = Vec2::ZERO;
        }
    }

    // Hover card + click select.
    st.hover = None;
    if let Some((_, satnum, text, p)) = best {
        let galley = painter.layout_no_wrap(text, theme::mono(11.0), TEXT);
        let size = galley.size() + Vec2::new(16.0, 10.0);
        let mut at = p + Vec2::new(14.0, -size.y / 2.0);
        if at.x + size.x > rect.right() - 4.0 {
            at.x = p.x - 14.0 - size.x;
        }
        at.y = at.y.clamp(rect.top() + 4.0, rect.bottom() - size.y - 4.0);
        let card = Rect::from_min_size(at, size);
        painter.rect(card, 4.0, theme::with_alpha(theme::RAISED, 235), Stroke::new(1.0, HAIRLINE));
        painter.galley(card.min + Vec2::new(8.0, 5.0), galley, TEXT);
        st.hover = Some(satnum.clone());
        if resp.clicked() {
            st.selected = Some(satnum.clone());
            let _ = tx.send(MountCmd::SelectTarget(Some(satnum)));
        }
    } else if resp.clicked() && !zoom_btn_hovered && resp.hover_pos().map_or(false, |p| p.distance(center) > radius) {
        st.selected = None;
        let _ = tx.send(MountCmd::SelectTarget(None));
    }

    // Corner readouts (full-size plot only).
    if compact {
        painter.text(rect.left_top() + Vec2::new(6.0, 4.0), Align2::LEFT_TOP, format!("{} visible", sky.n_visible), theme::mono(9.5), DIM);
        return;
    }
    painter.text(rect.left_top() + Vec2::new(10.0, 8.0), Align2::LEFT_TOP, &sky.utc_iso, theme::mono(12.5), TEXT);
    let err = sky.status.contains("failed") || sky.status.contains("not found");
    painter.text(
        rect.left_top() + Vec2::new(10.0, 26.0),
        Align2::LEFT_TOP,
        format!("{} visible · {}", sky.n_visible, sky.status),
        theme::mono(10.5),
        if err { RED } else { DIM },
    );
    painter.text(
        rect.left_top() + Vec2::new(10.0, 40.0),
        Align2::LEFT_TOP,
        format!("sky {:.1} ms · passes {:.0} ms", sky.compute_ms, passes.compute_ms),
        theme::mono(10.0),
        DIM,
    );
    {
        let adsb = shared.adsb.load();
        if adsb.mode != "off" {
            painter.text(
                rect.left_top() + Vec2::new(10.0, 54.0),
                Align2::LEFT_TOP,
                format!("ads-b: {} · {} aircraft · {} msgs", adsb.status, adsb.n_aircraft, adsb.n_msgs),
                theme::mono(10.0),
                if adsb.status.contains("giving up") || adsb.status.contains("not found") { RED } else { DIM },
            );
        }
    }
    // Legend.
    let mut ly = rect.bottom() - 14.0;
    for (name, col) in [("GEO", VIOLET), ("MEO", Color32::from_rgb(232, 150, 72)), ("LEO far", theme::sat_color(2400.0)), ("LEO near", theme::sat_color(400.0))] {
        painter.circle_filled(Pos2::new(rect.left() + 14.0, ly), 2.6, col);
        painter.text(Pos2::new(rect.left() + 22.0, ly), Align2::LEFT_CENTER, name, theme::sans(10.0), DIM);
        ly -= 14.0;
    }
    // Right-corner: mount pointing.
    painter.text(
        rect.right_top() + Vec2::new(-10.0, 8.0),
        Align2::RIGHT_TOP,
        format!("boresight {:7.3}° / {:6.3}°", mount.az, mount.el),
        theme::mono(11.0),
        AMBER,
    );

    // ---- Selected-object info pane (top-right, under the boresight line) ----
    let pane_w = 196.0;
    let px = rect.right() - pane_w - 8.0;
    let mut py = rect.top() + 26.0;
    let lb = Color32::from_rgb(150, 200, 255); // az
    let lg = Color32::from_rgb(150, 230, 150); // el
    let ly2 = Color32::from_rgb(235, 220, 140); // spot size
    if let Some(sel) = &st.selected {
        let mut title = String::new();
        let mut lines: Vec<(String, String, Color32)> = Vec::new();
        let mut kv = |lines: &mut Vec<(String, String, Color32)>, k: &str, v: String, c: Color32| lines.push((k.into(), v, c));
        if let Some(s) = sky.sats.iter().find(|s| &s.satnum == sel) {
            title = s.name.clone();
            kv(&mut lines, "norad", s.satnum.clone(), TEXT_2);
            kv(&mut lines, "intl desg", s.intl_desg.clone(), TEXT_2);
            kv(&mut lines, "epoch", fmt_unix(s.epoch_unix), TEXT_2);
            kv(&mut lines, "inclination", format!("{:.3}°", s.inclination_deg), TEXT_2);
            kv(&mut lines, "raan", format!("{:.3}°", s.raan_deg), TEXT_2);
            kv(&mut lines, "arg perigee", format!("{:.3}°", s.arg_perigee_deg), TEXT_2);
            kv(&mut lines, "mean anom", format!("{:.3}°", s.mean_anomaly_deg), TEXT_2);
            kv(&mut lines, "mean motion", format!("{:.4} rev/d", s.mean_motion_rev_day), TEXT_2);
            kv(&mut lines, "eccentricity", format!("{:.5}", s.eccentricity), TEXT_2);
            kv(&mut lines, "rev #", format!("{}", s.rev_number), TEXT_2);
            kv(&mut lines, "apogee", format!("{:.0} km", s.apogee_km), TEXT_2);
            kv(&mut lines, "perigee", format!("{:.0} km", s.perigee_km), TEXT_2);
            kv(&mut lines, "azimuth", format!("{:.2}°", (s.az + s.az_rate * age_s).rem_euclid(360.0)), lb);
            kv(&mut lines, "elevation", format!("{:.2}°", s.el + s.el_rate * age_s), lg);
            kv(&mut lines, "slant range", format!("{:.0} km", s.range_km), TEXT_2);
            kv(&mut lines, "rates", format!("{:+.3}/{:+.3}°/s", s.az_rate, s.el_rate), TEXT_2);
        } else if let Some(name) = sel.strip_prefix("body:") {
            if let Some(b) = sky.bodies.iter().find(|b| b.name == name) {
                title = b.name.clone();
                kv(&mut lines, "type", "solar system".into(), TEXT_2);
                kv(&mut lines, "azimuth", format!("{:.2}°", b.az), lb);
                kv(&mut lines, "elevation", format!("{:.2}°", b.el), lg);
                kv(&mut lines, "distance", if b.dist_km > 1e6 { format!("{:.4} au", b.dist_km / 1.495_978_707e8) } else { format!("{:.0} km", b.dist_km) }, TEXT_2);
            }
        } else if let Some(h) = sel.strip_prefix("star:") {
            let hip: i64 = h.parse().unwrap_or(-1);
            if let Some(s) = sky.stars.iter().find(|s| s.hip == hip) {
                title = sky.star_names.get(&hip).cloned().unwrap_or_else(|| format!("HIP {hip}"));
                kv(&mut lines, "type", "star".into(), TEXT_2);
                kv(&mut lines, "hip", format!("{hip}"), TEXT_2);
                kv(&mut lines, "v mag", format!("{:.2}", s.mag), TEXT_2);
                kv(&mut lines, "azimuth", format!("{:.2}°", s.az), lb);
                kv(&mut lines, "elevation", format!("{:.2}°", s.el), lg);
            }
        } else if let Some(k) = sel.strip_prefix("dso:") {
            if let Some(d) = sky.dsos.iter().find(|d| d.key == k) {
                title = d.name.clone();
                kv(&mut lines, "type", if d.messier { "Messier".into() } else { "NGC".into() }, TEXT_2);
                kv(&mut lines, "v mag", format!("{:.1}", d.mag), TEXT_2);
                kv(&mut lines, "azimuth", format!("{:.2}°", d.az), lb);
                kv(&mut lines, "elevation", format!("{:.2}°", d.el), lg);
            }
        } else if let Some(icao) = sel.strip_prefix("adsb:") {
            let adsb = shared.adsb.load();
            if let Some(a) = adsb.aircraft.iter().find(|a| a.icao == icao) {
                title = a.label.clone();
                kv(&mut lines, "type", "aircraft".into(), TEXT_2);
                kv(&mut lines, "icao", a.icao.clone(), TEXT_2);
                kv(&mut lines, "azimuth", format!("{:.2}°", a.az), lb);
                kv(&mut lines, "elevation", format!("{:.2}°", a.el), lg);
                kv(&mut lines, "range", format!("{:.1} km", a.range_km), TEXT_2);
                kv(&mut lines, "altitude", format!("{:.0} m", a.alt_m), TEXT_2);
                if let Some(s) = a.speed_kt {
                    kv(&mut lines, "speed", format!("{s:.0} kt"), TEXT_2);
                }
                if let Some(t) = a.track_deg {
                    kv(&mut lines, "track", format!("{t:.0}°"), TEXT_2);
                }
                kv(&mut lines, "fix age", format!("{:.1} s", a.age_s), TEXT_2);
            }
        } else if let Some(k) = sel.strip_prefix("launch:") {
            if let Some(l) = sky.launches.iter().find(|l| l.key == k) {
                title = l.name.clone();
                kv(&mut lines, "type", "launch traj".into(), TEXT_2);
                kv(&mut lines, "points", format!("{}", l.rows.len()), TEXT_2);
                if let Some(t) = &sky.target {
                    if &t.key == sel {
                        kv(&mut lines, "azimuth", format!("{:.2}°", t.az), lb);
                        kv(&mut lines, "elevation", format!("{:.2}°", t.el), lg);
                        kv(&mut lines, "rates", format!("{:+.3}/{:+.3}°/s", t.az_rate, t.el_rate), TEXT_2);
                    }
                }
            }
        }
        if !lines.is_empty() {
            py += info_pane(&painter, Pos2::new(px, py), pane_w, &title, ACCENT, &lines) + 6.0;
        }
    }
    // ---- Camera parameter / FOV cards, color-matched to the footprints ----
    for (i, c) in shared.cams.iter().enumerate() {
        if let Some(cam) = c.load().as_ref() {
            if !cam.connected {
                continue;
            }
            let s = shared.cam_settings[i].load();
            let col = CAM_COLORS[i % CAM_COLORS.len()];
            let fov_h = cam.deg_per_px * cam.height as f64;
            let (az_s, el_s) = if cam.fisheye {
                ("zenith".to_string(), format!("{:.1}°", 90.0))
            } else {
                (format!("{:.2}°", mount.az), format!("{:.2}°", mount.el))
            };
            let lines = vec![
                ("az".to_string(), az_s, lb),
                ("el".to_string(), el_s, lg),
                ("fov w".to_string(), format!("{:.3}°", cam.fov_deg), TEXT_2),
                ("fov h".to_string(), format!("{:.3}°", fov_h), TEXT_2),
                ("rotation".to_string(), format!("{:.1}°", s.rotation_deg), TEXT_2),
                ("spot size".to_string(), format!("{:.2}″/px", cam.deg_per_px * 3600.0), ly2),
            ];
            let name = cam.name.split(' ').next().unwrap_or("cam");
            py += info_pane(&painter, Pos2::new(px, py), pane_w, &format!("cam {} · {}", i + 1, name), col, &lines) + 6.0;
            if py > rect.bottom() - 60.0 {
                break;
            }
        }
    }
}

/// Reachability lattice over the sky disc (port of rendering_threads
/// _build_keepout_surface): +x = east, +y = south, r = (90 - el) / 90.
fn keepout_image(cfg: &crate::state::Config, n: usize) -> egui::ColorImage {
    use skytracker_core::transforms::{sky_to_mount, MountMode};
    let mode = crate::mount::parse_mount_mode(&cfg.mount_mode);
    let flips: &[bool] = if matches!(mode, MountMode::AltAz | MountMode::Passthrough) { &[false, true] } else { &[false] };
    let (azm_min, azm_max) = cfg.azm_limit;
    let (alt_min, alt_max) = cfg.alt_limit;
    let half = (n as f64 - 1.0) / 2.0;
    let mut px = vec![Color32::TRANSPARENT; n * n];
    for iy in 0..n {
        for ix in 0..n {
            let dx = (ix as f64 - half) / half;
            let dy = (iy as f64 - half) / half;
            let r = (dx * dx + dy * dy).sqrt();
            if r > 1.0 {
                continue;
            }
            let el = 90.0 * (1.0 - r);
            let az = dx.atan2(-dy).to_degrees().rem_euclid(360.0);
            let reachable = flips.iter().any(|&flip| {
                let (a, e) = if flip { ((az + 180.0).rem_euclid(360.0), 180.0 - el) } else { (az, el) };
                let (azm, alt) = sky_to_mount(mode, a, e, cfg.alignment_az, cfg.alignment_el, cfg.altaz_side_flip);
                azm_min <= azm && azm <= azm_max && alt_min <= alt && alt <= alt_max
            });
            if !reachable {
                px[iy * n + ix] = Color32::from_rgba_unmultiplied(255, 70, 70, 46);
            }
        }
    }
    egui::ColorImage { size: [n, n], pixels: px }
}

/// Gamma-stretch LUT (display only): out = 255 * (in/255)^gamma.
pub fn gamma_lut(gamma: f64) -> [u8; 256] {
    let g = gamma.clamp(0.05, 5.0);
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = ((i as f64 / 255.0).powf(g) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Upload the newest frame of `slot` as a texture (gamma applied when
/// enabled); returns the texture id + frame size when available.
pub fn cam_texture(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, slot: usize) -> Option<(egui::TextureId, usize, usize)> {
    let cam = shared.cam(slot)?;
    if cam.width == 0 || cam.height == 0 {
        return None;
    }
    while st.cam_tex.len() <= slot {
        st.cam_tex.push(None);
        st.cam_seq.push(u64::MAX);
    }
    let settings = shared.cam_settings[slot].load();
    // Re-upload on a new frame or when the gamma settings changed.
    let key = cam.seq ^ ((settings.gamma * 1000.0) as u64) << 40 ^ (settings.gamma_enabled as u64) << 63;
    if key != st.cam_seq[slot] {
        let img = if settings.gamma_enabled {
            let lut = gamma_lut(settings.gamma);
            let mapped: Vec<u8> = cam.data.iter().map(|&v| lut[v as usize]).collect();
            egui::ColorImage::from_gray([cam.width, cam.height], &mapped)
        } else {
            egui::ColorImage::from_gray([cam.width, cam.height], &cam.data)
        };
        match st.cam_tex[slot].as_mut() {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => st.cam_tex[slot] = Some(ui.ctx().load_texture(format!("cam{slot}"), img, egui::TextureOptions::LINEAR)),
        }
        st.cam_seq[slot] = key;
    }
    st.cam_tex[slot].as_ref().map(|t| (t.id(), cam.width, cam.height))
}

/// Draw a texture rotated by `angle_deg` about the centre of `rect`
/// (alignment rotation, positive = counter-clockwise like the Python view).
pub fn rotated_image(painter: &egui::Painter, tex: egui::TextureId, rect: Rect, angle_deg: f64, tint: Color32) {
    let c = rect.center();
    let (s, co) = (-(angle_deg.to_radians()) as f32).sin_cos();
    let rot = |p: Pos2| -> Pos2 {
        let d = p - c;
        Pos2::new(c.x + d.x * co - d.y * s, c.y + d.x * s + d.y * co)
    };
    let mut mesh = egui::Mesh::with_texture(tex);
    let corners = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()];
    let uvs = [Pos2::new(0.0, 0.0), Pos2::new(1.0, 0.0), Pos2::new(1.0, 1.0), Pos2::new(0.0, 1.0)];
    for (p, uv) in corners.iter().zip(uvs) {
        mesh.vertices.push(egui::epaint::Vertex { pos: rot(*p), uv, color: tint });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
}

/// Weighted overlay of several camera feeds into `rect` (co-boresighting):
/// painting layer k with alpha w_k / (w_1 + … + w_k) yields the exact
/// weighted mean of the layers, so the sliders shift which camera dominates.
/// `match_scale` sizes each feed by its plate scale (deg/px) relative to the
/// first camera; rotation is each feed's alignment rotation.
pub fn weighted_overlay(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, rect: Rect, cams: &[(usize, f32)], match_scale: bool) {
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, Color32::BLACK);
    let active: Vec<(usize, f32)> = cams.iter().copied().filter(|&(i, w)| w > 0.001 && shared.cam(i).map_or(false, |c| c.connected && c.width > 0)).collect();
    let Some(&(base_slot, _)) = active.first() else {
        p.text(rect.center(), Align2::CENTER_CENTER, "no feeds to overlay", theme::mono(11.0), DIM);
        return;
    };
    let base = shared.cam(base_slot).unwrap();
    let scale = (rect.width() / base.width as f32).min(rect.height() / base.height as f32);
    let mut wsum = 0.0f32;
    let mut names = Vec::new();
    for &(i, w) in &active {
        let Some(cam) = shared.cam(i) else { continue };
        let Some((tex, cw, ch)) = cam_texture(ui, shared, st, i) else { continue };
        let settings = shared.cam_settings[i].load();
        let k = if match_scale && base.deg_per_px > 0.0 && cam.deg_per_px > 0.0 {
            (cam.deg_per_px / base.deg_per_px) as f32
        } else {
            (base.width as f32 / cw as f32).min(base.height as f32 / ch as f32)
        };
        let size = Vec2::new(cw as f32 * scale * k, ch as f32 * scale * k);
        wsum += w;
        let alpha = ((w / wsum).clamp(0.0, 1.0) * 255.0) as u8;
        rotated_image(&p.with_clip_rect(rect), tex, Rect::from_center_size(rect.center(), size), settings.rotation_deg, Color32::from_white_alpha(alpha));
        names.push(format!("{} ×{:.2}", cam.name.split(' ').next().unwrap_or("cam"), w));
    }
    let c = rect.center();
    p.line_segment([c + Vec2::new(-16.0, 0.0), c + Vec2::new(16.0, 0.0)], Stroke::new(1.0, AMBER));
    p.line_segment([c + Vec2::new(0.0, -16.0), c + Vec2::new(0.0, 16.0)], Stroke::new(1.0, AMBER));
    p.text(rect.left_top() + Vec2::new(8.0, 8.0), Align2::LEFT_TOP, format!("combined · {}{}", names.join(" + "), if match_scale { " · plate-scale matched" } else { "" }), theme::mono(10.5), Color32::from_rgb(80, 200, 200));
}

pub fn camera_panel(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx_cam: &crossbeam_channel::Sender<CamCmd>, show_solve: bool) {
    // Camera selector.
    let n = shared.cams.len();
    ui.horizontal(|ui| {
        theme::section(ui, "camera");
        for i in 0..n {
            let c = shared.cam(i);
            let label = shared.config.cam.get(i).map(|c| c.name.split(' ').next().unwrap_or("cam").to_string()).unwrap_or_else(|| format!("cam {}", i + 1));
            let on = c.as_ref().map_or(false, |c| c.connected);
            if theme::mode_button(ui, &label, st.cam_view == i, if on { ACCENT } else { DIM }) {
                st.cam_view = i;
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(cam) = shared.cam(st.cam_view) {
                ui.label(egui::RichText::new(format!("{:.0} fps", cam.fps)).font(theme::mono(10.5)).color(TEXT_2));
                ui.label(egui::RichText::new(&cam.source).font(theme::mono(10.5)).color(DIM));
            }
        });
    });
    camera_view(ui, shared, st, st.cam_view, show_solve, None);
    camera_quick_controls(ui, shared, st.cam_view, tx_cam);
    let slot = st.cam_view;
    let Some(cam) = shared.cam(slot) else { return };
    ui.horizontal(|ui| {
        if theme::mode_button(ui, if cam.armed { "ARMED" } else { "ARM" }, cam.armed, RED) {
            let _ = tx_cam.send(if cam.armed { CamCmd::Disarm } else { CamCmd::Arm });
        }
        ui.add(egui::TextEdit::singleline(&mut st.capture_name).desired_width(110.0).hint_text("run name"));
        if ui.add_enabled(cam.armed, egui::Button::new("save run")).clicked() {
            let _ = tx_cam.send(CamCmd::Dump { name: st.capture_name.clone() });
        }
        ui.label(egui::RichText::new("(all connected cameras)").font(theme::sans(10.5)).color(DIM));
    });
    if let Some(d) = &cam.last_dump {
        ui.label(egui::RichText::new(d).font(theme::mono(10.0)).color(DIM));
    }
}

/// One camera's live view with the tracking / solve overlays. `height`
/// fixes the pane height (None = keep the frame aspect at the available width).
pub fn camera_view(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, slot: usize, show_solve: bool, height: Option<f32>) -> Option<Rect> {
    let mount = shared.mount.load();
    let Some(cam) = shared.cam(slot) else {
        ui.label(egui::RichText::new("waiting for frames…").color(DIM));
        return None;
    };
    if !cam.connected {
        let (r, p) = ui.allocate_painter(Vec2::new(ui.available_width(), height.unwrap_or(120.0)), Sense::hover());
        p.rect_filled(r.rect, 3.0, theme::BG);
        p.text(r.rect.center(), Align2::CENTER_CENTER, format!("{} · {}", cam.name, cam.source), theme::mono(11.0), DIM);
        return Some(r.rect);
    }
    let settings = shared.cam_settings[slot].load();
    let tex = cam_texture(ui, shared, st, slot);
    let avail = ui.available_width();
    let h = height.unwrap_or(avail * cam.height as f32 / cam.width as f32);
    let (r, p) = ui.allocate_painter(Vec2::new(avail, h), Sense::hover());
    p.rect_filled(r.rect, 0.0, Color32::BLACK);
    let p = p.with_clip_rect(r.rect);
    // Temporary pixel-zoom: mousewheel about the cursor, drag pans; purely a
    // display transform, independent of the hardware ROI.
    while st.cam_zoom.len() <= slot {
        st.cam_zoom.push(1.0);
        st.cam_pan.push(Vec2::ZERO);
    }
    let drag = ui.interact(r.rect, ui.id().with(("cam_pixel_zoom", slot)), Sense::drag());
    if let Some(ptr) = drag.hover_pos() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let old = st.cam_zoom[slot];
            let new = (old * (scroll * 0.0022).exp()).clamp(1.0, 16.0);
            if new != old {
                let cpos = r.rect.center() + st.cam_pan[slot];
                st.cam_pan[slot] = (ptr + (cpos - ptr) * (new / old)) - r.rect.center();
                st.cam_zoom[slot] = new;
            }
        }
    }
    if drag.dragged() && st.cam_zoom[slot] > 1.001 {
        st.cam_pan[slot] += drag.drag_delta();
    }
    if st.cam_zoom[slot] <= 1.001 {
        st.cam_zoom[slot] = 1.0;
        st.cam_pan[slot] = Vec2::ZERO;
    } else {
        let lim = Vec2::new(r.rect.width(), r.rect.height()) * st.cam_zoom[slot] * 0.6;
        st.cam_pan[slot] = st.cam_pan[slot].clamp(-lim, lim);
    }
    let zoom = st.cam_zoom[slot];
    // Fit the frame into the pane keeping aspect, then apply the pixel zoom.
    let scale = (r.rect.width() / cam.width as f32).min(r.rect.height() / cam.height as f32) * zoom;
    let img_size = Vec2::new(cam.width as f32 * scale, cam.height as f32 * scale);
    let img_rect = Rect::from_center_size(r.rect.center() + st.cam_pan[slot], img_size);
    if let Some((id, _, _)) = tex {
        rotated_image(&p, id, img_rect, settings.rotation_deg, Color32::WHITE);
    }
    let sx = img_rect.width() / cam.width as f32;
    let sy = img_rect.height() / cam.height as f32;
    let to_screen = |x: f64, y: f64| Pos2::new(img_rect.left() + x as f32 * sx, img_rect.top() + y as f32 * sy);
    let c = img_rect.center();
    // Reticle.
    let reticle = theme::with_alpha(AMBER, 200);
    p.line_segment([c + Vec2::new(-16.0, 0.0), c + Vec2::new(-5.0, 0.0)], Stroke::new(1.0, reticle));
    p.line_segment([c + Vec2::new(5.0, 0.0), c + Vec2::new(16.0, 0.0)], Stroke::new(1.0, reticle));
    p.line_segment([c + Vec2::new(0.0, -16.0), c + Vec2::new(0.0, -5.0)], Stroke::new(1.0, reticle));
    p.line_segment([c + Vec2::new(0.0, 5.0), c + Vec2::new(0.0, 16.0)], Stroke::new(1.0, reticle));
    // ROI box.
    if settings.roi_frac < 0.999 {
        let rw = img_rect.width() * settings.roi_frac as f32;
        let rh = img_rect.height() * settings.roi_frac as f32;
        let rc = Pos2::new(img_rect.left() + img_rect.width() * settings.roi_cx as f32, img_rect.top() + img_rect.height() * settings.roi_cy as f32);
        p.rect_stroke(Rect::from_center_size(rc, Vec2::new(rw, rh)), 0.0, Stroke::new(1.0, GREEN));
    }
    // Hotspot overlay (only on the hotspot camera).
    if slot == shared.hotspot_slot() && (mount.mode == "HOTSPOT" || mount.mode == "HANDOFF") {
        let gate = shared.config.hotspot_gate_radius as f32 * sx;
        p.circle_stroke(c, gate, Stroke::new(1.0, theme::with_alpha(GREEN, 60)));
        if let Some((cx, cy)) = mount.hotspot_centroid {
            let hp = to_screen(cx, cy);
            let col = if mount.hotspot_acquired { GREEN } else { theme::with_alpha(GREEN, 120) };
            p.circle_stroke(hp, 9.0, Stroke::new(1.4, col));
            p.line_segment([c, hp], Stroke::new(1.0, theme::with_alpha(col, 120)));
            p.text(hp + Vec2::new(12.0, 0.0), Align2::LEFT_CENTER, format!("snr {:.1}", mount.hotspot_snr), theme::mono(10.5), col);
        }
        p.text(r.rect.left_bottom() + Vec2::new(6.0, -6.0), Align2::LEFT_BOTTOM, format!("{}  {}", mount.mode, mount.hotspot_status), theme::mono(10.5), if mount.hotspot_acquired { GREEN } else { AMBER });
    }
    // Plate-solve overlay (solve camera).
    if show_solve && slot == shared.solve_slot() {
        let sv = shared.solve.load();
        if sv.frame_seq != 0 {
            for cpt in &sv.centroids {
                p.circle_stroke(to_screen(cpt[1], cpt[0]), 5.0, Stroke::new(0.8, theme::with_alpha(ACCENT, 140)));
            }
            for m in &sv.matched {
                p.circle_stroke(to_screen(m[1], m[0]), 7.0, Stroke::new(1.2, GREEN));
            }
        }
    }
    if zoom > 1.0 {
        let br = Rect::from_min_size(r.rect.right_bottom() + Vec2::new(-92.0, -24.0), Vec2::new(84.0, 18.0));
        let rb = ui.interact(br, ui.id().with(("cam_zoom_reset", slot)), Sense::click());
        p.rect(br, 4.0, theme::with_alpha(theme::RAISED, 230), Stroke::new(1.0, if rb.hovered() { ACCENT } else { HAIRLINE }));
        p.text(br.center(), Align2::CENTER_CENTER, format!("{zoom:.1}× · reset"), theme::mono(9.5), if rb.hovered() { TEXT } else { TEXT_2 });
        if rb.clicked() {
            st.cam_zoom[slot] = 1.0;
            st.cam_pan[slot] = Vec2::ZERO;
        }
    }
    if cam.armed {
        p.circle_filled(r.rect.right_top() + Vec2::new(-10.0, 10.0), 4.0, RED);
        p.text(r.rect.right_top() + Vec2::new(-18.0, 10.0), Align2::RIGHT_CENTER, format!("REC {}", cam.armed_frames), theme::mono(10.5), RED);
    }
    p.text(
        r.rect.left_top() + Vec2::new(6.0, 6.0),
        Align2::LEFT_TOP,
        format!("{}  FOV {:.2}°  {}×{}  #{}", cam.name, cam.fov_deg, cam.width, cam.height, cam.seq),
        theme::mono(10.0),
        theme::with_alpha(TEXT_2, 200),
    );
    if cam.deep_stars > 0 {
        p.text(r.rect.right_bottom() + Vec2::new(-6.0, -6.0), Align2::RIGHT_BOTTOM, format!("{} Tycho stars", cam.deep_stars), theme::mono(9.5), DIM);
    }
    Some(img_rect)
}

fn mode_color(mode: &str) -> Color32 {
    match mode {
        "RATE" => AMBER,
        "PROGRAM" | "HANDOFF" => ACCENT,
        "HOTSPOT" => GREEN,
        "MTI" => VIOLET,
        _ => TEXT_2,
    }
}

pub fn mount_panel(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx: &crossbeam_channel::Sender<MountCmd>) {
    let m = shared.mount.load();
    ui.horizontal(|ui| {
        theme::section(ui, "mount");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            theme::tag(ui, &m.mode, mode_color(&m.mode));
            ui.label(
                egui::RichText::new(format!("{} · {:.1} Hz", m.transport, m.actual_hz))
                    .font(theme::mono(10.5))
                    .color(if m.loop_dead { RED } else { DIM }),
            );
        });
    });
    ui.horizontal(|ui| {
        for mode in ["STANDBY", "RATE", "PROGRAM", "HANDOFF", "HOTSPOT", "MTI"] {
            if theme::mode_button(ui, mode, m.mode == mode, mode_color(mode)) {
                let _ = tx.send(MountCmd::SetMode(mode.to_string()));
            }
        }
        if theme::mode_button(ui, "STOP", m.stopped, RED) {
            let _ = tx.send(MountCmd::Stop);
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        theme::readout(ui, "azimuth", &format!("{:8.3}", m.az), "°", TEXT);
        ui.add_space(14.0);
        theme::readout(ui, "elevation", &format!("{:7.3}", m.el), "°", TEXT);
        ui.add_space(14.0);
        let ec = if m.az_error.abs() + m.el_error.abs() < 0.01 { GREEN } else { AMBER };
        theme::readout(ui, "error az / el", &format!("{:+.3} / {:+.3}", m.az_error, m.el_error), "°", ec);
    });
    ui.add_space(4.0);
    egui::Grid::new("mount_grid").num_columns(2).spacing([14.0, 3.0]).show(ui, |ui| {
        theme::kv(ui, "rate cmd", format!("{:+} / {:+}   gear {}", m.rate_cmd.0, m.rate_cmd.1, m.gear_ceiling));
        if m.bias != (0.0, 0.0) || m.parking {
            theme::kv_colored(ui, "bias", format!("{:+.2}° cross-el / {:+.2}° el  ({})", m.bias.0, m.bias.1, if m.bias_fine { "fine" } else { "coarse" }), AMBER);
        }
        if m.parking {
            theme::kv_colored(ui, "park", "driving to the configured offsets".to_string(), AMBER);
        }
        theme::kv(ui, "mount mode", format!("{}   (gamepad Options cycles)", m.mount_mode));
        if m.pointing_model_on {
            theme::kv_colored(ui, "pointing model", format!("ON  corr {:+.2}′ / {:+.2}′", m.pointing_corr.0 * 60.0, m.pointing_corr.1 * 60.0), GREEN);
        }
        if m.joystick_tare != (0.0, 0.0) {
            theme::kv(ui, "stick tare", format!("{:+.2} / {:+.2}", m.joystick_tare.0, m.joystick_tare.1));
        }
        theme::kv(ui, "joystick", format!("{}   {:+.2} / {:+.2}", m.joystick.clone().unwrap_or_else(|| "none".into()), m.stick.0, m.stick.1));
        let tname = {
            let sky = shared.sky.load();
            match (&m.target, &sky.target) {
                (Some(t), Some(tt)) if &tt.key == t => format!("{}  ({t})", tt.name),
                (Some(t), _) => t.clone(),
                (None, _) => "—".into(),
            }
        };
        theme::kv_colored(ui, "target", tname, if m.target.is_some() { ACCENT } else { DIM });
        if let Some((a, e)) = m.setpoint {
            theme::kv(ui, "setpoint", format!("{a:8.3}° / {e:7.3}°"));
        }
        if m.mode == "HANDOFF" {
            theme::kv_colored(ui, "handoff", format!("{} / {} detections", m.handoff_count, shared.config.handoff_min_frames), AMBER);
        }
        if m.mode == "HOTSPOT" {
            theme::kv_colored(ui, "hotspot", format!("{}  snr {:.1}", m.hotspot_status, m.hotspot_snr), if m.hotspot_acquired { GREEN } else { AMBER });
        }
        theme::kv(
            ui,
            "gains az",
            format!("{:.5} {:.5} {:.5}", m.gains[0][0], m.gains[0][1], m.gains[0][2]),
        );
        theme::kv(
            ui,
            "gains el",
            format!("{:.5} {:.5} {:.5}", m.gains[1][0], m.gains[1][1], m.gains[1][2]),
        );
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let tuning = m.autotune.is_some();
        if theme::mode_button(ui, if tuning { "autotune running" } else { "autotune" }, tuning, ACCENT) {
            let _ = tx.send(if tuning { MountCmd::AutotuneStop { revert: false } } else { MountCmd::AutotuneStart });
        }
        if tuning && ui.small_button("revert").clicked() {
            let _ = tx.send(MountCmd::AutotuneStop { revert: true });
        }
        if ui.small_button("tare").on_hover_text("zero the rate stick at its current position (gamepad: Square)").clicked() {
            let _ = tx.send(MountCmd::TareJoystick);
        }
        if ui.small_button("mode >").on_hover_text("cycle mount mode AltAz → AltAz-Side → Eq → Passthrough (gamepad: Options)").clicked() {
            let _ = tx.send(MountCmd::CycleMountMode);
        }
        if ui.small_button("park").on_hover_text("drive both axes to the configured offsets (gamepad: Triangle)").clicked() {
            let _ = tx.send(MountCmd::Park);
        }
        let step = if m.bias_fine { 0.01 } else { 0.1 };
        let ac = m.bias_frame == "alongcross";
        let (hn, vn) = if ac { ("InTk", "XTk") } else { ("Az", "El") };
        for (l, dx, dy) in [("◀", -step, 0.0), ("▶", step, 0.0), ("▲", 0.0, step), ("▼", 0.0, -step)] {
            let axis = if dx != 0.0 { hn } else { vn };
            if ui.small_button(l).on_hover_text(format!("operator bias {axis} {:+.2}° (gamepad D-pad; Share cycles the mode)", if dx != 0.0 { dx } else { dy })).clicked() {
                let _ = tx.send(MountCmd::Bias { daz: dx, del: dy });
            }
        }
        let mode_lbl = format!("{}·{}", if m.bias_fine { "fine" } else { "coarse" }, if ac { "trk" } else { "azel" });
        if ui.small_button(mode_lbl).on_hover_text("bias mode: coarse/azel → fine/azel → coarse/along-cross-track → fine/along-cross-track (gamepad: Share)").clicked() {
            let _ = tx.send(MountCmd::CycleBiasMode);
        }
        if (m.bias != (0.0, 0.0) || m.bias_it != 0.0 || m.bias_ct != 0.0) && ui.small_button("bias 0").clicked() {
            let _ = tx.send(MountCmd::BiasReset);
        }
        // Per-axis feed-forward (render_pid_diagnostics FF buttons).
        ui.separator();
        if theme::mode_button(ui, "FFaz", m.ff_azm, GREEN) {
            let _ = tx.send(MountCmd::SetFeedForward { az: !m.ff_azm, el: m.ff_alt });
        }
        if theme::mode_button(ui, "FFel", m.ff_alt, GREEN) {
            let _ = tx.send(MountCmd::SetFeedForward { az: m.ff_azm, el: !m.ff_alt });
        }
    });
    if m.bias_it != 0.0 || m.bias_ct != 0.0 {
        theme::kv_colored(ui, "bias trk", format!("in {:+.2}° / cross {:+.2}°", m.bias_it, m.bias_ct), AMBER);
    }
    if let Some(t) = &m.autotune {
        ui.label(egui::RichText::new(t).font(theme::mono(10.5)).color(ACCENT));
    }
    // Inline PID tuning: edits apply to the loop immediately (SetGains).
    // Per-mode gain profiles still load on mode change — "sync from loop"
    // re-reads the live gains after that.
    egui::CollapsingHeader::new(egui::RichText::new("PID GAINS").font(theme::sans(10.5)).color(DIM)).id_salt("pid_gains").default_open(false).show(ui, |ui| {
        let mut g = st.gains_edit.unwrap_or(m.gains);
        let before = g;
        egui::Grid::new("gains_grid").num_columns(4).spacing([8.0, 3.0]).show(ui, |ui| {
            ui.label(egui::RichText::new("").small());
            for h in ["P", "I", "D"] {
                ui.label(egui::RichText::new(h).color(DIM));
            }
            ui.end_row();
            for (i, name) in ["az", "el"].iter().enumerate() {
                ui.label(egui::RichText::new(*name).color(TEXT_2));
                for j in 0..3 {
                    ui.add(egui::DragValue::new(&mut g[i][j]).speed(0.00001).max_decimals(6));
                }
                ui.end_row();
            }
        });
        if g != before {
            st.gains_edit = Some(g);
            let _ = tx.send(MountCmd::SetGains { azm: (g[0][0], g[0][1], g[0][2]), alt: (g[1][0], g[1][1], g[1][2]) });
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("lead").font(theme::sans(10.0)).color(DIM));
            let mut lead = m.lead_time_s;
            if ui.add(egui::Slider::new(&mut lead, 0.0..=0.5).fixed_decimals(2).suffix(" s")).changed() {
                let _ = tx.send(MountCmd::SetLeadTime(lead));
            }
            let mut sf = m.star_filter;
            if ui.checkbox(&mut sf, egui::RichText::new("star filter").font(theme::sans(10.0)).color(TEXT_2)).on_hover_text("HOTSPOT: reject star-like near-zero-rate detections; off to deliberately track a star").changed() {
                let _ = tx.send(MountCmd::SetStarFilter(sf));
            }
        });
        ui.horizontal(|ui| {
            if st.gains_edit.is_some() {
                if ui.small_button("sync from loop").clicked() {
                    st.gains_edit = None;
                }
                ui.label(egui::RichText::new("edited · applying live").font(theme::sans(9.5)).color(AMBER));
            } else {
                ui.label(egui::RichText::new("live loop gains · drag to tune").font(theme::sans(9.5)).color(DIM));
            }
        });
    });
    // PID diagnostics: tracking-error and rate-command strip charts + navball
    // (render_tracking_strip_charts / render_pid_diagnostics / render_navball).
    {
        let t = st.history_t0.elapsed().as_secs_f64();
        let sample = (t, m.az_error, m.el_error, m.rate_cmd.0 as f64, m.rate_cmd.1 as f64);
        if st.history.back().map_or(true, |h| (h.0 - t).abs() > 1.0 / 30.0) {
            st.history.push_back(sample);
            while st.history.len() > 1800 {
                st.history.pop_front();
            }
        }
    }
    egui::CollapsingHeader::new(egui::RichText::new("DIAGNOSTICS").font(theme::sans(10.5)).color(DIM)).id_salt("diag").default_open(true).show(ui, |ui| {
        let tracking = matches!(m.mode.as_str(), "PROGRAM" | "HANDOFF" | "HOTSPOT");
        strip_chart(ui, &st.history, |h| (h.1, h.2), "error az / el  (°)", tracking, true);
        strip_chart(ui, &st.history, |h| (h.3, h.4), "rate cmd az / el  (steps)", tracking, false);
    });
    egui::CollapsingHeader::new(egui::RichText::new("NAVBALL").font(theme::sans(10.5)).color(DIM)).id_salt("navball").default_open(true).show(ui, |ui| {
        navball(ui, shared, st, &m);
    });
    egui::CollapsingHeader::new(egui::RichText::new("JOYSTICK").font(theme::sans(10.5)).color(DIM)).id_salt("joypanel").default_open(true).show(ui, |ui| {
        joystick_panel(ui, &m, shared.config.joystick_rate_stick.eq_ignore_ascii_case("left"));
    });
    ui.add_space(4.0);
    // Connection controls (mirrors render_connection_controls / ADS-B controls).
    if !st.conn_init {
        st.conn_init = true;
        st.conn_transport = shared.config.mount_transport.clone();
        st.conn_port = shared.config.serial_port.clone();
        st.conn_baud = shared.config.serial_baud;
        st.adsb_mode = shared.config.adsb_source_mode.clone();
        let _ = tx.send(MountCmd::ListPorts);
    }
    egui::CollapsingHeader::new(egui::RichText::new("CONNECTION").font(theme::sans(10.5)).color(DIM)).id_salt("conn").show(ui, |ui| {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("conn_tr").selected_text(st.conn_transport.clone()).width(70.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut st.conn_transport, "sim".into(), "sim");
                ui.selectable_value(&mut st.conn_transport, "serial".into(), "serial");
            });
            if st.conn_transport == "serial" {
                let ports = shared.serial_ports.load();
                egui::ComboBox::from_id_salt("conn_port").selected_text(if st.conn_port.is_empty() { "port".to_string() } else { st.conn_port.clone() }).width(90.0).show_ui(ui, |ui| {
                    for p in ports.iter() {
                        ui.selectable_value(&mut st.conn_port, p.clone(), p);
                    }
                });
                if ui.small_button("⟳").on_hover_text("rescan serial ports").clicked() {
                    let _ = tx.send(MountCmd::ListPorts);
                }
                ui.add(egui::DragValue::new(&mut st.conn_baud).speed(100.0).range(1200..=115200));
            }
            let connected = !m.loop_dead && m.connected;
            if theme::mode_button(ui, "CONNECT", connected && !m.transport.contains("disconnected"), GREEN) {
                let _ = tx.send(MountCmd::Connect { transport: st.conn_transport.clone(), port: st.conn_port.clone(), baud: st.conn_baud });
            }
            if ui.small_button("disconnect").clicked() {
                let _ = tx.send(MountCmd::Disconnect);
            }
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("ADS-B").color(TEXT_2));
            egui::ComboBox::from_id_salt("adsb_src").selected_text(st.adsb_mode.clone()).width(90.0).show_ui(ui, |ui| {
                for m in ["off", "rtlsdr", "dump1090", "sim"] {
                    ui.selectable_value(&mut st.adsb_mode, m.into(), m);
                }
            });
            if ui.small_button("connect").clicked() {
                shared.adsb_request.store(Arc::new(Some(st.adsb_mode.clone())));
            }
            if ui.small_button("disconnect").clicked() {
                shared.adsb_request.store(Arc::new(Some("off".into())));
            }
            let a = shared.adsb.load();
            ui.label(egui::RichText::new(format!("{} · {} ac", a.status, a.n_aircraft)).font(theme::mono(10.0)).color(DIM));
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("fit points").font(theme::sans(10.0)).color(DIM));
            let mut n = shared.adsb_fit_points.load(std::sync::atomic::Ordering::Relaxed) as i64;
            if ui.add(egui::Slider::new(&mut n, 2..=20)).on_hover_text("ADS-B linear-fit depth: recent fixes used for the trajectory prediction").changed() {
                shared.adsb_fit_points.store(n as usize, std::sync::atomic::Ordering::Relaxed);
                crate::mount::persist_config_key(&shared.config.path, "adsb_fit_points", serde_json::json!(n));
            }
        });
    });
    ui.add_space(4.0);
    theme::section(ui, "log");
    for s in m.status.iter().rev().take(6) {
        ui.label(egui::RichText::new(s).font(theme::mono(10.5)).color(TEXT_2));
    }
}

/// Compact mount readouts for camera-first layouts: mode buttons, az/el,
/// error, hotspot state — the essentials in ~150 px of height.
pub fn compact_mount(ui: &mut egui::Ui, shared: &Arc<Shared>, tx: &crossbeam_channel::Sender<MountCmd>) {
    let m = shared.mount.load();
    ui.horizontal_wrapped(|ui| {
        for mode in ["STANDBY", "RATE", "PROGRAM", "HANDOFF", "HOTSPOT"] {
            if theme::mode_button(ui, mode, m.mode == mode, mode_color(mode)) {
                let _ = tx.send(MountCmd::SetMode(mode.to_string()));
            }
        }
        if theme::mode_button(ui, "STOP", m.stopped, RED) {
            let _ = tx.send(MountCmd::Stop);
        }
    });
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("{:8.3}° / {:7.3}°", m.az, m.el)).font(theme::mono(15.0)).color(theme::TEXT));
        let ec = if m.az_error.abs() + m.el_error.abs() < 0.01 { GREEN } else { AMBER };
        ui.label(egui::RichText::new(format!("err {:+.3} / {:+.3}", m.az_error, m.el_error)).font(theme::mono(12.0)).color(ec));
    });
    ui.horizontal(|ui| {
        if let Some(t) = &m.target {
            ui.label(egui::RichText::new(format!("target {t}")).font(theme::mono(10.5)).color(ACCENT));
        }
        if m.mode == "HOTSPOT" || m.mode == "HANDOFF" {
            ui.label(egui::RichText::new(format!("{} snr {:.1}", m.hotspot_status, m.hotspot_snr)).font(theme::mono(10.5)).color(if m.hotspot_acquired { GREEN } else { AMBER }));
        }
    });
}

/// A two-trace strip chart over the last 30 s (az amber, el accent).
fn strip_chart(ui: &mut egui::Ui, hist: &std::collections::VecDeque<(f64, f64, f64, f64, f64)>, pick: impl Fn(&(f64, f64, f64, f64, f64)) -> (f64, f64), title: &str, live: bool, symmetric_auto: bool) {
    let (r, p) = ui.allocate_painter(Vec2::new(ui.available_width(), 64.0), Sense::hover());
    p.rect_filled(r.rect, 3.0, theme::BG);
    p.rect_stroke(r.rect, 3.0, Stroke::new(1.0, HAIRLINE));
    let window = 30.0;
    let t1 = hist.back().map(|h| h.0).unwrap_or(0.0);
    let t0 = t1 - window;
    let pts: Vec<(f64, f64, f64)> = hist.iter().filter(|h| h.0 >= t0).map(|h| { let (a, b) = pick(h); (h.0, a, b) }).collect();
    let mut amp = pts.iter().fold(0.0f64, |m, &(_, a, b)| m.max(a.abs()).max(b.abs()));
    if symmetric_auto {
        amp = if amp < 0.01 { 0.01 } else { amp * 1.15 };
    } else {
        amp = amp.max(1.0);
    }
    let inner = r.rect.shrink2(Vec2::new(4.0, 6.0));
    let y_of = |v: f64| inner.center().y - (v / amp) as f32 * (inner.height() / 2.0);
    let x_of = |t: f64| inner.left() + ((t - t0) / window) as f32 * inner.width();
    p.line_segment([Pos2::new(inner.left(), y_of(0.0)), Pos2::new(inner.right(), y_of(0.0))], Stroke::new(1.0, theme::with_alpha(HAIRLINE, 200)));
    let alpha = if live { 230 } else { 90 };
    for (idx, col) in [(1usize, AMBER), (2usize, ACCENT)] {
        let mut prev: Option<Pos2> = None;
        for &(t, a, b) in &pts {
            let v = if idx == 1 { a } else { b };
            let q = Pos2::new(x_of(t), y_of(v).clamp(inner.top(), inner.bottom()));
            if let Some(pp) = prev {
                p.line_segment([pp, q], Stroke::new(1.2, theme::with_alpha(col, alpha)));
            }
            prev = Some(q);
        }
    }
    p.text(r.rect.left_top() + Vec2::new(5.0, 3.0), Align2::LEFT_TOP, title, theme::mono(9.5), TEXT_2);
    p.text(r.rect.right_top() + Vec2::new(-5.0, 3.0), Align2::RIGHT_TOP, format!("±{:.3}", amp), theme::mono(9.5), DIM);
    if let Some(&(_, a, b)) = pts.last() {
        p.text(r.rect.right_bottom() + Vec2::new(-5.0, -3.0), Align2::RIGHT_BOTTOM, format!("{a:+.3}  {b:+.3}"), theme::mono(9.5), if live { TEXT } else { DIM });
    }
}

/// KSP-style navball: heading tape at the top, pitch ladder under a fixed
/// boresight reticle; the ball moves, the reticle stays. Target shown as a
/// green marker when a setpoint exists.
/// KSP-style navball (port of joystick_panels.render_navball): orthographic
/// ball with sky/ground hemispheres, meridians + parallels, cardinal letters,
/// pitch numerals, bezel, fixed "waterline" boresight reticle, the selected
/// target's trajectory + purple crosshair, ADS-B diamonds, and a green
/// HDG/PITCH readout. The hemisphere fill is a texture cached on quantized
/// elevation (the fill depends only on el); the grid is redrawn per frame.
fn navball(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, m: &crate::state::MountSnapshot) {
    let size = ui.available_width().min(260.0);
    let (r, p) = ui.allocate_painter(Vec2::new(ui.available_width(), size + 22.0), Sense::hover());
    let c = Pos2::new(r.rect.center().x, r.rect.top() + size / 2.0 - 4.0);
    let rad = size / 2.0 - 16.0;
    let connected = m.connected && !m.loop_dead;
    let (az, el) = if connected { (m.az, m.el) } else { (0.0, 15.0) };
    // Quantize the pointing so the cached fill + grid geometry snap (sub-pixel).
    let gaz = (az / 0.5).round() * 0.5;
    let gel = (el / 0.5).round() * 0.5;
    let (a0, e0) = (gaz.to_radians(), gel.to_radians());
    let (ce0, se0, sa0, ca0) = (e0.cos(), e0.sin(), a0.sin(), a0.cos());
    // World axes X=east Y=up Z=north; ball oriented so the pointing sits front-center.
    let zc = [ce0 * sa0, se0, ce0 * ca0];
    let yc = [-se0 * sa0, ce0, -se0 * ca0];
    let xc = [
        yc[1] * zc[2] - yc[2] * zc[1],
        yc[2] * zc[0] - yc[0] * zc[2],
        yc[0] * zc[1] - yc[1] * zc[0],
    ];
    let proj = move |az_deg: f64, el_deg: f64| -> (Pos2, f64) {
        let (ar, er) = (az_deg.to_radians(), el_deg.to_radians());
        let ce = er.cos();
        let v = [ce * ar.sin(), er.sin(), ce * ar.cos()];
        let z = v[0] * zc[0] + v[1] * zc[1] + v[2] * zc[2];
        let x = v[0] * xc[0] + v[1] * xc[1] + v[2] * xc[2];
        let y = v[0] * yc[0] + v[1] * yc[1] + v[2] * yc[2];
        (Pos2::new(c.x + x as f32 * rad, c.y - y as f32 * rad), z)
    };
    // Hemisphere fill: world-up in the camera frame is (0, cos e, sin e), so
    // the texture depends only on elevation — one-deep cache on gel.
    let key = (gel * 2.0).round() as i32;
    if st.navball_tex.as_ref().map_or(true, |(_, k)| *k != key) {
        const NT: usize = 192;
        let mut px = vec![Color32::TRANSPARENT; NT * NT];
        let half = (NT as f64 - 1.0) / 2.0;
        for iy in 0..NT {
            for ix in 0..NT {
                let nx = (ix as f64 - half) / half;
                let ny = (half - iy as f64) / half;
                let rr = nx * nx + ny * ny;
                if rr > 1.0 {
                    continue;
                }
                let nz = (1.0 - rr).sqrt();
                let vy = ce0 * ny + se0 * nz;
                let (br, bg, bb) = if vy >= 0.0 { (86.0, 150.0, 220.0) } else { (150.0, 110.0, 70.0) };
                let sh = 0.55 + 0.45 * nz; // spherical shading
                px[iy * NT + ix] = Color32::from_rgb((br * sh) as u8, (bg * sh) as u8, (bb * sh) as u8);
            }
        }
        let img = egui::ColorImage { size: [NT, NT], pixels: px };
        let tex = ui.ctx().load_texture("navball_ball", img, egui::TextureOptions::LINEAR);
        st.navball_tex = Some((tex, key));
    }
    let tint = if connected { Color32::WHITE } else { Color32::from_rgba_unmultiplied(255, 255, 255, 80) };
    if let Some((tex, _)) = &st.navball_tex {
        p.image(tex.id(), Rect::from_center_size(c, Vec2::splat(rad * 2.0)), Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), tint);
    }
    let clip = p.with_clip_rect(Rect::from_center_size(c, Vec2::splat(rad * 2.0 + 2.0)));
    let alpha: u8 = if connected { 255 } else { 80 };
    let wa = |col: Color32| theme::with_alpha(col, alpha);
    // Grid: azimuth meridians (tines) + elevation parallels + bright horizon.
    let draw_curve = |samples: &mut dyn Iterator<Item = (f64, f64)>, col: Color32, width: f32| {
        let mut run: Vec<Pos2> = Vec::new();
        for (a_d, e_d) in samples {
            let (pt, z) = proj(a_d, e_d);
            if z > 0.0 {
                run.push(pt);
            } else if run.len() >= 2 {
                clip.add(egui::Shape::line(std::mem::take(&mut run), Stroke::new(width, col)));
            } else {
                run.clear();
            }
        }
        if run.len() >= 2 {
            clip.add(egui::Shape::line(run, Stroke::new(width, col)));
        }
    };
    let meridian = wa(Color32::from_rgb(206, 210, 222));
    let sky_line = wa(Color32::from_rgb(225, 234, 246));
    let gnd_line = wa(Color32::from_rgb(212, 196, 176));
    for az_line in (0..360).step_by(30) {
        draw_curve(&mut (-17..=17).map(|e| (az_line as f64, e as f64 * 5.0)), meridian, 1.0);
    }
    for el_line in (-8..=8).map(|k| k * 10) {
        if el_line == 0 {
            continue;
        }
        let col = if el_line > 0 { sky_line } else { gnd_line };
        draw_curve(&mut (0..=72).map(|a| (a as f64 * 5.0, el_line as f64)), col, if el_line % 30 == 0 { 1.5 } else { 0.8 });
    }
    draw_curve(&mut (0..=72).map(|a| (a as f64 * 5.0, 0.0)), wa(Color32::from_rgb(250, 250, 252)), 2.2);
    // Pitch numerals in a gap between tines (green chips, KSP style).
    let gap_az = (gaz / 30.0).floor() * 30.0 + 15.0;
    for el_line in [-60i32, -30, 30, 60] {
        let (pt, z) = proj(gap_az, el_line as f64);
        if z > 0.0 {
            let galley = clip.layout_no_wrap(format!("{el_line:+}"), theme::mono(9.0), wa(Color32::from_rgb(130, 255, 140)));
            let bgr = Rect::from_center_size(pt, galley.size() + Vec2::new(6.0, 2.0));
            clip.rect_filled(bgr, 2.0, theme::with_alpha(Color32::from_rgb(16, 26, 18), alpha.saturating_sub(40)));
            clip.galley(bgr.min + Vec2::new(3.0, 1.0), galley, wa(Color32::from_rgb(130, 255, 140)));
        }
    }
    // Cardinal / intercardinal letters where their meridian meets the horizon.
    for (h, lbl) in [(0.0, "N"), (45.0, "NE"), (90.0, "E"), (135.0, "SE"), (180.0, "S"), (225.0, "SW"), (270.0, "W"), (315.0, "NW")] {
        let (pt, z) = proj(h, 0.0);
        if z <= 0.0 {
            continue;
        }
        let col = if lbl.len() == 1 { wa(Color32::from_rgb(255, 230, 110)) } else { wa(Color32::from_rgb(214, 218, 228)) };
        clip.text(pt + Vec2::new(0.0, -3.0), Align2::CENTER_BOTTOM, lbl, theme::sans(10.5), col);
    }
    if connected {
        // Selected target's arc (grey past, yellow future) — mirrors the skyplot.
        let passes = shared.passes.load();
        for w in passes.arc.windows(2) {
            if w[0].el <= 0.0 || w[1].el <= 0.0 {
                continue;
            }
            let (p0, z0) = proj(w[0].az, w[0].el);
            let (p1, z1) = proj(w[1].az, w[1].el);
            if z0 <= 0.0 || z1 <= 0.0 {
                continue;
            }
            let col = if w[1].t_rel_s < 0.0 { Color32::from_rgb(130, 130, 130) } else { Color32::from_rgb(255, 255, 0) };
            clip.line_segment([p0, p1], Stroke::new(1.6, col));
        }
        // Live target: purple KSP crosshair at the commanded setpoint.
        if let Some((taz, tel)) = m.setpoint {
            let (pt, z) = proj(taz, tel);
            if z > 0.0 {
                let purple = Color32::from_rgb(205, 95, 255);
                let rr = (rad * 0.075).max(6.0);
                clip.circle_stroke(pt, rr, Stroke::new(1.6, purple));
                for (dx, dy) in [(-1.0f32, 0.0f32), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                    clip.line_segment([pt + Vec2::new(dx * rr, dy * rr), pt + Vec2::new(dx * (rr + 6.0), dy * (rr + 6.0))], Stroke::new(1.6, purple));
                }
                clip.circle_filled(pt, 1.4, purple);
            }
        }
        // ADS-B aircraft: orange diamonds, white ring on the selection.
        let adsb = shared.adsb.load();
        for a in adsb.aircraft.iter() {
            if a.el <= 0.0 {
                continue;
            }
            let (pt, z) = proj(a.az, a.el);
            if z <= 0.0 {
                continue;
            }
            let d = (rad * 0.03).max(3.0);
            let ac = Color32::from_rgb(255, 170, 60);
            clip.add(egui::Shape::convex_polygon(
                vec![pt + Vec2::new(0.0, -d), pt + Vec2::new(d, 0.0), pt + Vec2::new(0.0, d), pt + Vec2::new(-d, 0.0)],
                ac,
                Stroke::NONE,
            ));
            if st.selected.as_deref() == Some(format!("adsb:{}", a.icao).as_str()) {
                clip.circle_stroke(pt, d + 3.0, Stroke::new(1.5, Color32::WHITE));
            }
        }
    }
    // Bezel ring, 30° ticks, fixed heading index.
    let bezel = wa(Color32::from_rgb(188, 190, 200));
    p.circle_stroke(c, rad + 1.0, Stroke::new(2.0, bezel));
    for deg in (0..360).step_by(30) {
        let ang = (deg as f32 - 90.0).to_radians();
        let (ca, sa) = (ang.cos(), ang.sin());
        p.line_segment(
            [Pos2::new(c.x + (rad + 2.0) * ca, c.y + (rad + 2.0) * sa), Pos2::new(c.x + (rad + 8.0) * ca, c.y + (rad + 8.0) * sa)],
            Stroke::new(2.0, bezel),
        );
    }
    p.add(egui::Shape::convex_polygon(
        vec![Pos2::new(c.x, c.y - rad + 1.0), Pos2::new(c.x - 6.0, c.y - rad - 9.0), Pos2::new(c.x + 6.0, c.y - rad - 9.0)],
        wa(Color32::from_rgb(255, 230, 110)),
        Stroke::NONE,
    ));
    // Fixed boresight "waterline" reticle (yellow aircraft marker).
    let yellow = wa(Color32::from_rgb(255, 214, 0));
    let r0 = (rad * 0.055).max(4.0);
    let wing = (rad * 0.20).max(10.0);
    let drop = (wing / 3.0).max(3.0);
    p.circle_stroke(c, r0, Stroke::new(2.0, yellow));
    for sx in [-1.0f32, 1.0] {
        let x_in = c.x + sx * (r0 + 2.0);
        let x_out = c.x + sx * (r0 + wing);
        p.line_segment([Pos2::new(x_in, c.y), Pos2::new(x_out, c.y)], Stroke::new(2.5, yellow));
        p.line_segment([Pos2::new(x_out, c.y), Pos2::new(x_out, c.y + drop)], Stroke::new(2.5, yellow));
    }
    p.line_segment([Pos2::new(c.x, c.y - r0 - 2.0), Pos2::new(c.x, c.y - r0 - (wing / 2.0).max(5.0))], Stroke::new(2.5, yellow));
    // Green HDG / PITCH instrument box (past-pole orientations normalized).
    let box_w = 2.0 * rad * 0.95;
    let bx = Rect::from_min_size(Pos2::new(c.x - box_w / 2.0, c.y + rad + 12.0), Vec2::new(box_w, 19.0));
    p.rect(bx, 2.0, Color32::from_rgb(16, 26, 18), Stroke::new(1.0, Color32::from_rgb(70, 110, 80)));
    let readout = if connected {
        let mut el_d = (el + 180.0).rem_euclid(360.0) - 180.0;
        let mut az_d = az;
        if el_d > 90.0 {
            az_d += 180.0;
            el_d = 180.0 - el_d;
        } else if el_d < -90.0 {
            az_d += 180.0;
            el_d = -180.0 - el_d;
        }
        format!("HDG {:05.1}°   PITCH {:+5.1}°", az_d.rem_euclid(360.0), el_d)
    } else {
        "HDG ---.-°   PITCH --.-°".to_string()
    };
    p.text(bx.center(), Align2::CENTER_CENTER, readout, theme::mono(11.0), if connected { Color32::from_rgb(130, 255, 140) } else { Color32::from_rgb(110, 130, 110) });
    if !connected {
        p.text(c, Align2::CENTER_CENTER, "mount offline", theme::mono(10.5), TEXT_2);
    }
}

/// Virtual-controller panel (port of render_joystick_status): a stylized
/// DualShock with live button/stick/trigger state, the rate-stick crosshair,
/// the tare marker, the adaptive-rate gear ladder, and a function legend
/// that highlights while a control is held.
pub fn joystick_panel(ui: &mut egui::Ui, m: &crate::state::MountSnapshot, rate_stick_left: bool) {
    let connected = m.joystick.is_some();
    ui.label(egui::RichText::new(format!("joystick: {}", m.joystick.clone().unwrap_or_else(|| "none".into()))).font(theme::mono(10.5)).color(if connected { TEXT_2 } else { RED }));
    let w = ui.available_width().min(340.0);
    let pad_h = w * 0.55;
    let (r, p) = ui.allocate_painter(Vec2::new(ui.available_width(), pad_h), Sense::hover());
    let body = Rect::from_center_size(Pos2::new(r.rect.center().x, r.rect.top() + pad_h * 0.60), Vec2::new(w * 0.94, pad_h * 0.70));
    let alpha: u8 = if connected { 255 } else { 90 };
    let wa = |col: Color32| theme::with_alpha(col, alpha);
    let btn = |bit: u32| connected && (m.joy_buttons >> bit) & 1 == 1;
    let off_fill = theme::with_alpha(theme::BG, alpha);
    p.rect(body, 26.0, theme::with_alpha(theme::RAISED, alpha.saturating_sub(55)), Stroke::new(1.0, wa(HAIRLINE)));
    let at = |fx: f32, fy: f32| Pos2::new(body.left() + body.width() * fx, body.top() + body.height() * fy);
    // Shoulders: L1/R1 buttons with L2/R2 analog fill bars above them.
    for (side, l2v, l1bit, l1, l2) in [(-1.0f32, m.joy_triggers.0, 9u32, "L1", "L2"), (1.0, m.joy_triggers.1, 10, "R1", "R2")] {
        let x = body.center().x + side * body.width() * 0.32;
        let l2r = Rect::from_center_size(Pos2::new(x, body.top() - 22.0), Vec2::new(46.0, 8.0));
        p.rect(l2r, 3.0, off_fill, Stroke::new(1.0, wa(HAIRLINE)));
        let fillw = l2r.width() * (l2v as f32).clamp(0.0, 1.0);
        if fillw > 0.5 {
            p.rect_filled(Rect::from_min_size(l2r.min, Vec2::new(fillw, l2r.height())), 3.0, AMBER);
        }
        p.text(l2r.right_center() + Vec2::new(4.0, 0.0), Align2::LEFT_CENTER, l2, theme::mono(7.5), wa(DIM));
        let l1r = Rect::from_center_size(Pos2::new(x, body.top() - 9.0), Vec2::new(46.0, 11.0));
        p.rect(l1r, 4.0, if btn(l1bit) { GREEN } else { off_fill }, Stroke::new(1.0, wa(HAIRLINE)));
        p.text(l1r.center(), Align2::CENTER_CENTER, l1, theme::mono(8.5), if btn(l1bit) { Color32::BLACK } else { wa(TEXT_2) });
    }
    // D-pad cross (bias).
    let dc = at(0.17, 0.36);
    let sq = w * 0.047;
    let off = w * 0.052;
    for (bit, dx, dy) in [(11u32, 0.0f32, -1.0f32), (12, 0.0, 1.0), (13, -1.0, 0.0), (14, 1.0, 0.0)] {
        let cr = Rect::from_center_size(dc + Vec2::new(dx * off, dy * off), Vec2::splat(sq));
        p.rect(cr, 2.0, if btn(bit) { GREEN } else { off_fill }, Stroke::new(1.0, wa(HAIRLINE)));
    }
    // Face buttons (PS symbols, PS colors; pressed = green fill).
    let fc = at(0.83, 0.36);
    let fr = w * 0.034;
    let foff = w * 0.058;
    for (bit, dx, dy, sym, col) in [
        (3u32, 0.0f32, -1.0f32, "△", Color32::from_rgb(64, 226, 160)),
        (1, 1.0, 0.0, "○", Color32::from_rgb(255, 102, 102)),
        (0, 0.0, 1.0, "✕", Color32::from_rgb(124, 178, 232)),
        (2, -1.0, 0.0, "□", Color32::from_rgb(255, 105, 248)),
    ] {
        let bc = fc + Vec2::new(dx * foff, dy * foff);
        p.circle_filled(bc, fr, if btn(bit) { GREEN } else { off_fill });
        p.circle_stroke(bc, fr, Stroke::new(1.2, wa(col)));
        p.text(bc, Align2::CENTER_CENTER, sym, theme::mono(10.0), if btn(bit) { Color32::BLACK } else { wa(col) });
    }
    // Share / Options / PS.
    for (bit, fx, lbl) in [(4u32, 0.35f32, "share"), (6, 0.65, "opts")] {
        let br = Rect::from_center_size(at(fx, 0.18), Vec2::new(32.0, 11.0));
        p.rect(br, 3.0, if btn(bit) { GREEN } else { off_fill }, Stroke::new(1.0, wa(HAIRLINE)));
        p.text(br.center(), Align2::CENTER_CENTER, lbl, theme::mono(7.5), if btn(bit) { Color32::BLACK } else { wa(DIM) });
    }
    {
        let bc = at(0.5, 0.42);
        p.circle_filled(bc, 7.0, if btn(5) { GREEN } else { off_fill });
        p.circle_stroke(bc, 7.0, Stroke::new(1.0, wa(HAIRLINE)));
        p.text(bc, Align2::CENTER_CENTER, "PS", theme::mono(7.0), if btn(5) { Color32::BLACK } else { wa(DIM) });
    }
    // Sticks: live crosshair boxes (the Python 2D crosshairs, on the pad);
    // the RATE stick is amber-tagged and shows the tare point.
    for (fx, vx, vy, tbit, is_rate) in [
        (0.36f32, m.joy_left.0, m.joy_left.1, 7u32, rate_stick_left),
        (0.64, m.joy_right.0, m.joy_right.1, 8, !rate_stick_left),
    ] {
        let sc = at(fx, 0.74);
        let sr = w * 0.083;
        p.circle_filled(sc, sr, off_fill);
        let ring = if is_rate { AMBER } else { HAIRLINE };
        p.circle_stroke(sc, sr, Stroke::new(if btn(tbit) { 2.2 } else { 1.2 }, if btn(tbit) { GREEN } else { wa(ring) }));
        let sclip = p.with_clip_rect(Rect::from_center_size(sc, Vec2::splat(sr * 2.0)));
        let cp = sc + Vec2::new((vx as f32).clamp(-1.0, 1.0) * sr * 0.72, -(vy as f32).clamp(-1.0, 1.0) * sr * 0.72);
        sclip.line_segment([Pos2::new(cp.x, sc.y - sr), Pos2::new(cp.x, sc.y + sr)], Stroke::new(1.0, wa(TEXT)));
        sclip.line_segment([Pos2::new(sc.x - sr, cp.y), Pos2::new(sc.x + sr, cp.y)], Stroke::new(1.0, wa(TEXT)));
        p.circle_filled(cp, 2.5, if is_rate { AMBER } else { wa(TEXT_2) });
        if is_rate {
            p.text(sc + Vec2::new(0.0, sr + 2.0), Align2::CENTER_TOP, "RATE", theme::mono(7.5), wa(AMBER));
            if m.joystick_tare != (0.0, 0.0) {
                let tp = sc + Vec2::new(m.joystick_tare.0 as f32 * sr * 0.72, m.joystick_tare.1 as f32 * sr * 0.72);
                p.circle_stroke(tp, 3.0, Stroke::new(1.2, wa(VIOLET)));
            }
        }
    }
    // Focus motor (render_joystick_status focus rows): centre-zero commanded
    // rate bar + live encoder read-back. L2 retracts, R2 extends.
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("focus (L2-/R2+)").font(theme::mono(10.0)).color(TEXT_2));
        let (fr, fp) = ui.allocate_painter(Vec2::new(100.0, 12.0), Sense::hover());
        fp.rect(fr.rect, 2.0, theme::with_alpha(theme::BG, alpha), Stroke::new(1.0, wa(HAIRLINE)));
        let cx = fr.rect.center().x;
        fp.line_segment([Pos2::new(cx, fr.rect.top()), Pos2::new(cx, fr.rect.bottom())], Stroke::new(1.0, wa(HAIRLINE)));
        if m.focus_rate != 0 {
            let fill = (m.focus_rate as f32 / 9.0).clamp(-1.0, 1.0) * (fr.rect.width() / 2.0);
            let col = if m.focus_rate > 0 { GREEN } else { AMBER };
            let x0 = if fill >= 0.0 { cx } else { cx + fill };
            fp.rect_filled(Rect::from_min_size(Pos2::new(x0, fr.rect.top() + 1.0), Vec2::new(fill.abs(), fr.rect.height() - 2.0)), 1.0, col);
        }
        ui.label(egui::RichText::new(format!("rate {:+}", m.focus_rate)).font(theme::mono(9.5)).color(if m.focus_rate > 0 { GREEN } else if m.focus_rate < 0 { AMBER } else { DIM }));
        ui.label(egui::RichText::new(format!("pos {}", m.focus_pos)).font(theme::mono(9.5)).color(if connected { GREEN } else { DIM }));
    });
    // Adaptive-rate gear ladder: green base gears, orange boost, grey outside RATE.
    let boosted = m.gear_ceiling > m.gear_base;
    ui.label(
        egui::RichText::new(format!("slew gear {}/{}{}", m.gear_ceiling, m.gear_max, if boosted && m.mode == "RATE" { "  BOOST" } else { "" }))
            .font(theme::mono(10.0))
            .color(if boosted && m.mode == "RATE" { AMBER } else { TEXT_2 }),
    );
    let (gr, gp) = ui.allocate_painter(Vec2::new(ui.available_width(), 15.0), Sense::hover());
    for sgear in 1..=m.gear_max {
        let seg = Rect::from_min_size(gr.rect.min + Vec2::new((sgear - 1) as f32 * 14.0, 1.0), Vec2::new(12.0, 13.0));
        let col = if m.mode == "RATE" && sgear <= m.gear_ceiling {
            if sgear <= m.gear_base { GREEN } else { AMBER }
        } else {
            theme::with_alpha(DIM, 60)
        };
        gp.rect(seg, 2.0, col, Stroke::new(1.0, HAIRLINE));
    }
    ui.label(egui::RichText::new("hold the stick pinned to wind up the boost gears").font(theme::sans(9.0)).color(DIM));
    // Function legend, live-highlighted while a control is held.
    egui::Grid::new("joy_legend").num_columns(2).spacing([10.0, 1.0]).show(ui, |ui| {
        let rows: [(&str, String, bool); 11] = [
            ("✕", "capture arm / save run".into(), btn(0)),
            ("○", "STOP → standby".into(), btn(1)),
            ("□", "tare rate stick".into(), btn(2)),
            ("△", "park".into(), btn(3)),
            ("L1", format!("feed-forward (az {} / el {})", if m.ff_azm { "on" } else { "off" }, if m.ff_alt { "on" } else { "off" }), btn(9)),
            ("L2/R2", "focus motor".into(), m.focus_rate != 0),
            ("L3/R3", "cycle tracking mode".into(), btn(7) || btn(8)),
            ("share", format!("bias mode ({}·{})", if m.bias_fine { "fine" } else { "coarse" }, if m.bias_frame == "alongcross" { "trk" } else { "azel" }), btn(4)),
            ("opts", format!("mount mode ({})", m.mount_mode), btn(6)),
            (
                "d-pad",
                if m.bias_frame == "alongcross" {
                    format!("bias in/cross {:+.2}°/{:+.2}°", m.bias_it, m.bias_ct)
                } else {
                    format!("bias az/el {:+.2}°/{:+.2}°", m.bias.0, m.bias.1)
                },
                btn(11) || btn(12) || btn(13) || btn(14),
            ),
            (if rate_stick_left { "L-stick" } else { "R-stick" }, "RATE slew (adaptive gearbox)".into(), false),
        ];
        for (k, v, active) in rows {
            ui.label(egui::RichText::new(k).font(theme::mono(9.5)).color(if active { GREEN } else { DIM }));
            ui.label(egui::RichText::new(v).font(theme::sans(9.5)).color(if active { GREEN } else { TEXT_2 }));
            ui.end_row();
        }
    });
}

pub fn sky_table(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, tx: &crossbeam_channel::Sender<MountCmd>) {
    use egui_extras::{Column, TableBuilder};
    let sky = shared.sky.load();
    let mask = shared.config.elevation_mask_deg;
    ui.horizontal(|ui| {
        theme::section(ui, &format!("visible now · {}", sky.n_visible));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.checkbox(&mut st.show_below_mask, "below mask");
        });
    });
    let mut rows: Vec<_> = sky.sats.iter().filter(|s| s.el > 0.0 && (st.show_below_mask || s.el >= mask)).collect();
    let (col, asc) = st.table_sort;
    rows.sort_by(|a, b| {
        let o = match col {
            0 => a.name.cmp(&b.name),
            1 => a.satnum.cmp(&b.satnum),
            2 => a.el.partial_cmp(&b.el).unwrap(),
            3 => a.az.partial_cmp(&b.az).unwrap(),
            4 => a.range_km.partial_cmp(&b.range_km).unwrap(),
            _ => a.az_rate.abs().partial_cmp(&b.az_rate.abs()).unwrap(),
        };
        if asc { o } else { o.reverse() }
    });
    let headers = ["Name", "NORAD", "El °", "Az °", "Range km", "Az rate °/s"];
    TableBuilder::new(ui)
        .striped(true)
        .sense(Sense::click())
        .column(Column::initial(190.0).at_least(120.0))
        .column(Column::initial(64.0))
        .column(Column::initial(60.0))
        .column(Column::initial(60.0))
        .column(Column::initial(80.0))
        .column(Column::remainder())
        .header(20.0, |mut h| {
            for (i, t) in headers.iter().enumerate() {
                h.col(|ui| {
                    let active = st.table_sort.0 == i;
                    let label = if active { format!("{t} {}", if st.table_sort.1 { "▲" } else { "▼" }) } else { t.to_string() };
                    if ui.add(egui::Label::new(egui::RichText::new(label).color(if active { TEXT } else { TEXT_2 }).font(theme::sans(11.5))).sense(Sense::click())).clicked() {
                        st.table_sort = if active { (i, !st.table_sort.1) } else { (i, i == 0 || i == 1) };
                    }
                });
            }
        })
        .body(|body| {
            body.rows(18.0, rows.len(), |mut row| {
                let s = rows[row.index()];
                let sel = st.selected.as_deref() == Some(s.satnum.as_str());
                row.set_selected(sel);
                let dim = s.el < mask;
                let c = if dim { DIM } else { TEXT };
                row.col(|ui| {
                    ui.label(egui::RichText::new(&s.name).color(if sel { ACCENT } else { c }));
                });
                row.col(|ui| { ui.label(egui::RichText::new(&s.satnum).font(theme::mono(11.0)).color(c)); });
                row.col(|ui| { ui.label(egui::RichText::new(format!("{:.1}", s.el)).font(theme::mono(11.0)).color(c)); });
                row.col(|ui| { ui.label(egui::RichText::new(format!("{:.1}", s.az)).font(theme::mono(11.0)).color(c)); });
                row.col(|ui| { ui.label(egui::RichText::new(format!("{:.0}", s.range_km)).font(theme::mono(11.0)).color(c)); });
                row.col(|ui| { ui.label(egui::RichText::new(format!("{:+.3}", s.az_rate)).font(theme::mono(11.0)).color(c)); });
                if row.response().clicked() {
                    st.selected = Some(s.satnum.clone());
                    let _ = tx.send(MountCmd::SelectTarget(Some(s.satnum.clone())));
                }
            });
        });
}
