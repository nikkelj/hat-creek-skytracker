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
    pub keepout_tex: Option<(egui::TextureHandle, String)>,
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
            keepout_tex: None,
        }
    }
}

/// Polar projection: az clockwise from north (up), el 90 at centre.
pub fn polar(center: Pos2, radius: f32, az: f64, el: f64) -> Pos2 {
    let r = radius * ((90.0 - el.clamp(-10.0, 90.0)) / 90.0) as f32;
    let a = az.to_radians() as f32;
    Pos2::new(center.x + r * a.sin(), center.y - r * a.cos())
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
    let (resp, painter) = ui.allocate_painter(avail, Sense::click());
    let rect = resp.rect;
    painter.rect_filled(rect, 0.0, BG);
    let radius = (rect.width().min(rect.height()) / 2.0 - 30.0).max(60.0);
    let center = rect.center();
    let mask = cfg.elevation_mask_deg;

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
                let want_label = st.show_labels && (selected || hovered || (above && el > 35.0 && s.range_km < 20_000.0 && n_labels < 36));
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
            if cam.fisheye || !cam.connected {
                continue;
            }
            let fov_r = radius * (cam.fov_deg / 90.0) as f32 / 2.0;
            let alpha = if i == shared.hotspot_slot() { 140 } else { 70 };
            painter.circle_stroke(mp, fov_r.max(3.0), Stroke::new(1.0, theme::with_alpha(AMBER, alpha)));
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
    } else if resp.clicked() && resp.hover_pos().map_or(false, |p| p.distance(center) > radius) {
        st.selected = None;
        let _ = tx.send(MountCmd::SelectTarget(None));
    }

    // Corner readouts.
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
    // Fit the frame into the pane keeping aspect.
    let scale = (r.rect.width() / cam.width as f32).min(r.rect.height() / cam.height as f32);
    let img_size = Vec2::new(cam.width as f32 * scale, cam.height as f32 * scale);
    let img_rect = Rect::from_center_size(r.rect.center(), img_size);
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
        for mode in ["STANDBY", "RATE", "PROGRAM", "HANDOFF", "HOTSPOT"] {
            if theme::mode_button(ui, mode, m.mode == mode, mode_color(mode)) {
                let _ = tx.send(MountCmd::SetMode(mode.to_string()));
            }
        }
        if theme::mode_button(ui, "STOP", false, RED) {
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
        if ui.small_button("gains…").clicked() {
            st.gains_edit = Some(if st.gains_edit.is_some() { st.gains_edit.unwrap() } else { m.gains });
            if st.gains_edit.is_some() && ui.input(|i| i.modifiers.shift) {
                st.gains_edit = None;
            }
        }
    });
    if let Some(t) = &m.autotune {
        ui.label(egui::RichText::new(t).font(theme::mono(10.5)).color(ACCENT));
    }
    if let Some(mut g) = st.gains_edit {
        let mut close = false;
        theme::card(ui, |ui| {
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
            ui.horizontal(|ui| {
                if ui.button("apply").clicked() {
                    let _ = tx.send(MountCmd::SetGains { azm: (g[0][0], g[0][1], g[0][2]), alt: (g[1][0], g[1][1], g[1][2]) });
                }
                if ui.button("close").clicked() {
                    close = true;
                }
            });
        });
        st.gains_edit = if close { None } else { Some(g) };
    }
    ui.add_space(4.0);
    theme::section(ui, "log");
    for s in m.status.iter().rev().take(6) {
        ui.label(egui::RichText::new(s).font(theme::mono(10.5)).color(TEXT_2));
    }
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
