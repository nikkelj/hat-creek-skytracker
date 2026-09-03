//! Visual language for the native app: one palette, one type scale, a few
//! reusable chrome helpers. Every screen draws from here so the app reads
//! as one instrument rather than a pile of widgets.
//!
//! Principles: near-black blue-grey ground, hairline structure, restrained
//! accent (selection / active only), amber reserved for the mount and
//! warnings, green for "tracking / good", red for errors and the mask.
//! Text carries hierarchy by weight and tone, not by colour.

use egui::{Color32, FontFamily, FontId, RichText, Rounding, Stroke, Vec2};

// ---- palette -------------------------------------------------------------
pub const BG: Color32 = Color32::from_rgb(9, 11, 15);        // sky / canvas ground
pub const PANEL: Color32 = Color32::from_rgb(15, 18, 24);    // panels
pub const RAISED: Color32 = Color32::from_rgb(22, 26, 34);   // cards, table header
pub const HAIRLINE: Color32 = Color32::from_rgb(34, 40, 52);
pub const TEXT: Color32 = Color32::from_rgb(226, 230, 236);
pub const TEXT_2: Color32 = Color32::from_rgb(150, 160, 175);
pub const DIM: Color32 = Color32::from_rgb(92, 102, 118);
pub const ACCENT: Color32 = Color32::from_rgb(92, 170, 220);  // selection / active (muted)
pub const AMBER: Color32 = Color32::from_rgb(240, 180, 60);   // mount, boresight, caution
pub const GREEN: Color32 = Color32::from_rgb(72, 200, 140);   // tracking / good
pub const RED: Color32 = Color32::from_rgb(224, 82, 99);      // error, mask
pub const VIOLET: Color32 = Color32::from_rgb(170, 120, 230); // GEO / distant

// Orbit-class colours for satellites (LEO warm-to-cool by range, MEO, GEO).
pub fn sat_color(range_km: f64) -> Color32 {
    if range_km > 20_000.0 {
        VIOLET
    } else if range_km > 3_000.0 {
        Color32::from_rgb(232, 150, 72)
    } else {
        // Low orbits: near (bright warm white) -> far (cool grey-blue).
        let t = (range_km / 2_500.0).clamp(0.0, 1.0) as f32;
        lerp(Color32::from_rgb(255, 236, 200), Color32::from_rgb(120, 150, 200), t)
    }
}

pub fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()), f(a.a(), b.a()))
}

pub fn with_alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

// ---- type scale ----------------------------------------------------------
pub fn mono(size: f32) -> FontId { FontId::new(size, FontFamily::Monospace) }
pub fn sans(size: f32) -> FontId { FontId::new(size, FontFamily::Proportional) }

/// Small-caps style section label ("MOUNT", "CAMERA 1").
pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(2.0);
    ui.label(RichText::new(title.to_uppercase()).font(sans(10.5)).color(DIM));
    ui.add_space(2.0);
}

/// Key/value line in the instrument panels: dim key, mono value.
pub fn kv(ui: &mut egui::Ui, key: &str, value: impl Into<String>) {
    kv_colored(ui, key, value, TEXT);
}

pub fn kv_colored(ui: &mut egui::Ui, key: &str, value: impl Into<String>, color: Color32) {
    ui.label(RichText::new(key).font(sans(12.0)).color(TEXT_2));
    ui.label(RichText::new(value.into()).font(mono(12.5)).color(color));
    ui.end_row();
}

/// Large numeric readout (az/el style): value in mono, unit dim.
pub fn readout(ui: &mut egui::Ui, label: &str, value: &str, unit: &str, color: Color32) {
    ui.vertical(|ui| {
        ui.label(RichText::new(label).font(sans(10.5)).color(DIM));
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.label(RichText::new(value).font(mono(20.0)).color(color));
            ui.label(RichText::new(unit).font(sans(11.0)).color(TEXT_2));
        });
    });
}

/// A pill status tag ("TRACKING", "STANDBY"): tinted fill + border.
pub fn tag(ui: &mut egui::Ui, text: &str, color: Color32) {
    let fill = with_alpha(color, 36);
    let r = egui::Frame::none()
        .fill(fill)
        .stroke(Stroke::new(1.0, with_alpha(color, 140)))
        .rounding(Rounding::same(3.0))
        .inner_margin(egui::Margin::symmetric(7.0, 2.0))
        .show(ui, |ui| {
            ui.label(RichText::new(text).font(mono(11.0)).color(color));
        });
    let _ = r;
}

/// Card container for instrument groups.
pub fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::none()
        .fill(RAISED)
        .stroke(Stroke::new(1.0, HAIRLINE))
        .rounding(Rounding::same(5.0))
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, add)
        .inner
}

/// Mode button row: highlighted when active, quiet otherwise.
pub fn mode_button(ui: &mut egui::Ui, label: &str, active: bool, color: Color32) -> bool {
    let (fill, stroke, text) = if active {
        (with_alpha(color, 48), Stroke::new(1.0, color), color)
    } else {
        (Color32::TRANSPARENT, Stroke::new(1.0, HAIRLINE), TEXT_2)
    };
    let b = egui::Button::new(RichText::new(label).font(mono(11.5)).color(text))
        .fill(fill)
        .stroke(stroke)
        .rounding(Rounding::same(3.0))
        .min_size(Vec2::new(0.0, 24.0));
    ui.add(b).clicked()
}

/// Install palette + spacing on the context, and upgrade the font stack
/// when the platform has nicer faces (Windows: Segoe UI + Cascadia/Consolas).
pub fn install(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = BG;
    v.faint_bg_color = RAISED;
    v.override_text_color = Some(TEXT);
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_2);
    v.widgets.inactive.bg_fill = RAISED;
    v.widgets.inactive.weak_bg_fill = RAISED;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_2);
    v.widgets.hovered.bg_fill = Color32::from_rgb(30, 36, 46);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(30, 36, 46);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, with_alpha(ACCENT, 120));
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.active.bg_fill = with_alpha(ACCENT, 60);
    v.widgets.active.weak_bg_fill = with_alpha(ACCENT, 60);
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.open.bg_fill = RAISED;
    v.selection.bg_fill = with_alpha(ACCENT, 70);
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.window_rounding = Rounding::same(6.0);
    v.menu_rounding = Rounding::same(4.0);
    v.widgets.inactive.rounding = Rounding::same(3.0);
    v.widgets.hovered.rounding = Rounding::same(3.0);
    v.widgets.active.rounding = Rounding::same(3.0);
    v.window_stroke = Stroke::new(1.0, HAIRLINE);
    v.striped = true;
    ctx.set_visuals(v);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 5.0);
    style.spacing.button_padding = Vec2::new(9.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(10.0);
    style.spacing.indent = 14.0;
    style.text_styles.insert(egui::TextStyle::Heading, sans(16.0));
    style.text_styles.insert(egui::TextStyle::Body, sans(12.5));
    style.text_styles.insert(egui::TextStyle::Button, sans(12.0));
    style.text_styles.insert(egui::TextStyle::Small, sans(10.5));
    style.text_styles.insert(egui::TextStyle::Monospace, mono(12.0));
    ctx.set_style(style);

    install_fonts(ctx);
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut changed = false;
    let try_font = |fonts: &mut egui::FontDefinitions, name: &str, candidates: &[&str], family: FontFamily| -> bool {
        for path in candidates {
            if let Ok(bytes) = std::fs::read(path) {
                fonts.font_data.insert(name.to_string(), egui::FontData::from_owned(bytes));
                fonts.families.entry(family.clone()).or_default().insert(0, name.to_string());
                return true;
            }
        }
        false
    };
    let win = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
    let f = |n: &str| format!("{win}\\Fonts\\{n}");
    changed |= try_font(&mut fonts, "ui-sans", &[&f("segoeui.ttf")], FontFamily::Proportional);
    changed |= try_font(
        &mut fonts,
        "ui-mono",
        &[&f("CascadiaMono.ttf"), &f("CascadiaCode.ttf"), &f("consola.ttf")],
        FontFamily::Monospace,
    );
    if changed {
        ctx.set_fonts(fonts);
    }
}
