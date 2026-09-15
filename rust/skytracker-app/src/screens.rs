//! Secondary screens: Passes, Align, Sim, Config. Each renders snapshots and
//! sends commands; none touches worker state directly.

use crate::state::{AlignCmd, CamCmd, CamSettings, FwCmd, MountCmd, Shared, SimSettings};
use crate::theme::{self, ACCENT, AMBER, DIM, GREEN, RED, TEXT, TEXT_2};
use crate::ui::UiState;
use egui::{RichText, Sense};
use std::sync::Arc;

pub struct PassesState {
    pub sort: (usize, bool),
    pub min_el: f64,
    pub leo_only: bool,
    pub hide_geo: bool,
    pub filter: String,
}

impl Default for PassesState {
    fn default() -> Self {
        PassesState { sort: (2, true), min_el: 20.0, leo_only: true, hide_geo: true, filter: String::new() }
    }
}

pub fn passes_screen(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, ps: &mut PassesState, tx: &crossbeam_channel::Sender<MountCmd>) {
    use egui_extras::{Column, TableBuilder};
    let passes = shared.passes.load();
    let now = crate::sky::now_unix();
    ui.horizontal(|ui| {
        theme::section(ui, &format!("upcoming passes · next {:.0} h", passes.horizon_h));
        ui.add_space(12.0);
        ui.label(RichText::new("min peak el").color(TEXT_2));
        ui.add(egui::Slider::new(&mut ps.min_el, 10.0..=80.0).suffix("°").show_value(true));
        ui.checkbox(&mut ps.leo_only, "LEO only");
        ui.checkbox(&mut ps.hide_geo, "hide GEO");
        ui.add(egui::TextEdit::singleline(&mut ps.filter).hint_text("filter name / NORAD").desired_width(160.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("refresh TLEs").on_hover_text("download the Celestrak active catalog into tle_cache.tle and reload").clicked() {
                shared.tle_refresh.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            ui.label(RichText::new(shared.tle_status.load().as_str()).font(theme::mono(10.5)).color(TEXT_2));
            let age = if passes.computed_unix > 0.0 { now - passes.computed_unix } else { 0.0 };
            ui.label(
                RichText::new(format!("{} passes · computed {:.0} s ago in {:.0} ms", passes.rows.len(), age, passes.compute_ms))
                    .font(theme::mono(10.5))
                    .color(DIM),
            );
        });
    });
    let filt = ps.filter.to_ascii_lowercase();
    let mut rows: Vec<_> = passes
        .rows
        .iter()
        .filter(|p| p.tca_el >= ps.min_el && p.los_unix > now)
        .filter(|p| !ps.hide_geo || p.apogee_km < 30_000.0)
        .filter(|p| !ps.leo_only || p.apogee_km < 3_000.0)
        .filter(|p| filt.is_empty() || p.name.to_ascii_lowercase().contains(&filt) || p.satnum.contains(&filt))
        .collect();
    let (col, asc) = ps.sort;
    rows.sort_by(|a, b| {
        let o = match col {
            0 => a.name.cmp(&b.name),
            1 => a.satnum.cmp(&b.satnum),
            2 => a.aos_unix.partial_cmp(&b.aos_unix).unwrap(),
            3 => a.tca_el.partial_cmp(&b.tca_el).unwrap(),
            4 => a.duration_s.partial_cmp(&b.duration_s).unwrap(),
            5 => a.max_rate_dps.partial_cmp(&b.max_rate_dps).unwrap(),
            6 => a.range_tca_km.partial_cmp(&b.range_tca_km).unwrap(),
            7 => a.apogee_km.partial_cmp(&b.apogee_km).unwrap(),
            _ => a.est_mag.unwrap_or(99.0).partial_cmp(&b.est_mag.unwrap_or(99.0)).unwrap(),
        };
        if asc { o } else { o.reverse() }
    });
    let headers = ["Name", "NORAD", "AOS (UTC)", "Peak el", "Duration", "Peak rate", "Range @ TCA", "Apogee", "Mag"];
    let widths = [210.0, 64.0, 210.0, 110.0, 70.0, 80.0, 100.0, 90.0, 50.0];
    let mut tb = TableBuilder::new(ui).striped(true).sense(Sense::click());
    for w in widths {
        tb = tb.column(Column::initial(w).at_least(40.0));
    }
    tb = tb.column(Column::remainder());
    tb.header(22.0, |mut h| {
        for (i, t) in headers.iter().enumerate() {
            h.col(|ui| {
                let active = ps.sort.0 == i;
                let label = if active { format!("{t} {}", if ps.sort.1 { "▲" } else { "▼" }) } else { t.to_string() };
                if ui.add(egui::Label::new(RichText::new(label).color(if active { TEXT } else { TEXT_2 })).sense(Sense::click())).clicked() {
                    ps.sort = if active { (i, !ps.sort.1) } else { (i, matches!(i, 0 | 1 | 2 | 8)) };
                }
            });
        }
        h.col(|ui| {
            ui.label(RichText::new("Track").color(TEXT_2));
        });
    })
    .body(|body| {
        body.rows(20.0, rows.len(), |mut row| {
            let p = rows[row.index()];
            let sel = st.selected.as_deref() == Some(p.satnum.as_str());
            row.set_selected(sel);
            let live = p.aos_unix <= now && now <= p.los_unix;
            let c = if live { GREEN } else { TEXT };
            let mono = |s: String| RichText::new(s).font(theme::mono(11.0)).color(c);
            row.col(|ui| { ui.label(RichText::new(&p.name).color(if sel { ACCENT } else { c })); });
            row.col(|ui| { ui.label(mono(p.satnum.clone())); });
            row.col(|ui| {
                let s = if live {
                    format!("now · LOS {}", crate::sky::hms_from_unix(p.los_unix))
                } else {
                    let dt = p.aos_unix - now;
                    format!("{}  (+{:02}:{:02})", crate::sky::hms_from_unix(p.aos_unix), (dt / 3600.0) as i64, ((dt % 3600.0) / 60.0) as i64)
                };
                ui.label(mono(s));
            });
            row.col(|ui| { ui.label(mono(format!("{:.1}° @ {:.0}°", p.tca_el, p.tca_az))); });
            row.col(|ui| { ui.label(mono(format!("{:.0}:{:02}", (p.duration_s / 60.0).floor(), (p.duration_s % 60.0) as i64))); });
            row.col(|ui| { ui.label(mono(format!("{:.2}°/s", p.max_rate_dps))); });
            row.col(|ui| { ui.label(mono(format!("{:.0} km", p.range_tca_km))); });
            row.col(|ui| { ui.label(mono(format!("{:.0} km", p.apogee_km))); });
            row.col(|ui| {
                match p.est_mag {
                    Some(m) => ui.label(mono(format!("{m:.1}"))),
                    None => ui.label(RichText::new("ecl").font(theme::mono(11.0)).color(DIM)),
                };
            });
            row.col(|ui| {
                ui.label(RichText::new(format!("{:.0}° → {:.0}°", p.aos_az, p.los_az)).font(theme::mono(10.5)).color(DIM));
            });
            if row.response().clicked() {
                st.selected = Some(p.satnum.clone());
                let _ = tx.send(MountCmd::SelectTarget(Some(p.satnum.clone())));
            }
        });
    });
}

pub struct AlignState {
    pub n_points: usize,
    pub supervised: bool,
}

impl Default for AlignState {
    fn default() -> Self {
        AlignState { n_points: 12, supervised: false }
    }
}

pub fn align_screen(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, als: &mut AlignState, tx_align: &crossbeam_channel::Sender<AlignCmd>, tx_mount: &crossbeam_channel::Sender<MountCmd>, tx_cam: &crossbeam_channel::Sender<crate::state::CamCmd>) {
    let solve = shared.solve.load();
    let align = shared.align.load();
    let mount = shared.mount.load();
    ui.columns(2, |cols| {
        // Left: camera with solve overlay + solve controls.
        let ui = &mut cols[0];
        crate::ui::camera_panel(ui, shared, st, tx_cam, true);
        ui.add_space(6.0);
        // Prerequisite checklist (alignment_ui readiness panel).
        ui.horizontal_wrapped(|ui| {
            let cam_ok = shared.cam(shared.solve_slot()).map_or(false, |c| c.connected && c.fps > 1.0);
            for (ok, label) in [
                (solve.db_loaded, "tetra3 db"),
                (mount.connected, "mount"),
                (cam_ok, "camera streaming"),
            ] {
                theme::tag(ui, &format!("{} {label}", if ok { "✓" } else { "✗" }), if ok { GREEN } else { RED });
            }
            ui.label(RichText::new(&solve.db_name).font(theme::mono(9.5)).color(DIM));
        });
        theme::section(ui, "plate solve");
        ui.horizontal(|ui| {
            if ui.add_enabled(solve.db_loaded && !solve.busy, egui::Button::new("solve now")).clicked() {
                let _ = tx_align.send(AlignCmd::SolveNow);
            }
            let mut cont = align.continuous;
            if ui.checkbox(&mut cont, "continuous").on_hover_text("background plate solve ~0.5 Hz (plate_solve_enabled)").changed() {
                let _ = tx_align.send(AlignCmd::Continuous(cont));
            }
            if ui.add_enabled(solve.last_ok, egui::Button::new("apply align")).on_hover_text("one-star alignment: fold this solve's azimuth error into alignment_azimuth (persisted)").clicked() {
                let _ = tx_align.send(AlignCmd::ApplyAlign);
            }
            ui.label(RichText::new(&solve.message).font(theme::mono(10.5)).color(if solve.last_ok { GREEN } else if solve.db_loaded { TEXT_2 } else { RED }));
        });
        if solve.frame_seq != 0 && solve.last_ok {
            egui::Grid::new("solve_grid").num_columns(2).spacing([14.0, 3.0]).show(ui, |ui| {
                theme::kv(ui, "RA / Dec", format!("{:.4}° / {:+.4}°", solve.ra_deg, solve.dec_deg));
                theme::kv(ui, "roll · FOV", format!("{:.2}° · {:.3}°", solve.roll_deg, solve.fov_deg));
                theme::kv(ui, "matches", format!("{} of {} centroids · rmse {:.1}″ · {:.0} ms", solve.matches, solve.n_centroids, solve.rmse_arcsec, solve.solve_ms));
                theme::kv_colored(ui, "true az / el", format!("{:.4}° / {:.4}°", solve.true_az, solve.true_el), GREEN);
                theme::kv_colored(ui, "mount az / el", format!("{:.4}° / {:.4}°", solve.mount_az, solve.mount_el), AMBER);
                let daz = ((solve.true_az - solve.mount_az + 540.0).rem_euclid(360.0) - 180.0) * 3600.0 * solve.mount_el.to_radians().cos();
                let del = (solve.true_el - solve.mount_el) * 3600.0;
                theme::kv_colored(ui, "pointing error", format!("{daz:+.0}″ cross · {del:+.0}″ el"), if daz.abs() + del.abs() < 120.0 { GREEN } else { AMBER });
            });
        }
        ui.add_space(6.0);
        theme::section(ui, "paddles (0.05°)");
        ui.horizontal(|ui| {
            for (l, daz, del) in [("← az", -0.05, 0.0), ("az →", 0.05, 0.0), ("el ↑", 0.0, 0.05), ("el ↓", 0.0, -0.05)] {
                if ui.button(l).clicked() {
                    let _ = tx_mount.send(MountCmd::Nudge { daz, del });
                }
            }
            ui.label(RichText::new(format!("{:8.3}° / {:7.3}°", mount.az, mount.el)).font(theme::mono(11.5)).color(AMBER));
        });

        // Right: alignment run.
        let ui = &mut cols[1];
        theme::section(ui, "pointing-model alignment");
        {
            let (on, t) = **shared.pointing.load();
            ui.horizontal(|ui| {
                if theme::mode_button(ui, if on { "MODEL ON" } else { "MODEL OFF" }, on, GREEN) {
                    let _ = tx_mount.send(MountCmd::PointingModel(!on));
                }
                ui.label(RichText::new(format!("{}  (applied to PROGRAM setpoints: command = desired − error)", skytracker_pointing::altaz::TERM_NAMES.iter().zip(t.iter()).map(|(n, v)| format!("{n} {:+.1}′", v * 60.0)).collect::<Vec<_>>().join(" "))).font(theme::mono(10.0)).color(TEXT_2));
            });
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("points").color(TEXT_2));
            ui.add(egui::DragValue::new(&mut als.n_points).range(4..=60));
            ui.checkbox(&mut als.supervised, "confirm each point");
            if !align.running {
                if ui.add_enabled(solve.db_loaded, egui::Button::new("start")).clicked() {
                    let _ = tx_align.send(AlignCmd::Start { n_points: als.n_points, supervised: als.supervised });
                }
                if ui.add_enabled(solve.db_loaded, egui::Button::new("quick refit")).on_hover_text("6 points, IA/IE free, other terms frozen at the live model").clicked() {
                    let _ = tx_align.send(AlignCmd::QuickRefit);
                }
                if align.n_failed > 0 {
                    if ui.button(format!("retry failed ({})", align.n_failed)).on_hover_text("re-run only the unsolved points, append + refit").clicked() {
                        let _ = tx_align.send(AlignCmd::RetryFailed);
                    }
                }
            } else if theme::mode_button(ui, "abort", true, RED) {
                let _ = tx_align.send(AlignCmd::Abort);
            }
        });
        if align.manual {
            theme::card(ui, |ui| {
                ui.horizontal(|ui| {
                    theme::tag(ui, "MANUAL", AMBER);
                    ui.label(RichText::new("point failed to solve — paddle the mount onto stars, or").font(theme::sans(10.5)).color(TEXT));
                    if theme::mode_button(ui, "skip point", true, AMBER) {
                        let _ = tx_align.send(AlignCmd::Skip);
                    }
                    if theme::mode_button(ui, "abort run", false, RED) {
                        let _ = tx_align.send(AlignCmd::Abort);
                    }
                });
            });
        }
        // Polar alignment (Eq): RA-axis sweep + plate solves + axis fit.
        ui.horizontal(|ui| {
            theme::section(ui, "polar align");
            if align.polar_running {
                theme::tag(ui, "SWEEPING", AMBER);
                if theme::mode_button(ui, "abort", false, RED) {
                    let _ = tx_align.send(AlignCmd::Abort);
                }
            } else if ui.add_enabled(solve.db_loaded, egui::Button::new("run sweep")).on_hover_text("slew the RA axis ±30° in 5 steps, solve each, fit the axis direction and report base-rotation / tilt knob corrections").clicked() {
                let _ = tx_align.send(AlignCmd::PolarStart { n_points: 5, sweep_deg: 60.0 });
            }
            if let Some((ax_az, ax_el, daz, del, n)) = align.polar {
                ui.label(RichText::new(format!("axis {ax_az:.3}°/{ax_el:.3}° · base {daz:+.3}° · tilt {del:+.3}° ({n} pts)")).font(theme::mono(10.0)).color(GREEN));
            }
        });
        ui.horizontal(|ui| {
            theme::tag(ui, if align.running { "RUNNING" } else { "IDLE" }, if align.running { GREEN } else { DIM });
            ui.label(RichText::new(&align.status).font(theme::mono(11.0)).color(TEXT));
            if align.n_points > 0 {
                ui.label(RichText::new(format!("{}/{}", align.point, align.n_points)).font(theme::mono(11.0)).color(TEXT_2));
            }
        });
        if align.status.contains("accept?") {
            ui.horizontal(|ui| {
                if theme::mode_button(ui, "accept point", true, GREEN) {
                    let _ = tx_align.send(AlignCmd::Accept);
                }
                if theme::mode_button(ui, "reject", false, RED) {
                    let _ = tx_align.send(AlignCmd::Reject);
                }
            });
        }
        // Mini skyplot of targets + samples.
        let size = egui::Vec2::new(ui.available_width().min(320.0), 220.0);
        let (r, p) = ui.allocate_painter(size, Sense::hover());
        let c = r.rect.center();
        let rad = r.rect.height() / 2.0 - 8.0;
        p.rect_filled(r.rect, 4.0, theme::BG);
        for el in [0.0, 30.0, 60.0] {
            p.circle_stroke(c, rad * ((90.0 - el) / 90.0) as f32, egui::Stroke::new(1.0, theme::HAIRLINE));
        }
        for (i, (az, el)) in align.targets.iter().enumerate() {
            let q = crate::ui::polar(c, rad, *az, *el);
            let done = i < align.point;
            p.circle_stroke(q, 4.0, egui::Stroke::new(1.0, if done { GREEN } else { DIM }));
            if i == align.point && align.running {
                p.circle_stroke(q, 7.0, egui::Stroke::new(1.2, AMBER));
            }
        }
        for s in &align.samples {
            let q = crate::ui::polar(c, rad, s.mount_az, s.mount_el);
            let t = crate::ui::polar(c, rad, s.true_az, s.true_el);
            p.circle_filled(q, 2.5, GREEN);
            p.line_segment([q, t], egui::Stroke::new(1.0, theme::with_alpha(ACCENT, 180)));
        }
        for (az, el) in &align.failed {
            let q = crate::ui::polar(c, rad, *az, *el);
            p.circle_stroke(q, 4.0, egui::Stroke::new(1.2, RED));
        }
        let mp = crate::ui::polar(c, rad, mount.az, mount.el);
        p.circle_stroke(mp, 5.0, egui::Stroke::new(1.2, AMBER));
        if let (Some(t), Some(rms)) = (align.terms, align.rms_arcsec) {
            theme::card(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("fit rms {rms:.1}″")).color(GREEN).font(theme::mono(12.5)));
                    if theme::mode_button(ui, "APPLY MODEL", shared.pointing.load().0, GREEN) {
                        let _ = tx_align.send(AlignCmd::ApplyModel);
                    }
                });
                let names = ["IA", "IE", "NPAE", "CA", "AN", "AW", "TF"];
                egui::Grid::new("terms").num_columns(7).spacing([10.0, 2.0]).show(ui, |ui| {
                    for n in names {
                        ui.label(RichText::new(n).color(DIM).font(theme::sans(10.5)));
                    }
                    ui.end_row();
                    for v in t {
                        ui.label(RichText::new(format!("{:+.1}′", v * 60.0)).font(theme::mono(11.0)));
                    }
                    ui.end_row();
                });
            });
        }
        theme::section(ui, "samples");
        egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
            for (i, s) in align.samples.iter().enumerate() {
                ui.label(
                    RichText::new(format!("{:2}  mount {:8.3} {:7.3}   true {:8.3} {:7.3}   Δ {:6.0}″", i + 1, s.mount_az, s.mount_el, s.true_az, s.true_el, s.residual_arcsec))
                        .font(theme::mono(10.5))
                        .color(TEXT_2),
                );
            }
        });
        theme::section(ui, "log");
        egui::ScrollArea::vertical().max_height(160.0).id_salt("align_log").stick_to_bottom(true).show(ui, |ui| {
            for l in &align.log {
                ui.label(RichText::new(l).font(theme::mono(10.5)).color(DIM));
            }
        });
    });
}

pub fn sim_screen(ui: &mut egui::Ui, shared: &Arc<Shared>) {
    let cur = shared.sim.load();
    let mut s: SimSettings = (**cur).clone();
    let mut changed = false;
    // Master sim switch (hw_sim "Simulation: ON/OFF"): transport + camera
    // source are worker-spawn-time choices, so this persists and asks for a
    // restart rather than pretending to hot-swap hardware.
    ui.horizontal(|ui| {
        let sim_on = shared.config.mount_transport.eq_ignore_ascii_case("sim") && shared.config.camera_source.eq_ignore_ascii_case("sim");
        if theme::mode_button(ui, if sim_on { "SIMULATION ON" } else { "SIMULATION OFF" }, sim_on, if sim_on { GREEN } else { DIM }) {
            let (mt, cs) = if sim_on { ("serial", "asi") } else { ("sim", "sim") };
            crate::mount::persist_config_key(&shared.config.path, "mount_transport", serde_json::json!(mt));
            crate::mount::persist_config_key(&shared.config.path, "camera_source", serde_json::json!(cs));
        }
        ui.label(RichText::new("switching persists mount_transport + camera_source — restart the app to apply").font(theme::sans(10.0)).color(DIM));
        let m = shared.mount.load();
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(match &m.target {
                    Some(t) => format!("target {t} · az {:.2}° el {:.2}° · {}", m.az, m.el, m.mode),
                    None => format!("no target · az {:.2}° el {:.2}° · {}", m.az, m.el, m.mode),
                })
                .font(theme::mono(10.5))
                .color(TEXT_2),
            );
        });
    });
    ui.columns(2, |cols| {
        let ui = &mut cols[0];
        theme::section(ui, "simulated mount");
        ui.label(RichText::new("Injected errors between what the mount reports and where the camera really looks. PROGRAM alone leaves these as residuals; HANDOFF/HOTSPOT nulls them optically.").color(TEXT_2).font(theme::sans(11.5)));
        egui::Grid::new("sim_mount").num_columns(2).spacing([14.0, 6.0]).show(ui, |ui| {
            ui.label(RichText::new("misalignment az").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.misalign_az_deg, -0.5..=0.5).suffix("°").fixed_decimals(3)).changed();
            ui.end_row();
            ui.label(RichText::new("misalignment el").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.misalign_el_deg, -0.5..=0.5).suffix("°").fixed_decimals(3)).changed();
            ui.end_row();
            ui.label(RichText::new("encoder noise (1σ)").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.encoder_noise_deg, 0.0..=0.01).suffix("°").fixed_decimals(4)).changed();
            ui.end_row();
            ui.label(RichText::new("rate noise (1σ)").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.rate_noise_dps, 0.0..=0.05).suffix("°/s").fixed_decimals(4)).changed();
            ui.end_row();
            ui.label(RichText::new("backlash").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.backlash_deg, 0.0..=0.05).suffix("°").fixed_decimals(4)).changed();
            ui.end_row();
            ui.label(RichText::new("periodic error amp").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.pe_amplitude_deg, 0.0..=0.05).suffix("°").fixed_decimals(4)).changed();
            ui.end_row();
            ui.label(RichText::new("periodic error period").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.pe_period_s, 10.0..=1200.0).suffix(" s")).changed();
            ui.end_row();
        });
        ui.add_space(8.0);
        theme::section(ui, "simulated camera");
        egui::Grid::new("sim_cam").num_columns(2).spacing([14.0, 6.0]).show(ui, |ui| {
            ui.label(RichText::new("background").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.background_level, 0.0..=60.0)).changed();
            ui.end_row();
            ui.label(RichText::new("read noise").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.read_noise, 0.0..=20.0)).changed();
            ui.end_row();
            ui.label(RichText::new("target brightness").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.target_brightness, 10.0..=255.0)).changed();
            ui.end_row();
            ui.label(RichText::new("star limit mag").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.star_limit_mag, 5.0..=11.0).fixed_decimals(1)).changed();
            ui.end_row();
            ui.label(RichText::new("Tycho deep catalogue").color(TEXT_2));
            changed |= ui.checkbox(&mut s.use_deep_catalog, "").changed();
            ui.end_row();
            ui.label(RichText::new("rng seed").color(TEXT_2));
            let mut seed = s.seed as i64;
            if ui.add(egui::DragValue::new(&mut seed).speed(1)).changed() {
                s.seed = seed.max(0) as u64;
                changed = true;
            }
            ui.end_row();
            ui.label(RichText::new("cam2 offset x / y px").color(TEXT_2));
            ui.horizontal(|ui| {
                changed |= ui.add(egui::DragValue::new(&mut s.cam2_dx_px).speed(0.5)).changed();
                changed |= ui.add(egui::DragValue::new(&mut s.cam2_dy_px).speed(0.5)).changed();
                changed |= ui.add(egui::DragValue::new(&mut s.cam2_rot_deg).speed(0.1).suffix("° rot")).changed();
            });
            ui.end_row();
            ui.label(RichText::new("serial latency").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.serial_latency_s, 0.0..=0.25).suffix(" s").fixed_decimals(3)).changed();
            ui.end_row();
            ui.label(RichText::new("serial garbage prob").color(TEXT_2));
            changed |= ui.add(egui::Slider::new(&mut s.serial_garbage_prob, 0.0..=0.2).fixed_decimals(3)).changed();
            ui.end_row();
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("reset to config.json").clicked() {
                s = shared.config.sim.clone();
                changed = true;
            }
            if ui.button("save to config.json").on_hover_text("persist these sim settings (sim_config)").clicked() {
                let mut c = shared.config.clone();
                c.sim = s.clone();
                match c.save() {
                    Ok(()) => {}
                    Err(e) => eprintln!("sim save failed: {e}"),
                }
            }
        });
        let ui = &mut cols[1];
        theme::section(ui, "what the sim does");
        for line in [
            "• Mount: byte-level NexStar responder with wall-clock physics (skytracker-core::sim), driven by the same core loop as hardware.",
            "• Camera: Tycho-2 stars (mag ≤ limit) + live satellites projected through camera 1's pinhole at the true boresight; Gaussian PSFs, read noise; real capture pump/ring at 100 FPS.",
            "• The projection is the exact inverse of the hotspot pixel→angle mapping, so HOTSPOT closes the loop with the configured signs (x +1, y −1) by construction. Hardware signs still need rig calibration.",
            "• Encoder read noise, rate noise and reversal backlash are injected into the byte-level responder live (skytracker-core::sim::SimNoise).",
        ] {
            ui.label(RichText::new(line).color(TEXT_2).font(theme::sans(11.5)));
        }
    });
    if changed {
        shared.sim.store(Arc::new(s));
    }
}

pub struct ConfigState {
    pub draft: Option<crate::state::Config>,
    pub message: String,
}

impl Default for ConfigState {
    fn default() -> Self {
        ConfigState { draft: None, message: String::new() }
    }
}

pub fn config_screen(ui: &mut egui::Ui, shared: &Arc<Shared>, cs: &mut ConfigState, tx: &crossbeam_channel::Sender<MountCmd>) {
    if cs.draft.is_none() {
        cs.draft = Some(shared.config.clone());
    }
    let c = cs.draft.as_mut().unwrap();
    ui.horizontal(|ui| {
        theme::section(ui, &format!("config · {}", c.path.display()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if theme::mode_button(ui, "save", true, GREEN) {
                cs.message = match c.save() {
                    Ok(()) => {
                        let _ = tx.send(MountCmd::SetGains { azm: c.azm_gains, alt: c.alt_gains });
                        let _ = tx.send(MountCmd::SetHotspotSigns { x: c.hotspot_x_sign, y: c.hotspot_y_sign });
                        "saved · gains + hotspot signs applied live; other fields take effect on restart".into()
                    }
                    Err(e) => format!("save failed: {e}"),
                };
            }
            if ui.button("revert").clicked() {
                *c = shared.config.clone();
                cs.message = "reverted".into();
            }
            ui.label(RichText::new(&cs.message).font(theme::mono(10.5)).color(TEXT_2));
        });
    });
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Hat Creek Skytracker").font(theme::sans(12.0)).color(TEXT));
            ui.label(RichText::new("· Jonathan Nikkel — @NikkelJonathan · native Rust app (rust-port)").font(theme::sans(11.0)).color(DIM));
        });
        ui.columns(3, |cols| {
            let ui = &mut cols[0];
            theme::section(ui, "site");
            egui::Grid::new("cfg_site").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                num(ui, "latitude °", &mut c.lat_deg, 1e-6, 7);
                num(ui, "longitude °", &mut c.lon_deg, 1e-6, 7);
                num(ui, "altitude m", &mut c.alt_m, 1.0, 1);
                num(ui, "elevation mask °", &mut c.elevation_mask_deg, 0.5, 1);
            });
            theme::section(ui, "mount");
            egui::Grid::new("cfg_mount").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                combo(ui, "mode", &mut c.mount_mode, &["AltAz", "AltAz-Side", "Passthrough", "Eq"]);
                boolean(ui, "pointing model", &mut c.pointing_model_enabled);
                combo(ui, "rate stick", &mut c.joystick_rate_stick, &["right", "left"]);
                let mut bc = c.joy_rate_base_ceiling as f64;
                num(ui, "gearbox base step", &mut bc, 1.0, 0);
                c.joy_rate_base_ceiling = bc.clamp(1.0, 9.0) as i32;
                num(ui, "gearbox wind-up s", &mut c.joy_rate_windup_delay_s, 0.1, 1);
                combo(ui, "transport", &mut c.mount_transport, &["sim", "serial"]);
                text(ui, "serial port", &mut c.serial_port);
                let mut baud = c.serial_baud as f64;
                num(ui, "baud", &mut baud, 1.0, 0);
                c.serial_baud = baud as u32;
                num(ui, "az offset °", &mut c.offsets.0, 0.01, 3);
                num(ui, "alt offset °", &mut c.offsets.1, 0.01, 3);
                num(ui, "alignment az °", &mut c.alignment_az, 0.01, 3);
                num(ui, "alignment el °", &mut c.alignment_el, 0.01, 3);
                boolean(ui, "alt-az side flip", &mut c.altaz_side_flip);
                num(ui, "az limit min", &mut c.azm_limit.0, 1.0, 1);
                num(ui, "az limit max", &mut c.azm_limit.1, 1.0, 1);
                num(ui, "el limit min", &mut c.alt_limit.0, 1.0, 1);
                num(ui, "el limit max", &mut c.alt_limit.1, 1.0, 1);
                num(ui, "loop Hz", &mut c.loop_hz, 1.0, 0);
            });
            let ui = &mut cols[1];
            theme::section(ui, "control");
            egui::Grid::new("cfg_ctrl").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                num(ui, "az P", &mut c.azm_gains.0, 1e-5, 6);
                num(ui, "az I", &mut c.azm_gains.1, 1e-5, 6);
                num(ui, "az D", &mut c.azm_gains.2, 1e-5, 6);
                num(ui, "el P", &mut c.alt_gains.0, 1e-5, 6);
                num(ui, "el I", &mut c.alt_gains.1, 1e-5, 6);
                num(ui, "el D", &mut c.alt_gains.2, 1e-5, 6);
                boolean(ui, "feed-forward az", &mut c.ff_azm);
                boolean(ui, "feed-forward el", &mut c.ff_alt);
                num(ui, "lead time s", &mut c.lead_time_s, 0.01, 2);
                boolean(ui, "continuous rate", &mut c.continuous_rate);
                num(ui, "guide rate max °/s", &mut c.guide_rate_max_dps, 0.1, 2);
                num(ui, "output filter τ s", &mut c.output_filter_tau, 0.01, 2);
            });
            theme::section(ui, "hotspot / handoff");
            egui::Grid::new("cfg_hs").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                num(ui, "SNR threshold", &mut c.hotspot_snr, 0.1, 1);
                num(ui, "gate radius px", &mut c.hotspot_gate_radius, 1.0, 0);
                num(ui, "coast time s", &mut c.hotspot_coast_s, 0.1, 1);
                num(ui, "max rate °/s", &mut c.hotspot_max_rate_dps, 0.1, 2);
                num(ui, "x sign", &mut c.hotspot_x_sign, 1.0, 0);
                num(ui, "y sign", &mut c.hotspot_y_sign, 1.0, 0);
                boolean(ui, "star filter", &mut c.hotspot_star_filter);
                num(ui, "rate gate °/s", &mut c.hotspot_rate_gate_dps, 0.01, 2);
                let mut hc = c.hotspot_camera_index as f64;
                num(ui, "hotspot camera (0-based)", &mut hc, 1.0, 0);
                c.hotspot_camera_index = hc.max(0.0) as usize;
                let mut pc = c.plate_solve_camera_index as f64;
                num(ui, "plate-solve camera", &mut pc, 1.0, 0);
                c.plate_solve_camera_index = pc.max(0.0) as usize;
                let mut hf = c.handoff_min_frames as f64;
                num(ui, "handoff frames", &mut hf, 1.0, 0);
                c.handoff_min_frames = hf.max(1.0) as u32;
            });
            let ui = &mut cols[2];
            for i in 0..c.cam.len() {
                theme::section(ui, &format!("camera {}", i + 1));
                egui::Grid::new(format!("cfg_cam{i}")).num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                    text(ui, "name", &mut c.cam[i].name);
                    combo(ui, "role", &mut c.cam[i].role, &["guide", "main", "bubble"]);
                    let mut proj = if c.cam[i].projection == crate::state::Projection::Pinhole { "pinhole".to_string() } else { "fisheye".to_string() };
                    combo(ui, "projection", &mut proj, &["pinhole", "fisheye"]);
                    c.cam[i].projection = if proj == "fisheye" { crate::state::Projection::FisheyeEquidistant } else { crate::state::Projection::Pinhole };
                    let mut hw = c.cam[i].asi_index as f64;
                    num(ui, "ASI index", &mut hw, 1.0, 0);
                    c.cam[i].asi_index = hw.max(0.0) as usize;
                    boolean(ui, "enabled", &mut c.cam[i].enabled);
                    num(ui, "pixel µm", &mut c.cam[i].pixel_um, 0.01, 2);
                    num(ui, "focal mm", &mut c.cam[i].focal_mm, 1.0, 2);
                    let mut sw = c.cam[i].sensor_w as f64;
                    num(ui, "sensor w px", &mut sw, 1.0, 0);
                    c.cam[i].sensor_w = sw.max(16.0) as usize;
                    let mut sh = c.cam[i].sensor_h as f64;
                    num(ui, "sensor h px", &mut sh, 1.0, 0);
                    c.cam[i].sensor_h = sh.max(16.0) as usize;
                    let mut b = c.cam[i].bin as f64;
                    num(ui, "bin", &mut b, 1.0, 0);
                    c.cam[i].bin = (b as usize).clamp(1, 8);
                    num(ui, "rotation °", &mut c.cam[i].alignment_rotation_deg, 1.0, 1);
                    let mut g = c.cam[i].gain as f64;
                    num(ui, "gain", &mut g, 1.0, 0);
                    c.cam[i].gain = g as i64;
                    let mut e = c.cam[i].exposure_us as f64;
                    num(ui, "exposure µs", &mut e, 10.0, 0);
                    c.cam[i].exposure_us = e.max(32.0) as i64;
                    num(ui, "gamma", &mut c.cam[i].gamma, 0.01, 2);
                    boolean(ui, "gamma enabled", &mut c.cam[i].gamma_enabled);
                    let mut db = c.cam[i].tetra3_db.clone().unwrap_or_default();
                    text(ui, "tetra3 db", &mut db);
                    c.cam[i].tetra3_db = if db.is_empty() { None } else { Some(db) };
                });
            }
            theme::section(ui, "sources & paths");
            egui::Grid::new("cfg_src").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                combo(ui, "camera source", &mut c.camera_source, &["sim", "asi"]);
                text(ui, "ASI dll", &mut c.asi_dll);
                text(ui, "captures dir", &mut c.captures_dir);
                text(ui, "tetra3 db dir", &mut c.tetra3_db_dir);
                boolean(ui, "vsync", &mut c.ui_vsync);
                num(ui, "star limit mag", &mut c.star_limit_mag, 0.1, 1);
                let mut gc = c.alignment_grid_search_cells as f64;
                num(ui, "align grid-search cells", &mut gc, 1.0, 0);
                c.alignment_grid_search_cells = gc.max(1.0) as usize;
                num(ui, "align target rms ′", &mut c.alignment_target_rms_arcmin, 0.1, 1);
                boolean(ui, "align pause on fail", &mut c.alignment_pause_on_fail);
                num(ui, "align settle s", &mut c.alignment_settle_s, 0.1, 1);
                num(ui, "align settle tol °", &mut c.alignment_settle_tol_deg, 0.005, 3);
                let mut sc2 = c.alignment_settle_cycles as f64;
                num(ui, "align settle cycles", &mut sc2, 1.0, 0);
                c.alignment_settle_cycles = sc2.max(1.0) as usize;
                let mut ms = c.max_stars as f64;
                num(ui, "max stars", &mut ms, 100.0, 0);
                c.max_stars = ms as usize;
            });
        });
    });
}

fn num(ui: &mut egui::Ui, label: &str, v: &mut f64, speed: f64, decimals: usize) {
    ui.label(RichText::new(label).color(TEXT_2));
    ui.add(egui::DragValue::new(v).speed(speed).max_decimals(decimals).min_decimals(decimals.min(3)));
    ui.end_row();
}

fn boolean(ui: &mut egui::Ui, label: &str, v: &mut bool) {
    ui.label(RichText::new(label).color(TEXT_2));
    ui.checkbox(v, "");
    ui.end_row();
}

fn text(ui: &mut egui::Ui, label: &str, v: &mut String) {
    ui.label(RichText::new(label).color(TEXT_2));
    ui.add(egui::TextEdit::singleline(v).desired_width(150.0));
    ui.end_row();
}

fn combo(ui: &mut egui::Ui, label: &str, v: &mut String, options: &[&str]) {
    ui.label(RichText::new(label).color(TEXT_2));
    egui::ComboBox::from_id_salt(label).selected_text(v.clone()).show_ui(ui, |ui| {
        for o in options {
            ui.selectable_value(v, o.to_string(), *o);
        }
    });
    ui.end_row();
}

// ---------------------------------------------------------------------------
// Cameras: the sensor-calibration view (port of camera_manager.render_sensor_calibration)
// ---------------------------------------------------------------------------

pub struct CamerasState {
    pub combined: bool,
    /// Per-camera dominance weight in the combined overlay (0 = excluded).
    pub weights: Vec<f32>,
    pub match_scale: bool,
    pub include: Vec<bool>,
    pub base: usize,
    pub roi_pick: Option<usize>,
    pub message: String,
    pub name_edits: Vec<Vec<String>>,
}

impl Default for CamerasState {
    fn default() -> Self {
        CamerasState { combined: false, weights: vec![1.0, 1.0, 0.0], match_scale: true, include: vec![true, true, false], base: 0, roi_pick: None, message: String::new(), name_edits: Vec::new() }
    }
}

const ROI_FRACS: [(f64, &str); 6] = [(1.0, "1.0"), (0.5, ".5"), (0.25, ".25"), (0.125, ".125"), (0.0625, ".063"), (0.03125, ".031")];

pub fn cameras_screen(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, cs: &mut CamerasState, tx_cam: &crossbeam_channel::Sender<CamCmd>, tx_fw: &crossbeam_channel::Sender<FwCmd>) {
    let n = shared.cams.len();
    while cs.include.len() < n {
        cs.include.push(false);
    }
    // ---- header row: combined view controls, capture, reset / save
    ui.horizontal(|ui| {
        theme::section(ui, "cameras · sensor calibration");
        ui.add_space(10.0);
        if theme::mode_button(ui, "COMBINED VIEW", cs.combined, theme::ACCENT) {
            cs.combined = !cs.combined;
        }
        if cs.combined {
            ui.checkbox(&mut cs.match_scale, "match plate scale");
            while cs.weights.len() < n {
                cs.weights.push(0.0);
            }
            for i in 0..n {
                let name = shared.config.cam[i].name.split(' ').next().unwrap_or("cam").to_string();
                ui.label(RichText::new(name).color(TEXT_2));
                ui.add(egui::Slider::new(&mut cs.weights[i], 0.0..=1.0).show_value(false).fixed_decimals(2));
            }
            ui.label(RichText::new("weights shift which camera dominates").font(theme::sans(10.0)).color(DIM));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if theme::mode_button(ui, "save to config", false, GREEN) {
                // Persist the live settings of every slot into config.json.
                let mut c = shared.config.clone();
                for (i, cam) in c.cam.iter_mut().enumerate() {
                    let s = shared.cam_settings[i].load();
                    cam.gain = s.gain;
                    cam.exposure_us = s.exposure_us;
                    cam.gamma = s.gamma;
                    cam.gamma_enabled = s.gamma_enabled;
                    cam.alignment_rotation_deg = s.rotation_deg;
                    cam.enabled = s.connected;
                    cam.asi_index = shared.cam_hw.lock().unwrap().get(i).copied().unwrap_or(cam.asi_index);
                }
                cs.message = match c.save() {
                    Ok(()) => "camera settings saved".into(),
                    Err(e) => format!("save failed: {e}"),
                };
                let _ = tx_fw.send(FwCmd::Save);
            }
            if ui.button("reset to config").clicked() {
                for (i, cam) in shared.config.cam.iter().enumerate() {
                    let mut s = (**shared.cam_settings[i].load()).clone();
                    let d = CamSettings::from_config(cam);
                    s.gain = d.gain;
                    s.exposure_us = d.exposure_us;
                    s.gamma = d.gamma;
                    s.gamma_enabled = d.gamma_enabled;
                    s.rotation_deg = d.rotation_deg;
                    s.roi_frac = 1.0;
                    s.roi_cx = 0.5;
                    s.roi_cy = 0.5;
                    shared.cam_settings[i].store(Arc::new(s));
                    let _ = tx_cam.send(CamCmd::Apply { slot: i });
                }
                cs.message = "settings reset to config.json".into();
            }
            let any_armed = (0..n).any(|i| shared.cam(i).map_or(false, |c| c.armed));
            if theme::mode_button(ui, if any_armed { "ARMED" } else { "ARM ALL" }, any_armed, RED) {
                let _ = tx_cam.send(if any_armed { CamCmd::Disarm } else { CamCmd::Arm });
            }
            ui.add(egui::TextEdit::singleline(&mut st.capture_name).desired_width(100.0).hint_text("run name"));
            if ui.add_enabled(any_armed, egui::Button::new("save run")).clicked() {
                let _ = tx_cam.send(CamCmd::Dump { name: st.capture_name.clone() });
            }
            ui.label(RichText::new(&cs.message).font(theme::mono(10.5)).color(TEXT_2));
        });
    });

    // ---- feeds
    let avail = ui.available_size();
    let feed_h = (avail.y * 0.5).clamp(220.0, 520.0);
    if cs.combined {
        combined_view(ui, shared, st, cs, feed_h);
    } else {
        ui.columns(n, |cols| {
            for (i, col) in cols.iter_mut().enumerate() {
                let rect = crate::ui::camera_view(col, shared, st, i, true, Some(feed_h));
                // Click in the pane sets the ROI centre (when a ROI size is active).
                if let Some(r) = rect {
                    let resp = col.interact(r, egui::Id::new(("roi_pick", i)), Sense::click());
                    if resp.clicked() {
                        if let Some(p) = resp.interact_pointer_pos() {
                            let mut s = (**shared.cam_settings[i].load()).clone();
                            if s.roi_frac < 0.999 {
                                s.roi_cx = ((p.x - r.left()) / r.width()).clamp(0.0, 1.0) as f64;
                                s.roi_cy = ((p.y - r.top()) / r.height()).clamp(0.0, 1.0) as f64;
                                shared.cam_settings[i].store(Arc::new(s));
                                let _ = tx_cam.send(CamCmd::Apply { slot: i });
                            }
                        }
                    }
                }
            }
        });
    }
    ui.add_space(6.0);

    // ---- per-camera controls
    ui.columns(n, |cols| {
        for (i, col) in cols.iter_mut().enumerate() {
            camera_controls(col, shared, i, tx_cam, n);
        }
    });
    ui.add_space(8.0);

    // ---- filter wheels
    filter_wheel_cards(ui, shared, cs, tx_fw);
}

fn camera_controls(ui: &mut egui::Ui, shared: &Arc<Shared>, i: usize, tx_cam: &crossbeam_channel::Sender<CamCmd>, n: usize) {
    let cfg = &shared.config.cam[i];
    let snap = shared.cam(i);
    let mut s = (**shared.cam_settings[i].load()).clone();
    let before = s.clone();
    theme::card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&cfg.name).font(theme::sans(13.0)).color(TEXT));
            theme::tag(ui, &cfg.role, if cfg.role == "main" { AMBER } else if cfg.role == "bubble" { theme::VIOLET } else { ACCENT });
            let hw = shared.cam_hw.lock().unwrap().get(i).copied().unwrap_or(0);
            ui.label(RichText::new(format!("hw #{hw}")).font(theme::mono(10.5)).color(DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let connected = snap.as_ref().map_or(false, |c| c.connected);
                if theme::mode_button(ui, if s.connected { "CONNECTED" } else { "CONNECT" }, connected, GREEN) {
                    s.connected = !s.connected;
                }
                if let Some(c) = &snap {
                    ui.label(RichText::new(format!("{:.0} fps · {}", c.fps, c.source)).font(theme::mono(10.0)).color(DIM));
                }
            });
        });
        egui::Grid::new(("cam_ctl", i)).num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
            ui.label(RichText::new("gain").color(TEXT_2));
            ui.add(egui::Slider::new(&mut s.gain, 0..=600));
            ui.end_row();
            ui.label(RichText::new("exposure").color(TEXT_2));
            // Logarithmic 32 µs .. 2 s: half the slider travel sits below ~8 ms,
            // so the microsecond region has real motion range.
            ui.add(
                egui::Slider::new(&mut s.exposure_us, 1..=2_000_000)
                    .logarithmic(true)
                    .custom_formatter(|v, _| fmt_exposure(v as i64))
                    .custom_parser(parse_exposure),
            );
            ui.end_row();
            ui.label(RichText::new("gamma").color(TEXT_2));
            ui.horizontal(|ui| {
                ui.checkbox(&mut s.gamma_enabled, "");
                ui.add(egui::Slider::new(&mut s.gamma, 0.05..=2.0).logarithmic(true).fixed_decimals(2));
            });
            ui.end_row();
            ui.label(RichText::new("rotation °").color(TEXT_2));
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut s.rotation_deg, -180.0..=180.0).fixed_decimals(1));
                if ui.small_button("0°").on_hover_text("recenter rotation").clicked() {
                    s.rotation_deg = 0.0;
                }
            });
            ui.end_row();
            ui.label(RichText::new("ROI").color(TEXT_2));
            ui.horizontal(|ui| {
                for (f, label) in ROI_FRACS {
                    if theme::mode_button(ui, label, (s.roi_frac - f).abs() < 1e-6, ACCENT) {
                        s.roi_frac = f;
                        if f >= 0.999 {
                            s.roi_cx = 0.5;
                            s.roi_cy = 0.5;
                        }
                    }
                }
                ui.label(RichText::new("click the feed to centre").font(theme::sans(10.0)).color(DIM));
            });
            ui.end_row();
        });
        ui.horizontal(|ui| {
            let (w, h) = cfg.frame_size();
            ui.label(RichText::new(format!("{:.1} µm × bin {} · f {:.0} mm · {}×{} · FOV {:.2}°", cfg.pixel_um, cfg.bin, cfg.focal_mm, w, h, cfg.fov_deg(w as f64))).font(theme::mono(10.0)).color(DIM));
            if n > 1 {
                let j = (i + 1) % n;
                if ui.small_button(format!("swap ↔ {}", shared.config.cam[j].name.split(' ').next().unwrap_or("cam"))).clicked() {
                    let _ = tx_cam.send(CamCmd::Swap { a: i, b: j });
                }
            }
        });
    });
    if s != before {
        shared.cam_settings[i].store(Arc::new(s));
        let _ = tx_cam.send(CamCmd::Apply { slot: i });
    }
}

/// Weighted overlay of the selected feeds (rotation applied), either fitted
/// or scaled to a common plate scale for co-boresighting; the per-camera
/// weight sliders shift which feed dominates the brightness.
fn combined_view(ui: &mut egui::Ui, shared: &Arc<Shared>, st: &mut UiState, cs: &mut CamerasState, h: f32) {
    let avail = ui.available_width();
    let (r, _p) = ui.allocate_painter(egui::Vec2::new(avail, h), Sense::hover());
    let cams: Vec<(usize, f32)> = (0..shared.cams.len()).map(|i| (i, cs.weights.get(i).copied().unwrap_or(0.0))).collect();
    crate::ui::weighted_overlay(ui, shared, st, r.rect, &cams, cs.match_scale);
}

fn filter_wheel_cards(ui: &mut egui::Ui, shared: &Arc<Shared>, cs: &mut CamerasState, tx_fw: &crossbeam_channel::Sender<FwCmd>) {
    let fw = shared.filter_wheels.load();
    ui.horizontal(|ui| {
        theme::section(ui, "filter wheels");
        ui.label(RichText::new(&fw.sdk).font(theme::mono(10.0)).color(DIM));
    });
    while cs.name_edits.len() < fw.wheels.len() {
        cs.name_edits.push(Vec::new());
    }
    let n = fw.wheels.len().max(1);
    ui.columns(n, |cols| {
        for (wi, w) in fw.wheels.iter().enumerate() {
            let ui = &mut cols[wi];
            theme::card(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&w.name).font(theme::sans(13.0)).color(TEXT));
                    theme::tag(ui, if w.kind == "efw" { "EFW" } else { "MANUAL" }, if w.kind == "efw" { ACCENT } else { AMBER });
                    let cam_name = shared.config.cam.get(w.camera).map(|c| c.name.clone()).unwrap_or_default();
                    ui.label(RichText::new(format!("on {cam_name}")).font(theme::sans(10.5)).color(TEXT_2));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let col = if w.moving { AMBER } else if w.connected { GREEN } else { DIM };
                        theme::tag(ui, if w.moving { "MOVING" } else if w.connected { "READY" } else { "OFFLINE" }, col);
                        ui.label(RichText::new(&w.message).font(theme::mono(10.0)).color(DIM));
                    });
                });
                // Slot buttons: current highlighted; click = goto (EFW) / mark (manual).
                ui.horizontal_wrapped(|ui| {
                    for (si, name) in w.slots.iter().enumerate() {
                        let label = if name.is_empty() { format!("{}", si + 1) } else { format!("{} {}", si + 1, name) };
                        let current = w.position == si;
                        let target = w.target == Some(si);
                        if theme::mode_button(ui, &label, current || target, if current { GREEN } else { AMBER }) {
                            let _ = tx_fw.send(if w.kind == "efw" { FwCmd::Goto { wheel: wi, slot: si } } else { FwCmd::MarkPosition { wheel: wi, slot: si } });
                        }
                    }
                });
                let cur = w.slots.get(w.position).cloned().unwrap_or_default();
                ui.label(RichText::new(format!("current filter: {}", if cur.is_empty() { "(unnamed)".to_string() } else { cur })).font(theme::mono(11.0)).color(TEXT));
                // Assignments editor.
                egui::CollapsingHeader::new(RichText::new("filter assignments").font(theme::sans(11.0)).color(TEXT_2)).id_salt(("fw_edit", wi)).show(ui, |ui| {
                    let edits = &mut cs.name_edits[wi];
                    while edits.len() < w.slots.len() {
                        edits.push(w.slots[edits.len()].clone());
                    }
                    egui::Grid::new(("fw_grid", wi)).num_columns(3).spacing([8.0, 3.0]).show(ui, |ui| {
                        for si in 0..w.slots.len() {
                            ui.label(RichText::new(format!("slot {}", si + 1)).color(TEXT_2));
                            ui.add(egui::TextEdit::singleline(&mut edits[si]).desired_width(110.0));
                            if edits[si] != w.slots[si] && ui.small_button("set").clicked() {
                                let _ = tx_fw.send(FwCmd::SetSlotName { wheel: wi, slot: si, name: edits[si].clone() });
                            }
                            ui.end_row();
                        }
                    });
                });
            });
        }
    });
}

/// 570 -> "570 µs", 57_000 -> "57.0 ms", 1_500_000 -> "1.50 s".
pub fn fmt_exposure(us: i64) -> String {
    if us < 1_000 {
        format!("{us} µs")
    } else if us < 1_000_000 {
        let n = format!("{:.3}", us as f64 / 1000.0);
        format!("{} ms", n.trim_end_matches('0').trim_end_matches('.'))
    } else {
        format!("{:.2} s", us as f64 / 1e6)
    }
}

/// Accepts "570", "570us", "5.7ms", "0.5s" (bare numbers = µs).
pub fn parse_exposure(text: &str) -> Option<f64> {
    let t = text.trim().to_ascii_lowercase().replace('µ', "u");
    let (num, mult) = if let Some(x) = t.strip_suffix("us") {
        (x, 1.0)
    } else if let Some(x) = t.strip_suffix("ms") {
        (x, 1_000.0)
    } else if let Some(x) = t.strip_suffix('s') {
        (x, 1_000_000.0)
    } else {
        (t.as_str(), 1.0)
    };
    num.trim().parse::<f64>().ok().map(|v| v * mult)
}
