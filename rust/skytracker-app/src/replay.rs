//! Replay / post-processing screen (port of post_process.py +
//! post_process_ui.py): browse saved capture runs, replay the camera frame
//! sequences in sync, adjust gamma / brightness / contrast, stabilize
//! (flow-method, skytracker-imaging), sharpen, stack the best N frames, draw
//! in-track / cross-track + meta overlays and annotations, and export MP4.
//!
//! Threading: the UI thread never touches the disk or the imaging engines.
//! One worker thread per camera decodes + processes frames on request and
//! publishes `ProcessedFrame`s through an `ArcSwapOption`; long jobs (export,
//! stack, library scan) run on their own threads and publish `JobStatus`.
//!
//! On-disk layout (written by capture_manager.py / skytracker-camera):
//!
//!   <captures>/<TARGET>_<NORAD>_<YYYY_MM_DD_HH_MM_SS>/   (or manual_<ts>/)
//!       Camera1_000000__<YYYY_MM_DD_HH_MM_SS.ffffff>[Z].bmp|png
//!       Camera2_000000__...
//!       trajectory.csv
//!       postproc.json      <- sidecar (display_name/favorite/tags/notes/annotations)

#![allow(dead_code)]

use crate::theme;
use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_channel::{Receiver, Sender};
use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use skytracker_imaging::enhance;
use skytracker_imaging::image::ImageF32;
use skytracker_imaging::metrics;
use skytracker_imaging::stabilize::{detect_reference_points, estimate_flow, StabilizeParams};
use skytracker_imaging::warp::{warp_affine, Border};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SIDECAR_NAME: &str = "postproc.json";
const SPEEDS: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];
const DECODE_CACHE: usize = 48;
const PREFETCH: usize = 3;
/// Default sharpening layers (sharpen.DEFAULT_LAYERS): (sigma px, amount).
const DEFAULT_LAYERS: [(f64, f64); 3] = [(1.0, 0.6), (2.5, 0.35), (6.0, 0.15)];

// ---------------------------------------------------------------------------
// Time helpers (UTC, no chrono)
// ---------------------------------------------------------------------------
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp as i64 + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn civil_to_epoch(y: i64, mo: u32, d: u32, h: u32, mi: u32, s: f64) -> f64 {
    days_from_civil(y, mo, d) as f64 * 86400.0 + h as f64 * 3600.0 + mi as f64 * 60.0 + s
}

/// Parse a frame-filename timestamp 'YYYY_MM_DD_HH_MM_SS[.ffffff]' -> epoch sec (UTC).
pub fn parse_frame_time(ts: &str) -> Option<f64> {
    let (date_part, frac) = match ts.split_once('.') {
        Some((a, b)) => (a, b),
        None => (ts, "0"),
    };
    let parts: Vec<&str> = date_part.split('_').collect();
    if parts.len() != 6 {
        return None;
    }
    let n: Vec<i64> = parts.iter().map(|p| p.parse::<i64>().ok()).collect::<Option<Vec<_>>>()?;
    if !frac.chars().all(|c| c.is_ascii_digit()) || frac.is_empty() {
        return None;
    }
    let f: f64 = format!("0.{frac}").parse().ok()?;
    if !(1..=12).contains(&n[1]) || !(1..=31).contains(&n[2]) || n[3] > 23 || n[4] > 59 || n[5] > 60 {
        return None;
    }
    Some(civil_to_epoch(n[0], n[1] as u32, n[2] as u32, n[3] as u32, n[4] as u32, n[5] as f64 + f))
}

/// Parse a trajectory.csv ISO timestamp ('2025-09-16T17:05:55.450644+00:00Z',
/// '...+00:00', '...Z', or naive = UTC) -> epoch sec.
pub fn parse_iso_time(s: &str) -> Option<f64> {
    let mut s = s.trim();
    if let Some(stripped) = s.strip_suffix('Z') {
        s = stripped;
    }
    if s.len() < 19 {
        return None;
    }
    let (main, offset_s) = {
        // Offset sign after the time part (position >= 19).
        let tail = &s[19..];
        if let Some(p) = tail.find(['+', '-']) {
            (&s[..19 + p], Some(&tail[p..]))
        } else {
            (s, None)
        }
    };
    let y: i64 = main.get(0..4)?.parse().ok()?;
    let mo: u32 = main.get(5..7)?.parse().ok()?;
    let d: u32 = main.get(8..10)?.parse().ok()?;
    let h: u32 = main.get(11..13)?.parse().ok()?;
    let mi: u32 = main.get(14..16)?.parse().ok()?;
    let sec: f64 = main.get(17..)?.parse().ok()?;
    let mut epoch = civil_to_epoch(y, mo, d, h, mi, sec);
    if let Some(off) = offset_s {
        let sign = if off.starts_with('-') { -1.0 } else { 1.0 };
        let body = &off[1..];
        let (oh, om) = match body.split_once(':') {
            Some((a, b)) => (a.parse::<f64>().ok()?, b.parse::<f64>().ok()?),
            None if body.len() == 4 => (body[..2].parse::<f64>().ok()?, body[2..].parse::<f64>().ok()?),
            None => (body.parse::<f64>().ok()?, 0.0),
        };
        epoch -= sign * (oh * 3600.0 + om * 60.0);
    }
    Some(epoch)
}

/// Compact UTC HH:MM:SS.mmm.
pub fn fmt_utc(epoch: f64) -> String {
    let secs = epoch.floor();
    let ms = ((epoch - secs) * 1000.0).floor() as i64;
    let day_s = secs.rem_euclid(86400.0) as i64;
    format!("{:02}:{:02}:{:02}.{:03}", day_s / 3600, (day_s / 60) % 60, day_s % 60, ms)
}

/// 'YYYY-MM-DD HH:MM:SS' UTC.
pub fn fmt_date(epoch: f64) -> String {
    let secs = epoch.floor() as i64;
    let days = secs.div_euclid(86400);
    let day_s = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", day_s / 3600, (day_s / 60) % 60, day_s % 60)
}

// ---------------------------------------------------------------------------
// Trajectory series
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub t: Vec<f64>,
    pub az: Vec<f64>, // unwrapped
    pub el: Vec<f64>,
    pub dist: Vec<f64>,
    pub px: Vec<f64>,
    pub py: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct TrajState {
    pub az: f64,
    pub el: f64,
    pub dist: f64,
    pub px: f64,
    pub py: f64,
}

fn interp1(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    // numpy.interp: clamped at the ends, linear inside (xs sorted ascending).
    if xs.is_empty() {
        return f64::NAN;
    }
    if x <= xs[0] {
        return ys[0];
    }
    let n = xs.len();
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let i = xs.partition_point(|&v| v <= x); // first index with xs[i] > x
    let (x0, x1) = (xs[i - 1], xs[i]);
    let (y0, y1) = (ys[i - 1], ys[i]);
    if x1 <= x0 {
        return y1;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

impl Trajectory {
    pub fn from_rows(mut rows: Vec<(f64, f64, f64, f64, f64, f64)>) -> Self {
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut tr = Trajectory::default();
        for (t, az, el, dist, px, py) in rows {
            tr.t.push(t);
            tr.az.push(az);
            tr.el.push(el);
            tr.dist.push(dist);
            tr.px.push(px);
            tr.py.push(py);
        }
        // np.unwrap(period=360)
        for i in 1..tr.az.len() {
            let mut d = tr.az[i] - tr.az[i - 1];
            while d > 180.0 {
                tr.az[i] -= 360.0;
                d -= 360.0;
            }
            while d < -180.0 {
                tr.az[i] += 360.0;
                d += 360.0;
            }
        }
        tr
    }

    pub fn load(csv_path: &Path) -> Trajectory {
        let Ok(text) = std::fs::read_to_string(csv_path) else {
            return Trajectory::default();
        };
        let mut lines = text.lines();
        let Some(header) = lines.next() else {
            return Trajectory::default();
        };
        let cols: Vec<String> = split_csv_line(header).iter().map(|c| c.trim().to_lowercase()).collect();
        let idx = |name: &str| cols.iter().position(|c| c == name);
        let (Some(it), Some(iaz), Some(iel)) = (idx("timestamp"), idx("azimuth_deg"), idx("altitude_deg")) else {
            return Trajectory::default();
        };
        let idist = idx("distance_km");
        let ipx = idx("pixel_x");
        let ipy = idx("pixel_y");
        let getf = |f: &[String], i: Option<usize>| -> f64 {
            i.and_then(|i| f.get(i)).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN)
        };
        let mut rows = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let f = split_csv_line(line);
            let (Some(ts), Some(az), Some(el)) = (f.get(it), f.get(iaz), f.get(iel)) else {
                continue;
            };
            let (Some(t), Ok(az), Ok(el)) = (parse_iso_time(ts), az.trim().parse::<f64>(), el.trim().parse::<f64>()) else {
                continue;
            };
            rows.push((t, az, el, getf(&f, idist), getf(&f, ipx), getf(&f, ipy)));
        }
        Trajectory::from_rows(rows)
    }

    pub fn valid(&self) -> bool {
        self.t.len() >= 2
    }

    pub fn interp(&self, t: f64) -> Option<TrajState> {
        if self.t.is_empty() {
            return None;
        }
        Some(TrajState {
            az: interp1(&self.t, &self.az, t).rem_euclid(360.0),
            el: interp1(&self.t, &self.el, t),
            dist: interp1(&self.t, &self.dist, t),
            px: interp1(&self.t, &self.px, t),
            py: interp1(&self.t, &self.py, t),
        })
    }

    /// Central-difference (az_rate, el_rate) deg/s around t.
    pub fn rate(&self, t: f64, dt: f64) -> (f64, f64) {
        if !self.valid() {
            return (0.0, 0.0);
        }
        let a0 = interp1(&self.t, &self.az, t - dt);
        let a1 = interp1(&self.t, &self.az, t + dt);
        let e0 = interp1(&self.t, &self.el, t - dt);
        let e1 = interp1(&self.t, &self.el, t + dt);
        ((a1 - a0) / (2.0 * dt), (e1 - e0) / (2.0 * dt))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrackVectors {
    pub anchor: [f64; 2],
    pub intrack: [f64; 2],
    pub cross_p: [f64; 2],
    pub cross_n: [f64; 2],
}

/// In-track / cross-track overlay vectors (output px) — port of
/// post_process.compute_track_vectors: on-sky velocity (az_rate*cos(el),
/// el_rate), anchored at the frame centre, image +y down.
pub fn compute_track_vectors(traj: &Trajectory, t: f64, out_w: f64, out_h: f64) -> Option<TrackVectors> {
    if !traj.valid() {
        return None;
    }
    let (az_rate, el_rate) = traj.rate(t, 0.5);
    let cos_el = traj.interp(t).map(|s| s.el.to_radians().cos()).unwrap_or(1.0).max(0.087);
    let vx = az_rate * cos_el;
    let vy = el_rate;
    let norm = vx.hypot(vy);
    if norm < 1e-9 {
        return None;
    }
    let (ux, uy) = (vx / norm, vy / norm);
    let (ix, iy) = (ux, -uy);
    let (cx, cy) = (-iy, ix);
    let l = 0.18 * out_w.min(out_h);
    let (ax, ay) = (out_w / 2.0, out_h / 2.0);
    Some(TrackVectors {
        anchor: [ax, ay],
        intrack: [ax + ix * l, ay + iy * l],
        cross_p: [ax + cx * l * 0.6, ay + cy * l * 0.6],
        cross_n: [ax - cx * l * 0.6, ay - cy * l * 0.6],
    })
}

// ---------------------------------------------------------------------------
// Runs + library
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct FrameRef {
    pub path: PathBuf,
    pub t: f64,
    pub seq: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CamFrames {
    /// 0-based camera index (Camera1_ -> 0).
    pub cam_index: usize,
    pub frames: Vec<FrameRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Annotation {
    Text { cam: usize, x: f64, y: f64, text: String },
    Arrow { cam: usize, x0: f64, y0: f64, x1: f64, y1: f64 },
    Box { cam: usize, x0: f64, y0: f64, x1: f64, y1: f64 },
}

impl Annotation {
    pub fn cam(&self) -> usize {
        match self {
            Annotation::Text { cam, .. } | Annotation::Arrow { cam, .. } | Annotation::Box { cam, .. } => *cam,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match self {
            Annotation::Text { cam, x, y, text } => serde_json::json!({"type": "text", "cam": cam, "x": x, "y": y, "text": text}),
            Annotation::Arrow { cam, x0, y0, x1, y1 } => serde_json::json!({"type": "arrow", "cam": cam, "x0": x0, "y0": y0, "x1": x1, "y1": y1}),
            Annotation::Box { cam, x0, y0, x1, y1 } => serde_json::json!({"type": "box", "cam": cam, "x0": x0, "y0": y0, "x1": x1, "y1": y1}),
        }
    }

    fn from_json(v: &serde_json::Value) -> Option<Annotation> {
        let cam = v["cam"].as_u64().unwrap_or(0) as usize;
        let f = |k: &str| v[k].as_f64();
        match v["type"].as_str()? {
            "text" => Some(Annotation::Text { cam, x: f("x")?, y: f("y")?, text: v["text"].as_str().unwrap_or("").to_string() }),
            "arrow" => Some(Annotation::Arrow { cam, x0: f("x0")?, y0: f("y0")?, x1: f("x1")?, y1: f("y1")? }),
            "box" => Some(Annotation::Box { cam, x0: f("x0")?, y0: f("y0")?, x1: f("x1")?, y1: f("y1")? }),
            _ => None,
        }
    }
}

/// postproc.json sidecar (unknown keys are preserved on save).
#[derive(Clone, Debug, Default)]
pub struct Sidecar {
    pub display_name: String,
    pub favorite: bool,
    pub tags: Vec<String>,
    pub notes: String,
    pub annotations: Vec<Annotation>,
    extra: serde_json::Map<String, serde_json::Value>,
}

impl Sidecar {
    pub fn load(path: &Path) -> Sidecar {
        let v: serde_json::Value = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let mut sc = Sidecar::default();
        if let serde_json::Value::Object(map) = v {
            sc.display_name = map.get("display_name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            sc.favorite = map.get("favorite").and_then(|v| v.as_bool()).unwrap_or(false);
            sc.tags = map
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                .unwrap_or_default();
            sc.notes = map.get("notes").and_then(|v| v.as_str()).unwrap_or("").to_string();
            sc.annotations = map
                .get("annotations")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(Annotation::from_json).collect())
                .unwrap_or_default();
            for (k, v) in map {
                if !matches!(k.as_str(), "display_name" | "favorite" | "tags" | "notes" | "annotations") {
                    sc.extra.insert(k, v);
                }
            }
        }
        sc
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut map = self.extra.clone();
        map.insert("display_name".into(), serde_json::Value::String(self.display_name.clone()));
        map.insert("favorite".into(), serde_json::Value::Bool(self.favorite));
        map.insert("tags".into(), serde_json::json!(self.tags));
        map.insert("notes".into(), serde_json::Value::String(self.notes.clone()));
        map.insert("annotations".into(), serde_json::Value::Array(self.annotations.iter().map(|a| a.to_json()).collect()));
        let text = serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".into());
        std::fs::write(path, text)
    }
}

#[derive(Clone, Debug)]
pub struct RunInfo {
    pub path: PathBuf,
    pub folder: String,
    pub target: String,
    pub norad: Option<String>,
    pub is_manual: bool,
    /// Folder timestamp (epoch s UTC), None when the folder name has none.
    pub start_epoch: Option<f64>,
    /// Cameras with frames, sorted by camera index.
    pub cams: Vec<CamFrames>,
    pub sidecar: Sidecar,
    pub trajectory: Arc<Trajectory>,
}

/// Parse 'Camera<N>_<seq>__<ts>[Z].(bmp|png|jpg)' -> (cam 0-based, seq, epoch).
pub fn parse_frame_name(name: &str) -> Option<(usize, u64, f64)> {
    let lower = name.to_ascii_lowercase();
    let stem_len = if lower.ends_with(".bmp") || lower.ends_with(".png") || lower.ends_with(".jpg") {
        name.len() - 4
    } else if lower.ends_with(".jpeg") {
        name.len() - 5
    } else {
        return None;
    };
    let stem = &name[..stem_len];
    let rest = stem.strip_prefix("Camera").or_else(|| stem.strip_prefix("camera"))?;
    let cam_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let cam: usize = rest[..cam_end].parse().ok()?;
    let rest = rest[cam_end..].strip_prefix('_')?;
    let seq_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let seq: u64 = rest[..seq_end].parse().ok()?;
    let ts = rest[seq_end..].strip_prefix("__")?;
    let ts = ts.strip_suffix('Z').unwrap_or(ts);
    let t = parse_frame_time(ts)?;
    if cam == 0 {
        return None;
    }
    Some((cam - 1, seq, t))
}

/// Split '<prefix>_<YYYY_MM_DD_HH_MM_SS>' -> (prefix, epoch).
fn parse_folder(name: &str) -> Option<(&str, f64)> {
    if name.len() < 21 {
        return None;
    }
    let split = name.len() - 20;
    if !name.is_char_boundary(split) || name.as_bytes()[split] != b'_' {
        return None;
    }
    let ts = &name[split + 1..];
    let t = parse_frame_time(ts)?;
    if ts.contains('.') {
        return None;
    }
    Some((&name[..split], t))
}

impl RunInfo {
    /// Index one run folder; None when it holds no camera frames.
    pub fn open(path: &Path) -> Option<RunInfo> {
        let folder = path.file_name()?.to_string_lossy().to_string();
        let is_manual = folder.to_lowercase().starts_with("manual_");
        let mut target = folder.clone();
        let mut norad = None;
        let mut start_epoch = None;
        if let Some((prefix, t)) = parse_folder(&folder) {
            start_epoch = Some(t);
            if is_manual {
                target = "Manual".into();
            } else {
                let toks: Vec<&str> = prefix.split('_').collect();
                if toks.len() > 1 && toks.last().map_or(false, |t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())) {
                    norad = Some(toks[toks.len() - 1].to_string());
                    target = toks[..toks.len() - 1].join("_");
                } else {
                    target = prefix.to_string();
                }
            }
        }
        let mut cams: HashMap<usize, Vec<FrameRef>> = HashMap::new();
        for entry in std::fs::read_dir(path).ok()?.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some((cam, seq, t)) = parse_frame_name(&name) {
                cams.entry(cam).or_default().push(FrameRef { path: entry.path(), t, seq });
            }
        }
        if cams.is_empty() {
            return None;
        }
        let mut cams: Vec<CamFrames> = cams.into_iter().map(|(cam_index, frames)| CamFrames { cam_index, frames }).collect();
        for c in cams.iter_mut() {
            c.frames.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        }
        cams.sort_by_key(|c| c.cam_index);
        let sidecar = Sidecar::load(&path.join(SIDECAR_NAME));
        let trajectory = Arc::new(Trajectory::load(&path.join("trajectory.csv")));
        Some(RunInfo { path: path.to_path_buf(), folder, target, norad, is_manual, start_epoch, cams, sidecar, trajectory })
    }

    pub fn display_name(&self) -> String {
        if self.sidecar.display_name.is_empty() {
            self.folder.clone()
        } else {
            self.sidecar.display_name.clone()
        }
    }

    pub fn t0(&self) -> Option<f64> {
        self.cams.iter().filter_map(|c| c.frames.first().map(|f| f.t)).fold(None, |m, t| Some(m.map_or(t, |m: f64| m.min(t))))
    }

    pub fn t1(&self) -> Option<f64> {
        self.cams.iter().filter_map(|c| c.frames.last().map(|f| f.t)).fold(None, |m, t| Some(m.map_or(t, |m: f64| m.max(t))))
    }

    pub fn duration(&self) -> f64 {
        match (self.t0(), self.t1()) {
            (Some(a), Some(b)) => b - a,
            _ => 0.0,
        }
    }

    pub fn total_frames(&self) -> usize {
        self.cams.iter().map(|c| c.frames.len()).sum()
    }

    /// Index of the frame camera slot `slot` shows at epoch t (last frame <= t, clamped).
    pub fn frame_index_at(&self, slot: usize, t: f64) -> Option<usize> {
        let frames = &self.cams.get(slot)?.frames;
        if frames.is_empty() {
            return None;
        }
        let i = frames.partition_point(|f| f.t <= t);
        Some(i.saturating_sub(1).min(frames.len() - 1))
    }

    pub fn sidecar_path(&self) -> PathBuf {
        self.path.join(SIDECAR_NAME)
    }
}

/// Scan a captures directory; runs with frames only, newest first.
pub fn scan_library(dir: &Path) -> Vec<RunInfo> {
    let mut runs = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return runs;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        if let Some(r) = RunInfo::open(&p) {
            runs.push(r);
        }
    }
    runs.sort_by(|a, b| {
        let ka = a.start_epoch.or_else(|| a.t0()).unwrap_or(f64::MIN);
        let kb = b.start_epoch.or_else(|| b.t0()).unwrap_or(f64::MIN);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal).then_with(|| b.folder.cmp(&a.folder))
    });
    runs
}

// ---------------------------------------------------------------------------
// Frame processing: LUT, decode, reduce, stabilize, sharpen, stack
// ---------------------------------------------------------------------------
/// Gamma stretch + brightness/contrast 256-LUT (post_process.FrameProcessor):
/// out = contrast * (255*(i/255)^(1/gamma) - 128) + 128 + brightness.
pub fn build_lut(gamma: f64, brightness: f64, contrast: f64) -> [u8; 256] {
    let g = gamma.max(1e-3);
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let stretched = 255.0 * (i as f64 / 255.0).powf(1.0 / g);
        let out = contrast * (stretched - 128.0) + 128.0 + brightness;
        *v = out.round().clamp(0.0, 255.0) as u8;
    }
    lut
}

pub fn is_default_params(gamma: f64, brightness: f64, contrast: f64) -> bool {
    gamma == 1.0 && brightness == 0.0 && contrast == 1.0
}

/// 8-bit grayscale frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Gray { w, h, data: vec![0; w * h] }
    }

    pub fn apply_lut(&self, lut: &[u8; 256]) -> Gray {
        Gray { w: self.w, h: self.h, data: self.data.iter().map(|&v| lut[v as usize]).collect() }
    }

    pub fn to_f32(&self) -> ImageF32 {
        ImageF32::from_vec(self.data.iter().map(|&v| v as f32).collect(), self.w, self.h)
    }

    pub fn to_f32_unit(&self) -> ImageF32 {
        ImageF32::from_vec(self.data.iter().map(|&v| v as f32 / 255.0).collect(), self.w, self.h)
    }

    pub fn from_f32(img: &ImageF32, scale: f32) -> Gray {
        Gray { w: img.w, h: img.h, data: img.data.iter().map(|&v| (v * scale).round().clamp(0.0, 255.0) as u8).collect() }
    }

    /// Integer-factor box downsample (the reduced decode of the Python replay).
    pub fn downsample(&self, f: usize) -> Gray {
        if f <= 1 {
            return self.clone();
        }
        let nw = (self.w / f).max(1);
        let nh = (self.h / f).max(1);
        let mut out = Gray::new(nw, nh);
        let inv = 1.0 / (f * f) as f32;
        for oy in 0..nh {
            for ox in 0..nw {
                let mut s: u32 = 0;
                for dy in 0..f {
                    let row = (oy * f + dy).min(self.h - 1) * self.w;
                    for dx in 0..f {
                        s += self.data[row + (ox * f + dx).min(self.w - 1)] as u32;
                    }
                }
                out.data[oy * nw + ox] = (s as f32 * inv + 0.5) as u8;
            }
        }
        out
    }

    pub fn to_rgb(&self) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(self.data.len() * 3);
        for &v in &self.data {
            rgb.extend_from_slice(&[v, v, v]);
        }
        rgb
    }
}

/// Decode a BMP/PNG/JPEG capture frame to 8-bit gray (luma).
pub fn load_gray(path: &Path) -> Result<Gray, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data = img.into_luma8().into_raw();
    if data.len() != w * h {
        return Err(format!("{}: unexpected buffer size", path.display()));
    }
    Ok(Gray { w, h, data })
}

/// Largest power-of-two reduction that still yields >= pane width (reduce_for).
pub fn reduce_for(full_w: usize, pane_w: usize) -> usize {
    if full_w == 0 || pane_w == 0 {
        return 1;
    }
    let mut r = 1;
    while r < 8 && full_w / (r * 2) >= pane_w {
        r *= 2;
    }
    r
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StabSettings {
    pub max_features: usize,
    pub ransac_threshold: f64,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
}

impl Default for StabSettings {
    fn default() -> Self {
        StabSettings { max_features: 600, ransac_threshold: 3.0, min_inliers: 8, min_inlier_ratio: 0.4 }
    }
}

impl StabSettings {
    fn params(&self) -> StabilizeParams {
        StabilizeParams {
            max_features: self.max_features,
            ransac_threshold: self.ransac_threshold,
            min_inliers: self.min_inliers,
            min_inlier_ratio: self.min_inlier_ratio,
            ..StabilizeParams::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StabInfo {
    pub ok: bool,
    pub inliers: usize,
    pub reason: Option<String>,
    pub m: Option<[[f64; 3]; 2]>,
}

/// Single-anchor flow stabilizer (stabilizer.Stabilizer method="flow" over
/// the Rust engine): features on the reference, LK + RANSAC similarity per
/// frame, warp onto the reference with black borders.
pub struct Stabilizer {
    settings: StabSettings,
    ref_gray: ImageF32,
    ref_points: Vec<[f32; 2]>,
}

impl Stabilizer {
    pub fn new(reference: &Gray, settings: StabSettings) -> Self {
        let ref_gray = reference.to_f32();
        let ref_points = detect_reference_points(&ref_gray, settings.max_features);
        Stabilizer { settings, ref_gray, ref_points }
    }

    pub fn estimate(&self, cur: &ImageF32) -> StabInfo {
        let est = estimate_flow(&self.ref_gray, &self.ref_points, cur, &self.settings.params());
        StabInfo { ok: est.m.is_some(), inliers: est.num_inliers, reason: est.reject_reason, m: est.m }
    }

    /// Warp `frame` onto the reference; passthrough (identity) when rejected.
    pub fn stabilize(&self, frame: &Gray) -> (Gray, StabInfo) {
        let cur = frame.to_f32();
        let info = self.estimate(&cur);
        match info.m {
            Some(m) => {
                let mf = [[m[0][0] as f32, m[0][1] as f32, m[0][2] as f32], [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32]];
                let warped = warp_affine(&cur, &mf, frame.w, frame.h, Border::Constant(0.0));
                (Gray::from_f32(&warped, 1.0), info)
            }
            None => (frame.clone(), info),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpenSettings {
    pub on: bool,
    /// Scales the default layer amounts (1.0 = sharpen.DEFAULT_LAYERS).
    pub strength: f64,
    pub stretch: bool,
}

impl Default for SharpenSettings {
    fn default() -> Self {
        SharpenSettings { on: false, strength: 1.0, stretch: false }
    }
}

/// sharpen.finish on a gray frame: multi-scale unsharp (+ optional auto-stretch).
pub fn finish_gray(g: &Gray, s: &SharpenSettings) -> Gray {
    let img = g.to_f32_unit();
    let layers: Vec<(f64, f64)> = DEFAULT_LAYERS.iter().map(|&(sig, amt)| (sig, amt * s.strength)).collect();
    let mut out = enhance::unsharp_layers(&img, &layers);
    if s.stretch {
        out = enhance::auto_stretch(&out, 0.25, 99.9, 0.25, 5.0);
    }
    Gray::from_f32(&out, 255.0)
}

pub fn finish_f32(img: &ImageF32, stretch: bool) -> ImageF32 {
    let out = enhance::unsharp_layers(img, &DEFAULT_LAYERS);
    if stretch {
        enhance::auto_stretch(&out, 0.25, 99.9, 0.25, 5.0)
    } else {
        out
    }
}

/// Coverage-weighted aligned mean (stacking.LuckyStacker, gray): frames are
/// aligned to `frames[reference]` (flow stabilizer) and averaged; frames
/// that fail to align are rejected. Returns (mean 0..255 f64, n_stacked, n_rejected).
pub fn stack_gray(frames: &[Gray], reference: usize, align: bool, settings: StabSettings) -> Option<(Vec<f64>, usize, usize)> {
    let r = frames.get(reference)?;
    let (w, h) = (r.w, r.h);
    let mut accum: Vec<f64> = r.data.iter().map(|&v| v as f64).collect();
    let mut weight: Vec<f64> = vec![1.0; w * h];
    let mut n_stacked = 1;
    let mut n_rejected = 0;
    let stab = if align { Some(Stabilizer::new(r, settings)) } else { None };
    let ones = ImageF32::from_vec(vec![1.0; w * h], w, h);
    for (i, f) in frames.iter().enumerate() {
        if i == reference {
            continue;
        }
        if f.w != w || f.h != h {
            n_rejected += 1;
            continue;
        }
        match &stab {
            Some(st) => {
                let cur = f.to_f32();
                let info = st.estimate(&cur);
                let Some(m) = info.m else {
                    n_rejected += 1;
                    continue;
                };
                let mf = [[m[0][0] as f32, m[0][1] as f32, m[0][2] as f32], [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32]];
                let warped = warp_affine(&cur, &mf, w, h, Border::Constant(0.0));
                let cover = warp_affine(&ones, &mf, w, h, Border::Constant(0.0));
                for k in 0..w * h {
                    let c = cover.data[k].clamp(0.0, 1.0) as f64;
                    accum[k] += warped.data[k] as f64 * c;
                    weight[k] += c;
                }
            }
            None => {
                for k in 0..w * h {
                    accum[k] += f.data[k] as f64;
                    weight[k] += 1.0;
                }
            }
        }
        n_stacked += 1;
    }
    let mean: Vec<f64> = accum.iter().zip(weight.iter()).map(|(a, wgt)| a / wgt.max(1e-6)).collect();
    Some((mean, n_stacked, n_rejected))
}

/// Garbage pre-cull content score (stacking.content_score, simple form):
/// mean absolute deviation from the frame median at half resolution. Pure
/// noise / blank sky collapses toward zero while a real target keeps a
/// prominent deviation, so sharpness (which noise fools) never ranks junk in.
pub fn content_score_gray(g: &Gray) -> f64 {
    let s = g.downsample(2);
    if s.data.is_empty() {
        return 0.0;
    }
    let mut hist = [0u32; 256];
    for &v in &s.data {
        hist[v as usize] += 1;
    }
    let half = s.data.len() / 2;
    let mut acc = 0usize;
    let mut med = 0u8;
    for (i, &c) in hist.iter().enumerate() {
        acc += c as usize;
        if acc > half {
            med = i as u8;
            break;
        }
    }
    let m = med as f64;
    s.data.iter().map(|&v| (v as f64 - m).abs()).sum::<f64>() / s.data.len() as f64
}

/// Intensity-weighted centroid of pixels above mean + 2σ
/// (stacking.brightness_centroid); None when nothing clears the sky cut.
pub fn brightness_centroid_gray(g: &Gray) -> Option<(f64, f64)> {
    let n = g.data.len();
    if n == 0 {
        return None;
    }
    let (mut sum, mut sq) = (0.0f64, 0.0f64);
    for &v in &g.data {
        let f = v as f64;
        sum += f;
        sq += f * f;
    }
    let mean = sum / n as f64;
    let std = (sq / n as f64 - mean * mean).max(0.0).sqrt();
    let thr = mean + 2.0 * std;
    let (mut wsum, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for y in 0..g.h {
        for x in 0..g.w {
            let wgt = g.data[y * g.w + x] as f64 - thr;
            if wgt > 0.0 {
                wsum += wgt;
                sx += x as f64 * wgt;
                sy += y as f64 * wgt;
            }
        }
    }
    if wsum <= 0.0 {
        None
    } else {
        Some((sx / wsum, sy / wsum))
    }
}

/// Fixed-size square crop centred on (cx, cy), off-frame area zero-padded
/// (stacking.crop_centered): PIPP centring shifts the target to the middle
/// so every cropped frame stacks without resizing.
pub fn crop_centered_gray(g: &Gray, cx: f64, cy: f64, size: usize) -> Gray {
    let size = size.max(2);
    let mut out = Gray::new(size, size);
    let sx0 = cx.round() as i64 - size as i64 / 2;
    let sy0 = cy.round() as i64 - size as i64 / 2;
    for oy in 0..size {
        let sy = sy0 + oy as i64;
        if sy < 0 || sy >= g.h as i64 {
            continue;
        }
        let src_row = sy as usize * g.w;
        for ox in 0..size {
            let sx = sx0 + ox as i64;
            if sx < 0 || sx >= g.w as i64 {
                continue;
            }
            out.data[oy * size + ox] = g.data[src_row + sx as usize];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Overlay baking for export (RGB u8): lines, arrows, boxes, 5x7 text
// ---------------------------------------------------------------------------
const COL_INTRACK: [u8; 3] = [80, 220, 90];
const COL_CROSSTRACK: [u8; 3] = [90, 170, 255];
const COL_META: [u8; 3] = [255, 235, 120];
const COL_ANNOT: [u8; 3] = [255, 90, 90];

/// Classic 5x7 font, columns LSB = top row, ASCII 32..=126.
const FONT5X7: [[u8; 5]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00], [0x00, 0x00, 0x5F, 0x00, 0x00], [0x00, 0x07, 0x00, 0x07, 0x00], [0x14, 0x7F, 0x14, 0x7F, 0x14],
    [0x24, 0x2A, 0x7F, 0x2A, 0x12], [0x23, 0x13, 0x08, 0x64, 0x62], [0x36, 0x49, 0x56, 0x20, 0x50], [0x00, 0x08, 0x07, 0x03, 0x00],
    [0x00, 0x1C, 0x22, 0x41, 0x00], [0x00, 0x41, 0x22, 0x1C, 0x00], [0x2A, 0x1C, 0x7F, 0x1C, 0x2A], [0x08, 0x08, 0x3E, 0x08, 0x08],
    [0x00, 0x80, 0x70, 0x30, 0x00], [0x08, 0x08, 0x08, 0x08, 0x08], [0x00, 0x00, 0x60, 0x60, 0x00], [0x20, 0x10, 0x08, 0x04, 0x02],
    [0x3E, 0x51, 0x49, 0x45, 0x3E], [0x00, 0x42, 0x7F, 0x40, 0x00], [0x72, 0x49, 0x49, 0x49, 0x46], [0x21, 0x41, 0x49, 0x4D, 0x33],
    [0x18, 0x14, 0x12, 0x7F, 0x10], [0x27, 0x45, 0x45, 0x45, 0x39], [0x3C, 0x4A, 0x49, 0x49, 0x31], [0x41, 0x21, 0x11, 0x09, 0x07],
    [0x36, 0x49, 0x49, 0x49, 0x36], [0x46, 0x49, 0x49, 0x29, 0x1E], [0x00, 0x00, 0x14, 0x00, 0x00], [0x00, 0x40, 0x34, 0x00, 0x00],
    [0x00, 0x08, 0x14, 0x22, 0x41], [0x14, 0x14, 0x14, 0x14, 0x14], [0x00, 0x41, 0x22, 0x14, 0x08], [0x02, 0x01, 0x59, 0x09, 0x06],
    [0x3E, 0x41, 0x5D, 0x59, 0x4E], [0x7C, 0x12, 0x11, 0x12, 0x7C], [0x7F, 0x49, 0x49, 0x49, 0x36], [0x3E, 0x41, 0x41, 0x41, 0x22],
    [0x7F, 0x41, 0x41, 0x41, 0x3E], [0x7F, 0x49, 0x49, 0x49, 0x41], [0x7F, 0x09, 0x09, 0x09, 0x01], [0x3E, 0x41, 0x41, 0x51, 0x73],
    [0x7F, 0x08, 0x08, 0x08, 0x7F], [0x00, 0x41, 0x7F, 0x41, 0x00], [0x20, 0x40, 0x41, 0x3F, 0x01], [0x7F, 0x08, 0x14, 0x22, 0x41],
    [0x7F, 0x40, 0x40, 0x40, 0x40], [0x7F, 0x02, 0x1C, 0x02, 0x7F], [0x7F, 0x04, 0x08, 0x10, 0x7F], [0x3E, 0x41, 0x41, 0x41, 0x3E],
    [0x7F, 0x09, 0x09, 0x09, 0x06], [0x3E, 0x41, 0x51, 0x21, 0x5E], [0x7F, 0x09, 0x19, 0x29, 0x46], [0x26, 0x49, 0x49, 0x49, 0x32],
    [0x03, 0x01, 0x7F, 0x01, 0x03], [0x3F, 0x40, 0x40, 0x40, 0x3F], [0x1F, 0x20, 0x40, 0x20, 0x1F], [0x3F, 0x40, 0x38, 0x40, 0x3F],
    [0x63, 0x14, 0x08, 0x14, 0x63], [0x03, 0x04, 0x78, 0x04, 0x03], [0x61, 0x59, 0x49, 0x4D, 0x43], [0x00, 0x7F, 0x41, 0x41, 0x41],
    [0x02, 0x04, 0x08, 0x10, 0x20], [0x00, 0x41, 0x41, 0x41, 0x7F], [0x04, 0x02, 0x01, 0x02, 0x04], [0x40, 0x40, 0x40, 0x40, 0x40],
    [0x00, 0x03, 0x07, 0x08, 0x00], [0x20, 0x54, 0x54, 0x78, 0x40], [0x7F, 0x28, 0x44, 0x44, 0x38], [0x38, 0x44, 0x44, 0x44, 0x28],
    [0x38, 0x44, 0x44, 0x28, 0x7F], [0x38, 0x54, 0x54, 0x54, 0x18], [0x00, 0x08, 0x7E, 0x09, 0x02], [0x18, 0xA4, 0xA4, 0x9C, 0x78],
    [0x7F, 0x08, 0x04, 0x04, 0x78], [0x00, 0x44, 0x7D, 0x40, 0x00], [0x20, 0x40, 0x40, 0x3D, 0x00], [0x7F, 0x10, 0x28, 0x44, 0x00],
    [0x00, 0x41, 0x7F, 0x40, 0x00], [0x7C, 0x04, 0x78, 0x04, 0x78], [0x7C, 0x08, 0x04, 0x04, 0x78], [0x38, 0x44, 0x44, 0x44, 0x38],
    [0xFC, 0x18, 0x24, 0x24, 0x18], [0x18, 0x24, 0x24, 0x18, 0xFC], [0x7C, 0x08, 0x04, 0x04, 0x08], [0x48, 0x54, 0x54, 0x54, 0x24],
    [0x04, 0x04, 0x3F, 0x44, 0x24], [0x3C, 0x40, 0x40, 0x20, 0x7C], [0x1C, 0x20, 0x40, 0x20, 0x1C], [0x3C, 0x40, 0x30, 0x40, 0x3C],
    [0x44, 0x28, 0x10, 0x28, 0x44], [0x4C, 0x90, 0x90, 0x90, 0x7C], [0x44, 0x64, 0x54, 0x4C, 0x44], [0x00, 0x08, 0x36, 0x41, 0x00],
    [0x00, 0x00, 0x77, 0x00, 0x00], [0x00, 0x41, 0x36, 0x08, 0x00], [0x02, 0x01, 0x02, 0x04, 0x02],
];

pub struct Canvas<'a> {
    pub rgb: &'a mut [u8],
    pub w: usize,
    pub h: usize,
}

impl<'a> Canvas<'a> {
    #[inline]
    fn put(&mut self, x: i64, y: i64, col: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 3;
        self.rgb[i..i + 3].copy_from_slice(&col);
    }

    fn dot(&mut self, x: i64, y: i64, col: [u8; 3], th: i64) {
        let r = th / 2;
        for dy in -r..=r {
            for dx in -r..=r {
                self.put(x + dx, y + dy, col);
            }
        }
    }

    pub fn line(&mut self, p0: [f64; 2], p1: [f64; 2], col: [u8; 3], th: i64) {
        let (x0, y0, x1, y1) = (p0[0], p0[1], p1[0], p1[1]);
        let n = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as i64).max(1);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            self.dot((x0 + (x1 - x0) * t).round() as i64, (y0 + (y1 - y0) * t).round() as i64, col, th);
        }
    }

    pub fn arrow(&mut self, p0: [f64; 2], p1: [f64; 2], col: [u8; 3], th: i64, tip_frac: f64) {
        self.line(p0, p1, col, th);
        let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len = dx.hypot(dy);
        if len < 1e-6 {
            return;
        }
        let tip = len * tip_frac;
        let ang = dy.atan2(dx);
        for s in [-1.0, 1.0] {
            let a = ang + s * std::f64::consts::PI / 6.0 + std::f64::consts::PI;
            self.line(p1, [p1[0] + tip * a.cos(), p1[1] + tip * a.sin()], col, th);
        }
    }

    pub fn rect(&mut self, p0: [f64; 2], p1: [f64; 2], col: [u8; 3], th: i64) {
        self.line(p0, [p1[0], p0[1]], col, th);
        self.line([p1[0], p0[1]], p1, col, th);
        self.line(p1, [p0[0], p1[1]], col, th);
        self.line([p0[0], p1[1]], p0, col, th);
    }

    /// Bitmap text with the top-left corner at (x, y), `scale` px per font px.
    pub fn text(&mut self, x: i64, y: i64, text: &str, col: [u8; 3], scale: i64) {
        let scale = scale.max(1);
        let mut cx = x;
        for ch in text.chars() {
            let code = ch as u32;
            let glyph = if (32..=126).contains(&code) { FONT5X7[(code - 32) as usize] } else { FONT5X7[('?' as u32 - 32) as usize] };
            for (gx, colbits) in glyph.iter().enumerate() {
                for gy in 0..8 {
                    if colbits >> gy & 1 == 1 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.put(cx + gx as i64 * scale + sx, y + gy as i64 * scale + sy, col);
                            }
                        }
                    }
                }
            }
            cx += 6 * scale;
        }
    }
}

/// Bake vectors / meta lines / annotations into an RGB frame (export path).
pub fn bake_overlays(rgb: &mut [u8], w: usize, h: usize, vectors: Option<&TrackVectors>, meta: &[String], annots: &[Annotation], cam: usize) {
    let mut c = Canvas { rgb, w, h };
    let th = ((h as f64 / 720.0).round() as i64).max(1);
    let scale = ((h as f64 / 540.0).round() as i64).max(1);
    let line_h = 9 * scale;
    if let Some(v) = vectors {
        c.arrow(v.anchor, v.intrack, COL_INTRACK, th, 0.18);
        c.line(v.cross_p, v.cross_n, COL_CROSSTRACK, th);
        c.text(v.intrack[0] as i64 + 3, v.intrack[1] as i64, "IN", COL_INTRACK, scale);
        c.text(v.cross_p[0] as i64 + 3, v.cross_p[1] as i64, "CROSS", COL_CROSSTRACK, scale);
    }
    let mut y = 6;
    for line in meta {
        c.text(6, y, line, COL_META, scale);
        y += line_h;
    }
    let (wf, hf) = (w as f64, h as f64);
    for a in annots.iter().filter(|a| a.cam() == cam) {
        match a {
            Annotation::Text { x, y, text, .. } => c.text((x * wf) as i64, (y * hf) as i64, text, COL_ANNOT, scale),
            Annotation::Arrow { x0, y0, x1, y1, .. } => c.arrow([x0 * wf, y0 * hf], [x1 * wf, y1 * hf], COL_ANNOT, th + 1, 0.2),
            Annotation::Box { x0, y0, x1, y1, .. } => c.rect([x0 * wf, y0 * hf], [x1 * wf, y1 * hf], COL_ANNOT, th + 1),
        }
    }
}

pub fn build_meta_lines(run: &RunInfo, cam_index: usize, t: f64, frame_idx: usize, n_frames: usize, gamma: f64, stabilize_on: bool) -> Vec<String> {
    let mut lines = vec![format!("Cam{}  {} UTC  [{}/{}]", cam_index + 1, fmt_utc(t), frame_idx + 1, n_frames)];
    if run.trajectory.valid() {
        if let Some(s) = run.trajectory.interp(t) {
            lines.push(format!("Az {:.2}  El {:.2}  Rng {:.0} km", s.az, s.el, s.dist));
        }
    }
    let mut flags = Vec::new();
    if gamma != 1.0 {
        flags.push(format!("gamma {gamma:.2}"));
    }
    if stabilize_on {
        flags.push("STAB".to_string());
    }
    if !flags.is_empty() {
        lines.push(flags.join("  "));
    }
    lines
}

// ---------------------------------------------------------------------------
// Camera workers
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub struct ProcParams {
    pub gamma: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub stabilize: bool,
    pub stab: StabSettings,
    pub reference_idx: usize,
    pub sharpen: SharpenSettings,
}

impl Default for ProcParams {
    fn default() -> Self {
        ProcParams { gamma: 1.0, brightness: 0.0, contrast: 1.0, stabilize: false, stab: StabSettings::default(), reference_idx: 0, sharpen: SharpenSettings::default() }
    }
}

#[derive(Clone, Debug)]
struct FrameRequest {
    frame_idx: usize,
    pane_w: usize,
    params: ProcParams,
    gen: u64,
    /// Playing: prefer the pre-decoded proxy frames (smooth) over a fresh
    /// full-resolution decode (sharp but ~100 ms per 6 MP BMP).
    playing: bool,
}

/// Whole-run reduced-resolution frames decoded once in the background
/// (rayon), so playback never waits on a 19 MB BMP decode. Capped at
/// PROXY_MAX_W px wide and ~PROXY_BUDGET bytes per camera.
pub struct Proxy {
    pub reduce: std::sync::atomic::AtomicUsize,
    pub frames: std::sync::Mutex<Vec<Option<Arc<Gray>>>>,
    pub done: std::sync::atomic::AtomicUsize,
    pub total: usize,
}

const PROXY_MAX_W: usize = 1024;
const PROXY_BUDGET: usize = 160 << 20;

impl Proxy {
    fn new(total: usize) -> Self {
        Proxy {
            reduce: std::sync::atomic::AtomicUsize::new(0),
            frames: std::sync::Mutex::new(vec![None; total]),
            done: std::sync::atomic::AtomicUsize::new(0),
            total,
        }
    }
    fn get(&self, idx: usize) -> Option<Arc<Gray>> {
        self.frames.lock().unwrap().get(idx).and_then(|f| f.clone())
    }
    pub fn reduce(&self) -> usize {
        self.reduce.load(Ordering::Relaxed)
    }
    pub fn progress(&self) -> (usize, usize) {
        (self.done.load(Ordering::Relaxed), self.total)
    }
}

/// Proxy reduction for a run: power of two so width <= PROXY_MAX_W and the
/// whole run fits the byte budget.
fn proxy_reduce_for(w: usize, h: usize, n: usize) -> usize {
    let mut r = 1;
    while w / r > PROXY_MAX_W || (w / r) * (h / r) * n > PROXY_BUDGET {
        r *= 2;
        if r >= 64 {
            break;
        }
    }
    r
}

fn spawn_proxy_builder(cam: Arc<CamFrames>, proxy: Arc<Proxy>, stop: Arc<AtomicBool>, ctx: Option<egui::Context>) {
    std::thread::Builder::new()
        .name(format!("replay-proxy{}", cam.cam_index + 1))
        .spawn(move || {
            use rayon::prelude::*;
            let frames = &cam.frames;
            if frames.is_empty() {
                return;
            }
            // First frame sets the geometry.
            let Ok(g0) = load_gray(&frames[0].path) else { return };
            let reduce = proxy_reduce_for(g0.w, g0.h, frames.len());
            proxy.reduce.store(reduce, Ordering::Relaxed);
            let store = |i: usize, g: Gray| {
                let r = if reduce > 1 { g.downsample(reduce) } else { g };
                proxy.frames.lock().unwrap()[i] = Some(Arc::new(r));
                proxy.done.fetch_add(1, Ordering::Relaxed);
            };
            store(0, g0);
            let pool = rayon::ThreadPoolBuilder::new().num_threads((num_cpus_hint() / 2).max(2)).build();
            let work = |i: usize| {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Ok(g) = load_gray(&frames[i].path) {
                    store(i, g);
                }
                if i % 8 == 0 {
                    if let Some(c) = &ctx {
                        c.request_repaint();
                    }
                }
            };
            match pool {
                Ok(p) => p.install(|| (1..frames.len()).into_par_iter().for_each(work)),
                Err(_) => (1..frames.len()).into_par_iter().for_each(work),
            }
            if let Some(c) = &ctx {
                c.request_repaint();
            }
        })
        .ok();
}

fn num_cpus_hint() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

pub struct ProcessedFrame {
    pub slot: usize,
    pub frame_idx: usize,
    pub t: f64,
    pub w: usize,
    pub h: usize,
    pub full_w: usize,
    pub full_h: usize,
    pub reduce: usize,
    pub gray: Vec<u8>,
    pub stab: Option<StabInfo>,
    pub gen: u64,
    pub seq: u64,
    pub proc_ms: f64,
    pub error: Option<String>,
}

struct CamWorker {
    tx: Sender<FrameRequest>,
    out: Arc<ArcSwapOption<ProcessedFrame>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    proxy: Arc<Proxy>,
}

impl Drop for CamWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Dropping tx disconnects the channel; the worker exits on its next recv.
    }
}

struct DecodeCache {
    map: HashMap<(usize, usize), Arc<Gray>>,
    order: VecDeque<(usize, usize)>,
    full: HashMap<usize, Arc<Gray>>,
    full_order: VecDeque<usize>,
}

impl DecodeCache {
    fn new() -> Self {
        DecodeCache { map: HashMap::new(), order: VecDeque::new(), full: HashMap::new(), full_order: VecDeque::new() }
    }

    fn get(&mut self, frames: &[FrameRef], idx: usize, reduce: usize, proxy: &Proxy) -> Result<Arc<Gray>, String> {
        let key = (idx, reduce);
        if let Some(g) = self.map.get(&key) {
            return Ok(g.clone());
        }
        // Proxy hit: derive from the pre-decoded reduced frame (cheap).
        let pr = proxy.reduce();
        if pr > 0 && reduce >= pr && reduce % pr == 0 {
            if let Some(p) = proxy.get(idx) {
                let g = if reduce > pr { Arc::new(p.downsample(reduce / pr)) } else { p };
                self.map.insert(key, g.clone());
                self.order.push_back(key);
                while self.order.len() > DECODE_CACHE {
                    if let Some(k) = self.order.pop_front() {
                        self.map.remove(&k);
                    }
                }
                return Ok(g);
            }
        }
        let full = match self.full.get(&idx) {
            Some(f) => f.clone(),
            None => {
                let g = Arc::new(load_gray(&frames[idx].path)?);
                // Keep a handful of full-res decodes (stabilize reference, prefetch).
                self.full.insert(idx, g.clone());
                self.full_order.push_back(idx);
                while self.full_order.len() > 6 {
                    if let Some(k) = self.full_order.pop_front() {
                        self.full.remove(&k);
                    }
                }
                g
            }
        };
        let reduced = if reduce <= 1 { full } else { Arc::new(full.downsample(reduce)) };
        self.map.insert(key, reduced.clone());
        self.order.push_back(key);
        while self.order.len() > DECODE_CACHE {
            if let Some(k) = self.order.pop_front() {
                self.map.remove(&k);
            }
        }
        Ok(reduced)
    }
}

struct StabCache {
    key: (usize, usize, [u8; 256], StabSettings),
    stab: Stabilizer,
}

fn spawn_cam_worker(slot: usize, cam: Arc<CamFrames>, ctx: Option<egui::Context>) -> CamWorker {
    let (tx, rx) = crossbeam_channel::unbounded::<FrameRequest>();
    let out: Arc<ArcSwapOption<ProcessedFrame>> = Arc::new(ArcSwapOption::from(None));
    let stop = Arc::new(AtomicBool::new(false));
    let proxy = Arc::new(Proxy::new(cam.frames.len()));
    spawn_proxy_builder(cam.clone(), proxy.clone(), stop.clone(), ctx.clone());
    let out2 = out.clone();
    let stop2 = stop.clone();
    let proxy2 = proxy.clone();
    let handle = std::thread::Builder::new()
        .name(format!("replay-cam{}", cam.cam_index + 1))
        .spawn(move || cam_worker_loop(slot, cam, rx, out2, stop2, ctx, proxy2))
        .ok();
    CamWorker { tx, out, stop, handle, proxy }
}

fn cam_worker_loop(slot: usize, cam: Arc<CamFrames>, rx: Receiver<FrameRequest>, out: Arc<ArcSwapOption<ProcessedFrame>>, stop: Arc<AtomicBool>, ctx: Option<egui::Context>, proxy: Arc<Proxy>) {
    let frames = &cam.frames;
    let mut cache = DecodeCache::new();
    let mut stab_cache: Option<StabCache> = None;
    let mut full_shape: Option<(usize, usize)> = None;
    let mut last: Option<FrameRequest> = None;
    let mut seq: u64 = 0;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let mut req = match rx.recv_timeout(Duration::from_millis(150)) {
            Ok(r) => r,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Idle: prefetch the next few frames at the last reduce.
                if let (Some(l), Some((fw, _))) = (&last, full_shape) {
                    let reduce = reduce_for(fw, l.pane_w);
                    for d in 1..=PREFETCH {
                        if !rx.is_empty() || stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let j = l.frame_idx + d;
                        if j < frames.len() {
                            let _ = cache.get(frames, j, reduce, &proxy);
                        }
                    }
                }
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        // Coalesce: only the newest request matters.
        while let Ok(r) = rx.try_recv() {
            req = r;
        }
        if frames.is_empty() {
            continue;
        }
        let idx = req.frame_idx.min(frames.len() - 1);
        let t0 = Instant::now();
        // Reduce factor needs the full frame width: decode full first time.
        let (fw, fh) = match full_shape {
            Some(s) => s,
            None => match cache.get(frames, idx, 1, &proxy) {
                Ok(g) => {
                    full_shape = Some((g.w, g.h));
                    (g.w, g.h)
                }
                Err(e) => {
                    seq += 1;
                    out.store(Some(Arc::new(ProcessedFrame { slot, frame_idx: idx, t: frames[idx].t, w: 0, h: 0, full_w: 0, full_h: 0, reduce: 1, gray: Vec::new(), stab: None, gen: req.gen, seq, proc_ms: 0.0, error: Some(e) })));
                    if let Some(c) = &ctx {
                        c.request_repaint();
                    }
                    last = Some(req);
                    continue;
                }
            },
        };
        let mut reduce = reduce_for(fw, req.pane_w.max(1));
        let pr = proxy.reduce();
        if req.playing && pr > 0 && proxy.get(idx).is_some() {
            // Smooth playback: use the proxy resolution (>= pane/2 px) rather
            // than a full decode per frame; a pause re-requests full quality.
            reduce = reduce.max(pr);
        }
        let raw = match cache.get(frames, idx, reduce, &proxy) {
            Ok(g) => g,
            Err(e) => {
                seq += 1;
                out.store(Some(Arc::new(ProcessedFrame { slot, frame_idx: idx, t: frames[idx].t, w: 0, h: 0, full_w: fw, full_h: fh, reduce, gray: Vec::new(), stab: None, gen: req.gen, seq, proc_ms: 0.0, error: Some(e) })));
                if let Some(c) = &ctx {
                    c.request_repaint();
                }
                last = Some(req);
                continue;
            }
        };
        let p = &req.params;
        let lut = build_lut(p.gamma, p.brightness, p.contrast);
        let mut proc = if is_default_params(p.gamma, p.brightness, p.contrast) { (*raw).clone() } else { raw.apply_lut(&lut) };
        let mut stab_info = None;
        if p.stabilize {
            let ref_idx = p.reference_idx.min(frames.len() - 1);
            let key = (ref_idx, reduce, lut, p.stab);
            let fresh = stab_cache.as_ref().map_or(true, |c| c.key != key);
            if fresh {
                if let Ok(ref_raw) = cache.get(frames, ref_idx, reduce, &proxy) {
                    let ref_proc = if is_default_params(p.gamma, p.brightness, p.contrast) { (*ref_raw).clone() } else { ref_raw.apply_lut(&lut) };
                    stab_cache = Some(StabCache { key, stab: Stabilizer::new(&ref_proc, p.stab) });
                }
            }
            if let Some(sc) = &stab_cache {
                if idx == ref_idx {
                    stab_info = Some(StabInfo { ok: true, inliers: 0, reason: None, m: Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) });
                } else {
                    let (warped, info) = sc.stab.stabilize(&proc);
                    proc = warped;
                    stab_info = Some(info);
                }
            }
        } else {
            stab_cache = None;
        }
        if p.sharpen.on {
            proc = finish_gray(&proc, &p.sharpen);
        }
        seq += 1;
        out.store(Some(Arc::new(ProcessedFrame {
            slot,
            frame_idx: idx,
            t: frames[idx].t,
            w: proc.w,
            h: proc.h,
            full_w: fw,
            full_h: fh,
            reduce,
            gray: proc.data,
            stab: stab_info,
            gen: req.gen,
            seq,
            proc_ms: t0.elapsed().as_secs_f64() * 1000.0,
            error: None,
        })));
        if let Some(c) = &ctx {
            c.request_repaint();
        }
        // Warm the next few frames while the queue is empty (a real request
        // arriving between decodes abandons the prefetch).
        for d in 1..=PREFETCH {
            if !rx.is_empty() || stop.load(Ordering::Relaxed) {
                break;
            }
            let j = idx + d;
            if j < frames.len() {
                let _ = cache.get(frames, j, reduce, &proxy);
            }
        }
        last = Some(req);
    }
}

// ---------------------------------------------------------------------------
// Jobs: export / stack
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default)]
pub struct JobStatus {
    pub kind: String,
    pub progress: f32,
    pub done: bool,
    pub message: String,
    pub error: Option<String>,
    pub out_path: Option<PathBuf>,
}

pub struct JobHandle {
    pub status: Arc<ArcSwap<JobStatus>>,
    pub cancel: Arc<AtomicBool>,
    started: Instant,
}

impl JobHandle {
    fn new(kind: &str) -> Self {
        JobHandle {
            status: Arc::new(ArcSwap::from_pointee(JobStatus { kind: kind.to_string(), message: "starting…".into(), ..Default::default() })),
            cancel: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
        }
    }

    fn set(status: &Arc<ArcSwap<JobStatus>>, f: impl FnOnce(&mut JobStatus)) {
        let mut s = (**status.load()).clone();
        f(&mut s);
        status.store(Arc::new(s));
    }
}

#[derive(Clone, Debug)]
pub struct ExportSpec {
    pub run: Arc<RunInfo>,
    pub slot: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub out_path: PathBuf,
    pub params: ProcParams,
    pub overlays: bool,
    pub annotations: Vec<Annotation>,
    pub fps: Option<f64>,
    /// Source-pixel crop [x, y, w, h] baked into the export (pane zoom view).
    pub crop: Option<[usize; 4]>,
}

fn frames_in_range(run: &RunInfo, slot: usize, t_start: f64, t_end: f64) -> Vec<(usize, FrameRef)> {
    let (lo, hi) = (t_start.min(t_end), t_start.max(t_end));
    run.cams.get(slot).map(|c| c.frames.iter().cloned().enumerate().filter(|(_, f)| f.t >= lo - 1e-9 && f.t <= hi + 1e-9).collect()).unwrap_or_default()
}

/// Median capture interval -> fps (1..60), 5 when undeterminable.
pub fn infer_fps(times: &[f64]) -> f64 {
    if times.len() < 2 {
        return 5.0;
    }
    let mut d: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if d.len() % 2 == 1 { d[d.len() / 2] } else { (d[d.len() / 2 - 1] + d[d.len() / 2]) / 2.0 };
    let fps = if med > 1e-3 { 1.0 / med } else { 5.0 };
    fps.clamp(1.0, 60.0)
}

/// Export an MP4 of the frame range with the full replay pipeline baked in.
/// Runs synchronously — call from a worker thread.
pub fn run_export(spec: &ExportSpec, status: &Arc<ArcSwap<JobStatus>>, cancel: &AtomicBool) -> Result<PathBuf, String> {
    #[cfg(not(feature = "mp4-export"))]
    {
        let _ = (spec, status, cancel);
        Err("MP4 export is not compiled in (build skytracker-app with the mp4-export feature)".into())
    }
    #[cfg(feature = "mp4-export")]
    {
        use skytracker_imaging::video::Mp4Encoder;
        let run = &spec.run;
        let frames = frames_in_range(run, spec.slot, spec.t_start, spec.t_end);
        if frames.is_empty() {
            return Err("No frames in the selected range".into());
        }
        let cam_index = run.cams[spec.slot].cam_index;
        let p = &spec.params;
        let lut = build_lut(p.gamma, p.brightness, p.contrast);
        let first = load_gray(&frames[0].1.path)?;
        // Optional view crop (source px), clamped to the frame; I420 needs even dims.
        let crop = spec.crop.map(|[x, y, cw, ch]| {
            let x = x.min(first.w.saturating_sub(2));
            let y = y.min(first.h.saturating_sub(2));
            let cw = cw.min(first.w - x) & !1;
            let ch = ch.min(first.h - y) & !1;
            [x, y, cw, ch]
        });
        let (w, h, ox, oy) = match crop {
            Some([x, y, cw, ch]) => (cw, ch, x, y),
            None => (first.w & !1, first.h & !1, 0, 0),
        };
        if w < 2 || h < 2 {
            return Err("frame too small".into());
        }
        // Annotations are normalized to the full frame: remap into crop space.
        let annots: Vec<Annotation> = match crop {
            Some([cx, cy, cw, ch]) => {
                let (fw, fh) = (first.w as f64, first.h as f64);
                let rx = |v: f64| (v * fw - cx as f64) / cw as f64;
                let ry = |v: f64| (v * fh - cy as f64) / ch as f64;
                spec.annotations
                    .iter()
                    .map(|a| match a {
                        Annotation::Text { cam, x, y, text } => Annotation::Text { cam: *cam, x: rx(*x), y: ry(*y), text: text.clone() },
                        Annotation::Arrow { cam, x0, y0, x1, y1 } => Annotation::Arrow { cam: *cam, x0: rx(*x0), y0: ry(*y0), x1: rx(*x1), y1: ry(*y1) },
                        Annotation::Box { cam, x0, y0, x1, y1 } => Annotation::Box { cam: *cam, x0: rx(*x0), y0: ry(*y0), x1: rx(*x1), y1: ry(*y1) },
                    })
                    .collect()
            }
            None => spec.annotations.clone(),
        };
        let fps = spec.fps.unwrap_or_else(|| infer_fps(&frames.iter().map(|f| f.1.t).collect::<Vec<_>>()));
        if let Some(dir) = spec.out_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let mut enc = Mp4Encoder::create(&spec.out_path, w, h, fps).map_err(|e| e.to_string())?;
        let mut stab: Option<Stabilizer> = None;
        let total = frames.len();
        let mut written = 0usize;
        for (i, (orig_idx, f)) in frames.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let raw = match load_gray(&f.path) {
                Ok(g) => g,
                Err(_) => continue,
            };
            let mut proc = if is_default_params(p.gamma, p.brightness, p.contrast) { raw } else { raw.apply_lut(&lut) };
            if p.stabilize {
                match &stab {
                    None => stab = Some(Stabilizer::new(&proc, p.stab)), // fresh anchor at range start
                    Some(s) => {
                        let (warped, _) = s.stabilize(&proc);
                        proc = warped;
                    }
                }
            }
            if p.sharpen.on {
                proc = finish_gray(&proc, &p.sharpen);
            }
            // Crop to the view rect / even dims.
            let mut rgb = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    let v = proc.data.get((y + oy) * proc.w + x + ox).copied().unwrap_or(0);
                    rgb.extend_from_slice(&[v, v, v]);
                }
            }
            if spec.overlays {
                let vectors = compute_track_vectors(&run.trajectory, f.t, w as f64, h as f64);
                let meta = build_meta_lines(run, cam_index, f.t, *orig_idx, run.cams[spec.slot].frames.len(), p.gamma, p.stabilize);
                bake_overlays(&mut rgb, w, h, vectors.as_ref(), &meta, &annots, spec.slot);
            }
            enc.write_rgb(&rgb).map_err(|e| e.to_string())?;
            written += 1;
            JobHandle::set(status, |s| {
                s.progress = (i + 1) as f32 / total as f32;
                s.message = format!("Exporting… {}/{} frames", i + 1, total);
            });
        }
        enc.finish().map_err(|e| e.to_string())?;
        if written == 0 {
            return Err("no frame could be decoded".into());
        }
        Ok(spec.out_path.clone())
    }
}

#[derive(Clone, Debug)]
pub struct StackSpec {
    pub run: Arc<RunInfo>,
    pub slot: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub keep_n: usize,
    pub stab: StabSettings,
    pub out_base: PathBuf,
    /// PIPP-style centring (stacking.pipp_center_stack): recentre each kept
    /// frame on the track target / brightest blob, crop to center_size².
    pub centered: bool,
    pub center_size: usize,
}

/// Lucky-imaging stack (stacking.stack_run, gray): grade frames in range by
/// Laplacian sharpness, keep the best N, align to the sharpest, coverage-
/// weighted mean; writes a 16-bit linear master PNG + finished 8-bit PNG.
pub fn run_stack(spec: &StackSpec, status: &Arc<ArcSwap<JobStatus>>, cancel: &AtomicBool) -> Result<PathBuf, String> {
    let frames = frames_in_range(&spec.run, spec.slot, spec.t_start, spec.t_end);
    if frames.is_empty() {
        return Err("No frames in the selected range".into());
    }
    let total = frames.len();
    // Grade: Laplacian sharpness + garbage pre-cull content score.
    let mut graded: Vec<(usize, f64, f64)> = Vec::with_capacity(total);
    for (k, (i, f)) in frames.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if let Ok(g) = load_gray(&f.path) {
            let score = metrics::sharpness_laplacian(&g.to_f32());
            graded.push((*i, score, content_score_gray(&g)));
        }
        JobHandle::set(status, |s| {
            s.progress = 0.4 * (k + 1) as f32 / total as f32;
            s.message = format!("Grading… {}/{}", k + 1, total);
        });
    }
    if graded.is_empty() {
        return Err("no frame could be decoded".into());
    }
    // Pre-cull empty/garbage frames (stacking.prefilter_garbage, simple form):
    // drop content scores below 0.25x the median — sharpness is fooled by
    // noise, so this runs before the ranking; the median guarantees survivors.
    let mut cvals: Vec<f64> = graded.iter().map(|g| g.2).collect();
    cvals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = 0.25 * cvals[cvals.len() / 2];
    let n_before = graded.len();
    graded.retain(|g| g.2 >= floor);
    let n_preculled = n_before - graded.len();
    if graded.is_empty() {
        return Err("all frames pre-culled as empty".into());
    }
    graded.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep: Vec<usize> = graded.iter().take(spec.keep_n.max(1)).map(|g| g.0).collect();
    let cam = &spec.run.cams[spec.slot];
    // Decode kept set (reference = sharpest, index 0); optional PIPP centring.
    let mut imgs: Vec<Gray> = Vec::with_capacity(keep.len());
    for (k, &i) in keep.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let mut g = load_gray(&cam.frames[i].path)?;
        if spec.centered {
            // Track target when the trajectory carries pixel coords, else the
            // brightest blob, else the geometric centre (blank frame guard).
            let traj_px = spec.run.trajectory.interp(cam.frames[i].t).and_then(|s| {
                (s.px.is_finite() && s.py.is_finite() && s.px >= 0.0 && s.py >= 0.0 && s.px < g.w as f64 && s.py < g.h as f64).then_some((s.px, s.py))
            });
            let (cx, cy) = traj_px
                .or_else(|| brightness_centroid_gray(&g))
                .unwrap_or((g.w as f64 / 2.0, g.h as f64 / 2.0));
            g = crop_centered_gray(&g, cx, cy, spec.center_size.max(16));
        }
        imgs.push(g);
        JobHandle::set(status, |s| {
            s.progress = 0.4 + 0.2 * (k + 1) as f32 / keep.len() as f32;
            s.message = format!("Decoding best {}…", keep.len());
        });
    }
    JobHandle::set(status, |s| {
        s.progress = 0.6;
        s.message = format!("Aligning + stacking {}…", imgs.len());
    });
    let (mean, n_stacked, n_rejected) = stack_gray(&imgs, 0, true, spec.stab).ok_or("nothing to stack")?;
    let (w, h) = (imgs[0].w, imgs[0].h);
    if let Some(dir) = spec.out_base.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    // 16-bit linear master.
    let master_path = PathBuf::from(format!("{}.png", spec.out_base.display()));
    let m16: Vec<u16> = mean.iter().map(|v| (v * 257.0).round().clamp(0.0, 65535.0) as u16).collect();
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w as u32, h as u32, m16)
        .ok_or("master buffer")?
        .save(&master_path)
        .map_err(|e| e.to_string())?;
    // Finished (sharpened + stretched) 8-bit.
    let lin = ImageF32::from_vec(mean.iter().map(|v| (*v / 255.0) as f32).collect(), w, h);
    let fin = finish_f32(&lin, true);
    let fin8 = Gray::from_f32(&fin, 255.0);
    let final_path = PathBuf::from(format!("{}_final.png", spec.out_base.display()));
    image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(w as u32, h as u32, fin8.data)
        .ok_or("final buffer")?
        .save(&final_path)
        .map_err(|e| e.to_string())?;
    JobHandle::set(status, |s| {
        s.progress = 1.0;
        let cull = if n_preculled > 0 { format!(", {n_preculled} pre-culled") } else { String::new() };
        s.message = format!("Stacked {n_stacked}/{} ({n_rejected} rejected{cull}) -> {}", keep.len(), final_path.display());
    });
    Ok(final_path)
}

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamAdjust {
    pub gamma: f64,
    pub brightness: f64,
    pub contrast: f64,
}

impl Default for CamAdjust {
    fn default() -> Self {
        CamAdjust { gamma: 1.0, brightness: 0.0, contrast: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnotMode {
    None,
    Text,
    Arrow,
    Box,
}

struct LoadedRun {
    run: Arc<RunInfo>,
    lib_index: usize,
    workers: Vec<CamWorker>,
    textures: Vec<Option<egui::TextureHandle>>,
    tex_seq: Vec<u64>,
    last_frame: Vec<Option<Arc<ProcessedFrame>>>,
    last_sent: Vec<Option<(usize, usize, ProcParams, bool)>>,
    pane_w: Vec<usize>,
    reference_idx: Vec<usize>,
    adjust: Vec<CamAdjust>,
    annotations: Vec<Annotation>,
    sidecar: Sidecar,
    t0: f64,
    t1: f64,
    /// Frames live in OneDrive as online-only placeholders: the first read
    /// of each frame is a cloud download (seconds per 18 MB BMP).
    cloud_only: bool,
}

/// Windows: FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS |
/// RECALL_ON_OPEN mark OneDrive Files-On-Demand placeholders.
fn is_cloud_placeholder(path: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if let Ok(m) = std::fs::metadata(path) {
            let a = m.file_attributes();
            return a & (0x1000 | 0x40000 | 0x400000) != 0;
        }
    }
    let _ = path;
    false
}

fn make_params(stabilize: bool, stab: StabSettings, sharpen: SharpenSettings, l: &LoadedRun, slot: usize) -> ProcParams {
    let a = l.adjust.get(slot).copied().unwrap_or_default();
    ProcParams {
        gamma: a.gamma,
        brightness: a.brightness,
        contrast: a.contrast,
        stabilize,
        stab,
        reference_idx: l.reference_idx.get(slot).copied().unwrap_or(0),
        sharpen,
    }
}

pub struct ReplayState {
    // library
    library: Option<Arc<Vec<RunInfo>>>,
    scan_rx: Option<Receiver<Vec<RunInfo>>>,
    scanned_dir: Option<PathBuf>,
    pub filter: String,
    pub favorites_only: bool,
    /// Group the run list under per-target collapsible headers (off = flat list).
    pub group_by_target: bool,
    // loaded run
    loaded: Option<LoadedRun>,
    // playback
    pub t: f64,
    pub playing: bool,
    pub speed_idx: usize,
    pub looping: bool,
    /// Playback is holding the playhead because the decoders have not
    /// caught up yet (proxy cache still building / cloud download).
    pub buffering: bool,
    pub in_marker: f64,
    pub out_marker: f64,
    last_tick: Instant,
    // processing
    pub active_cam: usize,
    pub stabilize: bool,
    pub stab: StabSettings,
    pub sharpen: SharpenSettings,
    pub overlays: bool,
    pub stack_n: usize,
    pub stack_centered: bool,
    pub stack_center_size: usize,
    // per-pane pixel zoom: a display transform (mousewheel about the cursor,
    // drag pans, right-click / chip resets); overlays ride the same transform
    pane_zoom: Vec<f32>,
    pane_pan: Vec<Vec2>,
    /// Visible source-pixel rect [x, y, w, h] per pane while zoomed (crop export).
    pane_view: Vec<Option<[usize; 4]>>,
    pub export_crop: bool,
    // annotation
    pub annot_mode: AnnotMode,
    annot_text: String,
    annot_pending: Option<(usize, f64, f64)>,
    annot_draft: Option<(usize, f64, f64, f64, f64)>,
    // jobs
    job: Option<JobHandle>,
    last_job_msg: String,
    last_job_ok: bool,
    req_gen: u64,
    rename_buf: String,
    notes_buf: String,
    tags_buf: String,
    /// Armed by the first "delete run" click; disarms after 3 s.
    delete_confirm: Option<Instant>,
}

impl Default for ReplayState {
    fn default() -> Self {
        ReplayState {
            library: None,
            scan_rx: None,
            scanned_dir: None,
            filter: String::new(),
            favorites_only: false,
            group_by_target: true,
            loaded: None,
            t: 0.0,
            playing: false,
            speed_idx: 2,
            looping: true,
            buffering: false,
            in_marker: 0.0,
            out_marker: 0.0,
            last_tick: Instant::now(),
            active_cam: 0,
            stabilize: false,
            stab: StabSettings::default(),
            sharpen: SharpenSettings::default(),
            overlays: true,
            stack_n: 20,
            stack_centered: false,
            stack_center_size: 512,
            pane_zoom: Vec::new(),
            pane_pan: Vec::new(),
            pane_view: Vec::new(),
            export_crop: false,
            annot_mode: AnnotMode::None,
            annot_text: String::new(),
            annot_pending: None,
            annot_draft: None,
            job: None,
            last_job_msg: String::new(),
            last_job_ok: true,
            req_gen: 0,
            rename_buf: String::new(),
            notes_buf: String::new(),
            tags_buf: String::new(),
            delete_confirm: None,
        }
    }
}

impl ReplayState {
    pub fn speed(&self) -> f64 {
        SPEEDS[self.speed_idx.min(SPEEDS.len() - 1)]
    }

    /// Library size after a scan (None until the scan has landed).
    pub fn library_len(&self) -> Option<usize> {
        self.library.as_ref().map(|l| l.len())
    }

    /// Index of the first library run whose folder name contains `needle`.
    pub fn find_run(&self, needle: &str) -> Option<usize> {
        let lib = self.library.as_ref()?;
        lib.iter().position(|r| r.path.to_string_lossy().contains(needle) || r.display_name().contains(needle))
    }

    /// One-line status for the headless benchmark: playhead, per-camera
    /// displayed frame index / process time, proxy cache progress.
    pub fn debug_status(&self) -> String {
        let Some(l) = &self.loaded else { return "no run".into() };
        let mut s = format!("t={:7.2}s play={}", self.t - l.t0, self.playing);
        for (slot, w) in l.workers.iter().enumerate() {
            let (pd, pt) = w.proxy.progress();
            let shown = l.last_frame[slot].as_ref().map(|f| (f.frame_idx, f.reduce, f.proc_ms)).unwrap_or((0, 0, 0.0));
            let want = l.run.frame_index_at(slot, self.t).unwrap_or(0);
            s.push_str(&format!("  cam{slot}: shown {} want {} /{} {:.0}ms proxy {pd}/{pt}", shown.0, want, shown.1, shown.2));
        }
        s
    }

    pub fn shown_frame_index(&self, slot: usize) -> Option<usize> {
        self.loaded.as_ref()?.last_frame.get(slot)?.as_ref().map(|f| f.frame_idx)
    }

    pub fn loaded_run(&self) -> Option<&Arc<RunInfo>> {
        self.loaded.as_ref().map(|l| &l.run)
    }

    pub fn job_running(&self) -> bool {
        self.job.as_ref().map_or(false, |j| !j.status.load().done)
    }

    /// Start (or restart) a background scan of `dir`.
    pub fn rescan(&mut self, dir: &Path) {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let dir = dir.to_path_buf();
        self.scanned_dir = Some(dir.clone());
        self.scan_rx = Some(rx);
        let _ = std::thread::Builder::new().name("replay-scan".into()).spawn(move || {
            let runs = scan_library(&dir);
            let _ = tx.send(runs);
        });
    }

    fn poll_scan(&mut self) {
        let got = match &self.scan_rx {
            Some(rx) => rx.try_recv().ok(),
            None => None,
        };
        if let Some(runs) = got {
            self.library = Some(Arc::new(runs));
            self.scan_rx = None;
        }
    }

    pub fn load_run(&mut self, lib_index: usize, ctx: Option<egui::Context>) {
        let Some(lib) = &self.library else { return };
        let Some(info) = lib.get(lib_index) else { return };
        let run = Arc::new(info.clone());
        let n = run.cams.len();
        let workers: Vec<CamWorker> = run.cams.iter().enumerate().map(|(slot, cam)| spawn_cam_worker(slot, Arc::new(cam.clone()), ctx.clone())).collect();
        let t0 = run.t0().unwrap_or(0.0);
        let t1 = run.t1().unwrap_or(t0);
        let cloud_only = run.cams.iter().filter_map(|c| c.frames.get(c.frames.len() / 2)).any(|f| is_cloud_placeholder(&f.path));
        self.t = t0;
        self.in_marker = t0;
        self.out_marker = t1;
        self.playing = false;
        self.active_cam = 0;
        self.annot_pending = None;
        self.annot_draft = None;
        self.annot_mode = AnnotMode::None;
        self.rename_buf = run.display_name();
        self.notes_buf = run.sidecar.notes.clone();
        self.tags_buf = run.sidecar.tags.join(", ");
        self.pane_zoom = vec![1.0; n];
        self.pane_pan = vec![Vec2::ZERO; n];
        self.pane_view = vec![None; n];
        self.export_crop = false;
        self.delete_confirm = None;
        self.loaded = Some(LoadedRun {
            annotations: run.sidecar.annotations.clone(),
            sidecar: run.sidecar.clone(),
            run,
            lib_index,
            workers,
            textures: (0..n).map(|_| None).collect(),
            tex_seq: vec![0; n],
            last_frame: (0..n).map(|_| None).collect(),
            last_sent: (0..n).map(|_| None).collect(),
            pane_w: vec![640; n],
            reference_idx: vec![0; n],
            adjust: vec![CamAdjust::default(); n],
            t0,
            t1,
            cloud_only,
        });
        self.last_tick = Instant::now();
    }

    pub fn close_run(&mut self) {
        self.loaded = None;
        self.playing = false;
    }

    fn save_sidecar(&mut self) {
        if let Some(l) = &mut self.loaded {
            l.sidecar.annotations = l.annotations.clone();
            if let Err(e) = l.sidecar.save(&l.run.sidecar_path()) {
                self.last_job_msg = format!("sidecar save failed: {e}");
                self.last_job_ok = false;
            }
            // Keep the library copy in sync so re-opening shows the edits.
            if let Some(lib) = &self.library {
                let mut v = (**lib).clone();
                if let Some(r) = v.get_mut(l.lib_index) {
                    r.sidecar = l.sidecar.clone();
                }
                self.library = Some(Arc::new(v));
            }
        }
    }

    fn tick_playback(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64().min(0.25);
        self.last_tick = now;
        let Some(l) = &self.loaded else { return };
        self.buffering = false;
        if self.playing && l.t1 > l.t0 {
            // Hold the playhead while the decoders are behind and the proxy
            // cache is still building: better a pause than skipping frames.
            let slot = self.active_cam.min(l.run.cams.len().saturating_sub(1));
            let next_t = self.t + dt * self.speed();
            if let (Some(w), Some(shown)) = (l.run.frame_index_at(slot, next_t), l.last_frame.get(slot).and_then(|f| f.as_ref())) {
                let (pd, pt) = l.workers[slot].proxy.progress();
                let proxies_building = pd < pt;
                if proxies_building && w > shown.frame_idx + 2 {
                    self.buffering = true;
                    return;
                }
            }
            self.t += dt * self.speed();
            if self.t >= l.t1 {
                if self.looping {
                    self.t = l.t0;
                } else {
                    self.t = l.t1;
                    self.playing = false;
                }
            }
        }
    }

    fn step_frames(&mut self, n: i64) {
        let Some(l) = &self.loaded else { return };
        let slot = self.active_cam.min(l.run.cams.len().saturating_sub(1));
        let Some(idx) = l.run.frame_index_at(slot, self.t) else { return };
        let frames = &l.run.cams[slot].frames;
        let j = (idx as i64 + n).clamp(0, frames.len() as i64 - 1) as usize;
        self.t = frames[j].t;
        self.playing = false;
    }

    fn fraction(&self) -> f64 {
        match &self.loaded {
            Some(l) if l.t1 > l.t0 => ((self.t - l.t0) / (l.t1 - l.t0)).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    fn seek_fraction(&mut self, f: f64) {
        if let Some(l) = &self.loaded {
            self.t = l.t0 + f.clamp(0.0, 1.0) * (l.t1 - l.t0);
        }
    }

    fn proc_params(&self, l: &LoadedRun, slot: usize) -> ProcParams {
        make_params(self.stabilize, self.stab, self.sharpen, l, slot)
    }

    fn export_dir(captures_dir: &Path) -> PathBuf {
        captures_dir.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("exports")
    }

    /// `<base>[_k]` such that neither `<base>.<ext>` nor `<base>_final.png` exists
    /// (no `with_extension`: run folders may contain dots).
    fn unique_base(base: &Path, ext: &str) -> PathBuf {
        let taken = |b: &str| Path::new(&format!("{b}.{ext}")).exists() || Path::new(&format!("{b}_final.png")).exists();
        let b0 = base.display().to_string();
        if !taken(&b0) {
            return base.to_path_buf();
        }
        let mut k = 1;
        loop {
            let b = format!("{b0}_{k}");
            if !taken(&b) {
                return PathBuf::from(b);
            }
            k += 1;
        }
    }

    fn start_export(&mut self, captures_dir: &Path, t_start: f64, t_end: f64, tag: &str) {
        let Some(l) = &self.loaded else { return };
        if self.job_running() {
            return;
        }
        let slot = self.active_cam.min(l.run.cams.len().saturating_sub(1));
        let cam_no = l.run.cams[slot].cam_index + 1;
        let base = Self::export_dir(captures_dir).join(format!("{}_cam{cam_no}_{tag}", l.run.folder));
        let out_path = PathBuf::from(format!("{}.mp4", Self::unique_base(&base, "mp4").display()));
        let spec = ExportSpec {
            run: l.run.clone(),
            slot,
            t_start,
            t_end,
            out_path,
            params: self.proc_params(l, slot),
            overlays: self.overlays,
            annotations: l.annotations.clone(),
            fps: None,
            crop: if self.export_crop { self.pane_view.get(slot).copied().flatten() } else { None },
        };
        let handle = JobHandle::new("export");
        let status = handle.status.clone();
        let cancel = handle.cancel.clone();
        let _ = std::thread::Builder::new().name("replay-export".into()).spawn(move || {
            let res = run_export(&spec, &status, &cancel);
            JobHandle::set(&status, |s| {
                s.done = true;
                match &res {
                    Ok(p) => {
                        s.progress = 1.0;
                        s.out_path = Some(p.clone());
                        s.message = format!("Saved {}", p.display());
                    }
                    Err(e) => {
                        s.error = Some(e.clone());
                        s.message = format!("Export failed: {e}");
                    }
                }
            });
        });
        self.job = Some(handle);
    }

    fn start_stack(&mut self, captures_dir: &Path) {
        let Some(l) = &self.loaded else { return };
        if self.job_running() {
            return;
        }
        let slot = self.active_cam.min(l.run.cams.len().saturating_sub(1));
        let cam_no = l.run.cams[slot].cam_index + 1;
        let base = Self::export_dir(captures_dir).join(format!("{}_cam{cam_no}_stack{}", l.run.folder, self.stack_n));
        let out_base = Self::unique_base(&base, "png");
        let spec = StackSpec { run: l.run.clone(), slot, t_start: self.in_marker, t_end: self.out_marker, keep_n: self.stack_n.max(1), stab: self.stab, out_base, centered: self.stack_centered, center_size: self.stack_center_size };
        let handle = JobHandle::new("stack");
        let status = handle.status.clone();
        let cancel = handle.cancel.clone();
        let _ = std::thread::Builder::new().name("replay-stack".into()).spawn(move || {
            let res = run_stack(&spec, &status, &cancel);
            JobHandle::set(&status, |s| {
                s.done = true;
                match &res {
                    Ok(p) => {
                        s.progress = 1.0;
                        s.out_path = Some(p.clone());
                    }
                    Err(e) => {
                        s.error = Some(e.clone());
                        s.message = format!("Stack failed: {e}");
                    }
                }
            });
        });
        self.job = Some(handle);
    }

    fn poll_job(&mut self) {
        if let Some(j) = &self.job {
            let s = j.status.load();
            if s.done {
                self.last_job_msg = s.message.clone();
                self.last_job_ok = s.error.is_none();
                let _ = j.started;
                self.job = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------
pub fn screen(ui: &mut egui::Ui, st: &mut ReplayState, captures_dir: &Path) {
    if st.scanned_dir.as_deref() != Some(captures_dir) {
        st.rescan(captures_dir);
    }
    st.poll_scan();
    st.poll_job();
    st.tick_playback();
    // Transport keys — only while no widget owns the keyboard (TextEdit focus).
    if st.loaded.is_some() && ui.ctx().memory(|m| m.focused()).is_none() {
        let (space, left, right) = ui.input(|i| (i.key_pressed(egui::Key::Space), i.key_pressed(egui::Key::ArrowLeft), i.key_pressed(egui::Key::ArrowRight)));
        if space {
            st.playing = !st.playing;
            st.last_tick = Instant::now();
        }
        if left {
            st.step_frames(-1);
        }
        if right {
            st.step_frames(1);
        }
    }
    if st.playing || st.job_running() || st.scan_rx.is_some() {
        ui.ctx().request_repaint_after(Duration::from_millis(16));
    }

    egui::SidePanel::left("replay_library")
        .resizable(true)
        .default_width(290.0)
        .width_range(230.0..=460.0)
        .frame(egui::Frame::none().fill(theme::PANEL).inner_margin(egui::Margin::same(8.0)))
        .show_inside(ui, |ui| library_panel(ui, st, captures_dir));

    if st.loaded.is_some() {
        egui::SidePanel::right("replay_controls")
            .resizable(true)
            .default_width(300.0)
            .width_range(260.0..=420.0)
            .frame(egui::Frame::none().fill(theme::PANEL).inner_margin(egui::Margin::same(8.0)))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("replay_controls_scroll").auto_shrink([false, false]).show(ui, |ui| controls_panel(ui, st, captures_dir));
            });
        egui::TopBottomPanel::bottom("replay_transport")
            .frame(egui::Frame::none().fill(theme::PANEL).inner_margin(egui::Margin::symmetric(10.0, 6.0)))
            .show_inside(ui, |ui| transport_panel(ui, st));
    }

    egui::CentralPanel::default().frame(egui::Frame::none().fill(theme::BG).inner_margin(egui::Margin::same(8.0))).show_inside(ui, |ui| {
        if st.loaded.is_some() {
            panes(ui, st);
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("Select a run from the library to replay it").font(theme::sans(14.0)).color(theme::DIM));
            });
        }
    });
}

fn library_panel(ui: &mut egui::Ui, st: &mut ReplayState, captures_dir: &Path) {
    ui.horizontal(|ui| {
        theme::section(ui, "Run library");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add_enabled(st.scan_rx.is_none(), egui::Button::new(egui::RichText::new("rescan").font(theme::mono(11.0)))).clicked() {
                st.rescan(captures_dir);
            }
            if st.scan_rx.is_some() {
                ui.spinner();
            }
        });
    });
    ui.label(egui::RichText::new(captures_dir.display().to_string()).font(theme::mono(10.0)).color(theme::DIM));
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut st.filter).hint_text("filter").desired_width(110.0));
        ui.checkbox(&mut st.favorites_only, egui::RichText::new("★ only").font(theme::sans(11.0)));
        ui.checkbox(&mut st.group_by_target, egui::RichText::new("group").font(theme::sans(11.0))).on_hover_text("Group runs by target (off = flat list)");
    });
    ui.add_space(4.0);
    let Some(lib) = st.library.clone() else {
        ui.label(egui::RichText::new("scanning…").color(theme::DIM));
        return;
    };
    if lib.is_empty() {
        ui.label(egui::RichText::new("no runs with camera frames found").color(theme::DIM));
        return;
    }
    let filter = st.filter.trim().to_lowercase();
    let selected = st.loaded.as_ref().map(|l| l.lib_index);
    let mut clicked: Option<usize> = None;
    let mut toggled_fav: Option<usize> = None;
    let visible: Vec<usize> = lib
        .iter()
        .enumerate()
        .filter(|(_, r)| !(st.favorites_only && !r.sidecar.favorite))
        .filter(|(_, r)| filter.is_empty() || r.folder.to_lowercase().contains(&filter) || r.display_name().to_lowercase().contains(&filter) || r.target.to_lowercase().contains(&filter))
        .map(|(i, _)| i)
        .collect();
    let group_by_target = st.group_by_target;
    egui::ScrollArea::vertical().id_salt("replay_lib_scroll").auto_shrink([false, false]).show(ui, |ui| {
        let mut card = |ui: &mut egui::Ui, i: usize, r: &RunInfo| {
            let is_sel = selected == Some(i);
            let fill = if is_sel { theme::with_alpha(theme::ACCENT, 40) } else { theme::RAISED };
            let stroke = if is_sel { Stroke::new(1.0, theme::ACCENT) } else { Stroke::new(1.0, theme::HAIRLINE) };
            let resp = egui::Frame::none()
                .fill(fill)
                .stroke(stroke)
                .rounding(egui::Rounding::same(4.0))
                .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let date = r.start_epoch.or_else(|| r.t0()).map(fmt_date).unwrap_or_else(|| "—".into());
                        ui.label(egui::RichText::new(date).font(theme::mono(11.0)).color(theme::TEXT_2));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let star = if r.sidecar.favorite { "★" } else { "☆" };
                            let col = if r.sidecar.favorite { theme::AMBER } else { theme::DIM };
                            if ui.add(egui::Button::new(egui::RichText::new(star).color(col)).frame(false)).clicked() {
                                toggled_fav = Some(i);
                            }
                        });
                    });
                    let name_col = if r.is_manual { theme::TEXT_2 } else { theme::TEXT };
                    let mut title = r.display_name();
                    if title == r.folder {
                        title = r.target.clone();
                        if let Some(n) = &r.norad {
                            title = format!("{title}  ·  {n}");
                        }
                    }
                    ui.label(egui::RichText::new(title).font(theme::sans(13.0)).color(name_col).strong());
                    let counts: Vec<String> = r.cams.iter().map(|c| format!("cam{} {}f", c.cam_index + 1, c.frames.len())).collect();
                    let traj = if r.trajectory.valid() { "  traj" } else { "" };
                    ui.label(egui::RichText::new(format!("{}  ·  {:.1}s{traj}", counts.join("  "), r.duration())).font(theme::mono(10.5)).color(theme::DIM));
                    if !r.sidecar.tags.is_empty() {
                        ui.label(egui::RichText::new(r.sidecar.tags.join(", ")).font(theme::sans(10.5)).color(theme::ACCENT));
                    }
                })
                .response;
            let resp = resp.interact(Sense::click());
            if resp.clicked() {
                clicked = Some(i);
            }
            ui.add_space(3.0);
        };
        if group_by_target {
            // Groups keep the newest-first run order; header order = first appearance.
            let mut order: Vec<&str> = Vec::new();
            let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
            for &i in &visible {
                let key = lib[i].target.as_str();
                groups.entry(key).or_insert_with(|| {
                    order.push(key);
                    Vec::new()
                }).push(i);
            }
            for key in order {
                let idxs = &groups[key];
                egui::CollapsingHeader::new(egui::RichText::new(format!("{key}  ({})", idxs.len())).font(theme::sans(12.0)).strong())
                    .id_salt(("replay_lib_group", key))
                    .default_open(true)
                    .show(ui, |ui| {
                        for &i in idxs {
                            card(ui, i, &lib[i]);
                        }
                    });
            }
        } else {
            for &i in &visible {
                card(ui, i, &lib[i]);
            }
        }
    });
    if let Some(i) = toggled_fav {
        let mut v = (*lib).clone();
        if let Some(r) = v.get_mut(i) {
            r.sidecar.favorite = !r.sidecar.favorite;
            let _ = r.sidecar.save(&r.sidecar_path());
            if let Some(l) = &mut st.loaded {
                if l.lib_index == i {
                    l.sidecar.favorite = r.sidecar.favorite;
                }
            }
        }
        st.library = Some(Arc::new(v));
    }
    if let Some(i) = clicked {
        // The star sits on top of the card: a favourite toggle is not a load.
        if selected != Some(i) && toggled_fav.is_none() {
            st.close_run();
            st.load_run(i, Some(ui.ctx().clone()));
        }
    }
}

fn fit_box(max_w: f32, max_h: f32, aspect: f32) -> Vec2 {
    let mut w = max_w;
    let mut h = w / aspect;
    if h > max_h {
        h = max_h;
        w = h * aspect;
    }
    Vec2::new(w.max(16.0), h.max(9.0))
}

fn panes(ui: &mut egui::Ui, st: &mut ReplayState) {
    let Some(l) = st.loaded.as_mut() else { return };
    let n = l.run.cams.len().min(3).max(1);
    let avail = ui.available_size();
    let gap = 10.0;
    let half_w = if n > 1 { (avail.x - gap * (n as f32 - 1.0)) / n as f32 } else { avail.x };
    let top_left = ui.cursor().min;
    let mut draft_commit: Option<Annotation> = None;
    let mut pending_text: Option<(usize, f64, f64)> = None;
    let mut new_active: Option<usize> = None;
    let mut set_ref: Option<usize> = None;
    let mut save_needed = false;

    // Pull fresh frames -> textures.
    for slot in 0..l.workers.len() {
        if let Some(f) = l.workers[slot].out.load_full() {
            if f.seq != l.tex_seq[slot] {
                l.tex_seq[slot] = f.seq;
                if f.error.is_none() && f.w > 0 && f.h > 0 {
                    let img = egui::ColorImage::from_gray([f.w, f.h], &f.gray);
                    match l.textures[slot].as_mut() {
                        Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                        None => l.textures[slot] = Some(ui.ctx().load_texture(format!("replay_cam{slot}"), img, egui::TextureOptions::LINEAR)),
                    }
                }
                l.last_frame[slot] = Some(f);
            }
        }
    }

    for slot in 0..n {
        let cam = &l.run.cams[slot];
        let (fw, fh) = l.last_frame[slot].as_ref().filter(|f| f.full_w > 0).map(|f| (f.full_w as f32, f.full_h as f32)).unwrap_or((16.0, 9.0));
        let size = fit_box(half_w, avail.y, fw / fh);
        let origin = Pos2::new(top_left.x + slot as f32 * (half_w + gap), top_left.y);
        let rect = Rect::from_min_size(origin, size);
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::BLACK);

        // Per-pane pixel zoom: mousewheel about the cursor, drag pans (unless an
        // annotation drag tool is armed), right-click / chip resets. A display
        // transform: the image rect scales/offsets and every overlay rides it.
        while st.pane_zoom.len() <= slot {
            st.pane_zoom.push(1.0);
            st.pane_pan.push(Vec2::ZERO);
            st.pane_view.push(None);
        }
        if let Some(ptr) = resp.hover_pos() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let old = st.pane_zoom[slot];
                let new = (old * (scroll * 0.0022).exp()).clamp(1.0, 16.0);
                if new != old {
                    let cpos = rect.center() + st.pane_pan[slot];
                    st.pane_pan[slot] = (ptr + (cpos - ptr) * (new / old)) - rect.center();
                    st.pane_zoom[slot] = new;
                }
            }
        }
        let pan_allowed = !matches!(st.annot_mode, AnnotMode::Arrow | AnnotMode::Box);
        if resp.dragged() && pan_allowed && st.pane_zoom[slot] > 1.001 {
            st.pane_pan[slot] += resp.drag_delta();
        }
        if resp.secondary_clicked() {
            st.pane_zoom[slot] = 1.0;
        }
        if st.pane_zoom[slot] <= 1.001 {
            st.pane_zoom[slot] = 1.0;
            st.pane_pan[slot] = Vec2::ZERO;
        } else {
            let lim = Vec2::new(rect.width(), rect.height()) * st.pane_zoom[slot] * 0.6;
            st.pane_pan[slot] = st.pane_pan[slot].clamp(-lim, lim);
        }
        let zoom = st.pane_zoom[slot];
        let img_rect = Rect::from_center_size(rect.center() + st.pane_pan[slot], size * zoom);
        // Decode target follows the zoom so a paused zoomed view sharpens up.
        l.pane_w[slot] = (size.x * zoom).round() as usize;
        // Visible source-pixel rect (crop export) while zoomed.
        st.pane_view[slot] = if zoom > 1.001 {
            l.last_frame[slot]
                .as_ref()
                .filter(|f| f.full_w > 0 && f.full_h > 0)
                .map(|f| {
                    let vis = rect.intersect(img_rect);
                    let fx = f.full_w as f32 / img_rect.width();
                    let fy = f.full_h as f32 / img_rect.height();
                    let x0 = (((vis.min.x - img_rect.min.x) * fx).floor().max(0.0)) as usize;
                    let y0 = (((vis.min.y - img_rect.min.y) * fy).floor().max(0.0)) as usize;
                    let x1 = (((vis.max.x - img_rect.min.x) * fx).ceil().max(0.0) as usize).min(f.full_w);
                    let y1 = (((vis.max.y - img_rect.min.y) * fy).ceil().max(0.0) as usize).min(f.full_h);
                    [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
                })
                .filter(|v| v[2] >= 2 && v[3] >= 2)
        } else {
            None
        };

        if let Some(tex) = &l.textures[slot] {
            painter.image(tex.id(), img_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        } else {
            let msg = match l.last_frame[slot].as_ref().and_then(|f| f.error.clone()) {
                Some(e) => format!("decode error: {e}"),
                None => "decoding…".to_string(),
            };
            painter.text(rect.left_top() + Vec2::new(8.0, 8.0), Align2::LEFT_TOP, msg, theme::mono(11.0), theme::DIM);
        }

        // Overlays: vectors, meta text, annotations, stab status.
        if let Some(f) = l.last_frame[slot].clone() {
            if st.overlays && f.error.is_none() {
                let (pw, ph) = (img_rect.width() as f64, img_rect.height() as f64);
                if let Some(v) = compute_track_vectors(&l.run.trajectory, f.t, pw, ph) {
                    let p = |q: [f64; 2]| Pos2::new(img_rect.min.x + q[0] as f32, img_rect.min.y + q[1] as f32);
                    let (a, it) = (p(v.anchor), p(v.intrack));
                    painter.arrow(a, it - a, Stroke::new(1.5, theme::GREEN));
                    painter.line_segment([p(v.cross_p), p(v.cross_n)], Stroke::new(1.5, theme::ACCENT));
                    painter.text(it + Vec2::new(4.0, 0.0), Align2::LEFT_CENTER, "IN", theme::mono(10.0), theme::GREEN);
                    painter.text(p(v.cross_p) + Vec2::new(4.0, 0.0), Align2::LEFT_CENTER, "CROSS", theme::mono(10.0), theme::ACCENT);
                }
                let gamma = l.adjust[slot].gamma;
                let meta = build_meta_lines(&l.run, cam.cam_index, f.t, f.frame_idx, cam.frames.len(), gamma, st.stabilize);
                let mut y = rect.min.y + 6.0;
                for line in meta {
                    painter.text(Pos2::new(rect.min.x + 6.0, y), Align2::LEFT_TOP, line, theme::mono(11.0), theme::AMBER);
                    y += 14.0;
                }
                let annot_col = theme::RED;
                for a in l.annotations.iter().filter(|a| a.cam() == slot) {
                    let p = |x: f64, y: f64| Pos2::new(img_rect.min.x + x as f32 * img_rect.width(), img_rect.min.y + y as f32 * img_rect.height());
                    match a {
                        Annotation::Text { x, y, text, .. } => {
                            painter.text(p(*x, *y), Align2::LEFT_BOTTOM, text, theme::sans(12.0), annot_col);
                        }
                        Annotation::Arrow { x0, y0, x1, y1, .. } => {
                            let (a0, a1) = (p(*x0, *y0), p(*x1, *y1));
                            painter.arrow(a0, a1 - a0, Stroke::new(2.0, annot_col));
                        }
                        Annotation::Box { x0, y0, x1, y1, .. } => {
                            painter.rect_stroke(Rect::from_two_pos(p(*x0, *y0), p(*x1, *y1)), 0.0, Stroke::new(2.0, annot_col));
                        }
                    }
                }
            }
            if let Some(s) = &f.stab {
                let (txt, col) = if s.ok { (format!("STAB ok · {} inliers", s.inliers), theme::GREEN) } else { (format!("STAB passthrough · {}", s.reason.clone().unwrap_or_default()), theme::AMBER) };
                painter.text(rect.right_bottom() + Vec2::new(-6.0, -6.0), Align2::RIGHT_BOTTOM, txt, theme::mono(10.0), col);
            }
            let (pd, pt) = l.workers[slot].proxy.progress();
            let cache_txt = if pd < pt { format!("  · caching {pd}/{pt}") } else { String::new() };
            painter.text(rect.left_bottom() + Vec2::new(6.0, -6.0), Align2::LEFT_BOTTOM, format!("{}x{} /{}  {:.0} ms{cache_txt}", f.w, f.h, f.reduce, f.proc_ms), theme::mono(9.5), theme::DIM);
        }
        // Draft annotation preview.
        if let Some((dc, x0, y0, x1, y1)) = st.annot_draft {
            if dc == slot {
                let p = |x: f64, y: f64| Pos2::new(img_rect.min.x + x as f32 * img_rect.width(), img_rect.min.y + y as f32 * img_rect.height());
                match st.annot_mode {
                    AnnotMode::Box => {
                        painter.rect_stroke(Rect::from_two_pos(p(x0, y0), p(x1, y1)), 0.0, Stroke::new(1.5, theme::RED));
                    }
                    _ => {
                        painter.line_segment([p(x0, y0), p(x1, y1)], Stroke::new(1.5, theme::RED));
                    }
                }
            }
        }
        if let Some((pc, x, y)) = st.annot_pending {
            if pc == slot {
                let p = Pos2::new(img_rect.min.x + x as f32 * img_rect.width(), img_rect.min.y + y as f32 * img_rect.height());
                painter.circle_stroke(p, 5.0, Stroke::new(1.5, theme::RED));
            }
        }

        // Frame chrome.
        let active = slot == st.active_cam;
        painter.rect_stroke(rect, 3.0, Stroke::new(if active { 1.5 } else { 1.0 }, if active { theme::ACCENT } else { theme::HAIRLINE }));
        let tag = format!("CAM {}", cam.cam_index + 1);
        painter.text(rect.right_top() + Vec2::new(-6.0, 6.0), Align2::RIGHT_TOP, tag, theme::mono(11.0), if active { theme::ACCENT } else { theme::TEXT_2 });
        // Zoom reset chip (right-click does the same).
        if zoom > 1.0 {
            let br = Rect::from_min_size(rect.right_top() + Vec2::new(-92.0, 22.0), Vec2::new(84.0, 18.0));
            let rb = ui.interact(br, ui.id().with(("replay_zoom_reset", slot)), Sense::click());
            painter.rect(br, 4.0, theme::with_alpha(theme::RAISED, 230), Stroke::new(1.0, if rb.hovered() { theme::ACCENT } else { theme::HAIRLINE }));
            painter.text(br.center(), Align2::CENTER_CENTER, format!("{zoom:.1}× · reset"), theme::mono(9.5), if rb.hovered() { theme::TEXT } else { theme::TEXT_2 });
            if rb.clicked() {
                st.pane_zoom[slot] = 1.0;
                st.pane_pan[slot] = Vec2::ZERO;
                st.pane_view[slot] = None;
            }
        }

        // Interaction: select pane, annotation tools (image-normalized coords).
        let norm = |p: Pos2| (((p.x - img_rect.min.x) / img_rect.width()).clamp(0.0, 1.0) as f64, ((p.y - img_rect.min.y) / img_rect.height()).clamp(0.0, 1.0) as f64);
        if resp.clicked() {
            new_active = Some(slot);
            if st.annot_mode == AnnotMode::Text {
                if let Some(p) = resp.interact_pointer_pos() {
                    let (x, y) = norm(p);
                    pending_text = Some((slot, x, y));
                }
            }
        }
        if resp.double_clicked() {
            set_ref = Some(slot);
        }
        if matches!(st.annot_mode, AnnotMode::Arrow | AnnotMode::Box) {
            if resp.drag_started() {
                new_active = Some(slot);
                if let Some(p) = resp.interact_pointer_pos() {
                    let (x, y) = norm(p);
                    st.annot_draft = Some((slot, x, y, x, y));
                }
            }
            if resp.dragged() {
                if let (Some(p), Some(d)) = (resp.interact_pointer_pos(), st.annot_draft.as_mut()) {
                    if d.0 == slot {
                        let (x, y) = norm(p);
                        d.3 = x;
                        d.4 = y;
                    }
                }
            }
            if resp.drag_stopped() {
                if let Some((dc, x0, y0, x1, y1)) = st.annot_draft.take() {
                    if dc == slot && ((x1 - x0).abs() > 0.005 || (y1 - y0).abs() > 0.005) {
                        draft_commit = Some(match st.annot_mode {
                            AnnotMode::Box => Annotation::Box { cam: slot, x0, y0, x1, y1 },
                            _ => Annotation::Arrow { cam: slot, x0, y0, x1, y1 },
                        });
                    }
                }
            }
        }
    }
    if let Some(a) = draft_commit {
        l.annotations.push(a);
        save_needed = true;
    }
    if let Some(s) = new_active {
        st.active_cam = s;
    }
    if pending_text.is_some() {
        st.annot_pending = pending_text;
    }
    if let Some(s) = set_ref {
        if let Some(idx) = l.run.frame_index_at(s, st.t) {
            l.reference_idx[s] = idx;
        }
    }

    // Requests: one per pane when the target frame or params changed.
    for slot in 0..n {
        let Some(idx) = l.run.frame_index_at(slot, st.t) else { continue };
        let params = make_params(st.stabilize, st.stab, st.sharpen, l, slot);
        let pane_w = l.pane_w[slot];
        let same = l.last_sent[slot].as_ref().map_or(false, |(i, w, p, pl)| *i == idx && *w == pane_w && *p == params && *pl == st.playing);
        if !same {
            st.req_gen += 1;
            let _ = l.workers[slot].tx.send(FrameRequest { frame_idx: idx, pane_w, params: params.clone(), gen: st.req_gen, playing: st.playing });
            l.last_sent[slot] = Some((idx, pane_w, params, st.playing));
        }
    }
    if save_needed {
        st.save_sidecar();
    }
}

fn scrubber(ui: &mut egui::Ui, st: &mut ReplayState) {
    let Some(l) = &st.loaded else { return };
    let (t0, t1) = (l.t0, l.t1);
    let w = ui.available_width();
    let (resp, p) = ui.allocate_painter(Vec2::new(w, 22.0), Sense::click_and_drag());
    let r = resp.rect;
    let track = Rect::from_min_max(Pos2::new(r.min.x + 6.0, r.center().y - 3.0), Pos2::new(r.max.x - 6.0, r.center().y + 3.0));
    p.rect_filled(track, 3.0, theme::RAISED);
    p.rect_stroke(track, 3.0, Stroke::new(1.0, theme::HAIRLINE));
    let span = (t1 - t0).max(1e-9);
    let xof = |t: f64| track.min.x + ((t - t0) / span).clamp(0.0, 1.0) as f32 * track.width();
    // Clip range shading.
    let (ci, co) = (xof(st.in_marker.min(st.out_marker)), xof(st.in_marker.max(st.out_marker)));
    p.rect_filled(Rect::from_min_max(Pos2::new(ci, track.min.y), Pos2::new(co, track.max.y)), 0.0, theme::with_alpha(theme::ACCENT, 70));
    for (m, col) in [(st.in_marker, theme::GREEN), (st.out_marker, theme::RED)] {
        let x = xof(m);
        p.line_segment([Pos2::new(x, r.min.y + 1.0), Pos2::new(x, r.max.y - 1.0)], Stroke::new(2.0, col));
    }
    // Frame ticks for the active cam (sparse).
    if let Some(c) = l.run.cams.get(st.active_cam) {
        let step = (c.frames.len() / 200).max(1);
        for f in c.frames.iter().step_by(step) {
            let x = xof(f.t);
            p.line_segment([Pos2::new(x, track.max.y), Pos2::new(x, track.max.y + 3.0)], Stroke::new(1.0, theme::DIM));
        }
    }
    let kx = xof(st.t);
    p.rect_filled(Rect::from_center_size(Pos2::new(kx, r.center().y), Vec2::new(8.0, 18.0)), 2.0, theme::TEXT);
    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let f = ((pos.x - track.min.x) / track.width()).clamp(0.0, 1.0) as f64;
            st.seek_fraction(f);
            st.playing = false;
        }
    }
}

fn transport_panel(ui: &mut egui::Ui, st: &mut ReplayState) {
    scrubber(ui, st);
    ui.horizontal(|ui| {
        if theme::mode_button(ui, "|<", false, theme::ACCENT) {
            st.step_frames(-1);
        }
        let play_label = if st.playing { "PAUSE" } else { "PLAY" };
        if theme::mode_button(ui, play_label, st.playing, theme::GREEN) {
            st.playing = !st.playing;
            st.last_tick = Instant::now();
        }
        if theme::mode_button(ui, ">|", false, theme::ACCENT) {
            st.step_frames(1);
        }
        if theme::mode_button(ui, &format!("x{}", st.speed()), false, theme::ACCENT) {
            st.speed_idx = (st.speed_idx + 1) % SPEEDS.len();
        }
        if theme::mode_button(ui, "LOOP", st.looping, theme::ACCENT) {
            st.looping = !st.looping;
        }
        ui.separator();
        if theme::mode_button(ui, "SET IN", false, theme::GREEN) {
            st.in_marker = st.t;
        }
        if theme::mode_button(ui, "SET OUT", false, theme::RED) {
            st.out_marker = st.t;
        }
        if theme::mode_button(ui, "RESET", false, theme::ACCENT) {
            if let Some(l) = &st.loaded {
                st.in_marker = l.t0;
                st.out_marker = l.t1;
            }
        }
        ui.separator();
        let (t0, dur) = st.loaded.as_ref().map(|l| (l.t0, l.t1 - l.t0)).unwrap_or((0.0, 0.0));
        ui.label(egui::RichText::new(format!("{} UTC", fmt_utc(st.t))).font(theme::mono(12.5)).color(theme::TEXT));
        ui.label(egui::RichText::new(format!("+{:.2}s / {:.1}s  ({:4.1}%)", st.t - t0, dur, st.fraction() * 100.0)).font(theme::mono(11.0)).color(theme::TEXT_2));
        ui.label(egui::RichText::new(format!("clip {:.1}s → {:.1}s", st.in_marker - t0, st.out_marker - t0)).font(theme::mono(11.0)).color(theme::DIM));
        if let Some(l) = &st.loaded {
            let frames: Vec<String> = (0..l.run.cams.len()).filter_map(|s| l.run.frame_index_at(s, st.t).map(|i| format!("c{} {}/{}", l.run.cams[s].cam_index + 1, i + 1, l.run.cams[s].frames.len()))).collect();
            ui.label(egui::RichText::new(frames.join("  ")).font(theme::mono(11.0)).color(theme::DIM));
            let (pd, pt): (usize, usize) = l.workers.iter().map(|w| w.proxy.progress()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            if pd < pt {
                let what = if l.cloud_only { "downloading from OneDrive" } else { "caching" };
                let txt = if st.buffering { format!("buffering · {what} {pd}/{pt}") } else { format!("{what} {pd}/{pt}") };
                ui.label(egui::RichText::new(txt).font(theme::mono(11.0)).color(theme::AMBER));
            }
            if l.cloud_only {
                ui.label(egui::RichText::new("online-only run: first playback pulls every frame from the cloud (right-click the folder → Always keep on this device)").font(theme::sans(10.5)).color(theme::TEXT_2));
            }
        }
    });
}

fn controls_panel(ui: &mut egui::Ui, st: &mut ReplayState, captures_dir: &Path) {
    let Some(run) = st.loaded.as_ref().map(|l| l.run.clone()) else { return };
    let n = run.cams.len();
    st.active_cam = st.active_cam.min(n.saturating_sub(1));

    // Run header + rename.
    theme::section(ui, "Run");
    ui.label(egui::RichText::new(run.display_name()).font(theme::sans(13.0)).strong());
    ui.label(egui::RichText::new(format!("{}{}", run.target, run.norad.as_ref().map(|n| format!("  ·  NORAD {n}")).unwrap_or_default())).font(theme::mono(11.0)).color(theme::TEXT_2));
    ui.label(egui::RichText::new(run.folder.clone()).font(theme::mono(10.0)).color(theme::DIM));
    ui.horizontal(|ui| {
        let r = ui.add(egui::TextEdit::singleline(&mut st.rename_buf).hint_text("display name").desired_width(170.0));
        if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            let name = st.rename_buf.trim().to_string();
            if let Some(l) = &mut st.loaded {
                l.sidecar.display_name = name;
            }
            st.save_sidecar();
        }
        if ui.button("close").clicked() {
            st.close_run();
        }
    });
    if st.loaded.is_none() {
        return;
    }
    // Notes + tags -> postproc.json (saved when the field loses focus).
    let mut save_meta = false;
    let r = ui.add(egui::TextEdit::multiline(&mut st.notes_buf).hint_text("notes").desired_rows(2).desired_width(f32::INFINITY).font(theme::sans(11.5)));
    save_meta |= r.lost_focus();
    let r = ui.add(egui::TextEdit::singleline(&mut st.tags_buf).hint_text("tags (comma separated)").desired_width(f32::INFINITY).font(theme::sans(11.5)));
    save_meta |= r.lost_focus();
    if save_meta {
        let tags: Vec<String> = st.tags_buf.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
        let dirty = st.loaded.as_mut().map_or(false, |l| {
            let d = l.sidecar.notes != st.notes_buf || l.sidecar.tags != tags;
            if d {
                l.sidecar.notes = st.notes_buf.clone();
                l.sidecar.tags = tags;
            }
            d
        });
        if dirty {
            st.save_sidecar();
        }
    }
    // Delete run: two-step confirm, the armed state expires after 3 s.
    let mut delete_now = false;
    ui.horizontal(|ui| {
        let armed = st.delete_confirm.map_or(false, |t| t.elapsed() < Duration::from_secs(3));
        if !armed {
            st.delete_confirm = None;
        }
        let label = if armed { "really delete?" } else { "delete run" };
        if ui.button(egui::RichText::new(label).font(theme::sans(11.0)).color(theme::RED)).clicked() {
            if armed {
                delete_now = true;
            } else {
                st.delete_confirm = Some(Instant::now());
            }
        }
        if armed {
            ui.label(egui::RichText::new("removes the whole run folder").font(theme::sans(10.5)).color(theme::AMBER));
            ui.ctx().request_repaint_after(Duration::from_millis(250)); // let the arm expire visibly
        }
    });
    if delete_now {
        let path = run.path.clone();
        st.close_run();
        st.delete_confirm = None;
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                st.last_job_msg = format!("deleted {}", path.display());
                st.last_job_ok = true;
            }
            Err(e) => {
                st.last_job_msg = format!("delete failed: {e}");
                st.last_job_ok = false;
            }
        }
        st.rescan(captures_dir);
        return;
    }
    ui.add_space(6.0);

    // Image adjust.
    theme::section(ui, "Image adjust");
    ui.horizontal(|ui| {
        for s in 0..n {
            if theme::mode_button(ui, &format!("CAM {}", run.cams[s].cam_index + 1), st.active_cam == s, theme::ACCENT) {
                st.active_cam = s;
            }
        }
    });
    let slot = st.active_cam;
    if let Some(l) = st.loaded.as_mut() {
        let mut copy_to_others = false;
        {
            let a = &mut l.adjust[slot];
            ui.spacing_mut().slider_width = 150.0;
            ui.add(egui::Slider::new(&mut a.gamma, 0.2..=5.0).text("gamma").fixed_decimals(2));
            ui.add(egui::Slider::new(&mut a.brightness, -128.0..=128.0).text("brightness").fixed_decimals(0));
            ui.add(egui::Slider::new(&mut a.contrast, 0.2..=3.0).text("contrast").fixed_decimals(2));
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    *a = CamAdjust::default();
                }
                if n > 1 && ui.button("copy to other cam").clicked() {
                    copy_to_others = true;
                }
            });
        }
        if copy_to_others {
            let v = l.adjust[slot];
            for (i, o) in l.adjust.iter_mut().enumerate() {
                if i != slot {
                    *o = v;
                }
            }
        }
    }
    ui.add_space(6.0);

    // Stabilize.
    theme::section(ui, "Stabilize");
    ui.horizontal(|ui| {
        if theme::mode_button(ui, "STABILIZE", st.stabilize, theme::GREEN) {
            st.stabilize = !st.stabilize;
        }
        if ui.add_enabled(st.stabilize, egui::Button::new("reference = here")).on_hover_text("Anchor the stabilizer on the active camera's current frame (double-click a pane does the same)").clicked() {
            if let Some(l) = st.loaded.as_mut() {
                if let Some(idx) = l.run.frame_index_at(slot, st.t) {
                    l.reference_idx[slot] = idx;
                }
            }
        }
    });
    if st.stabilize {
        let ref_idx = st.loaded.as_ref().map(|l| l.reference_idx[slot]).unwrap_or(0);
        ui.label(egui::RichText::new(format!("flow method · reference frame {}", ref_idx + 1)).font(theme::mono(10.5)).color(theme::DIM));
        egui::Grid::new("replay_stab_grid").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
            ui.label(egui::RichText::new("max features").color(theme::TEXT_2));
            ui.add(egui::DragValue::new(&mut st.stab.max_features).range(50..=3000).speed(10.0));
            ui.end_row();
            ui.label(egui::RichText::new("RANSAC px").color(theme::TEXT_2));
            ui.add(egui::DragValue::new(&mut st.stab.ransac_threshold).range(0.5..=20.0).speed(0.1).fixed_decimals(1));
            ui.end_row();
            ui.label(egui::RichText::new("min inliers").color(theme::TEXT_2));
            ui.add(egui::DragValue::new(&mut st.stab.min_inliers).range(3..=500));
            ui.end_row();
            ui.label(egui::RichText::new("min inlier ratio").color(theme::TEXT_2));
            ui.add(egui::DragValue::new(&mut st.stab.min_inlier_ratio).range(0.0..=1.0).speed(0.01).fixed_decimals(2));
            ui.end_row();
        });
    }
    ui.add_space(6.0);

    // Sharpen.
    theme::section(ui, "Sharpen");
    ui.horizontal(|ui| {
        if theme::mode_button(ui, "SHARPEN", st.sharpen.on, theme::GREEN) {
            st.sharpen.on = !st.sharpen.on;
        }
        ui.checkbox(&mut st.sharpen.stretch, "auto-stretch");
    });
    ui.add(egui::Slider::new(&mut st.sharpen.strength, 0.0..=3.0).text("strength").fixed_decimals(2));
    ui.add_space(6.0);

    // Overlays + annotation.
    theme::section(ui, "Overlays");
    ui.horizontal(|ui| {
        if theme::mode_button(ui, "OVERLAYS", st.overlays, theme::ACCENT) {
            st.overlays = !st.overlays;
        }
        ui.label(egui::RichText::new("vectors · meta · annotations").font(theme::sans(10.5)).color(theme::DIM));
    });
    ui.label(egui::RichText::new("Annotate (active pane)").font(theme::sans(11.0)).color(theme::TEXT_2));
    ui.horizontal(|ui| {
        for (mode, label) in [(AnnotMode::Text, "TEXT"), (AnnotMode::Arrow, "ARROW"), (AnnotMode::Box, "BOX")] {
            if theme::mode_button(ui, label, st.annot_mode == mode, theme::RED) {
                st.annot_mode = if st.annot_mode == mode { AnnotMode::None } else { mode };
                st.annot_pending = None;
                st.annot_draft = None;
            }
        }
        if ui.button("clear").clicked() {
            if let Some(l) = st.loaded.as_mut() {
                l.annotations.clear();
            }
            st.save_sidecar();
        }
    });
    match st.annot_mode {
        AnnotMode::Text => {
            ui.label(egui::RichText::new(if st.annot_pending.is_some() { "type the label, Enter to place" } else { "click a pane to place text" }).font(theme::sans(10.5)).color(theme::AMBER));
            let r = ui.add(egui::TextEdit::singleline(&mut st.annot_text).hint_text("label").desired_width(200.0));
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let (Some((cam, x, y)), false) = (st.annot_pending, st.annot_text.trim().is_empty()) {
                    let text = st.annot_text.trim().to_string();
                    if let Some(l) = st.loaded.as_mut() {
                        l.annotations.push(Annotation::Text { cam, x, y, text });
                    }
                    st.annot_pending = None;
                    st.annot_text.clear();
                    st.save_sidecar();
                }
            }
        }
        AnnotMode::Arrow | AnnotMode::Box => {
            ui.label(egui::RichText::new("drag on a pane to draw").font(theme::sans(10.5)).color(theme::AMBER));
        }
        AnnotMode::None => {}
    }
    let n_ann = st.loaded.as_ref().map(|l| l.annotations.len()).unwrap_or(0);
    ui.label(egui::RichText::new(format!("{n_ann} annotation(s) · saved to postproc.json")).font(theme::mono(10.0)).color(theme::DIM));
    ui.add_space(6.0);

    // Stack.
    theme::section(ui, "Stack (In → Out)");
    let n_range = st.loaded.as_ref().map(|l| frames_in_range(&l.run, slot, st.in_marker, st.out_marker).len()).unwrap_or(0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("best N").color(theme::TEXT_2));
        ui.add(egui::DragValue::new(&mut st.stack_n).range(1..=2000).speed(1.0));
        ui.label(egui::RichText::new(format!("of {n_range} in range")).font(theme::mono(10.5)).color(theme::DIM));
        for (label, frac) in [("25%", 0.25), ("50%", 0.5)] {
            if ui.small_button(label).on_hover_text(format!("best N = {label} of the frames in range")).clicked() {
                st.stack_n = ((frac * n_range as f64).ceil() as usize).max(1);
            }
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut st.stack_centered, egui::RichText::new("centered").font(theme::sans(11.0))).on_hover_text("PIPP-style: recentre each frame on the track target / brightest blob and crop before aligning");
        if st.stack_centered {
            ui.add(egui::DragValue::new(&mut st.stack_center_size).range(64..=4096).speed(8.0));
            ui.label(egui::RichText::new("px crop").font(theme::mono(10.5)).color(theme::DIM));
        }
    });
    if ui.add_enabled(!st.job_running(), egui::Button::new(format!("Stack best {}", st.stack_n))).on_hover_text("Pre-cull empty frames (content score), grade by Laplacian sharpness, keep the best N, align to the sharpest (flow), coverage-weighted mean → 16-bit master PNG + sharpened/stretched final PNG in exports/").clicked() {
        st.start_stack(captures_dir);
    }
    ui.add_space(6.0);

    // Export.
    theme::section(ui, "Export MP4");
    let mp4_ok = cfg!(feature = "mp4-export");
    if !mp4_ok {
        ui.label(egui::RichText::new("not compiled in (feature mp4-export)").color(theme::RED));
    }
    // Crop-to-view: offered while the active pane holds a pixel zoom.
    match st.pane_view.get(st.active_cam).copied().flatten() {
        Some(v) => {
            ui.checkbox(&mut st.export_crop, egui::RichText::new(format!("crop to view ({}×{} px)", v[2], v[3])).font(theme::sans(11.0))).on_hover_text("Bake the zoomed pane view into the export (annotations remapped)");
        }
        None => st.export_crop = false,
    }
    ui.horizontal(|ui| {
        if ui.add_enabled(mp4_ok && !st.job_running(), egui::Button::new("Export clip (In → Out)")).clicked() {
            let (a, b) = (st.in_marker, st.out_marker);
            st.start_export(captures_dir, a, b, if st.stabilize { "stab" } else { "clip" });
        }
        if ui.add_enabled(mp4_ok && !st.job_running(), egui::Button::new("Whole run")).clicked() {
            if let Some((a, b)) = st.loaded.as_ref().map(|l| (l.t0, l.t1)) {
                st.start_export(captures_dir, a, b, "run");
            }
        }
    });
    ui.label(egui::RichText::new("bakes gamma/B-C, stabilize, sharpen + overlays at full res; H.264").font(theme::sans(10.5)).color(theme::DIM));
    ui.label(egui::RichText::new(format!("→ {}", ReplayState::export_dir(captures_dir).display())).font(theme::mono(10.0)).color(theme::DIM));
    ui.add_space(4.0);

    // Job status.
    if let Some(j) = &st.job {
        let s = j.status.load();
        ui.add(egui::ProgressBar::new(s.progress).show_percentage().desired_width(ui.available_width()));
        ui.label(egui::RichText::new(format!("{}: {}", s.kind, s.message)).font(theme::mono(10.5)).color(theme::GREEN));
        if ui.button("cancel").clicked() {
            j.cancel.store(true, Ordering::Relaxed);
        }
    } else if !st.last_job_msg.is_empty() {
        let col = if st.last_job_ok { theme::GREEN } else { theme::RED };
        ui.label(egui::RichText::new(st.last_job_msg.clone()).font(theme::mono(10.5)).color(col));
    }
}

// ---------------------------------------------------------------------------
// Tests (non-UI)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// 24-bit BGR bottom-up BMP (the capture format) from a gray frame.
    fn write_bmp_gray(path: &Path, g: &Gray) {
        let (w, h) = (g.w, g.h);
        let row_bytes = w * 3;
        let row_padded = (row_bytes + 3) & !3;
        let data_size = row_padded * h;
        let file_size = 54 + data_size;
        let mut out = Vec::with_capacity(file_size);
        out.extend_from_slice(b"BM");
        out.extend_from_slice(&(file_size as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&54u32.to_le_bytes());
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&(w as i32).to_le_bytes());
        out.extend_from_slice(&(h as i32).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        out.extend_from_slice(&[0u8; 24]);
        let pad = [0u8; 3];
        for y in (0..h).rev() {
            for x in 0..w {
                let v = g.data[y * w + x];
                out.extend_from_slice(&[v, v, v]);
            }
            out.extend_from_slice(&pad[..row_padded - row_bytes]);
        }
        std::fs::write(path, out).unwrap();
    }

    /// Deterministic textured frame (blurred hash noise) with an (ox, oy) crop offset.
    fn textured(w: usize, h: usize, ox: usize, oy: usize, seed: u64) -> Gray {
        let (bw, bh) = (w * 2, h * 2);
        let mut base = vec![0u8; bw * bh];
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        for v in base.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (s >> 56) as u8;
        }
        // 3x3 box blur so features track well.
        let mut blurred = vec![0u8; bw * bh];
        for y in 1..bh - 1 {
            for x in 1..bw - 1 {
                let mut acc = 0u32;
                for dy in 0..3 {
                    for dx in 0..3 {
                        acc += base[(y + dy - 1) * bw + x + dx - 1] as u32;
                    }
                }
                blurred[y * bw + x] = (acc / 9) as u8;
            }
        }
        let mut g = Gray::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = blurred[(y + oy) * bw + x + ox];
            }
        }
        g
    }

    fn frame_name(cam: usize, seq: usize, t: f64) -> String {
        let secs = t.floor();
        let us = ((t - secs) * 1e6).round() as i64;
        let d = fmt_date(secs).replace(['-', ' ', ':'], "_");
        format!("Camera{cam}_{seq:06}__{d}.{us:06}.bmp")
    }

    const T0: f64 = 1_783_040_523.0; // 2026-07-03 01:02:03 UTC

    fn synth_library() -> PathBuf {
        let root = std::env::temp_dir().join(format!("replay_rs_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&root).unwrap();
        let sat = root.join("TESTSAT_99999_2026_07_03_01_02_03");
        std::fs::create_dir_all(&sat).unwrap();
        for i in 0..6 {
            let t = T0 + i as f64 / 5.0;
            write_bmp_gray(&sat.join(frame_name(1, i + 1, t)), &textured(96, 64, 4 + i, 6 + (i % 2), 7));
        }
        for i in 0..4 {
            let t = T0 + i as f64 / 3.0;
            write_bmp_gray(&sat.join(frame_name(2, i + 1, t)), &textured(64, 48, 2 + i, 3, 11));
        }
        let mut csv = String::from("timestamp,satellite_name,altitude_deg,azimuth_deg,distance_km,pixel_x,pixel_y,sequence_in_capture,camera_index\n");
        for i in 0..8 {
            let t = T0 - 1.0 + i as f64 / 2.0;
            let secs = t.floor();
            let iso = format!("{}T{}.{:06}+00:00", fmt_date(secs).replace(' ', "T").split('T').next().unwrap(), fmt_date(secs).split(' ').nth(1).unwrap(), ((t - secs) * 1e6).round() as i64);
            csv.push_str(&format!("{iso},TESTSAT,{},{},{},{},{},-1,-1\n", 40.0 + i as f64 * 0.5, 120.0 + i as f64, 800.0 - i as f64 * 5.0, 100.0 + i as f64 * 2.0, 80.0 - i as f64));
        }
        std::fs::write(sat.join("trajectory.csv"), csv).unwrap();
        let manual = root.join("manual_2026_07_03_02_00_00");
        std::fs::create_dir_all(&manual).unwrap();
        for i in 0..2 {
            let t = T0 + 3600.0 + i as f64 / 5.0;
            write_bmp_gray(&manual.join(frame_name(1, i + 1, t)), &textured(32, 24, i, 0, 3));
        }
        // A folder without frames must not surface.
        std::fs::create_dir_all(root.join("empty_2026_07_03_03_00_00")).unwrap();
        root
    }

    #[test]
    fn time_parsing() {
        let t = parse_frame_time("2025_09_16_03_47_22.342929").unwrap();
        assert!((t - 1757994442.342929).abs() < 1e-5, "{t}");
        assert_eq!(parse_frame_time("2026_07_03_03_00_00"), Some(civil_to_epoch(2026, 7, 3, 3, 0, 0.0)));
        let a = parse_iso_time("2025-09-16T17:05:55.450644+00:00Z").unwrap();
        let b = parse_iso_time("2025-09-16T17:05:55.450644+00:00").unwrap();
        let c = parse_iso_time("2025-09-16T17:05:55.450644Z").unwrap();
        assert!((a - b).abs() < 1e-6 && (a - c).abs() < 1e-6);
        let d = parse_iso_time("2025-09-16T19:05:55.450644+02:00").unwrap();
        assert!((d - a).abs() < 1e-6);
        assert_eq!(fmt_utc(1757994442.342929), "03:47:22.342");
        assert_eq!(fmt_date(1757994442.0), "2025-09-16 03:47:22");
        assert_eq!(parse_frame_name("Camera1_000069__2026_08_12_04_48_51.710298.bmp").map(|(c, s, _)| (c, s)), Some((0, 69)));
        assert_eq!(parse_frame_name("Camera2_000001__2026_07_03_03_00_00Z.png").map(|(c, s, _)| (c, s)), Some((1, 1)));
        assert!(parse_frame_name("trajectory.csv").is_none());
        assert!(parse_frame_name("YAOGAN_trackingvis_2026.png").is_none());
    }

    #[test]
    fn library_scan_synthetic() {
        let root = synth_library();
        let runs = scan_library(&root);
        assert_eq!(runs.len(), 2, "{:?}", runs.iter().map(|r| &r.folder).collect::<Vec<_>>());
        // Newest first: the manual run (02:00) before the sat run (01:02).
        assert!(runs[0].is_manual && runs[0].target == "Manual");
        let sat = &runs[1];
        assert_eq!(sat.target, "TESTSAT");
        assert_eq!(sat.norad.as_deref(), Some("99999"));
        assert_eq!(sat.cams.len(), 2);
        assert_eq!(sat.cams[0].cam_index, 0);
        assert_eq!(sat.cams[0].frames.len(), 6);
        assert_eq!(sat.cams[1].frames.len(), 4);
        assert!((sat.t0().unwrap() - T0).abs() < 1e-3);
        assert!((sat.duration() - 1.0).abs() < 1e-3, "{}", sat.duration());
        // frame_index_at: clamped + monotone, both cams synced by time.
        assert_eq!(sat.frame_index_at(0, T0 - 100.0), Some(0));
        assert_eq!(sat.frame_index_at(0, T0 + 100.0), Some(5));
        assert_eq!(sat.frame_index_at(0, T0 + 0.45), Some(2));
        assert_eq!(sat.frame_index_at(1, T0 + 0.5), Some(1));
        // Trajectory parsed + interpolated; vectors available.
        assert!(sat.trajectory.valid());
        let s = sat.trajectory.interp(T0).unwrap();
        assert!((s.az - 122.0).abs() < 1e-6 && (s.el - 41.0).abs() < 1e-6, "{s:?}");
        assert!(compute_track_vectors(&sat.trajectory, T0 + 0.5, 640.0, 360.0).is_some());
        // Sidecar round trip.
        let mut sc = sat.sidecar.clone();
        sc.display_name = "UNIT".into();
        sc.favorite = true;
        sc.annotations.push(Annotation::Box { cam: 0, x0: 0.1, y0: 0.2, x1: 0.3, y1: 0.4 });
        sc.save(&sat.sidecar_path()).unwrap();
        let fresh = RunInfo::open(&sat.path).unwrap();
        assert_eq!(fresh.display_name(), "UNIT");
        assert!(fresh.sidecar.favorite);
        assert_eq!(fresh.sidecar.annotations, sc.annotations);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lut_gamma_brightness_contrast() {
        let id = build_lut(1.0, 0.0, 1.0);
        assert!(id.iter().enumerate().all(|(i, &v)| v as usize == i));
        let g = build_lut(2.2, 0.0, 1.0);
        assert!(g[50] > 50 && g[0] == 0 && g[255] == 255);
        let b = build_lut(1.0, 40.0, 1.0);
        assert_eq!(b[50], 90);
        assert_eq!(b[250], 255);
        let c = build_lut(1.0, 0.0, 2.0);
        assert_eq!(c[128], 128);
        assert!(c[200] > 200 && c[50] < 50);
        assert!(is_default_params(1.0, 0.0, 1.0) && !is_default_params(1.2, 0.0, 1.0));
    }

    #[test]
    fn frame_loading_and_reduce() {
        let root = std::env::temp_dir().join(format!("replay_rs_load_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut g = Gray::new(37, 21);
        for (i, v) in g.data.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let p = root.join("Camera1_000001__2026_07_03_01_02_03.000000.bmp");
        write_bmp_gray(&p, &g);
        let back = load_gray(&p).unwrap();
        assert_eq!(back, g);
        assert_eq!(reduce_for(1936, 640), 2);
        assert_eq!(reduce_for(1936, 1000), 1);
        assert_eq!(reduce_for(3096, 300), 8);
        let d = g.downsample(2);
        assert_eq!((d.w, d.h), (18, 10));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stack_identical_frames_is_identity() {
        let f = textured(120, 90, 5, 7, 42);
        let frames = vec![f.clone(), f.clone(), f.clone(), f.clone()];
        let (mean, n_stacked, n_rejected) = stack_gray(&frames, 0, true, StabSettings::default()).unwrap();
        assert_eq!(n_stacked, 4, "rejected {n_rejected}");
        let out: Vec<u8> = mean.iter().map(|v| v.round() as u8).collect();
        let diff = out.iter().zip(f.data.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{diff} pixels differ");
        // Unaligned average of identical frames is exactly the frame too.
        let (mean2, ..) = stack_gray(&frames, 0, false, StabSettings::default()).unwrap();
        assert!(mean2.iter().zip(f.data.iter()).all(|(a, b)| (a - *b as f64).abs() < 1e-9));
    }

    #[test]
    fn precull_center_helpers() {
        // Content: a bright blob scores far above flat frames.
        let mut blob = Gray::new(64, 64);
        for y in 28..36 {
            for x in 28..36 {
                blob.data[y * 64 + x] = 250;
            }
        }
        let flat = Gray::new(64, 64);
        assert!(content_score_gray(&blob) > 10.0 * (content_score_gray(&flat) + 1e-9));
        // Centroid lands on the blob; a flat frame has no target.
        let (cx, cy) = brightness_centroid_gray(&blob).unwrap();
        assert!((cx - 31.5).abs() < 1.0 && (cy - 31.5).abs() < 1.0, "{cx} {cy}");
        assert!(brightness_centroid_gray(&flat).is_none());
        // Centred crop puts the blob centre on the output centre pixel, zero-padded.
        let c = crop_centered_gray(&blob, cx, cy, 16);
        assert_eq!((c.w, c.h), (16, 16));
        assert!(c.data[8 * 16 + 8] == 250);
        let edge = crop_centered_gray(&blob, 0.0, 0.0, 16);
        assert_eq!(edge.data[0], 0); // off-frame area padded
        // Centered stack through the job: output is center_size².
        let root = synth_library();
        let runs = scan_library(&root);
        let sat = Arc::new(runs.into_iter().find(|r| r.target == "TESTSAT").unwrap());
        let out_dir = root.join("exports");
        let status = Arc::new(ArcSwap::from_pointee(JobStatus::default()));
        let cancel = AtomicBool::new(false);
        let spec = StackSpec { run: sat.clone(), slot: 0, t_start: sat.t0().unwrap(), t_end: sat.t1().unwrap(), keep_n: 3, stab: StabSettings::default(), out_base: out_dir.join("stack_ctr"), centered: true, center_size: 48 };
        let final_path = run_stack(&spec, &status, &cancel).unwrap();
        let img = image::open(&final_path).unwrap();
        assert_eq!((img.width(), img.height()), (48, 48));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stabilizer_reduces_shift() {
        let base = textured(160, 120, 20, 20, 9);
        let shifted = textured(160, 120, 26, 17, 9); // +6, -3 px
        let st = Stabilizer::new(&base, StabSettings::default());
        let (warped, info) = st.stabilize(&shifted);
        assert!(info.ok, "{:?}", info.reason);
        let mad = |a: &Gray, b: &Gray| a.data.iter().zip(b.data.iter()).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.data.len() as f64;
        // Compare the interior (warp reveals a black border).
        let crop = |g: &Gray| {
            let mut o = Gray::new(120, 90);
            for y in 0..90 {
                for x in 0..120 {
                    o.data[y * 120 + x] = g.data[(y + 15) * 160 + x + 20];
                }
            }
            o
        };
        let before = mad(&crop(&base), &crop(&shifted));
        let after = mad(&crop(&base), &crop(&warped));
        assert!(after < before * 0.5, "before {before} after {after}");
    }

    #[test]
    fn export_and_stack_jobs() {
        let root = synth_library();
        let runs = scan_library(&root);
        let sat = Arc::new(runs.into_iter().find(|r| r.target == "TESTSAT").unwrap());
        let out_dir = root.join("exports");
        let status = Arc::new(ArcSwap::from_pointee(JobStatus::default()));
        let cancel = AtomicBool::new(false);
        // Stack best 4 of cam 1 over the whole run.
        let spec = StackSpec { run: sat.clone(), slot: 0, t_start: sat.t0().unwrap(), t_end: sat.t1().unwrap(), keep_n: 4, stab: StabSettings::default(), out_base: out_dir.join("stack_test"), centered: false, center_size: 512 };
        let final_path = run_stack(&spec, &status, &cancel).unwrap();
        assert!(final_path.exists() && out_dir.join("stack_test.png").exists());
        let master = image::open(out_dir.join("stack_test.png")).unwrap();
        assert_eq!((master.width(), master.height()), (96, 64));
        assert!(status.load().message.starts_with("Stacked"), "{}", status.load().message);
        // MP4 export of the first 5 frames with everything on.
        #[cfg(feature = "mp4-export")]
        {
            let mut params = ProcParams::default();
            params.gamma = 2.0;
            params.stabilize = true;
            params.sharpen.on = true;
            let t_end = sat.cams[0].frames[4].t;
            let spec = ExportSpec { run: sat.clone(), slot: 0, t_start: sat.t0().unwrap(), t_end, out_path: out_dir.join("clip.mp4"), params, overlays: true, annotations: vec![Annotation::Text { cam: 0, x: 0.5, y: 0.5, text: "T".into() }], fps: None, crop: None };
            let p = run_export(&spec, &status, &cancel).unwrap();
            let len = std::fs::metadata(&p).unwrap().len();
            assert!(len > 500, "mp4 too small: {len}");
            assert!((status.load().progress - 1.0).abs() < 1e-6);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn overlays_bake_and_fps() {
        let (w, h) = (200, 120);
        let mut rgb = vec![0u8; w * h * 3];
        let v = TrackVectors { anchor: [100.0, 60.0], intrack: [140.0, 40.0], cross_p: [110.0, 80.0], cross_n: [90.0, 40.0] };
        let ann = vec![Annotation::Text { cam: 0, x: 0.1, y: 0.5, text: "Hello 123".into() }, Annotation::Box { cam: 0, x0: 0.2, y0: 0.2, x1: 0.4, y1: 0.4 }, Annotation::Arrow { cam: 1, x0: 0.0, y0: 0.0, x1: 1.0, y1: 1.0 }];
        bake_overlays(&mut rgb, w, h, Some(&v), &["Cam1 00:00:00.000 UTC [1/5]".to_string()], &ann, 0);
        assert!(rgb.iter().any(|&b| b != 0));
        // Arrow for cam 1 must not be drawn on cam 0: the far corner stays black.
        let i = ((h - 1) * w + (w - 1)) * 3;
        assert_eq!(&rgb[i..i + 3], &[0, 0, 0]);
        assert!((infer_fps(&[0.0, 0.2, 0.4, 0.6]) - 5.0).abs() < 1e-9);
        assert_eq!(infer_fps(&[0.0]), 5.0);
        assert_eq!(infer_fps(&[0.0, 0.002]), 60.0); // 500 fps clamps to 60
        assert_eq!(infer_fps(&[0.0, 0.0005]), 5.0); // sub-ms interval: undeterminable
    }
}
