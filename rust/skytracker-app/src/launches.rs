//! Rocket launch trajectories (port of trajectory.read_launch_trajectories):
//! `launches/*.txt` = "#Unix T+0 Time (s)" + T0, then rows of
//! "dt x y z vx vy vz" (ECEF metres, m/s); `launches/*.csv` = columns
//! time (unix s), elevationDegs, azimuthDegs, rangeKm. Converted to
//! topocentric (unix, az, el, range_km) rows for the skyplot overlay and
//! PROGRAM tracking (`launch:<name>` selection keys).

use crate::state::LaunchTrack;
use skytracker_adsb::geom::{ecef_to_enu, enu_to_azel_range, geodetic_to_ecef};
use skytracker_astro::sgp4_pass::Observer;
use std::path::Path;

pub fn load_dir(dir: &Path, observer: &Observer) -> Vec<LaunchTrack> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    let (ox, oy, oz) = geodetic_to_ecef(observer.lat_deg, observer.lon_deg, observer.elevation_m);
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let rows: Vec<(f64, f64, f64, f64)> = match ext.to_ascii_lowercase().as_str() {
            "txt" => parse_txt(&text)
                .into_iter()
                .map(|(t, x, y, z)| {
                    let (e, n, u) = ecef_to_enu(x - ox, y - oy, z - oz, observer.lat_deg, observer.lon_deg);
                    let (az, el, rng) = enu_to_azel_range(e, n, u);
                    (t, az, el, rng)
                })
                .collect(),
            "csv" => parse_csv(&text),
            _ => continue,
        };
        if rows.len() >= 2 {
            out.push(LaunchTrack { key: format!("launch:{stem}"), name: stem.to_string(), rows });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// (unix, x, y, z) ECEF metres.
pub fn parse_txt(text: &str) -> Vec<(f64, f64, f64, f64)> {
    let mut t0: Option<f64> = None;
    let mut rows = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let f: Vec<f64> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if t0.is_none() && f.len() == 1 {
            t0 = Some(f[0]);
            continue;
        }
        if f.len() >= 4 {
            if let Some(t0) = t0 {
                rows.push((t0 + f[0], f[1], f[2], f[3]));
            }
        }
    }
    rows
}

/// (unix, az, el, range_km) from a CSV with time/elevationDegs/azimuthDegs/rangeKm.
pub fn parse_csv(text: &str) -> Vec<(f64, f64, f64, f64)> {
    let mut lines = text.lines();
    let Some(header) = lines.next() else { return Vec::new() };
    let cols: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
    let idx = |n: &str| cols.iter().position(|c| *c == n);
    let (Some(it), Some(ie), Some(ia), Some(ir)) = (idx("time"), idx("elevationDegs"), idx("azimuthDegs"), idx("rangeKm")) else { return Vec::new() };
    lines
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Some((f.get(it)?.trim().parse().ok()?, f.get(ia)?.trim().parse().ok()?, f.get(ie)?.trim().parse().ok()?, f.get(ir)?.trim().parse().ok()?))
        })
        .collect()
}

/// Linear interpolation of (az, el, range) at unix time `t` (None outside the track).
pub fn interp(l: &LaunchTrack, t: f64) -> Option<(f64, f64, f64)> {
    let rows = &l.rows;
    if rows.len() < 2 || t < rows[0].0 || t > rows[rows.len() - 1].0 {
        return None;
    }
    let i = rows.partition_point(|r| r.0 <= t).clamp(1, rows.len() - 1);
    let (a, b) = (rows[i - 1], rows[i]);
    let f = if b.0 > a.0 { (t - a.0) / (b.0 - a.0) } else { 0.0 };
    let daz = (b.1 - a.1 + 540.0).rem_euclid(360.0) - 180.0;
    Some(((a.1 + daz * f).rem_euclid(360.0), a.2 + (b.2 - a.2) * f, a.3 + (b.3 - a.3) * f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_txt_and_interpolates() {
        let text = "#Unix T+0 Time (s)\n1000.0\n\n# Delta T+ Time (s), ECEF ...\n0.0 1 2 3 0 0 0\n10.0 4 5 6 0 0 0\n";
        let rows = parse_txt(text);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], (1000.0, 1.0, 2.0, 3.0));
        let l = LaunchTrack { key: "launch:x".into(), name: "x".into(), rows: vec![(1000.0, 10.0, 5.0, 100.0), (1010.0, 20.0, 15.0, 200.0)] };
        let (az, el, r) = interp(&l, 1005.0).unwrap();
        assert!((az - 15.0).abs() < 1e-9 && (el - 10.0).abs() < 1e-9 && (r - 150.0).abs() < 1e-9);
        assert!(interp(&l, 999.0).is_none());
    }

    #[test]
    fn parses_repo_launch_files_if_present() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let obs = Observer { lat_deg: 34.874, lon_deg: -120.446, elevation_m: 100.0 };
        let l = load_dir(&root.join("launches"), &obs);
        for t in &l {
            assert!(t.rows.len() >= 2);
            assert!(t.rows.windows(2).all(|w| w[1].0 >= w[0].0), "{} times monotonic", t.name);
        }
    }
}
