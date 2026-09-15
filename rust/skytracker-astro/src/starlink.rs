//! Starlink public ephemerides (https://api.starlink.com/public-files/
//! ephemerides/): MANIFEST.txt lists the freshest file per satellite,
//! `MEME_<NORAD>_<NAME>_..._UNCLASSIFIED.txt` holds ~72 h of J2000 (MEME)
//! state vectors at 60 s steps plus covariance rows. Starlink maneuvers
//! often enough that TLEs go stale in hours — tracking from these
//! ephemerides survives the burns.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct EphemState {
    pub t_unix: f64,
    /// Geocentric position, km, MEME/J2000 axes (treated as ICRS: the frame
    /// bias is ~0.02″, far below TLE-vs-truth for these birds).
    pub pos_km: [f64; 3],
    pub vel_kms: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct Ephemeris {
    pub norad: String,
    pub name: String,
    pub start_unix: f64,
    pub stop_unix: f64,
    pub step_s: f64,
    pub states: Vec<EphemState>,
}

/// MANIFEST.txt -> NORAD -> (filename, satellite name). Lines look like
/// `MEME_58165_STARLINK-30814_2380241_Operational_1472006520_UNCLASSIFIED.txt`.
pub fn parse_manifest(text: &str) -> HashMap<String, (String, String)> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let f = line.trim();
        if f.is_empty() {
            continue;
        }
        let parts: Vec<&str> = f.split('_').collect();
        if parts.len() >= 3 && parts[0] == "MEME" {
            out.insert(parts[1].to_string(), (f.to_string(), parts[2].to_string()));
        }
    }
    out
}

/// `YYYYDDDHHMMSS.SSS` (UTC) -> unix seconds.
fn parse_epoch(s: &str) -> Option<f64> {
    let (intpart, frac) = match s.split_once('.') {
        Some((a, b)) => (a, format!("0.{b}").parse::<f64>().ok()?),
        None => (s, 0.0),
    };
    if intpart.len() != 13 {
        return None;
    }
    let year: i64 = intpart[0..4].parse().ok()?;
    let doy: i64 = intpart[4..7].parse().ok()?;
    let hh: i64 = intpart[7..9].parse().ok()?;
    let mm: i64 = intpart[9..11].parse().ok()?;
    let ss: i64 = intpart[11..13].parse().ok()?;
    // Howard Hinnant days_from_civil for Jan 1 of `year`.
    let (y, m, d) = (year, 1i64, 1i64);
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (m + 9) % 12;
    let doy0 = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy0;
    let days = era * 146_097 + doe - 719_468;
    Some(((days + doy - 1) * 86400 + hh * 3600 + mm * 60 + ss) as f64 + frac)
}

/// Parse a MEME ephemeris file: 4 header lines, then blocks of
/// `epoch x y z vx vy vz` followed by 3 covariance lines (skipped).
pub fn parse_ephemeris(text: &str, norad: &str, name: &str) -> Result<Ephemeris, String> {
    let mut step_s = 60.0;
    let mut states: Vec<EphemState> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix("ephemeris_start:") {
            if let Some(i) = rest.find("step_size:") {
                step_s = rest[i + 10..].trim().parse().unwrap_or(60.0);
            }
            continue;
        }
        if t.starts_with("created:") || t.starts_with("ephemeris_source:") || t.starts_with("UVW") {
            continue;
        }
        let cols: Vec<&str> = t.split_whitespace().collect();
        // State rows have 7 columns and a 13-digit integer epoch; covariance
        // rows have 7 exponent-notation columns — the epoch test separates them.
        if cols.len() == 7 && cols[0].split_once('.').map_or(false, |(a, _)| a.len() == 13 && a.chars().all(|c| c.is_ascii_digit())) {
            let t_unix = parse_epoch(cols[0]).ok_or_else(|| format!("bad epoch {}", cols[0]))?;
            let mut v = [0.0f64; 6];
            for (i, c) in cols[1..7].iter().enumerate() {
                v[i] = c.parse().map_err(|_| format!("bad number {c}"))?;
            }
            states.push(EphemState { t_unix, pos_km: [v[0], v[1], v[2]], vel_kms: [v[3], v[4], v[5]] });
        }
    }
    if states.len() < 2 {
        return Err(format!("only {} state rows", states.len()));
    }
    Ok(Ephemeris {
        norad: norad.to_string(),
        name: name.to_string(),
        start_unix: states.first().unwrap().t_unix,
        stop_unix: states.last().unwrap().t_unix,
        step_s,
        states,
    })
}

/// Cubic-Hermite interpolated geocentric position (km) at `t_unix`; None
/// outside the ephemeris span.
pub fn position_at(eph: &Ephemeris, t_unix: f64) -> Option<[f64; 3]> {
    let s = &eph.states;
    if t_unix < s.first()?.t_unix || t_unix > s.last()?.t_unix {
        return None;
    }
    let i = s.partition_point(|p| p.t_unix <= t_unix).min(s.len() - 1).max(1);
    let (a, b) = (&s[i - 1], &s[i]);
    let h = b.t_unix - a.t_unix;
    if h <= 0.0 {
        return Some(a.pos_km);
    }
    let u = ((t_unix - a.t_unix) / h).clamp(0.0, 1.0);
    let (u2, u3) = (u * u, u * u * u);
    let (h00, h10, h01, h11) = (2.0 * u3 - 3.0 * u2 + 1.0, u3 - 2.0 * u2 + u, -2.0 * u3 + 3.0 * u2, u3 - u2);
    let mut out = [0.0f64; 3];
    for k in 0..3 {
        out[k] = h00 * a.pos_km[k] + h10 * h * a.vel_kms[k] + h01 * b.pos_km[k] + h11 * h * b.vel_kms[k];
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_parses() {
        // 2026 DOY 238 = 2026-08-26, 02:41:42 UTC.
        let t = parse_epoch("2026238024142.000").unwrap();
        assert_eq!(t as i64 % 86400, 2 * 3600 + 41 * 60 + 42);
        let days = (t as i64) / 86400;
        // 2026-08-26 is 20691 days after 1970-01-01... assert via round trip:
        // day-of-year arithmetic: Jan 1 2026 unix / 86400 + 237.
        let jan1 = parse_epoch("2026001000000.000").unwrap() as i64 / 86400;
        assert_eq!(days, jan1 + 237);
    }

    #[test]
    fn manifest_and_file_parse() {
        let man = parse_manifest("MEME_58165_STARLINK-30814_2380241_Operational_1472006520_UNCLASSIFIED.txt\n\n");
        assert_eq!(man.get("58165").unwrap().1, "STARLINK-30814");
        let text = "created:2026-08-26 02:53:42 UTC\n\
ephemeris_start:2026-08-26 02:41:42 UTC ephemeris_stop:2026-08-29 02:41:42 UTC step_size:60\n\
ephemeris_source:blend\n\
UVW\n\
2026238024142.000 -6301.0394867531 2423.0414565903 1118.1878700790 -0.7285750589 -4.6813689446 5.9883496170\n\
4.6707008070e-07 -3.8301441937e-07 7.7378163772e-07 -1.8158714709e-10 -3.0323378290e-11 1.2330782645e-06 8.1966398458e-10\n\
-9.1118997424e-10 9.8806309418e-14 1.9512540419e-12 -4.6458088031e-10 4.0767862264e-10 -2.1698762285e-13 -8.2540761046e-13\n\
5.0466444694e-13 -6.9338141163e-13 3.0876558192e-13 1.6561528303e-09 -3.5825323486e-16 1.4687548089e-16 5.4309613222e-12\n\
2026238024242.000 -6330.5990327911 2136.9385193023 1474.7064931706 -0.2563696117 -4.8518304266 5.8911430010\n\
4.6707008070e-07 -3.8301441937e-07 7.7378163772e-07 -1.8158714709e-10 -3.0323378290e-11 1.2330782645e-06 8.1966398458e-10\n\
-9.1118997424e-10 9.8806309418e-14 1.9512540419e-12 -4.6458088031e-10 4.0767862264e-10 -2.1698762285e-13 -8.2540761046e-13\n\
5.0466444694e-13 -6.9338141163e-13 3.0876558192e-13 1.6561528303e-09 -3.5825323486e-16 1.4687548089e-16 5.4309613222e-12\n";
        let e = parse_ephemeris(text, "58165", "STARLINK-30814").unwrap();
        assert_eq!(e.states.len(), 2);
        assert!((e.stop_unix - e.start_unix - 60.0).abs() < 1e-6);
        // Node hit is exact; the midpoint stays near the LEO shell radius.
        let p0 = position_at(&e, e.start_unix).unwrap();
        assert!((p0[0] - -6301.0394867531).abs() < 1e-9);
        let pm = position_at(&e, e.start_unix + 30.0).unwrap();
        let r = (pm[0] * pm[0] + pm[1] * pm[1] + pm[2] * pm[2]).sqrt();
        assert!((6800.0..7100.0).contains(&r), "midpoint radius {r}");
        assert!(position_at(&e, e.start_unix - 1.0).is_none());
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    use crate::apparent::FrameContext;
    use crate::sgp4_pass::{satellite_altaz, Observer};
    use crate::tle::TleCatalog;
    use crate::time;

    /// Cross-check the ephemeris frame chain against SGP4 on the same
    /// satellite: run with SKYTRACKER_EPH_SAMPLE=<file> and a fresh
    /// tle_cache.tle. Agreement is bounded by TLE staleness (deg-level for
    /// Starlink); a frame-chain bug would be tens of degrees.
    #[test]
    #[ignore]
    fn ephemeris_vs_tle_cross_check() {
        let path = std::env::var("SKYTRACKER_EPH_SAMPLE").expect("set SKYTRACKER_EPH_SAMPLE");
        let tle_path = std::env::var("SKYTRACKER_TLE_CACHE").expect("set SKYTRACKER_TLE_CACHE");
        let text = std::fs::read_to_string(&path).unwrap();
        let norad = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().split('_').nth(1).unwrap().to_string();
        let eph = parse_ephemeris(&text, &norad, "sample").unwrap();
        let cat = TleCatalog::load(std::path::Path::new(&tle_path)).unwrap();
        let sat = cat.get(&norad).expect("satellite not in tle_cache");
        let observer = Observer { lat_deg: 34.874, lon_deg: -120.446, elevation_m: 100.0 };
        let geom = observer.geometry();
        let mut max_sep = 0.0f64;
        for k in 0..12 {
            let t_unix = eph.start_unix + 3600.0 * (k as f64) * 0.5;
            if t_unix > eph.stop_unix {
                break;
            }
            let jd_utc = 2440587.5 + t_unix / 86400.0;
            let jd_tt = time::utc_to_tt(jd_utc);
            let p = position_at(&eph, t_unix).unwrap();
            let ctx = FrameContext::new(jd_tt);
            let (el_e, az_e, rng_e) = ctx.altaz_range_of_position(&p, &geom);
            let (el_t, az_t, rng_t) = satellite_altaz(sat, jd_tt, &geom).unwrap();
            let daz = ((az_e - az_t + 540.0).rem_euclid(360.0) - 180.0) * el_t.to_radians().cos();
            let sep = (daz * daz + (el_e - el_t) * (el_e - el_t)).sqrt();
            eprintln!("t+{:>5.1}h  eph az/el {az_e:8.3}/{el_e:7.3} rng {rng_e:7.1}  tle {az_t:8.3}/{el_t:7.3} rng {rng_t:7.1}  sep {sep:7.4}°", (t_unix - eph.start_unix) / 3600.0);
            max_sep = max_sep.max(sep);
        }
        eprintln!("max separation {max_sep:.4}°");
        assert!(max_sep < 5.0, "frame-chain error? max separation {max_sep:.3}°");
    }
}
