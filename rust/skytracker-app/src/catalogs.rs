//! Catalogue loaders for the skyplot layers: IAU star names (HIP -> proper
//! name, catalogs/iau-csn.txt) and OpenNGC (Messier via the M cross-ref
//! column + the M45 patch, and NGC objects with a finite magnitude) —
//! ports of star_catalog.iau_star_names / celestial._load_openngc.

use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Dso {
    /// Selection key: "dso:M031" / "dso:NGC0224".
    pub key: String,
    pub name: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub mag: f64,
    pub messier: bool,
}

/// HIP -> IAU proper name. Token-based from the right (the name field may
/// contain spaces): ... Vmag band HIP HD RA Dec DATE [*].
pub fn iau_star_names(path: &Path) -> HashMap<i64, String> {
    let mut out = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else { return out };
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with('$') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 7 {
            continue;
        }
        let Some(date_idx) = (0..tokens.len()).rev().find(|&i| is_date(tokens[i])) else { continue };
        if date_idx < 4 {
            continue;
        }
        // ... HIP HD RA Dec DATE
        let hip: Option<i64> = tokens[date_idx - 4].parse().ok();
        let Some(hip) = hip else { continue };
        // Name = the fixed-width first column (18 chars), which keeps
        // multi-word names together.
        let name = line.get(..18).unwrap_or(tokens[0]).trim().to_string();
        if !name.is_empty() && name != "_" {
            out.entry(hip).or_insert(name);
        }
    }
    out
}

fn is_date(t: &str) -> bool {
    t.len() == 10 && t.as_bytes()[4] == b'-' && t.as_bytes()[7] == b'-' && t[..4].chars().all(|c| c.is_ascii_digit())
}

fn parse_hms(s: &str) -> Option<f64> {
    let p: Vec<f64> = s.split(':').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    if p.len() != 3 {
        return None;
    }
    Some((p[0] + p[1] / 60.0 + p[2] / 3600.0) * 15.0)
}

fn parse_dms(s: &str) -> Option<f64> {
    let neg = s.trim().starts_with('-');
    let p: Vec<f64> = s.trim().trim_start_matches(['+', '-']).split(':').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    if p.len() != 3 {
        return None;
    }
    let v = p[0] + p[1] / 60.0 + p[2] / 3600.0;
    Some(if neg { -v } else { v })
}

/// OpenNGC -> (Messier objects, NGC objects).
pub fn load_openngc(path: &Path) -> (Vec<Dso>, Vec<Dso>) {
    let mut messier = Vec::new();
    let mut ngc = Vec::new();
    let Ok(text) = std::fs::read_to_string(path) else { return (messier, ngc) };
    let mut lines = text.lines();
    let Some(header) = lines.next() else { return (messier, ngc) };
    let cols: Vec<&str> = header.split(';').collect();
    let idx = |n: &str| cols.iter().position(|c| *c == n);
    let (Some(i_name), Some(i_ra), Some(i_dec)) = (idx("Name"), idx("RA"), idx("Dec")) else { return (messier, ngc) };
    let i_v = idx("V-Mag");
    let i_b = idx("B-Mag");
    let i_m = idx("M");
    let i_common = idx("Common names");
    for line in lines {
        let f: Vec<&str> = line.split(';').collect();
        let get = |i: Option<usize>| i.and_then(|i| f.get(i)).map(|s| s.trim()).unwrap_or("");
        let (Some(ra), Some(dec)) = (parse_hms(get(Some(i_ra))), parse_dms(get(Some(i_dec)))) else { continue };
        let mag = get(i_v).parse::<f64>().ok().or_else(|| get(i_b).parse::<f64>().ok());
        let common = get(i_common).split(',').next().unwrap_or("").trim().to_string();
        let m = get(i_m);
        if !m.is_empty() {
            if let Ok(n) = m.parse::<u32>() {
                let label = format!("M{n}");
                let name = if common.is_empty() { label.clone() } else { format!("{label} {common}") };
                messier.push(Dso { key: format!("dso:M{m}"), name, ra_deg: ra, dec_deg: dec, mag: mag.unwrap_or(9.9), messier: true });
            }
        }
        let name = get(Some(i_name));
        if let (true, Some(mag)) = (name.starts_with("NGC"), mag) {
            let disp = if common.is_empty() { name.to_string() } else { format!("{name} {common}") };
            ngc.push(Dso { key: format!("dso:{name}"), name: disp, ra_deg: ra, dec_deg: dec, mag, messier: false });
        }
    }
    // M45 (Pleiades) is not in OpenNGC as an M object (celestial.MESSIER_PATCH).
    if !messier.iter().any(|d| d.key == "dso:M045") {
        messier.push(Dso { key: "dso:M045".into(), name: "M45 Pleiades".into(), ra_deg: 56.75, dec_deg: 24.1167, mag: 1.6, messier: true });
    }
    messier.sort_by(|a, b| a.key.cmp(&b.key));
    (messier, ngc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repo_catalogs_if_present() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let names = iau_star_names(&root.join("catalogs/iau-csn.txt"));
        if !names.is_empty() {
            assert_eq!(names.get(&32349).map(|s| s.as_str()), Some("Sirius"));
            assert_eq!(names.get(&7588).map(|s| s.as_str()), Some("Achernar"));
        }
        let (m, n) = load_openngc(&root.join("catalogs/openngc.csv"));
        if !m.is_empty() {
            assert!(m.iter().any(|d| d.key == "dso:M031" && d.name.contains("Andromeda")), "M31 present");
            assert!(m.iter().any(|d| d.key == "dso:M045"));
            assert!(n.len() > 1000);
        }
    }

    #[test]
    fn sexagesimal() {
        assert!((parse_hms("00:42:44.3").unwrap() - 10.6846).abs() < 1e-3);
        assert!((parse_dms("+41:16:09").unwrap() - 41.2692).abs() < 1e-3);
        assert!((parse_dms("-12:49:22.3").unwrap() + 12.8229).abs() < 1e-3);
    }
}
