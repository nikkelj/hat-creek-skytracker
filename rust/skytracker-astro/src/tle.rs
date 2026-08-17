//! TLE catalog handling: parse a Celestrak-style 3-line-element file into
//! propagation-ready satellites, keyed by the 5-character NORAD field
//! (`satnum_str` in python-sgp4 terms, alpha-5 compatible).
//!
//! HTTP fetch stays in Python (satellite_data.py) during the strangler
//! period — it is not performance-relevant; a native fetch arrives with
//! the Phase 7 app.

use crate::sgp4_pass::{SatError, Satellite};

pub struct TleCatalog {
    pub sats: Vec<Satellite>,
    /// NORAD field (line-1 cols 3-7, trimmed) -> index into `sats`.
    index: std::collections::HashMap<String, usize>,
}

/// Split raw TLE text into (name, line1, line2) triplets. Handles the
/// repo cache's `\r\r\n` endings by dropping blank lines, and both 3LE
/// (name line) and bare 2LE formats.
pub fn parse_tle_text(text: &str) -> Vec<(String, String, String)> {
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim_end())
        .filter(|l| !l.trim().is_empty())
        .collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("1 ")
            && i + 1 < lines.len()
            && lines[i + 1].starts_with("2 ")
        {
            out.push((String::new(), lines[i].to_string(), lines[i + 1].to_string()));
            i += 2;
        } else if i + 2 < lines.len()
            && lines[i + 1].starts_with("1 ")
            && lines[i + 2].starts_with("2 ")
        {
            out.push((
                lines[i].trim().to_string(),
                lines[i + 1].to_string(),
                lines[i + 2].to_string(),
            ));
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// The 5-character NORAD catalog field from line 1 (alpha-5 safe).
pub fn satnum_str(line1: &str) -> String {
    line1.get(2..7).unwrap_or("").trim().to_string()
}

impl TleCatalog {
    /// Parse a TLE file. Satellites whose elements fail SGP4
    /// initialization (decayed/malformed) are skipped, matching
    /// skyfield's tolerant loader.
    pub fn load(path: &std::path::Path) -> Result<Self, SatError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SatError::Tle(format!("read {path:?}: {e}")))?;
        Ok(Self::from_text(&text))
    }

    pub fn from_text(text: &str) -> Self {
        let mut sats = Vec::new();
        let mut index = std::collections::HashMap::new();
        for (name, l1, l2) in parse_tle_text(text) {
            let key = satnum_str(&l1);
            if key.is_empty() {
                continue;
            }
            match Satellite::from_tle(&name, &l1, &l2) {
                Ok(sat) => {
                    index.insert(key, sats.len());
                    sats.push(sat);
                }
                Err(_) => continue,
            }
        }
        TleCatalog { sats, index }
    }

    pub fn get(&self, satnum: &str) -> Option<&Satellite> {
        self.index.get(satnum).map(|&i| &self.sats[i])
    }

    pub fn len(&self) -> usize {
        self.sats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sats.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_crlf_cache_format() {
        let text = "CALSPHERE 1             \r\r\n1 00900U 64063C   26209.82932116  .00000536  00000+0  53569-3 0  9993\r\r\n2 00900  90.2197  72.7437 0025676 161.4653 332.3640 13.76663486 77152\r\r\n";
        let triplets = parse_tle_text(text);
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0].0, "CALSPHERE 1");
        assert_eq!(satnum_str(&triplets[0].1), "00900");
    }
}
