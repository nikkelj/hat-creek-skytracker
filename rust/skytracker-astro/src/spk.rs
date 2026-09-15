//! Minimal pure-Rust reader for JPL SPK ephemeris files (DAF container,
//! Type 2 Chebyshev-position segments) — exactly the subset `de421.bsp`
//! uses. Replaces jplephem for this app; validated segment-by-segment
//! against golden states recorded from jplephem (tests/golden/spk_states.npz).
//!
//! Format reference: NAIF SPK/DAF Required Reading. Only little-endian
//! ("LTL-IEEE") files are supported, which covers the shipped de421.bsp.

use std::path::Path;

#[derive(Debug)]
pub enum SpkError {
    Io(std::io::Error),
    Format(String),
    NoSegment { center: i32, target: i32 },
    OutOfRange { target: i32, jd_tdb: f64 },
}

impl std::fmt::Display for SpkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpkError::Io(e) => write!(f, "SPK I/O error: {e}"),
            SpkError::Format(s) => write!(f, "SPK format error: {s}"),
            SpkError::NoSegment { center, target } => {
                write!(f, "SPK: no segment {center} -> {target}")
            }
            SpkError::OutOfRange { target, jd_tdb } => {
                write!(f, "SPK: jd_tdb {jd_tdb} outside segment coverage for {target}")
            }
        }
    }
}

impl std::error::Error for SpkError {}

impl From<std::io::Error> for SpkError {
    fn from(e: std::io::Error) -> Self {
        SpkError::Io(e)
    }
}

/// One Type 2 segment: Chebyshev position records.
pub struct Segment {
    pub center: i32,
    pub target: i32,
    pub start_et_s: f64,
    pub end_et_s: f64,
    /// Record layout.
    init_s: f64,
    intlen_s: f64,
    rsize: usize,
    n_records: usize,
    /// All record data, `n_records * rsize` doubles.
    data: Vec<f64>,
}

pub struct Spk {
    pub segments: Vec<Segment>,
}

const RECORD_BYTES: usize = 1024;
const SECONDS_PER_DAY: f64 = 86400.0;
/// SPK epochs are TDB seconds past J2000 (JD 2451545.0 TDB).
const J2000_JD: f64 = 2451545.0;

fn f64_at(bytes: &[u8], word_index_1based: usize) -> f64 {
    let off = (word_index_1based - 1) * 8;
    f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
}

fn i32_at(bytes: &[u8], byte_off: usize) -> i32 {
    i32::from_le_bytes(bytes[byte_off..byte_off + 4].try_into().unwrap())
}

impl Spk {
    pub fn open(path: &Path) -> Result<Self, SpkError> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < RECORD_BYTES {
            return Err(SpkError::Format("file shorter than one DAF record".into()));
        }
        let idword = &bytes[0..8];
        if idword != b"DAF/SPK " {
            return Err(SpkError::Format(format!(
                "bad ID word {:?}",
                String::from_utf8_lossy(idword)
            )));
        }
        let nd = i32_at(&bytes, 8) as usize;
        let ni = i32_at(&bytes, 12) as usize;
        let fward = i32_at(&bytes, 76) as usize; // first summary record, 1-based
        let locfmt = &bytes[88..96];
        if locfmt != b"LTL-IEEE" {
            return Err(SpkError::Format(format!(
                "unsupported byte order {:?} (only LTL-IEEE)",
                String::from_utf8_lossy(locfmt)
            )));
        }
        if nd != 2 || ni != 6 {
            return Err(SpkError::Format(format!("unexpected ND={nd} NI={ni}")));
        }
        // Summary size in doubles: ND + ceil(NI/2).
        let ss = nd + ni.div_ceil(2);

        let mut segments = Vec::new();
        let mut record = fward;
        while record != 0 {
            let base = (record - 1) * RECORD_BYTES;
            let rec = &bytes[base..base + RECORD_BYTES];
            let next = f64_at(rec, 1) as usize;
            let nsum = f64_at(rec, 3) as usize;
            for k in 0..nsum {
                let s0 = 3 + k * ss; // first word of this summary (0-based in rec words)
                let start_et_s = f64_at(rec, s0 + 1);
                let end_et_s = f64_at(rec, s0 + 2);
                let ints_off = (s0 + 2) * 8; // byte offset of packed ints
                let target = i32_at(rec, ints_off);
                let center = i32_at(rec, ints_off + 4);
                let _frame = i32_at(rec, ints_off + 8);
                let data_type = i32_at(rec, ints_off + 12);
                let start_word = i32_at(rec, ints_off + 16) as usize;
                let end_word = i32_at(rec, ints_off + 20) as usize;
                if data_type != 2 {
                    return Err(SpkError::Format(format!(
                        "segment {center}->{target} has type {data_type}; only Type 2 supported"
                    )));
                }
                // Trailer: INIT, INTLEN, RSIZE, N (last four doubles).
                let init_s = f64_at(&bytes, end_word - 3);
                let intlen_s = f64_at(&bytes, end_word - 2);
                let rsize = f64_at(&bytes, end_word - 1) as usize;
                let n_records = f64_at(&bytes, end_word) as usize;
                let n_data = rsize * n_records;
                let mut data = Vec::with_capacity(n_data);
                for w in 0..n_data {
                    data.push(f64_at(&bytes, start_word + w));
                }
                segments.push(Segment {
                    center,
                    target,
                    start_et_s,
                    end_et_s,
                    init_s,
                    intlen_s,
                    rsize,
                    n_records,
                    data,
                });
            }
            record = next;
        }
        Ok(Spk { segments })
    }

    fn segment(&self, center: i32, target: i32) -> Result<&Segment, SpkError> {
        self.segments
            .iter()
            .find(|s| s.center == center && s.target == target)
            .ok_or(SpkError::NoSegment { center, target })
    }

    /// Position (km) and velocity (km/s) of `target` relative to `center`
    /// for one direct segment, at a TDB Julian date.
    pub fn state(&self, center: i32, target: i32, jd_tdb: f64) -> Result<([f64; 3], [f64; 3]), SpkError> {
        self.segment(center, target)?.state(jd_tdb)
    }

    /// Position/velocity of a body relative to the solar-system barycenter,
    /// chaining segments the way de421 lays them out (0 -> barycenter ->
    /// body). `target` uses NAIF ids (10 sun, 301 moon, 399 earth, 199/299/499
    /// planets, 1-9 barycenters).
    pub fn ssb_state(&self, target: i32, jd_tdb: f64) -> Result<([f64; 3], [f64; 3]), SpkError> {
        let chain: &[(i32, i32)] = match target {
            0 => return Ok(([0.0; 3], [0.0; 3])),
            1..=10 => &[(0, target)],
            199 => &[(0, 1), (1, 199)],
            299 => &[(0, 2), (2, 299)],
            301 => &[(0, 3), (3, 301)],
            399 => &[(0, 3), (3, 399)],
            499 => &[(0, 4), (4, 499)],
            other => {
                return Err(SpkError::NoSegment {
                    center: 0,
                    target: other,
                })
            }
        };
        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        for &(c, t) in chain {
            let (p, v) = self.state(c, t, jd_tdb)?;
            for i in 0..3 {
                pos[i] += p[i];
                vel[i] += v[i];
            }
        }
        Ok((pos, vel))
    }
}

impl Segment {
    /// Evaluate the Chebyshev record covering `jd_tdb`.
    pub fn state(&self, jd_tdb: f64) -> Result<([f64; 3], [f64; 3]), SpkError> {
        let et_s = (jd_tdb - J2000_JD) * SECONDS_PER_DAY;
        if et_s < self.start_et_s - 1e-6 || et_s > self.end_et_s + 1e-6 {
            return Err(SpkError::OutOfRange {
                target: self.target,
                jd_tdb,
            });
        }
        let mut idx = ((et_s - self.init_s) / self.intlen_s).floor() as isize;
        idx = idx.clamp(0, self.n_records as isize - 1);
        let rec = &self.data[idx as usize * self.rsize..(idx as usize + 1) * self.rsize];
        let mid = rec[0];
        let radius = rec[1];
        let n_coef = (self.rsize - 2) / 3;
        let tc = (et_s - mid) / radius; // in [-1, 1]

        // Chebyshev T_k(tc) and derivatives via recurrence.
        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        let mut t_km1 = 1.0; // T_0
        let mut t_k = tc; // T_1
        let mut dt_km1 = 0.0; // T_0'
        let mut dt_k = 1.0; // T_1'
        for axis in 0..3 {
            let coef = &rec[2 + axis * n_coef..2 + (axis + 1) * n_coef];
            pos[axis] = coef[0]; // T_0 term
            if n_coef > 1 {
                pos[axis] += coef[1] * tc;
                vel[axis] = coef[1];
            }
        }
        for k in 2..n_coef {
            let t_kp1 = 2.0 * tc * t_k - t_km1;
            let dt_kp1 = 2.0 * tc * dt_k + 2.0 * t_k - dt_km1;
            for axis in 0..3 {
                let c = self_coef(rec, n_coef, axis, k);
                pos[axis] += c * t_kp1;
                vel[axis] += c * dt_kp1;
            }
            t_km1 = t_k;
            t_k = t_kp1;
            dt_km1 = dt_k;
            dt_k = dt_kp1;
        }
        // d/dt = d/dtc * dtc/dt; radius is in seconds -> km/s.
        for v in vel.iter_mut() {
            *v /= radius;
        }
        Ok((pos, vel))
    }
}

#[inline]
fn self_coef(rec: &[f64], n_coef: usize, axis: usize, k: usize) -> f64 {
    rec[2 + axis * n_coef + k]
}
