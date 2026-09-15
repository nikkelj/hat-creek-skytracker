//! Minimal .npy/.npz reading for exactly the tetra3 database layout:
//! simple little-endian arrays ('<f4', '<u4', '<u2') plus the structured
//! `props_packed` record, whose descr is parsed field-by-field. Fails
//! loudly on anything unexpected — an unknown schema must never be
//! silently misread (the DB would "load" and solve garbage).

use std::collections::HashMap;
use std::io::Read;

#[derive(Debug)]
pub struct NpyError(pub String);

impl std::fmt::Display for NpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "npy: {}", self.0)
    }
}

impl std::error::Error for NpyError {}

fn err(s: impl Into<String>) -> NpyError {
    NpyError(s.into())
}

pub struct RawNpy {
    /// The descr string, e.g. "'<f4'" for simple arrays or "[('name', ...)]".
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// Parse one .npy member (v1.0/2.0 headers).
pub fn parse_npy(bytes: &[u8]) -> Result<RawNpy, NpyError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(err("bad magic"));
    }
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|_| err("header not utf8"))?;

    // descr: everything between "'descr':" and ", 'fortran_order'".
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split("'fortran_order'").next())
        .ok_or_else(|| err("no descr"))?
        .trim()
        .trim_end_matches(',')
        .trim()
        .to_string();

    if header.contains("'fortran_order': True") {
        return Err(err("fortran order unsupported"));
    }

    let shape_str = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| err("no shape"))?
        .trim()
        .trim_start_matches('(');
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();

    Ok(RawNpy {
        descr,
        shape,
        data: bytes[header_start + header_len..].to_vec(),
    })
}

/// Read every member of an .npz archive.
pub fn read_npz(path: &std::path::Path) -> Result<HashMap<String, RawNpy>, NpyError> {
    let file = std::fs::File::open(path).map_err(|e| err(format!("open {path:?}: {e}")))?;
    let mut archive =
        zip::ZipArchive::new(std::io::BufReader::new(file)).map_err(|e| err(e.to_string()))?;
    let mut out = HashMap::new();
    for i in 0..archive.len() {
        let mut member = archive.by_index(i).map_err(|e| err(e.to_string()))?;
        let name = member
            .name()
            .trim_end_matches(".npy")
            .to_string();
        let mut bytes = Vec::with_capacity(member.size() as usize);
        member
            .read_to_end(&mut bytes)
            .map_err(|e| err(e.to_string()))?;
        out.insert(name, parse_npy(&bytes)?);
    }
    Ok(out)
}

impl RawNpy {
    pub fn expect_descr(&self, want: &str) -> Result<(), NpyError> {
        if self.descr.trim_matches(|c| c == '\'' || c == '"') == want
            || self.descr == format!("'{want}'")
        {
            Ok(())
        } else {
            Err(err(format!("descr {} != expected {want}", self.descr)))
        }
    }

    pub fn as_f32(&self) -> Result<Vec<f32>, NpyError> {
        self.expect_descr("<f4")?;
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn as_u32(&self) -> Result<Vec<u32>, NpyError> {
        self.expect_descr("<u4")?;
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn as_u16(&self) -> Result<Vec<u16>, NpyError> {
        self.expect_descr("<u2")?;
        Ok(self
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// A parsed structured-record descr: field name -> (byte offset, format).
pub struct RecordLayout {
    pub fields: HashMap<String, (usize, String)>,
    pub itemsize: usize,
}

/// Parse a packed record descr like
/// `[('pattern_mode', '<U64'), ('pattern_size', '<u2'), ('range_ra', '<f4', (2,)), ...]`.
pub fn parse_record_descr(descr: &str) -> Result<RecordLayout, NpyError> {
    let mut fields = HashMap::new();
    let mut offset = 0usize;
    let inner = descr
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| err(format!("not a record descr: {descr}")))?;
    // Split on "), (" boundaries.
    for tuple in inner.split("), (") {
        let t = tuple.trim_matches(|c| c == '(' || c == ')' || c == ' ' || c == ',');
        let parts: Vec<&str> = t.split(',').map(|p| p.trim().trim_matches('\'')).collect();
        if parts.len() < 2 {
            return Err(err(format!("bad record field: {tuple}")));
        }
        let name = parts[0].to_string();
        let fmt = parts[1].to_string();
        let elem = format_size(&fmt)?;
        // Optional shape: remaining numeric parts multiply the size.
        let count: usize = parts[2..]
            .iter()
            .filter_map(|p| p.trim_matches(|c: char| !c.is_ascii_digit()).parse::<usize>().ok())
            .product::<usize>()
            .max(1);
        fields.insert(name, (offset, fmt));
        offset += elem * count;
    }
    Ok(RecordLayout {
        fields,
        itemsize: offset,
    })
}

fn format_size(fmt: &str) -> Result<usize, NpyError> {
    if let Some(n) = fmt.strip_prefix("<U") {
        return n
            .parse::<usize>()
            .map(|c| c * 4)
            .map_err(|_| err(format!("bad U format {fmt}")));
    }
    Ok(match fmt {
        "<u2" | "<i2" => 2,
        "<u4" | "<i4" | "<f4" => 4,
        "<u8" | "<i8" | "<f8" => 8,
        "?" | "|b1" => 1,
        other => return Err(err(format!("unsupported record format {other}"))),
    })
}

impl RecordLayout {
    pub fn get_u16(&self, data: &[u8], name: &str) -> Result<u16, NpyError> {
        let (off, fmt) = self.field(name)?;
        if fmt != "<u2" {
            return Err(err(format!("{name} is {fmt}, wanted <u2")));
        }
        Ok(u16::from_le_bytes(data[off..off + 2].try_into().unwrap()))
    }

    pub fn get_f32(&self, data: &[u8], name: &str) -> Result<f32, NpyError> {
        let (off, fmt) = self.field(name)?;
        if fmt != "<f4" {
            return Err(err(format!("{name} is {fmt}, wanted <f4")));
        }
        Ok(f32::from_le_bytes(data[off..off + 4].try_into().unwrap()))
    }

    pub fn get_bool(&self, data: &[u8], name: &str) -> Result<bool, NpyError> {
        let (off, fmt) = self.field(name)?;
        if fmt != "?" && fmt != "|b1" {
            return Err(err(format!("{name} is {fmt}, wanted bool")));
        }
        Ok(data[off] != 0)
    }

    pub fn get_ucs4(&self, data: &[u8], name: &str) -> Result<String, NpyError> {
        let (off, fmt) = self.field(name)?;
        let n: usize = fmt
            .strip_prefix("<U")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| err(format!("{name} is {fmt}, wanted <U*")))?;
        let mut s = String::new();
        for i in 0..n {
            let cp = u32::from_le_bytes(data[off + i * 4..off + i * 4 + 4].try_into().unwrap());
            if cp == 0 {
                break;
            }
            s.push(char::from_u32(cp).ok_or_else(|| err("bad UCS4"))?);
        }
        Ok(s)
    }

    fn field(&self, name: &str) -> Result<(usize, String), NpyError> {
        self.fields
            .get(name)
            .cloned()
            .ok_or_else(|| err(format!("record has no field {name}")))
    }
}
