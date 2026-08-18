//! Mode-S DF17/18 extended-squitter decoding — a faithful port of the
//! pyModeS subset adsb_receiver.py uses: CRC, ICAO, typecode, callsign
//! (TC 1-4), airborne position-with-reference (CPR local, TC 9-18/20-22),
//! altitude, and airborne velocity (TC 19). Algorithms mirror pyModeS
//! (py_common.py + decoder/bds/bds05/08/09) including its quirks (integer
//! speed truncation, the redundant zero-component velocity gates).
//!
//! Known deliberate divergence: Gillham-coded altitudes (Q-bit 0, only
//! used above 50,187 ft) decode to None instead of the gray-code value —
//! no optically trackable aircraft flies there.

/// A 112-bit Mode-S message as bits (MSB first).
struct Bits(u128);

impl Bits {
    fn parse(msg: &str) -> Option<Bits> {
        if msg.len() != 28 || !msg.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        u128::from_str_radix(msg, 16).ok().map(Bits)
    }

    /// Bit i (0-based from the front of the message).
    #[inline]
    fn bit(&self, i: usize) -> u32 {
        ((self.0 >> (111 - i)) & 1) as u32
    }

    /// Integer from bits [start, end).
    #[inline]
    fn int(&self, start: usize, end: usize) -> u32 {
        let width = end - start;
        ((self.0 >> (112 - end)) & ((1u128 << width) - 1)) as u32
    }
}

/// Mode-S CRC remainder over the full message (0 for a valid frame).
pub fn crc(msg: &str) -> Option<u32> {
    let bits = Bits::parse(msg)?;
    let mut bytes = [0u8; 14];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = ((bits.0 >> (104 - i * 8)) & 0xff) as u8;
    }
    const G: [u32; 4] = [0b1111_1111, 0b1111_1010, 0b0000_0100, 0b1000_0000];
    for ibyte in 0..bytes.len() - 3 {
        for ibit in 0..8 {
            let mask = 0x80u8 >> ibit;
            if bytes[ibyte] & mask != 0 {
                bytes[ibyte] ^= (G[0] >> ibit) as u8;
                bytes[ibyte + 1] ^= (0xff & ((G[0] << (8 - ibit)) | (G[1] >> ibit))) as u8;
                bytes[ibyte + 2] ^= (0xff & ((G[1] << (8 - ibit)) | (G[2] >> ibit))) as u8;
                bytes[ibyte + 3] ^= (0xff & ((G[2] << (8 - ibit)) | (G[3] >> ibit))) as u8;
            }
        }
    }
    Some(((bytes[11] as u32) << 16) | ((bytes[12] as u32) << 8) | bytes[13] as u32)
}

pub fn df(msg: &str) -> Option<u32> {
    Bits::parse(msg).map(|b| b.int(0, 5).min(24))
}

/// ICAO for DF17/18 (hex chars 2..8, uppercased).
pub fn icao(msg: &str) -> Option<String> {
    match df(msg)? {
        17 | 18 => Some(msg[2..8].to_ascii_uppercase()),
        _ => None,
    }
}

pub fn typecode(msg: &str) -> Option<u32> {
    match df(msg)? {
        17 | 18 => Some(Bits::parse(msg)?.int(32, 37)),
        _ => None,
    }
}

const CALLSIGN_CHARS: &[u8; 64] =
    b"#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######";

/// TC 1-4 callsign ('#' padding stripped, like pyModeS).
pub fn callsign(msg: &str) -> Option<String> {
    let tc = typecode(msg)?;
    if !(1..=4).contains(&tc) {
        return None;
    }
    let bits = Bits::parse(msg)?;
    let mut cs = String::new();
    for k in 0..8 {
        let idx = bits.int(40 + k * 6, 46 + k * 6) as usize;
        let ch = CALLSIGN_CHARS[idx] as char;
        if ch != '#' {
            cs.push(ch);
        }
    }
    Some(cs)
}

/// Airborne altitude in feet (TC 9-18 barometric 25-ft encoding, TC 20-22
/// GNSS metres converted). None for invalid/Gillham codes.
pub fn altitude_ft(msg: &str) -> Option<f64> {
    let tc = typecode(msg)?;
    if !(9..=18).contains(&tc) && !(20..=22).contains(&tc) {
        return None;
    }
    let bits = Bits::parse(msg)?;
    if tc >= 20 {
        return Some(bits.int(40, 52) as f64 * 3.28084);
    }
    // 12-bit field -> 13-bit altcode with M=0 inserted at position 6.
    let altbin = bits.int(40, 52);
    if altbin == 0 {
        return None;
    }
    let qbit = (altbin >> 4) & 1; // bit 8 of the 13-bit code == bit 7 of 12
    if qbit == 1 {
        // Drop the Q bit: upper 7 bits and lower 4 bits of the 12-bit field.
        let v = ((altbin >> 5) << 4) | (altbin & 0xf);
        Some((v as i64 * 25 - 1000) as f64)
    } else {
        None // Gillham 100-ft code (>50,187 ft): deliberately unsupported.
    }
}

fn cpr_nl(lat: f64) -> i64 {
    // np.isclose semantics: |a-b| <= 1e-8 + 1e-5*|b|.
    if lat.abs() <= 1e-8 {
        return 59;
    }
    if (lat.abs() - 87.0).abs() <= 1e-8 + 1e-5 * 87.0 {
        return 2;
    }
    if lat > 87.0 || lat < -87.0 {
        return 1;
    }
    let nz = 15.0;
    let a = 1.0 - (std::f64::consts::PI / (2.0 * nz)).cos();
    let b = (std::f64::consts::PI / 180.0 * lat.abs()).cos().powi(2);
    let nl = 2.0 * std::f64::consts::PI / (1.0 - a / b).acos();
    nl.floor() as i64
}

/// CPR local decode with a reference position (airborne, TC 9-18/20-22).
pub fn airborne_position_with_ref(msg: &str, lat_ref: f64, lon_ref: f64) -> Option<(f64, f64)> {
    let bits = Bits::parse(msg)?;
    let cprlat = bits.int(54, 71) as f64 / 131072.0;
    let cprlon = bits.int(71, 88) as f64 / 131072.0;
    let i = bits.bit(53) as i64;

    let d_lat = if i == 1 { 360.0 / 59.0 } else { 360.0 / 60.0 };
    let j = (lat_ref / d_lat).floor() + (0.5 + (lat_ref.rem_euclid(d_lat)) / d_lat - cprlat).floor();
    let lat = d_lat * (j + cprlat);

    let ni = cpr_nl(lat) - i;
    let d_lon = if ni > 0 { 360.0 / ni as f64 } else { 360.0 };
    let m = (lon_ref / d_lon).floor() + (0.5 + (lon_ref.rem_euclid(d_lon)) / d_lon - cprlon).floor();
    let lon = d_lon * (m + cprlon);
    Some((lat, lon))
}

pub struct Velocity {
    pub speed_kt: Option<f64>,
    pub track_deg: Option<f64>,
    pub vert_rate_fpm: Option<i64>,
}

/// TC 19 airborne velocity (pyModeS bds09.airborne_velocity semantics,
/// incl. its early-return when either raw component field is zero).
pub fn airborne_velocity(msg: &str) -> Option<Velocity> {
    if typecode(msg)? != 19 {
        return None;
    }
    let bits = Bits::parse(msg)?;
    let mb = |s: usize, e: usize| bits.int(32 + s, 32 + e);

    let subtype = mb(5, 8);
    if mb(14, 24) == 0 || mb(25, 35) == 0 {
        return None;
    }

    let (speed_kt, track_deg): (Option<f64>, Option<f64>) = if subtype == 1 || subtype == 2 {
        let mut v_ew = mb(14, 24) as i64;
        let mut v_ns = mb(25, 35) as i64;
        // (The zero case is unreachable past the gate above; kept for parity.)
        if v_ew == 0 || v_ns == 0 {
            (None, None)
        } else {
            let ew_sign = if mb(13, 14) == 1 { -1 } else { 1 };
            v_ew -= 1;
            if subtype == 2 {
                v_ew *= 4;
            }
            let ns_sign = if mb(24, 25) == 1 { -1 } else { 1 };
            v_ns -= 1;
            if subtype == 2 {
                v_ns *= 4;
            }
            let v_we = (ew_sign * v_ew) as f64;
            let v_sn = (ns_sign * v_ns) as f64;
            let spd = (v_sn * v_sn + v_we * v_we).sqrt().trunc(); // int() cast
            let mut trk = v_we.atan2(v_sn).to_degrees();
            if trk < 0.0 {
                trk += 360.0;
            }
            (Some(spd), Some(trk))
        }
    } else {
        let hdg = if mb(13, 14) == 0 {
            None
        } else {
            Some(mb(14, 24) as f64 / 1024.0 * 360.0)
        };
        let raw = mb(25, 35) as i64;
        let spd = if raw == 0 {
            None
        } else {
            let mut s = raw - 1;
            if subtype == 4 {
                s *= 4;
            }
            Some(s as f64)
        };
        (spd, hdg)
    };

    let vr_sign = if mb(36, 37) == 1 { -1i64 } else { 1 };
    let vr = mb(37, 46) as i64;
    let vert_rate_fpm = if vr == 0 { None } else { Some(vr_sign * (vr - 1) * 64) };

    Some(Velocity {
        speed_kt,
        track_deg,
        vert_rate_fpm,
    })
}

/// One decoded update, mirroring adsb_receiver.decode_adsb_message.
pub enum Decoded {
    Ident {
        icao: String,
        callsign: String,
    },
    Position {
        icao: String,
        lat: f64,
        lon: f64,
        alt_m: Option<f64>,
    },
    Velocity {
        icao: String,
        speed_kt: Option<f64>,
        track_deg: Option<f64>,
        vert_rate_fpm: Option<i64>,
    },
}

/// Full single-message decode (DF17/18, CRC-checked), or None.
pub fn decode_message(msg: &str, ref_lat: f64, ref_lon: f64) -> Option<Decoded> {
    if msg.len() != 28 {
        return None;
    }
    let d = df(msg)?;
    if d != 17 && d != 18 {
        return None;
    }
    if crc(msg)? != 0 {
        return None;
    }
    let icao = icao(msg)?;
    let tc = typecode(msg)?;
    if (1..=4).contains(&tc) {
        let cs = callsign(msg)?.replace('_', "").trim().to_string();
        return Some(Decoded::Ident { icao, callsign: cs });
    }
    if (9..=18).contains(&tc) || (20..=22).contains(&tc) {
        let alt_ft = altitude_ft(msg);
        let (lat, lon) = airborne_position_with_ref(msg, ref_lat, ref_lon)?;
        return Some(Decoded::Position {
            icao,
            lat,
            lon,
            alt_m: alt_ft.map(|a| a * 0.3048),
        });
    }
    if tc == 19 {
        let v = airborne_velocity(msg)?;
        return Some(Decoded::Velocity {
            icao,
            speed_kt: v.speed_kt,
            track_deg: v.track_deg,
            vert_rate_fpm: v.vert_rate_fpm,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // The two reference messages test_adsb.py uses (from the pyModeS docs).
    #[test]
    fn reference_position_message() {
        let msg = "8D40621D58C382D690C8AC2863A7";
        assert_eq!(crc(msg), Some(0));
        assert_eq!(df(msg), Some(17));
        assert_eq!(typecode(msg), Some(11));
        let (lat, lon) = airborne_position_with_ref(msg, 52.258, 3.918).unwrap();
        assert!((lat - 52.2572).abs() < 1e-3, "lat {lat}");
        assert!((lon - 3.91937).abs() < 1e-3, "lon {lon}");
        assert!((altitude_ft(msg).unwrap() - 38000.0).abs() < 1.0);
    }

    #[test]
    fn reference_ident_message() {
        let msg = "8D4840D6202CC371C32CE0576098";
        assert_eq!(crc(msg), Some(0));
        assert_eq!(callsign(msg).unwrap(), "KLM1023_");
    }
}
