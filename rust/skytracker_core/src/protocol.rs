//! NexStar "P" pass-through protocol encoding.
//!
//! Byte-for-byte port of the ACTIVE hand-controller path in `lib/auxstar.py`
//! (`NexstarHandController`). That path frames every command as an 8-byte 'P'
//! (0x50) pass-through packet:
//!
//! ```text
//!   0x50  msg_len  dest_id  msg_id  data0 data1 data2  resp_len
//! ```
//!
//! It does NOT use the lower 0x3b AUX framing / checksum (those appear only in
//! the commented examples at the bottom of auxstar.py and are unused by the
//! running tracker). Parity with the Python encoder is verified two ways:
//!   * Rust golden-byte unit tests in `tests/encoding.rs`, and
//!   * `test_rust_mount_parity.py`, which diffs these bytes against the bytes
//!     `NexstarHandController` actually writes to the wire.

/// Device target ids (subset used by the tracker). Mirrors `auxstar.Targets`.
pub mod targets {
    pub const ANY: u8 = 0x00;
    pub const HC: u8 = 0x04;
    pub const AZM: u8 = 0x10;
    pub const ALT: u8 = 0x11;
    pub const FOCUS: u8 = 0x12;
}

// (msg_id, cmd_bytes, resp_bytes) tuples mirroring entries in `auxstar.COMMANDS`.
pub const MC_GET_POSITION: (u8, u8, u8) = (0x01, 1, 3);
pub const MC_GOTO_FAST: (u8, u8, u8) = (0x02, 4, 0);
pub const MC_MOVE_POS: (u8, u8, u8) = (0x24, 2, 0);
pub const MC_MOVE_NEG: (u8, u8, u8) = (0x25, 2, 0);
pub const MC_SET_POS_GUIDERATE: (u8, u8, u8) = (0x06, 4, 0);
pub const MC_SET_NEG_GUIDERATE: (u8, u8, u8) = (0x07, 4, 0);
pub const MC_GET_VER: (u8, u8, u8) = (0xfe, 1, 2);

/// RATES table: fraction of a full revolution per second, index 0..=9.
/// Mirrors `auxstar.RATES`.
pub fn rate(idx: u8) -> f64 {
    match idx {
        0 => 0.0,
        1 => 1.0 / (360.0 * 60.0),
        2 => 2.0 / (360.0 * 60.0),
        3 => 5.0 / (360.0 * 60.0),
        4 => 15.0 / (360.0 * 60.0),
        5 => 30.0 / (360.0 * 60.0),
        6 => 1.0 / 360.0,
        7 => 2.0 / 360.0,
        8 => 5.0 / 360.0,
        9 => 10.0 / 360.0,
        _ => 0.0,
    }
}

/// Pack a fraction-of-revolution into 3 big-endian bytes (`auxstar.pack_int3`).
/// `int()` in Python truncates toward zero; Rust `as i32` does the same.
pub fn pack_int3(f: f64) -> [u8; 3] {
    let v = (f * (1u32 << 24) as f64) as i32;
    let b = v.to_be_bytes();
    [b[1], b[2], b[3]]
}

/// Unpack 3 big-endian bytes into a fraction of a revolution
/// (`auxstar.unpack_int3`). The implicit leading 0x00 keeps it unsigned.
pub fn unpack_int3(d: &[u8]) -> f64 {
    debug_assert!(d.len() >= 3);
    let v = ((d[0] as u32) << 16) | ((d[1] as u32) << 8) | (d[2] as u32);
    v as f64 / (1u32 << 24) as f64
}

/// Signed fraction of a revolution from degrees/minutes/seconds
/// (`auxstar.dms2f`). mm/ss are arcminutes/arcseconds -- fractions of a
/// *degree* -- so the angle in degrees is |dd| + mm/60 + ss/3600. The sign
/// convention matches f2dms, which carries the sign on the degrees term.
pub fn dms2f(dd: f64, mm: f64, ss: f64, sign: f64) -> f64 {
    let axis_sign = if dd < 0.0 { -1.0 } else { 1.0 };
    sign * axis_sign * (dd.abs() + mm / 60.0 + ss / 3600.0) / 360.0
}

/// Expected read length for a command: `resp_bytes + 1` for the trailing '#'.
/// Matches the `expected_response_length` passed to `_transact` in auxstar.
pub fn expected_response_len(resp_bytes: u8) -> usize {
    resp_bytes as usize + 1
}

fn passthrough(msg_len: u8, dest: u8, msg_id: u8, data: [u8; 3], resp_len: u8) -> Vec<u8> {
    vec![0x50, msg_len, dest, msg_id, data[0], data[1], data[2], resp_len]
}

pub fn encode_get_position(target: u8) -> Vec<u8> {
    let (id, clen, rlen) = MC_GET_POSITION;
    passthrough(clen, target, id, [0, 0, 0], rlen)
}

pub fn encode_get_version(target: u8) -> Vec<u8> {
    let (id, clen, rlen) = MC_GET_VER;
    passthrough(clen, target, id, [0, 0, 0], rlen)
}

/// `rate_step`: 0 = stop, 1..=9 positive, -1..=-9 negative. Sign selects the
/// MC_MOVE_POS / MC_MOVE_NEG opcode; magnitude goes in data0 (as in auxstar).
pub fn encode_slew_fixed(target: u8, rate_step: i32) -> Vec<u8> {
    let (id, clen, rlen) = if rate_step >= 0 { MC_MOVE_POS } else { MC_MOVE_NEG };
    let mag = rate_step.unsigned_abs().min(255) as u8;
    passthrough(clen, target, id, [mag, 0, 0], rlen)
}

/// Firmware guide-rate unit, CALIBRATED on the real AVX (bench_guiderate.py,
/// 2026-07-25): the 24-bit MC_SET_POS/NEG_GUIDERATE value is arcseconds per
/// second in Q10 fixed point (value = arcsec_per_sec * 1024). Full scale
/// (2^24 - 1) is 4.551 deg/s, matching the AVX max slew. Mirrors
/// `auxstar.GUIDE_COUNTS_PER_DPS`.
pub const GUIDE_COUNTS_PER_DPS: f64 = 3600.0 * 1024.0;
pub const GUIDE_RATE_MAX_DPS: f64 = ((1u32 << 24) - 1) as f64 / GUIDE_COUNTS_PER_DPS;

/// Continuous variable rate in deg/s via MC_SET_POS/NEG_GUIDERATE. Sign selects
/// the opcode; magnitude is packed as arcsec/s * 1024 (GUIDE_COUNTS_PER_DPS,
/// clamped to 24-bit full scale) -- mirrors `auxstar.hc_set_rate_dps`.
pub fn encode_set_guiderate(target: u8, dps: f64) -> Vec<u8> {
    let (id, clen, rlen) = if dps >= 0.0 {
        MC_SET_POS_GUIDERATE
    } else {
        MC_SET_NEG_GUIDERATE
    };
    // Round+clamp in COUNT space so full scale encodes exactly 0xFFFFFF
    // (clamping in deg/s and converting back can land one LSB short).
    let counts = (dps.abs() * GUIDE_COUNTS_PER_DPS).round().min(16777215.0);
    let mag = pack_int3(counts / 16777216.0);
    passthrough(clen, target, id, mag, rlen)
}

pub fn encode_goto_fast(target: u8, dd: f64, mm: f64, ss: f64) -> Vec<u8> {
    let (id, clen, rlen) = MC_GOTO_FAST;
    let fofr = pack_int3(dms2f(dd, mm, ss, 1.0));
    passthrough(clen, target, id, fofr, rlen)
}
