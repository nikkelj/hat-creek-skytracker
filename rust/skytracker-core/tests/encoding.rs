//! Pure-Rust golden-byte + closed-loop tests. No Python interpreter required:
//!   cargo test
//!
//! Cross-language byte parity against the real Python encoder lives in
//! `test_rust_mount_parity.py` at the repo root.

use skytracker_core::protocol::{self, targets};
use skytracker_core::sim::{LoopbackTransport, Mount, RecordingTransport, SimResponder};

#[test]
fn get_position_request_bytes() {
    // 0x50, len=1, dest=AZM, MC_GET_POSITION, 00 00 00, resp_len=3
    assert_eq!(
        protocol::encode_get_position(targets::AZM),
        vec![0x50, 0x01, 0x10, 0x01, 0x00, 0x00, 0x00, 0x03]
    );
    assert_eq!(
        protocol::encode_get_position(targets::ALT),
        vec![0x50, 0x01, 0x11, 0x01, 0x00, 0x00, 0x00, 0x03]
    );
}

#[test]
fn slew_fixed_request_bytes() {
    // positive -> MC_MOVE_POS (0x24)
    assert_eq!(
        protocol::encode_slew_fixed(targets::AZM, 5),
        vec![0x50, 0x02, 0x10, 0x24, 0x05, 0x00, 0x00, 0x00]
    );
    // negative -> MC_MOVE_NEG (0x25), magnitude in data0
    assert_eq!(
        protocol::encode_slew_fixed(targets::ALT, -9),
        vec![0x50, 0x02, 0x11, 0x25, 0x09, 0x00, 0x00, 0x00]
    );
    // stop
    assert_eq!(
        protocol::encode_slew_fixed(targets::AZM, 0),
        vec![0x50, 0x02, 0x10, 0x24, 0x00, 0x00, 0x00, 0x00]
    );
}

#[test]
fn get_version_request_bytes() {
    assert_eq!(
        protocol::encode_get_version(targets::HC),
        vec![0x50, 0x01, 0x04, 0xfe, 0x00, 0x00, 0x00, 0x02]
    );
}

#[test]
fn pack_unpack_roundtrip() {
    // 0x400000 / 2^24 == 0.25 (the value asserted in test_serial_transact.py)
    assert_eq!(protocol::pack_int3(0.25), [0x40, 0x00, 0x00]);
    assert!((protocol::unpack_int3(&[0x40, 0x00, 0x00]) - 0.25).abs() < 1e-9);
    for &f in &[0.0_f64, 0.1, 0.5, 0.999] {
        let b = protocol::pack_int3(f);
        assert!((protocol::unpack_int3(&b) - f).abs() < 1e-6, "f={f}");
    }
}

#[test]
fn short_read_is_an_error() {
    // Canned response too short -> ShortRead (the runaway-bug guard).
    let mut m = Mount::new(RecordingTransport::new(b"\x40\x00")); // 2 of 4 bytes
    assert!(m.hc_get_position(targets::ALT).is_err());
}

#[test]
fn sim_closed_loop_integrates_rate() {
    // Command max rate on AZM, advance 1s, position should be ~10 deg.
    // rate(9) = 10/360 rev/s -> 10 deg/s.
    let mut m = Mount::new(LoopbackTransport::new(SimResponder::new_manual(0.0, 0.0)));
    assert!(m.hc_slew_fixed(targets::AZM, 9).unwrap());
    m.io.responder.advance_time(1.0);
    let frac = m.hc_get_position(targets::AZM).unwrap();
    assert!((frac - 10.0 / 360.0).abs() < 1e-6, "frac={frac}");

    // ALT axis is independent and untouched.
    let alt = m.hc_get_position(targets::ALT).unwrap();
    assert!(alt.abs() < 1e-9, "alt={alt}");
}

#[test]
fn sim_negative_rate_and_wrap() {
    // Negative AZM rate from 0 should wrap to just under a full revolution.
    let mut m = Mount::new(LoopbackTransport::new(SimResponder::new_manual(0.0, 0.0)));
    m.hc_slew_fixed(targets::AZM, -9).unwrap();
    m.io.responder.advance_time(1.0);
    let frac = m.hc_get_position(targets::AZM).unwrap();
    // -10 deg wraps to 350 deg = 350/360.
    assert!((frac - 350.0 / 360.0).abs() < 1e-6, "frac={frac}");
}
