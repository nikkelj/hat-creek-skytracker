//! Pure-Rust hotspot tests mirroring test_hotspot.py. Run: cargo test
//! Cross-language parity (against the real numpy detector) is in
//! test_rust_hotspot_parity.py.

use ndarray::Array2;
use skytracker_core::hotspot::{detect_hotspot, pixel_offset_to_angles};

/// Noiseless gaussian blob over a flat background (f32, like a camera frame).
fn gaussian(w: usize, h: usize, cx: f64, cy: f64, amp: f64, sigma: f64, bg: f64) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let r2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
            a[[y, x]] = (bg + amp * (-(r2) / (2.0 * sigma * sigma)).exp()) as f32;
        }
    }
    a
}

#[test]
fn centroid_near_truth() {
    let img = gaussian(256, 200, 130.3, 90.7, 200.0, 3.0, 10.0);
    let d = detect_hotspot(&img.view(), None, 5.0, 12, 0.5, 3).expect("detection");
    assert!((d.cx - 130.3).abs() < 0.5, "cx={}", d.cx);
    assert!((d.cy - 90.7).abs() < 0.5, "cy={}", d.cy);
    assert!(d.snr > 5.0);
}

#[test]
fn flat_field_returns_none() {
    // Constant field: mad=0, std=0 -> noise=1, snr=0 < threshold -> None.
    let img = Array2::<f32>::from_elem((200, 200), 10.0);
    assert!(detect_hotspot(&img.view(), None, 5.0, 12, 0.5, 3).is_none());
}

#[test]
fn single_hot_pixel_rejected() {
    let mut img = Array2::<f32>::from_elem((128, 128), 10.0);
    img[[64, 80]] = 5000.0;
    assert!(detect_hotspot(&img.view(), None, 5.0, 12, 0.5, 3).is_none());
}

#[test]
fn gate_selects_dimmer_nearby_blob() {
    let dim = gaussian(200, 200, 50.0, 50.0, 150.0, 3.0, 10.0);
    let bright = gaussian(200, 200, 150.0, 150.0, 255.0, 3.0, 0.0);
    let img = &dim + &bright;

    // No gate -> brightest (near 150,150).
    let d_all = detect_hotspot(&img.view(), None, 5.0, 12, 0.5, 3).expect("all");
    assert!((d_all.cx - 150.0).abs() < 2.0, "cx={}", d_all.cx);

    // Gate around the dimmer blob -> locks it.
    let d_gate =
        detect_hotspot(&img.view(), Some((50.0, 50.0, 30.0)), 5.0, 12, 0.5, 3).expect("gate");
    assert!((d_gate.cx - 50.0).abs() < 2.0, "cx={}", d_gate.cx);
    assert!((d_gate.cy - 50.0).abs() < 2.0, "cy={}", d_gate.cy);
}

#[test]
fn geometry_magnitude_and_default_signs() {
    let px: f64 = 4.0;
    let fl: f64 = 1000.0;
    let ifov = ((px * 1e-3) / fl).to_degrees();
    let (az, el) = pixel_offset_to_angles(100.0, 100.0, px, fl, 0.0, 0.0, 1.0, -1.0, true);
    assert!((az - 100.0 * ifov).abs() < 1e-9);
    assert!((el - (-100.0 * ifov)).abs() < 1e-9);
}

#[test]
fn geometry_rotation_90() {
    let px: f64 = 4.0;
    let fl: f64 = 1000.0;
    let ifov = ((px * 1e-3) / fl).to_degrees();
    let (az, el) = pixel_offset_to_angles(100.0, 0.0, px, fl, 90.0, 0.0, 1.0, -1.0, true);
    assert!((el - 100.0 * ifov).abs() < 1e-9, "el={el}");
    assert!(az.abs() < 1e-9, "az={az}");
}

#[test]
fn geometry_cos_el_scaling() {
    let px = 4.0;
    let fl = 1000.0;
    let (az0, _) = pixel_offset_to_angles(100.0, 0.0, px, fl, 0.0, 0.0, 1.0, -1.0, true);
    let (az60, _) = pixel_offset_to_angles(100.0, 0.0, px, fl, 0.0, 60.0, 1.0, -1.0, true);
    assert!((az60 - az0 / 60.0_f64.to_radians().cos()).abs() < 1e-9);
}

#[test]
fn geometry_zero_focal_length_safe() {
    assert_eq!(
        pixel_offset_to_angles(10.0, 10.0, 4.0, 0.0, 0.0, 0.0, 1.0, -1.0, true),
        (0.0, 0.0)
    );
}

#[test]
fn nan_pixels_do_not_panic() {
    // Real ASI frames can contain NaN after debayer/hot-column handling. The
    // detector must never panic the control-loop thread over them.
    let mut img = gaussian(256, 200, 130.3, 90.7, 200.0, 3.0, 10.0);
    img[[0, 0]] = f32::NAN;
    img[[50, 60]] = f32::NAN;
    img[[199, 255]] = f32::NAN;
    // A few NaNs must leave the real blob detectable.
    let d = detect_hotspot(&img.view(), None, 5.0, 12, 0.5, 3).expect("detection");
    assert!((d.cx - 130.3).abs() < 1.0, "cx={}", d.cx);

    // Even an all-NaN frame must return cleanly (no detection is fine).
    let all_nan = Array2::<f32>::from_elem((64, 64), f32::NAN);
    let _ = detect_hotspot(&all_nan.view(), None, 5.0, 12, 0.5, 3);
}
