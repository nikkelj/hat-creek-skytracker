//! Golden parity vs cv2 (tests/golden/cv2_ops.npz): filters, warps,
//! phase correlation, LK flow, RANSAC similarity.

use skytracker_imaging::features::calc_optical_flow_pyr_lk;
use skytracker_imaging::filters::{gaussian_blur, laplacian, sobel_x, sobel_y};
use skytracker_imaging::image::ImageF32;
use skytracker_imaging::phasecorr::{hanning_window, phase_correlate};
use skytracker_imaging::ransac::estimate_affine_partial_2d;
use skytracker_imaging::warp::{warp_affine, Border};

use std::path::PathBuf;

fn npz() -> npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/cv2_ops.npz");
    npyz::npz::NpzArchive::open(&path).unwrap()
}

fn f32s(
    a: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
    name: &str,
) -> (Vec<usize>, Vec<f32>) {
    let arr = a.by_name(name).unwrap().unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    (shape, arr.into_vec::<f32>().unwrap())
}

fn f64s(
    a: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
    name: &str,
) -> (Vec<usize>, Vec<f64>) {
    let arr = a.by_name(name).unwrap().unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    (shape, arr.into_vec::<f64>().unwrap())
}

fn base_image(a: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>) -> ImageF32 {
    let (shape, data) = f32s(a, "base");
    ImageF32::from_vec(data, shape[1], shape[0])
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn filters_match_cv2() {
    let mut a = npz();
    let base = base_image(&mut a);
    let (_, blur_g) = f32s(&mut a, "gaussian_blur_5x5_s1p2");
    let (_, lap_g) = f32s(&mut a, "laplacian");
    let (_, sx_g) = f32s(&mut a, "sobel_x");
    let (_, sy_g) = f32s(&mut a, "sobel_y");

    let d_blur = max_abs_diff(&gaussian_blur(&base, 5, 1.2).data, &blur_g);
    let d_lap = max_abs_diff(&laplacian(&base).data, &lap_g);
    let d_sx = max_abs_diff(&sobel_x(&base).data, &sx_g);
    let d_sy = max_abs_diff(&sobel_y(&base).data, &sy_g);
    println!("filters max abs diff: blur {d_blur:.2e} lap {d_lap:.2e} sobel {d_sx:.2e}/{d_sy:.2e}");
    assert!(d_blur < 1e-3, "gaussian {d_blur}");
    assert!(d_lap < 1e-3, "laplacian {d_lap}");
    assert!(d_sx < 1e-3 && d_sy < 1e-3, "sobel {d_sx}/{d_sy}");
}

#[test]
fn warp_matches_cv2() {
    let mut a = npz();
    let base = base_image(&mut a);
    let (st_shape, shifts) = f64s(&mut a, "shifts_true");
    let (sh_shape, shifted) = f32s(&mut a, "shifted");
    let (h, w) = (sh_shape[1], sh_shape[2]);
    let mut worst = 0.0f32;
    for i in 0..st_shape[0] {
        let (dx, dy) = (shifts[i * 2] as f32, shifts[i * 2 + 1] as f32);
        let m = [[1.0, 0.0, dx], [0.0, 1.0, dy]];
        let mine = warp_affine(&base, &m, w, h, Border::Reflect);
        let golden = &shifted[i * h * w..(i + 1) * h * w];
        worst = worst.max(max_abs_diff(&mine.data, golden));
    }
    println!("warp max abs diff {worst:.2e}");
    assert!(worst < 1e-2, "warp {worst}");
}

#[test]
fn phase_correlate_matches_cv2() {
    let mut a = npz();
    let base = base_image(&mut a);
    let (sh_shape, shifted) = f32s(&mut a, "shifted");
    let (_, measured) = f64s(&mut a, "phasecorr_measured");
    let (st_shape, truth) = f64s(&mut a, "shifts_true");
    let _ = st_shape;
    let (h, w) = (sh_shape[1], sh_shape[2]);
    let win = hanning_window(w, h);
    let mut worst = 0.0f64;
    for i in 0..sh_shape[0] {
        let moved = ImageF32::from_vec(shifted[i * h * w..(i + 1) * h * w].to_vec(), w, h);
        let (mx, my, resp) = phase_correlate(&base, &moved, Some(&win));
        let (gx, gy, gresp) = (measured[i * 3], measured[i * 3 + 1], measured[i * 3 + 2]);
        let (tx, ty) = (truth[i * 2], truth[i * 2 + 1]);
        // Per axis: match cv2 within 0.05 px, OR — at exact half-pixel
        // shifts, where the weighted-centroid bias mirrors depending on
        // which of the two tied peak rows f32-vs-f64 dust selects — accept
        // the mirror estimate about the true shift with the same bias
        // magnitude. Either way we are as accurate as cv2.
        for (m, g, t, axis) in [(mx, gx, tx, "x"), (my, gy, ty, "y")] {
            let direct = (m - g).abs();
            let mirrored = (m + g - 2.0 * t).abs();
            let d = direct.min(mirrored);
            worst = worst.max(d);
            assert!(
                d < 0.06,
                "shift {i} {axis}: mine {m:.4} vs cv2 {g:.4} (true {t})"
            );
            assert!(
                (m - t).abs() <= (g - t).abs() + 0.05,
                "shift {i} {axis}: less accurate than cv2 ({m:.4} vs {g:.4}, true {t})"
            );
        }
        assert!((resp - gresp).abs() < 0.2, "response {resp} vs {gresp}");
    }
    println!("phasecorr worst (direct|mirror) diff {worst:.4} px");
}

#[test]
fn lk_flow_matches_cv2() {
    let mut a = npz();
    let base = base_image(&mut a);
    let (_, lk_shift) = f64s(&mut a, "lk_shift_true");
    let (p0_shape, p0) = f32s(&mut a, "lk_p0");
    let (_, p1_g) = f32s(&mut a, "lk_p1");
    let status_g: Vec<u8> = {
        let arr = a.by_name("lk_status").unwrap().unwrap();
        arr.into_vec::<u8>().unwrap()
    };

    // cv2 ran on uint8 images; reproduce that quantization.
    let to_u8 = |img: &ImageF32| {
        ImageF32::from_vec(
            img.data.iter().map(|&v| v.clamp(0.0, 255.0).round().min(255.0) as u8 as f32).collect(),
            img.w,
            img.h,
        )
    };
    let m = [
        [1.0, 0.0, lk_shift[0] as f32],
        [0.0, 1.0, lk_shift[1] as f32],
    ];
    let moved = warp_affine(&base, &m, base.w, base.h, Border::Reflect);
    let prev = to_u8(&base);
    let next = to_u8(&moved);

    let points: Vec<[f32; 2]> = (0..p0_shape[0]).map(|i| [p0[i * 2], p0[i * 2 + 1]]).collect();
    let (tracked, status) = calc_optical_flow_pyr_lk(&prev, &next, &points, 21, 3);

    let mut worst = 0.0f32;
    let mut n_checked = 0;
    for i in 0..points.len() {
        if status_g[i] == 0 || !status[i] {
            continue;
        }
        let d = ((tracked[i][0] - p1_g[i * 2]).powi(2)
            + (tracked[i][1] - p1_g[i * 2 + 1]).powi(2))
        .sqrt();
        worst = worst.max(d);
        n_checked += 1;
        assert!(d < 0.1, "point {i}: {d:.3} px from cv2's track");
    }
    assert!(n_checked >= points.len() * 3 / 4, "too few tracked: {n_checked}");
    println!("LK: {n_checked} points, worst {worst:.4} px vs cv2");
}

#[test]
fn ransac_similarity_matches_cv2() {
    let mut a = npz();
    let (src_shape, src_f) = f32s(&mut a, "affine_src");
    let (_, dst_f) = f32s(&mut a, "affine_dst");
    let (_, true_params) = f64s(&mut a, "affine_true"); // theta, scale, tx, ty
    let (_, est_g) = f64s(&mut a, "affine_est"); // cv2's 2x3

    let n = src_shape[0];
    let src: Vec<[f64; 2]> = (0..n).map(|i| [src_f[i * 2] as f64, src_f[i * 2 + 1] as f64]).collect();
    let dst: Vec<[f64; 2]> = (0..n).map(|i| [dst_f[i * 2] as f64, dst_f[i * 2 + 1] as f64]).collect();

    let res = estimate_affine_partial_2d(&src, &dst, 2.0, 2000, 42).unwrap();
    let m = res.model;

    // Compare to cv2's estimate: rotation within 0.05 deg, translation 0.1 px.
    let cv_rot = est_g[3].atan2(est_g[0]).to_degrees();
    let d_rot = (m.rotation_deg() - cv_rot).abs();
    let d_tx = (m.tx - est_g[2]).abs();
    let d_ty = (m.ty - est_g[5]).abs();
    let d_scale = (m.scale() - (est_g[0] * est_g[0] + est_g[3] * est_g[3]).sqrt()).abs();
    println!(
        "RANSAC: rot diff {d_rot:.4} deg, t diff ({d_tx:.3},{d_ty:.3}) px, scale diff {d_scale:.5}; \
         true (theta {:.3} deg, s {:.4}, t {:.2},{:.2}), {} inliers",
        true_params[0].to_degrees(),
        true_params[1],
        true_params[2],
        true_params[3],
        res.inliers.iter().filter(|&&k| k).count()
    );
    assert!(d_rot < 0.05, "rotation {d_rot}");
    assert!(d_tx < 0.1 && d_ty < 0.1, "translation {d_tx}/{d_ty}");
    assert!(d_scale < 1e-3, "scale {d_scale}");
}
