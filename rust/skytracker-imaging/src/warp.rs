//! cv2.warpAffine / cv2.remap parity for f32 images with INTER_LINEAR.
//! (On float images OpenCV uses the exact float bilinear path — no
//! fixed-point quantization — so parity is at float rounding level.)

use crate::image::{sample_bilinear_const, sample_bilinear_reflect, ImageF32};

#[derive(Clone, Copy)]
pub enum Border {
    Constant(f32),
    Reflect,
}

/// cv2.warpAffine(src, M, dsize): dst(x, y) = src(inv(M) . (x, y, 1)),
/// i.e. the 2x3 forward matrix is inverted internally (no
/// WARP_INVERSE_MAP flag).
pub fn warp_affine(src: &ImageF32, m: &[[f32; 3]; 2], w: usize, h: usize, border: Border) -> ImageF32 {
    // Invert the 2x3 affine.
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let inv_det = 1.0 / det;
    let a = m[1][1] * inv_det;
    let b = -m[0][1] * inv_det;
    let c = -m[1][0] * inv_det;
    let d = m[0][0] * inv_det;
    let tx = -(a * m[0][2] + b * m[1][2]);
    let ty = -(c * m[0][2] + d * m[1][2]);

    // cv2 computes source coordinates in fixed point with 5 fractional
    // bits (INTER_TAB_SIZE = 32) even for float images; quantize the same
    // way or sub-1/32 fractions (e.g. a 0.3 px shift) diverge.
    const TAB: f32 = 32.0;
    let mut out = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = (a * x as f32 + b * y as f32 + tx) * TAB;
            let sy = (c * x as f32 + d * y as f32 + ty) * TAB;
            let sx = sx.round() / TAB;
            let sy = sy.round() / TAB;
            *out.at_mut(y, x) = match border {
                Border::Constant(v) => sample_bilinear_const(src, sx, sy, v),
                Border::Reflect => sample_bilinear_reflect(src, sx, sy),
            };
        }
    }
    out
}

/// cv2.remap(src, map_x, map_y, INTER_LINEAR): dst(x,y) = src(map_x(x,y), map_y(x,y)).
pub fn remap(src: &ImageF32, map_x: &ImageF32, map_y: &ImageF32, border: Border) -> ImageF32 {
    let (w, h) = (map_x.w, map_x.h);
    let mut out = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = map_x.at(y, x);
            let sy = map_y.at(y, x);
            *out.at_mut(y, x) = match border {
                Border::Constant(v) => sample_bilinear_const(src, sx, sy, v),
                Border::Reflect => sample_bilinear_reflect(src, sx, sy),
            };
        }
    }
    out
}
