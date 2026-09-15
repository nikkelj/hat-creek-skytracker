//! OpenCV-parity filters on f32 images: Gaussian blur (separable, exact
//! getGaussianKernel weights), Laplacian (ksize=1 kernel), Sobel 3x3 —
//! all with BORDER_REFLECT_101, matching the cv2 defaults the stacking
//! pipeline uses.

use crate::image::{reflect101, ImageF32};

/// cv2.getGaussianKernel(ksize, sigma): normalized exp(-(i-c)^2 / (2s^2)).
/// For sigma <= 0 OpenCV derives sigma from ksize; the app always passes
/// explicit sigma, so that branch is omitted.
pub fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f32> {
    let c = (ksize as f64 - 1.0) / 2.0;
    let mut k: Vec<f64> = (0..ksize)
        .map(|i| (-(i as f64 - c).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k.iter().map(|&v| v as f32).collect()
}

/// Separable convolution with a symmetric 1-D kernel, BORDER_REFLECT_101.
pub fn sep_filter(img: &ImageF32, kernel: &[f32]) -> ImageF32 {
    let (w, h) = (img.w, img.h);
    let half = kernel.len() / 2;
    let mut tmp = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let xi = reflect101(x as isize + k as isize - half as isize, w);
                acc += img.at(y, xi) * kv;
            }
            *tmp.at_mut(y, x) = acc;
        }
    }
    let mut out = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let yi = reflect101(y as isize + k as isize - half as isize, h);
                acc += tmp.at(yi, x) * kv;
            }
            *out.at_mut(y, x) = acc;
        }
    }
    out
}

pub fn gaussian_blur(img: &ImageF32, ksize: usize, sigma: f64) -> ImageF32 {
    sep_filter(img, &gaussian_kernel(ksize, sigma))
}

/// Generic 3x3 filter, BORDER_REFLECT_101.
pub fn filter3x3(img: &ImageF32, k: &[[f32; 3]; 3]) -> ImageF32 {
    let (w, h) = (img.w, img.h);
    let mut out = ImageF32::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (dy, row) in k.iter().enumerate() {
                let yi = reflect101(y as isize + dy as isize - 1, h);
                for (dx, &kv) in row.iter().enumerate() {
                    let xi = reflect101(x as isize + dx as isize - 1, w);
                    acc += img.at(yi, xi) * kv;
                }
            }
            *out.at_mut(y, x) = acc;
        }
    }
    out
}

/// cv2.Laplacian(..., ksize=1): the 3x3 [[0,1,0],[1,-4,1],[0,1,0]] kernel.
pub fn laplacian(img: &ImageF32) -> ImageF32 {
    filter3x3(img, &[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
}

/// cv2.Sobel(..., dx=1, dy=0, ksize=3).
pub fn sobel_x(img: &ImageF32) -> ImageF32 {
    filter3x3(img, &[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
}

/// cv2.Sobel(..., dx=0, dy=1, ksize=3).
pub fn sobel_y(img: &ImageF32) -> ImageF32 {
    filter3x3(img, &[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
}
