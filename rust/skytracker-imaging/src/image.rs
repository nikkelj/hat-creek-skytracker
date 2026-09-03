//! Minimal f32 grayscale image type + OpenCV border conventions shared by
//! the imaging modules.

#[derive(Clone)]
pub struct ImageF32 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl ImageF32 {
    pub fn new(w: usize, h: usize) -> Self {
        ImageF32 {
            w,
            h,
            data: vec![0.0; w * h],
        }
    }

    pub fn from_vec(data: Vec<f32>, w: usize, h: usize) -> Self {
        assert_eq!(data.len(), w * h);
        ImageF32 { w, h, data }
    }

    #[inline]
    pub fn at(&self, y: usize, x: usize) -> f32 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn at_mut(&mut self, y: usize, x: usize) -> &mut f32 {
        &mut self.data[y * self.w + x]
    }
}

/// OpenCV BORDER_REFLECT_101 index mapping (gfedcb|abcdefgh|gfedcba):
/// mirror about the edge WITHOUT duplicating the edge sample. The default
/// border for filter2D / GaussianBlur / Sobel / Laplacian.
#[inline]
pub fn reflect101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    // Period of the reflection is 2(n-1).
    let period = 2 * (n - 1);
    i = i.rem_euclid(period);
    if i >= n {
        i = period - i;
    }
    i as usize
}

/// Bilinear sample with BORDER_CONSTANT (value) semantics at float coords.
#[inline]
pub fn sample_bilinear_const(img: &ImageF32, x: f32, y: f32, border: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let get = |yy: isize, xx: isize| -> f32 {
        if yy < 0 || xx < 0 || yy >= img.h as isize || xx >= img.w as isize {
            border
        } else {
            img.at(yy as usize, xx as usize)
        }
    };
    let (xi, yi) = (x0 as isize, y0 as isize);
    let v00 = get(yi, xi);
    let v01 = get(yi, xi + 1);
    let v10 = get(yi + 1, xi);
    let v11 = get(yi + 1, xi + 1);
    v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy) + v10 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Bilinear sample with BORDER_REFLECT (edge-duplicating, cv2 BORDER_REFLECT)
/// semantics — used by the golden warps rendered with borderMode=REFLECT.
#[inline]
pub fn sample_bilinear_reflect(img: &ImageF32, x: f32, y: f32) -> f32 {
    let reflect = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let period = 2 * n;
        let mut i = i.rem_euclid(period);
        if i >= n {
            i = period - 1 - i;
        }
        i as usize
    };
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let (xi, yi) = (x0 as isize, y0 as isize);
    let get = |yy: isize, xx: isize| -> f32 { img.at(reflect(yy, img.h), reflect(xx, img.w)) };
    let v00 = get(yi, xi);
    let v01 = get(yi, xi + 1);
    let v10 = get(yi + 1, xi);
    let v11 = get(yi + 1, xi + 1);
    v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy) + v10 * (1.0 - fx) * fy + v11 * fx * fy
}
