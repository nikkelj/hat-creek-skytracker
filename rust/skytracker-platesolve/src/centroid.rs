//! Port of tetra3.get_centroids_from_image for the parameter set the app
//! uses (the defaults): local-mean background subtraction (25-px boxcar,
//! reflect boundary), global root-square sigma threshold (sigma=2), 3x3
//! cross binary opening, 4-connectivity labeling, intensity-weighted
//! centroids with min_area=5 / max_area=100, sorted brightest first.
//! Gate: within 0.3 px of Python tetra3 on the golden frames.

pub struct CentroidParams {
    pub sigma: f64,
    pub filtsize: usize,
    pub min_area: usize,
    pub max_area: usize,
}

impl Default for CentroidParams {
    fn default() -> Self {
        CentroidParams {
            sigma: 2.0,
            filtsize: 25,
            min_area: 5,
            max_area: 100,
        }
    }
}

/// Separable boxcar mean filter with scipy's 'reflect' boundary
/// (edge sample duplicated: d c b a | a b c d).
fn uniform_filter(image: &[f32], h: usize, w: usize, size: usize) -> Vec<f32> {
    let half = size / 2;
    let reflect = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let mut i = i;
        // One reflection suffices for size << n.
        if i < 0 {
            i = -i - 1;
        }
        if i >= n {
            i = 2 * n - 1 - i;
        }
        i as usize
    };

    // Horizontal pass (running sum: O(n) instead of O(n * size)).
    let mut tmp = vec![0f32; h * w];
    for y in 0..h {
        let row = &image[y * w..(y + 1) * w];
        let mut acc = 0f64;
        for k in 0..size {
            acc += row[reflect(k as isize - half as isize, w)] as f64;
        }
        for x in 0..w {
            tmp[y * w + x] = (acc / size as f64) as f32;
            let leave = reflect(x as isize - half as isize, w);
            let enter = reflect(x as isize + half as isize + 1, w);
            acc += row[enter] as f64 - row[leave] as f64;
        }
    }
    // Vertical pass.
    let mut out = vec![0f32; h * w];
    for x in 0..w {
        let mut acc = 0f64;
        for k in 0..size {
            acc += tmp[reflect(k as isize - half as isize, h) * w + x] as f64;
        }
        for y in 0..h {
            out[y * w + x] = (acc / size as f64) as f32;
            let leave = reflect(y as isize - half as isize, h);
            let enter = reflect(y as isize + half as isize + 1, h);
            acc += tmp[enter * w + x] as f64 - tmp[leave * w + x] as f64;
        }
    }
    out
}

/// Binary erosion/dilation with the 3x3 cross (scipy default structure,
/// border_value=0 outside).
fn binary_open(mask: &mut [bool], h: usize, w: usize) {
    let at = |m: &[bool], y: isize, x: isize| -> bool {
        if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
            false
        } else {
            m[y as usize * w + x as usize]
        }
    };
    let mut eroded = vec![false; h * w];
    for y in 0..h as isize {
        for x in 0..w as isize {
            eroded[y as usize * w + x as usize] = at(mask, y, x)
                && at(mask, y - 1, x)
                && at(mask, y + 1, x)
                && at(mask, y, x - 1)
                && at(mask, y, x + 1);
        }
    }
    for y in 0..h as isize {
        for x in 0..w as isize {
            mask[y as usize * w + x as usize] = at(&eroded, y, x)
                || at(&eroded, y - 1, x)
                || at(&eroded, y + 1, x)
                || at(&eroded, y, x - 1)
                || at(&eroded, y, x + 1);
        }
    }
}

/// (y, x) centroids brightest-first, matching tetra3's pixel convention
/// (centre of top-left pixel is (0.5, 0.5)).
pub fn get_centroids(image_u8: &[u8], h: usize, w: usize, params: &CentroidParams) -> Vec<[f64; 2]> {
    let image: Vec<f32> = image_u8.iter().map(|&v| v as f32).collect();

    // Background subtraction (local mean).
    let bg = uniform_filter(&image, h, w, params.filtsize);
    let sub: Vec<f32> = image.iter().zip(bg.iter()).map(|(a, b)| a - b).collect();

    // Global root-square sigma -> threshold.
    let mean_sq: f64 = sub.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / sub.len() as f64;
    let threshold = mean_sq.sqrt() * params.sigma;

    let mut mask: Vec<bool> = sub.iter().map(|&v| (v as f64) > threshold).collect();
    binary_open(&mut mask, h, w);

    // 4-connectivity labeling via BFS flood fill in raster order (matches
    // scipy.ndimage.label region numbering).
    let mut label = vec![0u32; h * w];
    let mut regions: Vec<Vec<usize>> = Vec::new();
    let mut queue = Vec::new();
    for start in 0..h * w {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        let id = regions.len() as u32 + 1;
        let mut pixels = Vec::new();
        queue.clear();
        queue.push(start);
        label[start] = id;
        while let Some(p) = queue.pop() {
            pixels.push(p);
            let (y, x) = (p / w, p % w);
            let mut push = |q: usize| {
                if mask[q] && label[q] == 0 {
                    label[q] = id;
                    queue.push(q);
                }
            };
            if y > 0 {
                push(p - w);
            }
            if y + 1 < h {
                push(p + w);
            }
            if x > 0 {
                push(p - 1);
            }
            if x + 1 < w {
                push(p + 1);
            }
        }
        pixels.sort_unstable(); // raster order like labeled_comprehension positions
        regions.push(pixels);
    }

    // Stats + area filters, then sort brightest (sum) first.
    let mut spots: Vec<(f64, f64, f64)> = Vec::new(); // (m0, y, x)
    for pixels in &regions {
        let area = pixels.len();
        let mut m0 = 0f64;
        let mut mx = 0f64;
        let mut my = 0f64;
        for &p in pixels {
            let a = sub[p] as f64;
            let (y, x) = ((p / w) as f64, (p % w) as f64);
            m0 += a;
            mx += x * a;
            my += y * a;
        }
        if area < params.min_area || area > params.max_area {
            continue;
        }
        spots.push((m0, my / m0 + 0.5, mx / m0 + 0.5));
    }
    spots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    spots.iter().map(|&(_, y, x)| [y, x]).collect()
}
