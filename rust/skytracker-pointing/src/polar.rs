//! Polar-axis fit (port of polar_align.fit_polar_axis): the RA axis is the
//! normal to the best-fit plane through boresight directions swept while
//! only the RA axis turned — the smallest right-singular vector of the
//! centered points, oriented toward the pole-side reference.

use nalgebra::{DMatrix, Vector3};

pub fn cartesian_from_az_el(az_deg: f64, el_deg: f64) -> Vector3<f64> {
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    // Matches transformations.cartesian_from_az_el: x east, y north, z up.
    Vector3::new(el.cos() * az.sin(), el.cos() * az.cos(), el.sin())
}

pub fn az_el_from_cartesian(v: &Vector3<f64>) -> (f64, f64) {
    let az = v[0].atan2(v[1]).to_degrees().rem_euclid(360.0);
    let el = (v[2] / v.norm()).clamp(-1.0, 1.0).asin().to_degrees();
    (az, el)
}

/// Best-fit RA-axis direction (az_deg, el_deg); needs >= 3 samples.
pub fn fit_polar_axis(
    samples_azel: &[[f64; 2]],
    toward_az_deg: f64,
    toward_alt_deg: f64,
) -> Option<(f64, f64)> {
    if samples_azel.len() < 3 {
        return None;
    }
    let pts: Vec<Vector3<f64>> = samples_azel
        .iter()
        .map(|&[az, el]| cartesian_from_az_el(az, el))
        .collect();
    let centroid = pts.iter().sum::<Vector3<f64>>() / pts.len() as f64;
    let mut m = DMatrix::zeros(pts.len(), 3);
    for (i, p) in pts.iter().enumerate() {
        let c = p - centroid;
        m[(i, 0)] = c[0];
        m[(i, 1)] = c[1];
        m[(i, 2)] = c[2];
    }
    let svd = m.svd(false, true);
    let v_t = svd.v_t.as_ref()?;
    // Smallest singular value's right vector = plane normal. nalgebra does
    // not guarantee ordering, so pick the row with the smallest s.
    let mut min_i = 0;
    for i in 1..svd.singular_values.len() {
        if svd.singular_values[i] < svd.singular_values[min_i] {
            min_i = i;
        }
    }
    let mut axis = Vector3::new(v_t[(min_i, 0)], v_t[(min_i, 1)], v_t[(min_i, 2)]);
    axis /= axis.norm();
    if axis.dot(&cartesian_from_az_el(toward_az_deg, toward_alt_deg)) < 0.0 {
        axis = -axis;
    }
    Some(az_el_from_cartesian(&axis))
}
