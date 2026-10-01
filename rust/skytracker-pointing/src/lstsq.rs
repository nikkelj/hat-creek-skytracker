//! SVD least squares with numpy's `lstsq(rcond=None)` semantics: singular
//! values below eps * max(m, n) * s_max are treated as zero, and the
//! minimum-norm solution is returned. Also the design-matrix condition
//! number (np.linalg.cond: s_max / s_min).

use nalgebra::{DMatrix, DVector};

pub struct LstsqResult {
    pub x: Vec<f64>,
    pub cond: f64,
}

pub fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> LstsqResult {
    let (m, n) = a.shape();
    let svd = a.clone().svd(true, true);
    let s = &svd.singular_values;
    let s_max = s.iter().cloned().fold(0.0f64, f64::max);
    let s_min = s.iter().cloned().fold(f64::INFINITY, f64::min);
    let cond = if s_min > 0.0 && s_min.is_finite() {
        s_max / s_min
    } else {
        f64::INFINITY
    };
    let cutoff = f64::EPSILON * m.max(n) as f64 * s_max;

    let u = svd.u.as_ref().unwrap();
    let v_t = svd.v_t.as_ref().unwrap();
    // x = V . diag(1/s where s > cutoff) . U^T . b
    let utb = u.transpose() * b;
    let mut y = DVector::zeros(s.len());
    for i in 0..s.len() {
        if s[i] > cutoff {
            y[i] = utb[i] / s[i];
        }
    }
    let x = v_t.transpose() * y;
    LstsqResult {
        x: x.iter().copied().collect(),
        cond,
    }
}
