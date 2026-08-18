//! PyO3 bindings for skytracker-imaging (Phase 3b seam). Gray (2-D f32)
//! kernels only — the Python adapters keep cv2 for color paths and all
//! orchestration.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use skytracker_imaging::enhance;
use skytracker_imaging::gridshift;
use skytracker_imaging::image::ImageF32;
use skytracker_imaging::metrics;
use skytracker_imaging::stabilize::{self, StabilizeParams};
use skytracker_imaging::warp::Border;

fn to_image(arr: &PyReadonlyArray2<'_, f32>) -> ImageF32 {
    let a = arr.as_array();
    ImageF32::from_vec(a.iter().copied().collect(), a.shape()[1], a.shape()[0])
}

fn to_pyarray<'py>(py: Python<'py>, img: &ImageF32) -> Bound<'py, PyArray2<f32>> {
    let nd = ndarray::Array2::from_shape_vec((img.h, img.w), img.data.clone()).unwrap();
    PyArray2::from_owned_array_bound(py, nd)
}

/// Sharpness score: method "laplacian" or "tenengrad" (stacking.sharpness
/// numeric core; ROI/scale handled by the Python caller).
#[pyfunction]
fn imaging_sharpness(py: Python<'_>, gray: PyReadonlyArray2<'_, f32>, method: &str) -> PyResult<f64> {
    let img = to_image(&gray);
    let m = method.to_string();
    py.allow_threads(move || match m.as_str() {
        "laplacian" => Ok(metrics::sharpness_laplacian(&img)),
        "tenengrad" => Ok(metrics::sharpness_tenengrad(&img)),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown method {other}"
        ))),
    })
}

/// INTER_AREA downscale (for the sharpness scale<1 pass).
#[pyfunction]
fn imaging_resize_area<'py>(
    py: Python<'py>,
    gray: PyReadonlyArray2<'py, f32>,
    nw: usize,
    nh: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let img = to_image(&gray);
    let out = py.allow_threads(move || metrics::resize_area(&img, nw, nh));
    Ok(to_pyarray(py, &out))
}

/// Intensity-weighted centroid (cx, cy) or None (stacking.brightness_centroid core).
#[pyfunction]
#[pyo3(signature = (gray, threshold = None))]
fn imaging_brightness_centroid(
    py: Python<'_>,
    gray: PyReadonlyArray2<'_, f32>,
    threshold: Option<f64>,
) -> PyResult<Option<(f64, f64)>> {
    let img = to_image(&gray);
    Ok(py.allow_threads(move || metrics::brightness_centroid(&img, threshold)))
}

/// Per-point local shifts (stacking.measure_local_shifts).
#[pyfunction]
#[pyo3(signature = (ref_gray, cur_gray, points, patch, min_response = 0.0, max_shift = None))]
fn imaging_measure_local_shifts<'py>(
    py: Python<'py>,
    ref_gray: PyReadonlyArray2<'py, f32>,
    cur_gray: PyReadonlyArray2<'py, f32>,
    points: Vec<[f64; 2]>,
    patch: usize,
    min_response: f64,
    max_shift: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let r = to_image(&ref_gray);
    let c = to_image(&cur_gray);
    let shifts = py.allow_threads(move || {
        gridshift::measure_local_shifts(&r, &c, &points, patch, min_response, max_shift)
    });
    let flat: Vec<f64> = shifts.iter().flatten().copied().collect();
    let nd = ndarray::Array2::from_shape_vec((shifts.len(), 2), flat).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, nd))
}

/// Dense grid warp (stacking.warp_by_grid, gray plane).
#[pyfunction]
fn imaging_warp_by_grid<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray2<'py, f32>,
    rows: usize,
    cols: usize,
    shifts: Vec<[f64; 2]>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let img = to_image(&frame);
    let out = py.allow_threads(move || {
        gridshift::warp_by_grid(&img, rows, cols, &shifts, Border::Reflect)
    });
    Ok(to_pyarray(py, &out))
}

/// Reference feature detection for the flow stabilizer: (N, 2) x/y.
#[pyfunction]
fn imaging_detect_flow_reference<'py>(
    py: Python<'py>,
    ref_gray: PyReadonlyArray2<'py, f32>,
    max_features: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let img = to_image(&ref_gray);
    let pts = py.allow_threads(move || stabilize::detect_reference_points(&img, max_features));
    let flat: Vec<f32> = pts.iter().flatten().copied().collect();
    let nd = ndarray::Array2::from_shape_vec((pts.len(), 2), flat).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, nd))
}

/// Flow-method transform estimate. Returns (M_2x3_rows | None, inliers, reason | None).
#[pyfunction]
#[pyo3(signature = (ref_gray, ref_points, cur_gray, min_inliers = 8, min_inlier_ratio = 0.4,
                    scale_tol = 0.12, max_rotation_deg = 20.0, max_translation_frac = 0.6))]
#[allow(clippy::too_many_arguments)]
fn imaging_estimate_flow(
    py: Python<'_>,
    ref_gray: PyReadonlyArray2<'_, f32>,
    ref_points: Vec<[f32; 2]>,
    cur_gray: PyReadonlyArray2<'_, f32>,
    min_inliers: usize,
    min_inlier_ratio: f64,
    scale_tol: f64,
    max_rotation_deg: f64,
    max_translation_frac: f64,
) -> PyResult<(Option<[[f64; 3]; 2]>, usize, Option<String>)> {
    let r = to_image(&ref_gray);
    let c = to_image(&cur_gray);
    let params = StabilizeParams {
        max_features: ref_points.len().max(1),
        ransac_threshold: 3.0,
        min_inliers,
        min_inlier_ratio,
        scale_tol,
        max_rotation_deg,
        max_translation_frac,
    };
    let est =
        py.allow_threads(move || stabilize::estimate_flow(&r, &ref_points, &c, &params));
    Ok((est.m, est.num_inliers, est.reject_reason))
}

/// Multi-scale unsharp + optional auto-stretch on a [0,1] gray image
/// (sharpen.unsharp_layers / auto_stretch composition).
#[pyfunction]
#[pyo3(signature = (img01, layers, stretch = true, black_pct = 0.25, white_pct = 99.9,
                    target_median = 0.25, max_gamma = 5.0))]
#[allow(clippy::too_many_arguments)]
fn imaging_finish_gray<'py>(
    py: Python<'py>,
    img01: PyReadonlyArray2<'py, f32>,
    layers: Vec<(f64, f64)>,
    stretch: bool,
    black_pct: f64,
    white_pct: f64,
    target_median: f64,
    max_gamma: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let img = to_image(&img01);
    let out = py.allow_threads(move || {
        let sharp = enhance::unsharp_layers(&img, &layers);
        if stretch {
            enhance::auto_stretch(&sharp, black_pct, white_pct, target_median, max_gamma)
        } else {
            sharp
        }
    });
    Ok(to_pyarray(py, &out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(imaging_sharpness, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_resize_area, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_brightness_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_measure_local_shifts, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_warp_by_grid, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_detect_flow_reference, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_estimate_flow, m)?)?;
    m.add_function(wrap_pyfunction!(imaging_finish_gray, m)?)?;
    m.add("IMAGING_AVAILABLE", true)?;
    Ok(())
}
