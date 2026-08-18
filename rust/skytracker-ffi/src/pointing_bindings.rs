//! PyO3 bindings for skytracker-pointing (Phase 2b seam). Fit entry points
//! only — the Python model classes keep applying corrections; their `fit`
//! classmethods route here behind the flag.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use skytracker_pointing::{altaz, eq, polar};

fn free_indices(free_terms: Option<Vec<String>>, names: &[&str; 7]) -> Option<Vec<usize>> {
    // Mirrors Python: `if not free_idx` (None, empty, or unrecognized) -> full fit.
    let list = free_terms?;
    let idx: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| list.iter().any(|t| t == *n))
        .map(|(i, _)| i)
        .collect();
    if idx.is_empty() {
        None
    } else {
        Some(idx)
    }
}

fn seed_array(seed_terms: Option<&Bound<'_, PyDict>>, names: &[&str; 7]) -> PyResult<[f64; 7]> {
    let mut seed = [0.0; 7];
    if let Some(d) = seed_terms {
        for (i, name) in names.iter().enumerate() {
            if let Some(v) = d.get_item(name)? {
                seed[i] = v.extract::<f64>()?;
            }
        }
    }
    Ok(seed)
}

fn stats_dict<'py>(
    py: Python<'py>,
    stats: &skytracker_pointing::fit::FitStats,
    names: &[&str; 7],
) -> PyResult<Bound<'py, PyDict>> {
    let terms = PyDict::new_bound(py);
    for (i, name) in names.iter().enumerate() {
        terms.set_item(name, stats.terms[i])?;
    }
    let out = PyDict::new_bound(py);
    out.set_item("terms", terms)?;
    out.set_item("n_samples", stats.n_samples)?;
    out.set_item("n_rejected", stats.n_rejected)?;
    out.set_item("design_cond", stats.design_cond)?;
    out.set_item("rms_before_deg", stats.rms_before_deg)?;
    out.set_item("rms_after_deg", stats.rms_after_deg)?;
    out.set_item("rms_before_arcmin", stats.rms_before_deg * 60.0)?;
    out.set_item("rms_after_arcmin", stats.rms_after_deg * 60.0)?;
    Ok(out)
}

/// Alt-az 7-term fit; returns the stats dict (terms included) shaped like
/// PointingModel.fit's stats.
#[pyfunction]
#[pyo3(signature = (samples, remove_refraction = false, seed_terms = None,
                    free_terms = None, robust = false, robust_sigma = 4.0,
                    robust_floor_deg = 30.0 / 3600.0))]
#[allow(clippy::too_many_arguments)]
fn fit_pointing_model<'py>(
    py: Python<'py>,
    samples: Vec<[f64; 4]>,
    remove_refraction: bool,
    seed_terms: Option<Bound<'py, PyDict>>,
    free_terms: Option<Vec<String>>,
    robust: bool,
    robust_sigma: f64,
    robust_floor_deg: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let names = &altaz::TERM_NAMES;
    let seed = seed_array(seed_terms.as_ref(), names)?;
    let free_idx = free_indices(free_terms, names);
    let stats = py.allow_threads(|| {
        altaz::fit_altaz(
            &samples,
            remove_refraction,
            seed,
            free_idx,
            robust,
            robust_sigma,
            robust_floor_deg,
        )
    });
    stats_dict(py, &stats, names)
}

/// Equatorial 7-term fit at the given latitude.
#[pyfunction]
#[pyo3(signature = (samples, lat_deg, seed_terms = None, free_terms = None,
                    robust = false, robust_sigma = 4.0,
                    robust_floor_deg = 30.0 / 3600.0))]
#[allow(clippy::too_many_arguments)]
fn fit_eq_pointing_model<'py>(
    py: Python<'py>,
    samples: Vec<[f64; 4]>,
    lat_deg: f64,
    seed_terms: Option<Bound<'py, PyDict>>,
    free_terms: Option<Vec<String>>,
    robust: bool,
    robust_sigma: f64,
    robust_floor_deg: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let names = &eq::TERM_NAMES;
    let seed = seed_array(seed_terms.as_ref(), names)?;
    let free_idx = free_indices(free_terms, names);
    let stats = py.allow_threads(|| {
        eq::fit_eq(
            &samples,
            lat_deg,
            seed,
            free_idx,
            robust,
            robust_sigma,
            robust_floor_deg,
        )
    });
    stats_dict(py, &stats, names)
}

/// Polar-axis plane fit: (axis_az_deg, axis_el_deg), or None for < 3 samples.
#[pyfunction]
#[pyo3(signature = (samples_azel, toward_az_deg = 0.0, toward_alt_deg = 45.0))]
fn fit_polar_axis(
    samples_azel: Vec<[f64; 2]>,
    toward_az_deg: f64,
    toward_alt_deg: f64,
) -> Option<(f64, f64)> {
    polar::fit_polar_axis(&samples_azel, toward_az_deg, toward_alt_deg)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_pointing_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_eq_pointing_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_polar_axis, m)?)?;
    m.add("POINTING_AVAILABLE", true)?;
    Ok(())
}
