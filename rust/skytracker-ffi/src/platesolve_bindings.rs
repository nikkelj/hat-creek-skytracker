//! PyO3 bindings for the skytracker-platesolve crate (Phase 2 seam).

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skytracker_platesolve::centroid::{get_centroids, CentroidParams};
use skytracker_platesolve::db::SolverDatabase;
use skytracker_platesolve::solve::{solve_from_centroids, SolveParams};

#[pyclass]
pub struct PlateSolver {
    db: SolverDatabase,
}

#[pymethods]
impl PlateSolver {
    /// `PlateSolver(db_path)` — path to a tetra3 .npz pattern database
    /// (loaded as-is; fails loudly on unknown schema).
    #[new]
    fn new(py: Python<'_>, db_path: String) -> PyResult<Self> {
        let db = py
            .allow_threads(|| SolverDatabase::load(std::path::Path::new(&db_path)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PlateSolver { db })
    }

    /// tetra3.get_centroids_from_image with the app's default parameters:
    /// (N, 2) ndarray of (y, x), brightest first.
    fn get_centroids<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = image.as_array();
        let (h, w) = (arr.shape()[0], arr.shape()[1]);
        let data: Vec<u8> = arr.iter().copied().collect();
        let cents =
            py.allow_threads(|| get_centroids(&data, h, w, &CentroidParams::default()));
        let flat: Vec<f64> = cents.iter().flatten().copied().collect();
        let nd = ndarray::Array2::from_shape_vec((cents.len(), 2), flat).unwrap();
        Ok(PyArray2::from_owned_array_bound(py, nd))
    }

    /// Solve an image: centroid extraction + lost-in-space solve. Returns
    /// a dict shaped like tetra3's solve_from_image result (the subset of
    /// keys plate_solver.py reads), or None if no match.
    #[pyo3(signature = (image, fov_estimate, fov_max_error = None,
                        pattern_checking_stars = 8, match_radius = 0.01,
                        match_threshold = 1e-3))]
    fn solve_from_image<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
        fov_estimate: f64,
        fov_max_error: Option<f64>,
        pattern_checking_stars: usize,
        match_radius: f64,
        match_threshold: f64,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let arr = image.as_array();
        let (h, w) = (arr.shape()[0], arr.shape()[1]);
        let data: Vec<u8> = arr.iter().copied().collect();
        let params = SolveParams {
            fov_estimate_deg: fov_estimate,
            fov_max_error_deg: fov_max_error,
            pattern_checking_stars,
            match_radius,
            match_threshold,
            distortion: 0.0,
        };
        let solution = py.allow_threads(|| {
            let cents = get_centroids(&data, h, w, &CentroidParams::default());
            solve_from_centroids(&self.db, &cents, h, w, &params)
        });
        match solution {
            None => Ok(None),
            Some(s) => {
                let out = PyDict::new_bound(py);
                out.set_item("RA", s.ra_deg)?;
                out.set_item("Dec", s.dec_deg)?;
                out.set_item("Roll", s.roll_deg)?;
                out.set_item("FOV", s.fov_deg)?;
                out.set_item("distortion", s.distortion)?;
                out.set_item("RMSE", s.rmse_arcsec)?;
                out.set_item("Matches", s.matches)?;
                out.set_item("Prob", s.prob)?;
                let flat: Vec<f64> = s.matched_centroids.iter().flatten().copied().collect();
                let nd = ndarray::Array2::from_shape_vec((s.matched_centroids.len(), 2), flat)
                    .unwrap();
                out.set_item("matched_centroids", PyArray2::from_owned_array_bound(py, nd))?;
                Ok(Some(out))
            }
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlateSolver>()?;
    m.add("PLATESOLVE_AVAILABLE", true)?;
    Ok(())
}
