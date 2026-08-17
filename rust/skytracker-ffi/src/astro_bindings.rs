//! PyO3 bindings for the skytracker-astro engine (Phase 1 strangler seam).
//!
//! One `AstroEngine` instance is long-lived on the Python side (the
//! adapter holds it as a module global); heavy calls release the GIL and
//! fan out with rayon inside the engine.

use std::sync::Mutex;

use numpy::PyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skytracker_astro::engine::{Engine, TrackTarget};
use skytracker_astro::sgp4_pass::Observer;

fn to_pyerr(e: skytracker_astro::engine::EngineError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn rows_to_array<'py>(py: Python<'py>, rows: &[[f64; 8]]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let arr = ndarray::Array2::from_shape_vec((rows.len(), 8), flat).unwrap();
    PyArray2::from_owned_array_bound(py, arr)
}

#[pyclass]
pub struct AstroEngine {
    inner: Mutex<Engine>,
}

#[pymethods]
impl AstroEngine {
    /// `AstroEngine(de421_path=None)` — pass the .bsp path to enable
    /// solar-system bodies and fixed-target apparent places.
    #[new]
    #[pyo3(signature = (de421_path = None))]
    fn new(de421_path: Option<String>) -> PyResult<Self> {
        let engine = Engine::new(de421_path.as_deref().map(std::path::Path::new))
            .map_err(to_pyerr)?;
        Ok(AstroEngine {
            inner: Mutex::new(engine),
        })
    }

    /// Parse a TLE file (Celestrak 3LE; tolerant of \r\r\n). Returns the
    /// number of satellites loaded. Call again to refresh after download.
    fn load_tle_file(&self, py: Python<'_>, path: String) -> PyResult<usize> {
        py.allow_threads(|| {
            self.inner
                .lock()
                .unwrap()
                .load_tle_file(std::path::Path::new(&path))
        })
        .map_err(to_pyerr)
    }

    /// Bulk trajectory precompute: dict {satnum_str: (n, 8) ndarray} in the
    /// canonical column order [time_tt, alt, az, dist_km, px, py, az_rate,
    /// el_rate]. Satellites that fail to propagate are omitted.
    #[pyo3(signature = (satnums, times_tt, lat_deg, lon_deg, elevation_m, cx, cy, radius))]
    #[allow(clippy::too_many_arguments)]
    fn precompute_trajectories<'py>(
        &self,
        py: Python<'py>,
        satnums: Vec<String>,
        times_tt: Vec<f64>,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let observer = Observer {
            lat_deg,
            lon_deg,
            elevation_m,
        };
        let results = py
            .allow_threads(|| {
                self.inner.lock().unwrap().precompute(
                    &satnums, &times_tt, &observer, cx, cy, radius,
                )
            })
            .map_err(to_pyerr)?;
        let out = PyDict::new_bound(py);
        for (satnum, rows) in &results {
            out.set_item(satnum, rows_to_array(py, rows))?;
        }
        Ok(out)
    }

    /// One satellite's rows on an arbitrary (e.g. fine) time grid.
    #[pyo3(signature = (satnum, times_tt, lat_deg, lon_deg, elevation_m, cx, cy, radius))]
    #[allow(clippy::too_many_arguments)]
    fn satellite_rows<'py>(
        &self,
        py: Python<'py>,
        satnum: String,
        times_tt: Vec<f64>,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let observer = Observer {
            lat_deg,
            lon_deg,
            elevation_m,
        };
        let rows = py
            .allow_threads(|| {
                self.inner.lock().unwrap().satellite_rows(
                    &satnum, &times_tt, &observer, cx, cy, radius,
                )
            })
            .map_err(to_pyerr)?;
        Ok(rows_to_array(py, &rows))
    }

    /// Which satnums rise above min_alt_deg at any sample time.
    #[pyo3(signature = (satnums, times_tt, lat_deg, lon_deg, elevation_m, min_alt_deg))]
    fn visible_satnums(
        &self,
        py: Python<'_>,
        satnums: Vec<String>,
        times_tt: Vec<f64>,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
        min_alt_deg: f64,
    ) -> PyResult<Vec<String>> {
        let observer = Observer {
            lat_deg,
            lon_deg,
            elevation_m,
        };
        py.allow_threads(|| {
            self.inner.lock().unwrap().visible_satnums(
                &satnums, &times_tt, &observer, min_alt_deg,
            )
        })
        .map_err(to_pyerr)
    }

    /// Apparent topocentric (alt_deg, az_deg, dist_km) of a named body.
    #[pyo3(signature = (name, jd_tt, lat_deg, lon_deg, elevation_m))]
    fn body_altaz_dist(
        &self,
        py: Python<'_>,
        name: String,
        jd_tt: f64,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
    ) -> PyResult<(f64, f64, f64)> {
        let observer = Observer {
            lat_deg,
            lon_deg,
            elevation_m,
        };
        py.allow_threads(|| {
            self.inner
                .lock()
                .unwrap()
                .body_altaz_dist(&name, jd_tt, &observer)
        })
        .map_err(to_pyerr)
    }

    /// 8-column tracking rows for a named body (celestial.build_trajectory).
    #[pyo3(signature = (name, times_tt, lat_deg, lon_deg, elevation_m))]
    fn body_rows<'py>(
        &self,
        py: Python<'py>,
        name: String,
        times_tt: Vec<f64>,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let observer = Observer {
            lat_deg,
            lon_deg,
            elevation_m,
        };
        let rows = py
            .allow_threads(|| {
                self.inner.lock().unwrap().target_rows(
                    &TrackTarget::Body(name),
                    &times_tt,
                    &observer,
                )
            })
            .map_err(to_pyerr)?;
        Ok(rows_to_array(py, &rows))
    }

    /// 8-column tracking rows for a fixed ICRS RA/Dec target (stars, DSOs).
    #[pyo3(signature = (ra_deg, dec_deg, times_tt, lat_deg, lon_deg, elevation_m))]
    fn fixed_rows<'py>(
        &self,
        py: Python<'py>,
        ra_deg: f64,
        dec_deg: f64,
        times_tt: Vec<f64>,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !(-90.0..=90.0).contains(&dec_deg) {
            return Err(PyValueError::new_err("dec_deg outside [-90, 90]"));
        }
        let observer = Observer {
            lat_deg,
            lon_deg,
            elevation_m,
        };
        let rows = py
            .allow_threads(|| {
                self.inner.lock().unwrap().target_rows(
                    &TrackTarget::FixedRadec { ra_deg, dec_deg },
                    &times_tt,
                    &observer,
                )
            })
            .map_err(to_pyerr)?;
        Ok(rows_to_array(py, &rows))
    }

    /// GAST in hours at a TT Julian date (exposed for spot checks).
    #[staticmethod]
    fn gast_hours(jd_tt: f64) -> f64 {
        skytracker_astro::time::gast_hours(jd_tt)
    }
}

/// Register the astro classes on the module (called from bindings.rs).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AstroEngine>()?;
    m.add("ASTRO_ENGINE_AVAILABLE", true)?;
    Ok(())
}
