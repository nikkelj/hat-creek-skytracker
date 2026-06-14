//! PyO3 bindings (compiled only with the `extension-module` feature).
//!
//! Exposes the request encoders (for the byte-diff parity test) and a `SimMount`
//! class that wraps `Mount<LoopbackTransport>` and matches the method-level API
//! of `simulator.SimMount`, so it can later be swapped into the control loop.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::hotspot as core_hotspot;
use crate::pid::PidController as CorePid;
use crate::protocol;
use crate::sim::{LoopbackTransport, Mount, MountError, SimResponder};

fn to_pyerr(e: MountError) -> PyErr {
    PyIOError::new_err(e.to_string())
}

#[pyfunction]
fn encode_get_position(py: Python<'_>, target: u8) -> Bound<'_, PyBytes> {
    PyBytes::new_bound(py, &protocol::encode_get_position(target))
}

#[pyfunction]
fn encode_get_version(py: Python<'_>, target: u8) -> Bound<'_, PyBytes> {
    PyBytes::new_bound(py, &protocol::encode_get_version(target))
}

#[pyfunction]
fn encode_slew_fixed(py: Python<'_>, target: u8, rate: i32) -> Bound<'_, PyBytes> {
    PyBytes::new_bound(py, &protocol::encode_slew_fixed(target, rate))
}

#[pyfunction]
fn encode_goto_fast(py: Python<'_>, target: u8, dd: f64, mm: f64, ss: f64) -> Bound<'_, PyBytes> {
    PyBytes::new_bound(py, &protocol::encode_goto_fast(target, dd, mm, ss))
}

#[pyfunction]
fn pack_int3(py: Python<'_>, f: f64) -> Bound<'_, PyBytes> {
    PyBytes::new_bound(py, &protocol::pack_int3(f))
}

#[pyfunction]
fn unpack_int3(d: Vec<u8>) -> f64 {
    protocol::unpack_int3(&d)
}

/// Byte-level simulated mount. Drop-in for the used subset of
/// `simulator.SimMount`, but every call drives a real NexStar encode/transact/
/// parse round-trip through the in-memory loopback.
#[pyclass]
struct SimMount {
    mount: Mount<LoopbackTransport>,
}

#[pymethods]
impl SimMount {
    #[new]
    #[pyo3(signature = (az0_deg = 0.0, el0_deg = 0.0))]
    fn new(az0_deg: f64, el0_deg: f64) -> Self {
        let responder = SimResponder::new_wall(az0_deg, el0_deg);
        SimMount {
            mount: Mount::new(LoopbackTransport::new(responder)),
        }
    }

    /// `target_value` is the raw device id (use the module-level AZM/ALT consts).
    fn hc_get_position(&mut self, target_value: u8) -> PyResult<f64> {
        self.mount.hc_get_position(target_value).map_err(to_pyerr)
    }

    fn hc_slew_fixed(&mut self, target_value: u8, rate: i32) -> PyResult<bool> {
        self.mount
            .hc_slew_fixed(target_value, rate)
            .map_err(to_pyerr)
    }

    #[getter]
    fn az_true_deg(&self) -> f64 {
        self.mount.io.responder.az_true_deg
    }

    #[getter]
    fn el_true_deg(&self) -> f64 {
        self.mount.io.responder.el_true_deg
    }
}

/// PID controller. Port of `control.py::PIDController`; same constructor
/// defaults and `compute_pid_output` semantics.
#[pyclass]
struct PidController {
    inner: CorePid,
}

#[pymethods]
impl PidController {
    #[new]
    #[pyo3(signature = (p_gain = 1.0, i_gain = 0.0, d_gain = 0.0, axis_name = String::new(), feed_forward_enabled = false))]
    fn new(
        p_gain: f64,
        i_gain: f64,
        d_gain: f64,
        axis_name: String,
        feed_forward_enabled: bool,
    ) -> Self {
        PidController {
            inner: CorePid::new(p_gain, i_gain, d_gain, axis_name, feed_forward_enabled),
        }
    }

    #[pyo3(signature = (error_degrees, dt_seconds, measurement_degrees = None))]
    fn compute_pid_output(
        &mut self,
        error_degrees: f64,
        dt_seconds: f64,
        measurement_degrees: Option<f64>,
    ) -> (f64, i32) {
        self.inner
            .compute_pid_output(error_degrees, dt_seconds, measurement_degrees)
    }

    fn set_feed_forward_rate(&mut self, feed_forward_rate_deg_per_sec: f64) {
        self.inner
            .set_feed_forward_rate(feed_forward_rate_deg_per_sec);
    }

    fn set_feed_forward_enabled(&mut self, enabled: bool) {
        self.inner.set_feed_forward_enabled(enabled);
    }

    fn update_gains(&mut self, p_gain: f64, i_gain: f64, d_gain: f64) {
        self.inner.update_gains(p_gain, i_gain, d_gain);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn integral_error(&self) -> f64 {
        self.inner.integral_error
    }
    #[getter]
    fn previous_error(&self) -> f64 {
        self.inner.previous_error
    }
    #[getter]
    fn current_feed_forward_rate(&self) -> f64 {
        self.inner.current_feed_forward_rate
    }
    #[getter]
    fn current_pid_output(&self) -> f64 {
        self.inner.current_pid_output
    }
    #[getter]
    fn current_position_error(&self) -> f64 {
        self.inner.current_position_error
    }
}

/// Result of a successful hot-spot detection (image pixel coordinates).
/// Mirrors `hotspot.Detection`.
#[pyclass]
#[derive(Clone)]
struct Detection {
    #[pyo3(get)]
    cx: f64,
    #[pyo3(get)]
    cy: f64,
    #[pyo3(get)]
    peak: f64,
    #[pyo3(get)]
    background: f64,
    #[pyo3(get)]
    noise: f64,
    #[pyo3(get)]
    snr: f64,
    #[pyo3(get)]
    n_pixels: usize,
}

#[pymethods]
impl Detection {
    fn __repr__(&self) -> String {
        format!(
            "Detection(cx={:.2}, cy={:.2}, snr={:.1}, n={})",
            self.cx, self.cy, self.snr, self.n_pixels
        )
    }
}

/// Detect the brightest compact hot spot. Mirrors `hotspot.detect_hotspot`.
/// `image` must be a 2D float32 intensity map (apply `hotspot.to_intensity`
/// first for color/mono camera frames).
#[pyfunction]
#[pyo3(signature = (
    image,
    gate_center = None,
    gate_radius = None,
    snr_threshold = 5.0,
    centroid_radius = 12,
    half_max_fraction = 0.5,
    min_pixels = 3
))]
fn detect_hotspot(
    image: PyReadonlyArray2<'_, f32>,
    gate_center: Option<(f64, f64)>,
    gate_radius: Option<f64>,
    snr_threshold: f64,
    centroid_radius: usize,
    half_max_fraction: f64,
    min_pixels: usize,
) -> Option<Detection> {
    let view = image.as_array();
    let shape = view.shape();
    let (h, w) = (shape[0], shape[1]);
    let mut a = ndarray::Array2::<f64>::zeros((h, w));
    for ((i, j), v) in view.indexed_iter() {
        a[[i, j]] = *v as f64;
    }

    let gate = match (gate_center, gate_radius) {
        (Some((gx, gy)), Some(r)) => Some((gx, gy, r)),
        _ => None,
    };

    core_hotspot::detect_hotspot(
        &a.view(),
        gate,
        snr_threshold,
        centroid_radius,
        half_max_fraction,
        min_pixels,
    )
    .map(|d| Detection {
        cx: d.cx,
        cy: d.cy,
        peak: d.peak,
        background: d.background,
        noise: d.noise,
        snr: d.snr,
        n_pixels: d.n_pixels,
    })
}

/// Pixel offset -> (az_error_deg, el_error_deg). Mirrors
/// `hotspot.pixel_offset_to_angles`.
#[pyfunction]
#[pyo3(signature = (
    dx_pix,
    dy_pix,
    pixel_size_um,
    focal_length_mm,
    rotation_deg = 0.0,
    el_deg = 0.0,
    x_sign = 1.0,
    y_sign = -1.0,
    apply_cos_el = true
))]
#[allow(clippy::too_many_arguments)]
fn pixel_offset_to_angles(
    dx_pix: f64,
    dy_pix: f64,
    pixel_size_um: f64,
    focal_length_mm: f64,
    rotation_deg: f64,
    el_deg: f64,
    x_sign: f64,
    y_sign: f64,
    apply_cos_el: bool,
) -> (f64, f64) {
    core_hotspot::pixel_offset_to_angles(
        dx_pix,
        dy_pix,
        pixel_size_um,
        focal_length_mm,
        rotation_deg,
        el_deg,
        x_sign,
        y_sign,
        apply_cos_el,
    )
}

#[pymodule]
fn skytracker_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_get_position, m)?)?;
    m.add_function(wrap_pyfunction!(encode_get_version, m)?)?;
    m.add_function(wrap_pyfunction!(encode_slew_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(encode_goto_fast, m)?)?;
    m.add_function(wrap_pyfunction!(pack_int3, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_int3, m)?)?;
    m.add_function(wrap_pyfunction!(detect_hotspot, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_offset_to_angles, m)?)?;
    m.add_class::<SimMount>()?;
    m.add_class::<PidController>()?;
    m.add_class::<Detection>()?;

    // Device target ids, so Python can pass them without re-deriving the map.
    m.add("ANY", protocol::targets::ANY)?;
    m.add("HC", protocol::targets::HC)?;
    m.add("AZM", protocol::targets::AZM)?;
    m.add("ALT", protocol::targets::ALT)?;
    m.add("FOCUS", protocol::targets::FOCUS)?;
    Ok(())
}
