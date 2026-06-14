//! PyO3 bindings (compiled only with the `extension-module` feature).
//!
//! Exposes the request encoders (for the byte-diff parity test) and a `SimMount`
//! class that wraps `Mount<LoopbackTransport>` and matches the method-level API
//! of `simulator.SimMount`, so it can later be swapped into the control loop.

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

#[pymodule]
fn skytracker_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_get_position, m)?)?;
    m.add_function(wrap_pyfunction!(encode_get_version, m)?)?;
    m.add_function(wrap_pyfunction!(encode_slew_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(encode_goto_fast, m)?)?;
    m.add_function(wrap_pyfunction!(pack_int3, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_int3, m)?)?;
    m.add_class::<SimMount>()?;
    m.add_class::<PidController>()?;

    // Device target ids, so Python can pass them without re-deriving the map.
    m.add("ANY", protocol::targets::ANY)?;
    m.add("HC", protocol::targets::HC)?;
    m.add("AZM", protocol::targets::AZM)?;
    m.add("ALT", protocol::targets::ALT)?;
    m.add("FOCUS", protocol::targets::FOCUS)?;
    Ok(())
}
