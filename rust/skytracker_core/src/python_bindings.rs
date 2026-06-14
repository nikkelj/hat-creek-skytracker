//! PyO3 bindings (compiled only with the `extension-module` feature).
//!
//! Exposes the request encoders (for the byte-diff parity test) and a `SimMount`
//! class that wraps `Mount<LoopbackTransport>` and matches the method-level API
//! of `simulator.SimMount`, so it can later be swapped into the control loop.

// Several exported functions deliberately mirror the Python API names
// (AzAlt2AzEl, etc.) so they are drop-in from Python.
#![allow(non_snake_case)]

use std::sync::Arc;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::controller::{Frame, Inputs, LoopState, Mode, Setpoint};
use crate::core_loop::{run_cycle, Command, Shared};
use crate::hotspot as core_hotspot;
use crate::pid::PidController as CorePid;
use crate::protocol;
use crate::sim::{LoopbackTransport, Mount, MountError, SimResponder};
use crate::transforms::{self as core_tf, MountMode};

fn parse_mode(s: &str) -> PyResult<Mode> {
    match s.to_ascii_lowercase().as_str() {
        "standby" => Ok(Mode::Standby),
        "rate" | "rate_control" => Ok(Mode::Rate),
        "program" => Ok(Mode::Program),
        "handoff" => Ok(Mode::Handoff),
        "hotspot" => Ok(Mode::Hotspot),
        "mti" => Ok(Mode::Mti),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown mode: {other}"
        ))),
    }
}

fn mode_str(m: Mode) -> &'static str {
    match m {
        Mode::Standby => "standby",
        Mode::Rate => "rate",
        Mode::Program => "program",
        Mode::Handoff => "handoff",
        Mode::Hotspot => "hotspot",
        Mode::Mti => "mti",
    }
}

fn parse_mount_mode(s: &str) -> PyResult<MountMode> {
    match s.to_ascii_lowercase().as_str() {
        "altaz" => Ok(MountMode::AltAz),
        "passthrough" => Ok(MountMode::Passthrough),
        "eq" => Ok(MountMode::Eq),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown mount mode: {other}"
        ))),
    }
}

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

// --- coordinate transforms (port of transformations.py) ---

#[pyfunction]
fn cartesian_from_az_el(az_deg: f64, el_deg: f64) -> (f64, f64, f64) {
    let v = core_tf::cartesian_from_az_el(az_deg, el_deg);
    (v[0], v[1], v[2])
}

#[pyfunction]
fn az_el_from_cartesian(x: f64, y: f64, z: f64) -> (f64, f64) {
    core_tf::az_el_from_cartesian(&[x, y, z])
}

#[pyfunction]
fn AzAlt2AzEl(azm: f64, alt: f64, alignment_azimuth: f64, alignment_elevation: f64) -> (f64, f64) {
    core_tf::az_alt_to_az_el(azm, alt, alignment_azimuth, alignment_elevation)
}

#[pyfunction]
fn apply_rotation_to_az_el(az: f64, el: f64, rotation_angle: f64) -> (f64, f64) {
    core_tf::apply_rotation_to_az_el(az, el, rotation_angle)
}

#[pyfunction]
fn AzAlt2AzEl_AltAz(azm: f64, alt: f64, alignment_azimuth: f64) -> (f64, f64) {
    core_tf::az_alt_to_az_el_altaz(azm, alt, alignment_azimuth)
}

#[pyfunction]
fn AzEl2AzAlt_AltAz(az: f64, el: f64, alignment_azimuth: f64, alignment_elevation: f64) -> (f64, f64) {
    core_tf::az_el_to_az_alt_altaz(az, el, alignment_azimuth, alignment_elevation)
}

#[pyfunction]
fn AzAlt2AzEl_Passthrough(azm: f64, alt: f64) -> (f64, f64) {
    core_tf::az_alt_to_az_el_passthrough(azm, alt)
}

#[pyfunction]
fn AzEl2AzAlt_Passthrough(az: f64, el: f64) -> (f64, f64) {
    core_tf::az_el_to_az_alt_passthrough(az, el)
}

#[pyfunction]
#[pyo3(signature = (alt_tel_deg, az_tel_deg, lat_deg = 45.0))]
fn telescope_to_local_elev_az(alt_tel_deg: f64, az_tel_deg: f64, lat_deg: f64) -> (f64, f64) {
    core_tf::telescope_to_local_elev_az(alt_tel_deg, az_tel_deg, lat_deg)
}

#[pyfunction]
#[pyo3(signature = (el_local_deg, az_local_deg, lat_deg = 45.0))]
fn local_elev_az_to_telescope(el_local_deg: f64, az_local_deg: f64, lat_deg: f64) -> (f64, f64) {
    core_tf::local_elev_az_to_telescope(el_local_deg, az_local_deg, lat_deg)
}

#[pyfunction]
#[pyo3(signature = (el_deg, az_deg, lat_deg = 45.0, lst_hours = 17.89))]
fn altaz_local_to_radec(el_deg: f64, az_deg: f64, lat_deg: f64, lst_hours: f64) -> (f64, f64) {
    core_tf::altaz_local_to_radec(el_deg, az_deg, lat_deg, lst_hours)
}

/// Deterministic, sim-backed control loop for driving/validating the loop from
/// Python with no hardware. Wraps `Mount<LoopbackTransport>` (the byte-level
/// sim) + `LoopState`; Python pushes inputs/frames via setters and advances the
/// loop with `step(dt)`, exactly mirroring how the real threaded loop is fed.
///
/// The real hardware loop (serial transport + background thread) is the same
/// `core_loop::CoreLoop` already validated in Rust; this class exposes the
/// message-passing surface from Python.
#[pyclass]
struct SimCoreLoop {
    mount: Mount<LoopbackTransport>,
    state: LoopState,
    shared: Arc<Shared>,
    now: f64,
    frame_seq: u64,
}

#[pymethods]
impl SimCoreLoop {
    #[new]
    #[pyo3(signature = (az0_deg = 0.0, el0_deg = 0.0))]
    fn new(az0_deg: f64, el0_deg: f64) -> Self {
        let mut inputs = Inputs::default();
        inputs.connected = true;
        SimCoreLoop {
            mount: Mount::new(LoopbackTransport::new(SimResponder::new_manual(az0_deg, el0_deg))),
            state: LoopState::new(),
            shared: Shared::new(inputs),
            now: 0.0,
            frame_seq: 0,
        }
    }

    fn set_connected(&self, v: bool) {
        self.shared.inputs.lock().unwrap().connected = v;
    }
    fn set_stopped(&self, v: bool) {
        self.shared.inputs.lock().unwrap().stopped = v;
    }
    fn set_mode(&self, mode: &str) -> PyResult<()> {
        self.shared.inputs.lock().unwrap().mode = parse_mode(mode)?;
        Ok(())
    }
    fn set_gains(&self, azm_p: f64, azm_i: f64, azm_d: f64, alt_p: f64, alt_i: f64, alt_d: f64) {
        let mut i = self.shared.inputs.lock().unwrap();
        i.azm_gains = (azm_p, azm_i, azm_d);
        i.alt_gains = (alt_p, alt_i, alt_d);
    }
    fn set_limits(&self, azm_min: f64, azm_max: f64, alt_min: f64, alt_max: f64) {
        let mut i = self.shared.inputs.lock().unwrap();
        i.azm_limit = (azm_min, azm_max);
        i.alt_limit = (alt_min, alt_max);
    }
    fn set_offsets(&self, azm: f64, alt: f64) {
        self.shared.inputs.lock().unwrap().offsets = (azm, alt);
    }
    fn set_rate_cmd(&self, azm: i32, alt: i32) {
        self.shared.inputs.lock().unwrap().rate_cmd = (azm, alt);
    }
    fn set_mount_mode(&self, mode: &str) -> PyResult<()> {
        self.shared.inputs.lock().unwrap().mount_mode = parse_mount_mode(mode)?;
        Ok(())
    }
    fn set_alignment(&self, az: f64, el: f64) {
        let mut i = self.shared.inputs.lock().unwrap();
        i.alignment_az = az;
        i.alignment_el = el;
    }
    #[pyo3(signature = (az_deg, el_deg, ff_az_dps = 0.0, ff_el_dps = 0.0))]
    fn set_setpoint(&self, az_deg: f64, el_deg: f64, ff_az_dps: f64, ff_el_dps: f64) {
        self.shared.inputs.lock().unwrap().setpoint = Some(Setpoint {
            az_deg,
            el_deg,
            ff_az_dps,
            ff_el_dps,
        });
    }
    fn clear_setpoint(&self) {
        self.shared.inputs.lock().unwrap().setpoint = None;
    }
    fn set_ff_enabled(&self, azm: bool, alt: bool) {
        let mut i = self.shared.inputs.lock().unwrap();
        i.ff_azm_enabled = azm;
        i.ff_alt_enabled = alt;
    }
    #[allow(clippy::too_many_arguments)]
    fn set_hotspot_params(
        &self,
        snr_threshold: f64,
        gate_radius: f64,
        coast_time_s: f64,
        x_sign: f64,
        y_sign: f64,
        pixel_size_um: f64,
        focal_length_mm: f64,
        rotation_deg: f64,
    ) {
        self.shared.inputs.lock().unwrap().hotspot = crate::controller::HotspotParams {
            snr_threshold,
            gate_radius,
            coast_time_s,
            x_sign,
            y_sign,
            pixel_size_um,
            focal_length_mm,
            rotation_deg,
        };
    }

    /// Publish a frame (2D float32 intensity map) for HOTSPOT mode.
    fn push_frame(&mut self, image: PyReadonlyArray2<'_, f32>) {
        let view = image.as_array();
        let (h, w) = (view.shape()[0], view.shape()[1]);
        let mut data = Vec::with_capacity(h * w);
        for i in 0..h {
            for j in 0..w {
                data.push(view[[i, j]]);
            }
        }
        self.frame_seq += 1;
        *self.shared.frame.lock().unwrap() = Some(Arc::new(Frame {
            data: Arc::new(data),
            h,
            w,
            seq: self.frame_seq,
        }));
    }

    fn submit_stop(&self) {
        self.shared.commands.lock().unwrap().push_back(Command::Stop);
    }
    fn submit_slew(&self, azm: i32, alt: i32) {
        self.shared
            .commands
            .lock()
            .unwrap()
            .push_back(Command::Slew { azm, alt });
    }
    fn submit_goto(&self, azm_deg: f64, alt_deg: f64) {
        self.shared
            .commands
            .lock()
            .unwrap()
            .push_back(Command::GotoMount { azm_deg, alt_deg });
    }

    /// Advance the simulated clock by `dt` seconds and run one control cycle.
    fn step(&mut self, dt: f64) {
        self.now += dt;
        self.mount.io.responder.advance_time(dt);
        run_cycle(&mut self.mount, &mut self.state, &self.shared, self.now);
    }

    /// True simulated pointing (for test inspection).
    #[getter]
    fn az_true_deg(&self) -> f64 {
        self.mount.io.responder.az_true_deg
    }
    #[getter]
    fn el_true_deg(&self) -> f64 {
        self.mount.io.responder.el_true_deg
    }

    /// Current output snapshot as a dict (the UI's read surface).
    fn snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let o = self.shared.outputs.lock().unwrap();
        let d = PyDict::new_bound(py);
        let _ = d.set_item("azm", o.azm);
        let _ = d.set_item("alt", o.alt);
        let _ = d.set_item("azm_raw", o.azm_raw);
        let _ = d.set_item("alt_raw", o.alt_raw);
        let _ = d.set_item("fresh", o.fresh);
        let _ = d.set_item("azm_error", o.azm_error);
        let _ = d.set_item("alt_error", o.alt_error);
        let _ = d.set_item("azm_rate_cmd", o.azm_rate_cmd);
        let _ = d.set_item("alt_rate_cmd", o.alt_rate_cmd);
        let _ = d.set_item("azm_pid_output", o.azm_pid_output);
        let _ = d.set_item("alt_pid_output", o.alt_pid_output);
        let _ = d.set_item("hotspot_acquired", o.hotspot_acquired);
        let _ = d.set_item("hotspot_status", o.hotspot_status.clone());
        let _ = d.set_item("hotspot_snr", o.hotspot_snr);
        let _ = d.set_item("hotspot_centroid", o.hotspot_centroid);
        let _ = d.set_item("requested_mode", o.requested_mode.map(mode_str));
        let _ = d.set_item("status_msgs", o.status_msgs.clone());
        d
    }

    /// Drain and return the accumulated status messages.
    fn drain_status(&self) -> Vec<String> {
        std::mem::take(&mut self.shared.outputs.lock().unwrap().status_msgs)
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
    m.add_function(wrap_pyfunction!(detect_hotspot, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_offset_to_angles, m)?)?;
    m.add_function(wrap_pyfunction!(cartesian_from_az_el, m)?)?;
    m.add_function(wrap_pyfunction!(az_el_from_cartesian, m)?)?;
    m.add_function(wrap_pyfunction!(AzAlt2AzEl, m)?)?;
    m.add_function(wrap_pyfunction!(apply_rotation_to_az_el, m)?)?;
    m.add_function(wrap_pyfunction!(AzAlt2AzEl_AltAz, m)?)?;
    m.add_function(wrap_pyfunction!(AzEl2AzAlt_AltAz, m)?)?;
    m.add_function(wrap_pyfunction!(AzAlt2AzEl_Passthrough, m)?)?;
    m.add_function(wrap_pyfunction!(AzEl2AzAlt_Passthrough, m)?)?;
    m.add_function(wrap_pyfunction!(telescope_to_local_elev_az, m)?)?;
    m.add_function(wrap_pyfunction!(local_elev_az_to_telescope, m)?)?;
    m.add_function(wrap_pyfunction!(altaz_local_to_radec, m)?)?;
    m.add_class::<SimMount>()?;
    m.add_class::<PidController>()?;
    m.add_class::<Detection>()?;
    m.add_class::<SimCoreLoop>()?;

    // Device target ids, so Python can pass them without re-deriving the map.
    m.add("ANY", protocol::targets::ANY)?;
    m.add("HC", protocol::targets::HC)?;
    m.add("AZM", protocol::targets::AZM)?;
    m.add("ALT", protocol::targets::ALT)?;
    m.add("FOCUS", protocol::targets::FOCUS)?;
    Ok(())
}
