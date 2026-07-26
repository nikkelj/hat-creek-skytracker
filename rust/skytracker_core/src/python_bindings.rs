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

use crate::controller::{Frame, HotspotParams, Inputs, LoopState, Mode, Setpoint};
use crate::core_loop::{run_cycle, Command, CoreLoop as ThreadedLoop, Shared};
use crate::hotspot as core_hotspot;
use crate::pid::PidController as CorePid;
use crate::protocol;
use crate::serial::SerialTransport;
use crate::sim::{LoopbackTransport, Mount, MountError, SimResponder, Transport};
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
        "altaz_side" | "altaz-side" | "altazside" => Ok(MountMode::AltAzSide),
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
    let arr = image.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    // One contiguous f32 copy (decouples this crate's ndarray version from the
    // numpy crate's and normalizes layout). Detection then works in f32.
    let buf: Vec<f32> = arr.iter().copied().collect();
    let view = ndarray::ArrayView2::from_shape((h, w), &buf).ok()?;

    let gate = match (gate_center, gate_radius) {
        (Some((gx, gy)), Some(r)) => Some((gx, gy, r)),
        _ => None,
    };

    core_hotspot::detect_hotspot(
        &view,
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
fn AzAlt2AzEl_AltAzSide(azm: f64, alt: f64, alignment_azimuth: f64) -> (f64, f64) {
    core_tf::az_alt_to_az_el_altaz_side(azm, alt, alignment_azimuth)
}

#[pyfunction]
fn AzEl2AzAlt_AltAzSide(az: f64, el: f64, alignment_azimuth: f64) -> (f64, f64) {
    core_tf::az_el_to_az_alt_altaz_side(az, el, alignment_azimuth)
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
    fn set_lead_time(&self, secs: f64) {
        self.shared.inputs.lock().unwrap().lead_time_sec = secs;
    }
    fn set_continuous_rate(&self, enabled: bool, max_dps: f64) {
        let mut i = self.shared.inputs.lock().unwrap();
        i.continuous_rate = enabled;
        i.guide_rate_max_dps = max_dps;
    }
    fn set_handoff_min_frames(&self, n: u32) {
        self.shared.inputs.lock().unwrap().handoff_min_frames = n;
    }
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (snr_threshold, gate_radius, coast_time_s, x_sign, y_sign, pixel_size_um, focal_length_mm, rotation_deg, max_rate_dps = 2.0, star_filter = false, rate_gate_dps = 0.15))]
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
        max_rate_dps: f64,
        star_filter: bool,
        rate_gate_dps: f64,
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
            max_rate_dps,
            star_filter,
            rate_gate_dps,
        };
    }

    /// Feedback low-pass time constant for both PIDs (seconds; 0 disables).
    fn set_output_filter(&self, tau_seconds: f64) {
        self.shared.inputs.lock().unwrap().output_filter_tau = tau_seconds.max(0.0);
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
        let _ = d.set_item("handoff_detection_count", o.handoff_detection_count);
        let _ = d.set_item("requested_mode", o.requested_mode.map(mode_str));
        let _ = d.set_item("status_msgs", o.status_msgs.clone());
        let _ = d.set_item("cycle_count", o.cycle_count);
        let _ = d.set_item("loop_dead", o.loop_dead);
        d
    }

    /// Drain and return the accumulated status messages.
    fn drain_status(&self) -> Vec<String> {
        std::mem::take(&mut self.shared.outputs.lock().unwrap().status_msgs)
    }
}

// --- shared input/output helpers (used by both SimCoreLoop and CoreLoop) ---

fn h_set_connected(sh: &Shared, v: bool) {
    sh.inputs.lock().unwrap().connected = v;
}
fn h_set_stopped(sh: &Shared, v: bool) {
    sh.inputs.lock().unwrap().stopped = v;
}
fn h_set_mode(sh: &Shared, s: &str) -> PyResult<()> {
    sh.inputs.lock().unwrap().mode = parse_mode(s)?;
    Ok(())
}
fn h_set_gains(sh: &Shared, ap: f64, ai: f64, ad: f64, lp: f64, li: f64, ld: f64) {
    let mut i = sh.inputs.lock().unwrap();
    i.azm_gains = (ap, ai, ad);
    i.alt_gains = (lp, li, ld);
}
fn h_set_limits(sh: &Shared, amin: f64, amax: f64, lmin: f64, lmax: f64) {
    let mut i = sh.inputs.lock().unwrap();
    i.azm_limit = (amin, amax);
    i.alt_limit = (lmin, lmax);
}
fn h_set_offsets(sh: &Shared, azm: f64, alt: f64) {
    sh.inputs.lock().unwrap().offsets = (azm, alt);
}
fn h_set_rate_cmd(sh: &Shared, azm: i32, alt: i32) {
    sh.inputs.lock().unwrap().rate_cmd = (azm, alt);
}
fn h_set_mount_mode(sh: &Shared, s: &str) -> PyResult<()> {
    sh.inputs.lock().unwrap().mount_mode = parse_mount_mode(s)?;
    Ok(())
}
fn h_set_alignment(sh: &Shared, az: f64, el: f64) {
    let mut i = sh.inputs.lock().unwrap();
    i.alignment_az = az;
    i.alignment_el = el;
}
fn h_set_setpoint(sh: &Shared, az: f64, el: f64, ff_az: f64, ff_el: f64) {
    sh.inputs.lock().unwrap().setpoint = Some(Setpoint {
        az_deg: az,
        el_deg: el,
        ff_az_dps: ff_az,
        ff_el_dps: ff_el,
    });
}
fn h_clear_setpoint(sh: &Shared) {
    sh.inputs.lock().unwrap().setpoint = None;
}
fn h_set_ff_enabled(sh: &Shared, azm: bool, alt: bool) {
    let mut i = sh.inputs.lock().unwrap();
    i.ff_azm_enabled = azm;
    i.ff_alt_enabled = alt;
}
fn h_set_lead_time(sh: &Shared, secs: f64) {
    sh.inputs.lock().unwrap().lead_time_sec = secs;
}
fn h_set_continuous_rate(sh: &Shared, enabled: bool, max_dps: f64) {
    let mut i = sh.inputs.lock().unwrap();
    i.continuous_rate = enabled;
    i.guide_rate_max_dps = max_dps;
}
fn h_set_handoff_min_frames(sh: &Shared, n: u32) {
    sh.inputs.lock().unwrap().handoff_min_frames = n;
}
#[allow(clippy::too_many_arguments)]
fn h_set_hotspot_params(
    sh: &Shared,
    snr: f64,
    gate: f64,
    coast: f64,
    xs: f64,
    ys: f64,
    px: f64,
    fl: f64,
    rot: f64,
    max_rate_dps: f64,
    star_filter: bool,
    rate_gate_dps: f64,
) {
    sh.inputs.lock().unwrap().hotspot = HotspotParams {
        snr_threshold: snr,
        gate_radius: gate,
        coast_time_s: coast,
        max_rate_dps,
        x_sign: xs,
        y_sign: ys,
        pixel_size_um: px,
        focal_length_mm: fl,
        rotation_deg: rot,
        star_filter,
        rate_gate_dps,
    };
}
fn h_set_output_filter(sh: &Shared, tau_seconds: f64) {
    sh.inputs.lock().unwrap().output_filter_tau = tau_seconds.max(0.0);
}
fn h_push_frame(sh: &Shared, seq: u64, image: PyReadonlyArray2<'_, f32>) {
    let view = image.as_array();
    let (h, w) = (view.shape()[0], view.shape()[1]);
    let mut data = Vec::with_capacity(h * w);
    for i in 0..h {
        for j in 0..w {
            data.push(view[[i, j]]);
        }
    }
    *sh.frame.lock().unwrap() = Some(Arc::new(Frame {
        data: Arc::new(data),
        h,
        w,
        seq,
    }));
}
fn h_submit(sh: &Shared, cmd: Command) {
    sh.commands.lock().unwrap().push_back(cmd);
}
fn h_drain_status(sh: &Shared) -> Vec<String> {
    std::mem::take(&mut sh.outputs.lock().unwrap().status_msgs)
}
fn h_snapshot<'py>(py: Python<'py>, sh: &Shared) -> Bound<'py, PyDict> {
    let o = sh.outputs.lock().unwrap();
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
    let _ = d.set_item("handoff_detection_count", o.handoff_detection_count);
    let _ = d.set_item("requested_mode", o.requested_mode.map(mode_str));
    let _ = d.set_item("status_msgs", o.status_msgs.clone());
    let _ = d.set_item("actual_hz", o.actual_hz);
    let _ = d.set_item("cycle_ms", o.cycle_ms);
    let _ = d.set_item("cycle_count", o.cycle_count);
    let _ = d.set_item("loop_dead", o.loop_dead);
    d
}

/// Transport that bridges to a Python mount object (e.g. simulator.SimMount):
/// decodes each 8-byte 'P' request and calls the object's `hc_*` methods,
/// encoding the reply. Lets the Rust loop drive the existing HW simulator for
/// in-UI validation. Acquires the GIL per call (acceptable in sim mode); the
/// real hardware path uses `SerialTransport` and never touches Python.
struct PyMountTransport {
    mount: Py<PyAny>,
    azm: Py<PyAny>,
    alt: Py<PyAny>,
    focus: Py<PyAny>,
    pending: std::collections::VecDeque<u8>,
}

impl PyMountTransport {
    fn new(py: Python<'_>, mount: Py<PyAny>) -> PyResult<Self> {
        let targets = py.import_bound("lib.auxstar")?.getattr("Targets")?;
        Ok(PyMountTransport {
            mount,
            azm: targets.getattr("AZM")?.unbind(),
            alt: targets.getattr("ALT")?.unbind(),
            focus: targets.getattr("FOCUS")?.unbind(),
            pending: std::collections::VecDeque::new(),
        })
    }
}

impl Transport for PyMountTransport {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        if data.len() < 8 || data[0] != 0x50 {
            return Ok(());
        }
        let dest = data[2];
        let msg_id = data[3];
        let d = [data[4], data[5], data[6]];
        let resp_len = data[7] as usize;

        let resp: Vec<u8> = Python::with_gil(|py| {
            let mount = self.mount.bind(py);
            let target = match dest {
                0x11 => self.alt.bind(py),
                0x12 => self.focus.bind(py),
                _ => self.azm.bind(py),
            };
            if msg_id == protocol::MC_GET_POSITION.0 {
                let frac: f64 = mount
                    .call_method1("hc_get_position", (target,))
                    .and_then(|r| r.extract())
                    .unwrap_or(0.0);
                let mut v = protocol::pack_int3(frac).to_vec();
                v.push(b'#');
                v
            } else if msg_id == protocol::MC_MOVE_POS.0 || msg_id == protocol::MC_MOVE_NEG.0 {
                let sign = if msg_id == protocol::MC_MOVE_POS.0 { 1 } else { -1 };
                let rate = sign * (d[0] as i32);
                let _ = mount.call_method1("hc_slew_fixed", (target, rate));
                vec![b'#']
            } else if msg_id == protocol::MC_SET_POS_GUIDERATE.0
                || msg_id == protocol::MC_SET_NEG_GUIDERATE.0
            {
                let sign = if msg_id == protocol::MC_SET_POS_GUIDERATE.0 { 1.0 } else { -1.0 };
                let dps = sign * protocol::unpack_int3(&d) * 360.0;
                let _ = mount.call_method1("hc_set_rate_dps", (target, dps));
                vec![b'#']
            } else if msg_id == protocol::MC_GOTO_FAST.0 {
                let deg = protocol::unpack_int3(&d) * 360.0;
                let _ = mount.call_method1("hc_goto_fast", (target, deg, 0.0_f64, 0.0_f64));
                vec![b'#']
            } else {
                let mut v = vec![0u8; resp_len];
                v.push(b'#');
                v
            }
        });
        self.pending.extend(resp);
        Ok(())
    }

    fn read(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.pending.pop_front() {
                Some(b) => out.push(b),
                None => break,
            }
        }
        out
    }
}

/// Threaded control loop for real use: owns a background thread driving either a
/// serial port (`open_serial`) or a Python mount object (`wrap_mount`, for the
/// HW simulator). Same push/snapshot surface as `SimCoreLoop`.
#[pyclass]
struct CoreLoop {
    inner: ThreadedLoop,
    frame_seq: u64,
}

#[pymethods]
impl CoreLoop {
    #[staticmethod]
    #[pyo3(signature = (port, target_hz = 15.0))]
    fn open_serial(port: &str, target_hz: f64) -> PyResult<Self> {
        let st = SerialTransport::open(port, 9600, 250)
            .map_err(|e| PyIOError::new_err(format!("open {port}: {e}")))?;
        let boxed: Box<dyn Transport + Send> = Box::new(st);
        let mut inputs = Inputs::default();
        inputs.connected = true;
        let shared = Shared::new(inputs);
        let inner = ThreadedLoop::spawn(Mount::new(boxed), shared, target_hz);
        Ok(CoreLoop { inner, frame_seq: 0 })
    }

    #[staticmethod]
    #[pyo3(signature = (mount, target_hz = 15.0))]
    fn wrap_mount(py: Python<'_>, mount: Py<PyAny>, target_hz: f64) -> PyResult<Self> {
        let pt = PyMountTransport::new(py, mount)?;
        let boxed: Box<dyn Transport + Send> = Box::new(pt);
        let mut inputs = Inputs::default();
        inputs.connected = true;
        let shared = Shared::new(inputs);
        let inner = ThreadedLoop::spawn(Mount::new(boxed), shared, target_hz);
        Ok(CoreLoop { inner, frame_seq: 0 })
    }

    fn set_connected(&self, v: bool) {
        h_set_connected(self.inner.shared(), v);
    }
    fn set_stopped(&self, v: bool) {
        h_set_stopped(self.inner.shared(), v);
    }
    fn set_mode(&self, mode: &str) -> PyResult<()> {
        h_set_mode(self.inner.shared(), mode)
    }
    fn set_gains(&self, ap: f64, ai: f64, ad: f64, lp: f64, li: f64, ld: f64) {
        h_set_gains(self.inner.shared(), ap, ai, ad, lp, li, ld);
    }
    fn set_limits(&self, amin: f64, amax: f64, lmin: f64, lmax: f64) {
        h_set_limits(self.inner.shared(), amin, amax, lmin, lmax);
    }
    fn set_offsets(&self, azm: f64, alt: f64) {
        h_set_offsets(self.inner.shared(), azm, alt);
    }
    fn set_rate_cmd(&self, azm: i32, alt: i32) {
        h_set_rate_cmd(self.inner.shared(), azm, alt);
    }
    fn set_mount_mode(&self, mode: &str) -> PyResult<()> {
        h_set_mount_mode(self.inner.shared(), mode)
    }
    fn set_alignment(&self, az: f64, el: f64) {
        h_set_alignment(self.inner.shared(), az, el);
    }
    #[pyo3(signature = (az_deg, el_deg, ff_az_dps = 0.0, ff_el_dps = 0.0))]
    fn set_setpoint(&self, az_deg: f64, el_deg: f64, ff_az_dps: f64, ff_el_dps: f64) {
        h_set_setpoint(self.inner.shared(), az_deg, el_deg, ff_az_dps, ff_el_dps);
    }
    fn clear_setpoint(&self) {
        h_clear_setpoint(self.inner.shared());
    }
    fn set_ff_enabled(&self, azm: bool, alt: bool) {
        h_set_ff_enabled(self.inner.shared(), azm, alt);
    }
    fn set_lead_time(&self, secs: f64) {
        h_set_lead_time(self.inner.shared(), secs);
    }
    fn set_continuous_rate(&self, enabled: bool, max_dps: f64) {
        h_set_continuous_rate(self.inner.shared(), enabled, max_dps);
    }
    fn set_handoff_min_frames(&self, n: u32) {
        h_set_handoff_min_frames(self.inner.shared(), n);
    }
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (snr_threshold, gate_radius, coast_time_s, x_sign, y_sign, pixel_size_um, focal_length_mm, rotation_deg, max_rate_dps = 2.0, star_filter = false, rate_gate_dps = 0.15))]
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
        max_rate_dps: f64,
        star_filter: bool,
        rate_gate_dps: f64,
    ) {
        h_set_hotspot_params(
            self.inner.shared(),
            snr_threshold,
            gate_radius,
            coast_time_s,
            x_sign,
            y_sign,
            pixel_size_um,
            focal_length_mm,
            rotation_deg,
            max_rate_dps,
            star_filter,
            rate_gate_dps,
        );
    }

    /// Feedback low-pass time constant for both PIDs (seconds; 0 disables).
    fn set_output_filter(&self, tau_seconds: f64) {
        h_set_output_filter(self.inner.shared(), tau_seconds);
    }
    fn push_frame(&mut self, image: PyReadonlyArray2<'_, f32>) {
        self.frame_seq += 1;
        h_push_frame(self.inner.shared(), self.frame_seq, image);
    }
    fn submit_stop(&self) {
        h_submit(self.inner.shared(), Command::Stop);
    }
    fn submit_slew(&self, azm: i32, alt: i32) {
        h_submit(self.inner.shared(), Command::Slew { azm, alt });
    }
    fn submit_goto(&self, azm_deg: f64, alt_deg: f64) {
        h_submit(self.inner.shared(), Command::GotoMount { azm_deg, alt_deg });
    }
    fn snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        h_snapshot(py, self.inner.shared())
    }
    fn drain_status(&self) -> Vec<String> {
        h_drain_status(self.inner.shared())
    }
    /// Stop the loop thread. Releases the GIL during the join so a Python-mount
    /// bridge loop (which needs the GIL each cycle) can't deadlock.
    fn stop(&mut self, py: Python<'_>) {
        py.allow_threads(|| self.inner.stop());
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
    m.add_function(wrap_pyfunction!(AzAlt2AzEl_AltAzSide, m)?)?;
    m.add_function(wrap_pyfunction!(AzEl2AzAlt_AltAzSide, m)?)?;
    m.add_function(wrap_pyfunction!(AzAlt2AzEl_Passthrough, m)?)?;
    m.add_function(wrap_pyfunction!(AzEl2AzAlt_Passthrough, m)?)?;
    m.add_function(wrap_pyfunction!(telescope_to_local_elev_az, m)?)?;
    m.add_function(wrap_pyfunction!(local_elev_az_to_telescope, m)?)?;
    m.add_function(wrap_pyfunction!(altaz_local_to_radec, m)?)?;
    m.add_class::<SimMount>()?;
    m.add_class::<PidController>()?;
    m.add_class::<Detection>()?;
    m.add_class::<SimCoreLoop>()?;
    m.add_class::<CoreLoop>()?;

    // Device target ids, so Python can pass them without re-deriving the map.
    m.add("ANY", protocol::targets::ANY)?;
    m.add("HC", protocol::targets::HC)?;
    m.add("AZM", protocol::targets::AZM)?;
    m.add("ALT", protocol::targets::ALT)?;
    m.add("FOCUS", protocol::targets::FOCUS)?;
    Ok(())
}
