//! PyO3 bindings for skytracker-camera (Phase 4 seam): a camera pipeline
//! Python can feed (sim frames) or attach to ASI hardware, with the whole
//! per-frame path (stamping, ring, capture retention, BMP dump) in Rust.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use skytracker_camera::asi::{AsiSdk, AsiSource, ASI_IMG_RAW8};
use skytracker_camera::pump::{Pump, PushShared, PushSource};

#[pyclass]
pub struct CameraPipeline {
    pump: Option<Pump>,
    push: Option<Arc<PushShared>>,
    asi_stop: Option<Arc<AtomicBool>>,
}

#[pymethods]
impl CameraPipeline {
    /// A push-fed pipeline (the hardware simulator or tests drive it).
    #[staticmethod]
    #[pyo3(signature = (ring_capacity = 1000))]
    fn push_source(ring_capacity: usize) -> CameraPipeline {
        let (source, shared) = PushSource::new();
        CameraPipeline {
            pump: Some(Pump::spawn(source, ring_capacity)),
            push: Some(shared),
            asi_stop: None,
        }
    }

    /// An ASI-hardware pipeline (rig only; timing truth validated there).
    #[staticmethod]
    #[pyo3(signature = (dll_path, camera_index, width, height,
                        exposure_us, gain, ring_capacity = 1000))]
    fn open_asi(
        dll_path: String,
        camera_index: i32,
        width: usize,
        height: usize,
        exposure_us: i64,
        gain: i64,
        ring_capacity: usize,
    ) -> PyResult<CameraPipeline> {
        let sdk = AsiSdk::load(&dll_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let info = sdk
            .camera_info(camera_index)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let id = info.camera_id;
        let setup = || -> Result<(), skytracker_camera::asi::AsiError> {
            sdk.open(id)?;
            sdk.set_roi(id, width as i32, height as i32, 1, ASI_IMG_RAW8)?;
            sdk.set_control(id, skytracker_camera::asi::ASI_EXPOSURE, exposure_us, false)?;
            sdk.set_control(id, skytracker_camera::asi::ASI_GAIN, gain, false)?;
            sdk.start_video(id)
        };
        setup().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let stopped = Arc::new(AtomicBool::new(false));
        let source = AsiSource {
            sdk,
            camera_id: id,
            width,
            height,
            channels: 1,
            wait_ms: 1000,
            stopped: stopped.clone(),
        };
        Ok(CameraPipeline {
            pump: Some(Pump::spawn(source, ring_capacity)),
            push: None,
            asi_stop: Some(stopped),
        })
    }

    /// Feed one mono frame (push pipelines only).
    fn push_frame_mono(
        &self,
        frame: PyReadonlyArray2<'_, u8>,
        capture_time_s: f64,
    ) -> PyResult<()> {
        let push = self
            .push
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("not a push pipeline"))?;
        let arr = frame.as_array();
        let (h, w) = (arr.shape()[0], arr.shape()[1]);
        push.push(arr.iter().copied().collect(), w, h, 1, capture_time_s);
        Ok(())
    }

    /// Feed one RGB frame (push pipelines only).
    fn push_frame_rgb(
        &self,
        frame: PyReadonlyArray3<'_, u8>,
        capture_time_s: f64,
    ) -> PyResult<()> {
        let push = self
            .push
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("not a push pipeline"))?;
        let arr = frame.as_array();
        let (h, w, c) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
        push.push(arr.iter().copied().collect(), w, h, c, capture_time_s);
        Ok(())
    }

    /// Latest frame as (mono 2-D ndarray, utc_midpoint_s, seq), or None.
    /// (Display pull; RGB frames return their first channel here — the
    /// tracker consumes mono.)
    fn latest_frame<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(Bound<'py, PyArray2<u8>>, f64, u64)> {
        let pump = self.pump.as_ref()?;
        let f = pump.ring.latest()?;
        let mut mono = vec![0u8; f.width * f.height];
        if f.channels == 1 {
            mono.copy_from_slice(&f.data);
        } else {
            for i in 0..f.width * f.height {
                mono[i] = f.data[i * f.channels];
            }
        }
        let nd = Array2::from_shape_vec((f.height, f.width), mono).unwrap();
        Some((
            PyArray2::from_owned_array_bound(py, nd),
            f.utc_midpoint_s,
            f.seq,
        ))
    }

    fn frames_pumped(&self) -> u64 {
        self.pump.as_ref().map_or(0, |p| p.frames_pumped())
    }

    fn ring_len(&self) -> usize {
        self.pump.as_ref().map_or(0, |p| p.ring.len())
    }

    fn frames_dropped(&self) -> u64 {
        self.push.as_ref().map_or(0, |s| s.dropped())
    }

    fn arm_capture(&self) {
        if let Some(p) = &self.pump {
            p.recorder.arm();
        }
    }

    /// Disarm + dump BMPs into `dir` (rayon-parallel, off the GIL).
    /// Returns (count, per-frame utc midpoints).
    fn disarm_and_dump(&self, py: Python<'_>, dir: String) -> PyResult<(usize, Vec<f64>)> {
        let pump = self
            .pump
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("pipeline closed"))?;
        let recorder = pump.recorder.clone();
        py.allow_threads(move || recorder.disarm_and_dump(std::path::Path::new(&dir)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn close(&mut self, py: Python<'_>) {
        if let Some(s) = &self.push {
            s.close();
        }
        if let Some(stop) = &self.asi_stop {
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if let Some(pump) = self.pump.take() {
            py.allow_threads(move || pump.join());
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CameraPipeline>()?;
    m.add("CAMERA_AVAILABLE", true)?;
    Ok(())
}
