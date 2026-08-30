//! ZWO ASI SDK binding (ASICamera2.dll) — the used surface only, loaded
//! at runtime with libloading so builds/tests need neither the DLL nor
//! hardware. Signatures follow the ASI Camera SDK v1.x C API (stable ABI,
//! same DLL the repo ships in lib/).
//!
//! Timing semantics (exposure-midpoint accuracy vs the SDK's internal
//! buffering) are hardware truth: validate on the rig before promotion.

#![allow(non_snake_case, dead_code)]

use libloading::{Library, Symbol};
use std::ffi::c_int;

pub const ASI_IMG_RAW8: c_int = 0;
pub const ASI_IMG_RGB24: c_int = 1;
pub const ASI_IMG_RAW16: c_int = 2;
pub const ASI_IMG_Y8: c_int = 3;

pub const ASI_GAIN: c_int = 0;
pub const ASI_EXPOSURE: c_int = 1;
pub const ASI_BANDWIDTHOVERLOAD: c_int = 6;
pub const ASI_HIGH_SPEED_MODE: c_int = 14;

#[repr(C)]
#[derive(Clone)]
pub struct AsiCameraInfo {
    pub name: [u8; 64],
    pub camera_id: c_int,
    pub max_height: i64,
    pub max_width: i64,
    pub is_color_cam: c_int,
    pub bayer_pattern: c_int,
    pub supported_bins: [c_int; 16],
    pub supported_video_format: [c_int; 8],
    pub pixel_size: f64,
    pub mechanical_shutter: c_int,
    pub st4_port: c_int,
    pub is_cooler_cam: c_int,
    pub is_usb3_host: c_int,
    pub is_usb3_camera: c_int,
    pub elec_per_adu: f32,
    pub bit_depth: c_int,
    pub is_trigger_cam: c_int,
    pub unused: [u8; 16],
}

pub struct AsiSdk {
    lib: Library,
}

#[derive(Debug)]
pub struct AsiError(pub String);

impl std::fmt::Display for AsiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ASI SDK: {}", self.0)
    }
}

impl std::error::Error for AsiError {}

fn check(code: c_int, what: &str) -> Result<(), AsiError> {
    if code == 0 {
        Ok(())
    } else {
        Err(AsiError(format!("{what} failed with code {code}")))
    }
}

impl AsiSdk {
    /// Load ASICamera2.dll from the given path (e.g. lib/ASICamera2.dll).
    pub fn load(dll_path: &str) -> Result<Self, AsiError> {
        let lib = unsafe { Library::new(dll_path) }
            .map_err(|e| AsiError(format!("load {dll_path}: {e}")))?;
        Ok(AsiSdk { lib })
    }

    unsafe fn sym<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, AsiError> {
        unsafe {
            self.lib
                .get(name)
                .map_err(|e| AsiError(format!("{}: {e}", String::from_utf8_lossy(name))))
        }
    }

    pub fn num_cameras(&self) -> Result<i32, AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn() -> c_int> =
                self.sym(b"ASIGetNumOfConnectedCameras")?;
            Ok(f())
        }
    }

    pub fn camera_info(&self, index: i32) -> Result<AsiCameraInfo, AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut AsiCameraInfo, c_int) -> c_int> =
                self.sym(b"ASIGetCameraProperty")?;
            let mut info: AsiCameraInfo = std::mem::zeroed();
            check(f(&mut info, index), "ASIGetCameraProperty")?;
            Ok(info)
        }
    }

    pub fn open(&self, camera_id: i32) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int) -> c_int> = self.sym(b"ASIOpenCamera")?;
            check(f(camera_id), "ASIOpenCamera")?;
            let g: Symbol<unsafe extern "C" fn(c_int) -> c_int> = self.sym(b"ASIInitCamera")?;
            check(g(camera_id), "ASIInitCamera")
        }
    }

    pub fn close(&self, camera_id: i32) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int) -> c_int> = self.sym(b"ASICloseCamera")?;
            check(f(camera_id), "ASICloseCamera")
        }
    }

    pub fn set_roi(
        &self,
        camera_id: i32,
        width: i32,
        height: i32,
        bin: i32,
        img_type: c_int,
    ) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int) -> c_int> =
                self.sym(b"ASISetROIFormat")?;
            check(f(camera_id, width, height, bin, img_type), "ASISetROIFormat")
        }
    }

    pub fn set_control(&self, camera_id: i32, control: c_int, value: i64, auto: bool) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int, c_int, i64, c_int) -> c_int> =
                self.sym(b"ASISetControlValue")?;
            check(f(camera_id, control, value, auto as c_int), "ASISetControlValue")
        }
    }

    pub fn start_video(&self, camera_id: i32) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int) -> c_int> =
                self.sym(b"ASIStartVideoCapture")?;
            check(f(camera_id), "ASIStartVideoCapture")
        }
    }

    pub fn stop_video(&self, camera_id: i32) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int) -> c_int> =
                self.sym(b"ASIStopVideoCapture")?;
            check(f(camera_id), "ASIStopVideoCapture")
        }
    }

    /// Blocking frame fetch (the "ultra-fast" direct path the Python side
    /// used via ctypes): fills `buf`, waits up to `wait_ms`.
    pub fn get_video_data(&self, camera_id: i32, buf: &mut [u8], wait_ms: i32) -> Result<(), AsiError> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(c_int, *mut u8, i64, c_int) -> c_int> =
                self.sym(b"ASIGetVideoData")?;
            check(
                f(camera_id, buf.as_mut_ptr(), buf.len() as i64, wait_ms),
                "ASIGetVideoData",
            )
        }
    }
}

/// Hardware frame source: pulls ASIGetVideoData in a tight loop.
/// Compiles and ships now; timing truth (exposure-midpoint accuracy vs
/// SDK-internal buffering) is validated on the rig before promotion.
pub struct AsiSource {
    pub sdk: AsiSdk,
    pub camera_id: i32,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub wait_ms: i32,
    pub stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl crate::pump::FrameSource for AsiSource {
    fn next_frame(&mut self) -> Option<(Vec<u8>, usize, usize, usize, f64)> {
        if self.stopped.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = self.sdk.stop_video(self.camera_id);
            let _ = self.sdk.close(self.camera_id);
            return None;
        }
        let mut buf = vec![0u8; self.width * self.height * self.channels];
        let t0 = std::time::Instant::now();
        match self.sdk.get_video_data(self.camera_id, &mut buf, self.wait_ms) {
            Ok(()) => Some((
                buf,
                self.width,
                self.height,
                self.channels,
                t0.elapsed().as_secs_f64(),
            )),
            Err(_) => {
                // Dropped/timed-out frame: keep pumping (matches the
                // Python thread's tolerate-and-continue behavior).
                std::thread::sleep(std::time::Duration::from_millis(2));
                self.next_frame()
            }
        }
    }
}
