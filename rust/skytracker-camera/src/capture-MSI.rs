//! Armed-capture recording + parallel BMP dump (ports capture_manager's
//! CaptureDumpThread/worker-pool semantics, minus the GIL): while armed,
//! every pumped frame is retained; on disarm the frames are written as
//! BMPs by rayon workers alongside a trajectory.csv the Python side
//! composes (frame timestamps are returned to the caller for that).

use crate::ring::Frame;
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;

/// Minimal 24-bit BGR bottom-up BMP writer (what capture dumps use).
pub fn write_bmp(path: &Path, frame: &Frame) -> std::io::Result<()> {
    let (w, h, c) = (frame.width, frame.height, frame.channels);
    let row_bytes = w * 3;
    let row_padded = (row_bytes + 3) & !3;
    let data_size = row_padded * h;
    let file_size = 54 + data_size;

    let mut out = Vec::with_capacity(file_size);
    // BITMAPFILEHEADER
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    // BITMAPINFOHEADER
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 24]); // compression..colors, all zero

    let pad = [0u8; 3];
    for y in (0..h).rev() {
        for x in 0..w {
            let px = match c {
                1 => {
                    let v = frame.data[y * w + x];
                    [v, v, v]
                }
                _ => {
                    let i = (y * w + x) * c;
                    // stored RGB -> BMP wants BGR
                    [frame.data[i + 2], frame.data[i + 1], frame.data[i]]
                }
            };
            out.extend_from_slice(&px);
        }
        out.extend_from_slice(&pad[..row_padded - row_bytes]);
    }
    std::fs::File::create(path)?.write_all(&out)
}

pub struct CaptureRecorder {
    frames: std::sync::Mutex<Vec<Frame>>,
    armed: std::sync::atomic::AtomicBool,
}

impl Default for CaptureRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureRecorder {
    pub fn new() -> Self {
        CaptureRecorder {
            frames: std::sync::Mutex::new(Vec::new()),
            armed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn arm(&self) {
        self.frames.lock().unwrap().clear();
        self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_armed(&self) -> bool {
        self.armed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Called by the pump for every frame while armed (cheap: Arc clone).
    pub fn offer(&self, frame: &Frame) {
        if self.is_armed() {
            self.frames.lock().unwrap().push(frame.clone());
        }
    }

    /// Disarm and dump all recorded frames as frame_NNNNN.bmp into `dir`
    /// with rayon workers. Returns (count, utc timestamps) for the
    /// caller's trajectory.csv.
    pub fn disarm_and_dump(&self, dir: &Path) -> std::io::Result<(usize, Vec<f64>)> {
        self.armed.store(false, std::sync::atomic::Ordering::SeqCst);
        let frames: Vec<Frame> = std::mem::take(&mut *self.frames.lock().unwrap());
        std::fs::create_dir_all(dir)?;
        let results: Vec<std::io::Result<()>> = frames
            .par_iter()
            .enumerate()
            .map(|(i, f)| write_bmp(&dir.join(format!("frame_{i:05}.bmp")), f))
            .collect();
        for r in results {
            r?;
        }
        Ok((frames.len(), frames.iter().map(|f| f.utc_midpoint_s).collect()))
    }
}
