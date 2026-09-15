//! Frame ring buffer with exposure-midpoint timestamps — the Rust
//! replacement for camera_buffer.CircularBuffer + the stamping logic in
//! CameraThread. Designed for a single producer (the capture pump) and
//! cheap concurrent readers (display pull, capture flush): frames are
//! Arc'd so a reader never blocks the pump and nothing is copied until
//! it crosses to Python.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct Frame {
    /// Mono8 or RGB24 pixel data (row-major, `channels` planes interleaved).
    pub data: Arc<Vec<u8>>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// UTC unix seconds, back-dated to the exposure midpoint.
    pub utc_midpoint_s: f64,
    /// Wall-clock capture duration (exposure + readout + convert), seconds.
    pub capture_time_s: f64,
    /// Monotonic frame counter from the pump.
    pub seq: u64,
}

pub fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// camera_buffer.exposure_midpoint_utc: stamp now minus half the measured
/// capture time (lands on the exposure midpoint to within readout latency).
pub fn exposure_midpoint(capture_time_s: f64, now_s: f64) -> f64 {
    now_s - capture_time_s.max(0.0) / 2.0
}

pub struct Ring {
    inner: Mutex<VecDeque<Frame>>,
    capacity: usize,
}

impl Ring {
    pub fn new(capacity: usize) -> Self {
        Ring {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn push(&self, frame: Frame) {
        let mut q = self.inner.lock().unwrap();
        if q.len() == self.capacity {
            q.pop_front();
        }
        q.push_back(frame);
    }

    pub fn latest(&self) -> Option<Frame> {
        self.inner.lock().unwrap().back().cloned()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The newest `n` frames, oldest first (for capture flushes).
    pub fn last_n(&self, n: usize) -> Vec<Frame> {
        let q = self.inner.lock().unwrap();
        let start = q.len().saturating_sub(n);
        q.iter().skip(start).cloned().collect()
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }
}
