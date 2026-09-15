//! The capture pump: a dedicated thread pulling frames from a source at
//! full rate into the ring, stamping exposure midpoints, feeding the
//! armed-capture recorder — the whole per-frame path with zero Python
//! involvement. Sources: the ASI SDK (hardware) or a push queue (sim /
//! tests / Python-rendered frames).

use crate::capture::CaptureRecorder;
use crate::ring::{exposure_midpoint, now_unix, Frame, Ring};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Where frames come from.
pub trait FrameSource: Send + 'static {
    /// Block until the next frame's pixel data is available. Returns
    /// (data, width, height, channels, capture_time_s) or None on shutdown.
    fn next_frame(&mut self) -> Option<(Vec<u8>, usize, usize, usize, f64)>;
}

/// A push-fed source: the sim (or a test) hands frames in; the pump
/// consumes them. Bounded to 4 pending so a stalled pump applies
/// backpressure instead of ballooning.
pub struct PushSource {
    shared: Arc<PushShared>,
}

pub struct PushShared {
    queue: Mutex<std::collections::VecDeque<(Vec<u8>, usize, usize, usize, f64)>>,
    cv: Condvar,
    closed: AtomicBool,
    dropped: AtomicU64,
}

impl PushSource {
    pub fn new() -> (PushSource, Arc<PushShared>) {
        let shared = Arc::new(PushShared {
            queue: Mutex::new(std::collections::VecDeque::new()),
            cv: Condvar::new(),
            closed: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
        });
        (
            PushSource {
                shared: shared.clone(),
            },
            shared,
        )
    }
}

impl PushShared {
    pub fn push(&self, data: Vec<u8>, w: usize, h: usize, c: usize, capture_time_s: f64) {
        let mut q = self.queue.lock().unwrap();
        if q.len() >= 4 {
            q.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        q.push_back((data, w, h, c, capture_time_s));
        self.cv.notify_one();
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.cv.notify_all();
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl FrameSource for PushSource {
    fn next_frame(&mut self) -> Option<(Vec<u8>, usize, usize, usize, f64)> {
        let mut q = self.shared.queue.lock().unwrap();
        loop {
            if let Some(item) = q.pop_front() {
                return Some(item);
            }
            if self.shared.closed.load(Ordering::SeqCst) {
                return None;
            }
            q = self.shared.cv.wait(q).unwrap();
        }
    }
}

pub struct Pump {
    pub ring: Arc<Ring>,
    pub recorder: Arc<CaptureRecorder>,
    frames_pumped: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Pump {
    /// Spawn the pump thread over any source.
    pub fn spawn(mut source: impl FrameSource, ring_capacity: usize) -> Pump {
        let ring = Arc::new(Ring::new(ring_capacity));
        let recorder = Arc::new(CaptureRecorder::new());
        let frames_pumped = Arc::new(AtomicU64::new(0));

        let ring2 = ring.clone();
        let rec2 = recorder.clone();
        let count2 = frames_pumped.clone();
        let handle = std::thread::Builder::new()
            .name("camera-pump".into())
            .spawn(move || {
                let mut seq = 0u64;
                while let Some((data, w, h, c, capture_time_s)) = source.next_frame() {
                    let now = now_unix();
                    let frame = Frame {
                        data: Arc::new(data),
                        width: w,
                        height: h,
                        channels: c,
                        utc_midpoint_s: exposure_midpoint(capture_time_s, now),
                        capture_time_s,
                        seq,
                    };
                    rec2.offer(&frame);
                    ring2.push(frame);
                    seq += 1;
                    count2.store(seq, Ordering::Relaxed);
                }
            })
            .expect("spawn camera pump");

        Pump {
            ring,
            recorder,
            frames_pumped,
            handle: Some(handle),
        }
    }

    pub fn frames_pumped(&self) -> u64 {
        self.frames_pumped.load(Ordering::Relaxed)
    }

    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
