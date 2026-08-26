//! Armed-capture recording + parallel BMP dump (ports capture_manager's
//! CaptureDumpThread/worker-pool semantics, minus the GIL): while armed,
//! every pumped frame is retained; on disarm the frames are written as
//! BMPs by rayon workers alongside a trajectory.csv the Python side
//! composes (frame timestamps are returned to the caller for that).

use crate::ring::Frame;
use std::io::Write;
use std::path::Path;

/// Minimal bottom-up BMP writer (what capture dumps use). Mono frames are
/// written as 8-bit palette BMPs (3× smaller and 3× faster than inflating
/// to 24-bit — a 5-minute capture is disk-space-bound); RGB frames stay
/// 24-bit BGR.
pub fn write_bmp(path: &Path, frame: &Frame) -> std::io::Result<()> {
    let (w, h, c) = (frame.width, frame.height, frame.channels);
    if c == 1 {
        return write_bmp_gray8(path, frame);
    }
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

/// 8-bit grayscale palette BMP (BI_RGB, 256-entry linear palette).
fn write_bmp_gray8(path: &Path, frame: &Frame) -> std::io::Result<()> {
    let (w, h) = (frame.width, frame.height);
    let row_padded = (w + 3) & !3;
    let data_size = row_padded * h;
    let header_size = 54 + 256 * 4;
    let file_size = header_size + data_size;

    let mut out = Vec::with_capacity(file_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(header_size as u32).to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&8u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    out.extend_from_slice(&(data_size as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 8]); // ppm x/y
    out.extend_from_slice(&256u32.to_le_bytes()); // palette entries
    out.extend_from_slice(&0u32.to_le_bytes());
    for v in 0u16..256 {
        out.extend_from_slice(&[v as u8, v as u8, v as u8, 0]);
    }
    let pad = [0u8; 3];
    for y in (0..h).rev() {
        out.extend_from_slice(&frame.data[y * w..y * w + w]);
        out.extend_from_slice(&pad[..row_padded - w]);
    }
    std::fs::File::create(path)?.write_all(&out)
}

/// Streaming spool recorder. While armed, every offered frame goes through a
/// bounded queue to a writer thread that streams `frame_NNNNN.bmp` into the
/// spool directory AS IT ARRIVES — so a capture is disk-bound, not RAM-bound,
/// and can run for many minutes uninterrupted. RAM holds at most `capacity`
/// frames (config `buffer_size`); if the disk falls behind, the NEWEST
/// incoming frame is skipped (capture continues at whatever rate IO
/// sustains) — already-buffered frames are never ring-dropped and memory
/// never grows past the cap.
pub struct CaptureRecorder {
    armed: std::sync::atomic::AtomicBool,
    dropped: std::sync::atomic::AtomicUsize,
    written: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    inner: std::sync::Mutex<Option<Spool>>,
}

struct Spool {
    dir: std::path::PathBuf,
    /// Present while armed; dropping it closes the channel so the writer
    /// drains the queue and exits.
    tx: Option<std::sync::mpsc::SyncSender<Frame>>,
    writer: Option<std::thread::JoinHandle<std::io::Result<Vec<f64>>>>,
}

impl Default for CaptureRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureRecorder {
    pub fn new() -> Self {
        CaptureRecorder {
            armed: std::sync::atomic::AtomicBool::new(false),
            dropped: std::sync::atomic::AtomicUsize::new(0),
            written: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inner: std::sync::Mutex::new(None),
        }
    }

    /// Arm into `dir`: creates it and starts the writer thread. `capacity`
    /// bounds the in-RAM queue in frames.
    pub fn arm_spool(&self, dir: &Path, capacity: usize) -> std::io::Result<()> {
        use std::sync::atomic::Ordering;
        std::fs::create_dir_all(dir)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(capacity.max(2));
        let wdir = dir.to_path_buf();
        let written = self.written.clone();
        written.store(0, Ordering::SeqCst);
        self.dropped.store(0, Ordering::SeqCst);
        let writer = std::thread::Builder::new().name("capture-spool".into()).spawn(move || {
            let mut times = Vec::new();
            let mut i = 0usize;
            while let Ok(f) = rx.recv() {
                write_bmp(&wdir.join(format!("frame_{i:05}.bmp")), &f)?;
                times.push(f.utc_midpoint_s);
                written.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
            Ok(times)
        })?;
        *self.inner.lock().unwrap() = Some(Spool { dir: dir.to_path_buf(), tx: Some(tx), writer: Some(writer) });
        self.armed.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn is_armed(&self) -> bool {
        self.armed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Frames the writer has committed to disk so far.
    pub fn written(&self) -> usize {
        self.written.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Frames skipped because the queue was full (disk slower than capture).
    pub fn dropped(&self) -> usize {
        self.dropped.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Called by the pump for every frame while armed (cheap: Arc clone).
    pub fn offer(&self, frame: &Frame) {
        use std::sync::mpsc::TrySendError;
        if !self.is_armed() {
            return;
        }
        let inner = self.inner.lock().unwrap();
        if let Some(tx) = inner.as_ref().and_then(|sp| sp.tx.as_ref()) {
            match tx.try_send(frame.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    self.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Err(TrySendError::Disconnected(_)) => {}
            }
        }
    }

    /// Stop accepting frames immediately; the writer keeps draining what is
    /// already queued. Non-blocking (call `finish` to join).
    pub fn disarm(&self) {
        self.armed.store(false, std::sync::atomic::Ordering::SeqCst);
        if let Some(sp) = self.inner.lock().unwrap().as_mut() {
            sp.tx = None;
        }
    }

    /// Join the writer (blocks until the queue is flushed to disk). Returns
    /// the spool dir, the written frames' UTC midpoints (in file order), and
    /// the dropped count.
    pub fn finish(&self) -> std::io::Result<(std::path::PathBuf, Vec<f64>, usize)> {
        self.disarm();
        let sp = self.inner.lock().unwrap().take();
        let Some(mut sp) = sp else {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no spool armed"));
        };
        sp.tx = None;
        let times = match sp.writer.take() {
            Some(h) => h.join().map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "spool writer panicked"))??,
            None => Vec::new(),
        };
        Ok((sp.dir, times, self.dropped()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn frame(t: f64) -> Frame {
        Frame {
            data: Arc::new(vec![7u8; 64 * 48]),
            width: 64,
            height: 48,
            channels: 1,
            utc_midpoint_s: t,
            capture_time_s: 0.001,
            seq: (t * 1000.0) as u64,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("sk_spool_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn spool_writes_all_when_io_keeps_up() {
        let rec = CaptureRecorder::new();
        let dir = tmp("all");
        rec.arm_spool(&dir, 64).unwrap();
        for i in 0..200 {
            rec.offer(&frame(1000.0 + i as f64));
            std::thread::sleep(std::time::Duration::from_micros(300));
        }
        let (sdir, times, dropped) = rec.finish().unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(times.len(), 200);
        assert_eq!(std::fs::read_dir(&sdir).unwrap().count(), 200);
        assert!(times.windows(2).all(|w| w[0] < w[1]), "times must stay in offer order");
        let _ = std::fs::remove_dir_all(sdir);
    }

    #[test]
    fn spool_bounds_memory_and_drops_newest_when_full() {
        let rec = CaptureRecorder::new();
        let dir = tmp("full");
        rec.arm_spool(&dir, 4).unwrap();
        // Flood far faster than the writer can drain a 4-deep queue.
        for i in 0..2000 {
            rec.offer(&frame(2000.0 + i as f64));
        }
        let (sdir, times, dropped) = rec.finish().unwrap();
        assert_eq!(times.len() + dropped, 2000, "every offer is either written or counted dropped");
        assert!(times.len() >= 4, "queued frames always flush");
        assert_eq!(std::fs::read_dir(&sdir).unwrap().count(), times.len());
        let _ = std::fs::remove_dir_all(sdir);
    }

    #[test]
    fn gray8_bmp_roundtrips_through_image_crate() {
        let dir = tmp("gray8");
        std::fs::create_dir_all(&dir).unwrap();
        let mut data = vec![0u8; 61 * 33]; // odd width exercises row padding
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        let f = Frame { data: Arc::new(data.clone()), width: 61, height: 33, channels: 1, utc_midpoint_s: 0.0, capture_time_s: 0.0, seq: 0 };
        let path = dir.join("g.bmp");
        write_bmp(&path, &f).unwrap();
        let img = image::open(&path).unwrap().to_luma8();
        assert_eq!((img.width(), img.height()), (61, 33));
        assert_eq!(img.as_raw().as_slice(), data.as_slice(), "pixel-exact after the palette round-trip");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Sustained-rate probe for THIS machine's disk: run with
    /// `cargo test -p skytracker-camera --release -- --ignored bmp_throughput --nocapture`.
    #[test]
    #[ignore]
    fn bmp_throughput() {
        let dir = tmp("rate");
        std::fs::create_dir_all(&dir).unwrap();
        let f = Frame {
            data: Arc::new(vec![42u8; 1248 * 936]),
            width: 1248,
            height: 936,
            channels: 1,
            utc_midpoint_s: 0.0,
            capture_time_s: 0.01,
            seq: 0,
        };
        let n = 120;
        let t0 = std::time::Instant::now();
        for i in 0..n {
            write_bmp(&dir.join(format!("f{i:04}.bmp")), &f).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        let mbps = (n as f64 * (1248.0 * 936.0 + 1078.0)) / 1e6 / dt;
        let fps = n as f64 / dt;
        eprintln!("bmp write: {fps:.1} frames/s, {mbps:.0} MB/s (1248x936 mono -> 8-bit BMP)");
        let _ = std::fs::remove_dir_all(dir);
        assert!(fps > 5.0);
    }
}
