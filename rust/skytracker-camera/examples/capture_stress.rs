//! Capture stress harness: three synthetic cameras at the rig's real frame
//! sizes and rates, through the real Pump -> Ring -> CaptureRecorder path,
//! with the app's 100 Hz consumer loop mimicked. Reports throughput, drops,
//! ring depth and write latency every 5 s; the process RSS is sampled from
//! outside (scratchpad/memwatch.ps1).
//!
//!   cargo run --release -p skytracker-camera --example capture_stress -- \
//!       <seconds> <out_dir> [ring_cap] [spool_cap] [ring_mb] [spool_mb]

use skytracker_camera::capture::CaptureRecorder;
use skytracker_camera::pump::{Pump, PushSource};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Cam {
    name: &'static str,
    w: usize,
    h: usize,
    fps: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let secs: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let out = std::path::PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "stress_out".into()));
    let ring_cap: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(600);
    let spool_cap: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let ring_mb: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX / (1024 * 1024));
    let spool_mb: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX / (1024 * 1024));
    let cams = [
        Cam { name: "guide", w: 3096, h: 2080, fps: 11.0 },
        Cam { name: "main", w: 1608, h: 1104, fps: 6.7 },
        Cam { name: "bubble", w: 968, h: 548, fps: 20.0 },
    ];
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    println!("stress: {secs:.0} s, ring_cap {ring_cap} / {ring_mb} MB, spool_cap {spool_cap} / {spool_mb} MB, out {}", out.display());
    let nominal: f64 = cams.iter().map(|c| (c.w * c.h) as f64 * c.fps).sum::<f64>() / 1e6;
    println!("nominal input rate {nominal:.1} MB/s ({:.1} GB per 10 min)", nominal * 600.0 / 1024.0);

    let stop = Arc::new(AtomicBool::new(false));
    let mut pumps = Vec::new();
    let mut recs = Vec::new();
    let mut offered = Vec::new();
    let mut handles = Vec::new();
    for c in &cams {
        let (src, push) = PushSource::new();
        let pump = Arc::new(Pump::spawn_with_budget(src, ring_cap, ring_mb.saturating_mul(1024 * 1024)));
        let rec = Arc::new(CaptureRecorder::new());
        let queue = (spool_mb.saturating_mul(1024 * 1024) / (c.w * c.h)).clamp(2, spool_cap.max(2));
        println!("  {}: spool queue {queue} frames", c.name);
        rec.arm_spool(&out.join(c.name), queue).unwrap();
        let off = Arc::new(AtomicU64::new(0));
        // Producer: a fresh buffer per frame like the ASI path, at the hardware rate.
        {
            let stop = stop.clone();
            let (w, h, fps) = (c.w, c.h, c.fps);
            handles.push(std::thread::spawn(move || {
                let period = Duration::from_secs_f64(1.0 / fps);
                let mut next = Instant::now();
                let mut i = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    let mut data = vec![0u8; w * h];
                    // Touch the buffer so the pages are really committed (as a real
                    // frame would be) and vary content a little.
                    for row in data.chunks_mut(w).step_by(64) {
                        row[0] = (i & 0xff) as u8;
                    }
                    push.push(data, w, h, 1, 1.0 / fps);
                    i += 1;
                    next += period;
                    let now = Instant::now();
                    if next > now {
                        std::thread::sleep(next - now);
                    } else {
                        next = now;
                    }
                }
                push.close();
            }));
        }
        // Consumer: the app's run_slot loop — poll the ring at 100 Hz, offer new frames.
        {
            let stop = stop.clone();
            let pump = pump.clone();
            let rec = rec.clone();
            let off = off.clone();
            handles.push(std::thread::spawn(move || {
                let mut last_seq = u64::MAX;
                while !stop.load(Ordering::Relaxed) {
                    if let Some(f) = pump.ring.latest() {
                        if f.seq != last_seq {
                            last_seq = f.seq;
                            rec.offer(&f);
                            off.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }));
        }
        pumps.push(pump);
        recs.push(rec);
        offered.push(off);
    }

    let t0 = Instant::now();
    let mut last_written = vec![0usize; cams.len()];
    let mut last_t = t0;
    while t0.elapsed().as_secs_f64() < secs {
        std::thread::sleep(Duration::from_secs(5));
        let now = Instant::now();
        let dt = (now - last_t).as_secs_f64();
        let mut line = format!("t={:5.0}s", t0.elapsed().as_secs_f64());
        let mut mbps = 0.0;
        for (i, c) in cams.iter().enumerate() {
            let w = recs[i].written();
            let d = recs[i].dropped();
            let o = offered[i].load(Ordering::Relaxed);
            let rl = pumps[i].ring.len();
            let rb = pumps[i].ring.bytes() / (1024 * 1024);
            mbps += (w - last_written[i]) as f64 * (c.w * c.h) as f64 / dt / 1e6;
            last_written[i] = w;
            let failed = recs[i].failed().map(|e| format!(" FAILED[{e}]")).unwrap_or_default();
            line += &format!("  {}: off {o} wr {w} drop {d} ring {rl}/{rb}MB{failed}", c.name);
        }
        line += &format!("  disk {mbps:.0} MB/s");
        println!("{line}");
        last_t = now;
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    println!("finishing (join writers, drain queues)…");
    let tf = Instant::now();
    let mut total_bytes = 0u64;
    for (i, c) in cams.iter().enumerate() {
        let t = Instant::now();
        match recs[i].finish() {
            Ok((dir, times, dropped)) => {
                let n = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
                let bytes: u64 = std::fs::read_dir(&dir).map(|d| d.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
                total_bytes += bytes;
                let span = times.last().copied().unwrap_or(0.0) - times.first().copied().unwrap_or(0.0);
                println!(
                    "  {}: {} frames on disk ({:.2} GB), {} dropped, span {:.1} s, effective {:.2} fps (nominal {:.1}), finish took {:.2} s",
                    c.name, n, bytes as f64 / 1e9, dropped, span, times.len() as f64 / span.max(1e-9), c.fps, t.elapsed().as_secs_f64()
                );
            }
            Err(e) => println!("  {}: finish FAILED: {e}", c.name),
        }
    }
    println!("total {:.2} GB in {:.0} s; finish phase {:.2} s", total_bytes as f64 / 1e9, secs, tf.elapsed().as_secs_f64());
}
