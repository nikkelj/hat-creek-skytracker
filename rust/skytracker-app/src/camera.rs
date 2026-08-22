//! Camera worker: a synthetic star-field source (the hardware simulator's
//! essence in Rust — catalog stars projected through camera 1's pinhole at
//! the mount's live pointing, Gaussian PSFs + read noise) pumped through
//! the real skytracker-camera pipeline at ~100 FPS, publishing the latest
//! frame for the 120 Hz display. Swapping in `CameraPipeline::open_asi` at
//! rig time changes only the source.

use crate::state::{CamSnapshot, Shared};
use skytracker_camera::pump::{Pump, PushShared, PushSource};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const CAM_W: usize = 640;
pub const CAM_H: usize = 480;
pub const CAM_FPS: f64 = 100.0;

pub fn spawn(shared: Arc<Shared>) {
    std::thread::Builder::new()
        .name("camera-sim".into())
        .spawn(move || run(shared))
        .expect("spawn camera worker");
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }
}

fn render(shared: &Shared, buf: &mut [u8], rng: &mut Lcg, fov_deg: f64) {
    let mount = shared.mount.load();
    let sky = shared.sky.load();
    let (w, h) = (CAM_W as f64, CAM_H as f64);
    // Background + noise.
    for v in buf.iter_mut() {
        let n = rng.next() * 2.0 - 1.0;
        *v = (12.0 + n * 3.0).clamp(0.0, 255.0) as u8;
    }
    let fov_h = fov_deg * h / w;
    let scale = w / fov_deg; // px per degree
    let cos_el = mount.el.to_radians().cos().max(0.05);
    let half_diag = (fov_deg * fov_deg + fov_h * fov_h).sqrt() / 2.0 + 0.2;

    let mut draw = |az: f64, el: f64, amp: f32, sigma: f32| {
        let daz = (az - mount.az + 540.0).rem_euclid(360.0) - 180.0;
        let del = el - mount.el;
        if daz.abs() * cos_el > half_diag || del.abs() > half_diag {
            return;
        }
        let x = w / 2.0 + daz * cos_el * scale;
        let y = h / 2.0 - del * scale;
        if x < -5.0 || y < -5.0 || x > w + 5.0 || y > h + 5.0 {
            return;
        }
        let r = (sigma * 3.0).ceil() as i64;
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        for yy in (yi - r).max(0)..=(yi + r).min(CAM_H as i64 - 1) {
            for xx in (xi - r).max(0)..=(xi + r).min(CAM_W as i64 - 1) {
                let dx = xx as f64 - x;
                let dy = yy as f64 - y;
                let g = amp * (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp();
                let i = yy as usize * CAM_W + xx as usize;
                buf[i] = (buf[i] as f32 + g).clamp(0.0, 255.0) as u8;
            }
        }
    };

    for s in &sky.stars {
        let amp = (250.0 * 10f64.powf(-0.4 * (s.mag - 1.0))).clamp(25.0, 250.0) as f32;
        draw(s.az, s.el, amp, 1.3);
    }
    for s in &sky.sats {
        if s.el > 0.0 {
            draw(s.az, s.el, 230.0, 1.8);
        }
    }
}

fn run(shared: Arc<Shared>) {
    let (source, push): (PushSource, Arc<PushShared>) = PushSource::new();
    let pump = Pump::spawn(source, 600);
    let fov_deg = shared.config.cam1_fov_deg(CAM_W as f64).max(0.3);
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let period = Duration::from_secs_f64(1.0 / CAM_FPS);
    let mut fps_t0 = Instant::now();
    let mut fps_n = 0u64;
    let mut fps = 0.0;
    let mut last_seq = u64::MAX;
    loop {
        let t0 = Instant::now();
        let mut buf = vec![0u8; CAM_W * CAM_H];
        render(&shared, &mut buf, &mut rng, fov_deg);
        let render_s = t0.elapsed().as_secs_f64();
        push.push(buf, CAM_W, CAM_H, 1, render_s.max(1.0 / CAM_FPS));

        // Publish the newest pumped frame (Arc, no copy).
        if let Some(f) = pump.ring.latest() {
            if f.seq != last_seq {
                last_seq = f.seq;
                fps_n += 1;
                if fps_t0.elapsed() >= Duration::from_secs(1) {
                    fps = fps_n as f64 / fps_t0.elapsed().as_secs_f64();
                    fps_n = 0;
                    fps_t0 = Instant::now();
                }
                shared.cam.store(Arc::new(Some(CamSnapshot {
                    data: f.data.clone(),
                    width: f.width,
                    height: f.height,
                    seq: f.seq,
                    fps,
                    utc_midpoint_s: f.utc_midpoint_s,
                    fov_deg,
                })));
            }
        }
        let sleep = period.saturating_sub(t0.elapsed());
        if !sleep.is_zero() {
            std::thread::sleep(sleep);
        }
    }
}
