//! skytracker-camera — the capture pipeline (Phase 4 of the Rust port).
//!
//! The Python capture path (CameraThread + CircularBuffer + pygame
//! conversion, all under the GIL) capped at 4-10 FPS against cameras
//! capable of 50-100. Here the whole per-frame path — source pull,
//! exposure-midpoint stamping, ring buffer, armed-capture retention,
//! BMP dump — runs in Rust threads; Python touches frames only when it
//! pulls one for display. The ASI SDK binding loads ASICamera2.dll at
//! runtime (hardware timing truth to be validated on the rig); the push
//! source lets the hardware simulator drive the identical pipeline.

pub mod asi;
pub mod capture;
pub mod pump;
pub mod ring;

#[cfg(test)]
mod tests {
    use crate::pump::{FrameSource, Pump};
    use crate::ring::exposure_midpoint;

    struct Synthetic {
        n: usize,
        emitted: usize,
    }

    impl FrameSource for Synthetic {
        fn next_frame(&mut self) -> Option<(Vec<u8>, usize, usize, usize, f64)> {
            if self.emitted >= self.n {
                return None;
            }
            self.emitted += 1;
            Some((vec![7u8; 640 * 480], 640, 480, 1, 0.010))
        }
    }

    #[test]
    fn pump_throughput_unthrottled() {
        // 500 VGA frames through the full pipeline; must sustain >1000 FPS
        // internally (the real-world cap is then the camera, not us).
        let t0 = std::time::Instant::now();
        let pump = Pump::spawn(
            Synthetic {
                n: 500,
                emitted: 0,
            },
            1000,
        );
        while pump.frames_pumped() < 500 {
            std::thread::yield_now();
        }
        let fps = 500.0 / t0.elapsed().as_secs_f64();
        assert_eq!(pump.ring.len(), 500);
        let latest = pump.ring.latest().unwrap();
        assert_eq!(latest.seq, 499);
        pump.join();
        println!("internal pump rate: {fps:.0} FPS");
        assert!(fps > 1000.0, "pump too slow: {fps:.0} FPS");
    }

    #[test]
    fn midpoint_backdating() {
        let m = exposure_midpoint(0.5, 100.0);
        assert!((m - 99.75).abs() < 1e-12);
        assert_eq!(exposure_midpoint(-1.0, 100.0), 100.0);
    }
}
