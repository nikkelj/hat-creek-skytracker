//! Threaded fixed-rate control loop — Rust port of `MountControlThread`.
//!
//! Owns the `Mount` (the only writer of the serial link) and a `LoopState`,
//! and runs `poll -> drain commands -> step -> actuate -> publish` at a fixed
//! cadence on its own OS thread. The decision logic lives in
//! `controller::LoopState::step` (pure); this module is the I/O + threading +
//! shared-state shell.
//!
//! `run_cycle` is factored out so tests can drive the loop deterministically
//! against the in-memory `Mount<LoopbackTransport>` sim with no thread or wall
//! clock (`tests/core_loop.rs`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::controller::{Frame, Inputs, LoopState, Mode};
use crate::protocol::targets;
use crate::sim::{Mount, Transport};

/// Manual command pushed by the UI (goto / slew / stop). All serial traffic —
/// tracking and manual — flows through the loop so the port has one owner.
#[derive(Debug, Clone)]
pub enum Command {
    Stop,
    Slew { azm: i32, alt: i32 },
    GotoMount { azm_deg: f64, alt_deg: f64 },
}

/// Snapshot the UI polls for display.
#[derive(Default, Clone)]
pub struct Outputs {
    pub azm: f64,
    pub alt: f64,
    pub azm_raw: f64,
    pub alt_raw: f64,
    pub fresh: bool,

    pub azm_error: f64,
    pub alt_error: f64,
    pub azm_rate_cmd: i32,
    pub alt_rate_cmd: i32,
    pub azm_pid_output: f64,
    pub alt_pid_output: f64,

    pub hotspot_acquired: bool,
    pub hotspot_status: String,
    pub hotspot_snr: f64,
    pub hotspot_centroid: Option<(f64, f64)>,

    pub requested_mode: Option<Mode>,
    pub status_msgs: Vec<String>,

    pub actual_hz: f64,
    pub cycle_ms: f64,
}

/// State shared between the Python shim (writers) and the loop thread (reader/
/// writer of outputs). Frame is cloned out (Arc bump) so the camera shim is
/// never blocked by detection.
pub struct Shared {
    pub inputs: Mutex<Inputs>,
    pub outputs: Mutex<Outputs>,
    pub frame: Mutex<Option<Arc<Frame>>>,
    pub commands: Mutex<VecDeque<Command>>,
    pub stop: AtomicBool,
}

impl Shared {
    pub fn new(inputs: Inputs) -> Arc<Self> {
        Arc::new(Shared {
            inputs: Mutex::new(inputs),
            outputs: Mutex::new(Outputs::default()),
            frame: Mutex::new(None),
            commands: Mutex::new(VecDeque::new()),
            stop: AtomicBool::new(false),
        })
    }
}

fn apply_command<T: Transport>(mount: &mut Mount<T>, cmd: &Command) {
    let _ = match cmd {
        Command::Stop => {
            let _ = mount.hc_slew_fixed(targets::AZM, 0);
            mount.hc_slew_fixed(targets::ALT, 0)
        }
        Command::Slew { azm, alt } => {
            let _ = mount.hc_slew_fixed(targets::AZM, *azm);
            mount.hc_slew_fixed(targets::ALT, *alt)
        }
        Command::GotoMount { azm_deg, alt_deg } => {
            let _ = mount.hc_goto_fast(targets::AZM, *azm_deg, 0.0, 0.0);
            mount.hc_goto_fast(targets::ALT, *alt_deg, 0.0, 0.0)
        }
    };
}

/// One control cycle: poll position, drain commands, run the decision step,
/// actuate, publish outputs. A failed poll (short read) skips the cycle without
/// leaving the mount open-loop — the safety stance from MountControlThread.
pub fn run_cycle<T: Transport>(
    mount: &mut Mount<T>,
    state: &mut LoopState,
    shared: &Shared,
    now: f64,
) {
    let inputs = shared.inputs.lock().unwrap().clone();

    if !inputs.connected {
        return;
    }

    // 1) Single position poll per cycle (degrees), offset-applied.
    let azm_raw = match mount.hc_get_position(targets::AZM) {
        Ok(v) => v * 360.0,
        Err(_) => return, // skip cycle on comm fault
    };
    let alt_raw = match mount.hc_get_position(targets::ALT) {
        Ok(v) => v * 360.0,
        Err(_) => return,
    };
    let current_azm = azm_raw - inputs.offsets.0;
    let current_alt = alt_raw - inputs.offsets.1;

    // 2) Drain manual commands (they share the wire with tracking).
    let drained: Vec<Command> = {
        let mut q = shared.commands.lock().unwrap();
        q.drain(..).collect()
    };
    for cmd in &drained {
        apply_command(mount, cmd);
    }

    // 3) Frame (Arc clone so the camera shim isn't blocked during detection).
    let frame = shared.frame.lock().unwrap().clone();

    // 4) Decision step (pure).
    let out = state.step(&inputs, frame.as_deref(), current_azm, current_alt, now);

    // 5) Actuate (only when the step commands a rate).
    if let Some(r) = out.azm_rate_cmd {
        let _ = mount.hc_slew_fixed(targets::AZM, r);
    }
    if let Some(r) = out.alt_rate_cmd {
        let _ = mount.hc_slew_fixed(targets::ALT, r);
    }

    // 6) Publish snapshot.
    let mut o = shared.outputs.lock().unwrap();
    o.azm = current_azm;
    o.alt = current_alt;
    o.azm_raw = azm_raw;
    o.alt_raw = alt_raw;
    o.fresh = true;
    o.azm_error = out.azm_error;
    o.alt_error = out.alt_error;
    o.azm_rate_cmd = out.azm_rate_cmd.unwrap_or(o.azm_rate_cmd);
    o.alt_rate_cmd = out.alt_rate_cmd.unwrap_or(o.alt_rate_cmd);
    o.azm_pid_output = out.azm_pid_output;
    o.alt_pid_output = out.alt_pid_output;
    o.hotspot_acquired = out.hotspot_acquired;
    if !out.hotspot_status.is_empty() {
        o.hotspot_status = out.hotspot_status.to_string();
    }
    o.hotspot_snr = out.hotspot_snr;
    o.hotspot_centroid = out.hotspot_centroid;
    o.requested_mode = out.requested_mode;
    if let Some(msg) = out.status_msg {
        o.status_msgs.push(msg);
    }
}

/// The owning handle for the running loop thread.
pub struct CoreLoop {
    shared: Arc<Shared>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CoreLoop {
    /// Spawn the loop on its own thread. `mount` (any transport) and the
    /// `LoopState` move into the thread; the thread holds no GIL.
    pub fn spawn<T: Transport + Send + 'static>(
        mount: Mount<T>,
        shared: Arc<Shared>,
        target_hz: f64,
    ) -> Self {
        let period = Duration::from_secs_f64(1.0 / target_hz);
        let thread_shared = Arc::clone(&shared);
        let mut mount = mount;
        let mut state = LoopState::new();

        let handle = std::thread::Builder::new()
            .name("CoreLoop".into())
            .spawn(move || {
                let start = Instant::now();
                let mut count = 0u32;
                let mut window = Instant::now();
                while !thread_shared.stop.load(Ordering::Relaxed) {
                    let cycle_start = Instant::now();
                    let now = start.elapsed().as_secs_f64();
                    run_cycle(&mut mount, &mut state, &thread_shared, now);

                    let elapsed = cycle_start.elapsed();
                    // rate stats
                    count += 1;
                    if window.elapsed() >= Duration::from_secs(1) {
                        let hz = count as f64 / window.elapsed().as_secs_f64();
                        let mut o = thread_shared.outputs.lock().unwrap();
                        o.actual_hz = hz;
                        o.cycle_ms = elapsed.as_secs_f64() * 1000.0;
                        count = 0;
                        window = Instant::now();
                    }
                    if let Some(sleep) = period.checked_sub(elapsed) {
                        std::thread::sleep(sleep);
                    }
                }
                // On shutdown, never leave the mount slewing.
                apply_command(&mut mount, &Command::Stop);
            })
            .expect("spawn CoreLoop");

        CoreLoop {
            shared,
            handle: Some(handle),
        }
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for CoreLoop {
    fn drop(&mut self) {
        // Signal stop but do NOT join here. When the transport is the Python
        // mount bridge, the loop thread needs the GIL to finish its cycle;
        // joining while the GIL is held (e.g. during GC) would deadlock.
        // Explicit `stop()` joins with the GIL released (see the PyO3 wrapper).
        self.shared.stop.store(true, Ordering::Relaxed);
    }
}
