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

    /// Monotonic count of loop cycles (including idle/faulted ones). The
    /// Python adapter watches this to detect a dead loop thread — a snapshot
    /// whose cycle_count stops advancing is stale no matter what `fresh` says.
    pub cycle_count: u64,
    /// Set when the loop thread has exited (clean stop or contained panic).
    pub loop_dead: bool,
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

/// What a single call to `run_cycle` did, so the loop thread can apply the
/// consecutive-fault safety policy from MountControlThread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleOutcome {
    /// Full cycle ran (poll + step + actuate + publish).
    Ran,
    /// Not connected — nothing to do.
    Idle,
    /// Comm fault (short read / timeout); the cycle was skipped.
    Fault,
}

fn bump_cycle_count(shared: &Shared) {
    let mut o = shared.outputs.lock().unwrap();
    o.cycle_count += 1;
}

/// One control cycle: poll position, drain commands, run the decision step,
/// actuate, publish outputs. A failed poll (short read) skips the cycle without
/// leaving the mount open-loop — the safety stance from MountControlThread.
/// The caller is responsible for acting on repeated `Fault` outcomes (the
/// threaded loop stops motion after `MAX_CONSECUTIVE_FAULTS`).
pub fn run_cycle<T: Transport>(
    mount: &mut Mount<T>,
    state: &mut LoopState,
    shared: &Shared,
    now: f64,
) -> CycleOutcome {
    bump_cycle_count(shared);
    let inputs = shared.inputs.lock().unwrap().clone();

    if !inputs.connected {
        return CycleOutcome::Idle;
    }

    // 1) Single position poll per cycle (degrees), offset-applied. On a comm
    //    fault, mark the snapshot stale so readers don't keep trusting the
    //    last position, and report the fault to the caller.
    let fault = |shared: &Shared| {
        shared.outputs.lock().unwrap().fresh = false;
        CycleOutcome::Fault
    };
    let azm_raw = match mount.hc_get_position(targets::AZM) {
        Ok(v) => v * 360.0,
        Err(_) => return fault(shared),
    };
    let alt_raw = match mount.hc_get_position(targets::ALT) {
        Ok(v) => v * 360.0,
        Err(_) => return fault(shared),
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

    // 5) Actuate (only when the step commands a rate). In continuous-rate mode
    //    PROGRAM/HOTSPOT issue the fine variable-rate (guide-rate) command using
    //    the PID's continuous output, falling back to the discrete MC_MOVE step
    //    above guide_rate_max_dps (e.g. the near-zenith keyhole). Manual
    //    RATE_CONTROL always uses the discrete joystick step.
    let continuous = inputs.continuous_rate
        && matches!(inputs.mode, Mode::Program | Mode::Hotspot);
    let max_dps = inputs.guide_rate_max_dps;
    if let Some(r) = out.azm_rate_cmd {
        let dps = out.azm_pid_output * 360.0;
        if continuous && dps.abs() <= max_dps {
            let _ = mount.hc_set_rate_dps(targets::AZM, dps);
        } else {
            let _ = mount.hc_slew_fixed(targets::AZM, r);
        }
    }
    if let Some(r) = out.alt_rate_cmd {
        let dps = out.alt_pid_output * 360.0;
        if continuous && dps.abs() <= max_dps {
            let _ = mount.hc_set_rate_dps(targets::ALT, dps);
        } else {
            let _ = mount.hc_slew_fixed(targets::ALT, r);
        }
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
    drop(o);
    CycleOutcome::Ran
}

/// Consecutive comm faults before the loop stops motion (MountControlThread's
/// safety watchdog, ported).
pub const MAX_CONSECUTIVE_FAULTS: u32 = 3;

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
                let mut consecutive_faults = 0u32;
                while !thread_shared.stop.load(Ordering::Relaxed) {
                    let cycle_start = Instant::now();
                    let now = start.elapsed().as_secs_f64();

                    // Contain panics (e.g. from pathological camera frames):
                    // an unwinding loop thread must never skip the safe-stop
                    // below and leave the mount slewing while Python keeps
                    // reading a stale-but-plausible snapshot.
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || run_cycle(&mut mount, &mut state, &thread_shared, now),
                    ));
                    match outcome {
                        Ok(CycleOutcome::Ran) | Ok(CycleOutcome::Idle) => {
                            consecutive_faults = 0;
                        }
                        Ok(CycleOutcome::Fault) => {
                            // Ported MountControlThread policy: after a few
                            // consecutive comm faults, never leave the mount
                            // running open-loop — keep commanding a stop
                            // (best effort on a possibly-dead link).
                            consecutive_faults += 1;
                            if consecutive_faults >= MAX_CONSECUTIVE_FAULTS {
                                apply_command(&mut mount, &Command::Stop);
                                if consecutive_faults == MAX_CONSECUTIVE_FAULTS {
                                    if let Ok(mut o) = thread_shared.outputs.lock() {
                                        o.status_msgs.push(format!(
                                            "CoreLoop: {consecutive_faults} consecutive comm faults - motion stopped"
                                        ));
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // A cycle panicked. Stop the mount, mark the loop
                            // dead so the adapter's watchdog surfaces it, and
                            // halt — the loop's state can no longer be trusted.
                            apply_command(&mut mount, &Command::Stop);
                            if let Ok(mut o) = thread_shared.outputs.lock() {
                                o.fresh = false;
                                o.loop_dead = true;
                                o.status_msgs.push(
                                    "CoreLoop: cycle panicked - motion stopped, loop halted"
                                        .to_string(),
                                );
                            }
                            return;
                        }
                    }

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
                if let Ok(mut o) = thread_shared.outputs.lock() {
                    o.loop_dead = true;
                }
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
