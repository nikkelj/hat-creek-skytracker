//! Mount transport abstraction + a byte-level simulated mount.
//!
//! `Mount<T: Transport>` is the Rust port of the used subset of
//! `auxstar.NexstarHandController`: it encodes a request, runs an atomic
//! write+read transaction, and parses the response — including the same
//! short-read => error semantics that fixed the tracking-runaway bug.
//!
//! `SimResponder` is the piece the existing Python sim lacks: a BYTE-LEVEL
//! NexStar endpoint. The Python `SimMount` duck-types the controller at the
//! method level (`hc_get_position` returns a number); it never speaks bytes,
//! so it cannot exercise a real serial round-trip. `SimResponder` decodes the
//! 8-byte 'P' requests and emits correct response bytes, reusing the same
//! rate-integration physics. Wrapped in `LoopbackTransport`, it lets a Rust
//! `Mount` (or, via PyO3, Python) drive a full encode -> transact -> parse loop
//! with no hardware.

use crate::protocol;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MountError {
    /// A timed-out / partial read. Callers skip the cycle (never run open-loop).
    ShortRead { got: usize, expected: usize },
}

impl std::fmt::Display for MountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MountError::ShortRead { got, expected } => {
                write!(f, "Short read: got {} of {} bytes", got, expected)
            }
        }
    }
}
impl std::error::Error for MountError {}

/// Anything that can carry a NexStar transaction (a real serial port, the
/// in-memory loopback below, or a bridge to a Python mount object).
pub trait Transport {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()>;
    /// Return up to `n` bytes. Fewer than `n` signals a short/timed-out read.
    fn read(&mut self, n: usize) -> Vec<u8>;
}

/// Lets `Mount<Box<dyn Transport + Send>>` hold any transport behind a trait
/// object, so a single loop type can drive a real serial port OR a Python
/// sim-mount bridge chosen at runtime.
impl Transport for Box<dyn Transport + Send> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        (**self).write(data)
    }
    fn read(&mut self, n: usize) -> Vec<u8> {
        (**self).read(n)
    }
}

/// High-level mount API over any transport. Mirrors the used subset of
/// `NexstarHandController`.
pub struct Mount<T: Transport> {
    pub io: T,
}

impl<T: Transport> Mount<T> {
    pub fn new(io: T) -> Self {
        Self { io }
    }

    fn transact(&mut self, req: &[u8], expected: usize) -> Result<Vec<u8>, MountError> {
        // NOTE: a real serial Transport must hold its lock across write+read so
        // the control thread and UI thread can't interleave bytes (the auxstar
        // RLock invariant). The loopback below is single-threaded by construction.
        let _ = self.io.write(req);
        let resp = self.io.read(expected);
        if resp.len() < expected {
            return Err(MountError::ShortRead {
                got: resp.len(),
                expected,
            });
        }
        Ok(resp)
    }

    pub fn hc_get_position(&mut self, target: u8) -> Result<f64, MountError> {
        let req = protocol::encode_get_position(target);
        let expected = protocol::expected_response_len(protocol::MC_GET_POSITION.2);
        let resp = self.transact(&req, expected)?;
        Ok(protocol::unpack_int3(&resp))
    }

    pub fn hc_slew_fixed(&mut self, target: u8, rate_step: i32) -> Result<bool, MountError> {
        let req = protocol::encode_slew_fixed(target, rate_step);
        let expected = protocol::expected_response_len(protocol::MC_MOVE_POS.2);
        let resp = self.transact(&req, expected)?;
        Ok(resp.first() == Some(&b'#'))
    }

    /// Continuous variable rate in deg/s (MC_SET_POS/NEG_GUIDERATE) -- the fine
    /// alternative to the 10-step hc_slew_fixed.
    pub fn hc_set_rate_dps(&mut self, target: u8, dps: f64) -> Result<bool, MountError> {
        let req = protocol::encode_set_guiderate(target, dps);
        let expected = protocol::expected_response_len(protocol::MC_SET_POS_GUIDERATE.2);
        let resp = self.transact(&req, expected)?;
        Ok(resp.first() == Some(&b'#'))
    }

    pub fn hc_goto_fast(
        &mut self,
        target: u8,
        dd: f64,
        mm: f64,
        ss: f64,
    ) -> Result<bool, MountError> {
        let req = protocol::encode_goto_fast(target, dd, mm, ss);
        let expected = protocol::expected_response_len(protocol::MC_GOTO_FAST.2);
        let resp = self.transact(&req, expected)?;
        Ok(resp.first() == Some(&b'#'))
    }
}

/// Time source for the sim physics. Wall clock in real use; manual in tests so
/// integration is deterministic.
enum Clock {
    Wall(std::time::Instant),
    Manual(f64),
}
impl Clock {
    fn now(&self) -> f64 {
        match self {
            Clock::Wall(start) => start.elapsed().as_secs_f64(),
            Clock::Manual(t) => *t,
        }
    }
}

/// Byte-level simulated mount. Integrates the rate commands the control loop
/// issues into a true pointing and reports an encoder position (optionally with
/// fixed axis misalignment). Mirrors `simulator.SimMount` physics.
pub struct SimResponder {
    pub az_true_deg: f64,
    pub el_true_deg: f64,
    /// Focus motor: a third rate-commanded axis on the same encoder/rate
    /// convention as az/el, integrated cleanly (no misalignment) so a focus
    /// read-back follows the focus move commands.
    pub focus_true_deg: f64,
    az_rate_dps: f64,
    el_rate_dps: f64,
    focus_rate_dps: f64,
    last_t: f64,
    clock: Clock,
    pub mis_az_deg: f64,
    pub mis_el_deg: f64,
}

impl SimResponder {
    pub fn new_wall(az0_deg: f64, el0_deg: f64) -> Self {
        Self {
            az_true_deg: az0_deg,
            el_true_deg: el0_deg,
            focus_true_deg: 0.0,
            az_rate_dps: 0.0,
            el_rate_dps: 0.0,
            focus_rate_dps: 0.0,
            last_t: 0.0,
            clock: Clock::Wall(std::time::Instant::now()),
            mis_az_deg: 0.0,
            mis_el_deg: 0.0,
        }
    }

    /// Manual-clock variant for deterministic tests. Advance time with
    /// `advance_time`.
    pub fn new_manual(az0_deg: f64, el0_deg: f64) -> Self {
        let mut s = Self::new_wall(az0_deg, el0_deg);
        s.clock = Clock::Manual(0.0);
        s
    }

    /// Manual-clock only: move the simulated time forward by `dt` seconds.
    pub fn advance_time(&mut self, dt: f64) {
        if let Clock::Manual(t) = &mut self.clock {
            *t += dt;
        }
    }

    fn advance(&mut self) {
        let now = self.clock.now();
        let dt = now - self.last_t;
        self.last_t = now;
        if dt <= 0.0 {
            return;
        }
        self.az_true_deg = (self.az_true_deg + self.az_rate_dps * dt).rem_euclid(360.0);
        self.el_true_deg += self.el_rate_dps * dt;
        self.focus_true_deg = (self.focus_true_deg + self.focus_rate_dps * dt).rem_euclid(360.0);
    }

    /// Decode one 8-byte 'P' request and produce the response bytes
    /// (`resp_bytes` data bytes followed by the trailing '#'). A malformed
    /// request yields an empty response, which the Mount surfaces as a
    /// ShortRead — exactly how a real timeout looks.
    pub fn respond(&mut self, req: &[u8]) -> Vec<u8> {
        if req.len() < 8 || req[0] != 0x50 {
            return Vec::new();
        }
        let dest = req[2];
        let msg_id = req[3];
        let data = [req[4], req[5], req[6]];
        let resp_len = req[7] as usize;

        if msg_id == protocol::MC_GET_POSITION.0 {
            self.advance();
            let (true_deg, mis) = if dest == protocol::targets::ALT {
                (self.el_true_deg, self.mis_el_deg)
            } else if dest == protocol::targets::FOCUS {
                (self.focus_true_deg, 0.0)
            } else {
                (self.az_true_deg, self.mis_az_deg)
            };
            let frac = (true_deg + mis).rem_euclid(360.0) / 360.0;
            let mut out = protocol::pack_int3(frac).to_vec();
            out.push(b'#');
            out
        } else if msg_id == protocol::MC_MOVE_POS.0 || msg_id == protocol::MC_MOVE_NEG.0 {
            self.advance();
            let sign = if msg_id == protocol::MC_MOVE_POS.0 { 1.0 } else { -1.0 };
            let dps = sign * protocol::rate(data[0]) * 360.0;
            if dest == protocol::targets::ALT {
                self.el_rate_dps = dps;
            } else if dest == protocol::targets::FOCUS {
                self.focus_rate_dps = dps;
            } else {
                self.az_rate_dps = dps;
            }
            vec![b'#']
        } else if msg_id == protocol::MC_SET_POS_GUIDERATE.0
            || msg_id == protocol::MC_SET_NEG_GUIDERATE.0
        {
            // Continuous variable rate: value packed in rev/sec, sign by opcode.
            self.advance();
            let sign = if msg_id == protocol::MC_SET_POS_GUIDERATE.0 {
                1.0
            } else {
                -1.0
            };
            let dps = sign * protocol::unpack_int3(&data) * 360.0;
            if dest == protocol::targets::ALT {
                self.el_rate_dps = dps;
            } else if dest == protocol::targets::FOCUS {
                self.focus_rate_dps = dps;
            } else {
                self.az_rate_dps = dps;
            }
            vec![b'#']
        } else if msg_id == protocol::MC_GOTO_FAST.0 {
            self.advance();
            let deg = protocol::unpack_int3(&data) * 360.0;
            if dest == protocol::targets::ALT {
                self.el_true_deg = deg;
                self.el_rate_dps = 0.0;
            } else if dest == protocol::targets::FOCUS {
                self.focus_true_deg = deg.rem_euclid(360.0);
                self.focus_rate_dps = 0.0;
            } else {
                self.az_true_deg = deg.rem_euclid(360.0);
                self.az_rate_dps = 0.0;
            }
            vec![b'#']
        } else if msg_id == protocol::MC_GET_VER.0 {
            // Arbitrary firmware "7.15" + hash.
            vec![0x07, 0x0f, b'#']
        } else {
            // Generic ack: resp_len data bytes (zeroed) + hash.
            let mut out = vec![0u8; resp_len];
            out.push(b'#');
            out
        }
    }
}

/// In-memory transport: feeds each written request straight to a `SimResponder`
/// and buffers the response bytes for the next read. Makes
/// `Mount<LoopbackTransport>` a fully self-contained closed mount.
pub struct LoopbackTransport {
    pub responder: SimResponder,
    buf: std::collections::VecDeque<u8>,
}

impl LoopbackTransport {
    pub fn new(responder: SimResponder) -> Self {
        Self {
            responder,
            buf: std::collections::VecDeque::new(),
        }
    }
}

impl Transport for LoopbackTransport {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        let resp = self.responder.respond(data);
        self.buf.extend(resp);
        Ok(())
    }

    fn read(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.buf.pop_front() {
                Some(b) => out.push(b),
                None => break,
            }
        }
        out
    }
}

/// Recording transport for tests: captures every byte written, returns a fixed
/// canned response. Used to assert request-byte encoding.
pub struct RecordingTransport {
    pub written: Vec<u8>,
    pub canned: Vec<u8>,
}

impl RecordingTransport {
    pub fn new(canned: &[u8]) -> Self {
        Self {
            written: Vec::new(),
            canned: canned.to_vec(),
        }
    }
}

impl Transport for RecordingTransport {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.written.extend_from_slice(data);
        Ok(())
    }
    fn read(&mut self, n: usize) -> Vec<u8> {
        self.canned.iter().take(n).copied().collect()
    }
}
