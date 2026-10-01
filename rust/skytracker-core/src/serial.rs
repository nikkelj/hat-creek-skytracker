//! Real serial `Transport` for the mount (the hardware path).
//!
//! Mirrors `lib/auxstar.py`'s serial settings: 9600 8N1 with a short timeout so
//! a missing response can never stall the control loop. A short read clears the
//! input buffer (matching `_transact`'s `reset_input_buffer`) so the next
//! command starts on a clean stream. One owner per port (the loop thread), so
//! no locking is needed here.
//!
//! Set `SKYTRACKER_SERIAL_TRACE=<file>` to hex-dump every write/read (capped)
//! for bench debugging of the wire protocol.
//!
//! Compiled only with the `serial` feature (it depends on the
//! `serialport` crate); cannot be unit-tested without a physical port.

use std::io::Write as _;
use std::time::Duration;

use crate::sim::Transport;

/// Bound the trace so a long session can't fill the disk.
const TRACE_MAX_LINES: u32 = 2000;

pub struct SerialTransport {
    port: Box<dyn serialport::SerialPort>,
    trace: Option<(std::fs::File, u32, std::time::Instant)>,
}

fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(":")
}

impl SerialTransport {
    pub fn open(path: &str, baud: u32, timeout_ms: u64) -> serialport::Result<Self> {
        let mut port = serialport::new(path, baud)
            .timeout(Duration::from_millis(timeout_ms))
            .data_bits(serialport::DataBits::Eight)
            .parity(serialport::Parity::None)
            .stop_bits(serialport::StopBits::One)
            .flow_control(serialport::FlowControl::None)
            .open()?;
        // pyserial asserts DTR and RTS on open (its dsrdtr/rtscts=False only
        // disable flow *control*, not the line states); serialport leaves both
        // low on Windows, and the mount's adapter needs them high to listen.
        port.write_data_terminal_ready(true)?;
        port.write_request_to_send(true)?;
        let trace = std::env::var("SKYTRACKER_SERIAL_TRACE").ok().and_then(|p| {
            std::fs::File::create(&p).ok().map(|mut f| {
                let _ = writeln!(f, "open {path} @ {baud} 8N1, timeout {timeout_ms} ms, DTR+RTS asserted");
                (f, 0u32, std::time::Instant::now())
            })
        });
        Ok(SerialTransport { port, trace })
    }

    fn trace_line(&mut self, line: &str) {
        if let Some((f, n, t0)) = self.trace.as_mut() {
            if *n < TRACE_MAX_LINES {
                let _ = writeln!(f, "{:9.3} {line}", t0.elapsed().as_secs_f64());
                *n += 1;
                if *n == TRACE_MAX_LINES {
                    let _ = writeln!(f, "(trace cap reached)");
                }
            }
        }
    }
}

impl Transport for SerialTransport {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        let r = self.port.write_all(data);
        match &r {
            Ok(()) => self.trace_line(&format!("TX {}", hex(data))),
            Err(e) => {
                let msg = format!("TX FAILED ({e}) {}", hex(data));
                self.trace_line(&msg);
            }
        }
        r
    }

    fn read(&mut self, n: usize) -> Vec<u8> {
        let mut buf = vec![0u8; n];
        let mut got = 0;
        while got < n {
            match self.port.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(_) => break,
            }
        }
        if got < n {
            // Short/timed-out read: flush so the next cycle starts clean.
            let _ = self.port.clear(serialport::ClearBuffer::Input);
            let msg = format!("RX SHORT {got}/{n} {}", hex(&buf[..got]));
            self.trace_line(&msg);
        } else {
            let msg = format!("RX {}", hex(&buf[..got]));
            self.trace_line(&msg);
        }
        buf.truncate(got);
        buf
    }
}
