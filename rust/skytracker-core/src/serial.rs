//! Real serial `Transport` for the mount (the hardware path).
//!
//! Mirrors `lib/auxstar.py`'s serial settings: 9600 8N1 with a short timeout so
//! a missing response can never stall the control loop. A short read clears the
//! input buffer (matching `_transact`'s `reset_input_buffer`) so the next
//! command starts on a clean stream. One owner per port (the loop thread), so
//! no locking is needed here.
//!
//! Compiled only with the `serial` feature (it depends on the
//! `serialport` crate); cannot be unit-tested without a physical port.

use std::time::Duration;

use crate::sim::Transport;

pub struct SerialTransport {
    port: Box<dyn serialport::SerialPort>,
}

impl SerialTransport {
    pub fn open(path: &str, baud: u32, timeout_ms: u64) -> serialport::Result<Self> {
        let port = serialport::new(path, baud)
            .timeout(Duration::from_millis(timeout_ms))
            .data_bits(serialport::DataBits::Eight)
            .parity(serialport::Parity::None)
            .stop_bits(serialport::StopBits::One)
            .flow_control(serialport::FlowControl::None)
            .open()?;
        Ok(SerialTransport { port })
    }
}

impl Transport for SerialTransport {
    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.port.write_all(data)
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
        }
        buf.truncate(got);
        buf
    }
}
