//! Mode S demodulation from 2 MS/s IQ magnitude (port of pyModeS
//! `extra/rtlreader.py`): noise floor from 100 us windows, 10 dB minimum
//! amplitude, the 8 us preamble pattern 1010000101000000 at 2 samples/us,
//! then 112 PPM bits decided by comparing each bit's two half-chips, framed
//! as hex and accepted when the DF/CRC rules hold (DF17 CRC == 0; DF20/21
//! 112-bit; DF4/5/11 56-bit).

use crate::modes;

pub const SAMPLES_PER_US: usize = 2;
pub const PREAMBLE_BITS: usize = 8;
pub const FRAME_BITS: usize = 112;
/// Amplitude tolerance between a pulse and the ideal 0/1 preamble pattern
/// (pyModeS th_amp_diff), for magnitudes normalised so a full-scale IQ
/// sample is ~1.0.
pub const PREAMBLE_TOLERANCE: f32 = 0.8;
const PREAMBLE: [f32; 16] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Stateful demodulator: accumulates magnitudes and yields hex messages.
pub struct Demodulator {
    pub buffer: Vec<f32>,
    pub noise_floor: f32,
    /// Process once this many samples are buffered (pyModeS buffer_size).
    pub buffer_size: usize,
}

impl Default for Demodulator {
    fn default() -> Self {
        Demodulator { buffer: Vec::with_capacity(1024 * 300), noise_floor: 1e6, buffer_size: 1024 * 200 }
    }
}

/// IQ u8 pairs (librtlsdr read_sync) -> magnitudes normalised to ~[0, 1.41].
pub fn magnitudes_u8(iq: &[u8]) -> Vec<f32> {
    iq.chunks_exact(2)
        .map(|p| {
            let i = (p[0] as f32 - 127.5) / 127.5;
            let q = (p[1] as f32 - 127.5) / 127.5;
            (i * i + q * q).sqrt()
        })
        .collect()
}

/// Minimum 100 us window mean = noise floor.
pub fn noise_floor(sig: &[f32]) -> f32 {
    let window = SAMPLES_PER_US * 100;
    sig.chunks_exact(window).map(|w| w.iter().sum::<f32>() / window as f32).fold(f32::INFINITY, f32::min)
}

fn check_preamble(p: &[f32]) -> bool {
    p.len() == 16 && p.iter().zip(PREAMBLE.iter()).all(|(a, b)| (a - b).abs() <= PREAMBLE_TOLERANCE)
}

fn check_msg(hex: &str) -> bool {
    let Some(df) = modes::df(hex) else { return false };
    match (df, hex.len()) {
        (17, 28) => modes::crc(hex) == Some(0),
        (20 | 21, 28) => true,
        (4 | 5 | 11, 14) => true,
        _ => false,
    }
}

/// Demodulate one magnitude buffer; returns accepted hex messages.
pub fn process(sig: &[f32], noise_floor: f32) -> Vec<String> {
    let min_sig_amp = 3.162 * noise_floor; // 10 dB SNR
    let mut out = Vec::new();
    let pre_len = PREAMBLE_BITS * SAMPLES_PER_US;
    let frame_len = (FRAME_BITS + 1) * SAMPLES_PER_US;
    let n = sig.len();
    let mut i = 0;
    while i < n {
        if sig[i] < min_sig_amp {
            i += 1;
            continue;
        }
        let frame_start = i + pre_len;
        if frame_start <= n && check_preamble(&sig[i..frame_start]) {
            let frame_end = (frame_start + frame_len).min(n);
            let pulses = &sig[frame_start..frame_end];
            let threshold = pulses.iter().cloned().fold(0.0f32, f32::max) * 0.2;
            let mut bits: Vec<u8> = Vec::with_capacity(FRAME_BITS);
            let mut j = 0;
            while j + 1 < pulses.len() {
                let (a, b) = (pulses[j], pulses[j + 1]);
                if a < threshold && b < threshold {
                    break;
                }
                bits.push(if a >= b { 1 } else { 0 });
                j += 2;
            }
            i = frame_start + j;
            if !bits.is_empty() {
                let hex = bits_to_hex(&bits);
                if check_msg(&hex) {
                    out.push(hex);
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

/// pyModeS bin2hex: pad to a multiple of 4 bits (left), hex upper-case.
pub fn bits_to_hex(bits: &[u8]) -> String {
    let mut s = String::with_capacity(bits.len() / 4 + 1);
    // Python: hex(int(bin, 2))[2:].zfill(len//4) — i.e. floor(len/4) digits;
    // trailing partial nibble dropped by truncation to whole frames (56/112).
    let usable = bits.len() / 4 * 4;
    for chunk in bits[..usable].chunks(4) {
        let v = chunk.iter().fold(0u8, |acc, &b| (acc << 1) | b);
        s.push(std::char::from_digit(v as u32, 16).unwrap().to_ascii_uppercase());
    }
    s
}

impl Demodulator {
    /// Push magnitudes; when the buffer is full, demodulate and drain it.
    pub fn push(&mut self, mags: &[f32]) -> Vec<String> {
        self.buffer.extend_from_slice(mags);
        if self.buffer.len() < self.buffer_size {
            return Vec::new();
        }
        self.noise_floor = noise_floor(&self.buffer).min(self.noise_floor);
        let msgs = process(&self.buffer, self.noise_floor);
        self.buffer.clear();
        msgs
    }
}

/// Synthesize the magnitude waveform of a Mode S frame (for tests / the
/// simulator): preamble + PPM bits at 2 samples/us, amplitude `amp`.
pub fn synth_frame(hex: &str, amp: f32, noise: f32, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / u32::MAX as f32 * 2.0) * noise
    };
    let mut sig: Vec<f32> = (0..400).map(|_| next()).collect();
    for &p in &PREAMBLE {
        sig.push(p * amp + next());
    }
    for c in hex.chars() {
        let v = c.to_digit(16).unwrap();
        for k in (0..4).rev() {
            let bit = (v >> k) & 1;
            if bit == 1 {
                sig.push(amp + next());
                sig.push(next());
            } else {
                sig.push(next());
                sig.push(amp + next());
            }
        }
    }
    sig.extend((0..400).map(|_| next()));
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    // DF17 airborne position message (pyModeS docs example) - CRC valid.
    const MSG: &str = "8D40621D58C382D690C8AC2863A7";

    #[test]
    fn hex_roundtrip() {
        let bits: Vec<u8> = MSG.chars().flat_map(|c| { let v = c.to_digit(16).unwrap() as u8; (0..4).rev().map(move |k| (v >> k) & 1) }).collect();
        assert_eq!(bits_to_hex(&bits), MSG);
    }

    #[test]
    fn demodulates_a_synthesized_frame() {
        let sig = synth_frame(MSG, 1.0, 0.02, 7);
        let nf = noise_floor(&sig);
        assert!(nf < 0.1, "noise floor {nf}");
        let msgs = process(&sig, nf);
        assert_eq!(msgs, vec![MSG.to_string()]);
    }

    #[test]
    fn streaming_demodulator_finds_frames_in_noise() {
        let mut d = Demodulator { buffer_size: 5000, ..Default::default() };
        let mut found = Vec::new();
        for seed in 0..5u64 {
            let sig = synth_frame(MSG, 0.8, 0.05, 11 + seed);
            found.extend(d.push(&sig));
        }
        assert!(found.iter().filter(|m| *m == MSG).count() >= 4, "{found:?}");
    }

    #[test]
    fn corrupted_frames_are_rejected() {
        let mut sig = synth_frame(MSG, 1.0, 0.02, 3);
        // Flip a data pulse pair in the middle of the frame.
        let k = 400 + 16 + 60 * 2;
        sig.swap(k, k + 1);
        let msgs = process(&sig, noise_floor(&sig));
        assert!(msgs.is_empty(), "{msgs:?}");
    }
}
