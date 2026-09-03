//! MP4 (H.264) export — replaces post_process.Mp4Exporter's
//! cv2.VideoWriter(mp4v) with openh264 (the sanctioned FFI exception) +
//! a pure-Rust MP4 muxer. Output upgrades from MPEG-4 Part 2 to H.264.
//!
//! Compiled only with the `mp4` feature.

use openh264::encoder::{Encoder, EncoderConfig};
use openh264::formats::YUVBuffer;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug)]
pub struct VideoError(pub String);

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "video: {}", self.0)
    }
}

impl std::error::Error for VideoError {}

fn err(e: impl std::fmt::Display) -> VideoError {
    VideoError(e.to_string())
}

const TIMESCALE: u32 = 90_000;

/// BT.601 full-range RGB -> packed I420 (Y then U then V planes).
fn rgb_to_i420(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let y_plane_len = w * h;
    let c_w = w / 2;
    let c_h = h / 2;
    let mut y_plane = vec![0u8; y_plane_len];
    let mut u_plane = vec![0u8; c_w * c_h];
    let mut v_plane = vec![0u8; c_w * c_h];
    // Studio-swing (limited range) BT.601: unflagged H.264 is decoded as
    // limited range, so full-range encoding reads back with a systematic
    // contrast error (~24 dB ceiling regardless of bitrate).
    for yy in 0..h {
        for xx in 0..w {
            let i = (yy * w + xx) * 3;
            let (r, g, b) = (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32);
            let yv = 16.0 + (0.299 * r + 0.587 * g + 0.114 * b) * 219.0 / 255.0;
            y_plane[yy * w + xx] = yv.round().clamp(0.0, 255.0) as u8;
        }
    }
    for cy in 0..c_h {
        for cx in 0..c_w {
            // Average the 2x2 block for chroma.
            let (mut us, mut vs) = (0.0f32, 0.0f32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let yy = (cy * 2 + dy).min(h - 1);
                    let xx = (cx * 2 + dx).min(w - 1);
                    let i = (yy * w + xx) * 3;
                    let (r, g, b) = (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32);
                    us += (-0.169 * r - 0.331 * g + 0.5 * b) * 224.0 / 255.0 + 128.0;
                    vs += (0.5 * r - 0.419 * g - 0.081 * b) * 224.0 / 255.0 + 128.0;
                }
            }
            u_plane[cy * c_w + cx] = (us / 4.0).round().clamp(0.0, 255.0) as u8;
            v_plane[cy * c_w + cx] = (vs / 4.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let mut packed = y_plane;
    packed.extend_from_slice(&u_plane);
    packed.extend_from_slice(&v_plane);
    packed
}

/// Split an Annex B bitstream into raw NAL units (no start codes).
fn split_nals(annex_b: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut i = 0;
    let mut start: Option<usize> = None;
    while i + 2 < annex_b.len() {
        if annex_b[i] == 0 && annex_b[i + 1] == 0 && annex_b[i + 2] == 1 {
            if let Some(s) = start {
                let mut end = i;
                if end > s && annex_b[end - 1] == 0 {
                    end -= 1; // 4-byte start code
                }
                nals.push(&annex_b[s..end]);
            }
            start = Some(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    if let Some(s) = start {
        nals.push(&annex_b[s..]);
    }
    nals
}

pub struct Mp4Encoder {
    encoder: Encoder,
    writer: Option<mp4::Mp4Writer<BufWriter<File>>>,
    out_path: std::path::PathBuf,
    width: usize,
    height: usize,
    fps: f64,
    track_id: Option<u32>,
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
    frames_written: u64,
}

impl Mp4Encoder {
    pub fn create(path: &Path, width: usize, height: usize, fps: f64) -> Result<Self, VideoError> {
        if width % 2 != 0 || height % 2 != 0 {
            return Err(VideoError("width/height must be even for I420".into()));
        }
        // Offline export wants constant quality, not rate control: the
        // bitrate controller holds its startup QP for a whole (short) GOP,
        // crushing noisy astro footage regardless of the target. Buffer-
        // based mode adjusts to content quality; the generous target keeps
        // it from starving (~0.3 bits/px/frame).
        let bps = ((width * height) as f64 * fps * 0.3) as u32;
        let config = EncoderConfig::new()
            .set_bitrate_bps(bps.clamp(1_000_000, 60_000_000))
            .max_frame_rate(fps as f32)
            .rate_control_mode(openh264::encoder::RateControlMode::Bufferbased);
        let encoder = Encoder::with_api_config(openh264::OpenH264API::from_source(), config)
            .map_err(err)?;
        let file = File::create(path).map_err(err)?;
        let writer = mp4::Mp4Writer::write_start(
            BufWriter::new(file),
            &mp4::Mp4Config {
                major_brand: (*b"isom").into(),
                minor_version: 512,
                compatible_brands: vec![
                    (*b"isom").into(),
                    (*b"iso2").into(),
                    (*b"avc1").into(),
                    (*b"mp41").into(),
                ],
                timescale: TIMESCALE,
            },
        )
        .map_err(err)?;
        Ok(Mp4Encoder {
            encoder,
            writer: Some(writer),
            out_path: path.to_path_buf(),
            width,
            height,
            fps,
            track_id: None,
            sps: None,
            pps: None,
            frames_written: 0,
        })
    }

    pub fn write_rgb(&mut self, rgb: &[u8]) -> Result<(), VideoError> {
        if rgb.len() != self.width * self.height * 3 {
            return Err(VideoError(format!(
                "frame size {} != {}x{}x3",
                rgb.len(),
                self.width,
                self.height
            )));
        }
        let packed = rgb_to_i420(rgb, self.width, self.height);
        let yuv = YUVBuffer::from_vec(packed, self.width, self.height);
        let bitstream = self.encoder.encode(&yuv).map_err(err)?;
        let annex_b = bitstream.to_vec();
        let nals = split_nals(&annex_b);

        let mut sample_data = Vec::new();
        let mut is_sync = false;
        for nal in &nals {
            if nal.is_empty() {
                continue;
            }
            match nal[0] & 0x1f {
                7 => {
                    self.sps.get_or_insert_with(|| nal.to_vec());
                    continue;
                }
                8 => {
                    self.pps.get_or_insert_with(|| nal.to_vec());
                    continue;
                }
                5 => is_sync = true,
                _ => {}
            }
            sample_data.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            sample_data.extend_from_slice(nal);
        }

        let writer = self.writer.as_mut().ok_or_else(|| VideoError("finished".into()))?;
        if self.track_id.is_none() {
            let (Some(sps), Some(pps)) = (&self.sps, &self.pps) else {
                return Err(VideoError("no SPS/PPS in first encoded frame".into()));
            };
            writer
                .add_track(&mp4::TrackConfig {
                    track_type: mp4::TrackType::Video,
                    timescale: TIMESCALE,
                    language: "und".to_string(),
                    media_conf: mp4::MediaConfig::AvcConfig(mp4::AvcConfig {
                        width: self.width as u16,
                        height: self.height as u16,
                        seq_param_set: sps.clone(),
                        pic_param_set: pps.clone(),
                    }),
                })
                .map_err(err)?;
            self.track_id = Some(1);
        }

        let duration = (TIMESCALE as f64 / self.fps).round() as u32;
        writer
            .write_sample(
                self.track_id.unwrap(),
                &mp4::Mp4Sample {
                    start_time: self.frames_written * duration as u64,
                    duration,
                    rendering_offset: 0,
                    is_sync,
                    bytes: mp4::Bytes::from(sample_data),
                },
            )
            .map_err(err)?;
        self.frames_written += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<u64, VideoError> {
        if let Some(mut writer) = self.writer.take() {
            writer.write_end().map_err(err)?;
        }
        let _ = &self.out_path;
        Ok(self.frames_written)
    }
}
