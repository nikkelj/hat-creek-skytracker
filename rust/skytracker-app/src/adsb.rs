//! ADS-B worker: aircraft from the RTL-SDR / Nooelec dongle (config
//! `adsb_source_mode` = "rtlsdr": librtlsdr via libloading at 2 MS/s on
//! 1090 MHz + the pyModeS-equivalent demodulator in skytracker-adsb), a
//! dump1090 SBS/BaseStation TCP feed ("dump1090", host/port) or a built-in
//! simulator ("sim"). Port of adsb_receiver.AdsbTracker's essentials: per-ICAO
//! fix history, linear ENU fit over the last N fixes for prediction, stale
//! pruning, topocentric az/el/range through skytracker-adsb's geometry.

use crate::state::{AdsbSnapshot, AircraftMark, Shared};
use skytracker_adsb::demod::{magnitudes_u8, Demodulator};
use skytracker_adsb::geom::{ecef_to_enu, enu_to_azel_range, geodetic_to_ecef};
use skytracker_adsb::modes::{decode_message, Decoded};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Aircraft {
    callsign: String,
    /// (unix s, lat, lon, alt_m)
    history: VecDeque<(f64, f64, f64, f64)>,
    last_seen: Instant,
    alt_m: Option<f64>,
    speed_kt: Option<f64>,
    track_deg: Option<f64>,
}

impl Aircraft {
    fn new() -> Self {
        Aircraft { callsign: String::new(), history: VecDeque::new(), last_seen: Instant::now(), alt_m: None, speed_kt: None, track_deg: None }
    }
}

pub fn spawn(shared: Arc<Shared>) {
    std::thread::Builder::new().name("adsb".into()).spawn(move || run(shared)).expect("spawn adsb worker");
}

/// Least-squares slope/intercept of y(t).
fn linfit(ts: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = ts.len() as f64;
    if ts.len() < 2 {
        return (0.0, ys.first().copied().unwrap_or(0.0));
    }
    let mt = ts.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxx: f64 = ts.iter().map(|t| (t - mt) * (t - mt)).sum();
    let sxy: f64 = ts.iter().zip(ys).map(|(t, y)| (t - mt) * (y - my)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (slope, my - slope * mt)
}

fn run(shared: Arc<Shared>) {
    let cfg = shared.config.clone();
    let mode = std::env::var("SKYTRACKER_ADSB").unwrap_or_else(|_| cfg.adsb_source_mode.clone()).to_ascii_lowercase();
    let (tx, rx) = crossbeam_channel::unbounded::<Msg>();
    let mut status = match mode.as_str() {
        "dump1090" | "sbs" => {
            let addr = format!("{}:{}", cfg.adsb_host, cfg.adsb_port);
            spawn_sbs_reader(addr.clone(), tx.clone());
            format!("dump1090 SBS {addr}")
        }
        "sim" => {
            spawn_sim_source(cfg.lat_deg, cfg.lon_deg, tx.clone());
            "simulated traffic".to_string()
        }
        "rtlsdr" => {
            spawn_rtlsdr_source(cfg.clone(), tx.clone());
            "RTL-SDR 1090 MHz".to_string()
        }
        "off" | "" | "none" => "off".to_string(),
        other => format!("{other}: unknown adsb_source_mode (rtlsdr | dump1090 | sim | off)"),
    };
    let mut aircraft: HashMap<String, Aircraft> = HashMap::new();
    let mut mode = mode;
    let mut tx = tx;
    let mut rx = rx;
    let (ox, oy, oz) = geodetic_to_ecef(cfg.lat_deg, cfg.lon_deg, cfg.alt_m);
    let mut last_pub = Instant::now() - Duration::from_secs(10);
    let stale = Duration::from_secs_f64(cfg.adsb_stale_timeout_s.max(5.0));
    let mut n_msgs: u64 = 0;
    loop {
        // Runtime source switch (UI): drop the old channel (source threads
        // exit on send error) and start the requested source.
        if let Some(req) = shared.adsb_request.swap(Arc::new(None)).as_ref().clone() {
            let (ntx, nrx) = crossbeam_channel::unbounded::<Msg>();
            tx = ntx;
            rx = nrx;
            mode = req.to_ascii_lowercase();
            aircraft.clear();
            status = match mode.as_str() {
                "dump1090" | "sbs" => {
                    let addr = format!("{}:{}", cfg.adsb_host, cfg.adsb_port);
                    spawn_sbs_reader(addr.clone(), tx.clone());
                    format!("dump1090 SBS {addr}")
                }
                "sim" => {
                    spawn_sim_source(cfg.lat_deg, cfg.lon_deg, tx.clone());
                    "simulated traffic".to_string()
                }
                "rtlsdr" => {
                    spawn_rtlsdr_source(cfg.clone(), tx.clone());
                    "RTL-SDR 1090 MHz".to_string()
                }
                _ => "off".to_string(),
            };
            n_msgs = 0;
        }
        // Drain SBS lines.
        let mut got = false;
        while let Ok(m) = rx.try_recv() {
            got = true;
            match m {
                Msg::Status(s) => status = s,
                Msg::Line(line) => {
                    n_msgs += 1;
                    parse_sbs(&line, &mut aircraft);
                }
                Msg::Dec(d) => {
                    n_msgs += 1;
                    ingest_decoded(d, &mut aircraft);
                }
            }
        }
        if !got {
            std::thread::sleep(Duration::from_millis(50));
        }
        if last_pub.elapsed() < Duration::from_millis(250) {
            continue;
        }
        last_pub = Instant::now();
        aircraft.retain(|_, a| a.last_seen.elapsed() < stale);
        let now = crate::sky::now_unix();
        let n_fit = shared.adsb_fit_points.load(std::sync::atomic::Ordering::Relaxed).max(2);
        let mut marks = Vec::new();
        for (icao, a) in &aircraft {
            if a.history.is_empty() {
                continue;
            }
            let enu: Vec<(f64, f64, f64, f64)> = a
                .history
                .iter()
                .map(|&(t, lat, lon, alt)| {
                    let (x, y, z) = geodetic_to_ecef(lat, lon, alt);
                    let (e, n, u) = ecef_to_enu(x - ox, y - oy, z - oz, cfg.lat_deg, cfg.lon_deg);
                    (t, e, n, u)
                })
                .collect();
            let last = *enu.last().unwrap();
            let (az, el, rng_km) = enu_to_azel_range(last.1, last.2, last.3);
            // Linear fit over the last N fixes -> prediction + rates.
            let fit = &enu[enu.len().saturating_sub(n_fit)..];
            let t0 = fit.last().unwrap().0;
            let ts: Vec<f64> = fit.iter().map(|p| p.0 - t0).collect();
            let (ve, e0) = linfit(&ts, &fit.iter().map(|p| p.1).collect::<Vec<_>>());
            let (vn, n0) = linfit(&ts, &fit.iter().map(|p| p.2).collect::<Vec<_>>());
            let (vu, u0) = linfit(&ts, &fit.iter().map(|p| p.3).collect::<Vec<_>>());
            let at = |dt: f64| enu_to_azel_range(e0 + ve * dt, n0 + vn * dt, u0 + vu * dt);
            let (az0, el0, _) = at(0.0);
            let (az1, el1, _) = at(1.0);
            let az_rate = (az1 - az0 + 540.0).rem_euclid(360.0) - 180.0;
            let el_rate = el1 - el0;
            let mut predicted = Vec::new();
            if fit.len() >= 2 {
                let mut dt = cfg.adsb_predict_step_s.max(0.5);
                while dt <= cfg.adsb_predict_horizon_s + 1e-6 {
                    let (paz, pel, _) = at(dt);
                    predicted.push((t0 + dt, paz, pel));
                    dt += cfg.adsb_predict_step_s.max(0.5);
                }
            }
            marks.push(AircraftMark {
                icao: icao.clone(),
                label: if a.callsign.trim().is_empty() { icao.to_uppercase() } else { a.callsign.trim().to_string() },
                az,
                el,
                range_km: rng_km,
                alt_m: a.alt_m.unwrap_or(last.3),
                speed_kt: a.speed_kt,
                track_deg: a.track_deg,
                t_fix_unix: last.0,
                age_s: now - last.0,
                fit_az: az0,
                fit_el: el0,
                fit_t_unix: t0,
                az_rate,
                el_rate,
                predicted,
                history: enu.iter().map(|p| {
                    let (a, e, _) = enu_to_azel_range(p.1, p.2, p.3);
                    (p.0, a, e)
                }).collect(),
            });
        }
        marks.sort_by(|a, b| b.el.partial_cmp(&a.el).unwrap());
        shared.adsb.store(Arc::new(AdsbSnapshot { status: status.clone(), n_aircraft: marks.len(), n_msgs, aircraft: marks, mode: mode.clone() }));
    }
}

/// SBS-1 CSV: MSG,type,...,HexIdent(4),...,Callsign(10),Altitude(11),
/// GroundSpeed(12),Track(13),Lat(14),Lon(15),...
fn parse_sbs(line: &str, aircraft: &mut HashMap<String, Aircraft>) {
    if !line.starts_with("MSG") {
        return;
    }
    let f: Vec<&str> = line.split(',').collect();
    if f.len() < 16 {
        return;
    }
    let icao = f[4].trim().to_ascii_lowercase();
    if icao.is_empty() {
        return;
    }
    let a = aircraft.entry(icao).or_insert_with(Aircraft::new);
    a.last_seen = Instant::now();
    let cs = f[10].trim();
    if !cs.is_empty() {
        a.callsign = cs.to_string();
    }
    if let Ok(v) = f[12].trim().parse::<f64>() {
        a.speed_kt = Some(v);
    }
    if let Ok(v) = f[13].trim().parse::<f64>() {
        a.track_deg = Some(v);
    }
    if let (Ok(lat), Ok(lon)) = (f[14].trim().parse::<f64>(), f[15].trim().parse::<f64>()) {
        let alt_m = f[11].trim().parse::<f64>().ok().map(|ft| ft * 0.3048).or(a.alt_m);
        if let Some(alt) = alt_m {
            a.alt_m = Some(alt);
        }
        let now = crate::sky::now_unix();
        a.history.push_back((now, lat, lon, alt_m.unwrap_or(0.0)));
        while a.history.len() > 240 {
            a.history.pop_front();
        }
    }
}

fn spawn_sbs_reader(addr: String, tx: crossbeam_channel::Sender<Msg>) {
    std::thread::Builder::new()
        .name("adsb-sbs".into())
        .spawn(move || loop {
            match std::net::TcpStream::connect_timeout(&addr.parse().unwrap_or_else(|_| "127.0.0.1:30003".parse().unwrap()), Duration::from_secs(3)) {
                Ok(stream) => {
                    let _ = tx.send(Msg::Status(format!("connected to {addr}")));
                    let reader = BufReader::new(stream);
                    for line in reader.lines() {
                        match line {
                            Ok(l) => {
                                if tx.send(Msg::Line(l)).is_err() {
                                    return;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    let _ = tx.send(Msg::Status(format!("lost {addr}, reconnecting")));
                }
                Err(e) => {
                    if tx.send(Msg::Status(format!("{addr}: {e} (retrying)"))).is_err() {
                        return;
                    }
                }
            }
            std::thread::sleep(Duration::from_secs(3));
        })
        .ok();
}

/// Two simulated airliners crossing the sky at 10–11 km, emitted as SBS lines.
fn spawn_sim_source(lat0: f64, lon0: f64, tx: crossbeam_channel::Sender<Msg>) {
    std::thread::Builder::new()
        .name("adsb-sim".into())
        .spawn(move || {
            let t_start = Instant::now();
            // (icao, callsign, heading deg, speed m/s, alt m, start offset east/north km)
            let planes = [
                ("a1b2c3", "UAL1234", 75.0f64, 240.0, 10_700.0, (-60.0, -20.0)),
                ("3c6444", "DLH402", 200.0f64, 230.0, 11_300.0, (25.0, 70.0)),
                ("abc123", "N73KX", 320.0f64, 95.0, 2_400.0, (30.0, -25.0)),
            ];
            loop {
                let t = t_start.elapsed().as_secs_f64();
                for (icao, cs, hdg, spd, alt, (e0, n0)) in planes {
                    // Wrap every 600 s so the traffic keeps coming back.
                    let tt = t % 600.0;
                    let e_km = e0 + spd * tt / 1000.0 * hdg.to_radians().sin();
                    let n_km = n0 + spd * tt / 1000.0 * hdg.to_radians().cos();
                    let lat = lat0 + n_km / 111.32;
                    let lon = lon0 + e_km / (111.32 * lat0.to_radians().cos());
                    let alt_ft = alt / 0.3048;
                    if tx
                        .send(Msg::Line(format!(
                            "MSG,3,1,1,{},1,,,,,{},{:.0},{:.0},{:.0},{:.6},{:.6},,,,,,",
                            icao.to_uppercase(), cs, alt_ft, spd * 1.94384, hdg, lat, lon
                        )))
                        .is_err()
                    {
                        return;
                    }
                }
                std::thread::sleep(Duration::from_millis(1000));
            }
        })
        .ok();
}

/// Events from a source thread to the tracker.
enum Msg {
    Status(String),
    /// SBS/BaseStation line (dump1090, sim).
    Line(String),
    /// Locally demodulated + decoded Mode S message (RTL-SDR).
    Dec(Decoded),
}

fn ingest_decoded(d: Decoded, aircraft: &mut HashMap<String, Aircraft>) {
    let now = crate::sky::now_unix();
    match d {
        Decoded::Ident { icao, callsign } => {
            let a = aircraft.entry(icao.to_ascii_lowercase()).or_insert_with(Aircraft::new);
            a.last_seen = Instant::now();
            if !callsign.is_empty() {
                a.callsign = callsign;
            }
        }
        Decoded::Position { icao, lat, lon, alt_m } => {
            let a = aircraft.entry(icao.to_ascii_lowercase()).or_insert_with(Aircraft::new);
            a.last_seen = Instant::now();
            if let Some(alt) = alt_m {
                a.alt_m = Some(alt);
            }
            a.history.push_back((now, lat, lon, a.alt_m.unwrap_or(0.0)));
            while a.history.len() > 240 {
                a.history.pop_front();
            }
        }
        Decoded::Velocity { icao, speed_kt, track_deg, .. } => {
            let a = aircraft.entry(icao.to_ascii_lowercase()).or_insert_with(Aircraft::new);
            a.last_seen = Instant::now();
            if speed_kt.is_some() {
                a.speed_kt = speed_kt;
            }
            if track_deg.is_some() {
                a.track_deg = track_deg;
            }
        }
    }
}

/// librtlsdr through libloading (`rtlsdr.dll` ships in the repo root):
/// 2 MS/s at 1090 MHz, synchronous reads, pyModeS-equivalent demodulation.
struct RtlSdr {
    _lib: libloading::Library,
    dev: *mut std::ffi::c_void,
    read_sync: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u8, std::os::raw::c_int, *mut std::os::raw::c_int) -> std::os::raw::c_int,
    close: unsafe extern "C" fn(*mut std::ffi::c_void) -> std::os::raw::c_int,
}
unsafe impl Send for RtlSdr {}

impl RtlSdr {
    fn open(dll: &str, index: u32, gain: &str) -> Result<Self, String> {
        use std::os::raw::{c_int, c_uint};
        type Dev = *mut std::ffi::c_void;
        unsafe {
            let lib = libloading::Library::new(dll).map_err(|e| format!("{dll}: {e}"))?;
            let get_count: libloading::Symbol<unsafe extern "C" fn() -> c_uint> = lib.get(b"rtlsdr_get_device_count\0").map_err(|e| e.to_string())?;
            let n = get_count();
            if index >= n {
                return Err(format!("device {index} not found ({n} RTL-SDR device(s))"));
            }
            let open: libloading::Symbol<unsafe extern "C" fn(*mut Dev, c_uint) -> c_int> = lib.get(b"rtlsdr_open\0").map_err(|e| e.to_string())?;
            let mut dev: Dev = std::ptr::null_mut();
            let r = open(&mut dev, index);
            if r != 0 || dev.is_null() {
                return Err(format!("rtlsdr_open failed ({r}) -- device in use by another program?"));
            }
            let set_rate: libloading::Symbol<unsafe extern "C" fn(Dev, c_uint) -> c_int> = lib.get(b"rtlsdr_set_sample_rate\0").map_err(|e| e.to_string())?;
            let set_freq: libloading::Symbol<unsafe extern "C" fn(Dev, c_uint) -> c_int> = lib.get(b"rtlsdr_set_center_freq\0").map_err(|e| e.to_string())?;
            let set_gain_mode: libloading::Symbol<unsafe extern "C" fn(Dev, c_int) -> c_int> = lib.get(b"rtlsdr_set_tuner_gain_mode\0").map_err(|e| e.to_string())?;
            let set_gain: libloading::Symbol<unsafe extern "C" fn(Dev, c_int) -> c_int> = lib.get(b"rtlsdr_set_tuner_gain\0").map_err(|e| e.to_string())?;
            let reset: libloading::Symbol<unsafe extern "C" fn(Dev) -> c_int> = lib.get(b"rtlsdr_reset_buffer\0").map_err(|e| e.to_string())?;
            set_rate(dev, 2_000_000);
            set_freq(dev, 1_090_000_000);
            match gain.trim().parse::<f64>() {
                Ok(db) => {
                    set_gain_mode(dev, 1);
                    set_gain(dev, (db * 10.0).round() as c_int);
                }
                Err(_) => {
                    set_gain_mode(dev, 0);
                }
            }
            reset(dev);
            let read_sync = *lib.get::<unsafe extern "C" fn(Dev, *mut u8, c_int, *mut c_int) -> c_int>(b"rtlsdr_read_sync\0").map_err(|e| e.to_string())?;
            let close = *lib.get::<unsafe extern "C" fn(Dev) -> c_int>(b"rtlsdr_close\0").map_err(|e| e.to_string())?;
            Ok(RtlSdr { _lib: lib, dev, read_sync, close })
        }
    }
    fn read(&self, buf: &mut [u8]) -> Result<usize, String> {
        let mut n: std::os::raw::c_int = 0;
        let r = unsafe { (self.read_sync)(self.dev, buf.as_mut_ptr(), buf.len() as std::os::raw::c_int, &mut n) };
        if r != 0 {
            return Err(format!("rtlsdr_read_sync failed ({r})"));
        }
        Ok(n.max(0) as usize)
    }
}

impl Drop for RtlSdr {
    fn drop(&mut self) {
        unsafe {
            (self.close)(self.dev);
        }
    }
}

fn spawn_rtlsdr_source(cfg: crate::state::Config, tx: crossbeam_channel::Sender<Msg>) {
    std::thread::Builder::new()
        .name("adsb-rtlsdr".into())
        .spawn(move || {
            let root = cfg.repo_root();
            let dll = if std::path::Path::new(&cfg.adsb_dll).is_absolute() { cfg.adsb_dll.clone() } else { root.join(&cfg.adsb_dll).to_string_lossy().to_string() };
            let mut failures = 0;
            loop {
                let sdr = match RtlSdr::open(&dll, cfg.adsb_device_index as u32, &cfg.adsb_gain) {
                    Ok(s) => s,
                    Err(e) => {
                        failures += 1;
                        let _ = tx.send(Msg::Status(format!("rtlsdr: {e}{}", if failures >= 3 { " (giving up; fix and restart)" } else { " (retrying)" })));
                        if failures >= 3 {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(1500));
                        continue;
                    }
                };
                let _ = tx.send(Msg::Status("rtlsdr: receiving 1090 MHz".into()));
                let mut demod = Demodulator::default();
                let mut buf = vec![0u8; 1024 * 100 * 2];
                let mut n_frames: u64 = 0;
                let mut last_report = Instant::now();
                loop {
                    match sdr.read(&mut buf) {
                        Ok(n) => {
                            let mags = magnitudes_u8(&buf[..n]);
                            for hex in demod.push(&mags) {
                                n_frames += 1;
                                if let Some(d) = decode_message(&hex, cfg.lat_deg, cfg.lon_deg) {
                                    if tx.send(Msg::Dec(d)).is_err() {
                                        return;
                                    }
                                }
                            }
                            if last_report.elapsed() > Duration::from_secs(5) {
                                last_report = Instant::now();
                                if tx.send(Msg::Status(format!("rtlsdr: receiving 1090 MHz · {n_frames} Mode S frames · noise {:.3}", demod.noise_floor))).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Msg::Status(format!("rtlsdr: {e} (restarting)")));
                            break;
                        }
                    }
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        })
        .ok();
}
