//! PyO3 bindings for skytracker-adsb (Phase 5 seam).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use skytracker_adsb::{geom, modes};

/// Decode one raw Mode-S hex message; returns the same dict shapes as
/// adsb_receiver.decode_adsb_message, or None.
#[pyfunction]
fn adsb_decode_message<'py>(
    py: Python<'py>,
    msg: &str,
    ref_lat: f64,
    ref_lon: f64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let decoded = modes::decode_message(msg, ref_lat, ref_lon);
    let Some(decoded) = decoded else {
        return Ok(None);
    };
    let out = PyDict::new_bound(py);
    match decoded {
        modes::Decoded::Ident { icao, callsign } => {
            out.set_item("icao", icao)?;
            out.set_item("kind", "ident")?;
            out.set_item("callsign", callsign)?;
        }
        modes::Decoded::Position {
            icao,
            lat,
            lon,
            alt_m,
        } => {
            out.set_item("icao", icao)?;
            out.set_item("kind", "position")?;
            out.set_item("lat", lat)?;
            out.set_item("lon", lon)?;
            out.set_item("alt_m", alt_m)?;
        }
        modes::Decoded::Velocity {
            icao,
            speed_kt,
            track_deg,
            vert_rate_fpm,
        } => {
            out.set_item("icao", icao)?;
            out.set_item("kind", "velocity")?;
            out.set_item("speed_kt", speed_kt)?;
            out.set_item("track_deg", track_deg)?;
            out.set_item("vert_rate", vert_rate_fpm)?;
        }
    }
    Ok(Some(out))
}

/// Observer-relative (az_deg, el_deg, range_km) for a geodetic target.
#[pyfunction]
fn adsb_geodetic_to_azel_range(
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
    obs_lat: f64,
    obs_lon: f64,
    obs_alt_m: f64,
) -> (f64, f64, f64) {
    geom::geodetic_to_azel_range(lat_deg, lon_deg, alt_m, obs_lat, obs_lon, obs_alt_m)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adsb_decode_message, m)?)?;
    m.add_function(wrap_pyfunction!(adsb_geodetic_to_azel_range, m)?)?;
    m.add("ADSB_AVAILABLE", true)?;
    Ok(())
}
