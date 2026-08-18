//! skytracker-adsb — ADS-B aircraft tracking engine (Phase 5 of the Rust
//! port): Mode-S DF17/18 extended-squitter decoding (the pyModeS subset
//! adsb_receiver.py uses, ported faithfully) and WGS84 topocentric
//! geometry. The SBS/dump1090 TCP source and the tracker orchestration
//! stay in Python during the strangler period; a native RTL-SDR demod
//! path arrives with the Phase 7 app.

pub mod geom;
pub mod modes;
