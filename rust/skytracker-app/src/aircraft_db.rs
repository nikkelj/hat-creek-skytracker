//! Airframe registry lookups: enrich ADS-B hexes with registration, type,
//! and owner from adsbdb.com (free, keyless). One lookup at a time, ~2 s
//! apart, only for hexes currently being received; results (including
//! negatives) persist in `<captures_dir>/aircraft_db.json` so repeat
//! sightings and offline field sessions cost nothing.

use crate::state::{AircraftInfo, Shared};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

const BASE_URL: &str = "https://api.adsbdb.com/v0/aircraft";

pub fn spawn(shared: Arc<Shared>, root: std::path::PathBuf) {
    std::thread::Builder::new()
        .name("aircraft-db".into())
        .spawn(move || run(shared, root))
        .expect("spawn aircraft-db worker");
}

fn run(shared: Arc<Shared>, root: std::path::PathBuf) {
    let cache_path = root.join(&shared.config.captures_dir).join("aircraft_db.json");
    let mut map: HashMap<String, Arc<AircraftInfo>> = std::fs::read_to_string(&cache_path)
        .ok()
        .and_then(|t| serde_json::from_str::<HashMap<String, AircraftInfo>>(&t).ok())
        .map(|m| m.into_iter().map(|(k, v)| (k, Arc::new(v))).collect())
        .unwrap_or_default();
    if !map.is_empty() {
        shared.aircraft_info.store(Arc::new(map.clone()));
    }
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(15)).build();

    loop {
        // One un-looked-up hex from the aircraft currently in view.
        let want: Option<String> = {
            let adsb = shared.adsb.load();
            adsb.aircraft
                .iter()
                .map(|a| a.icao.to_ascii_lowercase())
                .find(|h| h.len() == 6 && !map.contains_key(h))
        };
        let Some(hex) = want else {
            std::thread::sleep(Duration::from_secs(2));
            continue;
        };
        let info = match agent.get(&format!("{BASE_URL}/{}", hex.to_ascii_uppercase())).call() {
            Ok(resp) => resp
                .into_string()
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .map(|v| {
                    let a = &v["response"]["aircraft"];
                    let s = |k: &str| a[k].as_str().unwrap_or("").to_string();
                    AircraftInfo {
                        registration: s("registration"),
                        icao_type: s("icao_type"),
                        type_name: s("type"),
                        manufacturer: s("manufacturer"),
                        owner: s("registered_owner"),
                        country: s("registered_owner_country_name"),
                        known: a.is_object(),
                        fetched_unix: crate::sky::now_unix(),
                    }
                })
                .unwrap_or_else(|| AircraftInfo { fetched_unix: crate::sky::now_unix(), ..Default::default() }),
            // 404 = not in the registry: negative-cache it. Transport errors:
            // back off without caching so a dead network doesn't poison the DB.
            Err(ureq::Error::Status(_, _)) => AircraftInfo { fetched_unix: crate::sky::now_unix(), ..Default::default() },
            Err(_) => {
                std::thread::sleep(Duration::from_secs(30));
                continue;
            }
        };
        map.insert(hex, Arc::new(info));
        shared.aircraft_info.store(Arc::new(map.clone()));
        let plain: HashMap<&String, &AircraftInfo> = map.iter().map(|(k, v)| (k, v.as_ref())).collect();
        if let Ok(text) = serde_json::to_string_pretty(&plain) {
            let _ = std::fs::create_dir_all(cache_path.parent().unwrap());
            let _ = std::fs::write(&cache_path, text);
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}
