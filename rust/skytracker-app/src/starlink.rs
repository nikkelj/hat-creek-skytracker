//! Starlink public-ephemeris worker: fetches MANIFEST.txt once at startup
//! (11k filenames, one per satellite), then downloads individual ephemeris
//! files on demand when the user switches a selected satellite to
//! ephemeris tracking. Files are ~1.7 MB each — downloading all of them is
//! intractable, so only requested satellites are fetched. Downloads are
//! also cached under `<captures_dir>/ephemerides/` and re-fetched when the
//! manifest names a newer file.

use crate::state::Shared;
use skytracker_astro::starlink::{parse_ephemeris, parse_manifest};
use std::sync::Arc;
use std::time::Duration;

const BASE_URL: &str = "https://api.starlink.com/public-files/ephemerides/";

pub fn spawn(shared: Arc<Shared>, root: std::path::PathBuf) {
    std::thread::Builder::new()
        .name("starlink-ephem".into())
        .spawn(move || run(shared, root))
        .expect("spawn starlink worker");
}

fn run(shared: Arc<Shared>, root: std::path::PathBuf) {
    let base = shared
        .config
        .raw["starlink_ephemeris_url"]
        .as_str()
        .unwrap_or(BASE_URL)
        .trim_end_matches('/')
        .to_string();
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build();
    let cache_dir = root.join(&shared.config.captures_dir).join("ephemerides");

    // Manifest once at startup (retry a few times: the app often starts
    // before the network is up in the field).
    let mut manifest_tries = 0;
    loop {
        manifest_tries += 1;
        let fetched: Result<String, String> = agent
            .get(&format!("{base}/MANIFEST.txt"))
            .call()
            .map_err(|e| e.to_string())
            .and_then(|r| r.into_string().map_err(|e| e.to_string()));
        match fetched {
            Ok(text) => {
                let man = parse_manifest(&text);
                shared.ephem_status.store(Arc::new(format!("manifest: {} satellites", man.len())));
                shared.starlink_manifest.store(Arc::new(Some(Arc::new(man))));
                break;
            }
            Err(e) => {
                shared.ephem_status.store(Arc::new(format!("manifest fetch failed: {e}")));
                if manifest_tries >= 5 {
                    break;
                }
                std::thread::sleep(Duration::from_secs(30));
            }
        }
    }

    // Request loop: fetch + parse one ephemeris per request.
    loop {
        let req = (**shared.ephem_request.load()).clone();
        let Some(satnum) = req else {
            std::thread::sleep(Duration::from_millis(300));
            continue;
        };
        shared.ephem_request.store(Arc::new(None));
        let man = shared.starlink_manifest.load();
        let Some((file, name)) = man.as_ref().as_ref().and_then(|m| m.get(&satnum)).cloned() else {
            shared.ephem_status.store(Arc::new(format!("{satnum}: not in the Starlink manifest")));
            continue;
        };
        shared.ephem_status.store(Arc::new(format!("{name}: downloading ephemeris…")));
        let cached = cache_dir.join(&file);
        let text = if cached.exists() {
            std::fs::read_to_string(&cached).map_err(|e| e.to_string())
        } else {
            agent
                .get(&format!("{base}/{file}"))
                .call()
                .map_err(|e| e.to_string())
                .and_then(|r| r.into_string().map_err(|e| e.to_string()))
                .inspect(|t| {
                    // Cache; sweep older files for the same satellite.
                    let _ = std::fs::create_dir_all(&cache_dir);
                    let _ = std::fs::write(&cached, t);
                    if let Ok(rd) = std::fs::read_dir(&cache_dir) {
                        let prefix = format!("MEME_{satnum}_");
                        for e in rd.flatten() {
                            let n = e.file_name().to_string_lossy().to_string();
                            if n.starts_with(&prefix) && n != file {
                                let _ = std::fs::remove_file(e.path());
                            }
                        }
                    }
                })
        };
        match text.and_then(|t| parse_ephemeris(&t, &satnum, &name)) {
            Ok(eph) => {
                let span_h = (eph.stop_unix - eph.start_unix) / 3600.0;
                let n = eph.states.len();
                let mut map = (**shared.ephemerides.load()).clone();
                map.insert(satnum.clone(), Arc::new(eph));
                shared.ephemerides.store(Arc::new(map));
                shared.ephem_status.store(Arc::new(format!("{name}: ephemeris loaded ({n} states, {span_h:.0} h)")));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&cached);
                shared.ephem_status.store(Arc::new(format!("{name}: ephemeris failed: {e}")));
            }
        }
    }
}
