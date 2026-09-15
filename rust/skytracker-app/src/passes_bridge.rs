//! Bridge from skytracker-astro's pass prediction to the UI's PassRow.
//! The engine predicts over the whole catalog (rayon); the sky worker calls
//! this once a minute.

use crate::state::PassRow;
use skytracker_astro::engine::Engine;
use skytracker_astro::passes::PassParams;
use skytracker_astro::sgp4_pass::Observer;

pub const HORIZON_HOURS: f64 = 6.0;

pub fn compute(engine: &Engine, satnums: &[String], observer: &Observer, jd_tt: f64, mask_deg: f64) -> Vec<PassRow> {
    let params = PassParams {
        min_el_deg: mask_deg.max(0.0),
        horizon_hours: HORIZON_HOURS,
        coarse_step_s: 30.0,
        fine_step_s: 1.0,
        max_passes_per_sat: 3,
    };
    engine
        .predict_passes(satnums, observer, jd_tt, &params)
        .into_iter()
        .map(|p| PassRow {
            satnum: p.satnum,
            name: p.name,
            aos_unix: crate::sky::jd_tt_to_unix(p.aos_jd_tt),
            tca_unix: crate::sky::jd_tt_to_unix(p.tca_jd_tt),
            los_unix: crate::sky::jd_tt_to_unix(p.los_jd_tt),
            aos_az: p.aos_az,
            tca_az: p.tca_az,
            tca_el: p.tca_el,
            los_az: p.los_az,
            duration_s: p.duration_s,
            max_rate_dps: p.max_rate_dps,
            range_tca_km: p.range_tca_km,
            apogee_km: p.apogee_km,
            est_mag: p.est_mag,
        })
        .collect()
}
