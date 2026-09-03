//! tetra3 pattern-database loading (.npz, schema-checked). The database
//! file is used AS-IS — no migration — so the app's existing generated
//! databases (db_cam1_tyc etc.) keep working.

use crate::npy::{self, NpyError};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct DbProps {
    pub pattern_mode: String,
    pub pattern_size: usize,
    pub pattern_bins: u64,
    pub pattern_max_error: f64,
    pub max_fov_deg: f64,
    pub min_fov_deg: f64,
    pub pattern_stars_per_fov: usize,
    pub verification_stars_per_fov: usize,
    pub presort_patterns: bool,
    pub epoch_equinox: u16,
    pub epoch_proper_motion: f32,
}

pub struct SolverDatabase {
    /// (N, 6): ra rad, dec rad, x, y, z, magnitude — brightest first.
    pub star_table: Vec<[f32; 6]>,
    /// (M, 4): open-addressed hash table of star-index quadruples.
    pub pattern_catalog: Vec<[u32; 4]>,
    pub props: DbProps,
}

impl SolverDatabase {
    pub fn load(path: &Path) -> Result<Self, NpyError> {
        let mut members = npy::read_npz(path)?;

        let star_raw = members
            .remove("star_table")
            .ok_or_else(|| NpyError("no star_table".into()))?;
        if star_raw.shape.len() != 2 || star_raw.shape[1] != 6 {
            return Err(NpyError(format!("star_table shape {:?}", star_raw.shape)));
        }
        let flat = star_raw.as_f32()?;
        let star_table: Vec<[f32; 6]> = flat
            .chunks_exact(6)
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
            .collect();

        let pat_raw = members
            .remove("pattern_catalog")
            .ok_or_else(|| NpyError("no pattern_catalog".into()))?;
        if pat_raw.shape.len() != 2 || pat_raw.shape[1] != 4 {
            return Err(NpyError(format!(
                "pattern_catalog shape {:?}",
                pat_raw.shape
            )));
        }
        // Generated DBs store u32; the small bundled demo DBs may use u16.
        let pattern_catalog: Vec<[u32; 4]> = if pat_raw.descr.contains("<u4") {
            pat_raw
                .as_u32()?
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect()
        } else {
            pat_raw
                .as_u16()?
                .chunks_exact(4)
                .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                .collect()
        };

        let props_raw = members
            .remove("props_packed")
            .ok_or_else(|| NpyError("no props_packed".into()))?;
        let layout = npy::parse_record_descr(&props_raw.descr)?;
        if props_raw.data.len() < layout.itemsize {
            return Err(NpyError(format!(
                "props_packed too short: {} < {}",
                props_raw.data.len(),
                layout.itemsize
            )));
        }
        let d = &props_raw.data;
        let props = DbProps {
            pattern_mode: layout.get_ucs4(d, "pattern_mode")?,
            pattern_size: layout.get_u16(d, "pattern_size")? as usize,
            pattern_bins: layout.get_u16(d, "pattern_bins")? as u64,
            pattern_max_error: layout.get_f32(d, "pattern_max_error")? as f64,
            max_fov_deg: layout.get_f32(d, "max_fov")? as f64,
            min_fov_deg: layout.get_f32(d, "min_fov")? as f64,
            pattern_stars_per_fov: layout.get_u16(d, "pattern_stars_per_fov")? as usize,
            verification_stars_per_fov: layout.get_u16(d, "verification_stars_per_fov")? as usize,
            presort_patterns: layout.get_bool(d, "presort_patterns")?,
            epoch_equinox: layout.get_u16(d, "epoch_equinox")?,
            epoch_proper_motion: layout.get_f32(d, "epoch_proper_motion")?,
        };
        if props.pattern_mode != "edge_ratio" {
            return Err(NpyError(format!(
                "unsupported pattern_mode {:?} (only edge_ratio)",
                props.pattern_mode
            )));
        }
        if props.pattern_size != 4 {
            return Err(NpyError(format!(
                "unsupported pattern_size {} (only 4)",
                props.pattern_size
            )));
        }

        Ok(SolverDatabase {
            star_table,
            pattern_catalog,
            props,
        })
    }
}
