//! Tycho deep catalogue for the simulated camera (port of
//! star_catalog.DeepStarCatalog): narrow-FOV frames need mag 10-11 stars,
//! which Hipparcos doesn't carry. tyc_main.dat is pipe-delimited; col 5 =
//! VT mag, 8 = RA deg, 9 = Dec deg. Cone-mask around the boresight with a
//! dot product, then convert only that subset to alt/az.

use skytracker_astro::apparent::FrameContext;
use skytracker_astro::sgp4_pass::ObserverGeometry;
use std::io::BufRead;
use std::path::Path;

pub struct DeepCatalog {
    pub xyz: Vec<[f32; 3]>,
    pub mag: Vec<f32>,
}

pub struct DeepStar {
    pub az: f64,
    pub el: f64,
    pub mag: f64,
}

impl DeepCatalog {
    pub fn load(path: &Path, mag_limit: f64) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let reader = std::io::BufReader::with_capacity(1 << 20, f);
        let mut xyz = Vec::with_capacity(400_000);
        let mut mag = Vec::with_capacity(400_000);
        for line in reader.split(b'\n') {
            let line = line?;
            let mut it = line.split(|&b| b == b'|');
            let f5 = it.by_ref().nth(5);
            let f8 = it.by_ref().nth(2);
            let f9 = it.next();
            let (Some(m), Some(r), Some(d)) = (f5, f8, f9) else { continue };
            let parse = |b: &[u8]| std::str::from_utf8(b).ok()?.trim().parse::<f64>().ok();
            let (Some(m), Some(ra), Some(dec)) = (parse(m), parse(r), parse(d)) else { continue };
            if m > mag_limit {
                continue;
            }
            let (sr, cr) = ra.to_radians().sin_cos();
            let (sd, cd) = dec.to_radians().sin_cos();
            xyz.push([(cd * cr) as f32, (cd * sr) as f32, sd as f32]);
            mag.push(m as f32);
        }
        Ok(DeepCatalog { xyz, mag })
    }

    pub fn len(&self) -> usize {
        self.mag.len()
    }

    /// Stars within `half_cone_deg` of the boresight (az/el), as topocentric
    /// az/el/mag at `jd_tt`. Uses the ICRS->ITRS->ENU chain (precession,
    /// nutation, GAST); proper motion and aberration are omitted (sim use).
    pub fn stars_near(
        &self,
        jd_tt: f64,
        geom: &ObserverGeometry,
        az_deg: f64,
        el_deg: f64,
        half_cone_deg: f64,
        limiting_mag: f64,
        max_stars: usize,
    ) -> Vec<DeepStar> {
        let ctx = FrameContext::new(jd_tt);
        // Boresight ENU -> ITRS -> ICRS (inverse of altaz_from_icrs).
        let (sa, ca) = az_deg.to_radians().sin_cos();
        let (se, ce) = el_deg.to_radians().sin_cos();
        let (e, n, u) = (ce * sa, ce * ca, se);
        let d_itrs = [
            e * geom.east[0] + n * geom.north[0] + u * geom.up[0],
            e * geom.east[1] + n * geom.north[1] + u * geom.up[1],
            e * geom.east[2] + n * geom.north[2] + u * geom.up[2],
        ];
        let rz = skytracker_astro::apparent::rot_z(ctx.gast_rad);
        let mt = skytracker_astro::frames::transpose(&ctx.m);
        let b = skytracker_astro::frames::mat_vec(&mt, &skytracker_astro::frames::mat_vec(&rz, &d_itrs));
        let b = [b[0] as f32, b[1] as f32, b[2] as f32];
        let cos_lim = (half_cone_deg.to_radians().cos()) as f32;
        let mut idx: Vec<usize> = (0..self.xyz.len())
            .filter(|&i| {
                self.mag[i] <= limiting_mag as f32 && {
                    let p = self.xyz[i];
                    p[0] * b[0] + p[1] * b[1] + p[2] * b[2] >= cos_lim
                }
            })
            .collect();
        idx.sort_by(|&a, &c| self.mag[a].partial_cmp(&self.mag[c]).unwrap());
        idx.truncate(max_stars);
        idx.into_iter()
            .map(|i| {
                let p = self.xyz[i];
                let (alt, az) = ctx.altaz_from_icrs(&[p[0] as f64, p[1] as f64, p[2] as f64], geom);
                DeepStar { az, el: alt, mag: self.mag[i] as f64 }
            })
            .collect()
    }
}
