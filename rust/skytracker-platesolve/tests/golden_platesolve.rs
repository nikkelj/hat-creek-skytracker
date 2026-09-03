//! Golden-vector tests: hash keys bit-exact, centroids within 0.3 px.
//! (End-to-end solve parity vs live Python tetra3 runs in
//! test_rust_platesolve_parity.py, which can render solvable star fields.)

use skytracker_platesolve::centroid::{get_centroids, CentroidParams};
use skytracker_platesolve::hash::key_to_index;

use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn open(name: &str) -> npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>> {
    npyz::npz::NpzArchive::open(golden_dir().join(name)).unwrap()
}

#[test]
fn hash_keys_bit_exact() {
    let mut npz = open("tetra3_hashes.npz");
    let keys: Vec<u64> = npz
        .by_name("keys")
        .unwrap()
        .unwrap()
        .into_vec::<u64>()
        .unwrap();
    let read_u64 = |npz: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>,
                    name: &str| {
        npz.by_name(name)
            .unwrap()
            .unwrap()
            .into_vec::<u64>()
            .unwrap()
    };
    let bins = read_u64(&mut npz, "pattern_bins")[0];
    let catalog_len = read_u64(&mut npz, "catalog_length")[0];
    let indices = read_u64(&mut npz, "indices");
    let alt_bins = read_u64(&mut npz, "alt_bins")[0];
    let alt_max = read_u64(&mut npz, "alt_max_index")[0];
    let indices_alt = read_u64(&mut npz, "indices_alt");

    let n = indices.len();
    assert_eq!(keys.len(), n * 5);
    for i in 0..n {
        let key = &keys[i * 5..(i + 1) * 5];
        assert_eq!(
            key_to_index(key, bins, catalog_len),
            indices[i],
            "key {i} (live geometry)"
        );
        assert_eq!(
            key_to_index(key, alt_bins, alt_max),
            indices_alt[i],
            "key {i} (alt geometry)"
        );
    }
    println!("{n} hash keys bit-exact on both geometries");
}

#[test]
fn centroids_match_python() {
    let mut npz = open("tetra3_centroids.npz");
    let images = npz
        .by_name("images")
        .unwrap()
        .unwrap();
    let shape: Vec<usize> = images.shape().iter().map(|&d| d as usize).collect();
    let (n_img, h, w) = (shape[0], shape[1], shape[2]);
    let image_data: Vec<u8> = images.into_vec::<u8>().unwrap();
    let cents = npz
        .by_name("centroids_yx")
        .unwrap()
        .unwrap();
    let cshape: Vec<usize> = cents.shape().iter().map(|&d| d as usize).collect();
    let max_c = cshape[1];
    let cent_data: Vec<f64> = cents.into_vec::<f64>().unwrap();

    let mut worst = 0.0f64;
    for i in 0..n_img {
        let img = &image_data[i * h * w..(i + 1) * h * w];
        let got = get_centroids(img, h, w, &CentroidParams::default());
        // Golden rows: NaN-padded (y, x), brightest first.
        let golden: Vec<[f64; 2]> = (0..max_c)
            .map(|j| {
                [
                    cent_data[(i * max_c + j) * 2],
                    cent_data[(i * max_c + j) * 2 + 1],
                ]
            })
            .filter(|c| !c[0].is_nan())
            .collect();
        assert_eq!(got.len(), golden.len(), "image {i}: centroid count");
        for (k, (g, p)) in got.iter().zip(golden.iter()).enumerate() {
            let d = ((g[0] - p[0]).powi(2) + (g[1] - p[1]).powi(2)).sqrt();
            worst = worst.max(d);
            assert!(d < 0.3, "image {i} centroid {k}: {d:.3} px off");
        }
    }
    println!("centroids: {n_img} frames, worst {worst:.4} px");
}
