//! tetra3's pattern-hash functions, ported bit-exactly (numpy uint64
//! arithmetic wraps on overflow; Rust `wrapping_*` reproduces it).
//! Validated against 1,000 golden keys from the live db_cam1_tyc geometry
//! (tests/golden/tetra3_hashes.npz).

pub const MAGIC_RAND: u64 = 2654435761;

/// `_key_to_index`: key (length-p bin indices) -> randomized table index.
pub fn key_to_index(key: &[u64], bin_factor: u64, max_index: u64) -> u64 {
    let mut sum: u64 = 0;
    let mut power: u64 = 1; // bin_factor^i, wrapping like np.uint64
    for &k in key {
        sum = sum.wrapping_add(k.wrapping_mul(power));
        power = power.wrapping_mul(bin_factor);
    }
    sum.wrapping_mul(MAGIC_RAND) % max_index
}

/// `_get_table_index_from_hash`: quadratic probing over the open-addressed
/// pattern table; collect row indices until an all-zero row is hit.
pub fn probe_indices(hash_index: u64, table: &[[u32; 4]]) -> Vec<usize> {
    let max_ind = table.len() as u64;
    let mut found = Vec::new();
    let mut c: u64 = 0;
    loop {
        let i = (hash_index.wrapping_add(c.wrapping_mul(c))) % max_ind;
        let row = &table[i as usize];
        if row.iter().all(|&v| v == 0) {
            return found;
        }
        found.push(i as usize);
        c += 1;
    }
}
